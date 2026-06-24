"""Trading-capable MetaTrader 5 session helpers and operational utilities."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from math import floor, isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import pandas as pd
from pdmt5 import Mt5Config, Mt5RuntimeError, Mt5TradingClient, Mt5TradingError

from .history import drop_forming_rate_bar
from .sdk import build_config
from .utils import coerce_login as _coerce_login

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

_logger = logging.getLogger(__name__)

PositionSide = Literal["long", "short"]
OrderSide = Literal["BUY", "SELL"]
OrderFillingMode = Literal["IOC", "FOK", "RETURN"]
OrderTimeMode = Literal["GTC", "DAY", "SPECIFIED", "SPECIFIED_DAY"]
ExecutionStatus = Literal["executed", "dry_run", "skipped", "failed"]
ProjectionMode = Literal["add", "replace_symbol"]


class MarginVolume(TypedDict):
    """Affordable volume bounds derived from account margin and symbol constraints."""

    margin_free: float
    available_margin: float
    trade_margin: float
    buy_volume: float
    sell_volume: float
    volume_min: float
    volume_max: float
    volume_step: float


class OrderLimits(TypedDict):
    """Protective order prices derived from current quotes and ratio parameters."""

    entry: float
    stop_loss: float | None
    take_profit: float | None


class OrderExecutionResult(TypedDict):
    """Normalized result from market-order and position-management helpers."""

    status: ExecutionStatus
    symbol: str
    order_side: OrderSide
    volume: float
    retcode: int | None
    comment: str | None
    request: dict[str, object]
    response: dict[str, object] | None
    dry_run: bool


_ORDER_FILLING_MODES: frozenset[str] = frozenset({"IOC", "FOK", "RETURN"})
_ORDER_TIME_MODES: frozenset[str] = frozenset({
    "GTC",
    "DAY",
    "SPECIFIED",
    "SPECIFIED_DAY",
})
_SUCCESS_RETCODE_NAMES: tuple[str, ...] = (
    "TRADE_RETCODE_DONE",
    "TRADE_RETCODE_DONE_PARTIAL",
    "TRADE_RETCODE_PLACED",
)
_SUCCESS_RETCODE_FALLBACKS: frozenset[int] = frozenset({10008, 10009, 10010})

_ACCOUNT_SNAPSHOT_FIELDS = (
    "login",
    "balance",
    "equity",
    "margin",
    "margin_free",
    "margin_level",
    "leverage",
    "currency",
)
_SYMBOL_SNAPSHOT_FIELDS = (
    "symbol",
    "visible",
    "trade_mode",
    "digits",
    "point",
    "volume_min",
    "volume_max",
    "volume_step",
    "trade_contract_size",
    "trade_tick_size",
    "trade_tick_value",
    "trade_stops_level",
    "filling_mode",
)
_TICK_SNAPSHOT_FIELDS = ("symbol", "time", "bid", "ask", "last", "volume")
POSITION_COLUMNS = (
    "ticket",
    "time",
    "symbol",
    "type",
    "volume",
    "price_open",
    "sl",
    "tp",
    "price_current",
    "profit",
    "swap",
    "comment",
)

__all__ = [
    "POSITION_COLUMNS",
    "ExecutionStatus",
    "MarginVolume",
    "OrderExecutionResult",
    "OrderFillingMode",
    "OrderLimits",
    "OrderSide",
    "OrderTimeMode",
    "PositionSide",
    "ProjectionMode",
    "calculate_account_projected_margin_ratio",
    "calculate_margin_and_volume",
    "calculate_new_position_margin_ratio",
    "calculate_positions_margin",
    "calculate_positions_margin_by_symbol",
    "calculate_positions_margin_safe",
    "calculate_projected_margin_ratio",
    "calculate_spread_ratio",
    "calculate_symbol_group_margin_ratio",
    "calculate_trailing_stop_updates",
    "calculate_volume_by_margin",
    "close_open_positions",
    "create_trading_client",
    "detect_position_side",
    "determine_order_limits",
    "ensure_symbol_selected",
    "estimate_order_margin",
    "extract_tick_price",
    "fetch_latest_closed_rates_for_trading_client",
    "fetch_latest_closed_rates_indexed",
    "get_account_snapshot",
    "get_positions_frame",
    "get_symbol_snapshot",
    "get_tick_snapshot",
    "mt5_trading_session",
    "normalize_order_volume",
    "place_market_order",
    "update_sltp_for_open_positions",
    "update_trailing_stop_loss_for_open_positions",
]


def _minimum_stop_distance(
    symbol_info: dict[str, float | int | str | bool | None],
) -> float:
    """Return the minimum SL/TP distance in price units from broker stop level."""
    stops_level = symbol_info.get("trade_stops_level")
    point = symbol_info.get("point")
    if not isinstance(stops_level, int | float) or not isinstance(point, int | float):
        return 0.0
    level = float(stops_level)
    pt = float(point)
    if level <= 0 or pt <= 0:
        return 0.0
    return level * pt


def _validate_protective_prices(
    *,
    symbol: str,
    side: PositionSide,
    entry: float,
    stop_loss: float | None,
    take_profit: float | None,
    min_distance: float,
) -> None:
    """Validate SL/TP distances against broker stop-level constraints.

    Raises:
        Mt5TradingError: When a protective price is closer than ``min_distance``.
    """
    if min_distance <= 0:
        return
    if side == "long":
        if stop_loss is not None and (entry - stop_loss) < min_distance:
            msg = (
                f"Stop loss for {symbol!r} violates broker stop level "
                f"(minimum distance {min_distance})."
            )
            raise Mt5TradingError(msg)
        if take_profit is not None and (take_profit - entry) < min_distance:
            msg = (
                f"Take profit for {symbol!r} violates broker stop level "
                f"(minimum distance {min_distance})."
            )
            raise Mt5TradingError(msg)
        return
    if stop_loss is not None and (stop_loss - entry) < min_distance:
        msg = (
            f"Stop loss for {symbol!r} violates broker stop level "
            f"(minimum distance {min_distance})."
        )
        raise Mt5TradingError(msg)
    if take_profit is not None and (entry - take_profit) < min_distance:
        msg = (
            f"Take profit for {symbol!r} violates broker stop level "
            f"(minimum distance {min_distance})."
        )
        raise Mt5TradingError(msg)


def ensure_symbol_selected(client: Mt5TradingClient, symbol: str) -> None:
    """Ensure a symbol is visible in Market Watch before sending orders.

    Args:
        client: Connected ``Mt5TradingClient`` instance.
        symbol: Symbol to select.

    Raises:
        Mt5TradingError: If the symbol cannot be selected in Market Watch or
            ``symbol_select`` is unavailable on the client.
    """
    snapshot = get_symbol_snapshot(client, symbol)
    if snapshot.get("visible"):
        return
    select = getattr(client, "symbol_select", None)
    if not callable(select):
        msg = "MT5 client is missing required method: symbol_select"
        raise Mt5TradingError(msg)
    if select(symbol, enable=True):
        return
    last_error = getattr(client, "last_error", None)
    detail = f" ({last_error()})" if callable(last_error) else ""
    msg = f"Failed to select symbol {symbol!r} in Market Watch{detail}."
    raise Mt5TradingError(msg)


def _require_unit_ratio(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be between 0 and 1 inclusive."
        raise ValueError(msg)


def _require_protective_ratio(value: float, name: str) -> None:
    if not 0.0 <= value < 1.0:
        msg = f"{name} must be at least 0 and less than 1."
        raise ValueError(msg)


def _sum_position_volume(positions: pd.DataFrame, position_type: object) -> float:
    matched = positions.loc[positions["type"] == position_type, "volume"]
    if matched.empty:
        return 0.0
    return float(matched.to_numpy(dtype=float).sum())


def _resolve_config(
    *,
    config: Mt5Config | None,
    login: int | str | None,
    password: str | None,
    server: str | None,
    path: str | None,
    timeout: int | None,
) -> Mt5Config:
    if config is not None:
        return config
    return build_config(
        path=path,
        login=_coerce_login(login),
        password=password,
        server=server,
        timeout=timeout,
    )


def _normalize_order_side(side: str) -> OrderSide:
    normalized = side.upper()
    if normalized in {"BUY", "LONG"}:
        return "BUY"
    if normalized in {"SELL", "SHORT"}:
        return "SELL"
    msg = f"Unsupported order side: {side!r}. Expected 'BUY' or 'SELL'."
    raise ValueError(msg)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and isfinite(value)


def _is_positive_finite_number(value: object) -> bool:
    if not _is_finite_number(value):
        return False
    return float(cast("float | int", value)) > 0


def normalize_order_volume(
    volume: float,
    *,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> float:
    """Normalize a requested order volume to broker volume constraints.

    Returns:
        Volume floored to the nearest valid broker step from ``volume_min``,
        capped at ``volume_max`` when finite and positive, and rounded
        deterministically. Returns ``0.0`` when inputs or constraints are
        invalid, non-finite, or the capped request is below ``volume_min``.
    """
    if not _is_finite_number(volume):
        return 0.0
    if not _is_positive_finite_number(volume_min):
        return 0.0
    if not _is_positive_finite_number(volume_step):
        return 0.0
    has_volume_cap = _is_positive_finite_number(volume_max)
    capped = min(volume, volume_max) if has_volume_cap else volume
    if capped < volume_min:
        return 0.0
    steps = floor(((capped - volume_min) / volume_step) + 1e-12)
    normalized = volume_min + max(0, steps) * volume_step
    if has_volume_cap:
        normalized = min(normalized, volume_max)
    return round(normalized, 10)


def _position_side_from_order_side(side: str) -> PositionSide:
    normalized = side.lower()
    if normalized in {"long", "buy"}:
        return "long"
    if normalized in {"short", "sell"}:
        return "short"
    msg = f"Unsupported position side: {side!r}. Expected 'long' or 'short'."
    raise ValueError(msg)


def _snapshot_from_value(value: object, fields: tuple[str, ...]) -> dict[str, object]:
    if isinstance(value, pd.DataFrame):
        row: dict[str, object] = (
            {} if value.empty else cast("dict[str, object]", value.iloc[0].to_dict())
        )
    else:
        asdict = getattr(value, "_asdict", None)
        if callable(asdict):
            row = cast("dict[str, object]", asdict())
        elif isinstance(value, dict):
            typed_value = cast("dict[object, object]", value)
            row = {str(key): item for key, item in typed_value.items()}
        else:
            row = {
                field: getattr(value, field)
                for field in fields
                if hasattr(value, field)
            }
    if not fields:
        return row
    return {field: row.get(field) for field in fields}


def _call_snapshot_method(client: Mt5TradingClient, *names: str) -> object:
    for name in names:
        method = getattr(client, name, None)
        if callable(method):
            return method()
    msg = f"MT5 client is missing required method: {' or '.join(names)}"
    raise AttributeError(msg)


def _resolve_mt5_constant(
    mt5: object,
    prefix: str,
    value: str,
    allowed: frozenset[str],
) -> int:
    normalized = value.upper()
    if normalized not in allowed:
        msg = f"Unsupported {prefix.lower()} mode: {value!r}."
        raise ValueError(msg)
    name = f"{prefix}_{normalized}"
    try:
        return cast("int", getattr(mt5, name))
    except AttributeError as exc:
        msg = f"MT5 module is missing required constant: {name}"
        raise Mt5TradingError(msg) from exc


def _parse_digit_string(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    sign = 1
    if text[0] == "+":
        text = text[1:].strip()
    elif text[0] == "-":
        sign = -1
        text = text[1:].strip()
    return sign * int(text) if text.isdigit() else None


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        return _parse_digit_string(value)
    return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_price(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        return None
    price = float(value)
    if price <= 0 or not isfinite(price):
        return None
    return price


def extract_tick_price(tick: Mapping[str, object], key: str) -> float | None:
    """Return a positive finite float from tick[key], or None if invalid.

    Accepts int, float, or numeric string values. Returns None when the key is
    missing, the value is None, non-numeric, NaN, infinite, zero, or negative.
    Booleans are treated as non-numeric and return None.
    """
    value = tick.get(key)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        price = float(value)
    elif isinstance(value, str):
        try:
            price = float(value)
        except ValueError:
            return None
    else:
        return None
    if not isfinite(price) or price <= 0:
        return None
    return price


def _success_retcodes(mt5: object) -> frozenset[int]:
    values = {
        value
        for name in _SUCCESS_RETCODE_NAMES
        if isinstance(value := getattr(mt5, name, None), int)
    }
    return frozenset(values) or _SUCCESS_RETCODE_FALLBACKS


def _order_status_from_retcode(mt5: object, retcode: object) -> ExecutionStatus:
    normalized = _optional_int(retcode)
    if normalized is None:
        return "failed"
    if normalized not in _success_retcodes(mt5):
        return "failed"
    return "executed"


def _calculate_min_volume_if_affordable(
    client: Mt5TradingClient,
    symbol: str,
    available_margin: float,
    order_side: OrderSide,
) -> float:
    if available_margin <= 0:
        return 0.0
    symbol_info = get_symbol_snapshot(client, symbol)
    volume_min = float(symbol_info.get("volume_min") or 0.0)
    volume_max = float(symbol_info.get("volume_max") or 0.0)
    volume_step = float(symbol_info.get("volume_step") or volume_min or 0.0)
    if (
        volume_min <= 0
        or volume_step <= 0
        or (volume_max > 0 and volume_min > volume_max)
    ):
        msg = f"Invalid volume constraints for {symbol!r}."
        raise Mt5TradingError(msg)
    side = _normalize_order_side(order_side)
    price = extract_tick_price(
        get_tick_snapshot(client, symbol), "ask" if side == "BUY" else "bid"
    )
    if price is None:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5TradingError(msg)
    order_type = (
        client.mt5.ORDER_TYPE_BUY if side == "BUY" else client.mt5.ORDER_TYPE_SELL
    )
    min_margin = float(client.order_calc_margin(order_type, symbol, volume_min, price))
    return round(volume_min, 10) if 0 < min_margin <= available_margin else 0.0


def create_trading_client(
    *,
    config: Mt5Config | None = None,
    login: int | str | None = None,
    password: str | None = None,
    server: str | None = None,
    path: str | None = None,
    timeout: int | None = None,
    retry_count: int = 0,
) -> Mt5TradingClient:
    """Return an initialized and logged-in trading client."""
    mt5_config = _resolve_config(
        config=config,
        login=login,
        password=password,
        server=server,
        path=path,
        timeout=timeout,
    )
    client = Mt5TradingClient(config=mt5_config, retry_count=retry_count)
    try:
        client.initialize_and_login_mt5()
    except Exception:
        client.shutdown()
        raise
    return client


def detect_position_side(
    client: Mt5TradingClient,
    symbol: str,
) -> PositionSide | None:
    """Detect the net open position side for a symbol.

    Args:
        client: Connected ``Mt5TradingClient`` instance.
        symbol: Symbol to inspect.

    Returns:
        ``"long"`` when there are buy positions and no sell positions,
        ``"short"`` when there are sell positions and no buy positions, or
        ``None`` when no positions or mixed exposure exists.
    """
    positions = get_positions_frame(client, symbol=symbol)
    if positions.empty:
        return None

    buy_type = client.mt5.POSITION_TYPE_BUY
    sell_type = client.mt5.POSITION_TYPE_SELL
    buy_volume = _sum_position_volume(positions, buy_type)
    sell_volume = _sum_position_volume(positions, sell_type)
    if buy_volume > 0 and sell_volume == 0:
        return "long"
    if sell_volume > 0 and buy_volume == 0:
        return "short"
    return None


def get_account_snapshot(
    client: Mt5TradingClient,
) -> dict[str, float | int | str | None]:
    """Return normalized account state with stable keys."""
    value = _call_snapshot_method(client, "account_info_as_dict", "account_info")
    return cast(
        "dict[str, float | int | str | None]",
        _snapshot_from_value(value, _ACCOUNT_SNAPSHOT_FIELDS),
    )


def get_symbol_snapshot(
    client: Mt5TradingClient,
    symbol: str,
) -> dict[str, float | int | str | bool | None]:
    """Return normalized symbol metadata required for trading decisions."""
    method = getattr(client, "symbol_info_as_dict", None)
    value = method(symbol=symbol) if callable(method) else client.symbol_info(symbol)
    snapshot = _snapshot_from_value(value, _SYMBOL_SNAPSHOT_FIELDS)
    snapshot["symbol"] = snapshot.get("symbol") or symbol
    return cast("dict[str, float | int | str | bool | None]", snapshot)


def get_tick_snapshot(
    client: Mt5TradingClient,
    symbol: str,
) -> dict[str, float | int | None]:
    """Return normalized latest tick data, including bid, ask, and timestamp."""
    method = getattr(client, "symbol_info_tick_as_dict", None)
    value = (
        method(symbol=symbol) if callable(method) else client.symbol_info_tick(symbol)
    )
    snapshot = _snapshot_from_value(value, _TICK_SNAPSHOT_FIELDS)
    snapshot["symbol"] = snapshot.get("symbol") or symbol
    return cast("dict[str, float | int | None]", snapshot)


def get_positions_frame(
    client: Mt5TradingClient,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Return open positions as a DataFrame with stable baseline columns."""
    frame = client.positions_get_as_df(symbol=symbol)
    for column in POSITION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame


def _order_side_from_position_type(
    client: Mt5TradingClient,
    position_type: object,
) -> OrderSide | None:
    if position_type == client.mt5.POSITION_TYPE_BUY:
        return "BUY"
    if position_type == client.mt5.POSITION_TYPE_SELL:
        return "SELL"
    return None


def _ensure_rate_time_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time" in frame.columns:
        return frame
    if frame.index.name == "time" or isinstance(frame.index, pd.DatetimeIndex):
        normalized = frame.reset_index()
        if "time" not in normalized.columns and not normalized.empty:
            normalized = normalized.rename(columns={normalized.columns[0]: "time"})
        return normalized
    return frame


def estimate_order_margin(
    client: Mt5TradingClient,
    symbol: str,
    order_side: OrderSide | str,
    volume: float,
) -> float:
    """Estimate required margin for one order at the current market price.

    Returns:
        Positive finite margin required for the order at the current quote.

    Raises:
        Mt5TradingError: If volume, tick data, or margin estimation is invalid.
    """
    if not _is_positive_finite_number(volume):
        msg = "Volume must be a positive finite number to estimate order margin."
        raise Mt5TradingError(msg)
    side = _normalize_order_side(order_side)
    tick = get_tick_snapshot(client, symbol)
    price = extract_tick_price(tick, "ask" if side == "BUY" else "bid")
    if price is None:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5TradingError(msg)
    order_type = (
        client.mt5.ORDER_TYPE_BUY if side == "BUY" else client.mt5.ORDER_TYPE_SELL
    )
    raw_margin = client.order_calc_margin(order_type, symbol, volume, price)
    try:
        margin = float(raw_margin)
    except (TypeError, ValueError) as exc:
        msg = f"Margin estimate is invalid for {symbol!r}."
        raise Mt5TradingError(msg) from exc
    if margin <= 0 or not isfinite(margin):
        msg = f"Margin estimate is invalid for {symbol!r}."
        raise Mt5TradingError(msg)
    return margin


def calculate_positions_margin(
    client: Mt5TradingClient,
    *,
    symbols: Sequence[str] | None = None,
) -> float:
    """Return the sum of estimated current margin for open positions.

    Args:
        client: Connected ``Mt5TradingClient`` instance.
        symbols: Optional symbol filter. When omitted, all open positions are
            included.

    Returns:
        Total estimated margin, or ``0.0`` when no matching positions exist.
    """
    frame = get_positions_frame(client)
    if frame.empty or "symbol" not in frame.columns:
        return 0.0
    if symbols is not None:
        frame = frame[frame["symbol"].isin(list(symbols))]
    if frame.empty:
        return 0.0
    grouped_volumes: dict[tuple[str, OrderSide], float] = {}
    for _, row in frame.iterrows():
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            continue
        volume = row.get("volume")
        if not _is_positive_finite_number(volume):
            continue
        order_side = _order_side_from_position_type(client, row.get("type"))
        if order_side is None:
            continue
        key = (symbol, order_side)
        finite_volume = float(cast("float | int", volume))
        grouped_volumes[key] = grouped_volumes.get(key, 0.0) + finite_volume
    total = 0.0
    for (symbol, order_side), volume in grouped_volumes.items():
        total += estimate_order_margin(client, symbol, order_side, volume)
    return total


def calculate_positions_margin_by_symbol(
    client: Mt5TradingClient,
    *,
    symbols: Sequence[str],
    suppress_errors: bool = True,
) -> dict[str, float]:
    """Return per-symbol estimated margin for open positions.

    Computes margin for each unique input symbol independently using the strict
    :func:`calculate_positions_margin` helper. Duplicates are deduplicated in
    first-seen order.

    Args:
        client: Connected ``Mt5TradingClient`` instance.
        symbols: Symbols to compute margin for.
        suppress_errors: When ``True``, log and skip symbols that raise
            ``Mt5TradingError``, ``Mt5RuntimeError``, or ``AttributeError``.
            When ``False``, re-raise the first failure.

    Returns:
        Mapping of symbol to margin total in first-seen unique-symbol order.
        Returns an empty dict when ``symbols`` is empty or all symbols fail
        with ``suppress_errors=True``.

    Raises:
        Mt5TradingError: When a symbol raises ``Mt5TradingError`` and
            ``suppress_errors=False``.
        Mt5RuntimeError: When a symbol raises ``Mt5RuntimeError`` and
            ``suppress_errors=False``.
        AttributeError: When a symbol raises ``AttributeError`` and
            ``suppress_errors=False``.
    """
    result: dict[str, float] = {}
    for symbol in dict.fromkeys(symbols):
        try:
            result[symbol] = calculate_positions_margin(client, symbols=[symbol])
        except (Mt5TradingError, Mt5RuntimeError, AttributeError) as exc:
            if not suppress_errors:
                raise
            _logger.warning("Skipping margin for %r: %s", symbol, exc)
    return result


def calculate_positions_margin_safe(
    client: Mt5TradingClient,
    *,
    symbols: Sequence[str],
) -> float:
    """Return the total estimated margin for open positions across symbols.

    Internally calls :func:`calculate_positions_margin_by_symbol` with
    ``suppress_errors=True``. Failed symbols are silently skipped.

    Args:
        client: Connected ``Mt5TradingClient`` instance.
        symbols: Symbols to include.

    Returns:
        Sum of per-symbol margins; ``0.0`` when no symbols or all fail.
    """
    return sum(
        calculate_positions_margin_by_symbol(client, symbols=symbols).values(),
        0.0,
    )


def calculate_spread_ratio(client: Mt5TradingClient, symbol: str) -> float:
    """Return ``(ask - bid) / ((ask + bid) / 2)`` for the latest tick.

    Raises:
        Mt5TradingError: If bid or ask is unavailable.
    """
    tick = get_tick_snapshot(client, symbol)
    bid = extract_tick_price(tick, "bid")
    ask = extract_tick_price(tick, "ask")
    if bid is None or ask is None:
        msg = f"Tick bid/ask is unavailable for {symbol!r}."
        raise Mt5TradingError(msg)
    return (ask - bid) / ((ask + bid) / 2.0)


def calculate_new_position_margin_ratio(
    client: Mt5TradingClient,
    *,
    symbol: str,
    new_position_side: OrderSide | None = None,
    new_position_volume: float = 0.0,
) -> float:
    """Return total margin/equity ratio after an optional hypothetical position.

    Raises:
        Mt5TradingError: If equity or required tick data is invalid.
    """
    account = get_account_snapshot(client)
    equity = float(account.get("equity") or 0.0)
    if equity <= 0:
        msg = "Account equity must be positive to calculate margin ratio."
        raise Mt5TradingError(msg)
    margin = float(account.get("margin") or 0.0)
    if new_position_side is not None and new_position_volume > 0:
        side = _normalize_order_side(new_position_side)
        price = extract_tick_price(
            get_tick_snapshot(client, symbol), "ask" if side == "BUY" else "bid"
        )
        if price is None:
            msg = f"Tick price is unavailable for {symbol!r}."
            raise Mt5TradingError(msg)
        order_type = (
            client.mt5.ORDER_TYPE_BUY if side == "BUY" else client.mt5.ORDER_TYPE_SELL
        )
        margin += float(
            client.order_calc_margin(order_type, symbol, new_position_volume, price),
        )
    return margin / equity


def _account_equity(client: Mt5TradingClient) -> float:
    account = get_account_snapshot(client)
    return _required_account_number(account, "equity", allow_zero=False)


def _required_account_number(
    account: Mapping[str, object],
    field: str,
    *,
    allow_zero: bool,
) -> float:
    raw_value = account.get(field)
    if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
        msg = f"Account {field} must be a finite number to calculate margin ratio."
        raise Mt5TradingError(msg)
    value = float(raw_value)
    if (
        not isfinite(value)
        or (not allow_zero and value <= 0)
        or (allow_zero and value < 0)
    ):
        msg = (
            f"Account {field} must be a non-negative finite number."
            if allow_zero
            else f"Account {field} must be a positive finite number."
        )
        raise Mt5TradingError(msg)
    return value


def calculate_account_projected_margin_ratio(
    client: Mt5TradingClient,
    *,
    symbol: str | None = None,
    new_position_side: OrderSide | None = None,
    new_position_volume: float = 0.0,
) -> float:
    """Return account-wide current plus optional new-position margin over equity.

    Current exposure comes from the broker account snapshot ``margin`` field so
    unrelated open positions remain in the baseline. Optional projected
    exposure is added via :func:`estimate_order_margin` only when a symbol, side,
    and positive volume are all supplied.

    """
    account = get_account_snapshot(client)
    equity = _required_account_number(account, "equity", allow_zero=False)
    margin = _required_account_number(account, "margin", allow_zero=True)
    if symbol is not None and new_position_side is not None and new_position_volume > 0:
        margin += estimate_order_margin(
            client,
            symbol,
            new_position_side,
            new_position_volume,
        )
    return margin / equity


def calculate_projected_margin_ratio(
    client: Mt5TradingClient,
    *,
    symbol: str,
    new_position_side: OrderSide | None = None,
    new_position_volume: float = 0.0,
) -> float:
    """Return estimated current plus optional new-position margin over equity.

    Current exposure is estimated from open positions with
    :func:`calculate_positions_margin`. Optional projected exposure is added via
    :func:`estimate_order_margin`. Thresholds and guard actions are intentionally
    left to downstream applications.

    Account equity, position margin, and optional projected margin errors from
    the composed MT5 helpers propagate to the caller.
    """
    equity = _account_equity(client)
    margin = calculate_positions_margin(client, symbols=[symbol])
    if new_position_side is not None and new_position_volume > 0:
        margin += estimate_order_margin(
            client,
            symbol,
            new_position_side,
            new_position_volume,
        )
    return margin / equity


def calculate_symbol_group_margin_ratio(
    client: Mt5TradingClient,
    *,
    symbols: Sequence[str],
    new_symbol: str | None = None,
    new_position_side: OrderSide | None = None,
    new_position_volume: float = 0.0,
    suppress_errors: bool = True,
    projection_mode: ProjectionMode = "add",
) -> float:
    """Return estimated symbol-group margin over account equity.

    Per-symbol current exposure is summed with
    :func:`calculate_positions_margin_by_symbol`. When ``new_symbol`` is inside
    the input symbol group and candidate side/volume are provided, projected order
    margin is applied according to ``projection_mode``:

    - ``"add"`` (default): adds candidate margin to the group total.
    - ``"replace_symbol"``: subtracts current margin for ``new_symbol``, then
      adds candidate margin. Useful for reversal-style projections where the new
      order is intended to replace existing exposure for that symbol.

    If the candidate margin estimation fails, the subtraction is also skipped so
    the operation is atomic. Invalid equity always raises to fail closed.

    Raises:
        AttributeError: When symbol margin lookup or projected margin lookup
            fails and ``suppress_errors`` is ``False``.
        Mt5RuntimeError: When symbol margin lookup or projected margin lookup
            fails and ``suppress_errors`` is ``False``.
        Mt5TradingError: When account equity is invalid, or when symbol margin
            lookup or projected margin lookup fails and ``suppress_errors`` is
            ``False``.
    """
    equity = _account_equity(client)
    unique_symbols = list(dict.fromkeys(symbols))
    per_symbol = calculate_positions_margin_by_symbol(
        client,
        symbols=unique_symbols,
        suppress_errors=suppress_errors,
    )
    margin = sum(per_symbol.values(), 0.0)
    if (
        new_symbol in unique_symbols
        and new_position_side is not None
        and new_position_volume > 0
    ):
        try:
            candidate_margin = estimate_order_margin(
                client,
                new_symbol,
                new_position_side,
                new_position_volume,
            )
        except (Mt5TradingError, Mt5RuntimeError, AttributeError):
            if not suppress_errors:
                raise
            _logger.warning("Skipping projected margin for %r.", new_symbol)
        else:
            if projection_mode == "replace_symbol":
                margin -= per_symbol.get(new_symbol, 0.0)
            margin += candidate_margin
    return margin / equity


def calculate_margin_and_volume(
    client: Mt5TradingClient,
    symbol: str,
    unit_margin_ratio: float,
    preserved_margin_ratio: float,
) -> MarginVolume:
    """Calculate tradable margin and volumes from account free margin.

    Applies ``preserved_margin_ratio`` to keep a reserve off ``margin_free``,
    then allocates ``unit_margin_ratio`` of the remainder as the margin budget
    for proportional volume sizing on both buy and sell sides. A
    ``unit_margin_ratio`` of ``0`` requests exactly one minimum valid unit per
    side when the post-reserve margin can afford it.

    Args:
        client: Connected ``Mt5TradingClient`` instance.
        symbol: Symbol used for minimum-lot margin and volume calculations.
        unit_margin_ratio: Fraction of post-reserve margin to allocate per unit.
        preserved_margin_ratio: Fraction of ``margin_free`` to preserve.

    Returns:
        Dictionary with ``margin_free``, ``available_margin``, ``trade_margin``,
        ``buy_volume``, and ``sell_volume``. Negative ``margin_free`` values are
        clamped to ``0.0`` before sizing.
    """
    _require_unit_ratio(unit_margin_ratio, "unit_margin_ratio")
    _require_unit_ratio(preserved_margin_ratio, "preserved_margin_ratio")

    account = client.account_info_as_dict()
    margin_free = max(0.0, float(account.get("margin_free") or 0.0))
    available_margin = margin_free * (1.0 - preserved_margin_ratio)
    trade_margin = available_margin * unit_margin_ratio
    if unit_margin_ratio == 0:
        buy_volume = _calculate_min_volume_if_affordable(
            client,
            symbol,
            available_margin,
            "BUY",
        )
        sell_volume = _calculate_min_volume_if_affordable(
            client,
            symbol,
            available_margin,
            "SELL",
        )
    else:
        buy_volume = calculate_volume_by_margin(client, symbol, trade_margin, "BUY")
        sell_volume = calculate_volume_by_margin(client, symbol, trade_margin, "SELL")
    try:
        symbol_info = get_symbol_snapshot(client, symbol)
        volume_min = float(symbol_info.get("volume_min") or 0.0)
        volume_max = float(symbol_info.get("volume_max") or 0.0)
        volume_step = float(symbol_info.get("volume_step") or 0.0)
    except AttributeError:
        volume_min = volume_max = volume_step = 0.0
    return {
        "margin_free": margin_free,
        "available_margin": available_margin,
        "trade_margin": trade_margin,
        "buy_volume": float(buy_volume),
        "sell_volume": float(sell_volume),
        "volume_min": volume_min,
        "volume_max": volume_max,
        "volume_step": volume_step,
    }


def calculate_volume_by_margin(
    client: Mt5TradingClient,
    symbol: str,
    available_margin: float,
    order_side: OrderSide,
) -> float:
    """Calculate max normalized volume affordable for one side.

    Returns:
        Largest stepped volume whose actual margin (from ``order_calc_margin``)
        fits within ``available_margin``, rounded down to symbol volume
        constraints; ``0.0`` when no affordable step exists.

    Raises:
        Mt5TradingError: If symbol volume constraints or tick data are invalid.
    """
    if available_margin <= 0:
        return 0.0
    symbol_info = get_symbol_snapshot(client, symbol)
    volume_min = float(symbol_info.get("volume_min") or 0.0)
    volume_max = float(symbol_info.get("volume_max") or 0.0)
    volume_step = float(symbol_info.get("volume_step") or volume_min or 0.0)
    if volume_min <= 0 or volume_step <= 0:
        msg = f"Invalid volume constraints for {symbol!r}."
        raise Mt5TradingError(msg)
    side = _normalize_order_side(order_side)
    price = extract_tick_price(
        get_tick_snapshot(client, symbol), "ask" if side == "BUY" else "bid"
    )
    if price is None:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5TradingError(msg)
    order_type = (
        client.mt5.ORDER_TYPE_BUY if side == "BUY" else client.mt5.ORDER_TYPE_SELL
    )
    min_margin = float(client.order_calc_margin(order_type, symbol, volume_min, price))
    if min_margin <= 0 or min_margin > available_margin:
        return 0.0
    lo = 0
    hi = int(
        max(
            0,
            floor(
                (
                    (
                        min(available_margin / min_margin * volume_min, volume_max)
                        if volume_max > 0
                        else available_margin / min_margin * volume_min
                    )
                    - volume_min
                )
                / volume_step
                + 1e-12
            ),
        )
    )
    best = -1

    while lo <= hi:
        mid = (lo + hi) // 2
        normalized = round(volume_min + mid * volume_step, 10)
        actual = float(client.order_calc_margin(order_type, symbol, normalized, price))

        if actual > 0 and actual <= available_margin:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return round(volume_min + best * volume_step, 10) if best >= 0 else 0.0


def determine_order_limits(
    client: Mt5TradingClient,
    symbol: str,
    side: PositionSide | str,
    stop_loss_limit_ratio: float | None = None,
    take_profit_limit_ratio: float | None = None,
) -> OrderLimits:
    """Derive entry and protective order prices from current market quotes.

    Args:
        client: Connected ``Mt5TradingClient`` instance.
        symbol: Symbol used for the quote lookup.
        side: Position side as ``"long"``/``"short"`` (``"buy"``/``"sell"``
            aliases are accepted).
        stop_loss_limit_ratio: Relative distance from entry for stop loss in
            ``[0, 1)``. A value of ``0`` omits the stop loss.
        take_profit_limit_ratio: Relative distance from entry for take profit in
            ``[0, 1)``. A value of ``0`` omits the take profit.

    Returns:
        Dictionary with ``entry``, ``stop_loss``, and ``take_profit`` keys.
        Omitted protective levels are returned as ``None``.

    Raises:
        Mt5TradingError: If required tick data is invalid or computed SL/TP
            prices violate available ``trade_stops_level`` pre-validation.
    """
    stop_loss_ratio = stop_loss_limit_ratio or 0.0
    take_profit_ratio = take_profit_limit_ratio or 0.0
    _require_protective_ratio(stop_loss_ratio, "stop_loss_limit_ratio")
    _require_protective_ratio(take_profit_ratio, "take_profit_limit_ratio")
    normalized_side = _position_side_from_order_side(side)
    tick = get_tick_snapshot(client, symbol)
    entry_key = "ask" if normalized_side == "long" else "bid"
    entry = extract_tick_price(tick, entry_key)
    if entry is None:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5TradingError(msg)
    try:
        symbol_info = get_symbol_snapshot(client, symbol)
    except (AttributeError, KeyError, TypeError, ValueError):
        symbol_info = {}
    try:
        digits = int(symbol_info.get("digits") or 8)
    except (TypeError, ValueError):
        digits = 8
    min_distance = _minimum_stop_distance(symbol_info)

    stop_loss: float | None = None
    if stop_loss_ratio > 0:
        if normalized_side == "long":
            stop_loss = entry * (1.0 - stop_loss_ratio)
        else:
            stop_loss = entry * (1.0 + stop_loss_ratio)
        stop_loss = round(stop_loss, digits)

    take_profit: float | None = None
    if take_profit_ratio > 0:
        if normalized_side == "long":
            take_profit = entry * (1.0 + take_profit_ratio)
        else:
            take_profit = entry * (1.0 - take_profit_ratio)
        take_profit = round(take_profit, digits)

    _validate_protective_prices(
        symbol=symbol,
        side=normalized_side,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        min_distance=min_distance,
    )

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


def place_market_order(
    client: Mt5TradingClient,
    *,
    symbol: str,
    volume: float,
    order_side: OrderSide,
    order_filling_mode: OrderFillingMode = "IOC",
    order_time_mode: OrderTimeMode = "GTC",
    sl: float | None = None,
    tp: float | None = None,
    position: int | None = None,
    dry_run: bool = False,
) -> OrderExecutionResult:
    """Place one normalized market order or return a dry-run result.

    ``pdmt5.Mt5TradingClient.order_send()`` raises only when MT5 returns no
    response. When MT5 returns a response with a known non-success retcode, this
    helper returns ``status="failed"`` and keeps the normalized response
    details for callers to inspect.

    Returns:
        Normalized execution result containing request and response details.

    Raises:
        Mt5TradingError: If volume or required tick data is invalid.
    """
    if volume <= 0:
        msg = "volume must be positive."
        raise Mt5TradingError(msg)
    side = _normalize_order_side(order_side)
    if not dry_run:
        ensure_symbol_selected(client, symbol)
    tick = get_tick_snapshot(client, symbol)
    price = extract_tick_price(tick, "ask" if side == "BUY" else "bid")
    if price is None:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5TradingError(msg)
    request = {
        "action": client.mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": (
            client.mt5.ORDER_TYPE_BUY if side == "BUY" else client.mt5.ORDER_TYPE_SELL
        ),
        "price": price,
        "type_filling": _resolve_mt5_constant(
            client.mt5,
            "ORDER_FILLING",
            order_filling_mode,
            _ORDER_FILLING_MODES,
        ),
        "type_time": _resolve_mt5_constant(
            client.mt5,
            "ORDER_TIME",
            order_time_mode,
            _ORDER_TIME_MODES,
        ),
    }
    if sl is not None:
        request["sl"] = sl
    if tp is not None:
        request["tp"] = tp
    if position is not None:
        request["position"] = position
    if dry_run:
        return {
            "status": "dry_run",
            "symbol": symbol,
            "order_side": side,
            "volume": volume,
            "retcode": None,
            "comment": None,
            "request": cast("dict[str, object]", request),
            "response": None,
            "dry_run": True,
        }
    response = client.order_send(request)
    response_dict = _snapshot_from_value(response, ())
    raw_retcode = response_dict.get("retcode")
    retcode = _optional_int(raw_retcode)
    return {
        "status": _order_status_from_retcode(client.mt5, raw_retcode),
        "symbol": symbol,
        "order_side": side,
        "volume": volume,
        "retcode": retcode,
        "comment": _optional_str(response_dict.get("comment")),
        "request": cast("dict[str, object]", request),
        "response": response_dict,
        "dry_run": False,
    }


def _filter_positions(
    positions: pd.DataFrame,
    *,
    symbols: str | list[str] | None = None,
    tickets: list[int] | None = None,
) -> pd.DataFrame:
    frame = positions
    if symbols is not None:
        symbol_set = {symbols} if isinstance(symbols, str) else set(symbols)
        frame = frame.loc[frame["symbol"].isin(symbol_set)]
    if tickets is not None:
        frame = frame.loc[frame["ticket"].isin(tickets)]
    return frame


def close_open_positions(
    client: Mt5TradingClient,
    *,
    symbols: str | list[str] | None = None,
    tickets: list[int] | None = None,
    dry_run: bool = False,
) -> list[OrderExecutionResult]:
    """Close matching open positions.

    Returns:
        Normalized execution results for matching positions.
    """
    positions = _filter_positions(
        get_positions_frame(client),
        symbols=symbols,
        tickets=tickets,
    )
    results: list[OrderExecutionResult] = []
    for row in positions.to_dict("records"):
        pos_type = row["type"]
        side: OrderSide = "SELL" if pos_type == client.mt5.POSITION_TYPE_BUY else "BUY"
        result = place_market_order(
            client,
            symbol=str(row["symbol"]),
            volume=float(row["volume"]),
            order_side=side,
            position=int(row["ticket"]),
            dry_run=dry_run,
        )
        results.append(result)
    return results


def _symbol_digits(client: Mt5TradingClient, symbol: str) -> int | None:
    try:
        raw_digits = get_symbol_snapshot(client, symbol).get("digits")
        if raw_digits is None:
            return None
        digits = int(raw_digits)
    except (AttributeError, TypeError, ValueError):
        return None
    return digits if digits >= 0 else None


def _position_ticket(value: object) -> int | None:
    ticket = _optional_int(value)
    return ticket if ticket is not None and ticket > 0 else None


def _current_stop_loss(value: object) -> float | None:
    return _optional_price(value)


def _trailing_stop_loss(
    client: Mt5TradingClient,
    *,
    position_type: object,
    current_sl: float | None,
    bid: float | None,
    ask: float | None,
    digits: int,
    trailing_stop_ratio: float,
) -> float | None:
    next_sl: float | None = None
    if position_type == client.mt5.POSITION_TYPE_BUY:
        if bid is not None:
            next_sl = round(bid * (1.0 - trailing_stop_ratio), digits)
            if current_sl is not None and current_sl >= next_sl:
                next_sl = None
    elif position_type == client.mt5.POSITION_TYPE_SELL and ask is not None:
        next_sl = round(ask * (1.0 + trailing_stop_ratio), digits)
        if current_sl is not None and current_sl <= next_sl:
            next_sl = None
    return next_sl


def calculate_trailing_stop_updates(
    client: Mt5TradingClient,
    *,
    symbol: str,
    trailing_stop_ratio: float,
) -> dict[int, float]:
    """Return per-ticket trailing stop-loss updates for open symbol positions.

    Buy positions trail from bid using ``bid * (1 - trailing_stop_ratio)``.
    Sell positions trail from ask using ``ask * (1 + trailing_stop_ratio)``.
    Existing stop losses are preserved when they are already more favorable.
    Missing symbol metadata returns an empty update map. Positions with a
    missing side-specific tick price are skipped.
    """
    _require_protective_ratio(trailing_stop_ratio, "trailing_stop_ratio")
    positions = get_positions_frame(client, symbol=symbol)
    if positions.empty:
        return {}
    tick = get_tick_snapshot(client, symbol)
    bid = extract_tick_price(tick, "bid")
    ask = extract_tick_price(tick, "ask")
    digits = _symbol_digits(client, symbol)
    if digits is None:
        return {}

    updates: dict[int, float] = {}
    for row in positions.to_dict("records"):
        ticket = _position_ticket(row.get("ticket"))
        if ticket is None:
            continue
        next_sl = _trailing_stop_loss(
            client,
            position_type=row.get("type"),
            current_sl=_current_stop_loss(row.get("sl")),
            bid=bid,
            ask=ask,
            digits=digits,
            trailing_stop_ratio=trailing_stop_ratio,
        )
        if next_sl is None:
            continue
        updates[ticket] = next_sl
    return updates


def update_trailing_stop_loss_for_open_positions(
    client: Mt5TradingClient,
    *,
    symbol: str,
    trailing_stop_ratio: float,
    dry_run: bool = False,
) -> list[OrderExecutionResult]:
    """Update open positions whose trailing stop loss should move favorably.

    Returns:
        Normalized execution results for positions that need an SL update.
    """
    updates = calculate_trailing_stop_updates(
        client,
        symbol=symbol,
        trailing_stop_ratio=trailing_stop_ratio,
    )
    results: list[OrderExecutionResult] = []
    for ticket, stop_loss in updates.items():
        results.extend(
            update_sltp_for_open_positions(
                client,
                symbol=symbol,
                tickets=[ticket],
                stop_loss=stop_loss,
                dry_run=dry_run,
            ),
        )
    return results


def update_sltp_for_open_positions(
    client: Mt5TradingClient,
    *,
    symbol: str | None = None,
    tickets: list[int] | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    dry_run: bool = False,
) -> list[OrderExecutionResult]:
    """Update SL/TP for matching open positions.

    Returns:
        Normalized execution results for matching positions.
    """
    positions = _filter_positions(
        get_positions_frame(client),
        symbols=symbol,
        tickets=tickets,
    )
    results: list[OrderExecutionResult] = []
    for row in positions.to_dict("records"):
        request = {
            "action": client.mt5.TRADE_ACTION_SLTP,
            "symbol": row["symbol"],
            "position": row["ticket"],
        }
        sl = _optional_price(row.get("sl") if stop_loss is None else stop_loss)
        tp = _optional_price(row.get("tp") if take_profit is None else take_profit)
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        if dry_run:
            response = None
            status: ExecutionStatus = "dry_run"
        else:
            ensure_symbol_selected(client, str(row["symbol"]))
            response = _snapshot_from_value(client.order_send(request), ())
            status = _order_status_from_retcode(
                client.mt5,
                response.get("retcode"),
            )
        results.append(
            {
                "status": status,
                "symbol": str(row["symbol"]),
                "order_side": "BUY"
                if row["type"] == client.mt5.POSITION_TYPE_BUY
                else "SELL",
                "volume": float(row["volume"]),
                "retcode": None
                if response is None
                else _optional_int(response.get("retcode")),
                "comment": None
                if response is None
                else _optional_str(response.get("comment")),
                "request": cast("dict[str, object]", request),
                "response": response,
                "dry_run": dry_run,
            },
        )
    return results


def fetch_latest_closed_rates_for_trading_client(
    client: Mt5TradingClient,
    *,
    symbol: str,
    granularity: str,
    count: int,
) -> pd.DataFrame:
    """Fetch the latest closed bars from a connected trading client.

    Returns:
        Up to ``count`` closed bars ordered oldest to newest.

    Raises:
        ValueError: If ``count`` is not positive, rate data is empty or
            malformed, or the ``time`` column is missing.
        Mt5TradingError: If the trading client cannot fetch rate data.
    """
    if count <= 0:
        msg = "count must be positive."
        raise ValueError(msg)
    fetch_method = getattr(client, "fetch_latest_rates_as_df", None)
    if not callable(fetch_method):
        msg = "MT5 trading client cannot fetch rate data."
        raise Mt5TradingError(msg)
    fetched = fetch_method(symbol, granularity, count + 1)
    if not isinstance(fetched, pd.DataFrame):
        msg = (
            f"Malformed rate data for {symbol!r} at granularity {granularity!r}: "
            "expected a DataFrame."
        )
        raise ValueError(msg)  # noqa: TRY004
    frame = fetched
    frame = _ensure_rate_time_column(frame)
    if "time" not in frame.columns:
        msg = f"Rate data is missing a time column for {symbol!r}."
        raise ValueError(msg)
    closed = drop_forming_rate_bar(frame)
    if closed.empty:
        msg = (
            f"Rate data is empty for {symbol!r} at granularity {granularity!r} "
            f"with count {count}."
        )
        raise ValueError(msg)
    return closed.tail(count).reset_index(drop=True)


def _rate_time_to_utc(series: pd.Series, symbol: str) -> pd.DatetimeIndex:
    """Convert a rate time series to a UTC-aware DatetimeIndex.

    Handles MT5 epoch seconds (including object-dtype Python numbers), timezone-
    naive datetime-like values, and timezone-aware datetime-like values.

    Returns:
        UTC-aware DatetimeIndex.

    Raises:
        ValueError: If the time data is invalid, unparseable, or contains NaT.
    """
    try:
        arr = series.to_numpy()
        non_null = series.dropna()
        object_numbers = (
            pd.api.types.is_object_dtype(series)
            and non_null.map(
                lambda value: type(value) is not bool and isinstance(value, Real),
            ).all()
        )
        numeric_dtype = pd.api.types.is_numeric_dtype(
            series
        ) and not pd.api.types.is_bool_dtype(
            series,
        )
        if numeric_dtype or object_numbers:
            idx = pd.to_datetime(arr, unit="s", utc=True)
        else:
            idx = pd.to_datetime(arr, utc=True)
    except Exception as exc:
        msg = f"Rate data for {symbol!r} has invalid or unparseable time data."
        raise ValueError(msg) from exc
    if any(idx.isna()):
        msg = f"Rate data for {symbol!r} contains missing (NaT) timestamp values."
        raise ValueError(msg)
    return idx


def fetch_latest_closed_rates_indexed(
    client: Mt5TradingClient,
    *,
    symbol: str,
    granularity: str,
    count: int,
) -> pd.DataFrame:
    """Fetch the latest closed bars with a UTC DatetimeIndex from a trading client.

    Internally reuses :func:`fetch_latest_closed_rates_for_trading_client` for
    closed-bar detection and validation, then converts the ``time`` column to a
    UTC-aware :class:`~pandas.DatetimeIndex` named ``"time"`` and drops the
    original column. Intended for downstream time-series consumers that require
    a datetime index rather than a ``time`` column.

    Args:
        client: Connected trading client with rate-fetch capability.
        symbol: Symbol name.
        granularity: Timeframe string (for example ``"M1"``, ``"H1"``).
        count: Maximum number of closed bars to return.

    Returns:
        Up to ``count`` closed bars ordered oldest to newest, with a
        UTC-aware ``DatetimeIndex`` named ``"time"``. The original ``time``
        column is dropped.

    Raises:
        ValueError: If ``count`` is not positive, rate data is empty or
            malformed, the ``time`` column is missing, or timestamp data
            is invalid or unparseable.
    """
    frame = fetch_latest_closed_rates_for_trading_client(
        client,
        symbol=symbol,
        granularity=granularity,
        count=count,
    )
    if "time" not in frame.columns:
        msg = f"Rate data is missing a time column for {symbol!r}."
        raise ValueError(msg)
    idx = _rate_time_to_utc(frame["time"], symbol)
    idx.name = "time"
    result = frame.drop(columns=["time"])
    result.index = idx
    return result


@contextmanager
def mt5_trading_session(
    config: Mt5Config | None = None,
    *,
    login: int | str | None = None,
    password: str | None = None,
    server: str | None = None,
    path: str | None = None,
    timeout: int | None = None,
    retry_count: int = 0,
) -> Iterator[Mt5TradingClient]:
    """Open a trading-capable MT5 session and always shut down safely.

    Launches the MetaTrader 5 terminal using ``Mt5Config.path`` when set,
    initializes and logs in via ``initialize_and_login_mt5()``, yields a
    connected :class:`~pdmt5.Mt5TradingClient`, and calls ``shutdown()`` on
    exit even when an error is raised inside the context.

    Args:
        config: MT5 connection configuration. Defaults to an empty config that
            attaches to a running terminal.
        login: Optional trading account login.
        password: Optional trading account password.
        server: Optional trading server name.
        path: Optional terminal executable path.
        timeout: Optional connection timeout in milliseconds.
        retry_count: Number of initialization retries passed to
            ``Mt5TradingClient``.

    Yields:
        Connected ``Mt5TradingClient`` bound to the session.
    """
    client = create_trading_client(
        config=config,
        login=login,
        password=password,
        server=server,
        path=path,
        timeout=timeout,
        retry_count=retry_count,
    )
    try:
        yield client
    finally:
        client.shutdown()
