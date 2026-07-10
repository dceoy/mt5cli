"""Trading-capable MetaTrader 5 session helpers and operational utilities."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from math import floor, isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, cast

import pandas as pd
from pdmt5 import Mt5Config, Mt5RuntimeError

from .client import MT5Client, mt5_session
from .exceptions import Mt5OperationError
from .history import drop_forming_rate_bar
from .sdk import build_config
from .utils import coerce_login as _coerce_login
from .utils import parse_timeframe

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from typing import Any

_logger = logging.getLogger(__name__)


class _Mt5ClientProtocol(Protocol):
    """Minimal protocol for MT5 clients with methods required by mt5cli.

    This protocol describes the interface required by mt5cli trading helpers.
    It uses positional-only parameters to avoid structural subtyping issues with
    different client implementations that may use different parameter names.
    """

    @property
    def mt5(self) -> Any:  # noqa: ANN401
        """MT5 module with trading constants (POSITION_TYPE_*, ORDER_TYPE_*, etc.)."""
        ...

    def account_info_as_dict(self) -> dict[str, Any]:
        """Return account information as a dictionary."""
        ...

    def symbol_info(self, symbol: str, /) -> object:
        """Return symbol information."""
        ...

    def symbol_info_tick(self, symbol: str, /) -> object:
        """Return latest symbol tick information."""
        ...

    def positions_get_as_df(self, symbol: str | None = None) -> pd.DataFrame:
        """Return open positions as a DataFrame."""
        ...

    def order_calc_margin(
        self, /, action: int, symbol: str, volume: float, price: float
    ) -> Any:  # noqa: ANN401
        """Calculate required margin for an order."""
        ...

    def order_send(self, request: dict[str, Any], /) -> Any:  # noqa: ANN401
        """Send an order request and return the response."""
        ...

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        """Select/deselect a symbol in Market Watch."""
        ...

    def last_error(self) -> object:
        """Return the last error message or info."""
        ...


class _HistoryDealsClientProtocol(Protocol):
    """Minimal protocol for MT5 clients capable of retrieving history deals.

    Describes the single method required by
    :func:`fetch_recent_history_deals_for_trading_client`. The raw
    ``pdmt5.Mt5DataClient`` returned by :func:`create_trading_client`
    satisfies this protocol. ``mt5cli.sdk.Mt5CliClient`` (used via
    ``mt5_session()``) exposes ``history_deals()`` instead and does not
    satisfy this protocol.
    """

    def history_deals_get_as_df(
        self,
        date_from: datetime,
        date_to: datetime,
        group: str | None = None,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ) -> pd.DataFrame | None:
        """Return historical deals as a DataFrame, or None when none exist."""
        ...


PositionSide = Literal["long", "short"]
OrderSide = Literal["BUY", "SELL"]
OrderFillingMode = Literal["IOC", "FOK", "RETURN"]
OrderTimeMode = Literal["GTC", "DAY", "SPECIFIED", "SPECIFIED_DAY"]
ExecutionStatus = Literal[
    "filled",
    "partial_fill",
    "placed",
    "dry_run",
    "skipped",
    "rejected",
    "malformed",
    "failed",
]
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


@dataclass(frozen=True)
class OrderExecutionResult:
    """Serialization-safe normalized receipt for an execution request.

    ``filled_volume`` and ``filled_price`` are only populated when returned by
    the broker; a successful retcode never implies that the requested values
    were filled. ``response`` is diagnostic data, not the primary contract.
    """

    status: ExecutionStatus
    symbol: str
    order_side: OrderSide
    requested_volume: float
    filled_volume: float | None
    request_price: float | None
    filled_price: float | None
    order_ticket: int | None
    deal_ticket: int | None
    position_id: int | None
    magic: int | None
    retcode: int | None
    comment: str | None
    dry_run: bool
    request: dict[str, object]
    response: dict[str, object] | None

    @property
    def volume(self) -> float:
        """Deprecated shorthand retained for tabular consumers."""
        return self.requested_volume

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of this receipt."""
        return cast("dict[str, object]", asdict(self))

    def __getitem__(self, key: str) -> object:
        """Provide temporary mapping-style access for existing integrations.

        Returns:
            Receipt field selected by ``key``.
        """
        return self.to_dict()[key]


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
    "trade_exemode",
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
    "magic",
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
    "estimate_server_clock_offset_seconds",
    "extract_tick_price",
    "fetch_latest_closed_rates_for_trading_client",
    "fetch_latest_closed_rates_indexed",
    "fetch_recent_history_deals_for_trading_client",
    "get_account_snapshot",
    "get_positions_frame",
    "get_symbol_snapshot",
    "get_tick_snapshot",
    "mt5_trading_session",
    "normalize_order_volume",
    "place_market_order",
    "resolve_broker_filling_mode",
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
        Mt5OperationError: When a protective price is closer than ``min_distance``.
    """
    if min_distance <= 0:
        return
    if side == "long":
        if stop_loss is not None and (entry - stop_loss) < min_distance:
            msg = (
                f"Stop loss for {symbol!r} violates broker stop level "
                f"(minimum distance {min_distance})."
            )
            raise Mt5OperationError(msg)
        if take_profit is not None and (take_profit - entry) < min_distance:
            msg = (
                f"Take profit for {symbol!r} violates broker stop level "
                f"(minimum distance {min_distance})."
            )
            raise Mt5OperationError(msg)
        return
    if stop_loss is not None and (stop_loss - entry) < min_distance:
        msg = (
            f"Stop loss for {symbol!r} violates broker stop level "
            f"(minimum distance {min_distance})."
        )
        raise Mt5OperationError(msg)
    if take_profit is not None and (entry - take_profit) < min_distance:
        msg = (
            f"Take profit for {symbol!r} violates broker stop level "
            f"(minimum distance {min_distance})."
        )
        raise Mt5OperationError(msg)


def ensure_symbol_selected(client: _Mt5ClientProtocol, symbol: str) -> None:
    """Ensure a symbol is visible in Market Watch before sending orders.

    Args:
        client: Connected MT5 client instance.
        symbol: Symbol to select.

    Raises:
        Mt5OperationError: If the symbol cannot be selected in Market Watch or
            ``symbol_select`` is unavailable on the client.
    """
    snapshot = get_symbol_snapshot(client, symbol)
    if snapshot.get("visible"):
        return
    select = getattr(client, "symbol_select", None)
    if not callable(select):
        msg = "MT5 client is missing required method: symbol_select"
        raise Mt5OperationError(msg)
    if select(symbol, enable=True):
        return
    last_error = getattr(client, "last_error", None)
    detail = f" ({last_error()})" if callable(last_error) else ""
    msg = f"Failed to select symbol {symbol!r} in Market Watch{detail}."
    raise Mt5OperationError(msg)


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


def _call_snapshot_method(client: _Mt5ClientProtocol, *names: str) -> object:
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
        raise Mt5OperationError(msg) from exc


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


def _plain_value(value: object) -> object:  # noqa: PLR0911
    """Convert broker values into JSON-compatible diagnostic data.

    Returns:
        Value composed only of JSON-compatible primitives, mappings, and lists.
    """
    as_dict = getattr(value, "_asdict", None)
    if callable(as_dict):
        return _plain_value(as_dict())
    if isinstance(value, pd.DataFrame):
        return [_plain_value(row) for row in value.to_dict("records")]
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {str(key): _plain_value(item) for key, item in mapping.items()}
    if isinstance(value, list | tuple):
        sequence = cast("list[object] | tuple[object, ...]", value)
        return [_plain_value(item) for item in sequence]
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


def _response_mapping(response: object) -> dict[str, object] | None:
    """Normalize supported pdmt5 response shapes to a plain mapping.

    Returns:
        Stable diagnostic mapping, or ``None`` for an unsupported response.
    """
    if response is None:
        return None
    if isinstance(response, pd.DataFrame):
        if response.empty:
            return {}
        return cast("dict[str, object]", _plain_value(response.iloc[0].to_dict()))
    value = _plain_value(response)
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    fields = (
        "retcode",
        "comment",
        "order",
        "deal",
        "position",
        "volume",
        "price",
        "magic",
    )
    mapped = {
        field: getattr(response, field) for field in fields if hasattr(response, field)
    }
    return cast("dict[str, object]", _plain_value(mapped)) if mapped else None


def _receipt_status(
    mt5: object, retcode: int | None, response: dict[str, object] | None
) -> ExecutionStatus:
    """Map broker result categories without inferring fill details.

    Returns:
        Stable execution status.
    """
    if response is None:
        return "failed"
    if retcode is None:
        return "malformed"
    partial = _optional_int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))
    placed = _optional_int(getattr(mt5, "TRADE_RETCODE_PLACED", 10008))
    done = _optional_int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
    if retcode == partial:
        return "partial_fill"
    if retcode == placed:
        return "placed"
    if retcode == done or retcode in _success_retcodes(mt5):
        return "filled"
    return "rejected"


def _execution_receipt(
    *,
    mt5: object,
    symbol: str,
    order_side: OrderSide,
    request: dict[str, object],
    response: object = None,
    dry_run: bool = False,
    status: ExecutionStatus | None = None,
    error: BaseException | None = None,
) -> OrderExecutionResult:
    """Create the one public execution receipt from any broker response.

    Returns:
        Fully normalized execution receipt.
    """
    normalized_request = cast("dict[str, object]", _plain_value(request))
    normalized_response = _response_mapping(response)
    retcode = (
        None
        if normalized_response is None
        else _optional_int(normalized_response.get("retcode"))
    )
    if dry_run:
        resolved_status: ExecutionStatus = "dry_run"
        normalized_response = None
    elif error is not None:
        resolved_status = "failed"
    else:
        resolved_status = status or _receipt_status(mt5, retcode, normalized_response)
    comment = (
        str(error)
        if error is not None
        else None
        if normalized_response is None
        else _optional_str(normalized_response.get("comment"))
    )
    return OrderExecutionResult(
        status=resolved_status,
        symbol=symbol,
        order_side=order_side,
        requested_volume=_optional_price(request.get("volume")) or 0.0,
        filled_volume=None
        if normalized_response is None
        else _optional_price(normalized_response.get("volume")),
        request_price=_optional_price(request.get("price")),
        filled_price=None
        if normalized_response is None
        else _optional_price(normalized_response.get("price")),
        order_ticket=None
        if normalized_response is None
        else _optional_int(normalized_response.get("order")),
        deal_ticket=None
        if normalized_response is None
        else _optional_int(normalized_response.get("deal")),
        position_id=(
            _optional_int(request.get("position"))
            if normalized_response is None
            else _optional_int(normalized_response.get("position"))
            or _optional_int(request.get("position"))
        ),
        magic=_optional_int(request.get("magic"))
        if normalized_response is None
        else _optional_int(normalized_response.get("magic"))
        or _optional_int(request.get("magic")),
        retcode=retcode,
        comment=comment,
        dry_run=dry_run,
        request=normalized_request,
        response=normalized_response,
    )


def _calculate_min_volume_if_affordable(
    client: _Mt5ClientProtocol,
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
        raise Mt5OperationError(msg)
    side = _normalize_order_side(order_side)
    price = extract_tick_price(
        get_tick_snapshot(client, symbol), "ask" if side == "BUY" else "bid"
    )
    if price is None:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5OperationError(msg)
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
) -> MT5Client:
    """Create a connected :class:`MT5Client` for transitional internal use.

    New callers must use :func:`mt5cli.mt5_session`; it has explicit ownership
    semantics and cannot leak a low-level pdmt5 client.

    Returns:
        Connected public client.
    """
    mt5_config = _resolve_config(
        config=config,
        login=login,
        password=password,
        server=server,
        path=path,
        timeout=timeout,
    )
    client = MT5Client(config=mt5_config, retry_count=retry_count)
    client.__enter__()  # noqa: PLC2801
    return client


def detect_position_side(
    client: _Mt5ClientProtocol,
    symbol: str,
    *,
    magic: int | None = None,
) -> PositionSide | None:
    """Detect the net open position side for a symbol.

    Args:
        client: Connected MT5 client instance.
        symbol: Symbol to inspect.
        magic: Optional magic number filter applied fail-closed.

    Returns:
        ``"long"`` when there are buy positions and no sell positions,
        ``"short"`` when there are sell positions and no buy positions, or
        ``None`` when no positions or mixed exposure exists.
    """
    positions = _filter_positions(
        get_positions_frame(client, symbol=symbol), magic=magic
    )
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
    client: _Mt5ClientProtocol,
) -> dict[str, float | int | str | None]:
    """Return normalized account state with stable keys."""
    value = _call_snapshot_method(client, "account_info_as_dict", "account_info")
    return cast(
        "dict[str, float | int | str | None]",
        _snapshot_from_value(value, _ACCOUNT_SNAPSHOT_FIELDS),
    )


def get_symbol_snapshot(
    client: _Mt5ClientProtocol,
    symbol: str,
) -> dict[str, float | int | str | bool | None]:
    """Return normalized symbol metadata required for trading decisions."""
    method = getattr(client, "symbol_info_as_dict", None)
    value = method(symbol=symbol) if callable(method) else client.symbol_info(symbol)
    snapshot = _snapshot_from_value(value, _SYMBOL_SNAPSHOT_FIELDS)
    snapshot["symbol"] = snapshot.get("symbol") or symbol
    return cast("dict[str, float | int | str | bool | None]", snapshot)


def get_tick_snapshot(
    client: _Mt5ClientProtocol,
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


_SERVER_CLOCK_OFFSET_ROUNDING_SECONDS = 1800.0
_MAX_PLAUSIBLE_SERVER_CLOCK_OFFSET_SECONDS = 14 * 3600.0


def estimate_server_clock_offset_seconds(
    client: _Mt5ClientProtocol,
    symbol: str,
) -> float | None:
    """Estimate the broker server clock offset from true UTC.

    MT5 tick, bar, and deal timestamps are epoch values labeled in the
    broker server's wall clock, which is commonly UTC+2 or UTC+3 rather
    than true UTC. This reads the latest tick for ``symbol`` and compares
    its timestamp to the current UTC time, rounding to the nearest half
    hour: server offsets are whole or half-hour multiples, and rounding
    absorbs a few minutes of real tick staleness.

    On a closed market, weekend, holiday, or illiquid symbol the latest tick
    can be hours or days old, in which case the tick-vs-now delta reflects
    tick staleness rather than a true clock offset. Real-world broker server
    offsets fall within roughly UTC-12..UTC+14, so a rounded delta whose
    magnitude exceeds 14 hours is treated as an implausible, stale-tick
    reading and discarded (with a warning) rather than returned.

    No offset is applied anywhere automatically; pass the result to
    :func:`fetch_recent_history_deals_for_trading_client` (via
    ``server_clock_offset_seconds``) or use it directly when comparing
    MT5-labeled timestamps to wall-clock time.

    Args:
        client: Connected MT5 client exposing tick snapshot access.
        symbol: Symbol whose latest tick supplies the timestamp.

    Returns:
        Estimated offset in seconds (server time minus UTC), rounded to the
        nearest 1800 seconds, or ``None`` when no valid tick time is
        available or the estimate is implausibly large (likely a stale
        tick rather than a true clock offset).
    """
    snapshot = get_tick_snapshot(client, symbol)
    tick_epoch = extract_tick_price(snapshot, "time")
    if tick_epoch is None:
        _logger.warning(
            "Cannot estimate MT5 server clock offset for %s: no valid tick time.",
            symbol,
        )
        return None
    raw_offset_seconds = tick_epoch - datetime.now(UTC).timestamp()
    offset_seconds = (
        round(raw_offset_seconds / _SERVER_CLOCK_OFFSET_ROUNDING_SECONDS)
        * _SERVER_CLOCK_OFFSET_ROUNDING_SECONDS
    )
    if abs(offset_seconds) > _MAX_PLAUSIBLE_SERVER_CLOCK_OFFSET_SECONDS:
        _logger.warning(
            "Discarding implausible MT5 server clock offset for %s: %.0f seconds"
            " (likely a stale tick, not a clock offset).",
            symbol,
            offset_seconds,
        )
        return None
    _logger.info(
        "Estimated MT5 server clock offset for %s: %.0f seconds.",
        symbol,
        offset_seconds,
    )
    return offset_seconds


def get_positions_frame(
    client: _Mt5ClientProtocol,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Return open positions as a DataFrame with stable baseline columns."""
    frame = client.positions_get_as_df(symbol=symbol)
    for column in POSITION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame


def _supported_filling_modes(
    client: _Mt5ClientProtocol,
    *,
    symbol: str,
    preferred_default: OrderFillingMode,
) -> set[str] | None:
    snapshot = get_symbol_snapshot(client, symbol)
    filling_mode = _optional_int(snapshot.get("filling_mode"))
    trade_exemode = _optional_int(snapshot.get("trade_exemode"))
    if filling_mode is None and trade_exemode is None:
        _logger.debug(
            "Filling-mode metadata unavailable for %s; keeping preferred mode %s.",
            symbol,
            preferred_default,
        )
        return None

    supported: set[str] = set()
    if filling_mode is not None:
        fok_flag = getattr(client.mt5, "SYMBOL_FILLING_FOK", None)
        ioc_flag = getattr(client.mt5, "SYMBOL_FILLING_IOC", None)
        if isinstance(fok_flag, int) and filling_mode & fok_flag:
            supported.add("FOK")
        if isinstance(ioc_flag, int) and filling_mode & ioc_flag:
            supported.add("IOC")
    # MQL5 permits IOC/FOK for Request and Instant execution regardless of the
    # SYMBOL_FILLING_MODE bitmask, which only governs Market/Exchange execution.
    implicit_ioc_fok_modes = {
        mode
        for mode in (
            getattr(client.mt5, "SYMBOL_TRADE_EXECUTION_REQUEST", None),
            getattr(client.mt5, "SYMBOL_TRADE_EXECUTION_INSTANT", None),
        )
        if isinstance(mode, int)
    }
    if trade_exemode is not None and trade_exemode in implicit_ioc_fok_modes:
        supported.update({"IOC", "FOK"})
    market_execution = getattr(client.mt5, "SYMBOL_TRADE_EXECUTION_MARKET", None)
    if trade_exemode is not None and not (
        isinstance(market_execution, int) and trade_exemode == market_execution
    ):
        supported.add("RETURN")
    if supported:
        return supported

    _logger.debug(
        "Filling-mode metadata was unparseable for %s; keeping preferred mode %s.",
        symbol,
        preferred_default,
    )
    return None


def resolve_broker_filling_mode(
    client: _Mt5ClientProtocol,
    *,
    symbol: str,
    preferred_modes: Sequence[OrderFillingMode] = ("IOC", "FOK", "RETURN"),
    default_mode: OrderFillingMode = "IOC",
) -> OrderFillingMode:
    """Return the first broker-supported filling mode from a preferred order.

    Raises:
        ValueError: If any preferred or default mode name is unsupported.
    """
    preferred = [mode.upper() for mode in preferred_modes]
    for mode in [*preferred, default_mode]:
        if mode not in _ORDER_FILLING_MODES:
            msg = f"Unsupported order_filling mode: {mode!r}."
            raise ValueError(msg)

    preferred_default = cast(
        "OrderFillingMode", preferred[0] if preferred else default_mode
    )
    supported = _supported_filling_modes(
        client,
        symbol=symbol,
        preferred_default=preferred_default,
    )
    if supported is None:
        return preferred_default

    for mode in preferred:
        if mode in supported:
            return cast("OrderFillingMode", mode)
    fallback_mode = next(
        mode for mode in (default_mode, "IOC", "FOK", "RETURN") if mode in supported
    )
    return cast("OrderFillingMode", fallback_mode)


def _order_side_from_position_type(
    client: _Mt5ClientProtocol,
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
    client: _Mt5ClientProtocol,
    symbol: str,
    order_side: OrderSide | str,
    volume: float,
) -> float:
    """Estimate required margin for one order at the current market price.

    Returns:
        Positive finite margin required for the order at the current quote.

    Raises:
        Mt5OperationError: If volume, tick data, or margin estimation is invalid.
    """
    if not _is_positive_finite_number(volume):
        msg = "Volume must be a positive finite number to estimate order margin."
        raise Mt5OperationError(msg)
    side = _normalize_order_side(order_side)
    tick = get_tick_snapshot(client, symbol)
    price = extract_tick_price(tick, "ask" if side == "BUY" else "bid")
    if price is None:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5OperationError(msg)
    order_type = (
        client.mt5.ORDER_TYPE_BUY if side == "BUY" else client.mt5.ORDER_TYPE_SELL
    )
    raw_margin = client.order_calc_margin(order_type, symbol, volume, price)
    try:
        margin = float(raw_margin)
    except (TypeError, ValueError) as exc:
        msg = f"Margin estimate is invalid for {symbol!r}."
        raise Mt5OperationError(msg) from exc
    if margin <= 0 or not isfinite(margin):
        msg = f"Margin estimate is invalid for {symbol!r}."
        raise Mt5OperationError(msg)
    return margin


def calculate_positions_margin(
    client: _Mt5ClientProtocol,
    *,
    symbols: Sequence[str] | None = None,
) -> float:
    """Return the sum of estimated current margin for open positions.

    Args:
        client: Connected MT5 client instance.
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
    client: _Mt5ClientProtocol,
    *,
    symbols: Sequence[str],
    suppress_errors: bool = True,
) -> dict[str, float]:
    """Return per-symbol estimated margin for open positions.

    Computes margin for each unique input symbol independently using the strict
    :func:`calculate_positions_margin` helper. Duplicates are deduplicated in
    first-seen order.

    Args:
        client: Connected MT5 client instance.
        symbols: Symbols to compute margin for.
        suppress_errors: When ``True``, log and skip symbols that raise
            ``Mt5OperationError``, ``Mt5RuntimeError``, or ``AttributeError``.
            When ``False``, re-raise the first failure.

    Returns:
        Mapping of symbol to margin total in first-seen unique-symbol order.
        Returns an empty dict when ``symbols`` is empty or all symbols fail
        with ``suppress_errors=True``.

    Raises:
        Mt5OperationError: When a symbol raises ``Mt5OperationError`` and
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
        except (Mt5OperationError, Mt5RuntimeError, AttributeError) as exc:
            if not suppress_errors:
                raise
            _logger.warning("Skipping margin for %r: %s", symbol, exc)
    return result


def calculate_positions_margin_safe(
    client: _Mt5ClientProtocol,
    *,
    symbols: Sequence[str],
) -> float:
    """Return the total estimated margin for open positions across symbols.

    Internally calls :func:`calculate_positions_margin_by_symbol` with
    ``suppress_errors=True``. Failed symbols are silently skipped.

    Args:
        client: Connected MT5 client instance.
        symbols: Symbols to include.

    Returns:
        Sum of per-symbol margins; ``0.0`` when no symbols or all fail.
    """
    return sum(
        calculate_positions_margin_by_symbol(client, symbols=symbols).values(),
        0.0,
    )


def calculate_spread_ratio(client: _Mt5ClientProtocol, symbol: str) -> float:
    """Return ``(ask - bid) / ((ask + bid) / 2)`` for the latest tick.

    Raises:
        Mt5OperationError: If bid or ask is unavailable.
    """
    tick = get_tick_snapshot(client, symbol)
    bid = extract_tick_price(tick, "bid")
    ask = extract_tick_price(tick, "ask")
    if bid is None or ask is None:
        msg = f"Tick bid/ask is unavailable for {symbol!r}."
        raise Mt5OperationError(msg)
    return (ask - bid) / ((ask + bid) / 2.0)


def calculate_new_position_margin_ratio(
    client: _Mt5ClientProtocol,
    *,
    symbol: str,
    new_position_side: OrderSide | None = None,
    new_position_volume: float = 0.0,
) -> float:
    """Return total margin/equity ratio after an optional hypothetical position.

    Raises:
        Mt5OperationError: If equity or required tick data is invalid.
    """
    account = get_account_snapshot(client)
    equity = float(account.get("equity") or 0.0)
    if equity <= 0:
        msg = "Account equity must be positive to calculate margin ratio."
        raise Mt5OperationError(msg)
    margin = float(account.get("margin") or 0.0)
    if new_position_side is not None and new_position_volume > 0:
        side = _normalize_order_side(new_position_side)
        price = extract_tick_price(
            get_tick_snapshot(client, symbol), "ask" if side == "BUY" else "bid"
        )
        if price is None:
            msg = f"Tick price is unavailable for {symbol!r}."
            raise Mt5OperationError(msg)
        order_type = (
            client.mt5.ORDER_TYPE_BUY if side == "BUY" else client.mt5.ORDER_TYPE_SELL
        )
        margin += float(
            client.order_calc_margin(order_type, symbol, new_position_volume, price),
        )
    return margin / equity


def _account_equity(client: _Mt5ClientProtocol) -> float:
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
        raise Mt5OperationError(msg)
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
        raise Mt5OperationError(msg)
    return value


def calculate_account_projected_margin_ratio(
    client: _Mt5ClientProtocol,
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
    client: _Mt5ClientProtocol,
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


def _validate_projection_mode(projection_mode: str) -> ProjectionMode:
    if projection_mode not in {"add", "replace_symbol"}:
        msg = (
            f"Unsupported projection mode: {projection_mode!r}. "
            "Expected 'add' or 'replace_symbol'."
        )
        raise ValueError(msg)
    return cast("ProjectionMode", projection_mode)


def calculate_symbol_group_margin_ratio(
    client: _Mt5ClientProtocol,
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
        Mt5OperationError: When account equity is invalid, or when symbol margin
            lookup or projected margin lookup fails and ``suppress_errors`` is
            ``False``.
    """
    projection_mode = _validate_projection_mode(projection_mode)
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
        except (Mt5OperationError, Mt5RuntimeError, AttributeError):
            if not suppress_errors:
                raise
            _logger.warning("Skipping projected margin for %r.", new_symbol)
        else:
            if projection_mode == "replace_symbol":
                margin = max(0.0, margin - per_symbol.get(new_symbol, 0.0))
            margin += candidate_margin
    return margin / equity


def calculate_margin_and_volume(
    client: _Mt5ClientProtocol,
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
        client: Connected MT5 client instance.
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
    client: _Mt5ClientProtocol,
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
        Mt5OperationError: If symbol volume constraints or tick data are invalid.
    """
    if available_margin <= 0:
        return 0.0
    symbol_info = get_symbol_snapshot(client, symbol)
    volume_min = float(symbol_info.get("volume_min") or 0.0)
    volume_max = float(symbol_info.get("volume_max") or 0.0)
    volume_step = float(symbol_info.get("volume_step") or volume_min or 0.0)
    if volume_min <= 0 or volume_step <= 0:
        msg = f"Invalid volume constraints for {symbol!r}."
        raise Mt5OperationError(msg)
    side = _normalize_order_side(order_side)
    price = extract_tick_price(
        get_tick_snapshot(client, symbol), "ask" if side == "BUY" else "bid"
    )
    if price is None:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5OperationError(msg)
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
    client: _Mt5ClientProtocol,
    symbol: str,
    side: PositionSide | str,
    stop_loss_limit_ratio: float | None = None,
    take_profit_limit_ratio: float | None = None,
) -> OrderLimits:
    """Derive entry and protective order prices from current market quotes.

    Args:
        client: Connected MT5 client instance.
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
        Mt5OperationError: If required tick data is invalid or computed SL/TP
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
        raise Mt5OperationError(msg)
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


def place_market_order(  # noqa: C901, PLR0913
    client: _Mt5ClientProtocol,
    *,
    symbol: str,
    volume: float,
    order_side: OrderSide,
    order_filling_mode: OrderFillingMode = "IOC",
    order_time_mode: OrderTimeMode = "GTC",
    sl: float | None = None,
    tp: float | None = None,
    position: int | None = None,
    deviation: int | None = None,
    comment: str | None = None,
    magic: int | None = None,
    dry_run: bool = False,
) -> OrderExecutionResult:
    """Place one normalized market order or return a dry-run result.

    ``order_send()`` raises only when MT5 returns no response. When MT5 returns
    a response with a known non-success retcode, this helper returns
    ``status="failed"`` and keeps the normalized response details for callers
    to inspect.

    Returns:
        Normalized execution result containing request and response details.

    Raises:
        Mt5OperationError: If volume or required tick data is invalid.
    """
    if volume <= 0:
        msg = "volume must be positive."
        raise Mt5OperationError(msg)
    side = _normalize_order_side(order_side)
    if not dry_run:
        ensure_symbol_selected(client, symbol)
    tick = get_tick_snapshot(client, symbol)
    price = extract_tick_price(tick, "ask" if side == "BUY" else "bid")
    if price is None:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5OperationError(msg)
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
    if deviation is not None:
        request["deviation"] = deviation
    if comment is not None:
        request["comment"] = comment
    if magic is not None:
        request["magic"] = magic
    if dry_run:
        return _execution_receipt(
            mt5=client.mt5,
            symbol=symbol,
            order_side=side,
            request=cast("dict[str, object]", request),
            dry_run=True,
        )
    try:
        response = client.order_send(request)
    except Exception as exc:  # noqa: BLE001 - receipt is the execution contract
        return _execution_receipt(
            mt5=client.mt5,
            symbol=symbol,
            order_side=side,
            request=cast("dict[str, object]", request),
            error=exc,
        )
    return _execution_receipt(
        mt5=client.mt5,
        symbol=symbol,
        order_side=side,
        request=cast("dict[str, object]", request),
        response=response,
    )


def _filter_positions(
    positions: pd.DataFrame,
    *,
    symbols: str | list[str] | None = None,
    tickets: list[int] | None = None,
    magic: int | None = None,
) -> pd.DataFrame:
    frame = positions
    if symbols is not None:
        symbol_set = {symbols} if isinstance(symbols, str) else set(symbols)
        frame = frame.loc[frame["symbol"].isin(symbol_set)]
    if tickets is not None:
        frame = frame.loc[frame["ticket"].isin(tickets)]
    if magic is not None:
        if "magic" not in frame.columns:
            return frame.iloc[0:0].copy()
        frame = frame.loc[frame["magic"] == magic]
    return frame


def close_open_positions(
    client: _Mt5ClientProtocol,
    *,
    symbols: str | list[str] | None = None,
    tickets: list[int] | None = None,
    order_filling_mode: OrderFillingMode | None = None,
    deviation: int | None = None,
    comment: str | None = None,
    magic: int | None = None,
    dry_run: bool = False,
) -> list[OrderExecutionResult]:
    """Close matching open positions.

    When ``order_filling_mode`` is ``None``, the filling mode is resolved per
    symbol with :func:`resolve_broker_filling_mode` so closes are not rejected
    on brokers whose symbols do not support IOC.

    Returns:
        Normalized execution results for matching positions.
    """
    positions = _filter_positions(
        get_positions_frame(client),
        symbols=symbols,
        tickets=tickets,
        magic=magic,
    )
    results: list[OrderExecutionResult] = []
    resolved_filling_modes: dict[str, OrderFillingMode] = {}
    for row in positions.to_dict("records"):
        pos_type = row["type"]
        side: OrderSide = "SELL" if pos_type == client.mt5.POSITION_TYPE_BUY else "BUY"
        symbol = str(row["symbol"])
        if order_filling_mode is None:
            if symbol not in resolved_filling_modes:
                resolved_filling_modes[symbol] = resolve_broker_filling_mode(
                    client,
                    symbol=symbol,
                )
            filling_mode = resolved_filling_modes[symbol]
        else:
            filling_mode = order_filling_mode
        result = place_market_order(
            client,
            symbol=symbol,
            volume=float(row["volume"]),
            order_side=side,
            order_filling_mode=filling_mode,
            position=int(row["ticket"]),
            deviation=deviation,
            comment=comment,
            magic=magic,
            dry_run=dry_run,
        )
        results.append(result)
    return results


def _symbol_digits(client: _Mt5ClientProtocol, symbol: str) -> int | None:
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
    client: _Mt5ClientProtocol,
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
    client: _Mt5ClientProtocol,
    *,
    symbol: str,
    trailing_stop_ratio: float,
    magic: int | None = None,
) -> dict[int, float]:
    """Return per-ticket trailing stop-loss updates for open symbol positions.

    Buy positions trail from bid using ``bid * (1 - trailing_stop_ratio)``.
    Sell positions trail from ask using ``ask * (1 + trailing_stop_ratio)``.
    Existing stop losses are preserved when they are already more favorable.
    Missing symbol metadata returns an empty update map. Positions with a
    missing side-specific tick price are skipped.
    """
    _require_protective_ratio(trailing_stop_ratio, "trailing_stop_ratio")
    positions = _filter_positions(
        get_positions_frame(client, symbol=symbol), magic=magic
    )
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
    client: _Mt5ClientProtocol,
    *,
    symbol: str,
    trailing_stop_ratio: float,
    magic: int | None = None,
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
        magic=magic,
    )
    results: list[OrderExecutionResult] = []
    for ticket, stop_loss in updates.items():
        results.extend(
            update_sltp_for_open_positions(
                client,
                symbol=symbol,
                tickets=[ticket],
                stop_loss=stop_loss,
                magic=magic,
                dry_run=dry_run,
            ),
        )
    return results


def update_sltp_for_open_positions(
    client: _Mt5ClientProtocol,
    *,
    symbol: str | None = None,
    tickets: list[int] | None = None,
    magic: int | None = None,
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
        magic=magic,
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
        side: OrderSide = (
            "BUY" if row["type"] == client.mt5.POSITION_TYPE_BUY else "SELL"
        )
        if dry_run:
            results.append(
                _execution_receipt(
                    mt5=client.mt5,
                    symbol=str(row["symbol"]),
                    order_side=side,
                    request=cast("dict[str, object]", request),
                    dry_run=True,
                )
            )
            continue
        ensure_symbol_selected(client, str(row["symbol"]))
        try:
            response = client.order_send(request)
        except Exception as exc:  # noqa: BLE001 - receipt is the execution contract
            results.append(
                _execution_receipt(
                    mt5=client.mt5,
                    symbol=str(row["symbol"]),
                    order_side=side,
                    request=cast("dict[str, object]", request),
                    error=exc,
                )
            )
            continue
        results.append(
            _execution_receipt(
                mt5=client.mt5,
                symbol=str(row["symbol"]),
                order_side=side,
                request=cast("dict[str, object]", request),
                response=response,
            )
        )
    return results


def fetch_latest_closed_rates_for_trading_client(
    client: _Mt5ClientProtocol,
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
        Mt5OperationError: If the trading client cannot fetch rate data.
    """
    if count <= 0:
        msg = "count must be positive."
        raise ValueError(msg)
    timeframe = parse_timeframe(granularity)
    fetch_method = getattr(client, "fetch_latest_rates_as_df", None)
    copy_method = getattr(client, "copy_rates_from_pos_as_df", None)
    if callable(fetch_method):
        fetched = fetch_method(symbol, granularity, count + 1)
    elif callable(copy_method):
        fetched = copy_method(
            symbol=symbol, timeframe=timeframe, start_pos=0, count=count + 1
        )
    else:
        msg = "MT5 trading client cannot fetch rate data."
        raise Mt5OperationError(msg)
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
    client: _Mt5ClientProtocol,
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


def fetch_recent_history_deals_for_trading_client(
    client: MT5Client | _HistoryDealsClientProtocol,
    *,
    symbol: str | None = None,
    group: str | None = None,
    hours: float = 24.0,
    date_to: datetime | None = None,
    server_clock_offset_seconds: float | None = None,
) -> pd.DataFrame:
    """Fetch recent history deals from an already-connected trading client.

    Computes a trailing window ending at ``date_to`` (or ``datetime.now(UTC)``
    when omitted) and delegates to the client's ``history_deals_get_as_df``
    method. The object returned by :func:`create_trading_client` (a raw
    ``pdmt5.Mt5DataClient``) satisfies this protocol directly. Note that
    ``mt5cli.sdk.Mt5CliClient`` (used via ``mt5_session()``) exposes
    ``history_deals()``, not ``history_deals_get_as_df()``, and therefore does
    not satisfy this protocol; use this helper with trading-client sessions only.

    The returned DataFrame preserves every column from the underlying client
    (``time``, ``symbol``, ``type``, ``entry``, ``volume``, ``profit``,
    ``position_id``, etc.). No strategy-specific transformations are applied;
    downstream packages own entry/exit classification, Kelly fractions, and
    any other betting or signal semantics.

    MT5 deal timestamps are labeled in the broker server's wall clock, which
    is commonly UTC+2 or UTC+3 rather than true UTC, so a window computed
    from true UTC ``now`` can miss the most recent deals on such brokers.
    Pass ``server_clock_offset_seconds`` (see
    :func:`estimate_server_clock_offset_seconds`) to shift the window so it
    covers the true most-recent deals.

    Args:
        client: Connected ``pdmt5.Mt5DataClient`` (or compatible) with
            ``history_deals_get_as_df`` capability, as returned by
            :func:`create_trading_client`.
        symbol: Optional symbol filter passed to the underlying client.
        group: Optional symbol group filter passed to the underlying client.
        hours: Trailing window length in hours. Must be positive.
        date_to: Window end timestamp. Defaults to ``datetime.now(UTC)``.
        server_clock_offset_seconds: Optional broker server clock offset
            (server time minus UTC) applied to both ends of the window. Must
            be finite when provided. Defaults to ``None``, which keeps the
            window anchored to true UTC (current behavior, byte-identical
            when omitted).

    Returns:
        DataFrame ordered chronologically by ``time`` (when the column
        exists) with a ``RangeIndex``. Schema-preserving empty DataFrames
        (zero rows but columns present) are passed through with a reset
        index. Returns a bare empty DataFrame only when the underlying
        client returns ``None``.

    Raises:
        ValueError: If ``hours`` is not positive, or if
            ``server_clock_offset_seconds`` is not finite.

    Example::

        from mt5cli import (
            create_trading_client,
            estimate_server_clock_offset_seconds,
            fetch_recent_history_deals_for_trading_client,
        )

        client = create_trading_client(login=12345, server="Broker-Demo")
        try:
            offset = estimate_server_clock_offset_seconds(client, "JP225")
            deals_df = fetch_recent_history_deals_for_trading_client(
                client,
                symbol="JP225",
                hours=24,
                server_clock_offset_seconds=offset,
            )
        finally:
            client.shutdown()
    """
    if not isfinite(hours) or hours <= 0:
        msg = "hours must be finite and positive."
        raise ValueError(msg)
    if server_clock_offset_seconds is not None and not isfinite(
        server_clock_offset_seconds,
    ):
        msg = "server_clock_offset_seconds must be finite."
        raise ValueError(msg)
    end = date_to if date_to is not None else datetime.now(UTC)
    start = end - timedelta(hours=hours)
    if server_clock_offset_seconds is not None:
        offset = timedelta(seconds=server_clock_offset_seconds)
        start += offset
        end += offset
    if isinstance(client, MT5Client):
        raw = client.history_deals(
            date_from=start,
            date_to=end,
            group=group,
            symbol=symbol,
        )
    else:
        raw = client.history_deals_get_as_df(
            date_from=start,
            date_to=end,
            group=group,
            symbol=symbol,
        )
    if raw is None:
        return pd.DataFrame()
    if raw.empty:
        return raw.reset_index(drop=True)
    if "time" in raw.columns:
        raw = raw.sort_values("time")
    return raw.reset_index(drop=True)


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
) -> Iterator[MT5Client]:
    """Open a trading-capable MT5 session and always shut down safely.

    Launches the MetaTrader 5 terminal using ``Mt5Config.path`` when set,
    initializes and logs in via ``initialize_and_login_mt5()``, yields a
    connected client supporting required MT5 methods, and calls ``shutdown()``
    on exit even when an error is raised inside the context.

    Args:
        config: MT5 connection configuration. Defaults to an empty config that
            attaches to a running terminal.
        login: Optional trading account login.
        password: Optional trading account password.
        server: Optional trading server name.
        path: Optional terminal executable path.
        timeout: Optional connection timeout in milliseconds.
        retry_count: Number of initialization retries.

    Yields:
        Connected client supporting required MT5 trading methods.
    """
    _ = retry_count
    resolved = _resolve_config(
        config=config,
        login=login,
        password=password,
        server=server,
        path=path,
        timeout=timeout,
    )
    with mt5_session(resolved) as client:
        yield client
