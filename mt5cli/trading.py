"""Trading-capable MetaTrader 5 session helpers and operational utilities."""

from __future__ import annotations

from contextlib import contextmanager
from math import floor, isfinite
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd
from pdmt5 import Mt5Config, Mt5TradingClient, Mt5TradingError

from .sdk import build_config
from .utils import coerce_login as _coerce_login

if TYPE_CHECKING:
    from collections.abc import Iterator

PositionSide = Literal["long", "short"]
OrderSide = Literal["BUY", "SELL"]
OrderFillingMode = Literal["IOC", "FOK", "RETURN"]
OrderTimeMode = Literal["GTC", "DAY", "SPECIFIED", "SPECIFIED_DAY"]
_ORDER_FILLING_MODES: frozenset[str] = frozenset({"IOC", "FOK", "RETURN"})
_ORDER_TIME_MODES: frozenset[str] = frozenset({
    "GTC",
    "DAY",
    "SPECIFIED",
    "SPECIFIED_DAY",
})

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
    "OrderFillingMode",
    "OrderSide",
    "OrderTimeMode",
    "PositionSide",
    "calculate_margin_and_volume",
    "calculate_new_position_margin_ratio",
    "calculate_spread_ratio",
    "calculate_volume_by_margin",
    "close_open_positions",
    "create_trading_client",
    "detect_position_side",
    "determine_order_limits",
    "get_account_snapshot",
    "get_positions_frame",
    "get_symbol_snapshot",
    "get_tick_snapshot",
    "mt5_trading_session",
    "place_market_order",
    "update_sltp_for_open_positions",
]


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


def _optional_price(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        return None
    price = float(value)
    if price <= 0 or not isfinite(price):
        return None
    return price


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


def calculate_spread_ratio(client: Mt5TradingClient, symbol: str) -> float:
    """Return ``(ask - bid) / ((ask + bid) / 2)`` for the latest tick.

    Raises:
        Mt5TradingError: If bid or ask is unavailable or non-positive.
    """
    tick = get_tick_snapshot(client, symbol)
    bid = tick.get("bid")
    ask = tick.get("ask")
    if not isinstance(bid, int | float) or not isinstance(ask, int | float):
        msg = f"Tick bid/ask is unavailable for {symbol!r}."
        raise Mt5TradingError(msg)
    if bid <= 0 or ask <= 0:
        msg = f"Tick bid/ask must be positive for {symbol!r}."
        raise Mt5TradingError(msg)
    return (float(ask) - float(bid)) / ((float(ask) + float(bid)) / 2.0)


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
        tick = get_tick_snapshot(client, symbol)
        price = tick["ask"] if side == "BUY" else tick["bid"]
        if not isinstance(price, int | float) or price <= 0:
            msg = f"Tick price is unavailable for {symbol!r}."
            raise Mt5TradingError(msg)
        order_type = (
            client.mt5.ORDER_TYPE_BUY if side == "BUY" else client.mt5.ORDER_TYPE_SELL
        )
        margin += float(
            client.order_calc_margin(order_type, symbol, new_position_volume, price),
        )
    return margin / equity


def calculate_margin_and_volume(
    client: Mt5TradingClient,
    symbol: str,
    unit_margin_ratio: float,
    preserved_margin_ratio: float,
) -> dict[str, float]:
    """Calculate tradable margin and volumes from account free margin.

    Applies ``preserved_margin_ratio`` to keep a reserve off ``margin_free``,
    then allocates ``unit_margin_ratio`` of the remainder as the margin budget
    for volume sizing on both buy and sell sides.

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
    native_calculate_volume = getattr(client, "calculate_volume_by_margin", None)
    if callable(native_calculate_volume):
        buy_volume = float(
            cast(
                "float | int | str",
                native_calculate_volume(symbol, trade_margin, "BUY"),
            ),
        )
        sell_volume = float(
            cast(
                "float | int | str",
                native_calculate_volume(symbol, trade_margin, "SELL"),
            ),
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
        Affordable volume rounded down to symbol volume constraints.

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
    tick = get_tick_snapshot(client, symbol)
    price = tick["ask"] if side == "BUY" else tick["bid"]
    if not isinstance(price, int | float) or price <= 0:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5TradingError(msg)
    order_type = (
        client.mt5.ORDER_TYPE_BUY if side == "BUY" else client.mt5.ORDER_TYPE_SELL
    )
    min_margin = float(client.order_calc_margin(order_type, symbol, volume_min, price))
    if min_margin <= 0 or min_margin > available_margin:
        return 0.0
    raw_volume = available_margin / min_margin * volume_min
    capped = min(raw_volume, volume_max) if volume_max > 0 else raw_volume
    steps = floor(((capped - volume_min) / volume_step) + 1e-12)
    normalized = volume_min + max(0, steps) * volume_step
    return round(normalized, 10) if normalized >= volume_min else 0.0


def determine_order_limits(
    client: Mt5TradingClient,
    symbol: str,
    side: PositionSide | str,
    stop_loss_limit_ratio: float | None = None,
    take_profit_limit_ratio: float | None = None,
) -> dict[str, float | None]:
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
        Mt5TradingError: If required tick data is invalid.
    """
    stop_loss_ratio = stop_loss_limit_ratio or 0.0
    take_profit_ratio = take_profit_limit_ratio or 0.0
    _require_protective_ratio(stop_loss_ratio, "stop_loss_limit_ratio")
    _require_protective_ratio(take_profit_ratio, "take_profit_limit_ratio")
    normalized_side = _position_side_from_order_side(side)
    tick = get_tick_snapshot(client, symbol)
    entry_value = tick["ask"] if normalized_side == "long" else tick["bid"]
    if not isinstance(entry_value, int | float):
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5TradingError(msg)
    entry = float(entry_value)
    try:
        digits = int(get_symbol_snapshot(client, symbol).get("digits") or 8)
    except AttributeError:
        digits = 8

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
) -> dict[str, object]:
    """Place one normalized market order or return a dry-run result.

    Returns:
        Normalized execution result containing request and response details.

    Raises:
        Mt5TradingError: If volume or required tick data is invalid.
    """
    if volume <= 0:
        msg = "volume must be positive."
        raise Mt5TradingError(msg)
    side = _normalize_order_side(order_side)
    tick = get_tick_snapshot(client, symbol)
    price = tick["ask"] if side == "BUY" else tick["bid"]
    if not isinstance(price, int | float) or price <= 0:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5TradingError(msg)
    request = {
        "action": client.mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": (
            client.mt5.ORDER_TYPE_BUY if side == "BUY" else client.mt5.ORDER_TYPE_SELL
        ),
        "price": float(price),
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
            "request": request,
            "response": None,
            "dry_run": True,
        }
    response = client.order_send(request)
    response_dict = _snapshot_from_value(response, ())
    return {
        "status": "executed",
        "symbol": symbol,
        "order_side": side,
        "volume": volume,
        "retcode": response_dict.get("retcode"),
        "comment": response_dict.get("comment"),
        "request": request,
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
) -> list[dict[str, object]]:
    """Close matching open positions.

    Returns:
        Normalized execution results for matching positions.
    """
    positions = _filter_positions(
        get_positions_frame(client),
        symbols=symbols,
        tickets=tickets,
    )
    results: list[dict[str, object]] = []
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


def update_sltp_for_open_positions(
    client: Mt5TradingClient,
    *,
    symbol: str | None = None,
    tickets: list[int] | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    """Update SL/TP for matching open positions.

    Returns:
        Normalized execution results for matching positions.
    """
    positions = _filter_positions(
        get_positions_frame(client),
        symbols=symbol,
        tickets=tickets,
    )
    results: list[dict[str, object]] = []
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
            status = "dry_run"
        else:
            response = _snapshot_from_value(client.order_send(request), ())
            status = "executed"
        results.append(
            {
                "status": status,
                "symbol": row["symbol"],
                "order_side": "BUY"
                if row["type"] == client.mt5.POSITION_TYPE_BUY
                else "SELL",
                "volume": row["volume"],
                "retcode": None if response is None else response.get("retcode"),
                "comment": None if response is None else response.get("comment"),
                "request": request,
                "response": response,
                "dry_run": dry_run,
            },
        )
    return results


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
