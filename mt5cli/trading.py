"""Trading-capable MetaTrader 5 session helpers and operational utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from pdmt5 import Mt5Config, Mt5TradingClient

from .sdk import build_config

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

PositionSide = Literal["long", "short"]
OrderSide = Literal["long", "short"]

__all__ = [
    "OrderSide",
    "PositionSide",
    "calculate_margin_and_volume",
    "detect_position_side",
    "determine_order_limits",
    "mt5_trading_session",
]


def _require_unit_ratio(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be between 0 and 1 inclusive."
        raise ValueError(msg)


def _sum_position_volume(positions: pd.DataFrame, position_type: object) -> float:
    matched = positions.loc[positions["type"] == position_type, "volume"]
    if matched.empty:
        return 0.0
    return float(matched.to_numpy(dtype=float).sum())


def _normalize_order_side(side: str) -> OrderSide:
    normalized = side.lower()
    if normalized in {"long", "buy"}:
        return "long"
    if normalized in {"short", "sell"}:
        return "short"
    msg = (
        f"Unsupported order side: {side!r}. Expected 'long', 'short', 'buy', or 'sell'."
    )
    raise ValueError(msg)


def detect_position_side(
    client: Mt5TradingClient,
    symbol: str,
) -> PositionSide | None:
    """Detect the net open position side for a symbol.

    Args:
        client: Connected ``Mt5TradingClient`` instance.
        symbol: Symbol to inspect.

    Returns:
        ``"long"`` when net buy volume exceeds sell volume, ``"short"`` when
        net sell volume exceeds buy volume, or ``None`` when no positions exist
        or buy/sell volumes are exactly balanced.
    """
    positions = client.positions_get_as_df(symbol=symbol)
    if positions.empty:
        return None

    buy_type = client.mt5.POSITION_TYPE_BUY
    sell_type = client.mt5.POSITION_TYPE_SELL
    buy_volume = _sum_position_volume(positions, buy_type)
    sell_volume = _sum_position_volume(positions, sell_type)
    net_volume = buy_volume - sell_volume
    if net_volume > 0:
        return "long"
    if net_volume < 0:
        return "short"
    return None


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
        ``buy_volume``, and ``sell_volume``.
    """
    _require_unit_ratio(unit_margin_ratio, "unit_margin_ratio")
    _require_unit_ratio(preserved_margin_ratio, "preserved_margin_ratio")

    account = client.account_info_as_dict()
    margin_free = float(account.get("margin_free") or 0.0)
    available_margin = margin_free * (1.0 - preserved_margin_ratio)
    trade_margin = available_margin * unit_margin_ratio
    buy_volume = client.calculate_volume_by_margin(symbol, trade_margin, "BUY")
    sell_volume = client.calculate_volume_by_margin(symbol, trade_margin, "SELL")
    return {
        "margin_free": margin_free,
        "available_margin": available_margin,
        "trade_margin": trade_margin,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
    }


def determine_order_limits(
    client: Mt5TradingClient,
    symbol: str,
    side: OrderSide | str,
    stop_loss_limit_ratio: float,
    take_profit_limit_ratio: float,
) -> dict[str, float | None]:
    """Derive entry and protective order prices from current market quotes.

    Args:
        client: Connected ``Mt5TradingClient`` instance.
        symbol: Symbol used for the quote lookup.
        side: Position side as ``"long"``/``"short"`` (``"buy"``/``"sell"``
            aliases are accepted).
        stop_loss_limit_ratio: Relative distance from entry for stop loss. A
            value ``<= 0`` omits the stop loss.
        take_profit_limit_ratio: Relative distance from entry for take profit.
            A value ``<= 0`` omits the take profit.

    Returns:
        Dictionary with ``entry``, ``stop_loss``, and ``take_profit`` keys.
        Omitted protective levels are returned as ``None``.
    """
    normalized_side = _normalize_order_side(side)
    tick = client.symbol_info_tick_as_dict(symbol=symbol)
    entry = float(tick["ask"] if normalized_side == "long" else tick["bid"])

    stop_loss: float | None = None
    if stop_loss_limit_ratio > 0:
        if normalized_side == "long":
            stop_loss = entry * (1.0 - stop_loss_limit_ratio)
        else:
            stop_loss = entry * (1.0 + stop_loss_limit_ratio)

    take_profit: float | None = None
    if take_profit_limit_ratio > 0:
        if normalized_side == "long":
            take_profit = entry * (1.0 + take_profit_limit_ratio)
        else:
            take_profit = entry * (1.0 - take_profit_limit_ratio)

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


@contextmanager
def mt5_trading_session(
    config: Mt5Config | None = None,
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
        retry_count: Number of initialization retries passed to
            ``Mt5TradingClient``.

    Yields:
        Connected ``Mt5TradingClient`` bound to the session.
    """
    mt5_config = config or build_config()
    client = Mt5TradingClient(config=mt5_config, retry_count=retry_count)
    try:
        client.initialize_and_login_mt5()
        yield client
    finally:
        client.shutdown()
