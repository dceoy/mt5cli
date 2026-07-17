"""Trading-capable MetaTrader 5 session helpers and operational utilities."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from inspect import signature
from math import floor, isfinite
from numbers import Integral, Real
from operator import itemgetter
from time import sleep
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, cast

import pandas as pd
from pdmt5 import Mt5RuntimeError

from .exceptions import Mt5OperationError
from .history import drop_forming_rate_bar

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

_logger = logging.getLogger(__name__)
_DATETIME_TYPES = (datetime, pd.Timestamp)


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

    def copy_ticks_range(
        self,
        symbol: str,
        date_from: datetime | str,
        date_to: datetime | str,
        flags: int | str,
        /,
    ) -> pd.DataFrame:
        """Return UTC-labeled copied ticks for a date range."""
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
ClockStatus = Literal["calibrated", "uncalibrated"]
CalibrationStatus = Literal[
    "calibrated",
    "no_live_tick",
    "no_copied_ticks",
    "no_matching_event",
    "insufficient_agreement",
    "offset_disagreement",
    "implausible_offset",
]


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


class NormalizedTickSnapshot(TypedDict):
    """Latest tick with an explicit raw-vs-UTC timestamp distinction.

    ``raw_time`` preserves the numeric MT5 epoch exactly as labeled by the
    broker server. ``time_utc`` is populated only when a validated server
    clock offset could be applied (``clock_status == "calibrated"``);
    otherwise ``time_utc`` and ``server_clock_offset_seconds`` are ``None``
    and callers must not treat ``raw_time`` as UTC.
    """

    symbol: str
    bid: float | int | None
    ask: float | int | None
    last: float | int | None
    volume: float | int | None
    raw_time: float | int | None
    time_utc: datetime | None
    server_clock_offset_seconds: float | None
    clock_status: ClockStatus


@dataclass(frozen=True)
class TickClockCalibration:
    """Diagnostic record of one server-clock calibration attempt.

    ``offset_seconds`` is ``server labeled time - true UTC`` (``10800.0`` for
    a UTC+3 server wall clock), rounded to 30-minute increments, and is only
    set when ``status == "calibrated"``. ``sample_count`` counts distinct
    matched live/copied tick events that agreed on the offset, and
    ``evidence_symbols`` names the symbols whose copied UTC ticks provided
    that evidence. ``calibrated_at`` is the UTC epoch when the calibration
    was accepted.
    """

    status: CalibrationStatus
    offset_seconds: float | None
    sample_count: int
    evidence_symbols: tuple[str, ...]
    calibrated_at: float | None

    @property
    def calibrated(self) -> bool:
        """Whether this calibration produced a usable offset."""
        return self.status == "calibrated"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of this calibration."""
        return cast("dict[str, object]", asdict(self))


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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of this receipt."""
        return cast("dict[str, object]", asdict(self))


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
    "CalibrationStatus",
    "ClockStatus",
    "ExecutionStatus",
    "MarginVolume",
    "NormalizedTickSnapshot",
    "OrderExecutionResult",
    "OrderFillingMode",
    "OrderLimits",
    "OrderSide",
    "OrderTimeMode",
    "PositionSide",
    "ProjectionMode",
    "TickClockCalibration",
    "TickClockNormalizer",
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
    "detect_position_side",
    "determine_order_limits",
    "ensure_symbol_selected",
    "estimate_order_margin",
    "extract_tick_price",
    "fetch_latest_closed_rates_indexed",
    "get_account_snapshot",
    "get_positions_frame",
    "get_symbol_snapshot",
    "get_tick_snapshot",
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


def _require_tick_price(price: float | None, symbol: str) -> float:
    """Return a broker quote or raise the normalized missing-quote error.

    Raises:
        Mt5OperationError: If the quote is unavailable.
    """
    if price is None:
        msg = f"Tick price is unavailable for {symbol!r}."
        raise Mt5OperationError(msg)
    return price


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
    _ = allowed
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


def _optional_positive_int(value: object) -> int | None:
    """Return a positive integer identifier, or None for absent sentinels.

    Brokers use ``0`` (and occasionally negative values) as "no identifier"
    sentinels for order, deal, and position tickets. Those must not surface
    as valid identifiers on the normalized receipt.
    """
    parsed = _optional_int(value)
    return parsed if parsed is not None and parsed > 0 else None


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
    if isinstance(value, float) and not isfinite(value):
        return None
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
    if error is not None:
        resolved_status: ExecutionStatus = "failed"
    elif dry_run:
        resolved_status = "dry_run"
        normalized_response = None
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
        else _optional_positive_int(normalized_response.get("order")),
        deal_ticket=None
        if normalized_response is None
        else _optional_positive_int(normalized_response.get("deal")),
        position_id=(
            _optional_positive_int(request.get("position"))
            if normalized_response is None
            else _optional_positive_int(normalized_response.get("position"))
            or _optional_positive_int(request.get("position"))
        ),
        magic=(
            _optional_int(request.get("magic"))
            if normalized_response is None
            else (
                response_magic
                if (response_magic := _optional_int(normalized_response.get("magic")))
                is not None
                else _optional_int(request.get("magic"))
            )
        ),
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
    """Return normalized latest tick data with a numeric MT5 timestamp.

    The numeric ``time`` preserves the original value returned by MT5 without
    applying timezone or server-offset correction. Use
    :class:`TickClockNormalizer` when a validated UTC timestamp is required.
    """
    snapshot = _snapshot_from_value(
        _raw_tick_value(client, symbol),
        _TICK_SNAPSHOT_FIELDS,
    )
    snapshot["symbol"] = snapshot.get("symbol") or symbol
    snapshot["time"] = _numeric_tick_time(snapshot.get("time"))
    return cast("dict[str, float | int | None]", snapshot)


def _raw_tick_value(client: _Mt5ClientProtocol, symbol: str) -> object:
    """Fetch the latest raw tick value, preferring the dict-based accessor.

    Returns:
        The raw broker tick value (mapping, named tuple, or DataFrame).
    """
    method = getattr(client, "symbol_info_tick_as_dict", None)
    if callable(method):
        try:
            signature(method).bind(symbol=symbol, skip_to_datetime=True)
        except (TypeError, ValueError):
            return method(symbol=symbol)
        return method(symbol=symbol, skip_to_datetime=True)
    return client.symbol_info_tick(symbol)


def _numeric_tick_time(value: object) -> float | int | None:
    """Normalize an MT5 timestamp to its numeric epoch representation.

    Returns:
        The numeric timestamp, or ``None`` for an unsupported value.
    """
    numeric: float | None
    if value is None or isinstance(value, bool):
        numeric = None
    elif isinstance(value, _DATETIME_TYPES):
        if pd.isna(value):
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(UTC)
        numeric = timestamp.timestamp()
    elif isinstance(value, Integral):
        return int(value)
    elif isinstance(value, Real):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
    else:
        numeric = None
    return numeric if numeric is not None and isfinite(numeric) else None


_SERVER_CLOCK_OFFSET_ROUNDING_SECONDS = 1800.0
_MAX_PLAUSIBLE_SERVER_CLOCK_OFFSET_SECONDS = 14 * 3600.0


_CALIBRATION_TICK_FIELDS = (
    "symbol",
    "time",
    "time_msc",
    "bid",
    "ask",
    "last",
    "volume",
)
_TICK_MATCH_PRICE_FIELDS = ("bid", "ask", "last")
_PRICE_MATCH_RELATIVE_TOLERANCE = 1e-9
_OFFSET_RESIDUAL_TOLERANCE_SECONDS = 5.0
_MAX_FUTURE_SKEW_SECONDS = 120.0
_COPY_TICKS_ALL_FALLBACK = -1


@dataclass(frozen=True)
class _OffsetSample:
    """One live-vs-copied tick comparison attempt."""

    offset_seconds: float | None
    reason: CalibrationStatus
    live_key: tuple[object, ...] | None


_CALIBRATION_FAILURE_PRIORITY: tuple[CalibrationStatus, ...] = (
    "implausible_offset",
    "no_matching_event",
    "no_copied_ticks",
    "no_live_tick",
)


def _aggregate_failure_status(
    failures: list[CalibrationStatus],
) -> CalibrationStatus:
    """Pick the most informative failure reason from all rejected samples.

    Returns:
        The highest-priority observed failure reason, or ``no_live_tick``
        when no sample produced a reason at all.
    """
    for reason in _CALIBRATION_FAILURE_PRIORITY:
        if reason in failures:
            return reason
    return "no_live_tick"


def _tick_event_epoch(tick: Mapping[str, object]) -> float | None:
    """Return the event epoch in seconds, preferring millisecond precision.

    Returns:
        UTC-scale epoch seconds as labeled by the source, or ``None`` when
        neither ``time_msc`` nor ``time`` holds a positive numeric value.
    """
    msc = tick.get("time_msc")
    if isinstance(msc, _DATETIME_TYPES):
        epoch = _numeric_tick_time(msc)
        if epoch is not None and epoch > 0:
            return float(epoch)
    else:
        epoch = _numeric_tick_time(msc)
        if epoch is not None and epoch > 0:
            return float(epoch) / 1000.0
    epoch = _numeric_tick_time(tick.get("time"))
    if epoch is not None and epoch > 0:
        return float(epoch)
    return None


def _match_value(value: object) -> float | None:
    """Return a finite float for tick-field comparison, or None if absent."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Integral | Real):
        numeric = float(value)
        return numeric if isfinite(numeric) else None
    return None


def _ticks_match(live: Mapping[str, object], copied: Mapping[str, object]) -> bool:
    """Decide whether a live tick and a copied tick are the same market event.

    Every field present on both sides (``bid``, ``ask``, ``last``,
    ``volume``) must agree, and at least one positive price field must supply
    evidence; volume agreement or all-zero prices alone are insufficient.

    Returns:
        True when the two ticks plausibly describe the same market event.
    """
    matched_price_field = False
    for field in (*_TICK_MATCH_PRICE_FIELDS, "volume"):
        live_value = _match_value(live.get(field))
        copied_value = _match_value(copied.get(field))
        if live_value is None or copied_value is None:
            continue
        tolerance = _PRICE_MATCH_RELATIVE_TOLERANCE * max(
            1.0,
            abs(live_value),
            abs(copied_value),
        )
        if abs(live_value - copied_value) > tolerance:
            return False
        if field in _TICK_MATCH_PRICE_FIELDS and live_value > 0:
            matched_price_field = True
    return matched_price_field


def _recent_copied_ticks(
    client: _Mt5ClientProtocol,
    symbol: str,
    *,
    window_seconds: float,
    now_epoch: float,
) -> pd.DataFrame | None:
    """Fetch copied ticks for the trailing UTC window ending now.

    Returns:
        The copied-tick DataFrame, or ``None`` when retrieval fails or the
        client returns malformed (non-DataFrame) data.
    """
    flags = getattr(getattr(client, "mt5", None), "COPY_TICKS_ALL", None)
    if not isinstance(flags, int):
        flags = _COPY_TICKS_ALL_FALLBACK
    try:
        frame = cast(
            "object",
            client.copy_ticks_range(
                symbol,
                datetime.fromtimestamp(now_epoch - window_seconds, tz=UTC),
                datetime.fromtimestamp(now_epoch, tz=UTC),
                flags,
            ),
        )
    except (Mt5RuntimeError, Mt5OperationError) as exc:
        _logger.warning("Copied-tick retrieval failed for %s: %s", symbol, exc)
        return None
    return frame if isinstance(frame, pd.DataFrame) else None


def _usable_copied_ticks(
    frame: pd.DataFrame | None,
) -> list[tuple[dict[str, object], float]]:
    """Return copied rows that carry a usable event epoch.

    Returns:
        ``(row, epoch_seconds)`` pairs for every row with a positive
        timestamp, or an empty list when the frame is missing, empty, or has
        no row with a usable timestamp.
    """
    if frame is None or frame.empty:
        return []
    return [
        (row, epoch)
        for row in cast("list[dict[str, object]]", frame.to_dict("records"))
        if (epoch := _tick_event_epoch(row)) is not None
    ]


def _matching_ticks_newest_first(
    candidates: list[tuple[dict[str, object], float]],
    live: Mapping[str, object],
) -> list[tuple[dict[str, object], float]]:
    """Return every candidate matching the live tick's fields, newest first.

    An actively updating symbol can produce a newer copied tick while the
    live-vs-copied comparison is in flight, so every recent copied row is
    searched for the same market event as ``live`` instead of assuming the
    newest copied row is the relevant one. A newer row can also
    coincidentally share ``live``'s price/volume fields without being the
    same event, so every match is returned for offset-plausibility
    evaluation rather than only the newest.

    Returns:
        ``(row, epoch_seconds)`` pairs from ``candidates`` that match
        ``live``, ordered from most to least recent.
    """
    matches = [item for item in candidates if _ticks_match(live, item[0])]
    matches.sort(key=itemgetter(1), reverse=True)
    return matches


def _sample_clock_offset(
    client: _Mt5ClientProtocol,
    symbol: str,
    *,
    window_seconds: float,
) -> _OffsetSample:
    """Compare one live tick against recent copied UTC ticks.

    Every field-matching copied tick is evaluated, newest first, so a newer
    row that coincidentally shares the live tick's price/volume fields
    cannot shadow a plausible offset from an older exact match; recency
    only breaks ties among candidates that are themselves plausible.

    Returns:
        An :class:`_OffsetSample` whose ``offset_seconds`` is set only when
        a matching copied tick yields a plausible half-hour-aligned server
        clock offset.
    """
    live = _snapshot_from_value(
        _raw_tick_value(client, symbol),
        _CALIBRATION_TICK_FIELDS,
    )
    live_epoch = _tick_event_epoch(live)
    if live_epoch is None:
        return _OffsetSample(None, "no_live_tick", None)
    now_epoch = datetime.now(UTC).timestamp()
    candidates = _usable_copied_ticks(
        _recent_copied_ticks(
            client,
            symbol,
            window_seconds=window_seconds,
            now_epoch=now_epoch,
        ),
    )
    if not candidates:
        return _OffsetSample(None, "no_copied_ticks", None)
    matches = _matching_ticks_newest_first(candidates, live)
    if not matches:
        return _OffsetSample(None, "no_matching_event", None)
    saw_implausible = False
    for _, copied_epoch in matches:
        raw_offset = live_epoch - copied_epoch
        rounded = (
            round(raw_offset / _SERVER_CLOCK_OFFSET_ROUNDING_SECONDS)
            * _SERVER_CLOCK_OFFSET_ROUNDING_SECONDS
        )
        if abs(raw_offset - rounded) > _OFFSET_RESIDUAL_TOLERANCE_SECONDS:
            continue
        if abs(rounded) > _MAX_PLAUSIBLE_SERVER_CLOCK_OFFSET_SECONDS:
            saw_implausible = True
            continue
        live_key = (symbol, live_epoch, live.get("bid"), live.get("ask"))
        return _OffsetSample(float(rounded), "calibrated", live_key)
    return _OffsetSample(
        None,
        "implausible_offset" if saw_implausible else "no_matching_event",
        None,
    )


class TickClockNormalizer:
    """Connection-scoped UTC normalization for live MT5 tick timestamps.

    ``symbol_info_tick()`` timestamps may carry a broker server wall-clock
    label (for example UTC+2/UTC+3 on OANDA-style servers) instead of true
    UTC. This normalizer calibrates that offset by matching live ticks
    against recent UTC-labeled ``copy_ticks_range()`` data, requires repeated
    agreement across distinct tick events (and/or symbols), rounds accepted
    offsets to 30-minute increments within a plausible UTC offset range, and
    caches the result per client connection.

    A cached calibration is revalidated after ``max_calibration_age_seconds``
    (covering DST transitions on the broker side), immediately when a
    normalized timestamp lands implausibly far in the future (evidence the
    broker offset grew), and periodically via one confirming sample well
    before the cache would otherwise expire (evidence the broker offset
    shrank, which a future-skew check alone cannot catch since it looks like
    ordinary staleness rather than an anomaly). Failed calibrations are
    retried at most once per ``failed_calibration_retry_seconds`` rather than
    on every call. When no offset can be established safely, normalized
    snapshots fail closed with ``clock_status="uncalibrated"`` and
    ``time_utc=None`` instead of treating the raw timestamp as UTC.

    Keep one instance per MT5 connection/account; the calibration is a
    property of the broker server, not of an individual symbol.
    """

    def __init__(
        self,
        client: _Mt5ClientProtocol,
        symbols: Sequence[str] | None = None,
        *,
        samples_per_symbol: int = 3,
        min_agreeing_samples: int = 2,
        sample_interval_seconds: float = 1.0,
        copied_window_seconds: float = 300.0,
        max_calibration_age_seconds: float = 6 * 3600.0,
        revalidation_interval_seconds: float = 300.0,
        failed_calibration_retry_seconds: float = 30.0,
    ) -> None:
        """Initialize a normalizer bound to one connected MT5 client.

        Args:
            client: Connected MT5 client exposing ``symbol_info_tick`` and
                ``copy_ticks_range``.
            symbols: Optional default calibration symbols. Prefer several
                actively updating symbols; when omitted, calibration uses the
                symbol passed to :meth:`get_normalized_tick_snapshot`.
            samples_per_symbol: Live/copied comparisons attempted per symbol
                during one calibration.
            min_agreeing_samples: Distinct matched tick events that must agree
                on the same rounded offset before it is accepted.
            sample_interval_seconds: Pause between consecutive samples so an
                actively updating symbol can produce distinct tick events.
            copied_window_seconds: Trailing UTC window queried from
                ``copy_ticks_range`` per sample.
            max_calibration_age_seconds: Age after which a cached calibration
                is recomputed unconditionally (bounds DST-transition
                staleness).
            revalidation_interval_seconds: Minimum time between opportunistic
                single-sample checks of an otherwise still-fresh cached
                calibration. A disagreeing sample forces full recalibration,
                which catches an offset decrease (for example a UTC+3 to
                UTC+2 transition) well before ``max_calibration_age_seconds``
                would; an inconclusive or agreeing sample keeps the cached
                calibration.
            failed_calibration_retry_seconds: Minimum time between full
                recalibration attempts after a failed calibration, so a
                closed or illiquid market does not retry on every call.

        Raises:
            ValueError: If a numeric tuning parameter is not positive.
        """
        if samples_per_symbol < 1 or min_agreeing_samples < 1:
            msg = "samples_per_symbol and min_agreeing_samples must be >= 1."
            raise ValueError(msg)
        if sample_interval_seconds < 0:
            msg = "sample_interval_seconds must not be negative."
            raise ValueError(msg)
        if (
            copied_window_seconds <= 0
            or max_calibration_age_seconds <= 0
            or revalidation_interval_seconds <= 0
            or failed_calibration_retry_seconds <= 0
        ):
            msg = (
                "copied_window_seconds, max_calibration_age_seconds,"
                " revalidation_interval_seconds, and"
                " failed_calibration_retry_seconds must be positive."
            )
            raise ValueError(msg)
        self._client = client
        self._symbols = tuple(symbols) if symbols else ()
        self._samples_per_symbol = samples_per_symbol
        self._min_agreeing_samples = min_agreeing_samples
        self._sample_interval_seconds = sample_interval_seconds
        self._copied_window_seconds = copied_window_seconds
        self._max_calibration_age_seconds = max_calibration_age_seconds
        self._revalidation_interval_seconds = revalidation_interval_seconds
        self._failed_calibration_retry_seconds = failed_calibration_retry_seconds
        self._calibration: TickClockCalibration | None = None
        self._last_attempt_at: float | None = None

    @property
    def calibration(self) -> TickClockCalibration | None:
        """The most recent calibration attempt, or None before the first."""
        return self._calibration

    def invalidate(self) -> None:
        """Drop the cached calibration so the next call recalibrates."""
        self._calibration = None

    def calibrate(
        self,
        symbols: Sequence[str] | None = None,
    ) -> TickClockCalibration:
        """Measure the server clock offset from live-vs-copied tick evidence.

        Args:
            symbols: Symbols to sample; defaults to the constructor symbols.

        Returns:
            The calibration result, also cached on this normalizer. Failed
            calibrations are recorded for diagnostics but never reused.

        Raises:
            ValueError: If no calibration symbol is available.
        """
        resolved = tuple(symbols) if symbols is not None else self._symbols
        if not resolved:
            msg = "At least one symbol is required for tick clock calibration."
            raise ValueError(msg)
        matched, failures = self._collect_offset_samples(resolved)
        calibration = self._build_calibration(matched, failures)
        self._calibration = calibration
        self._last_attempt_at = datetime.now(UTC).timestamp()
        if calibration.calibrated:
            _logger.info(
                "Calibrated MT5 server clock offset: %.0f seconds"
                " (%d samples from %s).",
                calibration.offset_seconds,
                calibration.sample_count,
                ", ".join(calibration.evidence_symbols),
            )
        else:
            _logger.warning(
                "MT5 server clock calibration failed for %s: %s.",
                ", ".join(resolved),
                calibration.status,
            )
        return calibration

    def get_normalized_tick_snapshot(self, symbol: str) -> NormalizedTickSnapshot:
        """Return the latest tick with a validated UTC timestamp when possible.

        The raw fields mirror :func:`get_tick_snapshot`, fetched only after
        calibration is confirmed or refreshed; an initial or expired
        calibration takes multiple paced samples, and fetching the reported
        tick beforehand would report a snapshot that was already stale by the
        time calibration finished. ``time_utc`` is set only under a currently
        valid calibration; when the normalized time would land implausibly
        far in the future (evidence that the broker offset changed, e.g. a
        DST transition) the offset is recalibrated once, the tick is
        refetched, and the snapshot fails closed if it still cannot be
        validated.

        Args:
            symbol: Symbol whose latest tick is normalized.

        Returns:
            A :class:`NormalizedTickSnapshot`; on any calibration failure the
            snapshot carries ``clock_status="uncalibrated"``, a ``None``
            ``time_utc``, and a ``None`` offset while preserving raw fields.
        """
        now_epoch = datetime.now(UTC).timestamp()
        calibration = self._current_calibration(symbol, now_epoch=now_epoch)
        snapshot = get_tick_snapshot(self._client, symbol)
        raw_time = snapshot.get("time")
        normalized_epoch = _normalized_epoch(raw_time, calibration)
        if (
            normalized_epoch is not None
            and normalized_epoch - now_epoch > _MAX_FUTURE_SKEW_SECONDS
        ):
            _logger.warning(
                "Normalized tick time for %s is %.0f seconds in the future;"
                " recalibrating the MT5 server clock offset.",
                symbol,
                normalized_epoch - now_epoch,
            )
            calibration = self.calibrate(self._symbols or (symbol,))
            snapshot = get_tick_snapshot(self._client, symbol)
            raw_time = snapshot.get("time")
            now_epoch = datetime.now(UTC).timestamp()
            normalized_epoch = _normalized_epoch(raw_time, calibration)
            if (
                normalized_epoch is not None
                and normalized_epoch - now_epoch > _MAX_FUTURE_SKEW_SECONDS
            ):
                normalized_epoch = None
        if normalized_epoch is None:
            time_utc = None
            offset_seconds = None
        else:
            time_utc = datetime.fromtimestamp(normalized_epoch, tz=UTC)
            offset_seconds = calibration.offset_seconds
        return NormalizedTickSnapshot(
            symbol=str(snapshot.get("symbol") or symbol),
            bid=snapshot.get("bid"),
            ask=snapshot.get("ask"),
            last=snapshot.get("last"),
            volume=snapshot.get("volume"),
            raw_time=raw_time,
            time_utc=time_utc,
            server_clock_offset_seconds=offset_seconds,
            clock_status="calibrated" if time_utc is not None else "uncalibrated",
        )

    def _current_calibration(
        self,
        symbol: str,
        *,
        now_epoch: float,
    ) -> TickClockCalibration:
        cached = self._calibration
        if cached is not None and not cached.calibrated:
            if (
                self._last_attempt_at is not None
                and now_epoch - self._last_attempt_at
                < self._failed_calibration_retry_seconds
            ):
                return cached
            return self.calibrate(self._symbols or (symbol,))
        if (
            cached is None
            or cached.calibrated_at is None
            or now_epoch - cached.calibrated_at > self._max_calibration_age_seconds
        ):
            return self.calibrate(self._symbols or (symbol,))
        if (
            self._last_attempt_at is not None
            and now_epoch - self._last_attempt_at < self._revalidation_interval_seconds
        ):
            return cached
        revalidated = self._revalidate(cached, symbol, now_epoch=now_epoch)
        return revalidated if revalidated is not None else cached

    def _revalidate(
        self,
        cached: TickClockCalibration,
        symbol: str,
        *,
        now_epoch: float,
    ) -> TickClockCalibration | None:
        """Confirm or replace a still-fresh cached calibration with one sample.

        A future-skew check alone never catches a broker offset *decrease*
        (e.g. a UTC+3 to UTC+2 transition): the resulting normalized time
        looks stale rather than future, and staleness is also the expected
        symptom of a quiet market. This periodic single-sample check is
        symmetric evidence that also detects that case. An inconclusive
        sample (closed market, no matching event) leaves the cached
        calibration in place rather than discarding a still-working offset.

        Returns:
            A freshly recalibrated :class:`TickClockCalibration` when the
            fresh sample disagreed with ``cached``, or ``None`` when the
            cached calibration should be kept as-is.
        """
        sample = _sample_clock_offset(
            self._client,
            symbol,
            window_seconds=self._copied_window_seconds,
        )
        self._last_attempt_at = now_epoch
        if (
            sample.offset_seconds is None
            or sample.offset_seconds == cached.offset_seconds
        ):
            return None
        _logger.warning(
            "MT5 server clock offset for %s appears to have changed from"
            " %.0f to %.0f seconds; recalibrating.",
            symbol,
            cached.offset_seconds,
            sample.offset_seconds,
        )
        return self.calibrate(self._symbols or (symbol,))

    def _collect_offset_samples(
        self,
        symbols: tuple[str, ...],
    ) -> tuple[dict[tuple[object, ...], tuple[float, str]], list[CalibrationStatus]]:
        matched: dict[tuple[object, ...], tuple[float, str]] = {}
        failures: list[CalibrationStatus] = []
        for symbol in symbols:
            first_sample = True
            for _ in range(self._samples_per_symbol):
                if not first_sample and self._sample_interval_seconds > 0:
                    sleep(self._sample_interval_seconds)
                first_sample = False
                sample = _sample_clock_offset(
                    self._client,
                    symbol,
                    window_seconds=self._copied_window_seconds,
                )
                if sample.offset_seconds is None or sample.live_key is None:
                    failures.append(sample.reason)
                    continue
                matched[sample.live_key] = (sample.offset_seconds, symbol)
                if len({offset for offset, _ in matched.values()}) > 1:
                    return matched, failures
        return matched, failures

    def _build_calibration(
        self,
        matched: dict[tuple[object, ...], tuple[float, str]],
        failures: list[CalibrationStatus],
    ) -> TickClockCalibration:
        offsets = {offset for offset, _ in matched.values()}
        evidence = tuple(sorted({symbol for _, symbol in matched.values()}))
        if len(offsets) > 1:
            return TickClockCalibration(
                status="offset_disagreement",
                offset_seconds=None,
                sample_count=len(matched),
                evidence_symbols=evidence,
                calibrated_at=None,
            )
        if not matched:
            return TickClockCalibration(
                status=_aggregate_failure_status(failures),
                offset_seconds=None,
                sample_count=0,
                evidence_symbols=(),
                calibrated_at=None,
            )
        if len(matched) < self._min_agreeing_samples:
            return TickClockCalibration(
                status="insufficient_agreement",
                offset_seconds=None,
                sample_count=len(matched),
                evidence_symbols=evidence,
                calibrated_at=None,
            )
        return TickClockCalibration(
            status="calibrated",
            offset_seconds=next(iter(offsets)),
            sample_count=len(matched),
            evidence_symbols=evidence,
            calibrated_at=datetime.now(UTC).timestamp(),
        )


def _normalized_epoch(
    raw_time: float | None,
    calibration: TickClockCalibration,
) -> float | None:
    """Apply a calibrated offset to a raw tick epoch.

    Returns:
        ``raw_time - offset`` in UTC epoch seconds, or ``None`` when the
        calibration is unusable or the raw time is missing.
    """
    if not calibration.calibrated or calibration.offset_seconds is None:
        return None
    if raw_time is None:
        return None
    return float(raw_time) - calibration.offset_seconds


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


def _rate_time_to_utc(series: pd.Series, symbol: str) -> pd.DatetimeIndex:
    """Convert MT5 epoch or datetime-like rate timestamps to UTC.

    Returns:
        UTC-aware timestamps.

    Raises:
        ValueError: If timestamp data cannot be normalized.
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
        ) and not pd.api.types.is_bool_dtype(series)
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

    Missing or falsy account ``equity``/``margin`` values are normalized to
    ``0.0``, and the hypothetical order margin from ``order_calc_margin()`` is
    accepted as-is without a positivity check.

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
        price = _require_tick_price(price, symbol)
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

    A dry run reads the current side-appropriate quote into
    ``request["price"]`` and returns a ``status="dry_run"`` receipt without
    selecting the symbol or sending an order. Live orders return
    ``status="rejected"`` for a valid non-success retcode,
    ``status="malformed"`` when the broker response has no valid retcode, and
    ``status="failed"`` when preparation or submission raises; the normalized
    request and response stay available for inspection.

    Returns:
        Normalized execution result containing request and response details.

    Raises:
        Mt5OperationError: If volume is invalid.
        ValueError: If an order mode is unsupported.
    """
    if volume <= 0:
        msg = "volume must be positive."
        raise Mt5OperationError(msg)
    side = _normalize_order_side(order_side)
    if order_filling_mode not in _ORDER_FILLING_MODES:
        msg = f"Unsupported order_filling mode: {order_filling_mode!r}."
        raise ValueError(msg)
    if order_time_mode not in _ORDER_TIME_MODES:
        msg = f"Unsupported order_time mode: {order_time_mode!r}."
        raise ValueError(msg)
    request: dict[str, object] = {
        "symbol": symbol,
        "volume": volume,
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
    mt5: Any = object()
    try:
        mt5 = client.mt5
        request.update({
            "action": mt5.TRADE_ACTION_DEAL,
            "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
            "type_filling": _resolve_mt5_constant(
                mt5, "ORDER_FILLING", order_filling_mode, _ORDER_FILLING_MODES
            ),
            "type_time": _resolve_mt5_constant(
                mt5, "ORDER_TIME", order_time_mode, _ORDER_TIME_MODES
            ),
        })
        if dry_run:
            tick = get_tick_snapshot(client, symbol)
            price = extract_tick_price(tick, "ask" if side == "BUY" else "bid")
            request["price"] = _require_tick_price(price, symbol)
            return _execution_receipt(
                mt5=mt5,
                symbol=symbol,
                order_side=side,
                request=request,
                dry_run=True,
            )
        ensure_symbol_selected(client, symbol)
        tick = get_tick_snapshot(client, symbol)
        price = extract_tick_price(tick, "ask" if side == "BUY" else "bid")
        request["price"] = _require_tick_price(price, symbol)
        response = client.order_send(request)
    except Exception as exc:  # noqa: BLE001 - receipt is the execution contract
        return _execution_receipt(
            mt5=mt5,
            symbol=symbol,
            order_side=side,
            request=request,
            dry_run=dry_run,
            error=exc,
        )
    return _execution_receipt(
        mt5=mt5,
        symbol=symbol,
        order_side=side,
        request=request,
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
        symbol = str(row["symbol"])
        side: OrderSide = "BUY"
        mt5: Any = object()
        try:
            mt5 = client.mt5
            side = "SELL" if pos_type == mt5.POSITION_TYPE_BUY else "BUY"
            if order_filling_mode is None:
                if symbol not in resolved_filling_modes:
                    resolved_filling_modes[symbol] = resolve_broker_filling_mode(
                        client,
                        symbol=symbol,
                    )
                filling_mode = resolved_filling_modes[symbol]
            else:
                filling_mode = order_filling_mode
        except Exception as exc:  # noqa: BLE001 - receipt is the execution contract
            results.append(
                _execution_receipt(
                    mt5=mt5,
                    symbol=symbol,
                    order_side=side,
                    request={
                        "symbol": symbol,
                        "volume": float(row["volume"]),
                        "position": int(row["ticket"]),
                        **({"magic": magic} if magic is not None else {}),
                    },
                    dry_run=dry_run,
                    error=exc,
                )
            )
            continue
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
        symbol = str(row["symbol"])
        side: OrderSide = "BUY"
        mt5: Any = object()
        request: dict[str, object] = {}
        try:
            mt5 = client.mt5
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": row["symbol"],
                "position": row["ticket"],
            }
            sl = _optional_price(row.get("sl") if stop_loss is None else stop_loss)
            tp = _optional_price(row.get("tp") if take_profit is None else take_profit)
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            side = "BUY" if row["type"] == mt5.POSITION_TYPE_BUY else "SELL"
        except Exception as exc:  # noqa: BLE001 - receipt is the execution contract
            results.append(
                _execution_receipt(
                    mt5=mt5,
                    symbol=symbol,
                    order_side=side,
                    request=request,
                    dry_run=dry_run,
                    error=exc,
                )
            )
            continue
        if dry_run:
            results.append(
                _execution_receipt(
                    mt5=mt5,
                    symbol=symbol,
                    order_side=side,
                    request=request,
                    dry_run=True,
                )
            )
            continue
        try:
            ensure_symbol_selected(client, symbol)
            response = client.order_send(request)
        except Exception as exc:  # noqa: BLE001 - receipt is the execution contract
            results.append(
                _execution_receipt(
                    mt5=mt5,
                    symbol=symbol,
                    order_side=side,
                    request=request,
                    error=exc,
                )
            )
            continue
        results.append(
            _execution_receipt(
                mt5=mt5,
                symbol=symbol,
                order_side=side,
                request=request,
                response=response,
            )
        )
    return results


def fetch_latest_closed_rates_indexed(
    client: _Mt5ClientProtocol,
    *,
    symbol: str,
    granularity: str,
    count: int,
) -> pd.DataFrame:
    """Fetch the latest closed bars with a UTC DatetimeIndex from a client.

    Fetches one additional current bar, removes it, then converts the ``time``
    column to a UTC-aware :class:`~pandas.DatetimeIndex` named ``"time"`` and
    drops the original column. Intended for downstream time-series consumers
    that require a datetime index rather than a ``time`` column.

    Args:
        client: Connected public client with rate-fetch capability.
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
        Mt5OperationError: If the client cannot fetch rate data.
        TypeError: If the client returns a non-DataFrame payload.
    """
    if count <= 0:
        msg = "count must be positive."
        raise ValueError(msg)
    copy_rates = getattr(client, "copy_rates_from_pos", None)
    if not callable(copy_rates):
        msg = "MT5 client cannot fetch rate data."
        raise Mt5OperationError(msg)
    frame = copy_rates(symbol, granularity, 0, count + 1)
    if not isinstance(frame, pd.DataFrame):
        msg = "MT5 client returned malformed rate data."
        raise TypeError(msg)
    frame = drop_forming_rate_bar(_ensure_rate_time_column(frame))
    if frame.empty:
        msg = f"Rate data is empty for {symbol!r} at granularity {granularity!r}."
        raise ValueError(msg)
    frame = frame.tail(count).reset_index(drop=True)
    if "time" not in frame.columns:
        msg = f"Rate data is missing a time column for {symbol!r}."
        raise ValueError(msg)
    idx = _rate_time_to_utc(frame["time"], symbol)
    idx.name = "time"
    result = frame.drop(columns=["time"])
    result.index = idx
    return result
