"""Shared conversion helpers for MT5 symbols, timeframes, and date ranges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pdmt5.constants import get_timeframe_name as _get_timeframe_name

from .utils import parse_datetime, parse_tick_flags, parse_timeframe

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ensure_trade_server_time",
    "ensure_utc",
    "granularity_name",
    "normalize_symbol",
    "normalize_symbols",
    "parse_date_range",
    "parse_datetime",
    "parse_tick_flags",
    "parse_timeframe",
    "recent_window",
]


def normalize_symbol(symbol: str) -> str:
    """Normalize a broker symbol name for MT5 API calls.

    Strips surrounding whitespace while preserving broker-specific casing and
    suffixes (for example ``XAUUSDm``, ``US500.cash``, or ``EURUSD.r``).

    Args:
        symbol: Raw symbol name.

    Returns:
        Normalized symbol string.

    Raises:
        ValueError: If the symbol is empty after normalization.
    """
    normalized = symbol.strip()
    if not normalized:
        msg = "Symbol must not be empty."
        raise ValueError(msg)
    return normalized


def normalize_symbols(symbols: Sequence[str]) -> list[str]:
    """Normalize a sequence of broker symbol names.

    Args:
        symbols: Raw symbol names.

    Returns:
        List of normalized, de-duplicated symbols preserving first-seen order.
    """
    seen: set[str] = set()
    resolved: list[str] = []
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        if normalized not in seen:
            seen.add(normalized)
            resolved.append(normalized)
    return resolved


def ensure_utc(value: datetime | str) -> datetime:
    """Return a timezone-aware UTC datetime without guessing a timezone.

    ISO-8601 strings continue to use :func:`parse_datetime`, whose documented
    input contract treats a missing offset as UTC for CLI/user input. Datetime
    objects are stricter: a naive object has no timezone information and may be
    a pdmt5 trade-server wall-clock value, so attaching UTC would silently
    change its meaning.

    Args:
        value: Datetime instance or ISO 8601 string.

    Returns:
        UTC-aware datetime.

    Raises:
        ValueError: If a datetime object is timezone-naive.
    """
    if isinstance(value, str):
        return parse_datetime(value).astimezone(UTC)
    if value.tzinfo is None:
        msg = (
            "Cannot convert a timezone-naive datetime to UTC without an explicit "
            "timezone; pdmt5 timestamps may represent MT5 trade-server wall time."
        )
        raise ValueError(msg)
    return value.astimezone(UTC)


def ensure_trade_server_time(value: datetime | str) -> datetime:
    """Return a timezone-naive MT5 trade-server wall-clock datetime.

    ISO 8601 strings without an offset are parsed as naive wall-clock values.
    Datetime objects and strings carrying timezone information are rejected;
    mt5cli has no broker timezone with which to convert them safely.

    Args:
        value: Naive datetime or offset-free ISO 8601 string.

    Returns:
        The timezone-naive trade-server wall-clock datetime.

    Raises:
        ValueError: If the value is invalid or timezone-aware.
    """
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            msg = f"Invalid datetime format: '{value}'. Use ISO 8601 format."
            raise ValueError(msg) from None
    else:
        parsed = value
    if parsed.tzinfo is not None:
        msg = (
            "MT5 query bounds must be timezone-naive trade-server wall-clock "
            "datetimes; timezone-aware values require an explicit "
            "UTC-to-server-time conversion."
        )
        raise ValueError(msg)
    return parsed


def parse_date_range(
    date_from: datetime | str,
    date_to: datetime | str,
) -> tuple[datetime, datetime]:
    """Parse and validate an inclusive trade-server wall-clock range.

    Args:
        date_from: Range start as datetime or ISO 8601 string.
        date_to: Range end as datetime or ISO 8601 string.

    Returns:
        Tuple of timezone-naive ``(start, end)`` datetimes.

    Raises:
        ValueError: If either bound is timezone-aware or ``date_from`` is after
            ``date_to``.
    """
    start = ensure_trade_server_time(date_from)
    end = ensure_trade_server_time(date_to)
    if start > end:
        msg = (
            f"date_from ({start.isoformat()}) must not be after "
            f"date_to ({end.isoformat()})."
        )
        raise ValueError(msg)
    return start, end


def recent_window(
    *,
    hours: float | None = None,
    seconds: float | None = None,
    date_to: datetime | str | None = None,
) -> tuple[datetime, datetime]:
    """Build a trailing trade-server wall-clock window.

    Exactly one of ``hours`` or ``seconds`` must be provided.

    Args:
        hours: Trailing window length in hours.
        seconds: Trailing window length in seconds.
        date_to: Required naive MT5 trade-server window end.

    Returns:
        Tuple of timezone-naive ``(start, end)`` datetimes.

    Raises:
        ValueError: If neither or both window lengths are provided, a length is
            not positive, ``date_to`` is omitted, or ``date_to`` is
            timezone-aware.
    """
    if (hours is None) == (seconds is None):
        msg = "Provide exactly one of hours or seconds."
        raise ValueError(msg)
    if hours is not None:
        length = timedelta(hours=hours)
    else:
        length = timedelta(seconds=seconds if seconds is not None else 0)
    if length.total_seconds() <= 0:
        msg = "Window length must be positive."
        raise ValueError(msg)
    if date_to is None:
        msg = (
            "date_to is required because mt5cli cannot determine the current "
            "MT5 trade-server time."
        )
        raise ValueError(msg)
    end = ensure_trade_server_time(date_to)
    return end - length, end


def granularity_name(timeframe: int | str) -> str:
    """Return a short granularity label for a timeframe integer or name.

    Args:
        timeframe: MT5 timeframe as integer or name (for example ``M1``).

    Returns:
        Short name such as ``M1`` or the stringified integer when unknown.
    """
    tf = parse_timeframe(timeframe)
    try:
        name = _get_timeframe_name(tf)
    except ValueError:
        return str(tf)
    return name.removeprefix("TIMEFRAME_")
