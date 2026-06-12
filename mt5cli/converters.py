"""Shared conversion helpers for MT5 symbols, timeframes, and date ranges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pdmt5 import get_timeframe_name as _get_timeframe_name

from .utils import parse_datetime, parse_tick_flags, parse_timeframe

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
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

    Strips surrounding whitespace and uppercases the symbol so downstream
    applications can accept mixed-case input consistently.

    Args:
        symbol: Raw symbol name.

    Returns:
        Normalized symbol string.

    Raises:
        ValueError: If the symbol is empty after normalization.
    """
    normalized = symbol.strip().upper()
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
    """Return a timezone-aware UTC datetime.

    Args:
        value: Datetime instance or ISO 8601 string.

    Returns:
        UTC-aware datetime.
    """
    if isinstance(value, str):
        return parse_datetime(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_date_range(
    date_from: datetime | str,
    date_to: datetime | str,
) -> tuple[datetime, datetime]:
    """Parse and validate an inclusive UTC date range.

    Args:
        date_from: Range start as datetime or ISO 8601 string.
        date_to: Range end as datetime or ISO 8601 string.

    Returns:
        Tuple of UTC-aware ``(start, end)`` datetimes.

    Raises:
        ValueError: If ``date_from`` is after ``date_to``.
    """
    start = ensure_utc(date_from)
    end = ensure_utc(date_to)
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
    """Build a trailing UTC window ending at ``date_to`` or now.

    Exactly one of ``hours`` or ``seconds`` must be provided.

    Args:
        hours: Trailing window length in hours.
        seconds: Trailing window length in seconds.
        date_to: Window end. Defaults to current UTC time.

    Returns:
        Tuple of UTC-aware ``(start, end)`` datetimes.

    Raises:
        ValueError: If neither or both window lengths are provided, or if a
            length is not positive.
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
    end = ensure_utc(date_to) if date_to is not None else datetime.now(UTC)
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
