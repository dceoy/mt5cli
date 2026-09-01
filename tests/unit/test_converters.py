"""Tests for the :mod:`mt5cli.converters` module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from mt5cli.converters import (
    ensure_trade_server_time,
    ensure_utc,
    granularity_name,
    normalize_symbol,
    normalize_symbols,
    parse_date_range,
    recent_window,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" eurusd ", "eurusd"),
        ("GbpJpy", "GbpJpy"),
        ("XAUUSDm", "XAUUSDm"),
        ("US500.cash", "US500.cash"),
        ("EURUSD.r", "EURUSD.r"),
    ],
)
def test_normalize_symbol(raw: str, expected: str) -> None:
    """Symbol normalization trims whitespace and preserves broker casing."""
    assert normalize_symbol(raw) == expected


def test_normalize_symbols_deduplicates() -> None:
    """Symbol lists are normalized and de-duplicated in order."""
    assert normalize_symbols(["XAUUSDm", " XAUUSDm ", "EURUSD.r", "eurusd"]) == [
        "XAUUSDm",
        "EURUSD.r",
        "eurusd",
    ]


def test_parse_date_range_rejects_inverted_bounds() -> None:
    """Date ranges must not be inverted."""
    with pytest.raises(ValueError, match="must not be after"):
        parse_date_range("2024-02-01", "2024-01-01")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hours": 24},
        {"seconds": 3600},
    ],
    ids=["hours", "seconds"],
)
def test_recent_window_success_cases(kwargs: dict[str, int]) -> None:
    """Recent windows end at the provided timestamp for both duration inputs."""
    end = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
    start, resolved_end = recent_window(date_to=end, **kwargs)
    assert resolved_end == end
    assert start < end


def test_recent_window_requires_explicit_server_time() -> None:
    """Recent windows reject an end that mt5cli cannot validate."""
    with pytest.raises(ValueError, match="date_to is required"):
        recent_window(hours=1)


@pytest.mark.parametrize(
    "value",
    [
        datetime(2024, 1, 1, tzinfo=UTC),
        "2024-01-01T00:00:00+00:00",
        "2024-01-01T00:00:00Z",
    ],
    ids=["aware-datetime", "offset-string", "z-string"],
)
def test_trade_server_time_rejects_aware_bounds(value: datetime | str) -> None:
    """Trade-server query bounds fail closed when timezone-aware."""
    with pytest.raises(ValueError, match="timezone-aware"):
        ensure_trade_server_time(value)


def test_trade_server_time_parses_offset_free_string() -> None:
    """Offset-free ISO strings remain naive server wall-clock labels."""
    result = ensure_trade_server_time("2024-01-01T12:34:56")
    assert result == datetime(2024, 1, 1, 12, 34, 56, tzinfo=UTC).replace(tzinfo=None)
    assert result.tzinfo is None


def test_granularity_name_maps_timeframe_alias() -> None:
    """Granularity labels resolve MT5 timeframe aliases."""
    assert granularity_name("M1") == "M1"


def test_normalize_symbol_rejects_empty_value() -> None:
    """Empty symbols are rejected after trimming."""
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_symbol("   ")


def test_ensure_utc_rejects_naive_and_normalizes_aware_datetimes() -> None:
    """UTC coercion rejects naive and normalizes timezone-aware datetimes."""
    naive = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    aware = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone-naive"):
        ensure_utc(naive)
    assert ensure_utc(aware).tzinfo == UTC
    assert ensure_utc("2024-01-01T00:00:00+00:00").tzinfo == UTC


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one"),
        ({"hours": 1, "seconds": 1}, "exactly one"),
        ({"hours": 0}, "positive"),
    ],
    ids=["no-duration", "both-hours-and-seconds", "non-positive-duration"],
)
def test_recent_window_validation_errors(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Recent window helpers validate mutually exclusive length arguments."""
    with pytest.raises(ValueError, match=match):
        recent_window(**kwargs)  # type: ignore[arg-type]


def test_parse_date_range_returns_ordered_bounds() -> None:
    """Valid date ranges return ordered server wall-clock bounds."""
    start, end = parse_date_range("2024-01-01", "2024-02-01")
    assert start < end
    assert start.tzinfo is None
    assert end.tzinfo is None


def test_granularity_name_falls_back_for_unknown_timeframe(
    mocker: MockerFixture,
) -> None:
    """Unknown timeframe integers stringify as granularity labels."""
    mocker.patch(
        "mt5cli.converters._get_timeframe_name",
        side_effect=ValueError("unknown"),
    )
    assert granularity_name(1) == "1"
