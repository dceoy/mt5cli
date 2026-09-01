"""Tests for the :mod:`mt5cli.schemas` module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pandas as pd
import pytest

from mt5cli.exceptions import Mt5SchemaError
from mt5cli.schemas import (
    DEDUP_KEYS,
    REQUIRED_COLUMNS,
    TIME_COLUMNS,
    DataKind,
    ensure_utc_columns,
    normalize_dataframe,
    normalize_time_columns,
    schema_columns,
    validate_schema,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

_SAMPLE_FRAME_COLUMNS: dict[DataKind, dict[str, list[object]]] = {
    DataKind.rates: {
        "time": [datetime(2024, 1, 1, tzinfo=UTC)],
        "open": [1.1],
        "high": [1.2],
        "low": [1.0],
        "close": [1.15],
        "tick_volume": [10],
        "spread": [1],
        "real_volume": [0],
    },
    DataKind.ticks: {
        "time": [datetime(2024, 1, 1, tzinfo=UTC)],
        "bid": [1.1],
        "ask": [1.11],
        "last": [1.105],
        "volume": [1],
        "time_msc": [datetime(2024, 1, 1, tzinfo=UTC)],
        "flags": [2],
        "volume_real": [0.0],
    },
    DataKind.orders: {
        "ticket": [1],
        "time_setup": [datetime(2024, 1, 1, tzinfo=UTC)],
        "type": [0],
        "state": [1],
        "symbol": ["EURUSD"],
        "volume_current": [0.1],
        "price_open": [1.1],
    },
    DataKind.positions: {
        "ticket": [1],
        "time": [datetime(2024, 1, 1, tzinfo=UTC)],
        "type": [0],
        "symbol": ["EURUSD"],
        "volume": [0.1],
        "price_open": [1.1],
        "price_current": [1.11],
        "profit": [1.0],
    },
    DataKind.history_orders: {
        "ticket": [1],
        "time_setup": [datetime(2024, 1, 1, tzinfo=UTC)],
        "type": [0],
        "state": [3],
        "symbol": ["EURUSD"],
        "volume_initial": [0.1],
        "price_open": [1.1],
    },
    DataKind.symbols: {
        "symbol": ["EURUSD"],
        "time": [datetime(2024, 1, 1, tzinfo=UTC)],
        "point": [0.00001],
        "digits": [5],
        "trade_contract_size": [100000.0],
        "volume_min": [0.01],
        "volume_max": [100.0],
        "volume_step": [0.01],
        "trade_tick_size": [0.00001],
        "trade_tick_value": [1.0],
        "currency_profit": ["USD"],
    },
    DataKind.history_deals: {
        "ticket": [1],
        "order": [2],
        "time": [datetime(2024, 1, 1, tzinfo=UTC)],
        "type": [0],
        "entry": [0],
        "symbol": ["EURUSD"],
        "volume": [0.1],
        "price": [1.1],
        "profit": [0.0],
    },
}


def _sample_frame(kind: DataKind) -> pd.DataFrame:
    return pd.DataFrame(_SAMPLE_FRAME_COLUMNS[kind])


@pytest.mark.parametrize("kind", list(DataKind))
def test_required_columns_contract(kind: DataKind) -> None:
    """Each dataset kind exposes a non-empty required column contract."""
    assert REQUIRED_COLUMNS[kind]
    validate_schema(_sample_frame(kind), kind)


@pytest.mark.parametrize("kind", list(DataKind))
def test_normalize_dataframe_injects_storage_metadata(kind: DataKind) -> None:
    """Normalization accepts MT5 frames and optional storage metadata."""
    frame = _sample_frame(kind)
    normalized = normalize_dataframe(
        frame,
        kind,
        symbol="eurusd",
        timeframe="M1" if kind is DataKind.rates else None,
    )
    if kind is DataKind.rates:
        assert normalized.loc[0, "symbol"] == "eurusd"
        assert normalized.loc[0, "timeframe"] == 1
    validate_schema(normalized, kind)


def test_validate_schema_raises_for_missing_columns() -> None:
    """Schema validation fails fast on missing required columns."""
    with pytest.raises(Mt5SchemaError, match="missing required columns"):
        validate_schema(pd.DataFrame({"time": [1]}), DataKind.rates)


def test_schema_columns_and_extra_required_validation() -> None:
    """Schema helpers expose contracts and honor extra required columns."""
    assert schema_columns(DataKind.rates) == REQUIRED_COLUMNS[DataKind.rates]
    validate_schema(pd.DataFrame(), DataKind.rates)
    frame = _sample_frame(DataKind.rates)
    with pytest.raises(Mt5SchemaError, match="storage_symbol"):
        validate_schema(frame, DataKind.rates, extra_required=["storage_symbol"])


def test_dedup_keys_match_storage_deduplication_contract() -> None:
    """SQLite history dedup keys stay aligned with the documented schema contract."""
    assert DEDUP_KEYS[DataKind.rates][0] == ("symbol", "timeframe", "time")
    assert DEDUP_KEYS[DataKind.ticks][0] == ("symbol", "time_msc")


def test_time_columns_include_optional_order_fields() -> None:
    """Schema contracts document optional MT5 time columns per dataset kind."""
    assert "time_done" in TIME_COLUMNS[DataKind.orders]
    assert "time_setup_msc" in TIME_COLUMNS[DataKind.history_orders]


def test_normalize_dataframe_empty_and_tick_sort_paths() -> None:
    """Normalization handles empty frames and tick time_msc sorting."""
    empty = pd.DataFrame()
    assert normalize_dataframe(empty, DataKind.rates).empty

    ticks = _sample_frame(DataKind.ticks)
    ticks = pd.concat([ticks, ticks], ignore_index=True)
    sorted_ticks = normalize_dataframe(ticks, DataKind.ticks, sort=True)
    assert len(sorted_ticks) == 2
    unsorted_ticks = normalize_dataframe(ticks, DataKind.ticks, sort=False)
    assert len(unsorted_ticks) == 2


def test_normalize_dataframe_rate_timeframe_without_symbol() -> None:
    """Rate normalization can inject timeframe without symbol metadata."""
    frame = _sample_frame(DataKind.rates)
    normalized = normalize_dataframe(frame, DataKind.rates, timeframe="M1")
    assert "timeframe" in normalized.columns


def test_normalize_dataframe_keeps_existing_symbol_and_timeframe() -> None:
    """Normalization does not duplicate existing storage metadata columns."""
    frame = normalize_dataframe(
        _sample_frame(DataKind.rates),
        DataKind.rates,
        symbol="EURUSD",
        timeframe="M1",
    )
    normalized = normalize_dataframe(
        frame,
        DataKind.rates,
        symbol="GBPUSD",
        timeframe="H1",
    )
    assert normalized.loc[0, "symbol"] == "EURUSD"
    assert normalized.loc[0, "timeframe"] == 1


def test_normalize_time_columns_skips_absent_time_fields() -> None:
    """Time normalization ignores absent optional time columns."""
    frame = pd.DataFrame({"open": [1.0]})
    result = normalize_time_columns(frame, DataKind.rates)
    assert list(result.columns) == ["open"]


@pytest.mark.parametrize(
    ("col", "value", "kind", "expected"),
    [
        (
            "time",
            1704067200,
            DataKind.rates,
            pd.Timestamp("2024-01-01"),
        ),
        (
            "time_msc",
            1704067200000,
            DataKind.ticks,
            pd.Timestamp("2024-01-01"),
        ),
        (
            "time",
            datetime(2024, 1, 1, tzinfo=UTC),
            DataKind.rates,
            pd.Timestamp("2024-01-01T00:00:00+00:00"),
        ),
        (
            "time",
            "2024-01-01T00:00:00+00:00",
            DataKind.rates,
            pd.Timestamp("2024-01-01T00:00:00+00:00"),
        ),
        (
            "time",
            "2024-01-01T00:00:00",
            DataKind.rates,
            pd.Timestamp("2024-01-01"),
        ),
    ],
)
def test_normalize_time_columns_coerces_value(
    col: str,
    value: object,
    kind: DataKind,
    expected: pd.Timestamp,
) -> None:
    """Time values preserve naive MT5 labels and normalize aware values to UTC."""
    frame = pd.DataFrame({col: [value]})
    result = normalize_time_columns(frame, kind)
    assert result.loc[0, col] == expected


def test_normalize_time_columns_handles_optional_order_times() -> None:
    """Optional order/history time columns are normalized when present."""
    frame = pd.DataFrame({
        "time_setup": [1704067200],
        "time_setup_msc": [1704067200000],
        "time_done": [1704153600],
        "time_done_msc": [1704153600000],
    })
    result = normalize_time_columns(frame, DataKind.orders)
    assert result.loc[0, "time_setup"] == pd.Timestamp("2024-01-01")
    assert result.loc[0, "time_setup_msc"] == pd.Timestamp("2024-01-01")
    assert result.loc[0, "time_done"] == pd.Timestamp("2024-01-02")
    assert result.loc[0, "time_done_msc"] == pd.Timestamp("2024-01-02")


def test_normalize_time_columns_handles_mixed_timezone_values() -> None:
    """Mixed naive and aware values preserve labels and normalize aware values."""
    frame = pd.DataFrame({
        "time": [
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:00+09:00",
            "2024-01-01T00:00:00+00:00",
        ],
    })

    result = normalize_time_columns(frame, DataKind.rates)

    assert result.loc[0, "time"] == pd.Timestamp("2024-01-01")
    assert result.loc[1, "time"] == pd.Timestamp("2023-12-31T15:00:00+00:00")
    assert result.loc[2, "time"] == pd.Timestamp("2024-01-01T00:00:00+00:00")
    assert cast("pd.Timestamp", result.loc[1, "time"]).tzinfo == UTC
    assert cast("pd.Timestamp", result.loc[2, "time"]).tzinfo == UTC


def test_normalize_time_columns_handles_mixed_numeric_values() -> None:
    """Mixed MT5 time columns parse numeric values with their epoch units."""
    frame = pd.DataFrame({
        "time": pd.Series([1704067200, "2024-01-02"], dtype="object"),
        "time_msc": pd.Series(
            [1704067200000, "2024-01-02"],
            dtype="object",
        ),
    })

    result = normalize_time_columns(frame, DataKind.ticks)

    assert result.loc[0, "time"] == pd.Timestamp("2024-01-01")
    assert result.loc[0, "time_msc"] == pd.Timestamp("2024-01-01")
    assert result.loc[1, "time"] == pd.Timestamp("2024-01-02")
    assert result.loc[1, "time_msc"] == pd.Timestamp("2024-01-02")


def test_normalize_dataframe_sorts_ticks_by_time_msc(
    mocker: MockerFixture,
) -> None:
    """Tick frames without ``time`` can still sort on ``time_msc``."""
    mocker.patch("mt5cli.schemas.validate_schema")
    ticks = pd.concat([_sample_frame(DataKind.ticks)] * 2, ignore_index=True).drop(
        columns=["time"],
    )
    ticks.loc[0, "time_msc"] = datetime(2024, 1, 1, tzinfo=UTC)
    ticks.loc[1, "time_msc"] = datetime(2024, 1, 2, tzinfo=UTC)
    ticks = pd.concat([ticks.iloc[[1]], ticks.iloc[[0]]], ignore_index=True)
    normalized = normalize_dataframe(ticks, DataKind.ticks, sort=True)
    assert normalized.iloc[0]["time_msc"] <= normalized.iloc[1]["time_msc"]


def test_ensure_utc_columns_skips_missing_columns() -> None:
    """UTC column coercion ignores absent columns."""
    frame = _sample_frame(DataKind.rates)
    result = ensure_utc_columns(frame, ["time", "missing"])
    assert "time" in result.columns


def test_ensure_utc_columns_coerces_non_mt5_columns() -> None:
    """Non-MT5 columns still coerce to UTC datetimes."""
    frame = pd.DataFrame({"created_at": ["2024-01-01T00:00:00+00:00"]})
    result = ensure_utc_columns(frame, ["created_at"])
    assert result.loc[0, "created_at"] == pd.Timestamp("2024-01-01T00:00:00+00:00")
