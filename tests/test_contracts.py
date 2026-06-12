"""Contract tests for the mt5cli public API and dataset schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pdmt5 import Mt5RuntimeError, Mt5TradingError
from pytest_mock import MockerFixture  # noqa: TC002

from mt5cli import (
    DEDUP_KEYS,
    REQUIRED_COLUMNS,
    TIME_COLUMNS,
    DataKind,
    Dataset,
    MT5Client,
    Mt5CliError,
    Mt5ConnectionError,
    Mt5OperationError,
    Mt5SchemaError,
    build_config,
    call_with_normalized_errors,
    detect_format,
    ensure_utc,
    export_dataframe,
    export_dataframe_to_sqlite,
    granularity_name,
    is_recoverable_mt5_error,
    mt5_session,
    normalize_dataframe,
    normalize_mt5_exception,
    normalize_symbol,
    normalize_symbols,
    parse_date_range,
    recent_window,
    schema_columns,
    validate_schema,
)
from mt5cli.retry import retry_with_backoff
from mt5cli.schemas import ensure_utc_columns, normalize_time_columns

if TYPE_CHECKING:
    from pathlib import Path


def _sample_frame(kind: DataKind) -> pd.DataFrame:
    if kind is DataKind.rates:
        return pd.DataFrame({
            "time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "open": [1.1],
            "high": [1.2],
            "low": [1.0],
            "close": [1.15],
            "tick_volume": [10],
            "spread": [1],
            "real_volume": [0],
        })
    if kind is DataKind.ticks:
        return pd.DataFrame({
            "time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "bid": [1.1],
            "ask": [1.11],
            "last": [1.105],
            "volume": [1],
            "time_msc": [datetime(2024, 1, 1, tzinfo=UTC)],
            "flags": [2],
            "volume_real": [0.0],
        })
    if kind is DataKind.orders:
        return pd.DataFrame({
            "ticket": [1],
            "time_setup": [datetime(2024, 1, 1, tzinfo=UTC)],
            "type": [0],
            "state": [1],
            "symbol": ["EURUSD"],
            "volume_current": [0.1],
            "price_open": [1.1],
        })
    if kind is DataKind.positions:
        return pd.DataFrame({
            "ticket": [1],
            "time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "type": [0],
            "symbol": ["EURUSD"],
            "volume": [0.1],
            "price_open": [1.1],
            "price_current": [1.11],
            "profit": [1.0],
        })
    if kind is DataKind.history_orders:
        return pd.DataFrame({
            "ticket": [1],
            "time_setup": [datetime(2024, 1, 1, tzinfo=UTC)],
            "type": [0],
            "state": [3],
            "symbol": ["EURUSD"],
            "volume_initial": [0.1],
            "price_open": [1.1],
        })
    return pd.DataFrame({
        "ticket": [1],
        "order": [2],
        "time": [datetime(2024, 1, 1, tzinfo=UTC)],
        "type": [0],
        "entry": [0],
        "symbol": ["EURUSD"],
        "volume": [0.1],
        "price": [1.1],
        "profit": [0.0],
    })


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


def test_history_dedup_keys_match_schema_contract() -> None:
    """SQLite history dedup keys stay aligned with schema contracts."""
    assert DEDUP_KEYS[DataKind.rates][0] == ("symbol", "timeframe", "time")
    assert DEDUP_KEYS[DataKind.ticks][0] == ("symbol", "time_msc")
    assert Dataset.rates.table_name == "rates"


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


def test_recent_window_builds_trailing_bounds() -> None:
    """Recent windows end at the provided timestamp."""
    end = datetime(2024, 1, 2, tzinfo=UTC)
    start, resolved_end = recent_window(hours=24, date_to=end)
    assert resolved_end == end
    assert start < end


def test_granularity_name_maps_timeframe_alias() -> None:
    """Granularity labels resolve MT5 timeframe aliases."""
    assert granularity_name("M1") == "M1"


@pytest.mark.parametrize(
    "exc",
    [Mt5RuntimeError("init failed"), Mt5TradingError("trade failed")],
)
def test_is_recoverable_mt5_error(exc: Exception) -> None:
    """Recoverable MT5 errors are classified consistently."""
    assert is_recoverable_mt5_error(exc)


def test_normalize_mt5_exception_maps_types() -> None:
    """MT5 exceptions map to stable mt5cli types."""
    assert isinstance(
        normalize_mt5_exception(Mt5RuntimeError("x")),
        Mt5ConnectionError,
    )
    assert isinstance(
        normalize_mt5_exception(Mt5TradingError("x")),
        Mt5OperationError,
    )


def test_call_with_normalized_errors_reraises_mapped_type() -> None:
    """Normalized error helper re-raises mapped mt5cli exceptions."""

    def _raise() -> None:
        message = "boom"
        raise Mt5RuntimeError(message)

    with pytest.raises(Mt5ConnectionError):
        call_with_normalized_errors(_raise)


def test_retry_with_backoff_retries_recoverable_errors(
    mocker: MockerFixture,
) -> None:
    """Retry helper retries recoverable MT5 failures."""
    calls = {"count": 0}

    def _flaky() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            message = "transient"
            raise Mt5RuntimeError(message)
        return "ok"

    mocker.patch("mt5cli.retry.time.sleep")
    assert retry_with_backoff(_flaky, retry_count=1) == "ok"
    assert calls["count"] == 2


def test_public_api_exports_mt5_client() -> None:
    """MT5Client is the primary importable client abstraction."""
    client = MT5Client(config=build_config())
    assert isinstance(client, MT5Client)
    assert isinstance(client, MT5Client.__mro__[1])


def test_mt5_client_order_primitives_use_connected_client(
    mock_client: object,
) -> None:
    """Order check/send route through the same client fetch path as exports."""
    request = {"action": 1}
    client = MT5Client()
    client.order_check(request)
    client.order_send(request)
    assert mock_client.order_check_as_df.call_count == 1  # type: ignore[attr-defined]
    assert mock_client.order_send_as_df.call_count == 1  # type: ignore[attr-defined]


def test_storage_export_round_trip_csv(tmp_path: Path) -> None:
    """Storage helpers export normalized rate frames to CSV."""
    frame = normalize_dataframe(
        _sample_frame(DataKind.rates),
        DataKind.rates,
        symbol="EURUSD",
        timeframe="M1",
    )
    output = tmp_path / "rates.csv"
    export_dataframe(frame, output, detect_format(output))
    loaded = pd.read_csv(output)
    assert len(loaded) == 1
    assert "close" in loaded.columns


def test_normalize_symbol_rejects_empty_value() -> None:
    """Empty symbols are rejected after trimming."""
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_symbol("   ")


def test_ensure_utc_handles_naive_and_aware_datetimes() -> None:
    """UTC coercion accepts naive and timezone-aware datetimes."""
    naive = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    aware = datetime(2024, 1, 1, tzinfo=UTC)
    assert ensure_utc(naive).tzinfo == UTC
    assert ensure_utc(aware).tzinfo == UTC
    assert ensure_utc("2024-01-01T00:00:00+00:00").tzinfo == UTC


def test_recent_window_validation_errors() -> None:
    """Recent window helpers validate mutually exclusive length arguments."""
    with pytest.raises(ValueError, match="exactly one"):
        recent_window()
    with pytest.raises(ValueError, match="exactly one"):
        recent_window(hours=1, seconds=1)
    with pytest.raises(ValueError, match="positive"):
        recent_window(hours=0)


def test_recent_window_supports_seconds_argument() -> None:
    """Recent windows can be built from a seconds-based length."""
    end = datetime(2024, 1, 2, tzinfo=UTC)
    start, resolved_end = recent_window(seconds=3600, date_to=end)
    assert resolved_end == end
    assert start < end


def test_parse_date_range_returns_ordered_bounds() -> None:
    """Valid date ranges return UTC-aware bounds."""
    start, end = parse_date_range("2024-01-01", "2024-02-01")
    assert start < end


def test_granularity_name_falls_back_for_unknown_timeframe(
    mocker: MockerFixture,
) -> None:
    """Unknown timeframe integers stringify as granularity labels."""
    mocker.patch(
        "mt5cli.converters._get_timeframe_name",
        side_effect=ValueError("unknown"),
    )
    assert granularity_name(1) == "1"


def test_normalize_mt5_exception_passthrough_and_generic() -> None:
    """Normalization preserves mt5cli errors and wraps unknown exceptions."""
    original = Mt5CliError("known")
    assert normalize_mt5_exception(original) is original
    assert isinstance(normalize_mt5_exception(ValueError("x")), Mt5CliError)


def test_schema_columns_and_extra_required_validation() -> None:
    """Schema helpers expose contracts and honor extra required columns."""
    assert schema_columns(DataKind.rates) == REQUIRED_COLUMNS[DataKind.rates]
    validate_schema(pd.DataFrame(), DataKind.rates)
    frame = _sample_frame(DataKind.rates)
    with pytest.raises(Mt5SchemaError, match="storage_symbol"):
        validate_schema(frame, DataKind.rates, extra_required=["storage_symbol"])


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


def test_normalize_time_columns_converts_unix_seconds() -> None:
    """Numeric MT5 ``time`` values are interpreted as Unix seconds."""
    frame = pd.DataFrame({"time": [1704067200]})
    result = normalize_time_columns(frame, DataKind.rates)
    assert result.loc[0, "time"] == pd.Timestamp("2024-01-01T00:00:00+00:00")


def test_normalize_time_columns_converts_unix_milliseconds() -> None:
    """Numeric MT5 ``time_msc`` values are interpreted as Unix milliseconds."""
    frame = pd.DataFrame({"time_msc": [1704067200000]})
    result = normalize_time_columns(frame, DataKind.ticks)
    assert result.loc[0, "time_msc"] == pd.Timestamp("2024-01-01T00:00:00+00:00")


def test_normalize_time_columns_preserves_utc_datetimes() -> None:
    """Already-converted datetime values remain UTC-normalized."""
    aware = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame({"time": [aware]})
    result = normalize_time_columns(frame, DataKind.rates)
    assert result.loc[0, "time"] == pd.Timestamp("2024-01-01T00:00:00+00:00")


def test_normalize_time_columns_handles_optional_order_times() -> None:
    """Optional order/history time columns are normalized when present."""
    frame = pd.DataFrame({
        "time_setup": [1704067200],
        "time_setup_msc": [1704067200000],
        "time_done": [1704153600],
        "time_done_msc": [1704153600000],
    })
    result = normalize_time_columns(frame, DataKind.orders)
    assert result.loc[0, "time_setup"] == pd.Timestamp("2024-01-01T00:00:00+00:00")
    assert result.loc[0, "time_setup_msc"] == pd.Timestamp(
        "2024-01-01T00:00:00+00:00",
    )
    assert result.loc[0, "time_done"] == pd.Timestamp("2024-01-02T00:00:00+00:00")
    assert result.loc[0, "time_done_msc"] == pd.Timestamp(
        "2024-01-02T00:00:00+00:00",
    )


def test_time_columns_include_optional_order_fields() -> None:
    """Schema contracts document optional MT5 time columns per dataset kind."""
    assert "time_done" in TIME_COLUMNS[DataKind.orders]
    assert "time_setup_msc" in TIME_COLUMNS[DataKind.history_orders]


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


def test_normalize_time_columns_coerces_string_timestamps() -> None:
    """String timestamps are parsed with timezone-aware datetime coercion."""
    frame = pd.DataFrame({"time": ["2024-01-01T00:00:00+00:00"]})
    result = normalize_time_columns(frame, DataKind.rates)
    assert result.loc[0, "time"] == pd.Timestamp("2024-01-01T00:00:00+00:00")


def test_ensure_utc_columns_coerces_non_mt5_columns() -> None:
    """Non-MT5 columns still coerce to UTC datetimes."""
    frame = pd.DataFrame({"created_at": ["2024-01-01T00:00:00+00:00"]})
    result = ensure_utc_columns(frame, ["created_at"])
    assert result.loc[0, "created_at"] == pd.Timestamp("2024-01-01T00:00:00+00:00")


def test_mt5_session_yields_connected_client(mocker: MockerFixture) -> None:
    """Public mt5_session yields an MT5Client bound to a connected session."""
    connected = mocker.MagicMock()
    context = mocker.MagicMock()
    context.__enter__.return_value = connected
    context.__exit__.return_value = False
    mocker.patch("mt5cli.client.connected_client", return_value=context)
    with mt5_session(build_config()) as client:
        assert isinstance(client, MT5Client)


def test_retry_with_backoff_reraises_non_recoverable_errors() -> None:
    """Non-MT5 errors are not retried."""

    def _raise() -> None:
        message = "fatal"
        raise ValueError(message)

    with pytest.raises(ValueError, match="fatal"):
        retry_with_backoff(_raise, retry_count=2)


def test_storage_export_round_trip_sqlite(tmp_path: Path) -> None:
    """Storage helpers append deduplicated frames to SQLite."""
    frame = normalize_dataframe(
        _sample_frame(DataKind.rates),
        DataKind.rates,
        symbol="EURUSD",
        timeframe="M1",
    )
    output = tmp_path / "rates.db"
    export_dataframe_to_sqlite(
        frame,
        output,
        "rates",
        deduplicate_on=DEDUP_KEYS[DataKind.rates][0],
    )
    with __import__("sqlite3").connect(output) as conn:
        count = conn.execute("SELECT COUNT(*) FROM rates").fetchone()[0]
    assert count == 1
