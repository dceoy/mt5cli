"""Contract tests for the mt5cli public API and dataset schemas."""

from __future__ import annotations

import importlib
import sqlite3
from datetime import UTC, datetime
from importlib.metadata import requires
from typing import TYPE_CHECKING, get_type_hints
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import pandas as pd
import pytest
from pdmt5 import Mt5RuntimeError
from pytest_mock import MockerFixture  # noqa: TC002

import mt5cli
from mt5cli import (
    STABLE_SDK_EXPORTS,
    AccountSpec,
    ExecutionStatus,
    MarginVolume,
    MT5Client,
    Mt5CliError,
    Mt5ConnectionError,
    Mt5SchemaError,
    OrderExecutionResult,
    OrderLimits,
    RateTarget,
    build_config,
    build_rate_targets,
    calculate_account_projected_margin_ratio,
    calculate_margin_and_volume,
    calculate_positions_margin,
    calculate_projected_margin_ratio,
    calculate_symbol_group_margin_ratio,
    calculate_trailing_stop_updates,
    drop_forming_rate_bar,
    ensure_symbol_selected,
    extract_tick_price,
    fetch_latest_closed_rates,
    fetch_latest_closed_rates_for_trading_client,
    fetch_latest_closed_rates_indexed,
    load_rate_series_from_sqlite,
    mt5_session,
    mt5_trading_session,
    normalize_order_volume,
    place_market_order,
    resolve_account_spec,
    resolve_account_specs,
)
from mt5cli.converters import (
    ensure_utc,
    granularity_name,
    normalize_symbol,
    normalize_symbols,
    parse_date_range,
    recent_window,
)
from mt5cli.exceptions import (
    call_with_normalized_errors,
    is_recoverable_mt5_error,
    normalize_mt5_exception,
)
from mt5cli.history import (
    create_rate_compatibility_views,
    load_rate_data,
    resolve_rate_view_name,
)
from mt5cli.retry import retry_with_backoff
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
from mt5cli.utils import (
    Dataset,
    detect_format,
    export_dataframe,
    export_dataframe_to_sqlite,
)

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
    end = datetime(2024, 1, 2, tzinfo=UTC)
    start, resolved_end = recent_window(date_to=end, **kwargs)
    assert resolved_end == end
    assert start < end


def test_granularity_name_maps_timeframe_alias() -> None:
    """Granularity labels resolve MT5 timeframe aliases."""
    assert granularity_name("M1") == "M1"


@pytest.mark.parametrize(
    "exc",
    [Mt5RuntimeError("init failed")],
)
def test_is_recoverable_mt5_error(exc: Exception) -> None:
    """Recoverable MT5 errors are classified consistently."""
    assert is_recoverable_mt5_error(exc)


@pytest.mark.parametrize(
    ("exc", "expected_type"),
    [
        (Mt5RuntimeError("x"), Mt5ConnectionError),
    ],
)
def test_normalize_mt5_exception_maps_types(
    exc: Exception,
    expected_type: type[Mt5ConnectionError],
) -> None:
    """MT5 exceptions map to stable mt5cli types."""
    assert isinstance(normalize_mt5_exception(exc), expected_type)


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


@pytest.mark.parametrize(
    ("col", "value", "kind"),
    [
        ("time", 1704067200, DataKind.rates),
        ("time_msc", 1704067200000, DataKind.ticks),
        ("time", datetime(2024, 1, 1, tzinfo=UTC), DataKind.rates),
        ("time", "2024-01-01T00:00:00+00:00", DataKind.rates),
    ],
)
def test_normalize_time_columns_coerces_value(
    col: str,
    value: object,
    kind: DataKind,
) -> None:
    """Time column values are coerced to UTC timestamps regardless of input type."""
    frame = pd.DataFrame({col: [value]})
    result = normalize_time_columns(frame, kind)
    assert result.loc[0, col] == pd.Timestamp("2024-01-01T00:00:00+00:00")


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


def test_storage_module_does_not_exist() -> None:
    """mt5cli.storage re-export module has been removed."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mt5cli.storage")


class TestStableSdkContract:
    """Tests for the documented stable downstream SDK contract."""

    def test_stable_exports_are_subset_of_all(self) -> None:
        """Every stable export is also listed in the package __all__."""
        missing = sorted(STABLE_SDK_EXPORTS - set(mt5cli.__all__))
        assert not missing, f"STABLE_SDK_EXPORTS missing from __all__: {missing}"

    def test_stable_exports_cover_root_api(self) -> None:
        """STABLE_SDK_EXPORTS classifies every package-root symbol."""
        tier_metadata = {"STABLE_SDK_EXPORTS"}
        root_exports = set(mt5cli.__all__)

        missing_from_root = sorted(STABLE_SDK_EXPORTS - root_exports)
        assert not missing_from_root, (
            f"STABLE_SDK_EXPORTS missing from __all__: {missing_from_root}"
        )

        unclassified = sorted(root_exports - STABLE_SDK_EXPORTS - tier_metadata)
        assert not unclassified, (
            f"Root exports not in STABLE_SDK_EXPORTS: {unclassified}"
        )

    @pytest.mark.parametrize("name", sorted(STABLE_SDK_EXPORTS))
    def test_stable_exports_are_importable_from_package_root(self, name: str) -> None:
        """Stable SDK names resolve through ``from mt5cli import ...``."""
        assert hasattr(mt5cli, name), f"{name!r} missing from mt5cli package root"

    def test_drop_forming_rate_bar_from_package_root(self) -> None:
        """Closed-bar trimming is available from the stable package surface."""
        frame = pd.DataFrame({"time": [1, 2, 3], "close": [1.0, 1.1, 1.2]})
        closed = drop_forming_rate_bar(frame)
        assert list(closed["close"]) == [1.0, 1.1]
        assert len(closed) == 2

    def test_fetch_latest_closed_rates_from_package_root(self) -> None:
        """Single-client closed-bar helper drops the forming row."""
        client = MagicMock()
        client.latest_rates.return_value = pd.DataFrame(
            {"time": [1, 2, 3], "close": [1.0, 1.1, 1.2]},
        )

        result = fetch_latest_closed_rates(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        client.latest_rates.assert_called_once_with("EURUSD", "M1", 3, start_pos=0)
        assert list(result["close"]) == [1.0, 1.1]

    def test_fetch_latest_closed_rates_for_trading_client_from_package_root(
        self,
    ) -> None:
        """Trading-client closed-bar helper is importable from the stable surface."""
        client = MagicMock()
        client.fetch_latest_rates_as_df.return_value = pd.DataFrame(
            {"time": [1, 2, 3], "close": [1.0, 1.1, 1.2]},
        )

        result = fetch_latest_closed_rates_for_trading_client(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert list(result["close"]) == [1.0, 1.1]

    def test_normalize_order_volume_from_package_root(self) -> None:
        """Volume normalization helper is importable from the stable surface."""
        result = normalize_order_volume(
            0.25,
            volume_min=0.1,
            volume_max=1.0,
            volume_step=0.1,
        )
        assert abs(result - 0.2) < 1e-9

    def test_calculate_positions_margin_from_package_root(self) -> None:
        """Position margin helper is importable from the stable surface."""
        client = MagicMock()
        client.mt5.POSITION_TYPE_BUY = 0
        client.mt5.POSITION_TYPE_SELL = 1
        client.mt5.ORDER_TYPE_BUY = 10
        client.mt5.ORDER_TYPE_SELL = 11
        client.positions_get_as_df.return_value = pd.DataFrame()

        assert calculate_positions_margin(client) == 0

    def test_generic_trading_helpers_from_package_root(self) -> None:
        """New generic trading helpers resolve through the stable surface."""
        price = extract_tick_price({"bid": "1.2"}, "bid")
        assert price is not None
        assert abs(price - 1.2) < 1e-9
        assert callable(calculate_trailing_stop_updates)
        assert callable(calculate_account_projected_margin_ratio)
        assert callable(calculate_projected_margin_ratio)
        assert callable(calculate_symbol_group_margin_ratio)

    def test_load_rate_series_from_sqlite_requires_managed_views(
        self,
        tmp_path: Path,
    ) -> None:
        """Multi-series loading fails clearly when managed views are absent."""
        db_path = tmp_path / "empty-views.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )

        targets = build_rate_targets(["EURUSD"], ["M1"])
        with pytest.raises(ValueError, match="No rate compatibility view exists"):
            load_rate_series_from_sqlite(db_path, targets, count=10)

        assert targets == [RateTarget(symbol="EURUSD", timeframe=1)]

    def test_resolve_account_spec_from_package_root(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Account credential resolution uses generic ${ENV_VAR} placeholders."""
        monkeypatch.setenv("APP_MT5_LOGIN", "555")
        monkeypatch.setenv("APP_MT5_PASSWORD", "secret")
        account = AccountSpec(
            symbols=["EURUSD"],
            login="${APP_MT5_LOGIN}",
            password="${APP_MT5_PASSWORD}",
            server="Broker-Demo",
        )

        resolved = resolve_account_spec(account, timeout=3000)
        assert resolved.login == "555"
        assert resolved.password == "secret"  # noqa: S105
        assert resolved.timeout == 3000

        batch = resolve_account_specs([account], server="Override")
        assert batch[0].server == "Override"

    def test_mt5_trading_session_lifecycle_from_package_root(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Trading session helper initializes and always shuts down."""
        mock_client = MagicMock()
        mocker.patch(
            "mt5cli.trading.Mt5DataClient",
            return_value=mock_client,
        )

        with mt5_trading_session(login=12345, server="Broker-Demo") as client:
            assert client is mock_client
            mock_client.initialize_and_login_mt5.assert_called_once()

        mock_client.shutdown.assert_called_once()

    def test_trading_order_helpers_importable_from_package_root(self) -> None:
        """Order planning helpers resolve through the stable package surface."""
        assert callable(calculate_margin_and_volume)
        assert callable(ensure_symbol_selected)
        assert callable(place_market_order)
        margin_hints = get_type_hints(MarginVolume)
        limits_hints = get_type_hints(OrderLimits)
        execution_hints = get_type_hints(OrderExecutionResult)
        assert margin_hints["buy_volume"] is float
        assert limits_hints["stop_loss"] == float | None
        assert execution_hints["status"] == ExecutionStatus

    def test_mt5_trading_session_shuts_down_on_exception(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Trading session helper shuts down even when the body raises."""
        mock_client = MagicMock()
        mocker.patch(
            "mt5cli.trading.Mt5DataClient",
            return_value=mock_client,
        )

        message = "strategy error"
        with (
            pytest.raises(RuntimeError, match=message),
            mt5_trading_session(login=12345, server="Broker-Demo"),
        ):
            raise RuntimeError(message)

        mock_client.shutdown.assert_called_once()

    def test_fetch_latest_closed_rates_indexed_from_package_root(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Indexed closed-bar helper returns a UTC DatetimeIndex named 'time'."""
        client = MagicMock()
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=pd.DataFrame(
                {
                    "time": [1704067200, 1704153600, 1704240000],
                    "close": [1.0, 1.1, 1.2],
                },
            ),
        )

        result = fetch_latest_closed_rates_indexed(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "time"
        assert result.index.tz is not None
        assert "time" not in result.columns
        assert "close" in result.columns

    def test_rate_view_helpers_in_history_module(self, tmp_path: Path) -> None:
        """Rate view helpers are available from mt5cli.history."""
        db_path = tmp_path / "rates.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            create_rate_compatibility_views(conn)

        assert resolve_rate_view_name(db_path, "EURUSD", "M1") == "rate_EURUSD__1"
        missing = tmp_path / "missing.db"
        with pytest.raises(ValueError, match="SQLite database not found"):
            resolve_rate_view_name(missing, "EURUSD", "M1", require_existing=True)

    def test_load_rate_data_in_history_module(self, tmp_path: Path) -> None:
        """SQLite rate loading normalizes timestamps through mt5cli.history."""
        db_path = tmp_path / "view.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                'CREATE VIEW "rate_EURUSD__1" AS'
                " SELECT '2024-01-01T00:00:00+00:00' AS time, 1.1 AS close",
            )

        frame = load_rate_data(db_path, "rate_EURUSD__1")
        assert frame.index.name == "time"
        assert abs(float(frame.iloc[0]["close"]) - 1.1) < 1e-9


@pytest.mark.parametrize(
    "name",
    [
        "Mt5Config",
        "Mt5RuntimeError",
        "Mt5TradingClient",
        "Mt5TradingError",
        "TICK_FLAG_MAP",
        "TIMEFRAME_MAP",
    ],
)
def test_pdmt5_pass_through_names_removed_from_public_contract(name: str) -> None:
    """Removed pdmt5 pass-through names are not part of the public contract."""
    assert name not in STABLE_SDK_EXPORTS, (
        f"{name!r} should not be in STABLE_SDK_EXPORTS"
    )
    assert name not in mt5cli.__all__, f"{name!r} should not be in mt5cli.__all__"


def test_mt5cli_does_not_import_high_level_trading_symbols() -> None:
    """mt5cli doesn't import Mt5TradingClient or Mt5TradingError at module level."""
    trading_module = importlib.import_module("mt5cli.trading")
    module_dict = vars(trading_module)
    assert "Mt5TradingClient" not in module_dict, (
        "mt5cli.trading should not import Mt5TradingClient at module level"
    )
    assert "Mt5TradingError" not in module_dict, (
        "mt5cli.trading should not import Mt5TradingError at module level"
    )


# ---------------------------------------------------------------------------
# Packaging metadata
# ---------------------------------------------------------------------------


def test_parquet_extra_declares_pyarrow() -> None:
    """Package metadata lists pyarrow under the parquet optional extra."""
    reqs = requires("mt5cli") or []
    parquet_reqs = [r for r in reqs if "pyarrow" in r and "parquet" in r]
    assert parquet_reqs, "pyarrow not found in parquet optional extra"


def test_pyarrow_not_in_core_dependencies() -> None:
    """Pyarrow is not a core dependency; it belongs only in the parquet extra."""
    reqs = requires("mt5cli") or []
    core_reqs = [r for r in reqs if "extra ==" not in r]
    assert not any("pyarrow" in r for r in core_reqs), (
        "pyarrow should not appear in core dependencies"
    )
