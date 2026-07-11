"""Contract tests for the mt5cli public API and dataset schemas."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd
import pytest
from pdmt5 import Mt5RuntimeError
from pytest_mock import MockerFixture  # noqa: TC002

import mt5cli
import mt5cli.cli
import mt5cli.client
import mt5cli.history
import mt5cli.market_data
import mt5cli.observability
import mt5cli.trading
import mt5cli.utils
from mt5cli import (
    STABLE_SDK_EXPORTS,
    MT5Client,
    Mt5CliError,
    Mt5ConnectionError,
    Mt5SchemaError,
    mt5_session,
)
from mt5cli.contract import HistoryClient, ObservabilityClient
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
    """Normalization preserves mt5cli and unrelated application exceptions."""
    original = Mt5CliError("known")
    assert normalize_mt5_exception(original) is original
    application_error = ValueError("x")
    assert normalize_mt5_exception(application_error) is application_error


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


def test_retry_with_backoff_reraises_non_recoverable_errors() -> None:
    """Non-MT5 errors are not retried."""

    def _raise() -> None:
        message = "fatal"
        raise ValueError(message)

    with pytest.raises(ValueError, match="fatal"):
        retry_with_backoff(_raise, retry_count=2)


class TestStableSdkContract:
    """Tests for the documented stable downstream SDK contract."""

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

    @pytest.mark.parametrize(
        "name",
        [
            "create_trading_client",
            "mt5_trading_session",
            "fetch_latest_closed_rates_for_trading_client",
            "fetch_recent_history_deals_for_trading_client",
        ],
    )
    def test_removed_legacy_apis_are_not_public(self, name: str) -> None:
        """No public factory or session can reopen the raw-client lifecycle path."""
        assert name not in mt5cli.__all__
        assert name not in STABLE_SDK_EXPORTS
        assert not hasattr(mt5cli, name)

    def test_public_function_annotations_hide_raw_pdmt5_clients(self) -> None:
        """The root facade never exposes a low-level pdmt5 client type."""
        for name in mt5cli.__all__:
            value = getattr(mt5cli, name)
            if not callable(value) or inspect.isclass(value):
                continue
            annotations = inspect.get_annotations(value, eval_str=False)
            assert all(
                "Mt5DataClient" not in str(annotation)
                for annotation in annotations.values()
            ), name

    def test_documented_contract_identifies_runtime_export_set(self) -> None:
        """The public-contract document points to the same runtime authority."""
        contract_doc = (
            Path(__file__).parents[1] / "docs" / "api" / "public-contract.md"
        ).read_text(encoding="utf-8")
        assert "`mt5cli.STABLE_SDK_EXPORTS`" in contract_doc
        assert set(mt5cli.__all__) - {"STABLE_SDK_EXPORTS"} == set(STABLE_SDK_EXPORTS)
        undocumented = sorted(
            name for name in STABLE_SDK_EXPORTS if f"`{name}`" not in contract_doc
        )
        assert not undocumented, f"Undocumented stable exports: {undocumented}"


@pytest.mark.parametrize(
    "name",
    [
        "Mt5Config",
        "Mt5RuntimeError",
        "Mt5DataClient",
        "TIMEFRAME_MAP",
        "COPY_TICKS_MAP",
        "ORDER_TYPE_MAP",
    ],
)
def test_pdmt5_pass_through_names_stay_out_of_public_contract(name: str) -> None:
    """Downstream code must import pdmt5 types/constants directly from pdmt5.

    docs/api/public-contract.md documents that mt5cli is not a pass-through
    compatibility namespace for pdmt5; these names must never reappear on the
    mt5cli package root.
    """
    assert name not in STABLE_SDK_EXPORTS, (
        f"{name!r} should not be in STABLE_SDK_EXPORTS"
    )
    assert name not in mt5cli.__all__, f"{name!r} should not be in mt5cli.__all__"
    assert not hasattr(mt5cli, name), f"{name!r} should not be on the mt5cli root"


class TestSingleConnectionLifecycleContract:
    """Tests that mt5cli.client is the sole connection-lifecycle owner."""

    def test_sdk_compatibility_module_no_longer_exists(self) -> None:
        """The removed mt5cli.sdk module cannot be imported."""
        with pytest.raises(ModuleNotFoundError):
            import mt5cli.sdk  # pyright: ignore[reportMissingImports,reportUnusedImport]  # noqa: F401, PLC0415

    def test_package_root_mt5_session_is_the_client_module_session(self) -> None:
        """The package-root mt5_session is the same object as mt5cli.client's."""
        assert mt5_session is mt5cli.client.mt5_session

    def test_mt5_session_yields_mt5_client(self, mocker: MockerFixture) -> None:
        """The single public session factory always yields an MT5Client."""
        raw_client = mocker.MagicMock()
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)
        with mt5_session() as client:
            assert isinstance(client, MT5Client)

    def test_no_other_submodule_defines_a_public_mt5_session(self) -> None:
        """No mt5cli submodule other than client.py defines mt5_session."""
        submodules = [
            mt5cli.history,
            mt5cli.market_data,
            mt5cli.observability,
            mt5cli.trading,
            mt5cli.cli,
            mt5cli.utils,
        ]
        for module in submodules:
            if hasattr(module, "mt5_session"):
                assert module.mt5_session is mt5_session, (
                    f"{module.__name__} must re-export the canonical mt5_session, "
                    "not define an alternate one"
                )

    def test_no_public_factory_returns_a_raw_pdmt5_client(self) -> None:
        """No stable export's return annotation resolves to pdmt5.Mt5DataClient."""
        for name in STABLE_SDK_EXPORTS:
            value = getattr(mt5cli, name)
            if not callable(value) or inspect.isclass(value):
                continue
            annotations = inspect.get_annotations(value, eval_str=False)
            returns = str(annotations.get("return", ""))
            assert "Mt5DataClient" not in returns, name

    def test_caller_owned_client_is_not_initialized_or_shut_down(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A caller-owned client passed to mt5_session is used as-is."""
        raw_client = mocker.MagicMock()
        owned_client = MT5Client.from_connected_client(raw_client)
        with mt5_session(client=owned_client) as client:
            assert client is owned_client
        raw_client.initialize_and_login_mt5.assert_not_called()
        raw_client.shutdown.assert_not_called()


class TestHistoryAndObservabilityClientContracts:
    """Tests for the internal HistoryClient/ObservabilityClient protocols."""

    def test_protocols_are_defined(self) -> None:
        """HistoryClient/ObservabilityClient are defined as structural protocols."""
        assert issubclass(HistoryClient, Protocol)
        assert issubclass(ObservabilityClient, Protocol)

    @pytest.mark.parametrize(
        "method_name",
        [
            "copy_rates_range",
            "copy_ticks_range",
            "history_orders",
            "history_deals",
            "symbol_info_as_dict",
        ],
    )
    def test_mt5_client_satisfies_history_client_protocol(
        self,
        method_name: str,
    ) -> None:
        """MT5Client implements every method required by HistoryClient."""
        assert callable(getattr(MT5Client, method_name))

    @pytest.mark.parametrize(
        "method_name",
        ["account_info", "terminal_info", "positions", "orders"],
    )
    def test_mt5_client_satisfies_observability_client_protocol(
        self,
        method_name: str,
    ) -> None:
        """MT5Client implements every method required by ObservabilityClient."""
        assert callable(getattr(MT5Client, method_name))

    def test_update_history_does_not_annotate_client_as_object_or_mt5dataclient(
        self,
    ) -> None:
        """update_history's client parameter is a protocol, not object/Mt5DataClient."""
        annotations = inspect.get_annotations(mt5cli.update_history, eval_str=False)
        client_annotation = str(annotations["client"])
        assert client_annotation != "object"
        assert "Mt5DataClient" not in client_annotation

    def test_update_observability_does_not_annotate_client_as_object_or_mt5dataclient(
        self,
    ) -> None:
        """update_observability's client param is a protocol, not object/raw client."""
        annotations = inspect.get_annotations(
            mt5cli.update_observability, eval_str=False
        )
        client_annotation = str(annotations["client"])
        assert client_annotation != "object"
        assert "Mt5DataClient" not in client_annotation
