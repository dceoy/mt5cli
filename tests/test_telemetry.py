"""Tests for mt5cli.telemetry module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from pytest_mock import MockerFixture  # noqa: TC002

from mt5cli.telemetry import (
    _Mt5Metrics,  # type: ignore[reportPrivateUsage]
    configure_metrics,
    enable_otel_metrics,
)


class TestMt5Metrics:
    """Tests for _Mt5Metrics."""

    def test_default_instruments_are_noop(self) -> None:
        """Default _Mt5Metrics methods do not raise before configure is called."""
        m = _Mt5Metrics()
        m.record_account_state(
            login="123",
            server="demo",
            balance=1000.0,
            equity=1050.0,
            margin=100.0,
            margin_free=950.0,
            margin_level=1050.0,
        )

    def test_configure_calls_meter(self) -> None:
        """configure() calls create_histogram, create_counter, create_gauge on meter."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        assert meter.create_histogram.called
        assert meter.create_counter.called
        assert meter.create_gauge.called

    @pytest.mark.parametrize(
        (
            "method",
            "kwargs",
            "duration_attr",
            "failures_attr",
            "last_success_attr",
        ),
        [
            pytest.param(
                "record_history_update",
                {"dataset": "rates"},
                "_history_duration",
                "_history_failures",
                "_last_successful_update",
                id="history-update",
            ),
            pytest.param(
                "record_snapshot_update",
                {},
                "_snapshot_duration",
                "_snapshot_failures",
                None,
                id="snapshot-update",
            ),
        ],
    )
    def test_record_update_success(
        self,
        method: str,
        kwargs: dict[str, str],
        duration_attr: str,
        failures_attr: str,
        last_success_attr: str | None,
    ) -> None:
        """record_*_update records duration and timestamp on success."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        with getattr(m, method)(**kwargs):
            pass
        getattr(m, duration_attr).record.assert_called_once()  # type: ignore[reportPrivateUsage]
        getattr(m, failures_attr).add.assert_not_called()  # type: ignore[reportPrivateUsage]
        if last_success_attr is not None:
            getattr(m, last_success_attr).set.assert_called_once()  # type: ignore[reportPrivateUsage]

    @pytest.mark.parametrize(
        (
            "method",
            "kwargs",
            "exc",
            "duration_attr",
            "failures_attr",
            "failure_labels",
        ),
        [
            pytest.param(
                "record_history_update",
                {"dataset": "rates"},
                ValueError("boom"),
                "_history_duration",
                "_history_failures",
                {"dataset": "rates"},
                id="history-update",
            ),
            pytest.param(
                "record_snapshot_update",
                {},
                RuntimeError("snap fail"),
                "_snapshot_duration",
                "_snapshot_failures",
                {},
                id="snapshot-update",
            ),
        ],
    )
    def test_record_update_failure(
        self,
        method: str,
        kwargs: dict[str, str],
        exc: BaseException,
        duration_attr: str,
        failures_attr: str,
        failure_labels: dict[str, str],
    ) -> None:
        """record_*_update increments failure counter and re-raises on error."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        with (
            pytest.raises(type(exc), match=str(exc)),
            getattr(m, method)(**kwargs),
        ):
            raise exc
        getattr(m, failures_attr).add.assert_called_once_with(  # type: ignore[reportPrivateUsage]
            1,
            failure_labels,
        )
        getattr(m, duration_attr).record.assert_not_called()  # type: ignore[reportPrivateUsage]

    def test_add_history_rows(self) -> None:
        """add_history_rows increments the rows-written counter."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        m.add_history_rows(42, dataset="rates")
        m._history_rows.add.assert_called_once_with(  # type: ignore[reportPrivateUsage]
            42, {"dataset": "rates"}
        )

    @pytest.mark.parametrize(
        ("method", "kwargs", "gauge_attr", "expected_set_count"),
        [
            pytest.param(
                "record_position_state",
                {
                    "login": "42",
                    "server": "demo",
                    "symbol": "EURUSD",
                    "profit": 12.5,
                    "volume": 0.01,
                },
                "_position_profit",
                2,
                id="position-state",
            ),
            pytest.param(
                "record_terminal_state",
                {
                    "connected": 1.0,
                    "trade_allowed": 1.0,
                    "trade_expert": 0.0,
                },
                "_terminal_connected",
                3,
                id="terminal-state",
            ),
            pytest.param(
                "record_account_state",
                {
                    "login": "99",
                    "server": "live",
                    "balance": 5000.0,
                    "equity": 5100.0,
                    "margin": 200.0,
                    "margin_free": 4800.0,
                    "margin_level": 2550.0,
                },
                "_account_balance",
                5,
                id="account-state",
            ),
        ],
    )
    def test_record_state_emits_gauges(
        self,
        method: str,
        kwargs: dict[str, float | str],
        gauge_attr: str,
        expected_set_count: int,
    ) -> None:
        """record_*_state emits the expected gauge set calls after configure."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        getattr(m, method)(**kwargs)
        # Related gauges share the same create_gauge mock; verify total set calls.
        assert (  # type: ignore[reportPrivateUsage]
            getattr(m, gauge_attr).set.call_count == expected_set_count
        )

    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("record_history_update", {"dataset": "ticks"}),
            ("record_snapshot_update", {}),
        ],
    )
    def test_record_update_noop_before_configure(
        self,
        method: str,
        kwargs: dict[str, str],
    ) -> None:
        """record_*_update works without configure (no-op instruments)."""
        m = _Mt5Metrics()
        with getattr(m, method)(**kwargs):
            pass


class TestConfigureMetrics:
    """Tests for the module-level configure_metrics wiring function."""

    def test_configure_metrics_delegates_to_global_registry(
        self,
        mocker: MockerFixture,
    ) -> None:
        """configure_metrics(meter) delegates exactly once to the registry."""
        meter = MagicMock()
        configure_spy = mocker.patch("mt5cli.telemetry._metrics.configure")
        configure_metrics(meter)
        configure_spy.assert_called_once_with(meter)


class TestEnableOtelMetrics:
    """Tests for enable_otel_metrics."""

    def test_enable_raises_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """enable_otel_metrics raises ImportError when OTel is not installed."""
        monkeypatch.setattr("mt5cli.telemetry._OTEL_AVAILABLE", False)
        with pytest.raises(ImportError, match="opentelemetry-api"):
            enable_otel_metrics()

    def test_enable_configures_sdk_pipeline_with_readers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """enable_otel_metrics wires up an SDK MeterProvider with supplied readers."""
        mock_mod = MagicMock()
        monkeypatch.setattr("mt5cli.telemetry._OTEL_AVAILABLE", True)
        monkeypatch.setattr("mt5cli.telemetry._otel_metrics_mod", mock_mod)
        reader = InMemoryMetricReader()
        enable_otel_metrics("my-service", readers=[reader])
        mock_mod.set_meter_provider.assert_called_once()
        provider = mock_mod.set_meter_provider.call_args[0][0]
        assert provider.get_meter("my-service") is not None

    def test_enable_default_readers_uses_otlp(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """enable_otel_metrics with no readers creates an OTLP pipeline by default."""
        mock_mod = MagicMock()
        monkeypatch.setattr("mt5cli.telemetry._OTEL_AVAILABLE", True)
        monkeypatch.setattr("mt5cli.telemetry._otel_metrics_mod", mock_mod)
        monkeypatch.setattr("mt5cli.telemetry._OtelOTLPExporter", MagicMock())
        enable_otel_metrics("my-service")
        mock_mod.set_meter_provider.assert_called_once()

    def test_enable_default_readers_raises_when_otlp_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """enable_otel_metrics raises ImportError when the OTLP exporter is missing."""
        mock_mod = MagicMock()
        monkeypatch.setattr("mt5cli.telemetry._OTEL_AVAILABLE", True)
        monkeypatch.setattr("mt5cli.telemetry._otel_metrics_mod", mock_mod)
        monkeypatch.setattr("mt5cli.telemetry._OtelOTLPExporter", None)
        with pytest.raises(ImportError, match="opentelemetry-exporter-otlp-proto-http"):
            enable_otel_metrics()
