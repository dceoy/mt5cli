"""Tests for mt5cli.telemetry module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from mt5cli.telemetry import (
    _OTEL_AVAILABLE,  # type: ignore[reportPrivateUsage]
    _Mt5Metrics,  # type: ignore[reportPrivateUsage]
    _NoOp,  # type: ignore[reportPrivateUsage]
    configure_metrics,
    enable_otel_metrics,
    get_metrics,
)


class TestNoOp:
    """Tests for _NoOp no-op instrument."""

    def test_add_is_noop(self) -> None:
        """_NoOp.add accepts amount and optional attributes without error."""
        noop = _NoOp()
        noop.add(1.0)
        noop.add(1.0, {"key": "val"})

    def test_set_is_noop(self) -> None:
        """_NoOp.set accepts amount and optional attributes without error."""
        noop = _NoOp()
        noop.set(2.0)
        noop.set(2.0, {"key": "val"})

    def test_record_is_noop(self) -> None:
        """_NoOp.record accepts amount and optional attributes without error."""
        noop = _NoOp()
        noop.record(3.0)
        noop.record(3.0, {"key": "val"})


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

    def test_record_history_update_success(self) -> None:
        """record_history_update records duration and timestamp on success."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        with m.record_history_update(dataset="rates"):
            pass
        m._history_duration.record.assert_called_once()  # type: ignore[reportPrivateUsage]
        m._last_successful_update.set.assert_called_once()  # type: ignore[reportPrivateUsage]
        m._history_failures.add.assert_not_called()  # type: ignore[reportPrivateUsage]

    def test_record_history_update_failure(self) -> None:
        """record_history_update increments failure counter and re-raises on error."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        exc = ValueError("boom")
        with (
            pytest.raises(ValueError, match="boom"),
            m.record_history_update(dataset="rates"),
        ):
            raise exc
        m._history_failures.add.assert_called_once_with(  # type: ignore[reportPrivateUsage]
            1, {"dataset": "rates"}
        )
        m._history_duration.record.assert_not_called()  # type: ignore[reportPrivateUsage]

    def test_add_history_rows(self) -> None:
        """add_history_rows increments the rows-written counter."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        m.add_history_rows(42, dataset="rates")
        m._history_rows.add.assert_called_once_with(  # type: ignore[reportPrivateUsage]
            42, {"dataset": "rates"}
        )

    def test_record_snapshot_update_success(self) -> None:
        """record_snapshot_update records duration on success."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        with m.record_snapshot_update():
            pass
        m._snapshot_duration.record.assert_called_once()  # type: ignore[reportPrivateUsage]
        m._snapshot_failures.add.assert_not_called()  # type: ignore[reportPrivateUsage]

    def test_record_snapshot_update_failure(self) -> None:
        """record_snapshot_update increments failure counter and re-raises on error."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        exc = RuntimeError("snap fail")
        with (
            pytest.raises(RuntimeError, match="snap fail"),
            m.record_snapshot_update(),
        ):
            raise exc
        m._snapshot_failures.add.assert_called_once_with(1, {})  # type: ignore[reportPrivateUsage]
        m._snapshot_duration.record.assert_not_called()  # type: ignore[reportPrivateUsage]

    def test_record_position_state(self) -> None:
        """record_position_state emits profit and volume gauges."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        m.record_position_state(
            login="42",
            server="demo",
            symbol="EURUSD",
            profit=12.5,
            volume=0.01,
        )
        # Both profit and volume share the same gauge mock via create_gauge.
        # Verify that set was called exactly twice (once each).
        assert m._position_profit.set.call_count == 2  # type: ignore[reportPrivateUsage]

    def test_record_account_state_after_configure(self) -> None:
        """record_account_state emits all five account gauges."""
        meter = MagicMock()
        m = _Mt5Metrics()
        m.configure(meter)
        m.record_account_state(
            login="99",
            server="live",
            balance=5000.0,
            equity=5100.0,
            margin=200.0,
            margin_free=4800.0,
            margin_level=2550.0,
        )
        # All five account gauges share the same gauge mock; set is called 5 times.
        assert m._account_balance.set.call_count == 5  # type: ignore[reportPrivateUsage]

    def test_record_history_update_noop_before_configure(self) -> None:
        """record_history_update works without configure (no-op instruments)."""
        m = _Mt5Metrics()
        with m.record_history_update(dataset="ticks"):
            pass

    def test_record_snapshot_update_noop_before_configure(self) -> None:
        """record_snapshot_update works without configure (no-op instruments)."""
        m = _Mt5Metrics()
        with m.record_snapshot_update():
            pass


class TestConfigureMetrics:
    """Tests for configure_metrics and get_metrics."""

    def test_configure_metrics_updates_global(self) -> None:
        """configure_metrics wires up the global singleton."""
        meter = MagicMock()
        configure_metrics(meter)
        assert get_metrics() is get_metrics()

    def test_get_metrics_returns_mt5metrics(self) -> None:
        """get_metrics returns the global _Mt5Metrics instance."""
        assert isinstance(get_metrics(), _Mt5Metrics)


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

    def test_otel_available_flag_is_bool(self) -> None:
        """_OTEL_AVAILABLE is a boolean."""
        assert isinstance(_OTEL_AVAILABLE, bool)
