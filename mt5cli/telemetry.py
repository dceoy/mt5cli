"""Optional OpenTelemetry metrics for MT5 history and snapshot observability."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_otel_available_flag = False

try:
    import opentelemetry.metrics as _otel_metrics_mod
    from opentelemetry.sdk.metrics import MeterProvider as _OtelMeterProvider
    from opentelemetry.sdk.metrics.export import (
        PeriodicExportingMetricReader as _OtelPeriodicReader,
    )
    from opentelemetry.sdk.resources import Resource as _OtelResource

    _otel_available_flag = True
except ImportError:  # pragma: no cover
    _otel_metrics_mod = None  # type: ignore[assignment]
    _OtelMeterProvider = None  # type: ignore[assignment]
    _OtelPeriodicReader = None  # type: ignore[assignment]
    _OtelResource = None  # type: ignore[assignment]

_OTEL_AVAILABLE: bool = _otel_available_flag

try:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # type: ignore[import]
        OTLPMetricExporter as _OtelOTLPExporter,  # type: ignore[reportUnknownVariableType]
    )
except ImportError:  # pragma: no cover
    _OtelOTLPExporter = None  # type: ignore[assignment, misc]


class _NoOp:
    """No-op instrument that silently ignores all calls."""

    def add(
        self,
        amount: float,
        attributes: dict[str, str] | None = None,
    ) -> None:
        """No-op add."""

    def set(
        self,
        amount: float,
        attributes: dict[str, str] | None = None,
    ) -> None:
        """No-op set."""

    def record(
        self,
        amount: float,
        attributes: dict[str, str] | None = None,
    ) -> None:
        """No-op record."""


_NOOP: _NoOp = _NoOp()


class _Mt5Metrics:
    """MT5 metric instrument registry.

    Holds references to OTel instruments. All instruments are no-op until
    :meth:`configure` is called with a compatible meter object.
    """

    def __init__(self) -> None:
        self._history_duration: Any = _NOOP
        self._history_rows: Any = _NOOP
        self._history_failures: Any = _NOOP
        self._snapshot_duration: Any = _NOOP
        self._snapshot_failures: Any = _NOOP
        self._account_balance: Any = _NOOP
        self._account_equity: Any = _NOOP
        self._account_margin: Any = _NOOP
        self._account_margin_free: Any = _NOOP
        self._account_margin_level: Any = _NOOP
        self._position_profit: Any = _NOOP
        self._position_volume: Any = _NOOP
        self._last_successful_update: Any = _NOOP

    def configure(self, meter: Any) -> None:  # noqa: ANN401
        """Set up metric instruments from a meter object.

        Args:
            meter: An OpenTelemetry ``Meter`` or duck-typed compatible object
                that supports ``create_counter``, ``create_histogram``, and
                ``create_gauge``.
        """
        self._history_duration = meter.create_histogram(
            "mt5_history_update_duration_seconds",
            unit="s",
            description="Duration of incremental history update operations.",
        )
        self._history_rows = meter.create_counter(
            "mt5_history_update_rows_total",
            description="Rows written during incremental history updates.",
        )
        self._history_failures = meter.create_counter(
            "mt5_history_update_failures_total",
            description="Number of incremental history update failures.",
        )
        self._snapshot_duration = meter.create_histogram(
            "mt5_snapshot_update_duration_seconds",
            unit="s",
            description="Duration of snapshot update operations.",
        )
        self._snapshot_failures = meter.create_counter(
            "mt5_snapshot_update_failures_total",
            description="Number of snapshot update failures.",
        )
        self._account_balance = meter.create_gauge(
            "mt5_account_balance",
            description="Account balance.",
        )
        self._account_equity = meter.create_gauge(
            "mt5_account_equity",
            description="Account equity.",
        )
        self._account_margin = meter.create_gauge(
            "mt5_account_margin",
            description="Account margin used.",
        )
        self._account_margin_free = meter.create_gauge(
            "mt5_account_margin_free",
            description="Account free margin.",
        )
        self._account_margin_level = meter.create_gauge(
            "mt5_account_margin_level",
            description="Account margin level as a percentage.",
        )
        self._position_profit = meter.create_gauge(
            "mt5_position_profit",
            description="Floating profit for an open position.",
        )
        self._position_volume = meter.create_gauge(
            "mt5_position_volume",
            description="Volume of an open position.",
        )
        self._last_successful_update = meter.create_gauge(
            "mt5_last_successful_update_timestamp",
            description="Unix timestamp of the last successful history update.",
        )

    @contextmanager
    def record_history_update(
        self,
        *,
        dataset: str,
    ) -> Iterator[None]:
        """Context manager recording history update duration and failures.

        Args:
            dataset: Dataset label (e.g. ``"rates"``).

        Yields:
            None inside the update operation.
        """
        attrs = {"dataset": dataset}
        start = time.monotonic()
        try:
            yield
            self._history_duration.record(time.monotonic() - start, attrs)
            self._last_successful_update.set(time.time(), attrs)
        except Exception:
            self._history_failures.add(1, attrs)
            raise

    @contextmanager
    def record_snapshot_update(self) -> Iterator[None]:
        """Context manager recording snapshot update duration and failures.

        Yields:
            None inside the snapshot operation.
        """
        start = time.monotonic()
        try:
            yield
            self._snapshot_duration.record(time.monotonic() - start, {})
        except Exception:
            self._snapshot_failures.add(1, {})
            raise

    def record_account_state(
        self,
        *,
        login: str,
        server: str,
        balance: float,
        equity: float,
        margin: float,
        margin_free: float,
        margin_level: float,
    ) -> None:
        """Emit account metric gauges.

        Args:
            login: Account login number (as string; not a password or secret).
            server: Broker server name.
            balance: Account balance.
            equity: Account equity.
            margin: Margin used.
            margin_free: Free margin.
            margin_level: Margin level percentage.
        """
        attrs: dict[str, str] = {"login": login, "server": server}
        self._account_balance.set(balance, attrs)
        self._account_equity.set(equity, attrs)
        self._account_margin.set(margin, attrs)
        self._account_margin_free.set(margin_free, attrs)
        self._account_margin_level.set(margin_level, attrs)

    def record_position_state(
        self,
        *,
        login: str,
        server: str,
        symbol: str,
        profit: float,
        volume: float,
    ) -> None:
        """Emit position metric gauges.

        Args:
            login: Account login number (as string).
            server: Broker server name.
            symbol: Position symbol.
            profit: Floating profit/loss.
            volume: Position volume.
        """
        attrs: dict[str, str] = {"login": login, "server": server, "symbol": symbol}
        self._position_profit.set(profit, attrs)
        self._position_volume.set(volume, attrs)


_metrics = _Mt5Metrics()


def configure_metrics(meter: Any) -> None:  # noqa: ANN401
    """Configure MT5 metrics using the provided meter.

    Args:
        meter: An OpenTelemetry ``Meter`` or duck-typed compatible object.
    """
    _metrics.configure(meter)


def enable_otel_metrics(
    service_name: str = "mt5cli",
    readers: list[Any] | None = None,
) -> None:
    """Enable OTel metrics by wiring up an SDK ``MeterProvider`` pipeline.

    Requires the ``otel`` optional dependency group:
    ``pip install "mt5cli[otel]"``.

    Args:
        service_name: OTel meter/service name used for the ``Resource`` and
            the meter itself.
        readers: Optional list of metric readers. When *None* (the default),
            a :class:`~opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader`
            backed by an OTLP HTTP exporter is created automatically
            (reads the endpoint from ``OTEL_EXPORTER_OTLP_ENDPOINT``).
            Pass a custom list (e.g. ``InMemoryMetricReader`` for tests)
            to override.

    Raises:
        ImportError: If ``opentelemetry-api`` is not installed, or if
            ``readers`` is *None* and
            ``opentelemetry-exporter-otlp-proto-http`` is not installed.
    """
    if not _OTEL_AVAILABLE:
        msg = (
            "opentelemetry-api is not installed. "
            'Install it with: pip install "mt5cli[otel]"'
        )
        raise ImportError(msg)
    if readers is None:
        if _OtelOTLPExporter is None:
            msg = (
                "opentelemetry-exporter-otlp-proto-http is required for the "
                "default OTLP export pipeline. "
                'Install it with: pip install "mt5cli[otel]" or pass a '
                "custom readers list."
            )
            raise ImportError(msg)
        readers = [_OtelPeriodicReader(_OtelOTLPExporter())]  # type: ignore[misc]
    resource = _OtelResource.create({"service.name": service_name})  # type: ignore[union-attr]
    provider = _OtelMeterProvider(resource=resource, metric_readers=readers)  # type: ignore[misc]
    _otel_metrics_mod.set_meter_provider(provider)  # type: ignore[union-attr]
    meter = provider.get_meter(service_name)
    configure_metrics(meter)


def get_metrics() -> _Mt5Metrics:
    """Return the global :class:`_Mt5Metrics` instance.

    Returns:
        The global metric registry (no-op until :func:`configure_metrics` is
        called).
    """
    return _metrics
