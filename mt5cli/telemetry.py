"""Optional OpenTelemetry metrics for MT5 history and snapshot observability."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

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


class _InstrumentSpec(NamedTuple):
    """Declarative OTel instrument definition."""

    key: str
    kind: str  # "counter", "histogram", or "gauge"
    name: str
    description: str
    unit: str | None = None


_INSTRUMENT_SPECS: tuple[_InstrumentSpec, ...] = (
    _InstrumentSpec(
        "history_duration",
        "histogram",
        "mt5_history_update_duration_seconds",
        "Duration of incremental history update operations.",
        unit="s",
    ),
    _InstrumentSpec(
        "history_rows",
        "counter",
        "mt5_history_update_rows_total",
        "Rows written during incremental history updates.",
    ),
    _InstrumentSpec(
        "history_failures",
        "counter",
        "mt5_history_update_failures_total",
        "Number of incremental history update failures.",
    ),
    _InstrumentSpec(
        "snapshot_duration",
        "histogram",
        "mt5_snapshot_update_duration_seconds",
        "Duration of snapshot update operations.",
        unit="s",
    ),
    _InstrumentSpec(
        "snapshot_failures",
        "counter",
        "mt5_snapshot_update_failures_total",
        "Number of snapshot update failures.",
    ),
    _InstrumentSpec(
        "account_balance", "gauge", "mt5_account_balance", "Account balance."
    ),
    _InstrumentSpec("account_equity", "gauge", "mt5_account_equity", "Account equity."),
    _InstrumentSpec(
        "account_margin", "gauge", "mt5_account_margin", "Account margin used."
    ),
    _InstrumentSpec(
        "account_margin_free",
        "gauge",
        "mt5_account_margin_free",
        "Account free margin.",
    ),
    _InstrumentSpec(
        "account_margin_level",
        "gauge",
        "mt5_account_margin_level",
        "Account margin level as a percentage.",
    ),
    _InstrumentSpec(
        "position_profit",
        "gauge",
        "mt5_position_profit",
        "Floating profit for an open position.",
    ),
    _InstrumentSpec(
        "position_volume",
        "gauge",
        "mt5_position_volume",
        "Volume of an open position.",
    ),
    _InstrumentSpec(
        "terminal_connected",
        "gauge",
        "mt5_terminal_connected",
        "1 if the terminal is connected to the broker, 0 otherwise.",
    ),
    _InstrumentSpec(
        "terminal_trade_allowed",
        "gauge",
        "mt5_terminal_trade_allowed",
        "1 if trading is allowed by the broker server, 0 otherwise.",
    ),
    _InstrumentSpec(
        "terminal_trade_expert",
        "gauge",
        "mt5_terminal_trade_expert",
        "1 if Expert Advisor trading is enabled, 0 otherwise.",
    ),
    _InstrumentSpec(
        "last_successful_update",
        "gauge",
        "mt5_last_successful_update_timestamp",
        "Unix timestamp of the last successful history update.",
    ),
)


class _Mt5Metrics:
    """MT5 metric instrument registry.

    Holds references to OTel instruments keyed by :data:`_INSTRUMENT_SPECS`.
    All instruments are no-op until :meth:`configure` is called with a
    compatible meter object.
    """

    def __init__(self) -> None:
        self._instruments: dict[str, Any] = dict.fromkeys(
            (spec.key for spec in _INSTRUMENT_SPECS), _NOOP
        )

    def _instrument(self, key: str) -> Any:  # noqa: ANN401
        return self._instruments[key]

    def configure(self, meter: Any) -> None:  # noqa: ANN401
        """Set up metric instruments from a meter object.

        Args:
            meter: An OpenTelemetry ``Meter`` or duck-typed compatible object
                that supports ``create_counter``, ``create_histogram``, and
                ``create_gauge``.
        """
        for spec in _INSTRUMENT_SPECS:
            factory = getattr(meter, f"create_{spec.kind}")
            if spec.unit is not None:
                instrument = factory(
                    spec.name, unit=spec.unit, description=spec.description
                )
            else:
                instrument = factory(spec.name, description=spec.description)
            self._instruments[spec.key] = instrument

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
            self._instrument("history_duration").record(time.monotonic() - start, attrs)
            self._instrument("last_successful_update").set(time.time(), attrs)
        except Exception:
            self._instrument("history_failures").add(1, attrs)
            raise

    def add_history_rows(self, count: int, *, dataset: str) -> None:
        """Increment the history rows-written counter.

        Args:
            count: Number of rows written during this update.
            dataset: Dataset label (e.g. ``"rates"``).
        """
        self._instrument("history_rows").add(count, {"dataset": dataset})

    @contextmanager
    def record_snapshot_update(self) -> Iterator[None]:
        """Context manager recording snapshot update duration and failures.

        Yields:
            None inside the snapshot operation.
        """
        start = time.monotonic()
        try:
            yield
            self._instrument("snapshot_duration").record(time.monotonic() - start, {})
        except Exception:
            self._instrument("snapshot_failures").add(1, {})
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
        self._instrument("account_balance").set(balance, attrs)
        self._instrument("account_equity").set(equity, attrs)
        self._instrument("account_margin").set(margin, attrs)
        self._instrument("account_margin_free").set(margin_free, attrs)
        self._instrument("account_margin_level").set(margin_level, attrs)

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
        self._instrument("position_profit").set(profit, attrs)
        self._instrument("position_volume").set(volume, attrs)

    def record_terminal_state(
        self,
        *,
        connected: float,
        trade_allowed: float,
        trade_expert: float,
    ) -> None:
        """Emit terminal connection and trading status gauges.

        Args:
            connected: 1.0 if connected to the broker, 0.0 otherwise.
            trade_allowed: 1.0 if broker server allows trading, 0.0 otherwise.
            trade_expert: 1.0 if Expert Advisor trading is enabled, 0.0 otherwise.
        """
        self._instrument("terminal_connected").set(connected, {})
        self._instrument("terminal_trade_allowed").set(trade_allowed, {})
        self._instrument("terminal_trade_expert").set(trade_expert, {})


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
