# Telemetry

::: mt5cli.telemetry

## Enabling OpenTelemetry metrics

Install the optional exporter dependencies with:

```bash
uv add 'mt5cli[otel]'
```

Then enable the default OTLP HTTP pipeline:

```python
from mt5cli.telemetry import enable_otel_metrics

enable_otel_metrics(service_name="mt5cli")
```

When `readers=None`, `enable_otel_metrics()` builds a
`PeriodicExportingMetricReader` backed by the OTLP HTTP exporter and reads the
endpoint from `OTEL_EXPORTER_OTLP_ENDPOINT`.

If your application already owns an OpenTelemetry `Meter`, wire mt5cli into it
directly with `configure_metrics(meter)`.

## Emitted metric names

`enable_otel_metrics()` / `configure_metrics()` register these instruments:

- `mt5_history_update_duration_seconds`
- `mt5_history_update_rows_total`
- `mt5_history_update_failures_total`
- `mt5_snapshot_update_duration_seconds`
- `mt5_snapshot_update_failures_total`
- `mt5_account_balance`
- `mt5_account_equity`
- `mt5_account_margin`
- `mt5_account_margin_free`
- `mt5_account_margin_level`
- `mt5_position_profit`
- `mt5_position_volume`
- `mt5_terminal_connected`
- `mt5_terminal_trade_allowed`
- `mt5_terminal_trade_expert`
- `mt5_last_successful_update_timestamp`

The history metrics use a `dataset` attribute. Account and position gauges add
labels such as `login`, `server`, and `symbol` where applicable.
