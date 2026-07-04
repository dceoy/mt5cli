# mt5cli

[![CI/CD](https://github.com/dceoy/mt5cli/actions/workflows/ci.yml/badge.svg)](https://github.com/dceoy/mt5cli/actions/workflows/ci.yml)

Generic MT5 data and execution infrastructure for Python applications. Export from the CLI or import a small, stable Python API in downstream packages.

The [Public API Contract](docs/api/public-contract.md) lists stable SDK exports (`mt5cli.STABLE_SDK_EXPORTS`), CLI commands, internal helpers, and responsibilities that remain out of scope (strategy logic, backtests, optimization).

Built on top of [pdmt5](https://github.com/dceoy/pdmt5), a pandas-based data handler for MetaTrader 5.

## Architecture

- **pdmt5** — canonical MT5 client, DataFrame/trading primitives, and MT5 constant parsing (`TIMEFRAME_*`, `COPY_TICKS_*`, order types).
- **mt5cli** — public `MT5Client` API, standardized dataset schemas, storage helpers, CLI commands, and SQLite history collection built on pdmt5.
- **mt5api** — sibling HTTP adapter for remote MT5 access; not a dependency of mt5cli.

## Features

- **Multi-format export**: CSV, JSON, Parquet, and SQLite3 output formats
- **Auto-detection**: Format detection from file extensions
- **Comprehensive data access**: Rates, ticks, account info, symbols, orders, positions, and trading history
- **Flexible timeframes**: Named timeframes (M1, H1, D1, etc.) and numeric values
- **Connection management**: Optional credentials, server, and timeout configuration
- **SQLite rate loading**: Load mt5cli-managed rate tables/views for offline workflows

## Installation

```bash
pip install -U mt5cli MetaTrader5
```

Parquet export is not included by default. To enable it, install the `parquet` extra:

```bash
pip install -U "mt5cli[parquet]" MetaTrader5
```

## Python API (downstream packages)

Import `MT5Client` for generic MT5 data access, schema normalization, and optional order primitives.

```python
from datetime import UTC, datetime
from pathlib import Path

from mt5cli import (
    MT5Client,
    build_config,
    collect_history,
    mt5_session,
    update_history_with_config,
)
from mt5cli.schemas import DataKind, normalize_dataframe
from mt5cli.utils import Dataset, export_dataframe

# Persistent session for multiple calls
with mt5_session(build_config(login=12345, server="Broker-Demo")) as client:
    rates = client.copy_rates_range(
        "EURUSD",
        timeframe="H1",
        date_from="2024-01-01",
        date_to="2024-02-01",
    )
    positions = client.positions()
    check = client.order_check({"action": 1, "symbol": "EURUSD", "volume": 0.1})

# Normalize MT5 frames to the public schema contract before storage
closed_rates = normalize_dataframe(
    rates, DataKind.rates, symbol="EURUSD", timeframe="H1"
)
export_dataframe(closed_rates, Path("rates.csv"), "csv")

# Bulk SQLite history (same behavior as collect-history CLI command)
collect_history(
    Path("history.db"),
    symbols=["EURUSD"],
    date_from=datetime(2024, 1, 1, tzinfo=UTC),
    date_to=datetime(2024, 2, 1, tzinfo=UTC),
    datasets={Dataset.rates, Dataset.history_deals},
)

# Incremental append for automated pipelines
update_history_with_config(
    output="history.db",
    symbols=["EURUSD"],
    config=build_config(login=12345),
)
```

Schema contracts live in `mt5cli.schemas` (`DataKind`, `validate_schema`, `normalize_dataframe`). Export and storage helpers are in `mt5cli.utils` (`Dataset`, `export_dataframe`) and `mt5cli.history`.

`MT5Client.order_send()` is a live execution primitive: it can place real trades on the connected account. mt5cli does not implement strategy logic, signal generation, backtesting, or optimization — downstream applications must gate live execution explicitly.

### Trading lifecycle and state helpers

Trading applications can depend on `mt5cli` imports only; terminal path,
credentials, server, and timeout are forwarded to `pdmt5.Mt5Config`, numeric
login strings are coerced to integers, and empty login strings are treated as
unset. Pass `allow_whole_dollar_env=True` to expand `${ENV_VAR}` and bare
`$ENV_NAME` placeholders in connection string parameters before coercion.

```python
from mt5cli import (
    build_config,
    calculate_spread_ratio,
    create_trading_client,
    get_account_snapshot,
    mt5_trading_session,
)

# Login from environment — numeric string is coerced to int automatically
config = build_config(login="$MT5_LOGIN", allow_whole_dollar_env=True)

with mt5_trading_session(
    path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
    login="12345",
    password="from-env-or-secret-store",
    server="Broker-Demo",
) as client:
    account = get_account_snapshot(client)
    spread = calculate_spread_ratio(client, "EURUSD")

client = create_trading_client(login=12345, server="Broker-Demo")
try:
    positions = client.positions_get_as_df(symbol="EURUSD")
finally:
    client.shutdown()
```

## CLI usage

```bash
# Export account information to CSV
mt5cli -o account.csv account-info

# Export EURUSD M1 rates to Parquet
mt5cli -o rates.parquet rates-from --symbol EURUSD --timeframe M1 \
  --date-from 2024-01-01 --count 1000

# Export ticks to JSON
mt5cli -o ticks.json ticks-from --symbol EURUSD \
  --date-from 2024-01-01 --count 500 --flags ALL

# Export symbols to SQLite3 with custom table name
mt5cli -o data.db --table symbols symbols --group "*USD*"

# Export with connection credentials from env or placeholders
MT5_LOGIN=12345 MT5_PASSWORD=secret MT5_SERVER=MyBroker-Demo \
  mt5cli -o positions.csv positions
MT5_PATH="/path/to/terminal64.exe" mt5cli -o positions.csv positions
```

Run as a Python module:

```bash
python -m mt5cli -o account.csv account-info
```

## Commands

| Command                | Description                                                                                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rates-from`           | Export rates from a start date                                                                                                                        |
| `rates-from-pos`       | Export rates from a start position                                                                                                                    |
| `latest-rates`         | Export latest rates from a start position                                                                                                             |
| `rates-range`          | Export rates for a date range                                                                                                                         |
| `ticks-from`           | Export ticks from a start date                                                                                                                        |
| `ticks-range`          | Export ticks for a date range                                                                                                                         |
| `ticks-recent`         | Export ticks from a recent trailing window                                                                                                            |
| `account-info`         | Export account information                                                                                                                            |
| `terminal-info`        | Export terminal information                                                                                                                           |
| `version`              | Export MetaTrader 5 version information                                                                                                               |
| `last-error`           | Export the last error information                                                                                                                     |
| `symbols`              | Export symbol list                                                                                                                                    |
| `symbol-info`          | Export symbol details                                                                                                                                 |
| `symbol-info-tick`     | Export the last tick for a symbol                                                                                                                     |
| `minimum-margins`      | Export minimum-volume buy and sell margin requirements                                                                                                |
| `market-book`          | Export market depth (order book)                                                                                                                      |
| `orders`               | Export active orders                                                                                                                                  |
| `positions`            | Export open positions                                                                                                                                 |
| `history-orders`       | Export historical orders                                                                                                                              |
| `history-deals`        | Export historical deals                                                                                                                               |
| `recent-history-deals` | Export historical deals from a recent trailing window                                                                                                 |
| `mt5-summary`          | Export terminal/account status summary                                                                                                                |
| `order-check`          | Check funds sufficiency for a trade request                                                                                                           |
| `order-send`           | Send a raw trade request to the trade server (`--yes` required; expert path)                                                                          |
| `close-positions`      | Close open positions by `--symbol` or `--ticket` (`--yes` required for live; `--dry-run` available; optional `--deviation` / `--comment` / `--magic`) |
| `collect-history`      | Collect rates, history-orders, and history-deals for one or more symbols into a single SQLite database (ticks opt-in via `--dataset ticks`)           |
| `history-gaps`         | Export a SQLite-only one-row-per-gap report from managed rate compatibility views without connecting to MT5                                           |
| `grafana-schema`       | Create or refresh Grafana-ready views and indexes in an existing SQLite database (idempotent, no MT5 connection)                                      |
| `snapshot`             | Snapshot current account, position, order, and terminal state into SQLite for live Grafana dashboards                                                 |

Use `order-check` to validate a request payload before running `order-send --yes`.
`close-positions` is the safer high-level alternative that builds correct close
requests automatically. At least one `--symbol` or `--ticket` must be provided.
CLI connection flags fall back to `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`,
and `MT5_PATH` when unset, and explicit CLI values still win.

### `history-gaps`

Inspect collected SQLite rate views offline and export one row per detected gap.
For managed compatibility views, the command infers bar spacing from the view
name. Use `--granularity-seconds` for custom tables or views.

```bash
mt5cli -o gaps.json history-gaps --sqlite3 history.db
mt5cli -o eurusd.csv history-gaps --sqlite3 history.db --table rate_EURUSD__M1_1
```

### `collect-history`

Collect several historical datasets per symbol into one SQLite database in a single MT5 session. Pick datasets with repeatable `--dataset` (default: `rates`, `history-orders`, `history-deals`; add `--dataset ticks` when tick-level history is required — tick data can grow the SQLite database quickly), choose conflict behavior with `--if-exists append|replace|fail` (default: `fail`), and optionally derive `cash_events` / `positions_reconstructed` views from `history_deals` via `--with-views`.

```bash
mt5cli -o history.db collect-history \
  --symbol EURUSD --symbol GBPUSD \
  --date-from 2024-01-01 --date-to 2024-02-01 \
  --dataset rates --dataset history-deals \
  --timeframe M1 --flags ALL --if-exists append --with-views
```

History orders and deals are fetched per symbol and concatenated, so the symbol filter is applied consistently across all datasets. The `cash_events` view is derived from symbol-filtered `history_deals`, so account-level cash events with empty or non-matching symbols may be excluded. The `rates` table records the requested `timeframe` so appended runs at different timeframes remain distinguishable. The `positions_reconstructed` view aggregates trade deals by `position_id`, excludes positions without closing-side entries, and uses volume-weighted open/close prices; reversal deals (`DEAL_ENTRY_INOUT`) are reported via `volume_reversal` / `reversal_count` columns.

### Grafana-ready SQLite dashboards

mt5cli can prepare a SQLite database for use as a Grafana datasource (via the [SQLite plugin](https://grafana.com/grafana/plugins/frser-sqlite-datasource/) or similar). Most `grafana_*` views expose an integer epoch-second `time` column for use in Grafana time-series panels. Two views (`grafana_realized_pnl`, `grafana_trade_stats`) are static symbol-level summaries with no `time` column — use them in table or stat panels.

#### Prepare the schema (idempotent, no MT5 connection needed)

```bash
mt5cli -o history.db grafana-schema
```

This creates snapshot tables (`account_snapshots`, `position_snapshots`, `order_snapshots`, `terminal_snapshots`, `snapshot_runs`) and all `grafana_*` views and indexes in the SQLite database. Safe to run repeatedly — all operations are idempotent.

#### Snapshot current account state

```bash
mt5cli -o history.db snapshot \
  --symbol JP225 --symbol HK50 --symbol NL25 \
  --with-account --with-positions --with-orders --with-terminal \
  --with-grafana-schema
```

Appends one timestamped row per data type. Never places orders or modifies trading state. Run periodically (e.g. from a cron job or a loop) to build a time-series account history.

#### SDK usage

```python
from pdmt5 import Mt5DataClient, Mt5Config
from mt5cli import update_observability, update_observability_with_config

# Reuse an already-connected client
client = Mt5DataClient(config=Mt5Config(login=12345))
client.initialize_and_login_mt5()
try:
    update_observability(
        client=client,
        output="history.db",
        symbols=["EURUSD", "GBPUSD"],  # optional position/order filter
        include_account=True,
        include_positions=True,
        include_orders=True,
        include_terminal=True,
        with_grafana_schema=True,
    )
finally:
    client.shutdown()

# Standalone wrapper that opens/closes MT5 automatically
update_observability_with_config(
    output="history.db",
    config=Mt5Config(login=12345),
)
```

#### Available Grafana views

**Time-series views** (integer epoch-second `time` column; snapshot views also expose `run_id`):

| View                         | Source               | Description                                                |
| ---------------------------- | -------------------- | ---------------------------------------------------------- |
| `grafana_rates`              | `rates`              | OHLCV bars with integer epoch `time`                       |
| `grafana_ticks`              | `ticks`              | Tick data with integer epoch `time`                        |
| `grafana_history_deals`      | `history_deals`      | All deals with epoch `time`                                |
| `grafana_history_orders`     | `history_orders`     | All historical orders; adds epoch `time` from `time_setup` |
| `grafana_trade_deals`        | `history_deals`      | Trade deals only (`type IN (0,1)`)                         |
| `grafana_cash_events`        | `history_deals`      | Non-trade deals (deposits, dividends, etc.)                |
| `grafana_symbol_pnl`         | `history_deals`      | Per-close-deal profit/loss per symbol                      |
| `grafana_account_snapshots`  | `account_snapshots`  | Account balance/equity/margin time series                  |
| `grafana_position_snapshots` | `position_snapshots` | Open position snapshots over time                          |
| `grafana_order_snapshots`    | `order_snapshots`    | Active order snapshots over time                           |
| `grafana_terminal_snapshots` | `terminal_snapshots` | Terminal connectivity snapshots                            |

**Static summary views** (no `time` column; use in table or stat panels, not time-series):

| View                   | Source          | Description                           |
| ---------------------- | --------------- | ------------------------------------- |
| `grafana_realized_pnl` | `history_deals` | Cumulative realized PnL per symbol    |
| `grafana_trade_stats`  | `history_deals` | Win/loss counts and profit per symbol |

#### Example Grafana queries

```sql
-- Equity curve over time
SELECT time, equity FROM grafana_account_snapshots ORDER BY time;

-- Rolling balance by account login
SELECT time, login, balance FROM grafana_account_snapshots
WHERE login = $login ORDER BY time;

-- Open positions at latest successful snapshot
SELECT symbol, volume, profit FROM grafana_position_snapshots
WHERE run_id = (SELECT MAX(run_id) FROM snapshot_runs WHERE status = 'ok');

-- Realized PnL by symbol
SELECT symbol, total_profit FROM grafana_trade_stats ORDER BY total_profit DESC;
```

#### Grafana and telemetry API docs

The shipped Grafana helpers are documented in [`docs/api/grafana.md`](docs/api/grafana.md), including `publish_grafana_copy()` for creating a WAL-safe published SQLite copy for Grafana.

OpenTelemetry metrics are documented in [`docs/api/telemetry.md`](docs/api/telemetry.md), including `enable_otel_metrics()`, `configure_metrics()`, the `mt5cli[otel]` extra, and `OTEL_EXPORTER_OTLP_ENDPOINT`.

### Incremental history SDK

For automated pipelines, use the importable incremental API instead of re-fetching fixed date ranges:

```python
from pdmt5 import Mt5Config, Mt5DataClient
from mt5cli import update_history, update_history_with_config
from mt5cli.utils import Dataset

# Reuse an already-connected pdmt5 client (does not open/close MT5)
client = Mt5DataClient(config=Mt5Config(login=12345))
client.initialize_and_login_mt5()
try:
    update_history(
        client=client,
        output="history.db",
        symbols=["EURUSD", "GBPUSD"],
        datasets={Dataset.rates, Dataset.history_deals},
        timeframes=["M1", "H1"],  # default: all fixed MT5 timeframes
        lookback_hours=24,
        create_rate_views=True,
        with_views=True,
        include_account_events=True,
    )
finally:
    client.shutdown()

# Standalone wrapper that opens and closes MT5 for you
update_history_with_config(
    output="history.db",
    symbols=["EURUSD"],
    config=Mt5Config(login=12345),
)
```

- **`collect-history`**: explicit date-range export into SQLite.
- **`update_history`**: incremental append based on existing SQLite `MAX(time)` per symbol (and timeframe for rates); account-level deals use a separate cursor when `include_account_events=True`.
- **`rates` table**: normalized storage with `symbol` and `timeframe` columns.
- **Rate compatibility views**: mt5cli manages all `rate_*` views. Naming is `rate_<symbol>__<timeframe>` when a symbol has one timeframe, otherwise `rate_<symbol>__<granularity>_<timeframe>` (for example `rate_EURUSD__M1_1`). Stale `rate_*` views are dropped and recreated when rates change for offline downstream tools.
- **Rate view resolution**: use `resolve_rate_view_name()` / `resolve_rate_view_names()` to map symbols and granularities to existing SQLite compatibility views without creating databases. Both accept `None` (or a missing path) and return deterministic default names unless `require_existing=True`.
- **Rate view loading**: use `load_rate_data()` / `load_rate_data_from_connection()` to load a SQLite rate table or view into a `DatetimeIndex` DataFrame.
- **Multi-series rate loading**: use `build_rate_targets()` to build neutral `RateTarget(symbol, timeframe)` pairs, `resolve_rate_tables()` to map them to table/view names (pass `require_existing=True` for strict resolution), and `load_rate_series_from_sqlite()` to load them into a mapping keyed by `(symbol, integer timeframe)`. The loader requires existing managed views unless `explicit_tables` is supplied, and rejects duplicate `(symbol, timeframe)` targets.
- **Multi-account latest rates**: use `collect_latest_rates_for_accounts()` with `AccountSpec` to read the latest bars for several account groups, merged into a `(symbol, integer timeframe)` mapping. For long-running pollers, `collect_latest_rates_for_accounts_with_retries()` adds bounded exponential backoff that retries only recoverable MT5 errors and re-raises once `retry_count` is exhausted.
- **Latest closed bars**: use `collect_latest_closed_rates_for_accounts()` when downstream logic must exclude the still-forming current bar. It fetches `count + 1` bars at `start_pos=0`, drops the last row with `drop_forming_rate_bar()`, and validates each series is non-empty. `collect_latest_closed_rates_by_granularity()` returns the same data keyed by `(symbol, granularity_name)` such as `("EURUSD", "M1")`.

```python
from mt5cli import AccountSpec, collect_latest_closed_rates_by_granularity

rates = collect_latest_closed_rates_by_granularity(
    [AccountSpec(symbols=["EURUSD", "GBPUSD"], login=12345)],
    ["M1", "H1"],
    count=500,
    retry_count=3,
)
eurusd_m1 = rates["EURUSD", "M1"]  # closed bars only
```

- **Credential resolution**: use `resolve_account_spec()` / `resolve_account_specs()` to merge explicit override values over `AccountSpec` fields and expand `${ENV_VAR}` placeholders (via `substitute_env_placeholders()`), raising `ValueError` for missing variables. This keeps secrets out of plan/config files without coupling to any strategy code. For config dicts or nested structures loaded from YAML/TOML, use `substitute_mapping_values(data, keys={"login", "password"})` to expand placeholders only for caller-specified keys — key names are never hard-coded in mt5cli.
- **Throttled history updates**: use `ThrottledHistoryUpdater` to wrap `update_history()` with a minimum `interval_seconds` between successful runs (monotonic clock). Call `should_update()` / `update(client, symbols)` from an application loop; errors propagate by default, or pass `suppress_errors=True` to swallow recoverable `Mt5*Error`, `sqlite3.Error`, `ValueError`, `OSError`, and MT5 client capability errors for history API methods without advancing the throttle (other `AttributeError` / `TypeError` values always propagate). Pass `update_backend` to inject a custom history update callable (same keyword arguments as `update_history`) instead of monkey-patching `mt5cli.sdk.update_history`.
- **Trading session helpers**: use `mt5_trading_session()` for a trading-capable client that initializes/logs in via `Mt5Config.path` and always shuts down safely. Pair with `detect_position_side()`, `calculate_margin_and_volume()`, and `determine_order_limits()` for generic position and sizing utilities. Keep read-only collection on `mt5_session()` / `MT5Client`.
- **Granularity-keyed rate loading**: `load_rate_series_by_granularity()` builds targets with `build_rate_targets()`, loads them with `load_rate_series_from_sqlite()`, and returns a mapping keyed by `(symbol | None, granularity_name)` such as `("EURUSD", "M1")` to reduce downstream boilerplate.
- **MT5 session helper**: use the `mt5_session()` context manager to attach to (or, when `Mt5Config.path` is set, launch) an MT5 terminal, log in, and yield a connected `MT5Client` that shuts down on exit.
- **SQLite export helpers**: use `export_dataframe_to_sqlite()` for append mode, optional index export, and post-write deduplication by key columns.
- **Recent ticks and margins**: `recent_ticks()` and `minimum_margins()` SDK helpers (and matching CLI commands) cover common downstream read-only queries.

## Requirements

- Python 3.11+
- Windows OS (MetaTrader 5 requirement)
- MetaTrader 5 platform installed

### Migration note for downstream trading apps

Replace local MT5 lifecycle and trading helper code with mt5cli imports:

```python
# Before (local application helpers)
# with local_mt5_trading_session(config) as client:
#     side = local_detect_position_side(client, symbol)
#     sizing = local_calculate_margin_and_volume(client, symbol, unit_ratio, preserved_ratio)
#     limits = local_determine_order_limits(client, symbol, side, sl_ratio, tp_ratio)

# After (mt5cli shared layer)
from pdmt5 import Mt5Config
from mt5cli import (
    calculate_margin_and_volume,
    detect_position_side,
    determine_order_limits,
    mt5_trading_session,
)

with mt5_trading_session(
    Mt5Config(path=terminal_path, login=login), retry_count=2
) as client:
    side = detect_position_side(client, symbol)
    sizing = calculate_margin_and_volume(
        client, symbol, unit_margin_ratio=0.5, preserved_margin_ratio=0.2
    )
    if side is not None:
        limits = determine_order_limits(
            client,
            symbol,
            side,
            stop_loss_limit_ratio=0.01,
            take_profit_limit_ratio=0.02,
        )
```

Throttled history updates use a separate read-only session:

```python
from pdmt5 import Mt5Config, Mt5DataClient

from mt5cli import ThrottledHistoryUpdater

updater = ThrottledHistoryUpdater(
    output="history.db", interval_seconds=60, suppress_errors=True
)
client = Mt5DataClient(config=Mt5Config(login=login))
client.initialize_and_login_mt5()
try:
    updater.update(client, ["EURUSD"])
finally:
    client.shutdown()
```

Read-only collectors can keep using `mt5_session()` and `MT5Client`.

## Development

```bash
git clone https://github.com/dceoy/mt5cli.git
cd mt5cli
uv sync
```

## License

[MIT](LICENSE)
