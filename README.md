# mt5cli

[![CI/CD](https://github.com/dceoy/mt5cli/actions/workflows/ci.yml/badge.svg)](https://github.com/dceoy/mt5cli/actions/workflows/ci.yml)

Command-line tool for exporting MetaTrader 5 data to CSV, JSON, Parquet, and SQLite3.

Built on top of [pdmt5](https://github.com/dceoy/pdmt5), a pandas-based data handler for MetaTrader 5.

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

## Usage

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

# Export with connection credentials
mt5cli --login 12345 --password mypass --server MyBroker-Demo \
  -o positions.csv positions
```

Run as a Python module:

```bash
python -m mt5cli -o account.csv account-info
```

## Commands

| Command                | Description                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| `rates-from`           | Export rates from a start date                                                                               |
| `rates-from-pos`       | Export rates from a start position                                                                           |
| `latest-rates`         | Export latest rates from a start position                                                                    |
| `rates-range`          | Export rates for a date range                                                                                |
| `ticks-from`           | Export ticks from a start date                                                                               |
| `ticks-range`          | Export ticks for a date range                                                                                |
| `ticks-recent`         | Export ticks from a recent trailing window                                                                   |
| `account-info`         | Export account information                                                                                   |
| `terminal-info`        | Export terminal information                                                                                  |
| `version`              | Export MetaTrader 5 version information                                                                      |
| `last-error`           | Export the last error information                                                                            |
| `symbols`              | Export symbol list                                                                                           |
| `symbol-info`          | Export symbol details                                                                                        |
| `symbol-info-tick`     | Export the last tick for a symbol                                                                            |
| `minimum-margins`      | Export minimum-volume buy and sell margin requirements                                                       |
| `market-book`          | Export market depth (order book)                                                                             |
| `orders`               | Export active orders                                                                                         |
| `positions`            | Export open positions                                                                                        |
| `history-orders`       | Export historical orders                                                                                     |
| `history-deals`        | Export historical deals                                                                                      |
| `recent-history-deals` | Export historical deals from a recent trailing window                                                        |
| `mt5-summary`          | Export terminal/account status summary                                                                       |
| `order-check`          | Check funds sufficiency for a trade request                                                                  |
| `order-send`           | Send a trade request to the trade server (`--yes` required)                                                  |
| `collect-history`      | Bundle rates, ticks, history-orders, and history-deals for one or more symbols into a single SQLite database |

Use `order-check` to validate a request payload before running `order-send --yes`.

### `collect-history`

Collect several historical datasets per symbol into one SQLite database in a single MT5 session. Pick datasets with repeatable `--dataset` (default: all four), choose conflict behavior with `--if-exists append|replace|fail` (default: `fail`), and optionally derive `cash_events` / `positions_reconstructed` views from `history_deals` via `--with-views`.

```bash
mt5cli -o history.db collect-history \
  --symbol EURUSD --symbol GBPUSD \
  --date-from 2024-01-01 --date-to 2024-02-01 \
  --dataset rates --dataset history-deals \
  --timeframe M1 --flags ALL --if-exists append --with-views
```

History orders and deals are fetched per symbol and concatenated, so the symbol filter is applied consistently across all datasets. The `cash_events` view is derived from symbol-filtered `history_deals`, so account-level cash events with empty or non-matching symbols may be excluded. The `rates` table records the requested `timeframe` so appended runs at different timeframes remain distinguishable. The `positions_reconstructed` view aggregates trade deals by `position_id`, excludes positions without closing-side entries, and uses volume-weighted open/close prices; reversal deals (`DEAL_ENTRY_INOUT`) are reported via `volume_reversal` / `reversal_count` columns.

### Incremental history SDK

For automated pipelines, use the importable incremental API instead of re-fetching fixed date ranges:

```python
from pdmt5 import Mt5Config, Mt5DataClient
from mt5cli import Dataset, update_history, update_history_with_config

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
- **Rate compatibility views**: mt5cli manages all `rate_*` views. Naming is `rate_<symbol>__<timeframe>` when a symbol has one timeframe, otherwise `rate_<symbol>__<granularity>_<timeframe>` (for example `rate_EURUSD__M1_1`). Stale `rate_*` views are dropped and recreated when rates change for offline tools such as mteor optimize.
- **Rate view resolution**: use `resolve_rate_view_name()` / `resolve_rate_view_names()` to map symbols and granularities to existing SQLite compatibility views without creating databases. Both accept `None` (or a missing path) and return deterministic default names unless `require_existing=True`.
- **Rate view loading**: use `load_rate_data()` / `load_rate_data_from_connection()` to load a SQLite rate table or view into a `DatetimeIndex` DataFrame.
- **Multi-series rate loading**: use `build_rate_targets()` to build neutral `RateTarget(symbol, timeframe)` pairs, `resolve_rate_tables()` to map them to table/view names (pass `require_existing=True` for strict resolution), and `load_rate_series_from_sqlite()` to load them into a mapping keyed by `(symbol, integer timeframe)`. The loader requires existing managed views unless `explicit_tables` is supplied, and rejects duplicate `(symbol, timeframe)` targets.
- **Multi-account latest rates**: use `collect_latest_rates_for_accounts()` with `AccountSpec` to read the latest bars for several account groups, merged into a `(symbol, integer timeframe)` mapping. For long-running pollers, `collect_latest_rates_for_accounts_with_retries()` adds bounded exponential backoff that retries only `pdmt5.Mt5TradingError` / `pdmt5.Mt5RuntimeError` and re-raises once `retry_count` is exhausted.
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

- **Credential resolution**: use `resolve_account_spec()` / `resolve_account_specs()` to merge explicit override values over `AccountSpec` fields and expand `${ENV_VAR}` placeholders (via `substitute_env_placeholders()`), raising `ValueError` for missing variables. This keeps secrets out of plan/config files without coupling to any strategy code.
- **Throttled history updates**: use `ThrottledHistoryUpdater` to wrap `update_history()` with a minimum `interval_seconds` between successful runs (monotonic clock). Call `should_update()` / `update(client, symbols)` from an application loop; errors propagate by default, or pass `suppress_errors=True` to swallow recoverable `Mt5*Error`, `sqlite3.Error`, `ValueError`, `OSError`, and MT5 client capability errors for history API methods without advancing the throttle (other `AttributeError` / `TypeError` values always propagate).
- **Trading session helpers**: use `mt5_trading_session()` for a trading-capable `pdmt5.Mt5TradingClient` that initializes/logs in via `Mt5Config.path` and always shuts down safely. Pair with `detect_position_side()`, `calculate_margin_and_volume()`, and `determine_order_limits()` for generic position and sizing utilities. The read-only `mt5_session()` / `Mt5CliClient` SDK is unchanged.
- **Granularity-keyed rate loading**: `load_rate_series_by_granularity()` builds targets with `build_rate_targets()`, loads them with `load_rate_series_from_sqlite()`, and returns a mapping keyed by `(symbol | None, granularity_name)` such as `("EURUSD", "M1")` to reduce downstream boilerplate.
- **MT5 session helper**: use the `mt5_session()` context manager to attach to (or, when `Mt5Config.path` is set, launch) an MT5 terminal, log in, and yield a connected `Mt5CliClient` that shuts down on exit.
- **SQLite export helpers**: use `export_dataframe_to_sqlite()` for append mode, optional index export, and post-write deduplication by key columns.
- **Recent ticks and margins**: `recent_ticks()` and `minimum_margins()` SDK helpers (and matching CLI commands) cover common downstream read-only queries.

## Requirements

- Python 3.11+
- Windows OS (MetaTrader 5 requirement)
- MetaTrader 5 platform installed

### Migration note for mteor

Replace local MT5 lifecycle and trading helper code with mt5cli imports:

```python
# Before (local mteor helpers)
# with local_mt5_trading_session(config) as client:
#     side = local_detect_position_side(client, symbol)
#     sizing = local_calculate_margin_and_volume(client, symbol, unit_ratio, preserved_ratio)
#     limits = local_determine_order_limits(client, symbol, side, sl_ratio, tp_ratio)

# After (mt5cli shared layer)
from pdmt5 import Mt5Config
from mt5cli import (
    ThrottledHistoryUpdater,
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

updater = ThrottledHistoryUpdater(
    output="history.db", interval_seconds=60, suppress_errors=True
)
```

Read-only collectors can keep using `mt5_session()` and `Mt5CliClient` without changes.

## Development

```bash
git clone https://github.com/dceoy/mt5cli.git
cd mt5cli
uv sync
```

## License

[MIT](LICENSE)
