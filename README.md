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

| Command            | Description                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| `rates-from`       | Export rates from a start date                                                                               |
| `rates-from-pos`   | Export rates from a start position                                                                           |
| `rates-range`      | Export rates for a date range                                                                                |
| `ticks-from`       | Export ticks from a start date                                                                               |
| `ticks-range`      | Export ticks for a date range                                                                                |
| `account-info`     | Export account information                                                                                   |
| `terminal-info`    | Export terminal information                                                                                  |
| `version`          | Export MetaTrader 5 version information                                                                      |
| `last-error`       | Export the last error information                                                                            |
| `symbols`          | Export symbol list                                                                                           |
| `symbol-info`      | Export symbol details                                                                                        |
| `symbol-info-tick` | Export the last tick for a symbol                                                                            |
| `market-book`      | Export market depth (order book)                                                                             |
| `orders`           | Export active orders                                                                                         |
| `positions`        | Export open positions                                                                                        |
| `history-orders`   | Export historical orders                                                                                     |
| `history-deals`    | Export historical deals                                                                                      |
| `order-check`      | Check funds sufficiency for a trade request                                                                  |
| `order-send`       | Send a trade request to the trade server (`--yes` required)                                                  |
| `collect-history`  | Bundle rates, ticks, history-orders, and history-deals for one or more symbols into a single SQLite database |

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

## Requirements

- Python 3.11+
- Windows OS (MetaTrader 5 requirement)
- MetaTrader 5 platform installed

## Development

```bash
git clone https://github.com/dceoy/mt5cli.git
cd mt5cli
uv sync
```

## License

[MIT](LICENSE)
