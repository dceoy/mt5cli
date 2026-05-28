# mt5cli

Command-line tool for MetaTrader 5 data export.

## Overview

mt5cli is a CLI application that exports MetaTrader 5 trading data to multiple file formats. It is built on top of [pdmt5](https://github.com/dceoy/pdmt5), a pandas-based data handler for MetaTrader 5.

## Features

- **Multi-format export**: CSV, JSON, Parquet, and SQLite3 output formats
- **Auto-detection**: Format detection from file extensions
- **Comprehensive data access**: Rates, ticks, account info, symbols, orders, positions, and trading history
- **Flexible timeframes**: Named timeframes (M1, H1, D1, etc.) and numeric values
- **Connection management**: Optional credentials, server, and timeout configuration

## Installation

```bash
pip install mt5cli
```

## Quick Start

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

## Commands

### Rates

| Command          | Description                        |
| ---------------- | ---------------------------------- |
| `rates-from`     | Export rates from a start date     |
| `rates-from-pos` | Export rates from a start position |
| `rates-range`    | Export rates for a date range      |

### Ticks

| Command       | Description                    |
| ------------- | ------------------------------ |
| `ticks-from`  | Export ticks from a start date |
| `ticks-range` | Export ticks for a date range  |

### Information

| Command            | Description                             |
| ------------------ | --------------------------------------- |
| `account-info`     | Export account information              |
| `terminal-info`    | Export terminal information             |
| `version`          | Export MetaTrader 5 version information |
| `last-error`       | Export the last error information       |
| `symbols`          | Export symbol list                      |
| `symbol-info`      | Export symbol details                   |
| `symbol-info-tick` | Export the last tick for a symbol       |
| `market-book`      | Export market depth (order book)        |

### Trading

| Command          | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| `orders`         | Export active orders                                        |
| `positions`      | Export open positions                                       |
| `history-orders` | Export historical orders                                    |
| `history-deals`  | Export historical deals                                     |
| `order-check`    | Check funds sufficiency for a trade request                 |
| `order-send`     | Send a trade request to the trade server (`--yes` required) |

Use `order-check` to validate a request payload before running `order-send --yes`.

### Bulk Collection

| Command           | Description                                                                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `collect-history` | Collect rates, ticks, history-orders, and history-deals for one or more symbols into a single SQLite database (optional cash-event/position views) |

```bash
mt5cli -o history.db collect-history \
  --symbol EURUSD --symbol GBPUSD \
  --date-from 2024-01-01 --date-to 2024-02-01 \
  --dataset rates --dataset history-deals \
  --timeframe M1 --flags ALL --if-exists append --with-views
```

`collect-history` options:

| Option         | Default    | Description                                                                                   |
| -------------- | ---------- | --------------------------------------------------------------------------------------------- |
| `--symbol/-s`  | _required_ | Symbol to collect (repeat for multiple).                                                      |
| `--date-from`  | _required_ | Start date in ISO 8601.                                                                       |
| `--date-to`    | _required_ | End date in ISO 8601.                                                                         |
| `--dataset`    | all four   | Repeatable: `rates`, `ticks`, `history-orders`, `history-deals`.                              |
| `--timeframe`  | `M1`       | Rates timeframe; recorded in a `timeframe` column on the `rates` table.                       |
| `--flags`      | `ALL`      | Tick copy flags forwarded to `copy_ticks_range`.                                              |
| `--if-exists`  | `fail`     | `append`, `replace`, or `fail` when a target table already exists.                            |
| `--with-views` | off        | Add `cash_events` and `positions_reconstructed` views (requires the `history-deals` dataset). |

History orders and deals are fetched per symbol and concatenated, so the symbol filter is applied consistently across all datasets. The `cash_events` view is derived from symbol-filtered `history_deals`, so account-level cash events with empty or non-matching symbols may be excluded. The `positions_reconstructed` view excludes positions with no closing deal, uses volume-weighted open/close prices, and reports reversal deals (`DEAL_ENTRY_INOUT`) via `volume_reversal` / `reversal_count`.

## Global Options

| Option         | Description                                             |
| -------------- | ------------------------------------------------------- |
| `-o, --output` | Output file path (required)                             |
| `-f, --format` | Output format (auto-detected from extension if omitted) |
| `--table`      | Table name for SQLite3 output (default: "data")         |
| `--login`      | Trading account login                                   |
| `--password`   | Trading account password                                |
| `--server`     | Trading server name                                     |
| `--path`       | Path to MetaTrader5 terminal EXE file                   |
| `--timeout`    | Connection timeout in milliseconds                      |
| `--log-level`  | Logging level (DEBUG, INFO, WARNING, ERROR)             |

## Requirements

- Python 3.11+
- Windows OS (MetaTrader 5 requirement)
- MetaTrader 5 platform

## API Reference

Browse the API documentation for detailed module information:

- [CLI Module](api/cli.md) - CLI application with export commands and utility functions

## Development

This project follows strict code quality standards:

- Type hints required (strict mode)
- Comprehensive linting with Ruff
- Test coverage tracking
- Google-style docstrings

## License

MIT License - see [LICENSE](https://github.com/dceoy/mt5cli/blob/main/LICENSE) file for details.
