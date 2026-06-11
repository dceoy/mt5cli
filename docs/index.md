# mt5cli

Command-line tool for MetaTrader 5 data export.

## Overview

mt5cli is a CLI application that exports MetaTrader 5 trading data to multiple file formats. It is built on top of [pdmt5](https://github.com/dceoy/pdmt5), a pandas-based data handler for MetaTrader 5.

## Architecture

- **pdmt5** — canonical MT5 client, DataFrame/trading primitives, and MT5 constant parsing (`TIMEFRAME_*`, `COPY_TICKS_*`, order types).
- **mt5cli** — CLI commands, CSV/JSON/Parquet/SQLite export, SQLite history collection, rate views, and local batch/automation SDK helpers built on pdmt5.
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
pip install mt5cli
```

## Programmatic usage / SDK usage

mt5cli can be used as a small Python SDK for read-only MetaTrader 5 data collection. SDK functions return pandas DataFrames without writing files. Use `export_dataframe` or `export_dataframe_to_sqlite` when you need to persist results.

```python
from datetime import UTC, datetime
from pathlib import Path

from mt5cli import (
    Mt5CliClient,
    collect_history,
    copy_rates_range,
    export_dataframe,
    export_dataframe_to_sqlite,
    load_rate_data,
    minimum_margins,
    recent_ticks,
)
from mt5cli.history import resolve_rate_view_name

# One-off fetch with module-level helpers
rates = copy_rates_range(
    "EURUSD",
    timeframe="H1",
    date_from="2024-01-01",
    date_to="2024-02-01",
)
export_dataframe(rates, Path("rates.csv"), "csv")

# Resolve SQLite rate compatibility views for downstream tools
view = resolve_rate_view_name(Path("history.db"), "EURUSD", "M1", require_existing=True)
offline_rates = load_rate_data(Path("history.db"), view, count=1000)

# Recent tick window and minimum margin summary
ticks = recent_ticks("EURUSD", seconds=300)
margins = minimum_margins("EURUSD")

# Reuse one MT5 connection for multiple calls
with Mt5CliClient(login=12345, password="secret", server="Broker-Demo") as client:
    account = client.account_info()
    positions = client.positions()
    latest = client.latest_rates("EURUSD", "M1", count=100)
    summary = client.mt5_summary()
    summary_table = client.mt5_summary_as_df()

# Bulk SQLite collection (same behavior as the collect-history CLI command)
collect_history(
    Path("history.db"),
    symbols=["EURUSD", "GBPUSD"],
    date_from=datetime(2024, 1, 1, tzinfo=UTC),
    date_to=datetime(2024, 2, 1, tzinfo=UTC),
    timeframe="M1",
    flags="ALL",
    with_views=True,
)
```

Timeframes, tick flags, and ISO 8601 date strings are accepted wherever noted in the SDK API.

`Mt5CliClient.mt5_summary()` returns the SDK structured form as plain nested Python values. Use `Mt5CliClient.mt5_summary_as_df()` when you need a one-row DataFrame for export. The `mt5-summary` CLI command uses this tabular form, so nested terminal/account fields are JSON-encoded strings that are safe for CSV, JSON, Parquet, and SQLite output.

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
| `latest-rates`   | Export latest rates                |
| `rates-range`    | Export rates for a date range      |

### Ticks

| Command        | Description                         |
| -------------- | ----------------------------------- |
| `ticks-from`   | Export ticks from a start date      |
| `ticks-range`  | Export ticks for a date range       |
| `ticks-recent` | Export ticks from a trailing window |

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
| `minimum-margins`  | Export minimum-volume margin summary    |
| `market-book`      | Export market depth (order book)        |

### Trading

| Command                | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `orders`               | Export active orders                                        |
| `positions`            | Export open positions                                       |
| `history-orders`       | Export historical orders                                    |
| `history-deals`        | Export historical deals                                     |
| `recent-history-deals` | Export historical deals from a trailing window              |
| `mt5-summary`          | Export terminal/account status summary                      |
| `order-check`          | Check funds sufficiency for a trade request                 |
| `order-send`           | Send a trade request to the trade server (`--yes` required) |

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

See the [History schema diagram](api/history.md#entity-relationship-diagram) for a sample ER layout of the resulting database.

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

- [CLI Module](api/cli.md) - CLI application with export commands
- [SDK Module](api/sdk.md) - Programmatic read-only data collection API
- [Utils Module](api/utils.md) - Constants, parameter types, parsers, and export utilities

## Development

This project follows strict code quality standards:

- Type hints required (strict mode)
- Comprehensive linting with Ruff
- Test coverage tracking
- Google-style docstrings

## License

MIT License - see [LICENSE](https://github.com/dceoy/mt5cli/blob/main/LICENSE) file for details.
