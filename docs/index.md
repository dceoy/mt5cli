# mt5cli

Generic MT5 data and execution infrastructure for Python applications.

## Overview

mt5cli provides a stable `MT5Client` Python API, standardized dataset schemas, storage helpers, and a CLI for exporting MetaTrader 5 data. It is built on top of [pdmt5](https://github.com/dceoy/pdmt5), a pandas-based data handler for MetaTrader 5.

## Architecture

- **pdmt5** — canonical MT5 client, DataFrame/trading primitives, and MT5 constant parsing (`TIMEFRAME_*`, `COPY_TICKS_*`, order types).
- **mt5cli** — public `MT5Client` API, schema contracts, storage helpers, CLI commands, and SQLite history collection built on pdmt5.
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

Parquet export is not included by default. To enable it, install the `parquet` extra:

```bash
pip install "mt5cli[parquet]"
```

## Python API for downstream packages

Import `MT5Client` for generic MT5 data access, schema normalization, and optional order primitives.

```python
from datetime import UTC, datetime
from pathlib import Path

from mt5cli import (
    MT5Client,
    build_config,
    collect_history,
    mt5_session,
)
from mt5cli.history import load_rate_data, resolve_rate_view_name
from mt5cli.schemas import DataKind, normalize_dataframe
from mt5cli.sdk import minimum_margins, recent_ticks
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

# Offline rate loading from mt5cli-managed SQLite history
view = resolve_rate_view_name(Path("history.db"), "EURUSD", "M1", require_existing=True)
offline_rates = load_rate_data(Path("history.db"), view, count=1000)

# One-off helpers still work without instantiating a client
ticks = recent_ticks("EURUSD", seconds=300)
margins = minimum_margins("EURUSD")

collect_history(
    Path("history.db"),
    symbols=["EURUSD", "GBPUSD"],
    date_from=datetime(2024, 1, 1, tzinfo=UTC),
    date_to=datetime(2024, 2, 1, tzinfo=UTC),
    datasets={Dataset.rates, Dataset.history_deals},
)
```

Schema contracts live in `mt5cli.schemas` (`DataKind`, `validate_schema`, `normalize_dataframe`). Export and storage helpers are in `mt5cli.utils` (`Dataset`, `export_dataframe`) and `mt5cli.history`.

`MT5Client.order_send()` is a live execution primitive: it can place real trades on the connected account. mt5cli does not implement strategy logic, signal generation, backtesting, or optimization — downstream applications must gate live execution explicitly (the CLI requires `--yes` for `order-send`).

`MT5Client.mt5_summary()` returns structured nested Python values. Use `MT5Client.mt5_summary_as_df()` when you need a one-row DataFrame for export.

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

# Export with connection credentials from env or placeholders
MT5_LOGIN=12345 MT5_PASSWORD=secret MT5_SERVER=MyBroker-Demo \
  mt5cli -o positions.csv positions
mt5cli --login '${MT5_LOGIN}' --password '${MT5_PASSWORD}' --server '${MT5_SERVER}' \
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

### Trading State

| Command                | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| `orders`               | Export active orders                                                |
| `positions`            | Export open positions                                               |
| `history-orders`       | Export historical orders                                            |
| `history-deals`        | Export historical deals                                             |
| `recent-history-deals` | Export historical deals from a trailing window                      |
| `mt5-summary`          | Export terminal/account status summary                              |
| `order-check`          | Check funds sufficiency for a trade request (read-only, no `--yes`) |

### Execution (live / mutating)

These commands send requests to the live trade server and can place or close
real trades. Both require `--yes` for live execution.

| Command           | Description                                                                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `order-send`      | Send a **raw** trade request directly to MT5 (`--yes` required; expert path — no extra validation)                                                     |
| `close-positions` | Close open positions by `--symbol` or `--ticket` (`--yes` required for live; `--dry-run` to preview; optional `--deviation` / `--comment` / `--magic`) |

Use `order-check` (Trading State) to validate funds before running `order-send --yes`.
`close-positions` is the safer high-level alternative that builds correct close
requests automatically. `order-send` is the expert raw path — downstream
applications should prefer dedicated closing helpers or their own risk controls.

### Bulk Collection

| Command           | Description                                                                                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `collect-history` | Collect rates, history-orders, and history-deals (ticks opt-in via `--dataset ticks`) for one or more symbols into a single SQLite database (optional cash-event/position views) |
| `history-gaps`    | Export a SQLite-only one-row-per-gap report from managed rate compatibility views without connecting to MT5                                                                      |

```bash
mt5cli -o history.db collect-history \
  --symbol EURUSD --symbol GBPUSD \
  --date-from 2024-01-01 --date-to 2024-02-01 \
  --dataset rates --dataset history-deals \
  --timeframe M1 --flags ALL --if-exists append --with-views
```

`collect-history` options:

| Option         | Default                              | Description                                                                                                                |
| -------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `--symbol/-s`  | _required_                           | Symbol to collect (repeat for multiple).                                                                                   |
| `--date-from`  | _required_                           | Start date in ISO 8601.                                                                                                    |
| `--date-to`    | _required_                           | End date in ISO 8601.                                                                                                      |
| `--dataset`    | rates, history-orders, history-deals | Repeatable: `rates`, `ticks`, `history-orders`, `history-deals`. Ticks are opt-in: pass `--dataset ticks` to include them. |
| `--timeframe`  | `M1`                                 | Rates timeframe; recorded in a `timeframe` column on the `rates` table.                                                    |
| `--flags`      | `ALL`                                | Tick copy flags forwarded to `copy_ticks_range`.                                                                           |
| `--if-exists`  | `fail`                               | `append`, `replace`, or `fail` when a target table already exists.                                                         |
| `--with-views` | off                                  | Add `cash_events` and `positions_reconstructed` views (requires the `history-deals` dataset).                              |

History orders and deals are fetched per symbol and concatenated, so the symbol filter is applied consistently across all datasets. The `cash_events` view is derived from symbol-filtered `history_deals`, so account-level cash events with empty or non-matching symbols may be excluded. The `positions_reconstructed` view excludes positions with no closing deal, uses volume-weighted open/close prices, and reports reversal deals (`DEAL_ENTRY_INOUT`) via `volume_reversal` / `reversal_count`.

See the [History schema diagram](api/history.md#entity-relationship-diagram) for a sample ER layout of the resulting database.

## Global Options

| Option         | Description                                             |
| -------------- | ------------------------------------------------------- |
| `-o, --output` | Output file path (required)                             |
| `-f, --format` | Output format (auto-detected from extension if omitted) |
| `--table`      | Table name for SQLite3 output (default: "data")         |
| `--login`      | Trading account login                                   |
| `--password`   | Trading account password (`MT5_PASSWORD`)               |
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

- [CLI Module](api/cli.md) - CLI application with data export and execution commands
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
