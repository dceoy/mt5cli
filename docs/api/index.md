# API Reference

This section contains the complete API documentation for mt5cli.

## Modules

The mt5cli package consists of the following modules:

### [CLI](cli.md)

Command-line interface module providing typer-based commands for exporting MetaTrader 5 data to CSV, JSON, Parquet, and SQLite3 formats.

### [Utils](utils.md)

Utility module providing constants, enums, Click parameter types, and helper functions for parsing and exporting data.

### [SDK](sdk.md)

Programmatic SDK for read-only MetaTrader 5 data collection. Returns pandas DataFrames and provides `collect_history` for SQLite bulk collection.

### [Trading](trading.md)

Trading-capable session management and operational helpers built on `pdmt5.Mt5TradingClient`. Complements the read-only SDK without changing existing `Mt5CliClient` behavior.

### [History Collection (SQLite)](history.md)

SQLite storage helpers for the `collect-history` command schema, incremental updates, deduplication, indexes, and optional views.

## Architecture Overview

The package follows a simple architecture built on top of pdmt5:

1. **CLI Layer** (`cli.py`): Typer application with subcommands that delegate to the SDK and export results.
2. **SDK Layer** (`sdk.py`): Read-only data access functions, `Mt5CliClient`, and `collect_history` orchestration.
3. **Trading Layer** (`trading.py`): Trading-capable sessions and operational helpers on `Mt5TradingClient`.
4. **Utils Layer** (`utils.py`): Constants, enums, custom Click parameter types, parsing helpers, and format detection/export utilities.
5. **Data Layer** (via `pdmt5`): Uses `Mt5DataClient`, `Mt5TradingClient`, and `Mt5Config` from the pdmt5 package for MetaTrader 5 access.

## Usage Guidelines

All modules follow these conventions:

- **Type Safety**: All functions include comprehensive type hints
- **Error Handling**: User-friendly error messages via typer
- **Documentation**: Google-style docstrings with examples
- **Validation**: Custom Click parameter types for input validation

## Quick Start

```bash
# Export account information to CSV
mt5cli -o account.csv account-info

# Export EURUSD H1 rates to Parquet
mt5cli -o rates.parquet rates-from --symbol EURUSD --timeframe H1 \
  --date-from 2024-01-01 --count 1000

# Export ticks to JSON
mt5cli -o ticks.json ticks-from --symbol EURUSD \
  --date-from 2024-01-01 --count 500 --flags ALL

# Export to SQLite3 with custom table name
mt5cli -o data.db --table symbols symbols --group "*USD*"
```

## Python API

```python
from datetime import UTC, datetime
from pathlib import Path

from mt5cli import (
    Dataset,
    IfExists,
    Mt5CliClient,
    collect_history,
    copy_rates_range,
    detect_format,
    export_dataframe,
    export_dataframe_to_sqlite,
    minimum_margins,
    recent_ticks,
)
from mt5cli.history import resolve_rate_view_name

# Fetch rates programmatically
rates = copy_rates_range(
    "EURUSD",
    timeframe="H1",
    date_from="2024-01-01",
    date_to="2024-02-01",
)

# Detect output format from file extension
fmt = detect_format(Path("output.parquet"))  # Returns "parquet"

# Export a DataFrame
export_dataframe(rates, Path("output.csv"), "csv")

# Append to SQLite with deduplication
export_dataframe_to_sqlite(
    rates,
    Path("history.db"),
    "rates",
    if_exists=IfExists.APPEND,
    deduplicate_on=("symbol", "timeframe", "time"),
)

# Resolve rate compatibility views and fetch recent ticks
view = resolve_rate_view_name(Path("history.db"), "EURUSD", "M1")
ticks = recent_ticks("EURUSD", seconds=300)
margins = minimum_margins("EURUSD")

# Collect history into SQLite
collect_history(
    Path("history.db"),
    symbols=["EURUSD"],
    date_from=datetime(2024, 1, 1, tzinfo=UTC),
    date_to=datetime(2024, 2, 1, tzinfo=UTC),
)
```

## Examples

See individual module pages for detailed usage examples and code samples.
