# API Reference

This section contains the complete API documentation for mt5cli.

## Modules

The mt5cli package consists of the following modules:

### [CLI](cli.md)

Command-line interface module providing typer-based commands for exporting MetaTrader 5 data to CSV, JSON, Parquet, and SQLite3 formats.

## Architecture Overview

The package follows a simple architecture built on top of pdmt5:

1. **CLI Layer** (`cli.py`): Typer application with subcommands for each data type, custom Click parameter types for datetime/timeframe/tick flags parsing, and format detection/export utilities.
2. **Data Layer** (via `pdmt5`): Uses `Mt5DataClient` and `Mt5Config` from the pdmt5 package for all MetaTrader 5 data access.

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
from mt5cli import detect_format, export_dataframe
import pandas as pd

# Detect output format from file extension
fmt = detect_format(Path("output.parquet"))  # Returns "parquet"

# Export a DataFrame
df = pd.DataFrame({"symbol": ["EURUSD"], "bid": [1.1234]})
export_dataframe(df, Path("output.csv"), "csv")
```

## Examples

See individual module pages for detailed usage examples and code samples.
