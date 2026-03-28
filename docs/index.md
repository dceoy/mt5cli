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

| Command | Description |
|---------|-------------|
| `rates-from` | Export rates from a start date |
| `rates-from-pos` | Export rates from a start position |
| `rates-range` | Export rates for a date range |

### Ticks

| Command | Description |
|---------|-------------|
| `ticks-from` | Export ticks from a start date |
| `ticks-range` | Export ticks for a date range |

### Information

| Command | Description |
|---------|-------------|
| `account-info` | Export account information |
| `terminal-info` | Export terminal information |
| `symbols` | Export symbol list |
| `symbol-info` | Export symbol details |

### Trading

| Command | Description |
|---------|-------------|
| `orders` | Export active orders |
| `positions` | Export open positions |
| `history-orders` | Export historical orders |
| `history-deals` | Export historical deals |

## Global Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file path (required) |
| `-f, --format` | Output format (auto-detected from extension if omitted) |
| `--table` | Table name for SQLite3 output (default: "data") |
| `--login` | Trading account login |
| `--password` | Trading account password |
| `--server` | Trading server name |
| `--path` | Path to MetaTrader5 terminal EXE file |
| `--timeout` | Connection timeout in milliseconds |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) |

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
