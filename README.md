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

| Command            | Description                                 |
| ------------------ | ------------------------------------------- |
| `rates-from`       | Export rates from a start date              |
| `rates-from-pos`   | Export rates from a start position          |
| `rates-range`      | Export rates for a date range               |
| `ticks-from`       | Export ticks from a start date              |
| `ticks-range`      | Export ticks for a date range               |
| `account-info`     | Export account information                  |
| `terminal-info`    | Export terminal information                 |
| `version`          | Export MetaTrader 5 version information     |
| `last-error`       | Export the last error information           |
| `symbols`          | Export symbol list                          |
| `symbol-info`      | Export symbol details                       |
| `symbol-info-tick` | Export the last tick for a symbol           |
| `market-book`      | Export market depth (order book)            |
| `orders`           | Export active orders                        |
| `positions`        | Export open positions                       |
| `history-orders`   | Export historical orders                    |
| `history-deals`    | Export historical deals                     |
| `order-check`      | Check funds sufficiency for a trade request |
| `order-send`       | Send a trade request to the trade server (`--yes` required) |

Use `order-check` to validate a request payload before running `order-send --yes`.

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
