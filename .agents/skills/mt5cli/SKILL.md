---
name: mt5cli
description: Use the `mt5cli` CLI to export MetaTrader 5 data (rates, ticks, account, symbols, orders, positions, history) to CSV, JSON, Parquet, or SQLite3. Invoke when the user asks to export, dump, download, or fetch MT5 market data or account data to a file.
---

# mt5cli

Export MetaTrader 5 data to CSV, JSON, Parquet, or SQLite3 via the `mt5cli`
command. Output format is auto-detected from the file extension (`.csv`,
`.json`, `.parquet`/`.pq`, `.db`/`.sqlite`/`.sqlite3`) or overridden with
`--format/-f`.

## Requirements

- Python 3.11+ on Windows with MetaTrader 5 installed (pdmt5 requires the
  MT5 terminal).
- Install: `pip install -U mt5cli MetaTrader5`.
- In this repo, run via `uv run mt5cli ...` or `uv run python -m mt5cli ...`.

## Invocation shape

```
mt5cli [GLOBAL OPTIONS] -o OUTPUT COMMAND [COMMAND OPTIONS]
```

Global options MUST precede the subcommand.

### Global options (apply to every subcommand)

| Option                | Purpose                                                       |
| --------------------- | ------------------------------------------------------------- |
| `-o, --output PATH`   | Output file path (required).                                  |
| `-f, --format FORMAT` | `csv`, `json`, `parquet`, or `sqlite3` (auto from extension). |
| `--table NAME`        | Table name for SQLite3 output (default: `data`).              |
| `--login INT`         | MT5 trading account login.                                    |
| `--password TEXT`     | MT5 trading account password.                                 |
| `--server TEXT`       | MT5 trading server name.                                      |
| `--path TEXT`         | Path to MetaTrader 5 terminal EXE.                            |
| `--timeout INT`       | Connection timeout in milliseconds.                           |
| `--log-level LEVEL`   | `DEBUG`, `INFO`, `WARNING` (default), `ERROR`.                |

### Parameter value formats

- **Datetimes** (`--date-from`, `--date-to`): ISO 8601 (`2024-01-01` or
  `2024-01-01T12:00:00+00:00`). Naive values are treated as UTC.
- **Timeframe** (`--timeframe`): `M1`, `M2`, `M3`, `M4`, `M5`, `M6`, `M10`,
  `M12`, `M15`, `M20`, `M30`, `H1`, `H2`, `H3`, `H4`, `H6`, `H8`, `H12`,
  `D1`, `W1`, `MN1`, or the raw integer.
- **Tick flags** (`--flags`): `ALL`, `INFO`, `TRADE`, or the raw integer.

## Commands

| Command          | Required options                                       | Optional options                                    |
| ---------------- | ------------------------------------------------------ | --------------------------------------------------- |
| `rates-from`     | `--symbol`, `--timeframe`, `--date-from`, `--count`    | —                                                   |
| `rates-from-pos` | `--symbol`, `--timeframe`, `--start-pos`, `--count`    | —                                                   |
| `rates-range`    | `--symbol`, `--timeframe`, `--date-from`, `--date-to`  | —                                                   |
| `ticks-from`     | `--symbol`, `--date-from`, `--count`, `--flags`        | —                                                   |
| `ticks-range`    | `--symbol`, `--date-from`, `--date-to`, `--flags`      | —                                                   |
| `account-info`   | —                                                      | —                                                   |
| `terminal-info`  | —                                                      | —                                                   |
| `symbols`        | —                                                      | `--group` (e.g., `*USD*`)                           |
| `symbol-info`    | `--symbol`                                             | —                                                   |
| `orders`         | —                                                      | `--symbol`, `--group`, `--ticket`                   |
| `positions`      | —                                                      | `--symbol`, `--group`, `--ticket`                   |
| `history-orders` | —                                                      | `--date-from`, `--date-to`, `--group`, `--symbol`, `--ticket`, `--position` |
| `history-deals`  | —                                                      | `--date-from`, `--date-to`, `--group`, `--symbol`, `--ticket`, `--position` |

## Examples

```bash
# Account snapshot as CSV.
mt5cli -o account.csv account-info

# EURUSD M1 bars (1000 rows) from a start date to Parquet.
mt5cli -o rates.parquet rates-from \
  --symbol EURUSD --timeframe M1 --date-from 2024-01-01 --count 1000

# EURUSD tick stream for a date range to JSON.
mt5cli -o ticks.json ticks-range \
  --symbol EURUSD --date-from 2024-01-01 --date-to 2024-01-02 --flags ALL

# USD symbols into a named table in SQLite3.
mt5cli -o data.db --table symbols symbols --group "*USD*"

# Historical deals filtered by symbol, with explicit credentials.
mt5cli --login 12345 --password mypass --server MyBroker-Demo \
  -o deals.csv history-deals --symbol EURUSD --date-from 2024-01-01
```

## Guidelines

- Pick the output extension to avoid passing `--format`.
- Use `--table` only with SQLite3 outputs; it is otherwise ignored.
- `--count` is required for `rates-from`, `rates-from-pos`, and `ticks-from`.
  Prefer `rates-range` / `ticks-range` when a fixed window is known.
- Credentials (`--login`, `--password`, `--server`) are optional when the
  local MT5 terminal is already logged in.
- Reach for `--log-level DEBUG` when a command fails silently — MT5
  connection errors surface there.
- If the user asks to run from source in this repo, prefix with `uv run`
  (e.g., `uv run mt5cli -o out.csv account-info`).
