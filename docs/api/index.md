# API Reference

This section documents the mt5cli public Python API and CLI modules.

## Public API layers

| Module                                    | Purpose                                                                   |
| ----------------------------------------- | ------------------------------------------------------------------------- |
| [Client](client.md)                       | `MT5Client` session abstraction for data access and order primitives      |
| [Schemas](schemas.md)                     | Canonical DataFrame contracts and normalization helpers                   |
| [Storage](storage.md)                     | CSV/JSON/Parquet/SQLite export and history collection helpers             |
| [Converters](converters.md)               | Symbol, timeframe, timezone, and date-range utilities                     |
| [Exceptions](exceptions.md)               | Stable mt5cli exception types and MT5 error normalization                 |
| [SDK](sdk.md)                             | Module-level fetch helpers, multi-account collectors, incremental history |
| [Trading](trading.md)                     | Trading-capable sessions and operational helpers                          |
| [History Collection (SQLite)](history.md) | SQLite schema, incremental writes, dedup, and rate views                  |
| [CLI](cli.md)                             | Typer commands that delegate to the Python API                            |
| [Utils](utils.md)                         | Parsing helpers and Click parameter types                                 |

## Architecture overview

```mermaid
flowchart TD
    App["Downstream application"] --> Client["MT5Client"]
    CLI["mt5cli CLI"] --> Client
    Client --> SDK["sdk / pdmt5"]
    Client --> Schemas["schemas"]
    Storage["storage"] --> History["history SQLite"]
    Storage --> Utils["utils export"]
    SDK --> PDMT5["pdmt5.Mt5DataClient"]
```

Downstream packages should depend on the package root exports (`MT5Client`, `DataKind`, `normalize_dataframe`, `export_dataframe`, `collect_history`, etc.) rather than private modules.

## Quick start

```python
from mt5cli import MT5Client, build_config, mt5_session

with mt5_session(build_config(login=12345)) as client:
    rates = client.copy_rates_range("EURUSD", "H1", "2024-01-01", "2024-02-01")
    positions = client.positions()
```

```bash
mt5cli -o account.csv account-info
mt5cli -o rates.parquet rates-range --symbol EURUSD --timeframe H1 \
  --date-from 2024-01-01 --date-to 2024-02-01
```

See individual module pages for detailed usage examples.
