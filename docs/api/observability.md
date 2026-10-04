# Observability Module

::: mt5cli.observability

## Snapshot persistence (SQLite)

These helpers append timestamped account/position/order/terminal snapshot
rows into a SQLite database. The snapshot tables are created on demand with
idempotent DDL (`CREATE TABLE IF NOT EXISTS`).

| Symbol                             | Role                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| `update_observability`             | Append one timestamped snapshot row per data type from a connected client implementation    |
| `update_observability_with_config` | Standalone wrapper: opens/closes MT5 connection automatically around `update_observability` |

Both functions write to the SQLite path given by `output=`. The optional
`symbols` parameter filters `positions` / `orders` by symbol.

Pass the `MT5Client` yielded by `mt5_session()` directly to
`update_observability(client=...)`. This workflow calls the facade's canonical
data methods (`account_info`, `positions`, `orders`, `terminal_info`); callers
never need a pdmt5 client, and no raw pdmt5 method-name fallback logic is
involved.

```python
from mt5cli import mt5_session, update_observability

with mt5_session() as client:
    update_observability(client=client, output="observability.db")
```

The snapshot table DDL and row inserts (`create_snapshot_tables`,
`start_snapshot_run`, `insert_*_snapshot(s)`, `record_snapshot_run`) live in
this module alongside the orchestration that decides _when_ and _what_ to
snapshot.
