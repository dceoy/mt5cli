# Observability Module

::: mt5cli.observability

## Grafana observability (SQLite read model)

These helpers append timestamped account/position/order/terminal snapshot
rows into a SQLite database, preparing it as a Grafana datasource. All DDL is
idempotent (`CREATE TABLE IF NOT EXISTS`, `DROP VIEW IF EXISTS` + `CREATE
VIEW`, `CREATE INDEX IF NOT EXISTS`). Missing source tables are skipped with a
warning rather than raising an error.

| Symbol                             | Role                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| `update_observability`             | Append one timestamped snapshot row per data type from a connected client implementation    |
| `update_observability_with_config` | Standalone wrapper: opens/closes MT5 connection automatically around `update_observability` |

Both functions write to the SQLite path given by `output=`. The optional
`symbols` parameter filters `positions` / `orders` by symbol.
`with_grafana_schema=False` (default) skips Grafana view/index setup; run
`grafana-schema` once to set up the schema, then call `snapshot` repeatedly
without this flag.

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

Schema and persistence (table DDL, Grafana views, and row inserts) belong to
[Grafana](grafana.md); this module owns _when_ and _what_ to snapshot.
