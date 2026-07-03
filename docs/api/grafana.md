# Grafana

::: mt5cli.grafana

## Grafana-ready SQLite workflow

Use `ensure_grafana_schema(conn)` or the `grafana-schema` CLI command to create
the snapshot tables, `grafana_*` views, and supporting indexes in one step.

`publish_grafana_copy(source, target)` creates a consistent SQLite copy via the
SQLite backup API, which is useful when the primary database is running in WAL
mode and Grafana should read from a separate published file.

## Main APIs

- `ensure_grafana_schema()`: idempotently creates snapshot tables, Grafana
  views, and indexes.
- `create_grafana_views()`: rebuilds the shipped `grafana_*` views.
- `create_grafana_indexes()`: creates read-oriented indexes for Grafana
  queries.
- `publish_grafana_copy()`: writes an atomic, WAL-safe published copy for a
  Grafana datasource.

For end-to-end snapshot collection examples, see the Grafana and observability
section in the project `README.md`.
