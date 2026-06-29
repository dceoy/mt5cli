# Public API Contract

mt5cli is the canonical operational trading SDK and CLI/batch layer over pdmt5.
The intended dependency direction is:

```text
downstream app -> mt5cli -> pdmt5 -> MetaTrader 5
```

## Responsibility boundary

| Layer          | Owns                                                                                                                                                                               |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **pdmt5**      | MT5 core wrapper; DataFrame/dict conversion; canonical MT5 constants and parsers; direct low-level order primitives                                                                |
| **mt5cli**     | CLI/batch workflows; SQLite history collection; normalized datasets; closed-bar helpers; small downstream operational SDK; generic broker-facing margin/volume/order orchestration |
| **downstream** | Strategy logic; signals; risk policy; backtesting; optimization; YAML/application semantics                                                                                        |

Downstream code should import raw pdmt5 types and constants (such as
`Mt5Config`, `Mt5RuntimeError`, `TIMEFRAME_MAP`, `COPY_TICKS_MAP`) directly
from `pdmt5` when needed. mt5cli does not serve as a pass-through compatibility
namespace for pdmt5. mt5cli's trading helpers type their client parameter against
an internal protocol backed by `pdmt5.Mt5DataClient`; `Mt5TradingClient` is no
longer required. `Mt5TradingError` is conditionally imported where still present
in pdmt5, but mt5cli raises `Mt5OperationError` for all trading-related failures.

Note: the former `mt5cli` re-export `TICK_FLAG_MAP` corresponds to `COPY_TICKS_MAP`
in pdmt5 — the name changed, it was not simply moved.

Downstream packages should import from the package root (`from mt5cli import
...`). The contract set `STABLE_SDK_EXPORTS` in `mt5cli.contract` enumerates
every package-root symbol. Lower-level helpers (schema utilities, export
functions, parser helpers, low-level MT5 wrappers) are available directly from
their owning modules (`mt5cli.schemas`, `mt5cli.utils`, `mt5cli.converters`,
`mt5cli.sdk`, etc.) and are not part of the root SDK surface.

## Stable downstream SDK API

These names are exported from `mt5cli` and enumerated in
`mt5cli.STABLE_SDK_EXPORTS` (defined in `mt5cli.contract`).

### Session lifecycle and configuration

| Symbol                                          | Role                                                                                                                                                                                                                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MT5Client`                                     | Read-only data client with optional `order_check` / `order_send`                                                                                                                                                                                                                  |
| `build_config`                                  | Build `pdmt5.Mt5Config` from connection fields; `login` accepts `int \| str \| None` — numeric strings are coerced to `int`, blank strings are treated as unset, and `${ENV_VAR}` / `$ENV_NAME` placeholders in string parameters are expanded when `allow_whole_dollar_env=True` |
| `mt5_session`                                   | Context manager: initialize, login, yield client, shutdown                                                                                                                                                                                                                        |
| `create_trading_client`, `mt5_trading_session`  | Trading-capable MT5 client lifecycle; returns a raw `pdmt5.Mt5DataClient` (not `MT5Client`) supporting order execution, account management, and history deal retrieval                                                                                                            |
| `AccountSpec`                                   | Generic account group: symbols plus optional credentials                                                                                                                                                                                                                          |
| `resolve_account_spec`, `resolve_account_specs` | Merge overrides and expand `${ENV_VAR}` placeholders; opt-in `allow_whole_dollar_env` for bare `$NAME`                                                                                                                                                                            |

### Closed-bar rate helpers

MetaTrader 5 returns the still-forming bar as the last row when
`start_pos=0`. Use these helpers instead of reimplementing bar trimming or
timestamp normalization in downstream apps.

| Symbol                                           | Role                                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------------------- |
| `drop_forming_rate_bar`                          | Remove the last row from chronologically ordered rate data                      |
| `fetch_latest_closed_rates`                      | Single connected client: fetch `count + 1`, drop forming bar                    |
| `fetch_latest_closed_rates_for_trading_client`   | Closed bars from an active trading client session; returns RangeIndex           |
| `fetch_latest_closed_rates_indexed`              | Same as above but returns a UTC `DatetimeIndex` named `"time"` (no time column) |
| `collect_latest_closed_rates_for_accounts`       | Multi-account closed bars with optional retry wrapper                           |
| `collect_latest_closed_rates_by_granularity`     | Same data keyed by `(symbol, granularity_name)`                                 |
| `collect_latest_rates_for_accounts_with_retries` | Bounded exponential backoff for transient MT5 errors                            |

### SQLite history collection and rate loading

| Symbol                                                            | Role                                                                                         |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `collect_history`                                                 | One-shot date-range export into SQLite                                                       |
| `update_history`, `update_history_with_config`                    | Incremental append from `MAX(time)` cursors                                                  |
| `ThrottledHistoryUpdater`                                         | Minimum interval between successful incremental updates; optional `update_backend` injection |
| `RateTarget`, `build_rate_targets`                                | Neutral `(symbol, timeframe)` series descriptors                                             |
| `load_rate_series_from_sqlite`, `load_rate_series_by_granularity` | Load one or many series; fail clearly when managed views are missing                         |

See [History Collection (SQLite)](history.md) for schema, view naming, and ER
diagrams.

### Trading and sizing primitives (generic)

These helpers implement broker-facing calculations only. They do not encode
strategy entries, exits, Kelly sizing, or signal logic.

| Symbol                                                                                                                         | Role                                                              |
| ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| `get_account_snapshot`, `get_symbol_snapshot`, `get_tick_snapshot`, `get_positions_frame`                                      | Normalized account/symbol/tick/position views                     |
| `extract_tick_price`                                                                                                           | Positive finite bid/ask extraction from tick mappings             |
| `detect_position_side`                                                                                                         | Net long / short / flat from open positions                       |
| `calculate_spread_ratio`                                                                                                       | Relative bid-ask spread                                           |
| `calculate_margin_and_volume`, `calculate_volume_by_margin`, `calculate_new_position_margin_ratio`                             | Margin budget and volume sizing                                   |
| `normalize_order_volume`, `estimate_order_margin`, `calculate_positions_margin`                                                | Broker volume normalization and margin totals                     |
| `calculate_positions_margin_by_symbol`                                                                                         | Per-symbol margin map (resilient, first-seen order)               |
| `calculate_positions_margin_safe`                                                                                              | Summed total margin across symbols (failed symbols skipped)       |
| `calculate_projected_margin_ratio`                                                                                             | Estimated symbol-scoped margin/equity after optional new exposure |
| `calculate_account_projected_margin_ratio`                                                                                     | Account snapshot margin/equity after optional new exposure        |
| `calculate_symbol_group_margin_ratio`                                                                                          | Estimated symbol-group margin/equity with optional exposure       |
| `determine_order_limits`                                                                                                       | SL/TP price levels from ratios                                    |
| `calculate_trailing_stop_updates`                                                                                              | Per-ticket generic trailing stop-loss update plan                 |
| `ensure_symbol_selected`                                                                                                       | Select/verify Market Watch visibility                             |
| `fetch_recent_history_deals_for_trading_client`                                                                                | Recent deal history from a connected trading client               |
| `place_market_order`, `close_open_positions`, `update_sltp_for_open_positions`, `update_trailing_stop_loss_for_open_positions` | Order execution helpers (`dry_run` supported)                     |
| `MarginVolume`, `OrderLimits`, `OrderExecutionResult`                                                                          | Typed return contracts for order helpers                          |
| `OrderSide`, `OrderFillingMode`, `OrderTimeMode`, `PositionSide`, `ExecutionStatus`                                            | Typed enums for order helpers                                     |
| `ProjectionMode`                                                                                                               | Literal type for `calculate_symbol_group_margin_ratio` projection |

`calculate_symbol_group_margin_ratio` accepts an optional `projection_mode`
parameter (`"add"` by default). Pass `projection_mode="replace_symbol"` to
subtract current exposure for `new_symbol` before adding the candidate margin —
useful for reversal-style projections. mt5cli only calculates broker-facing
exposure; downstream applications own thresholds, risk guard actions, and
strategy policy.

`MT5Client.order_send()` and CLI `order-send --yes` are live execution paths.

Order helpers validate broker stop-level distance in `determine_order_limits()` and
raise `Mt5OperationError` when computed SL/TP prices are too close to the entry
quote. Validation uses `trade_stops_level * point` from the current quote and
symbol metadata as a pre-check only; it does not guarantee live order acceptance
after price movement and does not inspect `trade_freeze_level`. Live
`place_market_order()` and SL/TP updates call
`ensure_symbol_selected()` so hidden symbols are added to Market Watch before
sending requests. Failed, malformed, or unknown broker retcodes are fail-closed
and returned as `status="failed"` with normalized `request` / `response` details;
`dry_run=True` never calls `ensure_symbol_selected()` or `order_send()`.

### Grafana observability (SQLite read model)

These helpers prepare a SQLite database as a Grafana datasource. All DDL is
idempotent (`CREATE TABLE IF NOT EXISTS`, `DROP VIEW IF EXISTS` + `CREATE
VIEW`, `CREATE INDEX IF NOT EXISTS`). Missing source tables are skipped with a
warning rather than raising an error.

| Symbol                             | Role                                                                                            |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| `update_observability`             | Append one timestamped snapshot row per data type; accepts an already-connected `Mt5DataClient` |
| `update_observability_with_config` | Standalone wrapper: opens/closes MT5 connection automatically around `update_observability`     |

Both functions write to the SQLite path given by `output=`. The optional
`symbols` parameter filters `positions_get` / `orders_get` by symbol.
`with_grafana_schema=False` (default) skips Grafana view/index setup; run
`grafana-schema` once to set up the schema, then call `snapshot` repeatedly
without this flag.

**Snapshot tables** (created by `create_snapshot_tables` in `mt5cli.grafana`):

| Table                | Content                                   |
| -------------------- | ----------------------------------------- |
| `account_snapshots`  | Balance, equity, margin, free-margin, P&L |
| `position_snapshots` | Open positions: symbol, volume, profit, … |
| `order_snapshots`    | Active orders: symbol, type, price, …     |
| `terminal_snapshots` | Terminal connectivity and build info      |
| `snapshot_runs`      | Per-run status (`ok` / `error`) timestamp |

**Grafana time-series views** (integer epoch-second `time` column; snapshot views also expose `run_id`):

| View                         | Source                           |
| ---------------------------- | -------------------------------- |
| `grafana_rates`              | `rates` table                    |
| `grafana_ticks`              | `ticks` table                    |
| `grafana_history_deals`      | `history_deals`                  |
| `grafana_history_orders`     | `history_orders`                 |
| `grafana_trade_deals`        | `history_deals` trade types only |
| `grafana_cash_events`        | `history_deals` non-trade events |
| `grafana_symbol_pnl`         | Per-close-deal P&L per symbol    |
| `grafana_account_snapshots`  | `account_snapshots`              |
| `grafana_position_snapshots` | `position_snapshots`             |
| `grafana_order_snapshots`    | `order_snapshots`                |
| `grafana_terminal_snapshots` | `terminal_snapshots`             |

**Grafana static summary views** (no `time` column; use for table/stat panels, not time-series):

| View                   | Source                                |
| ---------------------- | ------------------------------------- |
| `grafana_realized_pnl` | Cumulative realized PnL per symbol    |
| `grafana_trade_stats`  | Win/loss counts and profit per symbol |

Lower-level helpers (`ensure_grafana_schema`, `create_grafana_views`,
`create_grafana_indexes`, `create_snapshot_tables`, `start_snapshot_run`,
`insert_account_snapshot`, `insert_position_snapshots`, `insert_order_snapshots`,
`insert_terminal_snapshot`, `record_snapshot_run`) are available directly from
`mt5cli.grafana` and are not part of the package-root stable surface.

### Errors

| Symbol                                                                     | Role                          |
| -------------------------------------------------------------------------- | ----------------------------- |
| `Mt5CliError`, `Mt5ConnectionError`, `Mt5OperationError`, `Mt5SchemaError` | Stable mt5cli exception types |

## Module-scoped helpers

Lower-level helpers are available from their owning modules and are not part
of the package-root stable surface. Import them directly when needed:

| Module              | Examples                                                                                                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mt5cli.grafana`    | `ensure_grafana_schema`, `create_grafana_views`, `create_grafana_indexes`, `create_snapshot_tables`, `start_snapshot_run`, `insert_account_snapshot`, `record_snapshot_run` |
| `mt5cli.history`    | `resolve_rate_view_name`, `resolve_rate_tables`, `load_rate_data`, `build_rate_view_name`                                                                                   |
| `mt5cli.sdk`        | `copy_rates_from`, `copy_ticks_from`, `account_info`, `symbols`, `mt5_summary`, `latest_rates`                                                                              |
| `mt5cli.schemas`    | `DataKind`, `normalize_dataframe`, `validate_schema`, `DEDUP_KEYS`                                                                                                          |
| `mt5cli.utils`      | `Dataset`, `IfExists`, `detect_format`, `export_dataframe`, `export_dataframe_to_sqlite`                                                                                    |
| `mt5cli.converters` | `normalize_symbol`, `ensure_utc`, `parse_date_range`, `granularity_name`                                                                                                    |
| `mt5cli.exceptions` | `normalize_mt5_exception`, `call_with_normalized_errors`, `is_recoverable_mt5_error`                                                                                        |

## CLI commands

The Typer application in `mt5cli.cli` exposes file-export commands documented in
[CLI Module](cli.md) and the project README. CLI commands:

- Require `-o/--output` and write CSV, JSON, Parquet, or SQLite.
- Accept global MT5 connection options (`--login`, `--password`, `--server`,
  `--path`, `--timeout`).
- Delegate to the same Python APIs described here; they are not duplicated
  business logic.

`grafana-schema` initializes Grafana views, indexes, and snapshot tables in the
target SQLite database without connecting to MT5. It is idempotent and safe to
run repeatedly.

`snapshot` appends one timestamped row per enabled data type
(`--with-account`, `--with-positions`, `--with-orders`, `--with-terminal`) and
never places orders or modifies trading state. Both commands require
`-o/--output` to point at a `.db` / SQLite file.

`order-send` is the expert raw-request path; it requires `--yes` and a fully
constructed request payload. `close-positions` is the safer high-level helper
that closes open positions by `--symbol` or `--ticket` using
`close_open_positions()`. Both `order-send --yes` and `close-positions --yes`
are live execution paths. `close-positions --dry-run` previews close orders
without placing them and does not require `--yes`.

## Internal helpers (not stable)

Do not import these for downstream contracts; they may change without a semver
notice:

| Module                   | Examples                                                                  |
| ------------------------ | ------------------------------------------------------------------------- |
| `mt5cli.sdk`             | `connected_client`, `_run_with_client`, private coercion helpers          |
| `mt5cli.history`         | `write_*_dataset`, `deduplicate_history_tables`, `parse_sqlite_timestamp` |
| `mt5cli.retry`           | `retry_with_backoff`                                                      |
| `mt5cli.cli`             | Typer command handlers and Click parameter types                          |
| Leading-underscore names | Any `_`-prefixed function or method                                       |

Use the package-root stable exports instead of reaching into submodule
internals.

## Explicitly out of scope

mt5cli must **not** implement downstream strategy or research responsibilities.
The following belong in consuming applications, not in mt5cli:

- Signal detection (for example AR-GARCH or other model-specific triggers)
- Backtesting, walk-forward analysis, or parameter optimization
- Strategy-specific risk policy, position sizing systems, or Kelly fractions
- Entry/exit decision logic or YAML strategy semantics
- Entry-deal classification, Kelly fractions, or betting-specific deal transformations
  (use `fetch_recent_history_deals_for_trading_client` to retrieve raw deal data, then
  apply downstream transformations in your own adapter layer)
- Application-specific credential schema keys wired into mt5cli internals

mt5cli provides connection lifecycle, normalized data access, SQLite history
machinery, closed-bar helpers, generic margin/volume/spread/SL/TP utilities, and
optional order primitives so downstream apps can focus on strategy code behind
their own adapter layer.

## Contract verification

`tests/test_contracts.py` asserts that every name in `STABLE_SDK_EXPORTS` is
importable from `mt5cli`, that all package-root exports are covered by the
stable set, and documents key closed-bar, SQLite loading, account-resolution,
and trading-session behaviors.
