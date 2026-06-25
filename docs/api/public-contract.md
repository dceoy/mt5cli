# Public API Contract

mt5cli is the canonical operational trading SDK and CLI/batch layer over pdmt5.
The intended dependency direction is:

```text
downstream app -> mt5cli -> pdmt5 -> MetaTrader 5
mt5api         -> pdmt5 -> MetaTrader 5
```

mt5cli must not depend on mt5api.

## Responsibility boundary

| Layer            | Owns                                                                                                                                          |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **pdmt5**        | MT5 core wrapper; DataFrame/dict conversion; canonical MT5 constants and parsers; direct low-level order primitives                           |
| **mt5cli**       | CLI/batch workflows; SQLite history collection; normalized datasets; closed-bar helpers; small downstream operational SDK; generic broker-facing margin/volume/order orchestration |
| **mt5api**       | HTTP/FastAPI adapter over pdmt5; not a dependency of mt5cli                                                                                   |
| **downstream**   | Strategy logic; signals; risk policy; backtesting; optimization; YAML/application semantics                                                   |

Downstream code should import raw pdmt5 types and constants (such as
`Mt5Config`, `Mt5TradingClient`, `Mt5RuntimeError`, `Mt5TradingError`,
`TIMEFRAME_MAP`, `COPY_TICKS_MAP`) directly from `pdmt5` when needed.
mt5cli does not serve as a pass-through compatibility namespace for pdmt5.

Downstream packages should import from the package root (`from mt5cli import
...`) and use the public tier sets in `mt5cli.contract` to distinguish API
stability. CLI commands mirror the same behavior but are not importable Python
APIs.

## Public API tiers

mt5cli classifies package-root imports by intended downstream use:

| Tier             | Contract set               | Meaning                                                                                                                                     |
| ---------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Stable core      | `STABLE_SDK_EXPORTS`       | Preferred SDK surface for downstream MT5 infrastructure adapters. Changes require a deliberate compatibility path.                          |
| Secondary public | `SECONDARY_PUBLIC_EXPORTS` | Public helpers for CLI/export/schema integrations and lower-level MT5 wrappers. Importable, but less central to the downstream trading SDK. |

## Stable downstream SDK API

These names are exported from `mt5cli` and covered by the contract in
`mt5cli.STABLE_SDK_EXPORTS` (defined in `mt5cli.contract`).

### Session lifecycle and configuration

| Symbol                                          | Role                                                                                                                                                                                                                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MT5Client`                                     | Read-only data client with optional `order_check` / `order_send`                                                                                                                                                                                                                  |
| `build_config`                                  | Build `pdmt5.Mt5Config` from connection fields; `login` accepts `int \| str \| None` — numeric strings are coerced to `int`, blank strings are treated as unset, and `${ENV_VAR}` / `$ENV_NAME` placeholders in string parameters are expanded when `allow_whole_dollar_env=True` |
| `mt5_session`                                   | Context manager: initialize, login, yield client, shutdown                                                                                                                                                                                                                        |
| `create_trading_client`, `mt5_trading_session`  | Trading-capable `pdmt5.Mt5TradingClient` lifecycle                                                                                                                                                                                                                                |
| `AccountSpec`                                   | Generic account group: symbols plus optional credentials                                                                                                                                                                                                                          |
| `resolve_account_spec`, `resolve_account_specs` | Merge overrides and expand `${ENV_VAR}` placeholders; opt-in `allow_whole_dollar_env` for bare `$NAME`                                                                                                                                                                            |
| `substitute_env_placeholders`                   | Replace `${NAME}` substrings from the environment; opt-in `allow_whole_dollar_env` for whole-value `$NAME`                                                                                                                                                                        |
| `substitute_mapping_values`                     | Recursively traverse a dict/list/scalar structure and substitute `${ENV_VAR}` placeholders for caller-selected mapping keys only; optionally normalise blank strings to `None` for a separate caller-selected key set; does not hard-code any application-specific key names      |

Credential resolution is generic: any environment variable name may appear inside
`${...}`. mt5cli does not hard-code application-specific keys such as
`mt5_login` or `mt5_exe`.

Pass `allow_whole_dollar_env=True` to `substitute_env_placeholders()`,
`substitute_mapping_values()`, `resolve_account_spec()`, `resolve_account_specs()`,
and `build_config()` to additionally expand strings whose entire value is a bare
`$ENV_NAME` identifier.
Partial strings such as `"plan$pass"`, `"abc$ENV"`, or `"$ENV-suffix"` are
**never** expanded — only an exact `$IDENTIFIER` whole-string match qualifies.
Default is `False` to preserve backward compatibility.

### Closed-bar rate helpers

MetaTrader 5 returns the still-forming bar as the last row when
`start_pos=0`. Use these helpers instead of reimplementing bar trimming or
timestamp normalization in downstream apps.

| Symbol                                           | Role                                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------------------- |
| `drop_forming_rate_bar`                          | Remove the last row from chronologically ordered rate data                      |
| `fetch_latest_closed_rates`                      | Single connected client: fetch `count + 1`, drop forming bar                    |
| `fetch_latest_closed_rates_for_trading_client`   | Closed bars from an active `Mt5TradingClient` session; returns RangeIndex       |
| `fetch_latest_closed_rates_indexed`              | Same as above but returns a UTC `DatetimeIndex` named `"time"` (no time column) |
| `collect_latest_closed_rates_for_accounts`       | Multi-account closed bars with optional retry wrapper                           |
| `collect_latest_closed_rates_by_granularity`     | Same data keyed by `(symbol, granularity_name)`                                 |
| `collect_latest_rates_for_accounts_with_retries` | Bounded exponential backoff for transient MT5 errors                            |

### SQLite history collection and rate loading

| Symbol                                                                                                                        | Role                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `collect_history`                                                                                                             | One-shot date-range export into SQLite                                                       |
| `update_history`, `update_history_with_config`                                                                                | Incremental append from `MAX(time)` cursors                                                  |
| `ThrottledHistoryUpdater`                                                                                                     | Minimum interval between successful incremental updates; optional `update_backend` injection |
| `resolve_history_datasets`, `resolve_history_timeframes`, `resolve_history_tick_flags`                                        | History pipeline configuration                                                               |
| `build_rate_view_name`, `resolve_rate_table_name`, `resolve_rate_view_name`, `resolve_rate_view_names`, `resolve_rate_tables` | Map symbols/timeframes to mt5cli-managed table or view names                                 |
| `RateTarget`, `build_rate_targets`                                                                                            | Neutral `(symbol, timeframe)` series descriptors                                             |
| `load_rate_data`, `load_rate_data_from_connection`                                                                            | Load one table/view into a time-indexed DataFrame                                            |
| `load_rate_series_from_sqlite`, `load_rate_series_by_granularity`                                                             | Load one or many series; fail clearly when managed views are missing                         |

Pass `require_existing=True` to rate view resolution helpers when downstream
code must fail instead of receiving a best-guess view name. Multi-series loaders
require existing managed `rate_*__*` views unless `explicit_tables` is supplied.

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
raise `Mt5TradingError` when computed SL/TP prices are too close to the entry
quote. Validation uses `trade_stops_level * point` from the current quote and
symbol metadata as a pre-check only; it does not guarantee live order acceptance
after price movement and does not inspect `trade_freeze_level`. Live
`place_market_order()` and SL/TP updates call
`ensure_symbol_selected()` so hidden symbols are added to Market Watch before
sending requests. Failed, malformed, or unknown broker retcodes are fail-closed
and returned as `status="failed"` with normalized `request` / `response` details;
`dry_run=True` never calls `ensure_symbol_selected()` or `order_send()`.

### Errors

| Symbol                                                                               | Role                                         |
| ------------------------------------------------------------------------------------ | -------------------------------------------- |
| `Mt5CliError`, `Mt5ConnectionError`, `Mt5OperationError`, `Mt5SchemaError`           | Stable mt5cli exception types                |
| `normalize_mt5_exception`, `call_with_normalized_errors`, `is_recoverable_mt5_error` | Error normalization and retry classification |

## Secondary public exports

These names remain importable from `mt5cli` and are covered by
`SECONDARY_PUBLIC_EXPORTS`, but they are oriented toward CLI/export/schema
integrations, parsing, and lower-level MT5 access rather than the stable core
SDK surface. Prefer the stable symbols above for downstream infrastructure
adapters.

### Read-only MT5 data wrappers

Module-level helpers open a transient connection per call. Prefer `mt5_session`
or `MT5Client` when making many requests in one process.

| Area                 | Symbols                                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| Rates                | `copy_rates_from`, `copy_rates_from_pos`, `copy_rates_range`, `latest_rates`, `collect_latest_rates` |
| Ticks                | `copy_ticks_from`, `copy_ticks_range`, `recent_ticks`                                                |
| Account / terminal   | `account_info`, `terminal_info`, `mt5_version`, `last_error`, `mt5_summary`, `mt5_summary_as_df`     |
| Symbols / market     | `symbols`, `symbol_info`, `symbol_info_tick`, `market_book`, `minimum_margins`                       |
| Trading state (read) | `orders`, `positions`, `history_orders`, `history_deals`, `recent_history_deals`                     |
| Multi-account rates  | `collect_latest_rates_for_accounts`                                                                  |

Use `mt5_version` for MetaTrader 5 terminal version data. The name `version` at
the package root refers to `importlib.metadata.version` (package metadata), not
the MT5 SDK helper.

### Schema, export, and parser helpers

| Area                 | Symbols                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------- |
| Dataset contracts    | `DataKind`, `Dataset`, `IfExists`, `DEDUP_KEYS`, `REQUIRED_COLUMNS`, `TIME_COLUMNS`, `KNOWN_MT5_TIME_COLUMNS` |
| Schema normalization | `normalize_dataframe`, `normalize_time_columns`, `schema_columns`, `validate_schema`                          |
| Export helpers       | `detect_format`, `export_dataframe`, `export_dataframe_to_sqlite`                                             |
| Symbol parsing       | `normalize_symbol`, `normalize_symbols`                                                                       |
| Time parsing         | `ensure_utc`, `parse_date_range`, `parse_datetime`, `recent_window`                                           |
| MT5 parsing maps     | `granularity_name`, `parse_tick_flags`, `parse_timeframe`                                                     |
| Trading data shapes  | `POSITION_COLUMNS`                                                                                            |

## CLI commands

The Typer application in `mt5cli.cli` exposes file-export commands documented in
[CLI Module](cli.md) and the project README. CLI commands:

- Require `-o/--output` and write CSV, JSON, Parquet, or SQLite.
- Accept global MT5 connection options (`--login`, `--password`, `--server`,
  `--path`, `--timeout`).
- Delegate to the same Python APIs described here; they are not duplicated
  business logic.

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
- Application-specific credential schema keys wired into mt5cli internals

mt5cli provides connection lifecycle, normalized data access, SQLite history
machinery, closed-bar helpers, generic margin/volume/spread/SL/TP utilities, and
optional order primitives so downstream apps can focus on strategy code behind
their own adapter layer.

## Contract verification

`tests/test_contracts.py` asserts that every name in the stable and secondary
tier sets is importable from `mt5cli`, documents key closed-bar, rate-view,
SQLite loading, account-resolution, and trading-session behaviors, and keeps the
tier sets aligned with `__all__`.
