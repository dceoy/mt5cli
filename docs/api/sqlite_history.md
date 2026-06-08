# SQLite History Module

::: mt5cli.sqlite_history

## `collect-history` schema

The `collect-history` command (and the matching `collect_history` SDK function) writes
selected MT5 datasets into one SQLite database. Each dataset becomes a table; column
names and types mirror the pdmt5 DataFrame schema for that export, with two additions:

- `symbol` is prepended on every table.
- `timeframe` is prepended on `rates` so appended runs at different bar sizes stay
  distinguishable.

SQLite does not declare foreign keys. Rows are linked logically by `symbol`, time
windows, and (for deals) `position_id` / `order`. Duplicate rows are removed on
append using dataset-specific keys (for example `ticket` on history tables, or
`(symbol, timeframe, time)` on rates).

Optional views are created when `--with-views` is set and the `history-deals` dataset
was written.

### Entity-relationship diagram

Sample layout for a full collection with `--with-views`:

```mermaid
erDiagram
    rates {
        TEXT symbol "dedup key"
        INTEGER timeframe "dedup key"
        TEXT time "dedup key"
        REAL open
        REAL high
        REAL low
        REAL close
        INTEGER tick_volume
        INTEGER spread
        INTEGER real_volume
    }

    ticks {
        TEXT symbol "dedup key"
        TEXT time "dedup key"
        INTEGER time_msc "dedup key (preferred)"
        REAL bid
        REAL ask
        REAL last
        INTEGER volume
        INTEGER flags
        REAL volume_real
    }

    history_orders {
        INTEGER ticket "dedup key"
        TEXT symbol
        TEXT time
        INTEGER type
        INTEGER state
        REAL volume_initial
        REAL price_open
        REAL price_current
        INTEGER magic
    }

    history_deals {
        INTEGER ticket "dedup key"
        INTEGER order
        INTEGER position_id "groups position view"
        TEXT symbol
        TEXT time
        INTEGER type "0/1 trade, else cash event"
        INTEGER entry "0 IN, 1 OUT, 2 INOUT, 3 OUT_BY"
        REAL volume
        REAL price
        REAL profit
        REAL commission
        REAL swap
        REAL fee
    }

    cash_events {
        INTEGER ticket
        TEXT symbol
        TEXT time
        INTEGER type
        REAL profit
    }

    positions_reconstructed {
        INTEGER position_id
        TEXT symbol
        TEXT open_time
        TEXT close_time
        INTEGER direction
        REAL volume_open
        REAL volume_close
        REAL volume_reversal
        REAL open_price
        REAL close_price
        REAL total_profit
        INTEGER reversal_count
        INTEGER deals_count
    }

    rates ||--o{ history_deals : "symbol (logical)"
    ticks ||--o{ history_deals : "symbol (logical)"
    history_orders ||--o{ history_deals : "order ~ ticket (logical)"
    history_deals ||--|| cash_events : "VIEW: type NOT IN (0,1)"
    history_deals ||--o{ positions_reconstructed : "VIEW: GROUP BY position_id"
```

### Tables and views

| Object                    | Kind  | Source               | Notes                                                                                       |
| ------------------------- | ----- | -------------------- | ------------------------------------------------------------------------------------------- |
| `rates`                   | table | `copy_rates_range`   | Indexed on `(symbol, timeframe, time)` when columns exist.                                  |
| `ticks`                   | table | `copy_ticks_range`   | Indexed on `(symbol, time)` when columns exist.                                             |
| `history_orders`          | table | `history_orders_get` | Fetched per `--symbol`, then concatenated.                                                  |
| `history_deals`           | table | `history_deals_get`  | Fetched per `--symbol`, then concatenated. Indexed on `(position_id, symbol)` when present. |
| `cash_events`             | view  | `history_deals`      | Non-trade deal types (deposits, balance ops, etc.). Requires `type` column.                 |
| `positions_reconstructed` | view  | `history_deals`      | One row per closed `position_id`; volume-weighted prices and reversal stats.                |

Column sets can vary with terminal and pdmt5 version. Views are skipped with a warning
when required columns are missing.

### Incremental collection

The `update_history` SDK path uses the same base tables and optional
`cash_events` / `positions_reconstructed` views. It additionally maintains
`rate_<symbol>__<timeframe>` compatibility views when `create_rate_views=True`.
