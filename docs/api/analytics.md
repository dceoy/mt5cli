# Analytics

::: mt5cli.analytics

## Static dashboard workflow

mt5cli keeps SQLite as the canonical local store and publishes browser-facing
Parquet datasets from it, so a static site (for example DuckDB-Wasm) can query
the data without a live API or direct SQLite access. Visualization stays
outside mt5cli.

```bash
mt5cli -o dist/data/manifest.json publish-dashboard --sqlite3 history.db
```

```python
from mt5cli import publish_dashboard

manifest = publish_dashboard("history.db", "dist/data")
```

Publication never connects to MetaTrader 5 and never modifies the source
database (analytics views are created as connection-local TEMP views). It
requires the `parquet` extra (`pip install "mt5cli[parquet]"`).

| File                        | Source view                                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `trades.parquet`            | `analytics_trades`                                                                                    |
| `daily_pnl.parquet`         | `analytics_daily_pnl`                                                                                 |
| `strategy_stats.parquet`    | `analytics_strategy_stats`                                                                            |
| `equity.parquet`            | `analytics_equity`                                                                                    |
| `account_snapshots.parquet` | `account_snapshots` joined to successful runs                                                         |
| `manifest.json`             | `schema_version`, `generated_at`, `mt5cli_version`, and per dataset `name`, `file`, `rows`, `columns` |

Datasets whose source tables are missing are skipped and omitted from the
manifest. The manifest is written last.

## Analytics views

`create_analytics_views(conn)` creates the views idempotently in an existing
history database (`temporary=True` creates TEMP views for read-only databases).

- `analytics_trades`: one row per completed trade leg (the canonical leg model; `positions_reconstructed` remains the position-level compatibility view over raw deals). A leg is the stretch of
  a `position_id` between reversals: a `DEAL_ENTRY_INOUT` deal closes the
  current leg and opens the next one (`leg_index`). Its volume, commission and
  fee are split pro rata between the two legs; its swap goes wholly to the
  closing leg (MT5 reports `DEAL_SWAP` for the position being closed), and
  profit is realized on the closed part only. A leg is complete once its closing volume covers its opening
  volume (1e-9 lot tolerance), so partially closed legs and the still-open
  leg after a reversal are excluded. For positions without reversals, leg 0
  matches `positions_reconstructed`. Columns: `position_id`, `symbol`,
  `leg_index`, `magic`, `side`, `open_time`, `close_time`, `close_date`,
  `holding_seconds`, `volume`, `entry_price`, `exit_price` (volume-weighted),
  `reversal_count`, `deals_count`, `profit`, `commission`, `swap`, `fee`,
  `net_profit`. Numeric timestamps stay numeric epoch values; textual
  timestamps remain timezone-naive MT5 trade-server wall-clock values. No
  implicit UTC conversion is performed.
- `analytics_realized_events`: every deal portion, each at its own timestamp
  (cash basis: an entry's commission lands on the entry date and a partial
  close's profit on the day it is realized), including partially closed and
  still-open legs. Events of a completed leg sum to that leg's `net_profit`,
  so cumulative equity also reflects entry costs and partial-close profit of
  legs that have not completed yet.
- `analytics_daily_pnl`: per `date`, `symbol`, `magic` from the realized
  events: `event_count`, `profit`, `commission`, `swap`, `fee`, `net_profit`.
- `analytics_strategy_stats`: per `symbol`, `magic` from the trade legs
  (trade count, wins, losses, win rate, gross profit/loss, profit factor), plus
  `avg_holding_seconds`, `first_open_time`, `last_close_time`.
- `analytics_equity`: one row per realized event in time order with
  `cumulative_net_profit` (realized P&L only).

Net P/L is:

```text
net_profit = COALESCE(profit, 0) + COALESCE(commission, 0)
           + COALESCE(swap, 0) + COALESCE(fee, 0)
```

Missing optional deal columns are treated as zero (or NULL for `magic`). A trade
is a win when `net_profit > 0` and a loss when `net_profit < 0`; break-even
trades count as neither. `gross_profit` and `gross_loss` sum positive and
negative trade `net_profit`, and `profit_factor = gross_profit / ABS(gross_loss)`
is NULL without losses. Metrics needing unavailable state (such as MAE/MFE) are
out of scope.

Parquet columns with a fixed meaning are written with stable types (`Int64`,
`float64`, `string`) even for empty tables; time columns keep the stored
representation.
