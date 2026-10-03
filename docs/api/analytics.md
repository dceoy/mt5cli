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

- `analytics_trades`: one row per closed position, reusing the
  `positions_reconstructed` reconstruction (partial closes and
  `DEAL_ENTRY_INOUT` reversals) plus per-position `magic`, `commission`,
  `swap`, and `fee`. Columns: `position_id`, `symbol`, `magic`, `side`,
  `open_time`, `close_time` (epoch seconds), `close_date` (UTC),
  `holding_seconds`, `volume`, `entry_price`, `exit_price` (volume-weighted),
  `reversal_count`, `deals_count`, `profit`, `commission`, `swap`, `fee`,
  `net_profit`.
- `analytics_daily_pnl`: per UTC close `date`, `symbol`, `magic`.
- `analytics_strategy_stats`: per `symbol`, `magic`, plus
  `avg_holding_seconds`, `first_open_time`, `last_close_time`.
- `analytics_equity`: per closed trade in close order with
  `cumulative_net_profit` (realized P&L only).

Net P/L is `COALESCE(profit, 0) + COALESCE(commission, 0) + COALESCE(swap, 0)

- COALESCE(fee, 0)`. Missing optional deal columns are treated as zero (or NULL
for `magic`). A trade is a win when `net_profit > 0`and a loss when`net_profit < 0`; break-even trades count as neither. `gross_profit`and`gross_loss`sum positive and negative trade`net_profit`, and
`profit_factor = gross_profit / ABS(gross_loss)` is NULL without losses.
  Metrics needing unavailable state (such as MAE/MFE) are out of scope.

The existing `grafana_*` views are unchanged.
