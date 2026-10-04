"""Canonical trading analytics views and static-dashboard Parquet publication."""

from __future__ import annotations

import json
import logging
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .history import (
    VOLUME_EPSILON,
    create_deal_portions_view,
    get_table_columns,
    open_existing_sqlite_database,
)
from .utils import export_dataframe

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"

# Fixed-type columns are cast so Parquet types do not depend on the data. Time
# columns stay as stored because they are epoch numbers or naive text.
_INT_COLUMNS = frozenset({
    "position_id",
    "ticket",
    "leg_index",
    "event_count",
    "magic",
    "reversal_count",
    "deals_count",
    "holding_seconds",
    "trade_count",
    "wins",
    "losses",
    "run_id",
    "login",
    "leverage",
})
_FLOAT_COLUMNS = frozenset({
    "volume",
    "entry_price",
    "exit_price",
    "profit",
    "commission",
    "swap",
    "fee",
    "net_profit",
    "win_rate",
    "avg_trade",
    "gross_profit",
    "gross_loss",
    "profit_factor",
    "cumulative_net_profit",
    "avg_holding_seconds",
    "balance",
    "equity",
    "margin",
    "margin_free",
    "margin_level",
})
_STRING_COLUMNS = frozenset({
    "symbol",
    "side",
    "role",
    "date",
    "close_date",
    "currency",
})
_VIEW_DATASETS: tuple[tuple[str, str], ...] = (
    ("trades", "analytics_trades"),
    ("daily_pnl", "analytics_daily_pnl"),
    ("strategy_stats", "analytics_strategy_stats"),
    ("equity", "analytics_equity"),
)
_ACCOUNT_SNAPSHOTS_SQL = (
    'SELECT r."observed_at" AS "time", s.*'
    ' FROM "account_snapshots" s JOIN "snapshot_runs" r ON s."run_id" = r."run_id"'
    ' WHERE r."status" = \'ok\' ORDER BY r."observed_at"'
)


def _create_view(
    conn: sqlite3.Connection,
    name: str,
    select_sql: str,
    *,
    temporary: bool,
) -> None:
    conn.execute(f'DROP VIEW IF EXISTS {"temp." if temporary else ""}"{name}"')
    conn.execute(f'CREATE {"TEMP " if temporary else ""}VIEW "{name}" AS {select_sql}')


_CLOSE_DATE_SQL = (
    "CASE WHEN typeof({col}) IN ('integer', 'real')"
    " THEN date({col}, 'unixepoch') ELSE date({col}) END"
)
_TRADES_SQL = (
    "SELECT position_id, symbol, leg_index, magic, side, open_time, close_time,"  # noqa: S608
    f" {_CLOSE_DATE_SQL.format(col='close_time')} AS close_date,"
    " CASE"
    " WHEN typeof(open_time) IN ('integer', 'real')"
    " AND typeof(close_time) IN ('integer', 'real')"
    " THEN close_time - open_time"
    " WHEN typeof(open_time) = 'text' AND typeof(close_time) = 'text'"
    " THEN CAST(ROUND((julianday(close_time) - julianday(open_time))"
    " * 86400) AS INTEGER)"
    " END AS holding_seconds,"
    " volume, entry_price, exit_price, reversal_count, deals_count,"
    " profit, commission, swap, fee,"
    " profit + commission + swap + fee AS net_profit"
    " FROM (SELECT position_id, symbol, leg AS leg_index,"
    " COALESCE(MIN(CASE WHEN role = 'entry' THEN magic END), MIN(magic))"
    " AS magic,"
    " CASE MIN(CASE WHEN role = 'entry' THEN type END)"
    " WHEN 0 THEN 'buy' WHEN 1 THEN 'sell' END AS side,"
    " MIN(CASE WHEN role = 'entry' THEN time END) AS open_time,"
    " MAX(CASE WHEN role = 'exit' THEN time END) AS close_time,"
    " SUM(CASE WHEN role = 'entry' THEN volume ELSE 0 END) AS volume,"
    " SUM(CASE WHEN role = 'entry' THEN price * volume END)"
    " / NULLIF(SUM(CASE WHEN role = 'entry' THEN volume ELSE 0 END), 0)"
    " AS entry_price,"
    " SUM(CASE WHEN role = 'exit' THEN price * volume END)"
    " / NULLIF(SUM(CASE WHEN role = 'exit' THEN volume ELSE 0 END), 0)"
    " AS exit_price,"
    " SUM(entry = 2) AS reversal_count, COUNT(DISTINCT ticket) AS deals_count,"
    " SUM(profit) AS profit, SUM(commission) AS commission,"
    " SUM(swap) AS swap, SUM(fee) AS fee"
    " FROM deal_portions GROUP BY position_id, symbol, leg"
    " HAVING SUM(CASE WHEN role = 'exit' THEN volume ELSE 0 END)"
    f" >= SUM(CASE WHEN role = 'entry' THEN volume ELSE 0 END) - {VOLUME_EPSILON})"
)
# Cash basis: every deal portion lands at its own timestamp, including portions
# of legs that are still open (trades/strategy stats keep completed legs only).
_REALIZED_EVENTS_SQL = (
    "SELECT p.time AS time,"  # noqa: S608
    f" {_CLOSE_DATE_SQL.format(col='p.time')} AS date,"
    " p.ticket AS ticket, p.position_id AS position_id, p.symbol AS symbol,"
    " p.leg_magic AS magic, p.leg AS leg_index, p.role AS role,"
    " p.profit AS profit, p.commission AS commission, p.swap AS swap,"
    " p.fee AS fee, p.profit + p.commission + p.swap + p.fee AS net_profit"
    " FROM (SELECT *,"
    " COALESCE(MIN(CASE WHEN role = 'entry' THEN magic END) OVER leg_window,"
    " MIN(magic) OVER leg_window) AS leg_magic"
    " FROM deal_portions"
    " WINDOW leg_window AS (PARTITION BY position_id, symbol, leg)) p"
)

_METRICS_SQL = (
    "COUNT(*) AS trade_count,"
    " SUM(CASE WHEN net_profit > 0 THEN 1 ELSE 0 END) AS wins,"
    " SUM(CASE WHEN net_profit < 0 THEN 1 ELSE 0 END) AS losses,"
    " SUM(CASE WHEN net_profit > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) AS win_rate,"
    " SUM(profit) AS profit, SUM(commission) AS commission,"
    " SUM(swap) AS swap, SUM(fee) AS fee,"
    " SUM(net_profit) AS net_profit, AVG(net_profit) AS avg_trade,"
    " SUM(CASE WHEN net_profit > 0 THEN net_profit ELSE 0 END) AS gross_profit,"
    " SUM(CASE WHEN net_profit < 0 THEN net_profit ELSE 0 END) AS gross_loss,"
    " SUM(CASE WHEN net_profit > 0 THEN net_profit ELSE 0 END)"
    " / NULLIF(ABS(SUM(CASE WHEN net_profit < 0 THEN net_profit ELSE 0 END)), 0)"
    " AS profit_factor"
)

_DAILY_PNL_SQL = (
    "SELECT date, symbol, magic, COUNT(*) AS event_count,"
    " SUM(profit) AS profit, SUM(commission) AS commission,"
    " SUM(swap) AS swap, SUM(fee) AS fee, SUM(net_profit) AS net_profit"
    " FROM analytics_realized_events WHERE date IS NOT NULL"
    " GROUP BY date, symbol, magic"
)
_STRATEGY_STATS_SQL = (
    f"SELECT symbol, magic, {_METRICS_SQL},"  # noqa: S608
    " AVG(holding_seconds) AS avg_holding_seconds,"
    " MIN(open_time) AS first_open_time, MAX(close_time) AS last_close_time"
    " FROM analytics_trades GROUP BY symbol, magic"
)
_EQUITY_SQL = (
    "SELECT time, date, ticket, position_id, symbol, magic, leg_index, role,"
    " net_profit,"
    " SUM(net_profit) OVER (ORDER BY time, ticket, role, leg_index)"
    " AS cumulative_net_profit"
    " FROM analytics_realized_events"
)


def create_analytics_views(
    conn: sqlite3.Connection,
    *,
    temporary: bool = False,
) -> bool:
    """Create the canonical ``analytics_*`` views idempotently.

    ``analytics_trades`` has one row per completed trade leg. A leg is the
    stretch of a ``position_id`` between reversals: a ``DEAL_ENTRY_INOUT`` deal
    closes the current leg and opens the next (``leg_index``); its volume,
    commission and fee are split pro rata and its swap goes to the closing leg.
    A leg is complete once its closing volume covers
    its opening volume, so partially closed legs are excluded. For positions
    without reversals, leg 0 matches ``positions_reconstructed``. ``net_profit``
    is ``profit + commission + swap + fee`` with NULL treated as zero, and
    ``magic``/``commission``/``swap``/``fee`` come from ``history_deals``.
    ``analytics_realized_events`` lists every deal portion, including partially
    closed and still-open legs, at its own timestamp (cash basis) and feeds
    ``analytics_daily_pnl`` and ``analytics_equity``; events of a completed leg
    sum to its ``net_profit``. ``analytics_strategy_stats`` aggregates the legs by
    ``symbol`` and ``magic``. Stored trade-server wall-clock timestamps are
    preserved without implicit UTC conversion. A trade is a win when
    ``net_profit > 0`` and a loss when ``net_profit < 0``; break-even trades
    count as neither.

    Args:
        conn: SQLite connection holding ``history_deals``.
        temporary: Create connection-local TEMP views so a read-only database
            is never modified.

    Returns:
        True if the views were created, False if ``history_deals`` lacks the
        columns required for position reconstruction.
    """
    deal_columns = get_table_columns(conn, "history_deals")
    if not create_deal_portions_view(conn, deal_columns, temporary=temporary):
        return False
    for name, sql in (
        ("analytics_trades", _TRADES_SQL),
        ("analytics_realized_events", _REALIZED_EVENTS_SQL),
        ("analytics_daily_pnl", _DAILY_PNL_SQL),
        ("analytics_strategy_stats", _STRATEGY_STATS_SQL),
        ("analytics_equity", _EQUITY_SQL),
    ):
        _create_view(conn, name, sql, temporary=temporary)
    return True


def _account_snapshots_available(conn: sqlite3.Connection) -> bool:
    return {"run_id", "observed_at", "status"}.issubset(
        get_table_columns(conn, "snapshot_runs"),
    ) and "run_id" in get_table_columns(conn, "account_snapshots")


def _apply_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast fixed-type columns so empty or NULL-containing frames keep types.

    Returns:
        The frame with nullable ``Int64``, ``float64`` and ``string`` columns.
    """
    dtypes: dict[str, str] = {}
    for column in frame.columns:
        if column in _INT_COLUMNS:
            dtypes[column] = "Int64"
        elif column in _FLOAT_COLUMNS:
            dtypes[column] = "float64"
        elif column in _STRING_COLUMNS:
            dtypes[column] = "string"
    return frame.astype(dtypes)


def _collect_dataset_frames(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    if create_analytics_views(conn, temporary=True):
        for dataset, view in _VIEW_DATASETS:
            frames[dataset] = _apply_schema(
                pd.read_sql_query(  # pyright: ignore[reportUnknownMemberType]
                    f'SELECT * FROM "{view}"',  # noqa: S608
                    conn,
                ),
            )
    else:
        logger.warning("Skipping trade analytics: history_deals is missing or invalid")
    if _account_snapshots_available(conn):
        frames["account_snapshots"] = _apply_schema(
            pd.read_sql_query(  # pyright: ignore[reportUnknownMemberType]
                _ACCOUNT_SNAPSHOTS_SQL,
                conn,
            ),
        )
    else:
        logger.warning("Skipping account_snapshots: snapshot tables are missing")
    return frames


def _mt5cli_version() -> str | None:
    try:
        return version("mt5cli")
    except PackageNotFoundError:  # pragma: no cover
        return None


def publish_dashboard(
    source: str | Path,
    output_dir: str | Path,
    *,
    manifest_name: str = MANIFEST_NAME,
) -> dict[str, Any]:
    """Publish Parquet datasets and a manifest for static dashboards.

    Reads an existing SQLite history database without modifying it or
    connecting to MetaTrader 5, and writes ``trades``, ``daily_pnl``,
    ``strategy_stats``, ``equity`` and ``account_snapshots`` Parquet files plus
    a manifest into ``output_dir``. Datasets whose source tables are missing
    are skipped and omitted from the manifest. The manifest is written last
    so readers never see it before the datasets it lists.

    Args:
        source: Path to the SQLite history database.
        output_dir: Directory receiving the Parquet files and manifest.
        manifest_name: File name of the manifest inside ``output_dir``.

    Returns:
        The manifest as a dictionary.

    Raises:
        ValueError: If the source database is invalid or no dataset can be built.
    """
    conn, _ = open_existing_sqlite_database(source)
    with closing(conn):
        frames = _collect_dataset_frames(conn)
    if not frames:
        msg = f"No analytics datasets could be built from {source}"
        raise ValueError(msg)
    dataset_files = {f"{name}.parquet" for name in frames}
    manifest_basename = Path(manifest_name).name
    if manifest_basename in dataset_files:
        msg = f"Manifest name collides with dashboard dataset: {manifest_basename}"
        raise ValueError(msg)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mt5cli_version": _mt5cli_version(),
        "datasets": [
            {
                "name": name,
                "file": f"{name}.parquet",
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
            }
            for name, frame in frames.items()
        ],
    }
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        staged = Path(tmp)
        for name, frame in frames.items():
            export_dataframe(frame, staged / f"{name}.parquet", "parquet")
        (staged / manifest_name).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        for name in [f"{name}.parquet" for name in frames] + [manifest_name]:
            (staged / name).replace(out_dir / name)
    logger.info("Published %d dashboard datasets to %s", len(frames), out_dir)
    return manifest
