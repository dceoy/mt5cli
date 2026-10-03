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
    create_positions_reconstructed_view,
    get_table_columns,
    open_existing_sqlite_database,
)
from .utils import export_dataframe

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"

_TRADE_DEAL_TYPES_SQL = "(0, 1)"
_COST_COLUMNS = ("commission", "swap", "fee")
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


def time_col_expr(col: str) -> str:
    """Return SQL converting a TEXT or numeric time column to epoch seconds."""
    return (
        f"CASE WHEN typeof(\"{col}\") IN ('integer', 'real')"
        f' THEN CAST("{col}" AS INTEGER)'
        f" ELSE CAST(strftime('%s', \"{col}\") AS INTEGER) END"
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


def _trades_select_sql(deal_columns: set[str]) -> str:
    """Build the analytics_trades query over positions_reconstructed.

    Returns:
        SELECT statement joining reconstructed positions to per-position costs.
    """
    magic = (
        'COALESCE(MIN(CASE WHEN "entry" = 0 THEN "magic" END), MIN("magic"))'
        if "magic" in deal_columns
        else "NULL"
    )
    costs = ", ".join(
        f'SUM(COALESCE("{col}", 0)) AS "{col}"'
        if col in deal_columns
        else f'0 AS "{col}"'
        for col in _COST_COLUMNS
    )
    close_date = (
        "CASE WHEN typeof(p.close_time) IN ('integer', 'real')"
        " THEN date(p.close_time, 'unixepoch')"
        " ELSE date(p.close_time) END"
    )
    holding_seconds = (
        "CASE"
        " WHEN typeof(p.open_time) IN ('integer', 'real')"
        " AND typeof(p.close_time) IN ('integer', 'real')"
        " THEN p.close_time - p.open_time"
        " WHEN typeof(p.open_time) = 'text' AND typeof(p.close_time) = 'text'"
        " THEN CAST(ROUND((julianday(p.close_time) - julianday(p.open_time))"
        " * 86400) AS INTEGER)"
        " END"
    )
    return (
        "SELECT p.position_id AS position_id, p.symbol AS symbol,"  # noqa: S608
        " c.magic AS magic,"
        " CASE p.direction WHEN 0 THEN 'buy' WHEN 1 THEN 'sell' END AS side,"
        " p.open_time AS open_time, p.close_time AS close_time,"
        f" {close_date} AS close_date,"
        f" {holding_seconds} AS holding_seconds,"
        " p.volume_open AS volume, p.open_price AS entry_price,"
        " p.close_price AS exit_price, p.reversal_count AS reversal_count,"
        " p.deals_count AS deals_count,"
        " COALESCE(p.total_profit, 0) AS profit,"
        " c.commission AS commission, c.swap AS swap, c.fee AS fee,"
        " COALESCE(p.total_profit, 0) + COALESCE(c.commission, 0)"
        " + COALESCE(c.swap, 0) + COALESCE(c.fee, 0) AS net_profit"
        " FROM positions_reconstructed p"
        " JOIN (SELECT position_id, symbol,"
        f" {magic} AS magic, {costs}"
        " FROM history_deals"
        f" WHERE type IN {_TRADE_DEAL_TYPES_SQL} AND position_id != 0"
        " GROUP BY position_id, symbol) c"
        " ON c.position_id = p.position_id AND c.symbol IS p.symbol"
        " WHERE p.reversal_count > 0 OR p.volume_close >= p.volume_open"
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
    f"SELECT close_date AS date, symbol, magic, {_METRICS_SQL}"  # noqa: S608
    " FROM analytics_trades WHERE close_date IS NOT NULL"
    " GROUP BY close_date, symbol, magic"
)
_STRATEGY_STATS_SQL = (
    f"SELECT symbol, magic, {_METRICS_SQL},"  # noqa: S608
    " AVG(holding_seconds) AS avg_holding_seconds,"
    " MIN(open_time) AS first_open_time, MAX(close_time) AS last_close_time"
    " FROM analytics_trades GROUP BY symbol, magic"
)
_EQUITY_SQL = (
    "SELECT close_time AS time, close_date AS date, position_id, symbol, magic,"
    " net_profit,"
    " SUM(net_profit) OVER (ORDER BY close_time, position_id)"
    " AS cumulative_net_profit"
    " FROM analytics_trades WHERE close_time IS NOT NULL"
)


def create_analytics_views(
    conn: sqlite3.Connection,
    *,
    temporary: bool = False,
) -> bool:
    """Create the canonical ``analytics_*`` views idempotently.

    ``analytics_trades`` joins the existing ``positions_reconstructed``
    reconstruction with completed partial-close sequences and
    ``DEAL_ENTRY_INOUT`` reversals, plus
    per-position ``magic``, ``commission``, ``swap`` and ``fee`` totals from
    ``history_deals``. ``net_profit`` is ``profit + commission + swap + fee``
    with NULL treated as zero. Stored trade-server wall-clock timestamps are
    preserved without implicit UTC conversion. ``analytics_daily_pnl``,
    ``analytics_strategy_stats`` and ``analytics_equity`` aggregate that view
    by ``symbol`` and ``magic``. A trade is a win when ``net_profit > 0`` and
    a loss when ``net_profit < 0``; break-even trades count as neither.

    Args:
        conn: SQLite connection holding ``history_deals``.
        temporary: Create connection-local TEMP views so a read-only database
            is never modified.

    Returns:
        True if the views were created, False if ``history_deals`` lacks the
        columns required by the position reconstruction.
    """
    deal_columns = get_table_columns(conn, "history_deals")
    if not create_positions_reconstructed_view(conn, deal_columns, temporary=temporary):
        return False
    for name, sql in (
        ("analytics_trades", _trades_select_sql(deal_columns)),
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


def _collect_dataset_frames(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    if create_analytics_views(conn, temporary=True):
        for dataset, view in _VIEW_DATASETS:
            frames[dataset] = pd.read_sql_query(  # pyright: ignore[reportUnknownMemberType]
                f'SELECT * FROM "{view}"',  # noqa: S608
                conn,
            )
    else:
        logger.warning("Skipping trade analytics: history_deals is missing or invalid")
    if _account_snapshots_available(conn):
        frames["account_snapshots"] = pd.read_sql_query(  # pyright: ignore[reportUnknownMemberType]
            _ACCOUNT_SNAPSHOTS_SQL,
            conn,
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
