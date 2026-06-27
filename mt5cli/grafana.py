"""Grafana-oriented SQLite views, indexes, and snapshot tables."""

from __future__ import annotations

import contextlib
import datetime
import logging
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import cast

from .history import get_table_columns

logger = logging.getLogger(__name__)

_TRADE_DEAL_TYPES_SQL = "(0, 1)"

_GRAFANA_VIEW_NAMES = (
    "grafana_rates",
    "grafana_ticks",
    "grafana_history_deals",
    "grafana_history_orders",
    "grafana_trade_deals",
    "grafana_cash_events",
    "grafana_realized_pnl",
    "grafana_symbol_pnl",
    "grafana_trade_stats",
    "grafana_account_snapshots",
    "grafana_position_snapshots",
    "grafana_order_snapshots",
    "grafana_terminal_snapshots",
)


def _to_epoch_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return int(value.timestamp())
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _time_col_expr(col: str) -> str:
    return (
        f"CASE WHEN typeof(\"{col}\") IN ('integer', 'real')"
        f' THEN CAST("{col}" AS INTEGER)'
        f" ELSE CAST(strftime('%s', \"{col}\") AS INTEGER) END"
    )


def _create_view_safe(
    conn: sqlite3.Connection,
    name: str,
    select_sql: str,
) -> None:
    try:
        conn.execute(f'DROP VIEW IF EXISTS "{name}"')
        conn.execute(f'CREATE VIEW "{name}" AS {select_sql}')
    except sqlite3.Error as exc:
        logger.warning("Skipping view %s: %s", name, exc)


def _other_cols(all_cols: set[str], exclude: set[str]) -> list[str]:
    return sorted(all_cols - exclude)


# ---------------------------------------------------------------------------
# Snapshot table DDL
# ---------------------------------------------------------------------------

_SNAPSHOT_TABLE_DDLS: list[str] = [
    """CREATE TABLE IF NOT EXISTS snapshot_runs (
        run_id INTEGER PRIMARY KEY,
        observed_at INTEGER NOT NULL,
        status TEXT NOT NULL,
        detail TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS account_snapshots (
        run_id INTEGER NOT NULL,
        login INTEGER,
        currency TEXT,
        balance REAL,
        equity REAL,
        margin REAL,
        margin_free REAL,
        margin_level REAL,
        profit REAL,
        leverage INTEGER
    )""",
    """CREATE TABLE IF NOT EXISTS position_snapshots (
        run_id INTEGER NOT NULL,
        login INTEGER,
        ticket INTEGER,
        position_id INTEGER,
        symbol TEXT,
        type INTEGER,
        volume REAL,
        price_open REAL,
        price_current REAL,
        profit REAL,
        swap REAL,
        comment TEXT,
        magic INTEGER
    )""",
    """CREATE TABLE IF NOT EXISTS order_snapshots (
        run_id INTEGER NOT NULL,
        login INTEGER,
        ticket INTEGER,
        symbol TEXT,
        type INTEGER,
        volume_current REAL,
        price_open REAL,
        price_current REAL,
        state INTEGER,
        comment TEXT,
        magic INTEGER,
        time_setup INTEGER
    )""",
    """CREATE TABLE IF NOT EXISTS terminal_snapshots (
        run_id INTEGER NOT NULL,
        name TEXT,
        connected INTEGER,
        community_account INTEGER,
        trade_allowed INTEGER,
        trade_expert INTEGER,
        path TEXT,
        company TEXT,
        language TEXT
    )""",
]


def create_snapshot_tables(conn: sqlite3.Connection) -> None:
    """Create snapshot tables idempotently."""
    for ddl in _SNAPSHOT_TABLE_DDLS:
        conn.execute(ddl)


def start_snapshot_run(conn: sqlite3.Connection, observed_at: int) -> int:
    """Insert a snapshot_runs row with status 'running' and return its run_id.

    Returns:
        The auto-assigned run_id for the new row.
    """
    cursor = conn.execute(
        "INSERT INTO snapshot_runs (observed_at, status) VALUES (?, 'running')",
        (observed_at,),
    )
    return cast("int", cursor.lastrowid)


# ---------------------------------------------------------------------------
# View builders
# ---------------------------------------------------------------------------


def _build_grafana_rates(conn: sqlite3.Connection) -> None:
    cols = get_table_columns(conn, "rates")
    required = {"time", "symbol", "timeframe"}
    if not required.issubset(cols):
        logger.warning(
            "Skipping grafana_rates: rates table missing columns %s",
            sorted(required - cols),
        )
        return
    time_expr = _time_col_expr("time")
    others = _other_cols(cols, {"time"})
    other_sql = ", ".join(f'"{c}"' for c in others)
    _create_view_safe(
        conn,
        "grafana_rates",
        f'SELECT {time_expr} AS "time", {other_sql} FROM "rates"',  # noqa: S608
    )


def _build_grafana_ticks(conn: sqlite3.Connection) -> None:
    cols = get_table_columns(conn, "ticks")
    required = {"time", "symbol"}
    if not required.issubset(cols):
        logger.warning(
            "Skipping grafana_ticks: ticks table missing columns %s",
            sorted(required - cols),
        )
        return
    time_expr = _time_col_expr("time")
    others = _other_cols(cols, {"time"})
    other_sql = ", ".join(f'"{c}"' for c in others)
    _create_view_safe(
        conn,
        "grafana_ticks",
        f'SELECT {time_expr} AS "time", {other_sql} FROM "ticks"',  # noqa: S608
    )


def _build_grafana_history_deals(conn: sqlite3.Connection) -> None:
    cols = get_table_columns(conn, "history_deals")
    if "time" not in cols:
        logger.warning("Skipping grafana_history_deals: history_deals.time is missing")
        return
    time_expr = _time_col_expr("time")
    others = _other_cols(cols, {"time"})
    other_sql = ", ".join(f'"{c}"' for c in others)
    _create_view_safe(
        conn,
        "grafana_history_deals",
        f'SELECT {time_expr} AS "time", {other_sql} FROM "history_deals"',  # noqa: S608
    )


def _build_grafana_history_orders(conn: sqlite3.Connection) -> None:
    cols = get_table_columns(conn, "history_orders")
    if "time_setup" not in cols:
        logger.warning(
            "Skipping grafana_history_orders: history_orders.time_setup is missing"
        )
        return
    time_expr = _time_col_expr("time_setup")
    others = _other_cols(cols, set())
    other_sql = ", ".join(f'"{c}"' for c in others)
    _create_view_safe(
        conn,
        "grafana_history_orders",
        f'SELECT {time_expr} AS "time", {other_sql} FROM "history_orders"',  # noqa: S608
    )


def _build_grafana_trade_deals(conn: sqlite3.Connection) -> None:
    cols = get_table_columns(conn, "history_deals")
    required = {"time", "type"}
    if not required.issubset(cols):
        logger.warning(
            "Skipping grafana_trade_deals: history_deals missing columns %s",
            sorted(required - cols),
        )
        return
    time_expr = _time_col_expr("time")
    others = _other_cols(cols, {"time"})
    other_sql = ", ".join(f'"{c}"' for c in others)
    _create_view_safe(
        conn,
        "grafana_trade_deals",
        f'SELECT {time_expr} AS "time", {other_sql}'  # noqa: S608
        f' FROM "history_deals" WHERE "type" IN {_TRADE_DEAL_TYPES_SQL}',
    )


def _build_grafana_cash_events(conn: sqlite3.Connection) -> None:
    cols = get_table_columns(conn, "history_deals")
    required = {"time", "type"}
    if not required.issubset(cols):
        logger.warning(
            "Skipping grafana_cash_events: history_deals missing columns %s",
            sorted(required - cols),
        )
        return
    time_expr = _time_col_expr("time")
    others = _other_cols(cols, {"time"})
    other_sql = ", ".join(f'"{c}"' for c in others)
    _create_view_safe(
        conn,
        "grafana_cash_events",
        f'SELECT {time_expr} AS "time", {other_sql}'  # noqa: S608
        f' FROM "history_deals" WHERE "type" NOT IN {_TRADE_DEAL_TYPES_SQL}',
    )


def _build_grafana_realized_pnl(conn: sqlite3.Connection) -> None:
    cols = get_table_columns(conn, "history_deals")
    required = {"symbol", "profit", "type", "entry"}
    if not required.issubset(cols):
        logger.warning(
            "Skipping grafana_realized_pnl: history_deals missing columns %s",
            sorted(required - cols),
        )
        return
    _create_view_safe(
        conn,
        "grafana_realized_pnl",
        'SELECT "symbol",'  # noqa: S608
        ' SUM("profit") AS cumulative_pnl, COUNT(*) AS deal_count'
        ' FROM "history_deals"'
        f' WHERE "type" IN {_TRADE_DEAL_TYPES_SQL}'
        ' AND "entry" IN (1, 2, 3)'
        ' AND "symbol" IS NOT NULL AND "symbol" != \'\''
        ' GROUP BY "symbol"',
    )


def _build_grafana_symbol_pnl(conn: sqlite3.Connection) -> None:
    cols = get_table_columns(conn, "history_deals")
    required = {"time", "symbol", "profit", "type", "entry"}
    if not required.issubset(cols):
        logger.warning(
            "Skipping grafana_symbol_pnl: history_deals missing columns %s",
            sorted(required - cols),
        )
        return
    time_expr = _time_col_expr("time")
    select_parts = [f'{time_expr} AS "time"', '"symbol"', '"profit"']
    if "volume" in cols:
        select_parts.append('"volume"')
    if "price" in cols:
        select_parts.append('"price"')
    select_sql = ", ".join(select_parts)
    _create_view_safe(
        conn,
        "grafana_symbol_pnl",
        f'SELECT {select_sql} FROM "history_deals"'  # noqa: S608
        f' WHERE "type" IN {_TRADE_DEAL_TYPES_SQL}'
        ' AND "entry" IN (1, 2, 3)'
        ' AND "symbol" IS NOT NULL AND "symbol" != \'\'',
    )


def _build_grafana_trade_stats(conn: sqlite3.Connection) -> None:
    cols = get_table_columns(conn, "history_deals")
    required = {"symbol", "profit", "type"}
    if not required.issubset(cols):
        logger.warning(
            "Skipping grafana_trade_stats: history_deals missing columns %s",
            sorted(required - cols),
        )
        return
    has_entry = "entry" in cols
    entry_filter = ' AND "entry" IN (1, 2, 3)' if has_entry else ""
    _create_view_safe(
        conn,
        "grafana_trade_stats",
        'SELECT "symbol",'  # noqa: S608
        " COUNT(*) AS total_deals,"
        ' SUM(CASE WHEN "profit" > 0 THEN 1 ELSE 0 END) AS winning_deals,'
        ' SUM(CASE WHEN "profit" <= 0 THEN 1 ELSE 0 END) AS losing_deals,'
        ' SUM("profit") AS total_profit,'
        ' AVG("profit") AS avg_profit,'
        ' MAX("profit") AS max_profit,'
        ' MIN("profit") AS min_profit'
        ' FROM "history_deals"'
        f' WHERE "type" IN {_TRADE_DEAL_TYPES_SQL}'
        f"{entry_filter}"
        ' AND "symbol" IS NOT NULL AND "symbol" != \'\''
        ' GROUP BY "symbol"',
    )


def _build_snapshot_view(
    conn: sqlite3.Connection,
    view_name: str,
    table_name: str,
) -> None:
    cols = get_table_columns(conn, table_name)
    if not cols:
        logger.warning("Skipping %s: %s table missing", view_name, table_name)
        return
    if "run_id" not in cols:
        logger.warning("Skipping %s: %s missing run_id column", view_name, table_name)
        return
    others = _other_cols(cols, {"run_id"})
    run_cols = get_table_columns(conn, "snapshot_runs")
    if {"run_id", "observed_at", "status"}.issubset(run_cols):
        other_sql = (", " + ", ".join(f's."{c}"' for c in others)) if others else ""
        select_cols = f'r."observed_at" AS "time", s."run_id"{other_sql}'
        _create_view_safe(
            conn,
            view_name,
            f'SELECT {select_cols} FROM "{table_name}" s'  # noqa: S608
            f' JOIN "snapshot_runs" r ON s."run_id" = r."run_id"'
            f" WHERE r.\"status\" = 'ok'",
        )
    else:
        logger.warning("Skipping %s: snapshot_runs missing required columns", view_name)


def _build_grafana_account_snapshots(conn: sqlite3.Connection) -> None:
    _build_snapshot_view(conn, "grafana_account_snapshots", "account_snapshots")


def _build_grafana_position_snapshots(conn: sqlite3.Connection) -> None:
    _build_snapshot_view(conn, "grafana_position_snapshots", "position_snapshots")


def _build_grafana_order_snapshots(conn: sqlite3.Connection) -> None:
    _build_snapshot_view(conn, "grafana_order_snapshots", "order_snapshots")


def _build_grafana_terminal_snapshots(conn: sqlite3.Connection) -> None:
    _build_snapshot_view(conn, "grafana_terminal_snapshots", "terminal_snapshots")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_grafana_views(conn: sqlite3.Connection) -> None:
    """Create all Grafana-facing views idempotently.

    Missing source tables cause the affected view to be skipped with a warning;
    other views are unaffected. Stale views whose source table or required
    columns have disappeared are dropped before rebuild.
    """
    for name in _GRAFANA_VIEW_NAMES:
        conn.execute(f'DROP VIEW IF EXISTS "{name}"')
    _build_grafana_rates(conn)
    _build_grafana_ticks(conn)
    _build_grafana_history_deals(conn)
    _build_grafana_history_orders(conn)
    _build_grafana_trade_deals(conn)
    _build_grafana_cash_events(conn)
    _build_grafana_realized_pnl(conn)
    _build_grafana_symbol_pnl(conn)
    _build_grafana_trade_stats(conn)
    _build_grafana_account_snapshots(conn)
    _build_grafana_position_snapshots(conn)
    _build_grafana_order_snapshots(conn)
    _build_grafana_terminal_snapshots(conn)


def create_grafana_indexes(conn: sqlite3.Connection) -> None:
    """Create Grafana query performance indexes idempotently."""
    rates_cols = get_table_columns(conn, "rates")
    if {"time", "symbol", "timeframe"}.issubset(rates_cols):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rates_time_symbol_timeframe"
            ' ON "rates"("time", "symbol", "timeframe")',
        )

    ticks_cols = get_table_columns(conn, "ticks")
    if {"time", "symbol"}.issubset(ticks_cols):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticks_time_symbol"
            ' ON "ticks"("time", "symbol")',
        )

    deals_cols = get_table_columns(conn, "history_deals")
    if {"time", "symbol"}.issubset(deals_cols):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_deals_time_symbol"
            ' ON "history_deals"("time", "symbol")',
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_deals_symbol_time"
            ' ON "history_deals"("symbol", "time")',
        )

    orders_cols = get_table_columns(conn, "history_orders")
    if {"time_setup", "symbol"}.issubset(orders_cols):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_orders_time_setup_symbol"
            ' ON "history_orders"("time_setup", "symbol")',
        )

    if {"run_id", "login"}.issubset(get_table_columns(conn, "account_snapshots")):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_account_snapshots_time_login"
            ' ON "account_snapshots"("run_id", "login")',
        )
    if {"run_id", "symbol"}.issubset(get_table_columns(conn, "position_snapshots")):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_position_snapshots_time_symbol"
            ' ON "position_snapshots"("run_id", "symbol")',
        )
    if {"run_id", "symbol"}.issubset(get_table_columns(conn, "order_snapshots")):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_order_snapshots_time_symbol"
            ' ON "order_snapshots"("run_id", "symbol")',
        )
    if {"observed_at", "status"}.issubset(get_table_columns(conn, "snapshot_runs")):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshot_runs_time_status"
            ' ON "snapshot_runs"("observed_at", "status")',
        )


def ensure_grafana_schema(conn: sqlite3.Connection) -> None:
    """Create snapshot tables, Grafana views, and indexes idempotently."""
    create_snapshot_tables(conn)
    create_grafana_views(conn)
    create_grafana_indexes(conn)


def publish_grafana_copy(
    source: str | Path,
    target: str | Path,
) -> Path:
    """Publish a consistent SQLite copy for Grafana using the backup API.

    Uses the SQLite online backup API for a WAL-safe, consistent snapshot of
    the source database. Writes to a temporary file beside the target, then
    atomically replaces it so that a previous published copy is preserved if
    publishing fails.

    Args:
        source: Path to the source SQLite database.
        target: Destination path for the published copy.

    Returns:
        The resolved absolute target path.

    Raises:
        FileNotFoundError: If the source database does not exist.
    """
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_str = tempfile.mkstemp(
        dir=target_path.parent,
        suffix=".tmp",
        prefix=target_path.name + ".",
    )
    tmp_path = Path(tmp_str)
    try:
        os.close(tmp_fd)
        with (
            sqlite3.connect(source_path) as src,
            sqlite3.connect(tmp_path) as dst,
        ):
            src.backup(dst)
        tmp_path.replace(target_path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise

    logger.info("Published Grafana copy: %s -> %s", source_path, target_path)
    return target_path.resolve()


# ---------------------------------------------------------------------------
# Snapshot insert helpers
# ---------------------------------------------------------------------------


def insert_account_snapshot(
    conn: sqlite3.Connection,
    run_id: int,
    row: dict[str, object],
) -> None:
    """Append one account state row to account_snapshots."""
    conn.execute(
        "INSERT INTO account_snapshots"
        " (run_id, login, currency, balance, equity,"
        "  margin, margin_free, margin_level, profit, leverage)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            row.get("login"),
            row.get("currency"),
            row.get("balance"),
            row.get("equity"),
            row.get("margin"),
            row.get("margin_free"),
            row.get("margin_level"),
            row.get("profit"),
            row.get("leverage"),
        ),
    )


def insert_position_snapshots(
    conn: sqlite3.Connection,
    run_id: int,
    login: int | None,
    rows: list[dict[str, object]],
) -> None:
    """Append position rows to position_snapshots; no-op when rows is empty."""
    if not rows:
        return
    conn.executemany(
        "INSERT INTO position_snapshots"
        " (run_id, login, ticket, position_id, symbol, type, volume,"
        "  price_open, price_current, profit, swap, comment, magic)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                run_id,
                login,
                r.get("ticket"),
                r.get("position_id"),
                r.get("symbol"),
                r.get("type"),
                r.get("volume"),
                r.get("price_open"),
                r.get("price_current"),
                r.get("profit"),
                r.get("swap"),
                r.get("comment"),
                r.get("magic"),
            )
            for r in rows
        ],
    )


def insert_order_snapshots(
    conn: sqlite3.Connection,
    run_id: int,
    login: int | None,
    rows: list[dict[str, object]],
) -> None:
    """Append order rows to order_snapshots; no-op when rows is empty."""
    if not rows:
        return
    conn.executemany(
        "INSERT INTO order_snapshots"
        " (run_id, login, ticket, symbol, type, volume_current,"
        "  price_open, price_current, state, comment, magic, time_setup)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                run_id,
                login,
                r.get("ticket"),
                r.get("symbol"),
                r.get("type"),
                r.get("volume_current"),
                r.get("price_open"),
                r.get("price_current"),
                r.get("state"),
                r.get("comment"),
                r.get("magic"),
                _to_epoch_int(r.get("time_setup")),
            )
            for r in rows
        ],
    )


def insert_terminal_snapshot(
    conn: sqlite3.Connection,
    run_id: int,
    row: dict[str, object],
) -> None:
    """Append one terminal state row to terminal_snapshots."""
    conn.execute(
        "INSERT INTO terminal_snapshots"
        " (run_id, name, connected, community_account,"
        "  trade_allowed, trade_expert, path, company, language)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            row.get("name"),
            row.get("connected"),
            row.get("community_account"),
            row.get("trade_allowed"),
            row.get("trade_expert"),
            row.get("path"),
            row.get("company"),
            row.get("language"),
        ),
    )


def record_snapshot_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    detail: str | None = None,
) -> None:
    """Finalize a snapshot run by setting its status."""
    conn.execute(
        "UPDATE snapshot_runs SET status = ?, detail = ? WHERE run_id = ?",
        (status, detail, run_id),
    )
