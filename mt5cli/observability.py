"""Observability snapshot orchestration for account, position, order, terminal state.

This module owns *when* and *what* to snapshot, using the canonical
:class:`~mt5cli.contract.ObservabilityClient` contract rather than raw pdmt5
method names, and the SQLite snapshot tables the results are persisted into.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .client import mt5_session
from .telemetry import get_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from pdmt5 import Mt5Config

    from .contract import ObservabilityClient

logger = logging.getLogger(__name__)

__all__ = [
    "ObservabilitySnapshot",
    "capture_observability_snapshot",
    "create_snapshot_tables",
    "insert_account_snapshot",
    "insert_order_snapshots",
    "insert_position_snapshots",
    "insert_terminal_snapshot",
    "persist_observability_snapshot",
    "record_snapshot_run",
    "start_snapshot_run",
    "update_observability",
    "update_observability_with_config",
]


# ---------------------------------------------------------------------------
# Snapshot tables
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


def _to_epoch_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp())
    if isinstance(value, (int, float)):
        return int(value)
    return None


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


def _emit_account_metrics(row: dict[str, object]) -> None:
    login = str(row.get("login", ""))
    server = str(row.get("server", ""))
    get_metrics().record_account_state(
        login=login,
        server=server,
        balance=float(row.get("balance") or 0.0),  # type: ignore[arg-type]
        equity=float(row.get("equity") or 0.0),  # type: ignore[arg-type]
        margin=float(row.get("margin") or 0.0),  # type: ignore[arg-type]
        margin_free=float(row.get("margin_free") or 0.0),  # type: ignore[arg-type]
        margin_level=float(row.get("margin_level") or 0.0),  # type: ignore[arg-type]
    )


def _emit_position_metrics(
    rows: list[dict[str, object]],
    login: int | None,
) -> None:
    m = get_metrics()
    login_str = str(login) if login is not None else ""
    # Aggregate profit and volume by symbol so hedging accounts (multiple open
    # positions sharing the same symbol) emit a single gauge value per symbol
    # instead of overwriting with each row's value.
    totals: dict[str, tuple[float, float]] = {}
    for r in rows:
        symbol = str(r.get("symbol", ""))
        profit = float(r.get("profit") or 0.0)  # type: ignore[arg-type]
        volume = float(r.get("volume") or 0.0)  # type: ignore[arg-type]
        if symbol in totals:
            prev_p, prev_v = totals[symbol]
            totals[symbol] = (prev_p + profit, prev_v + volume)
        else:
            totals[symbol] = (profit, volume)
    for symbol, (profit, volume) in totals.items():
        m.record_position_state(
            login=login_str,
            server="",
            symbol=symbol,
            profit=profit,
            volume=volume,
        )


def _persist_account(
    conn: sqlite3.Connection,
    df: pd.DataFrame | None,
    run_id: int,
) -> int | None:
    if df is None:
        return None
    if df.empty:
        logger.warning("account_info returned empty frame; skipping account snapshot")
        return None
    row = cast("dict[str, object]", df.iloc[0].to_dict())
    insert_account_snapshot(conn, run_id, row)
    _emit_account_metrics(row)
    login_val = row.get("login")
    return int(login_val) if login_val is not None else None  # type: ignore[arg-type]


def _persist_positions(
    conn: sqlite3.Connection,
    df: pd.DataFrame | None,
    run_id: int,
    login: int | None,
    symbols: Sequence[str] | None,
) -> None:
    if df is None:
        return
    if symbols is not None and not df.empty and "symbol" in df.columns:
        df = df[df["symbol"].isin(symbols)].reset_index(drop=True)
    raw = df.to_dict(orient="records") if not df.empty else []
    rows = cast("list[dict[str, object]]", raw)
    insert_position_snapshots(conn, run_id, login, rows)
    _emit_position_metrics(rows, login)


def _persist_orders(
    conn: sqlite3.Connection,
    df: pd.DataFrame | None,
    run_id: int,
    login: int | None,
    symbols: Sequence[str] | None,
) -> None:
    if df is None:
        return
    if symbols is not None and not df.empty and "symbol" in df.columns:
        df = df[df["symbol"].isin(symbols)].reset_index(drop=True)
    raw = df.to_dict(orient="records") if not df.empty else []
    rows = cast("list[dict[str, object]]", raw)
    insert_order_snapshots(conn, run_id, login, rows)


def _emit_terminal_metrics(row: dict[str, object]) -> None:
    get_metrics().record_terminal_state(
        connected=float(row.get("connected") or 0.0),  # type: ignore[arg-type]
        trade_allowed=float(row.get("trade_allowed") or 0.0),  # type: ignore[arg-type]
        trade_expert=float(row.get("trade_expert") or 0.0),  # type: ignore[arg-type]
    )


def _persist_terminal(
    conn: sqlite3.Connection,
    df: pd.DataFrame | None,
    run_id: int,
) -> None:
    if df is None:
        return
    if df.empty:
        logger.warning("terminal_info returned empty frame; skipping terminal snapshot")
        return
    row = cast("dict[str, object]", df.iloc[0].to_dict())
    insert_terminal_snapshot(conn, run_id, row)
    _emit_terminal_metrics(row)


@dataclass(frozen=True)
class ObservabilitySnapshot:
    """MT5-independent captured account/position/order/terminal state.

    Produced by :func:`capture_observability_snapshot` on the MT5 connection
    thread; consumed by :func:`persist_observability_snapshot`, which never
    touches MT5 or any connection-scoped MT5 state and may run on a different
    thread. A field is None when its ``include_*`` flag was False at capture
    time, distinct from an empty DataFrame returned by MT5 itself.
    """

    observed_at: int
    account: pd.DataFrame | None
    positions: pd.DataFrame | None
    orders: pd.DataFrame | None
    terminal: pd.DataFrame | None
    symbols: tuple[str, ...] | None


def capture_observability_snapshot(
    *,
    client: ObservabilityClient,
    symbols: Sequence[str] | None = None,
    include_account: bool = True,
    include_positions: bool = True,
    include_orders: bool = True,
    include_terminal: bool = True,
) -> ObservabilitySnapshot:
    """Capture current account/position/order/terminal state as plain data.

    Reads MT5 state via ``client`` and returns it as a serialization-safe,
    MT5-independent snapshot. Performs no SQLite access. Never places orders
    or modifies trading state. Pass the result to
    :func:`persist_observability_snapshot` -- from any thread, without MT5
    access -- to write it.

    Args:
        client: Connected MT5 client implementation.
        symbols: Optional symbol filter for positions and orders. When None,
            all positions and orders are captured.
        include_account: Capture account info.
        include_positions: Capture open positions.
        include_orders: Capture active orders.
        include_terminal: Capture terminal info.

    Returns:
        The captured snapshot.
    """
    return ObservabilitySnapshot(
        observed_at=int(datetime.now(UTC).timestamp()),
        account=client.account_info() if include_account else None,
        positions=client.positions() if include_positions else None,
        orders=client.orders() if include_orders else None,
        terminal=client.terminal_info() if include_terminal else None,
        symbols=tuple(symbols) if symbols is not None else None,
    )


def persist_observability_snapshot(
    snapshot: ObservabilitySnapshot,
    *,
    output: Path | str,
) -> None:
    """Persist a :func:`capture_observability_snapshot` result into SQLite.

    Opens and owns its own SQLite connection (WAL journal mode,
    ``synchronous=NORMAL``). Never accesses MT5 or any connection-scoped MT5
    state, so it is safe to call from a dedicated persistence-writer thread
    that holds no MT5 client.

    Args:
        snapshot: A snapshot from :func:`capture_observability_snapshot`.
        output: SQLite database path.
    """
    with closing(sqlite3.connect(Path(output))) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        create_snapshot_tables(conn)
        with get_metrics().record_snapshot_update():
            run_id = start_snapshot_run(conn, snapshot.observed_at)
            try:
                login = _persist_account(conn, snapshot.account, run_id)
                _persist_positions(
                    conn,
                    snapshot.positions,
                    run_id,
                    login,
                    snapshot.symbols,
                )
                _persist_orders(conn, snapshot.orders, run_id, login, snapshot.symbols)
                _persist_terminal(conn, snapshot.terminal, run_id)
                record_snapshot_run(conn, run_id, "ok")
            except Exception:
                record_snapshot_run(conn, run_id, "error")
                conn.commit()
                raise


def update_observability(
    *,
    client: ObservabilityClient,
    output: Path | str,
    symbols: Sequence[str] | None = None,
    include_account: bool = True,
    include_positions: bool = True,
    include_orders: bool = True,
    include_terminal: bool = True,
) -> None:
    """Snapshot current account/position/order/terminal state into SQLite.

    Reads the current MT5 state and appends timestamped snapshot rows. Never
    places orders or modifies trading state.

    Args:
        client: Connected MT5 client implementation.
        output: SQLite database path.
        symbols: Optional symbol filter for positions and orders. When None,
            all positions and orders are snapshotted.
        include_account: Snapshot account info into ``account_snapshots``.
        include_positions: Snapshot open positions into ``position_snapshots``.
        include_orders: Snapshot active orders into ``order_snapshots``.
        include_terminal: Snapshot terminal info into ``terminal_snapshots``.
    """
    snapshot = capture_observability_snapshot(
        client=client,
        symbols=symbols,
        include_account=include_account,
        include_positions=include_positions,
        include_orders=include_orders,
        include_terminal=include_terminal,
    )
    persist_observability_snapshot(snapshot, output=output)


def update_observability_with_config(
    *,
    output: Path | str,
    config: Mt5Config | None = None,
    symbols: Sequence[str] | None = None,
    include_account: bool = True,
    include_positions: bool = True,
    include_orders: bool = True,
    include_terminal: bool = True,
) -> None:
    """Snapshot current MT5 state, opening and closing the MT5 connection.

    Convenience wrapper around :func:`update_observability` for standalone use.

    Args:
        output: SQLite database path.
        config: MT5 connection configuration. Defaults to an empty config that
            attaches to a running terminal.
        symbols: Optional symbol filter for positions and orders.
        include_account: Snapshot account info.
        include_positions: Snapshot open positions.
        include_orders: Snapshot active orders.
        include_terminal: Snapshot terminal info.
    """
    with mt5_session(config) as client:
        update_observability(
            client=client,
            output=output,
            symbols=symbols,
            include_account=include_account,
            include_positions=include_positions,
            include_orders=include_orders,
            include_terminal=include_terminal,
        )
