"""Observability snapshot orchestration for account, position, order, terminal state.

Snapshot persistence (SQLite schema and inserts) and Grafana-facing views
belong to :mod:`mt5cli.grafana`; this module owns *when* and *what* to
snapshot, using the canonical :class:`~mt5cli.contract.ObservabilityClient`
contract rather than raw pdmt5 method names.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .client import mt5_session
from .grafana import (
    create_snapshot_tables,
    ensure_grafana_schema,
    insert_account_snapshot,
    insert_order_snapshots,
    insert_position_snapshots,
    insert_terminal_snapshot,
    record_snapshot_run,
    start_snapshot_run,
)
from .telemetry import get_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pdmt5 import Mt5Config

    from .contract import ObservabilityClient

logger = logging.getLogger(__name__)

__all__ = [
    "update_observability",
    "update_observability_with_config",
]


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


def _snapshot_account(
    conn: sqlite3.Connection,
    client: ObservabilityClient,
    run_id: int,
) -> int | None:
    df = client.account_info()
    if df.empty:
        logger.warning("account_info returned empty frame; skipping account snapshot")
        return None
    row = cast("dict[str, object]", df.iloc[0].to_dict())
    insert_account_snapshot(conn, run_id, row)
    _emit_account_metrics(row)
    login_val = row.get("login")
    return int(login_val) if login_val is not None else None  # type: ignore[arg-type]


def _snapshot_positions(
    conn: sqlite3.Connection,
    client: ObservabilityClient,
    run_id: int,
    login: int | None,
    symbols: Sequence[str] | None,
) -> None:
    df = client.positions()
    if symbols is not None and not df.empty and "symbol" in df.columns:
        df = df[df["symbol"].isin(symbols)].reset_index(drop=True)
    raw = df.to_dict(orient="records") if not df.empty else []
    rows = cast("list[dict[str, object]]", raw)
    insert_position_snapshots(conn, run_id, login, rows)
    _emit_position_metrics(rows, login)


def _snapshot_orders(
    conn: sqlite3.Connection,
    client: ObservabilityClient,
    run_id: int,
    login: int | None,
    symbols: Sequence[str] | None,
) -> None:
    df = client.orders()
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


def _snapshot_terminal(
    conn: sqlite3.Connection,
    client: ObservabilityClient,
    run_id: int,
) -> None:
    df = client.terminal_info()
    if df.empty:
        logger.warning("terminal_info returned empty frame; skipping terminal snapshot")
        return
    row = cast("dict[str, object]", df.iloc[0].to_dict())
    insert_terminal_snapshot(conn, run_id, row)
    _emit_terminal_metrics(row)


def update_observability(
    *,
    client: ObservabilityClient,
    output: Path | str,
    symbols: Sequence[str] | None = None,
    include_account: bool = True,
    include_positions: bool = True,
    include_orders: bool = True,
    include_terminal: bool = True,
    with_grafana_schema: bool = False,
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
        with_grafana_schema: Ensure Grafana views and indexes exist. Defaults
            to ``False``; run ``grafana-schema`` once to set up the schema,
            then use ``snapshot`` repeatedly without this flag.
    """
    observed_at = int(datetime.now(UTC).timestamp())
    with closing(sqlite3.connect(Path(output))) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        if with_grafana_schema:
            ensure_grafana_schema(conn)
        else:
            create_snapshot_tables(conn)
        with get_metrics().record_snapshot_update():
            run_id = start_snapshot_run(conn, observed_at)
            login: int | None = None
            try:
                if include_account:
                    login = _snapshot_account(conn, client, run_id)
                if include_positions:
                    _snapshot_positions(conn, client, run_id, login, symbols)
                if include_orders:
                    _snapshot_orders(conn, client, run_id, login, symbols)
                if include_terminal:
                    _snapshot_terminal(conn, client, run_id)
                record_snapshot_run(conn, run_id, "ok")
            except Exception:
                record_snapshot_run(conn, run_id, "error")
                conn.commit()
                raise


def update_observability_with_config(
    *,
    output: Path | str,
    config: Mt5Config | None = None,
    symbols: Sequence[str] | None = None,
    include_account: bool = True,
    include_positions: bool = True,
    include_orders: bool = True,
    include_terminal: bool = True,
    with_grafana_schema: bool = False,
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
        with_grafana_schema: Ensure Grafana views and indexes exist.
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
            with_grafana_schema=with_grafana_schema,
        )
