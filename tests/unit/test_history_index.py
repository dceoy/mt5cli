"""Tests for SQLite history cursor indexes."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from mt5cli import rates

if TYPE_CHECKING:
    from pathlib import Path


def test_mark_timestamp_contract_creates_cursor_indexes(tmp_path: Path) -> None:
    """Canonical history finalization creates indexes used by cursor queries."""
    db_path = tmp_path / "history.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE rates("
            "symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
        )
        conn.execute("CREATE TABLE ticks(symbol TEXT, time TEXT, bid REAL)")
        conn.execute(
            "CREATE TABLE history_orders(symbol TEXT, time TEXT, ticket INTEGER)"
        )
        conn.execute(
            "CREATE TABLE history_deals("
            "symbol TEXT, time TEXT, type INTEGER, ticket INTEGER)"
        )
        conn.execute(
            "INSERT INTO rates(symbol, timeframe, time, close) "
            "VALUES ('EURUSD', 1, '2024-01-01T00:00:00', 1.0)"
        )

    rates._mark_rate_timestamp_contract(db_path)  # pyright: ignore[reportPrivateUsage]

    with sqlite3.connect(db_path) as conn:
        index_names = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert {
            "idx_rates_symbol_timeframe_history_cursor",
            "idx_ticks_symbol_history_cursor",
            "idx_history_orders_symbol_history_cursor",
            "idx_history_deals_symbol_history_cursor",
            "idx_history_deals_history_cursor",
        }.issubset(index_names)

        plan = conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT time FROM rates WHERE "
            "COALESCE(strftime('%Y-%m-%dT%H:%M:%f', time), "
            "strftime('%Y-%m-%dT%H:%M:%f', time, 'unixepoch')) IS NOT NULL "
            "AND symbol = ? AND timeframe = ? ORDER BY "
            "COALESCE(strftime('%Y-%m-%dT%H:%M:%f', time), "
            "strftime('%Y-%m-%dT%H:%M:%f', time, 'unixepoch')) DESC, "
            "ROWID DESC LIMIT 1",
            ("EURUSD", 1),
        ).fetchall()
        assert any(
            "idx_rates_symbol_timeframe_history_cursor" in str(row) for row in plan
        )
