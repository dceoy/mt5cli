"""Tests for SQLite history cursor index acceleration."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

import mt5cli.history_index as history_index
from mt5cli import ThrottledHistoryUpdater
from mt5cli.contract import HistoryClient
from mt5cli.history import _sqlite_normalized_time_expression

if TYPE_CHECKING:
    from pathlib import Path


def _create_history_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
    )
    conn.execute("CREATE TABLE ticks(symbol TEXT, bid REAL)")
    conn.execute("CREATE TABLE history_orders(symbol TEXT, time TEXT, ticket INTEGER)")
    conn.execute(
        "CREATE TABLE history_deals(symbol TEXT, time TEXT, type INTEGER, ticket INTEGER)"
    )


def test_ensure_incremental_cursor_indexes_creates_matching_expression_indexes(
    tmp_path: Path,
) -> None:
    """Cursor indexes match normalized timestamp queries and skip invalid schemas."""
    db_path = tmp_path / "history.db"
    with sqlite3.connect(db_path) as conn:
        _create_history_tables(conn)
        conn.executemany(
            "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
            [
                ("EURUSD", 1, "2024-01-01T00:00:00", 1.0),
                ("EURUSD", 1, "2024-01-01T00:01:00", 1.1),
            ],
        )

    history_index._ensure_incremental_cursor_indexes(db_path)  # pyright: ignore[reportPrivateUsage]

    with sqlite3.connect(db_path) as conn:
        index_names = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert index_names == {
            "idx_rates_symbol_timeframe_history_cursor",
            "idx_history_orders_symbol_history_cursor",
            "idx_history_deals_symbol_history_cursor",
            "idx_history_deals_history_cursor",
        }

        time_expr = _sqlite_normalized_time_expression(  # pyright: ignore[reportPrivateUsage]
            "time"
        )
        plan = conn.execute(
            "EXPLAIN QUERY PLAN "
            f"SELECT time FROM rates WHERE {time_expr} IS NOT NULL "  # noqa: S608
            f"AND symbol = ? AND timeframe = ? ORDER BY {time_expr} DESC, "
            "ROWID DESC LIMIT 1",
            ("EURUSD", 1),
        ).fetchall()
        assert any(
            "idx_rates_symbol_timeframe_history_cursor" in str(row)
            for row in plan
        )


def test_ensure_incremental_cursor_indexes_ignores_missing_database(
    tmp_path: Path,
) -> None:
    """A database that has not been created yet needs no index work."""
    history_index._ensure_incremental_cursor_indexes(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "missing.db"
    )


def test_ensure_incremental_cursor_indexes_is_best_effort_on_sqlite_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Index creation failure does not turn a successful history sync into failure."""
    db_path = tmp_path / "history.db"
    db_path.touch()
    monkeypatch.setattr(
        history_index.sqlite3,
        "connect",
        MagicMock(side_effect=sqlite3.OperationalError("locked")),
    )

    with caplog.at_level("WARNING"):
        history_index._ensure_incremental_cursor_indexes(  # pyright: ignore[reportPrivateUsage]
            db_path
        )

    assert "Could not create SQLite incremental history cursor indexes" in caplog.text


def test_throttled_updater_indexes_only_successful_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful updates create indexes while throttled calls do no extra DB work."""
    backend = MagicMock()
    ensure_indexes = MagicMock()
    monkeypatch.setattr(
        history_index,
        "_ensure_incremental_cursor_indexes",
        ensure_indexes,
    )
    updater = ThrottledHistoryUpdater(
        output=tmp_path / "history.db",
        datasets=set(),
        interval_seconds=60,
        update_backend=backend,
    )
    client = cast("HistoryClient", MagicMock())
    date_to = datetime(2024, 1, 1)

    assert updater.update(client, ["EURUSD"], date_to=date_to)
    ensure_indexes.assert_called_once_with(updater.output)

    ensure_indexes.reset_mock()
    assert not updater.update(client, ["EURUSD"], date_to=date_to)
    ensure_indexes.assert_not_called()
