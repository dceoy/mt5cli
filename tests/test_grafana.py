"""Tests for mt5cli.grafana module."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from mt5cli.grafana import (
    _build_snapshot_view,  # type: ignore[reportPrivateUsage]
    _create_view_safe,  # type: ignore[reportPrivateUsage]
    create_grafana_indexes,
    create_grafana_views,
    create_snapshot_tables,
    ensure_grafana_schema,
    insert_account_snapshot,
    insert_order_snapshots,
    insert_position_snapshots,
    insert_terminal_snapshot,
    record_snapshot_run,
)


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """Yield an in-memory SQLite connection for each test."""
    with sqlite3.connect(":memory:") as c:
        yield c


def _get_names(conn: sqlite3.Connection, type_: str) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type=?",
            (type_,),
        ).fetchall()
    }


def _make_rates_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE rates"
        " (time TEXT, symbol TEXT, timeframe INTEGER,"
        "  open REAL, high REAL, low REAL, close REAL)"
    )


def _make_ticks_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE ticks (time TEXT, symbol TEXT, bid REAL, ask REAL)")


def _make_history_deals_full(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE history_deals"
        " (time TEXT, symbol TEXT, profit REAL, type INTEGER,"
        "  entry INTEGER, volume REAL, price REAL, ticket INTEGER, position_id INTEGER)"
    )


def _make_history_deals_minimal(conn: sqlite3.Connection) -> None:
    """history_deals with only time, type, symbol, profit — no entry/volume/price."""
    conn.execute(
        "CREATE TABLE history_deals (time TEXT, symbol TEXT, profit REAL, type INTEGER)"
    )


def _make_history_orders_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE history_orders"
        " (time_setup TEXT, symbol TEXT, ticket INTEGER, type INTEGER)"
    )


# ---------------------------------------------------------------------------
# TestSnapshotTables
# ---------------------------------------------------------------------------


class TestSnapshotTables:
    """Tests for create_snapshot_tables."""

    def test_creates_all_five_tables(self, conn: sqlite3.Connection) -> None:
        """All five snapshot tables are created."""
        create_snapshot_tables(conn)
        tables = _get_names(conn, "table")
        assert "snapshot_runs" in tables
        assert "account_snapshots" in tables
        assert "position_snapshots" in tables
        assert "order_snapshots" in tables
        assert "terminal_snapshots" in tables

    def test_is_idempotent(self, conn: sqlite3.Connection) -> None:
        """Calling create_snapshot_tables twice does not raise."""
        create_snapshot_tables(conn)
        create_snapshot_tables(conn)
        tables = _get_names(conn, "table")
        assert "snapshot_runs" in tables


# ---------------------------------------------------------------------------
# TestCreateViewSafe
# ---------------------------------------------------------------------------


class TestCreateViewSafe:
    """Tests for _create_view_safe."""

    def test_creates_view_successfully(self, conn: sqlite3.Connection) -> None:
        """A valid select SQL creates the named view."""
        _create_view_safe(conn, "test_view", "SELECT 1 AS val")
        views = _get_names(conn, "view")
        assert "test_view" in views

    def test_replaces_existing_view(self, conn: sqlite3.Connection) -> None:
        """Calling again with a new SQL replaces the existing view."""
        _create_view_safe(conn, "test_view", "SELECT 1 AS val")
        _create_view_safe(conn, "test_view", "SELECT 2 AS val")
        result = conn.execute("SELECT val FROM test_view").fetchone()
        assert result == (2,)

    def test_logs_warning_on_sqlite_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """sqlite3.Error during CREATE VIEW logs a warning instead of raising."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [
            None,
            sqlite3.OperationalError("parse error"),
        ]
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            _create_view_safe(mock_conn, "bad_view", "SELECT 1")
        assert "Skipping view bad_view" in caplog.text
        assert "parse error" in caplog.text


# ---------------------------------------------------------------------------
# TestGrafanaViews
# ---------------------------------------------------------------------------


class TestGrafanaViews:
    """Tests for create_grafana_views and individual view builders."""

    def test_all_views_created_with_full_schema(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """All 13 Grafana views are created when all source tables are present."""
        _make_rates_table(conn)
        _make_ticks_table(conn)
        _make_history_deals_full(conn)
        _make_history_orders_table(conn)
        create_snapshot_tables(conn)
        create_grafana_views(conn)
        views = _get_names(conn, "view")
        expected = {
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
        }
        assert expected.issubset(views)

    def test_grafana_rates_skipped_when_table_absent(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_rates is skipped when rates table is missing."""
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_rates" not in _get_names(conn, "view")

    def test_grafana_rates_skipped_when_required_cols_missing(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_rates is skipped when rates table lacks required columns."""
        conn.execute("CREATE TABLE rates (open REAL)")
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_rates" not in _get_names(conn, "view")
        assert "Skipping grafana_rates" in caplog.text

    def test_grafana_ticks_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_ticks is skipped when ticks table lacks required columns."""
        conn.execute("CREATE TABLE ticks (bid REAL)")
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_ticks" not in _get_names(conn, "view")
        assert "Skipping grafana_ticks" in caplog.text

    def test_grafana_history_deals_skipped_when_time_missing(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_history_deals is skipped when history_deals.time is missing."""
        conn.execute("CREATE TABLE history_deals (symbol TEXT)")
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_history_deals" not in _get_names(conn, "view")
        assert "Skipping grafana_history_deals" in caplog.text

    def test_grafana_history_orders_skipped_when_time_setup_missing(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_history_orders is skipped when time_setup is absent."""
        conn.execute("CREATE TABLE history_orders (symbol TEXT)")
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_history_orders" not in _get_names(conn, "view")
        assert "Skipping grafana_history_orders" in caplog.text

    def test_grafana_trade_deals_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_trade_deals is skipped when history_deals missing time/type."""
        conn.execute("CREATE TABLE history_deals (symbol TEXT)")
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_trade_deals" not in _get_names(conn, "view")

    def test_grafana_cash_events_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_cash_events is skipped when history_deals missing time/type."""
        conn.execute("CREATE TABLE history_deals (symbol TEXT)")
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_cash_events" not in _get_names(conn, "view")

    def test_grafana_realized_pnl_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_realized_pnl is skipped when history_deals missing required cols."""
        conn.execute("CREATE TABLE history_deals (time TEXT, type INTEGER)")
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_realized_pnl" not in _get_names(conn, "view")
        assert "Skipping grafana_realized_pnl" in caplog.text

    def test_grafana_symbol_pnl_skipped_when_required_cols_missing(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_symbol_pnl is skipped when required columns are absent."""
        conn.execute("CREATE TABLE history_deals (time TEXT, type INTEGER)")
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_symbol_pnl" not in _get_names(conn, "view")
        assert "Skipping grafana_symbol_pnl" in caplog.text

    def test_grafana_symbol_pnl_without_volume_and_price(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """grafana_symbol_pnl is created with only required columns."""
        conn.execute(
            "CREATE TABLE history_deals"
            " (time TEXT, symbol TEXT, profit REAL, type INTEGER, entry INTEGER)"
        )
        create_grafana_views(conn)
        assert "grafana_symbol_pnl" in _get_names(conn, "view")

    def test_grafana_symbol_pnl_with_volume_and_price(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """grafana_symbol_pnl includes volume and price columns when present."""
        _make_history_deals_full(conn)
        create_grafana_views(conn)
        assert "grafana_symbol_pnl" in _get_names(conn, "view")
        # View columns include volume and price
        cols = {row[1] for row in conn.execute("PRAGMA table_info(grafana_symbol_pnl)")}
        assert "volume" in cols
        assert "price" in cols

    def test_grafana_trade_stats_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_trade_stats is skipped when history_deals missing required cols."""
        conn.execute("CREATE TABLE history_deals (time TEXT)")
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_trade_stats" not in _get_names(conn, "view")
        assert "Skipping grafana_trade_stats" in caplog.text

    def test_grafana_trade_stats_without_entry_col(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """grafana_trade_stats is created without entry filter when entry is absent."""
        _make_history_deals_minimal(conn)
        create_grafana_views(conn)
        assert "grafana_trade_stats" in _get_names(conn, "view")

    def test_grafana_trade_stats_with_entry_col(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """grafana_trade_stats includes entry filter when entry column is present."""
        _make_history_deals_full(conn)
        create_grafana_views(conn)
        assert "grafana_trade_stats" in _get_names(conn, "view")

    def test_snapshot_views_skipped_when_snapshot_tables_absent(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Snapshot views are skipped when snapshot tables are not created."""
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        views = _get_names(conn, "view")
        assert "grafana_account_snapshots" not in views
        assert "grafana_position_snapshots" not in views
        assert "grafana_order_snapshots" not in views
        assert "grafana_terminal_snapshots" not in views

    def test_build_snapshot_view_with_only_observed_at_col(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """_build_snapshot_view creates a valid view when table has only observed_at."""
        conn.execute("CREATE TABLE only_time (observed_at INTEGER NOT NULL)")
        _build_snapshot_view(conn, "test_view", "only_time")
        views = _get_names(conn, "view")
        assert "test_view" in views


# ---------------------------------------------------------------------------
# TestGrafanaIndexes
# ---------------------------------------------------------------------------


class TestGrafanaIndexes:
    """Tests for create_grafana_indexes."""

    def test_all_indexes_created_with_full_schema(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """All 9 indexes are created when all source tables are present."""
        _make_rates_table(conn)
        _make_ticks_table(conn)
        _make_history_deals_full(conn)
        _make_history_orders_table(conn)
        create_snapshot_tables(conn)
        create_grafana_indexes(conn)
        indexes = _get_names(conn, "index")
        assert "idx_rates_time_symbol_timeframe" in indexes
        assert "idx_ticks_time_symbol" in indexes
        assert "idx_history_deals_time_symbol" in indexes
        assert "idx_history_deals_symbol_time" in indexes
        assert "idx_history_orders_time_setup_symbol" in indexes
        assert "idx_account_snapshots_time_login" in indexes
        assert "idx_position_snapshots_time_symbol" in indexes
        assert "idx_order_snapshots_time_symbol" in indexes
        assert "idx_snapshot_runs_time_status" in indexes

    def test_no_indexes_created_when_tables_absent(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """No indexes are created when tables are absent."""
        create_grafana_indexes(conn)
        indexes = _get_names(conn, "index")
        assert not any(name.startswith("idx_") for name in indexes)

    def test_indexes_for_snapshot_tables_skipped_when_absent(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """Snapshot table indexes are skipped when snapshot tables don't exist."""
        _make_history_deals_full(conn)
        create_grafana_indexes(conn)
        indexes = _get_names(conn, "index")
        assert "idx_account_snapshots_time_login" not in indexes
        assert "idx_position_snapshots_time_symbol" not in indexes
        assert "idx_order_snapshots_time_symbol" not in indexes
        assert "idx_snapshot_runs_time_status" not in indexes

    def test_rates_index_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """Rates index is skipped when required columns are absent."""
        conn.execute("CREATE TABLE rates (open REAL)")
        create_grafana_indexes(conn)
        indexes = _get_names(conn, "index")
        assert "idx_rates_time_symbol_timeframe" not in indexes

    def test_ticks_index_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """Ticks index is skipped when required columns are absent."""
        conn.execute("CREATE TABLE ticks (bid REAL)")
        create_grafana_indexes(conn)
        indexes = _get_names(conn, "index")
        assert "idx_ticks_time_symbol" not in indexes

    def test_deals_indexes_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """history_deals indexes are skipped when required columns are absent."""
        conn.execute("CREATE TABLE history_deals (ticket INTEGER)")
        create_grafana_indexes(conn)
        indexes = _get_names(conn, "index")
        assert "idx_history_deals_time_symbol" not in indexes

    def test_orders_index_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """history_orders index is skipped when required columns are absent."""
        conn.execute("CREATE TABLE history_orders (ticket INTEGER)")
        create_grafana_indexes(conn)
        indexes = _get_names(conn, "index")
        assert "idx_history_orders_time_setup_symbol" not in indexes

    def test_indexes_are_idempotent(self, conn: sqlite3.Connection) -> None:
        """Creating indexes twice does not raise (IF NOT EXISTS)."""
        _make_rates_table(conn)
        create_grafana_indexes(conn)
        create_grafana_indexes(conn)
        indexes = _get_names(conn, "index")
        assert "idx_rates_time_symbol_timeframe" in indexes


# ---------------------------------------------------------------------------
# TestEnsureGrafanaSchema
# ---------------------------------------------------------------------------


class TestEnsureGrafanaSchema:
    """Tests for ensure_grafana_schema."""

    def test_creates_all_tables_views_and_indexes(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """ensure_grafana_schema creates snapshot tables, views, and indexes."""
        _make_rates_table(conn)
        _make_history_deals_full(conn)
        ensure_grafana_schema(conn)
        tables = _get_names(conn, "table")
        assert "snapshot_runs" in tables
        assert "account_snapshots" in tables
        views = _get_names(conn, "view")
        assert "grafana_rates" in views
        assert "grafana_account_snapshots" in views
        indexes = _get_names(conn, "index")
        assert "idx_rates_time_symbol_timeframe" in indexes

    def test_is_idempotent(self, conn: sqlite3.Connection) -> None:
        """Calling ensure_grafana_schema twice does not raise."""
        ensure_grafana_schema(conn)
        ensure_grafana_schema(conn)


# ---------------------------------------------------------------------------
# TestSnapshotInserts
# ---------------------------------------------------------------------------


class TestSnapshotInserts:
    """Tests for snapshot insert helpers."""

    @pytest.fixture(autouse=True)
    def setup_tables(self, conn: sqlite3.Connection) -> None:
        """Create snapshot tables before each insert test."""
        create_snapshot_tables(conn)

    def test_insert_account_snapshot(self, conn: sqlite3.Connection) -> None:
        """insert_account_snapshot appends a row with correct values."""
        row: dict[str, object] = {
            "login": 12345,
            "currency": "USD",
            "balance": 10000.0,
            "equity": 9800.0,
            "margin": 200.0,
            "margin_free": 9800.0,
            "margin_level": 4900.0,
            "profit": -200.0,
            "leverage": 100,
        }
        insert_account_snapshot(conn, 1700000000, row)
        result = conn.execute(
            "SELECT login, currency, balance FROM account_snapshots"
        ).fetchone()
        assert result == (12345, "USD", 10000.0)

    def test_insert_account_snapshot_partial_row(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """insert_account_snapshot works when some fields are missing (uses None)."""
        insert_account_snapshot(conn, 1700000000, {"login": 1})
        result = conn.execute(
            "SELECT login, currency FROM account_snapshots"
        ).fetchone()
        assert result == (1, None)

    def test_insert_position_snapshots_with_rows(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """insert_position_snapshots appends each position row."""
        rows: list[dict[str, object]] = [
            {"ticket": 1, "symbol": "EURUSD", "volume": 0.1, "profit": 10.0},
            {"ticket": 2, "symbol": "GBPUSD", "volume": 0.2, "profit": -5.0},
        ]
        insert_position_snapshots(conn, 1700000000, 12345, rows)
        count = conn.execute("SELECT COUNT(*) FROM position_snapshots").fetchone()[0]
        assert count == 2

    def test_insert_position_snapshots_noop_when_empty(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """insert_position_snapshots is a no-op when rows is empty."""
        insert_position_snapshots(conn, 1700000000, 12345, [])
        count = conn.execute("SELECT COUNT(*) FROM position_snapshots").fetchone()[0]
        assert count == 0

    def test_insert_order_snapshots_with_rows(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """insert_order_snapshots appends each order row."""
        rows: list[dict[str, object]] = [
            {"ticket": 10, "symbol": "EURUSD", "type": 2, "volume_current": 0.1},
        ]
        insert_order_snapshots(conn, 1700000000, 12345, rows)
        count = conn.execute("SELECT COUNT(*) FROM order_snapshots").fetchone()[0]
        assert count == 1

    def test_insert_order_snapshots_noop_when_empty(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """insert_order_snapshots is a no-op when rows is empty."""
        insert_order_snapshots(conn, 1700000000, 12345, [])
        count = conn.execute("SELECT COUNT(*) FROM order_snapshots").fetchone()[0]
        assert count == 0

    def test_insert_terminal_snapshot(self, conn: sqlite3.Connection) -> None:
        """insert_terminal_snapshot appends a terminal info row."""
        row: dict[str, object] = {
            "name": "MetaTrader 5",
            "connected": 1,
            "community_account": 0,
            "trade_allowed": 1,
            "trade_expert": 1,
            "path": "/mt5",
            "company": "Broker",
            "language": "en",
        }
        insert_terminal_snapshot(conn, 1700000000, row)
        result = conn.execute(
            "SELECT name, connected FROM terminal_snapshots"
        ).fetchone()
        assert result == ("MetaTrader 5", 1)

    def test_record_snapshot_run_with_detail(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """record_snapshot_run stores status and detail text."""
        record_snapshot_run(conn, 1700000000, "error", "RuntimeError: boom")
        row = conn.execute("SELECT status, detail FROM snapshot_runs").fetchone()
        assert row == ("error", "RuntimeError: boom")

    def test_record_snapshot_run_without_detail(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """record_snapshot_run stores None for detail when omitted."""
        record_snapshot_run(conn, 1700000000, "ok")
        row = conn.execute("SELECT status, detail FROM snapshot_runs").fetchone()
        assert row == ("ok", None)
