"""Tests for mt5cli.grafana module."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

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
    publish_grafana_copy,
    record_snapshot_run,
    start_snapshot_run,
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


def _make_history_deals_symbol_pnl_minimal(conn: sqlite3.Connection) -> None:
    """history_deals with entry but without volume and price columns."""
    conn.execute(
        "CREATE TABLE history_deals"
        " (time TEXT, symbol TEXT, profit REAL, type INTEGER, entry INTEGER)"
    )


def _make_history_orders_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE history_orders"
        " (time_setup TEXT, symbol TEXT, ticket INTEGER, type INTEGER)"
    )


_TIMESTAMP_TIME_SETUP: pd.Timestamp = pd.Timestamp("2024-01-15 10:30:00", tz="UTC")


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

    def test_stale_view_dropped_when_source_table_disappears(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """create_grafana_views drops a previously created view whose source is gone."""
        _make_ticks_table(conn)
        create_grafana_views(conn)
        assert "grafana_ticks" in _get_names(conn, "view")
        conn.execute("DROP TABLE ticks")
        create_grafana_views(conn)
        assert "grafana_ticks" not in _get_names(conn, "view")

    def test_grafana_rates_skipped_when_table_absent(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """grafana_rates is skipped when rates table is missing."""
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert "grafana_rates" not in _get_names(conn, "view")

    @pytest.mark.parametrize(
        ("ddl", "view_name"),
        [
            ("CREATE TABLE rates (open REAL)", "grafana_rates"),
            ("CREATE TABLE ticks (bid REAL)", "grafana_ticks"),
            ("CREATE TABLE history_deals (symbol TEXT)", "grafana_history_deals"),
            ("CREATE TABLE history_orders (symbol TEXT)", "grafana_history_orders"),
            ("CREATE TABLE history_deals (symbol TEXT)", "grafana_trade_deals"),
            ("CREATE TABLE history_deals (symbol TEXT)", "grafana_cash_events"),
            (
                "CREATE TABLE history_deals (time TEXT, type INTEGER)",
                "grafana_realized_pnl",
            ),
            (
                (
                    "CREATE TABLE history_deals"
                    " (time TEXT, symbol TEXT, profit REAL, type INTEGER)"
                ),
                "grafana_realized_pnl",
            ),
            (
                "CREATE TABLE history_deals (time TEXT, type INTEGER)",
                "grafana_symbol_pnl",
            ),
            ("CREATE TABLE history_deals (time TEXT)", "grafana_trade_stats"),
        ],
        ids=[
            "rates-cols",
            "ticks-cols",
            "history_deals-time",
            "history_orders-time_setup",
            "trade_deals-cols",
            "cash_events-cols",
            "realized_pnl-cols",
            "realized_pnl-entry",
            "symbol_pnl-cols",
            "trade_stats-cols",
        ],
    )
    def test_grafana_view_skipped_when_required_cols_missing(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
        ddl: str,
        view_name: str,
    ) -> None:
        """A grafana view is skipped (with warning) when source columns are missing."""
        conn.execute(ddl)
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            create_grafana_views(conn)
        assert view_name not in _get_names(conn, "view")
        assert f"Skipping {view_name}" in caplog.text

    @pytest.mark.parametrize(
        ("setup_deals", "optional_cols"),
        [
            pytest.param(
                _make_history_deals_symbol_pnl_minimal, set[str](), id="minimal"
            ),
            pytest.param(
                _make_history_deals_full,
                {"volume", "price"},
                id="full",
            ),
        ],
    )
    def test_grafana_symbol_pnl_schema(
        self,
        conn: sqlite3.Connection,
        setup_deals: Callable[[sqlite3.Connection], None],
        optional_cols: set[str],
    ) -> None:
        """grafana_symbol_pnl is created and includes optional columns when present."""
        setup_deals(conn)
        create_grafana_views(conn)
        assert "grafana_symbol_pnl" in _get_names(conn, "view")
        cols = {row[1] for row in conn.execute("PRAGMA table_info(grafana_symbol_pnl)")}
        assert optional_cols.issubset(cols)

    @pytest.mark.parametrize(
        "setup_deals",
        [_make_history_deals_minimal, _make_history_deals_full],
        ids=["minimal", "full"],
    )
    def test_grafana_trade_stats_static_summary(
        self,
        conn: sqlite3.Connection,
        setup_deals: Callable[[sqlite3.Connection], None],
    ) -> None:
        """grafana_trade_stats is a static summary view with no time column."""
        setup_deals(conn)
        create_grafana_views(conn)
        assert "grafana_trade_stats" in _get_names(conn, "view")
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(grafana_trade_stats)")
        }
        assert "time" not in cols
        assert "symbol" in cols

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

    def test_build_snapshot_view_with_only_run_id_col(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """_build_snapshot_view exposes time and run_id when table has only run_id."""
        create_snapshot_tables(conn)
        conn.execute("CREATE TABLE only_run (run_id INTEGER NOT NULL)")
        run_id = start_snapshot_run(conn, 1000)
        record_snapshot_run(conn, run_id, "ok")
        conn.execute("INSERT INTO only_run (run_id) VALUES (?)", (run_id,))
        _build_snapshot_view(conn, "test_view", "only_run")
        assert "test_view" in _get_names(conn, "view")
        cols = {row[1] for row in conn.execute("PRAGMA table_info(test_view)")}
        assert "time" in cols
        assert "run_id" in cols

    @pytest.mark.parametrize(
        ("use_snapshot_tables", "table_ddl", "table_name", "log_fragment"),
        [
            pytest.param(
                False,
                "CREATE TABLE only_run (run_id INTEGER NOT NULL)",
                "only_run",
                "snapshot_runs missing required columns",
                id="snapshot-runs-wrong-columns",
            ),
            pytest.param(
                True,
                "CREATE TABLE no_run_id (symbol TEXT)",
                "no_run_id",
                "missing run_id column",
                id="table-missing-run-id",
            ),
        ],
    )
    def test_build_snapshot_view_skips_negative_cases(
        self,
        conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
        use_snapshot_tables: bool,
        table_ddl: str,
        table_name: str,
        log_fragment: str,
    ) -> None:
        """_build_snapshot_view skips view creation for invalid table/run metadata."""
        if use_snapshot_tables:
            create_snapshot_tables(conn)
        else:
            conn.execute("CREATE TABLE snapshot_runs (foo TEXT)")
        conn.execute(table_ddl)
        with caplog.at_level(logging.WARNING, logger="mt5cli.grafana"):
            _build_snapshot_view(conn, "test_view", table_name)
        assert "test_view" not in _get_names(conn, "view")
        assert log_fragment in caplog.text

    @pytest.mark.parametrize(
        ("started_at", "status", "message", "keep_row"),
        [
            pytest.param(
                1000,
                "error",
                "terminal offline",
                False,
                id="excludes-failed-run-rows",
            ),
            pytest.param(2000, "ok", None, True, id="includes-ok-run-rows"),
        ],
    )
    def test_snapshot_view_filters_rows_by_run_status(
        self,
        conn: sqlite3.Connection,
        started_at: int,
        status: str,
        message: str | None,
        keep_row: bool,
    ) -> None:
        """Snapshot views expose rows only from successful runs."""
        create_snapshot_tables(conn)
        run_id = start_snapshot_run(conn, started_at)
        conn.execute(
            "INSERT INTO account_snapshots"
            " (run_id, login, balance, equity, margin, margin_free, profit)"
            " VALUES (?, 12345, 10000.0, 9800.0, 200.0, 9600.0, -200.0)",
            (run_id,),
        )
        record_snapshot_run(conn, run_id, status, message)
        create_grafana_views(conn)
        rows = conn.execute(
            "SELECT time, run_id, login FROM grafana_account_snapshots"
        ).fetchall()
        if keep_row:
            assert rows == [(started_at, run_id, 12345)]
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(grafana_account_snapshots)")
            }
            assert "run_id" in cols
        else:
            assert rows == []

    @pytest.mark.parametrize(
        ("observed_at", "runs", "expected_rows"),
        [
            pytest.param(
                3000,
                [("error", 99), ("ok", 12345)],
                [(12345,)],
                id="same-second-error-and-ok-exposes-only-ok-row",
            ),
            pytest.param(
                4000,
                [("ok", 1), ("ok", 2)],
                [(1,), (2,)],
                id="same-second-two-ok-runs-no-duplication",
            ),
        ],
    )
    def test_snapshot_view_same_second_runs(
        self,
        conn: sqlite3.Connection,
        observed_at: int,
        runs: list[tuple[str, int]],
        expected_rows: list[tuple[int]],
    ) -> None:
        """Snapshot views join same-second rows by run_id."""
        create_snapshot_tables(conn)
        for status, login in runs:
            run_id = start_snapshot_run(conn, observed_at)
            conn.execute(
                "INSERT INTO account_snapshots (run_id, login) VALUES (?, ?)",
                (run_id, login),
            )
            record_snapshot_run(conn, run_id, status)
        create_grafana_views(conn)
        rows = conn.execute("SELECT login FROM grafana_account_snapshots").fetchall()
        assert rows == expected_rows


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

    @pytest.mark.parametrize(
        ("ddl", "index_name"),
        [
            ("CREATE TABLE rates (open REAL)", "idx_rates_time_symbol_timeframe"),
            ("CREATE TABLE ticks (bid REAL)", "idx_ticks_time_symbol"),
            (
                "CREATE TABLE history_deals (ticket INTEGER)",
                "idx_history_deals_time_symbol",
            ),
            (
                "CREATE TABLE history_orders (ticket INTEGER)",
                "idx_history_orders_time_setup_symbol",
            ),
        ],
        ids=["rates", "ticks", "deals", "orders"],
    )
    def test_index_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
        ddl: str,
        index_name: str,
    ) -> None:
        """An index is skipped when the source table lacks required columns."""
        conn.execute(ddl)
        create_grafana_indexes(conn)
        assert index_name not in _get_names(conn, "index")

    def test_snapshot_indexes_skipped_when_cols_missing(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """Snapshot table indexes are skipped when required columns are absent."""
        conn.execute("CREATE TABLE account_snapshots (foo TEXT)")
        conn.execute("CREATE TABLE position_snapshots (foo TEXT)")
        conn.execute("CREATE TABLE order_snapshots (foo TEXT)")
        conn.execute("CREATE TABLE snapshot_runs (foo TEXT)")
        create_grafana_indexes(conn)
        indexes = _get_names(conn, "index")
        assert "idx_account_snapshots_time_login" not in indexes
        assert "idx_position_snapshots_time_symbol" not in indexes
        assert "idx_order_snapshots_time_symbol" not in indexes
        assert "idx_snapshot_runs_time_status" not in indexes

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

    @pytest.mark.parametrize(
        ("insert_func", "row", "select_sql", "expected"),
        [
            (
                insert_account_snapshot,
                {
                    "login": 12345,
                    "currency": "USD",
                    "balance": 10000.0,
                    "equity": 9800.0,
                    "margin": 200.0,
                    "margin_free": 9800.0,
                    "margin_level": 4900.0,
                    "profit": -200.0,
                    "leverage": 100,
                },
                "SELECT login, currency, balance FROM account_snapshots",
                (12345, "USD", 10000.0),
            ),
            (
                insert_terminal_snapshot,
                {
                    "name": "MetaTrader 5",
                    "connected": 1,
                    "community_account": 0,
                    "trade_allowed": 1,
                    "trade_expert": 1,
                    "path": "/mt5",
                    "company": "Broker",
                    "language": "en",
                },
                "SELECT name, connected FROM terminal_snapshots",
                ("MetaTrader 5", 1),
            ),
        ],
        ids=["account", "terminal"],
    )
    def test_insert_single_snapshot(
        self,
        conn: sqlite3.Connection,
        insert_func: Callable[[sqlite3.Connection, int, dict[str, object]], None],
        row: dict[str, object],
        select_sql: str,
        expected: tuple[object, ...],
    ) -> None:
        """insert_account_snapshot and insert_terminal_snapshot append one row."""
        run_id = start_snapshot_run(conn, 1700000000)
        insert_func(conn, run_id, row)
        result = conn.execute(select_sql).fetchone()
        assert result == expected

    def test_insert_account_snapshot_partial_row(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """insert_account_snapshot works when some fields are missing (uses None)."""
        run_id = start_snapshot_run(conn, 1700000000)
        insert_account_snapshot(conn, run_id, {"login": 1})
        result = conn.execute(
            "SELECT login, currency FROM account_snapshots"
        ).fetchone()
        assert result == (1, None)

    @pytest.mark.parametrize(
        ("insert_func", "table", "rows", "expected_count"),
        [
            (
                insert_position_snapshots,
                "position_snapshots",
                [
                    {"ticket": 1, "symbol": "EURUSD", "volume": 0.1, "profit": 10.0},
                    {"ticket": 2, "symbol": "GBPUSD", "volume": 0.2, "profit": -5.0},
                ],
                2,
            ),
            (
                insert_position_snapshots,
                "position_snapshots",
                [],
                0,
            ),
            (
                insert_order_snapshots,
                "order_snapshots",
                [
                    {
                        "ticket": 10,
                        "symbol": "EURUSD",
                        "type": 2,
                        "volume_current": 0.1,
                    },
                ],
                1,
            ),
            (
                insert_order_snapshots,
                "order_snapshots",
                [],
                0,
            ),
        ],
        ids=[
            "positions-with-rows",
            "positions-empty-noop",
            "orders-with-rows",
            "orders-empty-noop",
        ],
    )
    def test_insert_snapshot_rows(
        self,
        conn: sqlite3.Connection,
        insert_func: Callable[
            [sqlite3.Connection, int, int | None, list[dict[str, object]]],
            None,
        ],
        table: str,
        rows: list[dict[str, object]],
        expected_count: int,
    ) -> None:
        """insert_*_snapshots appends each row and is a no-op when empty."""
        run_id = start_snapshot_run(conn, 1700000000)
        insert_func(conn, run_id, 12345, rows)
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table}"  # noqa: S608
        ).fetchone()[0]
        assert count == expected_count

    @pytest.mark.parametrize(
        ("time_setup", "expected_stored"),
        [
            (_TIMESTAMP_TIME_SETUP, int(_TIMESTAMP_TIME_SETUP.timestamp())),
            (1705314600, 1705314600),
            ("not_a_time", None),
        ],
        ids=["timestamp", "int", "unknown-string"],
    )
    def test_insert_order_snapshots_normalizes_time_setup(
        self,
        conn: sqlite3.Connection,
        time_setup: object,
        expected_stored: int | None,
    ) -> None:
        """insert_order_snapshots stores epoch int, int as-is, or None for unknown."""
        run_id = start_snapshot_run(conn, 1700000000)
        rows: list[dict[str, object]] = [{"ticket": 10, "time_setup": time_setup}]
        insert_order_snapshots(conn, run_id, 12345, rows)
        stored = conn.execute("SELECT time_setup FROM order_snapshots").fetchone()[0]
        assert stored == expected_stored

    def test_start_snapshot_run_returns_incrementing_ids(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """start_snapshot_run returns a unique run_id for each call."""
        run1 = start_snapshot_run(conn, 1700000000)
        run2 = start_snapshot_run(conn, 1700000000)
        assert run1 != run2

    @pytest.mark.parametrize(
        ("status", "detail", "expected"),
        [
            pytest.param(
                "error",
                "RuntimeError: boom",
                ("error", "RuntimeError: boom"),
                id="with-detail",
            ),
            pytest.param("ok", None, ("ok", None), id="without-detail"),
        ],
    )
    def test_record_snapshot_run(
        self,
        conn: sqlite3.Connection,
        status: str,
        detail: str | None,
        expected: tuple[str, str | None],
    ) -> None:
        """record_snapshot_run stores status and optional detail text."""
        run_id = start_snapshot_run(conn, 1700000000)
        record_snapshot_run(conn, run_id, status, detail)
        row = conn.execute("SELECT status, detail FROM snapshot_runs").fetchone()
        assert row == expected


# ---------------------------------------------------------------------------
# TestPublishGrafanaCopy
# ---------------------------------------------------------------------------


def _make_source_db(path: Path) -> None:
    """Create a minimal source SQLite database with snapshot tables."""
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        create_snapshot_tables(conn)
        conn.execute(
            "INSERT INTO snapshot_runs (observed_at, status) VALUES (?, 'ok')",
            (1700000000,),
        )


class TestPublishGrafanaCopy:
    """Tests for publish_grafana_copy."""

    @pytest.mark.parametrize(
        ("target_rel", "stale_content", "required_tables"),
        [
            pytest.param(
                Path("out") / "grafana.db",
                None,
                frozenset({"snapshot_runs", "account_snapshots"}),
                id="fresh-target",
            ),
            pytest.param(
                Path("grafana.db"),
                b"stale",
                frozenset({"snapshot_runs"}),
                id="overwrite-stale-target",
            ),
        ],
    )
    def test_publish_creates_valid_sqlite_target(
        self,
        tmp_path: Path,
        target_rel: Path,
        stale_content: bytes | None,
        required_tables: frozenset[str],
    ) -> None:
        """publish_grafana_copy creates or replaces a valid SQLite target."""
        source = tmp_path / "src.db"
        target = tmp_path / target_rel
        _make_source_db(source)
        if stale_content is not None:
            target.write_bytes(stale_content)
        result = publish_grafana_copy(source, target)
        assert target.exists()
        assert isinstance(result, Path)
        assert result == target.resolve()
        with sqlite3.connect(target) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert required_tables <= tables

    def test_target_can_be_opened_readonly(self, tmp_path: Path) -> None:
        """Published target can be opened with uri=True in read-only mode."""
        source = tmp_path / "src.db"
        target = tmp_path / "grafana.db"
        _make_source_db(source)
        publish_grafana_copy(source, target)
        uri = f"file:{target}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            row = conn.execute("SELECT status FROM snapshot_runs").fetchone()
        assert row == ("ok",)

    def test_same_path_raises(self, tmp_path: Path) -> None:
        """publish_grafana_copy raises ValueError when source equals target."""
        db = tmp_path / "history.db"
        _make_source_db(db)
        with pytest.raises(ValueError, match="must differ from the source"):
            publish_grafana_copy(db, db)

    def test_source_not_found_raises(self, tmp_path: Path) -> None:
        """publish_grafana_copy raises FileNotFoundError when source is absent."""
        with pytest.raises(FileNotFoundError):
            publish_grafana_copy(tmp_path / "missing.db", tmp_path / "out.db")

    def test_preserve_old_target_on_backup_failure(self, tmp_path: Path) -> None:
        """Old target is preserved when the backup fails."""
        source = tmp_path / "src.db"
        target = tmp_path / "grafana.db"
        _make_source_db(source)
        original_content = b"original_data"
        target.write_bytes(original_content)
        with patch("sqlite3.connect") as mock_connect:
            mock_src = MagicMock()
            mock_src.__enter__ = MagicMock(return_value=mock_src)
            mock_src.__exit__ = MagicMock(return_value=False)
            mock_src.backup.side_effect = sqlite3.OperationalError("backup failed")
            mock_connect.return_value = mock_src
            with pytest.raises(sqlite3.OperationalError, match="backup failed"):
                publish_grafana_copy(source, target)
        assert target.read_bytes() == original_content

    def test_temp_file_cleaned_up_on_failure(self, tmp_path: Path) -> None:
        """Temporary file is removed when backup raises an exception."""
        source = tmp_path / "src.db"
        target = tmp_path / "grafana.db"
        _make_source_db(source)
        with patch("sqlite3.connect") as mock_connect:
            mock_src = MagicMock()
            mock_src.__enter__ = MagicMock(return_value=mock_src)
            mock_src.__exit__ = MagicMock(return_value=False)
            mock_src.backup.side_effect = sqlite3.OperationalError("fail")
            mock_connect.return_value = mock_src
            with pytest.raises(sqlite3.OperationalError):
                publish_grafana_copy(source, target)
        tmp_files = list(tmp_path.glob("grafana.db.*.tmp"))
        assert not tmp_files, "Temp file should be cleaned up on failure"

    def test_fresh_target_has_readable_permissions(self, tmp_path: Path) -> None:
        """Published copy is readable by the owner."""
        import stat as _stat  # noqa: PLC0415

        source = tmp_path / "src.db"
        target = tmp_path / "grafana.db"
        _make_source_db(source)
        publish_grafana_copy(source, target)
        mode = target.stat().st_mode & 0o777
        assert bool(mode & _stat.S_IRUSR), "owner must be able to read"

    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="Windows does not support Unix-style group/other permission bits",
    )
    def test_overwrite_preserves_existing_target_mode(self, tmp_path: Path) -> None:
        """Overwriting an existing target preserves that target's file mode."""
        source = tmp_path / "src.db"
        target = tmp_path / "grafana.db"
        _make_source_db(source)
        target.write_bytes(b"old")
        target.chmod(0o640)
        publish_grafana_copy(source, target)
        mode = target.stat().st_mode & 0o777
        assert mode == 0o640
