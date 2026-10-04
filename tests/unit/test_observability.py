"""Tests for mt5cli.observability module."""

from __future__ import annotations

import inspect
import logging
import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest

import mt5cli.observability as observability_mod
from mt5cli.observability import (
    create_snapshot_tables,
    insert_account_snapshot,
    insert_order_snapshots,
    insert_position_snapshots,
    insert_terminal_snapshot,
    record_snapshot_run,
    start_snapshot_run,
    update_observability,
    update_observability_with_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from pytest_mock import MockerFixture


def test_update_observability_does_not_annotate_client_as_object_or_mt5dataclient() -> (
    None
):
    """update_observability uses its protocol rather than the raw client."""
    annotations = inspect.get_annotations(update_observability, eval_str=False)
    client_annotation = str(annotations["client"])
    assert client_annotation != "object"
    assert "Mt5DataClient" not in client_annotation


class TestUpdateObservability:
    """Tests for update_observability and update_observability_with_config."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Mock client returning minimal valid frames via canonical method names."""
        client = MagicMock()
        client.account_info.return_value = pd.DataFrame([
            {
                "login": 12345,
                "currency": "USD",
                "balance": 10000.0,
                "equity": 10000.0,
                "margin": 0.0,
                "margin_free": 10000.0,
                "margin_level": 0.0,
                "profit": 0.0,
                "leverage": 100,
            }
        ])
        client.positions.return_value = pd.DataFrame()
        client.orders.return_value = pd.DataFrame()
        client.terminal_info.return_value = pd.DataFrame([
            {
                "name": "MetaTrader 5",
                "connected": 1,
                "community_account": 0,
                "trade_allowed": 1,
                "trade_expert": 1,
                "path": "/mt5",
                "company": "Broker",
                "language": "en",
            }
        ])
        return client

    def test_update_observability_creates_snapshot_tables(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Snapshot tables are created in the output database."""
        output = tmp_path / "obs.db"
        update_observability(client=mock_client, output=output)
        with sqlite3.connect(output) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "snapshot_runs" in tables
        assert "account_snapshots" in tables
        assert "position_snapshots" in tables

    def test_update_observability_records_ok_on_success(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """snapshot_runs records 'ok' status on a successful run."""
        output = tmp_path / "obs.db"
        update_observability(client=mock_client, output=output)
        with sqlite3.connect(output) as conn:
            row = conn.execute("SELECT status FROM snapshot_runs").fetchone()
        assert row == ("ok",)

    def test_update_observability_capture_failure_never_touches_sqlite(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """An MT5 capture failure propagates without creating the output database.

        Capture (MT5-bound) now runs entirely before persistence opens any
        SQLite connection, so a client failure during capture never reaches
        ``snapshot_runs`` -- there is no run to record an error against.
        """
        mock_client.account_info.side_effect = RuntimeError("boom")
        output = tmp_path / "obs.db"
        with pytest.raises(RuntimeError, match="boom"):
            update_observability(client=mock_client, output=output)
        assert not output.exists()

    def test_update_observability_persist_failure_records_error(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """snapshot_runs records 'error' and re-raises when persistence fails."""
        mocker.patch.object(
            observability_mod,
            "insert_position_snapshots",
            side_effect=RuntimeError("boom"),
        )
        output = tmp_path / "obs.db"
        with pytest.raises(RuntimeError, match="boom"):
            update_observability(client=mock_client, output=output)
        with sqlite3.connect(output) as conn:
            row = conn.execute("SELECT status FROM snapshot_runs").fetchone()
        assert row == ("error",)

    @pytest.mark.parametrize(
        ("kwarg", "method"),
        [
            ("include_account", "account_info"),
            ("include_positions", "positions"),
            ("include_orders", "orders"),
            ("include_terminal", "terminal_info"),
        ],
    )
    def test_update_observability_skips_when_disabled(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        kwarg: str,
        method: str,
    ) -> None:
        """include_X=False does not call the corresponding client method."""
        update_observability(
            client=mock_client,
            output=tmp_path / "obs.db",
            **{kwarg: False},  # type: ignore[arg-type]
        )
        getattr(mock_client, method).assert_not_called()

    @pytest.mark.parametrize(
        ("method", "table", "row", "expected_count"),
        [
            (
                "positions",
                "position_snapshots",
                {
                    "ticket": 1,
                    "position_id": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "price_open": 1.1,
                    "price_current": 1.1,
                    "profit": 0.0,
                    "swap": 0.0,
                    "comment": "",
                    "magic": 0,
                },
                1,
            ),
            (
                "orders",
                "order_snapshots",
                {
                    "ticket": 10,
                    "symbol": "EURUSD",
                    "type": 2,
                    "volume_current": 0.1,
                    "price_open": 1.2,
                    "price_current": 1.1,
                    "state": 1,
                    "comment": "",
                    "magic": 0,
                    "time_setup": 1700000000,
                },
                1,
            ),
        ],
        ids=["positions", "orders"],
    )
    def test_update_observability_writes_snapshot_rows(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        method: str,
        table: str,
        row: dict[str, object],
        expected_count: int,
    ) -> None:
        """Non-empty snapshots are written to the corresponding snapshot table."""
        getattr(mock_client, method).return_value = pd.DataFrame([row])
        output = tmp_path / "obs.db"
        update_observability(client=mock_client, output=output)
        with sqlite3.connect(output) as conn:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608
            ).fetchone()[0]
        assert count == expected_count

    @pytest.mark.parametrize(
        ("method", "table", "rows"),
        [
            (
                "positions",
                "position_snapshots",
                [
                    {"ticket": 1, "symbol": "EURUSD", "volume": 0.1, "profit": 0.0},
                    {"ticket": 2, "symbol": "USDJPY", "volume": 0.2, "profit": 0.0},
                ],
            ),
            (
                "orders",
                "order_snapshots",
                [
                    {"ticket": 10, "symbol": "EURUSD", "volume_current": 0.1},
                    {"ticket": 11, "symbol": "USDJPY", "volume_current": 0.5},
                ],
            ),
        ],
        ids=["positions", "orders"],
    )
    def test_update_observability_symbol_filter(
        self,
        tmp_path: Path,
        method: str,
        table: str,
        rows: list[dict[str, object]],
    ) -> None:
        """Symbol filter fetches all rows in one call and filters client-side."""
        client = MagicMock()
        client.account_info.return_value = pd.DataFrame([{"login": 1}])
        client.positions.return_value = pd.DataFrame()
        client.orders.return_value = pd.DataFrame()
        client.terminal_info.return_value = pd.DataFrame()
        getattr(client, method).return_value = pd.DataFrame(rows)
        output = tmp_path / "obs.db"
        update_observability(client=client, output=output, symbols=["EURUSD", "GBPUSD"])
        assert getattr(client, method).call_count == 1
        with sqlite3.connect(output) as conn:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608
            ).fetchone()[0]
        assert count == 1

    def test_update_observability_symbol_filter_no_symbol_col(
        self,
        tmp_path: Path,
    ) -> None:
        """Symbol filter is skipped when positions df has no symbol column."""
        client = MagicMock()
        client.account_info.return_value = pd.DataFrame([{"login": 1}])
        # No symbol column in positions — all rows pass through unfiltered
        client.positions.return_value = pd.DataFrame([
            {"ticket": 1, "volume": 0.1},
        ])
        client.orders.return_value = pd.DataFrame()
        client.terminal_info.return_value = pd.DataFrame()
        output = tmp_path / "obs.db"
        update_observability(client=client, output=output, symbols=["EURUSD"])
        with sqlite3.connect(output) as conn:
            count = conn.execute("SELECT COUNT(*) FROM position_snapshots").fetchone()[
                0
            ]
        assert count == 1

    def test_update_observability_account_none_login(
        self,
        tmp_path: Path,
    ) -> None:
        """Account row with no login key returns None login for downstream helpers."""
        client = MagicMock()
        client.account_info.return_value = pd.DataFrame([{"balance": 10000.0}])
        client.positions.return_value = pd.DataFrame()
        client.orders.return_value = pd.DataFrame()
        client.terminal_info.return_value = pd.DataFrame()
        output = tmp_path / "obs.db"
        update_observability(client=client, output=output)
        with sqlite3.connect(output) as conn:
            row = conn.execute("SELECT login FROM account_snapshots").fetchone()
        assert row is not None
        assert row[0] is None

    @pytest.mark.parametrize(
        ("method", "table", "message"),
        [
            (
                "account_info",
                "account_snapshots",
                "account_info returned empty frame",
            ),
            (
                "terminal_info",
                "terminal_snapshots",
                "terminal_info returned empty frame",
            ),
        ],
        ids=["account", "terminal"],
    )
    def test_update_observability_empty_logs_warning(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        method: str,
        table: str,
        message: str,
    ) -> None:
        """Empty snapshot frames log a warning and write no rows."""
        getattr(mock_client, method).return_value = pd.DataFrame()
        with caplog.at_level(logging.WARNING, logger="mt5cli.observability"):
            update_observability(client=mock_client, output=tmp_path / "obs.db")
        assert message in caplog.text
        with sqlite3.connect(tmp_path / "obs.db") as conn:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608
            ).fetchone()[0]
        assert count == 0

    def test_update_observability_with_config_opens_and_closes_connection(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """update_observability_with_config manages the MT5 connection lifecycle."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.return_value = pd.DataFrame()
        mock_client.positions_get_as_df.return_value = pd.DataFrame()
        mock_client.orders_get_as_df.return_value = pd.DataFrame()
        mock_client.terminal_info_as_df.return_value = pd.DataFrame()
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=mock_client)
        update_observability_with_config(output=tmp_path / "obs.db")
        mock_client.initialize_and_login_mt5.assert_called_once()
        mock_client.shutdown.assert_called_once()

    def test_update_observability_with_config_passes_symbols(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """update_observability_with_config forwards symbols to update_observability."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.return_value = pd.DataFrame()
        mock_client.positions_get_as_df.return_value = pd.DataFrame()
        mock_client.orders_get_as_df.return_value = pd.DataFrame()
        mock_client.terminal_info_as_df.return_value = pd.DataFrame()
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=mock_client)
        spy = mocker.patch("mt5cli.observability.update_observability")
        update_observability_with_config(
            output=tmp_path / "obs.db",
            symbols=["EURUSD"],
            include_account=False,
        )
        spy.assert_called_once()
        call_kwargs = spy.call_args.kwargs
        assert call_kwargs["symbols"] == ["EURUSD"]
        assert call_kwargs["include_account"] is False

    def test_update_observability_invokes_snapshot_telemetry(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """update_observability calls record_snapshot_update on the global metrics."""
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_snapshot_update.return_value = mock_cm
        mocker.patch("mt5cli.observability.get_metrics", return_value=mock_metrics)
        update_observability(client=mock_client, output=tmp_path / "obs.db")
        mock_metrics.record_snapshot_update.assert_called_once()

    def test_update_observability_emits_account_metrics(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """_snapshot_account emits account gauges via get_metrics."""
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_snapshot_update.return_value = mock_cm
        mocker.patch("mt5cli.observability.get_metrics", return_value=mock_metrics)
        update_observability(client=mock_client, output=tmp_path / "obs.db")
        mock_metrics.record_account_state.assert_called_once()

    def test_update_observability_emits_terminal_metrics(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """_snapshot_terminal emits connected/trade gauges via get_metrics."""
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_snapshot_update.return_value = mock_cm
        mocker.patch("mt5cli.observability.get_metrics", return_value=mock_metrics)
        update_observability(client=mock_client, output=tmp_path / "obs.db")
        mock_metrics.record_terminal_state.assert_called_once_with(
            connected=1.0, trade_allowed=1.0, trade_expert=1.0
        )

    def test_update_observability_aggregates_same_symbol_positions(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Same-symbol positions are summed before emitting gauges (hedging)."""
        mock_client = MagicMock()
        mock_client.account_info.return_value = pd.DataFrame([
            {
                "login": 1,
                "server": "demo",
                "balance": 1000.0,
                "equity": 1000.0,
                "margin": 0.0,
                "margin_free": 1000.0,
                "margin_level": 0.0,
            }
        ])
        mock_client.positions.return_value = pd.DataFrame([
            {"ticket": 1, "symbol": "EURUSD", "profit": 10.0, "volume": 0.1},
            {"ticket": 2, "symbol": "EURUSD", "profit": -5.0, "volume": 0.2},
            {"ticket": 3, "symbol": "GBPUSD", "profit": 3.0, "volume": 0.05},
        ])
        mock_client.orders.return_value = pd.DataFrame()
        mock_client.terminal_info.return_value = pd.DataFrame()
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_snapshot_update.return_value = mock_cm
        mocker.patch("mt5cli.observability.get_metrics", return_value=mock_metrics)
        update_observability(client=mock_client, output=tmp_path / "obs.db")
        calls = mock_metrics.record_position_state.call_args_list
        # Two EURUSD positions should be collapsed to one call; GBPUSD is one call.
        assert len(calls) == 2
        by_symbol = {c.kwargs["symbol"]: c.kwargs for c in calls}
        assert abs(float(by_symbol["EURUSD"]["profit"]) - 5.0) < 1e-9
        assert abs(float(by_symbol["EURUSD"]["volume"]) - 0.3) < 1e-9
        assert abs(float(by_symbol["GBPUSD"]["profit"]) - 3.0) < 1e-9
        assert abs(float(by_symbol["GBPUSD"]["volume"]) - 0.05) < 1e-9


_TIMESTAMP_TIME_SETUP: pd.Timestamp = pd.Timestamp("2024-01-15 10:30:00", tz="UTC")


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
