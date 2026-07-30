"""Tests for mt5cli.observability module."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest

import mt5cli.observability as observability_mod
from mt5cli.observability import update_observability, update_observability_with_config

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


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

    def test_update_observability_records_error_on_failure(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """snapshot_runs records 'error' and re-raises when a snapshot fails."""
        mock_client.account_info.side_effect = RuntimeError("boom")
        output = tmp_path / "obs.db"
        with pytest.raises(RuntimeError, match="boom"):
            update_observability(client=mock_client, output=output)
        with sqlite3.connect(output) as conn:
            row = conn.execute("SELECT status FROM snapshot_runs").fetchone()
        assert row == ("error",)

    @pytest.mark.parametrize(
        ("with_grafana_schema", "expected_call_count"),
        [
            pytest.param(False, 0, id="grafana-schema-disabled"),
            pytest.param(True, 1, id="grafana-schema-enabled"),
        ],
    )
    def test_update_observability_grafana_schema_gate(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
        with_grafana_schema: bool,
        expected_call_count: int,
    ) -> None:
        """with_grafana_schema controls whether ensure_grafana_schema is called."""
        spy = mocker.spy(observability_mod, "ensure_grafana_schema")
        update_observability(
            client=mock_client,
            output=tmp_path / "obs.db",
            with_grafana_schema=with_grafana_schema,
        )
        assert spy.call_count == expected_call_count

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
