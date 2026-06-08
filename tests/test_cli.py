"""Tests for mt5cli.cli module."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture  # noqa: TC002
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from mt5cli.cli import (
    _execute_export,  # type: ignore[reportPrivateUsage]
    _ExportContext,  # type: ignore[reportPrivateUsage]
    _sdk_client,  # type: ignore[reportPrivateUsage]
    app,
    main,
)

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def normalize_cli_output(output: str) -> str:
    """Normalize CLI output for cross-platform assertions."""
    return " ".join(_ANSI_ESCAPE_RE.sub("", output).split())


# ---------------------------------------------------------------------------
# _execute_export
# ---------------------------------------------------------------------------


class TestExecuteExport:
    """Tests for _execute_export."""

    def test_shutdown_on_error(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test that shutdown is called even when fetch raises."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.side_effect = RuntimeError("boom")
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        ctx = MagicMock()
        ctx.obj = _ExportContext(
            output=tmp_path / "out.csv",
            output_format="csv",
            table="data",
            config=MagicMock(),
        )
        with pytest.raises(RuntimeError, match="boom"):
            _execute_export(ctx, _sdk_client(ctx).account_info)
        mock_client.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# CLI commands via CliRunner
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client(mocker: MockerFixture) -> MagicMock:
    """Create and patch a mock Mt5DataClient for CLI tests."""
    client = MagicMock()
    sample_df = pd.DataFrame({"col": [1]})
    client.copy_rates_from_as_df.return_value = sample_df
    client.copy_rates_from_pos_as_df.return_value = sample_df
    client.copy_rates_range_as_df.return_value = sample_df
    client.copy_ticks_from_as_df.return_value = sample_df
    client.copy_ticks_range_as_df.return_value = sample_df
    client.account_info_as_df.return_value = sample_df
    client.terminal_info_as_df.return_value = sample_df
    client.symbols_get_as_df.return_value = sample_df
    client.symbol_info_as_df.return_value = sample_df
    client.orders_get_as_df.return_value = sample_df
    client.positions_get_as_df.return_value = sample_df
    client.history_orders_get_as_df.return_value = sample_df
    client.history_deals_get_as_df.return_value = sample_df
    client.version_as_df.return_value = sample_df
    client.last_error_as_df.return_value = sample_df
    client.symbol_info_tick_as_df.return_value = sample_df
    client.market_book_get_as_df.return_value = sample_df
    client.order_check_as_df.return_value = sample_df
    client.order_send_as_df.return_value = sample_df
    client.version.return_value = (5, 0, 1)
    client.terminal_info.return_value = {"connected": True}
    client.account_info.return_value = {"login": 123}
    client.symbols_total.return_value = 42
    mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
    return client


class TestCommands:
    """Tests for all CLI subcommands via CliRunner."""

    def test_account_info(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test account-info command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            ["-o", str(output), "account-info"],
        )
        assert result.exit_code == 0, result.output
        mock_client.account_info_as_df.assert_called_once()
        assert output.exists()

    def test_terminal_info(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test terminal-info command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            ["-o", str(output), "terminal-info"],
        )
        assert result.exit_code == 0, result.output
        mock_client.terminal_info_as_df.assert_called_once()

    def test_symbols(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test symbols command."""
        output = tmp_path / "out.json"
        result = runner.invoke(
            app,
            ["-o", str(output), "symbols", "--group", "*USD*"],
        )
        assert result.exit_code == 0, result.output
        mock_client.symbols_get_as_df.assert_called_once_with(
            group="*USD*",
        )

    def test_symbol_info(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test symbol-info command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            ["-o", str(output), "symbol-info", "--symbol", "EURUSD"],
        )
        assert result.exit_code == 0, result.output
        mock_client.symbol_info_as_df.assert_called_once_with(
            symbol="EURUSD",
        )

    def test_rates_from(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test rates-from command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "rates-from",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "M1",
                "--date-from",
                "2024-01-01",
                "--count",
                "100",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.copy_rates_from_as_df.assert_called_once_with(
            symbol="EURUSD",
            timeframe=1,
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            count=100,
        )

    def test_rates_from_pos(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test rates-from-pos command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "rates-from-pos",
                "--symbol",
                "GBPUSD",
                "--timeframe",
                "H1",
                "--start-pos",
                "0",
                "--count",
                "50",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.copy_rates_from_pos_as_df.assert_called_once_with(
            symbol="GBPUSD",
            timeframe=16385,
            start_pos=0,
            count=50,
        )

    def test_latest_rates(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test latest-rates command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "latest-rates",
                "--symbol",
                "GBPUSD",
                "--timeframe",
                "H1",
                "--count",
                "50",
                "--start-pos",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.copy_rates_from_pos_as_df.assert_called_once_with(
            symbol="GBPUSD",
            timeframe=16385,
            start_pos=2,
            count=50,
        )

    def test_rates_range(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test rates-range command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "rates-range",
                "--symbol",
                "USDJPY",
                "--timeframe",
                "D1",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.copy_rates_range_as_df.assert_called_once_with(
            symbol="USDJPY",
            timeframe=16408,
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            date_to=datetime(2024, 2, 1, tzinfo=UTC),
        )

    def test_ticks_from(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test ticks-from command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "ticks-from",
                "--symbol",
                "EURUSD",
                "--date-from",
                "2024-01-01",
                "--count",
                "100",
                "--flags",
                "ALL",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.copy_ticks_from_as_df.assert_called_once_with(
            symbol="EURUSD",
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            count=100,
            flags=1,
        )

    def test_ticks_range(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test ticks-range command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "ticks-range",
                "--symbol",
                "EURUSD",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
                "--flags",
                "INFO",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.copy_ticks_range_as_df.assert_called_once_with(
            symbol="EURUSD",
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            date_to=datetime(2024, 2, 1, tzinfo=UTC),
            flags=2,
        )

    def test_ticks_recent(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test ticks-recent command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "ticks-recent",
                "--symbol",
                "EURUSD",
                "--seconds",
                "120",
                "--date-to",
                "2024-01-02",
                "--count",
                "500",
                "--flags",
                "ALL",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.copy_ticks_from_as_df.assert_called_once_with(
            symbol="EURUSD",
            date_from=datetime(2024, 1, 2, tzinfo=UTC) - timedelta(seconds=120),
            count=500,
            flags=1,
        )
        mock_client.copy_ticks_range_as_df.assert_not_called()

    def test_minimum_margins(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test minimum-margins command."""
        sym = MagicMock(volume_min=0.01)
        account = MagicMock(currency="USD")
        tick = MagicMock(ask=1.1010, bid=1.1000)
        mock_client.symbol_info.return_value = sym
        mock_client.account_info.return_value = account
        mock_client.symbol_info_tick.return_value = tick
        mock_client.order_calc_margin.side_effect = [12.5, 12.4]
        mock_client.mt5.ORDER_TYPE_BUY = 0
        mock_client.mt5.ORDER_TYPE_SELL = 1
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            ["-o", str(output), "minimum-margins", "--symbol", "EURUSD"],
        )
        assert result.exit_code == 0, result.output
        mock_client.symbol_info.assert_called_once_with("EURUSD")
        mock_client.order_calc_margin.assert_any_call(0, "EURUSD", 0.01, 1.1010)
        mock_client.order_calc_margin.assert_any_call(1, "EURUSD", 0.01, 1.1000)

    def test_orders(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test orders command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "orders",
                "--symbol",
                "EURUSD",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.orders_get_as_df.assert_called_once()

    def test_positions(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test positions command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            ["-o", str(output), "positions"],
        )
        assert result.exit_code == 0, result.output
        mock_client.positions_get_as_df.assert_called_once()

    def test_history_orders(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test history-orders command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "history-orders",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.history_orders_get_as_df.assert_called_once()

    def test_history_deals(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test history-deals command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "history-deals",
                "--ticket",
                "12345",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.history_deals_get_as_df.assert_called_once()

    def test_recent_history_deals(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test recent-history-deals command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "recent-history-deals",
                "--hours",
                "6",
                "--date-to",
                "2024-01-02",
                "--symbol",
                "EURUSD",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.history_deals_get_as_df.assert_called_once_with(
            date_from=datetime(2024, 1, 1, 18, tzinfo=UTC),
            date_to=datetime(2024, 1, 2, tzinfo=UTC),
            group=None,
            symbol="EURUSD",
            ticket=None,
            position=None,
        )

    def test_mt5_summary(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test mt5-summary command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(app, ["-o", str(output), "mt5-summary"])
        assert result.exit_code == 0, result.output
        mock_client.version.assert_called_once()
        mock_client.terminal_info.assert_called_once()
        mock_client.account_info.assert_called_once()
        mock_client.symbols_total.assert_called_once()

    def test_version(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test version command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(app, ["-o", str(output), "version"])
        assert result.exit_code == 0, result.output
        mock_client.version_as_df.assert_called_once()

    def test_last_error(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test last-error command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(app, ["-o", str(output), "last-error"])
        assert result.exit_code == 0, result.output
        mock_client.last_error_as_df.assert_called_once()

    def test_symbol_info_tick(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test symbol-info-tick command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            ["-o", str(output), "symbol-info-tick", "--symbol", "EURUSD"],
        )
        assert result.exit_code == 0, result.output
        mock_client.symbol_info_tick_as_df.assert_called_once_with(
            symbol="EURUSD",
        )

    def test_market_book(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test market-book command."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            ["-o", str(output), "market-book", "--symbol", "EURUSD"],
        )
        assert result.exit_code == 0, result.output
        mock_client.market_book_get_as_df.assert_called_once_with(
            symbol="EURUSD",
        )

    def test_order_check(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test order-check command with inline JSON."""
        output = tmp_path / "out.csv"
        request = json.dumps({"action": 1, "symbol": "EURUSD", "volume": 0.1})
        result = runner.invoke(
            app,
            ["-o", str(output), "order-check", "--request", request],
        )
        assert result.exit_code == 0, result.output
        mock_client.order_check_as_df.assert_called_once_with(
            request={"action": 1, "symbol": "EURUSD", "volume": 0.1},
        )

    def test_order_check_file_reference(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test order-check command with file-based JSON."""
        output = tmp_path / "out.csv"
        req_path = tmp_path / "req.json"
        req_path.write_text(
            json.dumps({"action": 2, "symbol": "EURUSD"}),
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            ["-o", str(output), "order-check", "--request", f"@{req_path}"],
        )
        assert result.exit_code == 0, result.output
        mock_client.order_check_as_df.assert_called_once_with(
            request={"action": 2, "symbol": "EURUSD"},
        )

    def test_order_check_invalid_request(
        self,
        tmp_path: Path,
        mock_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test order-check rejects invalid JSON."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            ["-o", str(output), "order-check", "--request", "not-json"],
        )
        assert result.exit_code != 0
        assert "Invalid JSON request" in normalize_cli_output(result.output)

    def test_order_check_missing_request_file(
        self,
        tmp_path: Path,
        mock_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test order-check rejects a missing request file."""
        output = tmp_path / "out.csv"
        missing = tmp_path / "missing.json"
        result = runner.invoke(
            app,
            ["-o", str(output), "order-check", "--request", f"@{missing}"],
        )
        assert result.exit_code != 0
        assert "Failed to read JSON request file" in normalize_cli_output(
            result.output,
        )

    def test_order_send(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test order-send command with file-based JSON."""
        output = tmp_path / "out.csv"
        req_path = tmp_path / "req.json"
        req_path.write_text(
            json.dumps({"action": 2, "symbol": "EURUSD"}),
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "order-send",
                "--request",
                f"@{req_path}",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.order_send_as_df.assert_called_once_with(
            request={"action": 2, "symbol": "EURUSD"},
        )

    def test_order_send_inline_json(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test order-send command with inline JSON."""
        output = tmp_path / "out.csv"
        request = json.dumps({"action": 1, "symbol": "EURUSD", "volume": 0.1})
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "order-send",
                "--request",
                request,
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.order_send_as_df.assert_called_once_with(
            request={"action": 1, "symbol": "EURUSD", "volume": 0.1},
        )

    def test_order_send_requires_yes(
        self,
        tmp_path: Path,
        mock_client: MagicMock,
    ) -> None:
        """Test order-send requires explicit confirmation."""
        output = tmp_path / "out.csv"
        request = json.dumps({"action": 1, "symbol": "EURUSD"})
        result = runner.invoke(
            app,
            ["-o", str(output), "order-send", "--request", request],
        )
        assert result.exit_code != 0
        assert "Pass --yes to send a live trade request" in normalize_cli_output(
            result.output,
        )
        mock_client.order_send_as_df.assert_not_called()

    def test_order_send_invalid_request(
        self,
        tmp_path: Path,
        mock_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test order-send rejects invalid JSON."""
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            ["-o", str(output), "order-send", "--request", "[1,2]", "--yes"],
        )
        assert result.exit_code != 0
        assert "must be a JSON object" in normalize_cli_output(result.output)


# ---------------------------------------------------------------------------
# Callback / shared options
# ---------------------------------------------------------------------------


class TestCallback:
    """Tests for callback (shared options)."""

    def test_format_detection_error(self, tmp_path: Path) -> None:
        """Test that bad extension triggers a user-friendly error."""
        output = tmp_path / "out.xyz"
        result = runner.invoke(
            app,
            ["-o", str(output), "account-info"],
        )
        assert result.exit_code != 0
        assert "Cannot detect format" in normalize_cli_output(result.output)

    def test_connection_args_forwarded(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test that connection arguments reach Mt5Config."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.return_value = pd.DataFrame({"a": [1]})
        mocker.patch(
            "mt5cli.sdk.Mt5DataClient",
            return_value=mock_client,
        )
        mock_config = mocker.patch("mt5cli.cli.Mt5Config")
        output = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "--login",
                "123",
                "--password",
                "pw",
                "--server",
                "srv",
                "-o",
                str(output),
                "account-info",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_config.assert_called_once_with(
            path=None,
            login=123,
            password="pw",
            server="srv",
            timeout=None,
        )

    def test_explicit_format(
        self,
        tmp_path: Path,
        mock_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test explicit --format flag."""
        output = tmp_path / "out.txt"
        result = runner.invoke(
            app,
            ["-o", str(output), "--format", "json", "account-info"],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_sqlite3_with_table(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test SQLite3 output with custom table name."""
        mock_client = MagicMock()
        mock_client.symbols_get_as_df.return_value = pd.DataFrame(
            {"s": ["EURUSD"]},
        )
        mocker.patch(
            "mt5cli.sdk.Mt5DataClient",
            return_value=mock_client,
        )
        output = tmp_path / "out.db"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "--table",
                "symbols",
                "symbols",
                "--group",
                "*USD*",
            ],
        )
        assert result.exit_code == 0, result.output
        with sqlite3.connect(output) as conn:
            result_df = pd.read_sql(  # type: ignore[reportUnknownMemberType]
                "SELECT * FROM symbols",
                conn,
            )
        assert len(result_df) == 1


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


_DEALS_FIXTURE: dict[str, list[object]] = {
    "ticket": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "position_id": [100, 100, 100, 0, 200, 200, 300, 400, 400, 500, 500, 600, 600, 600],
    "symbol": [
        "EURUSD",
        "EURUSD",
        "EURUSD",
        "",
        "EURUSD",
        "EURUSD",
        "GBPUSD",
        "GBPUSD",
        "GBPUSD",
        "EURUSD",
        "EURUSD",
        "GBPUSD",
        "GBPUSD",
        "GBPUSD",
    ],
    "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    # type: 0=BUY, 1=SELL, 2=BALANCE
    "type": [0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 1, 1],
    # entry: 0=IN, 1=OUT, 2=INOUT (reversal), 3=OUT_BY
    "entry": [0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 3, 0, 2, 1],
    "volume": [1.0, 3.0, 4.0, 0.0, 2.0, 2.0, 5.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0],
    "price": [
        1.10,
        1.20,
        1.50,
        0.0,
        2.00,
        2.20,
        1.30,
        1.30,
        1.40,
        1.00,
        1.05,
        1.10,
        9.99,
        1.40,
    ],
    "profit": [0.0, 0.0, 10.0, 5.0, 0.0, 8.0, 0.0, 0.0, -1.0, 0.0, 3.0, 0.0, -2.0, 7.0],
}


def _build_history_client(mocker: MockerFixture) -> MagicMock:
    """Build a mocked Mt5DataClient with per-symbol history results."""
    client = MagicMock()

    def _rates(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({
            "time": [1],
            "open": [1.0],
            "symbol_arg": [kwargs.get("symbol")],
        })

    def _ticks(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({
            "time": [1],
            "bid": [1.0],
            "symbol_arg": [kwargs.get("symbol")],
        })

    client.copy_rates_range_as_df.side_effect = _rates
    client.copy_ticks_range_as_df.side_effect = _ticks

    def _orders(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"ticket": [10], "symbol": [kwargs.get("symbol")]})

    def _deals(**kwargs: object) -> pd.DataFrame:
        sym = kwargs.get("symbol")
        df = pd.DataFrame(_DEALS_FIXTURE)
        return df[df["symbol"] == sym].reset_index(drop=True)

    client.history_orders_get_as_df.side_effect = _orders
    client.history_deals_get_as_df.side_effect = _deals
    mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
    return client


class TestCollectHistory:
    """Tests for the collect-history command."""

    @pytest.fixture
    def history_client(self, mocker: MockerFixture) -> MagicMock:
        """Create a mocked Mt5DataClient with history-style DataFrames."""
        return _build_history_client(mocker)

    def test_collect_history_writes_all_tables(
        self,
        tmp_path: Path,
        history_client: MagicMock,
    ) -> None:
        """Test that collect-history writes rates, ticks, and history tables."""
        output = tmp_path / "history.db"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "collect-history",
                "--symbol",
                "EURUSD",
                "--symbol",
                "GBPUSD",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
            ],
        )
        assert result.exit_code == 0, result.output
        assert history_client.copy_rates_range_as_df.call_count == 2
        assert history_client.copy_ticks_range_as_df.call_count == 2
        history_client.copy_ticks_range_as_df.assert_any_call(
            symbol="EURUSD",
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            date_to=datetime(2024, 2, 1, tzinfo=UTC),
            flags=1,
        )
        with sqlite3.connect(output) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
        assert {"rates", "ticks", "history_orders", "history_deals"} <= tables

    def test_collect_history_history_fetched_per_symbol(
        self,
        tmp_path: Path,
        history_client: MagicMock,
    ) -> None:
        """Test that history-orders and history-deals are fetched per symbol."""
        output = tmp_path / "history.db"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "collect-history",
                "--symbol",
                "EURUSD",
                "--symbol",
                "GBPUSD",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
            ],
        )
        assert result.exit_code == 0, result.output
        assert history_client.history_orders_get_as_df.call_count == 2
        assert history_client.history_deals_get_as_df.call_count == 2
        history_client.history_orders_get_as_df.assert_any_call(
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            date_to=datetime(2024, 2, 1, tzinfo=UTC),
            symbol="EURUSD",
        )
        history_client.history_deals_get_as_df.assert_any_call(
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            date_to=datetime(2024, 2, 1, tzinfo=UTC),
            symbol="GBPUSD",
        )

    @pytest.mark.parametrize(
        ("selected", "expected_tables", "excluded_calls"),
        [
            (
                ["rates", "history-deals"],
                {"rates", "history_deals"},
                ("copy_ticks_range_as_df", "history_orders_get_as_df"),
            ),
            (
                ["ticks", "history-orders"],
                {"ticks", "history_orders"},
                ("copy_rates_range_as_df", "history_deals_get_as_df"),
            ),
        ],
    )
    def test_collect_history_dataset_selection(
        self,
        tmp_path: Path,
        history_client: MagicMock,
        selected: list[str],
        expected_tables: set[str],
        excluded_calls: tuple[str, ...],
    ) -> None:
        """Test that --dataset limits which datasets are fetched and written."""
        output = tmp_path / "history.db"
        args = [
            "-o",
            str(output),
            "collect-history",
            "--symbol",
            "EURUSD",
            "--date-from",
            "2024-01-01",
            "--date-to",
            "2024-02-01",
        ]
        for name in selected:
            args.extend(["--dataset", name])
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        for name in excluded_calls:
            getattr(history_client, name).assert_not_called()
        with sqlite3.connect(output) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
        assert expected_tables <= tables
        assert tables.isdisjoint(
            {"rates", "ticks", "history_orders", "history_deals"} - expected_tables
        )

    def test_collect_history_rates_table_has_timeframe(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test that the rates table carries the requested timeframe value."""
        output = tmp_path / "history.db"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "collect-history",
                "--symbol",
                "EURUSD",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
                "--timeframe",
                "H1",
                "--dataset",
                "rates",
            ],
        )
        assert result.exit_code == 0, result.output
        with sqlite3.connect(output) as conn:
            rows = conn.execute(
                "SELECT DISTINCT timeframe FROM rates",
            ).fetchall()
        assert rows == [(16385,)]

    def test_collect_history_if_exists_append(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test that --if-exists=append accumulates rows across runs."""
        output = tmp_path / "history.db"
        common = [
            "-o",
            str(output),
            "collect-history",
            "--symbol",
            "EURUSD",
            "--date-from",
            "2024-01-01",
            "--date-to",
            "2024-02-01",
            "--dataset",
            "rates",
        ]
        first = runner.invoke(app, common)
        second = runner.invoke(app, [*common, "--if-exists", "append"])
        assert first.exit_code == 0, first.output
        assert second.exit_code == 0, second.output
        with sqlite3.connect(output) as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM rates").fetchone()
        assert count == 2

    def test_collect_history_if_exists_fail(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test that --if-exists=fail rejects writing into an existing table."""
        output = tmp_path / "history.db"
        common = [
            "-o",
            str(output),
            "collect-history",
            "--symbol",
            "EURUSD",
            "--date-from",
            "2024-01-01",
            "--date-to",
            "2024-02-01",
            "--dataset",
            "rates",
        ]
        first = runner.invoke(app, common)
        second = runner.invoke(app, [*common, "--if-exists", "fail"])
        assert first.exit_code == 0, first.output
        assert second.exit_code != 0

    def test_collect_history_ticks_default_flags_all(
        self,
        tmp_path: Path,
        history_client: MagicMock,
    ) -> None:
        """Test that --flags defaults to ALL for ticks."""
        output = tmp_path / "history.db"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "collect-history",
                "--symbol",
                "EURUSD",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
            ],
        )
        assert result.exit_code == 0, result.output
        history_client.copy_ticks_range_as_df.assert_called_once_with(
            symbol="EURUSD",
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            date_to=datetime(2024, 2, 1, tzinfo=UTC),
            flags=1,
        )

    def test_collect_history_with_views(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test that --with-views creates cash_events and positions views."""
        output = tmp_path / "history.db"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "collect-history",
                "--symbol",
                "EURUSD",
                "--symbol",
                "GBPUSD",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
                "--with-views",
            ],
        )
        assert result.exit_code == 0, result.output
        with sqlite3.connect(output) as conn:
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                ).fetchall()
            }
            cash = conn.execute("SELECT type FROM cash_events").fetchall()
            positions = {
                row[0]: row
                for row in conn.execute(
                    "SELECT position_id, volume_open, volume_close,"
                    " volume_reversal, open_price, close_price, reversal_count"
                    " FROM positions_reconstructed",
                ).fetchall()
            }
        assert {"cash_events", "positions_reconstructed"} <= views
        assert all(row[0] not in {0, 1} for row in cash)
        # Position 100 (BUY 1@1.10 + BUY 3@1.20 then SELL 4@1.50) is closed.
        # Position 200 (BUY 2@2.00 then SELL 2@2.20) is closed.
        # Position 400 (reversal-only with non-trade deal type) stays excluded.
        assert set(positions) == {100, 200, 500, 600}
        pos_100 = positions[100]
        tol = 1e-9
        assert abs(pos_100[1] - 4.0) < tol  # volume_open
        assert abs(pos_100[2] - 4.0) < tol  # volume_close
        assert abs(pos_100[3] - 0.0) < tol  # volume_reversal
        # Volume-weighted open: (1*1.10 + 3*1.20) / 4 = 1.175
        assert abs(pos_100[4] - 1.175) < tol
        # Volume-weighted close: (4*1.50) / 4 = 1.50
        assert abs(pos_100[5] - 1.50) < tol
        assert pos_100[6] == 0  # reversal_count
        pos_500 = positions[500]
        assert abs(pos_500[2] - 2.0) < tol  # OUT_BY contributes to close volume
        assert abs(pos_500[5] - 1.05) < tol
        pos_600 = positions[600]
        assert abs(pos_600[1] - 3.0) < tol
        assert abs(pos_600[2] - 4.0) < tol  # reversal + close volumes
        assert abs(pos_600[3] - 1.0) < tol
        assert abs(pos_600[4] - 1.10) < tol
        assert abs(pos_600[5] - 3.5475) < tol
        assert pos_600[6] == 1

    def test_collect_history_filters_history_symbols_exactly(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test that history wildcard results are filtered to exact symbols."""
        client = MagicMock()
        client.history_orders_get_as_df.return_value = pd.DataFrame({
            "ticket": [1, 2],
            "symbol": ["EURUSD", "EURUSDm"],
        })
        client.history_deals_get_as_df.return_value = pd.DataFrame({
            "ticket": [3, 4],
            "symbol": ["EURUSD", "EURUSDm"],
        })
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
        output = tmp_path / "history.db"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "collect-history",
                "--symbol",
                "EURUSD",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
                "--dataset",
                "history-orders",
                "--dataset",
                "history-deals",
            ],
        )
        assert result.exit_code == 0, result.output
        with sqlite3.connect(output) as conn:
            order_symbols = conn.execute(
                "SELECT DISTINCT symbol FROM history_orders",
            ).fetchall()
            deal_symbols = conn.execute(
                "SELECT DISTINCT symbol FROM history_deals",
            ).fetchall()
        assert order_symbols == [("EURUSD",)]
        assert deal_symbols == [("EURUSD",)]

    def test_collect_history_requires_sqlite_format(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test that non-SQLite output is rejected."""
        output = tmp_path / "history.csv"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "collect-history",
                "--symbol",
                "EURUSD",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
            ],
        )
        assert result.exit_code != 0
        assert "requires SQLite3" in normalize_cli_output(result.output)

    def test_collect_history_requires_symbol(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test that at least one --symbol is required."""
        output = tmp_path / "history.db"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "collect-history",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
            ],
        )
        assert result.exit_code != 0

    def test_collect_history_views_skipped_when_columns_missing(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that views are not created when required columns are missing."""
        client = MagicMock()
        client.copy_rates_range_as_df.return_value = pd.DataFrame({"x": [1]})
        client.copy_ticks_range_as_df.return_value = pd.DataFrame({"x": [1]})
        client.history_orders_get_as_df.return_value = pd.DataFrame({"x": [1]})
        client.history_deals_get_as_df.return_value = pd.DataFrame({"x": [1]})
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
        output = tmp_path / "history.db"
        with caplog.at_level(logging.WARNING, logger="mt5cli.sdk"):
            result = runner.invoke(
                app,
                [
                    "-o",
                    str(output),
                    "collect-history",
                    "--symbol",
                    "EURUSD",
                    "--date-from",
                    "2024-01-01",
                    "--date-to",
                    "2024-02-01",
                    "--with-views",
                ],
            )
        assert result.exit_code == 0, result.output
        with sqlite3.connect(output) as conn:
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                ).fetchall()
            }
        assert "cash_events" not in views
        assert "positions_reconstructed" not in views
        assert "Skipping cash_events view" in caplog.text
        assert "Skipping positions_reconstructed view" in caplog.text

    def test_collect_history_skips_empty_history_without_columns(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test that empty no-column history results do not fail collection."""
        client = MagicMock()
        client.copy_rates_range_as_df.return_value = pd.DataFrame({"time": [1]})
        client.history_deals_get_as_df.return_value = pd.DataFrame()
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
        output = tmp_path / "history.db"
        result = runner.invoke(
            app,
            [
                "-o",
                str(output),
                "collect-history",
                "--symbol",
                "EURUSD",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-02-01",
                "--dataset",
                "rates",
                "--dataset",
                "history-deals",
            ],
        )
        assert result.exit_code == 0, result.output
        with sqlite3.connect(output) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
        assert "rates" in tables
        assert "history_deals" not in tables

    def test_collect_history_warns_when_views_requested_without_deals(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that --with-views warns when history_deals is not written."""
        output = tmp_path / "history.db"
        with caplog.at_level(logging.WARNING, logger="mt5cli.sdk"):
            result = runner.invoke(
                app,
                [
                    "-o",
                    str(output),
                    "collect-history",
                    "--symbol",
                    "EURUSD",
                    "--date-from",
                    "2024-01-01",
                    "--date-to",
                    "2024-02-01",
                    "--dataset",
                    "rates",
                    "--with-views",
                ],
            )
        assert result.exit_code == 0, result.output
        assert (
            "--with-views ignored: history_deals table was not written" in caplog.text
        )


class TestMain:
    """Tests for the main entry point."""

    def test_main_invokes_app(self, mocker: MockerFixture) -> None:
        """Test that main() calls the typer app."""
        mock_app = mocker.patch("mt5cli.cli.app")
        main()
        mock_app.assert_called_once()
