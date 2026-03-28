"""Tests for mt5cli.cli module."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture  # noqa: TC002
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from mt5cli.cli import (
    DATETIME_TYPE,
    TICK_FLAG_MAP,
    TICK_FLAGS_TYPE,
    TIMEFRAME_MAP,
    TIMEFRAME_TYPE,
    _execute_export,  # type: ignore[reportPrivateUsage]
    _ExportContext,  # type: ignore[reportPrivateUsage]
    app,
    detect_format,
    export_dataframe,
    main,
    parse_datetime,
    parse_tick_flags,
    parse_timeframe,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


class TestDetectFormat:
    """Tests for detect_format."""

    def test_explicit_format_returned(self, tmp_path: Path) -> None:
        """Test that explicit format overrides extension."""
        result = detect_format(tmp_path / "data.txt", explicit_format="csv")
        assert result == "csv"

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("data.csv", "csv"),
            ("data.json", "json"),
            ("data.parquet", "parquet"),
            ("data.pq", "parquet"),
            ("data.db", "sqlite3"),
            ("data.sqlite", "sqlite3"),
            ("data.sqlite3", "sqlite3"),
            ("DATA.CSV", "csv"),
            ("DATA.JSON", "json"),
            ("DATA.PARQUET", "parquet"),
        ],
    )
    def test_auto_detect_from_extension(
        self,
        tmp_path: Path,
        filename: str,
        expected: str,
    ) -> None:
        """Test format auto-detection from file extension."""
        result = detect_format(tmp_path / filename)
        assert result == expected

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        """Test that unknown extension raises ValueError."""
        with pytest.raises(ValueError, match="Cannot detect format"):
            detect_format(tmp_path / "data.xyz")


# ---------------------------------------------------------------------------
# export_dataframe
# ---------------------------------------------------------------------------


class TestExportDataframe:
    """Tests for export_dataframe."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    def test_export_csv(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test CSV export."""
        output = tmp_path / "out.csv"
        export_dataframe(sample_df, output, "csv")
        result = pd.read_csv(output)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_export_json(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test JSON export."""
        output = tmp_path / "out.json"
        export_dataframe(sample_df, output, "json")
        with output.open() as f:
            records = json.load(f)
        assert len(records) == 3
        assert records[0]["a"] == 1

    def test_export_parquet(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test Parquet export."""
        output = tmp_path / "out.parquet"
        export_dataframe(sample_df, output, "parquet")
        result = pd.read_parquet(output)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_export_sqlite3(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test SQLite3 export."""
        output = tmp_path / "out.db"
        export_dataframe(sample_df, output, "sqlite3", table_name="test_table")
        with sqlite3.connect(output) as conn:
            result = pd.read_sql(  # type: ignore[reportUnknownMemberType]
                "SELECT * FROM test_table",
                conn,
            )
        pd.testing.assert_frame_equal(result, sample_df)

    def test_unsupported_format_raises(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            export_dataframe(sample_df, tmp_path / "out.txt", "xml")


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


class TestParseDatetime:
    """Tests for parse_datetime."""

    def test_valid_date(self) -> None:
        """Test parsing a date string."""
        result = parse_datetime("2024-01-15")
        assert result == datetime(2024, 1, 15, tzinfo=UTC)

    def test_valid_datetime_with_tz(self) -> None:
        """Test parsing a datetime with timezone."""
        result = parse_datetime("2024-01-15T12:00:00+00:00")
        assert result == datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

    def test_invalid_format_raises(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid datetime"):
            parse_datetime("not-a-date")


class TestParseTimeframe:
    """Tests for parse_timeframe."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("M1", 1), ("h1", 16385), ("D1", 16408), ("MN1", 49153)],
    )
    def test_named_timeframe(self, value: str, expected: int) -> None:
        """Test parsing named timeframes."""
        assert parse_timeframe(value) == expected

    def test_integer_timeframe(self) -> None:
        """Test parsing integer timeframe."""
        assert parse_timeframe("42") == 42

    def test_invalid_timeframe_raises(self) -> None:
        """Test that invalid timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            parse_timeframe("INVALID")


class TestParseTickFlags:
    """Tests for parse_tick_flags."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("ALL", 1), ("info", 2), ("TRADE", 4)],
    )
    def test_named_flag(self, value: str, expected: int) -> None:
        """Test parsing named tick flags."""
        assert parse_tick_flags(value) == expected

    def test_integer_flag(self) -> None:
        """Test parsing integer tick flag."""
        assert parse_tick_flags("7") == 7

    def test_invalid_flag_raises(self) -> None:
        """Test that invalid flag raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tick flags"):
            parse_tick_flags("INVALID")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module constants."""

    def test_timeframe_map_has_expected_keys(self) -> None:
        """Test that TIMEFRAME_MAP contains standard timeframes."""
        for key in ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"):
            assert key in TIMEFRAME_MAP

    def test_tick_flag_map_has_expected_keys(self) -> None:
        """Test that TICK_FLAG_MAP contains standard flags."""
        assert set(TICK_FLAG_MAP) == {"ALL", "INFO", "TRADE"}


# ---------------------------------------------------------------------------
# Click ParamTypes
# ---------------------------------------------------------------------------


class TestDateTimeType:
    """Tests for _DateTimeType."""

    def test_convert_string(self) -> None:
        """Test converting a string to datetime."""
        result = DATETIME_TYPE.convert("2024-06-15", None, None)
        assert result == datetime(2024, 6, 15, tzinfo=UTC)

    def test_convert_datetime_passthrough(self) -> None:
        """Test that datetime values pass through unchanged."""
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        assert DATETIME_TYPE.convert(dt, None, None) is dt

    def test_convert_invalid(self) -> None:
        """Test that invalid values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid datetime"):
            DATETIME_TYPE.convert("bad", None, None)


class TestTimeframeType:
    """Tests for _TimeframeType."""

    def test_convert_string(self) -> None:
        """Test converting a string to timeframe integer."""
        assert TIMEFRAME_TYPE.convert("H1", None, None) == 16385

    def test_convert_int_passthrough(self) -> None:
        """Test that integer values pass through unchanged."""
        assert TIMEFRAME_TYPE.convert(42, None, None) == 42

    def test_convert_invalid(self) -> None:
        """Test that invalid values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid timeframe"):
            TIMEFRAME_TYPE.convert("bad", None, None)


class TestTickFlagsType:
    """Tests for _TickFlagsType."""

    def test_convert_string(self) -> None:
        """Test converting a string to tick flags integer."""
        assert TICK_FLAGS_TYPE.convert("ALL", None, None) == 1

    def test_convert_int_passthrough(self) -> None:
        """Test that integer values pass through unchanged."""
        assert TICK_FLAGS_TYPE.convert(7, None, None) == 7

    def test_convert_invalid(self) -> None:
        """Test that invalid values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid tick flags"):
            TICK_FLAGS_TYPE.convert("bad", None, None)


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
        mocker.patch("mt5cli.cli.Mt5DataClient", return_value=mock_client)
        ctx = MagicMock()
        ctx.obj = _ExportContext(
            output=tmp_path / "out.csv",
            output_format="csv",
            table="data",
            config=MagicMock(),
        )
        with pytest.raises(RuntimeError, match="boom"):
            _execute_export(ctx, lambda c: c.account_info_as_df())
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
    mocker.patch("mt5cli.cli.Mt5DataClient", return_value=client)
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
        assert "Cannot detect format" in result.output

    def test_connection_args_forwarded(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test that connection arguments reach Mt5Config."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.return_value = pd.DataFrame({"a": [1]})
        mocker.patch(
            "mt5cli.cli.Mt5DataClient",
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
            "mt5cli.cli.Mt5DataClient",
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


class TestMain:
    """Tests for the main entry point."""

    def test_main_invokes_app(self, mocker: MockerFixture) -> None:
        """Test that main() calls the typer app."""
        mock_app = mocker.patch("mt5cli.cli.app")
        main()
        mock_app.assert_called_once()
