"""Tests for mt5cli.utils module."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import mt5cli.utils

if TYPE_CHECKING:
    from pathlib import Path

from mt5cli.utils import (
    DATETIME_TYPE,
    REQUEST_TYPE,
    TICK_FLAGS_TYPE,
    TIMEFRAME_TYPE,
    Dataset,
    IfExists,
    detect_format,
    export_dataframe,
    export_dataframe_to_sqlite,
    parse_datetime,
    parse_request,
    parse_tick_flags,
    parse_timeframe,
)

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

    def test_export_parquet_without_pyarrow(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a clear error is raised when pyarrow is not installed."""
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        with pytest.raises(ImportError, match="mt5cli\\[parquet\\]"):
            export_dataframe(sample_df, tmp_path / "out.parquet", "parquet")

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


class TestExportDataframeToSqlite:
    """Tests for export_dataframe_to_sqlite."""

    def test_append_preserves_existing_rows(self, tmp_path: Path) -> None:
        """Test append mode keeps prior rows in the SQLite table."""
        output = tmp_path / "append.db"
        first = pd.DataFrame({"id": [1], "value": ["a"]})
        second = pd.DataFrame({"id": [2], "value": ["b"]})
        export_dataframe_to_sqlite(first, output, "items", if_exists=IfExists.REPLACE)
        export_dataframe_to_sqlite(second, output, "items", if_exists=IfExists.APPEND)
        with sqlite3.connect(output) as conn:
            result = pd.read_sql(  # type: ignore[reportUnknownMemberType]
                "SELECT id, value FROM items ORDER BY id",
                conn,
            )
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame({"id": [1, 2], "value": ["a", "b"]}),
        )

    def test_deduplicate_keeps_latest_row(self, tmp_path: Path) -> None:
        """Test deduplication keeps the latest ROWID for key columns."""
        output = tmp_path / "dedup.db"
        first = pd.DataFrame({
            "symbol": ["EURUSD", "EURUSD"],
            "time": ["2024-01-01", "2024-01-01"],
            "bid": [1.0, 1.1],
        })
        second = pd.DataFrame({
            "symbol": ["EURUSD"],
            "time": ["2024-01-01"],
            "bid": [1.2],
        })
        export_dataframe_to_sqlite(
            first,
            output,
            "ticks",
            if_exists=IfExists.REPLACE,
            deduplicate_on=("symbol", "time"),
        )
        export_dataframe_to_sqlite(
            second,
            output,
            "ticks",
            if_exists=IfExists.APPEND,
            deduplicate_on=("symbol", "time"),
        )
        with sqlite3.connect(output) as conn:
            result = pd.read_sql(  # type: ignore[reportUnknownMemberType]
                "SELECT symbol, time, bid FROM ticks",
                conn,
            )
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            pd.DataFrame({
                "symbol": ["EURUSD"],
                "time": ["2024-01-01"],
                "bid": [1.2],
            }),
        )

    def test_default_if_exists_appends_without_dropping_rows(
        self,
        tmp_path: Path,
    ) -> None:
        """Test the default append mode keeps prior rows."""
        output = tmp_path / "default-append.db"
        first = pd.DataFrame({"id": [1], "value": ["a"]})
        second = pd.DataFrame({"id": [2], "value": ["b"]})
        export_dataframe_to_sqlite(first, output, "items")
        export_dataframe_to_sqlite(second, output, "items")
        with sqlite3.connect(output) as conn:
            result = pd.read_sql(  # type: ignore[reportUnknownMemberType]
                "SELECT id, value FROM items ORDER BY id",
                conn,
            )
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame({"id": [1, 2], "value": ["a", "b"]}),
        )

    def test_writes_index_with_label(self, tmp_path: Path) -> None:
        """Test optional index export with a custom label."""
        output = tmp_path / "index.db"
        frame = pd.DataFrame(
            {"value": [1.0]}, index=pd.Index(["EURUSD"], name="symbol")
        )
        export_dataframe_to_sqlite(
            frame,
            output,
            "margins",
            if_exists=IfExists.REPLACE,
            index=True,
            index_label="symbol",
        )
        with sqlite3.connect(output) as conn:
            result = pd.read_sql(  # type: ignore[reportUnknownMemberType]
                "SELECT symbol, value FROM margins",
                conn,
            )
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame({"symbol": ["EURUSD"], "value": [1.0]}),
        )


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
        """Test parsing supported integer timeframes."""
        assert parse_timeframe("1") == 1
        assert parse_timeframe(16385) == 16385

    def test_unsupported_integer_timeframe_raises(self) -> None:
        """Test that unsupported integer timeframes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            parse_timeframe("42")

    def test_invalid_timeframe_raises(self) -> None:
        """Test that invalid timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            parse_timeframe("INVALID")


class TestParseTickFlags:
    """Tests for parse_tick_flags."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("ALL", -1), ("info", 1), ("TRADE", 2), ("COPY_TICKS_ALL", -1)],
    )
    def test_named_flag(self, value: str, expected: int) -> None:
        """Test parsing named tick flags."""
        assert parse_tick_flags(value) == expected

    def test_integer_flag(self) -> None:
        """Test parsing supported integer tick flags."""
        assert parse_tick_flags("-1") == -1
        assert parse_tick_flags(2) == 2

    def test_unsupported_integer_flag_raises(self) -> None:
        """Test that unsupported integer tick flags raise ValueError."""
        with pytest.raises(ValueError, match="Invalid tick flags"):
            parse_tick_flags("7")

    def test_invalid_flag_raises(self) -> None:
        """Test that invalid flag raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tick flags"):
            parse_tick_flags("INVALID")


# ---------------------------------------------------------------------------
# parse_request
# ---------------------------------------------------------------------------


class TestParseRequest:
    """Tests for parse_request."""

    def test_inline_json(self) -> None:
        """Test parsing an inline JSON object string."""
        result = parse_request('{"action": 1, "symbol": "EURUSD"}')
        assert result == {"action": 1, "symbol": "EURUSD"}

    def test_file_reference(self, tmp_path: Path) -> None:
        """Test parsing JSON from a file via the @path syntax."""
        path = tmp_path / "req.json"
        path.write_text('{"action": 2}', encoding="utf-8")
        result = parse_request(f"@{path}")
        assert result == {"action": 2}

    def test_invalid_json_raises(self) -> None:
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON request"):
            parse_request("not json")

    def test_non_object_raises(self) -> None:
        """Test that a non-object JSON raises ValueError."""
        with pytest.raises(ValueError, match="must be a JSON object"):
            parse_request("[1, 2, 3]")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Test that a missing request file raises ValueError."""
        path = tmp_path / "missing.json"
        with pytest.raises(ValueError, match="Failed to read JSON request file"):
            parse_request(f"@{path}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module constants."""

    def test_timeframe_map_is_private_in_utils(self) -> None:
        """TIMEFRAME_MAP is a private implementation detail; not a public attribute."""
        assert not hasattr(mt5cli.utils, "TIMEFRAME_MAP")

    def test_tick_flag_map_absent_from_utils(self) -> None:
        """TICK_FLAG_MAP is not exposed by mt5cli.utils."""
        assert not hasattr(mt5cli.utils, "TICK_FLAG_MAP")

    @pytest.mark.parametrize(
        ("dataset", "expected"),
        [
            (Dataset.rates, "rates"),
            (Dataset.ticks, "ticks"),
            (Dataset.history_orders, "history_orders"),
            (Dataset.history_deals, "history_deals"),
        ],
    )
    def test_dataset_table_name(self, dataset: Dataset, expected: str) -> None:
        """Test dataset SQLite table names."""
        assert dataset.table_name == expected


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

    def test_convert_int(self) -> None:
        """Test converting supported integer timeframe values."""
        assert TIMEFRAME_TYPE.convert(16385, None, None) == 16385

    def test_convert_unsupported_int(self) -> None:
        """Test that unsupported integer values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid timeframe"):
            TIMEFRAME_TYPE.convert(42, None, None)

    def test_convert_invalid(self) -> None:
        """Test that invalid values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid timeframe"):
            TIMEFRAME_TYPE.convert("bad", None, None)

    @pytest.mark.parametrize("value", [True, False, None, 1.5])
    def test_convert_invalid_types(self, value: object) -> None:
        """Test that bool, float, and None values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid timeframe"):
            TIMEFRAME_TYPE.convert(value, None, None)


class TestTickFlagsType:
    """Tests for _TickFlagsType."""

    def test_convert_string(self) -> None:
        """Test converting a string to tick flags integer."""
        assert TICK_FLAGS_TYPE.convert("ALL", None, None) == -1

    def test_convert_int(self) -> None:
        """Test converting supported integer tick flag values."""
        assert TICK_FLAGS_TYPE.convert(2, None, None) == 2

    def test_convert_unsupported_int(self) -> None:
        """Test that unsupported integer values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid tick flags"):
            TICK_FLAGS_TYPE.convert(7, None, None)

    @pytest.mark.parametrize("value", [True, False, None, 1.5])
    def test_convert_invalid_types(self, value: object) -> None:
        """Test that bool, float, and None values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid tick flags"):
            TICK_FLAGS_TYPE.convert(value, None, None)

    def test_convert_invalid(self) -> None:
        """Test that invalid values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid tick flags"):
            TICK_FLAGS_TYPE.convert("bad", None, None)


class TestRequestType:
    """Tests for _RequestType."""

    def test_convert_string(self) -> None:
        """Test converting a JSON string to a request dictionary."""
        assert REQUEST_TYPE.convert('{"action": 1}', None, None) == {"action": 1}

    def test_convert_invalid(self) -> None:
        """Test that invalid values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid JSON request"):
            REQUEST_TYPE.convert("bad", None, None)
