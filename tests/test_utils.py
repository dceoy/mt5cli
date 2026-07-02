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

    @pytest.mark.parametrize(
        ("filename", "output_format", "reader"),
        [
            ("out.csv", "csv", "csv"),
            ("out.json", "json", "json"),
            ("out.parquet", "parquet", "parquet"),
            ("out.db", "sqlite3", "sqlite3"),
        ],
        ids=["csv", "json", "parquet", "sqlite3"],
    )
    def test_export_round_trip(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        filename: str,
        output_format: str,
        reader: str,
    ) -> None:
        """Test CSV/JSON/Parquet/SQLite3 exports round-trip the sample DataFrame."""
        output = tmp_path / filename
        export_dataframe(sample_df, output, output_format, table_name="test_table")
        if reader == "csv":
            result = pd.read_csv(output)
            pd.testing.assert_frame_equal(result, sample_df)
        elif reader == "json":
            with output.open() as f:
                records = json.load(f)
            assert len(records) == 3
            assert records[0]["a"] == 1
        elif reader == "sqlite3":
            with sqlite3.connect(output) as conn:
                result = pd.read_sql(  # type: ignore[reportUnknownMemberType]
                    "SELECT * FROM test_table",
                    conn,
                )
            pd.testing.assert_frame_equal(result, sample_df)
        else:
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

    @pytest.mark.parametrize(
        ("first_if_exists", "second_if_exists"),
        [
            pytest.param(IfExists.REPLACE, IfExists.APPEND, id="replace-then-append"),
            pytest.param(None, None, id="default-append"),
        ],
    )
    def test_append_preserves_existing_rows(
        self,
        tmp_path: Path,
        first_if_exists: IfExists | None,
        second_if_exists: IfExists | None,
    ) -> None:
        """Test explicit append and default append modes keep prior rows."""
        output = tmp_path / "append.db"
        first = pd.DataFrame({"id": [1], "value": ["a"]})
        second = pd.DataFrame({"id": [2], "value": ["b"]})
        if first_if_exists is None:
            export_dataframe_to_sqlite(first, output, "items")
        else:
            export_dataframe_to_sqlite(
                first,
                output,
                "items",
                if_exists=first_if_exists,
            )
        if second_if_exists is None:
            export_dataframe_to_sqlite(second, output, "items")
        else:
            export_dataframe_to_sqlite(
                second,
                output,
                "items",
                if_exists=second_if_exists,
            )
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

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2024-01-15", datetime(2024, 1, 15, tzinfo=UTC)),
            ("2024-01-15T12:00:00+00:00", datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)),
        ],
        ids=["date", "datetime-with-tz"],
    )
    def test_valid_inputs(self, value: str, expected: datetime) -> None:
        """Test parsing valid date and datetime strings."""
        assert parse_datetime(value) == expected

    def test_invalid_format_raises(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid datetime"):
            parse_datetime("not-a-date")


class TestParseTimeframe:
    """Tests for parse_timeframe."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("M1", 1),
            ("h1", 16385),
            ("D1", 16408),
            ("MN1", 49153),
            ("1", 1),
            (16385, 16385),
        ],
        ids=["M1", "h1", "D1", "MN1", "int-string-1", "int-16385"],
    )
    def test_valid_timeframe(self, value: str | int, expected: int) -> None:
        """Test parsing valid string and integer timeframe values."""
        assert parse_timeframe(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["42", "INVALID"],
        ids=["unsupported-integer", "invalid-string"],
    )
    def test_invalid_timeframe_raises(self, value: str) -> None:
        """Test that invalid timeframe values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            parse_timeframe(value)


class TestParseTickFlags:
    """Tests for parse_tick_flags."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("ALL", -1),
            ("info", 1),
            ("TRADE", 2),
            ("COPY_TICKS_ALL", -1),
            ("-1", -1),
            (2, 2),
        ],
        ids=["ALL", "info", "TRADE", "COPY_TICKS_ALL", "int-string--1", "int-2"],
    )
    def test_valid_flag(self, value: str | int, expected: int) -> None:
        """Test parsing valid string and integer tick flag values."""
        assert parse_tick_flags(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["7", "INVALID"],
        ids=["unsupported-integer", "invalid-string"],
    )
    def test_invalid_flag_raises(self, value: str) -> None:
        """Test that invalid tick flag values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid tick flags"):
            parse_tick_flags(value)


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

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            pytest.param("not json", "Invalid JSON request", id="invalid-json"),
            pytest.param("[1, 2, 3]", "must be a JSON object", id="non-object"),
        ],
    )
    def test_invalid_request_raises(self, value: str, match: str) -> None:
        """Test that invalid JSON and non-object requests raise ValueError."""
        with pytest.raises(ValueError, match=match):
            parse_request(value)

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

    @pytest.mark.parametrize("name", ["TIMEFRAME_MAP", "TICK_FLAG_MAP"])
    def test_private_maps_absent_from_utils(self, name: str) -> None:
        """Private pdmt5 maps are not exposed as public mt5cli.utils attributes."""
        assert not hasattr(mt5cli.utils, name)

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

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("H1", 16385), (16385, 16385)],
        ids=["string", "int"],
    )
    def test_convert_valid(self, value: str | int, expected: int) -> None:
        """Test converting valid string and integer timeframe values."""
        assert TIMEFRAME_TYPE.convert(value, None, None) == expected

    @pytest.mark.parametrize(
        "value",
        [42, "bad"],
        ids=["unsupported-int", "invalid-string"],
    )
    def test_convert_invalid(self, value: object) -> None:
        """Test that unsupported int and invalid string values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid timeframe"):
            TIMEFRAME_TYPE.convert(value, None, None)

    @pytest.mark.parametrize("value", [True, False, None, 1.5])
    def test_convert_invalid_types(self, value: object) -> None:
        """Test that bool, float, and None values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid timeframe"):
            TIMEFRAME_TYPE.convert(value, None, None)


class TestTickFlagsType:
    """Tests for _TickFlagsType."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("ALL", -1), (2, 2)],
        ids=["string", "int"],
    )
    def test_convert_valid(self, value: str | int, expected: int) -> None:
        """Test converting valid string and integer tick flag values."""
        assert TICK_FLAGS_TYPE.convert(value, None, None) == expected

    @pytest.mark.parametrize(
        "value",
        [7, "bad"],
        ids=["unsupported-int", "invalid-string"],
    )
    def test_convert_invalid(self, value: object) -> None:
        """Test that unsupported int and invalid string values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid tick flags"):
            TICK_FLAGS_TYPE.convert(value, None, None)

    @pytest.mark.parametrize("value", [True, False, None, 1.5])
    def test_convert_invalid_types(self, value: object) -> None:
        """Test that bool, float, and None values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid tick flags"):
            TICK_FLAGS_TYPE.convert(value, None, None)


class TestRequestType:
    """Tests for _RequestType."""

    def test_convert_string(self) -> None:
        """Test converting a JSON string to a request dictionary."""
        assert REQUEST_TYPE.convert('{"action": 1}', None, None) == {"action": 1}

    def test_convert_invalid(self) -> None:
        """Test that invalid values raise BadParameter."""
        with pytest.raises(Exception, match="Invalid JSON request"):
            REQUEST_TYPE.convert("bad", None, None)
