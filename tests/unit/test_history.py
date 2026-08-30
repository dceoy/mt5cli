"""Tests for mt5cli.history module."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

from pdmt5 import TIMEFRAME_MAP, Mt5RuntimeError

from mt5cli import history
from mt5cli.exceptions import Mt5ConnectionError
from mt5cli.history import (
    DEFAULT_HISTORY_DATASETS,
    DEFAULT_HISTORY_TIMEFRAMES,
    DedupScope,
    RateTarget,
    ThrottledHistoryUpdater,
    append_dataframe,
    augment_written_columns_from_sqlite,
    build_rate_targets,
    collect_history,
    create_cash_events_view,
    create_history_indexes,
    create_positions_reconstructed_view,
    deduplicate_history_tables,
    drop_duplicates_in_table,
    drop_forming_rate_bar,
    filter_incremental_history_deals_frame,
    filter_trade_history_frame,
    get_history_deals_account_event_start_datetime,
    get_incremental_start_datetime,
    get_table_columns,
    load_incremental_start_datetimes,
    load_rate_data,
    load_rate_data_from_connection,
    parse_sqlite_timestamp,
    quote_sqlite_identifier,
    record_written_columns,
    report_rate_gaps,
    resolve_granularity_name,
    resolve_history_datasets,
    resolve_history_tick_flags,
    resolve_history_timeframes,
    update_history,
    update_history_with_config,
    write_collected_datasets,
    write_history_dataset,
    write_rates_dataset,
    write_streamed_frame,
    write_symbols_dataset,
)
from mt5cli.utils import Dataset, IfExists, parse_timeframe

_TEST_DATE_TO = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
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
    "type": [0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 1, 1],
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
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=client)
    return client


class TestQuoteSqliteIdentifier:
    """Tests for quote_sqlite_identifier."""

    @pytest.mark.parametrize(
        "symbol",
        ["EUR/USD", "US500.cash", "#US500"],
    )
    def test_quotes_broker_specific_symbols(self, symbol: str) -> None:
        """Test broker-specific symbols are safely quoted."""
        quoted = quote_sqlite_identifier(f"rate_{symbol}")
        assert quoted.startswith('"')
        assert quoted.endswith('"')


class TestLoadRateData:
    """Tests for SQLite rate-like table and view loading."""

    def test_loads_close_rates_from_path_with_count(self, tmp_path: Path) -> None:
        """Test loading the latest close-based rates in ascending time order."""
        db_path = tmp_path / "rates.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE rates(time TEXT, close REAL)")
            conn.executemany(
                "INSERT INTO rates(time, close) VALUES (?, ?)",
                [
                    ("2024-01-01T00:00:00+00:00", 1.0),
                    ("2024-01-01T00:02:00+00:00", 1.2),
                    ("2024-01-01T00:01:00+00:00", 1.1),
                ],
            )
        frame = load_rate_data(db_path, "rates", count=2)
        assert list(frame["close"]) == [1.1, 1.2]
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert frame.index.name == "time"
        assert frame.index.is_monotonic_increasing

    def test_loads_ask_bid_tick_like_rates_from_connection(
        self,
        tmp_path: Path,
    ) -> None:
        """Test loading tick-like tables with bid and ask columns."""
        db_path = tmp_path / "ticks.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE ticks(time TEXT, bid REAL, ask REAL)")
            conn.execute(
                "INSERT INTO ticks(time, bid, ask) VALUES (?, ?, ?)",
                ("2024-01-01T00:00:00+00:00", 1.0, 1.1),
            )
            frame = load_rate_data_from_connection(conn, "ticks")
            path_frame = load_rate_data(conn, "ticks")
        assert frame.iloc[0].to_dict() == {"bid": 1.0, "ask": 1.1}
        assert path_frame.iloc[0].to_dict() == {"bid": 1.0, "ask": 1.1}

    def test_loads_from_view(self, tmp_path: Path) -> None:
        """Test loading from a SQLite view."""
        db_path = tmp_path / "view.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE rates(time TEXT, close REAL)")
            conn.execute(
                "INSERT INTO rates(time, close) VALUES (?, ?)",
                ("2024-01-01T00:00:00+00:00", 1.0),
            )
            conn.execute("CREATE VIEW rate_view AS SELECT time, close FROM rates")
            frame = load_rate_data_from_connection(conn, "rate_view")
        assert list(frame["close"]) == [1.0]

    def test_loads_quoted_identifier(self, tmp_path: Path) -> None:
        """Test table names are quoted safely."""
        db_path = tmp_path / "quoted.db"
        table = 'rate "quoted"'
        quoted = quote_sqlite_identifier(table)
        with sqlite3.connect(db_path) as conn:
            conn.execute(f"CREATE TABLE {quoted}(time TEXT, close REAL)")
            conn.execute(
                f"INSERT INTO {quoted}(time, close) VALUES (?, ?)",  # noqa: S608
                ("2024-01-01T00:00:00+00:00", 1.0),
            )
            frame = load_rate_data_from_connection(conn, table)
        assert list(frame["close"]) == [1.0]

    @pytest.mark.parametrize(
        ("db_path", "match"),
        [
            pytest.param(
                "missing.db", "SQLite database not found", id="missing-database"
            ),
            pytest.param(".", "not a file", id="non-file"),
        ],
    )
    def test_rejects_missing_database_and_non_file(
        self,
        tmp_path: Path,
        db_path: str,
        match: str,
    ) -> None:
        """Test path validation for SQLite database inputs."""
        with pytest.raises(ValueError, match=match):
            load_rate_data(tmp_path / db_path, "rates")

    @pytest.mark.parametrize(
        ("table", "count", "match"),
        [
            ("", None, "must not be empty"),
            ("rates", 0, "count must be positive"),
            ("rates", -1, "count must be positive"),
        ],
    )
    def test_rejects_invalid_inputs(
        self,
        tmp_path: Path,
        table: str,
        count: int | None,
        match: str,
    ) -> None:
        """Test request validation."""
        db_path = tmp_path / "invalid-inputs.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE rates(time TEXT, close REAL)")
            with pytest.raises(ValueError, match=match):
                load_rate_data_from_connection(conn, table, count=count)

    @pytest.mark.parametrize(
        ("ddl", "match"),
        [
            ("CREATE TABLE rates(time TEXT, close REAL)", "contains no rows"),
            ("CREATE TABLE rates(close REAL)", "time column"),
            ("CREATE TABLE rates(time TEXT, open REAL)", "close, or both ask and bid"),
        ],
    )
    def test_rejects_invalid_tables(
        self,
        tmp_path: Path,
        ddl: str,
        match: str,
    ) -> None:
        """Test missing table, empty table, and invalid schemas."""
        db_path = tmp_path / "invalid-tables.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(ddl)
            with pytest.raises(ValueError, match=match):
                load_rate_data_from_connection(conn, "rates")
            with pytest.raises(ValueError, match="not found"):
                load_rate_data_from_connection(conn, "missing")

    def test_rejects_invalid_timestamp(self, tmp_path: Path) -> None:
        """Test unparsable timestamps fail clearly."""
        db_path = tmp_path / "invalid-time.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE rates(time TEXT, close REAL)")
            conn.execute("INSERT INTO rates(time, close) VALUES (?, ?)", ("bad", 1.0))
            with pytest.raises(ValueError, match="unparsable time"):
                load_rate_data_from_connection(conn, "rates")

    def test_loads_numeric_mt5_epoch_seconds(self, tmp_path: Path) -> None:
        """Test MT5-native integer timestamps are parsed as epoch seconds."""
        db_path = tmp_path / "epoch-rates.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE rates(time INTEGER, close REAL)")
            conn.execute(
                "INSERT INTO rates(time, close) VALUES (?, ?)",
                (1_704_067_200, 1.0),
            )
            frame = load_rate_data_from_connection(conn, "rates")
        assert frame.index[0] == pd.Timestamp("2024-01-01")
        assert list(frame["close"]) == [1.0]


class TestResolveHistorySettings:
    """Tests for history dataset and timeframe resolution."""

    @pytest.mark.parametrize(
        ("datasets", "expected"),
        [
            (
                None,
                {
                    Dataset.rates,
                    Dataset.history_orders,
                    Dataset.history_deals,
                },
            ),
            (
                cast("set[Dataset]", set()),
                cast("set[Dataset]", set()),
            ),
            ({Dataset.ticks}, {Dataset.ticks}),
            (set(Dataset), set(Dataset)),
        ],
        ids=["defaults", "empty", "explicit-ticks", "all-datasets"],
    )
    def test_resolve_history_datasets(
        self,
        datasets: set[Dataset] | None,
        expected: set[Dataset],
    ) -> None:
        """Test dataset resolution defaults and explicit selections."""
        resolved = resolve_history_datasets(datasets)
        assert resolved == expected
        if datasets is None:
            assert resolved == set(DEFAULT_HISTORY_DATASETS)
            assert Dataset.ticks not in resolved

    def test_resolve_history_timeframes_defaults(self) -> None:
        """Test default timeframes include all fixed MT5 values."""
        resolved = resolve_history_timeframes(None)
        assert len(resolved) == len(DEFAULT_HISTORY_TIMEFRAMES)
        assert not any(
            name.startswith("TIMEFRAME_") for name in DEFAULT_HISTORY_TIMEFRAMES
        )
        assert 1 in resolved
        assert TIMEFRAME_MAP["H1"] in resolved

    def test_resolve_history_timeframes_deduplicates_aliases(self) -> None:
        """Test duplicate aliases for the same timeframe are deduplicated."""
        assert resolve_history_timeframes(["M1", "1", "H1"]) == [1, TIMEFRAME_MAP["H1"]]

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [("ALL", -1), (2, 2)],
        ids=["named-all", "numeric"],
    )
    def test_resolve_history_tick_flags(
        self,
        flags: str | int,
        expected: int,
    ) -> None:
        """Test tick flag resolution accepts named and numeric flags."""
        assert resolve_history_tick_flags(flags) == expected

    @pytest.mark.parametrize(
        ("timeframe", "expected"),
        [(999, "999"), (1, "M1")],
        ids=["unknown-integer-fallback", "known-m1"],
    )
    def test_resolve_granularity_name_falls_back_to_integer(
        self,
        timeframe: int,
        expected: str,
    ) -> None:
        """Test unknown timeframe constants fall back to integer text."""
        assert resolve_granularity_name(timeframe) == expected

    def test_resolve_granularity_name_strips_official_prefix(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test official pdmt5 timeframe names are normalized to short aliases."""
        mocker.patch(
            "mt5cli.history._get_timeframe_name",
            return_value="TIMEFRAME_H1",
        )
        assert resolve_granularity_name(16385) == "H1"


class TestDropFormingRateBar:
    """Tests for drop_forming_rate_bar."""

    def test_drops_still_forming_last_bar(self) -> None:
        """Test the still-forming last bar is removed."""
        df_rate = pd.DataFrame(
            {"time": [1, 2, 3], "close": [1.1, 1.2, 1.3]},
            index=pd.Index(["a", "b", "c"], name="idx"),
        )

        result = drop_forming_rate_bar(df_rate)

        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame(
                {"time": [1, 2], "close": [1.1, 1.2]},
                index=pd.Index(["a", "b"], name="idx"),
            ),
        )
        assert df_rate.shape == (3, 2)

    @pytest.mark.parametrize(
        "df_rate",
        [
            pytest.param(
                pd.DataFrame(columns=["time", "close"]),
                id="empty-input",
            ),
            pytest.param(
                pd.DataFrame({"time": [1], "close": [1.1]}),
                id="single-forming-bar",
            ),
        ],
    )
    def test_returns_empty_frame_for_empty_result_cases(
        self,
        df_rate: pd.DataFrame,
    ) -> None:
        """Test empty and single-bar frames stay empty after dropping."""
        result = drop_forming_rate_bar(df_rate)

        assert result.empty
        assert list(result.columns) == ["time", "close"]


class TestParseSqliteTimestamp:
    """Tests for parse_sqlite_timestamp."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(None, None, id="none"),
            pytest.param(
                1_704_067_200,
                datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                id="mt5-epoch-seconds",
            ),
            pytest.param(
                datetime.fromisoformat("2024-01-01T00:00:00"),
                datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                id="naive-datetime",
            ),
            pytest.param(
                "Jan 1 2024",
                datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                id="pandas-string",
            ),
            pytest.param("not-a-datetime", None, id="invalid-string"),
            pytest.param(object(), None, id="unsupported-object"),
        ],
    )
    def test_parses_various_inputs(
        self,
        value: object,
        expected: datetime | None,
    ) -> None:
        """Test ISO, pandas-compatible, numeric, and datetime values."""
        assert parse_sqlite_timestamp(value) == expected

    def test_serialize_invalid_timestamp_returns_none(self) -> None:
        """Test invalid SQLite timestamp serialization returns None."""
        assert history._serialize_sqlite_timestamp(object()) is None  # type: ignore[reportPrivateUsage]

    def test_require_serialized_timestamp_raises_for_invalid(self) -> None:
        """Test invalid SQLite timestamp boundaries fail fast."""
        with pytest.raises(ValueError, match="Invalid SQLite timestamp boundary"):
            history._require_serialized_sqlite_timestamp(  # type: ignore[reportPrivateUsage]
                object()
            )


class TestIncrementalStart:
    """Tests for get_incremental_start_datetime."""

    def test_uses_max_time_scoped_by_symbol_and_timeframe(
        self,
        tmp_path: Path,
    ) -> None:
        """Test rates increment is scoped by symbol and timeframe."""
        db_path = tmp_path / "scoped-rates.db"
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, open) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-02T00:00:00", 1.0),
                    ("EURUSD", 16385, "2024-01-03T00:00:00", 1.1),
                    ("GBPUSD", 1, "2024-01-04T00:00:00", 1.2),
                ],
            )
            assert get_incremental_start_datetime(
                conn,
                Dataset.rates,
                symbol="EURUSD",
                timeframe=1,
                fallback_start=fallback,
            ) == datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
            assert get_incremental_start_datetime(
                conn,
                Dataset.rates,
                symbol="EURUSD",
                timeframe=16385,
                fallback_start=fallback,
            ) == datetime(2024, 1, 3, tzinfo=UTC).replace(tzinfo=None)

    def test_load_incremental_start_datetimes_batches_rates(
        self, tmp_path: Path
    ) -> None:
        """Test grouped rates resume query returns all symbol/timeframe pairs."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / "batch-rates.db") as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, open) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-02T00:00:00", 1.0),
                    ("GBPUSD", 1, "2024-01-03T00:00:00", 1.1),
                ],
            )
            starts = load_incremental_start_datetimes(
                conn,
                Dataset.rates,
                symbols=["EURUSD", "GBPUSD"],
                timeframes=[1],
                fallback_start=fallback,
            )
        assert starts["EURUSD", 1] == datetime(2024, 1, 2, tzinfo=UTC).replace(
            tzinfo=None
        )
        assert starts["GBPUSD", 1] == datetime(2024, 1, 3, tzinfo=UTC).replace(
            tzinfo=None
        )

    @pytest.mark.parametrize(
        (
            "dataset",
            "db_name",
            "ddl",
            "insert_sql",
            "insert_rows",
            "symbols",
            "timeframes",
            "expected_starts",
        ),
        [
            pytest.param(
                Dataset.rates,
                "duplicate-rate-groups",
                (
                    "CREATE TABLE rates("
                    " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)"
                ),
                (
                    "INSERT INTO rates(symbol, timeframe, time, open)"
                    " VALUES (?, ?, ?, ?)"
                ),
                [
                    ("EURUSD", 1, "2024-01-03T00:00:00", 1.2),
                    ("EURUSD", 1, "2024-01-02T00:00:00", 1.1),
                    ("GBPUSD", 1, "2024-01-04T00:00:00", 1.3),
                ],
                ["EURUSD", "GBPUSD"],
                [1],
                {
                    ("EURUSD", 1): datetime(2024, 1, 3, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                    ("GBPUSD", 1): datetime(2024, 1, 4, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                },
                id="rates-latest-row-per-group",
            ),
            pytest.param(
                Dataset.ticks,
                "duplicate-symbol-groups",
                "CREATE TABLE ticks(symbol TEXT, time TEXT)",
                "INSERT INTO ticks(symbol, time) VALUES (?, ?)",
                [
                    ("EURUSD", "2024-01-03T00:00:00"),
                    ("EURUSD", "2024-01-02T00:00:00"),
                    ("GBPUSD", "2024-01-04T00:00:00"),
                ],
                ["EURUSD", "GBPUSD"],
                None,
                {
                    ("EURUSD", None): datetime(2024, 1, 3, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                    ("GBPUSD", None): datetime(2024, 1, 4, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                },
                id="ticks-latest-row-per-symbol",
            ),
        ],
    )
    def test_load_incremental_start_prefers_latest_row_per_group(
        self,
        tmp_path: Path,
        dataset: Dataset,
        db_name: str,
        ddl: str,
        insert_sql: str,
        insert_rows: list[tuple[object, ...]],
        symbols: list[str],
        timeframes: list[int] | None,
        expected_starts: dict[tuple[str, int | None], datetime],
    ) -> None:
        """Test incremental resume keeps only the latest row per scoped group."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / f"{db_name}.db") as conn:
            conn.execute(ddl)
            conn.executemany(insert_sql, insert_rows)
            starts = load_incremental_start_datetimes(
                conn,
                dataset,
                symbols=symbols,
                timeframes=timeframes,
                fallback_start=fallback,
            )
        for key, expected in expected_starts.items():
            assert starts[key] == expected

    @pytest.mark.parametrize(
        (
            "dataset",
            "db_name",
            "ddl",
            "insert_sql",
            "insert_rows",
            "symbols",
            "timeframes",
            "expected_starts",
        ),
        [
            pytest.param(
                Dataset.rates,
                "numeric-rate-cursor",
                "CREATE TABLE rates( symbol TEXT, timeframe INTEGER, time, open REAL)",
                (
                    "INSERT INTO rates(symbol, timeframe, time, open)"
                    " VALUES (?, ?, ?, ?)"
                ),
                [
                    ("EURUSD", 1, 1704153600, 1.2),
                    ("GBPUSD", 1, "2024-01-03T00:00:00", 1.3),
                ],
                ["EURUSD", "GBPUSD"],
                [1],
                {
                    ("EURUSD", 1): datetime(2024, 1, 2, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                    ("GBPUSD", 1): datetime(2024, 1, 3, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                },
                id="rates-numeric-cursor",
            ),
            pytest.param(
                Dataset.ticks,
                "numeric-symbol-cursor",
                "CREATE TABLE ticks(symbol TEXT, time)",
                "INSERT INTO ticks(symbol, time) VALUES (?, ?)",
                [
                    ("EURUSD", 1704153600),
                    ("GBPUSD", "2024-01-03T00:00:00"),
                ],
                ["EURUSD", "GBPUSD"],
                None,
                {
                    ("EURUSD", None): datetime(2024, 1, 2, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                    ("GBPUSD", None): datetime(2024, 1, 3, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                },
                id="ticks-numeric-cursor",
            ),
        ],
    )
    def test_load_incremental_start_accepts_numeric_cursor(
        self,
        tmp_path: Path,
        dataset: Dataset,
        db_name: str,
        ddl: str,
        insert_sql: str,
        insert_rows: list[tuple[object, ...]],
        symbols: list[str],
        timeframes: list[int] | None,
        expected_starts: dict[tuple[str, int | None], datetime],
    ) -> None:
        """Test incremental resume preserves numeric epoch cursors."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / f"{db_name}.db") as conn:
            conn.execute(ddl)
            conn.executemany(insert_sql, insert_rows)
            starts = load_incremental_start_datetimes(
                conn,
                dataset,
                symbols=symbols,
                timeframes=timeframes,
                fallback_start=fallback,
            )
        for key, expected in expected_starts.items():
            assert starts[key] == expected

    @pytest.mark.parametrize(
        ("ddl", "missing_col"),
        [
            ("CREATE TABLE rates(symbol TEXT, time TEXT, open REAL)", "timeframe"),
            ("CREATE TABLE rates(timeframe INTEGER, time TEXT, open REAL)", "symbol"),
            ("CREATE TABLE rates(symbol TEXT, timeframe INTEGER, open REAL)", "time"),
        ],
    )
    def test_load_incremental_start_datetimes_requires_column(
        self,
        tmp_path: Path,
        ddl: str,
        missing_col: str,
    ) -> None:
        """Test rates tables missing a required column fail fast."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / f"rates-no-{missing_col}.db") as conn:
            conn.execute(ddl)
            with pytest.raises(ValueError, match=f"missing: {missing_col}") as exc_info:
                load_incremental_start_datetimes(
                    conn,
                    Dataset.rates,
                    symbols=["EURUSD"],
                    timeframes=[1],
                    fallback_start=fallback,
                )
            assert missing_col in str(exc_info.value)

    def test_load_incremental_start_datetimes_rejects_unrelated_rates_columns(
        self,
        tmp_path: Path,
    ) -> None:
        """Test rates tables with only unrelated columns fail fast."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / "rates-open-only.db") as conn:
            conn.execute("CREATE TABLE rates(open REAL)")
            with pytest.raises(ValueError, match="missing:") as exc_info:
                load_incremental_start_datetimes(
                    conn,
                    Dataset.rates,
                    symbols=["EURUSD"],
                    timeframes=[1],
                    fallback_start=fallback,
                )
            message = str(exc_info.value)
            assert "symbol" in message
            assert "timeframe" in message
            assert "time" in message

    @pytest.mark.parametrize(
        (
            "dataset",
            "ddl",
            "insert_sql",
            "insert_args",
            "symbols",
            "timeframes",
            "start_key",
            "db_name",
        ),
        [
            pytest.param(
                Dataset.ticks,
                "CREATE TABLE ticks(symbol TEXT, time TEXT)",
                "INSERT INTO ticks(symbol, time) VALUES (?, ?)",
                ("EURUSD", "not-a-datetime"),
                ["EURUSD"],
                None,
                ("EURUSD", None),
                "bad-max-time",
                id="ticks-unparseable-max-time",
            ),
            pytest.param(
                Dataset.rates,
                (
                    "CREATE TABLE rates("
                    " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)"
                ),
                (
                    "INSERT INTO rates(symbol, timeframe, time, open)"
                    " VALUES (?, ?, ?, ?)"
                ),
                ("EURUSD", 1, "not-a-datetime", 1.0),
                ["EURUSD"],
                [1],
                ("EURUSD", 1),
                "bad-rates-max-time",
                id="rates-unparseable-max-time",
            ),
        ],
    )
    def test_load_incremental_start_skips_unparseable_max_time(
        self,
        tmp_path: Path,
        dataset: Dataset,
        ddl: str,
        insert_sql: str,
        insert_args: tuple[object, ...],
        symbols: list[str],
        timeframes: list[int] | None,
        start_key: tuple[str, int | None],
        db_name: str,
    ) -> None:
        """Test grouped resume ignores rows whose MAX(time) cannot be parsed."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / f"{db_name}.db") as conn:
            conn.execute(ddl)
            conn.execute(insert_sql, insert_args)
            starts = load_incremental_start_datetimes(
                conn,
                dataset,
                symbols=symbols,
                timeframes=timeframes,
                fallback_start=fallback,
            )
        assert starts[start_key] == fallback

    def test_load_incremental_start_uses_table_max_without_symbol_column(
        self,
        tmp_path: Path,
    ) -> None:
        """Test grouped resume uses table-wide MAX(time) without symbol column."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / "batch-no-symbol.db") as conn:
            conn.execute("CREATE TABLE ticks(time TEXT)")
            conn.execute(
                "INSERT INTO ticks(time) VALUES (?)",
                ("2024-01-02T00:00:00",),
            )
            starts = load_incremental_start_datetimes(
                conn,
                Dataset.ticks,
                symbols=["EURUSD", "GBPUSD"],
                fallback_start=fallback,
            )
        expected = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
        assert starts["EURUSD", None] == expected
        assert starts["GBPUSD", None] == expected

    @pytest.mark.parametrize(
        (
            "db_name",
            "ddl",
            "table_name",
            "insert_sql",
            "insert_args",
            "loader_name",
            "loader_kwargs",
            "expected_key",
        ),
        [
            pytest.param(
                "grouped-rate-parse-none",
                (
                    "CREATE TABLE rates("
                    " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)"
                ),
                "rates",
                (
                    "INSERT INTO rates(symbol, timeframe, time, open)"
                    " VALUES (?, ?, ?, ?)"
                ),
                ("EURUSD", 1, "2024-01-03T00:00:00", 1.2),
                "_load_grouped_rate_start_datetimes",
                {"symbols": ["EURUSD"], "timeframes": [1]},
                ("EURUSD", 1),
                id="grouped-rate-parse-failure-fallback",
            ),
            pytest.param(
                "symbol-start-parse-none",
                "CREATE TABLE ticks(symbol TEXT, time TEXT)",
                "ticks",
                "INSERT INTO ticks(symbol, time) VALUES (?, ?)",
                ("EURUSD", "2024-01-03T00:00:00"),
                "_load_symbol_start_datetimes",
                {"symbols": ["EURUSD"]},
                ("EURUSD", None),
                id="symbol-scoped-parse-failure-fallback",
            ),
        ],
    )
    def test_incremental_start_skips_rows_when_timestamp_parse_fails(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        db_name: str,
        ddl: str,
        table_name: str,
        insert_sql: str,
        insert_args: tuple[object, ...],
        loader_name: str,
        loader_kwargs: dict[str, object],
        expected_key: tuple[str, int | None],
    ) -> None:
        """Test incremental starts fall back when a parsed row returns None."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / f"{db_name}.db") as conn:
            conn.execute(ddl)
            conn.execute(insert_sql, insert_args)
            mocker.patch.object(history, "parse_sqlite_timestamp", return_value=None)
            loader = getattr(history, loader_name)
            starts = loader(
                conn,
                table_name,
                fallback_start=fallback,
                **loader_kwargs,
            )
        assert starts[expected_key] == fallback

    @pytest.mark.parametrize(
        (
            "db_name",
            "ddl",
            "table_name",
            "insert_sql",
            "insert_rows",
            "loader_name",
            "loader_kwargs",
            "expected_starts",
        ),
        [
            pytest.param(
                "grouped-rate-mixed-formats",
                "CREATE TABLE rates( symbol TEXT, timeframe INTEGER, time, open REAL)",
                "rates",
                (
                    "INSERT INTO rates(symbol, timeframe, time, open)"
                    " VALUES (?, ?, ?, ?)"
                ),
                [
                    ("EURUSD", 1, "2024-01-02 00:00:00", 1.0),
                    ("EURUSD", 1, "2024-01-03T00:00:00", 1.1),
                    ("EURUSD", 1, 1704240000, 1.2),
                    ("GBPUSD", 1, 1704153600, 1.3),
                    ("GBPUSD", 1, "2024-01-04T00:00:00", 1.4),
                ],
                "_load_grouped_rate_start_datetimes",
                {"symbols": ["EURUSD", "GBPUSD"], "timeframes": [1]},
                {
                    ("EURUSD", 1): datetime(2024, 1, 3, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                    ("GBPUSD", 1): datetime(2024, 1, 4, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                },
                id="grouped-rate-mixed-timestamps",
            ),
            pytest.param(
                "symbol-mixed-formats",
                "CREATE TABLE ticks(symbol TEXT, time)",
                "ticks",
                "INSERT INTO ticks(symbol, time) VALUES (?, ?)",
                [
                    ("EURUSD", "2024-01-02 00:00:00"),
                    ("EURUSD", "2024-01-03T00:00:00"),
                    ("EURUSD", 1704240000),
                    ("GBPUSD", 1704153600),
                    ("GBPUSD", "2024-01-04T00:00:00"),
                ],
                "_load_symbol_start_datetimes",
                {"symbols": ["EURUSD", "GBPUSD"]},
                {
                    ("EURUSD", None): datetime(2024, 1, 3, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                    ("GBPUSD", None): datetime(2024, 1, 4, tzinfo=UTC).replace(
                        tzinfo=None
                    ),
                },
                id="symbol-scoped-mixed-timestamps",
            ),
        ],
    )
    def test_incremental_start_preserves_mixed_timestamp_semantics(
        self,
        tmp_path: Path,
        db_name: str,
        ddl: str,
        table_name: str,
        insert_sql: str,
        insert_rows: list[tuple[object, ...]],
        loader_name: str,
        loader_kwargs: dict[str, object],
        expected_starts: dict[tuple[str, int | None], datetime],
    ) -> None:
        """Test incremental resume preserves the selected row timestamp semantics."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / f"{db_name}.db") as conn:
            conn.execute(ddl)
            conn.executemany(insert_sql, insert_rows)
            loader = getattr(history, loader_name)
            starts = loader(
                conn,
                table_name,
                fallback_start=fallback,
                **loader_kwargs,
            )
        for key, expected in expected_starts.items():
            assert starts[key] == expected

    @pytest.mark.parametrize(
        (
            "db_name",
            "ddl",
            "table_name",
            "insert_sql",
            "insert_rows",
            "loader_name",
            "loader_kwargs",
            "expected",
        ),
        [
            pytest.param(
                "grouped-rate-aggregation",
                (
                    "CREATE TABLE rates("
                    " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)"
                ),
                "rates",
                (
                    "INSERT INTO rates(symbol, timeframe, time, open)"
                    " VALUES (?, ?, ?, ?)"
                ),
                [
                    ("EURUSD", 1, f"2024-01-01T{hour:02d}:00:00", float(hour))
                    for hour in range(24)
                ],
                "_load_grouped_rate_start_datetimes",
                {"symbols": ["EURUSD"], "timeframes": [1]},
                (
                    "GROUP BY symbol, timeframe",
                    ("EURUSD", 1),
                    datetime(2024, 1, 1, 23, tzinfo=UTC).replace(tzinfo=None),
                ),
                id="grouped-rate-sqlite-aggregation",
            ),
            pytest.param(
                "symbol-aggregation",
                "CREATE TABLE ticks(symbol TEXT, time TEXT)",
                "ticks",
                "INSERT INTO ticks(symbol, time) VALUES (?, ?)",
                [("EURUSD", f"2024-01-01T{hour:02d}:00:00") for hour in range(24)],
                "_load_symbol_start_datetimes",
                {"symbols": ["EURUSD"]},
                (
                    "GROUP BY symbol",
                    ("EURUSD", None),
                    datetime(2024, 1, 1, 23, tzinfo=UTC).replace(tzinfo=None),
                ),
                id="symbol-scoped-sqlite-aggregation",
            ),
        ],
    )
    def test_incremental_start_query_selects_latest_row_in_sqlite(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        db_name: str,
        ddl: str,
        table_name: str,
        insert_sql: str,
        insert_rows: list[tuple[object, ...]],
        loader_name: str,
        loader_kwargs: dict[str, object],
        expected: tuple[str, tuple[str, int | None], datetime],
    ) -> None:
        """Test incremental resume selects the latest scoped row in SQLite."""
        expected_group_by, expected_key, expected_start = expected
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / f"{db_name}.db") as conn:
            conn.execute(ddl)
            conn.executemany(insert_sql, insert_rows)
            execute_spy = mocker.spy(conn, "execute")
            loader = getattr(history, loader_name)
            starts = loader(
                conn,
                table_name,
                fallback_start=fallback,
                **loader_kwargs,
            )
            query = str(execute_spy.call_args[0][0])
        assert "MAX(" not in query
        assert expected_group_by not in query
        assert "ORDER BY" in query
        assert "LIMIT 1" in query
        assert starts[expected_key] == expected_start


class TestDeduplication:
    """Tests for SQLite deduplication helpers."""

    def test_append_dedup_keeps_latest_rowid(self, tmp_path: Path) -> None:
        """Test deduplication keeps the latest ROWID for stable keys."""
        db_path = tmp_path / "dedup.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, open) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 9.9),
                ],
            )
            deduplicate_history_tables(
                conn,
                {Dataset.rates: {"symbol", "timeframe", "time", "open"}},
                {Dataset.rates},
            )
            assert conn.execute("SELECT COUNT(*) FROM rates").fetchone() == (1,)
            assert conn.execute("SELECT open FROM rates").fetchone() == (9.9,)

    @pytest.mark.parametrize(
        ("table", "columns", "match"),
        [
            ("bad table", ["id"], "Invalid table name"),
            ("rates", ["bad column"], "Invalid column names"),
        ],
        ids=["invalid-table", "invalid-columns"],
    )
    def test_drop_duplicates_rejects_invalid_identifiers(
        self,
        table: str,
        columns: list[str],
        match: str,
    ) -> None:
        """Test invalid table or column names raise ValueError."""
        cursor = sqlite3.connect(":memory:").cursor()
        with pytest.raises(ValueError, match=match):
            drop_duplicates_in_table(cursor, table, columns)

    @pytest.mark.parametrize(
        ("dataset", "table_sql", "insert_sql", "rows", "columns"),
        [
            (
                Dataset.ticks,
                (
                    "CREATE TABLE ticks("
                    " symbol TEXT, time_msc INTEGER, time TEXT, bid REAL)"
                ),
                "INSERT INTO ticks(symbol, time_msc, time, bid) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 9.9),
                ],
                {"symbol", "time_msc", "time", "bid"},
            ),
            (
                Dataset.history_orders,
                (
                    "CREATE TABLE history_orders("
                    " ticket INTEGER, symbol TEXT, time TEXT, type INTEGER)"
                ),
                (
                    "INSERT INTO history_orders(ticket, symbol, time, type)"
                    " VALUES (?, ?, ?, ?)"
                ),
                [
                    (1, "EURUSD", "2024-01-01T00:00:00+00:00", 0),
                    (1, "EURUSD", "2024-01-01T00:00:00+00:00", 1),
                ],
                {"ticket", "symbol", "time", "type"},
            ),
            (
                Dataset.history_deals,
                (
                    "CREATE TABLE history_deals("
                    " ticket INTEGER, symbol TEXT, time TEXT, type INTEGER,"
                    " entry INTEGER)"
                ),
                (
                    "INSERT INTO history_deals(ticket, symbol, time, type, entry)"
                    " VALUES (?, ?, ?, ?, ?)"
                ),
                [
                    (1, "EURUSD", "2024-01-01T00:00:00+00:00", 0, 0),
                    (1, "EURUSD", "2024-01-01T00:00:00+00:00", 0, 1),
                ],
                {"ticket", "symbol", "time", "type", "entry"},
            ),
        ],
    )
    def test_deduplicates_non_rate_datasets_by_stable_keys(
        self,
        tmp_path: Path,
        dataset: Dataset,
        table_sql: str,
        insert_sql: str,
        rows: list[tuple[object, ...]],
        columns: set[str],
    ) -> None:
        """Test deduplication keys for ticks, orders, and deals."""
        with sqlite3.connect(tmp_path / f"{dataset.value}-dedup.db") as conn:
            conn.execute(table_sql)
            conn.executemany(insert_sql, rows)
            deduplicate_history_tables(conn, {dataset: columns}, {dataset})
            assert conn.execute(
                f"SELECT COUNT(*) FROM {dataset.table_name}",  # noqa: S608
            ).fetchone() == (1,)

    def test_scoped_dedup_preserves_older_rows(self, tmp_path: Path) -> None:
        """Test scoped deduplication only rewrites the appended boundary."""
        boundary = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / "scoped-dedup.db") as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, open) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", 1, "2024-01-02T00:00:00+00:00", 2.0),
                    ("EURUSD", 1, "2024-01-02T00:00:00+00:00", 9.9),
                ],
            )
            deduplicate_history_tables(
                conn,
                {Dataset.rates: {"symbol", "timeframe", "time", "open"}},
                {Dataset.rates},
                {
                    Dataset.rates: [
                        DedupScope(
                            "symbol = ? AND timeframe = ? AND time >= ?",
                            ("EURUSD", 1, boundary),
                            frozenset({"symbol", "timeframe", "time"}),
                        ),
                    ],
                },
            )
            rows = conn.execute(
                "SELECT time, open FROM rates ORDER BY time, open",
            ).fetchall()
        assert rows == [
            ("2024-01-01T00:00:00+00:00", 1.0),
            ("2024-01-02T00:00:00+00:00", 9.9),
        ]

    def test_scoped_dedup_matches_numeric_and_iso_times(self, tmp_path: Path) -> None:
        """Test scoped dedup collapses numeric epoch rows with canonical ISO writes."""
        boundary = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
        dedup_scopes: dict[Dataset, list[DedupScope]] = {}
        time_expr = history._sqlite_normalized_time_expression(  # type: ignore[reportPrivateUsage]
            "time"
        )
        with sqlite3.connect(tmp_path / "numeric-time-dedup.db") as conn:
            conn.execute(
                "CREATE TABLE rates( symbol TEXT, timeframe INTEGER, time, open REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, open) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", 1, 1704153600, 2.0),
                    ("EURUSD", 1, "2024-01-02T00:00:00+00:00", 9.9),
                ],
            )
            history._record_dedup_scope(  # type: ignore[reportPrivateUsage]
                dedup_scopes,
                Dataset.rates,
                DedupScope(
                    f"symbol = ? AND timeframe = ? AND {time_expr} >= ?",
                    ("EURUSD", 1, boundary.isoformat()),
                    frozenset({"symbol", "timeframe", "time"}),
                ),
            )
            deduplicate_history_tables(
                conn,
                {Dataset.rates: {"symbol", "timeframe", "time", "open"}},
                {Dataset.rates},
                dedup_scopes,
            )
            rows = conn.execute(
                "SELECT time, typeof(time), open FROM rates ORDER BY ROWID",
            ).fetchall()
        assert rows == [
            ("2024-01-01T00:00:00+00:00", "text", 1.0),
            ("2024-01-02T00:00:00+00:00", "text", 9.9),
        ]

    def test_unusable_scope_falls_back_to_table_dedup(self, tmp_path: Path) -> None:
        """Test scopes with missing columns do not break stable-key dedup."""
        boundary = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / "orders-without-time.db") as conn:
            conn.execute(
                "CREATE TABLE history_orders("
                " ticket INTEGER, symbol TEXT, time_setup TEXT, type INTEGER)",
            )
            conn.executemany(
                "INSERT INTO history_orders(ticket, symbol, time_setup, type)"
                " VALUES (?, ?, ?, ?)",
                [
                    (1, "EURUSD", "2024-01-01T00:00:00+00:00", 0),
                    (1, "EURUSD", "2024-01-01T00:00:01+00:00", 1),
                ],
            )
            deduplicate_history_tables(
                conn,
                {Dataset.history_orders: {"ticket", "symbol", "time_setup", "type"}},
                {Dataset.history_orders},
                {
                    Dataset.history_orders: [
                        DedupScope(
                            "symbol = ? AND time >= ?",
                            ("EURUSD", boundary),
                            frozenset({"symbol", "time"}),
                        ),
                    ],
                },
            )
            rows = conn.execute(
                "SELECT ticket, time_setup, type FROM history_orders",
            ).fetchall()
        assert rows == [(1, "2024-01-01T00:00:01+00:00", 1)]

    def test_partially_unusable_scopes_only_run_usable_scopes(
        self,
        tmp_path: Path,
    ) -> None:
        """Test mixed scope filtering skips only scopes with missing columns."""
        boundary = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / "partial-scope-filter.db") as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, open) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-02T00:00:00+00:00", 2.0),
                    ("EURUSD", 1, "2024-01-02T00:00:00+00:00", 9.9),
                    ("USDJPY", 1, "2024-01-02T00:00:00+00:00", 100.0),
                    ("USDJPY", 1, "2024-01-02T00:00:00+00:00", 101.0),
                ],
            )
            deduplicate_history_tables(
                conn,
                {Dataset.rates: {"symbol", "timeframe", "time", "open"}},
                {Dataset.rates},
                {
                    Dataset.rates: [
                        DedupScope(
                            "symbol = ? AND timeframe = ? AND time >= ?",
                            ("EURUSD", 1, boundary),
                            frozenset({"symbol", "timeframe", "time"}),
                        ),
                        DedupScope(
                            "symbol = ? AND timeframe = ? AND broker = ?",
                            ("USDJPY", 1, "demo"),
                            frozenset({"symbol", "timeframe", "broker"}),
                        ),
                    ],
                },
            )
            rows = conn.execute(
                "SELECT symbol, open FROM rates ORDER BY symbol, open",
            ).fetchall()
        assert rows == [
            ("EURUSD", 9.9),
            ("USDJPY", 100.0),
            ("USDJPY", 101.0),
        ]


class TestRateCompatibilityViews:
    """Tests for rate compatibility view creation."""


class TestDerivedViews:
    """Tests for cash_events and positions_reconstructed views."""

    def test_creates_views_when_columns_present(self, tmp_path: Path) -> None:
        """Test cash_events and positions_reconstructed views are created."""
        db_path = tmp_path / "derived-views.db"
        columns = {
            "ticket",
            "position_id",
            "symbol",
            "time",
            "type",
            "entry",
            "volume",
            "price",
            "profit",
        }
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE history_deals("
                " ticket INTEGER, position_id INTEGER, symbol TEXT, time INTEGER,"
                " type INTEGER, entry INTEGER, volume REAL, price REAL, profit REAL)",
            )
            conn.executemany(
                "INSERT INTO history_deals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (1, 100, "EURUSD", 1, 0, 0, 1.0, 1.1, 0.0),
                    (2, 100, "EURUSD", 2, 0, 1, 1.0, 1.2, 5.0),
                    (3, 0, "", 3, 2, 0, 0.0, 0.0, 10.0),
                ],
            )
            assert create_cash_events_view(conn, columns)
            assert create_positions_reconstructed_view(conn, columns)
            assert conn.execute("SELECT COUNT(*) FROM cash_events").fetchone() == (1,)
            assert conn.execute(
                "SELECT position_id FROM positions_reconstructed",
            ).fetchall() == [(100,)]


class TestFilterTradeHistoryFrame:
    """Tests for filter_trade_history_frame."""

    def test_includes_account_events_when_requested(self) -> None:
        """Test account events are kept alongside selected symbols."""
        frame = pd.DataFrame({
            "symbol": ["EURUSD", None, "OTHER"],
            "type": [0, 2, 0],
        })
        filtered = filter_trade_history_frame(
            frame,
            ["EURUSD"],
            include_account_events=True,
        )
        assert filtered["symbol"].tolist()[0] == "EURUSD"
        assert pd.isna(filtered["symbol"].tolist()[1])


class TestIncrementalHistoryDealsHelpers:
    """Tests for incremental history_deals helper functions."""

    @pytest.mark.parametrize(
        ("ddl", "insert_sql", "rows", "expected"),
        [
            pytest.param(
                (
                    "CREATE TABLE history_deals("
                    " ticket INTEGER, symbol TEXT, time TEXT, type INTEGER)"
                ),
                (
                    "INSERT INTO history_deals(ticket, symbol, time, type)"
                    " VALUES (?, ?, ?, ?)"
                ),
                [
                    (1, "EURUSD", "2024-01-05T00:00:00", 0),
                    (2, "", "2024-01-08T00:00:00", 2),
                ],
                datetime(2024, 1, 8, tzinfo=UTC).replace(tzinfo=None),
                id="uses-type-column",
            ),
            pytest.param(
                ("CREATE TABLE history_deals( ticket INTEGER, symbol TEXT, time TEXT)"),
                "INSERT INTO history_deals(ticket, symbol, time) VALUES (?, ?, ?)",
                [
                    (1, "EURUSD", "2024-01-05T00:00:00"),
                    (2, "", "2024-01-07T00:00:00"),
                ],
                datetime(2024, 1, 7, tzinfo=UTC).replace(tzinfo=None),
                id="falls-back-to-empty-symbol",
            ),
            pytest.param(
                "CREATE TABLE history_deals(ticket INTEGER, time TEXT)",
                "INSERT INTO history_deals(ticket, time) VALUES (?, ?)",
                [(1, "2024-01-05T00:00:00")],
                datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                id="without-identifying-columns",
            ),
        ],
    )
    def test_get_history_deals_account_event_start_datetime(
        self,
        tmp_path: Path,
        ddl: str,
        insert_sql: str,
        rows: list[tuple[object, ...]],
        expected: datetime,
    ) -> None:
        """Test account-event start resolution across identifying-column variants."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(tmp_path / "account-start.db") as conn:
            conn.execute(ddl)
            conn.executemany(insert_sql, rows)
            assert (
                get_history_deals_account_event_start_datetime(
                    conn,
                    fallback_start=fallback,
                )
                == expected
            )

    def test_filter_incremental_history_deals_frame(self) -> None:
        """Test incremental deal filtering applies per-symbol and account starts."""
        frame = pd.DataFrame({
            "ticket": [1, 2, 3, 4, 5],
            "symbol": ["EURUSD", "EURUSD", "GBPUSD", "OTHER", ""],
            "time": [
                "2024-01-05T00:00:00",
                "2024-01-11T00:00:00",
                "2024-01-02T00:00:00",
                "2024-01-02T00:00:00",
                "2024-01-03T00:00:00",
            ],
            "type": [0, 0, 0, 0, 2],
        })
        start_by_symbol = {
            "EURUSD": datetime(2024, 1, 10, tzinfo=UTC).replace(tzinfo=None),
            "GBPUSD": datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
        }
        filtered = filter_incremental_history_deals_frame(
            frame,
            ["EURUSD", "GBPUSD"],
            start_by_symbol,
            datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None),
        )
        assert filtered["ticket"].tolist() == [2, 3, 5]

    def test_filter_incremental_history_deals_rejects_aware_frame_times(
        self,
    ) -> None:
        """Incremental deal filtering rejects aware returned timestamps."""
        frame = pd.DataFrame({
            "ticket": [1],
            "symbol": ["EURUSD"],
            "time": ["2024-01-01T00:00:00+00:00"],
            "type": [0],
        })
        with pytest.raises(ValueError, match="timezone-aware"):
            filter_incremental_history_deals_frame(
                frame,
                ["EURUSD"],
                {"EURUSD": datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)},
                datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
            )

    @pytest.mark.parametrize(
        ("frame", "start_by_symbol", "account_event_start", "expected_tickets"),
        [
            pytest.param(
                pd.DataFrame({
                    "ticket": [1],
                    "symbol": ["EURUSD"],
                    "time": [
                        datetime(2024, 1, 5, tzinfo=UTC)
                        .replace(tzinfo=None)
                        .isoformat()
                    ],
                    "type": [2],
                }),
                {"EURUSD": datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)},
                datetime(2024, 1, 10, tzinfo=UTC).replace(tzinfo=None),
                [],
                id="excludes-symbolized-account-events-from-trade-cursor",
            ),
            pytest.param(
                pd.DataFrame({
                    "ticket": [1],
                    "time": ["2024-01-03T00:00:00"],
                    "type": [2],
                }),
                {"EURUSD": datetime(2024, 1, 10, tzinfo=UTC).replace(tzinfo=None)},
                datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None),
                [1],
                id="keeps-account-events-without-symbol-column",
            ),
        ],
    )
    def test_filter_incremental_account_event_edge_cases(
        self,
        frame: pd.DataFrame,
        start_by_symbol: dict[str, datetime],
        account_event_start: datetime,
        expected_tickets: list[int],
    ) -> None:
        """Test account-event rows are scoped by symbol column presence.

        A symbolized account-event row follows only the EURUSD trade cursor
        and is excluded when its time falls before account_event_start; when
        history_deals has no symbol column, account events are kept using
        account_event_start.
        """
        filtered = filter_incremental_history_deals_frame(
            frame,
            ["EURUSD"],
            start_by_symbol,
            account_event_start,
        )
        assert filtered["ticket"].tolist() == expected_tickets

    @pytest.mark.parametrize(
        "frame",
        [
            pytest.param(
                pd.DataFrame({
                    "ticket": [1],
                    "symbol": ["EURUSD"],
                    "time": ["not-a-datetime"],
                    "type": [0],
                }),
                id="unparseable-time",
            ),
            pytest.param(
                pd.DataFrame({
                    "ticket": [1],
                    "time": ["2024-01-03T00:00:00"],
                }),
                id="trade-rows-without-symbol-column",
            ),
            pytest.param(
                pd.DataFrame({
                    "ticket": [1],
                    "symbol": ["EURUSD"],
                    "type": [0],
                }),
                id="rows-without-time-column",
            ),
        ],
    )
    def test_filter_incremental_returns_empty_for_invalid_rows(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Test incremental filtering drops rows that cannot be evaluated."""
        filtered = filter_incremental_history_deals_frame(
            frame,
            ["EURUSD"],
            {"EURUSD": datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)},
            datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
        )
        assert filtered.empty


class TestIncrementalIntegration:
    """Integration tests for incremental write helpers."""

    def test_write_collected_datasets_and_edge_branches(
        self,
        tmp_path: Path,
    ) -> None:
        """Test collected dataset writer and helper edge branches."""
        client = MagicMock()
        client.copy_rates_range.return_value = pd.DataFrame()
        client.copy_ticks_range.return_value = pd.DataFrame({
            "time": ["2024-01-01T00:00:00"],
            "bid": [1.0],
        })
        client.history_orders.return_value = pd.DataFrame({
            "ticket": [1],
            "symbol": ["EURUSD"],
            "time": [1],
            "type": [0],
        })
        client.history_deals.return_value = pd.DataFrame({
            "ticket": [1],
            "symbol": ["EURUSD"],
            "time": [1],
            "type": [0],
        })
        client.symbol_info_as_dict.return_value = {
            "symbol": "EURUSD",
            "point": 0.00001,
            "digits": 5,
        }
        db_path = tmp_path / "collected-integration.db"
        start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        end = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(db_path) as conn:
            written_tables, _ = write_collected_datasets(
                conn,
                client,
                ["EURUSD"],
                {
                    Dataset.ticks,
                    Dataset.history_orders,
                    Dataset.history_deals,
                    Dataset.symbols,
                },
                1,
                1,
                start,
                end,
                IfExists.APPEND,
            )
            assert Dataset.symbols in written_tables
            symbols_row = conn.execute(
                "SELECT symbol, time, point, digits FROM symbols",
            ).fetchone()
            assert symbols_row == ("EURUSD", "2024-01-02T00:00:00", 0.00001, 5)
            assert (
                get_incremental_start_datetime(
                    conn,
                    Dataset.ticks,
                    symbol="EURUSD",
                    timeframe=None,
                    fallback_start=start,
                )
                == start
            )
            deduplicate_history_tables(
                conn,
                {Dataset.ticks: {"time"}},
                {Dataset.ticks},
            )
        filtered = filter_trade_history_frame(
            pd.DataFrame({"ticket": [1]}),
            ["EURUSD"],
            include_account_events=False,
        )
        assert len(filtered) == 1
        account_filtered = filter_trade_history_frame(
            pd.DataFrame({"symbol": ["", "EURUSD"], "type": [2, 0]}),
            ["EURUSD"],
            include_account_events=True,
        )
        assert len(account_filtered) == 2
        no_type_filtered = filter_trade_history_frame(
            pd.DataFrame({"symbol": ["", "EURUSD"]}),
            ["EURUSD"],
            include_account_events=True,
        )
        assert len(no_type_filtered) == 2

    def test_get_incremental_start_without_symbol_column(
        self,
        tmp_path: Path,
    ) -> None:
        """Test incremental start ignores missing symbol column filters."""
        db_path = tmp_path / "no-symbol-column.db"
        fallback = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE ticks(time TEXT)")
            conn.execute(
                "INSERT INTO ticks(time) VALUES (?)",
                ("2024-01-02T00:00:00",),
            )
            assert get_incremental_start_datetime(
                conn,
                Dataset.ticks,
                symbol="EURUSD",
                timeframe=None,
                fallback_start=fallback,
            ) == datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)

    def test_deduplicate_skips_unsupported_keys(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test deduplication logs when no stable key columns exist."""
        with (
            sqlite3.connect(tmp_path / "no-keys.db") as conn,
            caplog.at_level(
                logging.WARNING,
                logger="mt5cli.history",
            ),
        ):
            deduplicate_history_tables(conn, {Dataset.ticks: {"time"}}, {Dataset.ticks})
        assert "Skipping ticks deduplication" in caplog.text

    def test_write_rates_skips_empty_schema(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test rates writer logs the symbol when a frame has no columns."""
        client = MagicMock()
        client.copy_rates_range.return_value = pd.DataFrame()
        written_columns: dict[Dataset, set[str]] = {}
        with (
            caplog.at_level(logging.WARNING, logger="mt5cli.history"),
            sqlite3.connect(tmp_path / "empty-rates.db") as conn,
        ):
            assert not write_rates_dataset(
                conn,
                client,
                ["EURUSD"],
                1,
                datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None),
                IfExists.APPEND,
                written_columns,
            )
        assert (
            "Skipping rates for symbol=EURUSD: dataset returned no columns"
            in caplog.text
        )

    def test_write_symbols_dataset_snapshots_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        """Test symbols writer snapshots per-symbol metadata at snapshot_time."""

        def symbol_info_as_dict(*, symbol: str) -> dict[str, object]:
            return {
                "symbol": symbol,
                "point": 0.01 if symbol == "USDJPY" else 0.00001,
                "digits": 3 if symbol == "USDJPY" else 5,
                "trade_contract_size": 100000.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_tick_size": 0.001,
                "trade_tick_value": 1.0,
                "currency_profit": "JPY" if symbol == "USDJPY" else "USD",
            }

        client = MagicMock()
        client.symbol_info_as_dict.side_effect = symbol_info_as_dict
        snapshot_time = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        written_columns: dict[Dataset, set[str]] = {}
        with sqlite3.connect(tmp_path / "symbols-snapshot.db") as conn:
            assert write_symbols_dataset(
                conn,
                client,
                ["EURUSD", "USDJPY"],
                snapshot_time,
                IfExists.APPEND,
                written_columns,
            )
            rows = conn.execute(
                "SELECT symbol, time, point, digits, currency_profit"
                " FROM symbols ORDER BY symbol",
            ).fetchall()
        assert rows == [
            ("EURUSD", "2024-01-01T00:00:00", 0.00001, 5, "USD"),
            ("USDJPY", "2024-01-01T00:00:00", 0.01, 3, "JPY"),
        ]
        assert written_columns[Dataset.symbols] >= {
            "symbol",
            "time",
            "point",
            "digits",
            "currency_profit",
        }

    def test_write_symbols_dataset_nulls_metadata_for_zero_point(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test a missing/zero point persists a NULL row with a warning."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"symbol": "XAUUSD", "point": 0}
        written_columns: dict[Dataset, set[str]] = {}
        with (
            caplog.at_level(logging.WARNING, logger="mt5cli.history"),
            sqlite3.connect(tmp_path / "symbols-zero-point.db") as conn,
        ):
            assert write_symbols_dataset(
                conn,
                client,
                ["XAUUSD"],
                datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                IfExists.APPEND,
                written_columns,
            )
            row = conn.execute(
                "SELECT symbol, point, digits, currency_profit FROM symbols",
            ).fetchone()
        assert row == ("XAUUSD", None, None, None)
        assert "XAUUSD" in caplog.text
        assert "missing or zero point" in caplog.text

    @pytest.mark.parametrize(
        "error",
        [Mt5RuntimeError("unknown symbol"), Mt5ConnectionError("unknown symbol")],
    )
    def test_write_symbols_dataset_nulls_metadata_when_lookup_raises(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        error: Exception,
    ) -> None:
        """Test an unknown/invalid symbol persists NULL metadata, not an abort."""
        client = MagicMock()
        client.symbol_info_as_dict.side_effect = error
        written_columns: dict[Dataset, set[str]] = {}
        with (
            caplog.at_level(logging.WARNING, logger="mt5cli.history"),
            sqlite3.connect(tmp_path / "symbols-lookup-error.db") as conn,
        ):
            assert write_symbols_dataset(
                conn,
                client,
                ["BADSYM"],
                datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                IfExists.APPEND,
                written_columns,
            )
            row = conn.execute(
                "SELECT symbol, point, digits, currency_profit FROM symbols",
            ).fetchone()
        assert row == ("BADSYM", None, None, None)
        assert "BADSYM" in caplog.text
        assert "could not be retrieved" in caplog.text
        assert caplog.text.count("BADSYM") == 1

    def test_write_symbols_dataset_nulls_metadata_when_lookup_returns_none(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test a non-mapping lookup result persists NULL metadata, not a crash."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = None
        written_columns: dict[Dataset, set[str]] = {}
        with (
            caplog.at_level(logging.WARNING, logger="mt5cli.history"),
            sqlite3.connect(tmp_path / "symbols-lookup-none.db") as conn,
        ):
            assert write_symbols_dataset(
                conn,
                client,
                ["BADSYM"],
                datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                IfExists.APPEND,
                written_columns,
            )
            row = conn.execute(
                "SELECT symbol, point, digits, currency_profit FROM symbols",
            ).fetchone()
        assert row == ("BADSYM", None, None, None)
        assert "BADSYM" in caplog.text
        assert "could not be retrieved" in caplog.text

    def test_write_symbols_dataset_preserves_numeric_types_when_first_is_null(
        self,
        tmp_path: Path,
    ) -> None:
        """Test a NULL-first row does not poison later rows with TEXT affinity."""

        def symbol_info_as_dict(*, symbol: str) -> dict[str, object]:
            if symbol == "XAUUSD":
                return {"symbol": "XAUUSD", "point": 0}
            return {
                "symbol": symbol,
                "point": 0.00001,
                "digits": 5,
                "trade_contract_size": 100000.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_tick_size": 0.00001,
                "trade_tick_value": 1.0,
                "currency_profit": "USD",
            }

        client = MagicMock()
        client.symbol_info_as_dict.side_effect = symbol_info_as_dict
        written_columns: dict[Dataset, set[str]] = {}
        with sqlite3.connect(tmp_path / "symbols-null-first.db") as conn:
            assert write_symbols_dataset(
                conn,
                client,
                ["XAUUSD", "EURUSD"],
                datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                IfExists.APPEND,
                written_columns,
            )
            point, digits = conn.execute(
                "SELECT point, digits FROM symbols WHERE symbol = 'EURUSD'",
            ).fetchone()
        assert isinstance(point, float)
        assert abs(point - 0.00001) < 1e-9
        assert isinstance(digits, int)
        assert digits == 5

    def test_augment_written_columns_creates_new_entry(
        self,
        tmp_path: Path,
    ) -> None:
        """Test augment helper initializes dataset column maps."""
        db_path = tmp_path / "augment-new.db"
        written_columns: dict[Dataset, set[str]] = {}
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE rates(time TEXT)")
            augment_written_columns_from_sqlite(
                conn,
                {Dataset.rates},
                written_columns,
            )
        assert written_columns == {Dataset.rates: {"time"}}

    @pytest.mark.parametrize("flags", ["BAD", 7])
    def test_resolve_history_tick_flags_invalid(self, flags: str | int) -> None:
        """Test invalid tick flags raise ValueError."""
        with pytest.raises(ValueError, match="Invalid tick flags"):
            resolve_history_tick_flags(flags)

    def test_resolve_history_timeframes_invalid(self) -> None:
        """Test invalid timeframes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            resolve_history_timeframes(["BAD"])


class TestIncrementalHistoryDeals:
    """Tests for incremental history_deals account-event handling."""

    def test_write_history_dataset_fetches_account_events_once(
        self,
        tmp_path: Path,
    ) -> None:
        """Test write_history_dataset account-event path fetches once."""
        client = MagicMock()
        client.history_deals.return_value = pd.DataFrame({
            "ticket": [1, 2],
            "symbol": ["EURUSD", ""],
            "time": [1, 2],
            "type": [0, 2],
            "entry": [0, 0],
        })
        start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        end = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
        written_columns: dict[Dataset, set[str]] = {}
        with sqlite3.connect(tmp_path / "history-dataset-account.db") as conn:
            assert write_history_dataset(
                conn,
                client.history_deals,
                Dataset.history_deals,
                ["EURUSD"],
                start,
                end,
                IfExists.APPEND,
                written_columns,
                include_account_events=True,
            )
            rows = conn.execute(
                "SELECT ticket, symbol, type FROM history_deals ORDER BY ticket",
            ).fetchall()
        client.history_deals.assert_called_once()
        assert rows == [(1, "EURUSD", 0), (2, "", 2)]


class TestWriteHelpers:
    """Tests for SQLite write helper branches."""

    def test_append_dataframe_handles_wide_frames(self, tmp_path: Path) -> None:
        """Test wide DataFrames append without exceeding SQLite variable limits."""
        db_path = tmp_path / "wide-frame.db"
        columns = {f"col_{index}": [float(index)] for index in range(80)}
        frame = pd.DataFrame(columns)
        with sqlite3.connect(db_path) as conn:
            assert append_dataframe(conn, frame, "wide_rates", IfExists.APPEND)
            assert get_table_columns(conn, "wide_rates") == set(columns)

    def test_write_streamed_frame_and_column_tracking(self, tmp_path: Path) -> None:
        """Test append helpers track columns and skip empty frames."""
        db_path = tmp_path / "write-helpers.db"
        written_columns: dict[Dataset, set[str]] = {}
        with sqlite3.connect(db_path) as conn:
            assert not write_streamed_frame(
                conn,
                pd.DataFrame(),
                Dataset.rates,
                table_exists=False,
                if_exists=IfExists.APPEND,
                written_columns=written_columns,
            )
            assert write_streamed_frame(
                conn,
                pd.DataFrame({
                    "time": ["2024-01-01T00:00:00", "not-a-datetime"],
                    "open": [1.0, 2.0],
                }),
                Dataset.rates,
                table_exists=False,
                if_exists=IfExists.APPEND,
                written_columns=written_columns,
            )
            assert append_dataframe(
                conn,
                pd.DataFrame({"time": [1], "open": [1.0]}),
                "rates",
                IfExists.APPEND,
            )
            record_written_columns(
                written_columns,
                Dataset.rates,
                pd.DataFrame({"close": [1.1]}),
            )
            assert "close" in written_columns[Dataset.rates]
            augment_written_columns_from_sqlite(
                conn,
                {Dataset.rates},
                written_columns,
            )
            assert get_table_columns(conn, "rates") == {"time", "open"}
            create_history_indexes(conn, written_columns)


class TestRateSourceHelpers:
    """Tests for generic rate-source SDK helpers."""

    def test_rate_target_timeframe_int(self) -> None:
        """Test RateTarget resolves named and integer timeframes."""
        target = RateTarget(symbol="EURUSD", timeframe="M1")
        assert target.timeframe == 1
        assert target.timeframe_int == 1
        assert RateTarget(symbol="EURUSD", timeframe=16385).timeframe_int == 16385

    def test_build_rate_targets_row_major(self) -> None:
        """Test targets are built in row-major symbol/timeframe order."""
        targets = build_rate_targets(["EURUSD", "GBPUSD"], ["M1", "H1"])
        assert [(t.symbol, t.timeframe) for t in targets] == [
            ("EURUSD", 1),
            ("EURUSD", 16385),
            ("GBPUSD", 1),
            ("GBPUSD", 16385),
        ]

    def test_build_rate_targets_allows_missing_symbol(self) -> None:
        """Test missing symbols produce None-symbol targets when allowed."""
        targets = build_rate_targets([], ["M1", "H1"], allow_missing_symbol=True)
        assert [(t.symbol, t.timeframe) for t in targets] == [
            (None, 1),
            (None, 16385),
        ]

    @pytest.mark.parametrize(
        ("symbols", "timeframes", "match"),
        [
            (["EURUSD"], [], "At least one timeframe"),
            ([], ["M1"], "At least one symbol"),
        ],
    )
    def test_build_rate_targets_rejects_empty(
        self,
        symbols: list[str],
        timeframes: list[str],
        match: str,
    ) -> None:
        """Test target building input validation."""
        with pytest.raises(ValueError, match=match):
            build_rate_targets(symbols, timeframes)

    def test_report_rate_gaps_parses_numeric_sqlite_times_as_epoch_seconds(
        self,
        tmp_path: Path,
    ) -> None:
        """Numeric SQLite timestamps must be interpreted as Unix seconds."""
        db_path = tmp_path / "numeric-gaps.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE custom_rates(time INTEGER, close REAL)")
            conn.executemany(
                "INSERT INTO custom_rates(time, close) VALUES (?, ?)",
                [
                    (1704067200, 1.0),
                    (1704067320, 1.1),
                ],
            )
            result = report_rate_gaps(
                conn,
                "custom_rates",
                granularity_seconds=60,
            )

        assert len(result) == 1
        assert result.iloc[0]["missing_intervals"] == 1

    def test_report_rate_gaps_empty_schema_for_zero_gap_tables(
        self,
        tmp_path: Path,
    ) -> None:
        """Tables without gaps return the stable empty result schema."""
        db_path = tmp_path / "no-gaps.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE custom_rates(time TEXT, close REAL)")
            conn.executemany(
                "INSERT INTO custom_rates(time, close) VALUES (?, ?)",
                [
                    ("2024-01-01T00:00:00+00:00", 1.0),
                    ("2024-01-01T00:01:00+00:00", 1.1),
                ],
            )
            result = report_rate_gaps(
                conn,
                "custom_rates",
                granularity_seconds=60,
            )

        assert list(result.columns) == [
            "table",
            "symbol",
            "timeframe",
            "granularity",
            "granularity_seconds",
            "gap_start",
            "gap_end",
            "missing_intervals",
        ]
        assert result.empty

    def test_report_rate_gaps_filters_by_min_gap_intervals(
        self,
        tmp_path: Path,
    ) -> None:
        """Small gaps are filtered out when min_gap_intervals is raised."""
        db_path = tmp_path / "filtered-gaps.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE custom_rates(time TEXT, close REAL)")
            conn.executemany(
                "INSERT INTO custom_rates(time, close) VALUES (?, ?)",
                [
                    ("2024-01-01T00:00:00+00:00", 1.0),
                    ("2024-01-01T00:02:00+00:00", 1.1),
                ],
            )
            result = report_rate_gaps(
                conn,
                "custom_rates",
                granularity_seconds=60,
                min_gap_intervals=2,
            )

        assert result.empty

    def test_report_rate_gaps_computes_gaps_per_series_key(
        self,
        tmp_path: Path,
    ) -> None:
        """Managed rates tables must detect gaps within each symbol/timeframe series."""
        db_path = tmp_path / "series-gaps.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                "symbol TEXT, timeframe INTEGER, time TEXT, close REAL"
                ")",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("GBPUSD", 1, "2024-01-01T00:01:00+00:00", 1.1),
                    ("EURUSD", 1, "2024-01-01T00:02:00+00:00", 1.2),
                ],
            )

            result = report_rate_gaps(
                conn,
                "rates",
                granularity_seconds=60,
            )

        records = cast("list[dict[str, object]]", result.to_dict("records"))
        assert records == [
            {
                "table": "rates",
                "symbol": "EURUSD",
                "timeframe": 1,
                "granularity": "M1",
                "granularity_seconds": 60,
                "gap_start": datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                "gap_end": datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                "missing_intervals": 1,
            },
        ]

    def test_rate_gap_private_helpers_and_validation(self) -> None:
        """Private helpers should preserve schema and reject invalid inputs."""
        empty = history._empty_rate_gap_report()  # type: ignore[attr-defined]
        assert list(empty.columns) == [
            "table",
            "symbol",
            "timeframe",
            "granularity",
            "granularity_seconds",
            "gap_start",
            "gap_end",
            "missing_intervals",
        ]

        class _BadInt:
            def __int__(self) -> int:
                msg = "bad-int"
                raise ValueError(msg)

        assert history._coerce_optional_int(None) is None  # type: ignore[attr-defined]
        false_value: object = False
        assert history._coerce_optional_int(false_value) is None  # type: ignore[attr-defined]
        assert history._coerce_optional_int(" +7 ") == 7  # type: ignore[attr-defined]
        assert history._coerce_optional_int("bad") is None  # type: ignore[attr-defined]
        assert history._coerce_optional_int(object()) is None  # type: ignore[attr-defined]
        assert history._coerce_optional_int(_BadInt()) is None  # type: ignore[attr-defined]

        metadata = history._rate_gap_metadata(  # type: ignore[attr-defined]
            "custom_rates",
            pd.DataFrame({
                "symbol": ["EURUSD", "GBPUSD"],
                "timeframe": [1, 1],
            }),
        )
        assert metadata["symbol"] is None
        assert metadata["timeframe"] == 1
        assert metadata["granularity"] == "M1"

        fallback_metadata = history._rate_gap_metadata(  # type: ignore[attr-defined]
            "rate_USDJPY__M1_1",
            pd.DataFrame({"time": []}),
        )
        assert fallback_metadata["symbol"] is None
        assert fallback_metadata["timeframe"] is None
        assert fallback_metadata["granularity"] is None

        unique_symbol_metadata = history._rate_gap_metadata(  # type: ignore[attr-defined]
            "custom_rates",
            pd.DataFrame({"symbol": ["EURUSD"], "timeframe": [1]}),
        )
        assert unique_symbol_metadata["symbol"] == "EURUSD"

        multi_timeframe_metadata = history._rate_gap_metadata(  # type: ignore[attr-defined]
            "custom_rates",
            pd.DataFrame({"timeframe": [1, 5]}),
        )
        assert multi_timeframe_metadata["timeframe"] is None
        assert multi_timeframe_metadata["granularity"] is None

    @pytest.mark.parametrize(
        ("granularity_seconds", "min_gap_intervals", "match"),
        [
            pytest.param(
                0, 1, "granularity_seconds must be positive", id="bad-seconds"
            ),
            pytest.param(60, 0, "min_gap_intervals must be positive", id="bad-min-gap"),
        ],
    )
    def test_report_rate_gaps_rejects_invalid_parameters(
        self,
        tmp_path: Path,
        granularity_seconds: int,
        min_gap_intervals: int,
        match: str,
    ) -> None:
        """Gap reports reject non-positive granularity and min-gap values."""
        db_path = tmp_path / "invalid-gaps.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE custom_rates(time TEXT, close REAL)")
            conn.execute(
                "INSERT INTO custom_rates(time, close) VALUES (?, ?)",
                ("2024-01-01T00:00:00+00:00", 1.0),
            )
            with pytest.raises(ValueError, match=match):
                report_rate_gaps(
                    conn,
                    "custom_rates",
                    granularity_seconds=granularity_seconds,
                    min_gap_intervals=min_gap_intervals,
                )

    def test_report_rate_gaps_rejects_unparseable_timestamps(
        self,
        tmp_path: Path,
    ) -> None:
        """Unparseable table times should fail clearly."""
        db_path = tmp_path / "bad-times.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE custom_rates(time TEXT, close REAL)")
            conn.execute(
                "INSERT INTO custom_rates(time, close) VALUES (?, ?)",
                ("bad-time", 1.0),
            )
            with pytest.raises(ValueError, match="contains unparsable time values"):
                report_rate_gaps(
                    conn,
                    "custom_rates",
                    granularity_seconds=60,
                )

    def test_report_rate_gaps_rejects_mixed_timestamp_awareness(
        self,
        tmp_path: Path,
    ) -> None:
        """Gap reporting must not relabel naive values in a mixed series."""
        db_path = tmp_path / "mixed-times.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE custom_rates(time TEXT, close REAL)")
            conn.executemany(
                "INSERT INTO custom_rates(time, close) VALUES (?, ?)",
                [
                    ("2024-01-01T00:00:00", 1.0),
                    ("2024-01-01T00:02:00+00:00", 1.1),
                ],
            )
            with pytest.raises(ValueError, match="cannot mix timezone-naive"):
                report_rate_gaps(
                    conn,
                    "custom_rates",
                    granularity_seconds=60,
                )

    def test_report_rate_gaps_empty_and_single_row_sources_return_empty(
        self,
        tmp_path: Path,
    ) -> None:
        """Empty or single-row sources cannot produce gap rows."""
        db_path = tmp_path / "too-short.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE custom_rates(time TEXT, close REAL)")
            assert report_rate_gaps(conn, "custom_rates", granularity_seconds=60).empty
            conn.execute(
                "INSERT INTO custom_rates(time, close) VALUES (?, ?)",
                ("2024-01-01T00:00:00+00:00", 1.0),
            )
            assert report_rate_gaps(conn, "custom_rates", granularity_seconds=60).empty


class TestCollectHistory:
    """Tests for collect_history SDK function."""

    @pytest.fixture
    def history_client(self, mocker: MockerFixture) -> MagicMock:
        """Create a mocked Mt5DataClient with history-style DataFrames."""
        return _build_history_client(mocker)

    @pytest.mark.parametrize(
        (
            "datasets",
            "expected_rates_calls",
            "expected_ticks_calls",
            "required_tables",
            "forbidden_table",
        ),
        [
            pytest.param(
                None,
                2,
                0,
                {"rates", "history_orders", "history_deals"},
                "ticks",
                id="default-excludes-ticks",
            ),
            pytest.param(
                {Dataset.ticks},
                0,
                2,
                {"ticks"},
                "rates",
                id="explicit-ticks",
            ),
        ],
    )
    def test_collect_history_default_and_ticks_dataset(
        self,
        tmp_path: Path,
        history_client: MagicMock,
        datasets: set[Dataset] | None,
        expected_rates_calls: int,
        expected_ticks_calls: int,
        required_tables: set[str],
        forbidden_table: str,
    ) -> None:
        """Test default vs explicit ticks dataset selection for collect_history."""
        output = tmp_path / "history.db"
        if datasets is None:
            collect_history(
                output,
                ["EURUSD", "GBPUSD"],
                "2024-01-01",
                "2024-02-01",
            )
        else:
            collect_history(
                output,
                ["EURUSD", "GBPUSD"],
                "2024-01-01",
                "2024-02-01",
                datasets=datasets,
            )
        assert history_client.copy_rates_range_as_df.call_count == expected_rates_calls
        assert history_client.copy_ticks_range_as_df.call_count == expected_ticks_calls
        with sqlite3.connect(output) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
        assert required_tables <= tables
        assert forbidden_table not in tables

    def test_collect_history_with_views(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test that with_views creates cash_events and positions views."""
        output = tmp_path / "history.db"
        collect_history(
            output,
            ["EURUSD", "GBPUSD"],
            "2024-01-01",
            "2024-02-01",
            with_views=True,
        )
        with sqlite3.connect(output) as conn:
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                ).fetchall()
            }
            positions = {
                row[0]
                for row in conn.execute(
                    "SELECT position_id FROM positions_reconstructed",
                ).fetchall()
            }
        assert {"cash_events", "positions_reconstructed"} <= views
        assert set(positions) == {100, 200, 500, 600}

    def test_collect_history_rates_table_has_timeframe(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test that the rates table carries the requested timeframe value."""
        output = tmp_path / "history.db"
        collect_history(
            output,
            ["EURUSD"],
            "2024-01-01",
            "2024-02-01",
            datasets={Dataset.rates},
            timeframe="H1",
        )
        with sqlite3.connect(output) as conn:
            rows = conn.execute(
                "SELECT DISTINCT timeframe FROM rates",
            ).fetchall()
        assert rows == [(16385,)]

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
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=client)
        output = tmp_path / "history.db"
        with caplog.at_level(logging.WARNING, logger="mt5cli.history"):
            collect_history(
                output,
                ["EURUSD"],
                "2024-01-01",
                "2024-02-01",
                with_views=True,
            )
        with sqlite3.connect(output) as conn:
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                ).fetchall()
            }
        assert "cash_events" not in views
        assert "positions_reconstructed" not in views

    @pytest.mark.parametrize(
        ("datasets", "expected_message", "expected_level"),
        [
            pytest.param(
                {Dataset.history_deals},
                "Skipping history-deal views: no history_deals data was collected",
                logging.INFO,
                id="empty-deals-logs-info",
            ),
            pytest.param(
                set[Dataset](),
                "--with-views requires the history_deals dataset",
                logging.WARNING,
                id="missing-deals-dataset-logs-warning",
            ),
        ],
    )
    def test_collect_history_with_views_deal_skip_messages(
        self,
        tmp_path: Path,
        history_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
        datasets: set[Dataset],
        expected_message: str,
        expected_level: int,
    ) -> None:
        """Test collect_history reports history-deal view skips at the right level."""
        history_client.history_deals_get_as_df.side_effect = None
        history_client.history_deals_get_as_df.return_value = pd.DataFrame()
        with caplog.at_level(logging.INFO, logger="mt5cli.history"):
            collect_history(
                tmp_path / "history.db",
                ["EURUSD"],
                "2024-01-01",
                "2024-02-01",
                datasets=datasets,
                with_views=True,
            )
        records = [
            record for record in caplog.records if record.message == expected_message
        ]
        assert len(records) == 1
        assert records[0].levelno == expected_level
        if expected_level == logging.INFO:
            assert not [
                record for record in caplog.records if record.levelno >= logging.WARNING
            ]


class TestUpdateHistory:
    """Tests for update_history SDK functions."""

    @pytest.fixture
    def connected_client(self) -> MagicMock:
        """Create a connected mock client without MT5 lifecycle patching."""
        return MagicMock()

    def test_update_history_appends_incrementally(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test sequential SQLite history updates use existing max timestamps."""
        date_to = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
        first_expected_start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        second_expected_start = datetime(2024, 1, 1, 12, tzinfo=UTC).replace(
            tzinfo=None
        )
        rate_starts: list[datetime] = []
        deal_starts: list[datetime] = []

        def make_rates(**kwargs: object) -> pd.DataFrame:
            assert kwargs["symbol"] == "EURUSD"
            assert kwargs["timeframe"] == 1
            assert kwargs["date_to"] == date_to
            rate_starts.append(kwargs["date_from"])  # type: ignore[arg-type]
            return pd.DataFrame({
                "time": ["2024-01-01T12:00:00"],
                "open": [1.0 + len(rate_starts) / 10],
            })

        def make_deals(**kwargs: object) -> pd.DataFrame:
            assert kwargs["date_to"] == date_to
            deal_starts.append(kwargs["date_from"])  # type: ignore[arg-type]
            return pd.DataFrame({
                "ticket": [10],
                "position_id": [100],
                "symbol": ["EURUSD"],
                "time": ["2024-01-01T12:00:00"],
                "type": [0],
                "entry": [0],
                "volume": [1.0],
                "price": [1.1],
                "profit": [0.0],
            })

        connected_client.copy_rates_range.side_effect = make_rates
        connected_client.history_deals.side_effect = make_deals
        mocker.patch("mt5cli.client.Mt5DataClient")
        output = tmp_path / "incremental-history.db"

        for _ in range(2):
            update_history(
                client=connected_client,
                output=output,
                symbols=["EURUSD"],
                datasets={Dataset.rates, Dataset.history_deals},
                timeframes=["M1"],
                lookback_hours=24,
                date_to=date_to,
                with_views=True,
            )

        assert rate_starts == [first_expected_start, second_expected_start]
        assert deal_starts == [first_expected_start, first_expected_start]
        connected_client.initialize_and_login_mt5.assert_not_called()
        connected_client.shutdown.assert_not_called()
        with sqlite3.connect(output) as conn:
            assert conn.execute("SELECT COUNT(*) FROM rates").fetchone() == (1,)
            assert conn.execute("SELECT open FROM rates").fetchone() == (1.2,)
            assert conn.execute(
                "SELECT COUNT(*) FROM history_deals",
            ).fetchone() == (1,)
            assert conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'cash_events'",
            ).fetchone() == ("cash_events",)

    def test_update_history_deals_mixes_existing_and_new_symbol_cursors(
        self,
        connected_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Deal updates use one naive query window for mixed cursor sources."""
        output = tmp_path / "mixed-deal-cursors.db"
        with sqlite3.connect(output) as conn:
            conn.execute(
                "CREATE TABLE history_deals("
                "ticket INTEGER, position_id INTEGER, symbol TEXT, time TEXT, "
                "type INTEGER, entry INTEGER, volume REAL, price REAL, profit REAL)"
            )
            conn.execute(
                "INSERT INTO history_deals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (1, 100, "EURUSD", "2024-01-03T00:00:00", 0, 0, 1.0, 1.1, 0.0),
            )

        connected_client.history_deals.return_value = pd.DataFrame({
            "ticket": [1, 2, 3, 4],
            "position_id": [100, 101, 200, 0],
            "symbol": ["EURUSD", "EURUSD", "GBPUSD", ""],
            "time": [
                "2024-01-03T00:00:00",
                "2024-01-04T00:00:00",
                "2024-01-04T00:00:00",
                "2024-01-03T00:00:00",
            ],
            "type": [0, 0, 0, 2],
            "entry": [0, 1, 0, 0],
            "volume": [1.0, 1.0, 1.0, 0.0],
            "price": [1.1, 1.2, 1.3, 0.0],
            "profit": [0.0, 0.0, 0.0, 5.0],
        })

        update_history(
            client=connected_client,
            output=output,
            symbols=["EURUSD", "GBPUSD"],
            datasets={Dataset.history_deals},
            lookback_hours=24,
            date_to=datetime(2024, 1, 5, tzinfo=UTC).replace(tzinfo=None),
        )

        call_kwargs = connected_client.history_deals.call_args.kwargs
        assert call_kwargs["date_from"] == datetime(2024, 1, 3, tzinfo=UTC).replace(
            tzinfo=None
        )
        assert call_kwargs["date_to"] == datetime(2024, 1, 5, tzinfo=UTC).replace(
            tzinfo=None
        )
        assert call_kwargs["date_from"].tzinfo is None
        assert call_kwargs["date_to"].tzinfo is None
        with sqlite3.connect(output) as conn:
            symbols = {
                row[0]
                for row in conn.execute("SELECT DISTINCT symbol FROM history_deals")
            }
        assert symbols == {"EURUSD", "GBPUSD"}

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"symbols": []}, "At least one symbol"),
            (
                {"symbols": ["EURUSD"], "lookback_hours": 0},
                "lookback_hours must be positive",
            ),
            (
                {
                    "symbols": ["EURUSD"],
                    "datasets": {Dataset.rates},
                    "timeframes": ["BAD"],
                    "date_to": _TEST_DATE_TO,
                },
                "Invalid timeframe",
            ),
            (
                {
                    "symbols": ["EURUSD"],
                    "datasets": {Dataset.ticks},
                    "flags": "BAD",
                    "date_to": _TEST_DATE_TO,
                },
                "Invalid tick flags",
            ),
        ],
        ids=["empty-symbols", "non-positive-lookback", "bad-timeframe", "bad-flags"],
    )
    def test_update_history_rejects_invalid_inputs(
        self,
        connected_client: MagicMock,
        tmp_path: Path,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        """Test validation errors for incremental history updates."""
        output = tmp_path / "invalid-update.db"
        with pytest.raises(ValueError, match=match):
            update_history(
                client=connected_client,
                output=output,
                **kwargs,  # type: ignore[arg-type]
            )

    def test_update_history_noops_for_empty_datasets(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test empty dataset selection skips MT5 and SQLite writes."""
        writer = mocker.patch("mt5cli.history.write_incremental_datasets")
        connect = mocker.patch("mt5cli.history.sqlite3.connect")
        update_history(
            client=connected_client,
            output=tmp_path / "empty-datasets.db",
            symbols=["EURUSD"],
            datasets=set(),
        )
        writer.assert_not_called()
        connect.assert_not_called()

    @pytest.mark.parametrize(
        ("timeframes", "expected"),
        [
            (None, [parse_timeframe(t) for t in DEFAULT_HISTORY_TIMEFRAMES]),
            (["M1", "H1"], [1, 16385]),
        ],
        ids=["default", "specified"],
    )
    def test_update_history_resolves_timeframes(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
        timeframes: list[str] | None,
        expected: list[int],
    ) -> None:
        """Test update_history writes all default or specified rate timeframes."""
        timeframes_written: list[int] = []

        def capture(
            *args: object,
            **_kwargs: object,
        ) -> tuple[set[Dataset], dict[Dataset, set[str]]]:
            timeframes_written.extend(args[4])  # type: ignore[arg-type]
            return set(), {}

        mocker.patch("mt5cli.history.write_incremental_datasets", side_effect=capture)
        update_history(
            client=connected_client,
            output=tmp_path / "timeframes.db",
            symbols=["EURUSD"],
            datasets={Dataset.rates},
            timeframes=timeframes,
            lookback_hours=1,
            date_to=datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
        )
        assert timeframes_written == expected

    def test_update_history_updates_ticks_and_orders(
        self,
        connected_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test incremental update writes selected ticks and orders datasets."""
        date_to = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
        expected_start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)

        def make_ticks(**kwargs: object) -> pd.DataFrame:
            assert kwargs["symbol"] == "EURUSD"
            assert kwargs["date_from"] == expected_start
            assert kwargs["date_to"] == date_to
            assert kwargs["flags"] == -1
            return pd.DataFrame({
                "time": ["2024-01-01T12:00:00"],
                "time_msc": [1_704_110_400_000],
                "bid": [1.1],
            })

        def make_orders(**kwargs: object) -> pd.DataFrame:
            assert kwargs["symbol"] == "EURUSD"
            assert kwargs["date_from"] == expected_start
            assert kwargs["date_to"] == date_to
            return pd.DataFrame({
                "ticket": [1],
                "symbol": ["EURUSD"],
                "time": ["2024-01-01T12:00:00"],
                "type": [0],
            })

        connected_client.copy_ticks_range.side_effect = make_ticks
        connected_client.history_orders.side_effect = make_orders
        output = tmp_path / "ticks-orders.db"
        update_history(
            client=connected_client,
            output=output,
            symbols=["EURUSD"],
            datasets={Dataset.ticks, Dataset.history_orders},
            lookback_hours=24,
            date_to=date_to,
        )
        with sqlite3.connect(output) as conn:
            assert conn.execute("SELECT COUNT(*) FROM ticks").fetchone() == (1,)
            assert conn.execute(
                "SELECT COUNT(*) FROM history_orders",
            ).fetchone() == (1,)

    def test_update_history_syncs_rates_and_symbols_together(
        self,
        connected_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test rates and symbols metadata can be synced in the same update."""
        date_to = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)
        connected_client.copy_rates_range.return_value = pd.DataFrame({
            "time": ["2024-01-01T12:00:00"],
            "open": [1.1],
        })
        connected_client.symbol_info_as_dict.return_value = {
            "symbol": "EURUSD",
            "point": 0.00001,
            "digits": 5,
            "trade_contract_size": 100000.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "trade_tick_size": 0.00001,
            "trade_tick_value": 1.0,
            "currency_profit": "USD",
        }
        output = tmp_path / "rates-symbols.db"
        update_history(
            client=connected_client,
            output=output,
            symbols=["EURUSD"],
            datasets={Dataset.rates, Dataset.symbols},
            timeframes=["M1"],
            lookback_hours=24,
            date_to=date_to,
        )
        with sqlite3.connect(output) as conn:
            assert conn.execute("SELECT COUNT(*) FROM rates").fetchone() == (1,)
            assert conn.execute(
                "SELECT symbol, point, currency_profit FROM symbols",
            ).fetchone() == ("EURUSD", 0.00001, "USD")

    def test_update_history_with_config_validates_before_connecting(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test invalid inputs fail before MT5 is initialized."""
        mock_client = MagicMock()
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=mock_client)
        with pytest.raises(ValueError, match="lookback_hours must be positive"):
            update_history_with_config(
                output=tmp_path / "invalid-config.db",
                symbols=["EURUSD"],
                lookback_hours=0,
            )
        mock_client.initialize_and_login_mt5.assert_not_called()
        mock_client.shutdown.assert_not_called()

    def test_update_history_with_config_noops_for_empty_datasets(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test empty dataset selection skips MT5 initialization."""
        mock_client = MagicMock()
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=mock_client)
        updater = mocker.patch("mt5cli.history.update_history")
        update_history_with_config(
            output=tmp_path / "empty-config.db",
            symbols=["EURUSD"],
            datasets=set(),
        )
        mock_client.initialize_and_login_mt5.assert_not_called()
        mock_client.shutdown.assert_not_called()
        updater.assert_not_called()

    def test_update_history_requires_explicit_server_time(
        self,
        connected_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test update_history rejects an unknown server-time end."""
        with pytest.raises(ValueError, match="date_to is required"):
            update_history(
                client=connected_client,
                output=tmp_path / "missing-date-to.db",
                symbols=["EURUSD"],
                datasets={Dataset.rates},
                timeframes=["M1"],
                lookback_hours=12,
            )

    def test_update_history_default_datasets_exclude_ticks(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test update_history with datasets=None does not collect ticks."""
        datasets_written: list[set[Dataset]] = []

        def capture(
            *args: object,
            **_kwargs: object,
        ) -> tuple[set[Dataset], dict[Dataset, set[str]]]:
            datasets_written.append(args[3])  # type: ignore[arg-type]
            return set(), {}

        mocker.patch("mt5cli.history.write_incremental_datasets", side_effect=capture)
        update_history(
            client=connected_client,
            output=tmp_path / "default-datasets.db",
            symbols=["EURUSD"],
            datasets=None,
            timeframes=["M1"],
            lookback_hours=1,
            date_to=datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
        )
        assert len(datasets_written) == 1
        assert Dataset.ticks not in datasets_written[0]
        assert {
            Dataset.rates,
            Dataset.history_orders,
            Dataset.history_deals,
        } == datasets_written[0]


class TestThrottledHistoryUpdater:
    """Tests for the throttled incremental history updater."""

    def test_updates_every_call_when_interval_non_positive(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test interval_seconds <= 0 updates on every call."""
        update = mocker.patch("mt5cli.history.update_history")
        client = MagicMock()
        updater = ThrottledHistoryUpdater(output="history.db", interval_seconds=0)

        assert updater.update(client, ["EURUSD"], date_to=_TEST_DATE_TO) is True
        assert updater.update(client, ["EURUSD"], date_to=_TEST_DATE_TO) is True
        assert update.call_count == 2

    def test_throttles_within_interval(self, mocker: MockerFixture) -> None:
        """Test updates are skipped until the interval elapses."""
        update = mocker.patch("mt5cli.history.update_history")
        monotonic = mocker.patch("mt5cli.history.time.monotonic")
        # Calls: set(t=100), check(t=105), check(t=200), set(t=200).
        monotonic.side_effect = [100.0, 105.0, 200.0, 200.0]
        client = MagicMock()
        updater = ThrottledHistoryUpdater(output="history.db", interval_seconds=60)

        assert (
            updater.update(client, ["EURUSD"], date_to=_TEST_DATE_TO) is True
        )  # first update at t=100
        assert (
            updater.update(client, ["EURUSD"], date_to=_TEST_DATE_TO) is False
        )  # t=105, throttled
        assert (
            updater.update(client, ["EURUSD"], date_to=_TEST_DATE_TO) is True
        )  # t=200, elapsed
        assert update.call_count == 2

    def test_update_passes_expected_arguments(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test update_history is called with the configured arguments."""
        update = mocker.patch("mt5cli.history.update_history")
        client = MagicMock()
        updater = ThrottledHistoryUpdater(
            output="history.db",
            datasets={Dataset.rates},
            timeframes=["M1", "H1"],
            flags="INFO",
            lookback_hours=12.0,
            with_views=True,
            include_account_events=False,
        )

        updater.update(client, ["EURUSD", "GBPUSD"], date_to=_TEST_DATE_TO)

        update.assert_called_once_with(
            client=client,
            output="history.db",
            symbols=["EURUSD", "GBPUSD"],
            datasets={Dataset.rates},
            timeframes=["M1", "H1"],
            flags="INFO",
            lookback_hours=12.0,
            date_to=_TEST_DATE_TO,
            with_views=True,
            include_account_events=False,
        )

    def test_propagates_errors_by_default(self, mocker: MockerFixture) -> None:
        """Test MT5/SQLite errors propagate and do not advance the throttle."""
        mocker.patch(
            "mt5cli.history.update_history",
            side_effect=Mt5RuntimeError("boom"),
        )
        updater = ThrottledHistoryUpdater(output="history.db")

        with pytest.raises(Mt5RuntimeError, match="boom"):
            updater.update(MagicMock(), ["EURUSD"], date_to=_TEST_DATE_TO)

        assert updater.last_update_monotonic is None

    @pytest.mark.parametrize(
        "error",
        [
            Mt5RuntimeError("boom"),
            Mt5ConnectionError("normalized boom"),
            sqlite3.OperationalError("locked"),
            ValueError("invalid symbols"),
            OSError("disk full"),
        ],
    )
    def test_suppresses_errors_when_requested(
        self,
        mocker: MockerFixture,
        error: Exception,
    ) -> None:
        """Test suppress_errors swallows recoverable errors and returns False."""
        mocker.patch(
            "mt5cli.history.update_history",
            side_effect=error,
        )
        updater = ThrottledHistoryUpdater(
            output="history.db",
            suppress_errors=True,
        )

        assert updater.update(MagicMock(), ["EURUSD"], date_to=_TEST_DATE_TO) is False
        assert updater.last_update_monotonic is None

    @pytest.mark.parametrize(
        "error",
        [
            AttributeError("'dict' object has no attribute 'typo'"),
            TypeError("unsupported operand types"),
        ],
    )
    def test_suppress_errors_does_not_hide_programming_errors(
        self,
        mocker: MockerFixture,
        error: Exception,
    ) -> None:
        """Test generic AttributeError/TypeError still propagate when suppressed."""
        mocker.patch(
            "mt5cli.history.update_history",
            side_effect=error,
        )
        updater = ThrottledHistoryUpdater(
            output="history.db",
            suppress_errors=True,
        )

        with pytest.raises(type(error)):
            updater.update(MagicMock(), ["EURUSD"], date_to=_TEST_DATE_TO)

        assert updater.last_update_monotonic is None

    def test_suppresses_validation_errors_before_update(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test validation failures are suppressed without calling update_history."""
        update = mocker.patch("mt5cli.history.update_history")
        updater = ThrottledHistoryUpdater(
            output="history.db",
            suppress_errors=True,
        )

        assert updater.update(MagicMock(), []) is False
        update.assert_not_called()
        assert updater.last_update_monotonic is None

    def test_requires_explicit_server_time_before_backend(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test throttled updates reject an unknown server-time end."""
        update = mocker.patch("mt5cli.history.update_history")
        updater = ThrottledHistoryUpdater(output="history.db")

        with pytest.raises(ValueError, match="date_to is required"):
            updater.update(MagicMock(), ["EURUSD"])

        update.assert_not_called()
        assert updater.last_update_monotonic is None

    def test_default_update_backend_is_update_history(self) -> None:
        """Test the default backend resolves to update_history."""
        updater = ThrottledHistoryUpdater(output="history.db")
        assert updater.update_backend is update_history

    def test_falsy_callable_update_backend_is_preserved(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test only None selects the default backend, not falsy callables."""

        class FalsyCallable:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __bool__(self) -> bool:
                return False

            def __call__(self, **kwargs: object) -> None:
                self.calls.append(kwargs)

        falsy_backend = FalsyCallable()
        default_backend = mocker.patch("mt5cli.history.update_history")
        updater = ThrottledHistoryUpdater(
            output="history.db",
            update_backend=falsy_backend,
        )

        assert updater.update_backend is falsy_backend
        client = MagicMock()
        assert updater.update(client, ["EURUSD"], date_to=_TEST_DATE_TO) is True
        assert len(falsy_backend.calls) == 1
        assert falsy_backend.calls[0]["client"] is client
        assert falsy_backend.calls[0]["symbols"] == ["EURUSD"]
        default_backend.assert_not_called()

    def test_custom_update_backend_receives_expected_kwargs(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test a custom backend receives update_history keyword arguments."""
        backend = mocker.Mock()
        client = MagicMock()
        updater = ThrottledHistoryUpdater(
            output="history.db",
            datasets={Dataset.rates},
            timeframes=["M1", "H1"],
            flags="INFO",
            lookback_hours=12.0,
            with_views=True,
            include_account_events=False,
            update_backend=backend,
        )

        updater.update(client, ["EURUSD", "GBPUSD"], date_to=_TEST_DATE_TO)

        backend.assert_called_once_with(
            client=client,
            output="history.db",
            symbols=["EURUSD", "GBPUSD"],
            datasets={Dataset.rates},
            timeframes=["M1", "H1"],
            flags="INFO",
            lookback_hours=12.0,
            date_to=_TEST_DATE_TO,
            with_views=True,
            include_account_events=False,
        )

    def test_throttled_calls_do_not_invoke_custom_backend(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test throttled update cycles skip the injected backend."""
        backend = mocker.Mock()
        monotonic = mocker.patch("mt5cli.history.time.monotonic")
        monotonic.side_effect = [100.0, 105.0, 200.0, 200.0]
        client = MagicMock()
        updater = ThrottledHistoryUpdater(
            output="history.db",
            interval_seconds=60,
            update_backend=backend,
        )

        assert updater.update(client, ["EURUSD"], date_to=_TEST_DATE_TO) is True
        assert updater.update(client, ["EURUSD"], date_to=_TEST_DATE_TO) is False
        assert updater.update(client, ["EURUSD"], date_to=_TEST_DATE_TO) is True
        assert backend.call_count == 2

    def test_successful_custom_backend_advances_throttle(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test a successful custom backend updates _last_update_monotonic."""
        backend = mocker.Mock()
        monotonic = mocker.patch("mt5cli.history.time.monotonic", return_value=42.0)
        updater = ThrottledHistoryUpdater(
            output="history.db",
            update_backend=backend,
        )

        assert (
            updater.update(
                MagicMock(),
                ["EURUSD"],
                date_to=_TEST_DATE_TO,
            )
            is True
        )
        assert updater.last_update_monotonic is monotonic.return_value
        monotonic.assert_called_once()

    def test_failed_custom_backend_does_not_advance_throttle(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test a failing custom backend leaves _last_update_monotonic unchanged."""
        backend = mocker.Mock(side_effect=Mt5RuntimeError("boom"))
        updater = ThrottledHistoryUpdater(
            output="history.db",
            update_backend=backend,
        )

        with pytest.raises(Mt5RuntimeError, match="boom"):
            updater.update(
                MagicMock(),
                ["EURUSD"],
                date_to=_TEST_DATE_TO,
            )

        assert updater.last_update_monotonic is None

    @pytest.mark.parametrize(
        ("suppress_errors", "raises"),
        [
            (True, None),
            (False, Mt5RuntimeError),
        ],
        ids=["suppress", "propagate"],
    )
    def test_custom_backend_error_suppression(
        self,
        mocker: MockerFixture,
        suppress_errors: bool,
        raises: type[BaseException] | None,
    ) -> None:
        """suppress_errors controls whether recoverable backend errors propagate."""
        backend = mocker.Mock(side_effect=Mt5RuntimeError("boom"))
        updater = ThrottledHistoryUpdater(
            output="history.db",
            suppress_errors=suppress_errors,
            update_backend=backend,
        )
        if raises is None:
            assert (
                updater.update(
                    MagicMock(),
                    ["EURUSD"],
                    date_to=_TEST_DATE_TO,
                )
                is False
            )
        else:
            with pytest.raises(raises, match="boom"):
                updater.update(
                    MagicMock(),
                    ["EURUSD"],
                    date_to=_TEST_DATE_TO,
                )
        assert updater.last_update_monotonic is None


class TestUpdateHistoryTelemetry:
    """Tests for telemetry hooks in update_history."""

    def test_update_history_invokes_history_telemetry(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """update_history wraps write_incremental_datasets with telemetry."""
        mock_client = MagicMock()
        mock_client.copy_rates_range.return_value = pd.DataFrame()
        mock_client.history_orders.return_value = pd.DataFrame()
        mock_client.history_deals.return_value = pd.DataFrame()
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_history_update.return_value = mock_cm
        mocker.patch("mt5cli.history.get_metrics", return_value=mock_metrics)
        update_history(
            client=mock_client,
            output=tmp_path / "hist.db",
            symbols=["EURUSD"],
            date_to=_TEST_DATE_TO,
        )
        mock_metrics.record_history_update.assert_called_once_with(dataset="history")

    def test_update_history_emits_history_rows(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """update_history calls add_history_rows with the SQLite change delta."""
        mock_client = MagicMock()
        mock_client.copy_rates_range.return_value = pd.DataFrame()
        mock_client.history_orders.return_value = pd.DataFrame()
        mock_client.history_deals.return_value = pd.DataFrame()
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_history_update.return_value = mock_cm
        mocker.patch("mt5cli.history.get_metrics", return_value=mock_metrics)
        update_history(
            client=mock_client,
            output=tmp_path / "hist.db",
            symbols=["EURUSD"],
            date_to=_TEST_DATE_TO,
        )
        mock_metrics.add_history_rows.assert_called_once_with(0, dataset="history")


def test_sqlite_round_trip_preserves_naive_server_wall_clock() -> None:
    """SQLite storage must not relabel a naive server timestamp as UTC."""
    wall_clock = datetime(2024, 1, 1, 9, 30, tzinfo=UTC).replace(tzinfo=None)
    frame = pd.DataFrame({"time": [wall_clock], "close": [1.0]})
    with sqlite3.connect(":memory:") as conn:
        assert append_dataframe(conn, frame, "rates", IfExists.APPEND)
        stored = conn.execute("SELECT time FROM rates").fetchone()[0]
    assert stored == "2024-01-01T09:30:00"
    parsed = parse_sqlite_timestamp(stored)
    assert parsed == wall_clock
    assert parsed is not None
    assert parsed.tzinfo is None


def test_sqlite_round_trip_normalizes_explicit_offset_to_utc() -> None:
    """Aware values preserve their instant while storage normalizes to UTC."""
    aware = pd.Timestamp("2024-01-01T09:30:00+09:00").to_pydatetime()
    frame = pd.DataFrame({"time": [aware], "close": [1.0]})
    with sqlite3.connect(":memory:") as conn:
        assert append_dataframe(conn, frame, "rates", IfExists.APPEND)
        stored = conn.execute("SELECT time FROM rates").fetchone()[0]
    assert stored == "2024-01-01T00:30:00+00:00"
    assert parse_sqlite_timestamp(stored) == datetime(2024, 1, 1, 0, 30, tzinfo=UTC)


def test_incremental_rate_cursor_preserves_naive_wall_clock() -> None:
    """Incremental cursors return the stored naive server clock label unchanged."""
    fallback = datetime(2023, 12, 31, tzinfo=UTC).replace(tzinfo=None)
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
        )
        conn.execute(
            "INSERT INTO rates VALUES (?, ?, ?, ?)",
            ("EURUSD", 1, "2024-01-01T09:30:00", 1.0),
        )
        starts = load_incremental_start_datetimes(
            conn,
            Dataset.rates,
            symbols=["EURUSD"],
            timeframes=[1],
            fallback_start=fallback,
        )
    cursor = starts["EURUSD", 1]
    assert cursor == datetime(2024, 1, 1, 9, 30, tzinfo=UTC).replace(tzinfo=None)
    assert cursor.tzinfo is None


def test_incremental_cursor_rejects_aware_persisted_timestamp() -> None:
    """Managed incremental cursors reject explicit timezone-aware rows."""
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
        )
        conn.execute(
            "INSERT INTO rates VALUES (?, ?, ?, ?)",
            ("EURUSD", 1, "2024-01-01T09:30:00+00:00", 1.0),
        )
        with pytest.raises(ValueError, match="timezone-aware"):
            load_incremental_start_datetimes(
                conn,
                Dataset.rates,
                symbols=["EURUSD"],
                timeframes=[1],
                fallback_start=datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
            )
