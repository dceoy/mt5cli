"""Tests for mt5cli.history module."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture  # noqa: TC002

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

from pdmt5 import TIMEFRAME_MAP

from mt5cli import history
from mt5cli.history import (
    DEFAULT_HISTORY_DATASETS,
    DEFAULT_HISTORY_TIMEFRAMES,
    DedupScope,
    RateTarget,
    append_dataframe,
    augment_written_columns_from_sqlite,
    build_rate_targets,
    build_rate_view_name,
    create_cash_events_view,
    create_history_indexes,
    create_positions_reconstructed_view,
    create_rate_compatibility_views,
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
    load_rate_series_by_granularity,
    load_rate_series_from_sqlite,
    parse_sqlite_timestamp,
    quote_sqlite_identifier,
    record_written_columns,
    resolve_granularity_name,
    resolve_history_datasets,
    resolve_history_tick_flags,
    resolve_history_timeframes,
    resolve_rate_table_name,
    resolve_rate_tables,
    resolve_rate_view_name,
    resolve_rate_view_names,
    write_collected_datasets,
    write_history_dataset,
    write_incremental_datasets,
    write_rates_dataset,
    write_streamed_frame,
)
from mt5cli.utils import Dataset, IfExists


def _resolve_rate_view_single(
    conn_or_path: sqlite3.Connection | Path | str | None,
    *,
    granularity: str = "M1",
    require_existing: bool = False,
) -> str:
    """Resolve a single EURUSD rate view name via resolve_rate_view_name."""
    return resolve_rate_view_name(
        conn_or_path,
        "EURUSD",
        granularity,
        require_existing=require_existing,
    )


def _resolve_rate_view_batch(
    conn_or_path: sqlite3.Connection | Path | str | None,
    *,
    granularity: str = "M1",
    require_existing: bool = False,
) -> list[str]:
    """Resolve batch EURUSD rate view names via resolve_rate_view_names."""
    return resolve_rate_view_names(
        conn_or_path,
        ["EURUSD"],
        [granularity],
        require_existing=require_existing,
    )


class TestResolveRateViewName:
    """Tests for resolve_rate_view_name and resolve_rate_view_names."""

    def test_resolve_rate_table_name_returns_normalized_table(self) -> None:
        """Test canonical normalized rates table name is stable."""
        assert resolve_rate_table_name("EURUSD", "M1") == "rates"

    def test_resolve_rate_table_name_rejects_empty_symbol(self) -> None:
        """Test canonical rate table resolution validates symbols."""
        with pytest.raises(ValueError, match="symbol must not be empty"):
            resolve_rate_table_name(" ", "M1")

    def test_missing_database_path_does_not_create_file(self, tmp_path: Path) -> None:
        """Test resolving against a missing path does not create a database."""
        db_path = tmp_path / "missing.db"
        assert resolve_rate_view_name(db_path, "EURUSD", "M1") == "rate_EURUSD__1"
        assert not db_path.exists()

    def test_none_path_returns_default_name(self) -> None:
        """Test a None connection or path returns the deterministic default."""
        assert resolve_rate_view_name(None, "EURUSD", "M1") == "rate_EURUSD__1"
        assert resolve_rate_view_names(None, ["EURUSD"], ["M1", "H1"]) == [
            "rate_EURUSD__1",
            "rate_EURUSD__16385",
        ]

    @pytest.mark.parametrize(
        "resolve",
        [_resolve_rate_view_single, _resolve_rate_view_batch],
        ids=["single", "batch"],
    )
    def test_none_path_with_require_existing_raises(
        self,
        resolve: Callable[..., object],
    ) -> None:
        """Test a None path under strict mode raises a clear error."""
        with pytest.raises(ValueError, match="SQLite database not found"):
            resolve(None, require_existing=True)

    def test_no_rates_table_falls_back_to_single_timeframe_name(
        self,
        tmp_path: Path,
    ) -> None:
        """Test databases without a rates table use single-timeframe naming."""
        db_path = tmp_path / "no-rates.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE ticks(symbol TEXT, time TEXT)")
        assert resolve_rate_view_name(db_path, "EURUSD", "M1") == "rate_EURUSD__1"

    def test_single_timeframe_for_one_symbol(self, tmp_path: Path) -> None:
        """Test one stored timeframe resolves to the short view name."""
        db_path = tmp_path / "single-timeframe.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            create_rate_compatibility_views(conn)
        assert resolve_rate_view_name(db_path, "EURUSD", "M1") == "rate_EURUSD__1"

    def test_multiple_timeframes_for_one_symbol(self, tmp_path: Path) -> None:
        """Test multiple stored timeframes resolve to disambiguated view names."""
        db_path = tmp_path / "multi-timeframe.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", TIMEFRAME_MAP["H1"], "2024-01-01T01:00:00+00:00", 1.1),
                ],
            )
            create_rate_compatibility_views(conn)
        assert resolve_rate_view_name(db_path, "EURUSD", "M1") == "rate_EURUSD__M1_1"
        assert (
            resolve_rate_view_name(db_path, "EURUSD", "H1") == "rate_EURUSD__H1_16385"
        )

    def test_prefers_multi_name_when_both_candidate_views_exist(
        self,
        tmp_path: Path,
    ) -> None:
        """Test multi-timeframe metadata wins over stale single-timeframe views."""
        db_path = tmp_path / "stale-and-current-views.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", TIMEFRAME_MAP["H1"], "2024-01-01T01:00:00+00:00", 1.1),
                ],
            )
            conn.execute(
                'CREATE VIEW "rate_EURUSD__1" AS'
                " SELECT time, close FROM rates"
                " WHERE symbol = 'EURUSD' AND timeframe = 1",
            )
            conn.execute(
                'CREATE VIEW "rate_EURUSD__M1_1" AS'
                " SELECT time, close FROM rates"
                " WHERE symbol = 'EURUSD' AND timeframe = 1",
            )
        assert resolve_rate_view_name(db_path, "EURUSD", "M1") == "rate_EURUSD__M1_1"

    def test_prefers_existing_view_when_metadata_unavailable(
        self,
        tmp_path: Path,
    ) -> None:
        """Test an existing managed view is preferred without rates metadata."""
        db_path = tmp_path / "view-only.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE ticks(symbol TEXT, time TEXT)")
            conn.execute('CREATE VIEW "rate_EURUSD__M1_1" AS SELECT 1 AS close')
        assert resolve_rate_view_name(db_path, "EURUSD", "M1") == "rate_EURUSD__M1_1"

    def test_symbol_absent_from_rates_metadata_uses_candidate_pair(
        self,
        tmp_path: Path,
    ) -> None:
        """Test symbols missing from rates metadata still resolve known views."""
        db_path = tmp_path / "other-symbol-only.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("GBPUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            conn.execute(
                'CREATE VIEW "rate_EURUSD__1" AS'
                " SELECT time, close FROM rates"
                " WHERE symbol = 'EURUSD' AND timeframe = 1",
            )
        assert resolve_rate_view_name(db_path, "EURUSD", "M1") == "rate_EURUSD__1"

    def test_ignores_non_compatibility_rate_views(self, tmp_path: Path) -> None:
        """Test unrelated rate_* views without the __ separator are ignored."""
        db_path = tmp_path / "summary-view.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE ticks(symbol TEXT, time TEXT)")
            conn.execute('CREATE VIEW "rate_summary" AS SELECT 1 AS close')
        assert resolve_rate_view_name(db_path, "EURUSD", "M1") == "rate_EURUSD__1"

    @pytest.mark.parametrize(
        "resolve",
        [_resolve_rate_view_single, _resolve_rate_view_batch],
        ids=["single", "batch"],
    )
    def test_invalid_granularity_propagates_value_error(
        self,
        tmp_path: Path,
        resolve: Callable[..., object],
    ) -> None:
        """Test invalid granularities raise ValueError from parse_timeframe."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            resolve(tmp_path / "unused.db", granularity="BAD")

    def test_resolve_rate_view_names_for_multiple_pairs(self, tmp_path: Path) -> None:
        """Test batch resolution returns row-major symbol/granularity pairs."""
        db_path = tmp_path / "batch-resolve.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", TIMEFRAME_MAP["H1"], "2024-01-01T01:00:00+00:00", 1.1),
                    ("GBPUSD", 1, "2024-01-01T00:00:00+00:00", 1.2),
                ],
            )
            create_rate_compatibility_views(conn)
        assert resolve_rate_view_names(
            db_path,
            ["EURUSD", "GBPUSD"],
            ["M1", "H1"],
        ) == [
            "rate_EURUSD__M1_1",
            "rate_EURUSD__H1_16385",
            "rate_GBPUSD__1",
            "rate_GBPUSD__16385",
        ]

    @pytest.mark.parametrize(
        "symbol",
        ["EUR/USD", "US500.cash", "#US500"],
    )
    def test_supports_broker_specific_symbols(
        self,
        tmp_path: Path,
        symbol: str,
    ) -> None:
        """Test broker-specific symbols resolve to safely created view names."""
        db_path = tmp_path / "broker-symbol-resolve.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                (symbol, 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            create_rate_compatibility_views(conn)
        assert resolve_rate_view_name(db_path, symbol, "M1") == build_rate_view_name(
            symbol=symbol,
            granularity="M1",
            granularity_count=1,
            timeframe=1,
        )

    def test_accepts_open_sqlite_connection(self, tmp_path: Path) -> None:
        """Test resolver accepts an already-open SQLite connection."""
        db_path = tmp_path / "open-connection.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            create_rate_compatibility_views(conn)
            assert resolve_rate_view_name(conn, "EURUSD", "M1") == "rate_EURUSD__1"

    @pytest.mark.parametrize(
        "resolve",
        [_resolve_rate_view_single, _resolve_rate_view_batch],
        ids=["single", "batch"],
    )
    def test_require_existing_raises_when_database_missing(
        self,
        tmp_path: Path,
        resolve: Callable[..., object],
    ) -> None:
        """Test strict mode rejects missing database paths."""
        db_path = tmp_path / "missing.db"
        with pytest.raises(ValueError, match="SQLite database not found"):
            resolve(db_path, require_existing=True)

    @pytest.mark.parametrize(
        "resolve",
        [_resolve_rate_view_single, _resolve_rate_view_batch],
        ids=["single", "batch"],
    )
    def test_require_existing_raises_when_view_missing(
        self,
        tmp_path: Path,
        resolve: Callable[..., object],
    ) -> None:
        """Test strict mode rejects databases without matching rate views."""
        db_path = tmp_path / "no-view.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE ticks(symbol TEXT, time TEXT)")
        with pytest.raises(ValueError, match="No rate compatibility view exists"):
            resolve(db_path, require_existing=True)

    def test_require_existing_returns_existing_view(self, tmp_path: Path) -> None:
        """Test strict mode returns a view when one exists."""
        db_path = tmp_path / "existing-view.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            create_rate_compatibility_views(conn)
        assert (
            resolve_rate_view_name(
                db_path,
                "EURUSD",
                "M1",
                require_existing=True,
            )
            == "rate_EURUSD__1"
        )
        assert resolve_rate_view_names(
            db_path,
            ["EURUSD"],
            ["M1"],
            require_existing=True,
        ) == ["rate_EURUSD__1"]


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

    def test_load_rate_series_from_sqlite_table_style(
        self,
        tmp_path: Path,
    ) -> None:
        """Test public table-style loader returns one rate DataFrame."""
        db_path = tmp_path / "table-style.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE rates(time TEXT, close REAL)")
            conn.executemany(
                "INSERT INTO rates(time, close) VALUES (?, ?)",
                [
                    ("2024-01-01T00:00:00+00:00", 1.0),
                    ("2024-01-01T00:01:00+00:00", 1.1),
                ],
            )

        frame = load_rate_series_from_sqlite(db_path, table="rates", count=1)

        assert isinstance(frame, pd.DataFrame)
        assert list(frame["close"]) == [1.1]

    def test_load_rate_series_from_sqlite_requires_targets_without_table(self) -> None:
        """Test multi-series loading requires targets when table is omitted."""
        with pytest.raises(ValueError, match="targets are required"):
            load_rate_series_from_sqlite("unused.db", count=1)

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

    def test_rejects_missing_database_and_non_file(self, tmp_path: Path) -> None:
        """Test path validation for SQLite database inputs."""
        with pytest.raises(ValueError, match="SQLite database not found"):
            load_rate_data(tmp_path / "missing.db", "rates")
        with pytest.raises(ValueError, match="not a file"):
            load_rate_data(tmp_path, "rates")

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
        assert frame.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
        assert list(frame["close"]) == [1.0]


class TestResolveHistorySettings:
    """Tests for history dataset and timeframe resolution."""

    def test_resolve_history_datasets_defaults_and_empty(self) -> None:
        """Test dataset resolution excludes ticks by default."""
        resolved = resolve_history_datasets(None)
        assert resolved == set(DEFAULT_HISTORY_DATASETS)
        assert Dataset.ticks not in resolved
        assert {
            Dataset.rates,
            Dataset.history_orders,
            Dataset.history_deals,
        } == resolved
        assert resolve_history_datasets(set()) == set()

    def test_resolve_history_datasets_explicit_ticks(self) -> None:
        """Test that explicit ticks selection is honored."""
        assert resolve_history_datasets({Dataset.ticks}) == {Dataset.ticks}
        all_ds = resolve_history_datasets(set(Dataset))
        assert Dataset.ticks in all_ds

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
                datetime(2024, 1, 1, tzinfo=UTC),
                id="mt5-epoch-seconds",
            ),
            pytest.param(
                datetime.fromisoformat("2024-01-01T00:00:00"),
                datetime(2024, 1, 1, tzinfo=UTC),
                id="naive-datetime",
            ),
            pytest.param(
                "Jan 1 2024",
                datetime(2024, 1, 1, tzinfo=UTC),
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


class TestIncrementalStart:
    """Tests for get_incremental_start_datetime."""

    def test_uses_max_time_scoped_by_symbol_and_timeframe(
        self,
        tmp_path: Path,
    ) -> None:
        """Test rates increment is scoped by symbol and timeframe."""
        db_path = tmp_path / "scoped-rates.db"
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, open) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-02T00:00:00+00:00", 1.0),
                    ("EURUSD", 16385, "2024-01-03T00:00:00+00:00", 1.1),
                    ("GBPUSD", 1, "2024-01-04T00:00:00+00:00", 1.2),
                ],
            )
            assert get_incremental_start_datetime(
                conn,
                Dataset.rates,
                symbol="EURUSD",
                timeframe=1,
                fallback_start=fallback,
            ) == datetime(2024, 1, 2, tzinfo=UTC)
            assert get_incremental_start_datetime(
                conn,
                Dataset.rates,
                symbol="EURUSD",
                timeframe=16385,
                fallback_start=fallback,
            ) == datetime(2024, 1, 3, tzinfo=UTC)

    def test_load_incremental_start_datetimes_batches_rates(
        self, tmp_path: Path
    ) -> None:
        """Test grouped rates resume query returns all symbol/timeframe pairs."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "batch-rates.db") as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, open) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-02T00:00:00+00:00", 1.0),
                    ("GBPUSD", 1, "2024-01-03T00:00:00+00:00", 1.1),
                ],
            )
            starts = load_incremental_start_datetimes(
                conn,
                Dataset.rates,
                symbols=["EURUSD", "GBPUSD"],
                timeframes=[1],
                fallback_start=fallback,
            )
        assert starts["EURUSD", 1] == datetime(2024, 1, 2, tzinfo=UTC)
        assert starts["GBPUSD", 1] == datetime(2024, 1, 3, tzinfo=UTC)

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
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
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
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
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
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
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
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "batch-no-symbol.db") as conn:
            conn.execute("CREATE TABLE ticks(time TEXT)")
            conn.execute(
                "INSERT INTO ticks(time) VALUES (?)",
                ("2024-01-02T00:00:00+00:00",),
            )
            starts = load_incremental_start_datetimes(
                conn,
                Dataset.ticks,
                symbols=["EURUSD", "GBPUSD"],
                fallback_start=fallback,
            )
        expected = datetime(2024, 1, 2, tzinfo=UTC)
        assert starts["EURUSD", None] == expected
        assert starts["GBPUSD", None] == expected


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
        boundary = datetime(2024, 1, 2, tzinfo=UTC)
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

    def test_unusable_scope_falls_back_to_table_dedup(self, tmp_path: Path) -> None:
        """Test scopes with missing columns do not break stable-key dedup."""
        boundary = datetime(2024, 1, 1, tzinfo=UTC)
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
        boundary = datetime(2024, 1, 2, tzinfo=UTC)
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

    def test_creates_views_for_multiple_timeframes(self, tmp_path: Path) -> None:
        """Test rate views are created per symbol and timeframe."""
        db_path = tmp_path / "rate-views.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", 16385, "2024-01-01T01:00:00+00:00", 1.1),
                ],
            )
            create_rate_compatibility_views(conn)
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'"
                    " AND name LIKE 'rate_EURUSD%'",
                ).fetchall()
            }
            assert views == {"rate_EURUSD__M1_1", "rate_EURUSD__H1_16385"}
            assert conn.execute('SELECT close FROM "rate_EURUSD__M1_1"').fetchone() == (
                1.0,
            )

    def test_drops_stale_single_timeframe_view(self, tmp_path: Path) -> None:
        """Test stale rate_<symbol> views are removed when timeframes change."""
        db_path = tmp_path / "stale-rate-view.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            create_rate_compatibility_views(conn)
            assert conn.execute(
                "SELECT 1 FROM sqlite_master"
                " WHERE type='view' AND name = 'rate_EURUSD__1'",
            ).fetchone() == (1,)

            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", TIMEFRAME_MAP["H1"], "2024-01-01T01:00:00+00:00", 1.1),
            )
            create_rate_compatibility_views(conn)
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'"
                    " AND name LIKE 'rate_EURUSD%'",
                ).fetchall()
            }
        assert views == {"rate_EURUSD__M1_1", "rate_EURUSD__H1_16385"}
        assert "rate_EURUSD__1" not in views

    def test_does_not_drop_views_outside_rate_prefix(self, tmp_path: Path) -> None:
        """Test only literal rate_* views are dropped during recreation."""
        db_path = tmp_path / "rate-prefix-views.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            conn.execute("CREATE VIEW rate_EURUSD AS SELECT 1 AS close")
            conn.execute("CREATE VIEW rateX_custom AS SELECT 2 AS close")
            create_rate_compatibility_views(conn)
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                ).fetchall()
            }
            assert "rateX_custom" in views
            assert "rate_EURUSD__1" in views
            assert conn.execute("SELECT close FROM rate_EURUSD__1").fetchone() == (1.0,)

    @pytest.mark.parametrize(
        "symbol",
        ["EUR/USD", "US500.cash", "#US500"],
    )
    def test_supports_broker_specific_symbols(
        self,
        tmp_path: Path,
        symbol: str,
    ) -> None:
        """Test broker-specific symbols get readable rate views."""
        db_path = tmp_path / "broker-symbol.db"
        view_name = build_rate_view_name(
            symbol=symbol,
            granularity="M1",
            granularity_count=1,
            timeframe=1,
        )
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                (symbol, 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            create_rate_compatibility_views(conn)
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='view' AND name = ?",
                (view_name,),
            ).fetchone() == (1,)
            quoted = quote_sqlite_identifier(view_name)
            query = "SELECT close FROM " + quoted  # noqa: S608
            assert conn.execute(query).fetchone() == (1.0,)


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
                    (1, "EURUSD", "2024-01-05T00:00:00+00:00", 0),
                    (2, "", "2024-01-08T00:00:00+00:00", 2),
                ],
                datetime(2024, 1, 8, tzinfo=UTC),
                id="uses-type-column",
            ),
            pytest.param(
                ("CREATE TABLE history_deals( ticket INTEGER, symbol TEXT, time TEXT)"),
                "INSERT INTO history_deals(ticket, symbol, time) VALUES (?, ?, ?)",
                [
                    (1, "EURUSD", "2024-01-05T00:00:00+00:00"),
                    (2, "", "2024-01-07T00:00:00+00:00"),
                ],
                datetime(2024, 1, 7, tzinfo=UTC),
                id="falls-back-to-empty-symbol",
            ),
            pytest.param(
                "CREATE TABLE history_deals(ticket INTEGER, time TEXT)",
                "INSERT INTO history_deals(ticket, time) VALUES (?, ?)",
                [(1, "2024-01-05T00:00:00+00:00")],
                datetime(2024, 1, 1, tzinfo=UTC),
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
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
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
                "2024-01-05T00:00:00+00:00",
                "2024-01-11T00:00:00+00:00",
                "2024-01-02T00:00:00+00:00",
                "2024-01-02T00:00:00+00:00",
                "2024-01-03T00:00:00+00:00",
            ],
            "type": [0, 0, 0, 0, 2],
        })
        start_by_symbol = {
            "EURUSD": datetime(2024, 1, 10, tzinfo=UTC),
            "GBPUSD": datetime(2024, 1, 1, tzinfo=UTC),
        }
        filtered = filter_incremental_history_deals_frame(
            frame,
            ["EURUSD", "GBPUSD"],
            start_by_symbol,
            datetime(2024, 1, 2, tzinfo=UTC),
        )
        assert filtered["ticket"].tolist() == [2, 3, 5]

    @pytest.mark.parametrize(
        ("frame", "start_by_symbol", "account_event_start", "expected_tickets"),
        [
            pytest.param(
                pd.DataFrame({
                    "ticket": [1],
                    "symbol": ["EURUSD"],
                    "time": [datetime(2024, 1, 5, tzinfo=UTC).isoformat()],
                    "type": [2],
                }),
                {"EURUSD": datetime(2024, 1, 1, tzinfo=UTC)},
                datetime(2024, 1, 10, tzinfo=UTC),
                [],
                id="excludes-symbolized-account-events-from-trade-cursor",
            ),
            pytest.param(
                pd.DataFrame({
                    "ticket": [1],
                    "time": ["2024-01-03T00:00:00+00:00"],
                    "type": [2],
                }),
                {"EURUSD": datetime(2024, 1, 10, tzinfo=UTC)},
                datetime(2024, 1, 2, tzinfo=UTC),
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
                    "time": ["2024-01-03T00:00:00+00:00"],
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
            {"EURUSD": datetime(2024, 1, 1, tzinfo=UTC)},
            datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert filtered.empty


class TestIncrementalIntegration:
    """Integration tests for incremental write helpers."""

    def test_write_incremental_datasets_end_to_end(
        self,
        tmp_path: Path,
    ) -> None:
        """Test incremental dataset writer covers finalize branches."""
        client = MagicMock()
        client.copy_rates_range_as_df.return_value = pd.DataFrame({
            "time": ["2024-01-01T00:00:00+00:00"],
            "open": [1.0],
        })
        client.copy_ticks_range_as_df.return_value = pd.DataFrame()
        client.history_orders_get_as_df.return_value = pd.DataFrame({
            "ticket": [1],
            "symbol": ["EURUSD"],
            "time": ["2024-01-01T00:00:00+00:00"],
            "type": [0],
        })
        client.history_deals_get_as_df.return_value = pd.DataFrame({
            "ticket": [2],
            "position_id": [100],
            "symbol": ["EURUSD"],
            "time": ["2024-01-01T00:00:00+00:00"],
            "type": [0],
            "entry": [0],
            "volume": [1.0],
            "price": [1.1],
            "profit": [0.0],
        })
        db_path = tmp_path / "incremental-integration.db"
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        with sqlite3.connect(db_path) as conn:
            write_incremental_datasets(
                conn,
                client,
                ["EURUSD"],
                set(Dataset),
                [1],
                1,
                start,
                end,
                deduplicate=True,
                create_rate_views=True,
                with_views=True,
                include_account_events=True,
            )
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                ).fetchall()
            }
        assert {"rates", "history_orders", "history_deals"} <= tables
        assert "rate_EURUSD__1" in views
        assert "cash_events" in views

    def test_rate_view_names_do_not_collide_for_symbol_suffix(
        self,
        tmp_path: Path,
    ) -> None:
        """Test symbols resembling generated view suffixes stay distinct."""
        db_path = tmp_path / "rate-view-collision.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", 16385, "2024-01-01T01:00:00+00:00", 1.1),
                    ("EURUSD_M1", 1, "2024-01-01T02:00:00+00:00", 1.2),
                ],
            )
            create_rate_compatibility_views(conn)
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                ).fetchall()
            }
        assert views == {
            "rate_EURUSD__M1_1",
            "rate_EURUSD__H1_16385",
            "rate_EURUSD_M1__1",
        }

    def test_incremental_orders_without_time_deduplicate_by_ticket(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test incremental history_orders without time deduplicate safely."""

        def history_orders_get_as_df(**kwargs: object) -> pd.DataFrame:
            if kwargs["symbol"] == "GBPUSD":
                return pd.DataFrame()
            return pd.DataFrame({
                "ticket": [1, 1],
                "symbol": ["EURUSD", "EURUSD"],
                "time_setup": [
                    "2024-01-01T00:00:00+00:00",
                    "2024-01-01T00:00:01+00:00",
                ],
                "type": [0, 1],
            })

        client = MagicMock()
        client.history_orders_get_as_df.side_effect = history_orders_get_as_df
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        with (
            sqlite3.connect(tmp_path / "incremental-orders-without-time.db") as conn,
            caplog.at_level(logging.WARNING, logger="mt5cli.history"),
        ):
            write_incremental_datasets(
                conn,
                client,
                ["EURUSD", "GBPUSD"],
                {Dataset.history_orders},
                [],
                0,
                start,
                end,
                deduplicate=True,
                create_rate_views=False,
                with_views=False,
                include_account_events=False,
            )
            rows = conn.execute(
                "SELECT ticket, time_setup, type FROM history_orders",
            ).fetchall()
        assert rows == [(1, "2024-01-01T00:00:01+00:00", 1)]
        assert "Skipping history_orders: dataset returned no columns" in caplog.text

    def test_write_collected_datasets_and_edge_branches(
        self,
        tmp_path: Path,
    ) -> None:
        """Test collected dataset writer and helper edge branches."""
        client = MagicMock()
        client.copy_rates_range_as_df.return_value = pd.DataFrame()
        client.copy_ticks_range_as_df.return_value = pd.DataFrame({
            "time": ["2024-01-01T00:00:00+00:00"],
            "bid": [1.0],
        })
        client.history_orders_get_as_df.return_value = pd.DataFrame({
            "ticket": [1],
            "symbol": ["EURUSD"],
            "time": [1],
            "type": [0],
        })
        client.history_deals_get_as_df.return_value = pd.DataFrame({
            "ticket": [1],
            "symbol": ["EURUSD"],
            "time": [1],
            "type": [0],
        })
        db_path = tmp_path / "collected-integration.db"
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        with sqlite3.connect(db_path) as conn:
            write_collected_datasets(
                conn,
                client,
                ["EURUSD"],
                {Dataset.ticks, Dataset.history_orders, Dataset.history_deals},
                1,
                1,
                start,
                end,
                IfExists.APPEND,
            )
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
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE ticks(time TEXT)")
            conn.execute(
                "INSERT INTO ticks(time) VALUES (?)",
                ("2024-01-02T00:00:00+00:00",),
            )
            assert get_incremental_start_datetime(
                conn,
                Dataset.ticks,
                symbol="EURUSD",
                timeframe=None,
                fallback_start=fallback,
            ) == datetime(2024, 1, 2, tzinfo=UTC)

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
    ) -> None:
        """Test rates writer skips frames with no columns after normalization."""
        client = MagicMock()
        client.copy_rates_range_as_df.return_value = pd.DataFrame()
        written_columns: dict[Dataset, set[str]] = {}
        with sqlite3.connect(tmp_path / "empty-rates.db") as conn:
            assert not write_rates_dataset(
                conn,
                client,
                ["EURUSD"],
                1,
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                IfExists.APPEND,
                written_columns,
            )

    def test_finalize_with_views_warning_when_deals_missing(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test with_views warning when history_deals was not written."""
        client = MagicMock()
        client.copy_rates_range_as_df.return_value = pd.DataFrame()
        with (
            caplog.at_level(logging.WARNING, logger="mt5cli.history"),
            sqlite3.connect(tmp_path / "views-warning.db") as conn,
        ):
            write_incremental_datasets(
                conn,
                client,
                ["EURUSD"],
                {Dataset.rates, Dataset.history_deals},
                [1],
                0,
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                deduplicate=False,
                create_rate_views=False,
                with_views=True,
                include_account_events=True,
            )
        assert "with_views ignored" in caplog.text

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

    def test_create_rate_views_noop_without_required_columns(
        self,
        tmp_path: Path,
    ) -> None:
        """Test rate view creation skips tables missing required columns."""
        with sqlite3.connect(tmp_path / "missing-rate-columns.db") as conn:
            conn.execute("CREATE TABLE rates(open REAL)")
            create_rate_compatibility_views(conn)
            views = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view'",
            ).fetchall()
        assert views == []

    def test_incremental_history_noops_when_fetch_returns_empty(
        self,
        tmp_path: Path,
    ) -> None:
        """Test incremental history skips table tracking for empty writes."""
        client = MagicMock()
        client.history_orders_get_as_df.return_value = pd.DataFrame()
        client.history_deals_get_as_df.return_value = pd.DataFrame()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "empty-history.db") as conn:
            written_tables, _ = write_incremental_datasets(
                conn,
                client,
                ["EURUSD"],
                {Dataset.history_orders, Dataset.history_deals},
                [],
                0,
                start,
                end,
                deduplicate=False,
                create_rate_views=False,
                with_views=False,
                include_account_events=True,
            )
        assert written_tables == set()

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

    def test_fetches_per_symbol_when_account_events_disabled(
        self,
        tmp_path: Path,
    ) -> None:
        """Test history_deals are fetched per symbol without account events."""
        client = MagicMock()
        client.history_deals_get_as_df.return_value = pd.DataFrame({
            "ticket": [1],
            "symbol": ["EURUSD"],
            "time": [1],
            "type": [0],
            "entry": [0],
        })
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "per-symbol-deals.db") as conn:
            write_incremental_datasets(
                conn,
                client,
                ["EURUSD", "GBPUSD"],
                {Dataset.history_deals},
                [],
                0,
                start,
                end,
                deduplicate=False,
                create_rate_views=False,
                with_views=False,
                include_account_events=False,
            )
        assert client.history_deals_get_as_df.call_count == 2

    def test_skips_history_deals_when_fetch_returns_empty(
        self,
        tmp_path: Path,
    ) -> None:
        """Test incremental history_deals skips tracking when writes are empty."""
        client = MagicMock()
        client.history_deals_get_as_df.return_value = pd.DataFrame()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "empty-per-symbol-deals.db") as conn:
            written_tables, _ = write_incremental_datasets(
                conn,
                client,
                ["EURUSD", "GBPUSD"],
                {Dataset.history_deals},
                [],
                0,
                start,
                end,
                deduplicate=False,
                create_rate_views=False,
                with_views=False,
                include_account_events=False,
            )
        assert written_tables == set()

    def test_write_history_dataset_fetches_account_events_once(
        self,
        tmp_path: Path,
    ) -> None:
        """Test write_history_dataset account-event path fetches once."""
        client = MagicMock()
        client.history_deals_get_as_df.return_value = pd.DataFrame({
            "ticket": [1, 2],
            "symbol": ["EURUSD", ""],
            "time": [1, 2],
            "type": [0, 2],
            "entry": [0, 0],
        })
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        written_columns: dict[Dataset, set[str]] = {}
        with sqlite3.connect(tmp_path / "history-dataset-account.db") as conn:
            assert write_history_dataset(
                conn,
                client.history_deals_get_as_df,
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
        client.history_deals_get_as_df.assert_called_once()
        assert rows == [(1, "EURUSD", 0), (2, "", 2)]

    def test_fetches_account_events_once_for_multiple_symbols(
        self,
        tmp_path: Path,
    ) -> None:
        """Test account events are fetched once and not duplicated."""
        client = MagicMock()
        client.history_deals_get_as_df.return_value = pd.DataFrame({
            "ticket": [1, 2, 3, 4],
            "symbol": ["EURUSD", "GBPUSD", "OTHER", ""],
            "time": [
                "2024-01-01T00:00:00+00:00",
                "2024-01-01T01:00:00+00:00",
                "2024-01-01T02:00:00+00:00",
                "2024-01-01T03:00:00+00:00",
            ],
            "type": [0, 0, 0, 2],
            "entry": [0, 0, 0, 0],
        })
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 2, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "account-events.db") as conn:
            write_incremental_datasets(
                conn,
                client,
                ["EURUSD", "GBPUSD"],
                {Dataset.history_deals},
                [],
                0,
                start,
                end,
                deduplicate=False,
                create_rate_views=False,
                with_views=False,
                include_account_events=True,
            )
            rows = conn.execute(
                "SELECT ticket, symbol, type FROM history_deals ORDER BY ticket",
            ).fetchall()
            row_count = conn.execute("SELECT COUNT(*) FROM history_deals").fetchone()
        client.history_deals_get_as_df.assert_called_once()
        assert rows == [(1, "EURUSD", 0), (2, "GBPUSD", 0), (4, "", 2)]
        assert row_count == (3,)

    @pytest.mark.parametrize(
        ("frame", "ddl", "db_name"),
        [
            pytest.param(
                pd.DataFrame({
                    "ticket": [1],
                    "symbol": [""],
                    "time": ["2024-01-02T00:00:00+00:00"],
                }),
                "CREATE TABLE history_deals( ticket INTEGER, symbol TEXT, time TEXT)",
                "deals-without-type.db",
                id="dedup-scope-without-type-column",
            ),
            pytest.param(
                pd.DataFrame({
                    "ticket": [1],
                    "time": ["2024-01-02T00:00:00+00:00"],
                    "type": [2],
                }),
                "CREATE TABLE history_deals(ticket INTEGER, time TEXT, type INTEGER)",
                "no-symbol-deals.db",
                id="dedup-scope-without-symbol-column",
            ),
        ],
    )
    def test_account_event_dedup_scope_with_missing_column(
        self,
        tmp_path: Path,
        frame: pd.DataFrame,
        ddl: str,
        db_name: str,
    ) -> None:
        """Test account-event dedup falls back safely when a scope column is missing.

        Without a type column, dedup scope falls back to empty symbols; without
        a symbol column, per-symbol dedup scope is skipped. Both still write
        the row.
        """
        client = MagicMock()
        client.history_deals_get_as_df.return_value = frame
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 3, tzinfo=UTC)
        with sqlite3.connect(tmp_path / db_name) as conn:
            conn.execute(ddl)
            write_incremental_datasets(
                conn,
                client,
                ["EURUSD"],
                {Dataset.history_deals},
                [],
                0,
                start,
                end,
                deduplicate=True,
                create_rate_views=False,
                with_views=False,
                include_account_events=True,
            )
            assert conn.execute("SELECT COUNT(*) FROM history_deals").fetchone() == (1,)

    def test_creates_views_when_history_deals_written_with_with_views(
        self,
        tmp_path: Path,
    ) -> None:
        """Test derived views are rebuilt when history_deals is written."""
        client = MagicMock()
        client.history_deals_get_as_df.return_value = pd.DataFrame({
            "ticket": [1, 2],
            "position_id": [100, 100],
            "symbol": ["EURUSD", "EURUSD"],
            "time": ["2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00"],
            "type": [0, 1],
            "entry": [0, 1],
            "volume": [1.0, 1.0],
            "price": [1.1, 1.2],
            "profit": [0.0, 5.0],
        })
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 3, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "deals-with-views.db") as conn:
            write_incremental_datasets(
                conn,
                client,
                ["EURUSD"],
                {Dataset.history_deals},
                [],
                0,
                start,
                end,
                deduplicate=False,
                create_rate_views=False,
                with_views=True,
                include_account_events=False,
            )
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                ).fetchall()
            }
        assert {"cash_events", "positions_reconstructed"} <= views

    def test_applies_per_symbol_start_when_account_events_enabled(
        self,
        tmp_path: Path,
    ) -> None:
        """Test account-event fetch keeps per-symbol incremental boundaries."""
        client = MagicMock()
        client.history_deals_get_as_df.return_value = pd.DataFrame({
            "ticket": [1, 2, 3, 4, 5],
            "symbol": ["EURUSD", "EURUSD", "GBPUSD", "OTHER", ""],
            "time": [
                "2024-01-05T00:00:00+00:00",
                "2024-01-11T00:00:00+00:00",
                "2024-01-02T00:00:00+00:00",
                "2024-01-02T00:00:00+00:00",
                "2024-01-03T00:00:00+00:00",
            ],
            "type": [0, 0, 0, 0, 2],
            "entry": [0, 0, 0, 0, 0],
        })
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 15, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "per-symbol-account-events.db") as conn:
            conn.execute(
                "CREATE TABLE history_deals("
                " ticket INTEGER, symbol TEXT, time TEXT, type INTEGER, entry INTEGER)",
            )
            conn.executemany(
                "INSERT INTO history_deals(ticket, symbol, time, type, entry)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (100, "EURUSD", "2024-01-10T00:00:00+00:00", 0, 0),
                    (200, "GBPUSD", "2024-01-01T00:00:00+00:00", 0, 0),
                    (300, "", "2024-01-02T00:00:00+00:00", 2, 0),
                ],
            )
            write_incremental_datasets(
                conn,
                client,
                ["EURUSD", "GBPUSD"],
                {Dataset.history_deals},
                [],
                0,
                fallback,
                end,
                deduplicate=False,
                create_rate_views=False,
                with_views=False,
                include_account_events=True,
            )
            rows = conn.execute(
                "SELECT ticket, symbol, type FROM history_deals"
                " WHERE ticket IN (1, 2, 3, 4, 5) ORDER BY ticket",
            ).fetchall()
        client.history_deals_get_as_df.assert_called_once_with(
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            date_to=end,
        )
        assert rows == [(2, "EURUSD", 0), (3, "GBPUSD", 0), (5, "", 2)]


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

    def test_resolve_rate_tables_uses_explicit_tables(self) -> None:
        """Test explicit tables bypass view resolution when counts match."""
        targets = build_rate_targets([], ["M1", "H1"], allow_missing_symbol=True)
        assert resolve_rate_tables(None, targets, ["t1", "t2"]) == ["t1", "t2"]

    @pytest.mark.parametrize(
        ("targets", "explicit_tables", "path_kind", "require_existing", "match"),
        [
            pytest.param(
                build_rate_targets(["EURUSD"], ["M1"]),
                ["t1", "t2"],
                "none",
                False,
                "Expected 1 explicit table",
                id="mismatched-explicit-count",
            ),
            pytest.param(
                [],
                None,
                "none",
                False,
                "At least one rate target",
                id="empty-targets",
            ),
            pytest.param(
                build_rate_targets([], ["M1"], allow_missing_symbol=True),
                None,
                "none",
                False,
                "without a symbol",
                id="none-symbol-without-explicit",
            ),
            pytest.param(
                build_rate_targets(["EURUSD"], ["M1"]),
                None,
                "none",
                True,
                "SQLite database not found",
                id="none-path-require-existing",
            ),
            pytest.param(
                build_rate_targets(["EURUSD"], ["M1"]),
                None,
                "missing",
                True,
                "SQLite database not found",
                id="missing-db-require-existing",
            ),
        ],
    )
    def test_resolve_rate_tables_rejects_invalid_inputs(
        self,
        tmp_path: Path,
        targets: list[RateTarget],
        explicit_tables: list[str] | None,
        path_kind: str,
        require_existing: bool,
        match: str,
    ) -> None:
        """Test resolve_rate_tables input and strict-mode validation."""
        db_path = None if path_kind == "none" else tmp_path / "missing.db"
        with pytest.raises(ValueError, match=match):
            resolve_rate_tables(
                db_path,
                targets,
                explicit_tables=explicit_tables,
                require_existing=require_existing,
            )

    def test_resolve_rate_tables_resolves_view_names(self) -> None:
        """Test symbol targets resolve to default view names without a database."""
        targets = build_rate_targets(["EURUSD"], ["M1", "H1"])
        assert resolve_rate_tables(None, targets) == [
            "rate_EURUSD__1",
            "rate_EURUSD__16385",
        ]

    def test_resolve_rate_tables_missing_view_with_require_existing_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """Test strict mode rejects databases without managed rate views."""
        db_path = tmp_path / "no-views.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
        targets = build_rate_targets(["EURUSD"], ["M1"])
        with pytest.raises(ValueError, match="No rate compatibility view exists"):
            resolve_rate_tables(db_path, targets, require_existing=True)

    def test_resolve_rate_tables_with_require_existing_resolves_views(
        self,
        tmp_path: Path,
    ) -> None:
        """Test strict mode resolves existing managed rate views."""
        db_path = tmp_path / "strict-views.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            create_rate_compatibility_views(conn)
        targets = build_rate_targets(["EURUSD"], ["M1"])
        assert resolve_rate_tables(db_path, targets, require_existing=True) == [
            "rate_EURUSD__1",
        ]

    def test_resolve_rate_tables_batches_sqlite_metadata(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test resolving multiple targets loads SQLite metadata once."""
        db_path = tmp_path / "batch-rate-tables.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", 16385, "2024-01-01T01:00:00+00:00", 1.1),
                    ("GBPUSD", 1, "2024-01-01T00:00:00+00:00", 1.2),
                ],
            )
            create_rate_compatibility_views(conn)
        counts_spy = mocker.spy(history, "_load_rates_timeframe_counts")
        views_spy = mocker.spy(history, "_load_existing_rate_views")

        targets = build_rate_targets(["EURUSD", "GBPUSD"], ["M1", "H1"])
        assert resolve_rate_tables(db_path, targets) == [
            "rate_EURUSD__M1_1",
            "rate_EURUSD__H1_16385",
            "rate_GBPUSD__1",
            "rate_GBPUSD__16385",
        ]
        assert counts_spy.call_count == 1
        assert views_spy.call_count == 1

    def test_load_rate_series_from_sqlite(self, tmp_path: Path) -> None:
        """Test loading multiple rate series keyed by symbol and timeframe."""
        db_path = tmp_path / "series.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", 1, "2024-01-01T00:01:00+00:00", 1.1),
                ],
            )
            create_rate_compatibility_views(conn)
        targets = build_rate_targets(["EURUSD"], ["M1"])
        result = load_rate_series_from_sqlite(db_path, targets, count=2)
        assert set(result) == {("EURUSD", 1)}
        assert len(result["EURUSD", 1]) == 2

    def test_load_rate_series_by_granularity(self, tmp_path: Path) -> None:
        """Test loading rate series keyed by symbol and granularity name."""
        db_path = tmp_path / "granularity.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.executemany(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                [
                    ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
                    ("EURUSD", 16385, "2024-01-01T00:00:00+00:00", 1.1),
                ],
            )
            create_rate_compatibility_views(conn)

        result = load_rate_series_by_granularity(
            db_path,
            ["EURUSD"],
            ["M1", "H1"],
            count=1,
        )

        assert set(result) == {("EURUSD", "M1"), ("EURUSD", "H1")}

    def test_load_rate_series_by_granularity_explicit_tables(
        self,
        tmp_path: Path,
    ) -> None:
        """Test explicit tables with None-symbol targets key by granularity."""
        db_path = tmp_path / "granularity-explicit.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE custom_view(time TEXT, close REAL)")
            conn.execute(
                "INSERT INTO custom_view(time, close) VALUES (?, ?)",
                ("2024-01-01T00:00:00+00:00", 1.0),
            )

        result = load_rate_series_by_granularity(
            db_path,
            [],
            ["M1"],
            count=1,
            explicit_tables=["custom_view"],
            allow_missing_symbol=True,
        )

        assert set(result) == {(None, "M1")}

    def test_load_rate_series_reuses_path_connection(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test loading from a path opens SQLite once for resolve and reads."""
        db_path = tmp_path / "single-open-series.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
            create_rate_compatibility_views(conn)
        connect_spy = mocker.spy(history.sqlite3, "connect")

        result = load_rate_series_from_sqlite(
            db_path,
            build_rate_targets(["EURUSD"], ["M1"]),
            count=1,
        )

        assert set(result) == {("EURUSD", 1)}
        assert connect_spy.call_count == 1

    def test_load_rate_series_with_explicit_tables(self, tmp_path: Path) -> None:
        """Test explicit tables and None-symbol targets load series."""
        db_path = tmp_path / "explicit.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE custom_view(time TEXT, close REAL)")
            conn.execute(
                "INSERT INTO custom_view(time, close) VALUES (?, ?)",
                ("2024-01-01T00:00:00+00:00", 1.0),
            )
        targets = build_rate_targets([], ["M1"], allow_missing_symbol=True)
        result = load_rate_series_from_sqlite(
            db_path,
            targets,
            count=1,
            explicit_tables=["custom_view"],
        )
        assert set(result) == {(None, 1)}

    @pytest.mark.parametrize(
        ("targets", "count", "explicit_tables", "match"),
        [
            pytest.param(
                build_rate_targets(["EURUSD"], ["M1"]),
                0,
                None,
                "count must be positive",
                id="non-positive-count",
            ),
            pytest.param(
                [],
                1,
                None,
                "At least one rate target",
                id="empty-targets",
            ),
            pytest.param(
                build_rate_targets([], ["M1"], allow_missing_symbol=True),
                1,
                None,
                "without a symbol",
                id="none-symbol-without-explicit",
            ),
            pytest.param(
                [RateTarget("EURUSD", 1), RateTarget("EURUSD", "M1")],
                1,
                None,
                r"Duplicate rate target: \('EURUSD', 1\)",
                id="duplicate-targets",
            ),
            pytest.param(
                [RateTarget("EURUSD", 1), RateTarget("EURUSD", 1)],
                1,
                ["custom_view", "custom_view"],
                r"Duplicate rate target: \('EURUSD', 1\)",
                id="duplicate-targets-with-explicit",
            ),
        ],
    )
    def test_load_rate_series_rejects_invalid_inputs(
        self,
        targets: list[RateTarget],
        count: int,
        explicit_tables: list[str] | None,
        match: str,
    ) -> None:
        """Test load_rate_series_from_sqlite validation runs before opening SQLite."""
        with pytest.raises(ValueError, match=match):
            load_rate_series_from_sqlite(
                "unused.db",
                targets,
                count=count,
                explicit_tables=explicit_tables,
            )

    def test_load_rate_series_requires_existing_managed_views(
        self,
        tmp_path: Path,
    ) -> None:
        """Test loading without explicit tables requires managed rate views."""
        db_path = tmp_path / "no-managed-views.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            )
        targets = build_rate_targets(["EURUSD"], ["M1"])
        with pytest.raises(ValueError, match="No rate compatibility view exists"):
            load_rate_series_from_sqlite(db_path, targets, count=1)
