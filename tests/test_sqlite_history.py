"""Tests for mt5cli.sqlite_history module."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from mt5cli.sqlite_history import (
    DEFAULT_HISTORY_TIMEFRAMES,
    append_dataframe,
    augment_written_columns_from_sqlite,
    build_rate_view_name,
    create_cash_events_view,
    create_history_indexes,
    create_positions_reconstructed_view,
    create_rate_compatibility_views,
    deduplicate_history_tables,
    drop_duplicates_in_table,
    filter_incremental_history_deals_frame,
    filter_trade_history_frame,
    get_history_deals_account_event_start_datetime,
    get_incremental_start_datetime,
    get_table_columns,
    load_incremental_start_datetimes,
    parse_sqlite_timestamp,
    quote_sqlite_identifier,
    record_written_columns,
    resolve_granularity_name,
    resolve_history_datasets,
    resolve_history_tick_flags,
    resolve_history_timeframes,
    write_collected_datasets,
    write_history_dataset,
    write_incremental_datasets,
    write_rates_dataset,
    write_streamed_frame,
)
from mt5cli.utils import TIMEFRAME_MAP, Dataset, IfExists


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


class TestResolveHistorySettings:
    """Tests for history dataset and timeframe resolution."""

    def test_resolve_history_datasets_defaults_and_empty(self) -> None:
        """Test dataset resolution distinguishes None from empty selection."""
        assert resolve_history_datasets(None) == set(Dataset)
        assert resolve_history_datasets(set()) == set()

    def test_resolve_history_timeframes_defaults(self) -> None:
        """Test default timeframes include all fixed MT5 values."""
        resolved = resolve_history_timeframes(None)
        assert len(resolved) == len(DEFAULT_HISTORY_TIMEFRAMES)
        assert 1 in resolved
        assert TIMEFRAME_MAP["H1"] in resolved

    def test_resolve_history_timeframes_deduplicates_aliases(self) -> None:
        """Test duplicate aliases for the same timeframe are deduplicated."""
        assert resolve_history_timeframes(["M1", "1", "H1"]) == [1, TIMEFRAME_MAP["H1"]]

    def test_resolve_history_tick_flags(self) -> None:
        """Test tick flag resolution."""
        assert resolve_history_tick_flags("ALL") == 1
        assert resolve_history_tick_flags(2) == 2

    def test_resolve_granularity_name_falls_back_to_integer(self) -> None:
        """Test unknown timeframe constants fall back to integer text."""
        assert resolve_granularity_name(999) == "999"
        assert resolve_granularity_name(1) == "M1"


class TestParseSqliteTimestamp:
    """Tests for parse_sqlite_timestamp."""

    def test_parses_iso_and_pandas_strings(self) -> None:
        """Test ISO, pandas-compatible, numeric, and datetime values."""
        assert parse_sqlite_timestamp(None) is None
        assert parse_sqlite_timestamp(1_704_067_200) == datetime(2024, 1, 1, tzinfo=UTC)
        assert parse_sqlite_timestamp(
            datetime.fromisoformat("2024-01-01T00:00:00"),
        ) == datetime(2024, 1, 1, tzinfo=UTC)
        assert parse_sqlite_timestamp("Jan 1 2024") == datetime(2024, 1, 1, tzinfo=UTC)
        assert parse_sqlite_timestamp("not-a-datetime") is None
        assert parse_sqlite_timestamp(object()) is None


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

    def test_load_incremental_start_datetimes_requires_timeframe_column(
        self,
        tmp_path: Path,
    ) -> None:
        """Test rates tables without timeframe fail fast during incremental resume."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "legacy-rates.db") as conn:
            conn.execute("CREATE TABLE rates(symbol TEXT, time TEXT, open REAL)")
            conn.execute(
                "INSERT INTO rates(symbol, time, open) VALUES (?, ?, ?)",
                ("EURUSD", "2024-01-02T00:00:00+00:00", 1.0),
            )
            with pytest.raises(ValueError, match="timeframe"):
                load_incremental_start_datetimes(
                    conn,
                    Dataset.rates,
                    symbols=["EURUSD"],
                    timeframes=[1],
                    fallback_start=fallback,
                )

    def test_load_incremental_start_skips_unparseable_max_time(
        self,
        tmp_path: Path,
    ) -> None:
        """Test grouped resume ignores rows whose MAX(time) cannot be parsed."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "bad-max-time.db") as conn:
            conn.execute("CREATE TABLE ticks(symbol TEXT, time TEXT)")
            conn.execute(
                "INSERT INTO ticks(symbol, time) VALUES (?, ?)",
                ("EURUSD", "not-a-datetime"),
            )
            starts = load_incremental_start_datetimes(
                conn,
                Dataset.ticks,
                symbols=["EURUSD"],
                fallback_start=fallback,
            )
        assert starts["EURUSD", None] == fallback

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

    def test_load_incremental_start_skips_unparseable_rates_max_time(
        self,
        tmp_path: Path,
    ) -> None:
        """Test grouped rates resume ignores unparseable MAX(time) values."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "bad-rates-max-time.db") as conn:
            conn.execute(
                "CREATE TABLE rates("
                " symbol TEXT, timeframe INTEGER, time TEXT, open REAL)",
            )
            conn.execute(
                "INSERT INTO rates(symbol, timeframe, time, open) VALUES (?, ?, ?, ?)",
                ("EURUSD", 1, "not-a-datetime", 1.0),
            )
            starts = load_incremental_start_datetimes(
                conn,
                Dataset.rates,
                symbols=["EURUSD"],
                timeframes=[1],
                fallback_start=fallback,
            )
        assert starts["EURUSD", 1] == fallback


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

    def test_drop_duplicates_rejects_invalid_identifiers(self) -> None:
        """Test invalid table or column names raise ValueError."""
        cursor = sqlite3.connect(":memory:").cursor()
        with pytest.raises(ValueError, match="Invalid table name"):
            drop_duplicates_in_table(cursor, "bad table", ["id"])
        with pytest.raises(ValueError, match="Invalid column names"):
            drop_duplicates_in_table(cursor, "rates", ["bad column"])

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
                        (
                            "symbol = ? AND timeframe = ? AND time >= ?",
                            ("EURUSD", 1, boundary),
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

    def test_account_event_start_uses_type_column(self, tmp_path: Path) -> None:
        """Test account-event start uses non-trade deal types when available."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "account-start-type.db") as conn:
            conn.execute(
                "CREATE TABLE history_deals("
                " ticket INTEGER, symbol TEXT, time TEXT, type INTEGER)",
            )
            conn.executemany(
                "INSERT INTO history_deals(ticket, symbol, time, type)"
                " VALUES (?, ?, ?, ?)",
                [
                    (1, "EURUSD", "2024-01-05T00:00:00+00:00", 0),
                    (2, "", "2024-01-08T00:00:00+00:00", 2),
                ],
            )
            assert get_history_deals_account_event_start_datetime(
                conn,
                fallback_start=fallback,
            ) == datetime(2024, 1, 8, tzinfo=UTC)

    def test_account_event_start_falls_back_to_empty_symbol(
        self, tmp_path: Path
    ) -> None:
        """Test account-event start uses empty symbols when type is missing."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "account-start-symbol.db") as conn:
            conn.execute(
                "CREATE TABLE history_deals(ticket INTEGER, symbol TEXT, time TEXT)",
            )
            conn.executemany(
                "INSERT INTO history_deals(ticket, symbol, time) VALUES (?, ?, ?)",
                [
                    (1, "EURUSD", "2024-01-05T00:00:00+00:00"),
                    (2, "", "2024-01-07T00:00:00+00:00"),
                ],
            )
            assert get_history_deals_account_event_start_datetime(
                conn,
                fallback_start=fallback,
            ) == datetime(2024, 1, 7, tzinfo=UTC)

    def test_account_event_start_without_identifying_columns(
        self,
        tmp_path: Path,
    ) -> None:
        """Test account-event start falls back when type and symbol are missing."""
        fallback = datetime(2024, 1, 1, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "account-start-fallback.db") as conn:
            conn.execute("CREATE TABLE history_deals(ticket INTEGER, time TEXT)")
            conn.execute(
                "INSERT INTO history_deals(ticket, time) VALUES (?, ?)",
                (1, "2024-01-05T00:00:00+00:00"),
            )
            assert (
                get_history_deals_account_event_start_datetime(
                    conn,
                    fallback_start=fallback,
                )
                == fallback
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

    def test_filter_incremental_excludes_symbolized_account_events_from_trade_cursor(
        self,
    ) -> None:
        """Test account events are not kept by per-symbol trade cursors."""
        frame = pd.DataFrame({
            "ticket": [1],
            "symbol": ["EURUSD"],
            "time": ["2024-01-05T00:00:00+00:00"],
            "type": [2],
        })
        filtered = filter_incremental_history_deals_frame(
            frame,
            ["EURUSD"],
            {"EURUSD": datetime(2024, 1, 1, tzinfo=UTC)},
            datetime(2024, 1, 10, tzinfo=UTC),
        )
        assert filtered.empty

    def test_filter_incremental_rejects_rows_without_parseable_time(self) -> None:
        """Test incremental filtering drops rows when time cannot be parsed."""
        frame = pd.DataFrame({
            "ticket": [1],
            "symbol": ["EURUSD"],
            "time": ["not-a-datetime"],
            "type": [0],
        })
        filtered = filter_incremental_history_deals_frame(
            frame,
            ["EURUSD"],
            {"EURUSD": datetime(2024, 1, 1, tzinfo=UTC)},
            datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert filtered.empty

    def test_filter_incremental_keeps_account_events_without_symbol_column(
        self,
    ) -> None:
        """Test account events are kept when history_deals has no symbol column."""
        frame = pd.DataFrame({
            "ticket": [1],
            "time": ["2024-01-03T00:00:00+00:00"],
            "type": [2],
        })
        filtered = filter_incremental_history_deals_frame(
            frame,
            ["EURUSD"],
            {"EURUSD": datetime(2024, 1, 10, tzinfo=UTC)},
            datetime(2024, 1, 2, tzinfo=UTC),
        )
        assert filtered["ticket"].tolist() == [1]

    def test_filter_incremental_skips_trade_rows_without_symbol_column(self) -> None:
        """Test trade rows are excluded when symbol column is unavailable."""
        frame = pd.DataFrame({
            "ticket": [1],
            "time": ["2024-01-03T00:00:00+00:00"],
        })
        filtered = filter_incremental_history_deals_frame(
            frame,
            ["EURUSD"],
            {"EURUSD": datetime(2024, 1, 1, tzinfo=UTC)},
            datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert filtered.empty

    def test_filter_incremental_rejects_rows_without_time_column(self) -> None:
        """Test incremental filtering drops rows when time column is missing."""
        frame = pd.DataFrame({
            "ticket": [1],
            "symbol": ["EURUSD"],
            "type": [0],
        })
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
                logger="mt5cli.sqlite_history",
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
            caplog.at_level(logging.WARNING, logger="mt5cli.sqlite_history"),
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

    def test_resolve_history_tick_flags_invalid(self) -> None:
        """Test invalid tick flags raise ValueError."""
        with pytest.raises(ValueError, match="Invalid tick flags"):
            resolve_history_tick_flags("BAD")

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

    def test_records_account_event_dedup_scope_without_type_column(
        self,
        tmp_path: Path,
    ) -> None:
        """Test account-event dedup scope falls back to empty symbols."""
        client = MagicMock()
        client.history_deals_get_as_df.return_value = pd.DataFrame({
            "ticket": [1],
            "symbol": [""],
            "time": ["2024-01-02T00:00:00+00:00"],
        })
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 3, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "legacy-deals.db") as conn:
            conn.execute(
                "CREATE TABLE history_deals( ticket INTEGER, symbol TEXT, time TEXT)",
            )
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

    def test_skips_symbol_dedup_scope_when_symbol_column_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """Test account-event writes skip per-symbol dedup without symbol column."""
        client = MagicMock()
        client.history_deals_get_as_df.return_value = pd.DataFrame({
            "ticket": [1],
            "time": ["2024-01-02T00:00:00+00:00"],
            "type": [2],
        })
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 3, tzinfo=UTC)
        with sqlite3.connect(tmp_path / "no-symbol-deals.db") as conn:
            conn.execute(
                "CREATE TABLE history_deals(ticket INTEGER, time TEXT, type INTEGER)",
            )
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
