"""Tests for canonical stable rate APIs."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import inspect
import sqlite3
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from mt5cli.contract import HistoryClient

from mt5cli import rates
from mt5cli.history import RateTarget, create_history_indexes
from mt5cli.utils import Dataset


def test_update_history_does_not_annotate_client_as_object_or_mt5dataclient() -> None:
    """update_history uses the history protocol rather than the raw client."""
    annotations = inspect.get_annotations(rates.update_history, eval_str=False)
    client_annotation = str(annotations["client"])
    assert client_annotation != "object"
    assert "Mt5DataClient" not in client_annotation


def _create_canonical_database(
    path: Path,
    rows: list[tuple[str, int, str, float]],
) -> None:
    """Create a minimal normalized rates database for loader tests."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)",
        )
        conn.executemany(
            "INSERT INTO rates(symbol, timeframe, time, close) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.execute(
            "CREATE TABLE _mt5cli_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        )
        conn.execute(
            "INSERT INTO _mt5cli_metadata(key, value) VALUES (?, ?)",
            ("rates_timestamp_contract", "pdmt5-wall-clock-v1"),
        )


def _create_custom_database(path: Path) -> None:
    """Create a minimal explicit-table database for compatibility tests."""
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE custom_rates(time TEXT, close REAL)")
        conn.execute(
            "INSERT INTO custom_rates(time, close) VALUES (?, ?)",
            ("2024-01-01T00:00:00+00:00", 1.0),
        )


def test_canonical_loader_keeps_caller_connection_open_and_sorts_rows(
    tmp_path: Path,
) -> None:
    """Canonical loading uses the normalized table and preserves open connections."""
    db_path = tmp_path / "canonical.db"
    _create_canonical_database(
        db_path,
        [
            ("EURUSD", 1, "2024-01-01T00:01:00+00:00", 1.1),
            ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
        ],
    )

    with sqlite3.connect(db_path) as conn:
        result = rates.load_rate_series_from_sqlite(
            conn,
            [RateTarget("EURUSD", 1)],
            count=2,
        )
        assert list(result["EURUSD", 1]["close"]) == [1.0, 1.1]
        conn.execute("SELECT 1")


def test_canonical_loader_applies_count_after_timestamp_normalization(
    tmp_path: Path,
) -> None:
    """Canonical loading selects the newest rows by normalized UTC time."""
    db_path = tmp_path / "mixed-count-canonical.db"
    _create_canonical_database(
        db_path,
        [
            ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            ("EURUSD", 1, "2024-01-01T00:30:00+00:00", 2.0),
            ("EURUSD", 1, "2024-01-01T01:00:00+02:00", 3.0),
        ],
    )

    result = rates.load_rate_series_from_sqlite(
        db_path,
        [RateTarget("EURUSD", 1)],
        count=2,
    )

    assert list(result["EURUSD", 1]["close"]) == [1.0, 2.0]


def test_canonical_loader_returns_empty_series_from_path(tmp_path: Path) -> None:
    """A valid canonical table with no matching rows returns an empty frame."""
    db_path = tmp_path / "empty-canonical.db"
    _create_canonical_database(db_path, [])

    result = rates.load_rate_series_from_sqlite(
        db_path,
        [RateTarget("EURUSD", 1)],
        count=1,
    )

    assert result["EURUSD", 1].empty


def test_canonical_loader_rejects_missing_database(tmp_path: Path) -> None:
    """Canonical loading refuses to create a missing database path."""
    with pytest.raises(ValueError, match="SQLite database not found"):
        rates.load_rate_series_from_sqlite(
            tmp_path / "missing.db",
            [RateTarget("EURUSD", 1)],
            count=1,
        )


def test_canonical_loader_translates_open_errors(
    mocker: MockerFixture,
) -> None:
    """Canonical loading translates read-only connection errors to ValueError."""
    mocker.patch(
        "mt5cli.rates._open_existing_sqlite_database",
        side_effect=sqlite3.OperationalError("database is locked"),
    )

    with pytest.raises(ValueError, match="database could not be opened"):
        rates.load_rate_series_from_sqlite(
            "history.db",
            [RateTarget("EURUSD", 1)],
            count=1,
        )


def test_canonical_loader_rejects_missing_rates_table(tmp_path: Path) -> None:
    """Canonical loading translates SQLite schema errors to ValueError."""
    db_path = tmp_path / "invalid-canonical.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE unrelated(value INTEGER)")

    with pytest.raises(ValueError, match="canonical rates table"):
        rates.load_rate_series_from_sqlite(
            db_path,
            [RateTarget("EURUSD", 1)],
            count=1,
        )


def test_canonical_loader_rejects_rates_table_without_time(tmp_path: Path) -> None:
    """Canonical loading translates a missing time column to ValueError."""
    db_path = tmp_path / "missing-canonical-time.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE rates(symbol TEXT, timeframe INTEGER, close REAL)")

    with pytest.raises(ValueError, match="required time column"):
        rates.load_rate_series_from_sqlite(
            db_path,
            [RateTarget("EURUSD", 1)],
            count=1,
        )


def test_canonical_loader_rejects_target_without_symbol() -> None:
    """Canonical normalized loading requires a symbol on every target."""
    with (
        sqlite3.connect(":memory:") as conn,
        pytest.raises(
            ValueError,
            match="symbol is required",
        ),
    ):
        rates.load_rate_series_from_sqlite(
            conn,
            [RateTarget(None, 1)],
            count=1,
        )


@pytest.mark.parametrize(
    ("targets", "count", "match"),
    [
        pytest.param([RateTarget("EURUSD", 1)], 0, "count must be positive"),
        pytest.param(None, 1, "targets are required"),
        pytest.param([], 1, "At least one rate target"),
    ],
)
def test_canonical_loader_rejects_invalid_inputs(
    targets: list[RateTarget] | None,
    count: int,
    match: str,
) -> None:
    """Canonical loading validates inputs before opening SQLite."""
    with pytest.raises(ValueError, match=match):
        rates.load_rate_series_from_sqlite("unused.db", targets, count=count)


def test_canonical_loader_rejects_duplicate_targets() -> None:
    """Canonical loading rejects duplicate symbol/timeframe pairs."""
    targets = [RateTarget("EURUSD", 1), RateTarget("EURUSD", 1)]

    with pytest.raises(ValueError, match="Duplicate rate targets"):
        rates.load_rate_series_from_sqlite("unused.db", targets, count=1)


def test_explicit_table_loading_delegates_to_legacy_loader(tmp_path: Path) -> None:
    """Explicit table loading remains available through the legacy implementation."""
    db_path = tmp_path / "custom.db"
    _create_custom_database(db_path)

    frame = rates.load_rate_series_from_sqlite(db_path, table="custom_rates", count=1)

    assert isinstance(frame, pd.DataFrame)
    assert list(frame["close"]) == [1.0]


def test_explicit_tables_without_targets_use_legacy_validation(tmp_path: Path) -> None:
    """Explicit table lists still require target descriptors."""
    db_path = tmp_path / "custom-without-targets.db"
    _create_custom_database(db_path)

    with pytest.raises(ValueError, match="targets are required"):
        rates.load_rate_series_from_sqlite(
            db_path,
            count=1,
            explicit_tables=["custom_rates"],
        )


def test_explicit_target_loading_rejects_missing_count(tmp_path: Path) -> None:
    """Target-based explicit loading rejects an omitted row count."""
    db_path = tmp_path / "custom-missing-count.db"
    _create_custom_database(db_path)
    loader = cast("Any", rates.load_rate_series_from_sqlite)

    with pytest.raises(ValueError, match="count must be positive"):
        loader(
            db_path,
            [RateTarget(None, 1)],
            count=None,
            explicit_tables=["custom_rates"],
        )


def test_explicit_target_loading_supports_missing_symbols(tmp_path: Path) -> None:
    """Explicit table names can load symbol-less targets."""
    db_path = tmp_path / "custom-target.db"
    _create_custom_database(db_path)

    result = rates.load_rate_series_from_sqlite(
        db_path,
        [RateTarget(None, 1)],
        count=1,
        explicit_tables=["custom_rates"],
    )

    assert set(result) == {(None, 1)}


def test_load_by_granularity_reads_canonical_rates_without_views(
    tmp_path: Path,
) -> None:
    """Granularity loading queries canonical rates without compatibility views."""
    db_path = tmp_path / "granularity-canonical.db"
    _create_canonical_database(
        db_path,
        [
            ("EURUSD", 1, "2024-01-01T00:00:00+00:00", 1.0),
            ("EURUSD", 16385, "2024-01-01T00:00:00+00:00", 2.0),
        ],
    )

    result = rates.load_rate_series_by_granularity(
        db_path,
        ["EURUSD"],
        ["M1", "H1"],
        count=1,
    )

    assert set(result) == {("EURUSD", "M1"), ("EURUSD", "H1")}


def test_load_by_granularity_rejects_single_frame(
    mocker: MockerFixture,
) -> None:
    """Granularity loading rejects a table-style loader result."""
    mocker.patch(
        "mt5cli.rates.load_rate_series_from_sqlite", return_value=pd.DataFrame()
    )

    with pytest.raises(TypeError, match="Expected multiple rate series"):
        rates.load_rate_series_by_granularity(
            "unused.db",
            ["EURUSD"],
            ["M1"],
            count=1,
        )


def test_throttled_updater_uses_canonical_backend_and_preserves_custom_backend(
    mocker: MockerFixture,
) -> None:
    """The canonical updater defaults to rates.update_history exactly once."""
    default_updater = rates.ThrottledHistoryUpdater(output="history.db")
    assert default_updater.update_backend is rates.update_history

    custom_backend = mocker.Mock()
    custom_updater = rates.ThrottledHistoryUpdater(
        output="history.db",
        update_backend=custom_backend,
        interval_seconds=5,
        suppress_errors=True,
    )
    assert custom_updater.update_backend is custom_backend


def test_canonical_loader_returns_naive_datetime_index(tmp_path: Path) -> None:
    """Canonical naive server-clock series keep a DatetimeIndex without a zone."""
    db_path = tmp_path / "naive-index.db"
    _create_canonical_database(
        db_path,
        [
            ("EURUSD", 1, "2024-01-01T00:01:00", 1.1),
            ("EURUSD", 1, "2024-01-01T00:00:00", 1.0),
        ],
    )
    frame = rates.load_rate_series_from_sqlite(
        db_path,
        [RateTarget("EURUSD", 1)],
        count=2,
    )["EURUSD", 1]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.name == "time"
    assert frame.index.tz is None
    assert list(frame["close"]) == [1.0, 1.1]


def test_canonical_loader_normalizes_aware_index_to_utc(tmp_path: Path) -> None:
    """Canonical aware instants are represented by a UTC DatetimeIndex."""
    db_path = tmp_path / "aware-index.db"
    _create_canonical_database(
        db_path,
        [
            ("EURUSD", 1, "2024-01-01T09:00:00+09:00", 1.0),
            ("EURUSD", 1, "2024-01-01T00:01:00+00:00", 1.1),
        ],
    )
    frame = rates.load_rate_series_from_sqlite(
        db_path,
        [RateTarget("EURUSD", 1)],
        count=2,
    )["EURUSD", 1]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert str(frame.index.tz) == "UTC"
    assert list(frame["close"]) == [1.0, 1.1]


def test_canonical_loader_applies_limit_in_sqlite(tmp_path: Path) -> None:
    """Canonical loading bounds rows in SQLite before pandas materialization."""
    db_path = tmp_path / "limited.db"
    _create_canonical_database(
        db_path,
        [
            ("EURUSD", 1, f"2024-01-01T00:{minute:02d}:00", float(minute))
            for minute in range(10)
        ],
    )
    statements: list[str] = []
    with sqlite3.connect(db_path) as conn:
        conn.set_trace_callback(statements.append)
        frame = rates.load_rate_series_from_sqlite(
            conn,
            [RateTarget("EURUSD", 1)],
            count=2,
        )["EURUSD", 1]
    assert list(frame["close"]) == [8.0, 9.0]
    assert any("LIMIT 2" in statement for statement in statements)


def test_history_indexes_normalized_rate_cursor(tmp_path: Path) -> None:
    """History index creation accelerates normalized rate cursor queries."""
    db_path = tmp_path / "indexed.db"
    _create_canonical_database(
        db_path,
        [("EURUSD", 1, "2024-01-01T00:00:00", 1.0)],
    )

    with sqlite3.connect(db_path) as conn:
        create_history_indexes(
            conn,
            {Dataset.rates: {"symbol", "timeframe", "time"}},
        )
        plan = conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT time FROM rates WHERE "
            "COALESCE(strftime('%Y-%m-%dT%H:%M:%f', time), "
            "strftime('%Y-%m-%dT%H:%M:%f', time, 'unixepoch')) IS NOT NULL "
            "AND symbol = ? AND timeframe = ? ORDER BY "
            "COALESCE(strftime('%Y-%m-%dT%H:%M:%f', time), "
            "strftime('%Y-%m-%dT%H:%M:%f', time, 'unixepoch')) DESC, "
            "ROWID DESC LIMIT 1",
            ("EURUSD", 1),
        ).fetchall()
    assert any("idx_rates_symbol_timeframe_history_cursor" in str(row) for row in plan)


def _canonical_db(path: Path, *, mark_contract: bool = True) -> None:
    """Create the minimal canonical rate schema."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
        )
        if mark_contract:
            conn.execute(
                "CREATE TABLE _mt5cli_metadata ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO _mt5cli_metadata VALUES (?, ?)",
                (
                    "rates_timestamp_contract",
                    "pdmt5-wall-clock-v1",
                ),
            )


def test_canonical_loader_rejects_mixed_timestamp_awareness(tmp_path: Path) -> None:
    """One canonical series cannot mix server wall clocks with aware instants."""
    db_path = tmp_path / "mixed.db"
    _canonical_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO rates VALUES (?, ?, ?, ?)",
            [
                ("EURUSD", 1, "2024-01-01T00:00:00", 1.0),
                ("EURUSD", 1, "2024-01-01T00:01:00+00:00", 1.1),
            ],
        )

    with pytest.raises(ValueError, match="cannot mix timezone-naive"):
        rates.load_rate_series_from_sqlite(
            db_path,
            [RateTarget("EURUSD", 1)],
            count=2,
        )


@pytest.mark.parametrize(
    "timestamp",
    [
        "2024-01-01T00:00:00+00:00",
        "2024-01-01T00:00:00 UTC",
        "2024-01-01T00:00:00 GMT",
    ],
    ids=["numeric-offset", "utc-name", "gmt-name"],
)
def test_canonical_loader_rejects_unversioned_aware_timestamps(
    tmp_path: Path,
    timestamp: str,
) -> None:
    """Managed reads reject aware data without the current timestamp marker."""
    db_path = tmp_path / "unversioned.db"
    _canonical_db(db_path, mark_contract=False)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO rates VALUES (?, ?, ?, ?)",
            ("EURUSD", 1, timestamp, 1.0),
        )

    with pytest.raises(ValueError, match=r"unversioned timezone-aware"):
        rates.load_rate_series_from_sqlite(
            db_path,
            [RateTarget("EURUSD", 1)],
            count=1,
        )


def test_canonical_loader_defends_against_missing_time_in_query_result(
    mocker: MockerFixture,
) -> None:
    """A malformed query result still fails with the stable schema error."""
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
        )
        mocker.patch(
            "mt5cli.rates.pd.read_sql_query",
            return_value=pd.DataFrame({"close": [1.0]}),
        )
        with pytest.raises(ValueError, match="required time column"):
            rates.load_rate_series_from_sqlite(
                conn,
                [RateTarget("EURUSD", 1)],
                count=1,
            )


def test_explicit_tables_must_match_target_count() -> None:
    """Explicit custom table names map one-to-one to requested targets."""
    with pytest.raises(ValueError, match="Expected 1 explicit table"):
        rates.load_rate_series_from_sqlite(
            "unused.db",
            [RateTarget("EURUSD", 1)],
            count=1,
            explicit_tables=[],
        )


def test_explicit_tables_reject_duplicate_target_keys() -> None:
    """Explicit custom tables preserve unique target keys."""
    targets = [RateTarget("EURUSD", 1), RateTarget("EURUSD", 1)]
    with pytest.raises(ValueError, match="Duplicate rate targets"):
        rates.load_rate_series_from_sqlite(
            "unused.db",
            targets,
            count=1,
            explicit_tables=["one", "two"],
        )


def test_explicit_table_sql_errors_are_translated(mocker: MockerFixture) -> None:
    """SQLite errors from explicit tables use the stable ValueError surface."""
    mocker.patch(
        "mt5cli.rates._load_explicit_rate_series",
        side_effect=sqlite3.OperationalError("locked"),
    )
    with pytest.raises(ValueError, match="explicit rate table"):
        rates.load_rate_series_from_sqlite(
            "unused.db",
            table="custom_rates",
            count=1,
        )


def test_canonical_query_sql_errors_are_translated(mocker: MockerFixture) -> None:
    """Canonical SQLite errors use the stable ValueError surface."""
    with sqlite3.connect(":memory:") as conn:
        mocker.patch(
            "mt5cli.rates._load_canonical_rate_target",
            side_effect=sqlite3.OperationalError("locked"),
        )
        with pytest.raises(ValueError, match="canonical rates table"):
            rates.load_rate_series_from_sqlite(
                conn,
                [RateTarget("EURUSD", 1)],
                count=1,
            )


def test_update_history_forwards_canonical_arguments(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """The stable update wrapper forwards without compatibility-view options."""
    backend = mocker.patch("mt5cli.rates._legacy_update_history")
    client = mocker.MagicMock()
    output = tmp_path / "history.db"

    rates.update_history(
        client=client,
        output=output,
        symbols=["EURUSD"],
        date_to="2024-01-02",
    )

    backend.assert_called_once_with(
        client=client,
        output=output,
        symbols=["EURUSD"],
        datasets=None,
        timeframes=None,
        flags="ALL",
        lookback_hours=24.0,
        date_to="2024-01-02",
        deduplicate=True,
        with_views=False,
        include_account_events=True,
    )


def test_update_history_with_config_forwards_canonical_arguments(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """The managed-session wrapper forwards without compatibility-view options."""
    backend = mocker.patch("mt5cli.rates._legacy_update_history_with_config")
    output = tmp_path / "history.db"

    rates.update_history_with_config(
        output=output,
        symbols=["EURUSD"],
        date_to="2024-01-02",
    )

    backend.assert_called_once_with(
        output=output,
        symbols=["EURUSD"],
        config=None,
        datasets=None,
        timeframes=None,
        flags="ALL",
        lookback_hours=24.0,
        date_to="2024-01-02",
        deduplicate=True,
        with_views=False,
        include_account_events=True,
    )


def _create_rates_table(path: Path, *times: str) -> None:
    """Create a minimal canonical rates table with the requested timestamps."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
        )
        conn.executemany(
            "INSERT INTO rates VALUES ('EURUSD', 1, ?, 1.0)",
            [(value,) for value in times],
        )


def _set_contract(path: Path, value: str | None) -> None:
    """Create the metadata table and optionally set the timestamp contract."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _mt5cli_metadata ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        if value is not None:
            conn.execute(
                "INSERT INTO _mt5cli_metadata VALUES (?, ?)",
                (rates._RATE_TIMESTAMP_CONTRACT_KEY, value),
            )


def test_read_rate_timestamp_contract_states(tmp_path: Path) -> None:
    """Contract lookup distinguishes absent tables, absent keys, and values."""
    path = tmp_path / "contract.db"
    with sqlite3.connect(path) as conn:
        assert rates._read_rate_timestamp_contract(conn) is None
        conn.execute(
            "CREATE TABLE _mt5cli_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        assert rates._read_rate_timestamp_contract(conn) is None
        conn.execute(
            "INSERT INTO _mt5cli_metadata VALUES (?, ?)",
            (
                rates._RATE_TIMESTAMP_CONTRACT_KEY,
                rates._RATE_TIMESTAMP_CONTRACT_VALUE,
            ),
        )
        assert (
            rates._read_rate_timestamp_contract(conn)
            == rates._RATE_TIMESTAMP_CONTRACT_VALUE
        )


@pytest.mark.parametrize(
    ("table", "column", "create_sql", "insert_sql"),
    [
        (
            "rates",
            "time",
            'CREATE TABLE "rates" ("time" TEXT)',
            'INSERT INTO "rates" ("time") VALUES (?)',
        ),
        (
            "ticks",
            "time",
            'CREATE TABLE "ticks" ("time" TEXT)',
            'INSERT INTO "ticks" ("time") VALUES (?)',
        ),
        (
            "history_orders",
            "time_setup",
            'CREATE TABLE "history_orders" ("time_setup" TEXT)',
            'INSERT INTO "history_orders" ("time_setup") VALUES (?)',
        ),
        (
            "history_orders",
            "time_done",
            'CREATE TABLE "history_orders" ("time_done" TEXT)',
            'INSERT INTO "history_orders" ("time_done") VALUES (?)',
        ),
        (
            "history_deals",
            "time",
            'CREATE TABLE "history_deals" ("time" TEXT)',
            'INSERT INTO "history_deals" ("time") VALUES (?)',
        ),
        (
            "symbols",
            "time",
            'CREATE TABLE "symbols" ("time" TEXT)',
            'INSERT INTO "symbols" ("time") VALUES (?)',
        ),
    ],
)
def test_aware_timestamp_detection(
    tmp_path: Path,
    table: str,
    column: str,
    create_sql: str,
    insert_sql: str,
) -> None:
    """Managed timestamp detection covers every persisted text time column."""
    path = tmp_path / f"awareness-{table}-{column}.db"
    with sqlite3.connect(path) as conn:
        conn.execute(create_sql)
        assert not rates._managed_history_contains_aware_timestamp_text(conn)
        conn.execute(
            insert_sql,
            ("2024-01-01T09:30:00",),
        )
        assert not rates._managed_history_contains_aware_timestamp_text(conn)
        conn.execute(
            insert_sql,
            ("2024-01-01T00:30:00+00:00",),
        )
        assert rates._managed_history_contains_aware_timestamp_text(conn)


def test_validate_rate_timestamp_contract_states(tmp_path: Path) -> None:
    """Validation accepts safe states and rejects ambiguous or unknown contracts."""
    empty = tmp_path / "empty.db"
    with sqlite3.connect(empty) as conn:
        rates._validate_rate_timestamp_contract(conn)

    naive = tmp_path / "naive.db"
    _create_rates_table(naive, "2024-01-01T09:30:00")
    with sqlite3.connect(naive) as conn:
        rates._validate_rate_timestamp_contract(conn)

    current = tmp_path / "current.db"
    _create_rates_table(current, "2024-01-01T00:30:00+00:00")
    _set_contract(current, rates._RATE_TIMESTAMP_CONTRACT_VALUE)
    with sqlite3.connect(current) as conn:
        rates._validate_rate_timestamp_contract(conn)

    unknown = tmp_path / "unknown.db"
    _create_rates_table(unknown, "2024-01-01T09:30:00")
    _set_contract(unknown, "future-contract")
    with (
        sqlite3.connect(unknown) as conn,
        pytest.raises(ValueError, match=r"Unsupported .*timestamp contract"),
    ):
        rates._validate_rate_timestamp_contract(conn)

    legacy = tmp_path / "legacy.db"
    _create_rates_table(legacy, "2024-01-01T00:30:00+00:00")
    with (
        sqlite3.connect(legacy) as conn,
        pytest.raises(ValueError, match=r"mt5cli <= 1\.4\.1"),
    ):
        rates._validate_rate_timestamp_contract(conn)


def test_validate_existing_rate_database(tmp_path: Path) -> None:
    """Missing outputs are ignored while existing safe databases are validated."""
    rates._validate_existing_rate_database(tmp_path / "missing.db")
    path = tmp_path / "existing.db"
    _create_rates_table(path, "2024-01-01T09:30:00")
    rates._validate_existing_rate_database(path)


def test_mark_rate_timestamp_contract_states(tmp_path: Path) -> None:
    """Successful writes mark managed history without masking unknown contracts."""
    rates._mark_rate_timestamp_contract(tmp_path / "missing.db")

    no_rates = tmp_path / "no-rates.db"
    with sqlite3.connect(no_rates):
        pass
    rates._mark_rate_timestamp_contract(no_rates)

    path = tmp_path / "rates.db"
    _create_rates_table(path, "2024-01-01T09:30:00")
    rates._mark_rate_timestamp_contract(path)
    rates._mark_rate_timestamp_contract(path)
    with sqlite3.connect(path) as conn:
        assert (
            rates._read_rate_timestamp_contract(conn)
            == rates._RATE_TIMESTAMP_CONTRACT_VALUE
        )

    non_rates = tmp_path / "non-rates.db"
    with sqlite3.connect(non_rates) as conn:
        conn.execute("CREATE TABLE history_deals(time TEXT)")
    rates._mark_rate_timestamp_contract(non_rates)
    with sqlite3.connect(non_rates) as conn:
        assert (
            rates._read_rate_timestamp_contract(conn)
            == rates._RATE_TIMESTAMP_CONTRACT_VALUE
        )

    unknown = tmp_path / "unknown.db"
    _create_rates_table(unknown, "2024-01-01T09:30:00")
    _set_contract(unknown, "future-contract")
    with pytest.raises(ValueError, match=r"Unsupported .*timestamp contract"):
        rates._mark_rate_timestamp_contract(unknown)


def test_update_history_fails_before_touching_legacy_database(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """Incremental updates never mix new wall clocks into legacy aware storage."""
    path = tmp_path / "legacy.db"
    _create_rates_table(path, "2024-01-01T00:30:00+00:00")
    backend = mocker.patch("mt5cli.rates._legacy_update_history")

    with pytest.raises(ValueError, match="Recreate or explicitly migrate"):
        rates.update_history(
            client=cast("HistoryClient", object()),
            output=path,
            symbols=["EURUSD"],
            date_to="2024-01-02",
        )

    backend.assert_not_called()


def test_update_history_rejects_legacy_non_rates_database(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """Incremental updates reject legacy aware timestamps without rates."""
    path = tmp_path / "legacy-deals.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE history_deals(time TEXT)")
        conn.execute(
            "INSERT INTO history_deals VALUES (?)",
            ("2024-01-01T00:30:00+00:00",),
        )
    backend = mocker.patch("mt5cli.rates._legacy_update_history")

    with pytest.raises(ValueError, match=r"unversioned timezone-aware"):
        rates.update_history(
            client=cast("HistoryClient", object()),
            output=path,
            symbols=["EURUSD"],
            date_to="2024-01-02",
        )

    backend.assert_not_called()


def test_update_history_marks_successful_new_database(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """A successful canonical update records the timestamp representation."""
    path = tmp_path / "new.db"

    def write_rates(**_: object) -> None:
        _create_rates_table(path, "2024-01-01T09:30:00")

    mocker.patch("mt5cli.rates._legacy_update_history", side_effect=write_rates)
    rates.update_history(
        client=cast("HistoryClient", object()),
        output=path,
        symbols=["EURUSD"],
        date_to="2024-01-02",
    )

    with sqlite3.connect(path) as conn:
        assert (
            rates._read_rate_timestamp_contract(conn)
            == rates._RATE_TIMESTAMP_CONTRACT_VALUE
        )
