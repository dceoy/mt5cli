"""Tests for canonical stable rate APIs."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

from mt5cli import rates
from mt5cli.history import RateTarget
from mt5cli.utils import Dataset


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


def test_canonical_loader_sorts_mixed_timezone_rows(tmp_path: Path) -> None:
    """Canonical loading orders mixed naive and aware timestamps safely."""
    db_path = tmp_path / "mixed-canonical.db"
    _create_canonical_database(
        db_path,
        [
            ("EURUSD", 1, "2024-01-01T00:00:00", 1.0),
            ("EURUSD", 1, "2024-01-01T00:00:00+09:00", 2.0),
        ],
    )

    result = rates.load_rate_series_from_sqlite(
        db_path,
        [RateTarget("EURUSD", 1)],
        count=2,
    )
    times = [cast("pd.Timestamp", value) for value in result["EURUSD", 1]["time"]]

    assert times[0] == pd.Timestamp("2023-12-31T15:00:00+00:00")
    assert times[0].tzinfo == UTC
    assert times[1] == pd.Timestamp("2024-01-01")
    assert times[1].tzinfo is None


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


def test_explicit_table_loading_translates_sqlite_errors(
    mocker: MockerFixture,
) -> None:
    """Explicit-table SQLite errors are exposed as stable ValueErrors."""
    mocker.patch(
        "mt5cli.rates._legacy_load_rate_series_from_sqlite",
        side_effect=sqlite3.OperationalError("database is locked"),
    )

    with pytest.raises(ValueError, match="explicit rate table"):
        rates.load_rate_series_from_sqlite(
            "history.db",
            table="custom_rates",
            count=1,
        )


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


def test_update_history_forces_canonical_options_and_cleans_views(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """The stable update wrapper disables and cleans legacy rate views."""
    db_path = tmp_path / "update.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT)")
        conn.execute(
            "CREATE VIEW rate_EURUSD__M1_1 AS SELECT * FROM rates",
        )
        conn.execute("CREATE VIEW unrelated_view AS SELECT 1 AS value")

    legacy_update = mocker.patch("mt5cli.rates._legacy_update_history")
    client = mocker.MagicMock()
    date_to = datetime(2024, 1, 1, tzinfo=UTC)

    rates.update_history(
        client=client,
        output=db_path,
        symbols=["EURUSD"],
        datasets={Dataset.rates},
        timeframes=["M1"],
        flags="ALL",
        lookback_hours=1,
        date_to=date_to,
        deduplicate=False,
        with_views=True,
        include_account_events=False,
    )

    legacy_update.assert_called_once_with(
        client=client,
        output=db_path,
        symbols=["EURUSD"],
        datasets={Dataset.rates},
        timeframes=["M1"],
        flags="ALL",
        lookback_hours=1,
        date_to=date_to,
        deduplicate=False,
        create_rate_views=False,
        with_views=True,
        include_account_events=False,
    )
    with sqlite3.connect(db_path) as conn:
        views = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'view'",
            )
        }
    assert "rate_EURUSD__M1_1" not in views
    assert "unrelated_view" in views


def test_update_history_does_not_clean_views_after_failure(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """Failed updates do not mutate existing compatibility views."""
    db_path = tmp_path / "failed-update.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT)")
        conn.execute(
            "CREATE VIEW rate_EURUSD__M1_1 AS SELECT * FROM rates",
        )
    mocker.patch(
        "mt5cli.rates._legacy_update_history",
        side_effect=ValueError("update failed"),
    )

    with pytest.raises(ValueError, match="update failed"):
        rates.update_history(
            client=mocker.MagicMock(),
            output=db_path,
            symbols=["EURUSD"],
        )

    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type = 'view' AND name = 'rate_EURUSD__M1_1'",
        ).fetchone() == (1,)


def test_update_history_with_config_forces_canonical_options(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """The managed-session wrapper forwards the canonical view policy."""
    legacy_update = mocker.patch("mt5cli.rates._legacy_update_history_with_config")
    output = tmp_path / "config-update.db"

    rates.update_history_with_config(
        output=output,
        symbols=["EURUSD"],
        datasets={Dataset.history_deals},
        timeframes=["M1"],
        flags="INFO",
        lookback_hours=2,
        date_to="2024-01-01T00:00:00+00:00",
        deduplicate=False,
        with_views=True,
        include_account_events=False,
    )

    legacy_update.assert_called_once_with(
        output=output,
        symbols=["EURUSD"],
        config=None,
        datasets={Dataset.history_deals},
        timeframes=["M1"],
        flags="INFO",
        lookback_hours=2,
        date_to="2024-01-01T00:00:00+00:00",
        deduplicate=False,
        create_rate_views=False,
        with_views=True,
        include_account_events=False,
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
