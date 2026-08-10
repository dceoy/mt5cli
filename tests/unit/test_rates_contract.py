"""Focused contract coverage for canonical rate APIs."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from mt5cli import rates
from mt5cli.history import RateTarget

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _canonical_db(path: Path) -> None:
    """Create the minimal canonical rate schema."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
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
    """SQLite errors from explicit custom tables use the stable ValueError surface."""
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
    """SQLite errors after opening canonical storage use the stable ValueError surface."""
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

    rates.update_history(client=client, output=output, symbols=["EURUSD"])

    backend.assert_called_once_with(
        client=client,
        output=output,
        symbols=["EURUSD"],
        datasets=None,
        timeframes=None,
        flags="ALL",
        lookback_hours=24.0,
        date_to=None,
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

    rates.update_history_with_config(output=output, symbols=["EURUSD"])

    backend.assert_called_once_with(
        output=output,
        symbols=["EURUSD"],
        config=None,
        datasets=None,
        timeframes=None,
        flags="ALL",
        lookback_hours=24.0,
        date_to=None,
        deduplicate=True,
        with_views=False,
        include_account_events=True,
    )
