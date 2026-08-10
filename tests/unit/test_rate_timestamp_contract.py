"""Regression tests for the canonical SQLite rate timestamp contract."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, cast

import pytest

from mt5cli import rates, sdk

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from mt5cli.contract import HistoryClient


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


def test_sdk_exports_are_derived_from_public_imports() -> None:
    """The SDK export registry has no second manually maintained name list."""
    public_imports = {
        name
        for name in vars(sdk)
        if not name.startswith("_") and name != "STABLE_SDK_EXPORTS"
    }
    assert set(sdk.__all__) == public_imports
    assert frozenset(public_imports) == sdk.STABLE_SDK_EXPORTS
