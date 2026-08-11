"""Final edge coverage for canonical history behavior."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from mt5cli import history

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def test_gap_report_requires_interval_for_custom_table() -> None:
    """Custom tables without timeframe metadata require an explicit interval."""
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE custom_rates(time TEXT, close REAL)")
        conn.executemany(
            "INSERT INTO custom_rates VALUES (?, ?)",
            [
                ("2024-01-01T00:00:00", 1.0),
                ("2024-01-01T00:02:00", 1.1),
            ],
        )
        with pytest.raises(ValueError, match="Could not infer granularity"):
            history.report_rate_gaps(conn, "custom_rates")


def test_gap_report_uses_explicit_interval_for_custom_table() -> None:
    """An explicit interval bypasses timeframe inference for custom tables."""
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE custom_rates(time TEXT, close REAL)")
        conn.executemany(
            "INSERT INTO custom_rates VALUES (?, ?)",
            [
                ("2024-01-01T00:00:00", 1.0),
                ("2024-01-01T00:02:00", 1.1),
            ],
        )
        report = history.report_rate_gaps(
            conn,
            "custom_rates",
            granularity_seconds=60,
        )

    assert list(report["missing_intervals"]) == [1]


def test_gap_report_infers_supported_canonical_timeframe() -> None:
    """Canonical rate rows infer a supported timeframe and continue normally."""
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
        )
        conn.executemany(
            "INSERT INTO rates VALUES (?, ?, ?, ?)",
            [
                ("EURUSD", 1, "2024-01-01T00:00:00", 1.0),
                ("EURUSD", 1, "2024-01-01T00:02:00", 1.1),
            ],
        )
        report = history.report_rate_gaps(conn, "rates")

    assert list(report["granularity_seconds"]) == [60]
    assert list(report["missing_intervals"]) == [1]


def test_managed_update_returns_when_request_resolves_to_none(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """The managed-session wrapper preserves the resolved no-op path."""
    mocker.patch("mt5cli.history._resolve_update_history_request", return_value=None)
    session = mocker.patch("mt5cli.history.mt5_session")

    history.update_history_with_config(
        output=tmp_path / "history.db",
        symbols=["EURUSD"],
    )

    session.assert_not_called()


def test_managed_update_opens_session_and_delegates(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """A resolved managed update opens one session and delegates once."""
    mocker.patch(
        "mt5cli.history._resolve_update_history_request",
        return_value=mocker.MagicMock(),
    )
    client = mocker.MagicMock()
    session = mocker.patch("mt5cli.history.mt5_session")
    session.return_value.__enter__.return_value = client
    update = mocker.patch("mt5cli.history.update_history")
    output = tmp_path / "history.db"

    history.update_history_with_config(
        output=output,
        symbols=["EURUSD"],
    )

    session.assert_called_once_with(None)
    update.assert_called_once_with(
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
