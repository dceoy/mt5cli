"""Tests for the MT5-independent history capture/persist split (mteor #466)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd

from mt5cli import rates, update_history
from mt5cli.utils import Dataset

if TYPE_CHECKING:
    from pathlib import Path

DATE_TO = datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None)


def _make_client() -> MagicMock:
    client = MagicMock()
    client.copy_rates_range.return_value = pd.DataFrame({
        "time": ["2024-01-01T12:00:00"],
        "open": [1.1],
    })
    client.copy_ticks_range.return_value = pd.DataFrame({
        "time": ["2024-01-01T12:00:00"],
        "bid": [1.1],
        "ask": [1.2],
    })
    client.history_orders.return_value = pd.DataFrame({
        "ticket": [11],
        "symbol": ["EURUSD"],
        "time_setup": ["2024-01-01T12:00:00"],
    })
    client.history_deals.return_value = pd.DataFrame({
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
    client.symbol_info_as_dict.return_value = {"point": 0.0001, "digits": 5}
    return client


_ALL_DATASETS: set[Dataset] = {
    Dataset.rates,
    Dataset.ticks,
    Dataset.history_orders,
    Dataset.history_deals,
    Dataset.symbols,
}


def test_capture_never_creates_the_output_database(tmp_path: Path) -> None:
    """Capturing reads MT5 data only; it must not touch SQLite at all."""
    output = tmp_path / "capture-only.db"
    capture = rates.capture_history_datasets(
        client=_make_client(),
        output=output,
        symbols=["EURUSD"],
        datasets=_ALL_DATASETS,
        timeframes=["M1"],
        lookback_hours=24,
        date_to=DATE_TO,
    )
    assert capture is not None
    assert isinstance(capture, rates.HistoryCapture)
    assert not output.exists()


def test_capture_returns_none_for_no_selected_datasets(tmp_path: Path) -> None:
    """No selected datasets yields None, matching update_history's no-op."""
    output = tmp_path / "empty.db"
    capture = rates.capture_history_datasets(
        client=_make_client(),
        output=output,
        symbols=["EURUSD"],
        datasets=set(),
        date_to=DATE_TO,
    )
    assert capture is None


def test_persist_never_touches_the_client(tmp_path: Path) -> None:
    """Persisting a capture writes SQLite without any further MT5 access."""
    output = tmp_path / "persist.db"
    client = _make_client()
    capture = rates.capture_history_datasets(
        client=client,
        output=output,
        symbols=["EURUSD"],
        datasets=_ALL_DATASETS,
        timeframes=["M1"],
        lookback_hours=24,
        date_to=DATE_TO,
        with_views=True,
    )
    assert capture is not None
    calls_before = client.copy_rates_range.call_count + client.history_deals.call_count
    rates.persist_history_datasets(capture)
    calls_after = client.copy_rates_range.call_count + client.history_deals.call_count
    assert calls_after == calls_before
    assert output.exists()
    with sqlite3.connect(output) as conn:
        assert conn.execute("SELECT COUNT(*) FROM rates").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM history_deals").fetchone() == (1,)
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'cash_events'",
        ).fetchone() == ("cash_events",)


def test_capture_then_persist_matches_update_history(tmp_path: Path) -> None:
    """The split capture+persist path is equivalent to update_history's output."""
    fused_output = tmp_path / "fused.db"
    split_output = tmp_path / "split.db"

    update_history(
        client=_make_client(),
        output=fused_output,
        symbols=["EURUSD"],
        datasets={Dataset.rates, Dataset.history_deals},
        timeframes=["M1"],
        lookback_hours=24,
        date_to=DATE_TO,
    )

    capture = rates.capture_history_datasets(
        client=_make_client(),
        output=split_output,
        symbols=["EURUSD"],
        datasets={Dataset.rates, Dataset.history_deals},
        timeframes=["M1"],
        lookback_hours=24,
        date_to=DATE_TO,
    )
    assert capture is not None
    rates.persist_history_datasets(capture)

    def _rows(path: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
        with sqlite3.connect(path) as conn:
            r = conn.execute(
                "SELECT symbol, timeframe, time, open FROM rates",
            ).fetchall()
            d = conn.execute(
                "SELECT ticket, symbol, time, profit FROM history_deals",
            ).fetchall()
            return r, d

    assert _rows(fused_output) == _rows(split_output)


def test_capture_skips_unselected_datasets(tmp_path: Path) -> None:
    """A capture excluding rates fetches no rates and builds no rates group."""
    output = tmp_path / "symbols-only.db"
    client = _make_client()
    capture = rates.capture_history_datasets(
        client=client,
        output=output,
        symbols=["EURUSD"],
        datasets={Dataset.symbols},
        date_to=DATE_TO,
    )
    assert capture is not None
    assert capture.selected_datasets == frozenset({Dataset.symbols})
    client.copy_rates_range.assert_not_called()


def test_capture_reads_cursor_from_existing_database(tmp_path: Path) -> None:
    """Capture resumes from an existing MAX(time) cursor, not the lookback fallback."""
    output = tmp_path / "existing.db"
    with sqlite3.connect(output) as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, open REAL)",
        )
        conn.execute(
            "INSERT INTO rates VALUES ('EURUSD', 1, '2024-01-01T06:00:00', 1.0)",
        )
        conn.execute(
            "CREATE TABLE _mt5cli_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        )
        conn.execute(
            "INSERT INTO _mt5cli_metadata(key, value) VALUES "
            "('rates_timestamp_contract', 'pdmt5-wall-clock-v1')",
        )

    seen_starts: list[object] = []

    def make_rates(**kwargs: object) -> pd.DataFrame:
        seen_starts.append(kwargs["date_from"])
        return pd.DataFrame({"time": ["2024-01-01T12:00:00"], "open": [1.1]})

    client = MagicMock()
    client.copy_rates_range.side_effect = make_rates
    capture = rates.capture_history_datasets(
        client=client,
        output=output,
        symbols=["EURUSD"],
        datasets={Dataset.rates},
        timeframes=["M1"],
        lookback_hours=24,
        date_to=DATE_TO,
    )
    assert capture is not None
    assert seen_starts == [datetime(2024, 1, 1, 6, tzinfo=UTC).replace(tzinfo=None)]
