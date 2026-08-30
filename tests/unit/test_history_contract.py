"""Focused branch coverage for canonical history contracts."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from mt5cli import cli, history
from mt5cli.utils import Dataset, IfExists

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from unittest.mock import Mock

    from pytest_mock import MockerFixture


def test_timeframe_interval_and_canonical_table_resolution() -> None:
    """Canonical helpers cover known, unknown, and invalid metadata."""
    assert history.timeframe_interval_seconds(1) == 60
    assert history.timeframe_interval_seconds(999_999) is None
    assert history.resolve_rate_table_name("EURUSD", "M1") == "rates"
    with pytest.raises(ValueError, match="symbol must not be empty"):
        history.resolve_rate_table_name(" ", "M1")


def test_report_rate_gaps_rejects_unknown_timeframe() -> None:
    """Canonical gap inference fails clearly for unsupported timeframes."""
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
        )
        conn.executemany(
            "INSERT INTO rates VALUES (?, ?, ?, ?)",
            [
                ("EURUSD", 999_999, "2024-01-01T00:00:00", 1.0),
                ("EURUSD", 999_999, "2024-01-01T00:01:00", 1.1),
            ],
        )
        with pytest.raises(ValueError, match="Unsupported rate timeframe"):
            history.report_rate_gaps(conn, "rates")


def test_sqlite_timestamp_edge_contracts() -> None:
    """Timestamp helpers preserve semantics across defensive branches."""
    assert history.parse_sqlite_timestamp(float("nan")) is None
    assert history._serialize_sqlite_timestamp(object()) is None
    with pytest.raises(ValueError, match="Invalid SQLite timestamp boundary"):
        history._require_serialized_sqlite_timestamp(object())

    aware = datetime.fromisoformat("2024-01-01T09:00:00+09:00")
    naive = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        history._require_serialized_sqlite_timestamp(aware)
    assert history._require_serialized_sqlite_timestamp(naive) == (
        "2024-01-01T00:00:00"
    )

    frame = pd.DataFrame({"close": [1.0]})
    pd.testing.assert_frame_equal(
        history._canonicalize_sqlite_time_columns(frame),
        frame,
    )


def test_write_ticks_dataset_preserves_empty_frames(
    mocker: MockerFixture,
) -> None:
    """An empty tick response does not invent storage columns."""
    client = mocker.MagicMock()
    client.copy_ticks_range.return_value = pd.DataFrame()

    def fake_stream(
        conn: sqlite3.Connection,
        symbols: Sequence[str],
        dataset: Dataset,
        if_exists: IfExists,
        written_columns: dict[Dataset, set[str]],
        fetch_frame: Callable[[str], pd.DataFrame],
    ) -> bool:
        del conn, symbols, dataset, if_exists, written_columns
        frame = fetch_frame("EURUSD")
        assert frame.empty
        assert list(frame.columns) == []
        return False

    mocker.patch("mt5cli.history._stream_symbol_frames", side_effect=fake_stream)
    with sqlite3.connect(":memory:") as conn:
        assert not history.write_ticks_dataset(
            conn,
            client,
            ["EURUSD"],
            0,
            datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
            datetime(2024, 1, 2, tzinfo=UTC).replace(tzinfo=None),
            IfExists.APPEND,
            {},
        )


@pytest.mark.parametrize("written", [False, True])
def test_incremental_tick_branches(
    mocker: MockerFixture,
    written: bool,
) -> None:
    """Incremental tick updates cover written and empty responses."""
    start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    mocker.patch(
        "mt5cli.history.load_incremental_start_datetimes",
        return_value={("EURUSD", None): start},
    )
    mocker.patch(
        "mt5cli.history._fetch_ticks_frame",
        return_value=pd.DataFrame({"time": [start], "bid": [1.0]}),
    )
    mocker.patch("mt5cli.history.write_streamed_frame", return_value=written)
    written_tables: set[Dataset] = set()
    dedup_scopes: dict[Dataset, list[history.DedupScope]] = {}
    with sqlite3.connect(":memory:") as conn:
        group = history._capture_incremental_ticks(
            conn,
            mocker.MagicMock(),
            ["EURUSD"],
            0,
            start,
            start,
        )
        history._persist_captured_group(
            conn,
            group,
            {},
            written_tables,
            dedup_scopes,
        )
    assert (Dataset.ticks in written_tables) is written
    assert (Dataset.ticks in dedup_scopes) is written


@pytest.mark.parametrize("written", [False, True])
def test_incremental_symbol_branches(
    mocker: MockerFixture,
    written: bool,
) -> None:
    """Incremental symbol snapshots cover written and empty responses."""
    mocker.patch(
        "mt5cli.history._fetch_symbol_metadata_frame",
        return_value=pd.DataFrame({"symbol": ["EURUSD"], "point": [0.0001]}),
    )
    mocker.patch("mt5cli.history.write_streamed_frame", return_value=written)
    written_tables: set[Dataset] = set()
    with sqlite3.connect(":memory:") as conn:
        group = history._capture_incremental_symbols(
            mocker.MagicMock(),
            ["EURUSD"],
            datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
        )
        history._persist_captured_group(conn, group, {}, written_tables, {})
    assert (Dataset.symbols in written_tables) is written


@pytest.mark.parametrize("written", [False, True])
def test_incremental_order_branches(
    mocker: MockerFixture,
    written: bool,
) -> None:
    """Incremental order updates cover written and empty responses."""
    start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    mocker.patch(
        "mt5cli.history.load_incremental_start_datetimes",
        return_value={("EURUSD", None): start},
    )
    mocker.patch(
        "mt5cli.history._fetch_history_dataset_frame",
        return_value=pd.DataFrame({"time": [start], "ticket": [1]}),
    )
    mocker.patch("mt5cli.history.write_streamed_frame", return_value=written)
    written_tables: set[Dataset] = set()
    dedup_scopes: dict[Dataset, list[history.DedupScope]] = {}
    with sqlite3.connect(":memory:") as conn:
        group = history._capture_incremental_history_orders(
            conn,
            mocker.MagicMock(),
            ["EURUSD"],
            start,
            start,
        )
        history._persist_captured_group(
            conn,
            group,
            {},
            written_tables,
            dedup_scopes,
        )
    assert (Dataset.history_orders in written_tables) is written
    assert (Dataset.history_orders in dedup_scopes) is written


def _patch_account_event_dependencies(
    mocker: MockerFixture,
    *,
    written: bool,
    columns: set[str],
) -> tuple[datetime, Mock]:
    start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    mocker.patch(
        "mt5cli.history.load_incremental_start_datetimes",
        return_value={("EURUSD", None): start},
    )
    mocker.patch(
        "mt5cli.history.get_history_deals_account_event_start_datetime",
        return_value=start,
    )
    mocker.patch(
        "mt5cli.history.filter_incremental_history_deals_frame",
        return_value=pd.DataFrame({"time": [start]}),
    )
    mocker.patch("mt5cli.history.write_streamed_frame", return_value=written)
    mocker.patch("mt5cli.history.get_table_columns", return_value=columns)
    record = mocker.patch("mt5cli.history._record_dedup_scope")
    return start, record


def test_incremental_account_event_empty_write(mocker: MockerFixture) -> None:
    """Account-event updates return cleanly when no rows are written."""
    start, record = _patch_account_event_dependencies(
        mocker,
        written=False,
        columns={"symbol", "time"},
    )
    client = mocker.MagicMock()
    client.history_deals.return_value = pd.DataFrame()
    with sqlite3.connect(":memory:") as conn:
        group = history._capture_incremental_history_deals(
            conn,
            client,
            ["EURUSD"],
            start,
            start,
            include_account_events=True,
        )
        history._persist_captured_group(conn, group, {}, set(), {})
    assert not record.called


@pytest.mark.parametrize(
    "columns",
    [
        {"symbol", "time"},
        {"type", "time"},
        {"symbol", "type", "time"},
    ],
)
def test_incremental_account_event_scope_branches(
    mocker: MockerFixture,
    columns: set[str],
) -> None:
    """Account-event dedup scopes follow the available persisted schema."""
    start, record = _patch_account_event_dependencies(
        mocker,
        written=True,
        columns=columns,
    )
    client = mocker.MagicMock()
    client.history_deals.return_value = pd.DataFrame()
    written_tables: set[Dataset] = set()
    with sqlite3.connect(":memory:") as conn:
        group = history._capture_incremental_history_deals(
            conn,
            client,
            ["EURUSD"],
            start,
            start,
            include_account_events=True,
        )
        history._persist_captured_group(conn, group, {}, written_tables, {})
    assert Dataset.history_deals in written_tables
    assert record.called


@pytest.mark.parametrize("written", [False, True])
def test_incremental_trade_deal_branches(
    mocker: MockerFixture,
    written: bool,
) -> None:
    """Symbol-only deal updates cover written and empty responses."""
    start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    mocker.patch(
        "mt5cli.history.load_incremental_start_datetimes",
        return_value={("EURUSD", None): start},
    )
    mocker.patch(
        "mt5cli.history._fetch_history_dataset_frame",
        return_value=pd.DataFrame({"time": [start], "ticket": [1]}),
    )
    mocker.patch("mt5cli.history.write_streamed_frame", return_value=written)
    written_tables: set[Dataset] = set()
    dedup_scopes: dict[Dataset, list[history.DedupScope]] = {}
    with sqlite3.connect(":memory:") as conn:
        group = history._capture_incremental_history_deals(
            conn,
            mocker.MagicMock(),
            ["EURUSD"],
            start,
            start,
            include_account_events=False,
        )
        history._persist_captured_group(
            conn,
            group,
            {},
            written_tables,
            dedup_scopes,
        )
    assert (Dataset.history_deals in written_tables) is written
    assert (Dataset.history_deals in dedup_scopes) is written


@pytest.mark.parametrize("written", [False, True])
def test_write_incremental_ticks_branches(
    mocker: MockerFixture,
    written: bool,
) -> None:
    """Streaming tick updates cover written and empty responses."""
    start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    mocker.patch(
        "mt5cli.history.load_incremental_start_datetimes",
        return_value={("EURUSD", None): start},
    )
    mocker.patch("mt5cli.history.write_ticks_dataset", return_value=written)
    written_tables: set[Dataset] = set()
    dedup_scopes: dict[Dataset, list[history.DedupScope]] = {}
    with sqlite3.connect(":memory:") as conn:
        history._write_incremental_ticks(
            conn,
            mocker.MagicMock(),
            ["EURUSD"],
            0,
            start,
            start,
            {},
            written_tables,
            dedup_scopes,
        )
    assert (Dataset.ticks in written_tables) is written
    assert (Dataset.ticks in dedup_scopes) is written


@pytest.mark.parametrize("written", [False, True])
def test_write_incremental_symbols_branches(
    mocker: MockerFixture,
    written: bool,
) -> None:
    """Streaming symbol snapshots cover written and empty responses."""
    mocker.patch("mt5cli.history.write_symbols_dataset", return_value=written)
    written_tables: set[Dataset] = set()
    with sqlite3.connect(":memory:") as conn:
        history._write_incremental_symbols(
            conn,
            mocker.MagicMock(),
            ["EURUSD"],
            datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None),
            {},
            written_tables,
        )
    assert (Dataset.symbols in written_tables) is written


@pytest.mark.parametrize("written", [False, True])
def test_write_incremental_history_orders_branches(
    mocker: MockerFixture,
    written: bool,
) -> None:
    """Streaming order updates cover written and empty responses."""
    start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    mocker.patch(
        "mt5cli.history.load_incremental_start_datetimes",
        return_value={("EURUSD", None): start},
    )
    mocker.patch("mt5cli.history.write_history_dataset", return_value=written)
    written_tables: set[Dataset] = set()
    dedup_scopes: dict[Dataset, list[history.DedupScope]] = {}
    with sqlite3.connect(":memory:") as conn:
        history._write_incremental_history_orders(
            conn,
            mocker.MagicMock(),
            ["EURUSD"],
            start,
            start,
            {},
            written_tables,
            dedup_scopes,
        )
    assert (Dataset.history_orders in written_tables) is written
    assert (Dataset.history_orders in dedup_scopes) is written


@pytest.mark.parametrize("written", [False, True])
def test_write_incremental_history_deals_symbol_scoped_branches(
    mocker: MockerFixture,
    written: bool,
) -> None:
    """Streaming symbol-only deal updates cover written and empty responses."""
    start = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    mocker.patch(
        "mt5cli.history.load_incremental_start_datetimes",
        return_value={("EURUSD", None): start},
    )
    mocker.patch("mt5cli.history.write_history_dataset", return_value=written)
    written_tables: set[Dataset] = set()
    dedup_scopes: dict[Dataset, list[history.DedupScope]] = {}
    with sqlite3.connect(":memory:") as conn:
        history._write_incremental_history_deals(
            conn,
            mocker.MagicMock(),
            ["EURUSD"],
            start,
            start,
            {},
            written_tables,
            dedup_scopes,
            include_account_events=False,
        )
    assert (Dataset.history_deals in written_tables) is written
    assert (Dataset.history_deals in dedup_scopes) is written


def test_finalize_incremental_branch_matrix(mocker: MockerFixture) -> None:
    """Finalization covers deduplication and optional derived views."""
    mocker.patch("mt5cli.history.augment_written_columns_from_sqlite")
    mocker.patch("mt5cli.history.create_history_indexes")
    deduplicate = mocker.patch("mt5cli.history.deduplicate_history_tables")
    cash = mocker.patch("mt5cli.history.create_cash_events_view")
    positions = mocker.patch("mt5cli.history.create_positions_reconstructed_view")

    with sqlite3.connect(":memory:") as conn:
        history._finalize_incremental_writes(
            conn,
            {Dataset.history_deals},
            {Dataset.history_deals: {"type"}},
            {Dataset.history_deals},
            {},
            deduplicate=True,
            with_views=True,
        )
        history._finalize_incremental_writes(
            conn,
            {Dataset.history_deals},
            {},
            set(),
            {},
            deduplicate=False,
            with_views=True,
        )
        history._finalize_incremental_writes(
            conn,
            set(),
            {},
            set(),
            {},
            deduplicate=False,
            with_views=False,
        )

    deduplicate.assert_called_once()
    cash.assert_called_once()
    positions.assert_called_once()


def test_update_history_with_config_skips_empty_dataset_selection(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """Managed-session updates do not open MT5 when no datasets are selected."""
    session = mocker.patch("mt5cli.history.mt5_session")
    history.update_history_with_config(
        output=tmp_path / "history.db",
        symbols=["EURUSD"],
        datasets=set(),
    )
    assert not session.called


def test_history_gaps_cli_happy_path(
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """The canonical history-gaps command exports the concatenated report."""
    db_path = tmp_path / "history.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE rates(symbol TEXT, timeframe INTEGER, time TEXT, close REAL)"
        )
    export = mocker.patch("mt5cli.cli._execute_export")
    cli.history_gaps(
        mocker.MagicMock(),
        db_path,
        table=None,
        granularity_seconds=60,
        min_gap_intervals=1,
    )
    export.assert_called_once()
    assert export.call_args.args[1]().empty
