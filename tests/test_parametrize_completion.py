"""Parameterized regression coverage for remaining duplicate-shaped cases."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture  # noqa: TC002
from typer.testing import CliRunner

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

from mt5cli.cli import app
from mt5cli.grafana import (
    create_grafana_views,
    create_snapshot_tables,
    insert_account_snapshot,
    insert_terminal_snapshot,
    start_snapshot_run,
)
from mt5cli.history import (
    load_rate_data,
    resolve_granularity_name,
    resolve_history_tick_flags,
)
from mt5cli.sdk import AccountSpec, collect_latest_closed_rates_for_accounts
from mt5cli.trading import calculate_spread_ratio

runner = CliRunner()


def _get_names(conn: sqlite3.Connection, type_: str) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type=?",
            (type_,),
        ).fetchall()
    }


def _make_symbol_pnl_required_deals(conn: sqlite3.Connection) -> None:
    """Create history_deals with only columns required by grafana_symbol_pnl."""
    conn.execute(
        "CREATE TABLE history_deals"
        " (time TEXT, symbol TEXT, profit REAL, type INTEGER, entry INTEGER)"
    )


def _make_symbol_pnl_full_deals(conn: sqlite3.Connection) -> None:
    """Create history_deals with optional volume/price columns."""
    conn.execute(
        "CREATE TABLE history_deals"
        " (time TEXT, symbol TEXT, profit REAL, type INTEGER, entry INTEGER,"
        "  volume REAL, price REAL, ticket INTEGER, position_id INTEGER)"
    )


def _select_account_snapshot(conn: sqlite3.Connection) -> tuple[object, ...]:
    """Return the stable account snapshot assertion tuple."""
    row = conn.execute(
        "SELECT login, currency, balance FROM account_snapshots"
    ).fetchone()
    assert row is not None
    return tuple(row)


def _select_terminal_snapshot(conn: sqlite3.Connection) -> tuple[object, ...]:
    """Return the stable terminal snapshot assertion tuple."""
    row = conn.execute("SELECT name, connected FROM terminal_snapshots").fetchone()
    assert row is not None
    return tuple(row)


@pytest.mark.parametrize(
    ("command", "patch_update_observability"),
    [
        ("snapshot", True),
        ("grafana-schema", False),
    ],
    ids=["snapshot", "grafana-schema"],
)
@pytest.mark.parametrize(
    ("use_publish_copy", "expect_called"),
    [(True, True), (False, False)],
    ids=["with-publish-copy", "no-publish-copy"],
)
def test_publish_copy_option_gates_grafana_copy(
    tmp_path: Path,
    mocker: MockerFixture,
    command: str,
    patch_update_observability: bool,
    use_publish_copy: bool,
    expect_called: bool,
) -> None:
    """Test --publish-copy gates publish_grafana_copy for copy-capable commands."""
    if patch_update_observability:
        mocker.patch("mt5cli.cli.sdk.update_observability_with_config")
    mock_publish = mocker.patch("mt5cli.grafana.publish_grafana_copy")
    args = ["-o", str(tmp_path / "out.db"), command]
    if use_publish_copy:
        args += ["--publish-copy", str(tmp_path / "grafana.db")]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    if expect_called:
        mock_publish.assert_called_once()
    else:
        mock_publish.assert_not_called()


@pytest.mark.parametrize(
    ("rates_frame", "kwargs"),
    [
        (pd.DataFrame({"time": [1], "close": [1.1]}), {"count": 1}),
        (pd.DataFrame(columns=["time", "close"]), {"count": 1, "start_pos": 1}),
    ],
    ids=["forming-bar-only", "empty-start-pos-nonzero"],
)
def test_collect_latest_closed_rates_rejects_empty_effective_frames(
    mocker: MockerFixture,
    rates_frame: pd.DataFrame,
    kwargs: dict[str, object],
) -> None:
    """Test empty effective frames raise after start_pos/forming-bar handling."""
    mocker.patch(
        "mt5cli.sdk.collect_latest_rates_for_accounts_with_retries",
        return_value={("EURUSD", 1): rates_frame},
    )

    with pytest.raises(ValueError, match="Rate data is empty"):
        collect_latest_closed_rates_for_accounts(
            [AccountSpec(symbols=["EURUSD"])],
            ["M1"],
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("path_kind", "match"),
    [
        ("missing-file", "SQLite database not found"),
        ("directory", "not a file"),
    ],
    ids=["missing-file", "directory"],
)
def test_load_rate_data_rejects_invalid_database_paths(
    tmp_path: Path,
    path_kind: str,
    match: str,
) -> None:
    """Test load_rate_data validates missing and non-file SQLite paths."""
    db_path = tmp_path if path_kind == "directory" else tmp_path / "missing.db"
    with pytest.raises(ValueError, match=match):
        load_rate_data(db_path, "rates")


@pytest.mark.parametrize(
    ("flags", "expected"),
    [("ALL", -1), (2, 2)],
    ids=["all", "numeric"],
)
def test_resolve_history_tick_flags(flags: str | int, expected: int) -> None:
    """Test resolve_history_tick_flags accepts named and numeric tick flags."""
    assert resolve_history_tick_flags(flags) == expected


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [(999, "999"), (1, "M1")],
    ids=["unknown-integer", "known-m1"],
)
def test_resolve_granularity_name(timeframe: int, expected: str) -> None:
    """Test resolve_granularity_name handles known names and integer fallback."""
    assert resolve_granularity_name(timeframe) == expected


@pytest.mark.parametrize(
    ("tick", "expected"),
    [
        ({"bid": 99.0, "ask": 101.0}, 0.02),
        ({"bid": "99.0", "ask": "101.0"}, 0.02),
    ],
    ids=["numeric", "numeric-string"],
)
def test_calculate_spread_ratio(
    tick: dict[str, object],
    expected: float,
) -> None:
    """Test calculate_spread_ratio accepts numeric and numeric-string ticks."""
    client = MagicMock()
    client.symbol_info_tick_as_dict.return_value = tick

    assert abs(calculate_spread_ratio(client, "EURUSD") - expected) < 1e-9


@pytest.mark.parametrize(
    ("setup_deals", "expected_present_cols"),
    [
        (_make_symbol_pnl_required_deals, set()),
        (_make_symbol_pnl_full_deals, {"volume", "price"}),
    ],
    ids=["required-only", "with-volume-price"],
)
def test_grafana_symbol_pnl_optional_columns(
    setup_deals: Callable[[sqlite3.Connection], None],
    expected_present_cols: set[str],
) -> None:
    """Test grafana_symbol_pnl exposes optional columns when present."""
    with sqlite3.connect(":memory:") as conn:
        setup_deals(conn)
        create_grafana_views(conn)
        assert "grafana_symbol_pnl" in _get_names(conn, "view")
        cols = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(grafana_symbol_pnl)")
        }
    assert expected_present_cols <= cols


@pytest.mark.parametrize(
    ("insert_func", "row", "selector", "expected"),
    [
        (
            insert_account_snapshot,
            {
                "login": 12345,
                "currency": "USD",
                "balance": 10000.0,
                "equity": 9800.0,
                "margin": 200.0,
                "margin_free": 9800.0,
                "margin_level": 4900.0,
                "profit": -200.0,
                "leverage": 100,
            },
            _select_account_snapshot,
            (12345, "USD", 10000.0),
        ),
        (
            insert_terminal_snapshot,
            {
                "name": "MetaTrader 5",
                "connected": 1,
                "community_account": 0,
                "trade_allowed": 1,
                "trade_expert": 1,
                "path": "/mt5",
                "company": "Broker",
                "language": "en",
            },
            _select_terminal_snapshot,
            ("MetaTrader 5", 1),
        ),
    ],
    ids=["account", "terminal"],
)
def test_insert_single_snapshot(
    insert_func: Callable[[sqlite3.Connection, int, dict[str, object]], None],
    row: dict[str, object],
    selector: Callable[[sqlite3.Connection], tuple[object, ...]],
    expected: tuple[object, ...],
) -> None:
    """Test account and terminal snapshot helpers append one row."""
    with sqlite3.connect(":memory:") as conn:
        create_snapshot_tables(conn)
        run_id = start_snapshot_run(conn, 1700000000)
        insert_func(conn, run_id, row)
        result = selector(conn)
    assert result == expected
