"""Tests for mt5cli.sdk module."""

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
    from pathlib import Path

from mt5cli import sdk
from mt5cli.sdk import (
    Mt5CliClient,
    account_info,
    build_config,
    collect_history,
    copy_rates_from,
    copy_rates_from_pos,
    copy_rates_range,
    copy_ticks_from,
    copy_ticks_range,
    history_deals,
    history_orders,
    last_error,
    market_book,
    orders,
    positions,
    symbol_info,
    symbol_info_tick,
    symbols,
    terminal_info,
    update_history,
    update_history_with_config,
    version,
)
from mt5cli.sqlite_history import DEFAULT_HISTORY_TIMEFRAMES
from mt5cli.utils import Dataset

_DEALS_FIXTURE: dict[str, list[object]] = {
    "ticket": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "position_id": [100, 100, 100, 0, 200, 200, 300, 400, 400, 500, 500, 600, 600, 600],
    "symbol": [
        "EURUSD",
        "EURUSD",
        "EURUSD",
        "",
        "EURUSD",
        "EURUSD",
        "GBPUSD",
        "GBPUSD",
        "GBPUSD",
        "EURUSD",
        "EURUSD",
        "GBPUSD",
        "GBPUSD",
        "GBPUSD",
    ],
    "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "type": [0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 1, 1],
    "entry": [0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 3, 0, 2, 1],
    "volume": [1.0, 3.0, 4.0, 0.0, 2.0, 2.0, 5.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0],
    "price": [
        1.10,
        1.20,
        1.50,
        0.0,
        2.00,
        2.20,
        1.30,
        1.30,
        1.40,
        1.00,
        1.05,
        1.10,
        9.99,
        1.40,
    ],
    "profit": [0.0, 0.0, 10.0, 5.0, 0.0, 8.0, 0.0, 0.0, -1.0, 0.0, 3.0, 0.0, -2.0, 7.0],
}


@pytest.fixture
def mock_client(mocker: MockerFixture) -> MagicMock:
    """Create and patch a mock Mt5DataClient for SDK tests."""
    client = MagicMock()
    sample_df = pd.DataFrame({"col": [1]})
    client.copy_rates_from_as_df.return_value = sample_df
    client.copy_rates_from_pos_as_df.return_value = sample_df
    client.copy_rates_range_as_df.return_value = sample_df
    client.copy_ticks_from_as_df.return_value = sample_df
    client.copy_ticks_range_as_df.return_value = sample_df
    client.account_info_as_df.return_value = sample_df
    client.terminal_info_as_df.return_value = sample_df
    client.symbols_get_as_df.return_value = sample_df
    client.symbol_info_as_df.return_value = sample_df
    client.orders_get_as_df.return_value = sample_df
    client.positions_get_as_df.return_value = sample_df
    client.history_orders_get_as_df.return_value = sample_df
    client.history_deals_get_as_df.return_value = sample_df
    client.version_as_df.return_value = sample_df
    client.last_error_as_df.return_value = sample_df
    client.symbol_info_tick_as_df.return_value = sample_df
    client.market_book_get_as_df.return_value = sample_df
    mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
    return client


def _build_history_client(mocker: MockerFixture) -> MagicMock:
    """Build a mocked Mt5DataClient with per-symbol history results."""
    client = MagicMock()

    def _rates(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({
            "time": [1],
            "open": [1.0],
            "symbol_arg": [kwargs.get("symbol")],
        })

    def _ticks(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({
            "time": [1],
            "bid": [1.0],
            "symbol_arg": [kwargs.get("symbol")],
        })

    client.copy_rates_range_as_df.side_effect = _rates
    client.copy_ticks_range_as_df.side_effect = _ticks

    def _orders(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"ticket": [10], "symbol": [kwargs.get("symbol")]})

    def _deals(**kwargs: object) -> pd.DataFrame:
        sym = kwargs.get("symbol")
        df = pd.DataFrame(_DEALS_FIXTURE)
        return df[df["symbol"] == sym].reset_index(drop=True)

    client.history_orders_get_as_df.side_effect = _orders
    client.history_deals_get_as_df.side_effect = _deals
    mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
    return client


class TestConnectionLifecycle:
    """Tests for MT5 connection lifecycle helpers."""

    def test_connected_client_shuts_down(self, mocker: MockerFixture) -> None:
        """Test that _connected_client always shuts down."""
        mock_client = MagicMock()
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        config = MagicMock()
        with sdk._connected_client(config):  # type: ignore[reportPrivateUsage]
            mock_client.initialize_and_login_mt5.assert_called_once()
        mock_client.shutdown.assert_called_once()

    def test_connected_client_shutdown_on_init_failure(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that shutdown is called when initialize/login fails."""
        mock_client = MagicMock()
        mock_client.initialize_and_login_mt5.side_effect = RuntimeError(
            "login failed",
        )
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        with (
            pytest.raises(RuntimeError, match="login failed"),
            sdk._connected_client(MagicMock()),  # type: ignore[reportPrivateUsage]
        ):
            pass
        mock_client.shutdown.assert_called_once()

    def test_run_with_client_shutdown_on_error(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that shutdown is called even when fetch raises."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.side_effect = RuntimeError("boom")
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        with pytest.raises(RuntimeError, match="boom"):
            sdk._run_with_client(  # type: ignore[reportPrivateUsage]
                MagicMock(),
                lambda c: c.account_info_as_df(),
            )
        mock_client.shutdown.assert_called_once()

    def test_client_context_manager_reuses_connection(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that context-managed client reuses one connection."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.return_value = pd.DataFrame({"a": [1]})
        mock_client.terminal_info_as_df.return_value = pd.DataFrame({"b": [2]})
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        with Mt5CliClient() as client:
            client.account_info()
            client.terminal_info()
            assert client.config is not None
        mock_client.initialize_and_login_mt5.assert_called_once()
        mock_client.shutdown.assert_called_once()
        assert mock_client.account_info_as_df.call_count == 1
        assert mock_client.terminal_info_as_df.call_count == 1

    def test_client_context_manager_shutdown_on_init_failure(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that shutdown is called when context manager login fails."""
        mock_client = MagicMock()
        mock_client.initialize_and_login_mt5.side_effect = RuntimeError(
            "login failed",
        )
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        client = Mt5CliClient()
        with pytest.raises(RuntimeError, match="login failed"), client:
            pass
        mock_client.shutdown.assert_called_once()
        assert client._client is None  # type: ignore[reportPrivateUsage]

    def test_exit_without_enter_is_noop(self) -> None:
        """Test that __exit__ without __enter__ does not fail."""
        client = Mt5CliClient()
        client.__exit__(None, None, None)


class TestModuleFunctions:
    """Tests for module-level SDK wrappers."""

    @pytest.mark.parametrize(
        ("fn", "args", "method"),
        [
            (
                copy_rates_from,
                ("EURUSD", "M1", "2024-01-01", 10),
                "copy_rates_from_as_df",
            ),
            (
                copy_rates_from_pos,
                ("EURUSD", "M1", 0, 10),
                "copy_rates_from_pos_as_df",
            ),
            (
                copy_ticks_from,
                ("EURUSD", "2024-01-01", 10, "ALL"),
                "copy_ticks_from_as_df",
            ),
            (
                copy_ticks_range,
                ("EURUSD", "2024-01-01", "2024-02-01", "ALL"),
                "copy_ticks_range_as_df",
            ),
            (account_info, (), "account_info_as_df"),
            (terminal_info, (), "terminal_info_as_df"),
            (symbols, ("*USD*",), "symbols_get_as_df"),
            (symbol_info, ("EURUSD",), "symbol_info_as_df"),
            (orders, (), "orders_get_as_df"),
            (positions, (), "positions_get_as_df"),
            (history_orders, (), "history_orders_get_as_df"),
            (history_deals, (), "history_deals_get_as_df"),
            (version, (), "version_as_df"),
            (last_error, (), "last_error_as_df"),
            (symbol_info_tick, ("EURUSD",), "symbol_info_tick_as_df"),
            (market_book, ("EURUSD",), "market_book_get_as_df"),
        ],
    )
    def test_module_functions_delegate(
        self,
        mock_client: MagicMock,
        fn: object,
        args: tuple[object, ...],
        method: str,
    ) -> None:
        """Test module-level functions call the expected client methods."""
        config = build_config(login=123)
        result = fn(*args, config=config)  # type: ignore[operator]
        assert isinstance(result, pd.DataFrame)
        getattr(mock_client, method).assert_called_once()


class TestMt5CliClient:
    """Tests for Mt5CliClient SDK methods."""

    def test_copy_rates_range_returns_dataframe(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Test that copy_rates_range returns a DataFrame."""
        df = Mt5CliClient().copy_rates_range(
            "EURUSD",
            "D1",
            "2024-01-01",
            "2024-02-01",
        )
        assert isinstance(df, pd.DataFrame)
        mock_client.copy_rates_range_as_df.assert_called_once_with(
            symbol="EURUSD",
            timeframe=16408,
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            date_to=datetime(2024, 2, 1, tzinfo=UTC),
        )

    def test_copy_ticks_from_parses_flags(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Test that string tick flags are parsed."""
        Mt5CliClient().copy_ticks_from("EURUSD", "2024-01-01", 100, "INFO")
        mock_client.copy_ticks_from_as_df.assert_called_once_with(
            symbol="EURUSD",
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            count=100,
            flags=2,
        )

    def test_history_orders_accepts_string_dates(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Test that string datetime inputs are parsed."""
        Mt5CliClient().history_orders(
            date_from="2024-01-01",
            date_to="2024-02-01",
        )
        mock_client.history_orders_get_as_df.assert_called_once_with(
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            date_to=datetime(2024, 2, 1, tzinfo=UTC),
            group=None,
            symbol=None,
            ticket=None,
            position=None,
        )

    def test_module_function_delegates_to_client(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Test module-level copy_rates_range delegates to the client."""
        df = copy_rates_range(
            "USDJPY",
            "M1",
            "2024-01-01",
            "2024-02-01",
        )
        assert isinstance(df, pd.DataFrame)
        mock_client.copy_rates_range_as_df.assert_called_once()


class TestCollectHistory:
    """Tests for collect_history SDK function."""

    @pytest.fixture
    def history_client(self, mocker: MockerFixture) -> MagicMock:
        """Create a mocked Mt5DataClient with history-style DataFrames."""
        return _build_history_client(mocker)

    def test_collect_history_writes_all_tables(
        self,
        tmp_path: Path,
        history_client: MagicMock,
    ) -> None:
        """Test that collect_history writes rates, ticks, and history tables."""
        output = tmp_path / "history.db"
        collect_history(
            output,
            ["EURUSD", "GBPUSD"],
            "2024-01-01",
            "2024-02-01",
        )
        assert history_client.copy_rates_range_as_df.call_count == 2
        assert history_client.copy_ticks_range_as_df.call_count == 2
        with sqlite3.connect(output) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
        assert {"rates", "ticks", "history_orders", "history_deals"} <= tables

    def test_collect_history_with_views(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test that with_views creates cash_events and positions views."""
        output = tmp_path / "history.db"
        collect_history(
            output,
            ["EURUSD", "GBPUSD"],
            "2024-01-01",
            "2024-02-01",
            with_views=True,
        )
        with sqlite3.connect(output) as conn:
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                ).fetchall()
            }
            positions = {
                row[0]
                for row in conn.execute(
                    "SELECT position_id FROM positions_reconstructed",
                ).fetchall()
            }
        assert {"cash_events", "positions_reconstructed"} <= views
        assert set(positions) == {100, 200, 500, 600}

    def test_collect_history_rates_table_has_timeframe(
        self,
        tmp_path: Path,
        history_client: MagicMock,  # noqa: ARG002
    ) -> None:
        """Test that the rates table carries the requested timeframe value."""
        output = tmp_path / "history.db"
        collect_history(
            output,
            ["EURUSD"],
            "2024-01-01",
            "2024-02-01",
            datasets={Dataset.rates},
            timeframe="H1",
        )
        with sqlite3.connect(output) as conn:
            rows = conn.execute(
                "SELECT DISTINCT timeframe FROM rates",
            ).fetchall()
        assert rows == [(16385,)]

    def test_collect_history_views_skipped_when_columns_missing(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that views are not created when required columns are missing."""
        client = MagicMock()
        client.copy_rates_range_as_df.return_value = pd.DataFrame({"x": [1]})
        client.copy_ticks_range_as_df.return_value = pd.DataFrame({"x": [1]})
        client.history_orders_get_as_df.return_value = pd.DataFrame({"x": [1]})
        client.history_deals_get_as_df.return_value = pd.DataFrame({"x": [1]})
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
        output = tmp_path / "history.db"
        with caplog.at_level(logging.WARNING, logger="mt5cli.sdk"):
            collect_history(
                output,
                ["EURUSD"],
                "2024-01-01",
                "2024-02-01",
                with_views=True,
            )
        with sqlite3.connect(output) as conn:
            views = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view'",
                ).fetchall()
            }
        assert "cash_events" not in views
        assert "positions_reconstructed" not in views


class TestUpdateHistory:
    """Tests for update_history SDK functions."""

    @pytest.fixture
    def connected_client(self) -> MagicMock:
        """Create a connected mock client without MT5 lifecycle patching."""
        return MagicMock()

    def test_update_history_appends_incrementally(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test sequential SQLite history updates use existing max timestamps."""
        date_to = datetime(2024, 1, 2, tzinfo=UTC)
        first_expected_start = datetime(2024, 1, 1, tzinfo=UTC)
        second_expected_start = datetime(2024, 1, 1, 12, tzinfo=UTC)
        rate_starts: list[datetime] = []
        deal_starts: list[datetime] = []

        def make_rates(**kwargs: object) -> pd.DataFrame:
            assert kwargs["symbol"] == "EURUSD"
            assert kwargs["timeframe"] == 1
            assert kwargs["date_to"] == date_to
            rate_starts.append(kwargs["date_from"])  # type: ignore[arg-type]
            return pd.DataFrame({
                "time": ["2024-01-01T12:00:00+00:00"],
                "open": [1.0 + len(rate_starts) / 10],
            })

        def make_deals(**kwargs: object) -> pd.DataFrame:
            assert kwargs["date_to"] == date_to
            deal_starts.append(kwargs["date_from"])  # type: ignore[arg-type]
            return pd.DataFrame({
                "ticket": [10],
                "position_id": [100],
                "symbol": ["EURUSD"],
                "time": ["2024-01-01T12:00:00+00:00"],
                "type": [0],
                "entry": [0],
                "volume": [1.0],
                "price": [1.1],
                "profit": [0.0],
            })

        connected_client.copy_rates_range_as_df.side_effect = make_rates
        connected_client.history_deals_get_as_df.side_effect = make_deals
        mocker.patch("mt5cli.sdk.Mt5DataClient")
        output = tmp_path / "incremental-history.db"

        for _ in range(2):
            update_history(
                client=connected_client,
                output=output,
                symbols=["EURUSD"],
                datasets={Dataset.rates, Dataset.history_deals},
                timeframes=["M1"],
                lookback_hours=24,
                date_to=date_to,
                with_views=True,
            )

        assert rate_starts == [first_expected_start, second_expected_start]
        assert deal_starts == [first_expected_start, second_expected_start]
        connected_client.initialize_and_login_mt5.assert_not_called()
        connected_client.shutdown.assert_not_called()
        with sqlite3.connect(output) as conn:
            assert conn.execute("SELECT COUNT(*) FROM rates").fetchone() == (1,)
            assert conn.execute("SELECT open FROM rates").fetchone() == (1.2,)
            assert conn.execute(
                "SELECT COUNT(*) FROM history_deals",
            ).fetchone() == (1,)
            assert conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'cash_events'",
            ).fetchone() == ("cash_events",)

    def test_update_history_rejects_invalid_inputs(
        self,
        connected_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test validation errors for incremental history updates."""
        output = tmp_path / "invalid-update.db"
        with pytest.raises(ValueError, match="At least one symbol"):
            update_history(
                client=connected_client,
                output=output,
                symbols=[],
            )
        with pytest.raises(ValueError, match="lookback_hours must be positive"):
            update_history(
                client=connected_client,
                output=output,
                symbols=["EURUSD"],
                lookback_hours=0,
            )
        with pytest.raises(ValueError, match="Invalid timeframe"):
            update_history(
                client=connected_client,
                output=output,
                symbols=["EURUSD"],
                datasets={Dataset.rates},
                timeframes=["BAD"],
            )
        with pytest.raises(ValueError, match="Invalid tick flags"):
            update_history(
                client=connected_client,
                output=output,
                symbols=["EURUSD"],
                datasets={Dataset.ticks},
                flags="BAD",
            )

    def test_update_history_noops_for_empty_datasets(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test empty dataset selection skips MT5 and SQLite writes."""
        writer = mocker.patch("mt5cli.sdk.write_incremental_datasets")
        connect = mocker.patch("mt5cli.sdk.sqlite3.connect")
        update_history(
            client=connected_client,
            output=tmp_path / "empty-datasets.db",
            symbols=["EURUSD"],
            datasets=set(),
        )
        writer.assert_not_called()
        connect.assert_not_called()

    def test_update_history_uses_all_default_timeframes(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test that timeframes=None writes rates for all default MT5 timeframes."""
        timeframes_written: list[int] = []

        def capture(
            *args: object,
            **_kwargs: object,
        ) -> tuple[set[Dataset], dict[Dataset, set[str]]]:
            timeframes_written.extend(args[4])  # type: ignore[arg-type]
            return set(), {}

        mocker.patch("mt5cli.sdk.write_incremental_datasets", side_effect=capture)
        update_history(
            client=connected_client,
            output=tmp_path / "default-timeframes.db",
            symbols=["EURUSD"],
            datasets={Dataset.rates},
            timeframes=None,
            lookback_hours=1,
            date_to=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert len(timeframes_written) == len(DEFAULT_HISTORY_TIMEFRAMES)

    def test_update_history_uses_specified_timeframes(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test explicit timeframes limit rate updates."""
        timeframes_written: list[int] = []

        def capture(
            *args: object,
            **_kwargs: object,
        ) -> tuple[set[Dataset], dict[Dataset, set[str]]]:
            timeframes_written.extend(args[4])  # type: ignore[arg-type]
            return set(), {}

        mocker.patch("mt5cli.sdk.write_incremental_datasets", side_effect=capture)
        update_history(
            client=connected_client,
            output=tmp_path / "specific-timeframes.db",
            symbols=["EURUSD"],
            datasets={Dataset.rates},
            timeframes=["M1", "H1"],
            lookback_hours=1,
            date_to=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert timeframes_written == [1, 16385]

    def test_update_history_updates_ticks_and_orders(
        self,
        connected_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test incremental update writes selected ticks and orders datasets."""
        date_to = datetime(2024, 1, 2, tzinfo=UTC)
        expected_start = datetime(2024, 1, 1, tzinfo=UTC)

        def make_ticks(**kwargs: object) -> pd.DataFrame:
            assert kwargs["symbol"] == "EURUSD"
            assert kwargs["date_from"] == expected_start
            assert kwargs["date_to"] == date_to
            assert kwargs["flags"] == 1
            return pd.DataFrame({
                "time": ["2024-01-01T12:00:00+00:00"],
                "time_msc": [1_704_110_400_000],
                "bid": [1.1],
            })

        def make_orders(**kwargs: object) -> pd.DataFrame:
            assert kwargs["symbol"] == "EURUSD"
            assert kwargs["date_from"] == expected_start
            assert kwargs["date_to"] == date_to
            return pd.DataFrame({
                "ticket": [1],
                "symbol": ["EURUSD"],
                "time": ["2024-01-01T12:00:00+00:00"],
                "type": [0],
            })

        connected_client.copy_ticks_range_as_df.side_effect = make_ticks
        connected_client.history_orders_get_as_df.side_effect = make_orders
        output = tmp_path / "ticks-orders.db"
        update_history(
            client=connected_client,
            output=output,
            symbols=["EURUSD"],
            datasets={Dataset.ticks, Dataset.history_orders},
            lookback_hours=24,
            date_to=date_to,
        )
        with sqlite3.connect(output) as conn:
            assert conn.execute("SELECT COUNT(*) FROM ticks").fetchone() == (1,)
            assert conn.execute(
                "SELECT COUNT(*) FROM history_orders",
            ).fetchone() == (1,)

    def test_update_history_with_config_opens_and_closes_connection(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test update_history_with_config manages MT5 connection lifecycle."""
        mock_client = MagicMock()
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        updater = mocker.patch("mt5cli.sdk.update_history")
        update_history_with_config(
            output=tmp_path / "config-wrapper.db",
            symbols=["EURUSD"],
            datasets={Dataset.history_deals},
            lookback_hours=1,
            date_to=datetime(2024, 1, 1, tzinfo=UTC),
        )
        mock_client.initialize_and_login_mt5.assert_called_once()
        mock_client.shutdown.assert_called_once()
        updater.assert_called_once()
        assert updater.call_args.kwargs["client"] is mock_client

    def test_update_history_defaults_date_to_now(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test update_history uses current UTC time when date_to is omitted."""
        captured: dict[str, datetime] = {}

        def capture(
            *args: object,
            **_kwargs: object,
        ) -> tuple[set[Dataset], dict[Dataset, set[str]]]:
            captured["end"] = args[7]  # type: ignore[assignment]
            return set(), {}

        mocker.patch("mt5cli.sdk.write_incremental_datasets", side_effect=capture)
        before = datetime.now(UTC)
        update_history(
            client=connected_client,
            output=tmp_path / "now-default.db",
            symbols=["EURUSD"],
            datasets={Dataset.rates},
            timeframes=["M1"],
            lookback_hours=12,
        )
        after = datetime.now(UTC)
        assert before <= captured["end"] <= after
