"""Tests for mt5cli.sdk module."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple, cast
from unittest.mock import MagicMock, call

import pandas as pd
import pytest
from pdmt5 import Mt5RuntimeError, Mt5TradingError
from pytest_mock import MockerFixture  # noqa: TC002

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pdmt5 import Mt5Config, Mt5DataClient

from mt5cli import sdk
from mt5cli.history import DEFAULT_HISTORY_TIMEFRAMES, write_rates_dataset
from mt5cli.sdk import (
    AccountSpec,
    Mt5CliClient,
    ThrottledHistoryUpdater,
    account_info,
    build_config,
    collect_history,
    collect_latest_closed_rates_by_granularity,
    collect_latest_closed_rates_for_accounts,
    collect_latest_rates,
    collect_latest_rates_for_accounts,
    collect_latest_rates_for_accounts_with_retries,
    copy_rates_from,
    copy_rates_from_pos,
    copy_rates_range,
    copy_ticks_from,
    copy_ticks_range,
    fetch_latest_closed_rates,
    history_deals,
    history_orders,
    last_error,
    latest_rates,
    market_book,
    minimum_margins,
    mt5_session,
    mt5_summary,
    mt5_summary_as_df,
    orders,
    positions,
    recent_history_deals,
    recent_ticks,
    resolve_account_spec,
    resolve_account_specs,
    substitute_env_placeholders,
    substitute_mapping_values,
    symbol_info,
    symbol_info_tick,
    symbols,
    terminal_info,
    update_history,
    update_history_with_config,
    update_observability,
    update_observability_with_config,
    version,
)
from mt5cli.utils import Dataset, IfExists, coerce_login, parse_timeframe


class _TerminalInfo(NamedTuple):
    connected: bool
    path: str


class _AccountInfo(NamedTuple):
    login: int
    limits: dict[str, object]


class _MissingSummaryMethodClient:
    def version(self) -> tuple[int, int, int]:
        return (5, 0, 1)

    def terminal_info(self) -> dict[str, bool]:
        return {"connected": True}

    def symbols_total(self) -> int:
        return 42


class _NonCallableSummaryMethodClient:
    version = (5, 0, 1)


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
        with sdk.connected_client(config):  # type: ignore[reportPrivateUsage]
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
            sdk.connected_client(MagicMock()),  # type: ignore[reportPrivateUsage]
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

    def test_injected_client_is_reused_and_not_shutdown(self) -> None:
        """Test injected connected clients are not initialized or shut down."""
        connected = MagicMock()
        connected.account_info_as_df.return_value = pd.DataFrame({"a": [1]})
        connected.terminal_info_as_df.return_value = pd.DataFrame({"b": [2]})
        with Mt5CliClient.from_connected_client(connected) as client:
            result = client.account_info()
        assert result.to_dict("list") == {"a": [1]}
        connected.initialize_and_login_mt5.assert_not_called()
        connected.shutdown.assert_not_called()
        connected.account_info_as_df.assert_called_once()
        after_exit = client.terminal_info()
        assert after_exit.to_dict("list") == {"b": [2]}
        connected.terminal_info_as_df.assert_called_once()

    def test_constructor_injected_client_is_reused_and_not_shutdown(self) -> None:
        """Test constructor injection has the same non-owning lifecycle."""
        connected = MagicMock()
        connected.terminal_info_as_df.return_value = pd.DataFrame({"b": [2]})
        client = Mt5CliClient(client=connected)
        with client:
            result = client.terminal_info()
        assert result.to_dict("list") == {"b": [2]}
        connected.initialize_and_login_mt5.assert_not_called()
        connected.shutdown.assert_not_called()


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
            (latest_rates, ("EURUSD", "M1", 10), "copy_rates_from_pos_as_df"),
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

    @pytest.mark.parametrize(
        ("call", "expected_method", "expected_kwargs"),
        [
            pytest.param(
                lambda: Mt5CliClient().copy_rates_range(
                    "EURUSD",
                    "D1",
                    "2024-01-01",
                    "2024-02-01",
                ),
                "copy_rates_range_as_df",
                {
                    "symbol": "EURUSD",
                    "timeframe": 16408,
                    "date_from": datetime(2024, 1, 1, tzinfo=UTC),
                    "date_to": datetime(2024, 2, 1, tzinfo=UTC),
                },
                id="copy_rates_range-normalizes-dates-and-timeframe",
            ),
            pytest.param(
                lambda: Mt5CliClient().copy_ticks_from(
                    "EURUSD",
                    "2024-01-01",
                    100,
                    "INFO",
                ),
                "copy_ticks_from_as_df",
                {
                    "symbol": "EURUSD",
                    "date_from": datetime(2024, 1, 1, tzinfo=UTC),
                    "count": 100,
                    "flags": 1,
                },
                id="copy_ticks_from-parses-string-flags",
            ),
            pytest.param(
                lambda: Mt5CliClient().history_orders(
                    date_from="2024-01-01",
                    date_to="2024-02-01",
                ),
                "history_orders_get_as_df",
                {
                    "date_from": datetime(2024, 1, 1, tzinfo=UTC),
                    "date_to": datetime(2024, 2, 1, tzinfo=UTC),
                    "group": None,
                    "symbol": None,
                    "ticket": None,
                    "position": None,
                },
                id="history_orders-parses-string-dates",
            ),
            pytest.param(
                lambda: Mt5CliClient().latest_rates("EURUSD", "M1", 5, start_pos=2),
                "copy_rates_from_pos_as_df",
                {
                    "symbol": "EURUSD",
                    "timeframe": 1,
                    "start_pos": 2,
                    "count": 5,
                },
                id="latest_rates-wraps-copy_rates_from_pos",
            ),
        ],
    )
    def test_method_delegates_with_normalization(
        self,
        mock_client: MagicMock,
        call: Callable[[], object],
        expected_method: str,
        expected_kwargs: dict[str, object],
    ) -> None:
        """Mt5CliClient methods normalize inputs and forward them on."""
        result = call()
        assert isinstance(result, pd.DataFrame)
        getattr(mock_client, expected_method).assert_called_once_with(**expected_kwargs)

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

    def test_latest_rates_rejects_non_positive_count(self) -> None:
        """Test latest_rates validates count."""
        with pytest.raises(ValueError, match="count must be positive"):
            Mt5CliClient().latest_rates("EURUSD", "M1", 0)

    def test_collect_latest_rates_returns_mapping(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Test multi-target latest rate collection."""
        result = collect_latest_rates(["EURUSD", "GBPUSD"], ["M1", "H1"], count=3)
        assert set(result) == {
            ("EURUSD", 1),
            ("EURUSD", 16385),
            ("GBPUSD", 1),
            ("GBPUSD", 16385),
        }
        assert mock_client.copy_rates_from_pos_as_df.call_count == 4

    def test_collect_latest_rates_uses_single_transient_connection(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        """Test module helper opens one connection for all target pairs."""
        mt5_data_client = mocker.patch(
            "mt5cli.sdk.Mt5DataClient",
            return_value=mock_client,
        )

        collect_latest_rates(["EURUSD", "GBPUSD"], ["M1", "H1"], count=3)

        mt5_data_client.assert_called_once()
        mock_client.initialize_and_login_mt5.assert_called_once()
        mock_client.shutdown.assert_called_once()
        assert mock_client.copy_rates_from_pos_as_df.call_count == 4
        mock_client.copy_rates_from_pos_as_df.assert_has_calls(
            [
                call(symbol="EURUSD", timeframe=1, start_pos=0, count=3),
                call(symbol="EURUSD", timeframe=16385, start_pos=0, count=3),
                call(symbol="GBPUSD", timeframe=1, start_pos=0, count=3),
                call(symbol="GBPUSD", timeframe=16385, start_pos=0, count=3),
            ],
        )

    @pytest.mark.parametrize(
        ("symbols", "timeframes", "match"),
        [
            ([], ["M1"], "At least one symbol"),
            (["EURUSD"], [], "At least one timeframe"),
        ],
    )
    def test_collect_latest_rates_rejects_empty_inputs(
        self,
        symbols: list[str],
        timeframes: list[str],
        match: str,
    ) -> None:
        """Test multi-target latest rate input validation."""
        with pytest.raises(ValueError, match=match):
            Mt5CliClient().collect_latest_rates(symbols, timeframes, count=1)

    def test_recent_history_deals_uses_trailing_window(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Test recent_history_deals calculates date_from from hours."""
        result = recent_history_deals(
            6,
            date_to="2024-01-02T00:00:00+00:00",
            group="*",
            symbol="EURUSD",
        )
        assert isinstance(result, pd.DataFrame)
        mock_client.history_deals_get_as_df.assert_called_once_with(
            date_from=datetime(2024, 1, 1, 18, tzinfo=UTC),
            date_to=datetime(2024, 1, 2, tzinfo=UTC),
            group="*",
            symbol="EURUSD",
            ticket=None,
            position=None,
        )

    def test_recent_history_deals_defaults_date_to_now(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Test recent_history_deals uses current UTC time when date_to is omitted."""
        before = datetime.now(UTC)
        recent_history_deals(1.0)
        after = datetime.now(UTC)
        call_kwargs = mock_client.history_deals_get_as_df.call_args.kwargs
        assert before <= call_kwargs["date_to"] <= after
        assert call_kwargs["date_from"] == call_kwargs["date_to"] - timedelta(hours=1)

    def test_recent_history_deals_rejects_non_positive_hours(self) -> None:
        """Test recent_history_deals validates hours."""
        with pytest.raises(ValueError, match="hours must be positive"):
            Mt5CliClient().recent_history_deals(0)

    def test_mt5_summary_returns_status_mapping(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Test mt5_summary calls raw terminal/account status methods."""
        mock_client.version.return_value = (5, 0, 1)
        mock_client.terminal_info.return_value = {"connected": True}
        mock_client.account_info.return_value = {"login": 123}
        mock_client.symbols_total.return_value = 42
        assert mt5_summary() == {
            "version": [5, 0, 1],
            "terminal_info": {"connected": True},
            "account_info": {"login": 123},
            "symbols_total": 42,
        }

    def test_mt5_summary_normalizes_namedtuple_values(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Test mt5_summary returns structured plain Python values."""
        mock_client.version.return_value = (5, 0, 1)
        mock_client.terminal_info.return_value = _TerminalInfo(
            connected=True,
            path="terminal.exe",
        )
        mock_client.account_info.return_value = _AccountInfo(
            login=123,
            limits={"modes": ("netting", "hedging"), "servers": ["demo"]},
        )
        mock_client.symbols_total.return_value = 42

        assert mt5_summary() == {
            "version": [5, 0, 1],
            "terminal_info": {"connected": True, "path": "terminal.exe"},
            "account_info": {
                "login": 123,
                "limits": {"modes": ["netting", "hedging"], "servers": ["demo"]},
            },
            "symbols_total": 42,
        }

    def test_mt5_summary_as_df_stringifies_nested_values(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Test mt5_summary_as_df returns export-safe tabular values."""
        mock_client.version.return_value = (5, 0, 1)
        mock_client.terminal_info.return_value = _TerminalInfo(
            connected=True,
            path="terminal.exe",
        )
        mock_client.account_info.return_value = _AccountInfo(
            login=123,
            limits={"modes": ("netting", "hedging"), "servers": ["demo"]},
        )
        mock_client.symbols_total.return_value = 42

        result = mt5_summary_as_df()

        assert len(result) == 1
        assert result.iloc[0].to_dict() == {
            "version": "[5,0,1]",
            "terminal_info": '{"connected":true,"path":"terminal.exe"}',
            "account_info": (
                '{"limits":{"modes":["netting","hedging"],'
                '"servers":["demo"]},"login":123}'
            ),
            "symbols_total": 42,
        }

    @pytest.mark.parametrize(
        ("client_cls", "exc", "match"),
        [
            (
                _MissingSummaryMethodClient,
                AttributeError,
                "MT5 client is missing required method: account_info",
            ),
            (
                _NonCallableSummaryMethodClient,
                TypeError,
                "MT5 client attribute is not callable: version",
            ),
        ],
        ids=["missing-method", "non-callable-method"],
    )
    def test_mt5_summary_rejects_bad_client(
        self,
        client_cls: type[object],
        exc: type[BaseException],
        match: str,
    ) -> None:
        """Test mt5_summary fails clearly when a required method is bad."""
        client = Mt5CliClient(client=cast("Mt5DataClient", client_cls()))

        with pytest.raises(exc, match=match):
            client.mt5_summary()


class TestCollectHistory:
    """Tests for collect_history SDK function."""

    @pytest.fixture
    def history_client(self, mocker: MockerFixture) -> MagicMock:
        """Create a mocked Mt5DataClient with history-style DataFrames."""
        return _build_history_client(mocker)

    def test_collect_history_writes_default_tables(
        self,
        tmp_path: Path,
        history_client: MagicMock,
    ) -> None:
        """Test that collect_history default excludes ticks."""
        output = tmp_path / "history.db"
        collect_history(
            output,
            ["EURUSD", "GBPUSD"],
            "2024-01-01",
            "2024-02-01",
        )
        assert history_client.copy_rates_range_as_df.call_count == 2
        assert history_client.copy_ticks_range_as_df.call_count == 0
        with sqlite3.connect(output) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
        assert {"rates", "history_orders", "history_deals"} <= tables
        assert "ticks" not in tables

    def test_collect_history_explicit_ticks_dataset(
        self,
        tmp_path: Path,
        history_client: MagicMock,
    ) -> None:
        """Test that explicit datasets={Dataset.ticks} writes the ticks table."""
        output = tmp_path / "history.db"
        collect_history(
            output,
            ["EURUSD", "GBPUSD"],
            "2024-01-01",
            "2024-02-01",
            datasets={Dataset.ticks},
        )
        assert history_client.copy_ticks_range_as_df.call_count == 2
        assert history_client.copy_rates_range_as_df.call_count == 0
        with sqlite3.connect(output) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                ).fetchall()
            }
        assert "ticks" in tables
        assert "rates" not in tables

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
        assert deal_starts == [first_expected_start, first_expected_start]
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

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"symbols": []}, "At least one symbol"),
            (
                {"symbols": ["EURUSD"], "lookback_hours": 0},
                "lookback_hours must be positive",
            ),
            (
                {
                    "symbols": ["EURUSD"],
                    "datasets": {Dataset.rates},
                    "timeframes": ["BAD"],
                },
                "Invalid timeframe",
            ),
            (
                {
                    "symbols": ["EURUSD"],
                    "datasets": {Dataset.ticks},
                    "flags": "BAD",
                },
                "Invalid tick flags",
            ),
        ],
        ids=["empty-symbols", "non-positive-lookback", "bad-timeframe", "bad-flags"],
    )
    def test_update_history_rejects_invalid_inputs(
        self,
        connected_client: MagicMock,
        tmp_path: Path,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        """Test validation errors for incremental history updates."""
        output = tmp_path / "invalid-update.db"
        with pytest.raises(ValueError, match=match):
            update_history(
                client=connected_client,
                output=output,
                **kwargs,  # type: ignore[arg-type]
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

    @pytest.mark.parametrize(
        ("timeframes", "expected"),
        [
            (None, [parse_timeframe(t) for t in DEFAULT_HISTORY_TIMEFRAMES]),
            (["M1", "H1"], [1, 16385]),
        ],
        ids=["default", "specified"],
    )
    def test_update_history_resolves_timeframes(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
        timeframes: list[str] | None,
        expected: list[int],
    ) -> None:
        """Test update_history writes all default or specified rate timeframes."""
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
            output=tmp_path / "timeframes.db",
            symbols=["EURUSD"],
            datasets={Dataset.rates},
            timeframes=timeframes,
            lookback_hours=1,
            date_to=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert timeframes_written == expected

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
            assert kwargs["flags"] == -1
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
            timeframes=["M1"],
            flags="ALL",
            lookback_hours=1,
            date_to=datetime(2024, 1, 1, tzinfo=UTC),
            deduplicate=False,
            create_rate_views=False,
            with_views=True,
            include_account_events=False,
        )
        mock_client.initialize_and_login_mt5.assert_called_once()
        mock_client.shutdown.assert_called_once()
        updater.assert_called_once()
        assert updater.call_args.kwargs == {
            "client": mock_client,
            "output": tmp_path / "config-wrapper.db",
            "symbols": ["EURUSD"],
            "datasets": {Dataset.history_deals},
            "timeframes": ["M1"],
            "flags": "ALL",
            "lookback_hours": 1,
            "date_to": datetime(2024, 1, 1, tzinfo=UTC),
            "deduplicate": False,
            "create_rate_views": False,
            "with_views": True,
            "include_account_events": False,
        }

    def test_update_history_with_config_validates_before_connecting(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test invalid inputs fail before MT5 is initialized."""
        mock_client = MagicMock()
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        with pytest.raises(ValueError, match="lookback_hours must be positive"):
            update_history_with_config(
                output=tmp_path / "invalid-config.db",
                symbols=["EURUSD"],
                lookback_hours=0,
            )
        mock_client.initialize_and_login_mt5.assert_not_called()
        mock_client.shutdown.assert_not_called()

    def test_update_history_with_config_noops_for_empty_datasets(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test empty dataset selection skips MT5 initialization."""
        mock_client = MagicMock()
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        updater = mocker.patch("mt5cli.sdk.update_history")
        update_history_with_config(
            output=tmp_path / "empty-config.db",
            symbols=["EURUSD"],
            datasets=set(),
        )
        mock_client.initialize_and_login_mt5.assert_not_called()
        mock_client.shutdown.assert_not_called()
        updater.assert_not_called()

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

    def test_update_history_default_datasets_exclude_ticks(
        self,
        connected_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Test update_history with datasets=None does not collect ticks."""
        datasets_written: list[set[Dataset]] = []

        def capture(
            *args: object,
            **_kwargs: object,
        ) -> tuple[set[Dataset], dict[Dataset, set[str]]]:
            datasets_written.append(args[3])  # type: ignore[arg-type]
            return set(), {}

        mocker.patch("mt5cli.sdk.write_incremental_datasets", side_effect=capture)
        update_history(
            client=connected_client,
            output=tmp_path / "default-datasets.db",
            symbols=["EURUSD"],
            datasets=None,
            timeframes=["M1"],
            lookback_hours=1,
            date_to=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert len(datasets_written) == 1
        assert Dataset.ticks not in datasets_written[0]
        assert {
            Dataset.rates,
            Dataset.history_orders,
            Dataset.history_deals,
        } == datasets_written[0]


class TestRecentTicks:
    """Tests for recent_ticks helper."""

    def test_recent_ticks_uses_explicit_date_to_window(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test recent_ticks fetches the requested trailing window."""
        client = MagicMock()
        end = datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC)
        client.copy_ticks_from_as_df.return_value = pd.DataFrame({
            "time": [end],
            "bid": [1.0],
        })
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
        result = recent_ticks(
            "EURUSD",
            60,
            date_to=end,
            count=100,
            flags="INFO",
            config=build_config(login=123),
        )
        assert isinstance(result, pd.DataFrame)
        client.copy_ticks_from_as_df.assert_called_once_with(
            symbol="EURUSD",
            date_from=end - timedelta(seconds=60),
            count=100,
            flags=1,
        )
        client.copy_ticks_range_as_df.assert_not_called()

    def test_recent_ticks_uses_latest_tick_when_date_to_omitted(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test recent_ticks anchors the window on the latest tick time."""
        client = MagicMock()
        tick = MagicMock()
        tick.time = datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC)
        client.symbol_info_tick.return_value = tick
        client.copy_ticks_from_as_df.return_value = pd.DataFrame({
            "time": [1, 2],
            "bid": [1.0, 1.1],
        })
        client.copy_ticks_range_as_df.return_value = pd.DataFrame({
            "time": [1, 2, 3],
            "bid": [1.0, 1.1, 1.2],
        })
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
        result = Mt5CliClient().recent_ticks("EURUSD", 30, count=2, flags="ALL")
        assert len(result) == 2
        client.symbol_info_tick.assert_called_once_with("EURUSD")
        client.copy_ticks_from_as_df.assert_called_once()
        _, kwargs = client.copy_ticks_range_as_df.call_args
        assert kwargs["symbol"] == "EURUSD"
        assert kwargs["date_to"] == tick.time
        assert kwargs["date_from"] == tick.time - timedelta(seconds=30)
        assert kwargs["flags"] == -1

    def test_recent_ticks_rejects_unsupported_tick_time(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test recent_ticks raises when the latest tick time is unsupported."""
        client = MagicMock()
        tick = MagicMock()
        tick.time = object()
        client.symbol_info_tick.return_value = tick
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
        with pytest.raises(TypeError, match="Unsupported tick time value"):
            Mt5CliClient().recent_ticks("EURUSD", 30)

    @pytest.mark.parametrize(
        "tick_time",
        [
            "2024-01-02T12:00:00+00:00",
            1704196800,
        ],
    )
    def test_recent_ticks_coerces_string_and_unix_tick_times(
        self,
        mocker: MockerFixture,
        tick_time: str | int,
    ) -> None:
        """Test recent_ticks accepts string and unix tick timestamps."""
        client = MagicMock()
        tick = MagicMock()
        tick.time = tick_time
        client.symbol_info_tick.return_value = tick
        expected_end = (
            datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC)
            if isinstance(tick_time, str)
            else datetime.fromtimestamp(tick_time, tz=UTC)
        )
        client.copy_ticks_from_as_df.return_value = pd.DataFrame({
            "time": [expected_end],
        })
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
        Mt5CliClient().recent_ticks("EURUSD", 30)
        _, kwargs = client.copy_ticks_from_as_df.call_args
        assert kwargs["date_from"] == expected_end - timedelta(seconds=30)

    def test_recent_ticks_returns_full_frame_when_count_not_positive(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test non-positive count returns the full range without trimming."""
        client = MagicMock()
        end = datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC)
        client.copy_ticks_range_as_df.return_value = pd.DataFrame({
            "time": [1, 2, 3],
            "bid": [1.0, 1.1, 1.2],
        })
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)
        result = recent_ticks(
            "EURUSD",
            60,
            date_to=end,
            count=0,
            config=build_config(login=123),
        )
        assert len(result) == 3
        client.copy_ticks_from_as_df.assert_not_called()
        client.copy_ticks_range_as_df.assert_called_once_with(
            symbol="EURUSD",
            date_from=end - timedelta(seconds=60),
            date_to=end,
            flags=-1,
        )


class TestMinimumMargins:
    """Tests for minimum_margins helper."""

    def test_minimum_margins_shape(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test minimum_margins returns the expected summary columns."""
        client = MagicMock()
        sym = MagicMock(volume_min=0.01)
        account = MagicMock(currency="USD")
        tick = MagicMock(ask=1.1010, bid=1.1000)
        client.symbol_info.return_value = sym
        client.account_info.return_value = account
        client.symbol_info_tick.return_value = tick
        client.order_calc_margin.side_effect = [12.5, 12.4]
        client.mt5.ORDER_TYPE_BUY = 0
        client.mt5.ORDER_TYPE_SELL = 1
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=client)

        result = minimum_margins("EURUSD", config=build_config(login=123))

        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame([
                {
                    "symbol": "EURUSD",
                    "account_currency": "USD",
                    "volume_min": 0.01,
                    "buy_margin": 12.5,
                    "sell_margin": 12.4,
                }
            ]),
        )
        client.order_calc_margin.assert_any_call(0, "EURUSD", 0.01, 1.1010)
        client.order_calc_margin.assert_any_call(1, "EURUSD", 0.01, 1.1000)


class TestMt5Session:
    """Tests for the mt5_session context manager."""

    def test_yields_connected_client_and_shuts_down(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test mt5_session connects, yields a client wrapper, and shuts down."""
        mock_client = MagicMock()
        mt5_data_client = mocker.patch(
            "mt5cli.sdk.Mt5DataClient",
            return_value=mock_client,
        )

        with mt5_session(build_config(path="/opt/mt5/terminal64.exe")) as client:
            mock_client.initialize_and_login_mt5.assert_called_once()
            assert isinstance(client, Mt5CliClient)

        config = mt5_data_client.call_args.kwargs["config"]
        assert config.path == "/opt/mt5/terminal64.exe"
        mock_client.shutdown.assert_called_once()

    def test_default_config_attaches_to_running_terminal(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test mt5_session builds a default config when none is supplied."""
        mock_client = MagicMock()
        mt5_data_client = mocker.patch(
            "mt5cli.sdk.Mt5DataClient",
            return_value=mock_client,
        )

        with mt5_session():
            pass

        mt5_data_client.assert_called_once()
        mock_client.shutdown.assert_called_once()


class TestAccountSpec:
    """Tests for account configuration helpers."""

    def test_repr_omits_password(self) -> None:
        """Test AccountSpec repr does not expose plaintext passwords."""
        spec = AccountSpec(symbols=["EURUSD"], login=123, password="secret")

        assert "secret" not in repr(spec)
        assert "password" not in repr(spec)

    @pytest.mark.parametrize(
        ("login", "expected"),
        [
            (None, None),
            (123, 123),
            ("", None),
            ("   ", None),
            ("456", 456),
        ],
    )
    def test_coerce_login(
        self,
        login: int | str | None,
        expected: int | None,
    ) -> None:
        """Test login values are normalized for account configs."""
        assert coerce_login(login) == expected

    def test_coerce_login_rejects_non_numeric_string(self) -> None:
        """Test non-numeric login strings raise ValueError."""
        with pytest.raises(ValueError, match="invalid literal"):
            coerce_login("abc")


class TestCollectLatestRatesForAccounts:
    """Tests for collect_latest_rates_for_accounts."""

    def test_merges_results_across_accounts(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        """Test rates are collected and merged for each account group."""
        mt5_data_client = mocker.patch(
            "mt5cli.sdk.Mt5DataClient",
            return_value=mock_client,
        )
        accounts = [
            AccountSpec(symbols=["EURUSD"], login="123"),
            AccountSpec(symbols=["GBPUSD"], login=456),
        ]

        result = collect_latest_rates_for_accounts(accounts, ["M1"], count=2)

        assert set(result) == {("EURUSD", 1), ("GBPUSD", 1)}
        assert mt5_data_client.call_count == 2
        assert mock_client.initialize_and_login_mt5.call_count == 2
        assert mock_client.shutdown.call_count == 2

    def test_builds_config_from_account_and_base(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        """Test account fields override base_config, empty login falls back."""
        configs: list[object] = []

        def _record_config(*, config: object, **_: object) -> MagicMock:
            configs.append(config)
            return mock_client

        mocker.patch("mt5cli.sdk.Mt5DataClient", side_effect=_record_config)
        base = build_config(login=999, server="Base-Server", timeout=5000)
        accounts = [
            AccountSpec(symbols=["EURUSD"], login="", server="Acct-Server"),
        ]

        collect_latest_rates_for_accounts(accounts, ["M1"], count=1, base_config=base)

        assert len(configs) == 1
        config = cast("Mt5Config", configs[0])
        assert config.login == 999
        assert config.server == "Acct-Server"
        assert config.timeout == 5000

    @pytest.mark.parametrize(
        ("accounts", "timeframes", "count", "match"),
        [
            ([], ["M1"], 1, "At least one account"),
            ([AccountSpec(symbols=["EURUSD"])], [], 1, "At least one timeframe"),
            (
                [AccountSpec(symbols=[])],
                ["M1"],
                1,
                "Each account requires at least one symbol",
            ),
            (
                [AccountSpec(symbols=["EURUSD"])],
                ["M1"],
                0,
                "count must be positive",
            ),
        ],
    )
    def test_rejects_invalid_inputs(
        self,
        accounts: list[AccountSpec],
        timeframes: list[str],
        count: int,
        match: str,
    ) -> None:
        """Test input validation for account-level rate collection."""
        with pytest.raises(ValueError, match=match):
            collect_latest_rates_for_accounts(accounts, timeframes, count)

    def test_rejects_empty_symbols_before_connecting(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test all account symbols are validated before any MT5 connection."""
        mt5_data_client = mocker.patch("mt5cli.sdk.Mt5DataClient")
        accounts = [
            AccountSpec(symbols=["EURUSD"], login=123),
            AccountSpec(symbols=[], login=456),
        ]

        with pytest.raises(
            ValueError, match="Each account requires at least one symbol"
        ):
            collect_latest_rates_for_accounts(accounts, ["M1"], count=1)

        mt5_data_client.assert_not_called()


class TestCollectLatestRatesForAccountsWithRetries:
    """Tests for collect_latest_rates_for_accounts_with_retries."""

    def test_returns_result_on_first_success(self, mocker: MockerFixture) -> None:
        """Test no retry happens when the first attempt succeeds."""
        expected = {("EURUSD", 1): pd.DataFrame()}
        wrapped = mocker.patch(
            "mt5cli.sdk.collect_latest_rates_for_accounts",
            return_value=expected,
        )
        sleep = mocker.patch("mt5cli.sdk.time.sleep")
        accounts = [AccountSpec(symbols=["EURUSD"])]

        result = collect_latest_rates_for_accounts_with_retries(
            accounts,
            ["M1"],
            count=1,
            retry_count=3,
        )

        assert result is expected
        assert wrapped.call_count == 1
        sleep.assert_not_called()

    def test_retries_then_succeeds(self, mocker: MockerFixture) -> None:
        """Test transient MT5 errors are retried with exponential backoff."""
        expected = {("EURUSD", 1): pd.DataFrame()}
        wrapped = mocker.patch(
            "mt5cli.sdk.collect_latest_rates_for_accounts",
            side_effect=[
                Mt5TradingError("boom"),
                Mt5RuntimeError("boom"),
                expected,
            ],
        )
        sleep = mocker.patch("mt5cli.sdk.time.sleep")
        accounts = [AccountSpec(symbols=["EURUSD"])]

        result = collect_latest_rates_for_accounts_with_retries(
            accounts,
            ["M1"],
            count=1,
            retry_count=2,
            backoff_base=2,
        )

        assert result is expected
        assert wrapped.call_count == 3
        assert sleep.call_args_list == [call(2), call(4)]

    def test_reraises_after_exhausting_retries(self, mocker: MockerFixture) -> None:
        """Test the final error is re-raised once retries are exhausted."""
        wrapped = mocker.patch(
            "mt5cli.sdk.collect_latest_rates_for_accounts",
            side_effect=Mt5RuntimeError("boom"),
        )
        sleep = mocker.patch("mt5cli.sdk.time.sleep")
        accounts = [AccountSpec(symbols=["EURUSD"])]

        with pytest.raises(Mt5RuntimeError, match="boom"):
            collect_latest_rates_for_accounts_with_retries(
                accounts,
                ["M1"],
                count=1,
                retry_count=2,
            )

        assert wrapped.call_count == 3
        assert sleep.call_count == 2

    def test_does_not_retry_unrelated_errors(self, mocker: MockerFixture) -> None:
        """Test non-MT5 errors propagate without retrying."""
        wrapped = mocker.patch(
            "mt5cli.sdk.collect_latest_rates_for_accounts",
            side_effect=ValueError("bad input"),
        )
        sleep = mocker.patch("mt5cli.sdk.time.sleep")

        with pytest.raises(ValueError, match="bad input"):
            collect_latest_rates_for_accounts_with_retries(
                [AccountSpec(symbols=["EURUSD"])],
                ["M1"],
                count=1,
                retry_count=3,
            )

        assert wrapped.call_count == 1
        sleep.assert_not_called()


class TestCollectLatestClosedRatesForAccounts:
    """Tests for collect_latest_closed_rates_for_accounts."""

    def test_fetches_count_plus_one_and_drops_forming_bar(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test closed-bar collection requests one extra bar at start_pos=0."""
        df_rate = pd.DataFrame({"time": [1, 2, 3], "close": [1.1, 1.2, 1.3]})
        wrapped = mocker.patch(
            "mt5cli.sdk.collect_latest_rates_for_accounts_with_retries",
            return_value={("EURUSD", 1): df_rate},
        )
        accounts = [AccountSpec(symbols=["EURUSD"])]

        result = collect_latest_closed_rates_for_accounts(
            accounts,
            ["M1"],
            count=2,
            retry_count=1,
            backoff_base=3,
        )

        wrapped.assert_called_once_with(
            accounts,
            ["M1"],
            3,
            start_pos=0,
            base_config=None,
            retry_count=1,
            backoff_base=3,
        )
        pd.testing.assert_frame_equal(
            result["EURUSD", 1],
            pd.DataFrame({"time": [1, 2], "close": [1.1, 1.2]}),
        )

    def test_rejects_forming_bar_only_frames(self, mocker: MockerFixture) -> None:
        """Test empty results after dropping the forming bar raise ValueError."""
        mocker.patch(
            "mt5cli.sdk.collect_latest_rates_for_accounts_with_retries",
            return_value={("EURUSD", 1): pd.DataFrame({"time": [1], "close": [1.1]})},
        )

        with pytest.raises(ValueError, match="Rate data is empty"):
            collect_latest_closed_rates_for_accounts(
                [AccountSpec(symbols=["EURUSD"])],
                ["M1"],
                count=1,
            )

    def test_skips_extra_fetch_when_start_pos_nonzero(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test start_pos > 0 fetches count bars without dropping the last row."""
        df_rate = pd.DataFrame({"time": [1, 2], "close": [1.1, 1.2]})
        wrapped = mocker.patch(
            "mt5cli.sdk.collect_latest_rates_for_accounts_with_retries",
            return_value={("EURUSD", 1): df_rate},
        )

        result = collect_latest_closed_rates_for_accounts(
            [AccountSpec(symbols=["EURUSD"])],
            ["M1"],
            count=2,
            start_pos=1,
        )

        wrapped.assert_called_once_with(
            [AccountSpec(symbols=["EURUSD"])],
            ["M1"],
            2,
            start_pos=1,
            base_config=None,
            retry_count=0,
            backoff_base=2.0,
        )
        pd.testing.assert_frame_equal(result["EURUSD", 1], df_rate)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"count": 0}, "count must be positive"),
            ({"count": 1, "start_pos": -1}, "start_pos must be non-negative"),
        ],
        ids=["zero-count", "negative-start-pos"],
    )
    def test_rejects_invalid_inputs_before_fetching(
        self,
        mocker: MockerFixture,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        """Test invalid count/start_pos values are rejected before MT5 is called."""
        wrapped = mocker.patch(
            "mt5cli.sdk.collect_latest_rates_for_accounts_with_retries",
        )

        with pytest.raises(ValueError, match=match):
            collect_latest_closed_rates_for_accounts(
                [AccountSpec(symbols=["EURUSD"])],
                ["M1"],
                **kwargs,  # type: ignore[arg-type]
            )

        wrapped.assert_not_called()

    def test_rejects_empty_frames_with_start_pos_nonzero(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test empty upstream frames raise ValueError when start_pos > 0."""
        mocker.patch(
            "mt5cli.sdk.collect_latest_rates_for_accounts_with_retries",
            return_value={("EURUSD", 1): pd.DataFrame(columns=["time", "close"])},
        )

        with pytest.raises(ValueError, match="Rate data is empty"):
            collect_latest_closed_rates_for_accounts(
                [AccountSpec(symbols=["EURUSD"])],
                ["M1"],
                count=1,
                start_pos=1,
            )

    def test_processes_multiple_symbol_timeframe_pairs(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test each returned series is trimmed and validated independently."""
        mocker.patch(
            "mt5cli.sdk.collect_latest_rates_for_accounts_with_retries",
            return_value={
                ("EURUSD", 1): pd.DataFrame(
                    {"time": [1, 2, 3], "close": [1.1, 1.2, 1.3]},
                ),
                ("GBPUSD", 16385): pd.DataFrame(
                    {"time": [4, 5, 6], "close": [2.1, 2.2, 2.3]},
                ),
            },
        )

        result = collect_latest_closed_rates_for_accounts(
            [AccountSpec(symbols=["EURUSD", "GBPUSD"])],
            ["M1", "H1"],
            count=2,
        )

        assert set(result) == {("EURUSD", 1), ("GBPUSD", 16385)}
        pd.testing.assert_frame_equal(
            result["EURUSD", 1],
            pd.DataFrame({"time": [1, 2], "close": [1.1, 1.2]}),
        )
        pd.testing.assert_frame_equal(
            result["GBPUSD", 16385],
            pd.DataFrame({"time": [4, 5], "close": [2.1, 2.2]}),
        )


class TestFetchLatestClosedRates:
    """Tests for fetch_latest_closed_rates."""

    def test_fetches_extra_bar_and_drops_forming_row(self) -> None:
        """Test single-symbol closed-bar helper hides the forming bar."""
        client = MagicMock()
        client.latest_rates.return_value = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "close": [1.0, 1.1, 1.2],
            },
        )

        result = fetch_latest_closed_rates(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        client.latest_rates.assert_called_once_with(
            "EURUSD",
            "M1",
            3,
            start_pos=0,
        )
        assert list(result["close"]) == [1.0, 1.1]

    def test_raises_when_no_closed_bars_are_available(self) -> None:
        """Test empty closed-bar results raise an actionable ValueError."""
        client = MagicMock()
        client.latest_rates.return_value = pd.DataFrame({"close": [1.0]})

        with pytest.raises(ValueError, match="Rate data is empty"):
            fetch_latest_closed_rates(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=1,
            )

    def test_rejects_non_positive_count_before_fetching(self) -> None:
        """Test invalid count values fail before calling MT5."""
        client = MagicMock()

        with pytest.raises(ValueError, match="count must be positive"):
            fetch_latest_closed_rates(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=0,
            )

        client.latest_rates.assert_not_called()


class TestCollectLatestClosedRatesByGranularity:
    """Tests for collect_latest_closed_rates_by_granularity."""

    def test_rekeys_by_granularity_name(self, mocker: MockerFixture) -> None:
        """Test closed rates are keyed by symbol and granularity name."""
        df_rate = pd.DataFrame({"time": [1, 2], "close": [1.1, 1.2]})
        wrapped = mocker.patch(
            "mt5cli.sdk.collect_latest_closed_rates_for_accounts",
            return_value={("EURUSD", 1): df_rate},
        )

        result = collect_latest_closed_rates_by_granularity(
            [AccountSpec(symbols=["EURUSD"])],
            ["M1"],
            count=2,
        )

        wrapped.assert_called_once_with(
            [AccountSpec(symbols=["EURUSD"])],
            ["M1"],
            2,
            start_pos=0,
            base_config=None,
            retry_count=0,
            backoff_base=2.0,
        )
        assert ("EURUSD", "M1") in result
        pd.testing.assert_frame_equal(result["EURUSD", "M1"], df_rate)


class TestSubstituteEnvPlaceholders:
    """Tests for ${ENV_VAR} substitution."""

    @pytest.mark.parametrize(
        ("env", "input_", "allow_whole_dollar_env", "expected"),
        [
            pytest.param(
                {"MT5_LOGIN": "12345", "MT5_SERVER": "Broker-Demo"},
                "${MT5_LOGIN}",
                False,
                "12345",
                id="brace-substitution",
            ),
            pytest.param(
                {"MT5_LOGIN": "12345", "MT5_SERVER": "Broker-Demo"},
                "srv=${MT5_SERVER}!",
                False,
                "srv=Broker-Demo!",
                id="brace-substitution-embedded",
            ),
            pytest.param(
                {},
                "plain",
                False,
                "plain",
                id="plain-string-unchanged",
            ),
            pytest.param(
                {"MT5_PASSWORD": "secret"},
                "$MT5_PASSWORD",
                False,
                "$MT5_PASSWORD",
                id="whole-dollar-not-substituted-by-default",
            ),
            pytest.param(
                {"MT5_PASSWORD": "secret"},
                "$MT5_PASSWORD",
                True,
                "secret",
                id="whole-dollar-substituted-with-opt-in",
            ),
            pytest.param(
                {"pass": "secret", "ENV": "val"},
                "plan$pass",
                True,
                "plan$pass",
                id="partial-dollar-prefix-not-expanded",
            ),
            pytest.param(
                {"pass": "secret", "ENV": "val"},
                "abc$ENV",
                True,
                "abc$ENV",
                id="partial-env-suffix-not-expanded",
            ),
            pytest.param(
                {"ENV": "val"},
                "$ENV-suffix",
                True,
                "$ENV-suffix",
                id="whole-dollar-with-suffix-not-expanded",
            ),
            pytest.param(
                {"MT5_LOGIN": "12345"},
                "${MT5_LOGIN}",
                True,
                "12345",
                id="brace-substitution-with-opt-in",
            ),
        ],
    )
    def test_substitute_env_placeholders(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
        input_: str,
        allow_whole_dollar_env: bool,
        expected: str,
    ) -> None:
        """Handle ${ENV}, $ENV, plain, and partial forms of substitution."""
        for name, value in env.items():
            monkeypatch.setenv(name, value)

        result = substitute_env_placeholders(
            input_,
            allow_whole_dollar_env=allow_whole_dollar_env,
        )

        assert result == expected

    @pytest.mark.parametrize(
        ("input_", "allow_whole_dollar_env"),
        [
            pytest.param("${MT5_MISSING}", False, id="brace-missing"),
            pytest.param("$MT5_MISSING", True, id="whole-dollar-missing"),
        ],
    )
    def test_substitute_env_placeholders_raises_on_missing_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        input_: str,
        allow_whole_dollar_env: bool,
    ) -> None:
        """Missing env vars raise ValueError for both ${ENV} and $ENV (opt-in) forms."""
        monkeypatch.delenv("MT5_MISSING", raising=False)

        with pytest.raises(ValueError, match="'MT5_MISSING' is not set"):
            substitute_env_placeholders(
                input_,
                allow_whole_dollar_env=allow_whole_dollar_env,
            )


class TestResolveAccountSpec:
    """Tests for resolve_account_spec and resolve_account_specs."""

    def test_substitutes_env_placeholders_in_account(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test account string fields resolve ${ENV_VAR} placeholders."""
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        account = AccountSpec(
            symbols=["EURUSD"],
            login="${MT5_LOGIN}",
            password="${MT5_PASSWORD}",
        )
        monkeypatch.setenv("MT5_LOGIN", "999")

        resolved = resolve_account_spec(account)

        assert resolved.login == "999"
        assert resolved.password == "secret"  # noqa: S105
        assert resolved.symbols == ["EURUSD"]

    def test_explicit_overrides_take_precedence(self) -> None:
        """Test explicit override values win over account fields."""
        account = AccountSpec(symbols=["EURUSD"], login=111, server="Acct")

        resolved = resolve_account_spec(
            account,
            login=222,
            server="Override",
            timeout=5000,
        )

        assert resolved.login == 222
        assert resolved.server == "Override"
        assert resolved.timeout == 5000

    def test_resolves_string_login_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test string login overrides expand ${ENV_VAR} placeholders."""
        monkeypatch.setenv("MT5_LOGIN", "777")
        account = AccountSpec(symbols=["EURUSD"], login=111)

        resolved = resolve_account_spec(account, login="${MT5_LOGIN}")

        assert resolved.login == "777"

    def test_preserves_integer_login_without_coercion(self) -> None:
        """Test integer logins remain integers after resolution."""
        account = AccountSpec(symbols=["EURUSD"], login=111)

        resolved = resolve_account_spec(account)

        assert resolved.login == 111
        assert isinstance(resolved.login, int)

    def test_raises_on_missing_env_variable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test missing environment variables raise ValueError."""
        monkeypatch.delenv("MT5_NOPE", raising=False)
        account = AccountSpec(symbols=["EURUSD"], server="${MT5_NOPE}")

        with pytest.raises(ValueError, match="'MT5_NOPE' is not set"):
            resolve_account_spec(account)

    def test_resolve_account_specs_applies_to_all(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test resolve_account_specs resolves every account in order."""
        monkeypatch.setenv("MT5_SERVER", "Shared")
        accounts = [
            AccountSpec(symbols=["EURUSD"], server="${MT5_SERVER}"),
            AccountSpec(symbols=["GBPUSD"], server="Fixed"),
        ]

        resolved = resolve_account_specs(accounts, timeout=1000)

        assert [a.server for a in resolved] == ["Shared", "Fixed"]
        assert all(a.timeout == 1000 for a in resolved)

    @pytest.mark.parametrize(
        ("allow_whole_dollar_env", "expected"),
        [
            (True, "secret"),
            (False, "$MT5_PASSWORD"),
        ],
    )
    def test_resolve_account_spec_whole_dollar_password(
        self,
        monkeypatch: pytest.MonkeyPatch,
        allow_whole_dollar_env: bool,
        expected: str,
    ) -> None:
        """Test resolve_account_spec expands $ENV_NAME password only with opt-in."""
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        account = AccountSpec(symbols=["EURUSD"], password="$MT5_PASSWORD")

        resolved = resolve_account_spec(
            account, allow_whole_dollar_env=allow_whole_dollar_env
        )

        assert resolved.password == expected

    def test_resolve_account_specs_with_whole_dollar_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test resolve_account_specs threads allow_whole_dollar_env to each account."""
        monkeypatch.setenv("MT5_SERVER", "Broker-Demo")
        accounts = [
            AccountSpec(symbols=["EURUSD"], server="$MT5_SERVER"),
            AccountSpec(symbols=["GBPUSD"], server="Fixed"),
        ]

        resolved = resolve_account_specs(accounts, allow_whole_dollar_env=True)

        assert resolved[0].server == "Broker-Demo"
        assert resolved[1].server == "Fixed"

    def test_resolve_account_spec_whole_dollar_login(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test $ENV_NAME login string is expanded when allow_whole_dollar_env=True."""
        monkeypatch.setenv("MT5_LOGIN", "12345")
        account = AccountSpec(symbols=["EURUSD"], login="$MT5_LOGIN")

        resolved = resolve_account_spec(account, allow_whole_dollar_env=True)

        assert resolved.login == "12345"


class TestBuildConfigWholeDollarEnv:
    """Tests for build_config with allow_whole_dollar_env."""

    @pytest.mark.parametrize(
        ("env_var", "field", "env_value"),
        [
            ("MT5_SERVER", "server", "Broker-Demo"),
            ("MT5_PASSWORD", "password", "secret"),
            ("MT5_PATH", "path", "/opt/mt5/terminal64.exe"),
        ],
    )
    def test_build_config_substitutes_field_with_opt_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_var: str,
        field: str,
        env_value: str,
    ) -> None:
        """Test build_config expands $ENV_NAME fields when opt-in is enabled."""
        monkeypatch.setenv(env_var, env_value)

        config = build_config(**{field: f"${env_var}"}, allow_whole_dollar_env=True)  # type: ignore[arg-type]

        assert getattr(config, field) == env_value

    def test_build_config_leaves_dollar_literal_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test build_config does not substitute $ENV without opt-in."""
        monkeypatch.setenv("MT5_SERVER", "Broker-Demo")

        config = build_config(server="$MT5_SERVER")

        assert config.server == "$MT5_SERVER"

    def test_build_config_none_params_not_substituted(
        self,
        monkeypatch: pytest.MonkeyPatch,  # noqa: ARG002
    ) -> None:
        """Test build_config with None params does not raise even with opt-in."""
        config = build_config(allow_whole_dollar_env=True)

        assert config.server is None
        assert config.password is None
        assert config.path is None


class TestThrottledHistoryUpdater:
    """Tests for the throttled incremental history updater."""

    def test_updates_every_call_when_interval_non_positive(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test interval_seconds <= 0 updates on every call."""
        update = mocker.patch("mt5cli.sdk.update_history")
        client = MagicMock()
        updater = ThrottledHistoryUpdater(output="history.db", interval_seconds=0)

        assert updater.update(client, ["EURUSD"]) is True
        assert updater.update(client, ["EURUSD"]) is True
        assert update.call_count == 2

    def test_throttles_within_interval(self, mocker: MockerFixture) -> None:
        """Test updates are skipped until the interval elapses."""
        update = mocker.patch("mt5cli.sdk.update_history")
        monotonic = mocker.patch("mt5cli.sdk.time.monotonic")
        # Calls: set(t=100), check(t=105), check(t=200), set(t=200).
        monotonic.side_effect = [100.0, 105.0, 200.0, 200.0]
        client = MagicMock()
        updater = ThrottledHistoryUpdater(output="history.db", interval_seconds=60)

        assert updater.update(client, ["EURUSD"]) is True  # first update at t=100
        assert updater.update(client, ["EURUSD"]) is False  # t=105, throttled
        assert updater.update(client, ["EURUSD"]) is True  # t=200, elapsed
        assert update.call_count == 2

    def test_update_passes_expected_arguments(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test update_history is called with the configured arguments."""
        update = mocker.patch("mt5cli.sdk.update_history")
        client = MagicMock()
        updater = ThrottledHistoryUpdater(
            output="history.db",
            datasets={Dataset.rates},
            timeframes=["M1", "H1"],
            flags="INFO",
            lookback_hours=12.0,
            with_views=True,
            include_account_events=False,
        )

        updater.update(client, ["EURUSD", "GBPUSD"])

        update.assert_called_once_with(
            client=client,
            output="history.db",
            symbols=["EURUSD", "GBPUSD"],
            datasets={Dataset.rates},
            timeframes=["M1", "H1"],
            flags="INFO",
            lookback_hours=12.0,
            with_views=True,
            include_account_events=False,
        )

    def test_propagates_errors_by_default(self, mocker: MockerFixture) -> None:
        """Test MT5/SQLite errors propagate and do not advance the throttle."""
        mocker.patch(
            "mt5cli.sdk.update_history",
            side_effect=Mt5RuntimeError("boom"),
        )
        updater = ThrottledHistoryUpdater(output="history.db")

        with pytest.raises(Mt5RuntimeError, match="boom"):
            updater.update(MagicMock(), ["EURUSD"])

        assert updater.last_update_monotonic is None

    @pytest.mark.parametrize(
        "error",
        [
            Mt5RuntimeError("boom"),
            Mt5TradingError("trade failed"),
            sqlite3.OperationalError("locked"),
            ValueError("invalid symbols"),
            OSError("disk full"),
            AttributeError(
                "'StubClient' object has no attribute 'copy_rates_range_as_df'",
                name="copy_rates_range_as_df",
            ),
            AttributeError(
                "MT5 client is missing required method: copy_ticks_range_as_df"
            ),
            TypeError("MT5 client attribute is not callable: history_orders_get_as_df"),
        ],
    )
    def test_suppresses_errors_when_requested(
        self,
        mocker: MockerFixture,
        error: Exception,
    ) -> None:
        """Test suppress_errors swallows recoverable errors and returns False."""
        mocker.patch(
            "mt5cli.sdk.update_history",
            side_effect=error,
        )
        updater = ThrottledHistoryUpdater(
            output="history.db",
            suppress_errors=True,
        )

        assert updater.update(MagicMock(), ["EURUSD"]) is False
        assert updater.last_update_monotonic is None

    @pytest.mark.parametrize(
        "error",
        [
            AttributeError("'dict' object has no attribute 'typo'"),
            TypeError("unsupported operand types"),
        ],
    )
    def test_suppress_errors_does_not_hide_programming_errors(
        self,
        mocker: MockerFixture,
        error: Exception,
    ) -> None:
        """Test generic AttributeError/TypeError still propagate when suppressed."""
        mocker.patch(
            "mt5cli.sdk.update_history",
            side_effect=error,
        )
        updater = ThrottledHistoryUpdater(
            output="history.db",
            suppress_errors=True,
        )

        with pytest.raises(type(error)):
            updater.update(MagicMock(), ["EURUSD"])

        assert updater.last_update_monotonic is None

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (AttributeError("MT5 client is missing required method: version"), True),
            (
                AttributeError(
                    "'Stub' object has no attribute 'copy_rates_range_as_df'",
                    name="copy_rates_range_as_df",
                ),
                True,
            ),
            (AttributeError("'dict' object has no attribute 'typo'"), False),
            (TypeError("MT5 client attribute is not callable: version"), True),
            (TypeError("unsupported operand types"), False),
            (TypeError("'NoneType' object is not callable"), False),
            (ValueError("invalid"), False),
        ],
    )
    def test_is_mt5_client_capability_error(
        self,
        error: BaseException,
        expected: bool,
    ) -> None:
        """Test MT5 client capability error detection."""
        assert sdk._is_mt5_client_capability_error(error) is expected  # type: ignore[reportPrivateUsage]

    def test_is_mt5_client_capability_error_for_non_callable_history_client(
        self,
    ) -> None:
        """Test non-callable history client attributes are capability errors."""
        client = MagicMock()
        client.copy_rates_range_as_df = None
        with (
            sqlite3.connect(":memory:") as conn,
            pytest.raises(TypeError, match="not callable") as exc_info,
        ):
            write_rates_dataset(
                conn,
                client,
                ["EURUSD"],
                1,
                datetime.now(UTC),
                datetime.now(UTC),
                IfExists.APPEND,
                {},
            )

        assert sdk._is_mt5_client_capability_error(exc_info.value) is True  # type: ignore[reportPrivateUsage]

    def test_suppresses_non_callable_history_client_method(
        self,
        tmp_path: Path,
    ) -> None:
        """Test suppress_errors swallows non-callable history client API attributes."""
        client = MagicMock()
        client.copy_rates_range_as_df = None
        updater = ThrottledHistoryUpdater(
            output=tmp_path / "history.db",
            datasets={Dataset.rates},
            timeframes=["M1"],
            suppress_errors=True,
        )

        assert updater.update(client, ["EURUSD"]) is False
        assert updater.last_update_monotonic is None

    def test_suppress_errors_does_not_hide_internal_client_type_error(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test TypeError raised inside a callable client method still propagates."""
        mocker.patch(
            "mt5cli.sdk.update_history",
            side_effect=TypeError("'int' object is not callable"),
        )
        updater = ThrottledHistoryUpdater(
            output="history.db",
            suppress_errors=True,
        )

        with pytest.raises(TypeError, match="not callable"):
            updater.update(MagicMock(), ["EURUSD"])

        assert updater.last_update_monotonic is None

    def test_suppresses_validation_errors_before_update(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test validation failures are suppressed without calling update_history."""
        update = mocker.patch("mt5cli.sdk.update_history")
        updater = ThrottledHistoryUpdater(
            output="history.db",
            suppress_errors=True,
        )

        assert updater.update(MagicMock(), []) is False
        update.assert_not_called()
        assert updater.last_update_monotonic is None

    def test_default_update_backend_is_update_history(self) -> None:
        """Test the default backend resolves to update_history."""
        updater = ThrottledHistoryUpdater(output="history.db")
        assert updater.update_backend is update_history

    def test_falsy_callable_update_backend_is_preserved(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test only None selects the default backend, not falsy callables."""

        class FalsyCallable:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __bool__(self) -> bool:
                return False

            def __call__(self, **kwargs: object) -> None:
                self.calls.append(kwargs)

        falsy_backend = FalsyCallable()
        default_backend = mocker.patch("mt5cli.sdk.update_history")
        updater = ThrottledHistoryUpdater(
            output="history.db",
            update_backend=falsy_backend,
        )

        assert updater.update_backend is falsy_backend
        client = MagicMock()
        assert updater.update(client, ["EURUSD"]) is True
        assert len(falsy_backend.calls) == 1
        assert falsy_backend.calls[0]["client"] is client
        assert falsy_backend.calls[0]["symbols"] == ["EURUSD"]
        default_backend.assert_not_called()

    def test_custom_update_backend_receives_expected_kwargs(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test a custom backend receives update_history keyword arguments."""
        backend = mocker.Mock()
        client = MagicMock()
        updater = ThrottledHistoryUpdater(
            output="history.db",
            datasets={Dataset.rates},
            timeframes=["M1", "H1"],
            flags="INFO",
            lookback_hours=12.0,
            with_views=True,
            include_account_events=False,
            update_backend=backend,
        )

        updater.update(client, ["EURUSD", "GBPUSD"])

        backend.assert_called_once_with(
            client=client,
            output="history.db",
            symbols=["EURUSD", "GBPUSD"],
            datasets={Dataset.rates},
            timeframes=["M1", "H1"],
            flags="INFO",
            lookback_hours=12.0,
            with_views=True,
            include_account_events=False,
        )

    def test_throttled_calls_do_not_invoke_custom_backend(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test throttled update cycles skip the injected backend."""
        backend = mocker.Mock()
        monotonic = mocker.patch("mt5cli.sdk.time.monotonic")
        monotonic.side_effect = [100.0, 105.0, 200.0, 200.0]
        client = MagicMock()
        updater = ThrottledHistoryUpdater(
            output="history.db",
            interval_seconds=60,
            update_backend=backend,
        )

        assert updater.update(client, ["EURUSD"]) is True
        assert updater.update(client, ["EURUSD"]) is False
        assert updater.update(client, ["EURUSD"]) is True
        assert backend.call_count == 2

    def test_successful_custom_backend_advances_throttle(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test a successful custom backend updates _last_update_monotonic."""
        backend = mocker.Mock()
        monotonic = mocker.patch("mt5cli.sdk.time.monotonic", return_value=42.0)
        updater = ThrottledHistoryUpdater(
            output="history.db",
            update_backend=backend,
        )

        assert updater.update(MagicMock(), ["EURUSD"]) is True
        assert updater.last_update_monotonic is monotonic.return_value
        monotonic.assert_called_once()

    def test_failed_custom_backend_does_not_advance_throttle(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test a failing custom backend leaves _last_update_monotonic unchanged."""
        backend = mocker.Mock(side_effect=Mt5RuntimeError("boom"))
        updater = ThrottledHistoryUpdater(
            output="history.db",
            update_backend=backend,
        )

        with pytest.raises(Mt5RuntimeError, match="boom"):
            updater.update(MagicMock(), ["EURUSD"])

        assert updater.last_update_monotonic is None

    @pytest.mark.parametrize(
        ("suppress_errors", "raises"),
        [
            (True, None),
            (False, Mt5RuntimeError),
        ],
        ids=["suppress", "propagate"],
    )
    def test_custom_backend_error_suppression(
        self,
        mocker: MockerFixture,
        suppress_errors: bool,
        raises: type[BaseException] | None,
    ) -> None:
        """suppress_errors controls whether recoverable backend errors propagate."""
        backend = mocker.Mock(side_effect=Mt5RuntimeError("boom"))
        updater = ThrottledHistoryUpdater(
            output="history.db",
            suppress_errors=suppress_errors,
            update_backend=backend,
        )
        if raises is None:
            assert updater.update(MagicMock(), ["EURUSD"]) is False
        else:
            with pytest.raises(raises, match="boom"):
                updater.update(MagicMock(), ["EURUSD"])
        assert updater.last_update_monotonic is None


class TestBuildConfigStringLogin:
    """Tests for build_config() string login coercion (issue #61)."""

    @pytest.mark.parametrize(
        ("login", "expected"),
        [
            (None, None),
            (12345, 12345),
            ("12345", 12345),
            (" 12345 ", 12345),
            ("", None),
            ("   ", None),
        ],
    )
    def test_coerces_login_from_string(
        self,
        login: int | str | None,
        expected: int | None,
    ) -> None:
        """Test build_config coerces string login to int or None."""
        config = build_config(login=login)
        assert config.login == expected

    def test_rejects_non_numeric_string_login(self) -> None:
        """Test build_config raises ValueError for non-numeric string login."""
        with pytest.raises(ValueError, match="invalid literal"):
            build_config(login="abc")

    def test_expands_dollar_brace_login_with_opt_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test build_config expands ${MT5_LOGIN} and coerces with opt-in."""
        monkeypatch.setenv("MT5_LOGIN", "12345")
        config = build_config(login="${MT5_LOGIN}", allow_whole_dollar_env=True)
        assert config.login == 12345

    def test_expands_whole_dollar_login_with_opt_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test build_config expands $MT5_LOGIN and coerces with opt-in."""
        monkeypatch.setenv("MT5_LOGIN", "99999")
        config = build_config(login="$MT5_LOGIN", allow_whole_dollar_env=True)
        assert config.login == 99999

    def test_missing_env_variable_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test build_config raises ValueError when referenced env var is not set."""
        monkeypatch.delenv("MT5_LOGIN", raising=False)
        with pytest.raises(ValueError, match="'MT5_LOGIN' is not set"):
            build_config(login="${MT5_LOGIN}", allow_whole_dollar_env=True)

    def test_env_expands_to_blank_becomes_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test build_config coerces blank env-expanded login to None."""
        monkeypatch.setenv("MT5_LOGIN", "")
        config = build_config(login="${MT5_LOGIN}", allow_whole_dollar_env=True)
        assert config.login is None

    def test_dollar_brace_login_not_expanded_without_opt_in(self) -> None:
        """Test ${MT5_LOGIN} is not expanded when allow_whole_dollar_env=False."""
        with pytest.raises(ValueError, match="invalid literal"):
            build_config(login="${MT5_LOGIN}")

    def test_integer_login_preserved_backward_compat(self) -> None:
        """Test existing int login callers remain backward-compatible."""
        config = build_config(login=54321)
        assert config.login == 54321

    def test_none_login_preserved_backward_compat(self) -> None:
        """Test existing None login callers remain backward-compatible."""
        config = build_config(login=None)
        assert config.login is None


class TestSubstituteMappingValues:
    """Tests for substitute_mapping_values() (issue #62)."""

    def test_substitutes_selected_keys_in_flat_dict(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test selected keys are substituted in a flat mapping."""
        monkeypatch.setenv("MT5_LOGIN", "12345")
        data: dict[str, object] = {
            "mt5_login": "${MT5_LOGIN}",
            "strategy_name": "${MT5_LOGIN}",
        }
        result = substitute_mapping_values(data, keys={"mt5_login"})
        assert result == {"mt5_login": "12345", "strategy_name": "${MT5_LOGIN}"}

    def test_preserves_non_selected_literal_dollar_signs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test literal dollar signs in non-selected fields are preserved exactly."""
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        data: dict[str, object] = {
            "mt5_password": "${MT5_PASSWORD}",
            "notes": "$NOT_EXPANDED",
        }
        result = substitute_mapping_values(data, keys={"mt5_password"})
        assert result == {"mt5_password": "secret", "notes": "$NOT_EXPANDED"}

    def test_nested_dict_traversal_substitutes_selected_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test selected keys inside nested dicts are substituted."""
        monkeypatch.setenv("MT5_SERVER", "Broker-Demo")
        data: dict[str, object] = {
            "outer": {
                "mt5_server": "${MT5_SERVER}",
                "other": "${MT5_SERVER}",
            }
        }
        result = substitute_mapping_values(data, keys={"mt5_server"})
        assert result == {
            "outer": {"mt5_server": "Broker-Demo", "other": "${MT5_SERVER}"}
        }

    def test_nested_list_traversal_substitutes_selected_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test selected keys inside list elements are substituted."""
        monkeypatch.setenv("MT5_LOGIN", "42")
        data: dict[str, object] = {
            "accounts": [
                {"mt5_login": "${MT5_LOGIN}", "name": "${MT5_LOGIN}"},
                {"mt5_login": "${MT5_LOGIN}", "name": "fixed"},
            ]
        }
        result = substitute_mapping_values(data, keys={"mt5_login"})
        assert result == {
            "accounts": [
                {"mt5_login": "42", "name": "${MT5_LOGIN}"},
                {"mt5_login": "42", "name": "fixed"},
            ]
        }

    def test_whole_dollar_expanded_with_opt_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test $ENV_NAME is expanded when allow_whole_dollar_env=True."""
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        data: dict[str, object] = {"mt5_password": "$MT5_PASSWORD"}
        result = substitute_mapping_values(
            data,
            keys={"mt5_password"},
            allow_whole_dollar_env=True,
        )
        assert result == {"mt5_password": "secret"}

    def test_whole_dollar_not_expanded_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test $ENV_NAME in a selected key is preserved when opt-in is False."""
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        data: dict[str, object] = {"mt5_password": "$MT5_PASSWORD"}
        result = substitute_mapping_values(data, keys={"mt5_password"})
        assert result == {"mt5_password": "$MT5_PASSWORD"}

    def test_blank_string_becomes_none_for_blank_keys(self) -> None:
        """Test blank strings are normalised to None for blank_string_keys_as_none."""
        data: dict[str, object] = {
            "mt5_login": "",
            "mt5_password": "  ",
            "other": "",
        }
        result = substitute_mapping_values(
            data,
            keys=set(),
            blank_string_keys_as_none={"mt5_login", "mt5_password"},
        )
        assert result == {"mt5_login": None, "mt5_password": None, "other": ""}

    def test_env_expanded_blank_becomes_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test env-expanded blank string is normalised to None."""
        monkeypatch.setenv("MT5_LOGIN", "")
        data: dict[str, object] = {"mt5_login": "${MT5_LOGIN}"}
        result = substitute_mapping_values(
            data,
            keys={"mt5_login"},
            blank_string_keys_as_none={"mt5_login"},
        )
        assert result == {"mt5_login": None}

    def test_missing_env_variable_raises_for_selected_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test missing env var for a selected key raises ValueError."""
        monkeypatch.delenv("MT5_MISSING", raising=False)
        data: dict[str, object] = {"mt5_login": "${MT5_MISSING}"}
        with pytest.raises(ValueError, match="'MT5_MISSING' is not set"):
            substitute_mapping_values(data, keys={"mt5_login"})

    def test_non_string_values_preserved(self) -> None:
        """Test non-string values under selected or non-selected keys are preserved."""
        data: dict[str, object] = {
            "mt5_login": 12345,
            "timeout": 5000,
            "enabled": True,
            "ratio": 1.5,
            "nothing": None,
        }
        result = substitute_mapping_values(
            data, keys={"mt5_login", "timeout", "enabled", "ratio", "nothing"}
        )
        assert result == data

    def test_caller_supplied_key_set_substitutes_correctly(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test helper works with any caller-supplied key set."""
        monkeypatch.setenv("APP_LOGIN", "77777")
        monkeypatch.setenv("APP_PASSWORD", "p4ss")
        data: dict[str, object] = {
            "app_login": "${APP_LOGIN}",
            "app_password": "${APP_PASSWORD}",
            "unrelated": "${APP_LOGIN}",
        }
        credential_keys = {"app_login", "app_password"}
        result = substitute_mapping_values(data, keys=credential_keys)
        assert result == {
            "app_login": "77777",
            "app_password": "p4ss",
            "unrelated": "${APP_LOGIN}",
        }

    def test_scalar_data_returned_unchanged(self) -> None:
        """Test a scalar (non-dict, non-list) value is returned as-is."""
        assert substitute_mapping_values("hello", keys={"x"}) == "hello"
        assert substitute_mapping_values(42, keys={"x"}) == 42
        assert substitute_mapping_values(None, keys={"x"}) is None

    def test_tuple_container_not_traversed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test tuple containers are returned as-is without traversal."""
        monkeypatch.setenv("MT5_LOGIN", "42")
        data: dict[str, object] = {"accounts": ({"mt5_login": "${MT5_LOGIN}"},)}
        result = substitute_mapping_values(data, keys={"mt5_login"})
        # tuple is returned as-is; inner dict is NOT visited
        assert result == {"accounts": ({"mt5_login": "${MT5_LOGIN}"},)}


class TestUpdateObservability:
    """Tests for update_observability and update_observability_with_config."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Mock client returning minimal valid frames."""
        client = MagicMock()
        client.account_info_as_df.return_value = pd.DataFrame([
            {
                "login": 12345,
                "currency": "USD",
                "balance": 10000.0,
                "equity": 10000.0,
                "margin": 0.0,
                "margin_free": 10000.0,
                "margin_level": 0.0,
                "profit": 0.0,
                "leverage": 100,
            }
        ])
        client.positions_get_as_df.return_value = pd.DataFrame()
        client.orders_get_as_df.return_value = pd.DataFrame()
        client.terminal_info_as_df.return_value = pd.DataFrame([
            {
                "name": "MetaTrader 5",
                "connected": 1,
                "community_account": 0,
                "trade_allowed": 1,
                "trade_expert": 1,
                "path": "/mt5",
                "company": "Broker",
                "language": "en",
            }
        ])
        return client

    def test_update_observability_creates_snapshot_tables(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Snapshot tables are created in the output database."""
        output = tmp_path / "obs.db"
        update_observability(client=mock_client, output=output)
        with sqlite3.connect(output) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "snapshot_runs" in tables
        assert "account_snapshots" in tables
        assert "position_snapshots" in tables

    def test_update_observability_records_ok_on_success(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """snapshot_runs records 'ok' status on a successful run."""
        output = tmp_path / "obs.db"
        update_observability(client=mock_client, output=output)
        with sqlite3.connect(output) as conn:
            row = conn.execute("SELECT status FROM snapshot_runs").fetchone()
        assert row == ("ok",)

    def test_update_observability_records_error_on_failure(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """snapshot_runs records 'error' and re-raises when a snapshot fails."""
        mock_client.account_info_as_df.side_effect = RuntimeError("boom")
        output = tmp_path / "obs.db"
        with pytest.raises(RuntimeError, match="boom"):
            update_observability(client=mock_client, output=output)
        with sqlite3.connect(output) as conn:
            row = conn.execute("SELECT status FROM snapshot_runs").fetchone()
        assert row == ("error",)

    def test_update_observability_skips_ensure_grafana_schema_when_disabled(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """with_grafana_schema=False does not call ensure_grafana_schema."""
        spy = mocker.spy(sdk, "ensure_grafana_schema")
        update_observability(
            client=mock_client,
            output=tmp_path / "obs.db",
            with_grafana_schema=False,
        )
        spy.assert_not_called()

    def test_update_observability_calls_ensure_grafana_schema_by_default(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """with_grafana_schema=True calls ensure_grafana_schema."""
        spy = mocker.spy(sdk, "ensure_grafana_schema")
        update_observability(
            client=mock_client,
            output=tmp_path / "obs.db",
            with_grafana_schema=True,
        )
        spy.assert_called_once()

    @pytest.mark.parametrize(
        ("kwarg", "method"),
        [
            ("include_account", "account_info_as_df"),
            ("include_positions", "positions_get_as_df"),
            ("include_orders", "orders_get_as_df"),
            ("include_terminal", "terminal_info_as_df"),
        ],
    )
    def test_update_observability_skips_when_disabled(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        kwarg: str,
        method: str,
    ) -> None:
        """include_X=False does not call the corresponding client method."""
        update_observability(
            client=mock_client,
            output=tmp_path / "obs.db",
            **{kwarg: False},  # type: ignore[arg-type]
        )
        getattr(mock_client, method).assert_not_called()

    @pytest.mark.parametrize(
        ("method", "table", "row", "expected_count"),
        [
            (
                "positions_get_as_df",
                "position_snapshots",
                {
                    "ticket": 1,
                    "position_id": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "price_open": 1.1,
                    "price_current": 1.1,
                    "profit": 0.0,
                    "swap": 0.0,
                    "comment": "",
                    "magic": 0,
                },
                1,
            ),
            (
                "orders_get_as_df",
                "order_snapshots",
                {
                    "ticket": 10,
                    "symbol": "EURUSD",
                    "type": 2,
                    "volume_current": 0.1,
                    "price_open": 1.2,
                    "price_current": 1.1,
                    "state": 1,
                    "comment": "",
                    "magic": 0,
                    "time_setup": 1700000000,
                },
                1,
            ),
        ],
        ids=["positions", "orders"],
    )
    def test_update_observability_writes_snapshot_rows(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        method: str,
        table: str,
        row: dict[str, object],
        expected_count: int,
    ) -> None:
        """Non-empty snapshots are written to the corresponding snapshot table."""
        getattr(mock_client, method).return_value = pd.DataFrame([row])
        output = tmp_path / "obs.db"
        update_observability(client=mock_client, output=output)
        with sqlite3.connect(output) as conn:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608
            ).fetchone()[0]
        assert count == expected_count

    @pytest.mark.parametrize(
        ("method", "table", "rows"),
        [
            (
                "positions_get_as_df",
                "position_snapshots",
                [
                    {"ticket": 1, "symbol": "EURUSD", "volume": 0.1, "profit": 0.0},
                    {"ticket": 2, "symbol": "USDJPY", "volume": 0.2, "profit": 0.0},
                ],
            ),
            (
                "orders_get_as_df",
                "order_snapshots",
                [
                    {"ticket": 10, "symbol": "EURUSD", "volume_current": 0.1},
                    {"ticket": 11, "symbol": "USDJPY", "volume_current": 0.5},
                ],
            ),
        ],
        ids=["positions", "orders"],
    )
    def test_update_observability_symbol_filter(
        self,
        tmp_path: Path,
        method: str,
        table: str,
        rows: list[dict[str, object]],
    ) -> None:
        """Symbol filter fetches all rows in one call and filters client-side."""
        client = MagicMock()
        client.account_info_as_df.return_value = pd.DataFrame([{"login": 1}])
        client.positions_get_as_df.return_value = pd.DataFrame()
        client.orders_get_as_df.return_value = pd.DataFrame()
        client.terminal_info_as_df.return_value = pd.DataFrame()
        getattr(client, method).return_value = pd.DataFrame(rows)
        output = tmp_path / "obs.db"
        update_observability(client=client, output=output, symbols=["EURUSD", "GBPUSD"])
        assert getattr(client, method).call_count == 1
        with sqlite3.connect(output) as conn:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608
            ).fetchone()[0]
        assert count == 1

    def test_update_observability_symbol_filter_no_symbol_col(
        self,
        tmp_path: Path,
    ) -> None:
        """Symbol filter is skipped when positions df has no symbol column."""
        client = MagicMock()
        client.account_info_as_df.return_value = pd.DataFrame([{"login": 1}])
        # No symbol column in positions — all rows pass through unfiltered
        client.positions_get_as_df.return_value = pd.DataFrame([
            {"ticket": 1, "volume": 0.1},
        ])
        client.orders_get_as_df.return_value = pd.DataFrame()
        client.terminal_info_as_df.return_value = pd.DataFrame()
        output = tmp_path / "obs.db"
        update_observability(client=client, output=output, symbols=["EURUSD"])
        with sqlite3.connect(output) as conn:
            count = conn.execute("SELECT COUNT(*) FROM position_snapshots").fetchone()[
                0
            ]
        assert count == 1

    def test_update_observability_account_none_login(
        self,
        tmp_path: Path,
    ) -> None:
        """Account row with no login key returns None login for downstream helpers."""
        client = MagicMock()
        client.account_info_as_df.return_value = pd.DataFrame([{"balance": 10000.0}])
        client.positions_get_as_df.return_value = pd.DataFrame()
        client.orders_get_as_df.return_value = pd.DataFrame()
        client.terminal_info_as_df.return_value = pd.DataFrame()
        output = tmp_path / "obs.db"
        update_observability(client=client, output=output)
        with sqlite3.connect(output) as conn:
            row = conn.execute("SELECT login FROM account_snapshots").fetchone()
        assert row is not None
        assert row[0] is None

    @pytest.mark.parametrize(
        ("method", "table", "message"),
        [
            (
                "account_info_as_df",
                "account_snapshots",
                "account_info_as_df returned empty frame",
            ),
            (
                "terminal_info_as_df",
                "terminal_snapshots",
                "terminal_info_as_df returned empty frame",
            ),
        ],
        ids=["account", "terminal"],
    )
    def test_update_observability_empty_logs_warning(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        method: str,
        table: str,
        message: str,
    ) -> None:
        """Empty snapshot frames log a warning and write no rows."""
        getattr(mock_client, method).return_value = pd.DataFrame()
        with caplog.at_level(logging.WARNING, logger="mt5cli.sdk"):
            update_observability(client=mock_client, output=tmp_path / "obs.db")
        assert message in caplog.text
        with sqlite3.connect(tmp_path / "obs.db") as conn:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608
            ).fetchone()[0]
        assert count == 0

    def test_update_observability_with_config_opens_and_closes_connection(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """update_observability_with_config manages the MT5 connection lifecycle."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.return_value = pd.DataFrame()
        mock_client.positions_get_as_df.return_value = pd.DataFrame()
        mock_client.orders_get_as_df.return_value = pd.DataFrame()
        mock_client.terminal_info_as_df.return_value = pd.DataFrame()
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        update_observability_with_config(output=tmp_path / "obs.db")
        mock_client.initialize_and_login_mt5.assert_called_once()
        mock_client.shutdown.assert_called_once()

    def test_update_observability_with_config_passes_symbols(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """update_observability_with_config forwards symbols to update_observability."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.return_value = pd.DataFrame()
        mock_client.positions_get_as_df.return_value = pd.DataFrame()
        mock_client.orders_get_as_df.return_value = pd.DataFrame()
        mock_client.terminal_info_as_df.return_value = pd.DataFrame()
        mocker.patch("mt5cli.sdk.Mt5DataClient", return_value=mock_client)
        spy = mocker.patch("mt5cli.sdk.update_observability")
        update_observability_with_config(
            output=tmp_path / "obs.db",
            symbols=["EURUSD"],
            include_account=False,
        )
        spy.assert_called_once()
        call_kwargs = spy.call_args.kwargs
        assert call_kwargs["symbols"] == ["EURUSD"]
        assert call_kwargs["include_account"] is False

    def test_update_observability_invokes_snapshot_telemetry(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """update_observability calls record_snapshot_update on the global metrics."""
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_snapshot_update.return_value = mock_cm
        mocker.patch("mt5cli.sdk.get_metrics", return_value=mock_metrics)
        update_observability(client=mock_client, output=tmp_path / "obs.db")
        mock_metrics.record_snapshot_update.assert_called_once()

    def test_update_observability_emits_account_metrics(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """_snapshot_account emits account gauges via get_metrics."""
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_snapshot_update.return_value = mock_cm
        mocker.patch("mt5cli.sdk.get_metrics", return_value=mock_metrics)
        update_observability(client=mock_client, output=tmp_path / "obs.db")
        mock_metrics.record_account_state.assert_called_once()

    def test_update_observability_emits_terminal_metrics(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """_snapshot_terminal emits connected/trade gauges via get_metrics."""
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_snapshot_update.return_value = mock_cm
        mocker.patch("mt5cli.sdk.get_metrics", return_value=mock_metrics)
        update_observability(client=mock_client, output=tmp_path / "obs.db")
        mock_metrics.record_terminal_state.assert_called_once_with(
            connected=1.0, trade_allowed=1.0, trade_expert=1.0
        )

    def test_update_observability_aggregates_same_symbol_positions(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Same-symbol positions are summed before emitting gauges (hedging)."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.return_value = pd.DataFrame([
            {
                "login": 1,
                "server": "demo",
                "balance": 1000.0,
                "equity": 1000.0,
                "margin": 0.0,
                "margin_free": 1000.0,
                "margin_level": 0.0,
            }
        ])
        mock_client.positions_get_as_df.return_value = pd.DataFrame([
            {"ticket": 1, "symbol": "EURUSD", "profit": 10.0, "volume": 0.1},
            {"ticket": 2, "symbol": "EURUSD", "profit": -5.0, "volume": 0.2},
            {"ticket": 3, "symbol": "GBPUSD", "profit": 3.0, "volume": 0.05},
        ])
        mock_client.orders_get_as_df.return_value = pd.DataFrame()
        mock_client.terminal_info_as_df.return_value = pd.DataFrame()
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_snapshot_update.return_value = mock_cm
        mocker.patch("mt5cli.sdk.get_metrics", return_value=mock_metrics)
        update_observability(client=mock_client, output=tmp_path / "obs.db")
        calls = mock_metrics.record_position_state.call_args_list
        # Two EURUSD positions should be collapsed to one call; GBPUSD is one call.
        assert len(calls) == 2
        by_symbol = {c.kwargs["symbol"]: c.kwargs for c in calls}
        assert abs(float(by_symbol["EURUSD"]["profit"]) - 5.0) < 1e-9
        assert abs(float(by_symbol["EURUSD"]["volume"]) - 0.3) < 1e-9
        assert abs(float(by_symbol["GBPUSD"]["profit"]) - 3.0) < 1e-9
        assert abs(float(by_symbol["GBPUSD"]["volume"]) - 0.05) < 1e-9


class TestUpdateHistoryTelemetry:
    """Tests for telemetry hooks in update_history."""

    def test_update_history_invokes_history_telemetry(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """update_history wraps write_incremental_datasets with telemetry."""
        mock_client = MagicMock()
        mock_client.copy_rates_range_as_df.return_value = pd.DataFrame()
        mock_client.history_orders_get_as_df.return_value = pd.DataFrame()
        mock_client.history_deals_get_as_df.return_value = pd.DataFrame()
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_history_update.return_value = mock_cm
        mocker.patch("mt5cli.sdk.get_metrics", return_value=mock_metrics)
        update_history(
            client=mock_client,
            output=tmp_path / "hist.db",
            symbols=["EURUSD"],
        )
        mock_metrics.record_history_update.assert_called_once_with(dataset="history")

    def test_update_history_emits_history_rows(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """update_history calls add_history_rows with the SQLite change delta."""
        mock_client = MagicMock()
        mock_client.copy_rates_range_as_df.return_value = pd.DataFrame()
        mock_client.history_orders_get_as_df.return_value = pd.DataFrame()
        mock_client.history_deals_get_as_df.return_value = pd.DataFrame()
        mock_metrics = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_metrics.record_history_update.return_value = mock_cm
        mocker.patch("mt5cli.sdk.get_metrics", return_value=mock_metrics)
        update_history(
            client=mock_client,
            output=tmp_path / "hist.db",
            symbols=["EURUSD"],
        )
        mock_metrics.add_history_rows.assert_called_once_with(0, dataset="history")
