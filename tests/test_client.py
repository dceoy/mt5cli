"""Tests for mt5cli.client module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple, cast
from unittest.mock import MagicMock, call

import pandas as pd
import pytest
from pdmt5 import Mt5RuntimeError
from pydantic import SecretStr

from mt5cli.client import (
    MT5Client,
    _connected_client,  # pyright: ignore[reportPrivateUsage]
    _run_with_client,  # pyright: ignore[reportPrivateUsage]
    build_config,
    mt5_session,
    substitute_env_placeholders,
    substitute_mapping_values,
)
from mt5cli.exceptions import Mt5ConnectionError
from mt5cli.marketdata import (
    collect_latest_rates,
    copy_rates_range,
    recent_history_deals,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from pdmt5 import Mt5DataClient
    from pytest_mock import MockerFixture


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


def _mt5_client_with_injected_client(connected: MagicMock) -> MT5Client:
    return MT5Client.from_connected_client(connected)


def test_order_check_and_order_send_route_through_connected_client(
    mock_client: MagicMock,
) -> None:
    """order_check/order_send fetch through the same connected client path."""
    request = {"action": 1}
    client = MT5Client()
    client.order_check(request)
    client.order_send(request)
    mock_client.order_check_as_df.assert_called_once_with(request=request)
    mock_client.order_send_as_df.assert_called_once_with(request=request)


def test_from_connected_client_binds_without_owning(mock_client: MagicMock) -> None:
    """from_connected_client wraps an injected client without shutting it down."""
    client = MT5Client.from_connected_client(mock_client)
    with client:
        client.order_check({"action": 1})
    mock_client.shutdown.assert_not_called()


def test_mt5_session_yields_connected_mt5_client(mocker: MockerFixture) -> None:
    """Public mt5_session yields an MT5Client bound to a connected session."""
    connected = mocker.MagicMock()
    context = mocker.MagicMock()
    context.__enter__.return_value = connected
    context.__exit__.return_value = False
    mocker.patch("mt5cli.client._connected_client", return_value=context)
    with mt5_session(build_config()) as client:
        assert isinstance(client, MT5Client)


def test_owned_mt5_session_initializes_and_shuts_down_once(
    mocker: MockerFixture,
) -> None:
    """An owned session has exactly one initialization and shutdown boundary."""
    raw_client = MagicMock()
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)

    with mt5_session(build_config()) as client:
        assert isinstance(client, MT5Client)
        raw_client.initialize_and_login_mt5.assert_called_once()

    raw_client.shutdown.assert_called_once()


def test_owned_mt5_session_shuts_down_when_body_raises(
    mocker: MockerFixture,
) -> None:
    """Owned sessions release the terminal while preserving body exceptions."""
    raw_client = MagicMock()
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)

    message = "body failure"
    with pytest.raises(RuntimeError, match=message), mt5_session(build_config()):
        raise RuntimeError(message)

    raw_client.initialize_and_login_mt5.assert_called_once()
    raw_client.shutdown.assert_called_once()


def test_session_body_exceptions_are_not_normalized(mocker: MockerFixture) -> None:
    """Exceptions raised inside the session body pass through unnormalized."""
    raw_client = MagicMock()
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)

    message = "body-level failure"
    with (
        pytest.raises(Mt5RuntimeError, match=message),
        mt5_session(build_config()),
    ):
        raise Mt5RuntimeError(message)

    raw_client.shutdown.assert_called_once()


def test_mt5_session_initialization_failure_is_normalized(
    mocker: MockerFixture,
) -> None:
    """mt5_session normalizes Mt5RuntimeError raised during initialization."""
    raw_client = MagicMock()
    raw_client.initialize_and_login_mt5.side_effect = Mt5RuntimeError("init failed")
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)

    with pytest.raises(Mt5ConnectionError, match="init failed"), mt5_session():
        pass

    raw_client.shutdown.assert_called_once()


def test_mt5_client_enter_initialization_failure_is_normalized(
    mocker: MockerFixture,
) -> None:
    """MT5Client.__enter__ follows the same init-error contract as mt5_session."""
    raw_client = MagicMock()
    raw_client.initialize_and_login_mt5.side_effect = Mt5RuntimeError("boom")
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)

    client = MT5Client()
    with pytest.raises(Mt5ConnectionError, match="boom"), client:
        pass

    raw_client.shutdown.assert_called_once()


def test_transient_client_operation_normalizes_mt5_runtime_error(
    mocker: MockerFixture,
) -> None:
    """A transient (unbound) MT5Client normalizes Mt5RuntimeError from an op."""
    raw_client = MagicMock()
    raw_client.account_info_as_df.side_effect = Mt5RuntimeError("op failed")
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)

    client = MT5Client()
    with pytest.raises(Mt5ConnectionError, match="op failed"):
        client.account_info()

    raw_client.shutdown.assert_called_once()


def test_persistent_client_operation_normalizes_mt5_runtime_error(
    mock_client: MagicMock,
) -> None:
    """A persistent (entered) MT5Client normalizes Mt5RuntimeError from an op."""
    mock_client.account_info_as_df.side_effect = Mt5RuntimeError("op failed")

    with (
        MT5Client() as client,
        pytest.raises(Mt5ConnectionError, match="op failed"),
    ):
        client.account_info()


def test_caller_owned_client_operation_normalizes_mt5_runtime_error(
    mock_client: MagicMock,
) -> None:
    """A caller-owned (from_connected_client) client normalizes Mt5RuntimeError."""
    mock_client.account_info_as_df.side_effect = Mt5RuntimeError("op failed")

    client = MT5Client.from_connected_client(mock_client)
    with pytest.raises(Mt5ConnectionError, match="op failed"):
        client.account_info()
    mock_client.shutdown.assert_not_called()


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_persistent_client_propagates_ordinary_application_errors(
    mock_client: MagicMock,
    error_type: type[Exception],
) -> None:
    """Ordinary application exceptions are not normalized on the persistent path."""
    mock_client.account_info_as_df.side_effect = error_type("application error")

    with MT5Client() as client, pytest.raises(error_type, match="application error"):
        client.account_info()


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_transient_client_propagates_ordinary_application_errors(
    mocker: MockerFixture,
    error_type: type[Exception],
) -> None:
    """Ordinary application exceptions are not normalized on the transient path."""
    raw_client = MagicMock()
    raw_client.account_info_as_df.side_effect = error_type("application error")
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)

    client = MT5Client()
    with pytest.raises(error_type, match="application error"):
        client.account_info()

    raw_client.shutdown.assert_called_once()


@pytest.mark.parametrize(
    "make_session",
    [mt5_session, MT5Client],
    ids=["mt5_session", "MT5Client"],
)
def test_cleanup_only_shutdown_failure_is_normalized(
    mocker: MockerFixture,
    make_session: Callable[[], AbstractContextManager[object]],
) -> None:
    """A shutdown failure after a successful body raises the stable error."""
    raw_client = MagicMock()
    raw_client.shutdown.side_effect = Mt5RuntimeError("shutdown failed")
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)

    with pytest.raises(Mt5ConnectionError, match="shutdown failed"), make_session():
        pass


@pytest.mark.parametrize(
    "make_session",
    [mt5_session, MT5Client],
    ids=["mt5_session", "MT5Client"],
)
def test_body_exception_survives_shutdown_failure(
    mocker: MockerFixture,
    make_session: Callable[[], AbstractContextManager[object]],
) -> None:
    """A session-body exception passes through even when shutdown also fails."""
    raw_client = MagicMock()
    raw_client.shutdown.side_effect = Mt5RuntimeError("shutdown failed")
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)

    message = "body failure"
    with pytest.raises(RuntimeError, match=message), make_session():
        raise RuntimeError(message)

    raw_client.shutdown.assert_called_once()


@pytest.mark.parametrize(
    "make_session",
    [mt5_session, MT5Client],
    ids=["mt5_session", "MT5Client"],
)
def test_init_failure_survives_shutdown_failure(
    mocker: MockerFixture,
    make_session: Callable[[], AbstractContextManager[object]],
) -> None:
    """The normalized init error stays primary when shutdown also fails."""
    raw_client = MagicMock()
    raw_client.initialize_and_login_mt5.side_effect = Mt5RuntimeError("init failed")
    raw_client.shutdown.side_effect = Mt5RuntimeError("shutdown failed")
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=raw_client)

    with pytest.raises(Mt5ConnectionError, match="init failed"), make_session():
        pass


def test_public_facade_symbol_snapshot(mock_client: MagicMock) -> None:
    """Facade symbol snapshot normalizes a one-row DataFrame to a plain mapping."""
    mock_client.symbol_info_as_df.return_value = pd.DataFrame([{"symbol": "EURUSD"}])
    client = MT5Client.from_connected_client(mock_client)
    assert client.symbol_info_as_dict("EURUSD") == {"symbol": "EURUSD"}


def test_mt5_session_yields_injected_client_without_reconnect(
    mock_client: MagicMock,
) -> None:
    """Caller-owned clients are yielded as-is without initialize/shutdown."""
    injected = MT5Client.from_connected_client(mock_client)
    with mt5_session(client=injected) as client:
        assert client is injected
    mock_client.shutdown.assert_not_called()


def test_mt5_client_operational_helpers_route_through_connected_client(
    mock_client: MagicMock,
) -> None:
    """Operational helpers on MT5Client delegate to the bound pdmt5 client."""
    mock_client.mt5 = MagicMock()
    mock_client.account_info_as_df.return_value = pd.DataFrame([{"login": 1}])
    mock_client.positions_get_as_df.return_value = pd.DataFrame([{"ticket": 1}])
    mock_client.order_calc_margin.return_value = 12.5
    mock_client.symbol_select.return_value = True

    client = MT5Client.from_connected_client(mock_client)

    assert client.mt5 is mock_client.mt5
    assert client.account_info_as_dict() == {"login": 1}
    assert len(client.positions_get_as_df("EURUSD")) == 1
    mock_client.positions_get_as_df.assert_called_once_with(
        symbol="EURUSD",
        group=None,
        ticket=None,
    )
    margin = client.order_calc_margin(0, "EURUSD", 0.1, 1.1)
    mock_client.order_calc_margin.assert_called_once_with(0, "EURUSD", 0.1, 1.1)
    assert margin is mock_client.order_calc_margin.return_value
    assert client.symbol_select("EURUSD", enable=False) is True


def test_account_info_as_dict_returns_empty_mapping_for_empty_frame(
    mock_client: MagicMock,
) -> None:
    """Empty account snapshots normalize to an empty mapping."""
    mock_client.account_info_as_df.return_value = pd.DataFrame()
    client = MT5Client.from_connected_client(mock_client)
    assert client.account_info_as_dict() == {}


class TestConnectionLifecycle:
    """Tests for MT5 connection lifecycle helpers."""

    def test_connected_client_shuts_down(self, mocker: MockerFixture) -> None:
        """Test that _connected_client always shuts down."""
        mock_client = MagicMock()
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=mock_client)
        config = MagicMock()
        with _connected_client(config):
            mock_client.initialize_and_login_mt5.assert_called_once()
        mock_client.shutdown.assert_called_once()

    def test_connected_client_shutdown_on_init_failure(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that shutdown is called when initialize/login fails."""
        mock_client = MagicMock()
        mock_client.initialize_and_login_mt5.side_effect = Mt5RuntimeError(
            "login failed",
        )
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=mock_client)
        with (
            pytest.raises(Mt5ConnectionError, match="login failed"),
            _connected_client(MagicMock()),
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
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=mock_client)
        with pytest.raises(RuntimeError, match="boom"):
            _run_with_client(
                MagicMock(),
                lambda c: c.account_info_as_df(),
            )
        mock_client.shutdown.assert_called_once()

    def test_connected_client_omits_retry_count_by_default(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that retry_count is left to the pdmt5 default when omitted."""
        mock_client = MagicMock()
        mt5_data_client = mocker.patch(
            "mt5cli.client.Mt5DataClient",
            return_value=mock_client,
        )
        config = MagicMock()
        with _connected_client(config):
            pass
        mt5_data_client.assert_called_once_with(config=config)

    def test_connected_client_forwards_retry_count(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that an explicit retry_count is forwarded to Mt5DataClient."""
        mock_client = MagicMock()
        mt5_data_client = mocker.patch(
            "mt5cli.client.Mt5DataClient",
            return_value=mock_client,
        )
        config = MagicMock()
        with _connected_client(config, retry_count=7):
            pass
        mt5_data_client.assert_called_once_with(config=config, retry_count=7)

    def test_client_context_manager_reuses_connection(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that context-managed client reuses one connection."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.return_value = pd.DataFrame({"a": [1]})
        mock_client.terminal_info_as_df.return_value = pd.DataFrame({"b": [2]})
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=mock_client)
        with MT5Client() as client:
            client.account_info()
            client.terminal_info()
            assert client.config is not None
        mock_client.initialize_and_login_mt5.assert_called_once()
        mock_client.shutdown.assert_called_once()
        assert mock_client.account_info_as_df.call_count == 1
        assert mock_client.terminal_info_as_df.call_count == 1

    def test_context_manager_forwards_configured_retry_count(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test __enter__ constructs Mt5DataClient with the configured retry."""
        mock_client = MagicMock()
        mt5_data_client = mocker.patch(
            "mt5cli.client.Mt5DataClient",
            return_value=mock_client,
        )
        client = MT5Client(retry_count=7)
        with client:
            pass
        mt5_data_client.assert_called_once_with(config=client.config, retry_count=7)

    def test_one_shot_call_forwards_configured_retry_count(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test a one-shot call without a context manager forwards retry_count."""
        mock_client = MagicMock()
        mock_client.account_info_as_df.return_value = pd.DataFrame({"a": [1]})
        mt5_data_client = mocker.patch(
            "mt5cli.client.Mt5DataClient",
            return_value=mock_client,
        )
        client = MT5Client(retry_count=7)
        client.account_info()
        mt5_data_client.assert_called_once_with(config=client.config, retry_count=7)
        mock_client.initialize_and_login_mt5.assert_called_once()
        mock_client.shutdown.assert_called_once()

    def test_client_context_manager_shutdown_on_init_failure(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that shutdown is called when context manager login fails."""
        mock_client = MagicMock()
        mock_client.initialize_and_login_mt5.side_effect = RuntimeError(
            "login failed",
        )
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=mock_client)
        client = MT5Client()
        with pytest.raises(RuntimeError, match="login failed"), client:
            pass
        mock_client.shutdown.assert_called_once()
        assert client._client is None  # pyright: ignore[reportPrivateUsage]

    def test_exit_without_enter_is_noop(self) -> None:
        """Test that __exit__ without __enter__ does not fail."""
        client = MT5Client()
        client.__exit__(None, None, None)

    @pytest.mark.parametrize(
        "make_client",
        [
            pytest.param(
                MT5Client.from_connected_client,
                id="from-connected-client-classmethod",
            ),
            pytest.param(
                _mt5_client_with_injected_client,
                id="constructor-client-kwarg",
            ),
        ],
    )
    def test_injected_client_is_reused_and_not_shutdown(
        self,
        make_client: Callable[[MagicMock], MT5Client],
    ) -> None:
        """Test injected connected clients are not initialized or shut down.

        Both the from_connected_client classmethod and the constructor's
        client kwarg must produce the same non-owning lifecycle: no login or
        shutdown of the injected client, and continued usability after the
        context manager exits.
        """
        connected = MagicMock()
        connected.account_info_as_df.return_value = pd.DataFrame({"a": [1]})
        connected.terminal_info_as_df.return_value = pd.DataFrame({"b": [2]})
        with make_client(connected) as client:
            result = client.account_info()
        assert result.to_dict("list") == {"a": [1]}
        connected.initialize_and_login_mt5.assert_not_called()
        connected.shutdown.assert_not_called()
        connected.account_info_as_df.assert_called_once()
        after_exit = client.terminal_info()
        assert after_exit.to_dict("list") == {"b": [2]}
        connected.terminal_info_as_df.assert_called_once()


class TestMT5ClientMethods:
    """Tests for MT5Client SDK methods."""

    @pytest.mark.parametrize(
        ("call", "expected_method", "expected_kwargs"),
        [
            pytest.param(
                lambda: MT5Client().copy_rates_range(
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
                lambda: MT5Client().copy_ticks_from(
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
                lambda: MT5Client().history_orders(
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
                lambda: MT5Client().latest_rates("EURUSD", "M1", 5, start_pos=2),
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
        """MT5Client methods normalize inputs and forward them on."""
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
            MT5Client().latest_rates("EURUSD", "M1", 0)

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
            "mt5cli.client.Mt5DataClient",
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
            MT5Client().collect_latest_rates(symbols, timeframes, count=1)

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
            MT5Client().recent_history_deals(0)

    @pytest.mark.parametrize(
        ("terminal_info_value", "account_info_value", "expected"),
        [
            (
                {"connected": True},
                {"login": 123},
                {
                    "version": [5, 0, 1],
                    "terminal_info": {"connected": True},
                    "account_info": {"login": 123},
                    "symbols_total": 42,
                },
            ),
            (
                _TerminalInfo(
                    connected=True,
                    path="terminal.exe",
                ),
                _AccountInfo(
                    login=123,
                    limits={"modes": ("netting", "hedging"), "servers": ["demo"]},
                ),
                {
                    "version": [5, 0, 1],
                    "terminal_info": {"connected": True, "path": "terminal.exe"},
                    "account_info": {
                        "login": 123,
                        "limits": {
                            "modes": ["netting", "hedging"],
                            "servers": ["demo"],
                        },
                    },
                    "symbols_total": 42,
                },
            ),
        ],
        ids=["raw-mappings", "namedtuple-normalization"],
    )
    def test_mt5_summary_success_cases(
        self,
        mock_client: MagicMock,
        terminal_info_value: object,
        account_info_value: object,
        expected: dict[str, object],
    ) -> None:
        """Test mt5_summary returns normalized plain-Python status mappings."""
        mock_client.version.return_value = (5, 0, 1)
        mock_client.terminal_info.return_value = terminal_info_value
        mock_client.account_info.return_value = account_info_value
        mock_client.symbols_total.return_value = 42

        from mt5cli.marketdata import mt5_summary  # noqa: PLC0415

        assert mt5_summary() == expected

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

        from mt5cli.marketdata import mt5_summary_as_df  # noqa: PLC0415

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
        client = MT5Client.from_connected_client(cast("Mt5DataClient", client_cls()))

        with pytest.raises(exc, match=match):
            client.mt5_summary()


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
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=client)
        from mt5cli.marketdata import recent_ticks  # noqa: PLC0415

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
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=client)
        result = MT5Client().recent_ticks("EURUSD", 30, count=2, flags="ALL")
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
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=client)
        with pytest.raises(TypeError, match="Unsupported tick time value"):
            MT5Client().recent_ticks("EURUSD", 30)

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
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=client)
        MT5Client().recent_ticks("EURUSD", 30)
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
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=client)
        from mt5cli.marketdata import recent_ticks  # noqa: PLC0415

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
        mocker.patch("mt5cli.client.Mt5DataClient", return_value=client)
        from mt5cli.marketdata import minimum_margins  # noqa: PLC0415

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

        actual = getattr(config, field)
        if isinstance(actual, SecretStr):
            actual = actual.get_secret_value()
        assert actual == env_value

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


class TestBuildConfigStringLogin:
    """Tests for build_config() string login coercion (issue #61)."""

    @pytest.mark.parametrize(
        ("login", "expected"),
        [
            pytest.param(None, None, id="none-passthrough"),
            pytest.param(12345, 12345, id="int-passthrough"),
            pytest.param(54321, 54321, id="int-login-backward-compat"),
            pytest.param("12345", 12345, id="numeric-string-coerced"),
            pytest.param(" 12345 ", 12345, id="whitespace-padded-string-coerced"),
            pytest.param("", None, id="empty-string-becomes-none"),
            pytest.param("   ", None, id="whitespace-only-string-becomes-none"),
        ],
    )
    def test_coerces_login_from_string(
        self,
        login: int | str | None,
        expected: int | None,
    ) -> None:
        """Test build_config coerces string login to int/None.

        Int and None logins are left unchanged.
        """
        config = build_config(login=login)
        assert config.login == expected

    def test_rejects_non_numeric_string_login(self) -> None:
        """Test build_config raises ValueError for non-numeric string login."""
        with pytest.raises(ValueError, match="invalid literal"):
            build_config(login="abc")

    @pytest.mark.parametrize(
        ("login_template", "env_value", "expected"),
        [
            pytest.param("${MT5_LOGIN}", "12345", 12345, id="dollar-brace-expands"),
            pytest.param("$MT5_LOGIN", "99999", 99999, id="whole-dollar-expands"),
            pytest.param("${MT5_LOGIN}", "", None, id="blank-env-becomes-none"),
        ],
    )
    def test_expands_login_env_placeholder_with_opt_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
        login_template: str,
        env_value: str,
        expected: int | None,
    ) -> None:
        """Test build_config expands env-placeholder logins with opt-in.

        Both ``${VAR}`` and whole-``$VAR`` syntax are expanded and coerced
        when allow_whole_dollar_env=True; a blank expansion coerces to None.
        """
        monkeypatch.setenv("MT5_LOGIN", env_value)
        config = build_config(login=login_template, allow_whole_dollar_env=True)
        assert config.login == expected

    def test_missing_env_variable_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test build_config raises ValueError when referenced env var is not set."""
        monkeypatch.delenv("MT5_LOGIN", raising=False)
        with pytest.raises(ValueError, match="'MT5_LOGIN' is not set"):
            build_config(login="${MT5_LOGIN}", allow_whole_dollar_env=True)

    def test_dollar_brace_login_not_expanded_without_opt_in(self) -> None:
        """Test ${MT5_LOGIN} is not expanded when allow_whole_dollar_env=False."""
        with pytest.raises(ValueError, match="invalid literal"):
            build_config(login="${MT5_LOGIN}")


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

    @pytest.mark.parametrize(
        ("allow_whole_dollar_env", "expected"),
        [
            pytest.param(None, "$MT5_PASSWORD", id="default-preserves-whole-dollar"),
            pytest.param(True, "secret", id="opt-in-expands-whole-dollar"),
        ],
    )
    def test_whole_dollar_env_handling(
        self,
        monkeypatch: pytest.MonkeyPatch,
        allow_whole_dollar_env: bool | None,
        expected: str,
    ) -> None:
        """Test $ENV_NAME handling for selected mapping keys."""
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        data: dict[str, object] = {"mt5_password": "$MT5_PASSWORD"}
        if allow_whole_dollar_env is None:
            result = substitute_mapping_values(data, keys={"mt5_password"})
        else:
            result = substitute_mapping_values(
                data,
                keys={"mt5_password"},
                allow_whole_dollar_env=allow_whole_dollar_env,
            )
        assert result == {"mt5_password": expected}

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
