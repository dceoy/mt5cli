"""Tests for mt5cli.client module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd

from mt5cli.client import MT5Client, mt5_session
from mt5cli.sdk import build_config

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


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
    mocker.patch("mt5cli.client.connected_client", return_value=context)
    with mt5_session(build_config()) as client:
        assert isinstance(client, MT5Client)


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
