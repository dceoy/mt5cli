"""Tests for mt5cli.client module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mt5cli.client import MT5Client, mt5_session
from mt5cli.sdk import build_config

if TYPE_CHECKING:
    from unittest.mock import MagicMock

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
