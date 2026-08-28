"""Tests for persistent MT5 session lifecycle helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pdmt5 import Mt5RuntimeError

from mt5cli import MT5Client, Mt5ConnectionError, switch_account


def _connected_client(raw_client: MagicMock) -> MT5Client:
    """Return an MT5Client facade bound to a caller-owned active client."""
    return MT5Client.from_connected_client(raw_client)


@pytest.mark.parametrize(
    ("active_login", "active_server", "requested_server", "expect_login"),
    [
        pytest.param(222, "Demo", "Demo", False, id="same-login-same-server"),
        pytest.param(222, "Demo", None, False, id="same-login-no-server-constraint"),
        pytest.param(111, "Demo", "Demo", True, id="different-login"),
        pytest.param(222, "Other", "Demo", True, id="same-login-different-server"),
    ],
)
def test_switch_account_pre_switch_match_decision(
    active_login: int,
    active_server: str | None,
    requested_server: str | None,
    expect_login: bool,
) -> None:
    """The pre-switch account-match check decides whether login() runs."""
    raw_client = MagicMock()
    active_account = SimpleNamespace(login=active_login, server=active_server)
    raw_client.account_info.side_effect = [
        active_account,
        SimpleNamespace(login=222, server=requested_server or active_server),
    ]
    raw_client.login.return_value = True
    client = _connected_client(raw_client)

    switch_account(client, login=222, server=requested_server)

    assert raw_client.login.called is expect_login
    raw_client.shutdown.assert_not_called()


def test_switch_account_uses_login_without_reinitializing_terminal() -> None:
    """Changing accounts calls login only and verifies the resulting account."""
    raw_client = MagicMock()
    raw_client.account_info.side_effect = [
        SimpleNamespace(login=111, server="Demo"),
        SimpleNamespace(login=222, server="Demo"),
    ]
    raw_client.login.return_value = True
    client = _connected_client(raw_client)

    switch_account(
        client,
        login=222,
        password="secret",
        server="Demo",
        timeout=60_000,
    )

    raw_client.login.assert_called_once_with(
        login=222,
        password="secret",
        server="Demo",
        timeout=60_000,
    )
    raw_client.initialize_and_login_mt5.assert_not_called()
    raw_client.shutdown.assert_not_called()


def test_switch_account_requires_active_persistent_session() -> None:
    """Transient clients cannot switch an account without opening a session."""
    client = MT5Client()

    with pytest.raises(Mt5ConnectionError, match="active persistent session"):
        switch_account(client, login=222)


def test_switch_account_normalizes_login_failure() -> None:
    """A failed MT5 login becomes the stable connection exception."""
    raw_client = MagicMock()
    raw_client.account_info.return_value = SimpleNamespace(login=111, server="Demo")
    raw_client.login.return_value = False
    raw_client.last_error.return_value = (-6, "authorization failed")
    client = _connected_client(raw_client)

    with pytest.raises(Mt5ConnectionError, match="authorization failed"):
        switch_account(client, login=222, server="Demo")


def test_switch_account_normalizes_runtime_error() -> None:
    """Underlying pdmt5 runtime errors use the stable connection exception.

    This also covers the no-active-account case: pdmt5's
    ``Mt5DataClient.account_info()`` raises ``Mt5RuntimeError`` rather than
    returning ``None`` when MT5 has no active account, so ``switch_account``
    requires an already-active account and surfaces that failure the same way
    as any other runtime error.
    """
    raw_client = MagicMock()
    raw_client.account_info.side_effect = Mt5RuntimeError("connection lost")
    client = _connected_client(raw_client)

    with pytest.raises(Mt5ConnectionError, match="connection lost"):
        switch_account(client, login=222, server="Demo")


def test_switch_account_fails_if_requested_account_is_not_active_after_login() -> None:
    """A successful login return cannot bypass active-account verification."""
    raw_client = MagicMock()
    raw_client.account_info.side_effect = [
        SimpleNamespace(login=111, server="Demo"),
        SimpleNamespace(login=222, server="Other"),
    ]
    raw_client.login.return_value = True
    client = _connected_client(raw_client)

    with pytest.raises(Mt5ConnectionError, match="did not activate"):
        switch_account(client, login=222, server="Demo")
