"""Persistent MT5 session lifecycle helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pdmt5 import Mt5RuntimeError

from .exceptions import Mt5ConnectionError, normalize_mt5_exception

if TYPE_CHECKING:
    from pdmt5 import Mt5DataClient

    from .client import MT5Client


def _account_matches(account: Any, *, login: int, server: str | None) -> bool:  # noqa: ANN401
    """Return whether an MT5 account snapshot matches the requested account."""
    if account is None or getattr(account, "login", None) != login:
        return False
    return server is None or getattr(account, "server", None) == server


def switch_account(
    client: MT5Client,
    *,
    login: int,
    password: str | None = None,
    server: str | None = None,
    timeout: int | None = None,
) -> None:
    """Switch an active persistent MT5 session without reinitializing the terminal.

    The configured account on ``client`` is intentionally left unchanged, but this
    does not restore the prior account. For a self-owned client
    (``with MT5Client(...) as client:``), exiting and re-entering reconnects using
    the original configuration. For clients obtained via ``mt5_session()`` or
    ``from_connected_client()`` (the common case), exiting and re-entering ``client``
    has no effect on the connection at all, so the switched account persists for the
    life of the terminal session.

    Args:
        client: An already-entered ``MT5Client`` with a persistent connection.
        login: Target MT5 account login.
        password: Optional target account password.
        server: Optional target trade server.
        timeout: Optional MT5 login timeout in milliseconds.

    Raises:
        Mt5ConnectionError: If the client is not active or the account switch fails.
    """
    connected = cast("Mt5DataClient | None", getattr(client, "_client", None))
    if connected is None:
        msg = "MT5 account switching requires an active persistent session."
        raise Mt5ConnectionError(msg)

    try:
        account = connected.account_info()
        if _account_matches(account, login=login, server=server):
            return
        login_succeeded = connected.login(
            login=login,
            password=password,
            server=server,
            timeout=timeout,
        )
        last_error = None if login_succeeded else connected.last_error()
        active_account = connected.account_info() if login_succeeded else None
    except Mt5RuntimeError as exc:
        normalized = normalize_mt5_exception(exc)
        raise normalized from exc

    if not login_succeeded:
        msg = f"MT5 account switch failed: {last_error}"
        raise Mt5ConnectionError(msg)
    if not _account_matches(active_account, login=login, server=server):
        msg = (
            "MT5 account switch did not activate the requested account: "
            f"requested login={login}, server={server!r}, "
            f"active login={getattr(active_account, 'login', None)!r}, "
            f"server={getattr(active_account, 'server', None)!r}"
        )
        raise Mt5ConnectionError(msg)
