"""Persistent MT5 session lifecycle helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pdmt5 import Mt5RuntimeError

from .client import MT5Client
from .exceptions import Mt5ConnectionError, normalize_mt5_exception

if TYPE_CHECKING:
    from pdmt5 import Mt5DataClient
    from pydantic import SecretStr


def _account_matches(account: Any, *, login: int, server: str | None) -> bool:  # noqa: ANN401
    """Return whether an MT5 account snapshot matches the requested account."""
    if account is None or getattr(account, "login", None) != login:
        return False
    return server is None or getattr(account, "server", None) == server


def switch_account(
    client: MT5Client,
    *,
    login: int,
    password: str | SecretStr | None = None,
    server: str | None = None,
    timeout: int | None = None,
) -> None:
    """Switch an active persistent MT5 session without reinitializing the terminal.

    The configured account on ``client`` is intentionally left unchanged. Exiting and
    re-entering the client therefore reconnects to its original configuration; this
    helper only changes the account selected by the currently active process-global
    MT5 terminal session.

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
        if not connected.login(
            login=login,
            password=password,
            server=server,
            timeout=timeout,
        ):
            error_message = f"MT5 account switch failed: {connected.last_error()}"
            raise Mt5RuntimeError(error_message)
        account = connected.account_info()
        if not _account_matches(account, login=login, server=server):
            error_message = (
                "MT5 account switch did not activate the requested account: "
                f"login={login}, server={server!r}"
            )
            raise Mt5RuntimeError(error_message)
    except Mt5RuntimeError as exc:
        normalized = normalize_mt5_exception(exc)
        raise normalized from exc
