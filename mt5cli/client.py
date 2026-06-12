"""Stable public client abstraction for MT5 data and execution operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self

from .sdk import Mt5CliClient, build_config, connected_client

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd
    from pdmt5 import Mt5Config, Mt5DataClient

__all__ = [
    "MT5Client",
    "build_config",
    "mt5_session",
]


class MT5Client(Mt5CliClient):
    """Public client for generic MT5 data access and order primitives.

    Extends the read-only SDK client with optional order check/send helpers and
    exposes the same connection lifecycle as :class:`~mt5cli.sdk.Mt5CliClient`.
    Downstream applications such as private trading packages should prefer this
    type over the legacy ``Mt5CliClient`` name.
    """

    def order_check(self, request: dict[str, Any]) -> pd.DataFrame:
        """Check funds sufficiency for a trade request.

        Args:
            request: MT5 order request dictionary.

        Returns:
            One-row DataFrame with the order-check result.
        """
        return self._fetch(lambda client: client.order_check_as_df(request=request))

    def order_send(self, request: dict[str, Any]) -> pd.DataFrame:
        """Send a trade request to the MT5 trade server.

        Args:
            request: MT5 order request dictionary.

        Returns:
            One-row DataFrame with the order-send result.
        """
        return self._fetch(lambda client: client.order_send_as_df(request=request))

    @classmethod
    def from_connected_client(cls, client: Mt5DataClient) -> Self:
        """Bind to an already-connected ``Mt5DataClient`` without owning it.

        Returns:
            Client wrapper bound to the injected connection.
        """
        return cls(client=client)


@contextmanager
def mt5_session(config: Mt5Config | None = None) -> Iterator[MT5Client]:
    """Open an MT5 terminal session and yield a connected :class:`MT5Client`.

    Args:
        config: MT5 connection configuration. Defaults to an empty config that
            attaches to a running terminal.

    Yields:
        Connected :class:`MT5Client` bound to the session.
    """
    mt5_config = config or build_config()
    with connected_client(mt5_config) as client:
        yield MT5Client.from_connected_client(client)
