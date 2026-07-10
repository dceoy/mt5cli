"""Stable public client abstraction for MT5 data and execution operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self, cast

from .sdk import Mt5CliClient, build_config, connected_client

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd
    from pdmt5 import Mt5Config

__all__ = [
    "MT5Client",
    "build_config",
    "mt5_session",
]


class MT5Client(Mt5CliClient):
    """The single public connected MT5 client.

    Extends the read-only SDK client with optional order check/send helpers and
    exposes the same connection lifecycle as :func:`mt5_session`.

    mt5cli intentionally exposes minimal execution primitives only. Trading
    decisions, signals, strategies, backtests, and optimization remain the
    responsibility of downstream applications.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout: int | None = None,
        retry_count: int = 3,
        config: Mt5Config | None = None,
    ) -> None:
        """Configure a client; use it as a context manager to connect."""
        super().__init__(
            path=path,
            login=login,
            password=password,
            server=server,
            timeout=timeout,
            retry_count=retry_count,
            config=config,
        )

    @classmethod
    def _from_connected_client(cls, client: object) -> Self:
        """Create a facade around an internally managed pdmt5 connection.

        Returns:
            Public facade that does not own the supplied connection.
        """
        instance = cls()
        instance._client = cast("Any", client)
        instance._owns_client = False
        return instance

    @classmethod
    def from_connected_client(cls, client: object) -> Self:
        """Bind the public facade to an externally owned connection.

        Returns:
            Public facade that never initializes or shuts down ``client``.
        """
        return cls._from_connected_client(client)

    @property
    def mt5(self) -> Any:  # noqa: ANN401
        """Return MT5 constants required by operational helpers.

        This intentionally exposes constants only, never the pdmt5 client.
        """
        return self._fetch_value(lambda client: client.mt5)

    def account_info_as_dict(self) -> dict[str, object]:
        """Return the current account snapshot as a plain mapping."""
        frame = self.account_info()
        return {} if frame.empty else cast("dict[str, object]", frame.iloc[0].to_dict())

    def symbol_info_as_dict(self, symbol: str) -> dict[str, object]:
        """Return one symbol snapshot as a plain mapping."""
        frame = self.symbol_info(symbol)
        return {} if frame.empty else cast("dict[str, object]", frame.iloc[0].to_dict())

    def positions_get_as_df(self, symbol: str | None = None) -> pd.DataFrame:
        """Return open positions in the canonical DataFrame schema."""
        return self.positions(symbol=symbol)

    def order_calc_margin(
        self, action: int, symbol: str, volume: float, price: float
    ) -> object:
        """Calculate broker margin for an order candidate.

        Returns:
            Broker-calculated margin value.
        """
        return self._fetch_value(
            lambda client: client.order_calc_margin(action, symbol, volume, price)
        )

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        """Select or deselect a symbol in Market Watch.

        Returns:
            Whether MT5 accepted the selection operation.
        """
        return bool(
            self._fetch_value(
                lambda client: client.symbol_select(symbol, enable=enable)
            )
        )

    def order_check(self, request: dict[str, Any]) -> pd.DataFrame:
        """Check funds sufficiency for a trade request.

        Args:
            request: MT5 order request dictionary.

        Returns:
            One-row DataFrame with the order-check result.
        """
        return self._fetch(lambda client: client.order_check_as_df(request=request))

    def order_send(self, request: dict[str, Any]) -> pd.DataFrame:
        """Send a live trade request to the MT5 trade server.

        Warning:
            This is a live execution primitive. A successful call can place,
            modify, or close real trades on the connected account. Downstream
            applications must gate usage explicitly (for example behind manual
            confirmation or application-specific risk controls). mt5cli does
            not implement strategy logic, signal generation, or trade sizing.

        Args:
            request: MT5 order request dictionary.

        Returns:
            One-row DataFrame with the order-send result.
        """
        return self._fetch(lambda client: client.order_send_as_df(request=request))


@contextmanager
def mt5_session(
    config: Mt5Config | None = None, *, client: MT5Client | None = None
) -> Iterator[MT5Client]:
    """Open an MT5 terminal session and yield a connected :class:`MT5Client`.

    Args:
        config: MT5 connection configuration. Defaults to an empty config that
            attaches to a running terminal.
        client: A caller-owned connected public client. It is yielded as-is and
            is never initialized or shut down by this context manager.

    Yields:
        Connected :class:`MT5Client` bound to the session.
    """
    if client is not None:
        yield client
        return
    mt5_config = config or build_config()
    with connected_client(mt5_config) as raw_client:
        yield MT5Client._from_connected_client(raw_client)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
