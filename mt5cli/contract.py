"""Structural contracts used internally by mt5cli."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd


class HistoryClient(Protocol):
    """Structural contract required by history collection and updates.

    :class:`~mt5cli.client.MT5Client` satisfies this protocol. It exists so
    that history collection code depends on canonical mt5cli method names
    instead of probing for raw pdmt5 method names.
    """

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: int | str,
        date_from: datetime | str,
        date_to: datetime | str,
    ) -> pd.DataFrame:
        """Return rates for a naive trade-server wall-clock date range."""
        ...

    def copy_ticks_range(
        self,
        symbol: str,
        date_from: datetime | str,
        date_to: datetime | str,
        flags: int | str,
    ) -> pd.DataFrame:
        """Return ticks for a naive trade-server wall-clock date range."""
        ...

    def history_orders(
        self,
        date_from: datetime | str | None = None,
        date_to: datetime | str | None = None,
        group: str | None = None,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ) -> pd.DataFrame:
        """Return historical orders for optional naive server-time bounds."""
        ...

    def history_deals(
        self,
        date_from: datetime | str | None = None,
        date_to: datetime | str | None = None,
        group: str | None = None,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ) -> pd.DataFrame:
        """Return historical deals for optional naive server-time bounds."""
        ...

    def symbol_info_as_dict(self, symbol: str) -> dict[str, object]:
        """Return one symbol snapshot as a plain mapping."""
        ...


class ObservabilityClient(Protocol):
    """Structural contract required by observability snapshot orchestration.

    :class:`~mt5cli.client.MT5Client` satisfies this protocol.
    """

    def account_info(self) -> pd.DataFrame:
        """Return account information."""
        ...

    def terminal_info(self) -> pd.DataFrame:
        """Return terminal information."""
        ...

    def positions(
        self,
        symbol: str | None = None,
        group: str | None = None,
        ticket: int | None = None,
    ) -> pd.DataFrame:
        """Return open positions."""
        ...

    def orders(
        self,
        symbol: str | None = None,
        group: str | None = None,
        ticket: int | None = None,
    ) -> pd.DataFrame:
        """Return active orders."""
        ...


__all__ = ["HistoryClient", "ObservabilityClient"]
