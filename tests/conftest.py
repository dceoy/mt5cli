"""Shared pytest fixtures for mt5cli tests."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import MagicMock

import pandas as pd
import pytest

if TYPE_CHECKING:
    from types import TracebackType

    from pytest_mock import MockerFixture

_DATAFRAME_METHODS = (
    "copy_rates_from_as_df",
    "copy_rates_from_pos_as_df",
    "copy_rates_range_as_df",
    "copy_ticks_from_as_df",
    "copy_ticks_range_as_df",
    "account_info_as_df",
    "terminal_info_as_df",
    "symbols_get_as_df",
    "symbol_info_as_df",
    "orders_get_as_df",
    "positions_get_as_df",
    "history_orders_get_as_df",
    "history_deals_get_as_df",
    "version_as_df",
    "last_error_as_df",
    "symbol_info_tick_as_df",
    "market_book_get_as_df",
    "order_check_as_df",
    "order_send_as_df",
)

_ORIGINAL_SQLITE_CONNECT = sqlite3.connect


class ClosingSqliteConnection(sqlite3.Connection):
    """SQLite connection that closes after context-manager exit in tests."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Commit or roll back the transaction, then close the connection."""
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()
        return False


def build_mock_mt5_data_client() -> MagicMock:
    """Return a MagicMock Mt5DataClient with common DataFrame stubs."""
    client = MagicMock()
    sample_df = pd.DataFrame({"col": [1]})
    for method_name in _DATAFRAME_METHODS:
        getattr(client, method_name).return_value = sample_df
    client.version.return_value = (5, 0, 1)
    client.terminal_info.return_value = {"connected": True, "paths": ["terminal.exe"]}
    client.account_info.return_value = {"login": 123, "limits": {"modes": ["demo"]}}
    client.symbols_total.return_value = 42
    return client


@pytest.fixture
def mock_client(mocker: MockerFixture) -> MagicMock:
    """Create and patch a mock Mt5DataClient for CLI and SDK tests."""
    client = build_mock_mt5_data_client()
    mocker.patch("mt5cli.client.Mt5DataClient", return_value=client)
    return client


@pytest.fixture(autouse=True)
def close_sqlite_context_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make test SQLite context managers close their connection handles."""

    def connect(
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> sqlite3.Connection:
        kwargs.setdefault("factory", ClosingSqliteConnection)
        return _ORIGINAL_SQLITE_CONNECT(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
