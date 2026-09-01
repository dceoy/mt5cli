"""Shared pytest fixtures for all mt5cli tests."""
# ruff: noqa: INP001

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any, Literal


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
