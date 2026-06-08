"""Compatibility shim for ``mt5cli.history``.

Deprecated: import from ``mt5cli.history`` instead.
"""

from __future__ import annotations

from . import history as _history


def __getattr__(name: str) -> object:
    return getattr(_history, name)


def __dir__() -> list[str]:
    return [item for item in dir(_history) if not item.startswith("_")]
