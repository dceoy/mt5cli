"""mt5cli: Generic MT5 data and execution infrastructure for Python applications.

Downstream packages should import from this module (``from mt5cli import ...``)
rather than private submodule helpers. See ``docs/api/public-contract.md`` for
the stable SDK contract, CLI surface, internal modules, and out-of-scope
strategy responsibilities.
"""
# pyright: reportUnusedImport=false, reportUnsupportedDunderAll=false
# ruff: noqa: F401, F403

from importlib.metadata import version as _package_version

from .sdk import *
from .sdk import STABLE_SDK_EXPORTS, __all__ as _SDK_EXPORTS

__version__ = _package_version(__package__) if __package__ else None
__all__ = [*_SDK_EXPORTS, "STABLE_SDK_EXPORTS"]
