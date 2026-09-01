"""Tests for the :mod:`mt5cli.sdk` module."""

from __future__ import annotations

import mt5cli
from mt5cli import STABLE_SDK_EXPORTS, sdk


def test_sdk_module_is_authoritative_stable_surface() -> None:
    """The package root re-exports the SDK source-of-truth declarations."""
    assert mt5cli.sdk.STABLE_SDK_EXPORTS is STABLE_SDK_EXPORTS
    assert set(sdk.__all__) == set(STABLE_SDK_EXPORTS)


def test_sdk_exports_are_derived_from_public_imports() -> None:
    """The SDK export registry has no second manually maintained name list."""
    public_imports = {
        name
        for name in vars(sdk)
        if not name.startswith("_") and name != "STABLE_SDK_EXPORTS"
    }
    assert set(sdk.__all__) == public_imports
    assert frozenset(public_imports) == sdk.STABLE_SDK_EXPORTS
