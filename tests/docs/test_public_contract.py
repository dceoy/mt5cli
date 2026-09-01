"""Tests linking the public API documentation to runtime exports."""

from __future__ import annotations

from pathlib import Path

import mt5cli
from mt5cli import STABLE_SDK_EXPORTS


def test_documented_contract_identifies_runtime_export_set() -> None:
    """The public-contract document points to the same runtime authority."""
    contract_doc = (
        Path(__file__).parents[2] / "docs" / "api" / "public-contract.md"
    ).read_text(encoding="utf-8")
    assert "`mt5cli.STABLE_SDK_EXPORTS`" in contract_doc
    assert set(mt5cli.__all__) - {"STABLE_SDK_EXPORTS"} == set(STABLE_SDK_EXPORTS)
    undocumented = sorted(
        name for name in STABLE_SDK_EXPORTS if f"`{name}`" not in contract_doc
    )
    assert not undocumented, f"Undocumented stable exports: {undocumented}"
