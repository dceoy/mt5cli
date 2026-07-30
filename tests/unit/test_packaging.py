"""Packaging metadata invariants for mt5cli."""

from __future__ import annotations

from importlib.metadata import requires

from packaging.requirements import Requirement


def _mt5cli_requirements() -> list[Requirement]:
    reqs = requires("mt5cli") or []
    return [Requirement(r) for r in reqs]


def test_parquet_extra_declares_pyarrow() -> None:
    """Package metadata lists pyarrow under the parquet optional extra."""
    parquet_reqs = [
        req
        for req in _mt5cli_requirements()
        if req.name == "pyarrow"
        and req.marker is not None
        and 'extra == "parquet"' in str(req.marker)
    ]
    assert parquet_reqs, "pyarrow not found in parquet optional extra"


def test_pyarrow_not_in_core_dependencies() -> None:
    """Pyarrow is not a core dependency; it belongs only in the parquet extra."""
    core_reqs = [
        req
        for req in _mt5cli_requirements()
        if req.marker is None or "extra ==" not in str(req.marker)
    ]
    assert not any(req.name == "pyarrow" for req in core_reqs), (
        "pyarrow should not appear in core dependencies"
    )
