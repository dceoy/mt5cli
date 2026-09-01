"""Architecture checks for production-module and unit-test ownership."""

from __future__ import annotations

from pathlib import Path

_BOOTSTRAP_MODULES = {"__init__.py", "__main__.py"}


def test_production_modules_have_exactly_one_aligned_unit_test() -> None:
    """Direct production modules and unit tests form an exact one-to-one set."""
    root = Path(__file__).parents[2]
    production_names = {
        path.stem
        for path in (root / "mt5cli").glob("*.py")
        if path.is_file() and path.name not in _BOOTSTRAP_MODULES
    }
    expected_names = {f"test_{name}.py" for name in production_names}

    unit_dir = root / "tests" / "unit"
    actual_names = {path.name for path in unit_dir.glob("test_*.py") if path.is_file()}
    allowed_names = {"__init__.py", "conftest.py", *expected_names}
    unexpected_names = {
        path.name
        for path in unit_dir.glob("*.py")
        if path.is_file() and path.name not in allowed_names
    }

    assert not expected_names - actual_names, (
        f"Missing module-aligned unit tests: {sorted(expected_names - actual_names)}"
    )
    assert not actual_names - expected_names, (
        f"Orphan unit tests: {sorted(actual_names - expected_names)}"
    )
    assert not unexpected_names, (
        f"Unexpected direct Python files under tests/unit: {sorted(unexpected_names)}"
    )
