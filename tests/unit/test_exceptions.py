"""Tests for the :mod:`mt5cli.exceptions` module."""

from __future__ import annotations

import pytest
from pdmt5 import Mt5RuntimeError

from mt5cli.exceptions import (
    Mt5CliError,
    Mt5ConnectionError,
    call_with_normalized_errors,
    is_recoverable_mt5_error,
    normalize_mt5_exception,
)


@pytest.mark.parametrize(
    "exc",
    [Mt5RuntimeError("init failed"), Mt5ConnectionError("normalized failure")],
)
def test_is_recoverable_mt5_error(exc: Exception) -> None:
    """Recoverable MT5 errors are classified consistently."""
    assert is_recoverable_mt5_error(exc)


@pytest.mark.parametrize(
    ("exc", "expected_type"),
    [
        (Mt5RuntimeError("x"), Mt5ConnectionError),
    ],
)
def test_normalize_mt5_exception_maps_types(
    exc: Exception,
    expected_type: type[Mt5ConnectionError],
) -> None:
    """MT5 exceptions map to stable mt5cli types."""
    assert isinstance(normalize_mt5_exception(exc), expected_type)


def test_call_with_normalized_errors_reraises_mapped_type() -> None:
    """Normalized error helper re-raises mapped mt5cli exceptions."""

    def _raise() -> None:
        message = "boom"
        raise Mt5RuntimeError(message)

    with pytest.raises(Mt5ConnectionError):
        call_with_normalized_errors(_raise)


def test_normalize_mt5_exception_passthrough_and_generic() -> None:
    """Normalization preserves mt5cli and unrelated application exceptions."""
    original = Mt5CliError("known")
    assert normalize_mt5_exception(original) is original
    application_error = ValueError("x")
    assert normalize_mt5_exception(application_error) is application_error
