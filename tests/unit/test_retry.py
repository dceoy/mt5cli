"""Tests for the :mod:`mt5cli.retry` module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pdmt5 import Mt5RuntimeError

from mt5cli.retry import retry_with_backoff

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_retry_with_backoff_retries_recoverable_errors(
    mocker: MockerFixture,
) -> None:
    """Retry helper retries recoverable MT5 failures."""
    calls = {"count": 0}

    def _flaky() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            message = "transient"
            raise Mt5RuntimeError(message)
        return "ok"

    mocker.patch("mt5cli.retry.time.sleep")
    assert retry_with_backoff(_flaky, retry_count=1) == "ok"
    assert calls["count"] == 2


def test_retry_with_backoff_reraises_non_recoverable_errors() -> None:
    """Non-MT5 errors are not retried."""

    def _raise() -> None:
        message = "fatal"
        raise ValueError(message)

    with pytest.raises(ValueError, match="fatal"):
        retry_with_backoff(_raise, retry_count=2)
