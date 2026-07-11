"""Retry and reconnect helpers for transient MT5 failures."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, TypeVar

from .exceptions import is_recoverable_mt5_error

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = [
    "retry_with_backoff",
]


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retry_count: int = 0,
    backoff_base: float = 2.0,
    operation: str = "MT5 operation",
) -> T:
    """Call ``fn`` with bounded exponential backoff on recoverable MT5 errors.

    Only ``pdmt5.Mt5RuntimeError`` and its normalized ``Mt5ConnectionError``
    form are retried. Other exceptions propagate immediately. The final
    failure is re-raised once retries are exhausted.

    Args:
        fn: Callable performing MT5 work.
        retry_count: Maximum number of retries after the first attempt. ``0``
            disables retries.
        backoff_base: Base for exponential backoff. The delay before retry
            attempt ``n`` (1-indexed) is ``backoff_base ** n`` seconds.
        operation: Label used in warning logs.

    Returns:
        Value returned by ``fn`` on success.
    """
    attempts = max(retry_count, 0) + 1
    for attempt in range(attempts - 1):
        try:
            return fn()
        except Exception as exc:
            if not is_recoverable_mt5_error(exc):
                raise
            delay = backoff_base ** (attempt + 1)
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                operation,
                attempt + 1,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    return fn()
