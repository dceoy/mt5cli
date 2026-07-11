"""Normalized exception types for MT5 and mt5cli operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pdmt5 import Mt5RuntimeError

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

__all__ = [
    "Mt5CliError",
    "Mt5ConnectionError",
    "Mt5OperationError",
    "Mt5SchemaError",
    "call_with_normalized_errors",
    "is_recoverable_mt5_error",
    "normalize_mt5_exception",
]

_NORMALIZABLE_MT5_ERRORS: tuple[type[BaseException], ...] = (Mt5RuntimeError,)


class Mt5CliError(Exception):
    """Base exception for mt5cli public API errors."""


class Mt5ConnectionError(Mt5CliError):
    """Raised when MT5 initialization, login, or shutdown fails."""


class Mt5OperationError(Mt5CliError):
    """Raised when an MT5 data or trading operation fails."""


class Mt5SchemaError(Mt5CliError):
    """Raised when a DataFrame does not match an expected dataset schema."""


_RECOVERABLE_MT5_ERRORS: tuple[type[BaseException], ...] = (
    Mt5RuntimeError,
    Mt5ConnectionError,
)


def is_recoverable_mt5_error(exc: BaseException) -> bool:
    """Return whether an exception is a transient MT5 failure worth retrying.

    Args:
        exc: Exception raised by MT5 or pdmt5.

    Returns:
        True for ``Mt5RuntimeError`` and its normalized ``Mt5ConnectionError``
        form.
    """
    return isinstance(exc, _RECOVERABLE_MT5_ERRORS)


def normalize_mt5_exception(exc: BaseException) -> BaseException:
    """Map pdmt5/MT5 exceptions to stable mt5cli exception types.

    Args:
        exc: Original exception from MT5 or pdmt5.

    Returns:
        ``Mt5ConnectionError`` for runtime failures; unrelated exceptions are
        returned unchanged.
    """
    if isinstance(exc, Mt5RuntimeError):
        return Mt5ConnectionError(str(exc))
    if isinstance(exc, Mt5CliError):
        return exc
    return exc


def call_with_normalized_errors(fn: Callable[[], T]) -> T:
    """Run ``fn`` and map recoverable MT5 errors to mt5cli types.

    Args:
        fn: Callable performing MT5 work.

    Returns:
        Value returned by ``fn``.
    """
    try:
        return fn()
    except _NORMALIZABLE_MT5_ERRORS as exc:
        normalized = normalize_mt5_exception(exc)
        raise normalized from exc
