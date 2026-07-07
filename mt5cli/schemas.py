"""Canonical DataFrame schemas for MT5 market and account datasets."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final

import pandas as pd

from .converters import normalize_symbol, parse_timeframe
from .exceptions import Mt5SchemaError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "DEDUP_KEYS",
    "KNOWN_MT5_TIME_COLUMNS",
    "REQUIRED_COLUMNS",
    "TIME_COLUMNS",
    "DataKind",
    "normalize_dataframe",
    "normalize_time_columns",
    "schema_columns",
    "validate_schema",
]

KNOWN_MT5_TIME_COLUMNS: Final[frozenset[str]] = frozenset({
    "time",
    "time_setup",
    "time_setup_msc",
    "time_done",
    "time_done_msc",
    "time_msc",
})

_TIME_COLUMN_NAMES = KNOWN_MT5_TIME_COLUMNS


class DataKind(StrEnum):
    """Supported MT5 dataset kinds with canonical column contracts."""

    rates = "rates"
    ticks = "ticks"
    orders = "orders"
    positions = "positions"
    history_orders = "history_orders"
    history_deals = "history_deals"
    symbols = "symbols"


REQUIRED_COLUMNS: dict[DataKind, frozenset[str]] = {
    DataKind.rates: frozenset({
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    }),
    DataKind.ticks: frozenset({
        "time",
        "bid",
        "ask",
        "last",
        "volume",
        "time_msc",
        "flags",
        "volume_real",
    }),
    DataKind.orders: frozenset({
        "ticket",
        "time_setup",
        "type",
        "state",
        "symbol",
        "volume_current",
        "price_open",
    }),
    DataKind.positions: frozenset({
        "ticket",
        "time",
        "type",
        "symbol",
        "volume",
        "price_open",
        "price_current",
        "profit",
    }),
    DataKind.history_orders: frozenset({
        "ticket",
        "time_setup",
        "type",
        "state",
        "symbol",
        "volume_initial",
        "price_open",
    }),
    DataKind.history_deals: frozenset({
        "ticket",
        "order",
        "time",
        "type",
        "entry",
        "symbol",
        "volume",
        "price",
        "profit",
    }),
    DataKind.symbols: frozenset({
        "symbol",
        "time",
        "point",
        "digits",
        "trade_contract_size",
        "volume_min",
        "volume_max",
        "volume_step",
        "trade_tick_size",
        "trade_tick_value",
        "currency_profit",
    }),
}

_OPTIONAL_TIME_COLUMNS_BY_KIND: dict[DataKind, frozenset[str]] = {
    DataKind.orders: frozenset({
        "time_setup_msc",
        "time_done",
        "time_done_msc",
    }),
    DataKind.history_orders: frozenset({
        "time_setup_msc",
        "time_done",
        "time_done_msc",
    }),
    DataKind.positions: frozenset({"time_msc"}),
}

TIME_COLUMNS: dict[DataKind, frozenset[str]] = {
    kind: (REQUIRED_COLUMNS[kind] & _TIME_COLUMN_NAMES)
    | _OPTIONAL_TIME_COLUMNS_BY_KIND.get(kind, frozenset())
    for kind in DataKind
}

DEDUP_KEYS: dict[DataKind, tuple[tuple[str, ...], ...]] = {
    DataKind.rates: (("symbol", "timeframe", "time"), ("symbol", "time")),
    DataKind.ticks: (("symbol", "time_msc"), ("symbol", "time")),
    DataKind.history_orders: (("ticket",), ("symbol", "time", "type")),
    DataKind.history_deals: (("ticket",), ("symbol", "time", "type", "entry")),
    DataKind.symbols: (("symbol", "time"),),
}


def schema_columns(kind: DataKind) -> frozenset[str]:
    """Return required column names for a dataset kind.

    Args:
        kind: Dataset kind.

    Returns:
        Required column names for ``kind``.
    """
    return REQUIRED_COLUMNS[kind]


def validate_schema(
    frame: pd.DataFrame,
    kind: DataKind,
    *,
    extra_required: Iterable[str] | None = None,
) -> None:
    """Validate that a DataFrame includes required columns for a dataset kind.

    Args:
        frame: DataFrame to validate.
        kind: Expected dataset kind.
        extra_required: Additional columns that must be present (for example
            ``symbol`` and ``timeframe`` on stored rate history).

    Raises:
        Mt5SchemaError: If required columns are missing.
    """
    if frame.empty and len(frame.columns) == 0:
        return
    required = set(REQUIRED_COLUMNS[kind])
    if extra_required is not None:
        required.update(extra_required)
    missing = required - set(frame.columns)
    if missing:
        msg = (
            f"{kind.value} schema is missing required columns: "
            f"{', '.join(sorted(missing))}."
        )
        raise Mt5SchemaError(msg)


def _coerce_mt5_time_column(series: pd.Series, column: str) -> pd.Series:
    """Coerce one MT5 time column to UTC-aware datetimes.

    Returns:
        Series with UTC-aware datetime values.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    if pd.api.types.is_numeric_dtype(series):
        unit = "ms" if column.endswith("_msc") else "s"
        return pd.to_datetime(series, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def normalize_time_columns(frame: pd.DataFrame, kind: DataKind) -> pd.DataFrame:
    """Coerce dataset time columns to UTC-aware datetimes when present.

    Any column in :data:`KNOWN_MT5_TIME_COLUMNS` that is present in ``frame``
    is normalized. Numeric MT5 epoch values use seconds for ``time``,
    ``time_setup``, and ``time_done``, and milliseconds for ``*_msc`` columns.

    Args:
        frame: Source DataFrame from MT5 or pdmt5.
        kind: Dataset kind (retained for API compatibility).

    Returns:
        DataFrame copy with normalized time columns.
    """
    del kind
    normalized = frame.copy()
    for column in normalized.columns:
        if column not in _TIME_COLUMN_NAMES:
            continue
        normalized[column] = _coerce_mt5_time_column(normalized[column], column)
    return normalized


def normalize_dataframe(
    frame: pd.DataFrame,
    kind: DataKind,
    *,
    symbol: str | None = None,
    timeframe: int | str | None = None,
    sort: bool = True,
) -> pd.DataFrame:
    """Normalize MT5 DataFrame columns, timestamps, and storage metadata.

    Ensures UTC timestamps, optionally injects ``symbol`` / ``timeframe`` for
    storage-oriented datasets, and sorts chronologically when a ``time`` column
    exists.

    Args:
        frame: Source DataFrame from MT5 or pdmt5.
        kind: Dataset kind guiding normalization rules.
        symbol: Optional symbol to inject when missing.
        timeframe: Optional timeframe integer or name to inject for rates.
        sort: Whether to sort by ``time`` or ``time_msc`` when present.

    Returns:
        Normalized DataFrame copy.
    """
    if frame.empty and len(frame.columns) == 0:
        return frame.copy()

    normalized = normalize_time_columns(frame, kind)

    if symbol is not None and "symbol" not in normalized.columns:
        normalized.insert(0, "symbol", normalize_symbol(symbol))

    if timeframe is not None and kind is DataKind.rates:
        tf = parse_timeframe(timeframe)
        if "timeframe" not in normalized.columns:
            insert_at = 1 if "symbol" in normalized.columns else 0
            normalized.insert(insert_at, "timeframe", tf)

    validate_schema(normalized, kind)

    if sort:
        if "time" in normalized.columns:
            normalized = normalized.sort_values("time", kind="stable")
        elif "time_msc" in normalized.columns:
            normalized = normalized.sort_values("time_msc", kind="stable")
        normalized = normalized.reset_index(drop=True)

    return normalized


def ensure_utc_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with selected columns coerced to UTC datetimes.

    Args:
        frame: Source DataFrame.
        columns: Column names to coerce.

    Returns:
        DataFrame copy with UTC-aware datetime columns.
    """
    normalized = frame.copy()
    for column in columns:
        if column not in normalized.columns:
            continue
        if column in _TIME_COLUMN_NAMES:
            normalized[column] = _coerce_mt5_time_column(normalized[column], column)
        else:
            normalized[column] = pd.to_datetime(
                normalized[column], utc=True, errors="coerce"
            )
    return normalized
