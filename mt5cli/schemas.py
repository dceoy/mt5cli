"""Canonical DataFrame schemas for MT5 market and account datasets."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Literal, cast

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
    """Return required column names for a dataset kind."""
    return REQUIRED_COLUMNS[kind]


def validate_schema(
    frame: pd.DataFrame,
    kind: DataKind,
    *,
    extra_required: Iterable[str] | None = None,
) -> None:
    """Validate that a DataFrame includes required columns for a dataset kind.

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


def _normalize_mt5_timestamp(value: object) -> object:
    """Normalize one parsed aware timestamp to UTC.

    Returns:
        UTC-normalized aware timestamp or the original naive value.
    """
    if isinstance(value, pd.Timestamp) and value.tzinfo is not None:
        return value.tz_convert("UTC")
    return value


def _normalize_parsed_mt5_times(series: pd.Series) -> pd.Series:
    """Normalize aware MT5 times to UTC while preserving naive wall-clock labels.

    Returns:
        Series with aware values converted to UTC and naive values unchanged.
    """
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")
    if pd.api.types.is_object_dtype(series.dtype):
        return series.map(_normalize_mt5_timestamp)
    return series


def _parse_mt5_time_value(value: object, *, unit: Literal["s", "ms"]) -> object:
    """Parse one mixed MT5 time value using the column's epoch unit.

    Returns:
        Parsed timestamp or ``NaT`` for an invalid value.
    """
    if pd.api.types.is_number(value):
        numeric_value = cast("float | int", value)
        return pd.to_datetime(numeric_value, unit=unit, errors="coerce")
    return cast(
        "pd.Timestamp",
        pd.to_datetime(cast("Any", value), errors="coerce", format="mixed"),
    )


def _mt5_time_sort_key(series: pd.Series) -> pd.Series:
    """Build UTC sort keys without changing stored MT5 timestamp values.

    Returns:
        UTC-aware comparable values for chronological sorting.
    """
    return pd.to_datetime(series, errors="coerce", format="mixed", utc=True)


def _coerce_mt5_time_column(series: pd.Series, column: str) -> pd.Series:
    """Coerce one MT5 time column without inventing a timezone.

    pdmt5 intentionally represents MT5 trade-server timestamps as
    timezone-naive wall-clock datetimes. Those values must remain naive because
    the broker/server UTC offset is not available at this boundary. Explicitly
    timezone-aware values are normalized to UTC. Numeric MT5 epoch fields are
    converted to the same naive wall-clock representation used by pdmt5.

    Returns:
        Parsed datetime series preserving the pdmt5 timestamp contract.
    """
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    unit: Literal["s", "ms"] = "ms" if column.endswith("_msc") else "s"
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit=unit, errors="coerce")
    if pd.api.types.is_object_dtype(series.dtype):
        parsed = series.map(
            lambda value: _parse_mt5_time_value(value, unit=unit),
        )
        return _normalize_parsed_mt5_times(parsed)
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except ValueError:
        parsed = series.map(
            lambda value: cast(
                "pd.Timestamp",
                pd.to_datetime(value, errors="coerce", format="mixed"),
            ),
        )
    return _normalize_parsed_mt5_times(parsed)


def normalize_time_columns(frame: pd.DataFrame, kind: DataKind) -> pd.DataFrame:
    """Normalize dataset time columns while preserving MT5 server-time semantics.

    Any column in :data:`KNOWN_MT5_TIME_COLUMNS` that is present in ``frame``
    is parsed. Timezone-naive values remain timezone-naive MT5 trade-server
    wall-clock labels. Explicitly aware values are normalized to UTC. Numeric
    fields use seconds for ``time``, ``time_setup``, and ``time_done``, and
    milliseconds for ``*_msc`` columns, producing naive values consistent with
    pdmt5's conversion contract.

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

    Preserves timezone-naive trade-server wall-clock timestamps from pdmt5,
    normalizes explicitly aware timestamps to UTC, optionally injects
    ``symbol`` / ``timeframe`` metadata, and sorts chronologically.

    Returns:
        Normalized DataFrame with canonical columns and ordering.
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
            normalized = normalized.sort_values(
                "time",
                key=_mt5_time_sort_key,
                kind="stable",
            )
        elif "time_msc" in normalized.columns:
            normalized = normalized.sort_values(
                "time_msc",
                key=_mt5_time_sort_key,
                kind="stable",
            )
        normalized = normalized.reset_index(drop=True)

    return normalized


def ensure_utc_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with selected datetime columns normalized safely.

    Known MT5 time columns preserve the pdmt5 server-wall-clock contract rather
    than being relabeled as UTC. Other columns are interpreted as generic
    datetimes and coerced to UTC.
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
