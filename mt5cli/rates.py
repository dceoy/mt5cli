"""Canonical normalized-rate persistence and loading APIs."""
# ruff: noqa: PLR0913

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, overload

import pandas as pd

from .history import RateTarget, load_rate_data, resolve_granularity_name
from .history import (
    ThrottledHistoryUpdater as _LegacyThrottledHistoryUpdater,
)
from .history import (
    open_existing_sqlite_database as _open_existing_sqlite_database,
)
from .history import (
    update_history as _legacy_update_history,
)
from .history import (
    update_history_with_config as _legacy_update_history_with_config,
)
from .schemas import (
    DataKind,
    normalize_time_columns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from pdmt5.dataframe import Mt5Config

    from .contract import HistoryClient
    from .history import UpdateHistoryBackend
    from .utils import Dataset

SqliteConnOrPath = sqlite3.Connection | Path | str


def _load_canonical_rate_target(
    conn: sqlite3.Connection,
    target: RateTarget,
    *,
    count: int,
) -> pd.DataFrame:
    """Load one rate series from canonical normalized storage."""
    if target.symbol is None:
        msg = "A symbol is required for canonical normalized rate loading."
        raise ValueError(msg)
    time_expr = (
        "COALESCE(strftime('%Y-%m-%dT%H:%M:%f', time), "
        "strftime('%Y-%m-%dT%H:%M:%f', time, 'unixepoch'))"
    )
    frame = pd.read_sql_query(  # type: ignore[reportUnknownMemberType]
        "SELECT * FROM rates WHERE symbol = ? AND timeframe = ? "
        f"ORDER BY {time_expr} DESC, ROWID DESC LIMIT ?",
        conn,
        params=(target.symbol, target.timeframe_int, count),
    )
    if "time" not in frame.columns:
        msg = "The canonical rates table is missing the required time column."
        raise ValueError(msg)
    if frame.empty:
        result = frame.drop(columns=["time"])
        result.index = pd.DatetimeIndex([], name="time")
        return result

    frame = normalize_time_columns(frame, DataKind.rates)
    awareness = [
        isinstance(value, pd.Timestamp) and value.tzinfo is not None
        for value in frame["time"]
    ]
    if any(awareness) and not all(awareness):
        msg = (
            "A canonical rate series cannot mix timezone-naive MT5 wall-clock "
            "timestamps with timezone-aware instants."
        )
        raise ValueError(msg)
    result = frame.drop(columns=["time"])
    result.index = pd.DatetimeIndex(frame["time"], name="time")
    return result.sort_index(kind="stable")



def _load_explicit_rate_series(
    conn_or_path: SqliteConnOrPath,
    targets: Sequence[RateTarget] | None,
    count: int | None,
    explicit_tables: Sequence[str] | None,
    *,
    table: str | None,
) -> dict[tuple[str | None, int], pd.DataFrame] | pd.DataFrame:
    """Load intentionally named custom tables without managed-view discovery."""
    if table is not None:
        return load_rate_data(conn_or_path, table, count=count)
    if targets is None:
        msg = "targets are required when explicit_tables is provided."
        raise ValueError(msg)
    if count is None or count <= 0:
        msg = "count must be positive."
        raise ValueError(msg)
    target_list = list(targets)
    tables = list(explicit_tables or ())
    if len(tables) != len(target_list):
        msg = (
            f"Expected {len(target_list)} explicit table(s) to match the "
            f"targets, got {len(tables)}."
        )
        raise ValueError(msg)
    keys = [(target.symbol, target.timeframe_int) for target in target_list]
    if len(set(keys)) != len(keys):
        msg = "Duplicate rate targets are not allowed."
        raise ValueError(msg)
    return {
        key: load_rate_data(conn_or_path, table_name, count=count)
        for key, table_name in zip(keys, tables, strict=True)
    }



if TYPE_CHECKING:

    @overload
    def load_rate_series_from_sqlite(
        conn_or_path: SqliteConnOrPath,
        targets: None = None,
        count: int | None = None,
        explicit_tables: None = None,
        *,
        table: str,
    ) -> pd.DataFrame: ...

    @overload
    def load_rate_series_from_sqlite(
        conn_or_path: SqliteConnOrPath,
        targets: None = None,
        count: int | None = None,
        explicit_tables: Sequence[str] | None = None,
        *,
        table: None = None,
    ) -> dict[tuple[str | None, int], pd.DataFrame]: ...

    @overload
    def load_rate_series_from_sqlite(
        conn_or_path: SqliteConnOrPath,
        targets: Sequence[RateTarget],
        count: int,
        explicit_tables: Sequence[str] | None = None,
        *,
        table: None = None,
    ) -> dict[tuple[str | None, int], pd.DataFrame]: ...


def load_rate_series_from_sqlite(
    conn_or_path: SqliteConnOrPath,
    targets: Sequence[RateTarget] | None = None,
    count: int | None = None,
    explicit_tables: Sequence[str] | None = None,
    *,
    table: str | None = None,
) -> dict[tuple[str | None, int], pd.DataFrame] | pd.DataFrame:
    """Load rate series from canonical storage or an explicit custom table.

    Normal symbol/timeframe targets query the normalized ``rates`` table
    directly. Explicit custom table loading remains supported without restoring
    per-series compatibility-view discovery.

    Returns:
        One explicit-table frame or target-keyed normalized rate frames.

    Raises:
        ValueError: If targets/count are invalid or canonical storage is absent.
    """
    if table is not None or explicit_tables is not None:
        try:
            return _load_explicit_rate_series(
                conn_or_path,
                targets,
                count,
                explicit_tables,
                table=table,
            )
        except (sqlite3.Error, pd.errors.DatabaseError) as exc:
            msg = "The explicit rate table is unavailable or invalid."
            raise ValueError(msg) from exc
    if count is None or count <= 0:
        msg = "count must be positive."
        raise ValueError(msg)
    if targets is None:
        msg = "targets are required when table is not provided."
        raise ValueError(msg)
    target_list = list(targets)
    if not target_list:
        msg = "At least one rate target is required."
        raise ValueError(msg)
    keys = [(target.symbol, target.timeframe_int) for target in target_list]
    if len(set(keys)) != len(keys):
        msg = "Duplicate rate targets are not allowed."
        raise ValueError(msg)

    try:
        conn, should_close = _open_existing_sqlite_database(conn_or_path)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        msg = "The canonical rates database could not be opened."
        raise ValueError(msg) from exc
    try:
        return {
            key: _load_canonical_rate_target(conn, target, count=count)
            for key, target in zip(keys, target_list, strict=True)
        }
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        msg = "The canonical rates table is unavailable or invalid."
        raise ValueError(msg) from exc
    finally:
        if should_close:
            conn.close()


def load_rate_series_by_granularity(
    conn_or_path: SqliteConnOrPath,
    symbols: Sequence[str],
    granularities: Sequence[int | str],
    count: int,
    *,
    explicit_tables: Sequence[str] | None = None,
    allow_missing_symbol: bool = False,
) -> dict[tuple[str | None, str], pd.DataFrame]:
    """Load rate series keyed by symbol and canonical granularity name.

    Returns:
        Mapping keyed by ``(symbol, granularity)``.

    Raises:
        TypeError: If explicit table loading returns a single DataFrame.
    """
    from .history import build_rate_targets  # noqa: PLC0415

    targets = build_rate_targets(
        symbols,
        granularities,
        allow_missing_symbol=allow_missing_symbol,
    )
    series = load_rate_series_from_sqlite(
        conn_or_path,
        targets,
        count,
        explicit_tables=explicit_tables,
    )
    if isinstance(series, pd.DataFrame):
        msg = "Expected multiple rate series."
        raise TypeError(msg)
    return {
        (symbol, resolve_granularity_name(timeframe)): frame
        for (symbol, timeframe), frame in series.items()
    }




def update_history(
    *,
    client: HistoryClient,
    output: Path | str,
    symbols: Sequence[str],
    datasets: set[Dataset] | None = None,
    timeframes: Sequence[int | str] | None = None,
    flags: int | str = "ALL",
    lookback_hours: float = 24.0,
    date_to: datetime | str | None = None,
    deduplicate: bool = True,
    with_views: bool = False,
    include_account_events: bool = True,
) -> None:
    """Incrementally update history without creating rate compatibility views.

    The stable wrapper preserves the existing update contract while forcing the
    canonical normalized-rate model and removing stale compatibility views.
    """
    _legacy_update_history(
        client=client,
        output=output,
        symbols=symbols,
        datasets=datasets,
        timeframes=timeframes,
        flags=flags,
        lookback_hours=lookback_hours,
        date_to=date_to,
        deduplicate=deduplicate,
        with_views=with_views,
        include_account_events=include_account_events,
    )


def update_history_with_config(
    *,
    output: Path | str,
    symbols: Sequence[str],
    config: Mt5Config | None = None,
    datasets: set[Dataset] | None = None,
    timeframes: Sequence[int | str] | None = None,
    flags: int | str = "ALL",
    lookback_hours: float = 24.0,
    date_to: datetime | str | None = None,
    deduplicate: bool = True,
    with_views: bool = False,
    include_account_events: bool = True,
) -> None:
    """Update managed-session history without rate compatibility views."""
    _legacy_update_history_with_config(
        output=output,
        symbols=symbols,
        config=config,
        datasets=datasets,
        timeframes=timeframes,
        flags=flags,
        lookback_hours=lookback_hours,
        date_to=date_to,
        deduplicate=deduplicate,
        with_views=with_views,
        include_account_events=include_account_events,
    )


class ThrottledHistoryUpdater(_LegacyThrottledHistoryUpdater):
    """Throttle canonical history updates without legacy rate views."""

    def __init__(
        self,
        *,
        output: Path | str,
        datasets: set[Dataset] | None = None,
        timeframes: Sequence[int | str] | None = None,
        flags: int | str = "ALL",
        lookback_hours: float = 24.0,
        with_views: bool = False,
        include_account_events: bool = True,
        interval_seconds: float = 0.0,
        suppress_errors: bool = False,
        update_backend: UpdateHistoryBackend | None = None,
    ) -> None:
        """Initialize with the canonical update backend by default."""
        super().__init__(
            output=output,
            datasets=datasets,
            timeframes=timeframes,
            flags=flags,
            lookback_hours=lookback_hours,
            with_views=with_views,
            include_account_events=include_account_events,
            interval_seconds=interval_seconds,
            suppress_errors=suppress_errors,
            update_backend=(
                update_history if update_backend is None else update_backend
            ),
        )


__all__ = [
    "ThrottledHistoryUpdater",
    "load_rate_series_by_granularity",
    "load_rate_series_from_sqlite",
    "update_history",
    "update_history_with_config",
]
