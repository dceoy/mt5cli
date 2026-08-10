"""Canonical normalized-rate persistence and loading APIs."""
# ruff: noqa: PLR0913

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, overload

import pandas as pd

from .history import (
    RateTarget,
    drop_rate_compatibility_views,
    resolve_granularity_name,
)
from .history import (
    ThrottledHistoryUpdater as _LegacyThrottledHistoryUpdater,
)
from .history import (
    load_rate_series_from_sqlite as _legacy_load_rate_series_from_sqlite,
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
    _mt5_time_sort_key,  # type: ignore[reportPrivateUsage]
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
    """Load one rate series directly from the normalized ``rates`` table.

    Returns:
        Chronological normalized rate frame.

    Raises:
        ValueError: If the target has no symbol.
    """
    if target.symbol is None:
        msg = "A symbol is required for canonical normalized rate loading."
        raise ValueError(msg)
    frame = pd.read_sql_query(  # type: ignore[reportUnknownMemberType]
        "SELECT * FROM rates WHERE symbol = ? AND timeframe = ?",
        conn,
        params=(target.symbol, target.timeframe_int),
    )
    if "time" not in frame.columns:
        msg = "The canonical rates table is missing the required time column."
        raise ValueError(msg)
    if frame.empty:
        return frame
    frame = normalize_time_columns(frame, DataKind.rates)
    return (
        frame
        .sort_values(
            "time",
            key=_mt5_time_sort_key,
            kind="stable",
            na_position="first",
        )
        .tail(count)
        .reset_index(drop=True)
    )


def _load_rate_series_through_legacy(
    conn_or_path: SqliteConnOrPath,
    targets: Sequence[RateTarget] | None,
    count: int | None,
    explicit_tables: Sequence[str] | None,
    *,
    table: str | None,
) -> dict[tuple[str | None, int], pd.DataFrame] | pd.DataFrame:
    """Load explicit-table data through the legacy storage implementation.

    Returns:
        One explicit-table frame or target-keyed legacy rate frames.

    Raises:
        ValueError: If a target-based explicit-table request omits ``count``.
    """
    if table is not None:
        return _legacy_load_rate_series_from_sqlite(
            conn_or_path,
            count=count,
            table=table,
        )
    if targets is None:
        return _legacy_load_rate_series_from_sqlite(
            conn_or_path,
            targets=None,
            count=count,
            explicit_tables=explicit_tables,
        )
    if count is None:
        msg = "count must be positive."
        raise ValueError(msg)
    return _legacy_load_rate_series_from_sqlite(
        conn_or_path,
        targets=targets,
        count=count,
        explicit_tables=explicit_tables,
    )


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
            return _load_rate_series_through_legacy(
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


def _remove_rate_compatibility_views(output: Path | str) -> None:
    """Drop legacy rate views after a canonical history update."""
    path = Path(output)
    if not path.is_file():
        return
    with closing(sqlite3.connect(path)) as conn, conn:
        drop_rate_compatibility_views(conn)


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
        create_rate_views=False,
        with_views=with_views,
        include_account_events=include_account_events,
    )
    _remove_rate_compatibility_views(output)


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
        create_rate_views=False,
        with_views=with_views,
        include_account_events=include_account_events,
    )
    _remove_rate_compatibility_views(output)


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
