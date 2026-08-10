"""Canonical normalized-rate persistence and loading APIs."""
# ruff: noqa: C901, PLR0913, S608

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .history import (
    RateTarget,
    ThrottledHistoryUpdater as _LegacyThrottledHistoryUpdater,
    drop_rate_compatibility_views,
    load_rate_series_from_sqlite as _legacy_load_rate_series_from_sqlite,
    resolve_granularity_name,
    update_history as _legacy_update_history,
    update_history_with_config as _legacy_update_history_with_config,
)
from .schemas import DataKind, normalize_time_columns

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from pdmt5.dataframe import Mt5Config

    from .contract import HistoryClient
    from .utils import Dataset

SqliteConnOrPath = sqlite3.Connection | Path | str


def _open_existing_database(
    conn_or_path: SqliteConnOrPath,
) -> tuple[sqlite3.Connection, bool]:
    """Open an existing database without creating a missing path.

    Returns:
        Connection and whether this function owns and must close it.

    Raises:
        ValueError: If a supplied database path does not exist.
    """
    if isinstance(conn_or_path, sqlite3.Connection):
        return conn_or_path, False
    path = Path(conn_or_path)
    if not path.is_file():
        msg = f"SQLite database not found: {path}"
        raise ValueError(msg)
    return sqlite3.connect(path), True


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
        "SELECT * FROM rates WHERE symbol = ? AND timeframe = ? "
        "ORDER BY time DESC LIMIT ?",
        conn,
        params=(target.symbol, target.timeframe_int, count),
    )
    if frame.empty:
        return frame
    frame = normalize_time_columns(frame, DataKind.rates)
    return frame.sort_values("time", kind="stable").reset_index(drop=True)


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
        TypeError: If the requested mode yields an unexpected shape.
        ValueError: If targets/count are invalid or canonical storage is absent.
    """
    if table is not None or explicit_tables is not None:
        return _legacy_load_rate_series_from_sqlite(
            conn_or_path,
            targets,
            count,
            explicit_tables,
            table=table,
        )
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

    conn, should_close = _open_existing_database(conn_or_path)
    try:
        return {
            key: _load_canonical_rate_target(conn, target, count=count)
            for key, target in zip(keys, target_list, strict=True)
        }
    except sqlite3.Error as exc:
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

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize with the canonical update backend by default."""
        kwargs.setdefault("update_backend", update_history)
        super().__init__(**kwargs)


__all__ = [
    "ThrottledHistoryUpdater",
    "load_rate_series_by_granularity",
    "load_rate_series_from_sqlite",
    "update_history",
    "update_history_with_config",
]
