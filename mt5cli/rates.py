"""Canonical normalized-rate persistence and loading APIs."""
# ruff: noqa: PLR0913
# pyright: reportPrivateUsage=false

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, overload

import pandas as pd

from .history import (
    _SQLITE_TEXT_TIME_COLUMNS,
    RateTarget,
    _sqlite_normalized_time_expression,
    get_table_columns,
    load_rate_data,
    quote_sqlite_identifier,
    resolve_granularity_name,
)
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
from .schemas import TIME_COLUMNS, DataKind, normalize_time_columns

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from pdmt5.dataframe import Mt5Config

    from .contract import HistoryClient
    from .history import UpdateHistoryBackend
    from .utils import Dataset

SqliteConnOrPath = sqlite3.Connection | Path | str

_RATE_TIMESTAMP_CONTRACT_TABLE = "_mt5cli_metadata"
_RATE_TIMESTAMP_CONTRACT_KEY = "rates_timestamp_contract"
_RATE_TIMESTAMP_CONTRACT_VALUE = "pdmt5-wall-clock-v1"
_MANAGED_TIMESTAMP_DATA_KINDS: tuple[DataKind, ...] = (
    DataKind.rates,
    DataKind.ticks,
    DataKind.history_orders,
    DataKind.history_deals,
    DataKind.symbols,
)
_CURSOR_INDEX_SPECS: tuple[tuple[str, DataKind, tuple[str, ...]], ...] = (
    (
        "idx_rates_symbol_timeframe_history_cursor",
        DataKind.rates,
        ("symbol", "timeframe"),
    ),
    ("idx_ticks_symbol_history_cursor", DataKind.ticks, ("symbol",)),
    (
        "idx_history_orders_symbol_history_cursor",
        DataKind.history_orders,
        ("symbol",),
    ),
    (
        "idx_history_deals_symbol_history_cursor",
        DataKind.history_deals,
        ("symbol",),
    ),
    ("idx_history_deals_history_cursor", DataKind.history_deals, ()),
)


def _read_rate_timestamp_contract(conn: sqlite3.Connection) -> str | None:
    """Return the persisted canonical rate timestamp contract, when present."""
    metadata_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (_RATE_TIMESTAMP_CONTRACT_TABLE,),
    ).fetchone()
    if metadata_table is None:
        return None
    row = conn.execute(
        "SELECT value FROM _mt5cli_metadata WHERE key = ?",
        (_RATE_TIMESTAMP_CONTRACT_KEY,),
    ).fetchone()
    return str(row[0]) if row is not None else None


def _existing_managed_timestamp_tables(
    conn: sqlite3.Connection,
) -> tuple[tuple[DataKind, set[str]], ...]:
    """Return managed history tables that exist in the connection."""
    tables: list[tuple[DataKind, set[str]]] = []
    for kind in _MANAGED_TIMESTAMP_DATA_KINDS:
        columns = get_table_columns(conn, kind.value)
        if columns:
            tables.append((kind, columns))
    return tuple(tables)


def _table_contains_aware_timestamp_text(
    conn: sqlite3.Connection,
    kind: DataKind,
    columns: set[str],
) -> bool:
    """Return whether a managed table has an explicit timezone suffix."""
    table = quote_sqlite_identifier(kind.value)
    time_columns = TIME_COLUMNS[kind] & _SQLITE_TEXT_TIME_COLUMNS & columns
    for column in sorted(time_columns):
        quoted_column = quote_sqlite_identifier(column)
        row = conn.execute(
            f"""
            SELECT 1
            FROM {table}
            WHERE typeof({quoted_column}) = 'text'
              AND (
                substr(trim({quoted_column}), -1) IN ('Z', 'z')
                OR upper(substr(trim({quoted_column}), -3)) IN ('UTC', 'GMT')
                OR instr(substr(trim({quoted_column}), 20), '+') > 0
                OR instr(substr(trim({quoted_column}), 20), '-') > 0
              )
            LIMIT 1
            """  # noqa: S608
        ).fetchone()
        if row is not None:
            return True
    return False


def _managed_history_contains_aware_timestamp_text(
    conn: sqlite3.Connection,
) -> bool:
    """Return whether managed history has explicit timezone suffixes."""
    return any(
        _table_contains_aware_timestamp_text(conn, kind, columns)
        for kind, columns in _existing_managed_timestamp_tables(conn)
    )


def _validate_rate_timestamp_contract(conn: sqlite3.Connection) -> None:
    """Reject legacy or unknown managed history timestamp representations.

    Raises:
        ValueError: If persisted metadata is unsupported or an unversioned managed
            table contains timezone-aware timestamp text with ambiguous semantics.
    """
    if not _existing_managed_timestamp_tables(conn):
        return
    contract = _read_rate_timestamp_contract(conn)
    if contract == _RATE_TIMESTAMP_CONTRACT_VALUE:
        return
    if contract is not None:
        msg = f"Unsupported managed history timestamp contract: {contract!r}."
        raise ValueError(msg)
    if _managed_history_contains_aware_timestamp_text(conn):
        msg = (
            "The managed history database uses an unversioned timezone-aware timestamp "
            "representation that may have been created by mt5cli <= 1.4.1, which "
            "silently relabeled pdmt5 trade-server wall-clock values as UTC. "
            "Recreate or explicitly migrate this history database before "
            "incrementally updating it."
        )
        raise ValueError(msg)


def _validate_existing_rate_database(output: Path | str) -> None:
    """Validate an existing output database before an incremental update."""
    path = Path(output)
    if not path.exists():
        return
    conn, _ = _open_existing_sqlite_database(path)
    try:
        _validate_rate_timestamp_contract(conn)
    finally:
        conn.close()


def _create_incremental_cursor_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes matching normalized incremental timestamp cursors."""
    time_expression = _sqlite_normalized_time_expression("time")
    for index_name, kind, prefix_columns in _CURSOR_INDEX_SPECS:
        columns = get_table_columns(conn, kind.value)
        if not {"time", *prefix_columns}.issubset(columns):
            continue
        index_columns = [
            *(quote_sqlite_identifier(column) for column in prefix_columns),
            time_expression,
        ]
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {quote_sqlite_identifier(index_name)} "
            f"ON {quote_sqlite_identifier(kind.value)} ({', '.join(index_columns)})"
        )


def _mark_rate_timestamp_contract(output: Path | str) -> None:
    """Persist the managed history timestamp contract after a successful write.

    Raises:
        ValueError: If the database already declares an unsupported contract.
    """
    path = Path(output)
    if not path.exists():
        return
    with sqlite3.connect(path) as conn:
        if not _existing_managed_timestamp_tables(conn):
            return
        existing = _read_rate_timestamp_contract(conn)
        if existing not in {None, _RATE_TIMESTAMP_CONTRACT_VALUE}:
            msg = f"Unsupported managed history timestamp contract: {existing!r}."
            raise ValueError(msg)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _mt5cli_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO _mt5cli_metadata(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (_RATE_TIMESTAMP_CONTRACT_KEY, _RATE_TIMESTAMP_CONTRACT_VALUE),
        )
        _create_incremental_cursor_indexes(conn)


def _load_canonical_rate_target(
    conn: sqlite3.Connection,
    target: RateTarget,
    *,
    count: int,
) -> pd.DataFrame:
    """Load one rate series from canonical normalized storage.

    Returns:
        Ascending rate frame indexed by ``time``.

    Raises:
        ValueError: If the target or canonical table schema is invalid.
    """
    if target.symbol is None:
        msg = "A symbol is required for canonical normalized rate loading."
        raise ValueError(msg)
    columns = get_table_columns(conn, "rates")
    if not columns:
        msg = "The canonical rates table is unavailable or invalid."
        raise ValueError(msg)
    if "time" not in columns:
        msg = "The canonical rates table is missing the required time column."
        raise ValueError(msg)
    frame = pd.read_sql_query(  # type: ignore[reportUnknownMemberType]
        "SELECT * FROM rates WHERE symbol = ? AND timeframe = ? "
        "ORDER BY COALESCE("
        "strftime('%Y-%m-%dT%H:%M:%f', time), "
        "strftime('%Y-%m-%dT%H:%M:%f', time, 'unixepoch')"
        ") DESC, ROWID DESC LIMIT ?",
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
    """Load intentionally named custom tables without managed-view discovery.

    Returns:
        A single explicit-table frame or target-keyed frames.

    Raises:
        ValueError: If explicit table inputs are inconsistent.
    """
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
    directly. Explicit custom table loading remains supported without managed
    per-series table discovery.

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
        _validate_rate_timestamp_contract(conn)
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
    """Incrementally update canonical SQLite history."""
    _validate_existing_rate_database(output)
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
    _mark_rate_timestamp_contract(output)


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
    """Update canonical history with a managed MT5 session."""
    _validate_existing_rate_database(output)
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
    _mark_rate_timestamp_contract(output)


class ThrottledHistoryUpdater(_LegacyThrottledHistoryUpdater):
    """Throttle canonical history updates."""

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
