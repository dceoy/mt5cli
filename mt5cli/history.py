"""SQLite storage helpers for the ``collect-history`` incremental data pipeline."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd

from .utils import (
    TIMEFRAME_MAP,
    Dataset,
    IfExists,
    parse_datetime,
    parse_tick_flags,
    parse_timeframe,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pdmt5 import Mt5DataClient

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_TIMEFRAMES: tuple[str, ...] = tuple(TIMEFRAME_MAP)

_HISTORY_DEDUP_KEYS: dict[Dataset, tuple[tuple[str, ...], ...]] = {
    Dataset.rates: (("symbol", "timeframe", "time"), ("symbol", "time")),
    Dataset.ticks: (("symbol", "time_msc"), ("symbol", "time")),
    Dataset.history_orders: (("ticket",), ("symbol", "time", "type")),
    Dataset.history_deals: (("ticket",), ("symbol", "time", "type", "entry")),
}

_TRADE_DEAL_TYPES: tuple[int, int] = (0, 1)
_TRADE_DEAL_TYPES_SQL = f"({', '.join(str(value) for value in _TRADE_DEAL_TYPES)})"

_POSITIONS_VIEW_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "position_id",
    "symbol",
    "time",
    "type",
    "entry",
    "volume",
    "price",
    "profit",
})


def quote_sqlite_identifier(identifier: str) -> str:
    """Return a safely quoted SQLite identifier using double quotes."""
    return '"' + identifier.replace('"', '""') + '"'


def resolve_history_datasets(datasets: set[Dataset] | None) -> set[Dataset]:
    """Resolve configured history datasets.

    Returns:
        All supported datasets when ``datasets`` is None, otherwise the
        configured selection (which may be empty).
    """
    if datasets is None:
        return set(Dataset)
    return set(datasets)


def resolve_history_timeframes(
    timeframes: Sequence[int | str] | None,
) -> list[int]:
    """Resolve rate timeframes, deduplicating aliases for the same integer.

    Returns:
        Ordered list of unique timeframe integers.
    """
    raw = timeframes if timeframes is not None else DEFAULT_HISTORY_TIMEFRAMES
    seen: set[int] = set()
    resolved: list[int] = []
    for value in raw:
        tf = value if isinstance(value, int) else parse_timeframe(str(value))
        if tf not in seen:
            seen.add(tf)
            resolved.append(tf)
    return resolved


def resolve_history_tick_flags(flags: int | str) -> int:
    """Resolve tick copy flags from an integer or name.

    Returns:
        Integer tick flag value.
    """
    if isinstance(flags, int):
        return flags
    return parse_tick_flags(flags)


def resolve_granularity_name(timeframe: int) -> str:
    """Return a granularity name for a timeframe integer when known."""
    for name, value in TIMEFRAME_MAP.items():
        if value == timeframe:
            return name
    return str(timeframe)


def build_rate_view_name(
    *,
    symbol: str,
    granularity: str,
    granularity_count: int,
    timeframe: int,
) -> str:
    """Return a collision-free offline optimize view name.

    View names always include the timeframe integer after a ``__`` separator so
    a symbol such as ``EURUSD_M1`` cannot collide with ``EURUSD`` at timeframe
    ``M1``.
    """
    if granularity_count == 1:
        return f"rate_{symbol}__{timeframe}"
    return f"rate_{symbol}__{granularity}_{timeframe}"


SqliteConnOrPath = sqlite3.Connection | Path | str


def _require_non_empty_identifier(identifier: str, kind: str) -> str:
    value = identifier.strip()
    if not value:
        msg = f"SQLite {kind} name must not be empty."
        raise ValueError(msg)
    return value


def _open_history_connection(
    conn_or_path: SqliteConnOrPath | None,
) -> tuple[sqlite3.Connection | None, bool]:
    """Open a read-only SQLite connection when given a path.

    Returns:
        A connection and whether the caller should close it. When ``conn_or_path``
        is None or the path does not exist, returns ``(None, False)`` without
        creating a database file.
    """
    if conn_or_path is None:
        return None, False
    if isinstance(conn_or_path, sqlite3.Connection):
        return conn_or_path, False
    path = Path(conn_or_path)
    if not path.exists():
        return None, False
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    return conn, True


def _open_existing_sqlite_database(
    conn_or_path: SqliteConnOrPath,
) -> tuple[sqlite3.Connection, bool]:
    """Open a read-only SQLite database or reuse an existing connection.

    Returns:
        Tuple of connection and whether the caller should close it.

    Raises:
        ValueError: If the database path does not exist or is not a file.
    """
    if isinstance(conn_or_path, sqlite3.Connection):
        return conn_or_path, False
    path = Path(conn_or_path)
    if not path.exists():
        msg = f"SQLite database not found: {path}"
        raise ValueError(msg)
    if not path.is_file():
        msg = f"SQLite database path is not a file: {path}"
        raise ValueError(msg)
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    return conn, True


def _validate_rate_load_request(table: str, count: int | None) -> str:
    table_name = _require_non_empty_identifier(table, "table or view")
    if count is not None and count <= 0:
        msg = "count must be positive when provided."
        raise ValueError(msg)
    return table_name


def _ensure_rate_columns(columns: set[str], table: str) -> None:
    if not columns:
        msg = f"SQLite table or view not found: {table}"
        raise ValueError(msg)
    if "time" not in columns:
        msg = f"SQLite table or view {table!r} must include a time column."
        raise ValueError(msg)
    if "close" not in columns and not {"ask", "bid"}.issubset(columns):
        msg = (
            f"SQLite table or view {table!r} must include close, "
            "or both ask and bid columns."
        )
        raise ValueError(msg)


def _parse_rate_time_index(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    parsed = frame["time"].map(parse_sqlite_timestamp)
    if parsed.isna().any():
        msg = f"SQLite table or view {table!r} contains unparsable time values."
        raise ValueError(msg)
    result = frame.drop(columns=["time"])
    result.index = pd.DatetimeIndex(parsed, name="time")
    return result.sort_index(kind="stable")


def load_rate_data_from_connection(
    connection: sqlite3.Connection,
    table: str,
    count: int | None = None,
) -> pd.DataFrame:
    """Load rate-like data from a SQLite table or view.

    Args:
        connection: Open SQLite connection.
        table: Source table or view name.
        count: Optional number of most recent rows to load.

    Returns:
        DataFrame indexed by ascending ``time``.

    Raises:
        ValueError: If inputs, schema, timestamps are invalid, or the table
            or view contains no rows.
    """
    table_name = _validate_rate_load_request(table, count)
    columns = get_table_columns(connection, table_name)
    _ensure_rate_columns(columns, table_name)
    quoted_table = quote_sqlite_identifier(table_name)
    if count is None:
        frame = cast(
            "pd.DataFrame",
            pd.read_sql_query(  # type: ignore[reportUnknownMemberType]
                f"SELECT * FROM {quoted_table} ORDER BY time ASC",  # noqa: S608
                connection,
            ),
        )
    else:
        frame = cast(
            "pd.DataFrame",
            pd.read_sql_query(  # type: ignore[reportUnknownMemberType]
                f"SELECT * FROM {quoted_table} ORDER BY time DESC LIMIT ?",  # noqa: S608
                connection,
                params=(count,),
            ),
        )
    if frame.empty:
        msg = f"SQLite table or view {table_name!r} contains no rows."
        raise ValueError(msg)
    return _parse_rate_time_index(frame, table_name)


def load_rate_data(
    conn_or_path: SqliteConnOrPath,
    table: str,
    count: int | None = None,
) -> pd.DataFrame:
    """Load rate-like data from a SQLite database path or connection.

    Args:
        conn_or_path: SQLite database path or open connection.
        table: Source table or view name.
        count: Optional number of most recent rows to load.

    Returns:
        DataFrame indexed by ascending ``time``.

    """
    conn, should_close = _open_existing_sqlite_database(conn_or_path)
    try:
        return load_rate_data_from_connection(conn, table, count=count)
    finally:
        if should_close:
            conn.close()


def _load_rates_timeframe_counts(conn: sqlite3.Connection) -> dict[str, int] | None:
    """Return distinct timeframe counts per symbol from the normalized rates table."""
    columns = get_table_columns(conn, Dataset.rates.table_name)
    if not {"symbol", "timeframe"}.issubset(columns):
        return None
    rows = conn.execute(
        "SELECT symbol, COUNT(DISTINCT timeframe) FROM rates GROUP BY symbol",
    ).fetchall()
    return {str(symbol): int(count) for symbol, count in rows}


def _load_existing_rate_views(conn: sqlite3.Connection) -> set[str]:
    """Return mt5cli-managed ``rate_*__*`` compatibility view names."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'view' AND name GLOB 'rate_*__*'",
    ).fetchall()
    return {str(row[0]) for row in rows}


def _rate_view_name_candidates(
    *,
    symbol: str,
    granularity: str,
    granularity_count: int,
    timeframe: int,
) -> list[str]:
    """Return candidate view names in preference order."""
    single = build_rate_view_name(
        symbol=symbol,
        granularity=granularity,
        granularity_count=1,
        timeframe=timeframe,
    )
    if granularity_count <= 1:
        return [single]
    multi = build_rate_view_name(
        symbol=symbol,
        granularity=granularity,
        granularity_count=granularity_count,
        timeframe=timeframe,
    )
    return [multi, single]


def _resolve_rate_view_name_from_context(
    *,
    symbol: str,
    timeframe: int,
    granularity_name: str,
    timeframe_counts: dict[str, int] | None,
    existing_views: set[str],
    require_existing: bool = False,
) -> str:
    """Resolve one rate view name using preloaded SQLite metadata.

    Returns:
        Preferred mt5cli-managed rate compatibility view name.

    Raises:
        ValueError: If ``require_existing`` is True and no managed view exists.
    """
    if timeframe_counts is None or symbol not in timeframe_counts:
        candidates = [
            build_rate_view_name(
                symbol=symbol,
                granularity=granularity_name,
                granularity_count=1,
                timeframe=timeframe,
            ),
            build_rate_view_name(
                symbol=symbol,
                granularity=granularity_name,
                granularity_count=2,
                timeframe=timeframe,
            ),
        ]
    else:
        candidates = _rate_view_name_candidates(
            symbol=symbol,
            granularity=granularity_name,
            granularity_count=timeframe_counts[symbol],
            timeframe=timeframe,
        )
    for candidate in candidates:
        if candidate in existing_views:
            return candidate
    if require_existing:
        msg = (
            f"No rate compatibility view exists for symbol {symbol!r} "
            f"and granularity {granularity_name!r}; "
            f"candidates: {', '.join(candidates)}."
        )
        raise ValueError(msg)
    return candidates[0]


def resolve_rate_view_name(
    conn_or_path: SqliteConnOrPath | None,
    symbol: str,
    granularity: str,
    *,
    require_existing: bool = False,
) -> str:
    """Resolve the mt5cli-managed rate compatibility view name.

    Args:
        conn_or_path: SQLite database path or open connection. When None or a
            non-existing path and ``require_existing`` is False, the deterministic
            default view name is returned without creating a database file.
        symbol: Symbol stored in the normalized ``rates`` table.
        granularity: Timeframe name (for example ``M1``) or integer string.
        require_existing: When True, require the database and a managed view to exist.

    Returns:
        View name such as ``rate_EURUSD__1`` or ``rate_EURUSD__M1_1``.

    Raises:
        ValueError: If ``require_existing`` is True and the database or view is missing.
    """
    timeframe = parse_timeframe(granularity)
    granularity_name = resolve_granularity_name(timeframe)
    conn, should_close = _open_history_connection(conn_or_path)
    try:
        if conn is None:
            if require_existing:
                path = (
                    conn_or_path
                    if isinstance(conn_or_path, (Path, str))
                    else "database"
                )
                msg = f"SQLite database not found: {path}"
                raise ValueError(msg)
            return build_rate_view_name(
                symbol=symbol,
                granularity=granularity_name,
                granularity_count=1,
                timeframe=timeframe,
            )
        return _resolve_rate_view_name_from_context(
            symbol=symbol,
            timeframe=timeframe,
            granularity_name=granularity_name,
            timeframe_counts=_load_rates_timeframe_counts(conn),
            existing_views=_load_existing_rate_views(conn),
            require_existing=require_existing,
        )
    finally:
        if should_close and conn is not None:
            conn.close()


def resolve_rate_view_names(
    conn_or_path: SqliteConnOrPath | None,
    symbols: Sequence[str],
    granularities: Sequence[str],
    *,
    require_existing: bool = False,
) -> list[str]:
    """Resolve rate compatibility view names for symbol and granularity pairs.

    Args:
        conn_or_path: SQLite database path or open connection. When None or a
            non-existing path and ``require_existing`` is False, deterministic
            default view names are returned without creating a database file.
        symbols: Symbols stored in the normalized ``rates`` table.
        granularities: Timeframe names (for example ``M1``) or integer strings.
        require_existing: When True, require the database and managed views to exist.

    Returns:
        View names in row-major order: every ``granularity`` for the first
        symbol, then every granularity for the next symbol, and so on.
    """
    conn, should_close = _open_history_connection(conn_or_path)
    try:
        if conn is None:
            return [
                resolve_rate_view_name(
                    conn_or_path,
                    symbol,
                    granularity,
                    require_existing=require_existing,
                )
                for symbol in symbols
                for granularity in granularities
            ]
        timeframe_counts = _load_rates_timeframe_counts(conn)
        existing_views = _load_existing_rate_views(conn)
        resolved: list[str] = []
        for symbol in symbols:
            for granularity in granularities:
                timeframe = parse_timeframe(granularity)
                resolved.append(
                    _resolve_rate_view_name_from_context(
                        symbol=symbol,
                        timeframe=timeframe,
                        granularity_name=resolve_granularity_name(timeframe),
                        timeframe_counts=timeframe_counts,
                        existing_views=existing_views,
                        require_existing=require_existing,
                    ),
                )
        return resolved
    finally:
        if should_close and conn is not None:
            conn.close()


@dataclass(frozen=True)
class RateTarget:
    """A single rate series identified by symbol and timeframe.

    Attributes:
        symbol: MT5 symbol name, or None when the rate series is addressed only
            by an explicit table (for example a custom SQLite view).
        timeframe: MT5 timeframe as an integer or name (for example ``M1``).
    """

    symbol: str | None
    timeframe: int | str

    def __post_init__(self) -> None:
        """Normalize accepted timeframe aliases to the stored integer value."""
        if not isinstance(self.timeframe, int):
            object.__setattr__(self, "timeframe", parse_timeframe(self.timeframe))

    @property
    def timeframe_int(self) -> int:
        """Return the timeframe as its integer MT5 value."""
        if isinstance(self.timeframe, int):
            return self.timeframe
        return parse_timeframe(self.timeframe)


def build_rate_targets(
    symbols: Sequence[str],
    timeframes: Sequence[int | str],
    *,
    allow_missing_symbol: bool = False,
) -> list[RateTarget]:
    """Build rate targets for every symbol and timeframe combination.

    Args:
        symbols: MT5 symbol names. May be empty when ``allow_missing_symbol``.
        timeframes: MT5 timeframes as integers or names (for example ``M1``).
        allow_missing_symbol: When True and ``symbols`` is empty, build targets
            with ``symbol=None`` for each timeframe instead of raising.

    Returns:
        Targets in row-major order: every timeframe for the first symbol, then
        every timeframe for the next symbol, and so on.

    Raises:
        ValueError: If ``timeframes`` is empty, or ``symbols`` is empty and
            ``allow_missing_symbol`` is False.
    """
    if not timeframes:
        msg = "At least one timeframe is required."
        raise ValueError(msg)
    if not symbols:
        if not allow_missing_symbol:
            msg = "At least one symbol is required."
            raise ValueError(msg)
        return [RateTarget(symbol=None, timeframe=tf) for tf in timeframes]
    return [
        RateTarget(symbol=symbol, timeframe=tf)
        for symbol in symbols
        for tf in timeframes
    ]


def resolve_rate_tables(
    conn_or_path: SqliteConnOrPath | None,
    targets: Sequence[RateTarget],
    explicit_tables: Sequence[str] | None = None,
) -> list[str]:
    """Resolve SQLite table or view names for rate targets.

    Args:
        conn_or_path: SQLite database path or open connection. May be None when
            ``explicit_tables`` is provided.
        targets: Rate targets to resolve.
        explicit_tables: Optional explicit table or view names. When provided,
            they are used as-is and must match the number of targets.

    Returns:
        Table or view names aligned with ``targets``.

    Raises:
        ValueError: If ``targets`` is empty, ``explicit_tables`` length does not
            match the target count, or a target without a symbol is resolved
            without an explicit table.
    """
    target_list = list(targets)
    if not target_list:
        msg = "At least one rate target is required."
        raise ValueError(msg)
    if explicit_tables is not None:
        tables = list(explicit_tables)
        if len(tables) != len(target_list):
            msg = (
                f"Expected {len(target_list)} explicit table(s) "
                f"to match the targets, got {len(tables)}."
            )
            raise ValueError(msg)
        return tables
    if any(target.symbol is None for target in target_list):
        msg = (
            "Cannot resolve a rate table for a target without a symbol; "
            "provide explicit_tables."
        )
        raise ValueError(msg)
    conn, should_close = _open_history_connection(conn_or_path)
    try:
        timeframe_counts = (
            _load_rates_timeframe_counts(conn) if conn is not None else None
        )
        existing_views = _load_existing_rate_views(conn) if conn is not None else set()
        resolved: list[str] = []
        for target in target_list:
            symbol = target.symbol
            if symbol is None:
                msg = (
                    "Cannot resolve a rate table for a target without a symbol; "
                    "provide explicit_tables."
                )
                raise ValueError(msg)
            timeframe = target.timeframe_int
            resolved.append(
                _resolve_rate_view_name_from_context(
                    symbol=symbol,
                    timeframe=timeframe,
                    granularity_name=resolve_granularity_name(timeframe),
                    timeframe_counts=timeframe_counts,
                    existing_views=existing_views,
                ),
            )
        return resolved
    finally:
        if should_close and conn is not None:
            conn.close()


def load_rate_series_from_sqlite(
    conn_or_path: SqliteConnOrPath,
    targets: Sequence[RateTarget],
    count: int,
    explicit_tables: Sequence[str] | None = None,
) -> dict[tuple[str | None, int], pd.DataFrame]:
    """Load multiple rate series from a SQLite database.

    Args:
        conn_or_path: SQLite database path or open connection.
        targets: Rate targets to load.
        count: Number of most recent rows to load per series.
        explicit_tables: Optional explicit table or view names matching targets.

    Returns:
        Mapping keyed by ``(symbol, timeframe_int)`` to each rate DataFrame.

    Raises:
        ValueError: If ``count`` is not positive, targets are empty, or table
            resolution fails.
    """
    if count <= 0:
        msg = "count must be positive."
        raise ValueError(msg)
    target_list = list(targets)
    if not target_list:
        msg = "At least one rate target is required."
        raise ValueError(msg)
    if explicit_tables is None and any(target.symbol is None for target in target_list):
        msg = (
            "Cannot resolve a rate table for a target without a symbol; "
            "provide explicit_tables."
        )
        raise ValueError(msg)
    tables = (
        resolve_rate_tables(None, target_list, explicit_tables)
        if explicit_tables is not None
        else None
    )
    conn, should_close = _open_existing_sqlite_database(conn_or_path)
    try:
        resolved_tables = tables or resolve_rate_tables(conn, target_list)
        return {
            (target.symbol, target.timeframe_int): load_rate_data_from_connection(
                conn,
                table,
                count=count,
            )
            for target, table in zip(target_list, resolved_tables, strict=True)
        }
    finally:
        if should_close:
            conn.close()


def get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return existing SQLite columns for a table."""
    quoted_table = quote_sqlite_identifier(table)
    rows = conn.execute(f"PRAGMA table_info({quoted_table})").fetchall()
    return {str(row[1]) for row in rows}


def _parse_string_sqlite_timestamp(value: str) -> datetime | None:
    try:
        return parse_datetime(value)
    except ValueError:
        parsed_ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed_ts):
            logger.warning("Ignoring unparseable history timestamp: %s", value)
            return None
        return parsed_ts.to_pydatetime()


def parse_sqlite_timestamp(value: object) -> datetime | None:
    """Parse a SQLite history timestamp value.

    Returns:
        Parsed timezone-aware datetime, or None when parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        return _parse_string_sqlite_timestamp(value)
    logger.warning("Ignoring unsupported history timestamp type: %s", type(value))
    return None


def get_history_deals_account_event_start_datetime(
    conn: sqlite3.Connection,
    *,
    fallback_start: datetime,
) -> datetime:
    """Return the next update start for account-level history_deals rows."""
    table = Dataset.history_deals.table_name
    columns = get_table_columns(conn, table)
    if "time" not in columns:
        return fallback_start
    if "type" in columns:
        where_clause = f"type NOT IN {_TRADE_DEAL_TYPES_SQL}"
    elif "symbol" in columns:
        where_clause = "symbol IS NULL OR symbol = ''"
    else:
        return fallback_start
    row = conn.execute(
        f"SELECT MAX(time) FROM {table} WHERE {where_clause}",  # noqa: S608
    ).fetchone()
    parsed = parse_sqlite_timestamp(row[0] if row else None)
    return parsed if parsed is not None else fallback_start


_REQUIRED_RATE_COLUMNS = frozenset({"symbol", "timeframe", "time"})


def _validate_rates_schema(columns: set[str]) -> None:
    """Validate an existing rates table has normalized incremental columns.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = _REQUIRED_RATE_COLUMNS - columns
    if missing:
        msg = (
            "The rates table must include symbol, timeframe, and time columns"
            " for incremental updates; "
            f"missing: {', '.join(sorted(missing))}."
        )
        raise ValueError(msg)


def load_incremental_start_datetimes(
    conn: sqlite3.Connection,
    dataset: Dataset,
    *,
    symbols: Sequence[str],
    timeframes: Sequence[int] | None = None,
    fallback_start: datetime,
) -> dict[tuple[str, int | None], datetime]:
    """Return next update start datetimes keyed by symbol and optional timeframe."""
    table = dataset.table_name
    columns = get_table_columns(conn, table)
    if dataset is Dataset.rates and columns:
        _validate_rates_schema(columns)

    if "time" not in columns:
        if dataset is Dataset.rates and timeframes is not None:
            return {
                (symbol, timeframe): fallback_start
                for symbol in symbols
                for timeframe in timeframes
            }
        return {(symbol, None): fallback_start for symbol in symbols}

    parsed_by_key: dict[tuple[str, int | None], datetime] = {}
    if (
        dataset is Dataset.rates
        and timeframes is not None
        and {"symbol", "timeframe"}.issubset(columns)
    ):
        symbol_placeholders = ", ".join("?" for _ in symbols)
        timeframe_placeholders = ", ".join("?" for _ in timeframes)
        grouped_rates_query = (
            "SELECT symbol, timeframe, MAX(time) FROM "  # noqa: S608
            f"{table} WHERE symbol IN ({symbol_placeholders})"
            f" AND timeframe IN ({timeframe_placeholders})"
            " GROUP BY symbol, timeframe"
        )
        rows = conn.execute(
            grouped_rates_query,
            [*symbols, *timeframes],
        ).fetchall()
        for row_symbol, row_timeframe, max_time in rows:
            parsed = parse_sqlite_timestamp(max_time)
            if parsed is not None:
                parsed_by_key[str(row_symbol), int(row_timeframe)] = parsed
        return {
            (symbol, timeframe): parsed_by_key.get(
                (symbol, timeframe),
                fallback_start,
            )
            for symbol in symbols
            for timeframe in timeframes
        }

    if "symbol" in columns:
        symbol_placeholders = ", ".join("?" for _ in symbols)
        rows = conn.execute(
            f"SELECT symbol, MAX(time) FROM {table}"  # noqa: S608
            f" WHERE symbol IN ({symbol_placeholders}) GROUP BY symbol",
            list(symbols),
        ).fetchall()
        for row_symbol, max_time in rows:
            parsed = parse_sqlite_timestamp(max_time)
            if parsed is not None:
                parsed_by_key[str(row_symbol), None] = parsed
        return {
            (symbol, None): parsed_by_key.get((symbol, None), fallback_start)
            for symbol in symbols
        }

    row = conn.execute(f"SELECT MAX(time) FROM {table}").fetchone()  # noqa: S608
    parsed = parse_sqlite_timestamp(row[0] if row else None)
    shared_start = parsed if parsed is not None else fallback_start
    return {(symbol, None): shared_start for symbol in symbols}


def get_incremental_start_datetime(
    conn: sqlite3.Connection,
    dataset: Dataset,
    *,
    symbol: str,
    timeframe: int | None,
    fallback_start: datetime,
) -> datetime:
    """Return the next update start datetime from existing MAX(time)."""
    timeframes = [timeframe] if timeframe is not None else None
    starts = load_incremental_start_datetimes(
        conn,
        dataset,
        symbols=[symbol],
        timeframes=timeframes,
        fallback_start=fallback_start,
    )
    return starts[symbol, timeframe]


def append_dataframe(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    table_name: str,
    if_exists: IfExists,
) -> bool:
    """Append a DataFrame to SQLite when it has a schema.

    Returns:
        True if a table was written, False if the frame had no columns.
    """
    if len(frame.columns) == 0:
        logger.warning("Skipping %s: dataset returned no columns", table_name)
        return False
    frame.to_sql(  # type: ignore[reportUnknownMemberType]
        table_name,
        conn,
        if_exists=if_exists.value,
        index=False,
        chunksize=50_000,
    )
    return True


def record_written_columns(
    written_columns: dict[Dataset, set[str]],
    dataset: Dataset,
    frame: pd.DataFrame,
) -> None:
    """Remember columns for datasets written during collection."""
    columns = set(frame.columns)
    if dataset in written_columns:
        written_columns[dataset].update(columns)
    else:
        written_columns[dataset] = columns


def augment_written_columns_from_sqlite(
    conn: sqlite3.Connection,
    datasets: set[Dataset],
    written_columns: dict[Dataset, set[str]],
) -> None:
    """Add existing table columns to the written column map."""
    for dataset in datasets:
        columns = get_table_columns(conn, dataset.table_name)
        if not columns:
            continue
        if dataset in written_columns:
            written_columns[dataset].update(columns)
        else:
            written_columns[dataset] = columns


def write_streamed_frame(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    dataset: Dataset,
    table_exists: bool,
    if_exists: IfExists,
    written_columns: dict[Dataset, set[str]],
) -> bool:
    """Write one streamed dataset frame and track table state.

    Returns:
        True if the dataset table exists after this write attempt.
    """
    write_mode = IfExists.APPEND if table_exists else if_exists
    if append_dataframe(conn, frame, dataset.table_name, write_mode):
        record_written_columns(written_columns, dataset, frame)
        return True
    return table_exists


def drop_duplicates_in_table(
    cursor: sqlite3.Cursor,
    table: str,
    ids: list[str],
    *,
    keep: Literal["first", "last"] = "last",
    scope_where: str | None = None,
    scope_params: tuple[object, ...] = (),
) -> None:
    """Remove duplicate rows, keeping the first or last ROWID per key group.

    Raises:
        ValueError: If the table or column names are invalid.
    """
    if not table.isidentifier():
        msg = f"Invalid table name: {table}"
        raise ValueError(msg)
    if invalid := {column for column in ids if not column.isidentifier()}:
        msg = f"Invalid column names: {', '.join(sorted(invalid))}"
        raise ValueError(msg)
    ids_csv = ", ".join(f'"{column}"' for column in ids)
    rowid_selector = "MIN" if keep == "first" else "MAX"
    if scope_where:
        delete_sql = (
            f"DELETE FROM {table} WHERE {scope_where} AND ROWID NOT IN"  # noqa: S608
            f" (SELECT {rowid_selector}(ROWID) FROM {table} WHERE {scope_where}"
            f" GROUP BY {ids_csv})"
        )
        cursor.execute(delete_sql, scope_params + scope_params)
        return
    cursor.execute(
        f"DELETE FROM {table} WHERE ROWID NOT IN"  # noqa: S608
        f" (SELECT {rowid_selector}(ROWID) FROM {table} GROUP BY {ids_csv})",
    )


DedupScope = tuple[str, tuple[object, ...]]


def _record_dedup_scope(
    dedup_scopes: dict[Dataset, list[DedupScope]],
    dataset: Dataset,
    scope_where: str,
    scope_params: tuple[object, ...],
) -> None:
    dedup_scopes.setdefault(dataset, []).append((scope_where, scope_params))


def deduplicate_history_tables(
    conn: sqlite3.Connection,
    written_columns: dict[Dataset, set[str]],
    written_tables: set[Dataset],
    dedup_scopes: dict[Dataset, list[DedupScope]] | None = None,
) -> None:
    """Deduplicate appended history tables by stable identifiers."""
    cursor = conn.cursor()
    for dataset in written_tables:
        columns = written_columns.get(dataset, set())
        table = dataset.table_name
        keys = next(
            (
                candidate
                for candidate in _HISTORY_DEDUP_KEYS[dataset]
                if set(candidate).issubset(columns)
            ),
            None,
        )
        if keys is None:
            logger.warning(
                "Skipping %s deduplication: no supported key columns",
                table,
            )
            continue
        scopes = dedup_scopes.get(dataset, []) if dedup_scopes else []
        if scopes:
            for scope_where, scope_params in scopes:
                drop_duplicates_in_table(
                    cursor,
                    table,
                    list(keys),
                    keep="last",
                    scope_where=scope_where,
                    scope_params=scope_params,
                )
            continue
        drop_duplicates_in_table(cursor, table, list(keys), keep="last")


def create_history_indexes(
    conn: sqlite3.Connection,
    written_columns: dict[Dataset, set[str]],
) -> None:
    """Create useful indexes for collected history tables when present."""
    if {"symbol", "timeframe", "time"}.issubset(
        written_columns.get(Dataset.rates, set()),
    ):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rates_symbol_timeframe_time"
            " ON rates(symbol, timeframe, time)",
        )
    if {"symbol", "time"}.issubset(written_columns.get(Dataset.ticks, set())):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, time)",
        )
    if {"position_id", "symbol"}.issubset(
        written_columns.get(Dataset.history_deals, set()),
    ):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_deals_position_symbol"
            " ON history_deals(position_id, symbol)",
        )


def _history_deals_account_event_mask(frame: pd.DataFrame) -> pd.Series:
    if "type" in frame.columns:
        return ~frame["type"].isin(_TRADE_DEAL_TYPES)
    if "symbol" in frame.columns:
        return frame["symbol"].isna() | frame["symbol"].eq("")
    return pd.Series(data=False, index=frame.index)


def _frame_parsed_times(frame: pd.DataFrame) -> pd.Series:
    if "time" not in frame.columns:
        return pd.Series([pd.NaT] * len(frame), index=frame.index, dtype="object")
    return frame["time"].map(parse_sqlite_timestamp)


def filter_incremental_history_deals_frame(
    frame: pd.DataFrame,
    symbols: Sequence[str],
    start_by_symbol: dict[str, datetime],
    account_event_start: datetime,
) -> pd.DataFrame:
    """Filter incrementally fetched history_deals by symbol and event start times.

    Returns:
        Rows for selected symbols at or after each symbol start, plus account
        events at or after ``account_event_start``.
    """
    if frame.empty:
        return frame.copy()
    parsed_times = _frame_parsed_times(frame)
    time_valid = parsed_times.notna()
    account_event_mask = _history_deals_account_event_mask(frame)
    account_keep = account_event_mask & (parsed_times >= account_event_start)
    trade_keep = pd.Series(data=False, index=frame.index)
    if "symbol" in frame.columns:
        for symbol in symbols:
            trade_keep |= (
                (frame["symbol"] == symbol)
                & (parsed_times >= start_by_symbol[symbol])
                & ~account_event_mask
            )
    keep = (account_keep | trade_keep) & time_valid
    return frame.loc[keep].copy()


def filter_trade_history_frame(
    frame: pd.DataFrame,
    symbols: Sequence[str],
    *,
    include_account_events: bool,
) -> pd.DataFrame:
    """Filter trade history rows to selected symbols and account events.

    Returns:
        Filtered history rows.
    """
    if "symbol" not in frame.columns:
        return frame
    symbol_mask = frame["symbol"].isin(symbols)
    if not include_account_events:
        return frame.loc[symbol_mask].copy()
    account_event_mask = _history_deals_account_event_mask(frame)
    return frame.loc[symbol_mask | account_event_mask].copy()


def create_cash_events_view(
    conn: sqlite3.Connection,
    deals_columns: set[str],
) -> bool:
    """Create the cash_events SQLite view derived from history_deals.

    Returns:
        True if the view was created, False if required columns are missing.
    """
    if "type" not in deals_columns:
        logger.warning("Skipping cash_events view: history_deals.type is missing")
        return False
    conn.execute("DROP VIEW IF EXISTS cash_events")
    conn.execute(
        "CREATE VIEW cash_events AS"  # noqa: S608
        f" SELECT * FROM history_deals WHERE type NOT IN {_TRADE_DEAL_TYPES_SQL}",
    )
    return True


def create_positions_reconstructed_view(
    conn: sqlite3.Connection,
    deals_columns: set[str],
) -> bool:
    """Create the positions_reconstructed SQLite view derived from history_deals.

    Returns:
        True if the view was created, False if required columns are missing.
    """
    if not _POSITIONS_VIEW_REQUIRED_COLUMNS.issubset(deals_columns):
        missing = ", ".join(sorted(_POSITIONS_VIEW_REQUIRED_COLUMNS - deals_columns))
        logger.warning(
            "Skipping positions_reconstructed view: history_deals missing columns: %s",
            missing,
        )
        return False
    conn.execute("DROP VIEW IF EXISTS positions_reconstructed")
    conn.execute(
        "CREATE VIEW positions_reconstructed AS"  # noqa: S608
        " SELECT"
        " position_id,"
        " symbol,"
        " MIN(CASE WHEN entry = 0 THEN time END) AS open_time,"
        " MAX(CASE WHEN entry IN (1, 2, 3) THEN time END) AS close_time,"
        " MIN(CASE WHEN entry = 0 THEN type END) AS direction,"
        " SUM(CASE WHEN entry = 0 THEN volume ELSE 0 END) AS volume_open,"
        " SUM(CASE WHEN entry IN (1, 2, 3) THEN volume ELSE 0 END) AS volume_close,"
        " SUM(CASE WHEN entry = 2 THEN volume ELSE 0 END) AS volume_reversal,"
        " CASE"
        " WHEN SUM(CASE WHEN entry = 0 THEN volume ELSE 0 END) > 0"
        " THEN SUM(CASE WHEN entry = 0 THEN price * volume ELSE 0 END)"
        " / SUM(CASE WHEN entry = 0 THEN volume ELSE 0 END)"
        " END AS open_price,"
        " CASE"
        " WHEN SUM(CASE WHEN entry IN (1, 2, 3) THEN volume ELSE 0 END) > 0"
        " THEN SUM(CASE WHEN entry IN (1, 2, 3) THEN price * volume ELSE 0 END)"
        " / SUM(CASE WHEN entry IN (1, 2, 3) THEN volume ELSE 0 END)"
        " END AS close_price,"
        " SUM(profit) AS total_profit,"
        " SUM(CASE WHEN entry = 2 THEN 1 ELSE 0 END) AS reversal_count,"
        " COUNT(*) AS deals_count"
        " FROM history_deals"
        f" WHERE type IN {_TRADE_DEAL_TYPES_SQL} AND position_id != 0"
        " GROUP BY position_id, symbol"
        " HAVING SUM(CASE WHEN entry IN (1, 2, 3) THEN 1 ELSE 0 END) > 0",
    )
    return True


def drop_rate_compatibility_views(conn: sqlite3.Connection) -> None:
    """Drop all mt5cli-managed ``rate_*`` compatibility views."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'view' AND name GLOB 'rate_*'",
    ).fetchall()
    for (view_name,) in rows:
        quoted_view_name = quote_sqlite_identifier(str(view_name))
        conn.execute(f"DROP VIEW IF EXISTS {quoted_view_name}")


def create_rate_compatibility_views(conn: sqlite3.Connection) -> None:
    """Create rate compatibility views from the normalized rates table."""
    columns = get_table_columns(conn, Dataset.rates.table_name)
    if not {"symbol", "timeframe", "time"}.issubset(columns):
        return
    drop_rate_compatibility_views(conn)
    select_columns = sorted(columns - {"symbol", "timeframe"})
    quoted_columns = ", ".join(f'"{column}"' for column in select_columns)
    rows = conn.execute(
        "SELECT DISTINCT symbol, timeframe FROM rates ORDER BY symbol, timeframe",
    ).fetchall()
    timeframes_by_symbol: dict[str, list[int]] = {}
    for symbol, timeframe in rows:
        timeframes_by_symbol.setdefault(str(symbol), []).append(int(timeframe))
    for symbol, timeframes in timeframes_by_symbol.items():
        for timeframe in timeframes:
            granularity = resolve_granularity_name(timeframe)
            view_name = build_rate_view_name(
                symbol=symbol,
                granularity=granularity,
                granularity_count=len(timeframes),
                timeframe=timeframe,
            )
            quoted_view_name = quote_sqlite_identifier(view_name)
            escaped_symbol = symbol.replace("'", "''")
            conn.execute(
                f"CREATE VIEW {quoted_view_name} AS"  # noqa: S608
                f" SELECT {quoted_columns} FROM rates"
                f" WHERE symbol = '{escaped_symbol}'"
                f" AND timeframe = {timeframe}",
            )


def write_rates_dataset(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    timeframe: int,
    date_from: datetime,
    date_to: datetime,
    if_exists: IfExists,
    written_columns: dict[Dataset, set[str]],
) -> bool:
    """Stream rates frames into SQLite.

    Returns:
        True if the rates table was written.
    """
    table_exists = False
    for sym in symbols:
        frame = client.copy_rates_range_as_df(
            symbol=sym,
            timeframe=timeframe,
            date_from=date_from,
            date_to=date_to,
        ).drop(columns=["symbol", "timeframe"], errors="ignore")
        if len(frame.columns) != 0:
            frame.insert(0, "symbol", sym)
            frame.insert(1, "timeframe", timeframe)
        table_exists = write_streamed_frame(
            conn,
            frame,
            Dataset.rates,
            table_exists,
            if_exists,
            written_columns,
        )
    return table_exists


def write_ticks_dataset(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    flags: int,
    date_from: datetime,
    date_to: datetime,
    if_exists: IfExists,
    written_columns: dict[Dataset, set[str]],
) -> bool:
    """Stream ticks frames into SQLite.

    Returns:
        True if the ticks table was written.
    """
    table_exists = False
    for sym in symbols:
        frame = client.copy_ticks_range_as_df(
            symbol=sym,
            date_from=date_from,
            date_to=date_to,
            flags=flags,
        ).drop(columns=["symbol"], errors="ignore")
        if len(frame.columns) != 0:
            frame.insert(0, "symbol", sym)
        table_exists = write_streamed_frame(
            conn,
            frame,
            Dataset.ticks,
            table_exists,
            if_exists,
            written_columns,
        )
    return table_exists


def write_history_dataset(
    conn: sqlite3.Connection,
    fetch: Callable[..., pd.DataFrame],
    dataset: Dataset,
    symbols: Sequence[str],
    date_from: datetime,
    date_to: datetime,
    if_exists: IfExists,
    written_columns: dict[Dataset, set[str]],
    *,
    include_account_events: bool = False,
) -> bool:
    """Stream a history dataset into SQLite.

    Returns:
        True if the target table was written.
    """
    table_exists = False
    if include_account_events:
        frame = filter_trade_history_frame(
            fetch(date_from=date_from, date_to=date_to),
            symbols,
            include_account_events=True,
        )
        return write_streamed_frame(
            conn,
            frame,
            dataset,
            table_exists,
            if_exists,
            written_columns,
        )
    for sym in symbols:
        frame = fetch(date_from=date_from, date_to=date_to, symbol=sym)
        frame = filter_trade_history_frame(
            frame,
            [sym],
            include_account_events=False,
        )
        table_exists = write_streamed_frame(
            conn,
            frame,
            dataset,
            table_exists,
            if_exists,
            written_columns,
        )
    return table_exists


def _write_incremental_rates(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    resolved_timeframes: list[int],
    fallback_start: datetime,
    end_date: datetime,
    written_columns: dict[Dataset, set[str]],
    written_tables: set[Dataset],
    dedup_scopes: dict[Dataset, list[DedupScope]],
) -> None:
    start_by_key = load_incremental_start_datetimes(
        conn,
        Dataset.rates,
        symbols=symbols,
        timeframes=resolved_timeframes,
        fallback_start=fallback_start,
    )
    for symbol in symbols:
        for timeframe in resolved_timeframes:
            start_date = start_by_key[symbol, timeframe]
            if write_rates_dataset(
                conn,
                client,
                [symbol],
                timeframe,
                start_date,
                end_date,
                IfExists.APPEND,
                written_columns,
            ):
                written_tables.add(Dataset.rates)
                _record_dedup_scope(
                    dedup_scopes,
                    Dataset.rates,
                    "symbol = ? AND timeframe = ? AND time >= ?",
                    (symbol, timeframe, start_date),
                )


def _write_incremental_ticks(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    resolved_tick_flags: int,
    fallback_start: datetime,
    end_date: datetime,
    written_columns: dict[Dataset, set[str]],
    written_tables: set[Dataset],
    dedup_scopes: dict[Dataset, list[DedupScope]],
) -> None:
    start_by_symbol = load_incremental_start_datetimes(
        conn,
        Dataset.ticks,
        symbols=symbols,
        fallback_start=fallback_start,
    )
    for symbol in symbols:
        start_date = start_by_symbol[symbol, None]
        if write_ticks_dataset(
            conn,
            client,
            [symbol],
            resolved_tick_flags,
            start_date,
            end_date,
            IfExists.APPEND,
            written_columns,
        ):
            written_tables.add(Dataset.ticks)
            _record_dedup_scope(
                dedup_scopes,
                Dataset.ticks,
                "symbol = ? AND time >= ?",
                (symbol, start_date),
            )


def _write_incremental_history_orders(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    fallback_start: datetime,
    end_date: datetime,
    written_columns: dict[Dataset, set[str]],
    written_tables: set[Dataset],
    dedup_scopes: dict[Dataset, list[DedupScope]],
) -> None:
    start_by_symbol = load_incremental_start_datetimes(
        conn,
        Dataset.history_orders,
        symbols=symbols,
        fallback_start=fallback_start,
    )
    for symbol in symbols:
        start_date = start_by_symbol[symbol, None]
        if write_history_dataset(
            conn,
            client.history_orders_get_as_df,
            Dataset.history_orders,
            [symbol],
            start_date,
            end_date,
            IfExists.APPEND,
            written_columns,
            include_account_events=False,
        ):
            written_tables.add(Dataset.history_orders)
            _record_dedup_scope(
                dedup_scopes,
                Dataset.history_orders,
                "symbol = ? AND time >= ?",
                (symbol, start_date),
            )


def _write_incremental_history_deals(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    fallback_start: datetime,
    end_date: datetime,
    written_columns: dict[Dataset, set[str]],
    written_tables: set[Dataset],
    dedup_scopes: dict[Dataset, list[DedupScope]],
    *,
    include_account_events: bool,
) -> None:
    if include_account_events:
        start_by_symbol = load_incremental_start_datetimes(
            conn,
            Dataset.history_deals,
            symbols=symbols,
            fallback_start=fallback_start,
        )
        account_event_start = get_history_deals_account_event_start_datetime(
            conn,
            fallback_start=fallback_start,
        )
        fetch_start = min([*start_by_symbol.values(), account_event_start])
        frame = filter_incremental_history_deals_frame(
            client.history_deals_get_as_df(
                date_from=fetch_start,
                date_to=end_date,
            ),
            symbols,
            {symbol: start_by_symbol[symbol, None] for symbol in symbols},
            account_event_start,
        )
        if write_streamed_frame(
            conn,
            frame,
            Dataset.history_deals,
            table_exists=False,
            if_exists=IfExists.APPEND,
            written_columns=written_columns,
        ):
            written_tables.add(Dataset.history_deals)
            columns = get_table_columns(conn, Dataset.history_deals.table_name)
            if "symbol" in columns:
                for symbol in symbols:
                    _record_dedup_scope(
                        dedup_scopes,
                        Dataset.history_deals,
                        "symbol = ? AND time >= ?",
                        (symbol, start_by_symbol[symbol, None]),
                    )
            if "type" in columns:
                _record_dedup_scope(
                    dedup_scopes,
                    Dataset.history_deals,
                    f"type NOT IN {_TRADE_DEAL_TYPES_SQL} AND time >= ?",
                    (account_event_start,),
                )
            if "type" not in columns and "symbol" in columns:
                _record_dedup_scope(
                    dedup_scopes,
                    Dataset.history_deals,
                    "(symbol IS NULL OR symbol = '') AND time >= ?",
                    (account_event_start,),
                )
        return
    start_by_symbol = load_incremental_start_datetimes(
        conn,
        Dataset.history_deals,
        symbols=symbols,
        fallback_start=fallback_start,
    )
    for symbol in symbols:
        start_date = start_by_symbol[symbol, None]
        if write_history_dataset(
            conn,
            client.history_deals_get_as_df,
            Dataset.history_deals,
            [symbol],
            start_date,
            end_date,
            IfExists.APPEND,
            written_columns,
            include_account_events=False,
        ):
            written_tables.add(Dataset.history_deals)
            _record_dedup_scope(
                dedup_scopes,
                Dataset.history_deals,
                "symbol = ? AND time >= ?",
                (symbol, start_date),
            )


def _finalize_incremental_writes(
    conn: sqlite3.Connection,
    selected_datasets: set[Dataset],
    written_columns: dict[Dataset, set[str]],
    written_tables: set[Dataset],
    dedup_scopes: dict[Dataset, list[DedupScope]],
    *,
    deduplicate: bool,
    create_rate_views: bool,
    with_views: bool,
) -> None:
    augment_written_columns_from_sqlite(conn, selected_datasets, written_columns)
    create_history_indexes(conn, written_columns)
    if deduplicate:
        deduplicate_history_tables(
            conn,
            written_columns,
            written_tables,
            dedup_scopes,
        )
    if create_rate_views and Dataset.rates in written_tables:
        create_rate_compatibility_views(conn)
    if with_views and Dataset.history_deals in selected_datasets:
        if Dataset.history_deals in written_columns:
            deal_columns = written_columns[Dataset.history_deals]
            create_cash_events_view(conn, deal_columns)
            create_positions_reconstructed_view(conn, deal_columns)
        if (
            Dataset.history_deals not in written_columns
            and Dataset.history_deals not in written_tables
        ):
            logger.warning(
                "with_views ignored: history_deals table was not available",
            )


def write_incremental_datasets(  # noqa: PLR0913
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    selected_datasets: set[Dataset],
    resolved_timeframes: list[int],
    resolved_tick_flags: int,
    fallback_start: datetime,
    end_date: datetime,
    *,
    deduplicate: bool,
    create_rate_views: bool,
    with_views: bool,
    include_account_events: bool,
) -> tuple[set[Dataset], dict[Dataset, set[str]]]:
    """Append selected datasets incrementally and refresh indexes and views.

    Returns:
        Written datasets and their columns.
    """
    written_columns: dict[Dataset, set[str]] = {}
    written_tables: set[Dataset] = set()
    dedup_scopes: dict[Dataset, list[DedupScope]] = {}
    if Dataset.rates in selected_datasets:
        _write_incremental_rates(
            conn,
            client,
            symbols,
            resolved_timeframes,
            fallback_start,
            end_date,
            written_columns,
            written_tables,
            dedup_scopes,
        )
    if Dataset.ticks in selected_datasets:
        _write_incremental_ticks(
            conn,
            client,
            symbols,
            resolved_tick_flags,
            fallback_start,
            end_date,
            written_columns,
            written_tables,
            dedup_scopes,
        )
    if Dataset.history_orders in selected_datasets:
        _write_incremental_history_orders(
            conn,
            client,
            symbols,
            fallback_start,
            end_date,
            written_columns,
            written_tables,
            dedup_scopes,
        )
    if Dataset.history_deals in selected_datasets:
        _write_incremental_history_deals(
            conn,
            client,
            symbols,
            fallback_start,
            end_date,
            written_columns,
            written_tables,
            dedup_scopes,
            include_account_events=include_account_events,
        )
    _finalize_incremental_writes(
        conn,
        selected_datasets,
        written_columns,
        written_tables,
        dedup_scopes,
        deduplicate=deduplicate,
        create_rate_views=create_rate_views,
        with_views=with_views,
    )
    return written_tables, written_columns


def write_collected_datasets(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    datasets: set[Dataset],
    timeframe: int,
    flags: int,
    date_from: datetime,
    date_to: datetime,
    if_exists: IfExists,
) -> tuple[set[Dataset], dict[Dataset, set[str]]]:
    """Collect selected datasets and stream each symbol frame into SQLite.

    Returns:
        Written datasets and their columns.
    """
    written_columns: dict[Dataset, set[str]] = {}
    written_tables: set[Dataset] = set()
    if Dataset.rates in datasets and write_rates_dataset(
        conn,
        client,
        symbols,
        timeframe,
        date_from,
        date_to,
        if_exists,
        written_columns,
    ):
        written_tables.add(Dataset.rates)
    if Dataset.ticks in datasets and write_ticks_dataset(
        conn,
        client,
        symbols,
        flags,
        date_from,
        date_to,
        if_exists,
        written_columns,
    ):
        written_tables.add(Dataset.ticks)
    if Dataset.history_orders in datasets and write_history_dataset(
        conn,
        client.history_orders_get_as_df,
        Dataset.history_orders,
        symbols,
        date_from,
        date_to,
        if_exists,
        written_columns,
        include_account_events=False,
    ):
        written_tables.add(Dataset.history_orders)
    if Dataset.history_deals in datasets and write_history_dataset(
        conn,
        client.history_deals_get_as_df,
        Dataset.history_deals,
        symbols,
        date_from,
        date_to,
        if_exists,
        written_columns,
        include_account_events=False,
    ):
        written_tables.add(Dataset.history_deals)
    return written_tables, written_columns
