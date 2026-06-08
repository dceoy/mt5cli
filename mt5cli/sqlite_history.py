"""SQLite helpers for incremental MT5 history collection."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

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
) -> str:
    """Return the offline optimize view name for a symbol and granularity."""
    if granularity_count == 1:
        return f"rate_{symbol}"
    return f"rate_{symbol}_{granularity}"


def get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return existing SQLite columns for a table."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
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


def get_incremental_start_datetime(
    conn: sqlite3.Connection,
    dataset: Dataset,
    *,
    symbol: str,
    timeframe: int | None,
    fallback_start: datetime,
) -> datetime:
    """Return the next update start datetime from existing MAX(time)."""
    table = dataset.table_name
    columns = get_table_columns(conn, table)
    if "time" not in columns:
        return fallback_start
    conditions: list[str] = []
    params: list[str | int] = []
    if "symbol" in columns:
        conditions.append("symbol = ?")
        params.append(symbol)
    if dataset is Dataset.rates and timeframe is not None and "timeframe" in columns:
        conditions.append("timeframe = ?")
        params.append(timeframe)
    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    row = conn.execute(
        f"SELECT MAX(time) FROM {table}{where_clause}",  # noqa: S608
        params,
    ).fetchone()
    parsed = parse_sqlite_timestamp(row[0] if row else None)
    return parsed if parsed is not None else fallback_start


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
        method="multi",
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
    cursor.execute(
        f"DELETE FROM {table} WHERE ROWID NOT IN"  # noqa: S608
        f" (SELECT {rowid_selector}(ROWID) FROM {table} GROUP BY {ids_csv})",
    )


def deduplicate_history_tables(
    conn: sqlite3.Connection,
    written_columns: dict[Dataset, set[str]],
) -> None:
    """Deduplicate appended history tables by stable identifiers."""
    cursor = conn.cursor()
    for dataset, columns in written_columns.items():
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
    if "type" in frame.columns:
        account_event_mask = ~frame["type"].isin(_TRADE_DEAL_TYPES)
    else:
        account_event_mask = frame["symbol"].isna() | frame["symbol"].eq("")
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


def create_rate_compatibility_views(conn: sqlite3.Connection) -> None:
    """Create rate compatibility views from the normalized rates table."""
    columns = get_table_columns(conn, Dataset.rates.table_name)
    if not {"symbol", "timeframe", "time"}.issubset(columns):
        return
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
            )
            quoted_view_name = quote_sqlite_identifier(view_name)
            escaped_symbol = symbol.replace("'", "''")
            conn.execute(f"DROP VIEW IF EXISTS {quoted_view_name}")
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
) -> None:
    for symbol in symbols:
        for timeframe in resolved_timeframes:
            start_date = get_incremental_start_datetime(
                conn,
                Dataset.rates,
                symbol=symbol,
                timeframe=timeframe,
                fallback_start=fallback_start,
            )
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


def _write_incremental_ticks(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    resolved_tick_flags: int,
    fallback_start: datetime,
    end_date: datetime,
    written_columns: dict[Dataset, set[str]],
    written_tables: set[Dataset],
) -> None:
    for symbol in symbols:
        start_date = get_incremental_start_datetime(
            conn,
            Dataset.ticks,
            symbol=symbol,
            timeframe=None,
            fallback_start=fallback_start,
        )
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


def _write_incremental_history_orders(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    fallback_start: datetime,
    end_date: datetime,
    written_columns: dict[Dataset, set[str]],
    written_tables: set[Dataset],
) -> None:
    for symbol in symbols:
        start_date = get_incremental_start_datetime(
            conn,
            Dataset.history_orders,
            symbol=symbol,
            timeframe=None,
            fallback_start=fallback_start,
        )
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


def _write_incremental_history_deals(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: Sequence[str],
    fallback_start: datetime,
    end_date: datetime,
    written_columns: dict[Dataset, set[str]],
    written_tables: set[Dataset],
    *,
    include_account_events: bool,
) -> None:
    for symbol in symbols:
        start_date = get_incremental_start_datetime(
            conn,
            Dataset.history_deals,
            symbol=symbol,
            timeframe=None,
            fallback_start=fallback_start,
        )
        if write_history_dataset(
            conn,
            client.history_deals_get_as_df,
            Dataset.history_deals,
            [symbol],
            start_date,
            end_date,
            IfExists.APPEND,
            written_columns,
            include_account_events=include_account_events,
        ):
            written_tables.add(Dataset.history_deals)


def _finalize_incremental_writes(
    conn: sqlite3.Connection,
    selected_datasets: set[Dataset],
    written_columns: dict[Dataset, set[str]],
    written_tables: set[Dataset],
    *,
    deduplicate: bool,
    create_rate_views: bool,
    with_views: bool,
) -> None:
    augment_written_columns_from_sqlite(conn, selected_datasets, written_columns)
    if deduplicate:
        deduplicate_history_tables(conn, written_columns)
    create_history_indexes(conn, written_columns)
    if create_rate_views and Dataset.rates in selected_datasets:
        create_rate_compatibility_views(conn)
    if with_views and Dataset.history_deals in written_columns:
        deal_columns = written_columns[Dataset.history_deals]
        create_cash_events_view(conn, deal_columns)
        create_positions_reconstructed_view(conn, deal_columns)
    elif with_views and Dataset.history_deals not in written_tables:
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
            include_account_events=include_account_events,
        )
    _finalize_incremental_writes(
        conn,
        selected_datasets,
        written_columns,
        written_tables,
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
