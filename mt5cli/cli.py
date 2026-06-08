"""Command-line interface for MetaTrader 5 data export."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from pdmt5 import Mt5Config, Mt5DataClient

from .utils import (
    DATETIME_TYPE,
    REQUEST_TYPE,
    TICK_FLAGS_TYPE,
    TIMEFRAME_TYPE,
    Dataset,
    IfExists,
    LogLevel,
    OutputFormat,
    detect_format,
    export_dataframe,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

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
_DATASET_TABLE_NAMES: dict[Dataset, str] = {
    Dataset.rates: "rates",
    Dataset.ticks: "ticks",
    Dataset.history_orders: "history_orders",
    Dataset.history_deals: "history_deals",
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Export context
# ---------------------------------------------------------------------------


@dataclass
class _ExportContext:
    """Shared context data passed from the callback to each subcommand."""

    output: Path
    output_format: str
    table: str
    config: Mt5Config


# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="mt5cli",
    help="Export MetaTrader5 data to CSV, JSON, Parquet, or SQLite3.",
)

_REQUEST_OPTION_HELP = (
    "Order request as a JSON object string, or '@path' to load JSON from a file."
)


def _get_export_context(ctx: typer.Context) -> _ExportContext:
    return cast("_ExportContext", ctx.obj)


def _execute_export(
    ctx: typer.Context,
    fetch_fn: Callable[[Mt5DataClient], pd.DataFrame],
) -> None:
    """Execute the common connect-fetch-export-shutdown workflow.

    Args:
        ctx: Typer context carrying shared options.
        fetch_fn: Callable that receives a connected client and returns a
            DataFrame.
    """
    export_ctx = _get_export_context(ctx)
    client = Mt5DataClient(config=export_ctx.config)
    client.initialize_and_login_mt5()
    try:
        df = fetch_fn(client)
        export_dataframe(
            df=df,
            output_path=export_ctx.output,
            output_format=export_ctx.output_format,
            table_name=export_ctx.table,
        )
        logger.info(
            "Exported %d rows to %s (%s)",
            len(df),
            export_ctx.output,
            export_ctx.output_format,
        )
    finally:
        client.shutdown()


@app.callback()
def _callback(  # pyright: ignore[reportUnusedFunction]
    ctx: typer.Context,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path."),
    ],
    fmt: Annotated[
        OutputFormat | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format (auto-detected from extension if omitted).",
        ),
    ] = None,
    table: Annotated[
        str,
        typer.Option(help="Table name for SQLite3 output."),
    ] = "data",
    login: Annotated[
        int | None,
        typer.Option(help="Trading account login."),
    ] = None,
    password: Annotated[
        str | None,
        typer.Option(help="Trading account password."),
    ] = None,
    server: Annotated[
        str | None,
        typer.Option(help="Trading server name."),
    ] = None,
    path: Annotated[
        str | None,
        typer.Option(help="Path to MetaTrader5 terminal EXE file."),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option(help="Connection timeout in milliseconds."),
    ] = None,
    log_level: Annotated[
        LogLevel,
        typer.Option("--log-level", help="Logging level."),
    ] = LogLevel.WARNING,
) -> None:
    """Configure shared options for all export commands.

    Raises:
        typer.BadParameter: If the output format cannot be determined.
    """
    logging.basicConfig(level=getattr(logging, log_level.value))
    try:
        output_format = detect_format(
            output,
            explicit_format=fmt.value if fmt is not None else None,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    ctx.obj = _ExportContext(
        output=output,
        output_format=output_format,
        table=table,
        config=Mt5Config(
            path=path,
            login=login,
            password=password,
            server=server,
            timeout=timeout,
        ),
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command()
def rates_from(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
    timeframe: Annotated[
        int,
        typer.Option(
            click_type=TIMEFRAME_TYPE,
            help="Timeframe (e.g., M1, H1, D1, or integer).",
        ),
    ],
    date_from: Annotated[
        datetime,
        typer.Option(
            click_type=DATETIME_TYPE,
            help="Start date in ISO 8601 format.",
        ),
    ],
    count: Annotated[int, typer.Option(help="Number of records.")],
) -> None:
    """Export rates from a start date."""
    _execute_export(
        ctx,
        lambda c: c.copy_rates_from_as_df(
            symbol=symbol,
            timeframe=timeframe,
            date_from=date_from,
            count=count,
        ),
    )


@app.command()
def rates_from_pos(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
    timeframe: Annotated[
        int,
        typer.Option(
            click_type=TIMEFRAME_TYPE,
            help="Timeframe.",
        ),
    ],
    start_pos: Annotated[int, typer.Option(help="Start position (0 = current bar).")],
    count: Annotated[int, typer.Option(help="Number of records.")],
) -> None:
    """Export rates from a start position."""
    _execute_export(
        ctx,
        lambda c: c.copy_rates_from_pos_as_df(
            symbol=symbol,
            timeframe=timeframe,
            start_pos=start_pos,
            count=count,
        ),
    )


@app.command()
def rates_range(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
    timeframe: Annotated[
        int,
        typer.Option(
            click_type=TIMEFRAME_TYPE,
            help="Timeframe.",
        ),
    ],
    date_from: Annotated[
        datetime,
        typer.Option(click_type=DATETIME_TYPE, help="Start date."),
    ],
    date_to: Annotated[
        datetime,
        typer.Option(click_type=DATETIME_TYPE, help="End date."),
    ],
) -> None:
    """Export rates for a date range."""
    _execute_export(
        ctx,
        lambda c: c.copy_rates_range_as_df(
            symbol=symbol,
            timeframe=timeframe,
            date_from=date_from,
            date_to=date_to,
        ),
    )


@app.command()
def ticks_from(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
    date_from: Annotated[
        datetime,
        typer.Option(click_type=DATETIME_TYPE, help="Start date."),
    ],
    count: Annotated[int, typer.Option(help="Number of ticks.")],
    flags: Annotated[
        int,
        typer.Option(
            click_type=TICK_FLAGS_TYPE,
            help="Tick flags (ALL, INFO, TRADE, or integer).",
        ),
    ],
) -> None:
    """Export ticks from a start date."""
    _execute_export(
        ctx,
        lambda c: c.copy_ticks_from_as_df(
            symbol=symbol,
            date_from=date_from,
            count=count,
            flags=flags,
        ),
    )


@app.command()
def ticks_range(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
    date_from: Annotated[
        datetime,
        typer.Option(click_type=DATETIME_TYPE, help="Start date."),
    ],
    date_to: Annotated[
        datetime,
        typer.Option(click_type=DATETIME_TYPE, help="End date."),
    ],
    flags: Annotated[
        int,
        typer.Option(click_type=TICK_FLAGS_TYPE, help="Tick flags."),
    ],
) -> None:
    """Export ticks for a date range."""
    _execute_export(
        ctx,
        lambda c: c.copy_ticks_range_as_df(
            symbol=symbol,
            date_from=date_from,
            date_to=date_to,
            flags=flags,
        ),
    )


@app.command()
def account_info(ctx: typer.Context) -> None:
    """Export account information."""
    _execute_export(ctx, lambda c: c.account_info_as_df())


@app.command()
def terminal_info(ctx: typer.Context) -> None:
    """Export terminal information."""
    _execute_export(ctx, lambda c: c.terminal_info_as_df())


@app.command()
def symbols(
    ctx: typer.Context,
    group: Annotated[
        str | None,
        typer.Option(help="Symbol group filter (e.g., *USD*)."),
    ] = None,
) -> None:
    """Export symbol list."""
    _execute_export(
        ctx,
        lambda c: c.symbols_get_as_df(group=group),
    )


@app.command()
def symbol_info(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export symbol details."""
    _execute_export(
        ctx,
        lambda c: c.symbol_info_as_df(symbol=symbol),
    )


@app.command()
def orders(
    ctx: typer.Context,
    symbol: Annotated[str | None, typer.Option(help="Symbol filter.")] = None,
    group: Annotated[str | None, typer.Option(help="Group filter.")] = None,
    ticket: Annotated[int | None, typer.Option(help="Ticket filter.")] = None,
) -> None:
    """Export active orders."""
    _execute_export(
        ctx,
        lambda c: c.orders_get_as_df(
            symbol=symbol,
            group=group,
            ticket=ticket,
        ),
    )


@app.command()
def positions(
    ctx: typer.Context,
    symbol: Annotated[str | None, typer.Option(help="Symbol filter.")] = None,
    group: Annotated[str | None, typer.Option(help="Group filter.")] = None,
    ticket: Annotated[int | None, typer.Option(help="Ticket filter.")] = None,
) -> None:
    """Export open positions."""
    _execute_export(
        ctx,
        lambda c: c.positions_get_as_df(
            symbol=symbol,
            group=group,
            ticket=ticket,
        ),
    )


@app.command()
def history_orders(
    ctx: typer.Context,
    date_from: Annotated[
        datetime | None,
        typer.Option(click_type=DATETIME_TYPE, help="Start date."),
    ] = None,
    date_to: Annotated[
        datetime | None,
        typer.Option(click_type=DATETIME_TYPE, help="End date."),
    ] = None,
    group: Annotated[str | None, typer.Option(help="Group filter.")] = None,
    symbol: Annotated[str | None, typer.Option(help="Symbol filter.")] = None,
    ticket: Annotated[int | None, typer.Option(help="Order ticket.")] = None,
    position: Annotated[int | None, typer.Option(help="Position ticket.")] = None,
) -> None:
    """Export historical orders."""
    _execute_export(
        ctx,
        lambda c: c.history_orders_get_as_df(
            date_from=date_from,
            date_to=date_to,
            group=group,
            symbol=symbol,
            ticket=ticket,
            position=position,
        ),
    )


@app.command()
def history_deals(
    ctx: typer.Context,
    date_from: Annotated[
        datetime | None,
        typer.Option(click_type=DATETIME_TYPE, help="Start date."),
    ] = None,
    date_to: Annotated[
        datetime | None,
        typer.Option(click_type=DATETIME_TYPE, help="End date."),
    ] = None,
    group: Annotated[str | None, typer.Option(help="Group filter.")] = None,
    symbol: Annotated[str | None, typer.Option(help="Symbol filter.")] = None,
    ticket: Annotated[int | None, typer.Option(help="Order ticket.")] = None,
    position: Annotated[int | None, typer.Option(help="Position ticket.")] = None,
) -> None:
    """Export historical deals."""
    _execute_export(
        ctx,
        lambda c: c.history_deals_get_as_df(
            date_from=date_from,
            date_to=date_to,
            group=group,
            symbol=symbol,
            ticket=ticket,
            position=position,
        ),
    )


@app.command()
def version(ctx: typer.Context) -> None:
    """Export MetaTrader5 version information."""
    _execute_export(ctx, lambda c: c.version_as_df())


@app.command()
def last_error(ctx: typer.Context) -> None:
    """Export the last error information."""
    _execute_export(ctx, lambda c: c.last_error_as_df())


@app.command()
def symbol_info_tick(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export the last tick for a symbol."""
    _execute_export(
        ctx,
        lambda c: c.symbol_info_tick_as_df(symbol=symbol),
    )


@app.command()
def market_book(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export market depth (order book) for a symbol."""
    _execute_export(
        ctx,
        lambda c: c.market_book_get_as_df(symbol=symbol),
    )


@app.command()
def order_check(
    ctx: typer.Context,
    request: Annotated[
        dict[str, Any],
        typer.Option(click_type=REQUEST_TYPE, help=_REQUEST_OPTION_HELP),
    ],
) -> None:
    """Check funds sufficiency for a trading operation."""
    _execute_export(
        ctx,
        lambda c: c.order_check_as_df(request=request),
    )


@app.command()
def order_send(
    ctx: typer.Context,
    request: Annotated[
        dict[str, Any],
        typer.Option(click_type=REQUEST_TYPE, help=_REQUEST_OPTION_HELP),
    ],
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Confirm the live trade request."),
    ] = False,
) -> None:
    """Send a trading operation request to the trade server.

    Raises:
        typer.BadParameter: If --yes is not provided.
    """
    if not yes:
        msg = "Pass --yes to send a live trade request."
        raise typer.BadParameter(msg, param_hint="--yes")
    _execute_export(
        ctx,
        lambda c: c.order_send_as_df(request=request),
    )


def _create_cash_events_view(
    conn: sqlite3.Connection,
    deals_columns: set[str],
) -> bool:
    """Create the cash_events SQLite view derived from history_deals.

    Args:
        conn: Open SQLite connection.
        deals_columns: Column names present in the history_deals table.

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


def _create_positions_reconstructed_view(
    conn: sqlite3.Connection,
    deals_columns: set[str],
) -> bool:
    """Create the positions_reconstructed SQLite view derived from history_deals.

    The view aggregates trade deals (``type IN (0, 1)``) by ``position_id`` and
    excludes positions that have no closing deal (``entry IN (1, 3)``), so
    still-open positions and reversal-only fragments are filtered out.

    Open/close prices are volume-weighted averages over the corresponding
    entry deals. Reversal deals (``DEAL_ENTRY_INOUT = 2``) are reported via
    ``volume_reversal`` and ``reversal_count``; they do not contribute to the
    open or close volume/price weights because a single reversal deal mixes a
    close of the existing direction with the open of the new direction.

    Args:
        conn: Open SQLite connection.
        deals_columns: Column names present in the history_deals table.

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
        " SUM(CASE WHEN entry IN (1, 3) THEN volume ELSE 0 END) AS volume_close,"
        " SUM(CASE WHEN entry = 2 THEN volume ELSE 0 END) AS volume_reversal,"
        " CASE"
        " WHEN SUM(CASE WHEN entry = 0 THEN volume ELSE 0 END) > 0"
        " THEN SUM(CASE WHEN entry = 0 THEN price * volume ELSE 0 END)"
        " / SUM(CASE WHEN entry = 0 THEN volume ELSE 0 END)"
        " END AS open_price,"
        " CASE"
        " WHEN SUM(CASE WHEN entry IN (1, 3) THEN volume ELSE 0 END) > 0"
        " THEN SUM(CASE WHEN entry IN (1, 3) THEN price * volume ELSE 0 END)"
        " / SUM(CASE WHEN entry IN (1, 3) THEN volume ELSE 0 END)"
        " END AS close_price,"
        " SUM(profit) AS total_profit,"
        " SUM(CASE WHEN entry = 2 THEN 1 ELSE 0 END) AS reversal_count,"
        " COUNT(*) AS deals_count"
        " FROM history_deals"
        f" WHERE type IN {_TRADE_DEAL_TYPES_SQL} AND position_id != 0"
        " GROUP BY position_id, symbol"
        " HAVING SUM(CASE WHEN entry IN (1, 3) THEN 1 ELSE 0 END) > 0",
    )
    return True


def _write_frame_to_sqlite(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    table_name: str,
    if_exists: IfExists,
) -> bool:
    """Write a non-empty-schema frame to SQLite.

    Args:
        conn: Open SQLite connection.
        frame: DataFrame to write.
        table_name: Target SQLite table name.
        if_exists: Table conflict behavior.

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


def _create_collect_history_indexes(
    conn: sqlite3.Connection,
    written_columns: dict[Dataset, set[str]],
) -> None:
    """Create useful indexes for collected history tables when present."""
    if {"symbol", "time"}.issubset(written_columns.get(Dataset.rates, set())):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rates_symbol_time ON rates(symbol, time)",
        )
    if {"symbol", "time"}.issubset(written_columns.get(Dataset.ticks, set())):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, time)",
        )
    if {"position_id", "symbol"}.issubset(
        written_columns.get(Dataset.history_deals, set())
    ):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_deals_position_symbol"
            " ON history_deals(position_id, symbol)",
        )


def _record_written_columns(
    written_columns: dict[Dataset, set[str]],
    dataset: Dataset,
    frame: pd.DataFrame,
) -> None:
    """Remember columns for datasets written during streaming collection."""
    columns = set(frame.columns)
    if dataset in written_columns:
        written_columns[dataset].update(columns)
    else:
        written_columns[dataset] = columns


def _write_streamed_frame(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    dataset: Dataset,
    table_exists: bool,
    if_exists: IfExists,
    written_columns: dict[Dataset, set[str]],
) -> bool:
    """Write one streamed dataset frame and track table state.

    Args:
        conn: Open SQLite connection.
        frame: DataFrame to write.
        dataset: Dataset being written.
        table_exists: Whether this dataset table has already been written.
        if_exists: Initial table conflict behavior.
        written_columns: Mutable map of columns written by dataset.

    Returns:
        True if the dataset table exists after this write attempt.
    """
    write_mode = IfExists.APPEND if table_exists else if_exists
    if _write_frame_to_sqlite(
        conn,
        frame,
        _DATASET_TABLE_NAMES[dataset],
        write_mode,
    ):
        _record_written_columns(written_columns, dataset, frame)
        return True
    return table_exists


def _write_rates_dataset(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: list[str],
    timeframe: int,
    date_from: datetime,
    date_to: datetime,
    if_exists: IfExists,
    written_columns: dict[Dataset, set[str]],
) -> bool:
    """Stream rates frames into SQLite.

    Args:
        conn: Open SQLite connection.
        client: Connected MT5 data client.
        symbols: Symbols to collect.
        timeframe: Rates timeframe integer.
        date_from: Start date.
        date_to: End date.
        if_exists: Initial table conflict behavior.
        written_columns: Mutable map of columns written by dataset.

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
        )
        frame.insert(0, "symbol", sym)
        frame.insert(1, "timeframe", timeframe)
        table_exists = _write_streamed_frame(
            conn,
            frame,
            Dataset.rates,
            table_exists,
            if_exists,
            written_columns,
        )
    return table_exists


def _write_ticks_dataset(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: list[str],
    flags: int,
    date_from: datetime,
    date_to: datetime,
    if_exists: IfExists,
    written_columns: dict[Dataset, set[str]],
) -> bool:
    """Stream ticks frames into SQLite.

    Args:
        conn: Open SQLite connection.
        client: Connected MT5 data client.
        symbols: Symbols to collect.
        flags: Tick copy flags integer.
        date_from: Start date.
        date_to: End date.
        if_exists: Initial table conflict behavior.
        written_columns: Mutable map of columns written by dataset.

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
        )
        frame.insert(0, "symbol", sym)
        table_exists = _write_streamed_frame(
            conn,
            frame,
            Dataset.ticks,
            table_exists,
            if_exists,
            written_columns,
        )
    return table_exists


def _write_history_dataset(
    conn: sqlite3.Connection,
    fetch: Callable[..., pd.DataFrame],
    dataset: Dataset,
    symbols: list[str],
    date_from: datetime,
    date_to: datetime,
    if_exists: IfExists,
    written_columns: dict[Dataset, set[str]],
) -> bool:
    """Stream a history dataset into SQLite with exact symbol filtering.

    Args:
        conn: Open SQLite connection.
        fetch: Bound history_orders_get_as_df / history_deals_get_as_df method.
        dataset: History dataset being written.
        symbols: Symbols to collect.
        date_from: Start date.
        date_to: End date.
        if_exists: Initial table conflict behavior.
        written_columns: Mutable map of columns written by dataset.

    Returns:
        True if the history table was written.
    """
    table_exists = False
    for sym in symbols:
        frame = fetch(date_from=date_from, date_to=date_to, symbol=sym)
        if "symbol" in frame.columns:
            frame = frame[frame["symbol"] == sym]
        table_exists = _write_streamed_frame(
            conn,
            frame,
            dataset,
            table_exists,
            if_exists,
            written_columns,
        )
    return table_exists


def _write_collected_datasets(
    conn: sqlite3.Connection,
    client: Mt5DataClient,
    symbols: list[str],
    datasets: set[Dataset],
    timeframe: int,
    flags: int,
    date_from: datetime,
    date_to: datetime,
    if_exists: IfExists,
) -> tuple[set[Dataset], dict[Dataset, set[str]]]:
    """Collect selected datasets and stream each symbol frame into SQLite.

    Args:
        conn: Open SQLite connection.
        client: Connected MT5 data client.
        symbols: Symbols to collect.
        datasets: Selected datasets to write.
        timeframe: Rates timeframe integer.
        flags: Tick copy flags integer.
        date_from: Start date.
        date_to: End date.
        if_exists: Initial table conflict behavior.

    Returns:
        Written datasets and their columns.
    """
    written_columns: dict[Dataset, set[str]] = {}
    written_tables: set[Dataset] = set()
    if Dataset.rates in datasets and _write_rates_dataset(
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
    if Dataset.ticks in datasets and _write_ticks_dataset(
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
    if Dataset.history_orders in datasets and _write_history_dataset(
        conn,
        client.history_orders_get_as_df,
        Dataset.history_orders,
        symbols,
        date_from,
        date_to,
        if_exists,
        written_columns,
    ):
        written_tables.add(Dataset.history_orders)
    if Dataset.history_deals in datasets and _write_history_dataset(
        conn,
        client.history_deals_get_as_df,
        Dataset.history_deals,
        symbols,
        date_from,
        date_to,
        if_exists,
        written_columns,
    ):
        written_tables.add(Dataset.history_deals)
    return written_tables, written_columns


@app.command()
def collect_history(
    ctx: typer.Context,
    symbol: Annotated[
        list[str],
        typer.Option(
            "--symbol",
            "-s",
            help="Symbol to collect (repeat for multiple symbols).",
        ),
    ],
    date_from: Annotated[
        datetime,
        typer.Option(click_type=DATETIME_TYPE, help="Start date."),
    ],
    date_to: Annotated[
        datetime,
        typer.Option(click_type=DATETIME_TYPE, help="End date."),
    ],
    dataset: Annotated[
        list[Dataset] | None,
        typer.Option(
            "--dataset",
            help=(
                "Dataset to include (repeat for multiple)."
                " Defaults to all: rates, ticks, history-orders, history-deals."
            ),
        ),
    ] = None,
    timeframe: Annotated[
        int,
        typer.Option(
            click_type=TIMEFRAME_TYPE,
            help="Rates timeframe (e.g., M1, H1, D1).",
        ),
    ] = 1,
    flags: Annotated[
        int,
        typer.Option(
            click_type=TICK_FLAGS_TYPE,
            help="Tick copy flags (ALL, INFO, TRADE, or integer).",
        ),
    ] = 1,
    if_exists: Annotated[
        IfExists,
        typer.Option(
            "--if-exists",
            help="Behavior when a target table already exists.",
        ),
    ] = IfExists.FAIL,
    with_views: Annotated[
        bool,
        typer.Option(
            "--with-views",
            help=(
                "Add cash_events and positions_reconstructed SQLite views"
                " derived from history_deals."
            ),
        ),
    ] = False,
) -> None:
    """Collect historical datasets into a single SQLite database.

    Tables written depend on ``--dataset``: ``rates``, ``ticks``,
    ``history_orders``, ``history_deals``. History datasets are fetched per
    symbol and concatenated. Rates rows carry the requested ``timeframe`` so
    appended runs at different timeframes remain distinguishable.

    With ``--with-views`` (requires the ``history-deals`` dataset), optional
    views ``cash_events`` and ``positions_reconstructed`` are derived from
    ``history_deals`` when the required columns are present.

    Raises:
        typer.BadParameter: If the output format is not SQLite3.
    """
    export_ctx = _get_export_context(ctx)
    if export_ctx.output_format != "sqlite3":
        msg = (
            "collect-history requires SQLite3 output."
            " Use a .db/.sqlite/.sqlite3 extension or --format sqlite3."
        )
        raise typer.BadParameter(msg)
    datasets = set(dataset) if dataset else set(Dataset)
    client = Mt5DataClient(config=export_ctx.config)
    client.initialize_and_login_mt5()
    try:
        with sqlite3.connect(export_ctx.output) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            written_tables, written_columns = _write_collected_datasets(
                conn,
                client,
                symbol,
                datasets,
                timeframe,
                flags,
                date_from,
                date_to,
                if_exists,
            )
            _create_collect_history_indexes(conn, written_columns)
            if with_views and Dataset.history_deals in written_tables:
                _create_cash_events_view(conn, written_columns[Dataset.history_deals])
                _create_positions_reconstructed_view(
                    conn,
                    written_columns[Dataset.history_deals],
                )
            elif with_views:
                logger.warning(
                    "--with-views ignored: history_deals table was not written"
                )
        logger.info(
            "Collected %s for %d symbol(s) into %s",
            ", ".join(sorted(ds.value for ds in datasets)),
            len(symbol),
            export_ctx.output,
        )
    finally:
        client.shutdown()


def main() -> None:
    """Run the mt5cli CLI."""
    app()
