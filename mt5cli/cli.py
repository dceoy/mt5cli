"""Command-line interface for MetaTrader 5 data and execution utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, cast

import pandas as pd
import typer
from pdmt5 import Mt5Config

from . import sdk
from .client import MT5Client
from .trading import OrderExecutionResult, close_open_positions, create_trading_client
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
    help=(
        "MT5 data and execution utilities — read market data, inspect account"
        " state, and send trade requests. Data commands write to CSV, JSON,"
        " Parquet, or SQLite3. Execution commands (order-send, close-positions)"
        " require --yes for live mutations."
    ),
)

_REQUEST_OPTION_HELP = (
    "Order request as a JSON object string, or '@path' to load JSON from a file."
)


def _get_export_context(ctx: typer.Context) -> _ExportContext:
    return cast("_ExportContext", ctx.obj)


def _execute_export(
    ctx: typer.Context,
    fetch_fn: Callable[[], pd.DataFrame],
) -> None:
    """Execute the common fetch-export workflow.

    Args:
        ctx: Typer context carrying shared options.
        fetch_fn: Callable that returns a DataFrame via the SDK layer.
    """
    export_ctx = _get_export_context(ctx)
    df = fetch_fn()
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


def _sdk_client(ctx: typer.Context) -> MT5Client:
    export_ctx = _get_export_context(ctx)
    return MT5Client(config=export_ctx.config)


def _export_command(
    ctx: typer.Context,
    fetch_fn: Callable[[MT5Client], pd.DataFrame],
) -> None:
    """Create an SDK client, fetch a DataFrame, and export it."""
    client = _sdk_client(ctx)
    _execute_export(ctx, lambda: fetch_fn(client))


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
    """Configure shared connection and output options.

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


@app.command(rich_help_panel="Data / Export")
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
    _export_command(
        ctx,
        lambda client: client.copy_rates_from(symbol, timeframe, date_from, count),
    )


@app.command(rich_help_panel="Data / Export")
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
    _export_command(
        ctx,
        lambda client: client.copy_rates_from_pos(
            symbol,
            timeframe,
            start_pos,
            count,
        ),
    )


@app.command(rich_help_panel="Data / Export")
def latest_rates(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
    timeframe: Annotated[
        int,
        typer.Option(
            click_type=TIMEFRAME_TYPE,
            help="Timeframe.",
        ),
    ],
    count: Annotated[int, typer.Option(help="Number of records.")],
    start_pos: Annotated[
        int,
        typer.Option(help="Start position (0 = current bar)."),
    ] = 0,
) -> None:
    """Export latest rates from a start position."""
    _export_command(
        ctx,
        lambda client: client.latest_rates(
            symbol,
            timeframe,
            count,
            start_pos=start_pos,
        ),
    )


@app.command(rich_help_panel="Data / Export")
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
    _export_command(
        ctx,
        lambda client: client.copy_rates_range(symbol, timeframe, date_from, date_to),
    )


@app.command(rich_help_panel="Data / Export")
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
    _export_command(
        ctx,
        lambda client: client.copy_ticks_from(symbol, date_from, count, flags),
    )


@app.command(rich_help_panel="Data / Export")
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
    _export_command(
        ctx,
        lambda client: client.copy_ticks_range(symbol, date_from, date_to, flags),
    )


@app.command(rich_help_panel="Data / Export")
def ticks_recent(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
    seconds: Annotated[
        float,
        typer.Option(help="Lookback window in seconds."),
    ],
    date_to: Annotated[
        datetime | None,
        typer.Option(click_type=DATETIME_TYPE, help="Window end date."),
    ] = None,
    count: Annotated[
        int,
        typer.Option(help="Maximum number of ticks to return."),
    ] = 10000,
    flags: Annotated[
        int,
        typer.Option(
            click_type=TICK_FLAGS_TYPE,
            help="Tick flags (ALL, INFO, TRADE, or integer).",
        ),
    ] = "ALL",  # pyright: ignore[reportArgumentType]
) -> None:
    """Export ticks from a recent time window."""
    _export_command(
        ctx,
        lambda client: client.recent_ticks(
            symbol,
            seconds,
            date_to=date_to,
            count=count,
            flags=flags,
        ),
    )


@app.command(rich_help_panel="Data / Export")
def account_info(ctx: typer.Context) -> None:
    """Export account information."""
    _export_command(ctx, lambda client: client.account_info())


@app.command(rich_help_panel="Data / Export")
def terminal_info(ctx: typer.Context) -> None:
    """Export terminal information."""
    _export_command(ctx, lambda client: client.terminal_info())


@app.command(rich_help_panel="Data / Export")
def symbols(
    ctx: typer.Context,
    group: Annotated[
        str | None,
        typer.Option(help="Symbol group filter (e.g., *USD*)."),
    ] = None,
) -> None:
    """Export symbol list."""
    _export_command(ctx, lambda client: client.symbols(group=group))


@app.command(rich_help_panel="Data / Export")
def symbol_info(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export symbol details."""
    _export_command(ctx, lambda client: client.symbol_info(symbol))


@app.command(rich_help_panel="Data / Export")
def minimum_margins(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export minimum-volume buy and sell margin requirements."""
    _export_command(ctx, lambda client: client.minimum_margins(symbol))


@app.command(rich_help_panel="Data / Export")
def orders(
    ctx: typer.Context,
    symbol: Annotated[str | None, typer.Option(help="Symbol filter.")] = None,
    group: Annotated[str | None, typer.Option(help="Group filter.")] = None,
    ticket: Annotated[int | None, typer.Option(help="Ticket filter.")] = None,
) -> None:
    """Export active orders."""
    _export_command(
        ctx,
        lambda client: client.orders(symbol=symbol, group=group, ticket=ticket),
    )


@app.command(rich_help_panel="Data / Export")
def positions(
    ctx: typer.Context,
    symbol: Annotated[str | None, typer.Option(help="Symbol filter.")] = None,
    group: Annotated[str | None, typer.Option(help="Group filter.")] = None,
    ticket: Annotated[int | None, typer.Option(help="Ticket filter.")] = None,
) -> None:
    """Export open positions."""
    _export_command(
        ctx,
        lambda client: client.positions(symbol=symbol, group=group, ticket=ticket),
    )


@app.command(rich_help_panel="Data / Export")
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
    _export_command(
        ctx,
        lambda client: client.history_orders(
            date_from=date_from,
            date_to=date_to,
            group=group,
            symbol=symbol,
            ticket=ticket,
            position=position,
        ),
    )


@app.command(rich_help_panel="Data / Export")
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
    _export_command(
        ctx,
        lambda client: client.history_deals(
            date_from=date_from,
            date_to=date_to,
            group=group,
            symbol=symbol,
            ticket=ticket,
            position=position,
        ),
    )


@app.command(rich_help_panel="Data / Export")
def recent_history_deals(
    ctx: typer.Context,
    hours: Annotated[float, typer.Option(help="Lookback window in hours.")],
    date_to: Annotated[
        datetime | None,
        typer.Option(click_type=DATETIME_TYPE, help="Window end date."),
    ] = None,
    group: Annotated[str | None, typer.Option(help="Group filter.")] = None,
    symbol: Annotated[str | None, typer.Option(help="Symbol filter.")] = None,
) -> None:
    """Export historical deals from a recent trailing window."""
    _export_command(
        ctx,
        lambda client: client.recent_history_deals(
            hours,
            date_to=date_to,
            group=group,
            symbol=symbol,
        ),
    )


@app.command(rich_help_panel="Data / Export")
def mt5_summary(ctx: typer.Context) -> None:
    """Export a compact terminal/account status summary."""
    _export_command(ctx, lambda client: client.mt5_summary_as_df())


@app.command(rich_help_panel="Data / Export")
def version(ctx: typer.Context) -> None:
    """Export MetaTrader5 version information."""
    _export_command(ctx, lambda client: client.version())


@app.command(rich_help_panel="Data / Export")
def last_error(ctx: typer.Context) -> None:
    """Export the last error information."""
    _export_command(ctx, lambda client: client.last_error())


@app.command(rich_help_panel="Data / Export")
def symbol_info_tick(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export the last tick for a symbol."""
    _export_command(ctx, lambda client: client.symbol_info_tick(symbol))


@app.command(rich_help_panel="Data / Export")
def market_book(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export market depth (order book) for a symbol."""
    _export_command(ctx, lambda client: client.market_book(symbol))


@app.command(rich_help_panel="Data / Export")
def order_check(
    ctx: typer.Context,
    request: Annotated[
        dict[str, Any],
        typer.Option(click_type=REQUEST_TYPE, help=_REQUEST_OPTION_HELP),
    ],
) -> None:
    """Check funds sufficiency for a trading operation."""
    _export_command(ctx, lambda client: client.order_check(request))


@app.command(rich_help_panel="Execution")
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
    """Send a raw trade request to the trade server (expert path, live execution).

    Passes the request JSON directly to MT5 ``order_send``. This is the
    low-level expert path — it places real trades on the connected account
    with no additional validation beyond what MT5 itself performs. Use
    ``order-check`` first to validate funds sufficiency. Prefer
    ``close-positions`` for closing open positions. ``--yes`` is required.

    Raises:
        typer.BadParameter: If --yes is not provided.
    """
    if not yes:
        msg = "Pass --yes to send a live trade request."
        raise typer.BadParameter(msg, param_hint="--yes")
    _export_command(ctx, lambda client: client.order_send(request))


_EXECUTION_RESULT_COLUMNS: list[str] = [
    "status",
    "symbol",
    "order_side",
    "volume",
    "retcode",
    "comment",
    "request",
    "response",
    "dry_run",
]


def _execution_results_to_df(results: list[OrderExecutionResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=_EXECUTION_RESULT_COLUMNS)
    rows = [
        {
            **r,
            "request": json.dumps(r["request"]),
            "response": json.dumps(r["response"]),
        }
        for r in results
    ]
    return pd.DataFrame(rows)


@app.command(rich_help_panel="Execution")
def close_positions(
    ctx: typer.Context,
    symbol: Annotated[
        list[str] | None,
        typer.Option(
            "--symbol",
            "-s",
            help="Symbol to close (repeat for multiple symbols).",
        ),
    ] = None,
    ticket: Annotated[
        list[int] | None,
        typer.Option(
            "--ticket",
            "-t",
            help="Position ticket to close (repeat for multiple tickets).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview close orders without executing them."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Confirm live position closing."),
    ] = False,
) -> None:
    """Close open positions by symbol or ticket.

    Delegates to :func:`mt5cli.trading.close_open_positions`. At least one
    ``--symbol`` or ``--ticket`` must be provided to avoid accidentally closing
    all positions. Use ``--dry-run`` to preview without executing; ``--yes`` is
    required for live execution.

    ``order-send`` is the expert raw-request path. ``close-positions`` is the
    safer high-level helper that builds correct close requests automatically.

    Raises:
        typer.BadParameter: If neither ``--symbol`` nor ``--ticket`` is given,
            or if ``--yes`` is missing for a live (non-dry-run) run.
    """
    if not symbol and not ticket:
        msg = "Provide at least one --symbol or --ticket to close positions."
        raise typer.BadParameter(msg)
    if not dry_run and not yes:
        msg = "Pass --yes to close live positions."
        raise typer.BadParameter(msg, param_hint="--yes")
    export_ctx = _get_export_context(ctx)
    client = create_trading_client(config=export_ctx.config)
    try:
        results = close_open_positions(
            client,
            symbols=list(symbol) if symbol else None,
            tickets=list(ticket) if ticket else None,
            dry_run=dry_run,
        )
    finally:
        client.shutdown()
    df = _execution_results_to_df(results)
    _execute_export(ctx, lambda: df)


@app.command(rich_help_panel="Collection")
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
                " Defaults to rates, history-orders, history-deals."
                " Ticks are opt-in: pass --dataset ticks to include them."
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
    ] = "ALL",  # pyright: ignore[reportArgumentType]
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

    Tables written depend on ``--dataset``: ``rates``, ``history_orders``,
    ``history_deals`` by default. ``ticks`` are opt-in: pass
    ``--dataset ticks`` to include them (tick data grows the database quickly).
    History datasets are fetched per symbol and concatenated. Rates rows carry
    the requested ``timeframe`` so appended runs at different timeframes remain
    distinguishable.

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
    datasets = set(dataset) if dataset is not None else None
    sdk.collect_history(
        output=export_ctx.output,
        symbols=symbol,
        date_from=date_from,
        date_to=date_to,
        datasets=datasets,
        timeframe=timeframe,
        flags=flags,
        if_exists=if_exists,
        with_views=with_views,
        config=export_ctx.config,
    )


@app.command(rich_help_panel="Collection")
def grafana_schema(ctx: typer.Context) -> None:
    """Create or refresh Grafana-ready views and indexes in a SQLite database.

    Idempotent — safe to run repeatedly on the same database. Requires SQLite
    output. Does not connect to MetaTrader 5.

    Raises:
        typer.BadParameter: If the output format is not SQLite3.
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415

    from .grafana import create_snapshot_tables, ensure_grafana_schema  # noqa: PLC0415

    export_ctx = _get_export_context(ctx)
    if export_ctx.output_format != "sqlite3":
        msg = (
            "grafana-schema requires SQLite3 output."
            " Use a .db/.sqlite/.sqlite3 extension or --format sqlite3."
        )
        raise typer.BadParameter(msg)
    with _sqlite3.connect(export_ctx.output) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        create_snapshot_tables(conn)
        ensure_grafana_schema(conn)
    logger.info("Grafana schema applied to %s", export_ctx.output)


@app.command(rich_help_panel="Collection")
def snapshot(
    ctx: typer.Context,
    symbol: Annotated[
        list[str] | None,
        typer.Option(
            "--symbol",
            "-s",
            help="Symbol filter for positions/orders (repeat for multiple).",
        ),
    ] = None,
    with_account: Annotated[
        bool,
        typer.Option("--with-account/--no-account", help="Snapshot account info."),
    ] = True,
    with_positions: Annotated[
        bool,
        typer.Option(
            "--with-positions/--no-positions", help="Snapshot open positions."
        ),
    ] = True,
    with_orders: Annotated[
        bool,
        typer.Option("--with-orders/--no-orders", help="Snapshot active orders."),
    ] = True,
    with_terminal: Annotated[
        bool,
        typer.Option("--with-terminal/--no-terminal", help="Snapshot terminal info."),
    ] = True,
    with_grafana_schema: Annotated[
        bool,
        typer.Option(
            "--with-grafana-schema/--no-grafana-schema",
            help="Ensure Grafana views and indexes exist.",
        ),
    ] = False,
) -> None:
    """Snapshot current account, position, order, and terminal state into SQLite.

    Appends a timestamped snapshot row for each data type. Never places
    orders or modifies trading state.

    Raises:
        typer.BadParameter: If the output format is not SQLite3.
    """
    export_ctx = _get_export_context(ctx)
    if export_ctx.output_format != "sqlite3":
        msg = (
            "snapshot requires SQLite3 output."
            " Use a .db/.sqlite/.sqlite3 extension or --format sqlite3."
        )
        raise typer.BadParameter(msg)
    sdk.update_observability_with_config(
        output=export_ctx.output,
        config=export_ctx.config,
        symbols=list(symbol) if symbol else None,
        include_account=with_account,
        include_positions=with_positions,
        include_orders=with_orders,
        include_terminal=with_terminal,
        with_grafana_schema=with_grafana_schema,
    )
    logger.info("Snapshot written to %s", export_ctx.output)


def main() -> None:
    """Run the mt5cli CLI."""
    app()
