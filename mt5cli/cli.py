"""Command-line interface for MetaTrader 5 data export."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, cast

import pandas as pd
import typer
from pdmt5 import Mt5Config

from . import sdk
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
    help="Export MetaTrader5 data to CSV, JSON, Parquet, or SQLite3.",
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


def _sdk_client(ctx: typer.Context) -> sdk.Mt5CliClient:
    export_ctx = _get_export_context(ctx)
    return sdk.Mt5CliClient(config=export_ctx.config)


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
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.copy_rates_from(symbol, timeframe, date_from, count),
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
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.copy_rates_from_pos(symbol, timeframe, start_pos, count),
    )


@app.command()
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
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.latest_rates(symbol, timeframe, count, start_pos=start_pos),
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
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.copy_rates_range(symbol, timeframe, date_from, date_to),
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
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.copy_ticks_from(symbol, date_from, count, flags),
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
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.copy_ticks_range(symbol, date_from, date_to, flags),
    )


@app.command()
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
    ] = 1,
) -> None:
    """Export ticks from a recent time window."""
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.recent_ticks(
            symbol,
            seconds,
            date_to=date_to,
            count=count,
            flags=flags,
        ),
    )


@app.command()
def account_info(ctx: typer.Context) -> None:
    """Export account information."""
    _execute_export(ctx, _sdk_client(ctx).account_info)


@app.command()
def terminal_info(ctx: typer.Context) -> None:
    """Export terminal information."""
    _execute_export(ctx, _sdk_client(ctx).terminal_info)


@app.command()
def symbols(
    ctx: typer.Context,
    group: Annotated[
        str | None,
        typer.Option(help="Symbol group filter (e.g., *USD*)."),
    ] = None,
) -> None:
    """Export symbol list."""
    client = _sdk_client(ctx)
    _execute_export(ctx, lambda: client.symbols(group=group))


@app.command()
def symbol_info(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export symbol details."""
    client = _sdk_client(ctx)
    _execute_export(ctx, lambda: client.symbol_info(symbol))


@app.command()
def minimum_margins(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export minimum-volume buy and sell margin requirements."""
    client = _sdk_client(ctx)
    _execute_export(ctx, lambda: client.minimum_margins(symbol))


@app.command()
def orders(
    ctx: typer.Context,
    symbol: Annotated[str | None, typer.Option(help="Symbol filter.")] = None,
    group: Annotated[str | None, typer.Option(help="Group filter.")] = None,
    ticket: Annotated[int | None, typer.Option(help="Ticket filter.")] = None,
) -> None:
    """Export active orders."""
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.orders(symbol=symbol, group=group, ticket=ticket),
    )


@app.command()
def positions(
    ctx: typer.Context,
    symbol: Annotated[str | None, typer.Option(help="Symbol filter.")] = None,
    group: Annotated[str | None, typer.Option(help="Group filter.")] = None,
    ticket: Annotated[int | None, typer.Option(help="Ticket filter.")] = None,
) -> None:
    """Export open positions."""
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.positions(symbol=symbol, group=group, ticket=ticket),
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
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.history_orders(
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
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.history_deals(
            date_from=date_from,
            date_to=date_to,
            group=group,
            symbol=symbol,
            ticket=ticket,
            position=position,
        ),
    )


@app.command()
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
    client = _sdk_client(ctx)
    _execute_export(
        ctx,
        lambda: client.recent_history_deals(
            hours,
            date_to=date_to,
            group=group,
            symbol=symbol,
        ),
    )


@app.command()
def mt5_summary(ctx: typer.Context) -> None:
    """Export a compact terminal/account status summary."""
    client = _sdk_client(ctx)
    _execute_export(ctx, lambda: pd.DataFrame([client.mt5_summary()]))


@app.command()
def version(ctx: typer.Context) -> None:
    """Export MetaTrader5 version information."""
    _execute_export(ctx, _sdk_client(ctx).version)


@app.command()
def last_error(ctx: typer.Context) -> None:
    """Export the last error information."""
    _execute_export(ctx, _sdk_client(ctx).last_error)


@app.command()
def symbol_info_tick(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export the last tick for a symbol."""
    client = _sdk_client(ctx)
    _execute_export(ctx, lambda: client.symbol_info_tick(symbol))


@app.command()
def market_book(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Option(help="Symbol name.")],
) -> None:
    """Export market depth (order book) for a symbol."""
    client = _sdk_client(ctx)
    _execute_export(ctx, lambda: client.market_book(symbol))


@app.command()
def order_check(
    ctx: typer.Context,
    request: Annotated[
        dict[str, Any],
        typer.Option(click_type=REQUEST_TYPE, help=_REQUEST_OPTION_HELP),
    ],
) -> None:
    """Check funds sufficiency for a trading operation."""
    export_ctx = _get_export_context(ctx)

    def _fetch() -> pd.DataFrame:
        return sdk._run_with_client(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            export_ctx.config,
            lambda c: c.order_check_as_df(request=request),
        )

    _execute_export(ctx, _fetch)


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
    export_ctx = _get_export_context(ctx)

    def _fetch() -> pd.DataFrame:
        return sdk._run_with_client(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            export_ctx.config,
            lambda c: c.order_send_as_df(request=request),
        )

    _execute_export(ctx, _fetch)


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


def main() -> None:
    """Run the mt5cli CLI."""
    app()
