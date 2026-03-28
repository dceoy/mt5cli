"""Command-line interface for MetaTrader 5 data export."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, cast

import click
import typer

from pdmt5 import Mt5Config, Mt5DataClient

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEFRAME_MAP: dict[str, int] = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M12": 12,
    "M15": 15,
    "M20": 20,
    "M30": 30,
    "H1": 16385,
    "H2": 16386,
    "H3": 16387,
    "H4": 16388,
    "H6": 16390,
    "H8": 16392,
    "H12": 16396,
    "D1": 16408,
    "W1": 32769,
    "MN1": 49153,
}

TICK_FLAG_MAP: dict[str, int] = {
    "ALL": 1,
    "INFO": 2,
    "TRADE": 4,
}

_FORMAT_EXTENSIONS: dict[str, str] = {
    ".csv": "csv",
    ".json": "json",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".db": "sqlite3",
    ".sqlite": "sqlite3",
    ".sqlite3": "sqlite3",
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OutputFormat(StrEnum):
    """Supported output file formats."""

    csv = "csv"
    json = "json"
    parquet = "parquet"
    sqlite3 = "sqlite3"


class LogLevel(StrEnum):
    """Logging verbosity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Click parameter types
# ---------------------------------------------------------------------------


class _DateTimeType(click.ParamType):
    """Click parameter type for ISO 8601 datetime strings."""

    name = "DATETIME"

    def convert(
        self,
        value: object,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> datetime:
        """Convert a string value to a timezone-aware datetime.

        Args:
            value: Raw value from the command line.
            param: Click parameter instance.
            ctx: Click context.

        Returns:
            Parsed datetime.
        """
        if isinstance(value, datetime):
            return value
        try:
            return parse_datetime(str(value))
        except ValueError as exc:
            self.fail(str(exc), param, ctx)


class _TimeframeType(click.ParamType):
    """Click parameter type for MT5 timeframe values."""

    name = "TIMEFRAME"

    def convert(
        self,
        value: object,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> int:
        """Convert a string or integer value to a timeframe integer.

        Args:
            value: Raw value from the command line.
            param: Click parameter instance.
            ctx: Click context.

        Returns:
            Integer timeframe value.
        """
        if isinstance(value, int):
            return value
        try:
            return parse_timeframe(str(value))
        except ValueError as exc:
            self.fail(str(exc), param, ctx)


class _TickFlagsType(click.ParamType):
    """Click parameter type for MT5 tick copy flags."""

    name = "FLAGS"

    def convert(
        self,
        value: object,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> int:
        """Convert a string or integer value to a tick flags integer.

        Args:
            value: Raw value from the command line.
            param: Click parameter instance.
            ctx: Click context.

        Returns:
            Integer tick flag value.
        """
        if isinstance(value, int):
            return value
        try:
            return parse_tick_flags(str(value))
        except ValueError as exc:
            self.fail(str(exc), param, ctx)


DATETIME_TYPE = _DateTimeType()
TIMEFRAME_TYPE = _TimeframeType()
TICK_FLAGS_TYPE = _TickFlagsType()

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
# Public utility functions
# ---------------------------------------------------------------------------


def detect_format(
    output_path: Path,
    explicit_format: str | None = None,
) -> str:
    """Detect the output format from a file extension or explicit format string.

    Args:
        output_path: Path to the output file.
        explicit_format: Explicitly specified format, if any.

    Returns:
        The detected format string.

    Raises:
        ValueError: If the format cannot be determined.
    """
    if explicit_format is not None:
        return explicit_format
    suffix = output_path.suffix.lower()
    if suffix in _FORMAT_EXTENSIONS:
        return _FORMAT_EXTENSIONS[suffix]
    msg = (
        f"Cannot detect format from extension '{suffix}'."
        " Use --format to specify the output format."
    )
    raise ValueError(msg)


def export_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    output_format: str,
    table_name: str = "data",
) -> None:
    """Export a pandas DataFrame to the specified file format.

    Args:
        df: DataFrame to export.
        output_path: Path to the output file.
        output_format: Output format (csv, json, parquet, or sqlite3).
        table_name: Table name for SQLite3 output.

    Raises:
        ValueError: If the output format is not supported.
    """
    if output_format == "csv":
        df.to_csv(output_path, index=False)
    elif output_format == "json":
        df.to_json(
            output_path,
            orient="records",
            date_format="iso",
            indent=2,
        )
    elif output_format == "parquet":
        df.to_parquet(output_path, index=False)
    elif output_format == "sqlite3":
        with sqlite3.connect(output_path) as conn:
            df.to_sql(  # type: ignore[reportUnknownMemberType]
                table_name,
                conn,
                if_exists="replace",
                index=False,
            )
    else:
        msg = f"Unsupported output format: {output_format}"
        raise ValueError(msg)


def parse_datetime(value: str) -> datetime:
    """Parse an ISO 8601 datetime string to a timezone-aware datetime.

    Args:
        value: ISO 8601 datetime string (e.g., '2024-01-01' or
            '2024-01-01T12:00:00+00:00').

    Returns:
        Parsed datetime with UTC timezone if no timezone is specified.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        msg = f"Invalid datetime format: '{value}'. Use ISO 8601 format."
        raise ValueError(msg) from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def parse_timeframe(value: str) -> int:
    """Parse a timeframe string or integer value.

    Args:
        value: Timeframe name (e.g., 'M1', 'H1', 'D1') or integer value.

    Returns:
        Integer timeframe value.

    Raises:
        ValueError: If the timeframe is invalid.
    """
    upper = value.upper()
    if upper in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[upper]
    try:
        return int(value)
    except ValueError:
        valid = ", ".join(TIMEFRAME_MAP)
        msg = f"Invalid timeframe: '{value}'. Use one of: {valid}, or an integer."
        raise ValueError(msg) from None


def parse_tick_flags(value: str) -> int:
    """Parse tick flags string or integer value.

    Args:
        value: Tick flag name (ALL, INFO, TRADE) or integer value.

    Returns:
        Integer tick flag value.

    Raises:
        ValueError: If the flag is invalid.
    """
    upper = value.upper()
    if upper in TICK_FLAG_MAP:
        return TICK_FLAG_MAP[upper]
    try:
        return int(value)
    except ValueError:
        valid = ", ".join(TICK_FLAG_MAP)
        msg = f"Invalid tick flags: '{value}'. Use one of: {valid}, or an integer."
        raise ValueError(msg) from None


# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="mt5cli",
    help="Export MetaTrader5 data to CSV, JSON, Parquet, or SQLite3.",
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


def main() -> None:
    """Run the mt5cli CLI."""
    app()
