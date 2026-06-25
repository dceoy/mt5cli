"""Utility constants, types, and functions for the mt5cli package."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

import click
from pdmt5 import COPY_TICKS_MAP, TIMEFRAME_MAP
from pdmt5 import parse_copy_ticks as _parse_copy_ticks
from pdmt5 import parse_timeframe as _parse_timeframe

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEFRAME_NAMES: tuple[str, ...] = tuple(
    name for name in TIMEFRAME_MAP if not name.startswith("TIMEFRAME_")
)
_TICK_FLAG_NAMES: tuple[str, ...] = tuple(
    name for name in COPY_TICKS_MAP if not name.startswith("COPY_TICKS_")
)

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


class Dataset(StrEnum):
    """Datasets supported by the ``collect-history`` command."""

    rates = "rates"
    ticks = "ticks"
    history_orders = "history-orders"
    history_deals = "history-deals"

    @property
    def table_name(self) -> str:
        """Return the SQLite table name for this dataset."""
        return self.value.replace("-", "_")


class IfExists(StrEnum):
    """SQLite table conflict behavior for the ``collect-history`` command."""

    APPEND = "append"
    REPLACE = "replace"
    FAIL = "fail"


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
        try:
            return parse_timeframe(value)
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
        try:
            return parse_tick_flags(value)
        except ValueError as exc:
            self.fail(str(exc), param, ctx)


class _RequestType(click.ParamType):
    """Click parameter type for JSON order requests."""

    name = "REQUEST"

    def convert(
        self,
        value: object,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> dict[str, Any]:
        """Convert a raw CLI value to an order request dictionary.

        Args:
            value: Raw value from the command line.
            param: Click parameter instance.
            ctx: Click context.

        Returns:
            Parsed request dictionary.
        """
        try:
            return parse_request(str(value))
        except ValueError as exc:
            self.fail(str(exc), param, ctx)


DATETIME_TYPE = _DateTimeType()
TIMEFRAME_TYPE = _TimeframeType()
TICK_FLAGS_TYPE = _TickFlagsType()
REQUEST_TYPE = _RequestType()

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


def coerce_login(login: int | str | None) -> int | None:
    """Coerce a login value to int, treating empty strings as unset.

    Returns:
        Integer login, or None when unset or an empty string.
    """
    if login is None or isinstance(login, int):
        return login
    text = login.strip()
    if not text:
        return None
    return int(text)


def export_dataframe_to_sqlite(
    df: pd.DataFrame,
    output_path: Path,
    table_name: str = "data",
    *,
    if_exists: IfExists = IfExists.APPEND,
    index: bool = False,
    index_label: str | None = None,
    deduplicate_on: Sequence[str] | None = None,
) -> None:
    """Write a DataFrame to SQLite with configurable append and deduplication.

    Args:
        df: DataFrame to export.
        output_path: SQLite database path.
        table_name: Target table name.
        if_exists: Conflict behavior when the table already exists.
        index: Whether to write the DataFrame index as a column.
        index_label: Column name for the index when ``index=True``.
        deduplicate_on: Optional key columns to deduplicate after writing,
            keeping the latest ``ROWID`` per key group. Deduplication scans the
            full table, so repeated appends cost O(table size); index the key
            columns when appending frequently.
    """
    with sqlite3.connect(output_path) as conn:
        df.to_sql(  # type: ignore[reportUnknownMemberType]
            table_name,
            conn,
            if_exists=if_exists.value,
            index=index,
            index_label=index_label,
        )
        if deduplicate_on:
            from .history import drop_duplicates_in_table  # noqa: PLC0415

            drop_duplicates_in_table(
                conn.cursor(),
                table_name,
                list(deduplicate_on),
                keep="last",
            )
            conn.commit()


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
        ImportError: If the parquet format is requested but pyarrow is not installed.
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
        try:
            __import__("pyarrow")
        except ImportError as exc:
            msg = (
                "Parquet export requires the optional dependency pyarrow. "
                'Install it with: pip install "mt5cli[parquet]"'
            )
            raise ImportError(msg) from exc
        df.to_parquet(output_path, index=False)
    elif output_format == "sqlite3":
        export_dataframe_to_sqlite(
            df,
            output_path,
            table_name,
            if_exists=IfExists.REPLACE,
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


def parse_timeframe(value: object) -> int:
    """Parse a timeframe string or integer value.

    Args:
        value: Timeframe name (e.g., 'M1', 'H1', 'D1') or integer value.

    Returns:
        Integer timeframe value.

    Raises:
        ValueError: If the timeframe is invalid.
    """
    try:
        return _parse_timeframe(value)
    except ValueError:
        display = value if isinstance(value, str) else repr(value)
        valid = ", ".join(TIMEFRAME_NAMES)
        msg = (
            f"Invalid timeframe: '{display}'. "
            f"Use one of: {valid}, or a supported integer."
        )
        raise ValueError(msg) from None


def parse_tick_flags(value: object) -> int:
    """Parse tick flags string or integer value.

    Args:
        value: Tick flag name (ALL, INFO, TRADE, COPY_TICKS_*) or integer value.

    Returns:
        Integer tick flag value compatible with MetaTrader 5 ``COPY_TICKS_*``.

    Raises:
        ValueError: If the flag is invalid.
    """
    try:
        return _parse_copy_ticks(value)
    except ValueError:
        display = value if isinstance(value, str) else repr(value)
        valid = ", ".join(_TICK_FLAG_NAMES)
        msg = (
            f"Invalid tick flags: '{display}'. "
            f"Use one of: {valid}, or a supported integer."
        )
        raise ValueError(msg) from None


def _is_request_dict(value: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict)


def parse_request(value: str) -> dict[str, Any]:
    """Parse a JSON-formatted order request string or file reference.

    Args:
        value: JSON object string, or '@path' to read JSON from a file.

    Returns:
        Parsed request dictionary.

    Raises:
        ValueError: If the request file cannot be read or the value is not a
            JSON object.
    """
    if value.startswith("@"):
        path = Path(value[1:])
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            msg = f"Failed to read JSON request file '{path}': {exc}"
            raise ValueError(msg) from exc
    else:
        text = value
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON request: {exc}"
        raise ValueError(msg) from exc
    if not _is_request_dict(parsed):
        msg = "Order request must be a JSON object."
        raise ValueError(msg)
    return parsed
