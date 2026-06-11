"""Programmatic SDK for MetaTrader 5 data collection."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypeVar, cast

import pandas as pd
from pdmt5 import Mt5Config, Mt5DataClient, Mt5RuntimeError, Mt5TradingError

from .history import (
    create_cash_events_view,
    create_history_indexes,
    create_positions_reconstructed_view,
    drop_forming_rate_bar,
    resolve_granularity_name,
    resolve_history_datasets,
    resolve_history_tick_flags,
    resolve_history_timeframes,
    write_collected_datasets,
    write_incremental_datasets,
)
from .utils import (
    Dataset,
    IfExists,
    parse_datetime,
    parse_tick_flags,
    parse_timeframe,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

T = TypeVar("T")

logger = logging.getLogger(__name__)

_RECOVERABLE_HISTORY_UPDATE_ERRORS: tuple[type[BaseException], ...] = (
    Mt5TradingError,
    Mt5RuntimeError,
    sqlite3.Error,
    ValueError,
    OSError,
)

_MT5_CLIENT_CAPABILITY_METHODS: frozenset[str] = frozenset({
    "copy_rates_range_as_df",
    "copy_ticks_range_as_df",
    "history_deals_get_as_df",
    "history_orders_get_as_df",
})
_MT5_HISTORY_MODULE = Path(__file__).with_name("history.py")
_MT5_HISTORY_CLIENT_CALL_FUNCTIONS: frozenset[str] = frozenset({
    "write_rates_dataset",
    "write_ticks_dataset",
    "write_history_dataset",
    "_write_incremental_history_deals",
})
_NON_CALLABLE_TYPE_ERROR = re.compile(r"^'[^']+' object is not callable$")


def _is_non_callable_history_client_type_error(exc: TypeError) -> bool:
    """Return whether a TypeError came from calling a history client API attribute."""
    if not _NON_CALLABLE_TYPE_ERROR.match(str(exc)):
        return False
    history_module = _MT5_HISTORY_MODULE.resolve()
    for frame, _lineno in traceback.walk_tb(exc.__traceback__):
        if (
            frame.f_code.co_name in _MT5_HISTORY_CLIENT_CALL_FUNCTIONS
            and Path(frame.f_code.co_filename).resolve() == history_module
        ):
            return True
    return False


def _is_mt5_client_capability_error(exc: BaseException) -> bool:
    """Return whether an error indicates an incompatible MT5 client API surface."""
    if isinstance(exc, AttributeError):
        msg = str(exc)
        if msg.startswith("MT5 client is missing required method:"):
            return True
        name = getattr(exc, "name", None)
        return isinstance(name, str) and name in _MT5_CLIENT_CAPABILITY_METHODS
    if isinstance(exc, TypeError):
        msg = str(exc)
        if msg.startswith("MT5 client attribute is not callable:"):
            return True
        return _is_non_callable_history_client_type_error(exc)
    return False


__all__ = [
    "AccountSpec",
    "Mt5CliClient",
    "ThrottledHistoryUpdater",
    "account_info",
    "build_config",
    "collect_history",
    "collect_latest_closed_rates_by_granularity",
    "collect_latest_closed_rates_for_accounts",
    "collect_latest_rates",
    "collect_latest_rates_for_accounts",
    "collect_latest_rates_for_accounts_with_retries",
    "copy_rates_from",
    "copy_rates_from_pos",
    "copy_rates_range",
    "copy_ticks_from",
    "copy_ticks_range",
    "history_deals",
    "history_orders",
    "last_error",
    "latest_rates",
    "market_book",
    "minimum_margins",
    "mt5_session",
    "mt5_summary",
    "mt5_summary_as_df",
    "orders",
    "positions",
    "recent_history_deals",
    "recent_ticks",
    "resolve_account_spec",
    "resolve_account_specs",
    "substitute_env_placeholders",
    "symbol_info",
    "symbol_info_tick",
    "symbols",
    "terminal_info",
    "update_history",
    "update_history_with_config",
    "version",
]


def _coerce_timeframe(timeframe: int | str) -> int:
    if isinstance(timeframe, int):
        return timeframe
    return parse_timeframe(timeframe)


def _coerce_tick_flags(flags: int | str) -> int:
    if isinstance(flags, int):
        return flags
    return parse_tick_flags(flags)


def _plain_mt5_value(value: object) -> object:
    asdict = getattr(value, "_asdict", None)
    if callable(asdict):
        return _plain_mt5_value(asdict())
    if isinstance(value, dict):
        typed_value = cast("dict[object, object]", value)
        return {key: _plain_mt5_value(item) for key, item in typed_value.items()}
    if isinstance(value, tuple):
        typed_value = cast("tuple[object, ...]", value)
        return [_plain_mt5_value(item) for item in typed_value]
    if isinstance(value, list):
        typed_value = cast("list[object]", value)
        return [_plain_mt5_value(item) for item in typed_value]
    return value


def _require_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return parse_datetime(value)


def _coerce_datetime(value: datetime | str | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return parse_datetime(value)


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        msg = f"{name} must be positive."
        raise ValueError(msg)


def _require_non_negative(value: int, name: str) -> None:
    if value < 0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)


def _call_required_client_method(client: Mt5DataClient, name: str) -> object:
    try:
        method = getattr(client, name)
    except AttributeError as exc:
        msg = f"MT5 client is missing required method: {name}"
        raise AttributeError(msg) from exc
    if not callable(method):
        msg = f"MT5 client attribute is not callable: {name}"
        raise TypeError(msg)
    return method()


def _mt5_summary_export_value(value: object) -> object:
    plain_value = _plain_mt5_value(value)
    if isinstance(plain_value, dict | list):
        return json.dumps(plain_value, sort_keys=True, separators=(",", ":"))
    return plain_value


def _coerce_tick_time(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return parse_datetime(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    msg = f"Unsupported tick time value: {value!r}"
    raise TypeError(msg)


def _filter_ticks_to_end(frame: pd.DataFrame, end: datetime) -> pd.DataFrame:
    if frame.empty or "time" not in frame.columns:
        return frame
    times = pd.to_datetime(frame["time"], utc=True)
    return frame.loc[times <= end].reset_index(drop=True)


def _fetch_recent_ticks(
    client: Mt5DataClient,
    symbol: str,
    seconds: float,
    date_to: datetime | None,
    count: int,
    flags: int,
) -> pd.DataFrame:
    if date_to is not None:
        end = date_to
    else:
        tick = client.symbol_info_tick(symbol)
        end = _coerce_tick_time(tick.time)
    start = end - timedelta(seconds=seconds)
    if count > 0:
        from_frame = _filter_ticks_to_end(
            client.copy_ticks_from_as_df(
                symbol=symbol,
                date_from=start,
                count=count,
                flags=flags,
            ),
            end,
        )
        if len(from_frame) < count:
            return from_frame
    frame = client.copy_ticks_range_as_df(
        symbol=symbol,
        date_from=start,
        date_to=end,
        flags=flags,
    )
    if count > 0 and len(frame) > count:
        return frame.tail(count).reset_index(drop=True)
    return frame


def _fetch_minimum_margins(client: Mt5DataClient, symbol: str) -> pd.DataFrame:
    sym = client.symbol_info(symbol)
    account = client.account_info()
    tick = client.symbol_info_tick(symbol)
    volume_min = sym.volume_min
    buy_margin = client.order_calc_margin(
        client.mt5.ORDER_TYPE_BUY,
        symbol,
        volume_min,
        tick.ask,
    )
    sell_margin = client.order_calc_margin(
        client.mt5.ORDER_TYPE_SELL,
        symbol,
        volume_min,
        tick.bid,
    )
    return pd.DataFrame([
        {
            "symbol": symbol,
            "account_currency": account.currency,
            "volume_min": volume_min,
            "buy_margin": buy_margin,
            "sell_margin": sell_margin,
        }
    ])


def build_config(
    *,
    path: str | None = None,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    timeout: int | None = None,
) -> Mt5Config:
    """Build an ``Mt5Config`` from optional connection parameters.

    Returns:
        Configured ``Mt5Config`` instance.
    """
    return Mt5Config(
        path=path,
        login=login,
        password=password,
        server=server,
        timeout=timeout,
    )


@contextmanager
def _connected_client(config: Mt5Config) -> Iterator[Mt5DataClient]:
    """Initialize MT5, yield a connected client, and always shut down.

    Args:
        config: MT5 connection configuration.

    Yields:
        Connected ``Mt5DataClient`` instance.
    """
    client = Mt5DataClient(config=config)
    try:
        client.initialize_and_login_mt5()
        yield client
    finally:
        client.shutdown()


def _run_with_client(
    config: Mt5Config,
    fetch_fn: Callable[[Mt5DataClient], T],
) -> T:
    """Connect, run ``fetch_fn``, and shut down safely.

    Args:
        config: MT5 connection configuration.
        fetch_fn: Callable receiving a connected client.

    Returns:
        Value returned by ``fetch_fn``.
    """
    with _connected_client(config) as client:
        return fetch_fn(client)


@contextmanager
def mt5_session(config: Mt5Config | None = None) -> Iterator[Mt5CliClient]:
    """Open an MT5 terminal session and yield a connected client.

    Launches the MetaTrader 5 terminal using ``Mt5Config.path`` (when set),
    logs in, yields a connected :class:`Mt5CliClient`, and always shuts the
    terminal down on exit.

    Args:
        config: MT5 connection configuration. Defaults to an empty config that
            attaches to a running terminal.

    Yields:
        Connected ``Mt5CliClient`` bound to the session.
    """
    mt5_config = config or build_config()
    with _connected_client(mt5_config) as client:
        yield Mt5CliClient.from_connected_client(client)


class Mt5CliClient:
    """Programmatic client for read-only MetaTrader 5 data access."""

    def __init__(
        self,
        *,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout: int | None = None,
        config: Mt5Config | None = None,
        client: Mt5DataClient | None = None,
    ) -> None:
        """Initialize the SDK client.

        Args:
            path: Path to MetaTrader5 terminal EXE file.
            login: Trading account login.
            password: Trading account password.
            server: Trading server name.
            timeout: Connection timeout in milliseconds.
            config: Optional pre-built ``Mt5Config`` (overrides other args).
            client: Optional already-connected ``Mt5DataClient``. Injected
                clients are reused as-is and are not initialized or shut down.
        """
        self._config = config or build_config(
            path=path,
            login=login,
            password=password,
            server=server,
            timeout=timeout,
        )
        self._client = client
        self._owns_client = client is None

    @classmethod
    def from_connected_client(cls, client: Mt5DataClient) -> Self:
        """Bind to an already-connected ``Mt5DataClient`` without owning it.

        The returned ``Mt5CliClient`` never initializes or shuts down the
        injected client, including when used as a context manager.

        Returns:
            Client wrapper bound to the injected connection.
        """
        return cls(client=client)

    @property
    def config(self) -> Mt5Config:
        """Return the underlying MT5 configuration."""
        return self._config

    def __enter__(self) -> Self:
        """Open a persistent MT5 connection for multiple calls.

        Returns:
            This client instance.
        """
        if self._client is not None:
            return self
        client = Mt5DataClient(config=self._config)
        try:
            client.initialize_and_login_mt5()
        except Exception:
            client.shutdown()
            raise
        self._client = client
        self._owns_client = True  # only set when this method created the client
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Shut down the persistent MT5 connection."""
        if self._client is not None and self._owns_client:
            self._client.shutdown()
            self._client = None

    def _fetch_value(self, fetch_fn: Callable[[Mt5DataClient], T]) -> T:
        if self._client is not None:
            return fetch_fn(self._client)
        return _run_with_client(self._config, fetch_fn)

    def _fetch(self, fetch_fn: Callable[[Mt5DataClient], pd.DataFrame]) -> pd.DataFrame:
        return self._fetch_value(fetch_fn)

    def copy_rates_from(
        self,
        symbol: str,
        timeframe: int | str,
        date_from: datetime | str,
        count: int,
    ) -> pd.DataFrame:
        """Return rates starting from a date."""
        tf = _coerce_timeframe(timeframe)
        start = _require_datetime(date_from)
        return self._fetch(
            lambda c: c.copy_rates_from_as_df(
                symbol=symbol,
                timeframe=tf,
                date_from=start,
                count=count,
            ),
        )

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: int | str,
        start_pos: int,
        count: int,
    ) -> pd.DataFrame:
        """Return rates starting from a bar position."""
        tf = _coerce_timeframe(timeframe)
        return self._fetch(
            lambda c: c.copy_rates_from_pos_as_df(
                symbol=symbol,
                timeframe=tf,
                start_pos=start_pos,
                count=count,
            ),
        )

    def latest_rates(
        self,
        symbol: str,
        timeframe: int | str,
        count: int,
        start_pos: int = 0,
    ) -> pd.DataFrame:
        """Return the latest rates from a bar position."""
        _require_positive(count, "count")
        return self.copy_rates_from_pos(symbol, timeframe, start_pos, count)

    def collect_latest_rates(
        self,
        symbols: Sequence[str],
        timeframes: Sequence[int | str],
        *,
        count: int,
        start_pos: int = 0,
    ) -> dict[tuple[str, int], pd.DataFrame]:
        """Return latest rates for each symbol/timeframe pair.

        Returns:
            Mapping keyed by ``(symbol, timeframe_int)``.

        Raises:
            ValueError: If ``count`` is not positive or inputs are empty.
        """
        _require_positive(count, "count")
        if not symbols:
            msg = "At least one symbol is required."
            raise ValueError(msg)
        if not timeframes:
            msg = "At least one timeframe is required."
            raise ValueError(msg)
        resolved_timeframes = [_coerce_timeframe(timeframe) for timeframe in timeframes]
        return self._fetch_value(
            lambda c: {
                (symbol, timeframe): c.copy_rates_from_pos_as_df(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_pos=start_pos,
                    count=count,
                )
                for symbol in symbols
                for timeframe in resolved_timeframes
            },
        )

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: int | str,
        date_from: datetime | str,
        date_to: datetime | str,
    ) -> pd.DataFrame:
        """Return rates for a date range."""
        tf = _coerce_timeframe(timeframe)
        start = _require_datetime(date_from)
        end = _require_datetime(date_to)
        return self._fetch(
            lambda c: c.copy_rates_range_as_df(
                symbol=symbol,
                timeframe=tf,
                date_from=start,
                date_to=end,
            ),
        )

    def copy_ticks_from(
        self,
        symbol: str,
        date_from: datetime | str,
        count: int,
        flags: int | str,
    ) -> pd.DataFrame:
        """Return ticks starting from a date."""
        start = _require_datetime(date_from)
        tick_flags = _coerce_tick_flags(flags)
        return self._fetch(
            lambda c: c.copy_ticks_from_as_df(
                symbol=symbol,
                date_from=start,
                count=count,
                flags=tick_flags,
            ),
        )

    def copy_ticks_range(
        self,
        symbol: str,
        date_from: datetime | str,
        date_to: datetime | str,
        flags: int | str,
    ) -> pd.DataFrame:
        """Return ticks for a date range."""
        start = _require_datetime(date_from)
        end = _require_datetime(date_to)
        tick_flags = _coerce_tick_flags(flags)
        return self._fetch(
            lambda c: c.copy_ticks_range_as_df(
                symbol=symbol,
                date_from=start,
                date_to=end,
                flags=tick_flags,
            ),
        )

    def account_info(self) -> pd.DataFrame:
        """Return account information."""
        return self._fetch(lambda c: c.account_info_as_df())

    def terminal_info(self) -> pd.DataFrame:
        """Return terminal information."""
        return self._fetch(lambda c: c.terminal_info_as_df())

    def symbols(self, group: str | None = None) -> pd.DataFrame:
        """Return the symbol list."""
        return self._fetch(lambda c: c.symbols_get_as_df(group=group))

    def symbol_info(self, symbol: str) -> pd.DataFrame:
        """Return details for one symbol."""
        return self._fetch(lambda c: c.symbol_info_as_df(symbol=symbol))

    def orders(
        self,
        symbol: str | None = None,
        group: str | None = None,
        ticket: int | None = None,
    ) -> pd.DataFrame:
        """Return active orders."""
        return self._fetch(
            lambda c: c.orders_get_as_df(
                symbol=symbol,
                group=group,
                ticket=ticket,
            ),
        )

    def positions(
        self,
        symbol: str | None = None,
        group: str | None = None,
        ticket: int | None = None,
    ) -> pd.DataFrame:
        """Return open positions."""
        return self._fetch(
            lambda c: c.positions_get_as_df(
                symbol=symbol,
                group=group,
                ticket=ticket,
            ),
        )

    def history_orders(
        self,
        date_from: datetime | str | None = None,
        date_to: datetime | str | None = None,
        group: str | None = None,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ) -> pd.DataFrame:
        """Return historical orders."""
        start = _coerce_datetime(date_from)
        end = _coerce_datetime(date_to)
        return self._fetch(
            lambda c: c.history_orders_get_as_df(
                date_from=start,
                date_to=end,
                group=group,
                symbol=symbol,
                ticket=ticket,
                position=position,
            ),
        )

    def history_deals(
        self,
        date_from: datetime | str | None = None,
        date_to: datetime | str | None = None,
        group: str | None = None,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ) -> pd.DataFrame:
        """Return historical deals."""
        start = _coerce_datetime(date_from)
        end = _coerce_datetime(date_to)
        return self._fetch(
            lambda c: c.history_deals_get_as_df(
                date_from=start,
                date_to=end,
                group=group,
                symbol=symbol,
                ticket=ticket,
                position=position,
            ),
        )

    def recent_history_deals(
        self,
        hours: float,
        date_to: datetime | str | None = None,
        group: str | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Return historical deals from a recent trailing window."""
        _require_positive(hours, "hours")
        end = _require_datetime(date_to) if date_to is not None else datetime.now(UTC)
        start = end - timedelta(hours=hours)
        return self.history_deals(
            date_from=start,
            date_to=end,
            group=group,
            symbol=symbol,
        )

    def version(self) -> pd.DataFrame:
        """Return MetaTrader5 version information."""
        return self._fetch(lambda c: c.version_as_df())

    def last_error(self) -> pd.DataFrame:
        """Return the last error information."""
        return self._fetch(lambda c: c.last_error_as_df())

    def symbol_info_tick(self, symbol: str) -> pd.DataFrame:
        """Return the last tick for a symbol."""
        return self._fetch(lambda c: c.symbol_info_tick_as_df(symbol=symbol))

    def market_book(self, symbol: str) -> pd.DataFrame:
        """Return market depth for a symbol."""
        return self._fetch(lambda c: c.market_book_get_as_df(symbol=symbol))

    def recent_ticks(
        self,
        symbol: str,
        seconds: float,
        *,
        date_to: datetime | str | None = None,
        count: int = 10000,
        flags: int | str = "ALL",
    ) -> pd.DataFrame:
        """Return ticks from a recent time window.

        Args:
            symbol: Symbol name.
            seconds: Lookback window in seconds ending at ``date_to``.
            date_to: Window end time. When ``None``, uses the latest
                ``symbol_info_tick().time`` rather than wall-clock now.
            count: Maximum ticks to return. Values ``<= 0`` return the full
                window without trimming. Positive values keep the most recent
                ticks; when the window is sparse, ``copy_ticks_from`` avoids
                fetching the entire range.
            flags: Tick flags as ``ALL``, ``INFO``, ``TRADE``, or an integer.

        Returns:
            Tick DataFrame with MT5 tick columns such as ``time``, ``bid``,
            ``ask``, ``last``, and ``volume``.
        """
        tick_flags = _coerce_tick_flags(flags)
        end = _coerce_datetime(date_to)
        return self._fetch(
            lambda c: _fetch_recent_ticks(
                c,
                symbol,
                seconds,
                end,
                count,
                tick_flags,
            ),
        )

    def minimum_margins(self, symbol: str) -> pd.DataFrame:
        """Return minimum-volume buy and sell margin requirements.

        Args:
            symbol: Symbol name.

        Returns:
            One-row DataFrame with columns ``symbol``, ``account_currency``,
            ``volume_min``, ``buy_margin``, and ``sell_margin``.
        """
        return self._fetch(lambda c: _fetch_minimum_margins(c, symbol))

    def mt5_summary(self) -> dict[str, object]:
        """Return a compact terminal/account status summary."""

        def _summary(client: Mt5DataClient) -> dict[str, object]:
            return {
                "version": _plain_mt5_value(
                    _call_required_client_method(client, "version"),
                ),
                "terminal_info": _plain_mt5_value(
                    _call_required_client_method(client, "terminal_info"),
                ),
                "account_info": _plain_mt5_value(
                    _call_required_client_method(client, "account_info"),
                ),
                "symbols_total": _plain_mt5_value(
                    _call_required_client_method(client, "symbols_total"),
                ),
            }

        return self._fetch_value(_summary)

    def mt5_summary_as_df(self) -> pd.DataFrame:
        """Return an export-safe one-row terminal/account summary DataFrame."""
        summary = self.mt5_summary()
        return pd.DataFrame(
            [
                {
                    key: _mt5_summary_export_value(value)
                    for key, value in summary.items()
                },
            ],
        )


def _resolve_incremental_settings(
    selected_datasets: set[Dataset],
    timeframes: Sequence[int | str] | None,
    flags: int | str,
) -> tuple[list[int], int]:
    """Resolve dataset-specific incremental update settings.

    Returns:
        Tuple of resolved rate timeframes and tick copy flags.

    Raises:
        ValueError: If timeframe or tick flag values are invalid.
    """
    resolved_timeframes: list[int] = []
    if Dataset.rates in selected_datasets:
        try:
            resolved_timeframes = resolve_history_timeframes(timeframes)
        except ValueError as exc:
            msg = str(exc)
            raise ValueError(msg) from exc
    resolved_tick_flags = 0
    if Dataset.ticks in selected_datasets:
        try:
            resolved_tick_flags = resolve_history_tick_flags(flags)
        except ValueError as exc:
            msg = str(exc)
            raise ValueError(msg) from exc
    return resolved_timeframes, resolved_tick_flags


@dataclass(frozen=True)
class _UpdateHistoryRequest:
    selected: set[Dataset]
    end: datetime
    fallback_start: datetime
    resolved_timeframes: list[int]
    resolved_tick_flags: int
    output_path: Path


def _resolve_update_history_request(
    *,
    output: Path | str,
    symbols: Sequence[str],
    datasets: set[Dataset] | None,
    timeframes: Sequence[int | str] | None,
    flags: int | str,
    lookback_hours: float,
    date_to: datetime | str | None,
) -> _UpdateHistoryRequest | None:
    """Validate and resolve incremental history update inputs.

    Returns:
        Resolved request parameters, or None when no datasets are selected.

    Raises:
        ValueError: If symbols are empty, lookback_hours is not positive, or
            timeframe/flag values are invalid.
    """
    if lookback_hours <= 0:
        msg = "lookback_hours must be positive."
        raise ValueError(msg)
    selected = resolve_history_datasets(datasets)
    if not selected:
        logger.info("Skipping SQLite history update: no datasets selected.")
        return None
    if not symbols:
        msg = "At least one symbol is required."
        raise ValueError(msg)

    if date_to is not None:
        resolved_end = _coerce_datetime(date_to)
    else:
        resolved_end = datetime.now(UTC)
    end = resolved_end if resolved_end is not None else datetime.now(UTC)
    fallback_start = end - timedelta(hours=lookback_hours)
    resolved_timeframes, resolved_tick_flags = _resolve_incremental_settings(
        selected,
        timeframes,
        flags,
    )
    return _UpdateHistoryRequest(
        selected=selected,
        end=end,
        fallback_start=fallback_start,
        resolved_timeframes=resolved_timeframes,
        resolved_tick_flags=resolved_tick_flags,
        output_path=Path(output),
    )


def update_history(  # noqa: PLR0913
    *,
    client: Mt5DataClient,
    output: Path | str,
    symbols: Sequence[str],
    datasets: set[Dataset] | None = None,
    timeframes: Sequence[int | str] | None = None,
    flags: int | str = "ALL",
    lookback_hours: float = 24.0,
    date_to: datetime | str | None = None,
    deduplicate: bool = True,
    create_rate_views: bool = True,
    with_views: bool = False,
    include_account_events: bool = True,
) -> None:
    """Incrementally append MT5 history into a SQLite database.

    Uses an already-connected ``Mt5DataClient`` and does not create or close
    the MT5 connection. For first-time tables, data is fetched from
    ``date_to - lookback_hours``. Subsequent runs resume from existing
    ``MAX(time)`` per symbol (and timeframe for rates); when
    ``include_account_events=True``, account-level deals use a separate cursor
    over ``type NOT IN (0, 1)`` / empty-symbol rows.

    Args:
        client: Connected MT5 data client.
        output: SQLite database path.
        symbols: Symbols to update.
        datasets: Datasets to include (defaults to all).
        timeframes: Rate timeframes to update (defaults to all fixed MT5
            timeframes when None).
        flags: Tick copy flags as integer or name (e.g. ``ALL``).
        lookback_hours: First-run lookback when a table has no prior rows.
        date_to: Optional update end datetime. Defaults to now (UTC).
        deduplicate: Remove duplicate rows after append, keeping latest ROWID.
        create_rate_views: Create ``rate_<symbol>__<timeframe>`` views.
        with_views: Create ``cash_events`` and ``positions_reconstructed`` views.
        include_account_events: Include account-level cash events in
            ``history_deals`` when True.
    """
    request = _resolve_update_history_request(
        output=output,
        symbols=symbols,
        datasets=datasets,
        timeframes=timeframes,
        flags=flags,
        lookback_hours=lookback_hours,
        date_to=date_to,
    )
    if request is None:
        return
    logger.info(
        "Updating history in SQLite: symbols=%s, datasets=%s, path=%s",
        list(symbols),
        sorted(dataset.value for dataset in request.selected),
        request.output_path,
    )
    with sqlite3.connect(request.output_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        write_incremental_datasets(
            conn,
            client,
            symbols,
            request.selected,
            request.resolved_timeframes,
            request.resolved_tick_flags,
            request.fallback_start,
            request.end,
            deduplicate=deduplicate,
            create_rate_views=create_rate_views,
            with_views=with_views,
            include_account_events=include_account_events,
        )


def update_history_with_config(  # noqa: PLR0913
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
    create_rate_views: bool = True,
    with_views: bool = False,
    include_account_events: bool = True,
) -> None:
    """Incrementally append MT5 history, opening and closing the MT5 connection.

    Convenience wrapper around :func:`update_history` for standalone use.
    """
    request = _resolve_update_history_request(
        output=output,
        symbols=symbols,
        datasets=datasets,
        timeframes=timeframes,
        flags=flags,
        lookback_hours=lookback_hours,
        date_to=date_to,
    )
    if request is None:
        return
    mt5_config = config or build_config()
    with _connected_client(mt5_config) as client:
        update_history(
            client=client,
            output=output,
            symbols=symbols,
            datasets=datasets,
            timeframes=timeframes,
            flags=flags,
            lookback_hours=lookback_hours,
            date_to=date_to,
            deduplicate=deduplicate,
            create_rate_views=create_rate_views,
            with_views=with_views,
            include_account_events=include_account_events,
        )


class ThrottledHistoryUpdater:
    """Throttled incremental SQLite history updater for long-running apps.

    Wraps :func:`update_history` with a minimum interval between successful
    updates, so a tight application loop can call :meth:`update` every
    iteration without re-fetching MT5 history more often than desired. Timing
    uses a monotonic clock, so it is unaffected by wall-clock changes.
    """

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
    ) -> None:
        """Initialize the throttled updater.

        Args:
            output: SQLite database path.
            datasets: Datasets to include (defaults to all).
            timeframes: Rate timeframes to update (defaults to all fixed MT5
                timeframes).
            flags: Tick copy flags as integer or name (e.g. ``ALL``).
            lookback_hours: First-run lookback when a table has no prior rows.
            with_views: Create ``cash_events`` and ``positions_reconstructed``
                views.
            include_account_events: Include account-level cash events.
            interval_seconds: Minimum seconds between successful updates. Values
                ``<= 0`` update on every call.
            suppress_errors: When True, recoverable errors (``Mt5TradingError``,
                ``Mt5RuntimeError``, ``sqlite3.Error``, ``ValueError``,
                ``OSError``, and MT5 client capability ``AttributeError`` /
                ``TypeError`` for history API methods) raised during an update
                are swallowed and :meth:`update` returns False without advancing
                the throttle. Other ``AttributeError`` / ``TypeError`` values
                always propagate. When False (default), recoverable errors
                propagate so callers control logging.
        """
        self.output = output
        self.datasets = datasets
        self.timeframes = timeframes
        self.flags = flags
        self.lookback_hours = lookback_hours
        self.with_views = with_views
        self.include_account_events = include_account_events
        self.interval_seconds = interval_seconds
        self.suppress_errors = suppress_errors
        self._last_update_monotonic: float | None = None

    @property
    def last_update_monotonic(self) -> float | None:
        """Return the monotonic timestamp of the last successful update."""
        return self._last_update_monotonic

    def should_update(self) -> bool:
        """Return whether enough time has elapsed to run another update.

        Returns:
            True when ``interval_seconds <= 0``, when no update has succeeded
            yet, or when at least ``interval_seconds`` have elapsed since the
            last successful update.
        """
        if self.interval_seconds <= 0 or self._last_update_monotonic is None:
            return True
        return (time.monotonic() - self._last_update_monotonic) >= self.interval_seconds

    def update(self, client: Mt5DataClient, symbols: Sequence[str]) -> bool:
        """Run a throttled incremental history update.

        Args:
            client: Connected MT5 data client.
            symbols: Symbols to update.

        Returns:
            True if an update ran successfully, False if it was throttled or
            (when ``suppress_errors`` is True) failed with a recoverable error.
            When ``suppress_errors`` is False, recoverable update failures
            propagate to the caller.

        Raises:
            AttributeError: MT5 client capability mismatch when
                ``suppress_errors`` is False, or any other attribute error.
            TypeError: MT5 client capability mismatch when ``suppress_errors``
                is False, or any other type error.
        """
        if not self.should_update():
            return False
        try:
            _resolve_update_history_request(
                output=self.output,
                symbols=symbols,
                datasets=self.datasets,
                timeframes=self.timeframes,
                flags=self.flags,
                lookback_hours=self.lookback_hours,
                date_to=None,
            )
            update_history(
                client=client,
                output=self.output,
                symbols=symbols,
                datasets=self.datasets,
                timeframes=self.timeframes,
                flags=self.flags,
                lookback_hours=self.lookback_hours,
                with_views=self.with_views,
                include_account_events=self.include_account_events,
            )
        except _RECOVERABLE_HISTORY_UPDATE_ERRORS:
            if self.suppress_errors:
                logger.warning("Suppressed history update error", exc_info=True)
                return False
            raise
        except (AttributeError, TypeError) as exc:
            if self.suppress_errors and _is_mt5_client_capability_error(exc):
                logger.warning("Suppressed history update error", exc_info=True)
                return False
            raise
        self._last_update_monotonic = time.monotonic()
        return True


def collect_history(
    output: Path,
    symbols: list[str],
    date_from: datetime | str,
    date_to: datetime | str,
    *,
    datasets: set[Dataset] | None = None,
    timeframe: int | str = 1,
    flags: int | str = 1,
    if_exists: IfExists = IfExists.FAIL,
    with_views: bool = False,
    config: Mt5Config | None = None,
) -> None:
    """Collect historical datasets into a single SQLite database.

    Args:
        output: SQLite database path.
        symbols: Symbols to collect.
        date_from: Start date.
        date_to: End date.
        datasets: Datasets to include (defaults to all).
        timeframe: Rates timeframe as integer or name (e.g. ``M1``).
        flags: Tick copy flags as integer or name (e.g. ``ALL``).
        if_exists: Behavior when a target table already exists.
        with_views: Create ``cash_events`` and ``positions_reconstructed`` views.
        config: MT5 connection configuration.
    """
    start = _require_datetime(date_from)
    end = _require_datetime(date_to)
    selected = datasets if datasets is not None else set(Dataset)
    tf = _coerce_timeframe(timeframe)
    tick_flags = _coerce_tick_flags(flags)
    mt5_config = config or build_config()
    with _connected_client(mt5_config) as client, sqlite3.connect(output) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        written_tables, written_columns = write_collected_datasets(
            conn,
            client,
            symbols,
            selected,
            tf,
            tick_flags,
            start,
            end,
            if_exists,
        )
        create_history_indexes(conn, written_columns)
        if with_views and Dataset.history_deals in written_tables:
            create_cash_events_view(conn, written_columns[Dataset.history_deals])
            create_positions_reconstructed_view(
                conn,
                written_columns[Dataset.history_deals],
            )
        elif with_views:
            logger.warning(
                "--with-views ignored: history_deals table was not written",
            )
    logger.info(
        "Collected %s for %d symbol(s) into %s",
        ", ".join(sorted(ds.value for ds in selected)),
        len(symbols),
        output,
    )


def _make_client(*, config: Mt5Config | None = None) -> Mt5CliClient:
    return Mt5CliClient(config=config) if config is not None else Mt5CliClient()


def copy_rates_from(
    symbol: str,
    timeframe: int | str,
    date_from: datetime | str,
    count: int,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return rates starting from a date."""
    return _make_client(config=config).copy_rates_from(
        symbol,
        timeframe,
        date_from,
        count,
    )


def copy_rates_from_pos(
    symbol: str,
    timeframe: int | str,
    start_pos: int,
    count: int,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return rates starting from a bar position."""
    return _make_client(config=config).copy_rates_from_pos(
        symbol,
        timeframe,
        start_pos,
        count,
    )


def latest_rates(
    symbol: str,
    timeframe: int | str,
    count: int,
    start_pos: int = 0,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return the latest rates from a bar position."""
    return _make_client(config=config).latest_rates(
        symbol,
        timeframe,
        count,
        start_pos=start_pos,
    )


def collect_latest_rates(
    symbols: Sequence[str],
    timeframes: Sequence[int | str],
    *,
    count: int,
    start_pos: int = 0,
    config: Mt5Config | None = None,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Return latest rates for each symbol/timeframe pair."""
    return _make_client(config=config).collect_latest_rates(
        symbols,
        timeframes,
        count=count,
        start_pos=start_pos,
    )


@dataclass(frozen=True)
class AccountSpec:
    """Connection parameters and symbols for one MT5 account group.

    Attributes:
        symbols: Symbols to load latest rates for under this account.
        login: Trading account login. String values are coerced to int when
            non-empty.
        password: Trading account password.
        server: Trading server name.
        path: Path to the MetaTrader5 terminal EXE file.
        timeout: Connection timeout in milliseconds.
    """

    symbols: Sequence[str]
    login: int | str | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)
    server: str | None = None
    path: str | None = None
    timeout: int | None = None


_ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")


def substitute_env_placeholders(value: str) -> str:
    """Replace ``${ENV_VAR}`` placeholders in a string with environment values.

    Args:
        value: String that may contain one or more ``${ENV_VAR}`` placeholders.

    Returns:
        The string with every placeholder replaced by its environment value.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """
    parts: list[str] = []
    last_end = 0
    for match in _ENV_PLACEHOLDER_PATTERN.finditer(value):
        parts.append(value[last_end : match.start()])
        name = match.group("name")
        if name not in os.environ:
            msg = f"Environment variable {name!r} is not set."
            raise ValueError(msg)
        parts.append(os.environ[name])
        last_end = match.end()
    parts.append(value[last_end:])
    return "".join(parts)


def _resolve_field(override: str | None, account_value: str | None) -> str | None:
    """Resolve a string field from an override or account value with env subst.

    Returns:
        The explicit override when provided, otherwise the account value, with
        any ``${ENV_VAR}`` placeholders substituted.
    """
    value = override if override is not None else account_value
    if value is None:
        return None
    return substitute_env_placeholders(value)


def _resolve_login(
    override: int | str | None,
    account_login: int | str | None,
) -> int | str | None:
    """Resolve a login from an override or account value with env substitution.

    Returns:
        The explicit override when provided, otherwise the account login.
        Integer values are preserved; string values have ``${ENV_VAR}``
        placeholders substituted.
    """
    if override is not None:
        if isinstance(override, int):
            return override
        return substitute_env_placeholders(override)
    if account_login is None or isinstance(account_login, int):
        return account_login
    return substitute_env_placeholders(account_login)


def resolve_account_spec(
    account: AccountSpec,
    *,
    login: int | str | None = None,
    password: str | None = None,
    server: str | None = None,
    path: str | None = None,
    timeout: int | None = None,
) -> AccountSpec:
    """Resolve an account's credentials from overrides and ``${ENV_VAR}`` values.

    Explicit override arguments take precedence over the corresponding
    :class:`AccountSpec` fields. The resolved string fields (``login``,
    ``password``, ``server``, ``path``) have any ``${ENV_VAR}`` placeholders
    substituted from the environment.

    Args:
        account: Source account specification.
        login: Optional explicit login override.
        password: Optional explicit password override.
        server: Optional explicit server override.
        path: Optional explicit terminal path override.
        timeout: Optional explicit connection timeout override.

    Returns:
        A new :class:`AccountSpec` with resolved credentials and the original
        symbols preserved. Raises ``ValueError`` (via
        :func:`substitute_env_placeholders`) if a referenced environment
        variable is not set.
    """
    return AccountSpec(
        symbols=account.symbols,
        login=_resolve_login(login, account.login),
        password=_resolve_field(password, account.password),
        server=_resolve_field(server, account.server),
        path=_resolve_field(path, account.path),
        timeout=timeout if timeout is not None else account.timeout,
    )


def resolve_account_specs(
    accounts: Sequence[AccountSpec],
    *,
    login: int | str | None = None,
    password: str | None = None,
    server: str | None = None,
    path: str | None = None,
    timeout: int | None = None,
) -> list[AccountSpec]:
    """Resolve credentials for multiple accounts.

    Applies the same overrides and ``${ENV_VAR}`` substitution as
    :func:`resolve_account_spec` to every account.

    Args:
        accounts: Source account specifications.
        login: Optional explicit login override applied to each account.
        password: Optional explicit password override applied to each account.
        server: Optional explicit server override applied to each account.
        path: Optional explicit terminal path override applied to each account.
        timeout: Optional explicit timeout override applied to each account.

    Returns:
        Resolved account specifications in the original order. Raises
        ``ValueError`` (via :func:`substitute_env_placeholders`) if a referenced
        environment variable is not set.
    """
    return [
        resolve_account_spec(
            account,
            login=login,
            password=password,
            server=server,
            path=path,
            timeout=timeout,
        )
        for account in accounts
    ]


def _coerce_login(login: int | str | None) -> int | None:
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


def _build_account_config(
    account: AccountSpec,
    base_config: Mt5Config | None,
) -> Mt5Config:
    """Build an ``Mt5Config`` for an account, falling back to ``base_config``.

    Returns:
        Merged MT5 configuration for the account.
    """
    login = _coerce_login(account.login)
    if login is None and base_config is not None:
        login = base_config.login
    return build_config(
        path=account.path or (base_config.path if base_config else None),
        login=login,
        password=account.password or (base_config.password if base_config else None),
        server=account.server or (base_config.server if base_config else None),
        timeout=account.timeout
        if account.timeout is not None
        else (base_config.timeout if base_config else None),
    )


def collect_latest_rates_for_accounts(
    accounts: Sequence[AccountSpec],
    timeframes: Sequence[int | str],
    count: int,
    *,
    start_pos: int = 0,
    base_config: Mt5Config | None = None,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Collect latest rates across multiple MT5 account groups.

    Each account is connected in turn, its symbols are read for every
    timeframe, and the resulting frames are merged into a single mapping.

    Args:
        accounts: Account groups to read. Each must define at least one symbol.
        timeframes: MT5 timeframes as integers or names (for example ``M1``).
        count: Number of most recent bars to read per symbol/timeframe.
        start_pos: Initial bar position offset.
        base_config: Optional base configuration whose fields fill any value not
            set on an individual account.

    Returns:
        Mapping keyed by ``(symbol, timeframe_int)``. When accounts share a
        symbol/timeframe pair, the last account processed wins.

    Raises:
        ValueError: If ``accounts``, ``timeframes``, or any account's symbols are
            empty, or ``count`` is not positive.
    """
    account_list = list(accounts)
    if not account_list:
        msg = "At least one account is required."
        raise ValueError(msg)
    if not timeframes:
        msg = "At least one timeframe is required."
        raise ValueError(msg)
    if any(not account.symbols for account in account_list):
        msg = "Each account requires at least one symbol."
        raise ValueError(msg)
    _require_positive(count, "count")
    result: dict[tuple[str, int], pd.DataFrame] = {}
    for account in account_list:
        config = _build_account_config(account, base_config)
        with Mt5CliClient(config=config) as client:
            result.update(
                client.collect_latest_rates(
                    account.symbols,
                    timeframes,
                    count=count,
                    start_pos=start_pos,
                ),
            )
    return result


def collect_latest_rates_for_accounts_with_retries(
    accounts: Sequence[AccountSpec],
    timeframes: Sequence[int | str],
    count: int,
    *,
    start_pos: int = 0,
    base_config: Mt5Config | None = None,
    retry_count: int = 0,
    backoff_base: float = 2.0,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Collect latest rates across accounts, retrying transient MT5 failures.

    Wraps :func:`collect_latest_rates_for_accounts` with bounded exponential
    backoff. Only ``pdmt5.Mt5TradingError`` and ``pdmt5.Mt5RuntimeError`` are
    retried; other exceptions propagate immediately. The final failure is
    re-raised once retries are exhausted.

    Args:
        accounts: Account groups to read. Each must define at least one symbol.
        timeframes: MT5 timeframes as integers or names (for example ``M1``).
        count: Number of most recent bars to read per symbol/timeframe.
        start_pos: Initial bar position offset.
        base_config: Optional base configuration whose fields fill any value not
            set on an individual account.
        retry_count: Maximum number of retries after the first attempt. ``0``
            disables retries.
        backoff_base: Base for exponential backoff. The delay before retry
            attempt ``n`` (1-indexed) is ``backoff_base ** n`` seconds.

    Returns:
        Mapping keyed by ``(symbol, timeframe_int)``. Propagates ``ValueError``
        for invalid inputs (see :func:`collect_latest_rates_for_accounts`) and
        re-raises the last ``pdmt5.Mt5TradingError`` or ``pdmt5.Mt5RuntimeError``
        once retries are exhausted.
    """
    attempts = max(retry_count, 0) + 1

    def _collect() -> dict[tuple[str, int], pd.DataFrame]:
        return collect_latest_rates_for_accounts(
            accounts,
            timeframes,
            count,
            start_pos=start_pos,
            base_config=base_config,
        )

    for attempt in range(attempts - 1):
        try:
            return _collect()
        except (Mt5TradingError, Mt5RuntimeError) as exc:
            delay = backoff_base ** (attempt + 1)
            logger.warning(
                "Rate collection failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt + 1,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    return _collect()


def collect_latest_closed_rates_for_accounts(
    accounts: Sequence[AccountSpec],
    timeframes: Sequence[int | str],
    count: int,
    *,
    start_pos: int = 0,
    base_config: Mt5Config | None = None,
    retry_count: int = 0,
    backoff_base: float = 2.0,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Collect latest closed rate bars across multiple MT5 account groups.

    When ``start_pos`` is ``0`` (the default), MetaTrader 5 includes the
    still-forming current bar as the last row. This helper fetches
    ``count + 1`` bars, drops that bar with :func:`drop_forming_rate_bar`, and
    validates that each resulting frame is non-empty. When ``start_pos`` is
    greater than zero the forming bar is not in range, so only ``count`` bars
    are fetched and no row is dropped.

    Wraps :func:`collect_latest_rates_for_accounts_with_retries` for transient
    MT5 error handling.

    Args:
        accounts: Account groups to read. Each must define at least one symbol.
        timeframes: MT5 timeframes as integers or names (for example ``M1``).
        count: Number of closed bars to return per symbol/timeframe.
        start_pos: Initial bar position offset passed to the underlying collector.
        base_config: Optional base configuration whose fields fill any value not
            set on an individual account.
        retry_count: Maximum number of retries after the first attempt. ``0``
            disables retries.
        backoff_base: Base for exponential backoff between retry attempts.

    Returns:
        Mapping keyed by ``(symbol, timeframe_int)``.

    Raises:
        ValueError: If inputs are invalid, or any series is empty (after
            dropping the still-forming bar when ``start_pos`` is ``0``).
    """
    _require_positive(count, "count")
    _require_non_negative(start_pos, "start_pos")
    fetch_count = count + 1 if start_pos == 0 else count
    loaded = collect_latest_rates_for_accounts_with_retries(
        accounts,
        timeframes,
        fetch_count,
        start_pos=start_pos,
        base_config=base_config,
        retry_count=retry_count,
        backoff_base=backoff_base,
    )
    result: dict[tuple[str, int], pd.DataFrame] = {}
    for key, df_rate in loaded.items():
        closed = drop_forming_rate_bar(df_rate) if start_pos == 0 else df_rate
        if closed.empty:
            symbol, timeframe = key
            msg = f"Rate data is empty for {symbol!r} at timeframe {timeframe}."
            raise ValueError(msg)
        result[key] = closed
    return result


def collect_latest_closed_rates_by_granularity(
    accounts: Sequence[AccountSpec],
    granularities: Sequence[int | str],
    count: int,
    *,
    start_pos: int = 0,
    base_config: Mt5Config | None = None,
    retry_count: int = 0,
    backoff_base: float = 2.0,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Collect latest closed rate bars keyed by symbol and granularity name.

    Thin wrapper around :func:`collect_latest_closed_rates_for_accounts` that
    rekeys the result by granularity name (for example ``M1``) instead of the
    integer timeframe.

    Args:
        accounts: Account groups to read. Each must define at least one symbol.
        granularities: MT5 timeframes as integers or names (for example ``M1``).
        count: Number of closed bars to return per symbol/timeframe.
        start_pos: Initial bar position offset passed to the underlying collector.
        base_config: Optional base configuration whose fields fill any value not
            set on an individual account.
        retry_count: Maximum number of retries after the first attempt. ``0``
            disables retries.
        backoff_base: Base for exponential backoff between retry attempts.

    Returns:
        Mapping keyed by ``(symbol, granularity_name)``. Propagates
        ``ValueError`` from :func:`collect_latest_closed_rates_for_accounts`.
    """
    loaded = collect_latest_closed_rates_for_accounts(
        accounts,
        granularities,
        count,
        start_pos=start_pos,
        base_config=base_config,
        retry_count=retry_count,
        backoff_base=backoff_base,
    )
    return {
        (symbol, resolve_granularity_name(timeframe)): frame
        for (symbol, timeframe), frame in loaded.items()
    }


def copy_rates_range(
    symbol: str,
    timeframe: int | str,
    date_from: datetime | str,
    date_to: datetime | str,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return rates for a date range."""
    return _make_client(config=config).copy_rates_range(
        symbol,
        timeframe,
        date_from,
        date_to,
    )


def copy_ticks_from(
    symbol: str,
    date_from: datetime | str,
    count: int,
    flags: int | str,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return ticks starting from a date."""
    return _make_client(config=config).copy_ticks_from(
        symbol,
        date_from,
        count,
        flags,
    )


def copy_ticks_range(
    symbol: str,
    date_from: datetime | str,
    date_to: datetime | str,
    flags: int | str,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return ticks for a date range."""
    return _make_client(config=config).copy_ticks_range(
        symbol,
        date_from,
        date_to,
        flags,
    )


def account_info(*, config: Mt5Config | None = None) -> pd.DataFrame:
    """Return account information."""
    return _make_client(config=config).account_info()


def terminal_info(*, config: Mt5Config | None = None) -> pd.DataFrame:
    """Return terminal information."""
    return _make_client(config=config).terminal_info()


def symbols(
    group: str | None = None,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return the symbol list."""
    return _make_client(config=config).symbols(group=group)


def symbol_info(
    symbol: str,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return details for one symbol."""
    return _make_client(config=config).symbol_info(symbol)


def orders(
    symbol: str | None = None,
    group: str | None = None,
    ticket: int | None = None,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return active orders."""
    return _make_client(config=config).orders(
        symbol=symbol,
        group=group,
        ticket=ticket,
    )


def positions(
    symbol: str | None = None,
    group: str | None = None,
    ticket: int | None = None,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return open positions."""
    return _make_client(config=config).positions(
        symbol=symbol,
        group=group,
        ticket=ticket,
    )


def history_orders(
    date_from: datetime | str | None = None,
    date_to: datetime | str | None = None,
    group: str | None = None,
    symbol: str | None = None,
    ticket: int | None = None,
    position: int | None = None,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return historical orders."""
    return _make_client(config=config).history_orders(
        date_from=date_from,
        date_to=date_to,
        group=group,
        symbol=symbol,
        ticket=ticket,
        position=position,
    )


def history_deals(
    date_from: datetime | str | None = None,
    date_to: datetime | str | None = None,
    group: str | None = None,
    symbol: str | None = None,
    ticket: int | None = None,
    position: int | None = None,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return historical deals."""
    return _make_client(config=config).history_deals(
        date_from=date_from,
        date_to=date_to,
        group=group,
        symbol=symbol,
        ticket=ticket,
        position=position,
    )


def recent_history_deals(
    hours: float,
    date_to: datetime | str | None = None,
    group: str | None = None,
    symbol: str | None = None,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return historical deals from a recent trailing window."""
    return _make_client(config=config).recent_history_deals(
        hours,
        date_to=date_to,
        group=group,
        symbol=symbol,
    )


def version(*, config: Mt5Config | None = None) -> pd.DataFrame:
    """Return MetaTrader5 version information."""
    return _make_client(config=config).version()


def last_error(*, config: Mt5Config | None = None) -> pd.DataFrame:
    """Return the last error information."""
    return _make_client(config=config).last_error()


def symbol_info_tick(
    symbol: str,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return the last tick for a symbol."""
    return _make_client(config=config).symbol_info_tick(symbol)


def market_book(
    symbol: str,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return market depth for a symbol."""
    return _make_client(config=config).market_book(symbol)


def recent_ticks(
    symbol: str,
    seconds: float,
    *,
    date_to: datetime | str | None = None,
    count: int = 10000,
    flags: int | str = "ALL",
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return ticks from a recent time window ending at ``date_to`` or now.

    See ``Mt5CliClient.recent_ticks`` for parameter and return details.
    """
    return _make_client(config=config).recent_ticks(
        symbol,
        seconds,
        date_to=date_to,
        count=count,
        flags=flags,
    )


def minimum_margins(
    symbol: str,
    *,
    config: Mt5Config | None = None,
) -> pd.DataFrame:
    """Return minimum-volume buy and sell margin requirements.

    See ``Mt5CliClient.minimum_margins`` for return details.
    """
    return _make_client(config=config).minimum_margins(symbol)


def mt5_summary(*, config: Mt5Config | None = None) -> dict[str, object]:
    """Return a compact terminal/account status summary."""
    return _make_client(config=config).mt5_summary()


def mt5_summary_as_df(*, config: Mt5Config | None = None) -> pd.DataFrame:
    """Return an export-safe terminal/account status summary DataFrame."""
    return _make_client(config=config).mt5_summary_as_df()
