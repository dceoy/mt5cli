"""Programmatic SDK for MetaTrader 5 data collection."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypeVar

from pdmt5 import Mt5Config, Mt5DataClient

from .sqlite_history import (
    create_cash_events_view,
    create_history_indexes,
    create_positions_reconstructed_view,
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

    import pandas as pd

T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = [
    "Mt5CliClient",
    "account_info",
    "build_config",
    "collect_history",
    "copy_rates_from",
    "copy_rates_from_pos",
    "copy_rates_range",
    "copy_ticks_from",
    "copy_ticks_range",
    "history_deals",
    "history_orders",
    "last_error",
    "market_book",
    "orders",
    "positions",
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


def _require_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return parse_datetime(value)


def _coerce_datetime(value: datetime | str | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return parse_datetime(value)


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
    ) -> None:
        """Initialize the SDK client.

        Args:
            path: Path to MetaTrader5 terminal EXE file.
            login: Trading account login.
            password: Trading account password.
            server: Trading server name.
            timeout: Connection timeout in milliseconds.
            config: Optional pre-built ``Mt5Config`` (overrides other args).
        """
        self._config = config or build_config(
            path=path,
            login=login,
            password=password,
            server=server,
            timeout=timeout,
        )
        self._client: Mt5DataClient | None = None

    @property
    def config(self) -> Mt5Config:
        """Return the underlying MT5 configuration."""
        return self._config

    def __enter__(self) -> Self:
        """Open a persistent MT5 connection for multiple calls.

        Returns:
            This client instance.
        """
        client = Mt5DataClient(config=self._config)
        try:
            client.initialize_and_login_mt5()
        except Exception:
            client.shutdown()
            raise
        self._client = client
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Shut down the persistent MT5 connection."""
        if self._client is not None:
            self._client.shutdown()
            self._client = None

    def _fetch(self, fetch_fn: Callable[[Mt5DataClient], pd.DataFrame]) -> pd.DataFrame:
        if self._client is not None:
            return fetch_fn(self._client)
        return _run_with_client(self._config, fetch_fn)

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
    ``MAX(time)`` values scoped by dataset, symbol, and timeframe.

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
        create_rate_views: Create ``rate_<symbol>[_<granularity>]`` views.
        with_views: Create ``cash_events`` and ``positions_reconstructed`` views.
        include_account_events: Include account-level cash events in
            ``history_deals`` when True.

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
        return
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
    output_path = Path(output)
    logger.info(
        "Updating history in SQLite: symbols=%s, datasets=%s, path=%s",
        list(symbols),
        sorted(dataset.value for dataset in selected),
        output_path,
    )
    with closing(sqlite3.connect(output_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        write_incremental_datasets(
            conn,
            client,
            symbols,
            selected,
            resolved_timeframes,
            resolved_tick_flags,
            fallback_start,
            end,
            deduplicate=deduplicate,
            create_rate_views=create_rate_views,
            with_views=with_views,
            include_account_events=include_account_events,
        )
        conn.commit()


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
