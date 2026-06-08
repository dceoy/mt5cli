"""Programmatic SDK for MetaTrader 5 data collection."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Self, TypeVar

from pdmt5 import Mt5Config, Mt5DataClient

from .utils import (
    Dataset,
    IfExists,
    parse_datetime,
    parse_tick_flags,
    parse_timeframe,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import pandas as pd

T = TypeVar("T")

logger = logging.getLogger(__name__)

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
def connected_client(config: Mt5Config) -> Iterator[Mt5DataClient]:
    """Initialize MT5, yield a connected client, and always shut down.

    Args:
        config: MT5 connection configuration.

    Yields:
        Connected ``Mt5DataClient`` instance.
    """
    client = Mt5DataClient(config=config)
    client.initialize_and_login_mt5()
    try:
        yield client
    finally:
        client.shutdown()


def run_with_client(
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
    with connected_client(config) as client:
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
        self._client = Mt5DataClient(config=self._config)
        self._client.initialize_and_login_mt5()
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
        return run_with_client(self._config, fetch_fn)

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


def _create_cash_events_view(
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


def _create_positions_reconstructed_view(
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

    Returns:
        True if the dataset table exists after this write attempt.
    """
    write_mode = IfExists.APPEND if table_exists else if_exists
    if _write_frame_to_sqlite(
        conn,
        frame,
        dataset.table_name,
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
    with connected_client(mt5_config) as client, sqlite3.connect(output) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        written_tables, written_columns = _write_collected_datasets(
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
        _create_collect_history_indexes(conn, written_columns)
        if with_views and Dataset.history_deals in written_tables:
            _create_cash_events_view(conn, written_columns[Dataset.history_deals])
            _create_positions_reconstructed_view(
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
