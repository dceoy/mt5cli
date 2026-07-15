"""Sole connection-lifecycle owner and stable public client abstraction.

This module owns the MT5 connection lifecycle end to end: building
``Mt5Config``, initializing and logging in to the terminal, and shutting it
down exactly once on exit. ``mt5_session()`` is the only public session
factory; every other mt5cli module that needs a connection imports the
private primitives defined here instead of duplicating lifecycle behavior.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

import pandas as pd
from pdmt5 import Mt5Config, Mt5DataClient, Mt5RuntimeError

from .exceptions import normalize_mt5_exception
from .utils import coerce_login as _coerce_login
from .utils import parse_datetime, parse_tick_flags, parse_timeframe

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Sequence

T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = [
    "MT5Client",
    "build_config",
    "mt5_session",
    "substitute_env_placeholders",
    "substitute_mapping_values",
]

_ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")
_WHOLE_DOLLAR_PATTERN = re.compile(r"^\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")


def substitute_env_placeholders(
    value: str,
    *,
    allow_whole_dollar_env: bool = False,
) -> str:
    """Replace ``${ENV_VAR}`` placeholders in a string with environment values.

    Args:
        value: String that may contain one or more ``${ENV_VAR}`` placeholders.
        allow_whole_dollar_env: When ``True``, a string that is exactly
            ``$ENV_NAME`` (the whole value and nothing else) is also expanded
            from the environment. Partial occurrences such as ``"plan$pass"``
            or ``"$ENV-suffix"`` are left unchanged.

    Returns:
        The string with every placeholder replaced by its environment value.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """
    if allow_whole_dollar_env:
        m = _WHOLE_DOLLAR_PATTERN.match(value)
        if m:
            name = m.group("name")
            if name not in os.environ:
                msg = f"Environment variable {name!r} is not set."
                raise ValueError(msg)
            return os.environ[name]
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


def substitute_mapping_values(
    data: object,
    *,
    keys: Collection[str],
    allow_whole_dollar_env: bool = False,
    blank_string_keys_as_none: Collection[str] = (),
) -> object:
    """Recursively substitute environment placeholders for selected mapping keys.

    Traverses nested dicts and lists, expanding ``${ENV_VAR}`` (and
    ``$ENV_NAME`` when ``allow_whole_dollar_env=True``) in string values
    whose immediate parent dict key is in ``keys``.  Fields whose key is
    not in ``keys`` are preserved exactly, including literal dollar signs.
    Strings that are direct elements of a list are never substituted;
    substitution only applies to strings that are immediate dict values.

    This is a generic downstream config utility.  Key names such as
    ``mt5_login`` or ``mt5_password`` must be supplied by the caller;
    mt5cli does not hard-code any application-specific key names.
    Callers are responsible for ensuring ``data`` has bounded nesting depth;
    deeply nested or self-referential structures will hit Python's recursion
    limit.

    Args:
        data: Arbitrarily nested dict/list/scalar value to process.
        keys: Mapping keys whose string values receive placeholder
            substitution.
        allow_whole_dollar_env: When ``True``, a string that is exactly
            ``$ENV_NAME`` (whole value) is also expanded from the
            environment in addition to ``${ENV_NAME}`` placeholders.
            Default ``False`` expands ``${ENV_NAME}`` only.
        blank_string_keys_as_none: Mapping keys for which blank strings
            (after any substitution) are normalised to ``None``.  A key
            may appear in ``blank_string_keys_as_none`` without also
            appearing in ``keys``.

    Returns:
        The processed value.  Dicts and lists are rebuilt into new
        containers with selected string values substituted and
        blank-normalised.  Scalar inputs (non-dict, non-list) are
        returned as-is.
    """
    keys_set: frozenset[str] = frozenset(keys)
    blank_keys_set: frozenset[str] = frozenset(blank_string_keys_as_none)

    def _visit(node: object, current_key: str | None) -> object:
        if isinstance(node, dict):
            typed = cast("dict[object, object]", node)
            return {
                k: _visit(v, k if isinstance(k, str) else None)
                for k, v in typed.items()
            }
        if isinstance(node, list):
            typed_list = cast("list[object]", node)
            return [_visit(item, None) for item in typed_list]
        if not isinstance(node, str):
            return node
        text = node
        if current_key in keys_set:
            text = substitute_env_placeholders(
                node, allow_whole_dollar_env=allow_whole_dollar_env
            )
        if current_key in blank_keys_set and not text.strip():
            return None
        return text

    return _visit(data, None)


def _coerce_timeframe(timeframe: int | str) -> int:
    return parse_timeframe(timeframe)


def _coerce_tick_flags(flags: int | str) -> int:
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
    login: int | str | None = None,
    password: str | None = None,
    server: str | None = None,
    timeout: int | None = None,
    allow_whole_dollar_env: bool = False,
) -> Mt5Config:
    """Build an ``Mt5Config`` from optional connection parameters.

    Args:
        path: Optional terminal executable path.
        login: Optional trading account login. Integers are preserved. String
            values are coerced: empty or whitespace-only strings become
            ``None``; numeric strings such as ``"12345"`` are converted to
            ``int``; non-numeric strings raise ``ValueError``. When
            ``allow_whole_dollar_env=True``, ``$ENV_NAME`` and
            ``${ENV_NAME}`` placeholders are expanded before coercion.
        password: Optional trading account password.
        server: Optional trading server name.
        timeout: Optional connection timeout in milliseconds.
        allow_whole_dollar_env: When ``True``, string parameters that are
            exactly ``$ENV_NAME`` are expanded from the environment. Applies
            to ``path``, ``login``, ``password``, and ``server``. Default
            ``False`` preserves existing behavior.

    Returns:
        Configured ``Mt5Config`` instance.
    """
    if allow_whole_dollar_env:
        if path is not None:
            path = substitute_env_placeholders(path, allow_whole_dollar_env=True)
        if isinstance(login, str):
            login = substitute_env_placeholders(login, allow_whole_dollar_env=True)
        if password is not None:
            password = substitute_env_placeholders(
                password, allow_whole_dollar_env=True
            )
        if server is not None:
            server = substitute_env_placeholders(server, allow_whole_dollar_env=True)
    return Mt5Config(
        path=path,
        login=_coerce_login(login),
        password=password,
        server=server,
        timeout=timeout,
    )


def _shutdown_client(client: Mt5DataClient, *, raise_on_error: bool) -> None:
    """Shut down a client, normalizing or logging shutdown failures.

    When ``raise_on_error`` is True (cleanup is the only failure), a shutdown
    failure is raised as the stable normalized exception. When False (an
    initialization or session-body exception is already propagating), the
    failure is logged so the primary exception passes through unchanged.

    """
    try:
        client.shutdown()
    except Exception as exc:
        if raise_on_error:
            normalized = normalize_mt5_exception(exc)
            raise normalized from exc
        logger.warning("MT5 shutdown failed during cleanup", exc_info=True)


@contextmanager
def _connected_client(
    config: Mt5Config,
    *,
    retry_count: int | None = None,
) -> Iterator[Mt5DataClient]:
    """Initialize MT5, yield a connected client, and always shut down.

    Private lifecycle primitive owned by this module. Other mt5cli modules
    that need a raw connected session import this function rather than
    duplicating lifecycle behavior; the raw ``Mt5DataClient`` it yields is
    never returned from a public mt5cli API.

    Args:
        config: MT5 connection configuration.
        retry_count: Number of MT5 initialization retries. Defaults to the
            pdmt5 client default when omitted.

    Yields:
        Connected ``Mt5DataClient`` instance.

    """
    client = (
        Mt5DataClient(config=config)
        if retry_count is None
        else Mt5DataClient(config=config, retry_count=retry_count)
    )
    try:
        client.initialize_and_login_mt5()
    except Exception as exc:
        _shutdown_client(client, raise_on_error=False)
        normalized = normalize_mt5_exception(exc)
        raise normalized from exc
    try:
        yield client
    except BaseException:
        _shutdown_client(client, raise_on_error=False)
        raise
    _shutdown_client(client, raise_on_error=True)


def _run_with_client(
    config: Mt5Config,
    fetch_fn: Callable[[Mt5DataClient], T],
    *,
    retry_count: int | None = None,
) -> T:
    """Connect, run ``fetch_fn``, and shut down safely.

    Args:
        config: MT5 connection configuration.
        fetch_fn: Callable receiving a connected client.
        retry_count: Number of MT5 initialization retries. Defaults to the
            pdmt5 client default when omitted.

    Returns:
        Value returned by ``fetch_fn``.

    """
    with _connected_client(config, retry_count=retry_count) as client:
        try:
            return fetch_fn(client)
        except Mt5RuntimeError as exc:
            normalized = normalize_mt5_exception(exc)
            raise normalized from exc


class _BaseMT5Client:
    """Private implementation base for read-only MetaTrader 5 data access.

    Not part of the public API. :class:`MT5Client` is the sole public client
    contract; this base exists only to keep read-only data methods separate
    from execution primitives within this module.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout: int | None = None,
        retry_count: int = 3,
        config: Mt5Config | None = None,
    ) -> None:
        """Initialize the SDK client.

        Args:
            path: Path to MetaTrader5 terminal EXE file.
            login: Trading account login.
            password: Trading account password.
            server: Trading server name.
            timeout: Connection timeout in milliseconds.
            retry_count: Number of MT5 initialization retries for sessions
                opened by this client.
            config: Optional pre-built ``Mt5Config`` (overrides other args).
        """
        self._config = config or build_config(
            path=path,
            login=login,
            password=password,
            server=server,
            timeout=timeout,
        )
        self._retry_count = retry_count
        self._client: Mt5DataClient | None = None
        self._owns_client = True

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
        client = Mt5DataClient(config=self._config, retry_count=self._retry_count)
        try:
            client.initialize_and_login_mt5()
        except Exception as exc:
            _shutdown_client(client, raise_on_error=False)
            normalized = normalize_mt5_exception(exc)
            raise normalized from exc
        self._client = client
        self._owns_client = True  # only set when this method created the client
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Shut down the persistent MT5 connection.

        A shutdown failure is raised as the stable normalized exception only
        when no exception is already propagating from the ``with`` body;
        otherwise it is logged so the body exception passes through.
        """
        if self._client is not None and self._owns_client:
            client = self._client
            self._client = None
            _shutdown_client(client, raise_on_error=exc is None)

    def _fetch_value(self, fetch_fn: Callable[[Mt5DataClient], T]) -> T:
        if self._client is not None:
            try:
                return fetch_fn(self._client)
            except Mt5RuntimeError as exc:
                raise normalize_mt5_exception(exc) from exc
        return _run_with_client(self._config, fetch_fn, retry_count=self._retry_count)

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

    def symbol_info_tick_as_dict(
        self,
        symbol: str,
        *,
        skip_to_datetime: bool = False,
    ) -> dict[str, object]:
        """Return the last tick as a plain mapping.

        Args:
            symbol: Symbol name.
            skip_to_datetime: Preserve numeric MT5 time fields when ``True``.

        Returns:
            Tick fields using the underlying pdmt5 conversion behavior.
        """
        return self._fetch_value(
            lambda c: cast(
                "dict[str, object]",
                c.symbol_info_tick_as_dict(
                    symbol=symbol,
                    skip_to_datetime=skip_to_datetime,
                ),
            )
        )

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


class MT5Client(_BaseMT5Client):
    """The single public connected MT5 client.

    Extends the read-only data access above with optional order check/send
    helpers and exposes the same connection lifecycle as :func:`mt5_session`.

    mt5cli intentionally exposes minimal execution primitives only. Trading
    decisions, signals, strategies, backtests, and optimization remain the
    responsibility of downstream applications.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout: int | None = None,
        retry_count: int = 3,
        config: Mt5Config | None = None,
    ) -> None:
        """Configure a client; use it as a context manager to connect."""
        super().__init__(
            path=path,
            login=login,
            password=password,
            server=server,
            timeout=timeout,
            retry_count=retry_count,
            config=config,
        )

    @classmethod
    def _from_connected_client(cls, client: object) -> Self:
        """Create a facade around an internally managed pdmt5 connection.

        Returns:
            Public facade that does not own the supplied connection.
        """
        instance = cls()
        instance._client = cast("Any", client)
        instance._owns_client = False
        return instance

    @classmethod
    def from_connected_client(cls, client: object) -> Self:
        """Bind the public facade to an externally owned connection.

        Returns:
            Public facade that never initializes or shuts down ``client``.
        """
        return cls._from_connected_client(client)

    @property
    def mt5(self) -> Any:  # noqa: ANN401
        """Return MT5 constants required by operational helpers.

        This intentionally exposes constants only, never the pdmt5 client.
        """
        return self._fetch_value(lambda client: client.mt5)

    def account_info_as_dict(self) -> dict[str, object]:
        """Return the current account snapshot as a plain mapping."""
        frame = self.account_info()
        return {} if frame.empty else cast("dict[str, object]", frame.iloc[0].to_dict())

    def symbol_info_as_dict(self, symbol: str) -> dict[str, object]:
        """Return one symbol snapshot as a plain mapping."""
        frame = self.symbol_info(symbol)
        return {} if frame.empty else cast("dict[str, object]", frame.iloc[0].to_dict())

    def positions_get_as_df(self, symbol: str | None = None) -> pd.DataFrame:
        """Return open positions in the canonical DataFrame schema."""
        return self.positions(symbol=symbol)

    def order_calc_margin(
        self, action: int, symbol: str, volume: float, price: float
    ) -> object:
        """Calculate broker margin for an order candidate.

        Returns:
            Broker-calculated margin value.
        """
        return self._fetch_value(
            lambda client: client.order_calc_margin(action, symbol, volume, price)
        )

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        """Select or deselect a symbol in Market Watch.

        Returns:
            Whether MT5 accepted the selection operation.
        """
        return bool(
            self._fetch_value(
                lambda client: client.symbol_select(symbol, enable=enable)
            )
        )

    def order_check(self, request: dict[str, Any]) -> pd.DataFrame:
        """Check funds sufficiency for a trade request.

        Args:
            request: MT5 order request dictionary.

        Returns:
            One-row DataFrame with the order-check result.
        """
        return self._fetch(lambda client: client.order_check_as_df(request=request))

    def order_send(self, request: dict[str, Any]) -> pd.DataFrame:
        """Send a live trade request to the MT5 trade server.

        Warning:
            This is a live execution primitive. A successful call can place,
            modify, or close real trades on the connected account. Downstream
            applications must gate usage explicitly (for example behind manual
            confirmation or application-specific risk controls). mt5cli does
            not implement strategy logic, signal generation, or trade sizing.

        Args:
            request: MT5 order request dictionary.

        Returns:
            One-row DataFrame with the order-send result.
        """
        return self._fetch(lambda client: client.order_send_as_df(request=request))


@contextmanager
def mt5_session(
    config: Mt5Config | None = None, *, client: MT5Client | None = None
) -> Iterator[MT5Client]:
    """Open an MT5 terminal session and yield a connected :class:`MT5Client`.

    This is the sole public MT5 session factory. It is the only supported way
    to obtain a connected :class:`MT5Client` for market data, history,
    observability, and execution workflows.

    Args:
        config: MT5 connection configuration. Defaults to an empty config that
            attaches to a running terminal.
        client: A caller-owned connected public client. It is yielded as-is and
            is never initialized or shut down by this context manager.

    Yields:
        Connected :class:`MT5Client` bound to the session.
    """
    if client is not None:
        yield client
        return
    mt5_config = config or build_config()
    with _connected_client(mt5_config) as raw_client:
        yield MT5Client._from_connected_client(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            raw_client
        )
