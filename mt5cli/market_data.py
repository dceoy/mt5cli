"""Market-data facade and multi-account rate collection helpers.

These are stateless convenience wrappers around :class:`~mt5cli.client.MT5Client`
for one-off reads. Downstream applications that need a persistent connection
across multiple calls should use :func:`mt5cli.mt5_session` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import SecretStr

from .client import (
    MT5Client,
    _require_positive,  # pyright: ignore[reportPrivateUsage]
    build_config,
    substitute_env_placeholders,
)
from .history import drop_forming_rate_bar, resolve_granularity_name
from .retry import retry_with_backoff
from .utils import coerce_login as _coerce_login

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    import pandas as pd
    from pdmt5 import Mt5Config


def _require_non_negative(value: int, name: str) -> None:
    if value < 0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)


__all__ = [
    "AccountSpec",
    "account_info",
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
    "fetch_latest_closed_rates",
    "history_deals",
    "history_orders",
    "last_error",
    "latest_rates",
    "market_book",
    "minimum_margins",
    "mt5_summary",
    "mt5_summary_as_df",
    "orders",
    "positions",
    "recent_history_deals",
    "recent_ticks",
    "resolve_account_spec",
    "resolve_account_specs",
    "symbol_info",
    "symbol_info_tick",
    "symbols",
    "terminal_info",
    "version",
]


def _make_client(*, config: Mt5Config | None = None) -> MT5Client:
    return MT5Client(config=config) if config is not None else MT5Client()


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


def fetch_latest_closed_rates(
    client: MT5Client,
    *,
    symbol: str,
    granularity: str,
    count: int,
) -> pd.DataFrame:
    """Fetch up to ``count`` most recent closed bars, oldest to newest.

    Returns:
        Closed rate bars ordered oldest to newest.

    Raises:
        ValueError: If ``count`` is not positive or no closed bars are returned.
    """
    _require_positive(count, "count")
    frame = client.latest_rates(symbol, granularity, count + 1, start_pos=0)
    closed = drop_forming_rate_bar(frame)
    if closed.empty:
        msg = (
            f"Rate data is empty for {symbol!r} at granularity {granularity!r} "
            f"with count {count}."
        )
        raise ValueError(msg)
    return closed.tail(count).reset_index(drop=True)


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


def _resolve_field(
    override: str | None,
    account_value: str | None,
    *,
    allow_whole_dollar_env: bool = False,
) -> str | None:
    """Resolve a string field from an override or account value with env subst.

    Returns:
        The explicit override when provided, otherwise the account value, with
        any ``${ENV_VAR}`` placeholders substituted.
    """
    value = override if override is not None else account_value
    if value is None:
        return None
    return substitute_env_placeholders(
        value, allow_whole_dollar_env=allow_whole_dollar_env
    )


def _resolve_login(
    override: int | str | None,
    account_login: int | str | None,
    *,
    allow_whole_dollar_env: bool = False,
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
        return substitute_env_placeholders(
            override, allow_whole_dollar_env=allow_whole_dollar_env
        )
    if account_login is None or isinstance(account_login, int):
        return account_login
    return substitute_env_placeholders(
        account_login, allow_whole_dollar_env=allow_whole_dollar_env
    )


def resolve_account_spec(
    account: AccountSpec,
    *,
    login: int | str | None = None,
    password: str | None = None,
    server: str | None = None,
    path: str | None = None,
    timeout: int | None = None,
    allow_whole_dollar_env: bool = False,
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
        allow_whole_dollar_env: When ``True``, string fields that are exactly
            ``$ENV_NAME`` are also expanded from the environment. Default
            ``False`` preserves existing behavior.

    Returns:
        A new :class:`AccountSpec` with resolved credentials and the original
        symbols preserved. Raises ``ValueError`` (via
        :func:`~mt5cli.client.substitute_env_placeholders`) if a referenced
        environment variable is not set.
    """
    return AccountSpec(
        symbols=account.symbols,
        login=_resolve_login(
            login, account.login, allow_whole_dollar_env=allow_whole_dollar_env
        ),
        password=_resolve_field(
            password, account.password, allow_whole_dollar_env=allow_whole_dollar_env
        ),
        server=_resolve_field(
            server, account.server, allow_whole_dollar_env=allow_whole_dollar_env
        ),
        path=_resolve_field(
            path, account.path, allow_whole_dollar_env=allow_whole_dollar_env
        ),
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
    allow_whole_dollar_env: bool = False,
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
        allow_whole_dollar_env: When ``True``, string fields that are exactly
            ``$ENV_NAME`` are also expanded from the environment. Default
            ``False`` preserves existing behavior.

    Returns:
        Resolved account specifications in the original order. Raises
        ``ValueError`` (via :func:`~mt5cli.client.substitute_env_placeholders`)
        if a referenced environment variable is not set.
    """
    return [
        resolve_account_spec(
            account,
            login=login,
            password=password,
            server=server,
            path=path,
            timeout=timeout,
            allow_whole_dollar_env=allow_whole_dollar_env,
        )
        for account in accounts
    ]


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
    base_password = base_config.password if base_config else None
    if isinstance(base_password, SecretStr):
        base_password = base_password.get_secret_value()
    return build_config(
        path=account.path or (base_config.path if base_config else None),
        login=login,
        password=account.password or base_password,
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
        with MT5Client(config=config) as client:
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
    backoff. Only ``pdmt5.Mt5RuntimeError`` is retried; other exceptions
    propagate immediately. The final failure is re-raised once retries are
    exhausted.

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
        re-raises the last ``pdmt5.Mt5RuntimeError`` once retries are
        exhausted.
    """

    def _collect() -> dict[tuple[str, int], pd.DataFrame]:
        return collect_latest_rates_for_accounts(
            accounts,
            timeframes,
            count,
            start_pos=start_pos,
            base_config=base_config,
        )

    return retry_with_backoff(
        _collect,
        retry_count=retry_count,
        backoff_base=backoff_base,
        operation="Rate collection",
    )


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

    See ``MT5Client.recent_ticks`` for parameter and return details.
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

    See ``MT5Client.minimum_margins`` for return details.
    """
    return _make_client(config=config).minimum_margins(symbol)


def mt5_summary(*, config: Mt5Config | None = None) -> dict[str, object]:
    """Return a compact terminal/account status summary."""
    return _make_client(config=config).mt5_summary()


def mt5_summary_as_df(*, config: Mt5Config | None = None) -> pd.DataFrame:
    """Return an export-safe terminal/account status summary DataFrame."""
    return _make_client(config=config).mt5_summary_as_df()
