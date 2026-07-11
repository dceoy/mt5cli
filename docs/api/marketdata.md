# Market Data Module

::: mt5cli.marketdata

## Resilient multi-account orchestration

This module ships strategy-agnostic helpers for building long-running
collectors on top of `MT5Client`. None of them depend on a particular trading
application.

### Retrying transient rate collection

`collect_latest_rates_for_accounts_with_retries()` wraps
`collect_latest_rates_for_accounts()` with bounded exponential backoff. Only
`pdmt5.Mt5RuntimeError` is retried; the final failure is re-raised once
`retry_count` is exhausted.

```python
from mt5cli import AccountSpec, collect_latest_rates_for_accounts_with_retries

accounts = [AccountSpec(symbols=["EURUSD"], login=12345)]
rates = collect_latest_rates_for_accounts_with_retries(
    accounts,
    ["M1", "H1"],
    count=500,
    retry_count=3,
    backoff_base=2,  # sleeps 2s, 4s, 8s between attempts
)
```

### Latest closed rate bars

MetaTrader 5 `start_pos=0` includes the still-forming current bar as the last
row. `fetch_latest_closed_rates()` handles one connected `MT5Client`.
Multi-account helpers fetch
`count + 1` bars, drop
that row with `drop_forming_rate_bar()`, and validate each series is non-empty. Returned frames are ordered
oldest-to-newest and may contain fewer than `count` rows only when MT5 returns
fewer closed bars.

```python
from mt5cli import (
    AccountSpec,
    collect_latest_closed_rates_by_granularity,
    fetch_latest_closed_rates,
)

closed = fetch_latest_closed_rates(
    client,
    symbol="EURUSD",
    granularity="M1",
    count=500,
)

rates = collect_latest_closed_rates_by_granularity(
    [AccountSpec(symbols=["EURUSD"], login=12345)],
    ["M1", "H1"],
    count=500,
    retry_count=3,
)
closed_m1 = rates["EURUSD", "M1"]
```

Use `collect_latest_closed_rates_by_granularity()` when callers prefer keys such
as `("EURUSD", "M1")` instead of integer timeframes.

### Resolving credentials and `${ENV_VAR}` placeholders

`resolve_account_spec()` / `resolve_account_specs()` merge explicit override
values over `AccountSpec` fields and expand `${ENV_VAR}` placeholders, keeping
secrets out of plan/config files. A missing environment variable raises
`ValueError`.

```python
import os

from mt5cli import AccountSpec, resolve_account_specs

os.environ["MT5_LOGIN"] = "12345"
os.environ["MT5_PASSWORD"] = "secret"
accounts = [
    AccountSpec(symbols=["EURUSD"], login="${MT5_LOGIN}", password="${MT5_PASSWORD}")
]

resolved = resolve_account_specs(accounts, server="Broker-Demo")
# resolved[0].login == "12345", resolved[0].server == "Broker-Demo"
```

Pass `allow_whole_dollar_env=True` to also expand strings whose **entire value**
is a bare `$ENV_NAME` identifier (no braces). This opt-in covers
`substitute_env_placeholders()`, `resolve_account_spec()`,
`resolve_account_specs()`, and `build_config()`. Note: `build_config` cannot
expand `login` because that parameter is `int | None`; use
`resolve_account_spec` for a string `login` placeholder. Partial strings such as
`"plan$pass"`, `"abc$ENV"`, or `"$ENV-suffix"` are never expanded — only an
exact `$IDENTIFIER` whole-string match qualifies. The default is `False` to
preserve backward compatibility.

```python
import os

from mt5cli import AccountSpec, resolve_account_specs

os.environ["MT5_PASSWORD"] = "secret"
accounts = [AccountSpec(symbols=["EURUSD"], password="$MT5_PASSWORD")]

resolved = resolve_account_specs(accounts, allow_whole_dollar_env=True)
# resolved[0].password == "secret"
```

## Trading-capable sessions

For order placement and trading calculations, use the dedicated
[Trading module](trading.md). Use `mt5_session()` / `MT5Client` for read-only
collection.
