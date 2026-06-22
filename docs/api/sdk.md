# SDK Module

::: mt5cli.sdk

## Resilient multi-account orchestration

The SDK ships strategy-agnostic helpers for building long-running collectors on
top of the read-only client. None of them depend on a particular trading
application.

### Retrying transient rate collection

`collect_latest_rates_for_accounts_with_retries()` wraps
`collect_latest_rates_for_accounts()` with bounded exponential backoff. Only
`pdmt5.Mt5TradingError` and `pdmt5.Mt5RuntimeError` are retried; the final
failure is re-raised once `retry_count` is exhausted.

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
row. `fetch_latest_closed_rates()` handles one connected `Mt5CliClient`; use
`fetch_latest_closed_rates_for_trading_client()` from an active
`Mt5TradingClient` session. Multi-account helpers fetch `count + 1` bars, drop
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
`resolve_account_specs()`, and `build_config()`. Partial strings such as
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

### Throttled incremental history updates

`ThrottledHistoryUpdater` wraps `update_history()` with a minimum interval
between successful runs (using a monotonic clock), so an application loop can
call it every iteration without over-fetching.

```python
from pdmt5 import Mt5Config, Mt5DataClient

from mt5cli import Dataset, ThrottledHistoryUpdater

updater = ThrottledHistoryUpdater(
    output="history.db",
    datasets={Dataset.rates},
    timeframes=["M1"],
    interval_seconds=60,  # <= 0 updates on every call
)

client = Mt5DataClient(config=Mt5Config(login=12345))
client.initialize_and_login_mt5()
try:
    while True:
        updater.update(client, ["EURUSD", "GBPUSD"])  # no-op until 60s elapse
        # ... do other work; break when shutting down ...
finally:
    client.shutdown()
```

Pass `update_backend` to substitute the default `update_history` implementation
without monkey-patching `mt5cli.sdk.update_history`. The callable receives the
same keyword arguments as `update_history` (`client`, `output`, `symbols`,
`datasets`, `timeframes`, `flags`, `lookback_hours`, `with_views`,
`include_account_events`). The resolved backend is stored on
`updater.update_backend` for inspection or subclassing.

```python
from mt5cli import ThrottledHistoryUpdater, update_history


def app_update_history(**kwargs) -> None:
    update_history(**kwargs)  # or delegate to application-specific logic


updater = ThrottledHistoryUpdater(
    output="history.db",
    interval_seconds=60,
    update_backend=app_update_history,
)
```

By default recoverable errors (`Mt5TradingError`, `Mt5RuntimeError`,
`sqlite3.Error`, `ValueError`, `OSError`, and MT5 client capability
`AttributeError` / `TypeError` for history API methods) propagate so the caller
controls logging; pass `suppress_errors=True` to swallow them and return
`False` without advancing the throttle. Other `AttributeError` / `TypeError`
values always propagate. Input validation (`_resolve_update_history_request`)
runs before any MT5 or SQLite calls, but when `suppress_errors=True` the
resulting `ValueError` is suppressed along with other recoverable errors.

## Trading-capable sessions

For order placement and trading calculations, use the dedicated
[Trading module](trading.md). The read-only `Mt5CliClient` and `mt5_session()`
helpers in this module are unchanged.
