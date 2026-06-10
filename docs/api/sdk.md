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
row. `collect_latest_closed_rates_for_accounts()` fetches `count + 1` bars,
drops that row with `drop_forming_rate_bar()`, and validates each series is
non-empty. Use `collect_latest_closed_rates_by_granularity()` when callers
prefer keys such as `("EURUSD", "M1")` instead of integer timeframes.

```python
from mt5cli import AccountSpec, collect_latest_closed_rates_by_granularity

rates = collect_latest_closed_rates_by_granularity(
    [AccountSpec(symbols=["EURUSD"], login=12345)],
    ["M1", "H1"],
    count=500,
    retry_count=3,
)
closed_m1 = rates["EURUSD", "M1"]
```

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

By default `Mt5TradingError`, `Mt5RuntimeError`, and `sqlite3.Error` propagate so
the caller controls logging; pass `suppress_errors=True` to swallow them and
return `False` without advancing the throttle.
