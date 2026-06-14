# Trading Module

::: mt5cli.trading

## Trading-capable MT5 sessions

`create_trading_client()` and `mt5_trading_session()` complement the read-only
`mt5_session()` helper in `sdk.py`. They return or yield an initialized
`pdmt5.Mt5TradingClient`, use `Mt5Config.path` to launch the terminal when
configured, and `mt5_trading_session()` always calls `shutdown()` on exit.

```python
from mt5cli import create_trading_client, mt5_trading_session

with mt5_trading_session(
    path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
    login="12345",
    password="secret",
    server="Broker-Demo",
    retry_count=2,
) as client:
    positions = client.positions_get_as_df(symbol="EURUSD")

client = create_trading_client(login=12345, server="Broker-Demo")
try:
    account = client.account_info_as_dict()
finally:
    client.shutdown()
```

`login` accepts `int`, numeric `str`, or an empty string; empty strings are
treated as unset. `path`, `password`, `server`, and `timeout` are forwarded to
`pdmt5.Mt5Config`, and omitted `timeout` values keep the lower-level default.
The read-only `Mt5CliClient` / `mt5_session()` API is unchanged.

## State and order helpers

These helpers are strategy-agnostic and do not depend on signal detection,
betting logic, or scheduling code in downstream applications.

```python
from mt5cli import (
    calculate_spread_ratio,
    calculate_margin_and_volume,
    close_open_positions,
    detect_position_side,
    determine_order_limits,
    get_account_snapshot,
    get_positions_frame,
    get_symbol_snapshot,
    get_tick_snapshot,
    place_market_order,
)

account = get_account_snapshot(client)
symbol = get_symbol_snapshot(client, "EURUSD")
tick = get_tick_snapshot(client, "EURUSD")
positions = get_positions_frame(client, "EURUSD")
side = detect_position_side(client, "EURUSD")
spread_ratio = calculate_spread_ratio(client, "EURUSD")
sizing = calculate_margin_and_volume(
    client,
    "EURUSD",
    unit_margin_ratio=0.5,
    preserved_margin_ratio=0.2,
)
limits = determine_order_limits(
    client,
    "EURUSD",
    side="long",
    stop_loss_limit_ratio=0.01,
    take_profit_limit_ratio=0.02,
)
preview = place_market_order(
    client,
    symbol="EURUSD",
    volume=sizing["buy_volume"],
    order_side="BUY",
    sl=limits["stop_loss"],
    tp=limits["take_profit"],
    dry_run=True,
)
closed = close_open_positions(client, symbols="EURUSD", dry_run=True)
```

`detect_position_side()` returns `long` for buy-only exposure, `short` for
sell-only exposure, and `None` for no positions or mixed long/short exposure.
`calculate_spread_ratio()` uses `(ask - bid) / ((ask + bid) / 2)` and raises
`Mt5TradingError` when bid or ask is missing or non-positive.

SL/TP ratios for `determine_order_limits()` must satisfy `0 <= ratio < 1`; `0`
omits that level. SL/TP prices are rounded with symbol `digits` metadata when
available. `unit_margin_ratio` and `preserved_margin_ratio` for
`calculate_margin_and_volume()` accept `0 <= ratio <= 1`; `unit_margin_ratio=0`
requests one minimum valid unit when the post-reserve margin can afford it.
Negative `margin_free` is clamped to `0.0` before sizing. Execution helpers
return normalized dictionaries containing the request, response, status,
retcode, and `dry_run` flag; `dry_run=True` never sends an order. Market order
helpers mark known non-success MT5 retcodes as `status="failed"` while keeping
the normalized response for inspection.

## Migration from mteor-local helpers

| mteor-local concern                                      | mt5cli replacement                              |
| -------------------------------------------------------- | ----------------------------------------------- |
| Manual terminal spawn/kill around trading code           | `mt5_trading_session()`                         |
| Local position-side detection                            | `detect_position_side()`                        |
| Local margin/volume sizing                               | `calculate_margin_and_volume()`                 |
| Local SL/TP price derivation                             | `determine_order_limits()`                      |
| Throttled SQLite history loop with ad-hoc error handling | `ThrottledHistoryUpdater(suppress_errors=True)` |

Keep read-only data collection on `mt5_session()` / `Mt5CliClient`; use
`mt5_trading_session()` only where order placement or trading calculations are
required.
