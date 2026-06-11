# Trading Module

::: mt5cli.trading

## Trading-capable MT5 sessions

`mt5_trading_session()` complements the read-only `mt5_session()` helper in
`sdk.py`. It yields a connected `pdmt5.Mt5TradingClient`, uses
`Mt5Config.path` to launch the terminal when configured, and always calls
`shutdown()` on exit.

```python
from pdmt5 import Mt5Config

from mt5cli import mt5_trading_session

with mt5_trading_session(
    Mt5Config(path=r"C:\Program Files\MetaTrader 5\terminal64.exe", login=12345),
    retry_count=2,
) as client:
    positions = client.positions_get_as_df(symbol="EURUSD")
```

The read-only `Mt5CliClient` / `mt5_session()` API is unchanged.

## Operational trading helpers

These helpers are strategy-agnostic and do not depend on signal detection,
betting logic, or scheduling code in downstream applications.

```python
from mt5cli import (
    calculate_margin_and_volume,
    detect_position_side,
    determine_order_limits,
)

side = detect_position_side(client, "EURUSD")
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
```

Protective ratios must satisfy ``0 <= ratio < 1``; ``0`` omits that level.
``calculate_margin_and_volume()`` clamps negative ``margin_free`` to ``0.0``
before sizing.

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
