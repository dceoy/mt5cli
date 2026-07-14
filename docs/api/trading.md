# Trading Module

::: mt5cli.trading

## Trading-capable MT5 sessions

`mt5_session()` is the single lifecycle API. It yields `MT5Client`, which
supports both market data and generic operational execution. It initializes and
logs in once, then shuts down once; passing `client=` yields a caller-owned
client without changing its lifecycle.

```python
from mt5cli import mt5_session

with mt5_session() as client:
    positions = client.positions(symbol="EURUSD")
    account = client.account_info()
```

`login` accepts `int`, numeric `str`, or an empty string; empty strings are
treated as unset. `path`, `password`, `server`, and `timeout` are forwarded to
`pdmt5.Mt5Config`, and omitted `timeout` values keep the lower-level default.
The same client also retrieves canonical deal history with
`client.history_deals()` and `client.recent_history_deals()`.

## State and order helpers

These helpers are strategy-agnostic and do not depend on signal detection,
betting logic, or scheduling code in downstream applications.

```python
from mt5cli import (
    calculate_positions_margin,
    calculate_spread_ratio,
    calculate_margin_and_volume,
    close_open_positions,
    detect_position_side,
    determine_order_limits,
    estimate_order_margin,
    fetch_latest_closed_rates_indexed,
    get_account_snapshot,
    get_positions_frame,
    get_symbol_snapshot,
    get_tick_snapshot,
    normalize_order_volume,
    place_market_order,
)

account = get_account_snapshot(client)
symbol = get_symbol_snapshot(client, "EURUSD")
tick = get_tick_snapshot(client, "EURUSD")
positions = get_positions_frame(client, "EURUSD")
side = detect_position_side(client, "EURUSD")
spread_ratio = calculate_spread_ratio(client, "EURUSD")
volume = normalize_order_volume(
    0.15,
    volume_min=symbol["volume_min"],
    volume_max=symbol["volume_max"],
    volume_step=symbol["volume_step"],
)
buy_margin = (
    estimate_order_margin(client, "EURUSD", "BUY", volume) if volume > 0 else 0.0
)
open_margin = calculate_positions_margin(client, symbols=["EURUSD"])
# Fetch closed bars with a UTC DatetimeIndex instead of a "time" column:
indexed_bars = fetch_latest_closed_rates_indexed(
    client,
    symbol="EURUSD",
    granularity="M1",
    count=100,
)
# indexed_bars.index is a UTC-aware DatetimeIndex named "time"
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
`Mt5OperationError` when bid or ask is missing or non-positive.
`normalize_order_volume()` returns `0.0` for invalid constraints or
sub-minimum requests; check the result before calling `estimate_order_margin()`,
which requires a positive finite volume. `calculate_positions_margin()` silently
skips rows with missing symbols, non-positive volumes, non-finite volumes, or
unsupported position types, but propagates `Mt5OperationError` from `estimate_order_margin()` when a valid row
encounters invalid tick data or margin results from the broker.

SL/TP ratios for `determine_order_limits()` must satisfy `0 <= ratio < 1`; `0`
omits that level. SL/TP prices are rounded with symbol `digits` metadata when
available. `determine_order_limits()` pre-validates computed SL/TP prices against
available `trade_stops_level * point` metadata when present; violations raise
`Mt5OperationError`. This is a planning helper only: it does not guarantee broker
acceptance because live validation can still depend on price movement, bid/ask
side, freeze levels, and server-side rules, and it does not validate
`trade_freeze_level`. When symbol metadata cannot be loaded, protective prices
still round with `digits=8` and stop-level validation is skipped.
`unit_margin_ratio` and `preserved_margin_ratio` for `calculate_margin_and_volume()`
accept `0 <= ratio <= 1`; `unit_margin_ratio=0` requests one minimum valid unit
when the post-reserve margin can afford it. Negative `margin_free` is clamped to
`0.0` before sizing. Execution helpers return frozen, typed
`OrderExecutionResult` receipts. Access fields directly and call `to_dict()`
only when serializing; `dry_run=True` never sends an order or mutates Market
Watch visibility.
A dry-run market order still reads the current side-appropriate quote into
`request["price"]`, so `request_price` is populated on the preview receipt,
while `response` and `filled_price` stay `None`.
`ensure_symbol_selected()` adds hidden symbols to Market Watch before live order
placement and SL/TP updates; dry runs never call it. Receipt statuses follow
the public contract: `filled`, `partial_fill`, and `placed` map to broker
success retcodes; unknown but valid integer retcodes are `rejected`; a broker
response with a missing or unparsable retcode is `malformed`; exceptions during
MT5 constant access, symbol preparation, tick retrieval, or order submission
are `failed`; `skipped` means no execution was attempted for a defined
operational reason; `dry_run` is a preview only. The raw `request` and
`response` mappings remain diagnostic fields — use the normalized direct fields
(`request_price`, `filled_price`, `order_ticket`, `deal_ticket`,
`position_id`, `retcode`, ...) for standard execution metadata. Broker sentinel
identifiers (`0` or negative order/deal/position values) are normalized to
`None`; close and SL/TP receipts keep the known positive position ticket from
the request when the broker response omits it.

## Order planning return contracts

```python
from mt5cli import MarginVolume, OrderLimits, OrderExecutionResult

sizing: MarginVolume = calculate_margin_and_volume(
    client,
    "EURUSD",
    unit_margin_ratio=0.5,
    preserved_margin_ratio=0.2,
)
limits: OrderLimits = determine_order_limits(
    client,
    "EURUSD",
    side="long",
    stop_loss_limit_ratio=0.01,
    take_profit_limit_ratio=0.02,
)
preview: OrderExecutionResult = place_market_order(
    client,
    symbol="EURUSD",
    volume=sizing["buy_volume"],
    order_side="BUY",
    sl=limits["stop_loss"],
    tp=limits["take_profit"],
    dry_run=True,
)
updates: list[OrderExecutionResult] = update_sltp_for_open_positions(
    client,
    symbol="EURUSD",
    stop_loss=limits["stop_loss"],
    dry_run=True,
)
```

Closes issue #33: strategy-neutral order planning and execution helpers exposed
through the stable package root without embedding entry/exit policy.

## Retrieving recent history deals

`MT5Client.recent_history_deals()` fetches canonical history deals from the
single connected client over a trailing time window.

The helper returns a chronologically sorted DataFrame with a `RangeIndex` and
all columns from the underlying client (`time`, `symbol`, `type`, `entry`,
`volume`, `profit`, `position_id`, etc.). It does **not** apply any
strategy-specific transformations — entry/exit classification, Kelly fractions,
and betting semantics belong in downstream applications.

```python
from mt5cli import mt5_session

with mt5_session() as client:
    deals_df = client.recent_history_deals(symbol="JP225", hours=24)
```

`hours` must be positive; `date_to` defaults to `datetime.now(UTC)`. An empty
or `None` result from the underlying client is normalized to an empty DataFrame.

Downstream packages own all strategy-specific transformations. mt5cli does not
provide entry-deal classification, Kelly sizing, or any betting-specific helpers.

## Broker server clock offset

MT5 tick, bar, and deal timestamps are epoch values labeled in the broker
server's wall clock, which is commonly UTC+2 or UTC+3 rather than true UTC.
Any code that mixes those timestamps with true UTC (freshness checks,
trailing history windows) silently inherits the broker's offset as a bias
unless it is measured and applied explicitly.

`get_tick_snapshot()` preserves the numeric MT5 epoch value in `time`; it does
not expose pdmt5's timezone-naive `Timestamp` conversion. Treat that number as
a broker wall-clock label, not guaranteed true UTC.

`estimate_server_clock_offset_seconds()` reads the latest tick for a symbol
and returns the broker's clock offset from true UTC, rounded to the nearest
half hour, or `None` when no valid tick time is available. Apply the result in
the downstream time-window calculation when comparing broker timestamps to
true UTC.

No offset is applied automatically by history retrieval helpers.

## Migration from application-local helpers

| Application-local concern                                | mt5cli replacement                                                     |
| -------------------------------------------------------- | ---------------------------------------------------------------------- |
| Manual terminal spawn/kill around trading code           | `with mt5_session(config) as client:`                                  |
| Local position-side detection                            | `detect_position_side()`                                               |
| Local margin/volume sizing                               | `calculate_margin_and_volume()`                                        |
| Local broker volume step normalization                   | `normalize_order_volume()`                                             |
| Local order or position margin estimation                | `estimate_order_margin()`, `calculate_positions_margin()`              |
| Local closed-bar fetch from a session                    | `fetch_latest_closed_rates()` or `fetch_latest_closed_rates_indexed()` |
| Local recent deal history fetch from a session           | `client.recent_history_deals()`                                        |
| Local broker server clock offset measurement             | `estimate_server_clock_offset_seconds()`                               |
| Local SL/TP price derivation                             | `determine_order_limits()`                                             |
| Throttled SQLite history loop with ad-hoc error handling | `ThrottledHistoryUpdater(suppress_errors=True)`                        |

Use `mt5_session()` / `MT5Client` for both data collection and generic
execution helpers.
