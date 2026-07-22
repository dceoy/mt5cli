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

## Timestamp sources and broker clock offsets

MetaQuotes documents `copy_ticks_range()` and `copy_rates_range()` results,
and the date bounds they accept, as UTC, and does not document a different
timezone contract for `symbol_info_tick()`. In production on at least one
OANDA-style broker/terminal, however, `symbol_info_tick()` returned epochs
carrying the broker's own wall-clock label (UTC+3), not true UTC: treating
`copy_ticks_range()` as an independent UTC reference for the live tick clock
was the direct cause of a production calibration failure (see
[dceoy/mteor#428](https://github.com/dceoy/mteor/issues/428)), because a
query window built from the host clock missed the live event entirely, off
by exactly the broker's offset. mt5cli does not call `copy_ticks_range()` or
`copy_rates_range()` to calibrate the live tick clock, and does not
independently verify their documented UTC contract on any broker or
terminal — treat copied-history timestamps as broker/terminal-dependent
absent evidence from your own broker/terminal combination.

`get_tick_snapshot()` preserves the numeric MT5 epoch value in `time`; it does
not expose pdmt5's timezone-naive `Timestamp` conversion or alter the instant,
and it never applies any offset correction. Any broker-specific offset
handling requires independent evidence — see `TickClockNormalizer` below —
and must be applied separately. Do not subtract a presumed UTC+2 or UTC+3
offset from a raw timestamp without that evidence.

No offset is applied automatically by history retrieval helpers.

## UTC normalization for live ticks

`TickClockNormalizer` provides the stable API for obtaining a latest tick with
a **validated** UTC timestamp, so strategy applications never compare a raw
server-labeled tick time against `datetime.now(UTC)` themselves.

```python
from mt5cli import TickClockNormalizer, mt5_session

with mt5_session() as client:
    normalizer = TickClockNormalizer(client, ["SING30", "EURUSD"])
    snapshot = normalizer.get_normalized_tick_snapshot("SING30")
    if snapshot["clock_status"] == "calibrated":
        freshness = now_utc - snapshot["time_utc"]  # safe wall-clock compare
```

A normalized snapshot keeps the raw/UTC distinction explicit:

```python
{
    "symbol": "SING30",
    "bid": ...,
    "ask": ...,
    "last": ...,
    "volume": ...,
    "raw_time": 1717243200,  # numeric MT5 epoch, as labeled by broker
    "time_utc": datetime(..., tzinfo=UTC),  # None unless calibrated
    "server_clock_offset_seconds": 10800.0,  # None unless calibrated
    "clock_status": "calibrated",  # or "uncalibrated"
}
```

### Calibration evidence

`copy_ticks_range()` is not used for calibration: its documented UTC contract
is broker/terminal-dependent and not independently verified here (see
above), so neither the query timestamps nor the returned historical tick
timestamps can serve as an out-of-band UTC reference, and historical copied
rows are never used to infer the server offset. Instead, the only
independent reference available is **this process's own clock**: every
calibration poll
fetches one live `symbol_info_tick()` value and stamps it with
`datetime.now(UTC)` at the moment it is received. Comparing that host receipt
time against the tick's own server-labeled epoch yields one candidate offset,
which must align to a 30-minute increment within the realistic UTC-14..UTC+14
range. `time_msc` is preferred over second-resolution `time` when available.

A single poll is never trusted on its own. A closed, stale, or illiquid
symbol keeps returning the exact same last-traded tick, so its epoch never
changes between polls — that ordinary tick age is not calibration evidence.
Only a poll whose epoch _differs_ from the previous poll (comparison is
inequality, not "must increase": a genuine broker offset **decrease**, e.g. a
UTC+3 to UTC+2 DST fallback, relabels the next fresh event with a numerically
_smaller_ epoch than the last one observed under the old, larger offset)
proves the broker produced a new event, and only then is its offset computed.

An offset is accepted only after `min_agreeing_samples` (default 2)
**distinct** fresh tick events agree on the same rounded offset — repeated
polls of one unchanged tick never count as evidence, so an illiquid symbol
cannot self-confirm. Evidence can come from multiple samples of one actively
updating symbol (paced by `sample_interval_seconds`) or across several
symbols passed to the constructor; a closed symbol among several configured
symbols simply contributes no evidence rather than blocking the others. Any
two fresh events that disagree on the rounded offset abort the calibration
(`offset_disagreement`) instead of being averaged or resolved by recency.

### Failure modes and fail-closed behavior

When calibration cannot be established safely — no symbol ever produced a
fresh (changed-epoch) tick, e.g. every configured symbol is closed or
illiquid (`not_advancing`), a fresh event's raw offset does not round cleanly
to a plausible bucket (`unstable_offset`), an unrealistic rounded delta
(`implausible_offset`), too few distinct agreeing events
(`insufficient_agreement`), disagreeing events (`offset_disagreement`), or
missing live data (`no_live_tick`) — snapshots fail closed: `clock_status` is
`"uncalibrated"`, `time_utc` and `server_clock_offset_seconds` are `None`,
and the raw fields remain available. A raw timestamp is never silently
treated as UTC. The same fail-closed rule applies after a successful
calibration: once periodic revalidation invalidates the cached offset (see
below), subsequent snapshots are normalized only by a successful full
recalibration — never by the refuted cache. `normalizer.calibration`
exposes the full
`TickClockCalibration` diagnostics: status, offset, distinct sample count,
evidence symbols, last calibration time, and (for troubleshooting a failed
attempt) the most recently considered sample's symbol, raw server-labeled
time, host observation time, and raw (pre-rounding) offset —
`to_dict()` for serialization, and all of it is included in the one
concise warning logged per failed calibration attempt.

### Caching and revalidation

Calibration is a property of the broker server clock, so it is cached on the
normalizer (keep one instance per MT5 connection/account) and reused across
symbols and calls. A cached offset is recomputed in three cases:

- After `max_calibration_age_seconds` (default 6 hours, bounding
  DST-transition staleness), unconditionally.
- Immediately when a normalized timestamp lands more than two minutes in the
  future — evidence that the broker offset grew, e.g. a UTC+2 to UTC+3
  transition.
- Periodically, at most once per `revalidation_interval_seconds` (default 5
  minutes), through one full revalidation round over every configured
  symbol. This is what catches a broker offset _decrease_ (e.g. UTC+3 to
  UTC+2): the resulting normalized time looks stale rather than future,
  which a future-skew check alone cannot distinguish from ordinary
  quiet-market staleness.

In a revalidation round each configured symbol is polled once and compared
against that symbol's last known observation (from the initial calibration
or a prior revalidation), and the whole round is evaluated before any
decision is made — revalidation never stops at the first confirming symbol.
A symbol is inconclusive — skipped, neither confirming nor refuting the
cache — when its fetch is unusable, when it has no prior observation to
compare against, or when its tick epoch is unchanged since that observation
(ordinary tick age, e.g. a closed or illiquid symbol). Every other advancing
sample is evidence the cache must be able to explain:

- an accepted offset equal to the cached one confirms the cache, but never
  cancels another symbol's refuting evidence from the same round;
- an accepted offset that differs from the cached one refutes the cache (a
  clean broker offset change) — unless the cached offset, widened by the
  elapsed time since that symbol's prior observation plus that prior
  observation's own inferred delay under the cached offset, still explains
  the disagreeing raw offset. That case is forgiven once per symbol and held as
  a pending disagreement instead of refuting immediately, since it cannot
  yet be told apart from a maximally delayed tick under the cached offset.
  A pending disagreement escalates to a refutation only if that same
  symbol's next conclusive round rounds to the identical bucket again; any
  inconclusive, differently-bucketed, or unusable round in between clears
  it instead of letting unrelated samples accumulate into false
  corroboration;
- an advancing sample with no usable offset (`unstable_offset` or
  `implausible_offset`) also refutes the cache, subject to the same
  once-per-symbol forgiveness above — but an unusable sample can never
  itself be held as, or match, a pending disagreement, since it carries no
  specific offset to corroborate. A fresh market event whose raw offset is
  neither reconciled with any plausible 30-minute bucket nor forgiven this
  way is contradictory evidence the cached calibration cannot explain.

Any refuting evidence invalidates the cached calibration and forces full
recalibration; when several symbols refute the cache in one round, the
reported evidence is deterministic (`implausible_offset` first, then
`unstable_offset`, then a clean disagreement). The recalibration either
re-establishes a validated offset or fails closed — a raw tick is never
normalized with a cached offset that advancing evidence has just refuted. A
wholly inconclusive round keeps the cache, and a round in which no symbol
had a comparable baseline does not consume the interval, so the next call
retries immediately instead of waiting a full
`revalidation_interval_seconds`.

If recalibration still cannot validate the timestamp, the snapshot fails
closed. Failed calibrations are recorded for diagnostics but never reused;
retries are bounded to at most once per `failed_calibration_retry_seconds`
(default 30 seconds) rather than on every call, so a closed or illiquid
market does not trigger a full resample on every snapshot request.

A genuinely stale tick (weekend, illiquid symbol) under a valid calibration
normalizes to its true **past** UTC instant — staleness is preserved, never
corrected to the present.

mt5cli owns MT5 timestamp normalization only. Freshness thresholds,
entry-blocking rules, and other strategy policy belong in downstream
applications.

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
| Local broker server clock offset measurement             | `TickClockNormalizer`                                                  |
| Local SL/TP price derivation                             | `determine_order_limits()`                                             |
| Throttled SQLite history loop with ad-hoc error handling | `ThrottledHistoryUpdater(suppress_errors=True)`                        |

Use `mt5_session()` / `MT5Client` for both data collection and generic
execution helpers.
