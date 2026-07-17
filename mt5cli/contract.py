"""Downstream SDK export tier for mt5cli."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd


class HistoryClient(Protocol):
    """Structural contract required by history collection and updates.

    :class:`~mt5cli.client.MT5Client` satisfies this protocol. It exists so
    that history collection code depends on canonical mt5cli method names
    instead of probing for raw pdmt5 method names.
    """

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: int | str,
        date_from: datetime | str,
        date_to: datetime | str,
    ) -> pd.DataFrame:
        """Return rates for a date range."""
        ...

    def copy_ticks_range(
        self,
        symbol: str,
        date_from: datetime | str,
        date_to: datetime | str,
        flags: int | str,
    ) -> pd.DataFrame:
        """Return ticks for a date range."""
        ...

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
        ...

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
        ...

    def symbol_info_as_dict(self, symbol: str) -> dict[str, object]:
        """Return one symbol snapshot as a plain mapping."""
        ...


class ObservabilityClient(Protocol):
    """Structural contract required by observability snapshot orchestration.

    :class:`~mt5cli.client.MT5Client` satisfies this protocol.
    """

    def account_info(self) -> pd.DataFrame:
        """Return account information."""
        ...

    def terminal_info(self) -> pd.DataFrame:
        """Return terminal information."""
        ...

    def positions(
        self,
        symbol: str | None = None,
        group: str | None = None,
        ticket: int | None = None,
    ) -> pd.DataFrame:
        """Return open positions."""
        ...

    def orders(
        self,
        symbol: str | None = None,
        group: str | None = None,
        ticket: int | None = None,
    ) -> pd.DataFrame:
        """Return active orders."""
        ...


STABLE_SDK_EXPORTS: frozenset[str] = frozenset({
    "AccountSpec",
    "MT5Client",
    "Mt5CliError",
    "Mt5ConnectionError",
    "Mt5OperationError",
    "Mt5SchemaError",
    "OrderFillingMode",
    "OrderSide",
    "OrderTimeMode",
    "PositionSide",
    "ProjectionMode",
    "CalibrationStatus",
    "ClockStatus",
    "ExecutionStatus",
    "MarginVolume",
    "NormalizedTickSnapshot",
    "OrderExecutionResult",
    "OrderLimits",
    "TickClockCalibration",
    "TickClockNormalizer",
    "RateTarget",
    "ThrottledHistoryUpdater",
    "build_config",
    "build_rate_targets",
    "calculate_account_projected_margin_ratio",
    "calculate_margin_and_volume",
    "calculate_new_position_margin_ratio",
    "calculate_projected_margin_ratio",
    "calculate_positions_margin",
    "calculate_positions_margin_by_symbol",
    "calculate_positions_margin_safe",
    "calculate_spread_ratio",
    "calculate_symbol_group_margin_ratio",
    "calculate_trailing_stop_updates",
    "calculate_volume_by_margin",
    "close_open_positions",
    "collect_history",
    "collect_latest_closed_rates_by_granularity",
    "collect_latest_closed_rates_for_accounts",
    "collect_latest_rates_for_accounts_with_retries",
    "detect_position_side",
    "determine_order_limits",
    "drop_forming_rate_bar",
    "ensure_symbol_selected",
    "estimate_order_margin",
    "extract_tick_price",
    "fetch_latest_closed_rates",
    "fetch_latest_closed_rates_indexed",
    "get_account_snapshot",
    "get_positions_frame",
    "get_symbol_snapshot",
    "get_tick_snapshot",
    "load_rate_series_by_granularity",
    "load_rate_series_from_sqlite",
    "mt5_session",
    "normalize_order_volume",
    "place_market_order",
    "report_rate_gaps",
    "resolve_broker_filling_mode",
    "resolve_account_spec",
    "resolve_account_specs",
    "update_history",
    "update_history_with_config",
    "update_observability",
    "update_observability_with_config",
    "update_sltp_for_open_positions",
    "update_trailing_stop_loss_for_open_positions",
})

__all__ = ["STABLE_SDK_EXPORTS", "HistoryClient", "ObservabilityClient"]
