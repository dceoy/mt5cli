"""Authoritative stable package-root SDK surface for mt5cli."""
# pyright: reportUnusedImport=false, reportUnsupportedDunderAll=false
# ruff: noqa: F401, PLE0605

from pdmt5.dataframe import Mt5Config

from .client import MT5Client, build_config, mt5_session, substitute_mapping_values
from .exceptions import (
    Mt5CliError,
    Mt5ConnectionError,
    Mt5OperationError,
    Mt5SchemaError,
)
from .grafana import ensure_grafana_schema
from .history import (
    RateTarget,
    build_rate_targets,
    collect_history,
    drop_forming_rate_bar,
    report_rate_gaps,
    resolve_history_timeframes,
)
from .lifecycle import switch_account
from .marketdata import (
    AccountSpec,
    collect_latest_closed_rates_by_granularity,
    collect_latest_closed_rates_for_accounts,
    collect_latest_rates_for_accounts_with_retries,
    fetch_latest_closed_rates,
    resolve_account_spec,
    resolve_account_specs,
)
from .observability import (
    ObservabilitySnapshot,
    capture_observability_snapshot,
    persist_observability_snapshot,
    update_observability,
    update_observability_with_config,
)
from .rates import (
    HistoryCapture,
    ThrottledHistoryUpdater,
    capture_history_datasets,
    load_rate_series_by_granularity,
    load_rate_series_from_sqlite,
    persist_history_datasets,
    update_history,
    update_history_with_config,
)
from .trading import (
    CalibrationStatus,
    ClockStatus,
    ExecutionStatus,
    MarginVolume,
    NormalizedTickSnapshot,
    OrderExecutionResult,
    OrderFillingMode,
    OrderLimits,
    OrderSide,
    OrderTimeMode,
    PositionSide,
    ProjectionMode,
    TickClockCalibration,
    TickClockNormalizer,
    calculate_account_projected_margin_ratio,
    calculate_margin_and_volume,
    calculate_positions_margin,
    calculate_positions_margin_by_symbol,
    calculate_positions_margin_safe,
    calculate_projected_margin_ratio,
    calculate_spread_ratio,
    calculate_symbol_group_margin_ratio,
    calculate_trailing_stop_updates,
    calculate_volume_by_margin,
    close_open_positions,
    detect_position_side,
    determine_order_limits,
    ensure_symbol_selected,
    estimate_order_margin,
    extract_tick_price,
    fetch_latest_closed_rates_indexed,
    get_account_snapshot,
    get_positions_frame,
    get_symbol_snapshot,
    get_tick_snapshot,
    normalize_order_volume,
    place_market_order,
    resolve_broker_filling_mode,
    update_sltp_for_open_positions,
    update_trailing_stop_loss_for_open_positions,
)
from .utils import Dataset, parse_timeframe

# Public imports above are the single stable-SDK declaration. Keep implementation
# helpers private (leading underscore) so the enumerable contract cannot drift from
# the package-root imports through a second manually maintained symbol list.
__all__ = sorted(name for name in globals() if not name.startswith("_"))
STABLE_SDK_EXPORTS: frozenset[str] = frozenset(__all__)
