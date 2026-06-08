"""mt5cli: Command-line tool and SDK for MetaTrader 5."""

from importlib.metadata import version

from .sdk import (
    Mt5CliClient,
    account_info,
    build_config,
    collect_history,
    copy_rates_from,
    copy_rates_from_pos,
    copy_rates_range,
    copy_ticks_from,
    copy_ticks_range,
    history_deals,
    history_orders,
    last_error,
    market_book,
    minimum_margins,
    orders,
    positions,
    recent_ticks,
    symbol_info,
    symbol_info_tick,
    symbols,
    terminal_info,
    update_history,
    update_history_with_config,
)
from .sdk import (
    version as mt5_version,
)
from .utils import (
    Dataset,
    IfExists,
    detect_format,
    export_dataframe,
    export_dataframe_to_sqlite,
)

__version__ = version(__package__) if __package__ else None

__all__ = [
    "Dataset",
    "IfExists",
    "Mt5CliClient",
    "account_info",
    "build_config",
    "collect_history",
    "copy_rates_from",
    "copy_rates_from_pos",
    "copy_rates_range",
    "copy_ticks_from",
    "copy_ticks_range",
    "detect_format",
    "export_dataframe",
    "export_dataframe_to_sqlite",
    "history_deals",
    "history_orders",
    "last_error",
    "market_book",
    "minimum_margins",
    "mt5_version",
    "orders",
    "positions",
    "recent_ticks",
    "symbol_info",
    "symbol_info_tick",
    "symbols",
    "terminal_info",
    "update_history",
    "update_history_with_config",
]
