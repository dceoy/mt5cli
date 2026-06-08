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
    orders,
    positions,
    symbol_info,
    symbol_info_tick,
    symbols,
    terminal_info,
)
from .sdk import (
    version as mt5_version,
)
from .utils import detect_format, export_dataframe

__version__ = version(__package__) if __package__ else None

__all__ = [
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
    "history_deals",
    "history_orders",
    "last_error",
    "market_book",
    "mt5_version",
    "orders",
    "positions",
    "symbol_info",
    "symbol_info_tick",
    "symbols",
    "terminal_info",
]
