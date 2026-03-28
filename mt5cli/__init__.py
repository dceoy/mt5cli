"""mt5cli: Command-line tool for MetaTrader 5."""

from importlib.metadata import version

from .cli import detect_format, export_dataframe

__version__ = version(__package__) if __package__ else None

__all__ = [
    "detect_format",
    "export_dataframe",
]
