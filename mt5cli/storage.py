"""Generic storage helpers for MT5 market and account history."""

from __future__ import annotations

from .history import (
    RateTarget,
    build_rate_targets,
    build_rate_view_name,
    drop_forming_rate_bar,
    load_rate_data,
    load_rate_data_from_connection,
    load_rate_series_by_granularity,
    load_rate_series_from_sqlite,
    resolve_rate_tables,
    resolve_rate_view_name,
    resolve_rate_view_names,
)
from .sdk import collect_history, update_history, update_history_with_config
from .utils import (
    Dataset,
    IfExists,
    OutputFormat,
    detect_format,
    export_dataframe,
    export_dataframe_to_sqlite,
)

__all__ = [
    "Dataset",
    "IfExists",
    "OutputFormat",
    "RateTarget",
    "build_rate_targets",
    "build_rate_view_name",
    "collect_history",
    "detect_format",
    "drop_forming_rate_bar",
    "export_dataframe",
    "export_dataframe_to_sqlite",
    "load_rate_data",
    "load_rate_data_from_connection",
    "load_rate_series_by_granularity",
    "load_rate_series_from_sqlite",
    "resolve_rate_tables",
    "resolve_rate_view_name",
    "resolve_rate_view_names",
    "update_history",
    "update_history_with_config",
]
