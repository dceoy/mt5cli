"""SQLite cursor indexes for throttled canonical history updates."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from .history import (
    _sqlite_normalized_time_expression,  # pyright: ignore[reportPrivateUsage]
    get_table_columns,
    quote_sqlite_identifier,
)
from .rates import ThrottledHistoryUpdater as _RatesThrottledHistoryUpdater
from .utils import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from .contract import HistoryClient

logger = logging.getLogger(__name__)

_CURSOR_INDEX_SPECS: tuple[
    tuple[str, Dataset, frozenset[str], tuple[str, ...]], ...
] = (
    (
        "idx_rates_symbol_timeframe_history_cursor",
        Dataset.rates,
        frozenset({"symbol", "timeframe", "time"}),
        ("symbol", "timeframe"),
    ),
    (
        "idx_ticks_symbol_history_cursor",
        Dataset.ticks,
        frozenset({"symbol", "time"}),
        ("symbol",),
    ),
    (
        "idx_history_orders_symbol_history_cursor",
        Dataset.history_orders,
        frozenset({"symbol", "time"}),
        ("symbol",),
    ),
    (
        "idx_history_deals_symbol_history_cursor",
        Dataset.history_deals,
        frozenset({"symbol", "time"}),
        ("symbol",),
    ),
    (
        "idx_history_deals_history_cursor",
        Dataset.history_deals,
        frozenset({"time"}),
        (),
    ),
)


def _ensure_incremental_cursor_indexes(output: Path | str) -> None:
    """Create persistent indexes matching normalized incremental cursor queries."""
    path = Path(output)
    if not path.is_file():
        return

    time_expression = _sqlite_normalized_time_expression("time")
    try:
        with sqlite3.connect(path) as conn:
            for (
                index_name,
                dataset,
                required_columns,
                prefix_columns,
            ) in _CURSOR_INDEX_SPECS:
                table_name = dataset.table_name
                columns = get_table_columns(conn, table_name)
                if not required_columns.issubset(columns):
                    continue
                quoted_index = quote_sqlite_identifier(index_name)
                quoted_table = quote_sqlite_identifier(table_name)
                index_columns = [
                    *(quote_sqlite_identifier(column) for column in prefix_columns),
                    time_expression,
                ]
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {quoted_index} "
                    f"ON {quoted_table} ({', '.join(index_columns)})"
                )
    except sqlite3.Error:
        logger.warning(
            "Could not create SQLite incremental history cursor indexes.",
            exc_info=True,
        )


class ThrottledHistoryUpdater(_RatesThrottledHistoryUpdater):
    """Canonical throttled updater with persistent incremental cursor indexes."""

    def update(
        self,
        client: HistoryClient,
        symbols: Sequence[str],
        *,
        date_to: datetime | str | None = None,
    ) -> bool:
        """Run an update and index its canonical cursor paths for later cycles.

        Returns:
            bool: Whether the underlying throttled history update ran successfully.
        """
        updated = super().update(client, symbols, date_to=date_to)
        if updated:
            _ensure_incremental_cursor_indexes(self.output)
        return updated


__all__ = ["ThrottledHistoryUpdater"]
