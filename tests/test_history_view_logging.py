"""Tests for empty history view log levels."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd

from mt5cli.history import collect_history, write_incremental_datasets
from mt5cli.utils import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from pytest_mock import MockerFixture


def _mock_mt5_session(mocker: MockerFixture, client: MagicMock) -> None:
    session = MagicMock()
    session.__enter__.return_value = client
    session.__exit__.return_value = False
    mocker.patch("mt5cli.history.mt5_session", return_value=session)


def test_incremental_empty_deals_view_skip_logs_at_info(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty incremental deals should not produce a view warning."""
    client = MagicMock()
    client.history_deals.return_value = pd.DataFrame()
    with (
        sqlite3.connect(tmp_path / "incremental-empty-deals-views.db") as conn,
        caplog.at_level(logging.INFO, logger="mt5cli.history"),
    ):
        write_incremental_datasets(
            conn,
            client,
            ["EURUSD"],
            {Dataset.history_deals},
            [],
            0,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            deduplicate=True,
            create_rate_views=False,
            with_views=True,
            include_account_events=True,
        )
    expected = "Skipping history-deal views: no history_deals data was available"
    records = [record for record in caplog.records if record.message == expected]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]


def test_collect_empty_deals_view_skip_logs_at_info(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mocker: MockerFixture,
) -> None:
    """Empty one-shot deals should make the view skip informational."""
    client = MagicMock()
    client.history_deals.return_value = pd.DataFrame()
    _mock_mt5_session(mocker, client)
    with caplog.at_level(logging.INFO, logger="mt5cli.history"):
        collect_history(
            tmp_path / "collected-empty-deals-views.db",
            ["EURUSD"],
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datasets={Dataset.history_deals},
            with_views=True,
        )
    expected = "Skipping history-deal views: no history_deals data was collected"
    records = [record for record in caplog.records if record.message == expected]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]


def test_collect_views_without_deals_dataset_logs_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mocker: MockerFixture,
) -> None:
    """Requesting deal-derived views without deals remains a warning."""
    _mock_mt5_session(mocker, MagicMock())
    with caplog.at_level(logging.WARNING, logger="mt5cli.history"):
        collect_history(
            tmp_path / "collected-without-deals.db",
            ["EURUSD"],
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datasets=set(),
            with_views=True,
        )
    expected = "--with-views requires the history_deals dataset"
    records = [record for record in caplog.records if record.message == expected]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
