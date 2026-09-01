"""Tests for the :mod:`mt5cli.contract` module."""

from __future__ import annotations

from typing import Protocol

from mt5cli.contract import HistoryClient, ObservabilityClient


def test_protocols_are_defined() -> None:
    """HistoryClient and ObservabilityClient are structural protocols."""
    assert issubclass(HistoryClient, Protocol)
    assert issubclass(ObservabilityClient, Protocol)
