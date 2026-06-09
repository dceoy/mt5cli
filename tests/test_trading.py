"""Tests for trading session helpers and operational utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
from pdmt5 import Mt5RuntimeError
from pytest_mock import MockerFixture  # noqa: TC002

from mt5cli.sdk import build_config
from mt5cli.trading import (
    calculate_margin_and_volume,
    detect_position_side,
    determine_order_limits,
    mt5_trading_session,
)


class TestDetectPositionSide:
    """Tests for detect_position_side."""

    def test_returns_none_when_no_positions(self) -> None:
        """Test None is returned when no open positions exist."""
        client = MagicMock()
        client.positions_get_as_df.return_value = pd.DataFrame()

        assert detect_position_side(client, "EURUSD") is None

    def test_returns_long_for_net_buy_volume(self) -> None:
        """Test long is returned when buy volume exceeds sell volume."""
        client = MagicMock()
        client.mt5.POSITION_TYPE_BUY = 0
        client.mt5.POSITION_TYPE_SELL = 1
        client.positions_get_as_df.return_value = pd.DataFrame(
            {
                "type": [0, 0, 1],
                "volume": [0.2, 0.1, 0.05],
            },
        )

        assert detect_position_side(client, "EURUSD") == "long"

    def test_returns_short_for_net_sell_volume(self) -> None:
        """Test short is returned when sell volume exceeds buy volume."""
        client = MagicMock()
        client.mt5.POSITION_TYPE_BUY = 0
        client.mt5.POSITION_TYPE_SELL = 1
        client.positions_get_as_df.return_value = pd.DataFrame(
            {
                "type": [1, 1],
                "volume": [0.3, 0.1],
            },
        )

        assert detect_position_side(client, "EURUSD") == "short"

    def test_returns_none_for_balanced_hedged_positions(self) -> None:
        """Test None is returned when buy and sell volumes net to zero."""
        client = MagicMock()
        client.mt5.POSITION_TYPE_BUY = 0
        client.mt5.POSITION_TYPE_SELL = 1
        client.positions_get_as_df.return_value = pd.DataFrame(
            {
                "type": [0, 1],
                "volume": [0.2, 0.2],
            },
        )

        assert detect_position_side(client, "EURUSD") is None


class TestCalculateMarginAndVolume:
    """Tests for calculate_margin_and_volume."""

    def test_calculates_margin_budget_and_volumes(self) -> None:
        """Test margin budget and buy/sell volumes are derived from ratios."""
        client = MagicMock()
        client.account_info_as_dict.return_value = {"margin_free": 1000.0}
        client.calculate_volume_by_margin.side_effect = [0.3, 0.2]

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.5,
            preserved_margin_ratio=0.2,
        )

        assert result == {
            "margin_free": 1000.0,
            "available_margin": 800.0,
            "trade_margin": 400.0,
            "buy_volume": 0.3,
            "sell_volume": 0.2,
        }
        client.calculate_volume_by_margin.assert_any_call("EURUSD", 400.0, "BUY")
        client.calculate_volume_by_margin.assert_any_call("EURUSD", 400.0, "SELL")

    @pytest.mark.parametrize(
        ("unit_ratio", "preserved_ratio"),
        [
            (-0.1, 0.0),
            (1.1, 0.0),
            (0.5, -0.1),
            (0.5, 1.1),
        ],
    )
    def test_rejects_invalid_ratios(
        self,
        unit_ratio: float,
        preserved_ratio: float,
    ) -> None:
        """Test invalid ratio values raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            calculate_margin_and_volume(
                MagicMock(),
                "EURUSD",
                unit_margin_ratio=unit_ratio,
                preserved_margin_ratio=preserved_ratio,
            )


class TestDetermineOrderLimits:
    """Tests for determine_order_limits."""

    @pytest.mark.parametrize(
        ("side", "expected_entry_key"),
        [
            ("long", "ask"),
            ("short", "bid"),
            ("buy", "ask"),
            ("sell", "bid"),
        ],
    )
    def test_uses_expected_quote_for_entry(
        self,
        side: str,
        expected_entry_key: str,
    ) -> None:
        """Test entry price is taken from ask for long/buy and bid for short/sell."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}

        result = determine_order_limits(
            client,
            "EURUSD",
            side,
            stop_loss_limit_ratio=0.0,
            take_profit_limit_ratio=0.0,
        )

        assert (
            result["entry"]
            == client.symbol_info_tick_as_dict.return_value[expected_entry_key]
        )
        assert result["stop_loss"] is None
        assert result["take_profit"] is None

    def test_calculates_long_protective_levels(self) -> None:
        """Test long stop loss and take profit are placed below/above entry."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}

        result = determine_order_limits(
            client,
            "EURUSD",
            "long",
            stop_loss_limit_ratio=0.02,
            take_profit_limit_ratio=0.03,
        )

        assert result == {
            "entry": 100.0,
            "stop_loss": 98.0,
            "take_profit": 103.0,
        }

    def test_calculates_short_protective_levels(self) -> None:
        """Test short stop loss and take profit are placed above/below entry."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}

        result = determine_order_limits(
            client,
            "EURUSD",
            "short",
            stop_loss_limit_ratio=0.02,
            take_profit_limit_ratio=0.03,
        )

        assert result == {
            "entry": 99.0,
            "stop_loss": 100.98,
            "take_profit": 96.03,
        }

    def test_rejects_unknown_side(self) -> None:
        """Test unsupported side values raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported order side"):
            determine_order_limits(
                MagicMock(),
                "EURUSD",
                "flat",
                stop_loss_limit_ratio=0.01,
                take_profit_limit_ratio=0.01,
            )


class TestMt5TradingSession:
    """Tests for the mt5_trading_session context manager."""

    def test_yields_connected_client_and_shuts_down(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test mt5_trading_session connects, yields a client, and shuts down."""
        mock_client = MagicMock()
        trading_client = mocker.patch(
            "mt5cli.trading.Mt5TradingClient",
            return_value=mock_client,
        )

        with mt5_trading_session(
            build_config(path="/opt/mt5/terminal64.exe"),
            retry_count=2,
        ) as client:
            mock_client.initialize_and_login_mt5.assert_called_once()
            assert client is mock_client

        trading_client.assert_called_once()
        assert trading_client.call_args.kwargs["retry_count"] == 2
        assert (
            trading_client.call_args.kwargs["config"].path == "/opt/mt5/terminal64.exe"
        )
        mock_client.shutdown.assert_called_once()

    def test_shuts_down_when_initialize_raises(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test shutdown is called when initialization fails."""
        mock_client = MagicMock()
        mock_client.initialize_and_login_mt5.side_effect = Mt5RuntimeError("boom")
        mocker.patch("mt5cli.trading.Mt5TradingClient", return_value=mock_client)

        with pytest.raises(Mt5RuntimeError, match="boom"), mt5_trading_session():
            pass

        mock_client.shutdown.assert_called_once()
