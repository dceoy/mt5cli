"""Tests for trading session helpers and operational utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import pytest
from numpy import int64 as np_int64
from pdmt5 import Mt5RuntimeError, Mt5TradingClient, Mt5TradingError
from pytest_mock import MockerFixture  # noqa: TC002

from mt5cli.sdk import build_config
from mt5cli.trading import (
    MarginVolume,
    OrderExecutionResult,
    OrderLimits,
    calculate_margin_and_volume,
    calculate_new_position_margin_ratio,
    calculate_positions_margin,
    calculate_spread_ratio,
    calculate_volume_by_margin,
    close_open_positions,
    create_trading_client,
    detect_position_side,
    determine_order_limits,
    ensure_symbol_selected,
    estimate_order_margin,
    fetch_latest_closed_rates_for_trading_client,
    fetch_latest_closed_rates_indexed,
    get_account_snapshot,
    get_positions_frame,
    get_symbol_snapshot,
    get_tick_snapshot,
    mt5_trading_session,
    normalize_order_volume,
    place_market_order,
    update_sltp_for_open_positions,
)


def _mock_trade_client() -> MagicMock:
    client = MagicMock()
    client.mt5.POSITION_TYPE_BUY = 0
    client.mt5.POSITION_TYPE_SELL = 1
    client.mt5.ORDER_TYPE_BUY = 10
    client.mt5.ORDER_TYPE_SELL = 11
    client.mt5.TRADE_ACTION_DEAL = 20
    client.mt5.TRADE_ACTION_SLTP = 21
    client.mt5.ORDER_FILLING_IOC = 30
    client.mt5.ORDER_TIME_GTC = 40
    client.mt5.TRADE_RETCODE_PLACED = 10008
    client.mt5.TRADE_RETCODE_DONE = 10009
    client.mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
    return client


def _assert_close(actual: object, expected: float) -> None:
    assert abs(float(cast("float", actual)) - expected) < 1e-9


def _request_from_result(result: OrderExecutionResult) -> dict[str, object]:  # noqa: FURB118
    return result["request"]


class TestDetectPositionSide:
    """Tests for detect_position_side."""

    def test_returns_none_when_no_positions(self) -> None:
        """Test None is returned when no open positions exist."""
        client = MagicMock()
        client.positions_get_as_df.return_value = pd.DataFrame()

        assert detect_position_side(client, "EURUSD") is None

    def test_returns_long_for_buy_only_exposure(self) -> None:
        """Test long is returned when only buy positions exist."""
        client = MagicMock()
        client.mt5.POSITION_TYPE_BUY = 0
        client.mt5.POSITION_TYPE_SELL = 1
        client.positions_get_as_df.return_value = pd.DataFrame(
            {
                "type": [0, 0],
                "volume": [0.2, 0.1],
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

    def test_returns_none_for_mixed_hedged_positions(self) -> None:
        """Test None is returned for mixed buy and sell exposure."""
        client = MagicMock()
        client.mt5.POSITION_TYPE_BUY = 0
        client.mt5.POSITION_TYPE_SELL = 1
        client.positions_get_as_df.return_value = pd.DataFrame(
            {
                "type": [0, 1],
                "volume": [0.3, 0.2],
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
        client.symbol_info_as_dict.side_effect = AttributeError("missing")

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
            "volume_min": 0.0,
            "volume_max": 0.0,
            "volume_step": 0.0,
        }
        client.calculate_volume_by_margin.assert_any_call("EURUSD", 400.0, "BUY")
        client.calculate_volume_by_margin.assert_any_call("EURUSD", 400.0, "SELL")

    @pytest.mark.parametrize(
        ("account_dict", "expected_margin_free"),
        [
            ({"margin_free": 0.0}, 0.0),
            ({}, 0.0),
            ({"margin_free": None}, 0.0),
        ],
    )
    def test_zero_or_missing_margin_free(
        self,
        account_dict: dict[str, float | None],
        expected_margin_free: float,
    ) -> None:
        """Test missing or zero margin_free yields zero trade margin."""
        client = MagicMock()
        client.account_info_as_dict.return_value = account_dict
        client.calculate_volume_by_margin.return_value = 0.0

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.5,
            preserved_margin_ratio=0.2,
        )

        assert result["margin_free"] == expected_margin_free
        client.calculate_volume_by_margin.assert_any_call("EURUSD", 0.0, "BUY")
        client.calculate_volume_by_margin.assert_any_call("EURUSD", 0.0, "SELL")

    def test_clamps_negative_margin_free_to_zero(self) -> None:
        """Test negative margin_free is clamped to zero before sizing."""
        client = MagicMock()
        client.account_info_as_dict.return_value = {"margin_free": -500.0}
        client.calculate_volume_by_margin.return_value = 0.0

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.5,
            preserved_margin_ratio=0.2,
        )

        expected_margin_free = 0.0
        assert result["margin_free"] == expected_margin_free
        client.calculate_volume_by_margin.assert_any_call("EURUSD", 0.0, "BUY")
        client.calculate_volume_by_margin.assert_any_call("EURUSD", 0.0, "SELL")

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
        client.symbol_info_as_dict.return_value = {}

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
        client.symbol_info_as_dict.return_value = {}

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
        with pytest.raises(ValueError, match="Unsupported position side"):
            determine_order_limits(
                MagicMock(),
                "EURUSD",
                "flat",
                stop_loss_limit_ratio=0.01,
                take_profit_limit_ratio=0.01,
            )

    @pytest.mark.parametrize(
        ("stop_loss_ratio", "take_profit_ratio"),
        [
            (-0.05, 0.01),
            (0.01, 2.0),
        ],
    )
    def test_rejects_invalid_protective_ratios(
        self,
        stop_loss_ratio: float,
        take_profit_ratio: float,
    ) -> None:
        """Test out-of-range protective ratios raise ValueError."""
        with pytest.raises(ValueError, match="must be at least 0 and less than 1"):
            determine_order_limits(
                MagicMock(),
                "EURUSD",
                "long",
                stop_loss_limit_ratio=stop_loss_ratio,
                take_profit_limit_ratio=take_profit_ratio,
            )

    @pytest.mark.parametrize(
        ("field", "ratio"),
        [
            ("stop_loss_limit_ratio", 1.0),
            ("take_profit_limit_ratio", 1.0),
        ],
    )
    def test_rejects_unit_boundary_protective_ratios(
        self,
        field: str,
        ratio: float,
    ) -> None:
        """Test protective ratios of exactly 1.0 are rejected."""
        kwargs = {
            "stop_loss_limit_ratio": 0.01,
            "take_profit_limit_ratio": 0.01,
            field: ratio,
        }
        with pytest.raises(ValueError, match="must be at least 0 and less than 1"):
            determine_order_limits(
                MagicMock(),
                "EURUSD",
                "long",
                **kwargs,
            )

    def test_uses_default_digits_when_symbol_snapshot_fails(self) -> None:
        """Test order limit rounding falls back when symbol metadata is missing."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.234567891, "bid": 1.0}
        client.symbol_info_as_dict.return_value = {"digits": "invalid"}

        result = determine_order_limits(
            client,
            "EURUSD",
            "long",
            stop_loss_limit_ratio=0.01,
            take_profit_limit_ratio=0.01,
        )

        _assert_close(result["stop_loss"], 1.22222221)

    def test_uses_default_digits_when_symbol_lookup_raises(self) -> None:
        """Test order limits fall back when symbol metadata lookup fails."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.234567891, "bid": 1.0}
        client.symbol_info_as_dict.side_effect = AttributeError("missing")

        result = determine_order_limits(
            client,
            "EURUSD",
            "long",
            stop_loss_limit_ratio=0.01,
            take_profit_limit_ratio=0.01,
        )

        _assert_close(result["stop_loss"], 1.22222221)

    def test_uses_tick_snapshot_fallback(self) -> None:
        """Test order limits use the normalized tick snapshot helper."""
        client = MagicMock()
        del client.symbol_info_tick_as_dict
        client.symbol_info_tick.return_value = SimpleNamespace(ask=1.2, bid=1.1)
        client.symbol_info_as_dict.return_value = {"digits": 4}

        result = determine_order_limits(
            client,
            "EURUSD",
            "long",
            stop_loss_limit_ratio=0.01,
        )

        _assert_close(result["entry"], 1.2)
        _assert_close(result["stop_loss"], 1.188)

    def test_rejects_missing_entry_tick(self) -> None:
        """Test missing entry prices raise a trading error."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": None, "bid": 1.1}

        with pytest.raises(Mt5TradingError, match="Tick price is unavailable"):
            determine_order_limits(client, "EURUSD", "long")

    def test_rejects_stop_loss_inside_broker_stop_level(self) -> None:
        """Test stop-loss prices closer than trade_stops_level raise Mt5TradingError."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.0, "bid": 0.99}
        client.symbol_info_as_dict.return_value = {
            "digits": 2,
            "trade_stops_level": 100,
            "point": 0.0001,
        }

        with pytest.raises(Mt5TradingError, match="Stop loss for 'EURUSD'"):
            determine_order_limits(
                client,
                "EURUSD",
                "long",
                stop_loss_limit_ratio=0.0001,
            )

    def test_accepts_stop_loss_exactly_at_minimum_stop_distance(self) -> None:
        """Test protective levels exactly at trade_stops_level distance pass."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.0, "bid": 0.99}
        client.symbol_info_as_dict.return_value = {
            "digits": 2,
            "trade_stops_level": 100,
            "point": 0.0001,
        }

        result = determine_order_limits(
            client,
            "EURUSD",
            "long",
            stop_loss_limit_ratio=0.01,
            take_profit_limit_ratio=0.0,
        )

        _assert_close(result["stop_loss"], 0.99)

    def test_allows_protective_levels_beyond_broker_stop_level(self) -> None:
        """Test SL/TP beyond trade_stops_level pass validation."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.0, "bid": 0.99}
        client.symbol_info_as_dict.return_value = {
            "digits": 2,
            "trade_stops_level": 10,
            "point": 0.0001,
        }

        result = determine_order_limits(
            client,
            "EURUSD",
            "long",
            stop_loss_limit_ratio=0.05,
            take_profit_limit_ratio=0.05,
        )

        _assert_close(result["stop_loss"], 0.95)
        _assert_close(result["take_profit"], 1.05)

    def test_rejects_take_profit_inside_broker_stop_level(self) -> None:
        """Test long take-profit inside trade_stops_level raises Mt5TradingError."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.0, "bid": 0.99}
        client.symbol_info_as_dict.return_value = {
            "digits": 2,
            "trade_stops_level": 100,
            "point": 0.0001,
        }

        with pytest.raises(Mt5TradingError, match="Take profit for 'EURUSD'"):
            determine_order_limits(
                client,
                "EURUSD",
                "long",
                take_profit_limit_ratio=0.0001,
            )

    def test_rejects_short_stop_loss_inside_broker_stop_level(self) -> None:
        """Test short stop-loss inside trade_stops_level raises Mt5TradingError."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.01, "bid": 1.0}
        client.symbol_info_as_dict.return_value = {
            "digits": 2,
            "trade_stops_level": 100,
            "point": 0.0001,
        }

        with pytest.raises(Mt5TradingError, match="Stop loss for 'EURUSD'"):
            determine_order_limits(
                client,
                "EURUSD",
                "short",
                stop_loss_limit_ratio=0.0001,
            )

    def test_rejects_short_take_profit_inside_broker_stop_level(self) -> None:
        """Test short take-profit inside trade_stops_level raises Mt5TradingError."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.01, "bid": 1.0}
        client.symbol_info_as_dict.return_value = {
            "digits": 2,
            "trade_stops_level": 100,
            "point": 0.0001,
        }

        with pytest.raises(Mt5TradingError, match="Take profit for 'EURUSD'"):
            determine_order_limits(
                client,
                "EURUSD",
                "short",
                take_profit_limit_ratio=0.0001,
            )

    def test_allows_short_protective_levels_beyond_broker_stop_level(self) -> None:
        """Test short SL/TP beyond trade_stops_level pass validation."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.01, "bid": 1.0}
        client.symbol_info_as_dict.return_value = {
            "digits": 2,
            "trade_stops_level": 10,
            "point": 0.0001,
        }

        result = determine_order_limits(
            client,
            "EURUSD",
            "short",
            stop_loss_limit_ratio=0.05,
            take_profit_limit_ratio=0.05,
        )

        _assert_close(result["stop_loss"], 1.05)
        _assert_close(result["take_profit"], 0.95)

    def test_ignores_non_positive_broker_stop_level(self) -> None:
        """Test zero trade_stops_level skips stop-distance validation."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.0, "bid": 0.99}
        client.symbol_info_as_dict.return_value = {
            "digits": 2,
            "trade_stops_level": 0,
            "point": 0.0001,
        }

        result = determine_order_limits(
            client,
            "EURUSD",
            "long",
            stop_loss_limit_ratio=0.0001,
            take_profit_limit_ratio=0.0001,
        )

        assert result["stop_loss"] is not None
        assert result["take_profit"] is not None

    """Tests for ensure_symbol_selected."""

    def test_skips_selection_when_symbol_is_visible(self) -> None:
        """Test visible symbols do not call symbol_select."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"visible": True}

        ensure_symbol_selected(client, "EURUSD")

        client.symbol_select.assert_not_called()

    def test_selects_hidden_symbol_before_trading(self) -> None:
        """Test hidden symbols are selected in Market Watch."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"visible": False}
        client.symbol_select.return_value = True

        ensure_symbol_selected(client, "EURUSD")

        client.symbol_select.assert_called_once_with("EURUSD", enable=True)

    def test_raises_when_symbol_selection_fails(self) -> None:
        """Test failed symbol selection raises Mt5TradingError."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"visible": False}
        client.symbol_select.return_value = False
        client.last_error.return_value = (1, "not found")

        with pytest.raises(Mt5TradingError, match="Failed to select symbol 'EURUSD'"):
            ensure_symbol_selected(client, "EURUSD")

    def test_raises_when_symbol_select_is_unavailable(self) -> None:
        """Test missing symbol_select raises Mt5TradingError."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"visible": False}
        del client.symbol_select

        with pytest.raises(
            Mt5TradingError,
            match="missing required method: symbol_select",
        ):
            ensure_symbol_selected(client, "EURUSD")


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


class TestCreateTradingClient:
    """Tests for create_trading_client."""

    def test_initializes_with_keyword_config(self, mocker: MockerFixture) -> None:
        """Test keyword configuration is forwarded to Mt5TradingClient."""
        mock_client = MagicMock()
        trading_client = mocker.patch(
            "mt5cli.trading.Mt5TradingClient",
            return_value=mock_client,
        )

        result = create_trading_client(
            login="12345",
            password="test-pass",
            server="Demo",
            path="/opt/terminal64.exe",
            retry_count=2,
        )

        assert result is mock_client
        config = trading_client.call_args.kwargs["config"]
        assert config.login == 12345
        assert config.password == ("test" + "-pass")
        assert config.server == "Demo"
        assert config.path == "/opt/terminal64.exe"
        assert trading_client.call_args.kwargs["retry_count"] == 2
        mock_client.initialize_and_login_mt5.assert_called_once()

    def test_empty_login_string_is_unset(self, mocker: MockerFixture) -> None:
        """Test empty login strings are treated as None."""
        trading_client = mocker.patch(
            "mt5cli.trading.Mt5TradingClient",
            return_value=MagicMock(),
        )

        create_trading_client(login=" ")

        config = trading_client.call_args.kwargs["config"]
        assert config.login is None

    def test_shutdown_on_initialization_failure(self, mocker: MockerFixture) -> None:
        """Test failed initialization shuts the client down."""
        mock_client = MagicMock()
        mock_client.initialize_and_login_mt5.side_effect = Mt5RuntimeError("boom")
        mocker.patch("mt5cli.trading.Mt5TradingClient", return_value=mock_client)

        with pytest.raises(Mt5RuntimeError, match="boom"):
            create_trading_client()

        mock_client.shutdown.assert_called_once()


class TestSnapshotsAndState:
    """Tests for normalized state helpers."""

    def test_account_snapshot_includes_missing_fields(self) -> None:
        """Test account snapshot has stable keys with None for missing values."""
        client = MagicMock()
        client.account_info_as_dict.return_value = {"login": 1, "equity": 100.0}

        result = get_account_snapshot(client)

        assert result["login"] == 1
        _assert_close(result["equity"], 100.0)
        assert result["currency"] is None

    def test_account_snapshot_supports_object_fallback(self) -> None:
        """Test account snapshots can read plain MT5-like objects."""

        class ObjectClient:
            def account_info(self) -> SimpleNamespace:
                return SimpleNamespace(login=7, currency="JPY")

        result = get_account_snapshot(cast("Mt5TradingClient", ObjectClient()))

        assert result["login"] == 7
        assert result["currency"] == "JPY"

    def test_account_snapshot_requires_supported_method(self) -> None:
        """Test missing account snapshot methods raise AttributeError."""
        client = object()

        with pytest.raises(AttributeError, match="account_info"):
            get_account_snapshot(cast("Mt5TradingClient", client))

    def test_symbol_and_tick_snapshots_fill_symbol(self) -> None:
        """Test symbol and tick snapshots expose stable fields."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"digits": 5, "visible": True}
        client.symbol_info_tick_as_dict.return_value = {"bid": 1.1, "ask": 1.2}

        assert get_symbol_snapshot(client, "EURUSD")["symbol"] == "EURUSD"
        assert get_tick_snapshot(client, "EURUSD")["symbol"] == "EURUSD"

    def test_symbol_and_tick_snapshots_use_object_fallbacks(self) -> None:
        """Test symbol and tick snapshots support non-dict MT5 values."""
        client = MagicMock()
        del client.symbol_info_as_dict
        del client.symbol_info_tick_as_dict
        client.symbol_info.return_value = SimpleNamespace(digits=3)
        client.symbol_info_tick.return_value = SimpleNamespace(bid=1.0, ask=1.1)

        assert get_symbol_snapshot(client, "USDJPY")["digits"] == 3
        _assert_close(get_tick_snapshot(client, "USDJPY")["ask"], 1.1)

    def test_positions_frame_adds_stable_columns(self) -> None:
        """Test missing position columns are added to empty frames."""
        client = MagicMock()
        client.positions_get_as_df.return_value = pd.DataFrame()

        result = get_positions_frame(client, symbol="EURUSD")

        assert "ticket" in result.columns
        assert "comment" in result.columns

    def test_calculate_spread_ratio(self) -> None:
        """Test spread ratio uses mid-price denominator."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"bid": 99.0, "ask": 101.0}

        _assert_close(calculate_spread_ratio(client, "EURUSD"), 0.02)

    def test_calculate_spread_ratio_rejects_missing_tick(self) -> None:
        """Test missing bid/ask raises a trading error."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"bid": None, "ask": 1.0}

        with pytest.raises(Mt5TradingError):
            calculate_spread_ratio(client, "EURUSD")

    def test_calculate_spread_ratio_rejects_non_positive_tick(self) -> None:
        """Test non-positive bid/ask raises a trading error."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"bid": 0.0, "ask": 1.0}

        with pytest.raises(Mt5TradingError):
            calculate_spread_ratio(client, "EURUSD")


class TestNormalizeOrderVolume:
    """Tests for normalize_order_volume."""

    def test_returns_exact_minimum_volume(self) -> None:
        """Test exact volume_min is returned unchanged."""
        _assert_close(
            normalize_order_volume(
                0.1,
                volume_min=0.1,
                volume_max=1.0,
                volume_step=0.1,
            ),
            0.1,
        )

    def test_floors_to_step_between_boundaries(self) -> None:
        """Test volume between steps floors down to the nearest valid step."""
        _assert_close(
            normalize_order_volume(
                0.25,
                volume_min=0.1,
                volume_max=1.0,
                volume_step=0.1,
            ),
            0.2,
        )

    def test_clamps_to_volume_max(self) -> None:
        """Test positive volume_max caps the normalized result."""
        _assert_close(
            normalize_order_volume(
                0.9,
                volume_min=0.1,
                volume_max=0.5,
                volume_step=0.1,
            ),
            0.5,
        )

    def test_returns_zero_below_volume_min(self) -> None:
        """Test sub-minimum requests return zero volume."""
        _assert_close(
            normalize_order_volume(
                0.05,
                volume_min=0.1,
                volume_max=1.0,
                volume_step=0.1,
            ),
            0.0,
        )

    def test_returns_zero_for_invalid_volume_min(self) -> None:
        """Test non-positive volume_min returns zero volume."""
        _assert_close(
            normalize_order_volume(
                1.0,
                volume_min=0.0,
                volume_max=1.0,
                volume_step=0.1,
            ),
            0.0,
        )

    def test_returns_zero_for_invalid_volume_step(self) -> None:
        """Test non-positive volume_step returns zero volume."""
        _assert_close(
            normalize_order_volume(
                1.0,
                volume_min=0.1,
                volume_max=1.0,
                volume_step=0.0,
            ),
            0.0,
        )

    def test_treats_non_positive_volume_max_as_no_cap(self) -> None:
        """Test volume_max <= 0 disables the maximum cap."""
        _assert_close(
            normalize_order_volume(
                2.5,
                volume_min=0.1,
                volume_max=0.0,
                volume_step=0.1,
            ),
            2.5,
        )

    def test_reapplies_volume_max_after_step_normalization(self) -> None:
        """Test post-step normalization cannot exceed volume_max."""
        _assert_close(
            normalize_order_volume(
                0.5,
                volume_min=0.1,
                volume_max=0.34,
                volume_step=0.12,
            ),
            0.34,
        )

    def test_returns_zero_for_non_finite_volume(self) -> None:
        """Test NaN or infinite requested volume returns zero."""
        _assert_close(
            normalize_order_volume(
                float("nan"),
                volume_min=0.1,
                volume_max=1.0,
                volume_step=0.1,
            ),
            0.0,
        )
        _assert_close(
            normalize_order_volume(
                float("inf"),
                volume_min=0.1,
                volume_max=1.0,
                volume_step=0.1,
            ),
            0.0,
        )

    def test_returns_zero_for_non_finite_constraints(self) -> None:
        """Test NaN or infinite volume_min/volume_step returns zero."""
        _assert_close(
            normalize_order_volume(
                1.0,
                volume_min=float("nan"),
                volume_max=1.0,
                volume_step=0.1,
            ),
            0.0,
        )
        _assert_close(
            normalize_order_volume(
                1.0,
                volume_min=0.1,
                volume_max=1.0,
                volume_step=float("inf"),
            ),
            0.0,
        )

    def test_treats_non_finite_volume_max_as_no_cap(self) -> None:
        """Test non-finite volume_max disables the maximum cap."""
        _assert_close(
            normalize_order_volume(
                2.5,
                volume_min=0.1,
                volume_max=float("nan"),
                volume_step=0.1,
            ),
            2.5,
        )


class TestEstimateOrderMargin:
    """Tests for estimate_order_margin."""

    def test_estimates_buy_margin_at_ask(self) -> None:
        """Test buy margin uses ask price and buy order type."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = 12.5

        margin = estimate_order_margin(client, "EURUSD", "BUY", 0.1)

        _assert_close(margin, 12.5)
        client.order_calc_margin.assert_called_once_with(10, "EURUSD", 0.1, 1.1010)

    def test_estimates_sell_margin_at_bid(self) -> None:
        """Test sell margin uses bid price and sell order type."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = 12.4

        margin = estimate_order_margin(client, "EURUSD", "SELL", 0.1)

        _assert_close(margin, 12.4)
        client.order_calc_margin.assert_called_once_with(11, "EURUSD", 0.1, 1.1000)

    def test_accepts_long_and_short_aliases(self) -> None:
        """Test long/short aliases normalize to buy/sell pricing."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.side_effect = [12.5, 12.4]

        estimate_order_margin(client, "EURUSD", "long", 0.1)
        estimate_order_margin(client, "EURUSD", "short", 0.1)

        client.order_calc_margin.assert_any_call(10, "EURUSD", 0.1, 1.1010)
        client.order_calc_margin.assert_any_call(11, "EURUSD", 0.1, 1.1000)

    def test_rejects_invalid_side(self) -> None:
        """Test unsupported order side raises ValueError."""
        client = _mock_trade_client()

        with pytest.raises(ValueError, match="Unsupported order side"):
            estimate_order_margin(client, "EURUSD", "HOLD", 0.1)

    def test_rejects_non_positive_volume(self) -> None:
        """Test non-positive volume raises Mt5TradingError."""
        client = _mock_trade_client()

        with pytest.raises(Mt5TradingError, match="positive finite number"):
            estimate_order_margin(client, "EURUSD", "BUY", 0.0)

    def test_rejects_nan_volume(self) -> None:
        """Test NaN volume raises Mt5TradingError without broker calls."""
        client = _mock_trade_client()

        with pytest.raises(Mt5TradingError, match="positive finite number"):
            estimate_order_margin(client, "EURUSD", "BUY", float("nan"))

        client.symbol_info_tick_as_dict.assert_not_called()
        client.order_calc_margin.assert_not_called()

    def test_rejects_infinite_volume(self) -> None:
        """Test infinite volume raises Mt5TradingError without broker calls."""
        client = _mock_trade_client()

        with pytest.raises(Mt5TradingError, match="positive finite number"):
            estimate_order_margin(client, "EURUSD", "BUY", float("inf"))

        client.symbol_info_tick_as_dict.assert_not_called()
        client.order_calc_margin.assert_not_called()

    def test_rejects_missing_tick_prices(self) -> None:
        """Test missing tick prices raise Mt5TradingError."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": None, "bid": 1.1000}

        with pytest.raises(Mt5TradingError, match="Tick price is unavailable"):
            estimate_order_margin(client, "EURUSD", "BUY", 0.1)

    def test_rejects_non_positive_tick_price(self) -> None:
        """Test non-positive tick prices raise Mt5TradingError."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 0.0, "bid": 1.1000}

        with pytest.raises(Mt5TradingError, match="Tick price is unavailable"):
            estimate_order_margin(client, "EURUSD", "BUY", 0.1)

    def test_rejects_non_finite_tick_price(self) -> None:
        """Test non-finite tick prices raise Mt5TradingError."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {
            "ask": float("inf"),
            "bid": 1.1000,
        }

        with pytest.raises(Mt5TradingError, match="Tick price is unavailable"):
            estimate_order_margin(client, "EURUSD", "BUY", 0.1)

    def test_rejects_invalid_margin_result(self) -> None:
        """Test non-positive margin estimates raise Mt5TradingError."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = 0.0

        with pytest.raises(Mt5TradingError, match="Margin estimate is invalid"):
            estimate_order_margin(client, "EURUSD", "BUY", 0.1)

    def test_rejects_non_finite_margin_result(self) -> None:
        """Test non-finite margin estimates raise Mt5TradingError."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = float("inf")

        with pytest.raises(Mt5TradingError, match="Margin estimate is invalid"):
            estimate_order_margin(client, "EURUSD", "BUY", 0.1)

    def test_rejects_none_margin_result(self) -> None:
        """Test None margin results raise Mt5TradingError."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = None

        with pytest.raises(Mt5TradingError, match="Margin estimate is invalid"):
            estimate_order_margin(client, "EURUSD", "BUY", 0.1)

    def test_rejects_non_numeric_margin_result(self) -> None:
        """Test non-numeric margin results raise Mt5TradingError."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = "invalid"

        with pytest.raises(Mt5TradingError, match="Margin estimate is invalid"):
            estimate_order_margin(client, "EURUSD", "BUY", 0.1)


class TestCalculatePositionsMargin:
    """Tests for calculate_positions_margin."""

    def test_returns_zero_for_empty_positions(self) -> None:
        """Test empty positions return zero total margin."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame()

        _assert_close(calculate_positions_margin(client), 0.0)

    def test_filters_by_symbols(self) -> None:
        """Test optional symbol filter limits summed positions."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"symbol": "EURUSD", "type": 0, "volume": 0.1},
                {"symbol": "USDJPY", "type": 1, "volume": 0.2},
            ],
        )
        client.symbol_info_tick_as_dict.side_effect = [
            {"ask": 1.1010, "bid": 1.1000},
            {"ask": 110.0, "bid": 109.0},
        ]
        client.order_calc_margin.side_effect = [12.5, 20.0]

        margin = calculate_positions_margin(client, symbols=["EURUSD"])

        _assert_close(margin, 12.5)
        assert client.order_calc_margin.call_count == 1

    def test_sums_mixed_buy_and_sell_positions(self) -> None:
        """Test mixed buy/sell exposure sums each side independently."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"symbol": "EURUSD", "type": 0, "volume": 0.1},
                {"symbol": "EURUSD", "type": 1, "volume": 0.2},
            ],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.side_effect = [12.5, 24.8]

        margin = calculate_positions_margin(client)

        _assert_close(margin, 37.3)

    def test_groups_positions_by_symbol_and_side(self) -> None:
        """Test repeated symbol/side pairs use one margin call with summed volume."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"symbol": "EURUSD", "type": 0, "volume": 0.1},
                {"symbol": "EURUSD", "type": 0, "volume": 0.2},
            ],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = 37.5

        margin = calculate_positions_margin(client)

        _assert_close(margin, 37.5)
        client.order_calc_margin.assert_called_once()
        args = client.order_calc_margin.call_args[0]
        assert args[0] == 10
        assert args[1] == "EURUSD"
        _assert_close(args[2], 0.3)
        _assert_close(args[3], 1.1010)

    def test_sums_multiple_symbols(self) -> None:
        """Test positions across symbols are all included."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"symbol": "EURUSD", "type": 0, "volume": 0.1},
                {"symbol": "GBPUSD", "type": 1, "volume": 0.3},
            ],
        )
        client.symbol_info_tick_as_dict.side_effect = [
            {"ask": 1.1010, "bid": 1.1000},
            {"ask": 1.3010, "bid": 1.3000},
        ]
        client.order_calc_margin.side_effect = [12.5, 30.0]

        margin = calculate_positions_margin(client)

        _assert_close(margin, 42.5)

    def test_propagates_invalid_tick_or_margin_errors(self) -> None:
        """Test invalid tick or margin data raises Mt5TradingError."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"symbol": "EURUSD", "type": 0, "volume": 0.1}],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": None, "bid": 1.1000}

        with pytest.raises(Mt5TradingError, match="Tick price is unavailable"):
            calculate_positions_margin(client)

    def test_skips_rows_with_invalid_symbol_volume_or_type(self) -> None:
        """Test malformed position rows are ignored when summing margin."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"symbol": "", "type": 0, "volume": 0.1},
                {"symbol": "EURUSD", "type": 0, "volume": 0.0},
                {"symbol": "EURUSD", "type": 2, "volume": 0.1},
                {"symbol": "EURUSD", "type": 0, "volume": 0.1},
            ],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = 12.5

        margin = calculate_positions_margin(client)

        _assert_close(margin, 12.5)
        client.order_calc_margin.assert_called_once()

    def test_skips_rows_with_non_finite_volume(self) -> None:
        """Test NaN and infinite position volumes are ignored."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"symbol": "EURUSD", "type": 0, "volume": float("nan")},
                {"symbol": "EURUSD", "type": 0, "volume": float("inf")},
                {"symbol": "EURUSD", "type": 0, "volume": 0.1},
            ],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = 12.5

        margin = calculate_positions_margin(client)

        _assert_close(margin, 12.5)
        client.order_calc_margin.assert_called_once()

    def test_returns_zero_when_all_volumes_are_non_finite(self) -> None:
        """Test all-invalid non-finite volumes return zero without broker calls."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"symbol": "EURUSD", "type": 0, "volume": float("nan")},
                {"symbol": "EURUSD", "type": 1, "volume": float("inf")},
            ],
        )

        _assert_close(calculate_positions_margin(client), 0.0)
        client.order_calc_margin.assert_not_called()
        client.symbol_info_tick_as_dict.assert_not_called()

    def test_returns_zero_when_symbol_filter_matches_nothing(self) -> None:
        """Test filtered symbol lists with no matches return zero."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"symbol": "EURUSD", "type": 0, "volume": 0.1}],
        )

        _assert_close(calculate_positions_margin(client, symbols=["GBPUSD"]), 0.0)
        client.order_calc_margin.assert_not_called()

    def test_returns_zero_for_empty_positions_with_symbol_filter(self) -> None:
        """Test empty positions with a symbol filter return zero."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame()

        _assert_close(calculate_positions_margin(client, symbols=["EURUSD"]), 0.0)

    def test_returns_zero_for_positions_without_symbol_column_with_symbol_filter(
        self,
    ) -> None:
        """Test positions missing a symbol column return zero when filtered."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"type": 0, "volume": 0.1}],
        )

        _assert_close(calculate_positions_margin(client, symbols=["EURUSD"]), 0.0)
        client.order_calc_margin.assert_not_called()


class TestVolumeAndExecution:
    """Tests for order planning and execution helpers."""

    def test_calculate_volume_by_margin_rounds_down_to_step(self) -> None:
        """Test affordable volume respects min, max, and step."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.return_value = 25.0

        _assert_close(
            calculate_volume_by_margin(
                client,
                "EURUSD",
                130.0,
                "BUY",
            ),
            0.5,
        )

    def test_calculate_volume_by_margin_caps_at_max_volume(self) -> None:
        """Test positive volume_max caps raw affordable volume exactly."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 0.3,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.return_value = 10.0

        _assert_close(calculate_volume_by_margin(client, "EURUSD", 100.0, "BUY"), 0.3)

    def test_calculate_volume_by_margin_ignores_zero_max_volume(self) -> None:
        """Test zero volume_max means uncapped volume normalization."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 0.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.return_value = 10.0

        _assert_close(calculate_volume_by_margin(client, "EURUSD", 35.0, "BUY"), 0.3)

    def test_calculate_volume_by_margin_returns_zero_when_unaffordable(self) -> None:
        """Test unaffordable minimum volume returns zero."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.return_value = 25.0

        _assert_close(
            calculate_volume_by_margin(
                client,
                "EURUSD",
                10.0,
                "BUY",
            ),
            0.0,
        )

    def test_calculate_volume_by_margin_never_returns_nonzero_below_volume_min(
        self,
    ) -> None:
        """Test non-zero affordable volume is never below volume_min."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.return_value = 10.0

        volume = calculate_volume_by_margin(client, "EURUSD", 35.0, "BUY")

        assert abs(volume) < 1e-9 or volume >= 0.1

    def test_calculate_volume_by_margin_returns_zero_without_margin(self) -> None:
        """Test non-positive available margin returns zero before MT5 calls."""
        client = _mock_trade_client()

        _assert_close(
            calculate_volume_by_margin(
                client,
                "EURUSD",
                0.0,
                "BUY",
            ),
            0.0,
        )
        client.order_calc_margin.assert_not_called()

    def test_calculate_volume_by_margin_rejects_invalid_constraints(self) -> None:
        """Test invalid symbol volume constraints raise a trading error."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.0,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }

        with pytest.raises(Mt5TradingError):
            calculate_volume_by_margin(client, "EURUSD", 100.0, "BUY")

    def test_calculate_volume_by_margin_rejects_bad_tick(self) -> None:
        """Test unavailable side price raises a trading error."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.0, "bid": None}

        with pytest.raises(Mt5TradingError):
            calculate_volume_by_margin(client, "EURUSD", 100.0, "SELL")

    def test_calculate_margin_and_volume_without_native_helper(self) -> None:
        """Test margin helper uses module volume calculation when needed."""

        class ClientWithoutNative:
            mt5 = SimpleNamespace(ORDER_TYPE_BUY=10, ORDER_TYPE_SELL=11)

            def account_info_as_dict(self) -> dict[str, float]:
                return {"margin_free": 100.0}

            def symbol_info_as_dict(self, *, symbol: str) -> dict[str, float]:
                assert symbol == "EURUSD"
                return {"volume_min": 0.1, "volume_max": 1.0, "volume_step": 0.1}

            def symbol_info_tick_as_dict(self, *, symbol: str) -> dict[str, float]:
                assert symbol == "EURUSD"
                return {"ask": 100.0, "bid": 100.0}

            def order_calc_margin(
                self,
                order_type: int,
                symbol: str,
                volume: float,
                price: float,
            ) -> float:
                assert order_type in {10, 11}
                assert symbol == "EURUSD"
                _assert_close(volume, 0.1)
                _assert_close(price, 100.0)
                return 10.0

        result = calculate_margin_and_volume(
            cast("Mt5TradingClient", ClientWithoutNative()),
            "EURUSD",
            unit_margin_ratio=0.5,
            preserved_margin_ratio=0.0,
        )

        _assert_close(result["buy_volume"], 0.5)
        _assert_close(result["sell_volume"], 0.5)

    def test_calculate_margin_and_volume_zero_ratio_uses_minimum_volume(
        self,
    ) -> None:
        """Test zero unit ratio requests one minimum volume when affordable."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"margin_free": 100.0}
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.side_effect = [10.0, 20.0]

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.0,
            preserved_margin_ratio=0.0,
        )

        _assert_close(result["available_margin"], 100.0)
        _assert_close(result["trade_margin"], 0.0)
        _assert_close(result["buy_volume"], 0.1)
        _assert_close(result["sell_volume"], 0.1)
        client.calculate_volume_by_margin.assert_not_called()

    def test_calculate_margin_and_volume_zero_ratio_rejects_unaffordable_side(
        self,
    ) -> None:
        """Test zero unit ratio returns zero for unaffordable minimum lots."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"margin_free": 15.0}
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.side_effect = [10.0, 20.0]

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.0,
            preserved_margin_ratio=0.0,
        )

        _assert_close(result["buy_volume"], 0.1)
        _assert_close(result["sell_volume"], 0.0)

    def test_calculate_margin_and_volume_zero_ratio_preserves_margin_first(
        self,
    ) -> None:
        """Test preserved margin reduces affordability before min sizing."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"margin_free": 100.0}
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.side_effect = [11.0, 9.0]

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.0,
            preserved_margin_ratio=0.9,
        )

        _assert_close(result["available_margin"], 10.0)
        _assert_close(result["buy_volume"], 0.0)
        _assert_close(result["sell_volume"], 0.1)

    def test_calculate_margin_and_volume_zero_ratio_without_available_margin(
        self,
    ) -> None:
        """Test zero unit ratio returns zero when preserved margin consumes funds."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"margin_free": 100.0}
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.0,
            preserved_margin_ratio=1.0,
        )

        _assert_close(result["buy_volume"], 0.0)
        _assert_close(result["sell_volume"], 0.0)
        client.order_calc_margin.assert_not_called()

    def test_calculate_margin_and_volume_zero_ratio_rejects_invalid_max_volume(
        self,
    ) -> None:
        """Test zero unit ratio still respects max-volume constraints."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"margin_free": 100.0}
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.2,
            "volume_max": 0.1,
            "volume_step": 0.1,
        }

        with pytest.raises(Mt5TradingError, match="Invalid volume constraints"):
            calculate_margin_and_volume(
                client,
                "EURUSD",
                unit_margin_ratio=0.0,
                preserved_margin_ratio=0.0,
            )

    def test_calculate_margin_and_volume_zero_ratio_rejects_bad_tick(self) -> None:
        """Test zero unit ratio validates tick data for minimum sizing."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"margin_free": 100.0}
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": None, "bid": 99.0}

        with pytest.raises(Mt5TradingError, match="Tick price is unavailable"):
            calculate_margin_and_volume(
                client,
                "EURUSD",
                unit_margin_ratio=0.0,
                preserved_margin_ratio=0.0,
            )

    def test_calculate_margin_and_volume_positive_ratio_uses_existing_behavior(
        self,
    ) -> None:
        """Test positive unit ratios keep proportional native sizing."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"margin_free": 100.0}
        client.calculate_volume_by_margin.side_effect = [0.4, 0.3]
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.5,
            preserved_margin_ratio=0.2,
        )

        _assert_close(result["trade_margin"], 40.0)
        _assert_close(result["buy_volume"], 0.4)
        _assert_close(result["sell_volume"], 0.3)
        client.calculate_volume_by_margin.assert_any_call("EURUSD", 40.0, "BUY")
        client.calculate_volume_by_margin.assert_any_call("EURUSD", 40.0, "SELL")

    def test_calculate_margin_and_volume_handles_missing_symbol_snapshot(
        self,
    ) -> None:
        """Test missing symbol metadata falls back to zero volume constraints."""
        client = MagicMock()
        client.account_info_as_dict.return_value = {"margin_free": 1000.0}
        client.calculate_volume_by_margin.side_effect = [0.3, 0.2]
        client.symbol_info_as_dict.side_effect = AttributeError("missing")

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.5,
            preserved_margin_ratio=0.2,
        )

        _assert_close(result["volume_min"], 0.0)
        _assert_close(result["volume_max"], 0.0)
        _assert_close(result["volume_step"], 0.0)

    def test_new_position_margin_ratio_adds_hypothetical_margin(self) -> None:
        """Test hypothetical order margin is added to account margin."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0, "margin": 50.0}
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.return_value = 25.0

        result = calculate_new_position_margin_ratio(
            client,
            symbol="EURUSD",
            new_position_side="BUY",
            new_position_volume=0.1,
        )
        _assert_close(result, 0.075)

    def test_new_position_margin_ratio_without_new_position(self) -> None:
        """Test current margin ratio can be calculated without a new order."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0, "margin": 50.0}

        _assert_close(
            calculate_new_position_margin_ratio(
                client,
                symbol="EURUSD",
            ),
            0.05,
        )
        client.order_calc_margin.assert_not_called()

    def test_new_position_margin_ratio_rejects_invalid_equity(self) -> None:
        """Test non-positive equity raises a trading error."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 0.0, "margin": 50.0}

        with pytest.raises(Mt5TradingError):
            calculate_new_position_margin_ratio(client, symbol="EURUSD")

    def test_new_position_margin_ratio_rejects_bad_tick(self) -> None:
        """Test missing hypothetical order price raises a trading error."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0, "margin": 50.0}
        client.symbol_info_tick_as_dict.return_value = {"ask": None, "bid": 1.0}

        with pytest.raises(Mt5TradingError):
            calculate_new_position_margin_ratio(
                client,
                symbol="EURUSD",
                new_position_side="BUY",
                new_position_volume=0.1,
            )

    def test_place_market_order_dry_run_does_not_send(self) -> None:
        """Test dry-run market orders return a request without sending."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"visible": False}
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
            dry_run=True,
        )

        assert result["status"] == "dry_run"
        assert _request_from_result(result)["type"] == client.mt5.ORDER_TYPE_BUY
        client.order_send.assert_not_called()
        client.symbol_select.assert_not_called()

    def test_place_market_order_supports_limits(self) -> None:
        """Test optional SL/TP values are included in the request."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
            sl=1.0,
            tp=1.4,
            dry_run=True,
        )

        _assert_close(_request_from_result(result)["sl"], 1.0)
        _assert_close(_request_from_result(result)["tp"], 1.4)

    def test_place_market_order_rejects_invalid_filling_mode(self) -> None:
        """Test MT5 filling mode names are validated before getattr."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        with pytest.raises(ValueError, match="Unsupported order_filling mode"):
            place_market_order(
                client,
                symbol="EURUSD",
                volume=0.1,
                order_side="BUY",
                order_filling_mode=cast("Any", "BAD"),
                dry_run=True,
            )

    def test_place_market_order_rejects_invalid_time_mode(self) -> None:
        """Test MT5 time mode names are validated before getattr."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        with pytest.raises(ValueError, match="Unsupported order_time mode"):
            place_market_order(
                client,
                symbol="EURUSD",
                volume=0.1,
                order_side="BUY",
                order_time_mode=cast("Any", "BAD"),
                dry_run=True,
            )

    def test_place_market_order_rejects_missing_mt5_constant(self) -> None:
        """Test missing MT5 constants fail with a controlled trading error."""
        client = _mock_trade_client()
        del client.mt5.ORDER_FILLING_IOC
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        with pytest.raises(Mt5TradingError, match="ORDER_FILLING_IOC"):
            place_market_order(
                client,
                symbol="EURUSD",
                volume=0.1,
                order_side="BUY",
                dry_run=True,
            )

    def test_place_market_order_rejects_invalid_volume(self) -> None:
        """Test non-positive volume raises a trading error."""
        with pytest.raises(Mt5TradingError):
            place_market_order(
                _mock_trade_client(),
                symbol="EURUSD",
                volume=0.0,
                order_side="BUY",
            )

    def test_place_market_order_rejects_bad_tick(self) -> None:
        """Test unavailable market order price raises a trading error."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": None, "bid": 1.1}

        with pytest.raises(Mt5TradingError):
            place_market_order(
                client,
                symbol="EURUSD",
                volume=0.1,
                order_side="BUY",
            )

    def test_place_market_order_rejects_unknown_side(self) -> None:
        """Test unsupported order sides raise ValueError."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        with pytest.raises(ValueError, match="Unsupported order side"):
            place_market_order(
                client,
                symbol="EURUSD",
                volume=0.1,
                order_side=cast("Any", "FLAT"),
            )

    def test_place_market_order_sends_and_normalizes_response(self) -> None:
        """Test live market order responses are normalized."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": 10009, "comment": "done"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="SELL",
        )

        assert result["status"] == "executed"
        assert result["retcode"] == 10009
        client.order_send.assert_called_once()

    def test_place_market_order_marks_failed_retcode(self) -> None:
        """Test order_send responses with failed retcodes are normalized."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": 10013, "comment": "invalid request"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["status"] == "failed"
        assert result["retcode"] == 10013

    def test_place_market_order_marks_failed_numpy_retcode(self) -> None:
        """Test numpy integer retcodes normalize to failed status."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": np_int64(10013), "comment": "invalid request"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["status"] == "failed"
        assert result["retcode"] == 10013

    def test_place_market_order_rejects_bool_retcode(self) -> None:
        """Test bool retcodes are not treated as integer broker codes."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": True, "comment": "weird"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["retcode"] is None
        assert result["status"] == "failed"

    def test_place_market_order_marks_failed_string_retcode(self) -> None:
        """Test digit-string failure retcodes normalize to failed status."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": "10013", "comment": "invalid request"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["retcode"] == 10013
        assert result["status"] == "failed"

    def test_place_market_order_marks_failed_whitespace_string_retcode(self) -> None:
        """Test whitespace-padded digit-string retcodes normalize to failed status."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": " 10013 ", "comment": "invalid request"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["retcode"] == 10013
        assert result["status"] == "failed"

    @pytest.mark.parametrize("retcode", ["+10013", "-10013"])
    def test_place_market_order_marks_signed_string_retcode_as_failed(
        self,
        retcode: str,
    ) -> None:
        """Test signed digit-string failure retcodes normalize to failed status."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": retcode, "comment": "invalid request"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        expected = 10013 if retcode.startswith("+") else -10013
        assert result["retcode"] == expected
        assert result["status"] == "failed"

    def test_place_market_order_marks_missing_retcode_as_failed(self) -> None:
        """Test live responses without retcode are fail-closed."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"comment": "missing retcode"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["retcode"] is None
        assert result["status"] == "failed"

    def test_place_market_order_marks_malformed_retcode_as_failed(self) -> None:
        """Test malformed non-None retcodes are fail-closed."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": "invalid", "comment": "invalid request"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["retcode"] is None
        assert result["status"] == "failed"

    def test_place_market_order_marks_empty_string_retcode_as_failed(self) -> None:
        """Test empty string retcodes are fail-closed."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": "   ", "comment": "invalid request"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["retcode"] is None
        assert result["status"] == "failed"

    def test_place_market_order_marks_object_retcode_as_failed(self) -> None:
        """Test unsupported retcode object types are fail-closed."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": object(), "comment": "invalid request"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["retcode"] is None
        assert result["status"] == "failed"

    def test_close_open_positions_filters_and_dry_runs(self) -> None:
        """Test close helper filters positions and builds opposite orders."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1},
                {"ticket": 2, "symbol": "USDJPY", "type": 1, "volume": 0.2},
            ],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = close_open_positions(client, symbols="EURUSD", dry_run=True)

        assert len(result) == 1
        assert result[0]["order_side"] == "SELL"
        assert _request_from_result(result[0])["position"] == 1

    def test_close_open_positions_filters_by_ticket(self) -> None:
        """Test close helper can filter by ticket."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1},
                {"ticket": 2, "symbol": "USDJPY", "type": 1, "volume": 0.2},
            ],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = close_open_positions(client, tickets=[2], dry_run=True)

        assert len(result) == 1
        assert result[0]["order_side"] == "BUY"

    def test_close_open_positions_sends_position_ticket(self) -> None:
        """Test live close orders include the position before order_send."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 9, "symbol": "EURUSD", "type": 0, "volume": 0.1}],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = SimpleNamespace(retcode=10009, comment="done")

        close_open_positions(client, tickets=[9])

        assert client.order_send.call_args.args[0]["position"] == 9

    def test_update_sltp_filters_and_dry_runs(self) -> None:
        """Test SL/TP updates filter positions and do not send in dry-run mode."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"visible": False}
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": 1.0,
                    "tp": 1.4,
                },
                {
                    "ticket": 2,
                    "symbol": "USDJPY",
                    "type": 1,
                    "volume": 0.2,
                    "sl": 100.0,
                    "tp": 99.0,
                },
            ],
        )

        result = update_sltp_for_open_positions(
            client,
            symbol="EURUSD",
            stop_loss=1.1,
            take_profit=1.3,
            dry_run=True,
        )

        assert len(result) == 1
        _assert_close(_request_from_result(result[0])["sl"], 1.1)
        _assert_close(_request_from_result(result[0])["tp"], 1.3)
        client.order_send.assert_not_called()
        client.symbol_select.assert_not_called()

    def test_update_sltp_selects_hidden_symbol_for_live_send(self) -> None:
        """Test live SL/TP updates ensure hidden symbols are selected first."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"visible": False}
        client.symbol_select.return_value = True
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": 1.0,
                    "tp": 1.4,
                },
            ],
        )
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": 10009, "comment": "updated"}],
        )

        update_sltp_for_open_positions(client, tickets=[1], stop_loss=1.1)

        client.symbol_select.assert_called_once_with("EURUSD", enable=True)
        client.order_send.assert_called_once()

    def test_update_sltp_sends_and_normalizes_response(self) -> None:
        """Test live SL/TP updates send requests and normalize responses."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": 1.0,
                    "tp": 1.4,
                },
            ],
        )
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": 10009, "comment": "updated"}],
        )

        result = update_sltp_for_open_positions(client, tickets=[1])

        assert result[0]["status"] == "executed"
        assert result[0]["retcode"] == 10009
        _assert_close(_request_from_result(result[0])["sl"], 1.0)

    def test_update_sltp_omits_invalid_existing_levels(self) -> None:
        """Test raw broker SL/TP sentinels are not forwarded to order_send."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": 0.0,
                    "tp": float("nan"),
                },
            ],
        )

        result = update_sltp_for_open_positions(client, tickets=[1], dry_run=True)

        request = _request_from_result(result[0])
        assert "sl" not in request
        assert "tp" not in request

    def test_update_sltp_omits_missing_and_non_numeric_existing_levels(self) -> None:
        """Test missing and non-numeric SL/TP levels are not forwarded."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": None,
                    "tp": "unset",
                },
            ],
        )

        result = update_sltp_for_open_positions(client, tickets=[1], dry_run=True)

        request = _request_from_result(result[0])
        assert "sl" not in request
        assert "tp" not in request

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

    def test_place_market_order_selects_hidden_symbol_for_live_send(self) -> None:
        """Test live market orders select hidden symbols before reading ticks."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"visible": False}
        client.symbol_select.return_value = True
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": 10009, "comment": "done"}],
        )
        call_order: list[str] = []

        def _record_select(*_args: object, **_kwargs: object) -> bool:
            call_order.append("symbol_select")
            return True

        def _record_tick(*_args: object, **_kwargs: object) -> dict[str, float]:
            call_order.append("tick")
            return {"ask": 1.2, "bid": 1.1}

        client.symbol_select.side_effect = _record_select
        client.symbol_info_tick_as_dict.side_effect = _record_tick

        place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        client.symbol_select.assert_called_once_with("EURUSD", enable=True)
        client.symbol_info_tick_as_dict.assert_called_once()
        client.order_send.assert_called_once()
        assert call_order == ["symbol_select", "tick"]

    def test_place_market_order_reads_ticks_after_hidden_symbol_selection(self) -> None:
        """Test live orders can read ticks only after hidden symbols are selected."""
        client = _mock_trade_client()
        selected = {"value": False}

        def _symbol_info_side_effect(**_kwargs: object) -> dict[str, bool]:
            return {"visible": selected["value"]}

        def _select_symbol(*_args: object, **_kwargs: object) -> bool:
            selected["value"] = True
            return True

        def _tick_side_effect(**_kwargs: object) -> dict[str, float | None]:
            if not selected["value"]:
                return {"ask": None, "bid": None}
            return {"ask": 1.2, "bid": 1.1}

        client.symbol_info_as_dict.side_effect = _symbol_info_side_effect
        client.symbol_select.side_effect = _select_symbol
        client.symbol_info_tick_as_dict.side_effect = _tick_side_effect
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": 10009, "comment": "done"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["status"] == "executed"
        client.symbol_select.assert_called_once_with("EURUSD", enable=True)
        client.order_send.assert_called_once()

    def test_update_sltp_marks_failed_retcode(self) -> None:
        """Test SL/TP updates normalize failed broker retcodes."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"visible": True}
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": 1.0,
                    "tp": 1.4,
                },
            ],
        )
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": 10013, "comment": "invalid stops"}],
        )

        result = update_sltp_for_open_positions(client, tickets=[1], stop_loss=1.1)

        assert result[0]["status"] == "failed"
        assert result[0]["retcode"] == 10013

    def test_update_sltp_marks_failed_numpy_retcode(self) -> None:
        """Test numpy integer retcodes normalize to failed SL/TP status."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"visible": True}
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": 1.0,
                    "tp": 1.4,
                },
            ],
        )
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": np_int64(10013), "comment": "invalid stops"}],
        )

        result = update_sltp_for_open_positions(client, tickets=[1], stop_loss=1.1)

        assert result[0]["status"] == "failed"
        assert result[0]["retcode"] == 10013

    def test_update_sltp_marks_failed_string_retcode(self) -> None:
        """Test digit-string failure retcodes normalize to failed SL/TP status."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"visible": True}
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": 1.0,
                    "tp": 1.4,
                },
            ],
        )
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": "10013", "comment": "invalid stops"}],
        )

        result = update_sltp_for_open_positions(client, tickets=[1], stop_loss=1.1)

        assert result[0]["retcode"] == 10013
        assert result[0]["status"] == "failed"

    def test_update_sltp_marks_malformed_retcode_as_failed(self) -> None:
        """Test malformed non-None SL/TP retcodes are fail-closed."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"visible": True}
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": 1.0,
                    "tp": 1.4,
                },
            ],
        )
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": "invalid", "comment": "invalid stops"}],
        )

        result = update_sltp_for_open_positions(client, tickets=[1], stop_loss=1.1)

        assert result[0]["retcode"] is None
        assert result[0]["status"] == "failed"

    def test_update_sltp_marks_missing_retcode_as_failed(self) -> None:
        """Test live SL/TP responses without retcode are fail-closed."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"visible": True}
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": 1,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": 1.0,
                    "tp": 1.4,
                },
            ],
        )
        client.order_send.return_value = pd.DataFrame(
            [{"comment": "missing retcode"}],
        )

        result = update_sltp_for_open_positions(client, tickets=[1], stop_loss=1.1)

        assert result[0]["retcode"] is None
        assert result[0]["status"] == "failed"

    def test_trading_typed_dict_exports(self) -> None:
        """Test order-planning TypedDict contracts are importable."""
        margin: MarginVolume = {
            "margin_free": 1.0,
            "available_margin": 1.0,
            "trade_margin": 0.5,
            "buy_volume": 0.1,
            "sell_volume": 0.1,
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        limits: OrderLimits = {
            "entry": 1.0,
            "stop_loss": 0.9,
            "take_profit": 1.1,
        }
        execution: OrderExecutionResult = {
            "status": "dry_run",
            "symbol": "EURUSD",
            "order_side": "BUY",
            "volume": 0.1,
            "retcode": None,
            "comment": None,
            "request": {"action": 20},
            "response": None,
            "dry_run": True,
        }
        _assert_close(margin["buy_volume"], 0.1)
        _assert_close(limits["entry"], 1.0)
        assert execution["status"] == "dry_run"

    def test_shuts_down_when_body_raises(self, mocker: MockerFixture) -> None:
        """Test shutdown is called when the context body raises."""
        mock_client = MagicMock()
        mocker.patch("mt5cli.trading.Mt5TradingClient", return_value=mock_client)

        body_error = "body error"
        with pytest.raises(RuntimeError, match=body_error), mt5_trading_session():
            raise RuntimeError(body_error)

        mock_client.shutdown.assert_called_once()


class TestFetchLatestClosedRatesForTradingClient:
    """Tests for fetch_latest_closed_rates_for_trading_client."""

    def test_fetches_extra_bar_and_drops_forming_row(self) -> None:
        """Test trading-client helper hides the forming bar."""
        client = MagicMock()
        client.fetch_latest_rates_as_df.return_value = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "close": [1.0, 1.1, 1.2],
            },
        )

        result = fetch_latest_closed_rates_for_trading_client(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        client.fetch_latest_rates_as_df.assert_called_once_with("EURUSD", "M1", 3)
        assert list(result["close"]) == [1.0, 1.1]
        assert list(result["time"]) == [1, 2]

    def test_falls_back_to_copy_rates_from_pos_as_df(self) -> None:
        """Test legacy trading clients without fetch helper still work."""
        client = MagicMock(spec=["copy_rates_from_pos_as_df", "mt5"])
        del client.fetch_latest_rates_as_df
        client.copy_rates_from_pos_as_df.return_value = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "close": [1.0, 1.1, 1.2],
            },
        )

        result = fetch_latest_closed_rates_for_trading_client(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        client.copy_rates_from_pos_as_df.assert_called_once_with(
            symbol="EURUSD",
            timeframe=1,
            start_pos=0,
            count=3,
        )
        assert list(result["close"]) == [1.0, 1.1]

    def test_accepts_numeric_epoch_timestamps(self) -> None:
        """Test numeric epoch timestamps are preserved in output."""
        client = MagicMock()
        client.fetch_latest_rates_as_df.return_value = pd.DataFrame(
            {
                "time": [1700000000, 1700000060, 1700000120],
                "close": [1.0, 1.1, 1.2],
            },
        )

        result = fetch_latest_closed_rates_for_trading_client(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert list(result["time"]) == [1700000000, 1700000060]

    def test_accepts_timezone_aware_timestamps_from_index(self) -> None:
        """Test timezone-aware timestamps in the index are exposed as a column."""
        client = MagicMock()
        frame = pd.DataFrame(
            {
                "close": [1.0, 1.1, 1.2],
            },
            index=pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:01:00Z",
                    "2024-01-01T00:02:00Z",
                ],
                utc=True,
            ),
        )
        frame.index.name = "time"
        client.fetch_latest_rates_as_df.return_value = frame

        result = fetch_latest_closed_rates_for_trading_client(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert "time" in result.columns
        assert len(result) == 2
        assert result["close"].tolist() == [1.0, 1.1]

    def test_accepts_unnamed_datetime_index(self) -> None:
        """Test unnamed DatetimeIndex values are exposed as a time column."""
        client = MagicMock()
        frame = pd.DataFrame(
            {"close": [1.0, 1.1, 1.2]},
            index=pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:01:00Z",
                    "2024-01-01T00:02:00Z",
                ],
                utc=True,
            ),
        )
        client.fetch_latest_rates_as_df.return_value = frame

        result = fetch_latest_closed_rates_for_trading_client(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert "time" in result.columns
        assert len(result) == 2

    def test_accepts_named_non_time_index(self) -> None:
        """Test non-time named indexes are left unchanged before validation."""
        client = MagicMock()
        frame = pd.DataFrame(
            {
                "close": [1.0, 1.1, 1.2],
                "bar_id": [1, 2, 3],
            },
        ).set_index("bar_id")
        client.fetch_latest_rates_as_df.return_value = frame

        with pytest.raises(ValueError, match="missing a time column"):
            fetch_latest_closed_rates_for_trading_client(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=2,
            )

    def test_raises_when_trading_client_cannot_fetch_rates(self) -> None:
        """Test missing rate-fetch methods raise Mt5TradingError."""
        client = MagicMock(spec=[])

        with pytest.raises(Mt5TradingError, match="cannot fetch rate data"):
            fetch_latest_closed_rates_for_trading_client(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=1,
            )

    def test_raises_when_time_column_is_missing(self) -> None:
        """Test malformed rate data without time raises ValueError."""
        client = MagicMock()
        client.fetch_latest_rates_as_df.return_value = pd.DataFrame(
            {"close": [1.0, 1.1, 1.2]},
        )

        with pytest.raises(ValueError, match="missing a time column"):
            fetch_latest_closed_rates_for_trading_client(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=2,
            )

    def test_raises_when_no_closed_bars_are_available(self) -> None:
        """Test empty closed-bar results raise an actionable ValueError."""
        client = MagicMock()
        client.fetch_latest_rates_as_df.return_value = pd.DataFrame(
            {"time": [1], "close": [1.0]},
        )

        with pytest.raises(ValueError, match="Rate data is empty"):
            fetch_latest_closed_rates_for_trading_client(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=1,
            )

    def test_raises_when_fetch_returns_none(self) -> None:
        """Test None fetch results raise a malformed rate data error."""
        client = MagicMock()
        client.fetch_latest_rates_as_df.return_value = None

        with pytest.raises(ValueError, match="Malformed rate data"):
            fetch_latest_closed_rates_for_trading_client(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=2,
            )

    def test_raises_when_fetch_returns_non_dataframe(self) -> None:
        """Test non-DataFrame fetch results raise a malformed rate data error."""
        client = MagicMock()
        client.fetch_latest_rates_as_df.return_value = [{"time": 1, "close": 1.0}]

        with pytest.raises(ValueError, match="Malformed rate data"):
            fetch_latest_closed_rates_for_trading_client(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=2,
            )

    def test_rejects_non_positive_count_before_fetching(self) -> None:
        """Test invalid count values fail before calling MT5."""
        client = MagicMock()

        with pytest.raises(ValueError, match="count must be positive"):
            fetch_latest_closed_rates_for_trading_client(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=0,
            )

        client.fetch_latest_rates_as_df.assert_not_called()

    def test_returns_range_index_and_time_column_for_backward_compat(self) -> None:
        """Test original helper returns RangeIndex with a time column."""
        client = MagicMock()
        client.fetch_latest_rates_as_df.return_value = pd.DataFrame(
            {"time": [1700000000, 1700003600, 1700007200], "close": [1.1, 1.2, 1.3]},
        )

        result = fetch_latest_closed_rates_for_trading_client(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert isinstance(result.index, pd.RangeIndex)
        assert "time" in result.columns
        assert len(result) == 2


class TestFetchLatestClosedRatesIndexed:
    """Tests for fetch_latest_closed_rates_indexed."""

    def test_converts_epoch_seconds_to_utc_datetime_index(
        self, mocker: MockerFixture
    ) -> None:
        """Test integer epoch second timestamps become a UTC DatetimeIndex."""
        frame = pd.DataFrame(
            {"time": [1700000000, 1700003600], "close": [1.1, 1.2]},
        )
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=frame,
        )

        result = fetch_latest_closed_rates_indexed(
            MagicMock(),
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "time"
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"
        assert "time" not in result.columns
        assert list(result["close"]) == [1.1, 1.2]

    def test_converts_float_epoch_seconds_to_utc_datetime_index(
        self, mocker: MockerFixture
    ) -> None:
        """Test float64 epoch second timestamps (after concat/NA upcast) become UTC."""
        frame = pd.DataFrame(
            {"time": [1700000000.0, 1700003600.0], "close": [1.1, 1.2]},
        )
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=frame,
        )

        result = fetch_latest_closed_rates_indexed(
            MagicMock(),
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"
        assert result.index[0].year == 2023

    def test_converts_naive_datetime_to_utc_datetime_index(
        self, mocker: MockerFixture
    ) -> None:
        """Test timezone-naive datetime values are localized to UTC."""
        from datetime import datetime  # noqa: PLC0415

        frame = pd.DataFrame(
            {
                "time": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 1, 0)],  # noqa: DTZ001
                "close": [1.1, 1.2],
            },
        )
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=frame,
        )

        result = fetch_latest_closed_rates_indexed(
            MagicMock(),
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"
        assert result.index[0].year == 2024

    def test_converts_aware_datetime_to_utc(self, mocker: MockerFixture) -> None:
        """Test timezone-aware datetime values are converted to UTC."""
        from datetime import datetime, timedelta, timezone  # noqa: PLC0415

        tz_plus5 = timezone(timedelta(hours=5))
        frame = pd.DataFrame(
            {
                "time": [datetime(2024, 1, 1, 5, 0, tzinfo=tz_plus5)],
                "close": [1.1],
            },
        )
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=frame,
        )

        result = fetch_latest_closed_rates_indexed(
            MagicMock(),
            symbol="EURUSD",
            granularity="M1",
            count=1,
        )

        assert isinstance(result.index, pd.DatetimeIndex)
        assert str(result.index.tz) == "UTC"
        assert result.index[0].hour == 0

    def test_raises_on_missing_time_column(self, mocker: MockerFixture) -> None:
        """Test missing time column after the underlying fetch raises ValueError."""
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=pd.DataFrame({"close": [1.1]}),
        )

        with pytest.raises(ValueError, match="missing a time column"):
            fetch_latest_closed_rates_indexed(
                MagicMock(),
                symbol="EURUSD",
                granularity="M1",
                count=1,
            )

    def test_raises_on_unparseable_time_column(self, mocker: MockerFixture) -> None:
        """Test unparseable time data raises a clear ValueError."""
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=pd.DataFrame({"time": ["not-a-date"], "close": [1.1]}),
        )

        with pytest.raises(ValueError, match="invalid or unparseable time data"):
            fetch_latest_closed_rates_indexed(
                MagicMock(),
                symbol="EURUSD",
                granularity="M1",
                count=1,
            )

    def test_drops_time_column_and_sets_index(self, mocker: MockerFixture) -> None:
        """Test the returned DataFrame has the DatetimeIndex and no time column."""
        frame = pd.DataFrame(
            {
                "time": [1700000000, 1700003600],
                "open": [1.0, 1.1],
                "close": [1.1, 1.2],
            },
        )
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=frame,
        )

        result = fetch_latest_closed_rates_indexed(
            MagicMock(),
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert "time" not in result.columns
        assert "open" in result.columns
        assert "close" in result.columns
        assert isinstance(result.index, pd.DatetimeIndex)
