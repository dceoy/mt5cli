"""Tests for trading session helpers and operational utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pdmt5 import Mt5RuntimeError, Mt5TradingClient, Mt5TradingError
from pytest_mock import MockerFixture  # noqa: TC002

from mt5cli.sdk import build_config
from mt5cli.trading import (
    calculate_margin_and_volume,
    calculate_new_position_margin_ratio,
    calculate_spread_ratio,
    calculate_volume_by_margin,
    close_open_positions,
    create_trading_client,
    detect_position_side,
    determine_order_limits,
    get_account_snapshot,
    get_positions_frame,
    get_symbol_snapshot,
    get_tick_snapshot,
    mt5_trading_session,
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
    return client


def _assert_close(actual: object, expected: float) -> None:
    assert abs(float(cast("float", actual)) - expected) < 1e-9


def _request_from_result(result: dict[str, object]) -> dict[str, object]:
    return cast("dict[str, object]", result["request"])


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
        client.symbol_info_as_dict.side_effect = AttributeError("missing")

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
        client.symbol_info_as_dict.side_effect = AttributeError("missing")

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

    def test_shuts_down_when_body_raises(self, mocker: MockerFixture) -> None:
        """Test shutdown is called when the context body raises."""
        mock_client = MagicMock()
        mocker.patch("mt5cli.trading.Mt5TradingClient", return_value=mock_client)

        body_error = "body error"
        with pytest.raises(RuntimeError, match=body_error), mt5_trading_session():
            raise RuntimeError(body_error)

        mock_client.shutdown.assert_called_once()
