"""Tests for trading session helpers and operational utilities."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast, get_args
from unittest.mock import MagicMock

import pandas as pd
import pytest
from numpy import float64 as np_float64
from numpy import int64 as np_int64
from pdmt5 import Mt5RuntimeError, Mt5TradingClient, Mt5TradingError
from pytest_mock import MockerFixture  # noqa: TC002

from mt5cli.exceptions import Mt5OperationError
from mt5cli.sdk import build_config
from mt5cli.trading import (
    MarginVolume,
    OrderExecutionResult,
    OrderLimits,
    OrderSide,
    ProjectionMode,
    calculate_account_projected_margin_ratio,
    calculate_margin_and_volume,
    calculate_new_position_margin_ratio,
    calculate_positions_margin,
    calculate_positions_margin_by_symbol,
    calculate_positions_margin_safe,
    calculate_projected_margin_ratio,
    calculate_spread_ratio,
    calculate_symbol_group_margin_ratio,
    calculate_trailing_stop_updates,
    calculate_volume_by_margin,
    close_open_positions,
    create_trading_client,
    detect_position_side,
    determine_order_limits,
    ensure_symbol_selected,
    estimate_order_margin,
    extract_tick_price,
    fetch_latest_closed_rates_for_trading_client,
    fetch_latest_closed_rates_indexed,
    fetch_recent_history_deals_for_trading_client,
    get_account_snapshot,
    get_positions_frame,
    get_symbol_snapshot,
    get_tick_snapshot,
    mt5_trading_session,
    normalize_order_volume,
    place_market_order,
    update_sltp_for_open_positions,
    update_trailing_stop_loss_for_open_positions,
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


_MISSING_RETCODE: object = (
    object()
)  # sentinel: absent "retcode" key (used in two parametrized retcode tests)


class TestDetectPositionSide:
    """Tests for detect_position_side."""

    @pytest.mark.parametrize(
        ("types", "volumes", "expected"),
        [
            ([], [], None),
            ([0, 0], [0.2, 0.1], "long"),
            ([1, 1], [0.3, 0.1], "short"),
            ([0, 1], [0.3, 0.2], None),
        ],
        ids=["no-positions", "buy-only", "sell-only", "mixed"],
    )
    def test_detect_position_side(
        self,
        types: list[int],
        volumes: list[float],
        expected: str | None,
    ) -> None:
        """detect_position_side reports net long/short/None from open positions."""
        client = MagicMock()
        client.mt5.POSITION_TYPE_BUY = 0
        client.mt5.POSITION_TYPE_SELL = 1
        client.positions_get_as_df.return_value = pd.DataFrame(
            {"type": types, "volume": volumes},
        )

        assert detect_position_side(client, "EURUSD") == expected


class TestCalculateMarginAndVolume:
    """Tests for calculate_margin_and_volume."""

    def test_calculates_margin_budget_and_volumes(self, mocker: MockerFixture) -> None:
        """Test margin budget and buy/sell volumes are derived from ratios."""
        client = MagicMock()
        client.account_info_as_dict.return_value = {"margin_free": 1000.0}
        client.symbol_info_as_dict.side_effect = AttributeError("missing")
        mock_calc_vol = mocker.patch(
            "mt5cli.trading.calculate_volume_by_margin", side_effect=[0.3, 0.2]
        )

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
        mock_calc_vol.assert_any_call(client, "EURUSD", 400.0, "BUY")
        mock_calc_vol.assert_any_call(client, "EURUSD", 400.0, "SELL")

    @pytest.mark.parametrize(
        ("account_dict", "expected_margin_free"),
        [
            ({"margin_free": 0.0}, 0.0),
            ({}, 0.0),
            ({"margin_free": None}, 0.0),
            ({"margin_free": -500.0}, 0.0),
        ],
        ids=["zero", "missing", "none", "negative"],
    )
    def test_zero_missing_or_negative_margin_free(
        self,
        account_dict: dict[str, float | None],
        expected_margin_free: float,
        mocker: MockerFixture,
    ) -> None:
        """Test missing or zero margin_free yields zero trade margin."""
        client = MagicMock()
        client.account_info_as_dict.return_value = account_dict
        client.symbol_info_as_dict.side_effect = AttributeError("missing")
        mock_calc_vol = mocker.patch(
            "mt5cli.trading.calculate_volume_by_margin", return_value=0.0
        )

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.5,
            preserved_margin_ratio=0.2,
        )

        assert result["margin_free"] == expected_margin_free
        _assert_close(result["buy_volume"], 0.0)
        _assert_close(result["sell_volume"], 0.0)
        mock_calc_vol.assert_any_call(client, "EURUSD", 0.0, "BUY")
        mock_calc_vol.assert_any_call(client, "EURUSD", 0.0, "SELL")

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

    @pytest.mark.parametrize(
        ("side", "expected"),
        [
            ("long", {"entry": 100.0, "stop_loss": 98.0, "take_profit": 103.0}),
            ("short", {"entry": 99.0, "stop_loss": 100.98, "take_profit": 96.03}),
        ],
        ids=["long", "short"],
    )
    def test_calculates_protective_levels(
        self,
        side: str,
        expected: dict[str, float | None],
    ) -> None:
        """Test long/short stop loss and take profit are placed below/above entry."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.symbol_info_as_dict.return_value = {}

        result = determine_order_limits(
            client,
            "EURUSD",
            side,
            stop_loss_limit_ratio=0.02,
            take_profit_limit_ratio=0.03,
        )

        assert result == expected

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

    @pytest.mark.parametrize(
        ("return_value", "side_effect"),
        [
            ({"digits": "invalid"}, None),
            (None, AttributeError("missing")),
        ],
    )
    def test_uses_default_digits_when_symbol_info_fails(
        self,
        return_value: dict[str, object] | None,
        side_effect: Exception | None,
    ) -> None:
        """Test order limit rounding falls back when symbol metadata is unavailable."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.234567891, "bid": 1.0}
        if side_effect is not None:
            client.symbol_info_as_dict.side_effect = side_effect
        else:
            client.symbol_info_as_dict.return_value = return_value

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

        with pytest.raises(Mt5OperationError, match="Tick price is unavailable"):
            determine_order_limits(client, "EURUSD", "long")

    def test_accepts_numeric_string_entry(self) -> None:
        """Test numeric string ask/bid values are accepted as entry prices."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {
            "ask": "1.1010",
            "bid": "1.1000",
        }
        client.symbol_info_as_dict.return_value = {"digits": 4}

        result = determine_order_limits(client, "EURUSD", "long")

        _assert_close(result["entry"], 1.1010)

    @pytest.mark.parametrize(
        ("side", "field", "bad_value"),
        [
            ("long", "ask", float("nan")),
            ("long", "ask", float("inf")),
            ("long", "ask", float("-inf")),
            ("long", "ask", 0.0),
            ("long", "ask", -1.0),
            ("long", "ask", True),
            ("long", "ask", False),
            ("long", "ask", "invalid"),
            ("short", "bid", float("nan")),
            ("short", "bid", float("inf")),
            ("short", "bid", float("-inf")),
            ("short", "bid", 0.0),
            ("short", "bid", -1.0),
            ("short", "bid", True),
            ("short", "bid", False),
            ("short", "bid", "invalid"),
        ],
    )
    def test_rejects_invalid_entry_tick_values(
        self,
        side: str,
        field: str,
        bad_value: object,
    ) -> None:
        """Test invalid entry tick values raise Mt5TradingError."""
        tick: dict[str, object] = {"ask": 1.1, "bid": 1.0}
        tick[field] = bad_value
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = tick

        with pytest.raises(Mt5OperationError, match="Tick price is unavailable"):
            determine_order_limits(client, "EURUSD", side)

    @pytest.mark.parametrize(
        ("side", "ask", "bid", "kwarg", "match"),
        [
            ("long", 1.0, 0.99, "stop_loss_limit_ratio", "Stop loss for 'EURUSD'"),
            ("long", 1.0, 0.99, "take_profit_limit_ratio", "Take profit for 'EURUSD'"),
            ("short", 1.01, 1.0, "stop_loss_limit_ratio", "Stop loss for 'EURUSD'"),
            ("short", 1.01, 1.0, "take_profit_limit_ratio", "Take profit for 'EURUSD'"),
        ],
    )
    def test_rejects_protective_level_inside_broker_stop_level(
        self,
        side: str,
        ask: float,
        bid: float,
        kwarg: str,
        match: str,
    ) -> None:
        """Test protective levels inside trade_stops_level raise Mt5TradingError."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": ask, "bid": bid}
        client.symbol_info_as_dict.return_value = {
            "digits": 2,
            "trade_stops_level": 100,
            "point": 0.0001,
        }

        with pytest.raises(Mt5OperationError, match=match):
            determine_order_limits(client, "EURUSD", side, **{kwarg: 0.0001})

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

    @pytest.mark.parametrize(
        ("side", "ask", "bid", "expected_sl", "expected_tp"),
        [
            ("long", 1.0, 0.99, 0.95, 1.05),
            ("short", 1.01, 1.0, 1.05, 0.95),
        ],
    )
    def test_allows_protective_levels_beyond_broker_stop_level(
        self,
        side: str,
        ask: float,
        bid: float,
        expected_sl: float,
        expected_tp: float,
    ) -> None:
        """Test SL/TP beyond trade_stops_level pass validation."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"ask": ask, "bid": bid}
        client.symbol_info_as_dict.return_value = {
            "digits": 2,
            "trade_stops_level": 10,
            "point": 0.0001,
        }

        result = determine_order_limits(
            client,
            "EURUSD",
            side,
            stop_loss_limit_ratio=0.05,
            take_profit_limit_ratio=0.05,
        )

        _assert_close(result["stop_loss"], expected_sl)
        _assert_close(result["take_profit"], expected_tp)

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

        with pytest.raises(Mt5OperationError, match="Failed to select symbol 'EURUSD'"):
            ensure_symbol_selected(client, "EURUSD")

    def test_raises_when_symbol_select_is_unavailable(self) -> None:
        """Test missing symbol_select raises Mt5TradingError."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"visible": False}
        del client.symbol_select

        with pytest.raises(
            Mt5OperationError,
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
            "mt5cli.trading.Mt5DataClient",
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
        """Test keyword configuration is forwarded to Mt5DataClient."""
        mock_client = MagicMock()
        trading_client = mocker.patch(
            "mt5cli.trading.Mt5DataClient",
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
            "mt5cli.trading.Mt5DataClient",
            return_value=MagicMock(),
        )

        create_trading_client(login=" ")

        config = trading_client.call_args.kwargs["config"]
        assert config.login is None

    def test_shutdown_on_initialization_failure(self, mocker: MockerFixture) -> None:
        """Test failed initialization shuts the client down."""
        mock_client = MagicMock()
        mock_client.initialize_and_login_mt5.side_effect = Mt5RuntimeError("boom")
        mocker.patch("mt5cli.trading.Mt5DataClient", return_value=mock_client)

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

    def test_calculate_spread_ratio_accepts_numeric_string_tick(self) -> None:
        """Test numeric string bid/ask values are accepted."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"bid": "99.0", "ask": "101.0"}

        _assert_close(calculate_spread_ratio(client, "EURUSD"), 0.02)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("bid", None),
            ("bid", float("nan")),
            ("bid", float("inf")),
            ("bid", float("-inf")),
            ("bid", 0.0),
            ("bid", -1.0),
            ("bid", True),
            ("bid", False),
            ("bid", "invalid"),
            ("ask", None),
            ("ask", float("nan")),
            ("ask", float("inf")),
            ("ask", float("-inf")),
            ("ask", 0.0),
            ("ask", -1.0),
            ("ask", True),
            ("ask", False),
            ("ask", "invalid"),
        ],
    )
    def test_calculate_spread_ratio_rejects_invalid_tick_values(
        self,
        field: str,
        bad_value: object,
    ) -> None:
        """Test invalid bid/ask values raise Mt5TradingError."""
        tick: dict[str, object] = {"bid": 100.0, "ask": 100.0}
        tick[field] = bad_value
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = tick

        with pytest.raises(Mt5OperationError, match="Tick bid/ask is unavailable"):
            calculate_spread_ratio(client, "EURUSD")


class TestNormalizeOrderVolume:
    """Tests for normalize_order_volume."""

    @pytest.mark.parametrize(
        ("volume", "volume_min", "volume_max", "volume_step", "expected"),
        [
            (0.1, 0.1, 1.0, 0.1, 0.1),
            (0.25, 0.1, 1.0, 0.1, 0.2),
            (0.9, 0.1, 0.5, 0.1, 0.5),
            (0.05, 0.1, 1.0, 0.1, 0.0),
            (1.0, 0.0, 1.0, 0.1, 0.0),
            (1.0, 0.1, 1.0, 0.0, 0.0),
            (2.5, 0.1, 0.0, 0.1, 2.5),
            (0.5, 0.1, 0.34, 0.12, 0.34),
        ],
        ids=[
            "exact-minimum",
            "floor-to-step",
            "clamp-to-max",
            "below-minimum",
            "invalid-volume-min",
            "invalid-volume-step",
            "non-positive-max-no-cap",
            "max-reapplied-after-step",
        ],
    )
    def test_normalize_order_volume_deterministic(
        self,
        volume: float,
        volume_min: float,
        volume_max: float,
        volume_step: float,
        expected: float,
    ) -> None:
        """normalize_order_volume floors, clamps, and validates constraints."""
        _assert_close(
            normalize_order_volume(
                volume,
                volume_min=volume_min,
                volume_max=volume_max,
                volume_step=volume_step,
            ),
            expected,
        )

    @pytest.mark.parametrize("volume", [float("nan"), float("inf")], ids=["nan", "inf"])
    def test_returns_zero_for_non_finite_volume(self, volume: float) -> None:
        """Test NaN or infinite requested volume returns zero."""
        _assert_close(
            normalize_order_volume(
                volume, volume_min=0.1, volume_max=1.0, volume_step=0.1
            ),
            0.0,
        )

    @pytest.mark.parametrize(
        ("volume_min", "volume_step"),
        [(float("nan"), 0.1), (0.1, float("inf"))],
        ids=["nan-min", "inf-step"],
    )
    def test_returns_zero_for_non_finite_constraints(
        self, volume_min: float, volume_step: float
    ) -> None:
        """Test NaN or infinite volume_min/volume_step returns zero."""
        _assert_close(
            normalize_order_volume(
                1.0, volume_min=volume_min, volume_max=1.0, volume_step=volume_step
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

    @pytest.mark.parametrize(
        ("side", "order_type", "expected_price", "expected_margin"),
        [
            ("BUY", 10, 1.1010, 12.5),
            ("SELL", 11, 1.1000, 12.4),
            ("long", 10, 1.1010, 12.5),
            ("short", 11, 1.1000, 12.4),
        ],
        ids=["buy", "sell", "long", "short"],
    )
    def test_estimates_margin_for_side(
        self,
        side: str,
        order_type: int,
        expected_price: float,
        expected_margin: float,
    ) -> None:
        """Test estimate_order_margin uses ask for buy/long and bid for sell/short."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = expected_margin

        margin = estimate_order_margin(client, "EURUSD", side, 0.1)

        _assert_close(margin, expected_margin)
        client.order_calc_margin.assert_called_once_with(
            order_type,
            "EURUSD",
            0.1,
            expected_price,
        )

    def test_rejects_invalid_side(self) -> None:
        """Test unsupported order side raises ValueError."""
        client = _mock_trade_client()

        with pytest.raises(ValueError, match="Unsupported order side"):
            estimate_order_margin(client, "EURUSD", "HOLD", 0.1)

    def test_rejects_non_positive_volume(self) -> None:
        """Test non-positive volume raises Mt5TradingError."""
        client = _mock_trade_client()

        with pytest.raises(Mt5OperationError, match="positive finite number"):
            estimate_order_margin(client, "EURUSD", "BUY", 0.0)

    @pytest.mark.parametrize("volume", [float("nan"), float("inf")], ids=["nan", "inf"])
    def test_rejects_non_finite_volume(self, volume: float) -> None:
        """Test NaN or infinite volume raises Mt5TradingError without broker calls."""
        client = _mock_trade_client()

        with pytest.raises(Mt5OperationError, match="positive finite number"):
            estimate_order_margin(client, "EURUSD", "BUY", volume)

        client.symbol_info_tick_as_dict.assert_not_called()
        client.order_calc_margin.assert_not_called()

    @pytest.mark.parametrize(
        "ask",
        [None, 0.0, float("inf")],
        ids=["missing", "non-positive", "non-finite"],
    )
    def test_rejects_invalid_tick_price(
        self,
        ask: float | None,
    ) -> None:
        """Test invalid ask prices raise Mt5OperationError."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": ask, "bid": 1.1000}

        with pytest.raises(Mt5OperationError, match="Tick price is unavailable"):
            estimate_order_margin(client, "EURUSD", "BUY", 0.1)

    @pytest.mark.parametrize(
        "margin_value",
        [0.0, float("inf"), None, "invalid"],
        ids=["zero", "inf", "none", "string"],
    )
    def test_rejects_invalid_margin_result(self, margin_value: object) -> None:
        """Test invalid margin estimates raise Mt5TradingError."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.1010, "bid": 1.1000}
        client.order_calc_margin.return_value = margin_value

        with pytest.raises(Mt5OperationError, match="Margin estimate is invalid"):
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

        with pytest.raises(Mt5OperationError, match="Tick price is unavailable"):
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

    @pytest.mark.parametrize(
        ("positions_records", "symbols_filter"),
        [
            ([{"symbol": "EURUSD", "type": 0, "volume": 0.1}], ["GBPUSD"]),
            (None, ["EURUSD"]),
            ([{"type": 0, "volume": 0.1}], ["EURUSD"]),
        ],
        ids=["symbol_mismatch", "empty_df", "no_symbol_column"],
    )
    def test_returns_zero_with_symbol_filter_and_no_match(
        self,
        positions_records: list[dict[str, object]] | None,
        symbols_filter: list[str],
    ) -> None:
        """Test symbol filter with no matching positions returns zero margin."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = (
            pd.DataFrame(positions_records)
            if positions_records is not None
            else pd.DataFrame()
        )

        _assert_close(calculate_positions_margin(client, symbols=symbols_filter), 0.0)
        client.order_calc_margin.assert_not_called()


class TestVolumeAndExecution:
    """Tests for order planning and execution helpers."""

    @pytest.mark.parametrize(
        ("volume_max", "margin_per_lot", "budget", "expected_volume"),
        [
            (1.0, 25.0, 130.0, 0.5),
            (0.3, 10.0, 100.0, 0.3),
            (0.0, 10.0, 35.0, 0.3),
            (1.0, 25.0, 10.0, 0.0),
        ],
        ids=[
            "rounds-down-to-step",
            "caps-at-max-volume",
            "ignores-zero-max-volume",
            "returns-zero-when-unaffordable",
        ],
    )
    def test_calculate_volume_by_margin_boundary(
        self,
        volume_max: float,
        margin_per_lot: float,
        budget: float,
        expected_volume: float,
    ) -> None:
        """Test volume caps, step rounding, and affordability under min/max/step."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": volume_max,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.return_value = margin_per_lot

        _assert_close(
            calculate_volume_by_margin(client, "EURUSD", budget, "BUY"),
            expected_volume,
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

        with pytest.raises(Mt5OperationError):
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

        with pytest.raises(Mt5OperationError):
            calculate_volume_by_margin(client, "EURUSD", 100.0, "SELL")

    def test_calculate_volume_by_margin_steps_down_when_margin_exceeds_budget(
        self,
    ) -> None:
        """Tiered margin: binary search returns the largest affordable step."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        # min_margin (0.1): 25.0 -> hi=4.
        # Search: mid=2 (0.3)->75<=130, mid=3 (0.4)->100<=130, mid=4 (0.5)->150>130.
        client.order_calc_margin.side_effect = [25.0, 75.0, 100.0, 150.0]

        result = calculate_volume_by_margin(client, "EURUSD", 130.0, "BUY")

        _assert_close(result, 0.4)

    def test_calculate_volume_by_margin_returns_zero_when_all_steps_unaffordable(
        self,
    ) -> None:
        """If all binary-search probes exceed budget, returns zero."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 0.5,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        # min_margin (0.1): 10.0 -> hi=4.
        # Search: mid=2 (0.3)->150>130->hi=1, mid=0 (0.1)->150>130->hi=-1.
        client.order_calc_margin.side_effect = [10.0, 150.0, 150.0]

        result = calculate_volume_by_margin(client, "EURUSD", 130.0, "BUY")

        _assert_close(result, 0.0)

    def test_calculate_volume_by_margin_binary_search_is_bounded(self) -> None:
        """Binary search finds the largest affordable volume in O(log n) MT5 calls."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.01,
            "volume_max": 1000.0,
            "volume_step": 0.01,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        # Steps 0-50000 cost 0.001 (affordable); steps 50001+ cost 2000.0 (not).
        # Total range: 99999 steps. Linear scan: ~50000 calls; binary search: ~17.
        affordable_step = 50000

        def _margin(_ot: int, _sym: str, volume: float, _px: float) -> float:
            step = round((volume - 0.01) / 0.01)
            return 0.001 if step <= affordable_step else 2000.0

        client.order_calc_margin.side_effect = _margin
        result = calculate_volume_by_margin(client, "EURUSD", 200.0, "BUY")

        _assert_close(result, 500.01)  # 0.01 + 50000 * 0.01
        assert client.order_calc_margin.call_count <= 25

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
                assert 0.1 <= volume <= 1.0
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

    @pytest.mark.parametrize(
        (
            "margin_free",
            "preserved_margin_ratio",
            "order_calc_margins",
            "expected_available_margin",
            "expected_buy_volume",
            "expected_sell_volume",
        ),
        [
            (100.0, 0.0, [10.0, 20.0], 100.0, 0.1, 0.1),
            (15.0, 0.0, [10.0, 20.0], 15.0, 0.1, 0.0),
            (100.0, 0.9, [11.0, 9.0], 10.0, 0.0, 0.1),
        ],
        ids=[
            "uses-minimum-volume",
            "rejects-unaffordable-side",
            "preserves-margin-first",
        ],
    )
    def test_calculate_margin_and_volume_zero_ratio_sizes_minimum_lots(
        self,
        margin_free: float,
        preserved_margin_ratio: float,
        order_calc_margins: list[float],
        expected_available_margin: float,
        expected_buy_volume: float,
        expected_sell_volume: float,
    ) -> None:
        """Test zero unit ratio sizes minimum lots against available margin."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"margin_free": margin_free}
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.side_effect = order_calc_margins

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.0,
            preserved_margin_ratio=preserved_margin_ratio,
        )

        _assert_close(result["available_margin"], expected_available_margin)
        _assert_close(result["buy_volume"], expected_buy_volume)
        _assert_close(result["sell_volume"], expected_sell_volume)
        client.calculate_volume_by_margin.assert_not_called()

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

        with pytest.raises(Mt5OperationError, match="Invalid volume constraints"):
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

        with pytest.raises(Mt5OperationError, match="Tick price is unavailable"):
            calculate_margin_and_volume(
                client,
                "EURUSD",
                unit_margin_ratio=0.0,
                preserved_margin_ratio=0.0,
            )

    def test_calculate_margin_and_volume_positive_ratio_avoids_oversized_native_volume(
        self,
    ) -> None:
        """Regression: module path is used even when native helper is present.

        The pdmt5 linear estimate (floor(40/10)*0.1 = 0.4) would overstate
        affordable volume because order_calc_margin returns 45.0 for 0.4 lots,
        exceeding the 40.0 budget.  The module's verified binary search returns
        0.3 (margin=30.0), which is safe.
        """
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"margin_free": 100.0}
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": 1.0,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.calculate_volume_by_margin.return_value = 0.4  # pdmt5 linear oversized

        def _mock_calc_margin(
            _order_type: int, _symbol: str, volume: float, _price: float
        ) -> float:
            if volume <= 0.3:
                return round(volume * 100.0, 10)  # 0.1→10, 0.2→20, 0.3→30
            return round(volume * 112.5, 10)  # 0.4→45 — exceeds 40.0 budget

        client.order_calc_margin.side_effect = _mock_calc_margin

        result = calculate_margin_and_volume(
            client,
            "EURUSD",
            unit_margin_ratio=0.5,
            preserved_margin_ratio=0.2,
        )

        _assert_close(result["trade_margin"], 40.0)
        _assert_close(result["buy_volume"], 0.3)
        _assert_close(result["sell_volume"], 0.3)
        client.calculate_volume_by_margin.assert_not_called()

    def test_calculate_margin_and_volume_positive_ratio_raises_on_missing_symbol_info(
        self,
    ) -> None:
        """Test missing symbol info propagates an error when ratio is positive."""
        client = MagicMock()
        client.account_info_as_dict.return_value = {"margin_free": 1000.0}
        client.symbol_info_as_dict.side_effect = AttributeError("missing")

        with pytest.raises(AttributeError, match="missing"):
            calculate_margin_and_volume(
                client,
                "EURUSD",
                unit_margin_ratio=0.5,
                preserved_margin_ratio=0.2,
            )

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

        with pytest.raises(Mt5OperationError):
            calculate_new_position_margin_ratio(client, symbol="EURUSD")

    def test_new_position_margin_ratio_rejects_bad_tick(self) -> None:
        """Test missing hypothetical order price raises a trading error."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0, "margin": 50.0}
        client.symbol_info_tick_as_dict.return_value = {"ask": None, "bid": 1.0}

        with pytest.raises(Mt5OperationError):
            calculate_new_position_margin_ratio(
                client,
                symbol="EURUSD",
                new_position_side="BUY",
                new_position_volume=0.1,
            )

    def test_projected_margin_ratio_empty_positions(self) -> None:
        """Test no current or projected exposure returns zero ratio."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        client.positions_get_as_df.return_value = pd.DataFrame()

        _assert_close(calculate_projected_margin_ratio(client, symbol="EURUSD"), 0.0)

    def test_projected_margin_ratio_current_exposure(self) -> None:
        """Test current position margin is divided by account equity."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"symbol": "EURUSD", "type": 0, "volume": 0.2}],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.101, "bid": 1.1}
        client.order_calc_margin.return_value = 50.0

        _assert_close(calculate_projected_margin_ratio(client, symbol="EURUSD"), 0.05)

    @pytest.mark.parametrize(
        ("side", "order_type", "price", "margin_return", "expected_ratio"),
        [
            ("BUY", 10, 1.101, 25.0, 0.025),
            ("SELL", 11, 1.1, 24.0, 0.024),
        ],
        ids=["buy", "sell"],
    )
    def test_projected_margin_ratio_adds_side_exposure(
        self,
        side: OrderSide,
        order_type: int,
        price: float,
        margin_return: float,
        expected_ratio: float,
    ) -> None:
        """Test projected buy/sell margin uses ask/bid pricing and adds to exposure."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        client.positions_get_as_df.return_value = pd.DataFrame()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.101, "bid": 1.1}
        client.order_calc_margin.return_value = margin_return

        result = calculate_projected_margin_ratio(
            client,
            symbol="EURUSD",
            new_position_side=side,
            new_position_volume=0.1,
        )

        _assert_close(result, expected_ratio)
        client.order_calc_margin.assert_called_once_with(
            order_type, "EURUSD", 0.1, price
        )

    @pytest.mark.parametrize(
        ("account", "kwargs", "candidate_margin", "expected_ratio"),
        [
            ({"equity": 10_000.0, "margin": 4500.0}, {}, None, 0.45),
            (
                {"equity": 10_000.0, "margin": 4500.0},
                {
                    "symbol": "EURUSD",
                    "new_position_side": "BUY",
                    "new_position_volume": 0.1,
                },
                1000.0,
                0.55,
            ),
            ({"equity": 10_000.0, "margin": 55.0}, {}, None, 0.0055),
        ],
    )
    def test_account_projected_margin_ratio_uses_account_margin_baseline(
        self,
        account: dict[str, object],
        kwargs: dict[str, object],
        candidate_margin: float | None,
        expected_ratio: float,
        mocker: MockerFixture,
    ) -> None:
        """Test account-wide exposure uses snapshot margin plus optional candidate."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = account
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"symbol": "GBPUSD", "type": 0, "volume": 2.0}],
        )
        mock_margin = mocker.patch(
            "mt5cli.trading.estimate_order_margin",
            return_value=candidate_margin,
        )

        result = calculate_account_projected_margin_ratio(client, **cast("Any", kwargs))

        _assert_close(result, expected_ratio)
        if candidate_margin is None:
            mock_margin.assert_not_called()
        else:
            mock_margin.assert_called_once_with(client, "EURUSD", "BUY", 0.1)
        client.positions_get_as_df.assert_not_called()

    @pytest.mark.parametrize(
        ("kwargs", "expected_ratio"),
        [
            ({"new_position_side": "BUY", "new_position_volume": 0.1}, 0.45),
            ({"symbol": "EURUSD", "new_position_volume": 0.1}, 0.45),
            ({"symbol": "EURUSD", "new_position_side": "BUY"}, 0.45),
            (
                {
                    "symbol": "EURUSD",
                    "new_position_side": "BUY",
                    "new_position_volume": -0.1,
                },
                0.45,
            ),
        ],
    )
    def test_account_projected_margin_ratio_skips_incomplete_candidate(
        self,
        kwargs: dict[str, object],
        expected_ratio: float,
        mocker: MockerFixture,
    ) -> None:
        """Test candidate margin is added only when symbol, side, and volume exist."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {
            "equity": 10_000.0,
            "margin": 4500.0,
        }
        mock_margin = mocker.patch("mt5cli.trading.estimate_order_margin")

        result = calculate_account_projected_margin_ratio(client, **cast("Any", kwargs))

        _assert_close(result, expected_ratio)
        mock_margin.assert_not_called()

    @pytest.mark.parametrize(
        ("account", "match"),
        [
            ({"margin": 4500.0}, "Account equity"),
            ({"equity": None, "margin": 4500.0}, "Account equity"),
            ({"equity": "10000", "margin": 4500.0}, "Account equity"),
            ({"equity": True, "margin": 4500.0}, "Account equity"),
            ({"equity": float("nan"), "margin": 4500.0}, "Account equity"),
            ({"equity": float("inf"), "margin": 4500.0}, "Account equity"),
            ({"equity": 0.0, "margin": 4500.0}, "Account equity"),
            ({"equity": -1.0, "margin": 4500.0}, "Account equity"),
            ({"equity": 10_000.0}, "Account margin"),
            ({"equity": 10_000.0, "margin": None}, "Account margin"),
            ({"equity": 10_000.0, "margin": "4500"}, "Account margin"),
            ({"equity": 10_000.0, "margin": True}, "Account margin"),
            ({"equity": 10_000.0, "margin": False}, "Account margin"),
            ({"equity": 10_000.0, "margin": float("nan")}, "Account margin"),
            ({"equity": 10_000.0, "margin": float("inf")}, "Account margin"),
            ({"equity": 10_000.0, "margin": -1.0}, "Account margin"),
        ],
    )
    def test_account_projected_margin_ratio_rejects_invalid_snapshot_fields(
        self,
        account: dict[str, object],
        match: str,
    ) -> None:
        """Test invalid account equity and margin fields fail closed."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = account

        with pytest.raises(Mt5OperationError, match=match):
            calculate_account_projected_margin_ratio(client)

    def test_account_projected_margin_ratio_propagates_candidate_margin_error(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test candidate margin errors are not suppressed."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {
            "equity": 10_000.0,
            "margin": 4500.0,
        }
        mocker.patch(
            "mt5cli.trading.estimate_order_margin",
            side_effect=Mt5OperationError("bad tick"),
        )

        with pytest.raises(Mt5OperationError, match="bad tick"):
            calculate_account_projected_margin_ratio(
                client,
                symbol="EURUSD",
                new_position_side="BUY",
                new_position_volume=0.1,
            )

    def test_symbol_group_margin_ratio_sums_group_exposure(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test symbol-group exposure sums current per-symbol margins."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin_by_symbol",
            return_value={"EURUSD": 25.0, "GBPUSD": 35.0},
        )

        result = calculate_symbol_group_margin_ratio(
            client,
            symbols=["EURUSD", "GBPUSD"],
        )

        _assert_close(result, 0.06)

    def test_symbol_group_margin_ratio_adds_projected_group_exposure(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test projected order margin is added when the symbol is in the group."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.101, "bid": 1.1}
        client.order_calc_margin.return_value = 15.0
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin_by_symbol",
            return_value={"EURUSD": 25.0},
        )

        result = calculate_symbol_group_margin_ratio(
            client,
            symbols=["EURUSD"],
            new_symbol="EURUSD",
            new_position_side="BUY",
            new_position_volume=0.1,
        )

        _assert_close(result, 0.04)

    def test_symbol_group_margin_ratio_suppresses_per_symbol_failures(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test suppressible per-symbol failures are skipped by the safe map."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin",
            side_effect=[Mt5TradingError("bad tick"), 30.0],
        )

        result = calculate_symbol_group_margin_ratio(
            client,
            symbols=["EURUSD", "GBPUSD"],
            suppress_errors=True,
        )

        _assert_close(result, 0.03)

    def test_symbol_group_margin_ratio_rejects_invalid_equity(self) -> None:
        """Test invalid equity fails closed for exposure helpers."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 0.0}

        with pytest.raises(Mt5OperationError, match="Account equity"):
            calculate_symbol_group_margin_ratio(client, symbols=["EURUSD"])

    def test_projected_margin_ratio_rejects_nonnumeric_equity(self) -> None:
        """Test nonnumeric equity fails closed for exposure helpers."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": "invalid"}

        with pytest.raises(Mt5OperationError, match="Account equity"):
            calculate_projected_margin_ratio(client, symbol="EURUSD")

    def test_symbol_group_margin_ratio_suppresses_projected_failure(
        self,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test projected margin failures can be skipped for safe group reads."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin_by_symbol",
            return_value={},
        )
        mocker.patch(
            "mt5cli.trading.estimate_order_margin",
            side_effect=Mt5OperationError("bad tick"),
        )

        with caplog.at_level(logging.WARNING, logger="mt5cli.trading"):
            result = calculate_symbol_group_margin_ratio(
                client,
                symbols=["EURUSD"],
                new_symbol="EURUSD",
                new_position_side="BUY",
                new_position_volume=0.1,
                suppress_errors=True,
            )

        _assert_close(result, 0.0)
        assert "Skipping projected margin" in caplog.text

    def test_symbol_group_margin_ratio_reraises_projected_failure(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test projected margin failures raise when suppression is disabled."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin_by_symbol",
            return_value={},
        )
        mocker.patch(
            "mt5cli.trading.estimate_order_margin",
            side_effect=Mt5OperationError("bad tick"),
        )

        with pytest.raises(Mt5OperationError, match="bad tick"):
            calculate_symbol_group_margin_ratio(
                client,
                symbols=["EURUSD"],
                new_symbol="EURUSD",
                new_position_side="BUY",
                new_position_volume=0.1,
                suppress_errors=False,
            )

    def test_symbol_group_margin_ratio_replace_symbol_subtracts_and_adds(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test replace_symbol mode subtracts current exposure and adds candidate."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.101, "bid": 1.1}
        client.order_calc_margin.return_value = 15.0
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin_by_symbol",
            return_value={"EURUSD": 25.0},
        )

        result = calculate_symbol_group_margin_ratio(
            client,
            symbols=["EURUSD"],
            new_symbol="EURUSD",
            new_position_side="BUY",
            new_position_volume=0.1,
            projection_mode="replace_symbol",
        )

        # margin = 25.0 - 25.0 (replaced) + 15.0 (candidate) = 15.0
        _assert_close(result, 0.015)

    def test_symbol_group_margin_ratio_replace_no_existing_adds_candidate(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test replace_symbol with no existing current margin still adds candidate."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.101, "bid": 1.1}
        client.order_calc_margin.return_value = 20.0
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin_by_symbol",
            return_value={"EURUSD": 0.0},
        )

        result = calculate_symbol_group_margin_ratio(
            client,
            symbols=["EURUSD"],
            new_symbol="EURUSD",
            new_position_side="BUY",
            new_position_volume=0.2,
            projection_mode="replace_symbol",
        )

        # margin = 0.0 - 0.0 + 20.0 = 20.0
        _assert_close(result, 0.02)

    def test_symbol_group_margin_ratio_replace_symbol_outside_group_unchanged(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test candidate symbol outside the group does not affect margin."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin_by_symbol",
            return_value={"EURUSD": 30.0},
        )

        result = calculate_symbol_group_margin_ratio(
            client,
            symbols=["EURUSD"],
            new_symbol="GBPUSD",
            new_position_side="BUY",
            new_position_volume=0.1,
            projection_mode="replace_symbol",
        )

        # GBPUSD is not in the group; no candidate margin applied
        _assert_close(result, 0.03)

    def test_symbol_group_margin_ratio_replace_no_candidate_returns_current(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test replace_symbol with no candidate side/volume returns current margin."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin_by_symbol",
            return_value={"EURUSD": 40.0},
        )

        result = calculate_symbol_group_margin_ratio(
            client,
            symbols=["EURUSD"],
            new_symbol="EURUSD",
            projection_mode="replace_symbol",
        )

        _assert_close(result, 0.04)

    def test_symbol_group_margin_ratio_replace_suppresses_candidate_failure(
        self,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test replace mode skips both subtract and add when estimation fails."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin_by_symbol",
            return_value={"EURUSD": 25.0},
        )
        mocker.patch(
            "mt5cli.trading.estimate_order_margin",
            side_effect=Mt5OperationError("bad tick"),
        )

        with caplog.at_level(logging.WARNING, logger="mt5cli.trading"):
            result = calculate_symbol_group_margin_ratio(
                client,
                symbols=["EURUSD"],
                new_symbol="EURUSD",
                new_position_side="BUY",
                new_position_volume=0.1,
                projection_mode="replace_symbol",
                suppress_errors=True,
            )

        # When candidate fails, neither subtraction nor addition is applied
        _assert_close(result, 0.025)
        assert "Skipping projected margin" in caplog.text

    def test_symbol_group_margin_ratio_replace_reraises_candidate_failure(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test replace mode re-raises candidate failure when suppress_errors=False."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0}
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin_by_symbol",
            return_value={"EURUSD": 25.0},
        )
        mocker.patch(
            "mt5cli.trading.estimate_order_margin",
            side_effect=Mt5OperationError("bad tick"),
        )

        with pytest.raises(Mt5OperationError, match="bad tick"):
            calculate_symbol_group_margin_ratio(
                client,
                symbols=["EURUSD"],
                new_symbol="EURUSD",
                new_position_side="BUY",
                new_position_volume=0.1,
                projection_mode="replace_symbol",
                suppress_errors=False,
            )

    def test_projection_mode_type_alias_is_importable(self) -> None:
        """Test ProjectionMode type alias is importable and has expected values."""
        assert ProjectionMode is not None
        args = get_args(ProjectionMode)
        assert "add" in args
        assert "replace_symbol" in args

    def test_invalid_projection_mode_raises_value_error(self) -> None:
        """Test that an unsupported projection_mode raises ValueError."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 10000.0}
        client.positions_get_as_df.return_value = pd.DataFrame(
            columns=["symbol", "type", "volume"]
        )
        with pytest.raises(ValueError, match="Unsupported projection mode"):
            calculate_symbol_group_margin_ratio(
                client,
                symbols=["EURUSD"],
                projection_mode="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_projection_mode_message_includes_value_and_accepted(
        self,
    ) -> None:
        """Test ValueError message contains the bad value and accepted modes."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 10000.0}
        client.positions_get_as_df.return_value = pd.DataFrame(
            columns=["symbol", "type", "volume"]
        )
        with pytest.raises(ValueError, match="'unknown'") as exc_info:
            calculate_symbol_group_margin_ratio(
                client,
                symbols=["EURUSD"],
                projection_mode="unknown",  # type: ignore[arg-type]
            )
        msg = str(exc_info.value)
        assert "add" in msg
        assert "replace_symbol" in msg

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

    @pytest.mark.parametrize(
        ("mode_kwarg", "match"),
        [
            ("order_filling_mode", "Unsupported order_filling mode"),
            ("order_time_mode", "Unsupported order_time mode"),
        ],
        ids=["filling-mode", "time-mode"],
    )
    def test_place_market_order_rejects_invalid_mode(
        self,
        mode_kwarg: str,
        match: str,
    ) -> None:
        """Test MT5 filling and time mode names are validated before getattr."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        with pytest.raises(ValueError, match=match):
            place_market_order(
                client,
                symbol="EURUSD",
                volume=0.1,
                order_side="BUY",
                **{mode_kwarg: cast("Any", "BAD")},
                dry_run=True,
            )

    def test_place_market_order_rejects_missing_mt5_constant(self) -> None:
        """Test missing MT5 constants fail with a controlled trading error."""
        client = _mock_trade_client()
        del client.mt5.ORDER_FILLING_IOC
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        with pytest.raises(Mt5OperationError, match="ORDER_FILLING_IOC"):
            place_market_order(
                client,
                symbol="EURUSD",
                volume=0.1,
                order_side="BUY",
                dry_run=True,
            )

    def test_place_market_order_rejects_invalid_volume(self) -> None:
        """Test non-positive volume raises a trading error."""
        with pytest.raises(Mt5OperationError):
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

        with pytest.raises(Mt5OperationError):
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

    @pytest.mark.parametrize(
        ("raw_retcode", "expected_retcode"),
        [
            (10013, 10013),
            (np_int64(10013), 10013),
            ("10013", 10013),
            (" 10013 ", 10013),
            ("+10013", 10013),
            ("-10013", -10013),
            (True, None),
            ("invalid", None),
            ("   ", None),
            (object(), None),
            (_MISSING_RETCODE, None),
        ],
        # ids required: repr(object()) is non-deterministic, breaking --lf/-k
        ids=[
            "int",
            "np-int",
            "str",
            "str-padded",
            "str-plus",
            "str-minus",
            "bool",
            "malformed",
            "empty-str",
            "object",
            "missing-key",
        ],
    )
    def test_place_market_order_normalizes_failed_retcode(
        self,
        raw_retcode: object,
        expected_retcode: int | None,
    ) -> None:
        """Test failed or malformed retcodes from order_send normalize correctly."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        response: dict[str, object] = {"comment": "x"}
        if raw_retcode is not _MISSING_RETCODE:
            response["retcode"] = raw_retcode
        client.order_send.return_value = pd.DataFrame([response])

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result["retcode"] == expected_retcode
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

    def test_calculate_trailing_stop_updates_no_positions(self) -> None:
        """Test empty position sets produce no trailing updates."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame()

        assert (
            calculate_trailing_stop_updates(
                client,
                symbol="EURUSD",
                trailing_stop_ratio=0.02,
            )
            == {}
        )

    @pytest.mark.parametrize(
        ("positions", "tick", "expected"),
        [
            pytest.param(
                [
                    {
                        "ticket": 1,
                        "symbol": "EURUSD",
                        "type": 0,
                        "volume": 0.1,
                        "sl": 1.0,
                    },
                    {
                        "ticket": 2,
                        "symbol": "EURUSD",
                        "type": 0,
                        "volume": 0.1,
                        "sl": 1.19,
                    },
                ],
                {"bid": 1.2, "ask": 1.201},
                {1: 1.188},
                id="buy-uses-bid-skips-already-favorable",
            ),
            pytest.param(
                [
                    {
                        "ticket": 3,
                        "symbol": "EURUSD",
                        "type": 1,
                        "volume": 0.1,
                        "sl": 1.3,
                    },
                    {
                        "ticket": 4,
                        "symbol": "EURUSD",
                        "type": 1,
                        "volume": 0.1,
                        "sl": 1.21,
                    },
                ],
                {"bid": 1.198, "ask": 1.2},
                {3: 1.212},
                id="sell-uses-ask-skips-already-favorable",
            ),
        ],
    )
    def test_calculate_trailing_stop_updates_by_side(
        self,
        positions: list[dict[str, object]],
        tick: dict[str, float],
        expected: dict[int, float],
    ) -> None:
        """Buy uses bid (improves up); sell uses ask (improves down)."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(positions)
        client.symbol_info_tick_as_dict.return_value = tick
        client.symbol_info_as_dict.return_value = {"digits": 4}

        result = calculate_trailing_stop_updates(
            client,
            symbol="EURUSD",
            trailing_stop_ratio=0.01,
        )

        assert result == expected

    @pytest.mark.parametrize(
        ("positions", "tick", "expected"),
        [
            pytest.param(
                [
                    {
                        "ticket": 1,
                        "symbol": "EURUSD",
                        "type": 0,
                        "volume": 0.1,
                        "sl": 1.0,
                    },
                ],
                {"bid": 1.2, "ask": 0.0},
                {1: 1.188},
                id="buy-ignores-invalid-ask",
            ),
            pytest.param(
                [
                    {
                        "ticket": 3,
                        "symbol": "EURUSD",
                        "type": 1,
                        "volume": 0.1,
                        "sl": 1.3,
                    },
                ],
                {"bid": 0.0, "ask": 1.2},
                {3: 1.212},
                id="sell-ignores-invalid-bid",
            ),
        ],
    )
    def test_calculate_trailing_stop_updates_ignores_opposite_side_price(
        self,
        positions: list[dict[str, object]],
        tick: dict[str, object],
        expected: dict[int, float],
    ) -> None:
        """Buy uses bid only; sell uses ask only (other-side price may be invalid)."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(positions)
        client.symbol_info_tick_as_dict.return_value = tick
        client.symbol_info_as_dict.return_value = {"digits": 4}

        result = calculate_trailing_stop_updates(
            client,
            symbol="EURUSD",
            trailing_stop_ratio=0.01,
        )

        assert result == expected

    def test_calculate_trailing_stop_updates_invalid_bid_or_ask(self) -> None:
        """Test invalid side-specific tick prices fail safely without updates."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1, "sl": 1.0},
                {"ticket": 2, "symbol": "EURUSD", "type": 1, "volume": 0.1, "sl": 1.3},
            ],
        )
        client.symbol_info_as_dict.return_value = {"digits": 4}

        client.symbol_info_tick_as_dict.return_value = {"bid": 0.0, "ask": None}

        assert (
            calculate_trailing_stop_updates(
                client,
                symbol="EURUSD",
                trailing_stop_ratio=0.01,
            )
            == {}
        )

    @pytest.mark.parametrize(
        ("tick", "expected"),
        [
            pytest.param(
                {"bid": 1.2, "ask": 0.0},
                {1: 1.188},
                id="invalid-ask-skips-sell-updates",
            ),
            pytest.param(
                {"bid": None, "ask": 1.2},
                {2: 1.212},
                id="invalid-bid-skips-buy-updates",
            ),
        ],
    )
    def test_calculate_trailing_stop_updates_mixed_positions_skip_invalid_side(
        self,
        tick: dict[str, object],
        expected: dict[int, float],
    ) -> None:
        """Test one invalid side price does not block the valid side."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1, "sl": 1.0},
                {"ticket": 2, "symbol": "EURUSD", "type": 1, "volume": 0.1, "sl": 1.3},
            ],
        )
        client.symbol_info_as_dict.return_value = {"digits": 4}
        client.symbol_info_tick_as_dict.return_value = tick

        result = calculate_trailing_stop_updates(
            client,
            symbol="EURUSD",
            trailing_stop_ratio=0.01,
        )

        assert result == expected

    def test_calculate_trailing_stop_updates_invalid_symbol_digits(self) -> None:
        """Test invalid symbol metadata fails safely without updates."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1, "sl": 1.0}],
        )
        client.symbol_info_tick_as_dict.return_value = {"bid": 1.2, "ask": 1.201}
        client.symbol_info_as_dict.return_value = {"digits": "bad"}

        assert (
            calculate_trailing_stop_updates(
                client,
                symbol="EURUSD",
                trailing_stop_ratio=0.01,
            )
            == {}
        )

    @pytest.mark.parametrize(
        "symbol_info",
        [{}, {"digits": None}],
        ids=["missing", "none"],
    )
    def test_calculate_trailing_stop_updates_missing_symbol_digits(
        self,
        symbol_info: dict[str, object],
    ) -> None:
        """Test missing or None symbol digits fail safely without rounded updates."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1, "sl": 1.0}],
        )
        client.symbol_info_tick_as_dict.return_value = {"bid": 1.2, "ask": 1.201}
        client.symbol_info_as_dict.return_value = symbol_info

        assert (
            calculate_trailing_stop_updates(
                client,
                symbol="EURUSD",
                trailing_stop_ratio=0.01,
            )
            == {}
        )

    def test_calculate_trailing_stop_updates_skips_invalid_rows(self) -> None:
        """Test invalid tickets and unknown position types are ignored."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {
                    "ticket": None,
                    "symbol": "EURUSD",
                    "type": 0,
                    "volume": 0.1,
                    "sl": 1.0,
                },
                {
                    "ticket": "5",
                    "symbol": "EURUSD",
                    "type": "unknown",
                    "volume": 0.1,
                    "sl": 1.0,
                },
            ],
        )
        client.symbol_info_tick_as_dict.return_value = {"bid": 1.2, "ask": 1.201}
        client.symbol_info_as_dict.return_value = {"digits": 4}

        assert (
            calculate_trailing_stop_updates(
                client,
                symbol="EURUSD",
                trailing_stop_ratio=0.01,
            )
            == {}
        )

    def test_update_trailing_stop_loss_dry_run(self) -> None:
        """Test trailing-stop update wrapper supports dry-run requests."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1, "sl": 1.0}],
        )
        client.symbol_info_tick_as_dict.return_value = {"bid": 1.2, "ask": 1.201}
        client.symbol_info_as_dict.return_value = {"digits": 4}

        result = update_trailing_stop_loss_for_open_positions(
            client,
            symbol="EURUSD",
            trailing_stop_ratio=0.01,
            dry_run=True,
        )

        assert len(result) == 1
        assert result[0]["status"] == "dry_run"
        _assert_close(_request_from_result(result[0])["sl"], 1.188)
        client.order_send.assert_not_called()

    def test_update_trailing_stop_loss_sends_changed_sl(self) -> None:
        """Test trailing-stop wrapper sends normalized SL/TP updates."""
        client = _mock_trade_client()
        client.symbol_select.return_value = True
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1, "sl": 1.0}],
        )
        client.symbol_info_tick_as_dict.return_value = {"bid": 1.2, "ask": 1.201}
        client.symbol_info_as_dict.return_value = {"digits": 4, "visible": True}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": 10009, "comment": "updated"}],
        )

        result = update_trailing_stop_loss_for_open_positions(
            client,
            symbol="EURUSD",
            trailing_stop_ratio=0.01,
        )

        assert result[0]["status"] == "executed"
        _assert_close(_request_from_result(result[0])["sl"], 1.188)
        client.order_send.assert_called_once()

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
        mocker.patch("mt5cli.trading.Mt5DataClient", return_value=mock_client)

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

    @pytest.mark.parametrize(
        ("raw_retcode", "expected_retcode"),
        [
            (10013, 10013),
            (np_int64(10013), 10013),
            ("10013", 10013),
            ("invalid", None),
            (_MISSING_RETCODE, None),
        ],
        ids=["int", "np-int", "str", "malformed", "missing-key"],
    )
    def test_update_sltp_normalizes_failed_retcode(
        self,
        raw_retcode: object,
        expected_retcode: int | None,
    ) -> None:
        """Test failed or malformed SL/TP retcodes normalize correctly.

        Exhaustive retcode variants are covered in
        test_place_market_order_normalizes_failed_retcode; this set is
        representative because both functions share the same normalization path.
        """
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
        response: dict[str, object] = {"comment": "x"}
        if raw_retcode is not _MISSING_RETCODE:
            response["retcode"] = raw_retcode
        client.order_send.return_value = pd.DataFrame([response])

        result = update_sltp_for_open_positions(client, tickets=[1], stop_loss=1.1)

        assert result[0]["retcode"] == expected_retcode
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
        mocker.patch("mt5cli.trading.Mt5DataClient", return_value=mock_client)

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

    def test_copy_rates_from_pos_fallback_drops_forming_bar(self) -> None:
        """Regression: client with only copy_rates_from_pos_as_df works end-to-end."""
        client = MagicMock(spec=["copy_rates_from_pos_as_df"])
        client.copy_rates_from_pos_as_df.return_value = pd.DataFrame(
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

        client.copy_rates_from_pos_as_df.assert_called_once_with(
            symbol="EURUSD", timeframe=1, start_pos=0, count=3
        )
        assert list(result["close"]) == [1.0, 1.1]
        assert list(result["time"]) == [1700000000, 1700000060]

    def test_copy_rates_from_pos_fallback_resolves_granularity(self) -> None:
        """Fallback path resolves granularity string to integer timeframe."""
        client = MagicMock(spec=["copy_rates_from_pos_as_df"])
        client.copy_rates_from_pos_as_df.return_value = pd.DataFrame(
            {"time": [1, 2, 3, 4], "close": [1.0, 1.1, 1.2, 1.3]},
        )

        fetch_latest_closed_rates_for_trading_client(
            client, symbol="USDJPY", granularity="H1", count=3
        )

        call_kwargs = client.copy_rates_from_pos_as_df.call_args.kwargs
        assert call_kwargs["symbol"] == "USDJPY"
        assert call_kwargs["timeframe"] == 16385
        assert call_kwargs["start_pos"] == 0
        assert call_kwargs["count"] == 4

    def test_copy_rates_from_pos_fallback_returns_count_closed_rows(self) -> None:
        """Fallback path trims to exactly count closed rows after forming-bar drop."""
        client = MagicMock(spec=["copy_rates_from_pos_as_df"])
        client.copy_rates_from_pos_as_df.return_value = pd.DataFrame(
            {"time": list(range(6)), "close": [float(i) for i in range(6)]},
        )

        result = fetch_latest_closed_rates_for_trading_client(
            client, symbol="EURUSD", granularity="M1", count=4
        )

        assert len(result) == 4
        assert list(result["close"]) == [1.0, 2.0, 3.0, 4.0]

    def test_copy_rates_from_pos_fallback_raises_on_invalid_granularity(self) -> None:
        """Invalid granularity raises ValueError before calling the fallback method."""
        client = MagicMock(spec=["copy_rates_from_pos_as_df"])

        with pytest.raises(ValueError, match="Invalid timeframe"):
            fetch_latest_closed_rates_for_trading_client(
                client, symbol="EURUSD", granularity="BADGRAN", count=1
            )

        client.copy_rates_from_pos_as_df.assert_not_called()

    def test_raises_when_trading_client_cannot_fetch_rates(self) -> None:
        """Test missing rate-fetch methods raise Mt5TradingError."""
        client = MagicMock(spec=[])

        with pytest.raises(Mt5OperationError, match="cannot fetch rate data"):
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

    @pytest.mark.parametrize(
        "timestamps",
        [
            [1700000000, 1700003600],
            [1700000000.0, 1700003600.0],
            [np_int64(1700000000), np_int64(1700003600)],
            [np_float64(1700000000.0), np_float64(1700003600.0)],
        ],
        ids=["integers", "floats", "numpy-integers", "numpy-floats"],
    )
    def test_converts_object_numeric_epoch_seconds_to_utc_datetime_index(
        self,
        mocker: MockerFixture,
        timestamps: list[int] | list[float] | list[np_int64] | list[np_float64],
    ) -> None:
        """Test object-dtype real numbers are interpreted as epoch seconds."""
        frame = pd.DataFrame(
            {
                "time": pd.Series(timestamps, dtype=object),
                "close": [1.1, 1.2],
            },
        )
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=frame,
        )

        result = fetch_latest_closed_rates_indexed(
            MagicMock(), symbol="EURUSD", granularity="M1", count=2
        )

        assert list(result.index) == list(
            pd.to_datetime([1700000000, 1700003600], unit="s", utc=True)
        )

    def test_parses_mixed_datetime_like_strings(self, mocker: MockerFixture) -> None:
        """Test object-dtype datetime strings retain datetime-like parsing."""
        timestamps = ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00+01:00"]
        frame = pd.DataFrame({"time": timestamps, "close": [1.1, 1.2]})
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=frame,
        )

        result = fetch_latest_closed_rates_indexed(
            MagicMock(), symbol="EURUSD", granularity="M1", count=2
        )

        assert list(result.index) == list(pd.to_datetime(timestamps, utc=True))

    def test_does_not_treat_bool_as_epoch_seconds(self, mocker: MockerFixture) -> None:
        """Test bool timestamps do not enter the numeric epoch-seconds path."""
        frame = pd.DataFrame(
            {"time": pd.Series([True, False], dtype=object), "close": [1.1, 1.2]},
        )
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=frame,
        )

        with pytest.raises(ValueError, match="invalid or unparseable time data"):
            fetch_latest_closed_rates_indexed(
                MagicMock(), symbol="EURUSD", granularity="M1", count=2
            )

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

    def test_raises_on_nat_time_column(self, mocker: MockerFixture) -> None:
        """Test NaT in the time column raises ValueError instead of silently passing."""
        mocker.patch(
            "mt5cli.trading.fetch_latest_closed_rates_for_trading_client",
            return_value=pd.DataFrame(
                {"time": [1700000000, None], "close": [1.1, 1.2]},
            ),
        )

        with pytest.raises(ValueError, match=r"missing.*NaT.*timestamp"):
            fetch_latest_closed_rates_indexed(
                MagicMock(),
                symbol="EURUSD",
                granularity="M1",
                count=2,
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

    def test_copy_rates_from_pos_fallback_produces_utc_datetime_index(self) -> None:
        """Fallback path via copy_rates_from_pos_as_df produces a UTC DatetimeIndex."""
        client = MagicMock(spec=["copy_rates_from_pos_as_df"])
        client.copy_rates_from_pos_as_df.return_value = pd.DataFrame(
            {
                "time": [1700000000, 1700003600, 1700007200],
                "close": [1.1, 1.2, 1.3],
            },
        )

        result = fetch_latest_closed_rates_indexed(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "time"
        assert str(result.index.tz) == "UTC"
        assert "time" not in result.columns
        assert list(result["close"]) == [1.1, 1.2]


class TestExtractTickPrice:
    """Tests for the public extract_tick_price helper."""

    @pytest.mark.parametrize(
        ("tick", "expected"),
        [
            ({"bid": 1.1000}, 1.1000),
            ({"bid": 2}, 2.0),
            ({"bid": "1.5"}, 1.5),
        ],
        ids=["float", "int", "numeric-string"],
    )
    def test_returns_valid_price(
        self, tick: dict[str, object], expected: float
    ) -> None:
        """Returns a valid positive float for numeric inputs."""
        result = extract_tick_price(tick, "bid")
        assert result is not None
        assert isinstance(result, float)
        _assert_close(result, expected)

    @pytest.mark.parametrize(
        "tick",
        [
            {},
            {"bid": None},
            {"bid": True},
            {"bid": "not_a_number"},
            {"bid": float("nan")},
            {"bid": float("inf")},
            {"bid": float("-inf")},
            {"bid": 0.0},
            {"bid": -1.0},
            {"bid": [1.0]},
        ],
        ids=[
            "missing-key",
            "none",
            "bool",
            "invalid-string",
            "nan",
            "inf",
            "neg-inf",
            "zero",
            "negative",
            "list",
        ],
    )
    def test_returns_none(self, tick: dict[str, object]) -> None:
        """Returns None for missing, invalid, or non-positive price values."""
        assert extract_tick_price(tick, "bid") is None


class TestCalculatePositionsMarginBySymbol:
    """Tests for calculate_positions_margin_by_symbol (#50)."""

    def test_all_symbols_succeed(self, mocker: MockerFixture) -> None:
        """Returns one entry per symbol in first-seen order when all calls succeed."""
        client = _mock_trade_client()
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin",
            side_effect=[12.5, 30.0],
        )

        result = calculate_positions_margin_by_symbol(
            client, symbols=["EURUSD", "GBPUSD"]
        )

        assert list(result.keys()) == ["EURUSD", "GBPUSD"]
        _assert_close(result["EURUSD"], 12.5)
        _assert_close(result["GBPUSD"], 30.0)

    def test_one_symbol_fails_suppress_errors_true(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Skips the failing symbol, emits a warning, and returns the successful one."""
        client = _mock_trade_client()
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin",
            side_effect=[Mt5TradingError("tick unavailable"), 30.0],
        )

        with caplog.at_level(logging.WARNING, logger="mt5cli.trading"):
            result = calculate_positions_margin_by_symbol(
                client, symbols=["EURUSD", "GBPUSD"], suppress_errors=True
            )

        assert result == {"GBPUSD": 30.0}
        assert any(
            "EURUSD" in record.getMessage() and record.levelno == logging.WARNING
            for record in caplog.records
        )

    @pytest.mark.parametrize(
        "exc",
        [
            Mt5TradingError("trading error"),
            Mt5RuntimeError("runtime error"),
            AttributeError("missing attr"),
        ],
    )
    def test_one_symbol_fails_suppress_errors_false(
        self, mocker: MockerFixture, exc: Exception
    ) -> None:
        """Re-raises the first failure for each caught exception type."""
        client = _mock_trade_client()
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin",
            side_effect=[exc, 30.0],
        )

        with pytest.raises(type(exc)):
            calculate_positions_margin_by_symbol(
                client, symbols=["EURUSD", "GBPUSD"], suppress_errors=False
            )

    def test_all_symbols_fail_suppress_errors_true(self, mocker: MockerFixture) -> None:
        """Returns an empty dict when all symbols fail with suppress_errors=True."""
        client = _mock_trade_client()
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin",
            side_effect=[Mt5TradingError("err1"), Mt5TradingError("err2")],
        )

        result = calculate_positions_margin_by_symbol(
            client, symbols=["EURUSD", "GBPUSD"], suppress_errors=True
        )

        assert result == {}

    def test_empty_symbol_list(self, mocker: MockerFixture) -> None:
        """Returns an empty dict for an empty input list without any broker calls."""
        client = _mock_trade_client()
        mock_calc = mocker.patch("mt5cli.trading.calculate_positions_margin")
        result = calculate_positions_margin_by_symbol(client, symbols=[])
        assert result == {}
        mock_calc.assert_not_called()

    def test_duplicate_symbols_preserve_first_seen_order(
        self, mocker: MockerFixture
    ) -> None:
        """Processes each unique symbol once in first-seen order."""
        client = _mock_trade_client()
        mock_calc = mocker.patch(
            "mt5cli.trading.calculate_positions_margin",
            return_value=12.5,
        )

        result = calculate_positions_margin_by_symbol(
            client, symbols=["EURUSD", "GBPUSD", "EURUSD"]
        )

        assert list(result.keys()) == ["EURUSD", "GBPUSD"]
        assert mock_calc.call_count == 2


class TestCalculatePositionsMarginSafe:
    """Tests for calculate_positions_margin_safe (#50)."""

    def test_returns_summed_total(self, mocker: MockerFixture) -> None:
        """Returns the sum of all per-symbol margins."""
        client = _mock_trade_client()
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin",
            side_effect=[12.5, 30.0],
        )

        total = calculate_positions_margin_safe(client, symbols=["EURUSD", "GBPUSD"])

        _assert_close(total, 42.5)

    def test_partial_failure_skips_and_sums(self, mocker: MockerFixture) -> None:
        """Sums only successful margins when one symbol raises."""
        client = _mock_trade_client()
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin",
            side_effect=[Mt5TradingError("tick unavailable"), 30.0],
        )

        total = calculate_positions_margin_safe(client, symbols=["EURUSD", "GBPUSD"])

        _assert_close(total, 30.0)

    def test_all_symbols_fail_returns_zero(self, mocker: MockerFixture) -> None:
        """Returns 0.0 when every symbol raises."""
        client = _mock_trade_client()
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin",
            side_effect=[Mt5TradingError("err1"), Mt5RuntimeError("err2")],
        )

        total = calculate_positions_margin_safe(client, symbols=["EURUSD", "GBPUSD"])

        _assert_close(total, 0.0)

    def test_empty_symbols_returns_zero(self) -> None:
        """Returns 0.0 for an empty symbol list."""
        client = _mock_trade_client()
        total = calculate_positions_margin_safe(client, symbols=[])
        _assert_close(total, 0.0)


class TestFetchRecentHistoryDealsForTradingClient:
    """Tests for fetch_recent_history_deals_for_trading_client."""

    def _fake_client(self, return_value: pd.DataFrame | None) -> MagicMock:
        client = MagicMock()
        client.history_deals_get_as_df.return_value = return_value
        return client

    def test_passes_correct_date_range_and_filters(self) -> None:
        """Calls history_deals_get_as_df with derived date_from/date_to."""
        anchor = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        client = self._fake_client(pd.DataFrame())

        fetch_recent_history_deals_for_trading_client(
            client,
            symbol="JP225",
            group="FX*",
            hours=6.0,
            date_to=anchor,
        )

        client.history_deals_get_as_df.assert_called_once_with(
            date_from=anchor - timedelta(hours=6.0),
            date_to=anchor,
            group="FX*",
            symbol="JP225",
        )

    def test_raises_for_zero_hours(self) -> None:
        """hours=0 raises ValueError."""
        client = self._fake_client(pd.DataFrame())
        with pytest.raises(ValueError, match="hours must be finite and positive"):
            fetch_recent_history_deals_for_trading_client(client, hours=0)

    def test_raises_for_negative_hours(self) -> None:
        """Negative hours raises ValueError."""
        client = self._fake_client(pd.DataFrame())
        with pytest.raises(ValueError, match="hours must be finite and positive"):
            fetch_recent_history_deals_for_trading_client(client, hours=-1.0)

    @pytest.mark.parametrize("bad_hours", [float("nan"), float("inf"), float("-inf")])
    def test_raises_for_non_finite_hours(self, bad_hours: float) -> None:
        """nan, inf, and -inf raise ValueError before reaching timedelta."""
        client = self._fake_client(pd.DataFrame())
        with pytest.raises(ValueError, match="hours must be finite and positive"):
            fetch_recent_history_deals_for_trading_client(client, hours=bad_hours)

    def test_none_result_returns_empty_dataframe(self) -> None:
        """None from underlying client becomes an empty DataFrame."""
        client = self._fake_client(None)
        result = fetch_recent_history_deals_for_trading_client(client, hours=24.0)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_dataframe_result_preserves_schema(self) -> None:
        """Empty DataFrame from client is returned with its columns intact."""
        schema_df = pd.DataFrame(columns=["time", "symbol", "profit", "volume"])
        client = self._fake_client(schema_df)
        result = fetch_recent_history_deals_for_trading_client(client, hours=24.0)
        assert result.empty
        assert list(result.columns) == ["time", "symbol", "profit", "volume"]

    def test_none_result_returns_bare_empty_dataframe(self) -> None:
        """None from client becomes a bare empty DataFrame (no columns)."""
        client = self._fake_client(None)
        result = fetch_recent_history_deals_for_trading_client(client, hours=24.0)
        assert result.empty
        assert list(result.columns) == []

    def test_sorts_by_time_and_resets_index(self) -> None:
        """Unsorted time rows are sorted chronologically and index is reset."""
        t1 = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)
        t2 = datetime(2024, 6, 1, 10, 0, tzinfo=UTC)
        t3 = datetime(2024, 6, 1, 11, 0, tzinfo=UTC)
        df = pd.DataFrame({"time": [t3, t1, t2], "profit": [3.0, 1.0, 2.0]})
        client = self._fake_client(df)

        result = fetch_recent_history_deals_for_trading_client(client, hours=24.0)

        assert list(result["time"]) == [t1, t2, t3]
        assert list(result.index) == [0, 1, 2]

    def test_preserves_all_columns(self) -> None:
        """No columns are dropped from the underlying client result."""
        anchor = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
        df = pd.DataFrame({
            "time": [anchor],
            "symbol": ["JP225"],
            "type": [0],
            "entry": [1],
            "volume": [0.1],
            "profit": [50.0],
            "position_id": [123456],
            "commission": [-0.5],
        })
        client = self._fake_client(df)

        result = fetch_recent_history_deals_for_trading_client(client, hours=24.0)

        assert set(result.columns) == {
            "time",
            "symbol",
            "type",
            "entry",
            "volume",
            "profit",
            "position_id",
            "commission",
        }

    def test_no_time_column_still_returns_data(self) -> None:
        """DataFrames without a time column are returned with RangeIndex."""
        df = pd.DataFrame({"profit": [1.0, 2.0], "ticket": [10, 11]})
        client = self._fake_client(df)

        result = fetch_recent_history_deals_for_trading_client(client, hours=24.0)

        assert list(result["profit"]) == [1.0, 2.0]
        assert list(result.index) == [0, 1]

    def test_defaults_date_to_to_utc_now(self, mocker: MockerFixture) -> None:
        """When date_to is omitted, the window end is datetime.now(UTC)."""
        frozen = datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)
        mock_dt = mocker.patch("mt5cli.trading.datetime")
        mock_dt.now.return_value = frozen
        client = self._fake_client(pd.DataFrame())

        fetch_recent_history_deals_for_trading_client(client, hours=1.0)

        mock_dt.now.assert_called_once_with(UTC)
        client.history_deals_get_as_df.assert_called_once_with(
            date_from=frozen - timedelta(hours=1.0),
            date_to=frozen,
            group=None,
            symbol=None,
        )


class TestCreateTradingClientHistoryDealsIntegration:
    """Verify create_trading_client() result satisfies fetch_recent_history_deals."""

    def test_create_trading_client_result_usable_with_history_deals_helper(
        self,
        mocker: MockerFixture,
    ) -> None:
        """create_trading_client() result passes directly to fetch_recent_history_deals.

        This exercises the intended SDK call path without a live MT5 terminal.
        The mock satisfies both _Mt5ClientProtocol and _HistoryDealsClientProtocol.
        """
        mock_raw_client = MagicMock()
        mocker.patch("mt5cli.trading.Mt5DataClient", return_value=mock_raw_client)

        anchor = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        expected_df = pd.DataFrame({"time": [anchor], "profit": [10.0]})
        mock_raw_client.history_deals_get_as_df.return_value = expected_df

        client = create_trading_client(login=12345, server="Demo")
        result = fetch_recent_history_deals_for_trading_client(
            client,
            symbol="EURUSD",
            hours=6.0,
            date_to=anchor,
        )

        mock_raw_client.history_deals_get_as_df.assert_called_once_with(
            date_from=anchor - timedelta(hours=6.0),
            date_to=anchor,
            group=None,
            symbol="EURUSD",
        )
        assert list(result["profit"]) == [10.0]
