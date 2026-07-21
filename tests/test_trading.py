"""Tests for trading session helpers and operational utilities."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_args
from unittest.mock import MagicMock

import pandas as pd
import pytest
from numpy import int64 as np_int64
from pdmt5 import Mt5RuntimeError

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from mt5cli import trading
from mt5cli.client import MT5Client
from mt5cli.exceptions import Mt5OperationError
from mt5cli.trading import (
    MarginVolume,
    OrderExecutionResult,
    OrderLimits,
    OrderSide,
    ProjectionMode,
    TickClockCalibration,
    TickClockNormalizer,
    _aggregate_failure_status,  # type: ignore[reportPrivateUsage]
    _execution_receipt,  # type: ignore[reportPrivateUsage]
    _filter_positions,  # type: ignore[reportPrivateUsage]
    _Mt5ClientProtocol,  # type: ignore[reportPrivateUsage]
    _optional_positive_int,  # type: ignore[reportPrivateUsage]
    _plain_value,  # type: ignore[reportPrivateUsage]
    _response_mapping,  # type: ignore[reportPrivateUsage]
    _snapshot_from_value,  # type: ignore[reportPrivateUsage]
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
    detect_position_side,
    determine_order_limits,
    ensure_symbol_selected,
    estimate_order_margin,
    extract_tick_price,
    fetch_latest_closed_rates_indexed,
    get_account_snapshot,
    get_positions_frame,
    get_symbol_snapshot,
    get_tick_snapshot,
    normalize_order_volume,
    place_market_order,
    resolve_broker_filling_mode,
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
    client.mt5.SYMBOL_FILLING_FOK = 1
    client.mt5.SYMBOL_FILLING_IOC = 2
    client.mt5.ORDER_TIME_GTC = 40
    client.mt5.SYMBOL_TRADE_EXECUTION_MARKET = 3
    client.mt5.SYMBOL_TRADE_EXECUTION_REQUEST = 4
    client.mt5.SYMBOL_TRADE_EXECUTION_INSTANT = 5
    client.mt5.TRADE_RETCODE_PLACED = 10008
    client.mt5.TRADE_RETCODE_DONE = 10009
    client.mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
    return client


def _assert_close(actual: object, expected: float) -> None:
    assert abs(float(cast("float", actual)) - expected) < 1e-9


def _request_from_result(result: OrderExecutionResult) -> dict[str, object]:
    return result.request


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

    def test_detect_position_side_filters_by_magic(self) -> None:
        """Magic-scoped side detection ignores foreign positions."""
        client = MagicMock()
        client.mt5.POSITION_TYPE_BUY = 0
        client.mt5.POSITION_TYPE_SELL = 1
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"type": 0, "volume": 0.3, "magic": 7},
                {"type": 1, "volume": 0.2, "magic": 9},
            ],
        )

        assert detect_position_side(client, "EURUSD", magic=7) == "long"
        assert detect_position_side(client, "EURUSD", magic=9) == "short"

    def test_detect_position_side_magic_is_fail_closed_without_magic_column(
        self,
    ) -> None:
        """Magic-scoped side detection returns None without magic metadata."""
        client = MagicMock()
        client.mt5.POSITION_TYPE_BUY = 0
        client.mt5.POSITION_TYPE_SELL = 1
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"type": 0, "volume": 0.3}],
        )

        assert detect_position_side(client, "EURUSD", magic=7) is None


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
        """Test invalid entry tick values raise Mt5OperationError."""
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
        """Test protective levels inside trade_stops_level raise Mt5OperationError."""
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

    @pytest.mark.parametrize(
        ("visible", "select_result", "expected_calls"),
        [
            (True, True, 0),
            (False, True, 1),
        ],
        ids=["already-visible", "select-hidden"],
    )
    def test_ensure_symbol_selected_success_cases(
        self,
        visible: bool,
        select_result: bool,
        expected_calls: int,
    ) -> None:
        """Test symbol selection only occurs when the symbol is hidden."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"visible": visible}
        client.symbol_select.return_value = select_result

        ensure_symbol_selected(client, "EURUSD")

        assert client.symbol_select.call_count == expected_calls
        if expected_calls:
            client.symbol_select.assert_called_once_with("EURUSD", enable=True)
        else:
            client.symbol_select.assert_not_called()

    def test_raises_when_symbol_selection_fails(self) -> None:
        """Test failed symbol selection raises Mt5OperationError."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"visible": False}
        client.symbol_select.return_value = False
        client.last_error.return_value = (1, "not found")

        with pytest.raises(Mt5OperationError, match="Failed to select symbol 'EURUSD'"):
            ensure_symbol_selected(client, "EURUSD")

    def test_raises_when_symbol_select_is_unavailable(self) -> None:
        """Test missing symbol_select raises Mt5OperationError."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"visible": False}
        del client.symbol_select

        with pytest.raises(
            Mt5OperationError,
            match="missing required method: symbol_select",
        ):
            ensure_symbol_selected(client, "EURUSD")


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

        result = get_account_snapshot(cast("_Mt5ClientProtocol", ObjectClient()))

        assert result["login"] == 7
        assert result["currency"] == "JPY"

    def test_account_snapshot_requires_supported_method(self) -> None:
        """Test missing account snapshot methods raise AttributeError."""
        client = object()

        with pytest.raises(AttributeError, match="account_info"):
            get_account_snapshot(cast("_Mt5ClientProtocol", client))

    def test_symbol_and_tick_snapshots_fill_symbol(self) -> None:
        """Test symbol and tick snapshots expose stable fields."""
        client = MagicMock()
        client.symbol_info_as_dict.return_value = {"digits": 5, "visible": True}
        client.symbol_info_tick_as_dict.return_value = {"bid": 1.1, "ask": 1.2}

        assert get_symbol_snapshot(client, "EURUSD")["symbol"] == "EURUSD"
        assert get_tick_snapshot(client, "EURUSD")["symbol"] == "EURUSD"

    def test_tick_snapshot_requests_raw_time_from_mt5_client(self) -> None:
        """Test the canonical client preserves pdmt5's numeric MT5 time."""

        class ConnectedClient:
            def symbol_info_tick_as_dict(
                self,
                symbol: str,
                skip_to_datetime: bool = False,
            ) -> dict[str, float]:
                assert symbol == "USOIL"
                assert skip_to_datetime is True
                return {"time": 1717250400, "bid": 78.5, "ask": 78.6}

        client = MT5Client.from_connected_client(ConnectedClient())

        snapshot = get_tick_snapshot(client, "USOIL")

        assert snapshot["time"] == 1717250400
        _assert_close(snapshot["bid"], 78.5)
        _assert_close(snapshot["ask"], 78.6)

    def test_tick_snapshot_supports_dict_method_without_skip_keyword(self) -> None:
        """Test compatible dictionary clients need not accept pdmt5 options."""

        class DictionaryClient:
            def symbol_info_tick_as_dict(self, *, symbol: str) -> dict[str, float]:
                assert symbol == "UKOIL"
                return {"time": 1717254000, "bid": 80.1, "ask": 80.2}

        snapshot = get_tick_snapshot(
            cast("_Mt5ClientProtocol", DictionaryClient()),
            "UKOIL",
        )

        assert snapshot["time"] == 1717254000

    @pytest.mark.parametrize(
        "tick_time",
        [
            pd.Timestamp("2024-06-01 15:00:00"),
            pd.Timestamp("2024-06-01 15:00:00", tz=UTC),
        ],
        ids=["naive", "timezone-aware"],
    )
    def test_tick_snapshot_normalizes_dataframe_timestamp_to_numeric(
        self,
        tick_time: pd.Timestamp,
    ) -> None:
        """Test DataFrame fallbacks cannot leak a naive pandas Timestamp."""
        client = MagicMock()
        del client.symbol_info_tick_as_dict
        client.symbol_info_tick.return_value = pd.DataFrame(
            [{"time": tick_time, "bid": 1.0}],
        )

        snapshot = get_tick_snapshot(client, "EURUSD")

        _assert_close(snapshot["time"], 1717254000.0)
        assert not isinstance(snapshot["time"], pd.Timestamp)

    @pytest.mark.parametrize(
        ("tick_time", "expected"),
        [
            (True, None),
            (float("inf"), None),
            (pd.NaT, None),
            ("1717254000", 1717254000.0),
            ("not-a-number", None),
            (object(), None),
        ],
        ids=[
            "boolean",
            "non-finite",
            "missing-datetime",
            "numeric-string",
            "non-numeric-string",
            "unsupported",
        ],
    )
    def test_tick_snapshot_time_always_matches_numeric_contract(
        self,
        tick_time: object,
        expected: float | None,
    ) -> None:
        """Test non-canonical time payloads normalize to a number or None."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = {"time": tick_time}

        snapshot = get_tick_snapshot(client, "EURUSD")

        if expected is None:
            assert snapshot["time"] is None
        else:
            _assert_close(snapshot["time"], expected)

    def test_symbol_and_tick_snapshots_use_object_fallbacks(self) -> None:
        """Test symbol and tick snapshots support non-dict MT5 values."""
        client = MagicMock()
        del client.symbol_info_as_dict
        del client.symbol_info_tick_as_dict
        client.symbol_info.return_value = SimpleNamespace(digits=3)
        client.symbol_info_tick.return_value = SimpleNamespace(bid=1.0, ask=1.1)

        assert get_symbol_snapshot(client, "USDJPY")["digits"] == 3
        _assert_close(get_tick_snapshot(client, "USDJPY")["ask"], 1.1)

    def test_account_snapshot_reads_dataframe_snapshots(self) -> None:
        """Test account snapshots accept one-row DataFrame broker payloads."""
        client = MagicMock()
        client.account_info_as_dict.return_value = pd.DataFrame(
            [{"login": 42, "equity": 100.0}],
        )

        result = get_account_snapshot(client)

        assert result["login"] == 42
        _assert_close(result["equity"], 100.0)

    def test_positions_frame_adds_stable_columns(self) -> None:
        """Test missing position columns are added to empty frames."""
        client = MagicMock()
        client.positions_get_as_df.return_value = pd.DataFrame()

        result = get_positions_frame(client, symbol="EURUSD")

        assert "ticket" in result.columns
        assert "comment" in result.columns

    @pytest.mark.parametrize(
        "tick",
        [
            {"bid": 99.0, "ask": 101.0},
            {"bid": "99.0", "ask": "101.0"},
        ],
        ids=["numeric", "numeric-string"],
    )
    def test_calculate_spread_ratio(self, tick: dict[str, object]) -> None:
        """Test spread ratio uses mid-price denominator for numeric ticks."""
        client = MagicMock()
        client.symbol_info_tick_as_dict.return_value = tick

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
        """Test invalid bid/ask values raise Mt5OperationError."""
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
            (2.5, 0.1, float("nan"), 0.1, 2.5),
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
            "non-finite-max-no-cap",
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

    @pytest.mark.parametrize(
        "volume",
        [0.0, float("nan"), float("inf")],
        ids=["zero", "nan", "inf"],
    )
    def test_rejects_invalid_volume(self, volume: float) -> None:
        """Test non-positive or non-finite volume raises Mt5OperationError."""
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
        """Test invalid margin estimates raise Mt5OperationError."""
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

    @pytest.mark.parametrize(
        (
            "positions_records",
            "tick_records",
            "margin_values",
            "symbols",
            "expected",
            "expected_margin_calls",
        ),
        [
            (
                [
                    {"symbol": "EURUSD", "type": 0, "volume": 0.1},
                    {"symbol": "USDJPY", "type": 1, "volume": 0.2},
                ],
                [
                    {"ask": 1.1010, "bid": 1.1000},
                    {"ask": 110.0, "bid": 109.0},
                ],
                [12.5, 20.0],
                ["EURUSD"],
                12.5,
                1,
            ),
            (
                [
                    {"symbol": "EURUSD", "type": 0, "volume": 0.1},
                    {"symbol": "EURUSD", "type": 1, "volume": 0.2},
                ],
                [
                    {"ask": 1.1010, "bid": 1.1000},
                    {"ask": 1.1010, "bid": 1.1000},
                ],
                [12.5, 24.8],
                None,
                37.3,
                2,
            ),
            (
                [
                    {"symbol": "EURUSD", "type": 0, "volume": 0.1},
                    {"symbol": "GBPUSD", "type": 1, "volume": 0.3},
                ],
                [
                    {"ask": 1.1010, "bid": 1.1000},
                    {"ask": 1.3010, "bid": 1.3000},
                ],
                [12.5, 30.0],
                None,
                42.5,
                2,
            ),
        ],
        ids=["filters-by-symbol", "mixed-buy-sell", "multiple-symbols"],
    )
    def test_sums_margin_for_normal_position_sets(
        self,
        positions_records: list[dict[str, object]],
        tick_records: list[dict[str, float]],
        margin_values: list[float],
        symbols: list[str] | None,
        expected: float,
        expected_margin_calls: int,
    ) -> None:
        """Test standard valid position sets sum per-group margins correctly."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(positions_records)
        client.symbol_info_tick_as_dict.side_effect = tick_records
        client.order_calc_margin.side_effect = margin_values

        margin = calculate_positions_margin(client, symbols=symbols)

        _assert_close(margin, expected)
        assert client.order_calc_margin.call_count == expected_margin_calls

    def test_propagates_invalid_tick_or_margin_errors(self) -> None:
        """Test invalid tick or margin data raises Mt5OperationError."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"symbol": "EURUSD", "type": 0, "volume": 0.1}],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": None, "bid": 1.1000}

        with pytest.raises(Mt5OperationError, match="Tick price is unavailable"):
            calculate_positions_margin(client)

    @pytest.mark.parametrize(
        "positions_records",
        [
            [
                {"symbol": "", "type": 0, "volume": 0.1},
                {"symbol": "EURUSD", "type": 0, "volume": 0.0},
                {"symbol": "EURUSD", "type": 2, "volume": 0.1},
                {"symbol": "EURUSD", "type": 0, "volume": 0.1},
            ],
            [
                {"symbol": "EURUSD", "type": 0, "volume": float("nan")},
                {"symbol": "EURUSD", "type": 0, "volume": float("inf")},
                {"symbol": "EURUSD", "type": 0, "volume": 0.1},
            ],
        ],
        ids=["invalid-symbol-volume-type", "non-finite-volume"],
    )
    def test_skips_invalid_rows_and_sums_remaining_valid_row(
        self,
        positions_records: list[dict[str, object]],
    ) -> None:
        """Test malformed or non-finite position rows are ignored when summing."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(positions_records)
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

    @pytest.mark.parametrize(
        ("volume_max", "budget", "margin_values", "expected"),
        [
            (1.0, 130.0, [25.0, 75.0, 100.0, 150.0], 0.4),
            (0.5, 130.0, [10.0, 150.0, 150.0], 0.0),
        ],
        ids=["steps-down-to-affordable-boundary", "all-steps-unaffordable"],
    )
    def test_calculate_volume_by_margin_binary_search_affordability_boundaries(
        self,
        volume_max: float,
        budget: float,
        margin_values: list[float],
        expected: float,
    ) -> None:
        """Binary search returns the largest affordable step or zero when none fit."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "volume_min": 0.1,
            "volume_max": volume_max,
            "volume_step": 0.1,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.side_effect = margin_values

        result = calculate_volume_by_margin(client, "EURUSD", budget, "BUY")

        _assert_close(result, expected)

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
            cast("_Mt5ClientProtocol", ClientWithoutNative()),
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

    @pytest.mark.parametrize(
        "account",
        [{"equity": 1000.0}, {"equity": 1000.0, "margin": None}],
        ids=["absent", "none"],
    )
    def test_new_position_margin_ratio_defaults_missing_margin_to_zero(
        self, account: dict[str, float | None]
    ) -> None:
        """Test absent or None account margin is normalized to zero."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = account

        _assert_close(
            calculate_new_position_margin_ratio(client, symbol="EURUSD"),
            0.0,
        )

    def test_new_position_margin_ratio_accepts_non_positive_order_margin(self) -> None:
        """Test non-positive hypothetical order margin is accepted as-is."""
        client = _mock_trade_client()
        client.account_info_as_dict.return_value = {"equity": 1000.0, "margin": 50.0}
        client.symbol_info_tick_as_dict.return_value = {"ask": 100.0, "bid": 99.0}
        client.order_calc_margin.return_value = 0.0

        _assert_close(
            calculate_new_position_margin_ratio(
                client,
                symbol="EURUSD",
                new_position_side="BUY",
                new_position_volume=0.1,
            ),
            0.05,
        )

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
            side_effect=[Mt5OperationError("bad tick"), 30.0],
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

    @pytest.mark.parametrize(
        ("suppress_errors", "expected_outcome"),
        [
            pytest.param(True, "suppressed", id="suppresses-projected-failure"),
            pytest.param(False, "raised", id="reraises-projected-failure"),
        ],
    )
    def test_symbol_group_margin_ratio_projected_failure(
        self,
        suppress_errors: bool,
        expected_outcome: str,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test add-mode projected margin failures suppress or reraise."""
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

        if suppress_errors:
            assert expected_outcome == "suppressed"
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
            return

        assert expected_outcome == "raised"
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

    @pytest.mark.parametrize(
        ("suppress_errors", "expected_ratio"),
        [
            pytest.param(True, 0.025, id="suppresses"),
            pytest.param(False, None, id="reraises"),
        ],
    )
    def test_symbol_group_margin_ratio_replace_mode_candidate_failure(
        self,
        suppress_errors: bool,
        expected_ratio: float | None,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test replace-mode candidate failures suppress or reraise."""
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

        if suppress_errors:
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
            _assert_close(result, cast("float", expected_ratio))
            assert "Skipping projected margin" in caplog.text
            return

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

        assert result.status == "dry_run"
        assert _request_from_result(result)["type"] == client.mt5.ORDER_TYPE_BUY
        client.order_send.assert_not_called()
        client.symbol_select.assert_not_called()

    @pytest.mark.parametrize(
        ("order_side", "expected_price"),
        [("BUY", 1.2), ("SELL", 1.1)],
        ids=["buy-uses-ask", "sell-uses-bid"],
    )
    def test_place_market_order_dry_run_populates_request_price(
        self,
        order_side: OrderSide,
        expected_price: float,
    ) -> None:
        """Test dry-run receipts quote the side-appropriate price."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"visible": False}
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side=order_side,
            dry_run=True,
        )

        assert result.status == "dry_run"
        _assert_close(result.request_price, expected_price)
        _assert_close(result.request["price"], expected_price)
        assert result.response is None
        assert result.filled_price is None
        client.symbol_select.assert_not_called()
        client.order_send.assert_not_called()

    def test_place_market_order_dry_run_bad_tick_returns_failed(self) -> None:
        """Test a dry run without a usable quote fails closed without sending."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": None, "bid": 1.1}

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
            dry_run=True,
        )

        assert result.status == "failed"
        assert result.dry_run is True
        assert "Tick price is unavailable" in str(result.comment)
        client.symbol_select.assert_not_called()
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

    def test_place_market_order_supports_optional_deviation_comment_and_magic(
        self,
    ) -> None:
        """Test optional request metadata is preserved for market orders."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
            deviation=7,
            comment="close-me",
            magic=42,
            dry_run=True,
        )

        assert _request_from_result(result)["deviation"] == 7
        assert _request_from_result(result)["comment"] == "close-me"
        assert _request_from_result(result)["magic"] == 42

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

    def test_place_market_order_receipts_missing_mt5_constant(self) -> None:
        """Test broker preparation failures produce a failed receipt."""
        client = _mock_trade_client()
        del client.mt5.ORDER_FILLING_IOC
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
            dry_run=True,
        )

        assert result.status == "failed"
        assert result.dry_run is True
        assert "ORDER_FILLING_IOC" in str(result.comment)

    def test_place_market_order_rejects_invalid_volume(self) -> None:
        """Test non-positive volume raises a trading error."""
        with pytest.raises(Mt5OperationError):
            place_market_order(
                _mock_trade_client(),
                symbol="EURUSD",
                volume=0.0,
                order_side="BUY",
            )

    def test_place_market_order_receipts_bad_tick(self) -> None:
        """Test unavailable broker tick price produces a failed receipt."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": None, "bid": 1.1}

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result.status == "failed"
        assert result.request_price is None

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

        assert result.status == "filled"
        assert result.retcode == 10009
        client.order_send.assert_called_once()

    def test_place_market_order_preserves_explicit_response_magic_zero(self) -> None:
        """Test an explicit response magic of zero is not replaced by the request."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": 10009, "comment": "done", "magic": 0}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="SELL",
            magic=42,
        )

        assert result.magic == 0

    @pytest.mark.parametrize(
        ("retcode", "expected_status"),
        [
            (10008, "placed"),
            (10010, "partial_fill"),
        ],
        ids=["placed", "partial-fill"],
    )
    def test_place_market_order_maps_broker_status_categories(
        self,
        retcode: int,
        expected_status: str,
    ) -> None:
        """Test broker retcodes map to placed and partial-fill statuses."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = pd.DataFrame(
            [{"retcode": retcode, "comment": "ok"}],
        )

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result.status == expected_status
        _assert_close(result.requested_volume, 0.1)

    def test_place_market_order_captures_send_errors_in_receipt(self) -> None:
        """Test live send failures normalize to failed execution receipts."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.side_effect = RuntimeError("send failed")

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result.status == "failed"
        assert "send failed" in str(result.comment)

    def test_place_market_order_marks_unmappable_response_as_failed(self) -> None:
        """Test unsupported broker response shapes normalize to failed receipts."""
        client = _mock_trade_client()
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = object()

        result = place_market_order(
            client,
            symbol="EURUSD",
            volume=0.1,
            order_side="BUY",
        )

        assert result.status == "failed"

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

        assert result.retcode == expected_retcode
        if expected_retcode is None:
            assert result.status == "malformed"
        else:
            assert result.status == "rejected"

    @pytest.mark.parametrize(
        ("symbol_info", "preferred_modes", "default_mode", "expected"),
        [
            (
                {"filling_mode": 2, "trade_exemode": 3},
                ("IOC", "FOK"),
                "IOC",
                "IOC",
            ),
            (
                {"filling_mode": 1, "trade_exemode": 3},
                ("IOC", "FOK"),
                "IOC",
                "FOK",
            ),
            (
                {"filling_mode": 0, "trade_exemode": 4},
                ("RETURN", "IOC"),
                "IOC",
                "RETURN",
            ),
            (
                {"filling_mode": None, "trade_exemode": None},
                ("RETURN", "FOK"),
                "IOC",
                "RETURN",
            ),
            (
                {"filling_mode": 2, "trade_exemode": 3},
                ("FOK",),
                "IOC",
                "IOC",
            ),
            (
                {"filling_mode": 2, "trade_exemode": 3},
                ("FOK",),
                "RETURN",
                "IOC",
            ),
        ],
        ids=[
            "ioc",
            "fok-fallback",
            "return",
            "default-fallback",
            "supported-default-fallback",
            "ignore-unsupported-default",
        ],
    )
    def test_resolve_broker_filling_mode(
        self,
        symbol_info: dict[str, object],
        preferred_modes: tuple[str, ...],
        default_mode: str,
        expected: str,
    ) -> None:
        """Test filling-mode resolution prefers supported modes then falls back."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = symbol_info

        result = resolve_broker_filling_mode(
            client,
            symbol="EURUSD",
            preferred_modes=cast("Any", preferred_modes),
            default_mode=cast("Any", default_mode),
        )

        assert result == expected

    @pytest.mark.parametrize(
        ("preferred_modes", "default_mode"),
        [
            pytest.param(("BAD",), "IOC", id="bad-preferred"),
            pytest.param(("IOC",), "BAD", id="bad-default"),
        ],
    )
    def test_resolve_broker_filling_mode_rejects_invalid_mode_names(
        self,
        preferred_modes: tuple[str, ...],
        default_mode: str,
    ) -> None:
        """Test invalid preferred/default filling mode names raise ValueError."""
        client = _mock_trade_client()

        with pytest.raises(ValueError, match="Unsupported order_filling mode"):
            resolve_broker_filling_mode(
                client,
                symbol="EURUSD",
                preferred_modes=cast("Any", preferred_modes),
                default_mode=cast("Any", default_mode),
            )

    def test_resolve_broker_filling_mode_keeps_preferred_when_metadata_missing(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Missing metadata should fail open to the caller-preferred mode."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {"filling_mode": None}

        with caplog.at_level(logging.DEBUG):
            result = resolve_broker_filling_mode(
                client,
                symbol="EURUSD",
                preferred_modes=("FOK", "IOC"),
            )

        assert result == "FOK"
        assert "keeping preferred mode" in caplog.text

    def test_resolve_broker_filling_mode_supports_return_without_bitmask(self) -> None:
        """RETURN should be allowed when execution mode is non-market."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "filling_mode": None,
            "trade_exemode": client.mt5.SYMBOL_TRADE_EXECUTION_REQUEST,
        }

        result = resolve_broker_filling_mode(
            client,
            symbol="EURUSD",
            preferred_modes=("RETURN", "FOK"),
        )

        assert result == "RETURN"

    @pytest.mark.parametrize(
        "execution_attr",
        ["SYMBOL_TRADE_EXECUTION_REQUEST", "SYMBOL_TRADE_EXECUTION_INSTANT"],
        ids=["request-execution", "instant-execution"],
    )
    @pytest.mark.parametrize(
        "filling_mode", [0, None], ids=["filling-mode-zero", "filling-mode-none"]
    )
    @pytest.mark.parametrize(
        ("preferred_modes", "expected"),
        [
            pytest.param(("IOC", "FOK"), "IOC", id="prefers-ioc"),
            pytest.param(("FOK", "IOC"), "FOK", id="prefers-fok"),
        ],
    )
    def test_resolve_broker_filling_mode_request_instant_execution_implies_ioc_fok(
        self,
        execution_attr: str,
        filling_mode: int | None,
        preferred_modes: tuple[str, str],
        expected: str,
    ) -> None:
        """Request/Instant execution supports IOC/FOK regardless of the mask.

        MQL5 permits IOC and FOK for Request and Instant execution even when
        ``SYMBOL_FILLING_MODE`` carries no matching bits (that bitmask only
        governs Market/Exchange execution), so the resolver must honor
        whichever of IOC/FOK is preferred without requiring mask bits.
        """
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "filling_mode": filling_mode,
            "trade_exemode": getattr(client.mt5, execution_attr),
        }

        result = resolve_broker_filling_mode(
            client,
            symbol="EURUSD",
            preferred_modes=cast("Any", preferred_modes),
        )

        assert result == expected

    def test_resolve_broker_filling_mode_keeps_preferred_when_metadata_unparseable(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unparseable metadata should still fail open to the preferred mode."""
        client = _mock_trade_client()
        client.symbol_info_as_dict.return_value = {
            "filling_mode": 0,
            "trade_exemode": client.mt5.SYMBOL_TRADE_EXECUTION_MARKET,
        }

        with caplog.at_level(logging.DEBUG):
            result = resolve_broker_filling_mode(
                client,
                symbol="EURUSD",
                preferred_modes=("FOK", "IOC"),
            )

        assert result == "FOK"
        assert "unparseable" in caplog.text

    @pytest.mark.parametrize(
        ("filter_kwargs", "expected_order_side", "expected_position"),
        [
            pytest.param(
                {"symbols": "EURUSD"},
                "SELL",
                1,
                id="filter-by-symbol",
            ),
            pytest.param(
                {"tickets": [2]},
                "BUY",
                2,
                id="filter-by-ticket",
            ),
        ],
    )
    def test_close_open_positions_filters_and_dry_runs(
        self,
        filter_kwargs: dict[str, object],
        expected_order_side: str,
        expected_position: int,
    ) -> None:
        """Test close helper filters positions and builds opposite orders."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1},
                {"ticket": 2, "symbol": "USDJPY", "type": 1, "volume": 0.2},
            ],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = close_open_positions(
            client, dry_run=True, **cast("Any", filter_kwargs)
        )

        assert len(result) == 1
        assert result[0].order_side == expected_order_side
        assert _request_from_result(result[0])["position"] == expected_position

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

    def test_close_open_positions_forwards_optional_request_fields(self) -> None:
        """Test close helper forwards deviation/comment/magic into dry-run requests."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 9, "symbol": "EURUSD", "type": 0, "volume": 0.1, "magic": 42}],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = close_open_positions(
            client,
            tickets=[9],
            deviation=8,
            comment="close-me",
            magic=42,
            dry_run=True,
        )

        request = _request_from_result(result[0])
        assert request["deviation"] == 8
        assert request["comment"] == "close-me"
        assert request["magic"] == 42

    def test_close_open_positions_magic_filter_is_fail_closed_without_column(
        self,
    ) -> None:
        """Test magic-scoped close operations skip rows without magic metadata."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 9, "symbol": "EURUSD", "type": 0, "volume": 0.1}],
        )

        result = close_open_positions(client, magic=42, dry_run=True)

        assert result == []
        client.order_send.assert_not_called()

    def test_close_open_positions_resolves_filling_mode_per_symbol(self) -> None:
        """Default closes resolve the broker filling mode instead of assuming IOC."""
        client = _mock_trade_client()
        client.mt5.ORDER_FILLING_FOK = 31
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1},
                {"ticket": 2, "symbol": "EURUSD", "type": 0, "volume": 0.2},
            ],
        )
        client.symbol_info_as_dict.return_value = {
            "filling_mode": client.mt5.SYMBOL_FILLING_FOK,
            "trade_exemode": client.mt5.SYMBOL_TRADE_EXECUTION_MARKET,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = close_open_positions(client, symbols="EURUSD", dry_run=True)

        assert [_request_from_result(r)["type_filling"] for r in result] == [31, 31]
        client.symbol_info_as_dict.assert_called_once()

    def test_close_open_positions_resolves_filling_mode_per_distinct_symbol(
        self,
    ) -> None:
        """Each distinct symbol gets its own resolved mode with one lookup each."""
        client = _mock_trade_client()
        client.mt5.ORDER_FILLING_FOK = 31
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1},
                {"ticket": 2, "symbol": "USDJPY", "type": 0, "volume": 0.2},
                {"ticket": 3, "symbol": "EURUSD", "type": 0, "volume": 0.3},
            ],
        )
        snapshots = {
            "EURUSD": {
                "filling_mode": client.mt5.SYMBOL_FILLING_FOK,
                "trade_exemode": client.mt5.SYMBOL_TRADE_EXECUTION_MARKET,
            },
            "USDJPY": {
                "filling_mode": (
                    client.mt5.SYMBOL_FILLING_FOK | client.mt5.SYMBOL_FILLING_IOC
                ),
                "trade_exemode": client.mt5.SYMBOL_TRADE_EXECUTION_MARKET,
            },
        }
        client.symbol_info_as_dict.side_effect = (
            lambda symbol: snapshots[symbol]  # pyright: ignore[reportUnknownLambdaType]
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = close_open_positions(client, dry_run=True)

        # Order matches the source positions frame: EURUSD, USDJPY, EURUSD.
        type_fillings = [_request_from_result(r)["type_filling"] for r in result]
        assert [r.symbol for r in result] == ["EURUSD", "USDJPY", "EURUSD"]
        assert type_fillings == [31, 30, 31]
        # One symbol_info lookup per unique symbol, not per position.
        assert client.symbol_info_as_dict.call_count == 2

    def test_close_open_positions_keeps_ioc_when_supported(self) -> None:
        """Symbols advertising IOC support keep the IOC close default."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1}],
        )
        client.symbol_info_as_dict.return_value = {
            "filling_mode": (
                client.mt5.SYMBOL_FILLING_FOK | client.mt5.SYMBOL_FILLING_IOC
            ),
            "trade_exemode": client.mt5.SYMBOL_TRADE_EXECUTION_MARKET,
        }
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = close_open_positions(client, symbols="EURUSD", dry_run=True)

        assert _request_from_result(result[0])["type_filling"] == 30

    @pytest.mark.parametrize(
        ("order_filling_mode", "expected"),
        [
            pytest.param("IOC", 30, id="ioc"),
            pytest.param("FOK", 31, id="fok"),
            pytest.param("RETURN", 32, id="return"),
        ],
    )
    def test_close_open_positions_forwards_explicit_filling_mode(
        self, order_filling_mode: str, expected: int
    ) -> None:
        """An explicit order_filling_mode bypasses broker resolution."""
        client = _mock_trade_client()
        client.mt5.ORDER_FILLING_FOK = 31
        client.mt5.ORDER_FILLING_RETURN = 32
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1}],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}

        result = close_open_positions(
            client,
            symbols="EURUSD",
            order_filling_mode=cast("Any", order_filling_mode),
            dry_run=True,
        )

        assert _request_from_result(result[0])["type_filling"] == expected
        client.symbol_info_as_dict.assert_not_called()

    def test_close_open_positions_captures_send_errors_in_receipt(self) -> None:
        """Test close failures normalize to failed execution receipts."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1}],
        )
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.side_effect = RuntimeError("close failed")

        results = close_open_positions(client, symbols="EURUSD")

        assert results[0].status == "failed"
        assert "close failed" in str(results[0].comment)

    def test_filter_positions_magic_is_fail_closed_without_magic_column(self) -> None:
        """Test direct magic filtering fails closed when the DataFrame lacks magic."""
        positions = pd.DataFrame([{"ticket": 1, "symbol": "EURUSD"}])

        result = _filter_positions(positions, magic=42)

        assert result.empty

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
        assert result[0].status == "dry_run"
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

        assert result[0].status == "filled"
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

        assert result[0].status == "filled"
        assert result[0].retcode == 10009
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

        assert result.status == "filled"
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

        assert result[0].retcode == expected_retcode
        if expected_retcode is None:
            assert result[0].status == "malformed"
        else:
            assert result[0].status == "rejected"

    def test_update_sltp_captures_send_errors_in_receipt(self) -> None:
        """Test live SL/TP updates normalize send failures to failed receipts."""
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
        client.order_send.side_effect = RuntimeError("sltp failed")

        result = update_sltp_for_open_positions(client, tickets=[1], stop_loss=1.1)

        assert result[0].status == "failed"
        assert "sltp failed" in str(result[0].comment)

    def test_update_sltp_captures_mt5_constant_access_failures_in_receipt(
        self,
    ) -> None:
        """Test request preparation failures (e.g. mt5 constant access) fail closed."""

        class _RaisingMt5Client(MagicMock):
            @property
            def mt5(self) -> Any:  # noqa: ANN401
                message = "mt5 constants unavailable"
                raise RuntimeError(message)

        client = _RaisingMt5Client()
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

        result = update_sltp_for_open_positions(client, tickets=[1], stop_loss=1.1)

        assert result[0].status == "failed"
        assert "mt5 constants unavailable" in str(result[0].comment)
        client.order_send.assert_not_called()

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
        execution = OrderExecutionResult(
            status="dry_run",
            symbol="EURUSD",
            order_side="BUY",
            requested_volume=0.1,
            filled_volume=None,
            request_price=None,
            filled_price=None,
            order_ticket=None,
            deal_ticket=None,
            position_id=None,
            magic=None,
            retcode=None,
            comment=None,
            dry_run=True,
            request={"action": 20},
            response=None,
        )
        _assert_close(margin["buy_volume"], 0.1)
        _assert_close(limits["entry"], 1.0)
        assert execution.status == "dry_run"


class TestExecutionNormalizationInternals:
    """Tests for execution receipt normalization helpers."""

    def test_plain_value_serializes_nested_structures(self) -> None:
        """Test diagnostic serialization handles nested broker payloads."""
        row = SimpleNamespace(_asdict=lambda: {"retcode": 1})
        value = {
            "when": datetime(2024, 1, 2, tzinfo=UTC),
            "rows": [row],
            "frame": pd.DataFrame([{"x": 1}]),
            "items": ({"y": 2},),
        }

        plain = cast("dict[str, object]", _plain_value(value))

        assert plain["when"] == "2024-01-02T00:00:00+00:00"
        assert plain["rows"] == [{"retcode": 1}]
        assert plain["frame"] == [{"x": 1}]
        assert plain["items"] == [{"y": 2}]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_plain_value_normalizes_non_finite_floats_to_none(
        self, value: float
    ) -> None:
        """Test non-finite floats become JSON-safe ``None``."""
        assert _plain_value(value) is None

    def test_snapshot_from_value_with_empty_fields_returns_full_row(self) -> None:
        """Test empty field filters return the full normalized row."""
        row = _snapshot_from_value(pd.DataFrame([{"a": 1, "b": 2}]), ())
        assert row == {"a": 1, "b": 2}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1, 1),
            ("123", 123),
            (0, None),
            ("0", None),
            (-1, None),
            (True, None),
            (None, None),
            ("", None),
            ("invalid", None),
            (1.5, None),
        ],
        ids=[
            "positive-int",
            "digit-string",
            "zero",
            "zero-string",
            "negative",
            "bool",
            "none",
            "empty-string",
            "invalid-string",
            "float",
        ],
    )
    def test_optional_positive_int(self, value: object, expected: int | None) -> None:
        """Test identifier normalization returns positive integers only."""
        assert _optional_positive_int(value) == expected

    def test_execution_receipt_keeps_positive_broker_identifiers(self) -> None:
        """Test filled responses expose positive order/deal/position tickets."""
        client = _mock_trade_client()

        result = _execution_receipt(
            mt5=client.mt5,
            symbol="EURUSD",
            order_side="BUY",
            request={"symbol": "EURUSD", "volume": 0.1},
            response={"retcode": 10009, "order": 11, "deal": 22, "position": 33},
        )

        assert result.status == "filled"
        assert result.order_ticket == 11
        assert result.deal_ticket == 22
        assert result.position_id == 33

    def test_execution_receipt_normalizes_zero_identifiers_to_none(self) -> None:
        """Test rejected responses with zero-sentinel identifiers expose None."""
        client = _mock_trade_client()

        result = _execution_receipt(
            mt5=client.mt5,
            symbol="EURUSD",
            order_side="BUY",
            request={"symbol": "EURUSD", "volume": 0.1},
            response={"retcode": 10013, "order": 0, "deal": 0, "position": 0},
        )

        assert result.status == "rejected"
        assert result.order_ticket is None
        assert result.deal_ticket is None
        assert result.position_id is None

    def test_execution_receipt_prefers_request_position_over_zero_sentinel(
        self,
    ) -> None:
        """Test close receipts keep the requested position ticket over zero."""
        client = _mock_trade_client()

        result = _execution_receipt(
            mt5=client.mt5,
            symbol="EURUSD",
            order_side="SELL",
            request={"symbol": "EURUSD", "volume": 0.1, "position": 123},
            response={"retcode": 10009, "order": 11, "deal": 22, "position": 0},
        )

        assert result.position_id == 123

    def test_execution_receipt_normalizes_malformed_identifiers_to_none(self) -> None:
        """Test negative and malformed identifiers are dropped from receipts."""
        client = _mock_trade_client()

        result = _execution_receipt(
            mt5=client.mt5,
            symbol="EURUSD",
            order_side="BUY",
            request={"symbol": "EURUSD", "volume": 0.1},
            response={
                "retcode": 10009,
                "order": -5,
                "deal": "invalid",
                "position": True,
            },
        )

        assert result.order_ticket is None
        assert result.deal_ticket is None
        assert result.position_id is None

    def test_response_mapping_handles_empty_and_dict_responses(self) -> None:
        """Test response normalization covers empty frames and plain mappings."""
        assert _response_mapping(pd.DataFrame()) == {}
        assert _response_mapping({"retcode": 10009}) == {"retcode": 10009}


class TestFetchLatestClosedRatesIndexed:
    """Tests for the public client-based closed-rate helper."""

    def test_returns_closed_rates_with_utc_index(self) -> None:
        """The last forming bar is removed and time becomes the index."""
        client = _mock_trade_client()
        client.copy_rates_from_pos.return_value = pd.DataFrame(
            {"time": [1_704_067_200, 1_704_067_260, 1_704_067_320], "close": [1, 2, 3]},
        )

        result = fetch_latest_closed_rates_indexed(
            client, symbol="EURUSD", granularity="M1", count=2
        )

        assert result.index.name == "time"
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None
        assert result["close"].tolist() == [1, 2]

    @pytest.mark.parametrize(
        ("client", "count", "error"),
        [
            (_mock_trade_client(), 0, "count must be positive"),
            (object(), 1, "cannot fetch rate data"),
        ],
    )
    def test_rejects_invalid_request(
        self, client: object, count: int, error: str
    ) -> None:
        """Invalid counts and client capabilities fail deterministically."""
        with pytest.raises((ValueError, Mt5OperationError), match=error):
            fetch_latest_closed_rates_indexed(
                cast("_Mt5ClientProtocol", client),
                symbol="EURUSD",
                granularity="M1",
                count=count,
            )

    @pytest.mark.parametrize(
        ("frame", "error"),
        [
            ("not-a-frame", "malformed rate data"),
            (pd.DataFrame({"time": [1]}), "Rate data is empty"),
            (pd.DataFrame({"close": [1, 2]}), "missing a time column"),
        ],
    )
    def test_rejects_malformed_rate_data(self, frame: object, error: str) -> None:
        """Malformed, empty, and time-less payloads are rejected."""
        client = _mock_trade_client()
        client.copy_rates_from_pos.return_value = frame

        with pytest.raises((TypeError, ValueError), match=error):
            fetch_latest_closed_rates_indexed(
                client, symbol="EURUSD", granularity="M1", count=1
            )

    def test_accepts_datetime_index_without_a_time_column(self) -> None:
        """Datetime indexes are normalized to the canonical time column."""
        client = _mock_trade_client()
        client.copy_rates_from_pos.return_value = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
        )

        result = fetch_latest_closed_rates_indexed(
            client, symbol="EURUSD", granularity="M1", count=2
        )

        assert result["close"].tolist() == [1, 2]

    def test_accepts_named_time_index(self) -> None:
        """A named time index remains a time column after normalization."""
        client = _mock_trade_client()
        index = pd.date_range("2024-01-01", periods=3, tz="UTC").rename("time")
        client.copy_rates_from_pos.return_value = pd.DataFrame(
            {"close": [1, 2, 3]}, index=index
        )

        result = fetch_latest_closed_rates_indexed(
            client, symbol="EURUSD", granularity="M1", count=2
        )

        assert result["close"].tolist() == [1, 2]

    @pytest.mark.parametrize(
        ("times", "error"),
        [
            ([{"bad": 1}, {"bad": 1}, {"bad": 1}], "invalid or unparseable"),
            ([None, None, None], "contains missing"),
        ],
    )
    def test_rejects_invalid_or_missing_rate_times(
        self, times: list[object], error: str
    ) -> None:
        """Timestamp normalization rejects invalid and missing values."""
        client = _mock_trade_client()
        client.copy_rates_from_pos.return_value = pd.DataFrame(
            {"time": times, "close": [1, 2, 3]},
        )

        with pytest.raises(ValueError, match=error):
            fetch_latest_closed_rates_indexed(
                client, symbol="EURUSD", granularity="M1", count=2
            )


class TestExecutionBrokerFailures:
    """Broker failures after input validation always produce receipts."""

    def test_close_continues_after_filling_mode_failure(self) -> None:
        """One failed close receipt does not prevent later positions being handled."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [
                {"ticket": 1, "symbol": "FAIL", "type": 0, "volume": 0.1},
                {"ticket": 2, "symbol": "EURUSD", "type": 0, "volume": 0.1},
            ],
        )

        def _symbol_info(*, symbol: str) -> dict[str, object]:
            if symbol == "FAIL":
                msg = "metadata unavailable"
                raise RuntimeError(msg)
            return {"visible": True, "filling_mode": 2}

        client.symbol_info_as_dict.side_effect = _symbol_info
        client.symbol_info_tick_as_dict.return_value = {"ask": 1.2, "bid": 1.1}
        client.order_send.return_value = {"retcode": 10009}

        results = close_open_positions(client)

        assert [result.status for result in results] == ["failed", "filled"]
        assert results[0].position_id == 1

    def test_sltp_symbol_selection_failure_is_a_receipt(self) -> None:
        """SL/TP selection failures preserve the known position context."""
        client = _mock_trade_client()
        client.positions_get_as_df.return_value = pd.DataFrame(
            [{"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 0.1}],
        )
        client.symbol_info_as_dict.return_value = {"visible": False}
        client.symbol_select.return_value = False

        results = update_sltp_for_open_positions(client, tickets=[1], stop_loss=1.0)

        assert results[0].status == "failed"
        assert results[0].position_id == 1


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
            side_effect=[Mt5OperationError("tick unavailable"), 30.0],
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
            Mt5OperationError("trading error"),
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
            side_effect=[Mt5OperationError("err1"), Mt5OperationError("err2")],
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
            side_effect=[Mt5OperationError("tick unavailable"), 30.0],
        )

        total = calculate_positions_margin_safe(client, symbols=["EURUSD", "GBPUSD"])

        _assert_close(total, 30.0)

    def test_all_symbols_fail_returns_zero(self, mocker: MockerFixture) -> None:
        """Returns 0.0 when every symbol raises."""
        client = _mock_trade_client()
        mocker.patch(
            "mt5cli.trading.calculate_positions_margin",
            side_effect=[Mt5OperationError("err1"), Mt5RuntimeError("err2")],
        )

        total = calculate_positions_margin_safe(client, symbols=["EURUSD", "GBPUSD"])

        _assert_close(total, 0.0)

    def test_empty_symbols_returns_zero(self) -> None:
        """Returns 0.0 for an empty symbol list."""
        client = _mock_trade_client()
        total = calculate_positions_margin_safe(client, symbols=[])
        _assert_close(total, 0.0)


_CLOCK_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_CLOCK_NOW_EPOCH = _CLOCK_NOW.timestamp()
_UTC_PLUS_2 = 7200.0
_UTC_PLUS_3 = 10800.0


def _freeze_clock(
    mocker: MockerFixture,
    now: datetime = _CLOCK_NOW,
) -> tuple[MagicMock, MagicMock]:
    mock_dt = mocker.patch("mt5cli.trading.datetime")
    mock_dt.now.return_value = now
    mock_dt.fromtimestamp.side_effect = datetime.fromtimestamp
    mock_sleep = mocker.patch("mt5cli.trading.sleep")
    return mock_dt, mock_sleep


def _live_tick(
    server_epoch: float,
    *,
    symbol: str = "SING30",
    bid: float | None = 1.1,
    ask: float | None = 1.2,
    last: float | None = 0.0,
    volume: int | None = 5,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "time": int(server_epoch),
        "time_msc": int(server_epoch * 1000),
        "bid": bid,
        "ask": ask,
        "last": last,
        "volume": volume,
    }


def _clock_client(live_ticks: list[dict[str, object]]) -> MagicMock:
    """Build an MT5 client mock returning ``live_ticks`` in call order.

    ``copy_ticks_range`` is intentionally left as a bare, unconfigured mock:
    the redesigned calibration never calls it (that was the root cause of
    the dceoy/mteor#428 ``no_matching_event`` failure), so any test that
    still exercised it would be silently exercising the wrong contract.
    """
    client = MagicMock()
    client.symbol_info_tick_as_dict.side_effect = live_ticks
    return client


class TestTickClockNormalizer:
    """Tests for TickClockNormalizer host-clock-based calibration.

    ``symbol_info_tick()`` and ``copy_ticks_range()`` share the same
    server-labeled epoch contract (see dceoy/mteor#428): a query window built
    from the host clock can miss the live event entirely by exactly the
    broker's offset, which is why v1.3.2 still failed calibration with
    ``no_matching_event``. These tests model the actual contract instead:
    calibration evidence comes only from live ticks, each stamped with this
    process's own ``datetime.now(UTC)`` at receipt time, so
    ``client.copy_ticks_range`` is never configured with data and every test
    is free to assert it was never called.
    """

    def test_active_utc_plus_three_server_calibrates(
        self,
        mocker: MockerFixture,
    ) -> None:
        """An OANDA-style UTC+3 server calibrates from advancing live ticks alone."""
        _, mock_sleep = _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])

        calibration = normalizer.calibrate()

        assert calibration.status == "calibrated"
        _assert_close(calibration.offset_seconds, _UTC_PLUS_3)
        assert calibration.sample_count == 2
        assert calibration.evidence_symbols == ("SING30",)
        _assert_close(calibration.calibrated_at, _CLOCK_NOW_EPOCH)
        assert calibration.last_sample_symbol == "SING30"
        _assert_close(calibration.last_sample_raw_offset_seconds, _UTC_PLUS_3 - 1.0)
        assert mock_sleep.call_count == 2
        client.copy_ticks_range.assert_not_called()

    def test_active_utc_plus_two_server_calibrates(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A UTC+2 server (pre-DST OANDA) calibrates just as reliably."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_2),
            _live_tick(event_1 + _UTC_PLUS_2),
            _live_tick(event_2 + _UTC_PLUS_2),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])

        calibration = normalizer.calibrate()

        assert calibration.status == "calibrated"
        _assert_close(calibration.offset_seconds, _UTC_PLUS_2)
        assert calibration.sample_count == 2
        client.copy_ticks_range.assert_not_called()

    def test_utc_native_broker_calibrates_zero_offset(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A true-UTC broker calibrates to exactly 0.0, not falsy-skipped."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0),
            _live_tick(event_1),
            _live_tick(event_2),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])

        calibration = normalizer.calibrate()

        assert calibration.status == "calibrated"
        _assert_close(calibration.offset_seconds, 0.0)
        assert calibration.sample_count == 2

    def test_network_processing_delay_still_resolves_offset(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A few seconds of receipt latency still round to the true offset."""
        _freeze_clock(mocker)
        latency = 2.0
        event_0 = _CLOCK_NOW_EPOCH - 3 - latency
        event_1 = _CLOCK_NOW_EPOCH - 2 - latency
        event_2 = _CLOCK_NOW_EPOCH - 1 - latency
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])

        calibration = normalizer.calibrate()

        assert calibration.status == "calibrated"
        _assert_close(calibration.offset_seconds, _UTC_PLUS_3)
        assert calibration.sample_count == 2

    def test_stale_single_symbol_does_not_calibrate(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A symbol whose tick never changes across polls yields no evidence."""
        _freeze_clock(mocker)
        stale_event = _CLOCK_NOW_EPOCH - 2 * 24 * 3600
        client = _clock_client([_live_tick(stale_event) for _ in range(3)])
        normalizer = TickClockNormalizer(client, ["SING30"])

        calibration = normalizer.calibrate()

        assert calibration.status == "not_advancing"
        assert calibration.offset_seconds is None
        assert calibration.sample_count == 0
        assert calibration.calibrated_at is None
        assert calibration.last_sample_symbol is None
        client.copy_ticks_range.assert_not_called()

    def test_closed_symbol_does_not_block_active_symbol(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A frozen symbol cannot prevent another configured symbol calibrating."""
        _freeze_clock(mocker)
        closed_event = _CLOCK_NOW_EPOCH - 2 * 24 * 3600
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(closed_event, symbol="CLOSED"),
            _live_tick(closed_event, symbol="CLOSED"),
            _live_tick(closed_event, symbol="CLOSED"),
            _live_tick(event_0 + _UTC_PLUS_3, symbol="ACTIVE"),
            _live_tick(event_1 + _UTC_PLUS_3, symbol="ACTIVE"),
            _live_tick(event_2 + _UTC_PLUS_3, symbol="ACTIVE"),
        ])
        normalizer = TickClockNormalizer(client, ["CLOSED", "ACTIVE"])

        calibration = normalizer.calibrate()

        assert calibration.status == "calibrated"
        _assert_close(calibration.offset_seconds, _UTC_PLUS_3)
        assert calibration.evidence_symbols == ("ACTIVE",)
        assert calibration.sample_count == 2

    def test_contradictory_observations_fail_closed(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Distinct advancing events that disagree are never averaged or accepted."""
        _freeze_clock(mocker)
        event_a0 = _CLOCK_NOW_EPOCH - 4
        event_a1 = _CLOCK_NOW_EPOCH - 3
        event_b0 = _CLOCK_NOW_EPOCH - 2
        event_b1 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_a0 + _UTC_PLUS_2, symbol="A"),
            _live_tick(event_a1 + _UTC_PLUS_2, symbol="A"),
            _live_tick(event_b0 + _UTC_PLUS_3, symbol="B"),
            _live_tick(event_b1 + _UTC_PLUS_3, symbol="B"),
        ])
        normalizer = TickClockNormalizer(client, ["A", "B"], samples_per_symbol=2)

        calibration = normalizer.calibrate()

        assert calibration.status == "offset_disagreement"
        assert calibration.offset_seconds is None
        assert calibration.sample_count == 2

    def test_unstable_offset_between_buckets_is_rejected(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A raw offset far from any 30-minute bucket is never accepted."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 2
        event_1 = event_0 + 600.0
        client = _clock_client([_live_tick(event_0), _live_tick(event_1)])
        normalizer = TickClockNormalizer(client, ["SING30"], samples_per_symbol=2)

        calibration = normalizer.calibrate()

        assert calibration.status == "unstable_offset"
        assert calibration.offset_seconds is None

    def test_implausible_offset_is_rejected(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A rounded offset outside the realistic timezone range is rejected."""
        _freeze_clock(mocker)
        huge_offset = 16 * 3600.0
        event_0 = _CLOCK_NOW_EPOCH - 2
        event_1 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0 + huge_offset),
            _live_tick(event_1 + huge_offset),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"], samples_per_symbol=2)

        calibration = normalizer.calibrate()

        assert calibration.status == "implausible_offset"
        assert calibration.offset_seconds is None

    def test_single_matched_sample_is_insufficient(
        self,
        mocker: MockerFixture,
    ) -> None:
        """One accepted sample alone never satisfies min_agreeing_samples."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 2
        event_1 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"], samples_per_symbol=2)

        calibration = normalizer.calibrate()

        assert calibration.status == "insufficient_agreement"
        assert calibration.offset_seconds is None
        assert calibration.sample_count == 1
        assert calibration.calibrated_at is None

    def test_single_poll_from_cold_start_has_no_baseline(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A single observation is insufficient: there is no prior tick to compare."""
        _freeze_clock(mocker)
        client = _clock_client([_live_tick(_CLOCK_NOW_EPOCH - 1 + _UTC_PLUS_3)])
        normalizer = TickClockNormalizer(client, ["SING30"], samples_per_symbol=1)

        calibration = normalizer.calibrate()

        assert calibration.status == "not_advancing"
        assert calibration.offset_seconds is None
        assert calibration.sample_count == 0

    def test_no_live_tick_yields_no_live_tick_status(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A tick with no usable timestamp aborts before any comparison."""
        _freeze_clock(mocker)
        client = _clock_client([
            {"symbol": "SING30", "time": None},
            {"symbol": "SING30"},
        ])
        normalizer = TickClockNormalizer(client, ["SING30"], samples_per_symbol=2)

        calibration = normalizer.calibrate()

        assert calibration.status == "no_live_tick"
        client.copy_ticks_range.assert_not_called()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"bid": None, "ask": None, "last": None},
            {"bid": 0.0, "ask": None, "last": None},
            {"bid": "1.1", "ask": None, "last": None},
            {"bid": float("nan"), "ask": None, "last": None},
        ],
        ids=["all-none", "zero-price", "string-price", "nan-price"],
    )
    def test_weak_price_evidence_is_rejected(
        self,
        mocker: MockerFixture,
        overrides: dict[str, object],
    ) -> None:
        """A changed epoch alone is not evidence without a usable price."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 2
        event_1 = _CLOCK_NOW_EPOCH - 1
        weak_tick = {**_live_tick(event_1), **overrides}
        client = _clock_client([_live_tick(event_0), weak_tick])
        normalizer = TickClockNormalizer(client, ["SING30"], samples_per_symbol=2)

        calibration = normalizer.calibrate()

        assert calibration.status == "not_advancing"
        assert calibration.offset_seconds is None

    def test_volume_changes_do_not_affect_offset_calibration(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Volume is not part of event identity or offset math."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3, volume=1),
            _live_tick(event_1 + _UTC_PLUS_3, volume=99),
            _live_tick(event_2 + _UTC_PLUS_3, volume=5),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])

        calibration = normalizer.calibrate()

        assert calibration.status == "calibrated"
        _assert_close(calibration.offset_seconds, _UTC_PLUS_3)

    def test_time_msc_datetime_and_time_only_fallbacks(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Live ticks may carry a datetime time_msc or only a time column."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 2
        event_1 = _CLOCK_NOW_EPOCH - 1
        tick_with_datetime_msc: dict[str, object] = {
            "symbol": "SING30",
            "time_msc": pd.Timestamp(event_0, unit="s", tz=UTC),
            "bid": 1.1,
            "ask": 1.2,
            "last": 0.0,
            "volume": 5,
        }
        tick_with_time_only: dict[str, object] = {
            "symbol": "SING30",
            "time": int(event_1),
            "bid": 1.1,
            "ask": 1.2,
            "last": 0.0,
            "volume": 5,
        }
        client = _clock_client([tick_with_datetime_msc, tick_with_time_only])
        normalizer = TickClockNormalizer(
            client,
            ["SING30"],
            samples_per_symbol=2,
            min_agreeing_samples=1,
        )

        calibration = normalizer.calibrate()

        assert calibration.status == "calibrated"
        _assert_close(calibration.offset_seconds, 0.0)

    def test_zero_epoch_datetime_time_msc_falls_back_to_time(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A datetime time_msc at the Unix epoch is not usable; time wins instead."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 2
        event_1 = _CLOCK_NOW_EPOCH - 1

        def _tick_with_epoch_zero_msc(event: float) -> dict[str, object]:
            return {
                "symbol": "SING30",
                "time_msc": pd.Timestamp(0, unit="s", tz=UTC),
                "time": int(event),
                "bid": 1.1,
                "ask": 1.2,
                "last": 0.0,
                "volume": 5,
            }

        client = _clock_client([
            _tick_with_epoch_zero_msc(event_0),
            _tick_with_epoch_zero_msc(event_1),
        ])
        normalizer = TickClockNormalizer(
            client,
            ["SING30"],
            samples_per_symbol=2,
            min_agreeing_samples=1,
        )

        calibration = normalizer.calibrate()

        assert calibration.status == "calibrated"
        _assert_close(calibration.offset_seconds, 0.0)

    def test_normalized_snapshot_reports_correct_utc_time(
        self,
        mocker: MockerFixture,
    ) -> None:
        """get_normalized_tick_snapshot returns a validated UTC instant."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])
        normalizer.calibrate()

        snapshot = normalizer.get_normalized_tick_snapshot("SING30")

        assert snapshot["symbol"] == "SING30"
        assert snapshot["clock_status"] == "calibrated"
        assert snapshot["raw_time"] == int(event_2 + _UTC_PLUS_3)
        assert snapshot["time_utc"] == datetime.fromtimestamp(event_2, tz=UTC)
        _assert_close(snapshot["server_clock_offset_seconds"], _UTC_PLUS_3)
        _assert_close(snapshot["bid"], 1.1)
        _assert_close(snapshot["ask"], 1.2)
        client.copy_ticks_range.assert_not_called()

    def test_offset_increase_from_utc2_to_utc3_triggers_recalibration(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A DST-style offset increase is caught by the future-skew guard."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_2),
            _live_tick(event_1 + _UTC_PLUS_2),
            _live_tick(event_2 + _UTC_PLUS_2),
            # Raw snapshot fetch: looks ~1h in the future under the stale UTC+2
            # offset, since the broker has since moved to UTC+3.
            _live_tick(event_2 + _UTC_PLUS_3),
            # Recalibration polls, all now labeled UTC+3:
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            # Refetched raw snapshot, now normalized correctly under UTC+3:
            _live_tick(event_2 + _UTC_PLUS_3),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])
        first = normalizer.calibrate()
        _assert_close(first.offset_seconds, _UTC_PLUS_2)

        snapshot = normalizer.get_normalized_tick_snapshot("SING30")

        assert snapshot["clock_status"] == "calibrated"
        _assert_close(snapshot["server_clock_offset_seconds"], _UTC_PLUS_3)
        assert snapshot["time_utc"] == datetime.fromtimestamp(event_2, tz=UTC)
        calibration = normalizer.calibration
        assert calibration is not None
        _assert_close(calibration.offset_seconds, _UTC_PLUS_3)
        assert calibration.sample_count == 2

    def test_offset_decrease_from_utc3_to_utc2_caught_by_periodic_revalidation(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A DST-style offset decrease is caught before the cache would expire.

        A future-skew check alone never catches this: applying the stale
        larger offset makes a fresh tick look stale, not future, which is
        indistinguishable from ordinary quiet-market staleness.
        """
        mock_dt, _ = _freeze_clock(mocker)
        later = datetime.fromtimestamp(_CLOCK_NOW_EPOCH + 61.0, tz=UTC)
        later_epoch = later.timestamp()
        raw_event = later_epoch - 1
        client = _clock_client([
            _live_tick(_CLOCK_NOW_EPOCH - 3 + _UTC_PLUS_3),
            _live_tick(_CLOCK_NOW_EPOCH - 2 + _UTC_PLUS_3),
            _live_tick(_CLOCK_NOW_EPOCH - 1 + _UTC_PLUS_3),
            # Periodic revalidation sample: the broker has since fallen back
            # to UTC+2, so this fresh event's label is numerically *smaller*
            # than the last observation taken under the old UTC+3 offset.
            _live_tick(later_epoch - 2 + _UTC_PLUS_2),
            # Full recalibration confirms UTC+2:
            _live_tick(later_epoch - 4 + _UTC_PLUS_2),
            _live_tick(later_epoch - 3 + _UTC_PLUS_2),
            _live_tick(later_epoch - 2 + _UTC_PLUS_2),
            # Raw snapshot fetch, normalized correctly under the new offset:
            _live_tick(raw_event + _UTC_PLUS_2),
        ])
        normalizer = TickClockNormalizer(
            client,
            ["SING30"],
            revalidation_interval_seconds=60.0,
        )
        first = normalizer.calibrate()
        _assert_close(first.offset_seconds, _UTC_PLUS_3)

        mock_dt.now.return_value = later
        snapshot = normalizer.get_normalized_tick_snapshot("SING30")

        assert snapshot["clock_status"] == "calibrated"
        _assert_close(snapshot["server_clock_offset_seconds"], _UTC_PLUS_2)
        assert snapshot["time_utc"] == datetime.fromtimestamp(raw_event, tz=UTC)
        calibration = normalizer.calibration
        assert calibration is not None
        _assert_close(calibration.offset_seconds, _UTC_PLUS_2)

    def test_revalidation_tolerates_tick_observed_well_after_its_event(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A changed epoch is evidence even when detected long after it occurred.

        An infrequently-traded symbol checked only once per
        ``revalidation_interval_seconds`` can easily surface a fresh event
        that itself happened well before it was polled: a changed epoch only
        proves the event occurred sometime since the previous poll, not that
        it occurred within the fixed residual tolerance of *this* poll. The
        extra slack is bounded by the elapsed time since the previous poll,
        so it cannot be mistaken for evidence of the wrong offset bucket.
        """
        mock_dt, _ = _freeze_clock(mocker)
        later = datetime.fromtimestamp(_CLOCK_NOW_EPOCH + 61.0, tz=UTC)
        later_epoch = later.timestamp()
        raw_event = later_epoch - 1
        client = _clock_client([
            _live_tick(_CLOCK_NOW_EPOCH - 3 + _UTC_PLUS_3),
            _live_tick(_CLOCK_NOW_EPOCH - 2 + _UTC_PLUS_3),
            _live_tick(_CLOCK_NOW_EPOCH - 1 + _UTC_PLUS_3),
            # Periodic revalidation sample: this fresh post-DST event
            # actually occurred 30 seconds before it was polled, so its raw
            # offset residual from the UTC+2 bucket is 30 seconds -- well
            # past the fixed 5-second tolerance on its own.
            _live_tick(later_epoch - 30.0 + _UTC_PLUS_2),
            # Full recalibration confirms UTC+2:
            _live_tick(later_epoch - 4 + _UTC_PLUS_2),
            _live_tick(later_epoch - 3 + _UTC_PLUS_2),
            _live_tick(later_epoch - 2 + _UTC_PLUS_2),
            # Raw snapshot fetch, normalized correctly under the new offset:
            _live_tick(raw_event + _UTC_PLUS_2),
        ])
        normalizer = TickClockNormalizer(
            client,
            ["SING30"],
            revalidation_interval_seconds=60.0,
        )
        first = normalizer.calibrate()
        _assert_close(first.offset_seconds, _UTC_PLUS_3)

        mock_dt.now.return_value = later
        snapshot = normalizer.get_normalized_tick_snapshot("SING30")

        assert snapshot["clock_status"] == "calibrated"
        _assert_close(snapshot["server_clock_offset_seconds"], _UTC_PLUS_2)
        calibration = normalizer.calibration
        assert calibration is not None
        _assert_close(calibration.offset_seconds, _UTC_PLUS_2)

    def test_revalidation_without_any_baseline_retries_on_the_next_call(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A wholly baseline-less revalidation round does not consume the interval.

        When no configured symbol has a prior observation to compare
        against, the round cannot confirm or refute the cached offset
        either way, so it must not count as a completed opportunistic check:
        the very next call retries immediately instead of waiting a full
        ``revalidation_interval_seconds`` for the first real comparison.
        """
        mock_dt, _ = _freeze_clock(mocker)
        later = datetime.fromtimestamp(_CLOCK_NOW_EPOCH + 61.0, tz=UTC)
        later_epoch = later.timestamp()
        client = _clock_client([
            _live_tick(_CLOCK_NOW_EPOCH - 3 + _UTC_PLUS_3, symbol="A"),
            _live_tick(_CLOCK_NOW_EPOCH - 2 + _UTC_PLUS_3, symbol="A"),
            _live_tick(_CLOCK_NOW_EPOCH - 1 + _UTC_PLUS_3, symbol="A"),
            # First call at `later`: "B" has no baseline yet, so this
            # revalidation round is wholly inconclusive.
            _live_tick(later_epoch - 5.0, symbol="B"),
            _live_tick(later_epoch - 5.0, symbol="B"),  # raw snapshot fetch
            # Second call, same clock reading: since the round above did not
            # consume the interval, revalidation retries immediately and now
            # confirms the cached offset from B's freshly established
            # baseline.
            _live_tick(later_epoch - 4.0 + _UTC_PLUS_3, symbol="B"),
            _live_tick(later_epoch - 4.0 + _UTC_PLUS_3, symbol="B"),  # raw snapshot
        ])
        normalizer = TickClockNormalizer(client, revalidation_interval_seconds=60.0)
        first = normalizer.calibrate(["A"])
        assert first.status == "calibrated"

        mock_dt.now.return_value = later
        normalizer.get_normalized_tick_snapshot("B")
        normalizer.get_normalized_tick_snapshot("B")

        assert [
            call.kwargs["symbol"]
            for call in client.symbol_info_tick_as_dict.call_args_list
        ] == ["A", "A", "A", "B", "B", "B", "B"]
        calibration = normalizer.calibration
        assert calibration is not None
        _assert_close(calibration.offset_seconds, _UTC_PLUS_3)

    def test_revalidation_uses_active_configured_symbol_when_one_is_closed(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A closed symbol cannot hide fresh connection-wide evidence."""
        mock_dt, _ = _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        closed_event = _CLOCK_NOW_EPOCH - 2 * 24 * 3600
        later = datetime.fromtimestamp(_CLOCK_NOW_EPOCH + 61.0, tz=UTC)
        later_event = later.timestamp() - 1
        client = _clock_client([
            _live_tick(closed_event, symbol="CLOSED"),
            _live_tick(closed_event, symbol="CLOSED"),
            _live_tick(closed_event, symbol="CLOSED"),
            _live_tick(event_0 + _UTC_PLUS_3, symbol="ACTIVE"),
            _live_tick(event_1 + _UTC_PLUS_3, symbol="ACTIVE"),
            _live_tick(event_2 + _UTC_PLUS_3, symbol="ACTIVE"),
            _live_tick(closed_event, symbol="CLOSED"),  # revalidation: skipped
            _live_tick(later_event + _UTC_PLUS_3, symbol="ACTIVE"),  # confirms cache
            _live_tick(later_event + _UTC_PLUS_3, symbol="ACTIVE"),  # raw snapshot
        ])
        normalizer = TickClockNormalizer(
            client,
            ["CLOSED", "ACTIVE"],
            revalidation_interval_seconds=60.0,
        )
        first = normalizer.calibrate()
        assert first.status == "calibrated"
        _assert_close(first.offset_seconds, _UTC_PLUS_3)
        assert first.evidence_symbols == ("ACTIVE",)

        mock_dt.now.return_value = later
        snapshot = normalizer.get_normalized_tick_snapshot("ACTIVE")

        assert snapshot["clock_status"] == "calibrated"
        assert snapshot["time_utc"] == datetime.fromtimestamp(later_event, tz=UTC)
        _assert_close(snapshot["server_clock_offset_seconds"], _UTC_PLUS_3)
        assert [
            call.kwargs["symbol"]
            for call in client.symbol_info_tick_as_dict.call_args_list
        ] == [
            "CLOSED",
            "CLOSED",
            "CLOSED",
            "ACTIVE",
            "ACTIVE",
            "ACTIVE",
            "CLOSED",
            "ACTIVE",
            "ACTIVE",
        ]

    def test_revalidation_skips_unusable_and_never_before_seen_symbols(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Revalidation tolerates a broken fetch and a symbol with no baseline."""
        mock_dt, _ = _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        later = datetime.fromtimestamp(_CLOCK_NOW_EPOCH + 61.0, tz=UTC)
        later_event = later.timestamp() - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3, symbol="SING30"),
            _live_tick(event_1 + _UTC_PLUS_3, symbol="SING30"),
            _live_tick(event_2 + _UTC_PLUS_3, symbol="SING30"),
            {"symbol": "NOFETCH"},  # revalidation: unusable fetch
            _live_tick(later_event, symbol="NEWSYM"),  # revalidation: no baseline
            _live_tick(later_event + _UTC_PLUS_3, symbol="SING30"),  # confirms cache
            _live_tick(later_event + _UTC_PLUS_3, symbol="SING30"),  # raw snapshot
        ])
        normalizer = TickClockNormalizer(
            client,
            ["NOFETCH", "NEWSYM", "SING30"],
            revalidation_interval_seconds=60.0,
        )
        first = normalizer.calibrate(["SING30"])
        assert first.status == "calibrated"

        mock_dt.now.return_value = later
        snapshot = normalizer.get_normalized_tick_snapshot("SING30")

        assert snapshot["clock_status"] == "calibrated"
        _assert_close(snapshot["server_clock_offset_seconds"], _UTC_PLUS_3)

    def test_revalidation_returns_none_when_every_candidate_is_inconclusive(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A wholly inconclusive revalidation round keeps the cached offset."""
        mock_dt, _ = _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        later = datetime.fromtimestamp(_CLOCK_NOW_EPOCH + 61.0, tz=UTC)
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),  # revalidation: identical, inconclusive
            _live_tick(event_2 + _UTC_PLUS_3),  # raw snapshot fetch
        ])
        normalizer = TickClockNormalizer(
            client,
            ["SING30"],
            revalidation_interval_seconds=60.0,
        )
        first = normalizer.calibrate()
        assert first.status == "calibrated"

        mock_dt.now.return_value = later
        snapshot = normalizer.get_normalized_tick_snapshot("SING30")

        assert snapshot["clock_status"] == "calibrated"
        _assert_close(snapshot["server_clock_offset_seconds"], _UTC_PLUS_3)
        calibration = normalizer.calibration
        assert calibration is not None
        _assert_close(calibration.calibrated_at, _CLOCK_NOW_EPOCH)

    def test_revalidation_continues_past_already_changed_candidate(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A second disagreeing candidate does not overwrite the first change."""
        mock_dt, _ = _freeze_clock(mocker)
        later = datetime.fromtimestamp(_CLOCK_NOW_EPOCH + 61.0, tz=UTC)
        later_epoch = later.timestamp()
        client = _clock_client([
            # Initial calibration, both symbols agreeing on UTC+3:
            _live_tick(_CLOCK_NOW_EPOCH - 4 + _UTC_PLUS_3, symbol="A"),
            _live_tick(_CLOCK_NOW_EPOCH - 3 + _UTC_PLUS_3, symbol="A"),
            _live_tick(_CLOCK_NOW_EPOCH - 2 + _UTC_PLUS_3, symbol="B"),
            _live_tick(_CLOCK_NOW_EPOCH - 1 + _UTC_PLUS_3, symbol="B"),
            # Revalidation: both A and B now disagree with the cached UTC+3
            # offset (broker has fallen back to UTC+2); A is seen first and
            # sets the pending change, B must not overwrite it.
            _live_tick(later_epoch - 2 + _UTC_PLUS_2, symbol="A"),
            _live_tick(later_epoch - 1 + _UTC_PLUS_2, symbol="B"),
            # Recalibration, both symbols now agreeing on UTC+2:
            _live_tick(later_epoch + _UTC_PLUS_2, symbol="A"),
            _live_tick(later_epoch + 1 + _UTC_PLUS_2, symbol="A"),
            _live_tick(later_epoch + 2 + _UTC_PLUS_2, symbol="B"),
            _live_tick(later_epoch + 3 + _UTC_PLUS_2, symbol="B"),
            # Raw snapshot fetch:
            _live_tick(later_epoch - 1 + _UTC_PLUS_2, symbol="A"),
        ])
        normalizer = TickClockNormalizer(
            client,
            ["A", "B"],
            samples_per_symbol=2,
            revalidation_interval_seconds=60.0,
        )
        first = normalizer.calibrate()
        assert first.status == "calibrated"
        _assert_close(first.offset_seconds, _UTC_PLUS_3)

        mock_dt.now.return_value = later
        snapshot = normalizer.get_normalized_tick_snapshot("A")

        assert snapshot["clock_status"] == "calibrated"
        _assert_close(snapshot["server_clock_offset_seconds"], _UTC_PLUS_2)
        calibration = normalizer.calibration
        assert calibration is not None
        _assert_close(calibration.offset_seconds, _UTC_PLUS_2)

    def test_cached_calibration_is_reused_across_symbols_and_calls(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Calibration is cached per connection, not per symbol or call."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3, bid=150.0),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])
        normalizer.calibrate()

        first = normalizer.get_normalized_tick_snapshot("SING30")
        second = normalizer.get_normalized_tick_snapshot("USDJPY")

        assert first["clock_status"] == "calibrated"
        assert second["clock_status"] == "calibrated"
        _assert_close(second["server_clock_offset_seconds"], _UTC_PLUS_3)
        assert client.symbol_info_tick_as_dict.call_count == 5

    def test_recalibrates_after_max_calibration_age(
        self,
        mocker: MockerFixture,
    ) -> None:
        """An aged calibration is recomputed, catching offset changes."""
        mock_dt, _ = _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        later = datetime.fromtimestamp(_CLOCK_NOW_EPOCH + 7 * 3600, tz=UTC)
        later_epoch = later.timestamp()
        later_event_0 = later_epoch - 3
        later_event_1 = later_epoch - 2
        later_event_2 = later_epoch - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            # Seven hours later the broker labels ticks UTC+2:
            _live_tick(later_event_0 + _UTC_PLUS_2),
            _live_tick(later_event_1 + _UTC_PLUS_2),
            _live_tick(later_event_2 + _UTC_PLUS_2),
            _live_tick(later_event_2 + _UTC_PLUS_2),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])
        first = normalizer.calibrate()
        _assert_close(first.offset_seconds, _UTC_PLUS_3)

        mock_dt.now.return_value = later
        snapshot = normalizer.get_normalized_tick_snapshot("SING30")

        assert snapshot["clock_status"] == "calibrated"
        _assert_close(snapshot["server_clock_offset_seconds"], _UTC_PLUS_2)
        assert snapshot["time_utc"] == datetime.fromtimestamp(later_event_2, tz=UTC)

    def test_persistent_future_skew_fails_closed(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A snapshot is never trusted when its UTC time stays in the future."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        future_raw = _CLOCK_NOW_EPOCH + _UTC_PLUS_3 + 10_000.0
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            _live_tick(future_raw),
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            _live_tick(future_raw),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])

        snapshot = normalizer.get_normalized_tick_snapshot("SING30")

        assert snapshot["clock_status"] == "uncalibrated"
        assert snapshot["time_utc"] is None
        assert snapshot["server_clock_offset_seconds"] is None
        calibration = normalizer.calibration
        assert calibration is not None
        assert calibration.calibrated

    def test_snapshot_without_raw_time_fails_closed_even_when_calibrated(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A calibrated clock cannot normalize a tick that lacks a time."""
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            {"symbol": "SING30", "bid": 1.1, "ask": 1.2},
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])
        normalizer.calibrate()

        snapshot = normalizer.get_normalized_tick_snapshot("SING30")

        assert snapshot["raw_time"] is None
        assert snapshot["time_utc"] is None
        assert snapshot["clock_status"] == "uncalibrated"
        calibration = normalizer.calibration
        assert calibration is not None
        assert calibration.status == "calibrated"

    def test_failed_calibration_retries_after_cooldown_elapses(
        self,
        mocker: MockerFixture,
    ) -> None:
        """A failed calibration is retried only once the retry cooldown passes."""
        mock_dt, _ = _freeze_clock(mocker)
        stale_event = _CLOCK_NOW_EPOCH - 2 * 24 * 3600
        client = _clock_client([_live_tick(stale_event) for _ in range(9)])
        normalizer = TickClockNormalizer(
            client,
            ["SING30"],
            failed_calibration_retry_seconds=30.0,
        )

        normalizer.get_normalized_tick_snapshot("SING30")
        assert client.symbol_info_tick_as_dict.call_count == 4

        # Still within the retry cooldown: no new calibration attempt, only
        # the raw snapshot fetch.
        normalizer.get_normalized_tick_snapshot("SING30")
        assert client.symbol_info_tick_as_dict.call_count == 5

        mock_dt.now.return_value = datetime.fromtimestamp(
            _CLOCK_NOW_EPOCH + 31.0,
            tz=UTC,
        )
        normalizer.get_normalized_tick_snapshot("SING30")
        assert client.symbol_info_tick_as_dict.call_count == 9

    def test_invalidate_forces_recalibration(
        self,
        mocker: MockerFixture,
    ) -> None:
        """invalidate() drops the cache so the next call recalibrates.

        A fresh full calibration must not compare its first poll against the
        previous attempt's last observation: that prior tick could be
        arbitrarily stale (the very reason `invalidate()` or a long idle gap
        forced a fresh attempt), so a changed epoch alone would not prove
        the current tick is fresh. The first poll after `invalidate()`
        therefore only establishes a new baseline, just like the very first
        cold-start calibration, and only the following polls can yield a
        sample.
        """
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        event_3 = _CLOCK_NOW_EPOCH
        event_4 = _CLOCK_NOW_EPOCH + 1
        event_5 = _CLOCK_NOW_EPOCH + 2
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            _live_tick(event_3 + _UTC_PLUS_3),
            _live_tick(event_4 + _UTC_PLUS_3),
            _live_tick(event_5 + _UTC_PLUS_3),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])
        assert normalizer.calibration is None
        normalizer.calibrate()

        normalizer.invalidate()

        assert normalizer.calibration is None
        calibration = normalizer.calibrate()
        assert calibration.status == "calibrated"
        _assert_close(calibration.offset_seconds, _UTC_PLUS_3)
        assert calibration.sample_count == 2
        assert client.symbol_info_tick_as_dict.call_count == 6

    def test_zero_sample_interval_never_sleeps(
        self,
        mocker: MockerFixture,
    ) -> None:
        """sample_interval_seconds=0 disables pacing between samples."""
        _, mock_sleep = _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 2
        event_1 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0),
            _live_tick(event_1),
        ])
        normalizer = TickClockNormalizer(
            client,
            ["SING30"],
            samples_per_symbol=2,
            sample_interval_seconds=0.0,
        )

        calibration = normalizer.calibrate()

        assert calibration.status == "insufficient_agreement"
        mock_sleep.assert_not_called()

    def test_raw_get_tick_snapshot_contract_is_preserved(
        self,
        mocker: MockerFixture,
    ) -> None:
        """get_tick_snapshot still returns the raw numeric MT5 timestamp."""
        _freeze_clock(mocker)
        raw = _CLOCK_NOW_EPOCH + _UTC_PLUS_3
        client = _clock_client([_live_tick(raw)])

        snapshot = get_tick_snapshot(client, "SING30")

        assert set(snapshot) == {"symbol", "time", "bid", "ask", "last", "volume"}
        assert snapshot["time"] == int(raw)
        client.copy_ticks_range.assert_not_called()

    def test_calibration_never_queries_copy_ticks_range(
        self,
        mocker: MockerFixture,
    ) -> None:
        """The redesigned calibration never depends on copy_ticks_range.

        This is the direct regression guard for dceoy/mteor#428: v1.3.2
        still queried ``copy_ticks_range()`` for calibration evidence and
        centered the window on the host clock, so an OANDA-style +3h server
        label always put the matching copied event outside the queried
        range. The fix removes that dependency entirely.
        """
        _freeze_clock(mocker)
        event_0 = _CLOCK_NOW_EPOCH - 3
        event_1 = _CLOCK_NOW_EPOCH - 2
        event_2 = _CLOCK_NOW_EPOCH - 1
        client = _clock_client([
            _live_tick(event_0 + _UTC_PLUS_3),
            _live_tick(event_1 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
            _live_tick(event_2 + _UTC_PLUS_3),
        ])
        normalizer = TickClockNormalizer(client, ["SING30"])

        normalizer.calibrate()
        normalizer.get_normalized_tick_snapshot("SING30")

        client.copy_ticks_range.assert_not_called()

    def test_calibrate_without_symbols_raises(self) -> None:
        """Calibration requires at least one symbol."""
        normalizer = TickClockNormalizer(MagicMock())
        with pytest.raises(ValueError, match="At least one symbol"):
            normalizer.calibrate()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"samples_per_symbol": 0},
            {"min_agreeing_samples": 0},
            {"sample_interval_seconds": -0.1},
            {"max_calibration_age_seconds": 0.0},
            {"revalidation_interval_seconds": 0.0},
            {"failed_calibration_retry_seconds": 0.0},
        ],
        ids=[
            "samples",
            "min-agreeing",
            "negative-interval",
            "max-age",
            "revalidation-interval",
            "failed-retry-interval",
        ],
    )
    def test_constructor_rejects_invalid_tuning(
        self,
        kwargs: dict[str, float],
    ) -> None:
        """Non-positive tuning parameters are rejected up front."""
        with pytest.raises(ValueError, match="must"):
            TickClockNormalizer(MagicMock(), ["SING30"], **kwargs)  # type: ignore[arg-type]

    def test_calibration_to_dict_and_aggregate_fallback(self) -> None:
        """Diagnostics serialize cleanly and the fallback reason is stable."""
        calibration = TickClockCalibration(
            status="calibrated",
            offset_seconds=_UTC_PLUS_3,
            sample_count=2,
            evidence_symbols=("SING30",),
            calibrated_at=_CLOCK_NOW_EPOCH,
            last_sample_symbol="SING30",
            last_sample_raw_time=_CLOCK_NOW_EPOCH + _UTC_PLUS_3,
            last_sample_host_time=_CLOCK_NOW_EPOCH,
            last_sample_raw_offset_seconds=_UTC_PLUS_3,
        )
        assert calibration.calibrated
        assert calibration.to_dict() == {
            "status": "calibrated",
            "offset_seconds": _UTC_PLUS_3,
            "sample_count": 2,
            "evidence_symbols": ("SING30",),
            "calibrated_at": _CLOCK_NOW_EPOCH,
            "last_sample_symbol": "SING30",
            "last_sample_raw_time": _CLOCK_NOW_EPOCH + _UTC_PLUS_3,
            "last_sample_host_time": _CLOCK_NOW_EPOCH,
            "last_sample_raw_offset_seconds": _UTC_PLUS_3,
        }
        assert _aggregate_failure_status([]) == "no_live_tick"

    def test_exported_through_trading_module_all(self) -> None:
        """The normalization API is part of the module's public surface."""
        assert "TickClockNormalizer" in trading.__all__
        assert "TickClockCalibration" in trading.__all__
        assert "NormalizedTickSnapshot" in trading.__all__
