"""Tests for mt5cli.market_data module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, call

import pandas as pd
import pytest
from pdmt5 import Mt5RuntimeError

from mt5cli.client import build_config
from mt5cli.market_data import (
    AccountSpec,
    account_info,
    collect_latest_closed_rates_by_granularity,
    collect_latest_closed_rates_for_accounts,
    collect_latest_rates_for_accounts,
    collect_latest_rates_for_accounts_with_retries,
    copy_rates_from,
    copy_rates_from_pos,
    copy_ticks_from,
    copy_ticks_range,
    fetch_latest_closed_rates,
    history_deals,
    history_orders,
    last_error,
    latest_rates,
    market_book,
    orders,
    positions,
    resolve_account_spec,
    resolve_account_specs,
    symbol_info,
    symbol_info_tick,
    symbols,
    terminal_info,
    version,
)
from mt5cli.utils import coerce_login

if TYPE_CHECKING:
    from pdmt5 import Mt5Config
    from pytest_mock import MockerFixture


class TestModuleFunctions:
    """Tests for module-level SDK wrappers."""

    @pytest.mark.parametrize(
        ("fn", "args", "method"),
        [
            (
                copy_rates_from,
                ("EURUSD", "M1", "2024-01-01", 10),
                "copy_rates_from_as_df",
            ),
            (
                copy_rates_from_pos,
                ("EURUSD", "M1", 0, 10),
                "copy_rates_from_pos_as_df",
            ),
            (
                copy_ticks_from,
                ("EURUSD", "2024-01-01", 10, "ALL"),
                "copy_ticks_from_as_df",
            ),
            (
                copy_ticks_range,
                ("EURUSD", "2024-01-01", "2024-02-01", "ALL"),
                "copy_ticks_range_as_df",
            ),
            (account_info, (), "account_info_as_df"),
            (terminal_info, (), "terminal_info_as_df"),
            (symbols, ("*USD*",), "symbols_get_as_df"),
            (symbol_info, ("EURUSD",), "symbol_info_as_df"),
            (orders, (), "orders_get_as_df"),
            (positions, (), "positions_get_as_df"),
            (history_orders, (), "history_orders_get_as_df"),
            (history_deals, (), "history_deals_get_as_df"),
            (version, (), "version_as_df"),
            (last_error, (), "last_error_as_df"),
            (symbol_info_tick, ("EURUSD",), "symbol_info_tick_as_df"),
            (market_book, ("EURUSD",), "market_book_get_as_df"),
            (latest_rates, ("EURUSD", "M1", 10), "copy_rates_from_pos_as_df"),
        ],
    )
    def test_module_functions_delegate(
        self,
        mock_client: MagicMock,
        fn: object,
        args: tuple[object, ...],
        method: str,
    ) -> None:
        """Test module-level functions call the expected client methods."""
        config = build_config(login=123)
        result = fn(*args, config=config)  # type: ignore[operator]
        assert isinstance(result, pd.DataFrame)
        getattr(mock_client, method).assert_called_once()


class TestAccountSpec:
    """Tests for account configuration helpers."""

    def test_repr_omits_password(self) -> None:
        """Test AccountSpec repr does not expose plaintext passwords."""
        spec = AccountSpec(symbols=["EURUSD"], login=123, password="secret")

        assert "secret" not in repr(spec)
        assert "password" not in repr(spec)

    @pytest.mark.parametrize(
        ("login", "expected"),
        [
            (None, None),
            (123, 123),
            ("", None),
            ("   ", None),
            ("456", 456),
        ],
    )
    def test_coerce_login(
        self,
        login: int | str | None,
        expected: int | None,
    ) -> None:
        """Test login values are normalized for account configs."""
        assert coerce_login(login) == expected

    def test_coerce_login_rejects_non_numeric_string(self) -> None:
        """Test non-numeric login strings raise ValueError."""
        with pytest.raises(ValueError, match="invalid literal"):
            coerce_login("abc")


class TestCollectLatestRatesForAccounts:
    """Tests for collect_latest_rates_for_accounts."""

    def test_merges_results_across_accounts(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        """Test rates are collected and merged for each account group."""
        mt5_data_client = mocker.patch(
            "mt5cli.client.Mt5DataClient",
            return_value=mock_client,
        )
        accounts = [
            AccountSpec(symbols=["EURUSD"], login="123"),
            AccountSpec(symbols=["GBPUSD"], login=456),
        ]

        result = collect_latest_rates_for_accounts(accounts, ["M1"], count=2)

        assert set(result) == {("EURUSD", 1), ("GBPUSD", 1)}
        assert mt5_data_client.call_count == 2
        assert mock_client.initialize_and_login_mt5.call_count == 2
        assert mock_client.shutdown.call_count == 2

    def test_builds_config_from_account_and_base(
        self,
        mock_client: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        """Test account fields override base_config, empty login falls back."""
        configs: list[object] = []

        def _record_config(*, config: object, **_: object) -> MagicMock:
            configs.append(config)
            return mock_client

        mocker.patch("mt5cli.client.Mt5DataClient", side_effect=_record_config)
        base = build_config(
            login=999, server="Base-Server", timeout=5000, password="base-pass"
        )
        accounts = [
            AccountSpec(symbols=["EURUSD"], login="", server="Acct-Server"),
        ]

        collect_latest_rates_for_accounts(accounts, ["M1"], count=1, base_config=base)

        assert len(configs) == 1
        config = cast("Mt5Config", configs[0])
        assert config.login == 999
        assert config.server == "Acct-Server"
        assert config.timeout == 5000
        assert config.password is not None
        assert config.password.get_secret_value() == "base-pass"  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        ("accounts", "timeframes", "count", "match"),
        [
            ([], ["M1"], 1, "At least one account"),
            ([AccountSpec(symbols=["EURUSD"])], [], 1, "At least one timeframe"),
            (
                [AccountSpec(symbols=[])],
                ["M1"],
                1,
                "Each account requires at least one symbol",
            ),
            (
                [AccountSpec(symbols=["EURUSD"])],
                ["M1"],
                0,
                "count must be positive",
            ),
        ],
    )
    def test_rejects_invalid_inputs(
        self,
        accounts: list[AccountSpec],
        timeframes: list[str],
        count: int,
        match: str,
    ) -> None:
        """Test input validation for account-level rate collection."""
        with pytest.raises(ValueError, match=match):
            collect_latest_rates_for_accounts(accounts, timeframes, count)

    def test_rejects_empty_symbols_before_connecting(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test all account symbols are validated before any MT5 connection."""
        mt5_data_client = mocker.patch("mt5cli.client.Mt5DataClient")
        accounts = [
            AccountSpec(symbols=["EURUSD"], login=123),
            AccountSpec(symbols=[], login=456),
        ]

        with pytest.raises(
            ValueError, match="Each account requires at least one symbol"
        ):
            collect_latest_rates_for_accounts(accounts, ["M1"], count=1)

        mt5_data_client.assert_not_called()


class TestCollectLatestRatesForAccountsWithRetries:
    """Tests for collect_latest_rates_for_accounts_with_retries."""

    def test_returns_result_on_first_success(self, mocker: MockerFixture) -> None:
        """Test no retry happens when the first attempt succeeds."""
        expected = {("EURUSD", 1): pd.DataFrame()}
        wrapped = mocker.patch(
            "mt5cli.market_data.collect_latest_rates_for_accounts",
            return_value=expected,
        )
        sleep = mocker.patch("mt5cli.retry.time.sleep")
        accounts = [AccountSpec(symbols=["EURUSD"])]

        result = collect_latest_rates_for_accounts_with_retries(
            accounts,
            ["M1"],
            count=1,
            retry_count=3,
        )

        assert result is expected
        assert wrapped.call_count == 1
        sleep.assert_not_called()

    def test_retries_then_succeeds(self, mocker: MockerFixture) -> None:
        """Test transient MT5 errors are retried with exponential backoff."""
        expected = {("EURUSD", 1): pd.DataFrame()}
        wrapped = mocker.patch(
            "mt5cli.market_data.collect_latest_rates_for_accounts",
            side_effect=[
                Mt5RuntimeError("boom"),
                Mt5RuntimeError("boom"),
                expected,
            ],
        )
        sleep = mocker.patch("mt5cli.retry.time.sleep")
        accounts = [AccountSpec(symbols=["EURUSD"])]

        result = collect_latest_rates_for_accounts_with_retries(
            accounts,
            ["M1"],
            count=1,
            retry_count=2,
            backoff_base=2,
        )

        assert result is expected
        assert wrapped.call_count == 3
        assert sleep.call_args_list == [call(2), call(4)]

    def test_reraises_after_exhausting_retries(self, mocker: MockerFixture) -> None:
        """Test the final error is re-raised once retries are exhausted."""
        wrapped = mocker.patch(
            "mt5cli.market_data.collect_latest_rates_for_accounts",
            side_effect=Mt5RuntimeError("boom"),
        )
        sleep = mocker.patch("mt5cli.retry.time.sleep")
        accounts = [AccountSpec(symbols=["EURUSD"])]

        with pytest.raises(Mt5RuntimeError, match="boom"):
            collect_latest_rates_for_accounts_with_retries(
                accounts,
                ["M1"],
                count=1,
                retry_count=2,
            )

        assert wrapped.call_count == 3
        assert sleep.call_count == 2

    def test_does_not_retry_unrelated_errors(self, mocker: MockerFixture) -> None:
        """Test non-MT5 errors propagate without retrying."""
        wrapped = mocker.patch(
            "mt5cli.market_data.collect_latest_rates_for_accounts",
            side_effect=ValueError("bad input"),
        )
        sleep = mocker.patch("mt5cli.retry.time.sleep")

        with pytest.raises(ValueError, match="bad input"):
            collect_latest_rates_for_accounts_with_retries(
                [AccountSpec(symbols=["EURUSD"])],
                ["M1"],
                count=1,
                retry_count=3,
            )

        assert wrapped.call_count == 1
        sleep.assert_not_called()


class TestCollectLatestClosedRatesForAccounts:
    """Tests for collect_latest_closed_rates_for_accounts."""

    def test_fetches_count_plus_one_and_drops_forming_bar(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test closed-bar collection requests one extra bar at start_pos=0."""
        df_rate = pd.DataFrame({"time": [1, 2, 3], "close": [1.1, 1.2, 1.3]})
        wrapped = mocker.patch(
            "mt5cli.market_data.collect_latest_rates_for_accounts_with_retries",
            return_value={("EURUSD", 1): df_rate},
        )
        accounts = [AccountSpec(symbols=["EURUSD"])]

        result = collect_latest_closed_rates_for_accounts(
            accounts,
            ["M1"],
            count=2,
            retry_count=1,
            backoff_base=3,
        )

        wrapped.assert_called_once_with(
            accounts,
            ["M1"],
            3,
            start_pos=0,
            base_config=None,
            retry_count=1,
            backoff_base=3,
        )
        pd.testing.assert_frame_equal(
            result["EURUSD", 1],
            pd.DataFrame({"time": [1, 2], "close": [1.1, 1.2]}),
        )

    @pytest.mark.parametrize(
        ("rates_frame", "kwargs"),
        [
            (pd.DataFrame({"time": [1], "close": [1.1]}), {"count": 1}),
            (pd.DataFrame(columns=["time", "close"]), {"count": 1, "start_pos": 1}),
        ],
        ids=["forming-bar-only", "empty-start-pos-nonzero"],
    )
    def test_rejects_empty_effective_frames(
        self,
        mocker: MockerFixture,
        rates_frame: pd.DataFrame,
        kwargs: dict[str, object],
    ) -> None:
        """Test empty effective frames raise after start_pos/forming-bar handling."""
        mocker.patch(
            "mt5cli.market_data.collect_latest_rates_for_accounts_with_retries",
            return_value={("EURUSD", 1): rates_frame},
        )

        with pytest.raises(ValueError, match="Rate data is empty"):
            collect_latest_closed_rates_for_accounts(
                [AccountSpec(symbols=["EURUSD"])],
                ["M1"],
                **kwargs,  # type: ignore[arg-type]
            )

    def test_skips_extra_fetch_when_start_pos_nonzero(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test start_pos > 0 fetches count bars without dropping the last row."""
        df_rate = pd.DataFrame({"time": [1, 2], "close": [1.1, 1.2]})
        wrapped = mocker.patch(
            "mt5cli.market_data.collect_latest_rates_for_accounts_with_retries",
            return_value={("EURUSD", 1): df_rate},
        )

        result = collect_latest_closed_rates_for_accounts(
            [AccountSpec(symbols=["EURUSD"])],
            ["M1"],
            count=2,
            start_pos=1,
        )

        wrapped.assert_called_once_with(
            [AccountSpec(symbols=["EURUSD"])],
            ["M1"],
            2,
            start_pos=1,
            base_config=None,
            retry_count=0,
            backoff_base=2.0,
        )
        pd.testing.assert_frame_equal(result["EURUSD", 1], df_rate)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"count": 0}, "count must be positive"),
            ({"count": 1, "start_pos": -1}, "start_pos must be non-negative"),
        ],
        ids=["zero-count", "negative-start-pos"],
    )
    def test_rejects_invalid_inputs_before_fetching(
        self,
        mocker: MockerFixture,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        """Test invalid count/start_pos values are rejected before MT5 is called."""
        wrapped = mocker.patch(
            "mt5cli.market_data.collect_latest_rates_for_accounts_with_retries",
        )

        with pytest.raises(ValueError, match=match):
            collect_latest_closed_rates_for_accounts(
                [AccountSpec(symbols=["EURUSD"])],
                ["M1"],
                **kwargs,  # type: ignore[arg-type]
            )

        wrapped.assert_not_called()

    def test_processes_multiple_symbol_timeframe_pairs(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test each returned series is trimmed and validated independently."""
        mocker.patch(
            "mt5cli.market_data.collect_latest_rates_for_accounts_with_retries",
            return_value={
                ("EURUSD", 1): pd.DataFrame(
                    {"time": [1, 2, 3], "close": [1.1, 1.2, 1.3]},
                ),
                ("GBPUSD", 16385): pd.DataFrame(
                    {"time": [4, 5, 6], "close": [2.1, 2.2, 2.3]},
                ),
            },
        )

        result = collect_latest_closed_rates_for_accounts(
            [AccountSpec(symbols=["EURUSD", "GBPUSD"])],
            ["M1", "H1"],
            count=2,
        )

        assert set(result) == {("EURUSD", 1), ("GBPUSD", 16385)}
        pd.testing.assert_frame_equal(
            result["EURUSD", 1],
            pd.DataFrame({"time": [1, 2], "close": [1.1, 1.2]}),
        )
        pd.testing.assert_frame_equal(
            result["GBPUSD", 16385],
            pd.DataFrame({"time": [4, 5], "close": [2.1, 2.2]}),
        )


class TestFetchLatestClosedRates:
    """Tests for fetch_latest_closed_rates."""

    def test_fetches_extra_bar_and_drops_forming_row(self) -> None:
        """Test single-symbol closed-bar helper hides the forming bar."""
        client = MagicMock()
        client.latest_rates.return_value = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "close": [1.0, 1.1, 1.2],
            },
        )

        result = fetch_latest_closed_rates(
            client,
            symbol="EURUSD",
            granularity="M1",
            count=2,
        )

        client.latest_rates.assert_called_once_with(
            "EURUSD",
            "M1",
            3,
            start_pos=0,
        )
        assert list(result["close"]) == [1.0, 1.1]

    def test_raises_when_no_closed_bars_are_available(self) -> None:
        """Test empty closed-bar results raise an actionable ValueError."""
        client = MagicMock()
        client.latest_rates.return_value = pd.DataFrame({"close": [1.0]})

        with pytest.raises(ValueError, match="Rate data is empty"):
            fetch_latest_closed_rates(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=1,
            )

    def test_rejects_non_positive_count_before_fetching(self) -> None:
        """Test invalid count values fail before calling MT5."""
        client = MagicMock()

        with pytest.raises(ValueError, match="count must be positive"):
            fetch_latest_closed_rates(
                client,
                symbol="EURUSD",
                granularity="M1",
                count=0,
            )

        client.latest_rates.assert_not_called()


class TestCollectLatestClosedRatesByGranularity:
    """Tests for collect_latest_closed_rates_by_granularity."""

    def test_rekeys_by_granularity_name(self, mocker: MockerFixture) -> None:
        """Test closed rates are keyed by symbol and granularity name."""
        df_rate = pd.DataFrame({"time": [1, 2], "close": [1.1, 1.2]})
        wrapped = mocker.patch(
            "mt5cli.market_data.collect_latest_closed_rates_for_accounts",
            return_value={("EURUSD", 1): df_rate},
        )

        result = collect_latest_closed_rates_by_granularity(
            [AccountSpec(symbols=["EURUSD"])],
            ["M1"],
            count=2,
        )

        wrapped.assert_called_once_with(
            [AccountSpec(symbols=["EURUSD"])],
            ["M1"],
            2,
            start_pos=0,
            base_config=None,
            retry_count=0,
            backoff_base=2.0,
        )
        assert ("EURUSD", "M1") in result
        pd.testing.assert_frame_equal(result["EURUSD", "M1"], df_rate)


class TestResolveAccountSpec:
    """Tests for resolve_account_spec and resolve_account_specs."""

    def test_substitutes_env_placeholders_in_account(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test account string fields resolve ${ENV_VAR} placeholders."""
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        account = AccountSpec(
            symbols=["EURUSD"],
            login="${MT5_LOGIN}",
            password="${MT5_PASSWORD}",
        )
        monkeypatch.setenv("MT5_LOGIN", "999")

        resolved = resolve_account_spec(account)

        assert resolved.login == "999"
        assert resolved.password == "secret"  # noqa: S105
        assert resolved.symbols == ["EURUSD"]

    def test_explicit_overrides_take_precedence(self) -> None:
        """Test explicit override values win over account fields."""
        account = AccountSpec(symbols=["EURUSD"], login=111, server="Acct")

        resolved = resolve_account_spec(
            account,
            login=222,
            server="Override",
            timeout=5000,
        )

        assert resolved.login == 222
        assert resolved.server == "Override"
        assert resolved.timeout == 5000

    def test_resolves_string_login_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test string login overrides expand ${ENV_VAR} placeholders."""
        monkeypatch.setenv("MT5_LOGIN", "777")
        account = AccountSpec(symbols=["EURUSD"], login=111)

        resolved = resolve_account_spec(account, login="${MT5_LOGIN}")

        assert resolved.login == "777"

    def test_preserves_integer_login_without_coercion(self) -> None:
        """Test integer logins remain integers after resolution."""
        account = AccountSpec(symbols=["EURUSD"], login=111)

        resolved = resolve_account_spec(account)

        assert resolved.login == 111
        assert isinstance(resolved.login, int)

    def test_raises_on_missing_env_variable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test missing environment variables raise ValueError."""
        monkeypatch.delenv("MT5_NOPE", raising=False)
        account = AccountSpec(symbols=["EURUSD"], server="${MT5_NOPE}")

        with pytest.raises(ValueError, match="'MT5_NOPE' is not set"):
            resolve_account_spec(account)

    def test_resolve_account_specs_applies_to_all(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test resolve_account_specs resolves every account in order."""
        monkeypatch.setenv("MT5_SERVER", "Shared")
        accounts = [
            AccountSpec(symbols=["EURUSD"], server="${MT5_SERVER}"),
            AccountSpec(symbols=["GBPUSD"], server="Fixed"),
        ]

        resolved = resolve_account_specs(accounts, timeout=1000)

        assert [a.server for a in resolved] == ["Shared", "Fixed"]
        assert all(a.timeout == 1000 for a in resolved)

    @pytest.mark.parametrize(
        ("allow_whole_dollar_env", "expected"),
        [
            (True, "secret"),
            (False, "$MT5_PASSWORD"),
        ],
    )
    def test_resolve_account_spec_whole_dollar_password(
        self,
        monkeypatch: pytest.MonkeyPatch,
        allow_whole_dollar_env: bool,
        expected: str,
    ) -> None:
        """Test resolve_account_spec expands $ENV_NAME password only with opt-in."""
        monkeypatch.setenv("MT5_PASSWORD", "secret")
        account = AccountSpec(symbols=["EURUSD"], password="$MT5_PASSWORD")

        resolved = resolve_account_spec(
            account, allow_whole_dollar_env=allow_whole_dollar_env
        )

        assert resolved.password == expected

    def test_resolve_account_specs_with_whole_dollar_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test resolve_account_specs threads allow_whole_dollar_env to each account."""
        monkeypatch.setenv("MT5_SERVER", "Broker-Demo")
        accounts = [
            AccountSpec(symbols=["EURUSD"], server="$MT5_SERVER"),
            AccountSpec(symbols=["GBPUSD"], server="Fixed"),
        ]

        resolved = resolve_account_specs(accounts, allow_whole_dollar_env=True)

        assert resolved[0].server == "Broker-Demo"
        assert resolved[1].server == "Fixed"

    def test_resolve_account_spec_whole_dollar_login(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test $ENV_NAME login string is expanded when allow_whole_dollar_env=True."""
        monkeypatch.setenv("MT5_LOGIN", "12345")
        account = AccountSpec(symbols=["EURUSD"], login="$MT5_LOGIN")

        resolved = resolve_account_spec(account, allow_whole_dollar_env=True)

        assert resolved.login == "12345"
