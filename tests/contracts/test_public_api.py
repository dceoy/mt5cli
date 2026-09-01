"""Cross-module tests for the public mt5cli API contract."""

from __future__ import annotations

import inspect

import pytest
from pdmt5.dataframe import Mt5Config

import mt5cli
import mt5cli.cli
import mt5cli.history
import mt5cli.marketdata
import mt5cli.observability
import mt5cli.trading
import mt5cli.utils
from mt5cli import STABLE_SDK_EXPORTS, trading


class TestStableSdkContract:
    """Tests for the documented stable downstream SDK contract."""

    def test_stable_exports_cover_root_api(self) -> None:
        """STABLE_SDK_EXPORTS classifies every package-root symbol."""
        tier_metadata = {"STABLE_SDK_EXPORTS"}
        root_exports = set(mt5cli.__all__)

        missing_from_root = sorted(STABLE_SDK_EXPORTS - root_exports)
        assert not missing_from_root, (
            f"STABLE_SDK_EXPORTS missing from __all__: {missing_from_root}"
        )
        unclassified = sorted(root_exports - STABLE_SDK_EXPORTS - tier_metadata)
        assert not unclassified, (
            f"Root exports not in STABLE_SDK_EXPORTS: {unclassified}"
        )

    def test_public_function_annotations_hide_raw_pdmt5_clients(self) -> None:
        """The root facade never exposes a low-level pdmt5 client type."""
        for name in mt5cli.__all__:
            value = getattr(mt5cli, name)
            if not callable(value) or inspect.isclass(value):
                continue
            annotations = inspect.get_annotations(value, eval_str=False)
            assert all(
                "Mt5DataClient" not in str(annotation)
                for annotation in annotations.values()
            ), name


@pytest.mark.parametrize("name", sorted(STABLE_SDK_EXPORTS))
def test_stable_exports_are_importable_from_package_root(name: str) -> None:
    """Stable SDK names resolve through ``from mt5cli import ...``."""
    assert hasattr(mt5cli, name), f"{name!r} missing from mt5cli package root"


@pytest.mark.parametrize(
    "name",
    [
        "create_trading_client",
        "mt5_trading_session",
        "fetch_latest_closed_rates_for_trading_client",
        "fetch_recent_history_deals_for_trading_client",
    ],
)
def test_removed_legacy_apis_are_not_public(name: str) -> None:
    """No public factory or session can reopen the raw-client lifecycle path."""
    assert name not in mt5cli.__all__
    assert name not in STABLE_SDK_EXPORTS
    assert not hasattr(mt5cli, name)


@pytest.mark.parametrize(
    "name",
    [
        "Mt5RuntimeError",
        "Mt5DataClient",
        "TIMEFRAME_MAP",
        "COPY_TICKS_MAP",
        "ORDER_TYPE_MAP",
    ],
)
def test_pdmt5_pass_through_names_stay_out_of_public_contract(name: str) -> None:
    """Low-level pdmt5 types/constants stay out of the stable package root."""
    assert name not in STABLE_SDK_EXPORTS, (
        f"{name!r} should not be in STABLE_SDK_EXPORTS"
    )
    assert name not in mt5cli.__all__, f"{name!r} should not be in mt5cli.__all__"
    assert not hasattr(mt5cli, name), f"{name!r} should not be on the mt5cli root"


def test_no_other_submodule_defines_a_public_mt5_session() -> None:
    """No submodule defines a second public mt5_session implementation."""
    submodules = [
        mt5cli.history,
        mt5cli.marketdata,
        mt5cli.observability,
        mt5cli.trading,
        mt5cli.cli,
        mt5cli.utils,
    ]
    for module in submodules:
        if hasattr(module, "mt5_session"):
            assert module.mt5_session is mt5cli.mt5_session, (
                f"{module.__name__} must re-export the canonical mt5_session, "
                "not define an alternate one"
            )


def test_no_public_factory_returns_a_raw_pdmt5_client() -> None:
    """No stable export returns a raw pdmt5 client."""
    for name in STABLE_SDK_EXPORTS:
        value = getattr(mt5cli, name)
        if not callable(value) or inspect.isclass(value):
            continue
        annotations = inspect.get_annotations(value, eval_str=False)
        returns = str(annotations.get("return", ""))
        assert "Mt5DataClient" not in returns, name


def test_stable_sdk_exposes_downstream_adapter_primitives() -> None:
    """Required downstream primitives are available from the package root."""
    for name in (
        "Dataset",
        "Mt5Config",
        "build_config",
        "ensure_grafana_schema",
        "parse_timeframe",
        "resolve_history_timeframes",
        "substitute_mapping_values",
    ):
        assert name in mt5cli.STABLE_SDK_EXPORTS
        assert hasattr(mt5cli, name)


def test_build_config_returns_public_config_type() -> None:
    """The stable config builder returns the stable root-exported config type."""
    config = mt5cli.build_config(login=12345, server="Demo")
    assert isinstance(config, Mt5Config)
    assert mt5cli.Mt5Config is Mt5Config


def test_stable_sdk_registry_matches_package_all() -> None:
    """The root module re-exports exactly the canonical SDK declaration."""
    assert set(mt5cli.__all__) == {
        *mt5cli.STABLE_SDK_EXPORTS,
        "STABLE_SDK_EXPORTS",
    }
    for name in mt5cli.STABLE_SDK_EXPORTS:
        assert hasattr(mt5cli, name)


def test_internal_protocols_stay_out_of_stable_sdk() -> None:
    """Internal structural protocols stay out of the stable SDK surface."""
    assert "ObservabilityClient" not in mt5cli.STABLE_SDK_EXPORTS
    assert not hasattr(mt5cli, "ObservabilityClient")


def test_legacy_margin_helper_is_removed() -> None:
    """The retired account-margin helper is absent from the public implementation."""
    name = "calculate_new_position_margin_ratio"
    assert name not in mt5cli.STABLE_SDK_EXPORTS
    assert not hasattr(mt5cli, name)
    assert not hasattr(trading, name)
