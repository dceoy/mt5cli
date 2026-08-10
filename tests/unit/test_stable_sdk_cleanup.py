"""Focused regression tests for the reduced stable SDK surface."""

from pdmt5 import Mt5Config

import mt5cli


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


def test_stable_sdk_registry_is_derived_from_root_exports() -> None:
    """The enumerable SDK contract matches the root module public bindings."""
    expected = frozenset(
        name
        for name in vars(mt5cli)
        if not name.startswith("_") and name != "STABLE_SDK_EXPORTS"
    )
    assert mt5cli.STABLE_SDK_EXPORTS == expected
    assert set(mt5cli.__all__) == {*expected, "STABLE_SDK_EXPORTS"}


def test_legacy_margin_helper_is_removed() -> None:
    """The retired account-margin helper is absent from the public implementation."""
    name = "calculate_new_position_margin_ratio"
    assert name not in mt5cli.STABLE_SDK_EXPORTS
    assert not hasattr(mt5cli, name)
    assert not hasattr(mt5cli.trading, name)
