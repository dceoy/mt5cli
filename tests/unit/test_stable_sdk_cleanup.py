"""Focused regression tests for the reduced stable SDK surface."""

import mt5cli


def test_stable_sdk_exposes_downstream_adapter_primitives() -> None:
    """Required downstream primitives are available from the package root."""
    for name in (
        "Dataset",
        "ensure_grafana_schema",
        "parse_timeframe",
        "resolve_history_timeframes",
        "substitute_mapping_values",
    ):
        assert name in mt5cli.STABLE_SDK_EXPORTS
        assert hasattr(mt5cli, name)


def test_pdmt5_config_type_stays_out_of_root_sdk() -> None:
    """Connection config remains an implementation type, not a pass-through export."""
    assert "Mt5Config" not in mt5cli.STABLE_SDK_EXPORTS
    assert not hasattr(mt5cli, "Mt5Config")


def test_legacy_margin_helper_is_removed() -> None:
    """The retired account-margin helper is absent from the public implementation."""
    name = "calculate_new_position_margin_ratio"
    assert name not in mt5cli.STABLE_SDK_EXPORTS
    assert not hasattr(mt5cli, name)
    assert not hasattr(mt5cli.trading, name)
