"""Focused regression tests for the reduced stable SDK surface."""

import mt5cli


def test_stable_sdk_exposes_downstream_adapter_primitives() -> None:
    """Required downstream primitives are available from the package root."""
    for name in (
        "Dataset",
        "Mt5Config",
        "ensure_grafana_schema",
        "parse_timeframe",
        "resolve_history_timeframes",
        "substitute_mapping_values",
    ):
        assert name in mt5cli.STABLE_SDK_EXPORTS
        assert hasattr(mt5cli, name)


def test_legacy_margin_helper_is_not_stable() -> None:
    """The retired account-margin helper is absent from the stable root API."""
    assert "calculate_new_position_margin_ratio" not in mt5cli.STABLE_SDK_EXPORTS
    assert not hasattr(mt5cli, "calculate_new_position_margin_ratio")
