"""Documentation contract tests for canonical rate storage."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "relative_path",
    [
        "README.md",
        "docs/api/history.md",
        "docs/api/public-contract.md",
        "docs/index.md",
    ],
)
def test_rate_docs_do_not_reference_removed_managed_views(
    relative_path: str,
    request: pytest.FixtureRequest,
) -> None:
    """Published docs describe only canonical managed rate storage."""
    root = request.config.rootpath
    text = (root / relative_path).read_text()
    stale_tokens = (
        "resolve_rate_view_name",
        "resolve_rate_view_names",
        "resolve_rate_tables",
        "Rate compatibility views",
        "mt5cli-managed rate tables/views",
        "managed rate legacy views",
    )
    assert not any(token in text for token in stale_tokens)
