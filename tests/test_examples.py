"""Tests for example files in examples/grafana/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "grafana"
_DASHBOARDS_DIR = _EXAMPLES_DIR / "dashboards"


class TestGrafanaExamples:
    """Validate structure and content of bundled Grafana example files."""

    def test_dashboard_json_files_are_valid_json(self) -> None:
        """All dashboard JSON files parse without error."""
        paths = list(_DASHBOARDS_DIR.glob("*.json"))
        assert paths, "No dashboard JSON files found"
        for path in paths:
            content = path.read_text(encoding="utf-8")
            obj = json.loads(content)
            assert isinstance(obj, dict), f"{path.name} root must be a JSON object"

    def test_dashboard_json_has_no_private_placeholders(self) -> None:
        """Dashboard JSON files contain no obvious credential placeholders."""
        private_patterns = ["password", "api_key", "apikey"]
        for path in _DASHBOARDS_DIR.glob("*.json"):
            content = path.read_text(encoding="utf-8").lower()
            for pat in private_patterns:
                assert pat not in content, f"{path.name} contains {pat!r}"

    def test_dashboard_json_uses_grafana_views(self) -> None:
        """All dashboard JSON files query grafana_* views."""
        for path in _DASHBOARDS_DIR.glob("*.json"):
            content = path.read_text(encoding="utf-8")
            assert "grafana_" in content, (
                f"{path.name} must contain queries against grafana_* views"
            )

    @pytest.mark.parametrize("field", ["uid", "title"])
    def test_dashboard_json_has_required_field(self, field: str) -> None:
        """All dashboard JSON files have a non-empty uid and title field."""
        for path in _DASHBOARDS_DIR.glob("*.json"):
            obj = json.loads(path.read_text(encoding="utf-8"))
            assert obj.get(field), f"{path.name} must have a {field}"

    def test_expected_dashboards_present(self) -> None:
        """The three expected dashboard files are present."""
        names = {p.name for p in _DASHBOARDS_DIR.glob("*.json")}
        assert "mt5cli-overview.json" in names
        assert "mt5cli-trades.json" in names
        assert "mt5cli-market.json" in names

    @pytest.mark.parametrize(
        "relative_path",
        [
            "README.md",
            "compose.yml",
            "provisioning/datasources/mt5cli-sqlite.yml",
            "provisioning/dashboards/mt5cli.yml",
        ],
        ids=["readme", "compose", "datasource-provisioning", "dashboard-provisioning"],
    )
    def test_expected_file_exists(self, relative_path: str) -> None:
        """Bundled Grafana example support files are present."""
        assert (_EXAMPLES_DIR / relative_path).is_file()
