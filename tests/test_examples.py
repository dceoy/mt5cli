"""Tests for example files in examples/grafana/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "grafana"
_DASHBOARDS_DIR = _EXAMPLES_DIR / "dashboards"


def _dashboard_json_files() -> list[Path]:
    """Return bundled Grafana dashboard JSON files in deterministic order."""
    return sorted(_DASHBOARDS_DIR.glob("*.json"))


@pytest.fixture(params=_dashboard_json_files(), ids=lambda path: path.name)
def dashboard_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Yield one bundled Grafana dashboard JSON file per test case."""
    return request.param


class TestGrafanaExamples:
    """Validate structure and content of bundled Grafana example files."""

    def test_dashboard_json_file_is_valid_json(self, dashboard_path: Path) -> None:
        """Each dashboard JSON file parses to a JSON object."""
        content = dashboard_path.read_text(encoding="utf-8")
        obj = json.loads(content)
        assert isinstance(obj, dict), (
            f"{dashboard_path.name} root must be a JSON object"
        )

    def test_dashboard_json_uses_grafana_views(self, dashboard_path: Path) -> None:
        """Each dashboard JSON file queries grafana_* views."""
        content = dashboard_path.read_text(encoding="utf-8")
        assert "grafana_" in content, (
            f"{dashboard_path.name} must contain queries against grafana_* views"
        )

    @pytest.mark.parametrize("field", ["uid", "title"])
    def test_dashboard_json_has_required_field(
        self,
        dashboard_path: Path,
        field: str,
    ) -> None:
        """Each dashboard JSON file has a non-empty uid and title field."""
        obj = json.loads(dashboard_path.read_text(encoding="utf-8"))
        assert obj.get(field), f"{dashboard_path.name} must have a {field}"

    def test_expected_dashboards_present(self) -> None:
        """The three expected dashboard files are present."""
        names = {p.name for p in _dashboard_json_files()}
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
