"""Parameterized regression tests for remaining duplicate test cases."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest
from typer.testing import CliRunner

from mt5cli.cli import app
from mt5cli.history import (
    load_rate_data,
    resolve_granularity_name,
    resolve_history_tick_flags,
)
from mt5cli.sdk import AccountSpec, collect_latest_closed_rates_for_accounts
from mt5cli.trading import calculate_spread_ratio

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

runner = CliRunner()


@pytest.mark.parametrize(
    ("command", "patch_snapshot_update", "use_publish_copy", "expect_called"),
    [
        ("snapshot", True, True, True),
        ("snapshot", True, False, False),
        ("grafana-schema", False, True, True),
        ("grafana-schema", False, False, False),
    ],
    ids=[
        "snapshot-with-publish-copy",
        "snapshot-no-publish-copy",
        "grafana-schema-with-publish-copy",
        "grafana-schema-no-publish-copy",
    ],
)
def test_publish_copy_option_gates_grafana_copy(
    tmp_path: Path,
    mocker: MockerFixture,
    command: str,
    patch_snapshot_update: bool,
    use_publish_copy: bool,
    expect_called: bool,
) -> None:
    """snapshot and grafana-schema gate publish-copy with the same option semantics."""
    if patch_snapshot_update:
        mocker.patch("mt5cli.cli.sdk.update_observability_with_config")
    mock_publish = mocker.patch("mt5cli.grafana.publish_grafana_copy")
    args = ["-o", str(tmp_path / "out.db"), command]
    if use_publish_copy:
        args += ["--publish-copy", str(tmp_path / "grafana.db")]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    if expect_called:
        mock_publish.assert_called_once()
    else:
        mock_publish.assert_not_called()


@pytest.mark.parametrize(
    ("upstream_frame", "kwargs"),
    [
        pytest.param(
            pd.DataFrame({"time": [1], "close": [1.1]}),
            {"count": 1},
            id="forming-bar-only",
        ),
        pytest.param(
            pd.DataFrame(columns=["time", "close"]),
            {"count": 1, "start_pos": 1},
            id="empty-start-pos-nonzero",
        ),
    ],
)
def test_collect_latest_closed_rates_rejects_empty_effective_frames(
    mocker: MockerFixture,
    upstream_frame: pd.DataFrame,
    kwargs: dict[str, object],
) -> None:
    """Closed-rate collection rejects empty frames after start-position handling."""
    mocker.patch(
        "mt5cli.sdk.collect_latest_rates_for_accounts_with_retries",
        return_value={("EURUSD", 1): upstream_frame},
    )

    with pytest.raises(ValueError, match="Rate data is empty"):
        collect_latest_closed_rates_for_accounts(
            [AccountSpec(symbols=["EURUSD"])],
            ["M1"],
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("path_kind", "match"),
    [
        pytest.param("missing-file", "SQLite database not found", id="missing-file"),
        pytest.param("directory", "not a file", id="directory"),
    ],
)
def test_load_rate_data_rejects_invalid_database_paths(
    tmp_path: Path,
    path_kind: str,
    match: str,
) -> None:
    """SQLite path validation rejects missing files and directories consistently."""
    path = tmp_path / "missing.db" if path_kind == "missing-file" else tmp_path

    with pytest.raises(ValueError, match=match):
        load_rate_data(path, "rates")


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        pytest.param("ALL", -1, id="all-alias"),
        pytest.param(2, 2, id="integer-pass-through"),
    ],
)
def test_resolve_history_tick_flags_mapping(flags: str | int, expected: int) -> None:
    """Tick flag resolution maps aliases and preserves explicit integers."""
    assert resolve_history_tick_flags(flags) == expected


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [
        pytest.param(999, "999", id="unknown-integer"),
        pytest.param(1, "M1", id="known-timeframe"),
    ],
)
def test_resolve_granularity_name_mapping(timeframe: int, expected: str) -> None:
    """Granularity names use known aliases and fall back to integer text."""
    assert resolve_granularity_name(timeframe) == expected


@pytest.mark.parametrize(
    "tick_payload",
    [
        pytest.param({"bid": 99.0, "ask": 101.0}, id="numeric-values"),
        pytest.param({"bid": "99.0", "ask": "101.0"}, id="numeric-strings"),
    ],
)
def test_calculate_spread_ratio_from_tick_payload(
    tick_payload: dict[str, object],
) -> None:
    """Spread ratio accepts numeric and numeric-string bid/ask payloads."""
    client = MagicMock()
    client.symbol_info_tick_as_dict.return_value = tick_payload

    assert abs(calculate_spread_ratio(client, "EURUSD") - 0.02) < 1e-9
