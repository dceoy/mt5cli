"""Tests for mt5cli.analytics."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mt5cli.analytics import create_analytics_views, publish_dashboard
from mt5cli.grafana import create_snapshot_tables, start_snapshot_run

if TYPE_CHECKING:
    from pathlib import Path

_DEALS_DDL = (
    "CREATE TABLE history_deals ("
    " ticket INTEGER, position_id INTEGER, symbol TEXT, time TEXT, type INTEGER,"
    " entry INTEGER, volume REAL, price REAL, profit REAL, commission REAL,"
    " swap REAL, fee REAL, magic INTEGER)"
)
_DEALS: list[tuple[object, ...]] = [
    # 100: buy, full close, NULL fee -> net 100 - 4 - 1 = 95
    (1, 100, "EURUSD", "2024-01-01 10:00:00", 0, 0, 1.0, 1.1, 0.0, -2.0, 0.0, None, 1),
    (
        2,
        100,
        "EURUSD",
        "2024-01-01 12:00:00",
        1,
        1,
        1.0,
        1.2,
        100.0,
        -2.0,
        -1.0,
        None,
        1,
    ),
    # 101: buy, two partial closes on different days -> net 40 - 1 - 2 - 0.5 = 36.5
    (3, 101, "EURUSD", "2024-01-02 09:00:00", 0, 0, 2.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1),
    (4, 101, "EURUSD", "2024-01-02 10:00:00", 1, 1, 1.0, 1.1, 10.0, None, 0.0, 0.0, 1),
    (
        5,
        101,
        "EURUSD",
        "2024-01-03 10:00:00",
        1,
        1,
        1.0,
        1.3,
        30.0,
        None,
        -2.0,
        -0.5,
        1,
    ),
    # 102: sell loss for another magic -> net -50 - 1 = -51
    (6, 102, "GBPUSD", "2024-01-03 11:00:00", 1, 0, 1.0, 1.5, 0.0, -0.5, 0.0, 0.0, 2),
    (
        7,
        102,
        "GBPUSD",
        "2024-01-03 13:00:00",
        0,
        1,
        1.0,
        1.55,
        -50.0,
        -0.5,
        0.0,
        0.0,
        2,
    ),
    # 103: reversal (DEAL_ENTRY_INOUT) closes the original position
    (8, 103, "EURUSD", "2024-01-04 09:00:00", 0, 0, 1.0, 1.1, 0.0, 0.0, 0.0, 0.0, 1),
    (9, 103, "EURUSD", "2024-01-04 10:00:00", 1, 2, 2.0, 1.1, 0.0, 0.0, 0.0, 0.0, 1),
    # 104: still open; 105: only partially closed; both are excluded
    (10, 104, "EURUSD", "2024-01-05 09:00:00", 0, 0, 1.0, 1.1, 0.0, 0.0, 0.0, 0.0, 1),
    (12, 105, "EURUSD", "2024-01-06 09:00:00", 0, 0, 2.0, 1.1, 0.0, 0.0, 0.0, 0.0, 1),
    (13, 105, "EURUSD", "2024-01-06 10:00:00", 1, 1, 1.0, 1.2, 5.0, 0.0, 0.0, 0.0, 1),
    # Balance row and position_id 0 are excluded.
    (11, 0, "", "2024-01-05 10:00:00", 2, 0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0, 0),
]


def _make_db(
    path: Path, ddl: str = _DEALS_DDL, rows: list[tuple[object, ...]] | None = None
) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute(ddl)
        data = _DEALS if rows is None else rows
        if data:
            marks = ", ".join("?" * len(data[0]))
            conn.executemany(f"INSERT INTO history_deals VALUES ({marks})", data)  # noqa: S608
    return path


def _query(path: Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(sql, conn)  # pyright: ignore[reportUnknownMemberType]


def _approx(expected: object) -> object:
    return pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """SQLite history database with the shared deal scenarios."""
    path = _make_db(tmp_path / "history.db")
    with sqlite3.connect(path) as conn:
        assert create_analytics_views(conn)
    return path


class TestAnalyticsTrades:
    """Tests for the canonical analytics_trades view."""

    def test_reconstructs_closed_positions_only(self, db: Path) -> None:
        """Open/partial positions, balance rows, and position_id 0 are excluded."""
        trades = _query(db, "SELECT * FROM analytics_trades ORDER BY position_id")
        assert trades["position_id"].tolist() == [100, 101, 102, 103]

    @pytest.mark.parametrize(
        ("position_id", "expected"),
        [
            (
                100,
                {
                    "side": "buy",
                    "magic": 1,
                    "volume": 1.0,
                    "profit": 100.0,
                    "commission": -4.0,
                    "swap": -1.0,
                    "fee": 0.0,
                    "net_profit": 95.0,
                    "holding_seconds": 7200,
                    "open_time": "2024-01-01 10:00:00",
                    "close_time": "2024-01-01 12:00:00",
                    "close_date": "2024-01-01",
                    "entry_price": 1.1,
                    "exit_price": 1.2,
                },
            ),
            (
                101,
                {
                    "side": "buy",
                    "magic": 1,
                    "volume": 2.0,
                    "profit": 40.0,
                    "commission": -1.0,
                    "swap": -2.0,
                    "fee": -0.5,
                    "net_profit": 36.5,
                    "holding_seconds": 90000,
                    "close_date": "2024-01-03",
                    "exit_price": 1.2,
                },
            ),
            (
                102,
                {
                    "side": "sell",
                    "magic": 2,
                    "net_profit": -51.0,
                    "close_date": "2024-01-03",
                },
            ),
            (103, {"reversal_count": 1, "deals_count": 2, "net_profit": 0.0}),
        ],
        ids=["full-close", "partial-closes", "sell-loss", "inout-reversal"],
    )
    def test_trade_values(
        self, db: Path, position_id: int, expected: dict[str, object]
    ) -> None:
        """Trade rows carry weighted prices, costs, and fee-inclusive net P/L."""
        row = _query(
            db,
            f"SELECT * FROM analytics_trades WHERE position_id = {position_id}",  # noqa: S608
        ).iloc[0]
        for column, value in expected.items():
            assert row[column] == _approx(value), column

    def test_scale_in_with_float_lots_counts_as_closed(self, tmp_path: Path) -> None:
        """Lots that sum with float error (0.1 + 0.2 vs 0.3) still close a trade."""
        rows: list[tuple[object, ...]] = [
            (
                1,
                20,
                "EURUSD",
                "2024-01-01 10:00:00",
                0,
                0,
                0.1,
                1.1,
                0.0,
                0.0,
                0.0,
                0.0,
                1,
            ),
            (
                2,
                20,
                "EURUSD",
                "2024-01-01 10:01:00",
                0,
                0,
                0.2,
                1.1,
                0.0,
                0.0,
                0.0,
                0.0,
                1,
            ),
            (
                3,
                20,
                "EURUSD",
                "2024-01-01 11:00:00",
                1,
                1,
                0.3,
                1.2,
                9.0,
                0.0,
                0.0,
                0.0,
                1,
            ),
        ]
        path = _make_db(tmp_path / "float.db", rows=rows)
        with sqlite3.connect(path) as conn:
            assert create_analytics_views(conn)
        trades = _query(path, "SELECT position_id, net_profit FROM analytics_trades")
        assert trades["position_id"].tolist() == [20]
        assert trades["net_profit"].tolist() == pytest.approx([9.0])

    def test_missing_optional_columns_default(self, tmp_path: Path) -> None:
        """Missing magic and cost columns become NULL magic and zero costs."""
        ddl = (
            "CREATE TABLE history_deals (ticket INTEGER, position_id INTEGER,"
            " symbol TEXT, time INTEGER, type INTEGER, entry INTEGER,"
            " volume REAL, price REAL, profit REAL)"
        )
        rows: list[tuple[object, ...]] = [
            (1, 7, "EURUSD", 1000, 0, 0, 1.0, 1.1, 0.0),
            (2, 7, "EURUSD", 2000, 1, 1, 1.0, 1.2, 5.0),
        ]
        path = _make_db(tmp_path / "old.db", ddl, rows)
        with sqlite3.connect(path) as conn:
            assert create_analytics_views(conn)
        row = _query(path, "SELECT * FROM analytics_trades").iloc[0]
        assert pd.isna(row["magic"])
        assert (row["commission"], row["swap"], row["fee"]) == (0, 0, 0)
        assert row["net_profit"] == _approx(5.0)
        assert row["holding_seconds"] == 1000

    def test_missing_required_columns_skips_views(self, tmp_path: Path) -> None:
        """History without reconstruction columns creates no analytics views."""
        path = _make_db(
            tmp_path / "bad.db", "CREATE TABLE history_deals (symbol TEXT)", []
        )
        with sqlite3.connect(path) as conn:
            assert not create_analytics_views(conn)
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        assert not {n for n in names if n.startswith("analytics_")}

    def test_idempotent(self, db: Path) -> None:
        """Rebuilding the views is safe."""
        with sqlite3.connect(db) as conn:
            assert create_analytics_views(conn)
            count = conn.execute("SELECT COUNT(*) FROM analytics_trades").fetchone()
        assert count == (4,)


class TestAggregateViews:
    """Tests for daily, strategy, and equity views."""

    def test_daily_pnl_groups_by_date_symbol_magic(self, db: Path) -> None:
        """Daily rows are split by close date, symbol, and magic."""
        daily = _query(db, "SELECT * FROM analytics_daily_pnl ORDER BY date, symbol")
        assert daily[["date", "symbol", "magic"]].to_numpy().tolist() == [
            ["2024-01-01", "EURUSD", 1],
            ["2024-01-03", "EURUSD", 1],
            ["2024-01-03", "GBPUSD", 2],
            ["2024-01-04", "EURUSD", 1],
        ]
        assert daily["net_profit"].tolist() == [95.0, 36.5, -51.0, 0.0]
        assert daily["trade_count"].tolist() == [1, 1, 1, 1]

    def test_strategy_stats_metrics(self, db: Path) -> None:
        """Strategy rows expose win/loss counts, rates, and profit factor."""
        stats = (
            _query(db, "SELECT * FROM analytics_strategy_stats")
            .set_index("symbol")
            .to_dict("index")
        )
        eur = stats["EURUSD"]
        assert (eur["trade_count"], eur["wins"], eur["losses"]) == (3, 2, 0)
        assert eur["win_rate"] == _approx(2 / 3)
        assert eur["net_profit"] == _approx(131.5)
        assert eur["avg_trade"] == _approx(131.5 / 3)
        assert eur["gross_profit"] == _approx(131.5)
        assert eur["profit_factor"] is None or pd.isna(eur["profit_factor"])
        gbp = stats["GBPUSD"]
        assert (gbp["wins"], gbp["losses"], gbp["magic"]) == (0, 1, 2)
        assert gbp["gross_loss"] == _approx(-51.0)
        assert gbp["profit_factor"] == 0
        assert eur["first_open_time"] == "2024-01-01 10:00:00"

    def test_equity_is_cumulative_in_close_order(self, db: Path) -> None:
        """Cumulative net profit accumulates by close time."""
        equity = _query(db, "SELECT * FROM analytics_equity ORDER BY time, position_id")
        assert equity["cumulative_net_profit"].tolist() == _approx([
            95.0,
            131.5,
            80.5,
            80.5,
        ])


class TestTemporaryViews:
    """Tests for read-only database support."""

    def test_temporary_views_leave_database_unchanged(self, tmp_path: Path) -> None:
        """TEMP views work on a read-only connection."""
        path = _make_db(tmp_path / "ro.db")
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            assert create_analytics_views(conn, temporary=True)
            assert conn.execute("SELECT COUNT(*) FROM analytics_trades").fetchone() == (
                4,
            )
        assert not {
            r[0]
            for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master")
        } & {"analytics_trades", "positions_reconstructed"}


class TestPublishDashboard:
    """Tests for publish_dashboard."""

    def test_publishes_parquet_and_manifest(self, tmp_path: Path) -> None:
        """Datasets round-trip through Parquet and are listed in the manifest."""
        path = _make_db(tmp_path / "history.db")
        with sqlite3.connect(path) as conn:
            create_snapshot_tables(conn)
            run_id = start_snapshot_run(conn, 1_700_000_000)
            conn.execute(
                "INSERT INTO account_snapshots (run_id, login, balance, equity)"
                " VALUES (?, 1, 1000.0, 1010.0)",
                (run_id,),
            )
            conn.execute("UPDATE snapshot_runs SET status = 'ok'")
        out = tmp_path / "dist" / "data"
        manifest = publish_dashboard(path, out)
        names = [d["name"] for d in manifest["datasets"]]
        assert names == [
            "trades",
            "daily_pnl",
            "strategy_stats",
            "equity",
            "account_snapshots",
        ]
        trades = pd.read_parquet(out / "trades.parquet")
        assert sorted(trades["position_id"]) == [100, 101, 102, 103]
        assert pd.read_parquet(out / "account_snapshots.parquet")[
            "equity"
        ].tolist() == [1010.0]
        written = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert written == manifest
        assert written["schema_version"] == 1
        assert written["generated_at"]
        trades_entry = written["datasets"][0]
        assert trades_entry["file"] == "trades.parquet"
        assert trades_entry["rows"] == 4
        assert "net_profit" in trades_entry["columns"]
        assert sorted(p.name for p in out.iterdir()) == sorted([
            *(f"{n}.parquet" for n in names),
            "manifest.json",
        ])
        assert "analytics_trades" not in {
            r[0]
            for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master")
        }

    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param([], id="empty-table"),
            pytest.param(
                [
                    (
                        1,
                        1,
                        "EURUSD",
                        "2024-01-01 10:00:00",
                        0,
                        0,
                        1.0,
                        1.1,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        None,
                    ),
                    (
                        2,
                        1,
                        "EURUSD",
                        "2024-01-01 11:00:00",
                        1,
                        1,
                        1.0,
                        1.2,
                        5.0,
                        0.0,
                        0.0,
                        0.0,
                        None,
                    ),
                ],
                id="null-magic",
            ),
        ],
    )
    def test_parquet_schema_is_stable(
        self, tmp_path: Path, rows: list[tuple[object, ...]]
    ) -> None:
        """Typed Parquet columns do not depend on whether rows or NULLs exist."""
        path = _make_db(tmp_path / "history.db", rows=rows)
        out = tmp_path / "out"
        publish_dashboard(path, out)
        trades = pq.read_schema(out / "trades.parquet")
        assert pa.types.is_int64(trades.field("position_id").type)
        assert pa.types.is_int64(trades.field("magic").type)
        assert pa.types.is_float64(trades.field("net_profit").type)
        assert pa.types.is_large_string(trades.field("symbol").type) or (
            pa.types.is_string(trades.field("symbol").type)
        )
        stats = pq.read_schema(out / "strategy_stats.parquet")
        assert pa.types.is_int64(stats.field("trade_count").type)
        assert pa.types.is_float64(stats.field("profit_factor").type)

    def test_rejects_manifest_dataset_collision(self, tmp_path: Path) -> None:
        """Manifest file names cannot overwrite generated Parquet datasets."""
        path = _make_db(tmp_path / "history.db")
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="collides"):
            publish_dashboard(path, out, manifest_name="trades.parquet")
        assert not out.exists()

    def test_skips_missing_sources(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Datasets without source tables are skipped with warnings."""
        path = _make_db(tmp_path / "history.db")
        manifest = publish_dashboard(path, tmp_path / "out", manifest_name="m.json")
        assert [d["name"] for d in manifest["datasets"]] == [
            "trades",
            "daily_pnl",
            "strategy_stats",
            "equity",
        ]
        assert (tmp_path / "out" / "m.json").exists()
        assert "Skipping account_snapshots" in caplog.text

    def test_snapshot_only_database(self, tmp_path: Path) -> None:
        """Trade datasets are skipped when history_deals is absent."""
        path = tmp_path / "snap.db"
        with sqlite3.connect(path) as conn:
            create_snapshot_tables(conn)
        manifest = publish_dashboard(path, tmp_path / "out")
        assert [d["name"] for d in manifest["datasets"]] == ["account_snapshots"]
        assert manifest["datasets"][0]["rows"] == 0

    def test_raises_when_nothing_to_publish(self, tmp_path: Path) -> None:
        """An empty database raises instead of writing an empty manifest."""
        path = tmp_path / "empty.db"
        sqlite3.connect(path).close()
        with pytest.raises(ValueError, match="No analytics datasets"):
            publish_dashboard(path, tmp_path / "out")
        assert not (tmp_path / "out").exists()

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        """A missing database is rejected without creating it."""
        with pytest.raises(ValueError, match="not found"):
            publish_dashboard(tmp_path / "nope.db", tmp_path / "out")
        assert not (tmp_path / "nope.db").exists()
