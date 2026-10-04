"""Tests for mt5cli.analytics."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from mt5cli.analytics import create_analytics_views, publish_dashboard
from mt5cli.history import create_positions_reconstructed_view, get_table_columns
from mt5cli.observability import create_snapshot_tables, start_snapshot_run

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
    # 104: still open; 105: only partially closed (no trade leg, but its
    # realized events still count)
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
        assert trades["net_profit"].tolist() == _approx([9.0])

    def test_missing_optional_columns_default(self, tmp_path: Path) -> None:
        """Missing ticket, magic and cost columns fall back to rowid, NULL and zero."""
        ddl = (
            "CREATE TABLE history_deals (position_id INTEGER,"
            " symbol TEXT, time INTEGER, type INTEGER, entry INTEGER,"
            " volume REAL, price REAL, profit REAL)"
        )
        rows: list[tuple[object, ...]] = [
            (7, "EURUSD", 1000, 0, 0, 1.0, 1.1, 0.0),
            (7, "EURUSD", 2000, 1, 1, 1.0, 1.2, 5.0),
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

    def test_daily_pnl_is_cash_basis_by_symbol_and_magic(self, db: Path) -> None:
        """Daily rows hold deal-time cash flows split by symbol and magic."""
        daily = _query(db, "SELECT * FROM analytics_daily_pnl ORDER BY date, symbol")
        assert daily[["date", "symbol", "magic"]].to_numpy().tolist() == [
            ["2024-01-01", "EURUSD", 1],
            ["2024-01-02", "EURUSD", 1],
            ["2024-01-03", "EURUSD", 1],
            ["2024-01-03", "GBPUSD", 2],
            ["2024-01-04", "EURUSD", 1],
            ["2024-01-05", "EURUSD", 1],
            ["2024-01-06", "EURUSD", 1],
        ]
        # Position 101 realizes 10 on Jan 2 and 30 on Jan 3, not 40 on Jan 3, and
        # position 105 shows its partial-close profit while still open.
        assert daily["net_profit"].tolist() == _approx([
            95.0,
            9.0,
            27.5,
            -51.0,
            0.0,
            0.0,
            5.0,
        ])
        assert daily["event_count"].tolist() == [2, 2, 1, 2, 3, 1, 2]

    def test_open_leg_events_do_not_wait_for_the_final_close(
        self, tmp_path: Path
    ) -> None:
        """A partial close is realized when it happens, before the leg completes."""
        rows: list[tuple[object, ...]] = [
            (
                1,
                9,
                "EURUSD",
                "2024-02-01 09:00:00",
                0,
                0,
                2.0,
                1.1,
                0.0,
                -1.0,
                0.0,
                0.0,
                3,
            ),
            (
                2,
                9,
                "EURUSD",
                "2024-02-01 10:00:00",
                1,
                1,
                1.0,
                1.2,
                5.0,
                0.0,
                0.0,
                0.0,
                3,
            ),
        ]
        path = _make_db(tmp_path / "open.db", rows=rows)
        with sqlite3.connect(path) as conn:
            assert create_analytics_views(conn)
        assert _query(path, "SELECT * FROM analytics_trades").empty
        daily = _query(path, "SELECT * FROM analytics_daily_pnl")
        assert daily["date"].tolist() == ["2024-02-01"]
        assert daily["magic"].tolist() == [3]
        assert daily["net_profit"].tolist() == _approx([4.0])

    def test_events_reconcile_with_trade_legs(self, db: Path) -> None:
        """Realized events of completed legs sum to each leg's net profit."""
        diff = _query(
            db,
            "SELECT t.position_id, t.net_profit - SUM(e.net_profit) AS diff"
            " FROM analytics_trades t JOIN analytics_realized_events e"
            " ON e.position_id = t.position_id AND e.leg_index = t.leg_index"
            " GROUP BY t.position_id, t.leg_index",
        )
        assert len(diff) == 4
        assert diff["diff"].tolist() == _approx([0.0] * 4)

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

    def test_equity_is_cumulative_in_event_order(self, db: Path) -> None:
        """Cumulative net profit accumulates by deal timestamp."""
        equity = _query(db, "SELECT * FROM analytics_equity ORDER BY time, ticket")
        assert equity["cumulative_net_profit"].tolist() == _approx([
            *[-2.0, 95.0, 94.0, 104.0, 131.5, 131.0],
            *[80.5] * 6,
            85.5,
        ])


class TestReversalLegs:
    """Tests for position reversals (DEAL_ENTRY_INOUT) split into trade legs."""

    @staticmethod
    def _rows(*deals: tuple[object, ...]) -> list[tuple[object, ...]]:
        return [
            (
                i,
                200,
                "EURUSD",
                time,
                kind,
                entry,
                volume,
                price,
                profit,
                comm,
                swap,
                0.0,
                1,
            )
            for i, (time, kind, entry, volume, price, profit, comm, swap) in enumerate(
                deals, start=1
            )
        ]

    def _legs(self, tmp_path: Path, rows: list[tuple[object, ...]]) -> pd.DataFrame:
        path = _make_db(tmp_path / "rev.db", rows=rows)
        with sqlite3.connect(path) as conn:
            assert create_analytics_views(conn)
        return _query(path, "SELECT * FROM analytics_trades ORDER BY leg_index")

    def test_reversal_then_close_splits_two_legs(self, tmp_path: Path) -> None:
        """A long reversed into a short and later closed yields two legs."""
        legs = self._legs(
            tmp_path,
            self._rows(
                ("2024-01-01 10:00:00", 0, 0, 1.0, 1.10, 0.0, -1.0, 0.0),
                ("2024-01-02 10:00:00", 1, 2, 2.0, 1.20, 10.0, -2.0, -0.3),
                ("2024-01-03 10:00:00", 0, 1, 1.0, 1.25, -4.0, -0.5, -0.2),
            ),
        )
        assert legs["leg_index"].tolist() == [0, 1]
        assert legs["side"].tolist() == ["buy", "sell"]
        assert legs["volume"].tolist() == _approx([1.0, 1.0])
        assert legs["close_date"].tolist() == ["2024-01-02", "2024-01-03"]
        assert legs["profit"].tolist() == _approx([10.0, -4.0])
        # Commission is split pro rata; the reversal's swap stays on the closed leg.
        assert legs["commission"].tolist() == _approx([-2.0, -1.5])
        assert legs["swap"].tolist() == _approx([-0.3, -0.2])
        assert legs["net_profit"].tolist() == _approx([7.7, -5.7])
        assert legs["reversal_count"].tolist() == [1, 1]
        assert legs["deals_count"].tolist() == [2, 2]

    def test_open_leg_after_reversal_is_excluded(self, tmp_path: Path) -> None:
        """The reopened leg is not a trade until it closes."""
        legs = self._legs(
            tmp_path,
            self._rows(
                ("2024-01-01 10:00:00", 0, 0, 1.0, 1.10, 0.0, -1.0, 0.0),
                ("2024-01-02 10:00:00", 1, 2, 2.0, 1.20, 10.0, -2.0, -0.3),
            ),
        )
        assert legs["leg_index"].tolist() == [0]
        assert legs["net_profit"].tolist() == _approx([10.0 - 2.0 / 2 - 1.0 - 0.3])

    def test_reversal_without_visible_prior_volume_keeps_its_swap(
        self, tmp_path: Path
    ) -> None:
        """A reversal with nothing to close in history does not lose its swap."""
        path = _make_db(
            tmp_path / "trunc.db",
            rows=self._rows(("2024-01-02 10:00:00", 1, 2, 2.0, 1.2, 10.0, -2.0, -0.6)),
        )
        with sqlite3.connect(path) as conn:
            assert create_analytics_views(conn)
        events = _query(path, "SELECT swap, commission FROM analytics_realized_events")
        assert events["swap"].sum() == _approx(-0.6)
        assert events["commission"].sum() == _approx(-2.0)

    def test_double_reversal_yields_three_legs(self, tmp_path: Path) -> None:
        """Each INOUT deal starts a new leg."""
        legs = self._legs(
            tmp_path,
            self._rows(
                ("2024-01-01 10:00:00", 0, 0, 1.0, 1.0, 0.0, 0.0, 0.0),
                ("2024-01-02 10:00:00", 1, 2, 2.0, 1.1, 5.0, 0.0, 0.0),
                ("2024-01-03 10:00:00", 0, 2, 2.0, 1.0, 7.0, 0.0, 0.0),
                ("2024-01-04 10:00:00", 1, 1, 1.0, 1.2, 9.0, 0.0, 0.0),
            ),
        )
        assert legs["side"].tolist() == ["buy", "sell", "buy"]
        assert legs["profit"].tolist() == _approx([5.0, 7.0, 9.0])

    def test_matches_positions_reconstructed(self, db: Path) -> None:
        """Leg 0 keeps the positions_reconstructed semantics of the position view."""
        with sqlite3.connect(db) as conn:
            columns = get_table_columns(conn, "history_deals")
            assert create_positions_reconstructed_view(conn, columns)
        compared = _query(
            db,
            "SELECT t.position_id, t.reversal_count,"
            " t.open_time = p.open_time AS open_ok,"
            " ABS(t.volume - p.volume_open) < 1e-9 AS volume_ok,"
            " ABS(t.entry_price - p.open_price) < 1e-9 AS entry_ok,"
            " t.close_time = p.close_time AS close_ok,"
            " ABS(t.exit_price - p.close_price) < 1e-9 AS exit_ok,"
            " ABS(t.profit - p.total_profit) < 1e-9 AS profit_ok"
            " FROM analytics_trades t JOIN positions_reconstructed p"
            " ON p.position_id = t.position_id"
            " WHERE t.leg_index = 0 ORDER BY t.position_id",
        )
        assert compared["position_id"].tolist() == [100, 101, 102, 103]
        # Without reversals every field matches; the reversal position (103)
        # keeps the same opening side and profit, while its closing fields use the
        # split INOUT portion instead of the whole deal.
        plain = compared[compared["reversal_count"] == 0]
        assert plain.iloc[:, 2:].to_numpy().tolist() == [[1] * 6] * 3
        reversed_ = compared[compared["reversal_count"] == 1].iloc[0]
        assert (reversed_["open_ok"], reversed_["volume_ok"]) == (1, 1)
        assert (reversed_["entry_ok"], reversed_["profit_ok"]) == (1, 1)


class TestIncompleteHistoryAndMixedMagic:
    """Tests for legs without a visible opening and legs mixing magics."""

    @pytest.mark.parametrize(
        ("rows", "expected_daily_net"),
        [
            pytest.param(
                [
                    (
                        1,
                        50,
                        "EURUSD",
                        "2024-03-01 10:00:00",
                        1,
                        1,
                        1.0,
                        1.2,
                        30.0,
                        -1.0,
                        0.0,
                        0.0,
                        7,
                    )
                ],
                29.0,
                id="close-only",
            ),
            pytest.param(
                [
                    (
                        1,
                        51,
                        "EURUSD",
                        "2024-03-01 10:00:00",
                        1,
                        2,
                        2.0,
                        1.2,
                        10.0,
                        -2.0,
                        -0.6,
                        0.0,
                        7,
                    )
                ],
                7.4,
                id="reversal-without-visible-opening",
            ),
        ],
    )
    def test_leg_without_visible_opening_is_not_a_trade(
        self, tmp_path: Path, rows: list[tuple[object, ...]], expected_daily_net: float
    ) -> None:
        """Cash flows stay in events and daily P/L but no trade is invented."""
        path = _make_db(tmp_path / "close_only.db", rows=rows)
        with sqlite3.connect(path) as conn:
            assert create_analytics_views(conn)
        assert _query(path, "SELECT * FROM analytics_trades").empty
        assert _query(path, "SELECT * FROM analytics_strategy_stats").empty
        daily = _query(path, "SELECT * FROM analytics_daily_pnl")
        assert daily["magic"].tolist() == [7]
        assert daily["net_profit"].tolist() == _approx([expected_daily_net])

    @pytest.mark.parametrize(
        "kinds",
        [
            # (type, entry, volume): capture starts with a 2-lot long already open
            pytest.param([(1, 1, 0.5), (1, 2, 2.0), (0, 1, 0.5)], id="partial-close"),
            # 1-lot long already open: the visible balance would fabricate a trade
            pytest.param([(1, 1, 0.7), (1, 2, 1.0), (0, 1, 0.7)], id="fabricated"),
            # first visible deal is an out-by, then a later entry
            pytest.param([(0, 3, 0.5), (0, 0, 1.0), (1, 1, 1.0)], id="out-by-first"),
        ],
    )
    def test_position_opened_before_capture_is_not_reconstructed(
        self, tmp_path: Path, kinds: list[tuple[int, int, float]]
    ) -> None:
        """Unknown pre-capture exposure keeps cash flows but yields no trade legs."""
        rows: list[tuple[object, ...]] = [
            (
                ticket,
                70,
                "EURUSD",
                f"2024-04-01 1{ticket}:00:00",
                kind,
                entry,
                volume,
                1.2,
                10.0 * ticket,
                -0.5 * ticket,
                -0.1 * ticket,
                0.0,
                7,
            )
            for ticket, (kind, entry, volume) in enumerate(kinds, start=1)
        ]
        raw_net = sum(10.0 * t - 0.5 * t - 0.1 * t for t in range(1, len(kinds) + 1))
        path = _make_db(tmp_path / "unseen.db", rows=rows)
        with sqlite3.connect(path) as conn:
            assert create_analytics_views(conn)
        assert _query(path, "SELECT * FROM analytics_trades").empty
        assert _query(path, "SELECT * FROM analytics_strategy_stats").empty
        events = _query(path, "SELECT net_profit FROM analytics_realized_events")
        assert events["net_profit"].sum() == _approx(raw_net)
        daily = _query(path, "SELECT net_profit FROM analytics_daily_pnl")
        assert daily["net_profit"].sum() == _approx(raw_net)
        equity = _query(
            path,
            "SELECT cumulative_net_profit FROM analytics_equity ORDER BY time, ticket",
        )
        assert equity["cumulative_net_profit"].iloc[-1] == _approx(raw_net)

    def test_known_start_partial_close_then_reversal_splits_correctly(
        self, tmp_path: Path
    ) -> None:
        """With the opening visible, the same sequence reconstructs two legs."""
        kinds = [(0, 0, 2.0), (1, 1, 0.5), (1, 2, 2.0), (0, 1, 0.5)]
        rows: list[tuple[object, ...]] = [
            (
                ticket,
                71,
                "EURUSD",
                f"2024-04-01 1{ticket}:00:00",
                kind,
                entry,
                volume,
                1.2,
                0.0,
                0.0,
                0.0,
                0.0,
                7,
            )
            for ticket, (kind, entry, volume) in enumerate(kinds, start=1)
        ]
        path = _make_db(tmp_path / "known.db", rows=rows)
        with sqlite3.connect(path) as conn:
            assert create_analytics_views(conn)
        legs = _query(path, "SELECT * FROM analytics_trades ORDER BY leg_index")
        assert legs["side"].tolist() == ["buy", "sell"]
        assert legs["volume"].tolist() == _approx([2.0, 0.5])

    def test_position_start_known_flag(self, tmp_path: Path) -> None:
        """deal_portions flags positions whose first visible deal is an entry-in."""
        rows: list[tuple[object, ...]] = [
            (
                1,
                80,
                "EURUSD",
                "2024-05-01 10:00:00",
                0,
                0,
                1.0,
                1.1,
                0.0,
                0.0,
                0.0,
                0.0,
                1,
            ),
            (
                2,
                80,
                "EURUSD",
                "2024-05-01 11:00:00",
                1,
                1,
                1.0,
                1.2,
                1.0,
                0.0,
                0.0,
                0.0,
                1,
            ),
            # same position_id on another symbol starts with a reversal
            (
                3,
                80,
                "GBPUSD",
                "2024-05-01 10:00:00",
                1,
                2,
                1.0,
                1.2,
                1.0,
                0.0,
                0.0,
                0.0,
                1,
            ),
            (
                4,
                81,
                "EURUSD",
                "2024-05-01 12:00:00",
                1,
                1,
                1.0,
                1.2,
                1.0,
                0.0,
                0.0,
                0.0,
                1,
            ),
        ]
        path = _make_db(tmp_path / "flag.db", rows=rows)
        with sqlite3.connect(path) as conn:
            assert create_analytics_views(conn)
        flags = _query(
            path,
            "SELECT DISTINCT position_id, symbol, position_start_known"
            " FROM deal_portions ORDER BY position_id, symbol",
        )
        assert flags.to_numpy().tolist() == [
            [80, "EURUSD", 1],
            [80, "GBPUSD", 0],
            [81, "EURUSD", 0],
        ]
        trades = _query(path, "SELECT position_id, symbol FROM analytics_trades")
        assert trades.to_numpy().tolist() == [[80, "EURUSD"]]

    def test_mixed_magic_scale_in(self, tmp_path: Path) -> None:
        """Events keep each deal's magic; the trade leg is marked mixed (NULL)."""
        rows: list[tuple[object, ...]] = [
            (
                1,
                60,
                "EURUSD",
                "2024-03-01 10:00:00",
                0,
                0,
                1.0,
                1.1,
                0.0,
                -1.0,
                0.0,
                0.0,
                100,
            ),
            (
                2,
                60,
                "EURUSD",
                "2024-03-01 11:00:00",
                0,
                0,
                1.0,
                1.1,
                0.0,
                -1.0,
                0.0,
                0.0,
                200,
            ),
            (
                3,
                60,
                "EURUSD",
                "2024-03-02 10:00:00",
                1,
                1,
                2.0,
                1.2,
                40.0,
                -2.0,
                0.0,
                0.0,
                100,
            ),
        ]
        path = _make_db(tmp_path / "mixed.db", rows=rows)
        with sqlite3.connect(path) as conn:
            assert create_analytics_views(conn)
        events = _query(path, "SELECT * FROM analytics_realized_events ORDER BY ticket")
        assert events["magic"].tolist() == [100, 200, 100]
        daily = _query(path, "SELECT * FROM analytics_daily_pnl ORDER BY date, magic")
        assert daily[["date", "magic"]].to_numpy().tolist() == [
            ["2024-03-01", 100],
            ["2024-03-01", 200],
            ["2024-03-02", 100],
        ]
        assert daily["net_profit"].tolist() == _approx([-1.0, -1.0, 38.0])
        trade = _query(path, "SELECT * FROM analytics_trades").iloc[0]
        assert pd.isna(trade["magic"])
        assert trade["magic_count"] == 2
        assert trade["net_profit"] == _approx(36.0)
        stats = _query(path, "SELECT * FROM analytics_strategy_stats")
        assert stats["trade_count"].tolist() == [1]
        assert pd.isna(stats["magic"].iloc[0])

    def test_single_magic_leg_keeps_its_magic(self, db: Path) -> None:
        """A leg whose entries share one magic is labeled with it."""
        trades = _query(
            db, "SELECT position_id, magic, magic_count FROM analytics_trades"
        )
        assert trades["magic_count"].tolist() == [1, 1, 1, 1]
        assert trades["magic"].tolist() == [1, 1, 2, 1]


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
        trades = pd.read_parquet(out / "trades.parquet")
        stats = pd.read_parquet(out / "strategy_stats.parquet")
        assert str(trades["position_id"].dtype) == "Int64"
        assert str(trades["magic"].dtype) == "Int64"
        assert str(trades["net_profit"].dtype) == "float64"
        assert str(trades["symbol"].dtype) in {"string", "str"}
        assert str(stats["trade_count"].dtype) == "Int64"
        assert str(stats["profit_factor"].dtype) == "float64"

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
