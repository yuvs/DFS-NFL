"""Hand-computed FanDuel scoring cases (2024+ rules)."""

import polars as pl
import pytest

from dfs_nfl import scoring


class TestOffense:
    def test_qb_big_game_with_300_bonus(self):
        # 325 pass yds, 3 pass TD, 1 INT, 20 rush yds:
        # 13.0 + 12 - 1 + 2.0 + 3 (300+ bonus) = 29.0
        assert scoring.offense_points(
            passing_yards=325, passing_tds=3, passing_interceptions=1,
            rushing_yards=20,
        ) == 29.0

    def test_qb_just_under_bonus(self):
        # 299 yds = 11.96, no bonus
        assert scoring.offense_points(passing_yards=299) == 11.96

    def test_qb_exactly_300_gets_bonus(self):
        assert scoring.offense_points(passing_yards=300) == 15.0

    def test_rb_hundred_yard_game(self):
        # 100 rush yds + bonus + 2 TD + 3 rec, 25 rec yds:
        # 10 + 3 + 12 + 1.5 + 2.5 = 29.0
        assert scoring.offense_points(
            rushing_yards=100, rushing_tds=2, receptions=3, receiving_yards=25,
        ) == 29.0

    def test_wr_half_ppr_and_rec_bonus(self):
        # 8 rec, 112 yds, 1 TD: 4.0 + 11.2 + 6 + 3 = 24.2
        assert scoring.offense_points(
            receptions=8, receiving_yards=112, receiving_tds=1,
        ) == 24.2

    def test_negative_plays(self):
        # 2 INT, 1 fumble lost, 10 pass yds: 0.4 - 2 - 2 = -3.6
        assert scoring.offense_points(
            passing_yards=10, passing_interceptions=2, fumbles_lost=1,
        ) == -3.6

    def test_two_point_and_return_td(self):
        assert scoring.offense_points(
            two_point_conversions=1, return_tds=1,
        ) == 8.0

    def test_double_bonus_rush_and_rec(self):
        # 105 rush + 101 rec, both bonuses: 10.5 + 10.1 + 3 + 3 = 26.6
        assert scoring.offense_points(
            rushing_yards=105, receiving_yards=101,
        ) == 26.6


class TestDst:
    def test_shutout(self):
        # 4 sacks, 2 INT, 1 fumble rec, 0 allowed: 4 + 4 + 2 + 10 = 20
        assert scoring.dst_points(
            sacks=4, interceptions=2, fumble_recoveries=1, points_allowed=0,
        ) == 20.0

    def test_blowup_defense(self):
        # 1 sack, 35 allowed: 1 - 4 = -3
        assert scoring.dst_points(sacks=1, points_allowed=35) == -3.0

    def test_all_brackets(self):
        expected = {0: 10, 3: 7, 6: 7, 7: 4, 13: 4, 14: 1, 20: 1,
                    21: 0, 27: 0, 28: -1, 34: -1, 35: -4, 62: -4}
        for pa, pts in expected.items():
            assert scoring.points_allowed_score(pa) == pts, pa

    def test_events(self):
        assert scoring.dst_points(
            safeties=1, blocked_kicks=1, tds=2, points_allowed=21,
        ) == 16.0


class TestExprsMatchScalars:
    def test_offense_expr_matches_scalar(self):
        rows = pl.DataFrame({
            "passing_yards": [325.0, 0.0, 0.0],
            "passing_tds": [3, 0, 0],
            "passing_interceptions": [1, 0, 0],
            "passing_2pt_conversions": [0, 0, 0],
            "rushing_yards": [20.0, 100.0, 0.0],
            "rushing_tds": [0, 2, 0],
            "rushing_2pt_conversions": [0, 0, 0],
            "receptions": [0, 3, 8],
            "receiving_yards": [0.0, 25.0, 112.0],
            "receiving_tds": [0, 0, 1],
            "receiving_2pt_conversions": [0, 0, 0],
            "fumbles_lost_total": [0, 0, 0],
            "special_teams_tds": [0, 0, 0],
            "fumble_recovery_tds": [0, 0, 0],
        })
        out = rows.with_columns(scoring.offense_points_expr().alias("fd"))
        assert out["fd"].to_list() == [29.0, 29.0, 24.2]

    def test_pa_bracket_expr(self):
        df = pl.DataFrame({"points_allowed": [0, 6, 7, 20, 27, 34, 50]})
        out = df.with_columns(scoring.points_allowed_score_expr().alias("s"))
        assert out["s"].to_list() == [10.0, 7.0, 4.0, 1.0, 0.0, -1.0, -4.0]

    def test_null_handling(self):
        row = pl.DataFrame({c: [None] for c in [
            "passing_yards", "passing_tds", "passing_interceptions",
            "passing_2pt_conversions", "rushing_yards", "rushing_tds",
            "rushing_2pt_conversions", "receptions", "receiving_yards",
            "receiving_tds", "receiving_2pt_conversions",
            "fumbles_lost_total", "special_teams_tds", "fumble_recovery_tds",
        ]}, schema={c: pl.Float64 for c in [
            "passing_yards", "passing_tds", "passing_interceptions",
            "passing_2pt_conversions", "rushing_yards", "rushing_tds",
            "rushing_2pt_conversions", "receptions", "receiving_yards",
            "receiving_tds", "receiving_2pt_conversions",
            "fumbles_lost_total", "special_teams_tds", "fumble_recovery_tds",
        ]})
        out = row.with_columns(scoring.offense_points_expr().alias("fd"))
        assert out["fd"].to_list() == [0.0]


def test_negative_points_allowed_raises():
    with pytest.raises(ValueError):
        scoring.points_allowed_score(-1)
