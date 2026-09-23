"""Play-by-play opportunity / expected-TD and team-context tables on a
tiny synthetic play log."""

import polars as pl

from dfs_nfl import pbp_features as pf


def synthetic_pbp() -> pl.DataFrame:
    """Two seasons, one game each between AAA (offense) and BBB.
    Season 1: 4 goal-line carries (2 TDs), 2 red-zone targets (1 TD).
    Season 2: same plays, no TDs — so its rates must come from season 1."""
    def plays(season, tds):
        rows = []
        for i in range(4):  # rusher R1 carries from the 2
            rows.append(dict(season=season, week=1, game_id=f"g{season}",
                             season_type="REG", posteam="AAA", defteam="BBB",
                             yardline_100=2, qb_kneel=0, qb_spike=0, sack=0,
                             pass_attempt=0, rush_attempt=1, qb_dropback=0,
                             receiver_player_id=None, rusher_player_id="R1",
                             passer_player_id=None, pass_touchdown=0,
                             rush_touchdown=int(i < 2 and tds), epa=0.1,
                             success=1, wp=0.5, pass_oe=None))
        for i in range(2):  # QB Q1 targets W1 from the 15
            rows.append(dict(season=season, week=1, game_id=f"g{season}",
                             season_type="REG", posteam="AAA", defteam="BBB",
                             yardline_100=15, qb_kneel=0, qb_spike=0, sack=0,
                             pass_attempt=1, rush_attempt=0, qb_dropback=1,
                             receiver_player_id="W1", rusher_player_id=None,
                             passer_player_id="Q1", pass_touchdown=int(i == 0 and tds),
                             rush_touchdown=0, epa=-0.2, success=0, wp=0.5,
                             pass_oe=5.0))
        # a kneel: never an opportunity, never a play for pace
        rows.append(dict(season=season, week=1, game_id=f"g{season}",
                         season_type="REG", posteam="AAA", defteam="BBB",
                         yardline_100=50, qb_kneel=1, qb_spike=0, sack=0,
                         pass_attempt=0, rush_attempt=1, qb_dropback=0,
                         receiver_player_id=None, rusher_player_id="Q1",
                         passer_player_id=None, pass_touchdown=0,
                         rush_touchdown=0, epa=-1.0, success=0, wp=0.9,
                         pass_oe=None))
        return rows
    return pl.DataFrame(plays(2020, True) + plays(2021, False))


def test_td_rates_use_prior_seasons():
    rates = pf.td_rate_table(pf._scoring_plays(synthetic_pbp()))
    r = {(s, k, b): v for s, k, b, v in rates.rows()}
    assert r[(2020, "rush", "gl2")] == 0.5      # own season (earliest)
    assert r[(2021, "rush", "gl2")] == 0.5      # prior season's rate, not 0
    assert r[(2021, "rec", "rz20")] == 0.5
    assert r[(2021, "pass", "rz20")] == 0.5


def test_player_opportunity_counts_and_expected_tds():
    po = pf.player_opportunity(synthetic_pbp()).filter(pl.col("season") == 2021)
    by = {row["player_id"]: row for row in po.to_dicts()}
    assert by["R1"]["gl_carries"] == 4 and by["R1"]["rz_carries"] == 4
    assert by["R1"]["gl_carry_share"] == 1.0
    assert abs(by["R1"]["x_rush_td"] - 2.0) < 1e-9   # 4 carries x 0.5
    assert by["W1"]["rz_targets"] == 2 and by["W1"]["i10_targets"] == 0
    assert abs(by["W1"]["x_rec_td"] - 1.0) < 1e-9
    assert abs(by["Q1"]["x_pass_td"] - 1.0) < 1e-9 and by["Q1"]["rz_attempts"] == 2
    assert abs(by["Q1"]["x_td"] - 1.0) < 1e-9        # the kneel adds nothing
    assert set(po.columns) == {"season", "week", "player_id", *pf.PLAYER_OPP_COLS}


def test_team_game_tables_exclude_kneels():
    off, dfn = pf.team_game_tables(synthetic_pbp())
    o = off.filter(pl.col("season") == 2021).row(0, named=True)
    assert o["off_plays"] == 6                       # 4 carries + 2 dropbacks
    assert abs(o["off_pass_rate"] - 2 / 6) < 1e-9
    assert o["off_pass_oe"] == 5.0
    d = dfn.filter(pl.col("season") == 2021).row(0, named=True)
    assert d["team"] == "BBB"
    assert abs(d["def_pass_epa"] + 0.2) < 1e-9 and abs(d["def_rush_epa"] - 0.1) < 1e-9


def test_team_trailing_is_shifted_and_covers_unplayed_weeks(monkeypatch):
    grid = pl.DataFrame({"season": [2021] * 3, "week": [1, 2, 3],
                         "team": ["AAA"] * 3}).with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    monkeypatch.setattr(pf, "_team_week_grid", lambda: grid)
    table = pl.DataFrame({"season": [2021, 2021], "week": [1, 2], "team": ["AAA"] * 2,
                          "off_plays": [60.0, 70.0]})
    tr = pf.team_trailing(table, ["off_plays"], prefix="team_")
    vals = dict(zip(tr["week"], tr["team_off_plays_ewma"]))
    assert vals[1] is None                 # nothing before week 1
    assert vals[2] == 60.0                 # week 2 sees week 1 only
    assert 60.0 < vals[3] < 70.0           # unplayed week 3 gets the blend
