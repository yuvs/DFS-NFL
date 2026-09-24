"""Team-level Vegas anchor: line fit, per-team factors, and the
production path (full week saves the line, a single game reuses it)."""

import json

import numpy as np
import polars as pl
import pytest

from dfs_nfl import models, predict


def _week(n_teams=10, noise=None, season=2026, week=3):
    """Two offense players per team whose sum lies on 20 + 2*implied,
    plus optional per-team noise and one DST row per team."""
    implied = np.linspace(17, 28, n_teams)
    noise = np.zeros(n_teams) if noise is None else np.asarray(noise, float)
    rows = []
    for i, (imp, eps) in enumerate(zip(implied, noise)):
        total = 20 + 2 * imp + eps
        for pos, share in [("QB", 0.6), ("WR", 0.4)]:
            rows.append({"season": season, "week": week, "game_id": f"g{i // 2}",
                         "team": f"T{i}", "position": pos, "player_id": f"{pos}{i}",
                         "implied_total": float(imp), "inj_status": 0,
                         "projection": total * share, "p10": total * share / 2,
                         "p50": total * share, "p90": total * share * 2})
        rows.append({"season": season, "week": week, "game_id": f"g{i // 2}",
                     "team": f"T{i}", "position": "DST", "player_id": f"T{i}",
                     "implied_total": float(imp), "inj_status": 0,
                     "projection": 7.0, "p10": 1.0, "p50": 7.0, "p90": 14.0})
    return pl.DataFrame(rows)


def test_line_recovered_and_on_line_teams_untouched():
    df = _week()
    sums = models.team_sums(df, "projection")
    assert sums.height == 10  # DST excluded from sums
    a, b = models.fit_anchor_line(sums)
    assert a == pytest.approx(20) and b == pytest.approx(2)
    f = models.anchor_factors(sums, (a, b))
    assert np.allclose(f["anchor_factor"], 1.0)


def test_factor_closes_lambda_of_the_gap_and_clips():
    sums = pl.DataFrame({"game_id": ["g", "g", "h"], "team": ["A", "B", "C"],
                         "implied_total": [20.0, 20.0, 20.0],
                         "team_proj": [70.0, 50.0, 400.0]})
    f = models.anchor_factors(sums, (20.0, 2.0), lam=0.75)["anchor_factor"].to_list()
    # A: 70 -> 60 + 0.25*10 = 62.5; B: 50 -> 57.5; C clipped at the floor
    assert f[0] * 70 == pytest.approx(62.5)
    assert f[1] * 50 == pytest.approx(57.5)
    assert f[2] == models.ANCHOR_CLIP[0]


def test_injured_players_dont_count_and_too_few_teams_gives_no_line():
    df = _week(n_teams=4)
    assert models.fit_anchor_line(models.team_sums(df, "projection")) is None
    out_row = df.filter(pl.col("player_id") == "QB0").with_columns(
        pl.lit("OUT0").alias("player_id"), pl.lit(3, dtype=pl.Int64).alias("inj_status"),
        pl.lit(99.0).alias("projection"))
    sums = models.team_sums(pl.concat([df, out_row]), "projection")
    t0 = sums.filter(pl.col("team") == "T0")["team_proj"][0]
    assert t0 == pytest.approx(20 + 2 * 17)


def test_apply_vegas_anchor_scales_offense_only():
    noise = [10, -10, 0, 0, 0, 0, 0, 0, 0, 0]
    df = _week(noise=noise).with_columns(pl.col("projection").alias("pred"))
    out = models.apply_vegas_anchor(df, "pred", "pred_anch", lam=1.0)
    dst = out.filter(pl.col("position") == "DST")
    assert (dst["pred"] == dst["pred_anch"]).all()
    t0 = out.filter((pl.col("team") == "T0") & (pl.col("position") != "DST"))
    t1 = out.filter((pl.col("team") == "T1") & (pl.col("position") != "DST"))
    assert t0["pred_anch"].sum() < t0["pred"].sum()   # above the line -> down
    assert t1["pred_anch"].sum() > t1["pred"].sum()   # below the line -> up
    # within-team shares are preserved
    assert t0["pred_anch"][0] / t0["pred_anch"].sum() == pytest.approx(0.6)


def test_production_full_week_saves_line_and_single_game_reuses_it(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "PROJECTIONS_DIR", tmp_path)
    noise = [8, -8, 0, 0, 0, 0, 0, 0, 0, 0]
    full = predict._anchor_to_vegas(_week(noise=noise), 2026, 3, teams=None)
    saved = json.loads(predict.anchor_line_path(2026, 3).read_text())
    assert saved["teams"] == 10
    # quantiles move with the projection; DST untouched
    t0 = full.filter((pl.col("team") == "T0") & (pl.col("position") == "QB"))
    assert t0["p90"][0] == pytest.approx(2 * t0["projection"][0])
    assert (full.filter(pl.col("position") == "DST")["anchor_factor"] == 1.0).all()

    one_game = _week(noise=noise).filter(pl.col("game_id") == "g0")
    single = predict._anchor_to_vegas(one_game, 2026, 3, teams=["T0", "T1"])
    got = single.filter(pl.col("position") == "QB").sort("team")["projection"].to_list()
    want = full.filter((pl.col("position") == "QB") & pl.col("team").is_in(["T0", "T1"])
                       ).sort("team")["projection"].to_list()
    assert got == pytest.approx(want)


def test_single_game_without_saved_line_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "PROJECTIONS_DIR", tmp_path)
    one_game = _week(noise=[8, -8] + [0] * 8).filter(pl.col("game_id") == "g0")
    out = predict._anchor_to_vegas(one_game, 2026, 3, teams=["T0", "T1"])
    assert (out["anchor_factor"] == 1.0).all()
    assert out["projection"].to_list() == one_game["projection"].to_list()
