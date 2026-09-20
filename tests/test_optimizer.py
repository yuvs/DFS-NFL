"""Optimizer constraint tests on a small synthetic slate."""

import polars as pl
import pytest

from dfs_nfl import optimizer


def synthetic_pool() -> pl.DataFrame:
    """4 teams x (1 QB, 2 RB, 3 WR, 2 TE) + 4 DSTs, all affordable."""
    rows = []
    teams = ["AAA", "BBB", "CCC", "DDD"]
    opps = {"AAA": "BBB", "BBB": "AAA", "CCC": "DDD", "DDD": "CCC"}
    for ti, team in enumerate(teams):
        specs = [("QB", 1), ("RB", 2), ("WR", 3), ("TE", 2)]
        for pos, count in specs:
            for k in range(count):
                proj = 20 - ti - k * 2.0 + {"QB": 0, "RB": -2, "WR": -4, "TE": -8}[pos]
                rows.append({
                    "fd_name": f"{team} {pos}{k}", "position": pos, "team": team,
                    "opponent_team": opps[team], "salary": 4000 + 200 * (10 - ti - k),
                    "projection": proj, "p10": proj - 5, "p90": proj + 6,
                    "sd": 4.0, "cv": 0.5, "injury": None,
                })
        rows.append({
            "fd_name": f"{team} D/ST", "position": "DST", "team": team,
            "opponent_team": opps[team], "salary": 3500, "projection": 7.0 - ti,
            "p10": 1.0, "p90": 14.0, "sd": 4.5, "cv": 0.6, "injury": None,
        })
    return pl.DataFrame(rows)


def test_optimal_lineup_valid():
    lu = optimizer.solve_lineup(synthetic_pool(), mode="cash")
    assert lu is not None
    optimizer.validate_lineup(lu)


def test_salary_cap_binds():
    pool = synthetic_pool().with_columns(pl.col("salary") * 2)
    lu = optimizer.solve_lineup(pool, mode="mean")
    assert lu is None or lu["salary"].sum() <= optimizer.SALARY_CAP


def test_lock_and_exclude():
    pool = synthetic_pool()
    lu = optimizer.solve_lineup(pool, locks=["BBB QB0"], excludes=["AAA WR0"])
    names = lu["fd_name"].to_list()
    assert "BBB QB0" in names and "AAA WR0" not in names


def test_unknown_lock_raises():
    with pytest.raises(ValueError):
        optimizer.solve_lineup(synthetic_pool(), locks=["Nobody Real"])


def test_stack_forces_qb_teammate():
    lu = optimizer.solve_lineup(synthetic_pool(), stack=True)
    qb_team = lu.filter(pl.col("position") == "QB")["team"][0]
    mates = lu.filter((pl.col("team") == qb_team)
                      & pl.col("position").is_in(["WR", "TE"]))
    assert mates.height >= 1


def test_dst_conflict_blocked():
    lu = optimizer.solve_lineup(synthetic_pool())
    dst_opp = lu.filter(pl.col("position") == "DST")["opponent_team"][0]
    facing = lu.filter((pl.col("team") == dst_opp) & (pl.col("position") != "DST"))
    assert facing.height == 0


def test_multiple_lineups_differ():
    pool = synthetic_pool()
    first = optimizer.solve_lineup(pool)
    second = optimizer.solve_lineup(pool, banned_sets=[first["idx"].to_list()],
                                    min_diff=2)
    assert second is not None
    overlap = len(set(first["idx"]) & set(second["idx"]))
    assert overlap <= optimizer.ROSTER_SIZE - 2


def test_injured_players_filtered(tmp_path):
    pool = synthetic_pool().with_columns(
        pl.when(pl.col("fd_name") == "AAA QB0").then(pl.lit("O"))
        .otherwise(pl.lit(None)).alias("injury")
    )
    # load_pool applies the filter; emulate it here
    filtered = pool.filter(
        pl.col("injury").is_null() | ~pl.col("injury").is_in(optimizer.OUT_TAGS))
    assert "AAA QB0" not in filtered["fd_name"].to_list()


def test_max_exposure_caps_appearances(monkeypatch):
    pool = synthetic_pool()
    monkeypatch.setattr(optimizer, "load_pool", lambda *a, **k: pool)
    n = 4
    lineups = optimizer.generate(2026, 1, "gpp", n, min_diff=1, locks=[],
                                 excludes=[], stack=False,
                                 allow_dst_conflict=False, slate_path=None,
                                 max_exposure=0.5)
    assert len(lineups) == n
    counts: dict[str, int] = {}
    for lu in lineups:
        for name in lu["fd_name"]:
            counts[name] = counts.get(name, 0) + 1
    assert max(counts.values()) <= 2


def test_max_exposure_exempts_locks(monkeypatch):
    pool = synthetic_pool()
    monkeypatch.setattr(optimizer, "load_pool", lambda *a, **k: pool)
    lineups = optimizer.generate(2026, 1, "gpp", 3, min_diff=1,
                                 locks=["AAA QB0"], excludes=[], stack=False,
                                 allow_dst_conflict=False, slate_path=None,
                                 max_exposure=0.34)
    assert len(lineups) == 3
    assert all("AAA QB0" in lu["fd_name"].to_list() for lu in lineups)
