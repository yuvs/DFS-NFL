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


def tournament_pool() -> pl.DataFrame:
    """Single-game export shape: 2 teams, MVP roster position, MVP salary."""
    rows = []
    for ti, team in enumerate(["AAA", "BBB"]):
        opp = "BBB" if team == "AAA" else "AAA"
        specs = [("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1), ("K", 1)]
        for pos, count in specs:
            for k in range(count):
                proj = 18 - ti - k * 2.0 + {"QB": 0, "RB": -2, "WR": -4,
                                            "TE": -8, "K": -10}[pos]
                salary = 5000 + 1000 * (8 - ti - k)
                rows.append({
                    "fd_name": f"{team} {pos}{k}", "position": pos, "team": team,
                    "opponent_team": opp, "salary": salary,
                    "mvp_salary": int(salary * 1.5),
                    "roster_position": "MVP - 1.5X Points/AnyFLEX",
                    "projection": proj, "p10": proj - 5, "p90": proj + 6,
                    "sd": 4.0, "cv": 0.5, "injury": None,
                })
        rows.append({
            "fd_name": f"{team} D/ST", "position": "DST", "team": team,
            "opponent_team": opp, "salary": 4000, "mvp_salary": 6000,
            "roster_position": "MVP - 1.5X Points/AnyFLEX",
            "projection": 7.0 - ti, "p10": 1.0, "p90": 14.0, "sd": 4.5,
            "cv": 0.6, "injury": None,
        })
    return pl.DataFrame(rows)


def test_tournament_lineup_valid():
    lu = optimizer.solve_lineup(tournament_pool(), mode="gpp")
    assert lu is not None
    optimizer.validate_lineup(lu)
    assert lu.height == optimizer.TOURNAMENT_SIZE
    assert lu["is_mvp"].sum() == 1
    assert lu["team"].n_unique() == 2
    spend = sum(r["mvp_salary"] if r["is_mvp"] else r["salary"]
                for r in lu.to_dicts())
    assert spend <= optimizer.SALARY_CAP


def test_tournament_mvp_scores_1_5x():
    lu = optimizer.solve_lineup(tournament_pool(), mode="mean")
    s = optimizer.lineup_summary(lu)
    mvp_proj = lu.filter(pl.col("is_mvp"))["projection"][0]
    plain = lu["projection"].sum() - mvp_proj
    assert s["projection"] == round(plain + 1.5 * mvp_proj, 1)


def test_format_detection_and_mismatch():
    assert optimizer.resolve_format(synthetic_pool()) == "classic"
    assert optimizer.resolve_format(tournament_pool()) == "tournament"
    with pytest.raises(ValueError, match="not tournament"):
        optimizer.resolve_format(synthetic_pool(), "tournament")
    with pytest.raises(ValueError, match="not classic"):
        optimizer.resolve_format(tournament_pool(), "classic")


def test_generate_enforces_requested_format(monkeypatch):
    monkeypatch.setattr(optimizer, "load_pool", lambda *a, **k: synthetic_pool())
    with pytest.raises(ValueError):
        optimizer.generate(2026, 1, "gpp", 1, min_diff=1, locks=[], excludes=[],
                           stack=False, allow_dst_conflict=False,
                           slate_path=None, fmt="tournament")


def test_tournament_lineups_differ_and_write(monkeypatch, tmp_path):
    monkeypatch.setattr(optimizer, "load_pool", lambda *a, **k: tournament_pool())
    monkeypatch.setattr(optimizer, "LINEUPS_DIR", tmp_path)
    lineups = optimizer.generate(2026, 1, "gpp", 3, min_diff=2, locks=[],
                                 excludes=[], stack=False,
                                 allow_dst_conflict=False, slate_path=None,
                                 fmt="tournament")
    assert len(lineups) == 3
    for a in lineups:
        for b in lineups:
            if a is not b:
                assert len(set(a["idx"]) & set(b["idx"])) <= optimizer.TOURNAMENT_SIZE - 2
    pool = tournament_pool().with_columns(
        pl.format("1-{}", pl.int_range(pl.len())).alias("fanduel_id"))
    lineups = [lu.join(pool.select("fd_name", "fanduel_id"), on="fd_name")
               for lu in lineups]
    stem = optimizer.write_lineups(lineups, 2026, 1, "gpp")
    assert stem.name == "2026_w01_gpp_tournament_AAA-BBB"
    upload = (tmp_path / "2026_w01_gpp_tournament_AAA-BBB_upload.csv").read_text().splitlines()
    assert upload[0].split(",") == optimizer.TOURNAMENT_UPLOAD_HEADER
    assert len(upload) == 4 and all(len(r.split(",")) == 6 for r in upload[1:])


def test_classic_stem_has_no_suffix():
    lu = optimizer.solve_lineup(synthetic_pool())
    assert optimizer.lineup_stem([lu], 2026, 1, "cash").name == "2026_w01_cash"


def two_qb_pool() -> pl.DataFrame:
    """Classic pool where team AAA lists a starter and a backup QB."""
    pool = synthetic_pool().with_columns(pl.lit(None, dtype=pl.Int32).alias("depth_rank"))
    backup = pool.filter(pl.col("fd_name") == "AAA QB0").with_columns(
        pl.lit("AAA QB1").alias("fd_name"), pl.lit(3000, dtype=pl.Int64).alias("salary"),
        pl.lit(30.0).alias("projection"), pl.lit(25.0).alias("p10"),
        pl.lit(36.0).alias("p90"), pl.lit(2, dtype=pl.Int32).alias("depth_rank"))
    starter = pl.when(pl.col("fd_name") == "AAA QB0").then(1).otherwise(None)
    return pl.concat([pool.with_columns(starter.cast(pl.Int32).alias("depth_rank")),
                      backup])


def test_backup_qb_by_depth_rank():
    dropped = optimizer.backup_qbs(two_qb_pool())["fd_name"].to_list()
    assert dropped == ["AAA QB1"]


def test_backup_qb_by_salary_when_no_depth_data():
    pool = two_qb_pool().with_columns(pl.lit(None, dtype=pl.Int32).alias("depth_rank"))
    dropped = optimizer.backup_qbs(pool)["fd_name"].to_list()
    assert dropped == ["AAA QB1"]  # $3,000 vs the starter's higher salary


def test_backup_qb_without_depth_column():
    dropped = optimizer.backup_qbs(two_qb_pool().drop("depth_rank"))["fd_name"].to_list()
    assert dropped == ["AAA QB1"]


def test_load_pool_drops_backup_unless_locked_or_allowed(tmp_path, capsys):
    sheet = tmp_path / "2026_w01_slate1.csv"
    two_qb_pool().with_columns(
        pl.lit("ok").alias("confidence"),
        pl.format("1-{}", pl.int_range(pl.len())).alias("fanduel_id"),
    ).write_csv(sheet)
    names = optimizer.load_pool(2026, 1, sheet)["fd_name"].to_list()
    assert "AAA QB1" not in names and "AAA QB0" in names
    assert "skipping 1 backup QB(s): AAA QB1 (AAA)" in capsys.readouterr().out
    kept = optimizer.load_pool(2026, 1, sheet, keep=["AAA QB1"])["fd_name"].to_list()
    assert "AAA QB1" in kept
    allowed = optimizer.load_pool(2026, 1, sheet, allow_backup_qb=True)["fd_name"].to_list()
    assert "AAA QB1" in allowed


def test_generate_skips_juicy_backup(tmp_path):
    sheet = tmp_path / "2026_w01_slate1.csv"
    two_qb_pool().with_columns(
        pl.lit("ok").alias("confidence"),
        pl.format("1-{}", pl.int_range(pl.len())).alias("fanduel_id"),
    ).write_csv(sheet)
    (lu,) = optimizer.generate(2026, 1, "gpp", 1, min_diff=1, locks=[], excludes=[],
                               stack=False, allow_dst_conflict=False, slate_path=sheet)
    assert "AAA QB1" not in lu["fd_name"].to_list()


def test_questionable_starter_still_blocks_backup_in_cash(tmp_path):
    """safe_only removes the Q starter, but the backup must not be
    promoted: if the starter suits up, the backup scores nothing."""
    sheet = tmp_path / "2026_w01_slate1.csv"
    two_qb_pool().with_columns(
        pl.lit("ok").alias("confidence"),
        pl.format("1-{}", pl.int_range(pl.len())).alias("fanduel_id"),
        pl.when(pl.col("fd_name") == "AAA QB0").then(pl.lit("Q"))
        .otherwise(pl.col("injury")).alias("injury"),
    ).write_csv(sheet)
    names = optimizer.load_pool(2026, 1, sheet, safe_only=True)["fd_name"].to_list()
    assert "AAA QB0" not in names and "AAA QB1" not in names


def test_out_starter_promotes_backup(tmp_path):
    sheet = tmp_path / "2026_w01_slate1.csv"
    two_qb_pool().with_columns(
        pl.lit("ok").alias("confidence"),
        pl.format("1-{}", pl.int_range(pl.len())).alias("fanduel_id"),
        pl.when(pl.col("fd_name") == "AAA QB0").then(pl.lit("O"))
        .otherwise(pl.col("injury")).alias("injury"),
    ).write_csv(sheet)
    names = optimizer.load_pool(2026, 1, sheet)["fd_name"].to_list()
    assert "AAA QB0" not in names and "AAA QB1" in names
