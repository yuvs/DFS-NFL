"""Tuesday accuracy job: score last week's projections against actuals.

Joins outputs/projections/{season}_w{week:02d}.parquet with the curated
label tables (refresh + rebuild labels first so the completed week is
in), prints per-position MAE / Spearman / top-5 hit rate and the worst
misses, and appends to outputs/accuracy_log.csv.

Lineup scorecard: every lineup built for the week (outputs/lineups/) is
scored on actual points and compared with two references built from the
same value sheet under the same roster rules — the hindsight-optimal
lineup (the ceiling: max actual points) and a lineup optimized on
FanDuel's own FPPG column (their trailing average, the public
baseline). Appends to outputs/lineup_log.csv. There is no salary
history before the saved slates, so this is the live lineup backtest.

Usage: uv run python -m dfs_nfl.scorecard <season> <week>
"""

from __future__ import annotations

import sys
from datetime import date

import polars as pl
from scipy.stats import spearmanr

from dfs_nfl.baselines import CURATED_DIR, OUTPUTS_DIR
from dfs_nfl.models import top_hit_rate
from dfs_nfl.predict import PROJECTIONS_DIR

LOG_PATH = OUTPUTS_DIR / "accuracy_log.csv"
LINEUP_LOG_PATH = OUTPUTS_DIR / "lineup_log.csv"


def actuals(season: int, week: int) -> pl.DataFrame:
    players = (
        pl.read_parquet(CURATED_DIR / "player_week.parquet")
        .filter((pl.col("season") == season) & (pl.col("week") == week))
        .select("player_id", "game_id", pl.col("fd_points").alias("actual"))
    )
    dst = (
        pl.read_parquet(CURATED_DIR / "dst_week.parquet")
        .filter((pl.col("season") == season) & (pl.col("week") == week))
        .select(pl.col("team").alias("player_id"), "game_id",
                pl.col("fd_points").alias("actual"))
    )
    return pl.concat([players, dst])


def score_week(season: int, week: int) -> pl.DataFrame:
    path = PROJECTIONS_DIR / f"{season}_w{week:02d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no projections at {path}")
    proj = pl.read_parquet(path)
    scored = proj.join(actuals(season, week), on=["player_id", "game_id"], how="inner")
    if not scored.height:
        raise ValueError("no projection rows matched actuals — is the week complete?")

    rows = []
    for pos in ["QB", "RB", "WR", "TE", "DST"]:
        sub = scored.filter(pl.col("position") == pos)
        if not sub.height:
            continue
        err = sub["projection"] - sub["actual"]
        r = spearmanr(sub["projection"], sub["actual"]).statistic if sub.height >= 8 else None
        rows.append({
            "run_date": date.today().isoformat(),
            "season": season, "week": week, "position": pos,
            "n": sub.height,
            "mae": round(float(err.abs().mean()), 3),
            "rmse": round(float((err ** 2).mean() ** 0.5), 3),
            "spearman": round(float(r), 3) if r == r and r is not None else None,
            "cov_p10": round(float((sub["actual"] < sub["p10"]).mean()), 3),
            "cov_p90": round(float((sub["actual"] < sub["p90"]).mean()), 3),
            "top5_hit": top_hit_rate(sub["projection"], sub["actual"]),
        })
    report = pl.DataFrame(rows)

    if LOG_PATH.exists():
        log = pl.read_csv(LOG_PATH).filter(
            ~((pl.col("season") == season) & (pl.col("week") == week))
        )
        # diagonal: older log rows predate columns added later (top5_hit)
        report = pl.concat([log, report], how="diagonal_relaxed")
    report.write_csv(LOG_PATH)

    print(f"\n{season} week {week} scorecard:\n")
    print(report.filter((pl.col("season") == season) & (pl.col("week") == week)))
    misses = scored.with_columns(
        (pl.col("projection") - pl.col("actual")).abs().alias("abs_err")
    ).sort("abs_err", descending=True)
    print("\nworst misses:")
    print(misses.select("name", "position", "team", "projection", "actual").head(8))
    try:
        score_lineups(season, week)
    except Exception as e:  # noqa: BLE001 - never fail the accuracy log on this
        print(f"lineup scorecard skipped: {type(e).__name__}: {e}")
    return report


def _lineup_actual(rows: pl.DataFrame) -> float:
    mult = (pl.when(pl.col("is_mvp")).then(1.5).otherwise(1.0)
            if "is_mvp" in rows.columns else pl.lit(1.0))
    return float((rows["actual"].fill_null(0.0) * rows.select(mult).to_series()).sum())


def _reference_lineup(pool: pl.DataFrame, score_col: str) -> float | None:
    """Actual points of the lineup that maximizes score_col over the pool
    under the same roster rules; None if infeasible or column missing."""
    from dfs_nfl.optimizer import solve_lineup, validate_lineup

    if score_col not in pool.columns:
        return None
    ref = pool.with_columns(pl.col(score_col).fill_null(0.0).alias("projection"))
    lu = solve_lineup(ref, mode="mean")
    if lu is None:
        return None
    validate_lineup(lu)
    return _lineup_actual(lu)


def score_lineups(season: int, week: int) -> pl.DataFrame | None:
    """Grade every lineup file for the week against actual points, the
    hindsight-optimal lineup and the FPPG-baseline lineup from its slate."""
    from dfs_nfl.optimizer import LINEUPS_DIR, load_pool, lineup_summary
    from dfs_nfl.value import VALUE_DIR

    files = sorted(f for f in LINEUPS_DIR.glob(f"{season}_w{week:02d}_*.csv")
                   if not f.stem.endswith("_upload"))
    if not files:
        print("no lineups to score for this week")
        return None
    act = actuals(season, week).select("player_id", "actual")
    refs: dict[str, tuple[float | None, float | None]] = {}
    rows = []
    for f in files:
        lus = pl.read_csv(f).join(act, on="player_id", how="left")
        slate = str(lus["fanduel_id"][0]).split("-")[0]
        fmt = "tournament" if "is_mvp" in lus.columns else "classic"
        mode = f.stem.split("_")[2]
        if slate not in refs:
            sheet = VALUE_DIR / f"{season}_w{week:02d}_slate{slate}.csv"
            pool = load_pool(season, week, sheet).join(act, on="player_id", how="left")
            refs[slate] = (_reference_lineup(pool, "actual"),
                           _reference_lineup(pool, "fppg"))
        best, fppg = refs[slate]
        for (k,), lu in lus.group_by("lineup", maintain_order=True):
            s = lineup_summary(lu)
            actual = _lineup_actual(lu)
            rows.append({
                "run_date": date.today().isoformat(),
                "season": season, "week": week, "file": f.stem,
                "format": fmt, "mode": mode, "lineup": int(k),
                "salary": s["salary"], "projected": s["projection"],
                "actual": round(actual, 2),
                "hindsight_max": None if best is None else round(best, 2),
                "captured": None if not best else round(actual / best, 3),
                "fppg_lineup": None if fppg is None else round(fppg, 2),
                "players_out": int(lu["actual"].is_null().sum()),
            })
    report = pl.DataFrame(rows)
    if LINEUP_LOG_PATH.exists():
        log = pl.read_csv(LINEUP_LOG_PATH).filter(
            ~((pl.col("season") == season) & (pl.col("week") == week))
        )
        report = pl.concat([log, report], how="diagonal_relaxed")
    report.write_csv(LINEUP_LOG_PATH)
    this = report.filter((pl.col("season") == season) & (pl.col("week") == week))
    print(f"\n{season} week {week} lineup scorecard "
          "(actual vs hindsight ceiling and FanDuel-FPPG lineup):")
    with pl.Config(tbl_rows=40, tbl_width_chars=160):
        print(this.select("file", "lineup", "projected", "actual", "hindsight_max",
                          "captured", "fppg_lineup", "players_out"))
    return this


def main() -> None:
    season, week = int(sys.argv[1]), int(sys.argv[2])
    score_week(season, week)


if __name__ == "__main__":
    main()
