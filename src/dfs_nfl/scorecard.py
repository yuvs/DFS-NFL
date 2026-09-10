"""Tuesday accuracy job: score last week's projections against actuals.

Joins outputs/projections/{season}_w{week:02d}.parquet with the curated
label tables (refresh + rebuild labels first so the completed week is
in), prints per-position MAE/Spearman and the worst misses, and appends
to outputs/accuracy_log.csv.

Usage: uv run python -m dfs_nfl.scorecard <season> <week>
"""

from __future__ import annotations

import sys
from datetime import date

import polars as pl
from scipy.stats import spearmanr

from dfs_nfl.baselines import CURATED_DIR, OUTPUTS_DIR
from dfs_nfl.predict import PROJECTIONS_DIR

LOG_PATH = OUTPUTS_DIR / "accuracy_log.csv"


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
        })
    report = pl.DataFrame(rows)

    if LOG_PATH.exists():
        log = pl.read_csv(LOG_PATH).filter(
            ~((pl.col("season") == season) & (pl.col("week") == week))
        )
        report = pl.concat([log.cast(report.schema), report], how="vertical_relaxed")
    report.write_csv(LOG_PATH)

    print(f"\n{season} week {week} scorecard:\n")
    print(report.filter((pl.col("season") == season) & (pl.col("week") == week)))
    misses = scored.with_columns(
        (pl.col("projection") - pl.col("actual")).abs().alias("abs_err")
    ).sort("abs_err", descending=True)
    print("\nworst misses:")
    print(misses.select("name", "position", "team", "projection", "actual").head(8))
    return report


def main() -> None:
    season, week = int(sys.argv[1]), int(sys.argv[2])
    score_week(season, week)


if __name__ == "__main__":
    main()
