"""Cold-start predictions for players with <2 career games.

The main models require trailing history; rookies and first-time
contributors get a pooled LightGBM prior instead, built from role and
pedigree: depth-chart rank, draft capital, rookie flag, Vegas context,
and whatever 0-1 prior games exist. Output rows carry
confidence="low" so downstream consumers can tell them apart.

Backtest compares against the naive prior (mean cold-start points by
position in the training data).

Usage: uv run python -m dfs_nfl.coldstart
"""

from __future__ import annotations

import lightgbm as lgb
import polars as pl

from dfs_nfl.baselines import CURATED_DIR, MIN_PRIOR_GAMES, OUTPUTS_DIR
from dfs_nfl.models import EVAL_SEASONS, to_matrix

POSITION_CODES = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}

FEATURES = [
    "position_code", "depth_rank", "draft_round", "draft_pick", "is_rookie",
    "prior_games", "fd_points_ewma",  # the 0-1 games that do exist
    "implied_total", "opp_implied_total", "total_line", "is_home", "week",
]

LGB_PARAMS = {
    "objective": "regression",
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 60,
    "feature_fraction": 0.9,
    "verbose": -1,
    "seed": 7,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
}
NUM_ROUNDS = 200


def cold_rows() -> pl.DataFrame:
    df = pl.read_parquet(CURATED_DIR / "features_offense.parquet")
    return (
        df.filter(pl.col("prior_games") < MIN_PRIOR_GAMES)
        .with_columns(
            pl.col("position").replace_strict(POSITION_CODES).alias("position_code"),
            pl.col("draft_round").fill_null(8),
        )
    )


def backtest() -> pl.DataFrame:
    df = cold_rows()
    eval_weeks = (
        df.filter(pl.col("season").is_in(EVAL_SEASONS))
        .select("season", "week").unique().sort(["season", "week"]).rows()
    )
    preds = []
    for season, week in eval_weeks:
        train = df.filter(
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") < week))
        )
        test = df.filter((pl.col("season") == season) & (pl.col("week") == week))
        if not test.height or train.height < 500:
            continue
        booster = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(to_matrix(train, FEATURES), label=train["fd_points"].to_numpy()),
            num_boost_round=NUM_ROUNDS,
        )
        naive = (
            train.group_by("position").agg(pl.col("fd_points").mean().alias("pred_naive"))
        )
        preds.append(
            test.select("player_id", "game_id", "season", "week", "position", "fd_points")
            .with_columns(pl.Series("pred_cold", booster.predict(to_matrix(test, FEATURES))))
            .join(naive, on="position", how="left")
        )
    return pl.concat(preds)


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    preds = backtest()
    preds.write_parquet(OUTPUTS_DIR / "coldstart_predictions.parquet")
    rows = []
    for pos in POSITION_CODES:
        sub = preds.filter(pl.col("position") == pos)
        if not sub.height:
            continue
        rows.append({
            "position": pos,
            "n": sub.height,
            "cold_mae": round(float((sub["pred_cold"] - sub["fd_points"]).abs().mean()), 3),
            "naive_mae": round(float((sub["pred_naive"] - sub["fd_points"]).abs().mean()), 3),
            "actual_mean": round(float(sub["fd_points"].mean()), 2),
        })
    report = pl.DataFrame(rows)
    report.write_csv(OUTPUTS_DIR / "coldstart_backtest.csv")
    print(f"Cold-start eval (<{MIN_PRIOR_GAMES} career games), "
          f"seasons {EVAL_SEASONS}, vs naive position mean\n")
    print(report)


if __name__ == "__main__":
    main()
