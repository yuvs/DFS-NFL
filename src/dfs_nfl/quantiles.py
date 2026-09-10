"""Floor/ceiling quantile models (p10 / p50 / p90) per position.

Same walk-forward protocol as models.py, LightGBM quantile objective.
Quantile crossings are fixed by sorting the three predictions per row.
Report: coverage calibration (share of actuals below each predicted
quantile — ideally 0.10 / 0.50 / 0.90) and mean p10-p90 width.

Usage: uv run python -m dfs_nfl.quantiles
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from dfs_nfl.baselines import MIN_PRIOR_GAMES, OUTPUTS_DIR
from dfs_nfl.models import (
    EVAL_SEASONS,
    LGB_PARAMS,
    POSITION_PARAMS,
    POSITIONS,
    feature_columns,
    load_features,
    to_matrix,
)

ALPHAS = [0.1, 0.5, 0.9]
# Training alphas are shifted where raw quantile boosters come out
# miscalibrated on 2024-25 (the p10 floor ran ~5-8 points too high;
# QB floors need a stronger shift than the rest).
TRAIN_ALPHA = {0.1: 0.055, 0.5: 0.5, 0.9: 0.92}
TRAIN_ALPHA_P10_QB = 0.04
NUM_ROUNDS = 200


def backtest_position(pos: str, baselines: pl.DataFrame) -> pl.DataFrame:
    df = load_features(pos).join(baselines, on=["player_id", "game_id"], how="left")
    cols = feature_columns(df)
    base_params = LGB_PARAMS | POSITION_PARAMS.get(pos, {}) | {"objective": "quantile"}
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
        test = df.filter(
            (pl.col("season") == season) & (pl.col("week") == week)
            & (pl.col("prior_games") >= MIN_PRIOR_GAMES)
        )
        if not test.height:
            continue
        x_train, y_train = to_matrix(train, cols), train["fd_points"].to_numpy()
        x_test = to_matrix(test, cols)
        qhat = np.sort(np.column_stack([
            lgb.train(
                base_params | {"alpha": TRAIN_ALPHA_P10_QB
                               if (a == 0.1 and pos == "QB")
                               else TRAIN_ALPHA[a]},
                lgb.Dataset(x_train, label=y_train),
                num_boost_round=NUM_ROUNDS,
            ).predict(x_test)
            for a in ALPHAS
        ]), axis=1)  # sort enforces p10 <= p50 <= p90
        preds.append(test.select(
            "player_id", "game_id", "season", "week", "position", "fd_points",
        ).with_columns(
            pl.Series("p10", qhat[:, 0]),
            pl.Series("p50", qhat[:, 1]),
            pl.Series("p90", qhat[:, 2]),
        ))
    return pl.concat(preds)


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    baselines = pl.read_parquet(OUTPUTS_DIR / "baseline_predictions.parquet").select(
        "player_id", "game_id", "pred_b1", "pred_b2",
    )
    all_preds, rows = [], []
    for pos in POSITIONS:
        preds = backtest_position(pos, baselines)
        all_preds.append(preds)
        rows.append({
            "position": pos,
            "n": preds.height,
            "cov_p10": round(float((preds["fd_points"] < preds["p10"]).mean()), 3),
            "cov_p50": round(float((preds["fd_points"] < preds["p50"]).mean()), 3),
            "cov_p90": round(float((preds["fd_points"] < preds["p90"]).mean()), 3),
            "mean_width": round(float((preds["p90"] - preds["p10"]).mean()), 2),
        })
    report = pl.DataFrame(rows)
    report.write_csv(OUTPUTS_DIR / "quantile_backtest.csv")
    pl.concat(all_preds).write_parquet(OUTPUTS_DIR / "quantile_predictions.parquet")
    print(f"Walk-forward quantile eval, seasons {EVAL_SEASONS} "
          f"(targets: 0.10 / 0.50 / 0.90)\n")
    print(report)


if __name__ == "__main__":
    main()
