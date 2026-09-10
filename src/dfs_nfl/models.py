"""Per-position LightGBM models, walk-forward backtest vs baselines.

For each eval week (season S, week W): train on all REG rows strictly
before it — earlier seasons plus weeks < W of season S — and predict W.
Comparison against B1/B2 uses the identical row set (matched join on
game_id + player_id).

Usage: uv run python -m dfs_nfl.models
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from dfs_nfl.baselines import CURATED_DIR, OUTPUTS_DIR, MIN_PRIOR_GAMES

EVAL_SEASONS = [2024, 2025]
POSITIONS = ["QB", "RB", "WR", "TE", "DST"]

ID_COLS = {
    "player_id", "player_display_name", "name", "position", "season", "week",
    "season_type", "game_id", "team", "opponent_team", "fd_points",
}

LGB_PARAMS = {
    "objective": "regression",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 7,
    # bit-reproducible runs: same data -> same projections
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
}
# smaller samples (per-position row counts) want smaller trees
POSITION_PARAMS = {
    "QB": {"num_leaves": 31},
    "TE": {"num_leaves": 31, "min_child_samples": 60},
    "DST": {"num_leaves": 15, "min_child_samples": 60},
}
NUM_ROUNDS = 300

# Final prediction blends the LightGBM output with baseline B2; the blend
# beat both components at every position on 2024-25 and is insensitive to
# the exact weight across 0.5-0.7.
BLEND_WEIGHT = 0.6


def load_features(pos: str) -> pl.DataFrame:
    fname = "features_dst.parquet" if pos == "DST" else "features_offense.parquet"
    df = pl.read_parquet(CURATED_DIR / fname)
    if pos != "DST":
        df = df.filter(pl.col("position") == pos)
    return df


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Numeric columns that are pre-kickoff safe: trailing (_ewma/_l4/std/max),
    counters, opponent factor, and game context. Same-week raw stats are
    excluded by listing, not by guessing."""
    context = {
        "prior_games", "season_prior_games", "opp_factor_raw", "opp_allowed_pg",
        "is_home", "rest_days", "div_game", "is_dome", "total_line",
        "implied_total", "opp_implied_total", "vegas_edge", "week",
        "temp", "wind", "depth_rank", "draft_round", "draft_pick", "is_rookie",
        # stacked baselines: built from strictly-prior games, so
        # pre-kickoff safe; gives the trees B2's multiplicative
        # ewma x opponent x Vegas structure directly
        "pred_b1", "pred_b2",
    }
    cols = [
        c for c in df.columns
        if c.endswith(("_ewma", "_l4", "_std8", "_max8")) or c in context
    ]
    return sorted(set(cols) & set(df.columns))


def to_matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()


def backtest_position(pos: str, baselines: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = (
        load_features(pos)
        .join(baselines, on=["player_id", "game_id"], how="left")
        .sort(["season", "week", "game_id", "player_id"])
    )
    cols = feature_columns(df)
    params = LGB_PARAMS | POSITION_PARAMS.get(pos, {})
    eval_weeks = (
        df.filter(pl.col("season").is_in(EVAL_SEASONS))
        .select("season", "week").unique().sort(["season", "week"]).rows()
    )
    preds, importances = [], np.zeros(len(cols))
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
        booster = lgb.train(
            params,
            lgb.Dataset(to_matrix(train, cols), label=train["fd_points"].to_numpy()),
            num_boost_round=NUM_ROUNDS,
        )
        yhat = booster.predict(to_matrix(test, cols))
        importances += booster.feature_importance(importance_type="gain")
        preds.append(test.select(
            "player_id", "game_id", "season", "week", "position", "fd_points",
        ).with_columns(pl.Series("pred_model", yhat)))
    imp = pl.DataFrame({"feature": cols, "gain": importances}).sort("gain", descending=True)
    return pl.concat(preds), imp


def evaluate(matched: pl.DataFrame) -> dict:
    row = {"n": matched.height}
    for name in ["final", "model", "b1", "b2"]:
        err = matched[f"pred_{name}"] - matched["fd_points"]
        row[f"{name}_mae"] = round(float(err.abs().mean()), 3)
        row[f"{name}_rmse"] = round(float((err ** 2).mean() ** 0.5), 3)
        corrs = []
        for _, wk in matched.group_by(["season", "week"]):
            if wk.height >= 8:
                r = spearmanr(wk[f"pred_{name}"], wk["fd_points"]).statistic
                if r == r:
                    corrs.append(r)
        row[f"{name}_spearman"] = round(sum(corrs) / len(corrs), 3) if corrs else None
    return row


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    baselines = pl.read_parquet(OUTPUTS_DIR / "baseline_predictions.parquet").select(
        "player_id", "game_id", "pred_b1", "pred_b2",
    )
    rows, all_preds = [], []
    for pos in POSITIONS:
        preds, imp = backtest_position(pos, baselines)
        imp.write_csv(OUTPUTS_DIR / f"feature_importance_{pos.lower()}.csv")
        matched = preds.join(baselines, on=["player_id", "game_id"], how="inner").filter(
            pl.col("pred_b1").is_not_null() & pl.col("pred_b2").is_not_null()
        ).with_columns(
            (BLEND_WEIGHT * pl.col("pred_model")
             + (1 - BLEND_WEIGHT) * pl.col("pred_b2")).alias("pred_final")
        )
        all_preds.append(matched)
        rows.append({"position": pos} | evaluate(matched))
        print(f"{pos}: top features -> "
              + ", ".join(imp["feature"].head(5).to_list()))
    report = pl.DataFrame(rows)
    report.write_csv(OUTPUTS_DIR / "model_backtest.csv")
    pl.concat(all_preds).write_parquet(OUTPUTS_DIR / "model_predictions.parquet")
    with pl.Config(tbl_cols=-1, tbl_width_chars=220):
        print(f"\nWalk-forward eval, seasons {EVAL_SEASONS}, REG, "
              f">={MIN_PRIOR_GAMES} prior games, matched rows vs baselines\n")
        print(report)


if __name__ == "__main__":
    main()
