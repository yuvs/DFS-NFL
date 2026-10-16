"""Floor/ceiling quantile models (p10 / p50 / p90) per position.

Same walk-forward protocol as models.py, LightGBM quantile objective.
Quantile crossings are fixed by sorting the three predictions per row.
Report: coverage calibration (share of actuals below each predicted
quantile — ideally 0.10 / 0.50 / 0.90) and mean p10-p90 width, scored
separately on EVAL_SEASONS and on the in-season HOLDOUT_SEASONS.

Usage:
  uv run python -m dfs_nfl.quantiles              # backtest + coverage
  uv run python -m dfs_nfl.quantiles calibrate    # sweep the p10 alphas

The calibrate pass is how TRAIN_ALPHA below gets set: it reports p10
coverage per candidate training alpha on both windows, so the choice is
read off the alpha-to-coverage curve instead of guessed.
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from dfs_nfl.baselines import OUTPUTS_DIR
from dfs_nfl.models import (
    EVAL_SEASONS,
    HOLDOUT_SEASONS,
    LGB_PARAMS,
    POSITION_PARAMS,
    POSITIONS,
    feature_columns,
    load_features,
    split_week,
    to_matrix,
    walk_forward_weeks,
)

ALPHAS = [0.1, 0.5, 0.9]
# Training alphas are shifted where raw quantile boosters come out
# miscalibrated. Set by the `calibrate` sweep below, minimax across the
# eval and holdout windows (see recommend()).
#
# 0.055 for the non-QB p10 is unchanged: it is at or within noise of the
# minimax pick for RB (worst gap 0.008), WR (0.005) and DST (0.020), and
# TE is a dead heat with 0.04 (0.025 vs 0.024) — not worth moving on.
#
# QB's p10 moved 0.04 -> 0.055 (2026-10). 0.04 was fit on 2024-25 alone,
# where it was nearly exact (coverage 0.098), but it under-covers badly
# on the 2026 holdout (0.051); 0.055 trades that for 0.131 / 0.091 and
# cuts the worst-case gap from 0.049 to 0.031.
#
# Do NOT try to tune the non-QB p10 tighter against coverage: 12% (RB),
# 23% (WR) and 24% (TE) of actual scores are *exactly* zero, so the
# empirical 10th percentile of those distributions is 0.0 and p10
# coverage lands inside that atom. It swings 0.02-0.05 on p10 changes
# too small to see (two runs correlated 0.99 in p10 differed that much
# in coverage), which makes it a knife-edge metric rather than a
# calibration signal. QB is the only position with a continuous 10th
# percentile (0.6% zeros, empirical q10 = 2.3), so it's the only one
# where this sweep means anything.
TRAIN_ALPHA = {0.1: 0.055, 0.5: 0.5, 0.9: 0.92}
TRAIN_ALPHA_P10_QB = 0.055
NUM_ROUNDS = 200

# Candidate p10 training alphas for the calibrate sweep.
P10_CANDIDATES = [0.04, 0.055, 0.07, 0.085, 0.10]

WINDOWS = {"eval": EVAL_SEASONS, "holdout": HOLDOUT_SEASONS}


def train_alpha(pos: str, a: float) -> float:
    """Training alpha for a nominal quantile at this position."""
    if a == 0.1 and pos == "QB":
        return TRAIN_ALPHA_P10_QB
    return TRAIN_ALPHA[a]


def _quantile_params(pos: str) -> dict:
    return LGB_PARAMS | POSITION_PARAMS.get(pos, {}) | {"objective": "quantile"}


def _fit_quantile(x_train: np.ndarray, y_train: np.ndarray,
                  x_test: np.ndarray, params: dict, alpha: float) -> np.ndarray:
    return lgb.train(
        params | {"alpha": alpha},
        lgb.Dataset(x_train, label=y_train),
        num_boost_round=NUM_ROUNDS,
    ).predict(x_test)


def backtest_position(pos: str, baselines: pl.DataFrame,
                      seasons: list[int]) -> pl.DataFrame:
    df = (load_features(pos)
          .join(baselines, on=["player_id", "game_id"], how="left")
          .sort(["season", "week", "game_id", "player_id"]))
    cols = feature_columns(df)
    params = _quantile_params(pos)
    preds = []
    for season, week in walk_forward_weeks(df, seasons):
        train, test = split_week(df, season, week)
        if not test.height:
            continue
        x_train, y_train = to_matrix(train, cols), train["fd_points"].to_numpy()
        x_test = to_matrix(test, cols)
        qhat = np.sort(np.column_stack([
            _fit_quantile(x_train, y_train, x_test, params, train_alpha(pos, a))
            for a in ALPHAS
        ]), axis=1)  # sort enforces p10 <= p50 <= p90
        preds.append(test.select(
            "player_id", "game_id", "season", "week", "position", "fd_points",
        ).with_columns(
            pl.Series("p10", qhat[:, 0]),
            pl.Series("p50", qhat[:, 1]),
            pl.Series("p90", qhat[:, 2]),
        ))
    return pl.concat(preds) if preds else pl.DataFrame()


def coverage(preds: pl.DataFrame, pos: str, window: str) -> dict:
    """Share of actuals below each predicted quantile, plus the
    week-clustered standard error of the p10 share (rows within a week
    share that week's game environment, so pooling them as independent
    understates the uncertainty).

    `pct_zero` is the share of actuals that are exactly 0.0 and it is
    the health warning on cov_p10: once it clears ~0.10 the true 10th
    percentile is inside that atom, so cov_p10 stops being a
    calibration signal (see the TRAIN_ALPHA note).
    """
    per_week = [float((g["fd_points"] < g["p10"]).mean())
                for _, g in preds.group_by(["season", "week"])]
    se = (float(np.std(per_week, ddof=1) / np.sqrt(len(per_week)))
          if len(per_week) > 1 else None)
    return {
        "window": window,
        "position": pos,
        "n": preds.height,
        "weeks": len(per_week),
        "pct_zero": round(float((preds["fd_points"] == 0).mean()), 3),
        "cov_p10": round(float((preds["fd_points"] < preds["p10"]).mean()), 3),
        "cov_p10_se": round(se, 4) if se is not None else None,
        "cov_p50": round(float((preds["fd_points"] < preds["p50"]).mean()), 3),
        "cov_p90": round(float((preds["fd_points"] < preds["p90"]).mean()), 3),
        "mean_width": round(float((preds["p90"] - preds["p10"]).mean()), 2),
    }


def calibrate_position(pos: str, baselines: pl.DataFrame,
                       seasons: list[int], window: str) -> list[dict]:
    """p10 coverage per candidate training alpha, walk-forward. Only the
    p10 booster is re-fit per candidate — p50/p90 don't move."""
    df = (load_features(pos)
          .join(baselines, on=["player_id", "game_id"], how="left")
          .sort(["season", "week", "game_id", "player_id"]))
    cols = feature_columns(df)
    params = _quantile_params(pos)
    # per candidate: one (below, n) pair per scored week
    tally: dict[float, list[tuple[int, int]]] = {a: [] for a in P10_CANDIDATES}
    for season, week in walk_forward_weeks(df, seasons):
        train, test = split_week(df, season, week)
        if not test.height:
            continue
        x_train, y_train = to_matrix(train, cols), train["fd_points"].to_numpy()
        x_test, y_test = to_matrix(test, cols), test["fd_points"].to_numpy()
        for a in P10_CANDIDATES:
            yhat = _fit_quantile(x_train, y_train, x_test, params, a)
            tally[a].append((int((y_test < yhat).sum()), len(y_test)))
    rows = []
    for a, cells in tally.items():
        n = sum(c[1] for c in cells)
        if not n:
            continue
        shares = [c[0] / c[1] for c in cells if c[1]]
        se = (float(np.std(shares, ddof=1) / np.sqrt(len(shares)))
              if len(shares) > 1 else None)
        rows.append({
            "window": window,
            "position": pos,
            "train_alpha": a,
            "n": n,
            "weeks": len(cells),
            "cov_p10": round(sum(c[0] for c in cells) / n, 4),
            "cov_p10_se": round(se, 4) if se is not None else None,
        })
    return rows


def load_baselines() -> pl.DataFrame:
    return pl.read_parquet(OUTPUTS_DIR / "baseline_predictions.parquet").select(
        "player_id", "game_id", "pred_b1", "pred_b2",
    )


def recommend(report: pl.DataFrame, target: float = 0.1) -> pl.DataFrame:
    """Per position, the candidate alpha with the smallest *worst* gap to
    target across the windows.

    Minimax rather than a pooled or n-weighted fit on purpose: the eval
    window outnumbers the holdout ~7:1 on rows, so averaging would let
    the older seasons decide a question that is specifically about
    whether the current environment has moved. Minimax refuses an alpha
    that is badly wrong in either regime.
    """
    gap = (report.with_columns((pl.col("cov_p10") - target).abs().alias("gap"))
           .group_by(["position", "train_alpha"])
           .agg(pl.col("gap").max().round(4).alias("worst_gap"),
                pl.len().alias("windows")))
    return (gap.sort(["position", "worst_gap", "train_alpha"])
            .group_by("position", maintain_order=True).first()
            .select("position", pl.col("train_alpha").alias("pick"),
                    "worst_gap", "windows"))


def run_calibrate() -> None:
    baselines = load_baselines()
    rows = []
    for pos in POSITIONS:
        for window, seasons in WINDOWS.items():
            got = calibrate_position(pos, baselines, seasons, window)
            rows += got
            for r in got:
                print(f"  {window:>7} {pos:<3} alpha {r['train_alpha']:.3f} -> "
                      f"cov_p10 {r['cov_p10']:.4f} "
                      f"(n {r['n']}, {r['weeks']} wks)", flush=True)
    report = pl.DataFrame(rows)
    report.write_csv(OUTPUTS_DIR / "quantile_calibration.csv")
    print(f"\np10 training-alpha sweep (target coverage 0.10)\n"
          f"  eval    = {EVAL_SEASONS}\n  holdout = {HOLDOUT_SEASONS}\n")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(report.pivot(values="cov_p10", index=["position", "train_alpha"],
                           on="window").sort(["position", "train_alpha"]))
        print("\nminimax pick (smallest worst-case gap to target):")
        print(recommend(report))


def run_backtest() -> None:
    baselines = load_baselines()
    all_preds, rows = [], []
    for pos in POSITIONS:
        for window, seasons in WINDOWS.items():
            preds = backtest_position(pos, baselines, seasons)
            if not preds.height:
                continue
            all_preds.append(preds.with_columns(pl.lit(window).alias("window")))
            rows.append(coverage(preds, pos, window))
    report = pl.DataFrame(rows)
    report.write_csv(OUTPUTS_DIR / "quantile_backtest.csv")
    pl.concat(all_preds).write_parquet(OUTPUTS_DIR / "quantile_predictions.parquet")
    print(f"Walk-forward quantile eval (targets 0.10 / 0.50 / 0.90)\n"
          f"  eval    = {EVAL_SEASONS}\n  holdout = {HOLDOUT_SEASONS}\n")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(report.sort(["window", "position"]))


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    if sys.argv[1:2] == ["calibrate"]:
        run_calibrate()
    else:
        run_backtest()


if __name__ == "__main__":
    main()
