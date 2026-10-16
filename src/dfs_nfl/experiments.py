"""Walk-forward comparison of modeling variants on the current feature
store (PLAN.md §4) — the same protocol as models.py, with four dials:

  residual    trees predict fd_points - pred_b2 (boost on what the
              baseline misses) instead of raw points
  decay       training-row weight 0.5 ** (seasons_ago / decay): recent
              seasons count more
  objective   regression (L2) | huber | tweedie (tweedie needs a
              non-negative label, so the target is shifted by the
              training minimum and shifted back)
  early_stop  rounds chosen by early stopping on the last full training
              season (re-fit on everything with that count) instead of
              the fixed NUM_ROUNDS; cached per (position, eval season)

Every variant is scored on the identical matched rows, both raw
("model") and blended with B2 ("final", what production ships). The
trainer itself lives in models.py (MODEL_SPEC picks the production
variant); this module is just the grid.

Usage: uv run python -m dfs_nfl.experiments [variant ...]
Writes outputs/experiments.csv and prints a leaderboard.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

from dfs_nfl.baselines import MIN_PRIOR_GAMES, OUTPUTS_DIR
from dfs_nfl.models import (  # noqa: F401 - re-exported for tests/callers
    BLEND_WEIGHT,
    EARLY_MAX_ROUNDS,
    EVAL_SEASONS,
    HOLDOUT_SEASONS,
    HUBER_DELTA,
    TWEEDIE_POWER,
    POSITIONS,
    Trainer,
    evaluate,
    feature_columns,
    load_features,
    sample_weights,
    split_week,
    variant_params,
    walk_forward_weeks,
)

# Each variant is scored on both windows: the 2024-25 primary that picks
# the production spec, and the in-season holdout as a current-environment
# sanity read (too few weeks to decide anything on its own).
WINDOWS = {"eval": EVAL_SEASONS, "holdout": HOLDOUT_SEASONS}
from dfs_nfl.pbp_features import QB_SKILL_COLS

VARIANTS: dict[str, dict] = {
    "base": {},
    "residual": {"residual": True},
    "decay": {"decay": 3.0},
    "resid_decay": {"residual": True, "decay": 3.0},
    "huber": {"objective": "huber"},
    "huber_resid_decay": {"residual": True, "decay": 3.0, "objective": "huber"},
    "tweedie_decay": {"objective": "tweedie", "decay": 3.0},
    "early": {"early_stop": True},
    "resid_decay_early": {"residual": True, "decay": 3.0, "early_stop": True},
    # production spec plus the starting-QB skill features: built into the
    # feature store but kept out of production (2026-09 A/B: MAE a hair
    # better, top-5 hit a hair worse, all within week-bootstrap noise)
    "huber_qb": {"objective": "huber", "extra_features": QB_SKILL_COLS},
}


def backtest_variant(name: str, spec: dict, baselines: pl.DataFrame) -> list[dict]:
    rows = []
    for pos in POSITIONS:
        df = (
            load_features(pos)
            .join(baselines, on=["player_id", "game_id"], how="left")
            .sort(["season", "week", "game_id", "player_id"])
        )
        cols = feature_columns(df) + spec.get("extra_features", [])
        for window, seasons in WINDOWS.items():
            t0 = time.time()
            trainer = Trainer(pos, spec)
            preds = []
            for season, week in walk_forward_weeks(df, seasons):
                train, test = split_week(df, season, week, MIN_PRIOR_GAMES)
                if not test.height:
                    continue
                yhat = trainer.fit_predict(train, test, cols, season)
                preds.append(test.select(
                    "player_id", "game_id", "season", "week", "position", "fd_points",
                    "pred_b1", "pred_b2",
                ).with_columns(pl.Series("pred_model", yhat)))
            if not preds:
                continue
            matched = pl.concat(preds).filter(
                pl.col("pred_b1").is_not_null() & pl.col("pred_b2").is_not_null()
            ).with_columns(
                (BLEND_WEIGHT * pl.col("pred_model")
                 + (1 - BLEND_WEIGHT) * pl.col("pred_b2")).alias("pred_final")
            )
            row = ({"variant": name, "window": window, "position": pos}
                   | evaluate(matched))
            row["rounds"] = round(float(np.mean(trainer.rounds_used)), 0)
            row["secs"] = round(time.time() - t0, 0)
            rows.append(row)
            print(f"  {name:>20} {window:>7} {pos:<3} "
                  f"model MAE {row['model_mae']:.3f} "
                  f"sp {row['model_spearman']:.3f} | final MAE {row['final_mae']:.3f} "
                  f"sp {row['final_spearman']:.3f} | rounds {row['rounds']:.0f} "
                  f"({row['secs']:.0f}s)", flush=True)
    return rows


def main() -> None:
    names = sys.argv[1:] or list(VARIANTS)
    baselines = pl.read_parquet(OUTPUTS_DIR / "baseline_predictions.parquet").select(
        "player_id", "game_id", "pred_b1", "pred_b2",
    )
    rows = []
    for name in names:
        rows += backtest_variant(name, VARIANTS[name], baselines)
    report = pl.DataFrame(rows)
    out = OUTPUTS_DIR / "experiments.csv"
    if out.exists() and sys.argv[1:]:
        # partial run: merge over any earlier rows for the same variants
        old = pl.read_csv(out).filter(~pl.col("variant").is_in(names))
        if "window" not in old.columns:
            # pre-holdout rows were scored on EVAL_SEASONS only
            old = old.with_columns(pl.lit("eval").alias("window"))
        report = pl.concat([old, report], how="diagonal_relaxed")
    report.write_csv(out)
    keep = ["variant", "position", "model_mae", "model_spearman",
            "final_mae", "final_rmse", "final_spearman", "rounds"]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        for window in WINDOWS:
            sub = report.filter(pl.col("window") == window)
            if not sub.height:
                continue
            print(f"\n=== {window} ({WINDOWS[window]}) ===")
            print(sub.select(keep).sort(["position", "final_mae"]))


if __name__ == "__main__":
    main()
