"""Per-position LightGBM models, walk-forward backtest vs baselines, and
the team-level Vegas anchor applied after the B2 blend.

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

# Final prediction blends the LightGBM output with baseline B2. Re-tuned
# for the huber model on 2024-25: 0.8 is best or within noise of best on
# MAE, RMSE and Spearman at every position (0.6 was right for the L2
# model; the stronger model now carries more of the weight).
BLEND_WEIGHT = 0.8

# Production training variant (see experiments.py for the grid):
#   residual   - predict fd_points - pred_b2 instead of raw points
#   decay      - row weight 0.5 ** (seasons_ago / decay)
#   objective  - regression | huber | tweedie
#   early_stop - rounds from early stopping on the last full train season
# Per-position overrides win over "default".
# Huber (delta 4 pts, ~one MAE) won the 2024-25 walk-forward grid at every
# position on MAE and Spearman: fantasy points are right-skewed, and L2
# spent the trees chasing boom weeks. Residual-over-B2 targets, recency
# weights and early stopping were neutral to slightly worse.
MODEL_SPEC: dict[str, dict] = {"default": {"objective": "huber"}}

# Team-level Vegas anchor, applied after the B2 blend to offense only.
# Each week, regress team projection sums on Vegas implied totals across
# that week's teams; a team's distance from the line is mostly noise
# (actual scoring barely follows it once the implied total is known), so
# every player on the team is scaled to close VEGAS_ANCHOR of that gap.
# The line is fit cross-sectionally on the same week's projections, so
# it needs no outcomes and absorbs the universe's scale (the backtest
# scores only active >=2-game players; production adds backups and
# cold-start rookies).
VEGAS_ANCHOR = 0.75
ANCHOR_CLIP = (0.75, 1.25)
ANCHOR_MIN_TEAMS = 8
# Doubtful/Out players don't count toward team sums (production still
# projects them; the backtest never sees them)
ANCHOR_MAX_INJ_STATUS = 1
OFFENSE = ["QB", "RB", "WR", "TE"]


def team_sums(rows: pl.DataFrame, pred: str) -> pl.DataFrame:
    """(season, week, game_id, team, implied_total, team_proj) over offense
    rows expected to play. `rows` needs season, week, game_id, team,
    position, implied_total and `pred`; inj_status is optional."""
    counted = rows.filter(pl.col("position").is_in(OFFENSE))
    if "inj_status" in counted.columns:
        counted = counted.filter(
            pl.col("inj_status").fill_null(0) <= ANCHOR_MAX_INJ_STATUS)
    return (counted.group_by("season", "week", "game_id", "team")
            .agg(pl.col("implied_total").first(), pl.col(pred).sum().alias("team_proj"))
            .filter(pl.col("implied_total").is_not_null()))


def fit_anchor_line(teams: pl.DataFrame) -> tuple[float, float] | None:
    """OLS team_proj = a + b * implied_total, or None with too few teams
    (a single-game slate: reuse the full week's line instead)."""
    if teams.height < ANCHOR_MIN_TEAMS:
        return None
    b, a = np.polyfit(teams["implied_total"].to_numpy().astype(float),
                      teams["team_proj"].to_numpy().astype(float), 1)
    return float(a), float(b)


def anchor_factors(teams: pl.DataFrame, line: tuple[float, float],
                   lam: float = VEGAS_ANCHOR) -> pl.DataFrame:
    """(game_id, team, anchor_factor): scale that moves team_proj lam of the
    way to the line, clipped to ANCHOR_CLIP."""
    a, b = line
    target = a + b * pl.col("implied_total")
    return teams.select(
        "game_id", "team",
        (1 - lam * (pl.col("team_proj") - target) / pl.col("team_proj"))
        .clip(*ANCHOR_CLIP).fill_nan(1.0).alias("anchor_factor"),
    )


def apply_vegas_anchor(rows: pl.DataFrame, pred: str, out: str,
                       lam: float = VEGAS_ANCHOR) -> pl.DataFrame:
    """Backtest form: fit the line per (season, week) and write the
    anchored prediction to `out`. DST rows and weeks with too few teams
    pass through unchanged."""
    teams = team_sums(rows, pred)
    factors = []
    for _, wk in teams.group_by("season", "week"):
        line = fit_anchor_line(wk)
        if line is not None:
            factors.append(anchor_factors(wk, line, lam))
    f = (pl.concat(factors) if factors
         else pl.DataFrame(schema={"game_id": pl.Utf8, "team": pl.Utf8,
                                   "anchor_factor": pl.Float64}))
    return (rows.join(f, on=["game_id", "team"], how="left")
            .with_columns(pl.when(pl.col("position").is_in(OFFENSE))
                          .then(pl.col(pred) * pl.col("anchor_factor").fill_null(1.0))
                          .otherwise(pl.col(pred)).alias(out))
            .drop("anchor_factor"))


def model_spec(pos: str) -> dict:
    return MODEL_SPEC.get(pos, MODEL_SPEC["default"])


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
        "inj_status", "inj_practice", "vacated_fd_pos", "vacated_fd_team",
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


HUBER_DELTA = 4.0            # roughly one MAE: quadratic inside, linear beyond
TWEEDIE_POWER = 1.3
EARLY_MAX_ROUNDS = 1500
EARLY_PATIENCE = 50


def variant_params(pos: str, spec: dict) -> dict:
    params = LGB_PARAMS | POSITION_PARAMS.get(pos, {})
    obj = spec.get("objective", "regression")
    params["objective"] = obj
    if obj == "huber":
        params["alpha"] = HUBER_DELTA
    if obj == "tweedie":
        params["tweedie_variance_power"] = TWEEDIE_POWER
    return params


def sample_weights(seasons: np.ndarray, decay: float | None) -> np.ndarray | None:
    if not decay:
        return None
    return 0.5 ** ((seasons.max() - seasons) / decay)


class Trainer:
    """One variant's fit/predict, with the target transform and the
    early-stopping round cache."""

    def __init__(self, pos: str, spec: dict):
        self.pos, self.spec = pos, spec
        self.params = variant_params(pos, spec)
        self.residual = spec.get("residual", False)
        self.decay = spec.get("decay")
        self.early = spec.get("early_stop", False)
        self.rounds_cache: dict[int, int] = {}
        self.rounds_used: list[int] = []

    # -- target transform ------------------------------------------------
    def _label(self, df: pl.DataFrame) -> np.ndarray:
        y = df["fd_points"].to_numpy().astype(float)
        if self.residual:
            y = y - df["pred_b2"].to_numpy().astype(float)
        return y

    def _fit(self, x, y, w, rounds, valid=None) -> lgb.Booster:
        y_shift = 0.0
        if self.params["objective"] == "tweedie":
            y_shift = float(y.min())
            y = y - y_shift
            if valid is not None:
                valid = (valid[0], valid[1] - y_shift, valid[2])
        self._shift = y_shift
        train_set = lgb.Dataset(x, label=y, weight=w)
        if valid is None:
            return lgb.train(self.params, train_set, num_boost_round=rounds)
        vx, vy, vw = valid
        return lgb.train(
            self.params, train_set, num_boost_round=rounds,
            valid_sets=[lgb.Dataset(vx, label=vy, weight=vw)],
            callbacks=[lgb.early_stopping(EARLY_PATIENCE, verbose=False)],
        )

    def _rounds(self, train: pl.DataFrame, cols: list[str], eval_season: int) -> int:
        if not self.early:
            return NUM_ROUNDS
        if eval_season in self.rounds_cache:
            return self.rounds_cache[eval_season]
        last = int(train["season"].max())
        fit = train.filter(pl.col("season") < last)
        val = train.filter(pl.col("season") == last)
        if fit.height < 1000 or val.height < 200:
            self.rounds_cache[eval_season] = NUM_ROUNDS
            return NUM_ROUNDS
        booster = self._fit(
            to_matrix(fit, cols), self._label(fit),
            sample_weights(fit["season"].to_numpy(), self.decay), EARLY_MAX_ROUNDS,
            valid=(to_matrix(val, cols), self._label(val),
                   sample_weights(val["season"].to_numpy(), self.decay)),
        )
        best = max(50, int(booster.best_iteration or NUM_ROUNDS))
        self.rounds_cache[eval_season] = best
        return best

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame,
                    cols: list[str], eval_season: int) -> np.ndarray:
        if self.residual:
            train = train.filter(pl.col("pred_b2").is_not_null())
        rounds = self._rounds(train, cols, eval_season)
        self.rounds_used.append(rounds)
        self.booster = self._fit(
            to_matrix(train, cols), self._label(train),
            sample_weights(train["season"].to_numpy(), self.decay), rounds,
        )
        yhat = self.booster.predict(to_matrix(test, cols)) + self._shift
        if self.residual:
            yhat = yhat + test["pred_b2"].fill_null(0.0).to_numpy()
        return yhat


def backtest_position(pos: str, baselines: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = (
        load_features(pos)
        .join(baselines, on=["player_id", "game_id"], how="left")
        .sort(["season", "week", "game_id", "player_id"])
    )
    cols = feature_columns(df)
    trainer = Trainer(pos, model_spec(pos))
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
        yhat = trainer.fit_predict(train, test, cols, season)
        importances += trainer.booster.feature_importance(importance_type="gain")
        preds.append(test.select(
            "player_id", "game_id", "season", "week", "position", "fd_points",
            "team", "implied_total",
            (pl.col("inj_status") if "inj_status" in test.columns
             else pl.lit(0)).cast(pl.Int64).alias("inj_status"),
        ).with_columns(pl.Series("pred_model", yhat)))
    imp = pl.DataFrame({"feature": cols, "gain": importances}).sort("gain", descending=True)
    return pl.concat(preds), imp


TOP_N, TOP_M = 5, 10


def top_hit_rate(pred, actual, n: int = TOP_N, m: int = TOP_M) -> float | None:
    """Share of the top-n by prediction that finished in the actual top-m
    (PLAN.md §4): the DFS question — did the model's favourites land?"""
    pred, actual = np.asarray(pred, dtype=float), np.asarray(actual, dtype=float)
    if len(pred) < max(n, m):
        return None
    picked = np.argsort(-pred, kind="stable")[:n]
    real = set(np.argsort(-actual, kind="stable")[:m])
    return float(np.mean([i in real for i in picked]))


def evaluate(matched: pl.DataFrame) -> dict:
    row = {"n": matched.height}
    for name in ["final", "blend", "model", "b1", "b2"]:
        if f"pred_{name}" not in matched.columns:
            continue
        err = matched[f"pred_{name}"] - matched["fd_points"]
        row[f"{name}_mae"] = round(float(err.abs().mean()), 3)
        row[f"{name}_rmse"] = round(float((err ** 2).mean() ** 0.5), 3)
        corrs, hits = [], []
        for _, wk in matched.group_by(["season", "week"]):
            if wk.height >= 8:
                r = spearmanr(wk[f"pred_{name}"], wk["fd_points"]).statistic
                if r == r:
                    corrs.append(r)
            h = top_hit_rate(wk[f"pred_{name}"], wk["fd_points"])
            if h is not None:
                hits.append(h)
        row[f"{name}_spearman"] = round(sum(corrs) / len(corrs), 3) if corrs else None
        row[f"{name}_top5"] = round(sum(hits) / len(hits), 3) if hits else None
    return row


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    baselines = pl.read_parquet(OUTPUTS_DIR / "baseline_predictions.parquet").select(
        "player_id", "game_id", "pred_b1", "pred_b2",
    )
    all_preds = []
    for pos in POSITIONS:
        preds, imp = backtest_position(pos, baselines)
        imp.write_csv(OUTPUTS_DIR / f"feature_importance_{pos.lower()}.csv")
        all_preds.append(preds.join(baselines, on=["player_id", "game_id"], how="inner").filter(
            pl.col("pred_b1").is_not_null() & pl.col("pred_b2").is_not_null()
        ).with_columns(
            (BLEND_WEIGHT * pl.col("pred_model")
             + (1 - BLEND_WEIGHT) * pl.col("pred_b2")).alias("pred_blend")
        ))
        print(f"{pos}: top features -> "
              + ", ".join(imp["feature"].head(5).to_list()))
    # the anchor needs every offensive position at once (team sums), so it
    # runs after the per-position loop; "blend" is the pre-anchor number
    matched_all = apply_vegas_anchor(pl.concat(all_preds), "pred_blend", "pred_final")
    rows = [{"position": pos}
            | evaluate(matched_all.filter(pl.col("position") == pos))
            for pos in POSITIONS]
    report = pl.DataFrame(rows)
    report.write_csv(OUTPUTS_DIR / "model_backtest.csv")
    matched_all.write_parquet(OUTPUTS_DIR / "model_predictions.parquet")
    with pl.Config(tbl_cols=-1, tbl_width_chars=220):
        print(f"\nWalk-forward eval, seasons {EVAL_SEASONS}, REG, "
              f">={MIN_PRIOR_GAMES} prior games, matched rows vs baselines\n")
        print(report)


if __name__ == "__main__":
    main()
