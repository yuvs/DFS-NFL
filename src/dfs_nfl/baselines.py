"""Walk-forward backtest of the two naive baselines — the bar any model
must beat (see PLAN.md §4).

  B1: mean FanDuel points over the player's previous 4 played games
  B2: EWMA (halflife 4 games) x shrunk opponent-allowed factor
      x shrunk Vegas implied-total factor

Both use only information from strictly earlier games, so the walk-forward
protocol is satisfied by construction.

Usage: uv run python -m dfs_nfl.baselines
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
CURATED_DIR = REPO_ROOT / "data" / "curated"
OUTPUTS_DIR = REPO_ROOT / "outputs"

EVAL_SEASONS = range(2021, 2026)
MIN_PRIOR_GAMES = 2
EWMA_HALFLIFE = 4.0
SHRINK = 0.5           # weight on (raw factor - 1)
FACTOR_CLAMP = (0.6, 1.5)
MIN_OPP_GAMES = 3      # prior same-season games before the opp factor kicks in


def load_panel() -> pl.DataFrame:
    """One row per entity-game: offensive players plus DST as position 'DST'."""
    players = (
        pl.read_parquet(CURATED_DIR / "player_week.parquet")
        .select(
            "player_id",
            pl.col("player_display_name").alias("name"),
            pl.col("position_group").alias("position"),
            "season", "week", "season_type", "game_id",
            "team", "opponent_team", "fd_points",
        )
    )
    dst = (
        pl.read_parquet(CURATED_DIR / "dst_week.parquet")
        .select(
            pl.col("team").alias("player_id"),
            (pl.col("team") + " D/ST").alias("name"),
            pl.lit("DST").alias("position"),
            "season", "week", "season_type", "game_id",
            "team", "opponent_team", "fd_points",
        )
    )
    return (
        pl.concat([players, dst])
        .filter(pl.col("season_type") == "REG")
        .sort(["player_id", "season", "week"])
    )


def add_trailing_predictions(panel: pl.DataFrame) -> pl.DataFrame:
    """B1 trailing mean and the EWMA core of B2, per entity, shifted one
    game so week W predictions use only games before W."""
    by_player = ["player_id"]
    return panel.with_columns(
        pl.col("fd_points").shift(1).rolling_mean(4, min_samples=1)
        .over(by_player).alias("pred_b1"),
        pl.col("fd_points").ewm_mean(half_life=EWMA_HALFLIFE).shift(1)
        .over(by_player).alias("ewma"),
        pl.int_range(pl.len()).over(by_player).alias("prior_games"),
    )


def opponent_allowed_factor(panel: pl.DataFrame) -> pl.DataFrame:
    """Per (season, week, opponent_team, position): opponent's mean FD points
    allowed per game to that position over PRIOR weeks of the same season,
    relative to the league average over those same weeks."""
    per_game = (
        panel.group_by(["season", "week", "opponent_team", "position"])
        .agg(pl.col("fd_points").sum().alias("allowed"))
    )
    # cumulative prior-week stats per (season, defense, position)
    prior = (
        per_game.sort(["season", "opponent_team", "position", "week"])
        .with_columns(
            pl.col("allowed").cum_sum().shift(1)
            .over(["season", "opponent_team", "position"]).alias("cum_allowed"),
            pl.int_range(pl.len())
            .over(["season", "opponent_team", "position"]).alias("n_prior"),
        )
        .with_columns(
            (pl.col("cum_allowed") / pl.col("n_prior")).alias("opp_allowed_pg"),
        )
    )
    # league average allowed per defense-game, over prior weeks of the season
    league = (
        per_game.group_by(["season", "week", "position"])
        .agg(pl.col("allowed").sum().alias("wk_total"), pl.len().alias("wk_n"))
        .sort(["season", "position", "week"])
        .with_columns(
            (pl.col("wk_total").cum_sum().shift(1) / pl.col("wk_n").cum_sum().shift(1))
            .over(["season", "position"]).alias("lg_allowed_pg"),
        )
    )
    return (
        prior.join(league, on=["season", "week", "position"], how="left")
        .with_columns(
            pl.when(
                (pl.col("n_prior") >= MIN_OPP_GAMES)
                & pl.col("lg_allowed_pg").is_not_null()
                & (pl.col("lg_allowed_pg") > 0)
            )
            .then(pl.col("opp_allowed_pg") / pl.col("lg_allowed_pg"))
            .otherwise(1.0)
            .alias("opp_factor_raw"),
        )
        .select("season", "week", "opponent_team", "position", "opp_factor_raw")
    )


def vegas_factor(panel: pl.DataFrame) -> pl.DataFrame:
    """Implied team total vs league average (prior weeks of same season).

    nflverse spread_line is home margin (positive = home favored), so
    implied home total = total/2 + spread/2. For offense the factor scales
    with own implied total; for DST it scales inversely with the
    opponent's implied total.
    """
    sched = pl.read_parquet(REPO_ROOT / "data" / "raw" / "schedules.parquet")
    implied = (
        sched.select("game_id", "season", "week", "home_team", "away_team",
                     "spread_line", "total_line")
        .with_columns(
            (pl.col("total_line") / 2 + pl.col("spread_line") / 2).alias("home_implied"),
            (pl.col("total_line") / 2 - pl.col("spread_line") / 2).alias("away_implied"),
        )
    )
    long = pl.concat([
        implied.select(
            "game_id", "season", "week",
            pl.col("home_team").alias("team"),
            pl.col("home_implied").alias("implied_total"),
            pl.col("away_implied").alias("opp_implied_total"),
        ),
        implied.select(
            "game_id", "season", "week",
            pl.col("away_team").alias("team"),
            pl.col("away_implied").alias("implied_total"),
            pl.col("home_implied").alias("opp_implied_total"),
        ),
    ])
    lg = (
        long.group_by(["season", "week"])
        .agg(pl.col("implied_total").mean().alias("wk_mean"), pl.len().alias("wk_n"))
        .sort(["season", "week"])
        .with_columns(
            ((pl.col("wk_mean") * pl.col("wk_n")).cum_sum().shift(1)
             / pl.col("wk_n").cum_sum().shift(1))
            .over("season").alias("lg_implied"),
        )
        .select("season", "week", "lg_implied")
    )
    return (
        long.join(lg, on=["season", "week"], how="left")
        .with_columns(
            pl.when(pl.col("lg_implied").is_not_null() & pl.col("implied_total").is_not_null())
            .then(pl.col("implied_total") / pl.col("lg_implied"))
            .otherwise(1.0).alias("vegas_factor_off_raw"),
            pl.when(pl.col("lg_implied").is_not_null() & pl.col("opp_implied_total").is_not_null()
                    & (pl.col("opp_implied_total") > 0))
            .then(pl.col("lg_implied") / pl.col("opp_implied_total"))
            .otherwise(1.0).alias("vegas_factor_dst_raw"),
        )
        .select("game_id", "team", "vegas_factor_off_raw", "vegas_factor_dst_raw")
    )


def shrink_clamp(col: str) -> pl.Expr:
    lo, hi = FACTOR_CLAMP
    return (1 + SHRINK * (pl.col(col).fill_null(1.0) - 1)).clip(lo, hi)


def build_predictions(panel: pl.DataFrame | None = None) -> pl.DataFrame:
    """B1/B2 for every row of the panel. `panel` may include synthetic
    future rows (null fd_points) — their trailing stats and factors come
    out correct because everything is shift(1)/prior-week based."""
    panel = add_trailing_predictions(load_panel() if panel is None else panel)
    panel = (
        panel
        .join(opponent_allowed_factor(panel),
              on=["season", "week", "opponent_team", "position"], how="left")
        .join(vegas_factor(panel), on=["game_id", "team"], how="left")
        .with_columns(
            shrink_clamp("opp_factor_raw").alias("opp_factor"),
            pl.when(pl.col("position") == "DST")
            .then(shrink_clamp("vegas_factor_dst_raw"))
            .otherwise(shrink_clamp("vegas_factor_off_raw"))
            .alias("vegas_factor"),
        )
        .with_columns(
            (pl.col("ewma") * pl.col("opp_factor") * pl.col("vegas_factor"))
            .alias("pred_b2"),
        )
    )
    return panel


def evaluate(panel: pl.DataFrame) -> pl.DataFrame:
    eval_df = panel.filter(
        pl.col("season").is_in(list(EVAL_SEASONS))
        & (pl.col("prior_games") >= MIN_PRIOR_GAMES)
        & pl.col("pred_b1").is_not_null()
        & pl.col("pred_b2").is_not_null()
    )

    rows = []
    for pos in ["QB", "RB", "WR", "TE", "DST"]:
        sub = eval_df.filter(pl.col("position") == pos)
        if not sub.height:
            continue
        row = {"position": pos, "n_player_weeks": sub.height}
        for b in ["b1", "b2"]:
            err = sub[f"pred_{b}"] - sub["fd_points"]
            row[f"{b}_mae"] = round(float(err.abs().mean()), 3)
            row[f"{b}_rmse"] = round(float((err ** 2).mean() ** 0.5), 3)
            # mean within-week Spearman across position-weeks with >=8 entities
            corrs = []
            for (_, _), wk in sub.group_by(["season", "week"]):
                if wk.height >= 8:
                    r = spearmanr(wk[f"pred_{b}"], wk["fd_points"]).statistic
                    if r == r:  # not NaN
                        corrs.append(r)
            row[f"{b}_spearman"] = round(sum(corrs) / len(corrs), 3) if corrs else None
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    panel = build_predictions()
    panel.write_parquet(OUTPUTS_DIR / "baseline_predictions.parquet")
    report = evaluate(panel)
    report.write_csv(OUTPUTS_DIR / "baseline_backtest.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(f"Eval: seasons {EVAL_SEASONS.start}–{EVAL_SEASONS.stop - 1} REG, "
              f">={MIN_PRIOR_GAMES} prior games\n")
        print(report)


if __name__ == "__main__":
    main()
