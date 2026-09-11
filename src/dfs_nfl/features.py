"""Feature store: pre-kickoff features for offense and DST (PLAN.md §3).

Every feature is computed from strictly earlier games (trailing stats are
shifted one game) or from pre-game information (Vegas lines, venue, rest),
so rows are safe for walk-forward training/prediction as-is.

Outputs (data/curated/):
  features_offense.parquet  features_dst.parquet

Usage: uv run python -m dfs_nfl.features
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dfs_nfl.baselines import (
    CURATED_DIR,
    REPO_ROOT,
    EWMA_HALFLIFE,
    load_panel,
    opponent_allowed_factor,
)

RAW_DIR = REPO_ROOT / "data" / "raw"

OFFENSE_TRAILING_COLS = [
    "fd_points", "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "carries", "rushing_yards", "rushing_tds",
    "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "target_share", "air_yards_share", "wopr", "snap_pct",
]
DST_TRAILING_COLS = [
    "fd_points", "event_points", "points_allowed", "def_sacks", "takeaways",
]


def trailing_exprs(cols: list[str], over: str = "player_id") -> list[pl.Expr]:
    """EWMA + trailing-4 mean per column, shifted so week W sees only <W.
    Trailing stats deliberately cross season boundaries (with EWMA decay)."""
    exprs = []
    for c in cols:
        base = pl.col(c).cast(pl.Float64)
        exprs.append(
            base.ewm_mean(half_life=EWMA_HALFLIFE).shift(1).over(over)
            .alias(f"{c}_ewma")
        )
        exprs.append(
            base.shift(1).rolling_mean(4, min_samples=1).over(over)
            .alias(f"{c}_l4")
        )
    exprs.append(
        pl.col("fd_points").cast(pl.Float64).shift(1)
        .rolling_std(8, min_samples=3).over(over).alias("fd_points_std8")
    )
    exprs.append(
        pl.col("fd_points").cast(pl.Float64).shift(1)
        .rolling_max(8, min_samples=1).over(over).alias("fd_points_max8")
    )
    exprs.append(pl.int_range(pl.len()).over(over).alias("prior_games"))
    return exprs


def game_context() -> pl.DataFrame:
    """Per (game_id, team): venue, rest, Vegas-derived numbers."""
    sched = pl.read_parquet(RAW_DIR / "schedules.parquet")
    base = sched.select(
        "game_id", "season", "week", "home_team", "away_team",
        "home_rest", "away_rest", "div_game", "roof", "spread_line",
        "total_line", "temp", "wind",
    ).with_columns(
        (pl.col("total_line") / 2 + pl.col("spread_line") / 2).alias("home_implied"),
        (pl.col("total_line") / 2 - pl.col("spread_line") / 2).alias("away_implied"),
        pl.col("roof").is_in(["dome", "closed"]).cast(pl.Int8).alias("is_dome"),
    ).with_columns(
        # indoor games: standard conditions instead of nulls
        pl.when(pl.col("is_dome") == 1).then(70).otherwise(pl.col("temp")).alias("temp"),
        pl.when(pl.col("is_dome") == 1).then(0).otherwise(pl.col("wind")).alias("wind"),
    )
    home = base.select(
        "game_id",
        pl.col("home_team").alias("team"),
        pl.lit(1, dtype=pl.Int8).alias("is_home"),
        pl.col("home_rest").alias("rest_days"),
        "div_game", "is_dome", "total_line", "temp", "wind",
        pl.col("home_implied").alias("implied_total"),
        pl.col("away_implied").alias("opp_implied_total"),
    )
    away = base.select(
        "game_id",
        pl.col("away_team").alias("team"),
        pl.lit(0, dtype=pl.Int8).alias("is_home"),
        pl.col("away_rest").alias("rest_days"),
        "div_game", "is_dome", "total_line", "temp", "wind",
        pl.col("away_implied").alias("implied_total"),
        pl.col("home_implied").alias("opp_implied_total"),
    )
    return pl.concat([home, away]).with_columns(
        (pl.col("implied_total") - pl.col("opp_implied_total")).alias("vegas_edge")
    )


def snap_pct_table() -> pl.DataFrame:
    """(season, week, gsis player_id) -> offensive snap share."""
    files = sorted((RAW_DIR / "snap_counts").glob("*.parquet"))
    snaps = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    # sorts before unique / aggregation instead of unique's arbitrary
    # tie-break, so repeated runs build identical features
    ids = (
        pl.read_parquet(RAW_DIR / "ff_playerids.parquet")
        .select("gsis_id", "pfr_id")
        .drop_nulls()
        .sort("gsis_id")
        .unique(subset="pfr_id", keep="first")
    )
    return (
        snaps.select("season", "week", "pfr_player_id", "offense_pct")
        .join(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .group_by("season", "week", pl.col("gsis_id").alias("player_id"))
        .agg(pl.col("offense_pct").cast(pl.Float64).max().alias("snap_pct"))
    )


def depth_rank_table() -> pl.DataFrame:
    """(season, week, gsis player_id) -> depth-chart rank at own position."""
    files = sorted((RAW_DIR / "depth_charts").glob("*.parquet"))
    dc = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return (
        dc.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
        .group_by(["season", "week", "gsis_id"])
        .agg(pl.col("depth_rank").min())
        .rename({"gsis_id": "player_id"})
    )


def player_draft_table() -> pl.DataFrame:
    """Static per-player draft capital and rookie-season info."""
    return (
        pl.read_parquet(RAW_DIR / "players.parquet")
        .select(
            pl.col("gsis_id").alias("player_id"),
            "rookie_season", "draft_round",
            # undrafted players sort after every drafted pick
            pl.col("draft_pick").fill_null(300).alias("draft_pick"),
        )
        .drop_nulls("player_id")
        .sort("player_id", "rookie_season")
        .unique(subset="player_id", keep="first")
    )


STATUS_ORD = {"Questionable": 1, "Doubtful": 2, "Out": 3}
PRACTICE_ORD = {
    "Full Participation in Practice": 0,
    "Limited Participation in Practice": 1,
    "Did Not Participate In Practice": 2,
}


def _read_injuries() -> pl.DataFrame:
    files = sorted((RAW_DIR / "injuries").glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int32,
                                    "team": pl.Utf8, "gsis_id": pl.Utf8,
                                    "position": pl.Utf8, "report_status": pl.Utf8,
                                    "practice_status": pl.Utf8})
    return (
        pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
        .filter(pl.col("game_type") == "REG")
        .with_columns(
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
        )
    )


def injury_status_table() -> pl.DataFrame:
    """(season, week, player_id) -> own report/practice status, ordinal.
    Reports finalize Friday before Sunday games, so they are
    pre-kickoff-safe under the Sunday-morning snapshot rule."""
    return (
        _read_injuries()
        .with_columns(
            pl.col("report_status").replace_strict(STATUS_ORD, default=0)
            .alias("inj_status"),
            pl.col("practice_status").replace_strict(PRACTICE_ORD, default=0)
            .alias("inj_practice"),
        )
        .group_by(["season", "week", pl.col("gsis_id").alias("player_id")])
        .agg(pl.col("inj_status").max(), pl.col("inj_practice").max())
    )


def vacated_table() -> pl.DataFrame:
    """Per (season, week, team, position): trailing FD points of teammates
    ruled Out/Doubtful that week — the opportunity they vacate. Only
    players who appeared within the last 5 schedule weeks count, so
    long-absent names stop inflating the signal."""
    played = (
        pl.read_parquet(CURATED_DIR / "player_week.parquet")
        .filter(pl.col("season_type") == "REG")
        .sort("player_id", "season", "week")
        .with_columns(
            pl.col("fd_points").ewm_mean(half_life=EWMA_HALFLIFE)
            .over("player_id").alias("fd_ewma_incl"),
            # 25 > max week, so consecutive seasons stay ~contiguous and
            # the recency gate below works across the offseason boundary
            (pl.col("season") * 25 + pl.col("week")).cast(pl.Int64).alias("wk_idx"),
        )
        .select("player_id", "wk_idx", "fd_ewma_incl")
        .sort("wk_idx")
    )
    out = (
        _read_injuries()
        .filter(pl.col("report_status").is_in(["Out", "Doubtful"])
                & pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
        .select(
            "season", "week", "team",
            pl.col("position").alias("position_out"),
            pl.col("gsis_id").alias("player_id"),
            (pl.col("season") * 25 + pl.col("week") - 1).cast(pl.Int64)
            .alias("asof_idx"),
        )
        .sort("asof_idx")
    )
    joined = out.join_asof(
        played, left_on="asof_idx", right_on="wk_idx",
        by="player_id", strategy="backward",
    ).filter(
        # active within ~10 schedule weeks; across the offseason boundary
        # this admits players who produced late in the prior season
        pl.col("fd_ewma_incl").is_not_null()
        & ((pl.col("asof_idx") - pl.col("wk_idx")) <= 10)
    )
    per_pos = (
        joined.group_by(["season", "week", "team", "position_out"])
        .agg(pl.col("fd_ewma_incl").sum().round(2).alias("vacated_fd_pos"))
    )
    per_team = (
        joined.group_by(["season", "week", "team"])
        .agg(pl.col("fd_ewma_incl").sum().round(2).alias("vacated_fd_team"))
    )
    return per_pos, per_team


def _with_opp_and_context(df: pl.DataFrame, panel_for_opp: pl.DataFrame) -> pl.DataFrame:
    opp = opponent_allowed_factor(panel_for_opp)
    return (
        df.join(opp, on=["season", "week", "opponent_team", "position"], how="left")
        .with_columns(pl.col("opp_factor_raw").fill_null(1.0))
        .join(game_context(), on=["game_id", "team"], how="left")
    )


def _panel_with_skeleton(skeleton: pl.DataFrame | None) -> pl.DataFrame:
    """Combined panel for opponent-allowed factors; skeleton rows (a not-
    yet-played week, null fd_points) are appended so the factor table has
    keys for that week, built from prior weeks only."""
    panel = load_panel()
    if skeleton is None:
        return panel
    return pl.concat(
        [panel, skeleton.select("player_id", "position", "season", "week",
                                "game_id", "team", "opponent_team")],
        how="diagonal_relaxed",
    )


def build_offense_features(skeleton: pl.DataFrame | None = None) -> pl.DataFrame:
    """Feature table for all history; pass `skeleton` (identity columns for
    an upcoming week, stats null) to also get prediction-ready rows for
    that week."""
    pw = pl.read_parquet(CURATED_DIR / "player_week.parquet").filter(
        pl.col("season_type") == "REG"
    )
    if skeleton is not None:
        # simulate the target week as unplayed (matters when re-running a
        # historical week): its real rows are replaced by the skeleton
        pw = pl.concat(
            [pw.join(skeleton.select("season", "week").unique(),
                     on=["season", "week"], how="anti"), skeleton],
            how="diagonal_relaxed",
        )
    pw = (
        pw.join(snap_pct_table(), on=["season", "week", "player_id"], how="left")
        .drop("position")
        .rename({"position_group": "position"})
        .sort(["player_id", "season", "week"])
        .with_columns(
            pl.int_range(pl.len()).over(["player_id", "season"]).alias("season_prior_games"),
        )
    )
    vac_pos, vac_team = vacated_table()
    pw = (
        pw.with_columns(trailing_exprs(OFFENSE_TRAILING_COLS))
        .join(depth_rank_table(), on=["season", "week", "player_id"], how="left")
        .join(player_draft_table(), on="player_id", how="left")
        .join(injury_status_table(), on=["season", "week", "player_id"], how="left")
        .join(vac_pos.rename({"position_out": "position"}),
              on=["season", "week", "team", "position"], how="left")
        .join(vac_team, on=["season", "week", "team"], how="left")
        .with_columns(
            (pl.col("season") == pl.col("rookie_season")).cast(pl.Int8)
            .fill_null(0).alias("is_rookie"),
            pl.col("inj_status").fill_null(0),
            pl.col("inj_practice").fill_null(0),
            pl.col("vacated_fd_pos").fill_null(0.0),
            pl.col("vacated_fd_team").fill_null(0.0),
        )
    )
    # opponent-allowed uses the combined panel so factors match baselines
    return _with_opp_and_context(pw, _panel_with_skeleton(skeleton)).sort(
        ["season", "week", "team"]
    )


def build_dst_features(skeleton: pl.DataFrame | None = None) -> pl.DataFrame:
    dst = pl.read_parquet(CURATED_DIR / "dst_week.parquet").filter(
        pl.col("season_type") == "REG"
    )
    if skeleton is not None:
        dst = pl.concat(
            [dst.join(skeleton.select("season", "week").unique(),
                      on=["season", "week"], how="anti"), skeleton],
            how="diagonal_relaxed",
        )
    dst = (
        dst.with_columns(
            pl.col("team").alias("player_id"),
            pl.lit("DST").alias("position"),
            (pl.col("def_interceptions").fill_null(0)
             + pl.col("fumble_recovery_opp").fill_null(0)).alias("takeaways"),
        )
        .sort(["player_id", "season", "week"])
        .with_columns(
            pl.int_range(pl.len()).over(["player_id", "season"]).alias("season_prior_games"),
        )
    )
    dst = dst.with_columns(trailing_exprs(DST_TRAILING_COLS))
    return _with_opp_and_context(dst, _panel_with_skeleton(skeleton)).sort(
        ["season", "week", "team"]
    )


def main() -> None:
    off = build_offense_features()
    off.write_parquet(CURATED_DIR / "features_offense.parquet")
    print(f"features_offense: {off.height:,} rows, {len(off.columns)} cols")
    dst = build_dst_features()
    dst.write_parquet(CURATED_DIR / "features_dst.parquet")
    print(f"features_dst: {dst.height:,} rows, {len(dst.columns)} cols")


if __name__ == "__main__":
    main()
