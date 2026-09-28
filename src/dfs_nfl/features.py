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
from dfs_nfl.pbp_features import (
    PLAYER_OPP_COLS,
    QB_SKILL_COLS,
    _team_week_grid,
    incumbent_ratings,
    player_opportunity,
    primary_passers,
    qb_game_table,
    qb_ratings,
    team_context_tables,
)

RAW_DIR = REPO_ROOT / "data" / "raw"

OFFENSE_TRAILING_COLS = [
    "fd_points", "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "carries", "rushing_yards", "rushing_tds",
    "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "target_share", "air_yards_share", "wopr", "snap_pct",
    # play-by-play opportunity: red-zone / goal-line usage, expected TDs
    *PLAYER_OPP_COLS,
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


def qb_depth_table() -> pl.DataFrame:
    """(season, week, team, qb_id, depth_rank) for QBs on each week's chart."""
    files = sorted((RAW_DIR / "depth_charts").glob("*.parquet"))
    dc = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return (
        dc.filter(pl.col("position") == "QB")
        .group_by(["season", "week", "team", pl.col("gsis_id").alias("qb_id")])
        .agg(pl.col("depth_rank").min())
        .with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    )


def expected_starters(games: pl.DataFrame,
                      extra_outs: pl.DataFrame | None = None) -> pl.DataFrame:
    """(season, week, team, qb_id): the best-ranked QB on that week's depth
    chart who isn't ruled Doubtful/Out; weeks without a usable chart fall
    back to the passer with the most dropbacks in the team's last game."""
    grid = _team_week_grid()
    outs = injury_status_table(extra_outs).filter(pl.col("inj_status") >= 2).select(
        "season", "week", pl.col("player_id").alias("qb_id"))
    chart = (
        qb_depth_table()
        .join(outs, on=["season", "week", "qb_id"], how="anti")
        .sort("depth_rank")
        .unique(subset=["season", "week", "team"], keep="first", maintain_order=True)
        .select("season", "week", "team", "qb_id")
    )
    last = primary_passers(games).with_columns(
        ((pl.col("season") - 2010) * 25 + pl.col("week")).alias("g_idx")).sort("g_idx")
    fallback = (
        grid.with_columns(((pl.col("season") - 2010) * 25 + pl.col("week") - 1).alias("g_idx"))
        .sort("g_idx")
        .join_asof(last.select("team", "g_idx", "qb_id"), on="g_idx", by="team",
                   strategy="backward")
        .select("season", "week", "team", pl.col("qb_id").alias("fb_id"))
    )
    return (
        grid.join(chart, on=["season", "week", "team"], how="left")
        .join(fallback, on=["season", "week", "team"], how="left")
        .select("season", "week", "team",
                pl.coalesce("qb_id", "fb_id").alias("qb_id"))
        .filter(pl.col("qb_id").is_not_null())
    )


def qb_skill_table(extra_outs: pl.DataFrame | None = None) -> pl.DataFrame:
    """(season, week, team) -> the expected starter's rating (qb_epa,
    qb_cpoe) and its gap to the passers behind the team's trailing
    offense stats (qb_*_delta: ~0 most weeks, large when a backup or a
    returning starter takes over)."""
    games = qb_game_table()
    starters = expected_starters(games, extra_outs)
    rated = qb_ratings(games, starters)
    inc = incumbent_ratings(games, starters.select("season", "week", "team"))
    return (
        rated.join(inc, on=["season", "week", "team"], how="left")
        .with_columns(
            (pl.col("qb_epa") - pl.col("inc_epa")).fill_null(0.0).alias("qb_epa_delta"),
            (pl.col("qb_cpoe") - pl.col("inc_cpoe")).fill_null(0.0).alias("qb_cpoe_delta"),
        )
        .select("season", "week", "team", *QB_SKILL_COLS)
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


INJURY_SCHEMA = {"season": pl.Int32, "week": pl.Int32, "team": pl.Utf8,
                 "gsis_id": pl.Utf8, "position": pl.Utf8,
                 "report_status": pl.Utf8, "practice_status": pl.Utf8}


def _read_injuries(extra_outs: pl.DataFrame | None = None) -> pl.DataFrame:
    """nflverse injury reports (the official Wed-Fri reports). extra_outs
    — (season, week, team, gsis_id, position) — are appended as 'Out':
    game-day inactives that only show up in a slate export."""
    files = sorted((RAW_DIR / "injuries").glob("*.parquet"))
    inj = pl.DataFrame(schema=INJURY_SCHEMA)
    if files:
        inj = (
            pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
            .filter(pl.col("game_type") == "REG")
            .select(list(INJURY_SCHEMA))
            .with_columns(
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
            )
        )
    if extra_outs is not None and extra_outs.height:
        extra = extra_outs.select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
            "team", "gsis_id", "position",
            pl.lit("Out").alias("report_status"),
            pl.lit(None, dtype=pl.Utf8).alias("practice_status"),
        )
        inj = pl.concat([inj, extra], how="vertical_relaxed")
    return inj


def injury_status_table(extra_outs: pl.DataFrame | None = None) -> pl.DataFrame:
    """(season, week, player_id) -> own report/practice status, ordinal.
    Reports finalize Friday before Sunday games, so they are
    pre-kickoff-safe under the Sunday-morning snapshot rule."""
    return (
        _read_injuries(extra_outs)
        .with_columns(
            pl.col("report_status").replace_strict(STATUS_ORD, default=0)
            .alias("inj_status"),
            pl.col("practice_status").replace_strict(PRACTICE_ORD, default=0)
            .alias("inj_practice"),
        )
        .group_by(["season", "week", pl.col("gsis_id").alias("player_id")])
        .agg(pl.col("inj_status").max(), pl.col("inj_practice").max())
    )


def vacated_table(extra_outs: pl.DataFrame | None = None) -> pl.DataFrame:
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
        _read_injuries(extra_outs)
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


def _with_opp_and_context(df: pl.DataFrame, panel_for_opp: pl.DataFrame,
                          dst: bool = False) -> pl.DataFrame:
    """Opponent-allowed factor, Vegas/venue context, and the play-by-play
    team tables: pace/pass tendency of the offense and split-EPA allowed
    by the defense. For players the offense is their own team and the
    defense is the opponent; for DST the roles flip."""
    opp = opponent_allowed_factor(panel_for_opp)
    off_tbl, def_tbl = team_context_tables()
    own, other = (def_tbl, off_tbl) if dst else (off_tbl, def_tbl)
    other = other.rename({c: c.replace("team_", "opp_", 1)
                          for c in other.columns if c.startswith("team_")})
    return (
        df.join(opp, on=["season", "week", "opponent_team", "position"], how="left")
        .with_columns(pl.col("opp_factor_raw").fill_null(1.0))
        .join(game_context(), on=["game_id", "team"], how="left")
        .join(own, on=["season", "week", "team"], how="left")
        .join(other.rename({"team": "opponent_team"}),
              on=["season", "week", "opponent_team"], how="left")
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


def build_offense_features(skeleton: pl.DataFrame | None = None,
                           extra_outs: pl.DataFrame | None = None) -> pl.DataFrame:
    """Feature table for all history; pass `skeleton` (identity columns for
    an upcoming week, stats null) to also get prediction-ready rows for
    that week. extra_outs adds game-day inactives (see _read_injuries)
    to the injury and vacated-opportunity features."""
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
        .join(player_opportunity(), on=["season", "week", "player_id"], how="left")
        .with_columns([
            # a played game with no red-zone work is a 0, not unknown;
            # skeleton (unplayed) rows stay null so trailing shifts work
            pl.when(pl.col("fd_points").is_not_null())
            .then(pl.col(c).fill_null(0.0)).otherwise(None).alias(c)
            for c in PLAYER_OPP_COLS
        ])
        .drop("position")
        .rename({"position_group": "position"})
        .sort(["player_id", "season", "week"])
        .with_columns(
            pl.int_range(pl.len()).over(["player_id", "season"]).alias("season_prior_games"),
        )
    )
    vac_pos, vac_team = vacated_table(extra_outs)
    pw = (
        pw.with_columns(trailing_exprs(OFFENSE_TRAILING_COLS))
        .join(depth_rank_table(), on=["season", "week", "player_id"], how="left")
        .join(player_draft_table(), on="player_id", how="left")
        .join(injury_status_table(extra_outs), on=["season", "week", "player_id"], how="left")
        .join(vac_pos.rename({"position_out": "position"}),
              on=["season", "week", "team", "position"], how="left")
        .join(vac_team, on=["season", "week", "team"], how="left")
        .join(qb_skill_table(extra_outs), on=["season", "week", "team"], how="left")
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
    return _with_opp_and_context(dst, _panel_with_skeleton(skeleton), dst=True).sort(
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
