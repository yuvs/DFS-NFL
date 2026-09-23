"""Play-by-play derived per-game tables (PLAN.md §3): red-zone / goal-line
opportunity and expected touchdowns per player, team pace and pass
tendencies, and opponent split-EPA allowed.

Everything here is a raw per-game value; features.py turns player tables
into shifted trailing averages, and team tables come out of
team_trailing() already shifted (week W sees games < W only).

Expected TDs: each target / carry / pass attempt is credited the league
TD rate for its (play kind, yardline bucket), with rates taken from
seasons strictly before the play's season (the first season uses its
own). Trailing expected TDs are a far steadier read on scoring
opportunity than trailing actual TDs.
"""

from __future__ import annotations

import polars as pl

from dfs_nfl.baselines import EWMA_HALFLIFE, REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"

PLAYER_OPP_COLS = [
    "rz_targets", "i10_targets", "rz_target_share", "x_rec_td",
    "gl_carries", "rz_carries", "gl_carry_share", "x_rush_td",
    "rz_attempts", "x_pass_td", "x_td",
]
TEAM_OFF_COLS = ["off_plays", "off_pass_rate", "off_pass_oe", "off_epa"]
TEAM_DEF_COLS = ["def_pass_epa", "def_rush_epa", "def_success_allowed"]


def read_pbp(reg_only: bool = True) -> pl.DataFrame:
    files = sorted((RAW_DIR / "pbp").glob("pbp_*.parquet"))
    pbp = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    if reg_only:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    return pbp


def _bucket() -> pl.Expr:
    y = pl.col("yardline_100")
    return (
        pl.when(y <= 2).then(pl.lit("gl2"))
        .when(y <= 5).then(pl.lit("gl5"))
        .when(y <= 10).then(pl.lit("i10"))
        .when(y <= 20).then(pl.lit("rz20"))
        .when(y <= 40).then(pl.lit("m40"))
        .otherwise(pl.lit("far"))
        .alias("bucket")
    )


def _scoring_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Targets, carries and pass attempts with their kind, bucket and
    whether they scored. Sacks, kneels and spikes are not opportunities."""
    base = pbp.filter(
        pl.col("yardline_100").is_not_null()
        & (pl.col("qb_kneel").fill_null(0) == 0)
        & (pl.col("qb_spike").fill_null(0) == 0)
    ).with_columns(_bucket())
    targets = base.filter(
        (pl.col("pass_attempt") == 1) & (pl.col("sack").fill_null(0) == 0)
        & pl.col("receiver_player_id").is_not_null()
    ).select(
        "season", "week", "game_id", pl.col("posteam").alias("team"),
        pl.col("receiver_player_id").alias("player_id"),
        pl.lit("rec").alias("kind"), "bucket", "yardline_100",
        pl.col("pass_touchdown").fill_null(0).alias("td"),
    )
    carries = base.filter(
        (pl.col("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null()
    ).select(
        "season", "week", "game_id", pl.col("posteam").alias("team"),
        pl.col("rusher_player_id").alias("player_id"),
        pl.lit("rush").alias("kind"), "bucket", "yardline_100",
        pl.col("rush_touchdown").fill_null(0).alias("td"),
    )
    attempts = base.filter(
        (pl.col("pass_attempt") == 1) & (pl.col("sack").fill_null(0) == 0)
        & pl.col("passer_player_id").is_not_null()
    ).select(
        "season", "week", "game_id", pl.col("posteam").alias("team"),
        pl.col("passer_player_id").alias("player_id"),
        pl.lit("pass").alias("kind"), "bucket", "yardline_100",
        pl.col("pass_touchdown").fill_null(0).alias("td"),
    )
    return pl.concat([targets, carries, attempts])


def td_rate_table(plays: pl.DataFrame) -> pl.DataFrame:
    """(season, kind, bucket) -> league TD rate from strictly earlier
    seasons; the earliest season falls back to its own rate."""
    per = (
        plays.group_by(["season", "kind", "bucket"])
        .agg(pl.len().alias("n"), pl.col("td").sum().alias("tds"))
        .sort(["kind", "bucket", "season"])
        .with_columns(
            pl.col("n").cum_sum().shift(1).over(["kind", "bucket"]).alias("n_prior"),
            pl.col("tds").cum_sum().shift(1).over(["kind", "bucket"]).alias("tds_prior"),
        )
        .with_columns(
            (pl.coalesce("tds_prior", "tds") / pl.coalesce("n_prior", "n"))
            .alias("td_rate")
        )
    )
    return per.select("season", "kind", "bucket", "td_rate")


def player_opportunity(pbp: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per (season, week, player_id): red-zone / goal-line usage, team
    shares of it, and expected TDs by role."""
    plays = _scoring_plays(read_pbp() if pbp is None else pbp)
    plays = plays.join(td_rate_table(plays), on=["season", "kind", "bucket"], how="left")
    y = pl.col("yardline_100")
    keys = ["season", "week", "game_id", "team", "player_id"]
    rec = plays.filter(pl.col("kind") == "rec").group_by(keys).agg(
        (y <= 20).sum().alias("rz_targets"),
        (y <= 10).sum().alias("i10_targets"),
        pl.col("td_rate").sum().alias("x_rec_td"),
    )
    rush = plays.filter(pl.col("kind") == "rush").group_by(keys).agg(
        (y <= 5).sum().alias("gl_carries"),
        (y <= 20).sum().alias("rz_carries"),
        pl.col("td_rate").sum().alias("x_rush_td"),
    )
    pas = plays.filter(pl.col("kind") == "pass").group_by(keys).agg(
        (y <= 20).sum().alias("rz_attempts"),
        pl.col("td_rate").sum().alias("x_pass_td"),
    )
    team = plays.group_by(["game_id", "team"]).agg(
        ((pl.col("kind") == "rec") & (y <= 20)).sum().alias("team_rz_targets"),
        ((pl.col("kind") == "rush") & (y <= 5)).sum().alias("team_gl_carries"),
    )
    out = (
        rec.join(rush, on=keys, how="full", coalesce=True)
        .join(pas, on=keys, how="full", coalesce=True)
        .join(team, on=["game_id", "team"], how="left")
    )
    count_cols = ["rz_targets", "i10_targets", "x_rec_td", "gl_carries",
                  "rz_carries", "x_rush_td", "rz_attempts", "x_pass_td"]
    return (
        out.with_columns([pl.col(c).fill_null(0).cast(pl.Float64) for c in count_cols])
        .with_columns(
            pl.when(pl.col("team_rz_targets") > 0)
            .then(pl.col("rz_targets") / pl.col("team_rz_targets"))
            .otherwise(0.0).alias("rz_target_share"),
            pl.when(pl.col("team_gl_carries") > 0)
            .then(pl.col("gl_carries") / pl.col("team_gl_carries"))
            .otherwise(0.0).alias("gl_carry_share"),
            (pl.col("x_rec_td") + pl.col("x_rush_td") + pl.col("x_pass_td")).alias("x_td"),
        )
        .select("season", "week", "player_id", *PLAYER_OPP_COLS)
    )


def team_game_tables(pbp: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per (season, week, team): offensive pace/tendency and defensive
    split-EPA allowed, one row per team-game."""
    pbp = read_pbp() if pbp is None else pbp
    real = pbp.filter(
        ((pl.col("qb_dropback") == 1) | (pl.col("rush_attempt") == 1))
        & (pl.col("qb_kneel").fill_null(0) == 0)
        & (pl.col("qb_spike").fill_null(0) == 0)
        & pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null()
    )
    neutral = pl.col("wp").is_between(0.2, 0.8)
    off = real.group_by(["season", "week", pl.col("posteam").alias("team")]).agg(
        pl.len().cast(pl.Float64).alias("off_plays"),
        pl.col("qb_dropback").mean().alias("off_pass_rate"),
        pl.col("pass_oe").filter(neutral).mean().alias("off_pass_oe"),
        pl.col("epa").mean().alias("off_epa"),
    )
    dfn = real.group_by(["season", "week", pl.col("defteam").alias("team")]).agg(
        pl.col("epa").filter(pl.col("qb_dropback") == 1).mean().alias("def_pass_epa"),
        pl.col("epa").filter(pl.col("rush_attempt") == 1).mean().alias("def_rush_epa"),
        pl.col("success").mean().alias("def_success_allowed"),
    )
    return off, dfn


def _team_week_grid() -> pl.DataFrame:
    sched = pl.read_parquet(RAW_DIR / "schedules.parquet").filter(
        pl.col("game_type") == "REG"
    )
    return pl.concat([
        sched.select("season", "week", pl.col("home_team").alias("team")),
        sched.select("season", "week", pl.col("away_team").alias("team")),
    ]).unique().with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)
    )


def team_trailing(table: pl.DataFrame, cols: list[str],
                  prefix: str = "") -> pl.DataFrame:
    """Shifted EWMA of each col per team over the full schedule grid, so an
    unplayed (future) team-week gets the value through its last game.
    Output columns: {prefix}{col}_ewma keyed by (season, week, team)."""
    grid = _team_week_grid()
    joined = (
        grid.join(table.with_columns(pl.col("season").cast(pl.Int32),
                                     pl.col("week").cast(pl.Int32)),
                  on=["season", "week", "team"], how="left")
        .sort(["team", "season", "week"])
    )
    exprs = [
        pl.col(c).cast(pl.Float64).ewm_mean(half_life=EWMA_HALFLIFE, ignore_nulls=True)
        .shift(1).over("team").alias(f"{prefix}{c}_ewma")
        for c in cols
    ]
    return joined.select("season", "week", "team", *exprs)


def team_context_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(offense trailing keyed by team, defense trailing keyed by team).
    Join offense on a player's own team and defense on his opponent;
    for DST the roles flip."""
    off, dfn = team_game_tables()
    return (team_trailing(off, TEAM_OFF_COLS, prefix="team_"),
            team_trailing(dfn, TEAM_DEF_COLS, prefix="team_"))
