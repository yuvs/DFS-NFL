"""Download nflverse data to data/raw/ as parquet, partitioned by season.

Idempotent: re-running overwrites each season's file, so the current season
can be refreshed weekly with the same command.

Usage: uv run python -m dfs_nfl.ingest [first_season] [last_season]
"""

from __future__ import annotations

import sys
from pathlib import Path

import nflreadpy as nfl
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# PBP columns: DST points-allowed reconstruction, plus the play-level
# fields behind the opportunity features (red-zone / goal-line usage,
# expected TDs), team pace/pass rate and opponent split-EPA allowed.
# Columns missing from an older season are simply skipped.
PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "play_type", "touchdown", "td_team", "return_touchdown",
    "field_goal_result", "extra_point_result", "two_point_conv_result",
    "defensive_two_point_conv",
    # play situation
    "play_id", "drive", "down", "yardline_100", "goal_to_go",
    "game_seconds_remaining", "half_seconds_remaining", "wp",
    # play type / outcome
    "pass", "rush", "qb_dropback", "qb_kneel", "qb_spike", "sack",
    "pass_attempt", "rush_attempt", "complete_pass", "interception",
    "fumble_lost", "two_point_attempt", "pass_touchdown", "rush_touchdown",
    "air_yards", "yards_gained", "epa", "success", "xpass", "pass_oe",
    # who
    "passer_player_id", "rusher_player_id", "receiver_player_id",
    "td_player_id",
]


def load_pbp(year: int) -> pl.DataFrame:
    df = nfl.load_pbp([year])
    return df.select([c for c in PBP_COLS if c in df.columns])


SOURCES = {
    "player_stats": lambda yr: nfl.load_player_stats([yr]),
    "team_stats": lambda yr: nfl.load_team_stats([yr]),
    "pbp": load_pbp,
    "snap_counts": lambda yr: nfl.load_snap_counts([yr]),
    "depth_charts": lambda yr: load_depth_charts_normalized(yr),
    "injuries": lambda yr: nfl.load_injuries([yr]).select(
        "season", "game_type", "week", "team", "gsis_id", "position",
        "report_status", "practice_status",
    ),
}


def load_depth_charts_normalized(year: int) -> pl.DataFrame:
    """Normalize both depth-chart schemas (weekly <=2024; timestamped
    snapshots 2025+) to: season, week, team, gsis_id, position, depth_rank."""
    dc = nfl.load_depth_charts([year])
    if "depth_team" in dc.columns:  # old weekly schema
        return (
            dc.filter(pl.col("game_type") == "REG")
            .select(
                "season", "week",
                pl.col("club_code").alias("team"),
                "gsis_id", "position",
                pl.col("depth_team").cast(pl.Int32, strict=False).alias("depth_rank"),
            )
            .drop_nulls(["gsis_id", "depth_rank"])
            .group_by(["season", "week", "team", "gsis_id", "position"])
            .agg(pl.col("depth_rank").min())
        )
    # new snapshot schema: assign each snapshot to the first schedule week
    # whose latest game date is on/after the snapshot date
    sched = (
        nfl.load_schedules()
        .filter((pl.col("season") == year) & (pl.col("game_type") == "REG"))
        .group_by("week")
        .agg(pl.col("gameday").max().alias("week_end"))
        .sort("week")
    )
    weeks = sched.rows()  # [(week, week_end 'YYYY-MM-DD'), ...]
    snap = dc.with_columns(pl.col("dt").cast(pl.Utf8).str.slice(0, 10).alias("snap_date"))
    week_expr = pl.lit(None, dtype=pl.Int32)
    for week, week_end in reversed(weeks):
        week_expr = (
            pl.when(pl.col("snap_date") <= week_end)
            .then(pl.lit(week, dtype=pl.Int32))
            .otherwise(week_expr)
        )
    return (
        snap.with_columns(week_expr.alias("week"), pl.lit(year, dtype=pl.Int32).alias("season"))
        .drop_nulls(["gsis_id", "week", "pos_rank"])
        .select(
            "season", "week", "team", "gsis_id",
            pl.col("pos_abb").alias("position"),
            pl.col("pos_rank").cast(pl.Int32, strict=False).alias("depth_rank"),
        )
        .group_by(["season", "week", "team", "gsis_id", "position"])
        .agg(pl.col("depth_rank").min())
    )


def ingest_season(year: int) -> None:
    for name, loader in SOURCES.items():
        out_dir = RAW_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{name}_{year}.parquet"
        df = loader(year)
        df.write_parquet(out)
        print(f"  {out.relative_to(REPO_ROOT)}: {df.height:,} rows")


def ingest_schedules(first: int = 2016, last: int | None = None) -> None:
    """All seasons from `first` on — no upper cap, so future seasons'
    schedules (and their betting lines) arrive as soon as nflverse has them."""
    out = RAW_DIR / "schedules.parquet"
    df = nfl.load_schedules().filter(pl.col("season") >= first)
    if last is not None:
        df = df.filter(pl.col("season") <= last)
    df.write_parquet(out)
    print(f"  {out.relative_to(REPO_ROOT)}: {df.height:,} rows")


def ingest_playerids() -> None:
    out = RAW_DIR / "ff_playerids.parquet"
    df = nfl.load_ff_playerids()
    df.write_parquet(out)
    print(f"  {out.relative_to(REPO_ROOT)}: {df.height:,} rows")


def ingest_players() -> None:
    out = RAW_DIR / "players.parquet"
    df = nfl.load_players().select(
        "gsis_id", "display_name", "position", "birth_date",
        "draft_year", "draft_round", "draft_pick", "rookie_season",
    )
    df.write_parquet(out)
    print(f"  {out.relative_to(REPO_ROOT)}: {df.height:,} rows")


def main() -> None:
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
    last = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    for year in range(first, last + 1):
        print(f"season {year}")
        ingest_season(year)
    print("schedules")
    ingest_schedules(first, last)
    print("playerids")
    ingest_playerids()
    print("players")
    ingest_players()


if __name__ == "__main__":
    main()
