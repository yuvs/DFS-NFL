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

# PBP is only needed for DST points-allowed reconstruction; keep the slim
# column set so 10 seasons stay small on disk.
PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "play_type", "touchdown", "td_team", "return_touchdown",
    "field_goal_result", "extra_point_result", "two_point_conv_result",
    "defensive_two_point_conv",
]

SOURCES = {
    "player_stats": lambda yr: nfl.load_player_stats([yr]),
    "team_stats": lambda yr: nfl.load_team_stats([yr]),
    "pbp": lambda yr: nfl.load_pbp([yr]).select(PBP_COLS),
}


def ingest_season(year: int) -> None:
    for name, loader in SOURCES.items():
        out_dir = RAW_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{name}_{year}.parquet"
        df = loader(year)
        df.write_parquet(out)
        print(f"  {out.relative_to(REPO_ROOT)}: {df.height:,} rows")


def ingest_schedules(first: int, last: int) -> None:
    out = RAW_DIR / "schedules.parquet"
    df = nfl.load_schedules().filter(pl.col("season").is_between(first, last))
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


if __name__ == "__main__":
    main()
