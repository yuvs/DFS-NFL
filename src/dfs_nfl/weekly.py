"""One-command weekly run: refresh data, score the last completed week,
project the next one.

Figures out the current season/week from the schedule (first REG week
with an unplayed game), so a cron entry needs no arguments:

  uv run python -m dfs_nfl.weekly

Cron examples (see README): Sunday 10am ET final projections, Tuesday
9am ET scorecard + refresh.
"""

from __future__ import annotations

import polars as pl

from dfs_nfl.features import RAW_DIR
from dfs_nfl.predict import PROJECTIONS_DIR, predict_week, refresh_data
from dfs_nfl.scorecard import score_week


def next_week() -> tuple[int, int]:
    sched = (
        pl.read_parquet(RAW_DIR / "schedules.parquet")
        .filter(pl.col("game_type") == "REG")
    )
    upcoming = sched.filter(pl.col("home_score").is_null())
    if not upcoming.height:
        raise SystemExit("no unplayed games in schedules — refresh or extend seasons")
    row = upcoming.sort(["season", "week"]).row(0, named=True)
    return row["season"], row["week"]


def main() -> None:
    # schedules first, so next_week() sees the freshest results/season
    from dfs_nfl.ingest import ingest_schedules

    ingest_schedules()
    season, week = next_week()
    print(f"current target: {season} week {week}")
    refresh_data(season)

    prior = PROJECTIONS_DIR / f"{season}_w{week - 1:02d}.parquet"
    if week > 1 and prior.exists():
        try:
            score_week(season, week - 1)
        except ValueError as e:
            print(f"scorecard skipped: {e}")

    predict_week(season, week, refresh=False)


if __name__ == "__main__":
    main()
