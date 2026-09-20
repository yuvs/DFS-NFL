"""One-command weekly run: refresh data, score the last completed week,
project the next one — and, when a FanDuel slate CSV for the target
week is present in data/raw/fanduel/, also build the value sheet and
cash/GPP lineups.

Figures out the current season/week from the schedule (first REG week
with an unplayed game), so a scheduled job needs no arguments:

  uv run python -m dfs_nfl.weekly

Automation: launchd agents (com.dfsnfl.weekly.*) run this Tuesday 9am
and Sunday 10am ET — see README.
"""

from __future__ import annotations

import polars as pl

from dfs_nfl.features import RAW_DIR
from dfs_nfl.predict import PROJECTIONS_DIR, predict_week, refresh_data
from dfs_nfl.scorecard import score_week
from dfs_nfl.value import FANDUEL_DIR, build_value_sheet, slate_date


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


def week_slate_file(season: int, week: int):
    """Newest slate CSV whose filename date falls on one of the target
    week's game days, or None. Guards against silently building lineups
    from last week's salaries."""
    sched = pl.read_parquet(RAW_DIR / "schedules.parquet").filter(
        (pl.col("season") == season) & (pl.col("week") == week)
    )
    game_days = set(sched["gameday"].to_list())
    matches = [f for f in sorted(FANDUEL_DIR.glob("*.csv"),
                                 key=lambda p: p.stat().st_mtime)
               if slate_date(f) in game_days]
    return matches[-1] if matches else None


def build_slate_outputs(season: int, week: int) -> None:
    slate = week_slate_file(season, week)
    if slate is None:
        print(f"no slate CSV for {season} w{week} in {FANDUEL_DIR} — "
              "download the FanDuel players list to get value + lineups")
        return
    from dfs_nfl.config import load_config
    from dfs_nfl.optimizer import generate, lineup_summary, write_lineups

    build_value_sheet(season, week, slate)
    lu_cfg = load_config()["lineups"]
    plans = (("cash", lu_cfg["cash"], False),
             ("gpp", lu_cfg["gpp"], lu_cfg["gpp_stack"]))
    for mode, n, stack in plans:
        if n < 1:
            continue
        lineups = generate(season, week, mode, n, min_diff=lu_cfg["min_diff"],
                           locks=[], excludes=[], stack=stack,
                           allow_dst_conflict=False, slate_path=None,
                           max_exposure=lu_cfg["max_exposure"])
        for i, lu in enumerate(lineups, 1):
            s = lineup_summary(lu)
            print(f"  {mode} lineup {i}: proj {s['projection']} "
                  f"range {s['p10_total']}-{s['p90_total']} ${s['salary']:,}")
        if lineups:
            write_lineups(lineups, season, week, mode)


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
    build_slate_outputs(season, week)


if __name__ == "__main__":
    main()
