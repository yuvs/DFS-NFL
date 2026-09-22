"""One-command weekly run: refresh data, score the last completed week,
project the next one — and, for every FanDuel slate CSV of the target
week present in data/raw/fanduel/, build its value sheet and lineups:
classic exports get the cash/GPP counts from config.toml, single-game
exports get the tournament (MVP + 5 FLEX) counts.

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
from dfs_nfl.value import (FANDUEL_DIR, build_value_sheet, slate_date, slate_id,
                           value_sheet_path)


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


def week_slate_files(season: int, week: int) -> list:
    """Slate CSVs whose filename date falls on one of the target week's
    game days (oldest first; one per slate ID, newest re-download wins).
    Guards against silently building lineups from last week's salaries."""
    sched = pl.read_parquet(RAW_DIR / "schedules.parquet").filter(
        (pl.col("season") == season) & (pl.col("week") == week)
    )
    game_days = set(sched["gameday"].to_list())
    newest: dict[str, object] = {}
    for f in sorted(FANDUEL_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime):
        if slate_date(f) in game_days:
            newest[slate_id(f)] = f
    return sorted(newest.values(), key=lambda p: p.stat().st_mtime)


def week_slate_file(season: int, week: int):
    """Newest slate CSV for the week, or None."""
    files = week_slate_files(season, week)
    return files[-1] if files else None


def lineup_plans(fmt: str, lu_cfg: dict) -> list[tuple[str, int, bool, float]]:
    """(mode, count, stack, max_exposure) builds for a value sheet of the
    given format ('classic' or 'tournament'), from the [lineups] config.
    Stacks only apply to classic builds — a tournament lineup is one
    game, so it's stacked by construction."""
    if fmt == "tournament":
        cap = lu_cfg["tournament_max_exposure"]
        return [("cash", lu_cfg["tournament_cash"], False, cap),
                ("gpp", lu_cfg["tournament_gpp"], False, cap)]
    cap = lu_cfg["max_exposure"]
    return [("cash", lu_cfg["cash"], False, cap),
            ("gpp", lu_cfg["gpp"], lu_cfg["gpp_stack"], cap)]


def build_slate_outputs(season: int, week: int) -> None:
    slates = week_slate_files(season, week)
    if not slates:
        print(f"no slate CSV for {season} w{week} in {FANDUEL_DIR} — "
              "download the FanDuel players list to get value + lineups")
        return
    from dfs_nfl.config import load_config
    from dfs_nfl.optimizer import (generate, is_tournament, lineup_summary,
                                   write_lineups)

    lu_cfg = load_config()["lineups"]
    for slate in slates:
        sheet = build_value_sheet(season, week, slate)
        sheet_path = value_sheet_path(season, week, slate)
        fmt = "tournament" if is_tournament(sheet) else "classic"
        print(f"{slate.name}: {fmt} slate")
        for mode, n, stack, cap in lineup_plans(fmt, lu_cfg):
            if n < 1:
                continue
            lineups = generate(season, week, mode, n, min_diff=lu_cfg["min_diff"],
                               locks=[], excludes=[], stack=stack,
                               allow_dst_conflict=False, slate_path=sheet_path,
                               max_exposure=cap, fmt=fmt)
            for i, lu in enumerate(lineups, 1):
                s = lineup_summary(lu)
                print(f"  {fmt} {mode} lineup {i}: proj {s['projection']} "
                      f"range {s['p10_total']}-{s['p90_total']} ${s['salary']:,}")
            if lineups:
                stem = write_lineups(lineups, season, week, mode)
                print(f"  wrote {stem}.csv and {stem}_upload.csv")


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
