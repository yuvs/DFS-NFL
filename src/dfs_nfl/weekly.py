"""One-command weekly run: refresh data, score the last completed week,
project the next one — and, for every FanDuel slate CSV of the target
week present in data/raw/fanduel/, build its value sheet and lineups:
classic exports get the cash/GPP counts from config.toml, single-game
exports get the tournament (MVP + 5 FLEX) counts.

Single-game slates are re-projected at slate time: the injury report and
depth chart are refreshed and the two teams are projected into their own
file (outputs/projections/{season}_w{week}_slate{id}), so a Monday
inactive (a WR1 ruled out) reaches the vacated-opportunity features
instead of being missed by the Sunday-morning projection.

Figures out the current season/week from the schedule (first REG week
with an unplayed game), so a scheduled job needs no arguments:

  uv run python -m dfs_nfl.weekly

Automation: launchd agents (com.dfsnfl.weekly.*) run this Tuesday 9am
and Sunday 10am ET — see README.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from dfs_nfl.features import RAW_DIR
from dfs_nfl.predict import (PREGAME_SOURCES, PROJECTIONS_DIR, predict_week,
                             projection_path, refresh_data, refresh_sources)
from dfs_nfl.scorecard import score_week
from dfs_nfl.value import (FANDUEL_DIR, build_value_sheet, load_slate, match_slate,
                           slate_date, slate_id, value_sheet_path)


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


# nflverse gameday/gametime are US Eastern
KICKOFF_TZ = ZoneInfo("America/New_York")


def slate_first_kickoff(season: int, week: int, teams: list[str]) -> datetime | None:
    """Earliest scheduled kickoff among the week's games involving the
    slate's teams, or None if none of them are on the schedule."""
    games = pl.read_parquet(RAW_DIR / "schedules.parquet").filter(
        (pl.col("season") == season) & (pl.col("week") == week)
        & (pl.col("home_team").is_in(teams) | pl.col("away_team").is_in(teams))
        & pl.col("gametime").is_not_null()
    )
    if not games.height:
        return None
    return min(
        datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=KICKOFF_TZ)
        for d, t in games.select("gameday", "gametime").rows()
    )


def slate_started(season: int, week: int, teams: list[str],
                  now: datetime | None = None) -> bool:
    """True once the slate's first game has kicked off. Lineups for it
    can no longer be entered, and rebuilding would overwrite the files
    the lineup scorecard grades. Unknown teams never count as started."""
    kickoff = slate_first_kickoff(season, week, teams)
    if kickoff is None:
        print(f"  no scheduled games found for {teams}; building anyway")
        return False
    return (now or datetime.now(KICKOFF_TZ)) >= kickoff


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


def slate_inactives(season: int, week: int, raw: pl.DataFrame) -> pl.DataFrame:
    """Slate players tagged O/IR/D/NA, mapped to gsis ids through the
    week's main projections. FanDuel's game-day tags (NA = declared
    inactive) arrive after nflverse's Friday injury report, so this is
    the only place a Monday-night scratch is visible before kickoff."""
    from dfs_nfl.optimizer import OUT_TAGS

    tagged = raw.filter(pl.col("injury").is_in(OUT_TAGS)
                        & (pl.col("position") != "DST"))
    proj = pl.read_parquet(projection_path(season, week))
    matched, _ = match_slate(tagged, proj)
    return matched.filter(pl.col("player_id").is_not_null()).select(
        pl.lit(season, dtype=pl.Int32).alias("season"),
        pl.lit(week, dtype=pl.Int32).alias("week"),
        "team", pl.col("player_id").alias("gsis_id"), "position",
    )


def reproject_slate(season: int, week: int, slate, raw: pl.DataFrame):
    """Fresh injuries + depth chart plus the slate's own inactive tags,
    then project just the slate's teams into a slate-tagged file. Falls
    back to the week's main projections (returns None) if the
    re-projection fails for any reason."""
    teams = sorted(raw["team"].unique().to_list())
    tag = f"_slate{slate_id(slate)}"
    try:
        refresh_sources(season, PREGAME_SOURCES)
        outs = slate_inactives(season, week, raw)
        if outs.height:
            print(f"  slate inactives applied: {outs.height}")
        predict_week(season, week, refresh=False, teams=teams, tag=tag,
                     extra_outs=outs)
    except Exception as e:  # noqa: BLE001 - never block lineups on this
        print(f"  slate re-projection failed ({type(e).__name__}: {e}); "
              "using the week's main projections")
        return None
    return projection_path(season, week, tag)


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
        raw = load_slate(slate)
        fmt = "tournament" if is_tournament(raw) else "classic"
        teams = sorted(raw["team"].unique().to_list())
        if slate_started(season, week, teams):
            print(f"{slate.name}: {fmt} slate already started; skipped "
                  "(existing value sheet and lineups kept)")
            continue
        print(f"{slate.name}: {fmt} slate")
        proj = None
        if fmt == "tournament":
            proj = reproject_slate(season, week, slate, raw)
        sheet = build_value_sheet(season, week, slate, projections_path=proj)
        sheet_path = value_sheet_path(season, week, slate)
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


def refresh_backtest_inputs() -> None:
    """Rebuild the baseline predictions and the curated feature store.

    Production projects from features built on the fly, so these two
    files only feed the backtest harnesses (models / experiments /
    quantiles / coldstart) — which means they used to sit at whatever
    week they were last built by hand, and the in-season holdout read
    off them was silently short of the completed weeks. Both rebuilds
    are a couple of seconds, so the weekly run just redoes them.
    """
    from dfs_nfl.baselines import main as build_baselines
    from dfs_nfl.features import main as build_features

    for label, fn in (("baselines", build_baselines), ("features", build_features)):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - never block the week's lineups
            print(f"  {label} rebuild failed ({type(e).__name__}: {e}); "
                  "backtest inputs left as they were")


def main() -> None:
    # schedules first, so next_week() sees the freshest results/season
    from dfs_nfl.ingest import ingest_schedules

    ingest_schedules()
    season, week = next_week()
    print(f"current target: {season} week {week}")
    refresh_data(season)

    print("rebuilding backtest inputs (baselines + feature store)...")
    refresh_backtest_inputs()

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
