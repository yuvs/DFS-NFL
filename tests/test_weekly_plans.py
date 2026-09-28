"""Config -> per-slate build plans (no data or model needed)."""

from dfs_nfl.config import DEFAULTS
from dfs_nfl.weekly import lineup_plans


def test_tournament_plans_use_tournament_keys():
    cfg = dict(DEFAULTS["lineups"], tournament_cash=2, tournament_gpp=7,
               tournament_max_exposure=0.8, gpp_stack=True, max_exposure=0.5)
    plans = lineup_plans("tournament", cfg)
    assert plans == [("cash", 2, False, 0.8), ("gpp", 7, False, 0.8)]


def test_classic_plans_use_classic_keys():
    cfg = dict(DEFAULTS["lineups"], cash=1, gpp=9, gpp_stack=True, max_exposure=0.5)
    plans = lineup_plans("classic", cfg)
    assert plans == [("cash", 1, False, 0.5), ("gpp", 9, True, 0.5)]


def test_defaults_cover_every_key():
    for key in ("cash", "gpp", "gpp_stack", "min_diff", "max_exposure",
                "tournament_cash", "tournament_gpp", "tournament_max_exposure"):
        assert key in DEFAULTS["lineups"]


def _schedule(tmp_path, monkeypatch):
    import polars as pl
    from dfs_nfl import weekly
    (tmp_path / "raw").mkdir()
    pl.DataFrame({
        "season": [2026] * 3, "week": [3] * 3,
        "gameday": ["2026-09-24", "2026-09-27", "2026-09-27"],
        "gametime": ["20:15", "13:00", "16:25"],
        "home_team": ["GB", "DET", "KC"], "away_team": ["ATL", "BAL", "DAL"],
    }).write_parquet(tmp_path / "raw" / "schedules.parquet")
    monkeypatch.setattr(weekly, "RAW_DIR", tmp_path / "raw")
    return weekly


def test_slate_started_uses_earliest_kickoff(tmp_path, monkeypatch):
    from datetime import datetime
    weekly = _schedule(tmp_path, monkeypatch)
    tz = weekly.KICKOFF_TZ
    sunday_10am = datetime(2026, 9, 27, 10, 0, tzinfo=tz)
    sunday_2pm = datetime(2026, 9, 27, 14, 0, tzinfo=tz)
    # Thursday single-game slate is over by Sunday
    assert weekly.slate_started(2026, 3, ["ATL", "GB"], now=sunday_10am)
    # Sunday main slate: open at 10am, locked once the 1pm games start,
    # even though its 4:25 game hasn't kicked off
    main = ["BAL", "DAL", "DET", "KC"]
    assert not weekly.slate_started(2026, 3, main, now=sunday_10am)
    assert weekly.slate_started(2026, 3, main, now=sunday_2pm)
    # exactly at kickoff counts as started
    assert weekly.slate_started(2026, 3, ["DET"], now=datetime(2026, 9, 27, 13, 0, tzinfo=tz))


def test_unknown_teams_are_never_skipped(tmp_path, monkeypatch):
    from datetime import datetime
    weekly = _schedule(tmp_path, monkeypatch)
    assert weekly.slate_first_kickoff(2026, 3, ["XXX"]) is None
    assert not weekly.slate_started(2026, 3, ["XXX"],
                                    now=datetime(2027, 1, 1, tzinfo=weekly.KICKOFF_TZ))
