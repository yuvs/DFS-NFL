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
