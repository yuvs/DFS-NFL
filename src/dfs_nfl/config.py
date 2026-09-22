"""Repo-root config.toml, merged over defaults. Missing file or missing
keys fall back silently so automation never breaks on a config edit."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.toml"

DEFAULTS = {
    "lineups": {
        # classic (9-man) builds from full-slate exports
        "cash": 1, "gpp": 3, "gpp_stack": True, "min_diff": 2,
        "max_exposure": 1.0,
        # tournament (MVP + 5 FLEX) builds from single-game exports
        "tournament_cash": 1, "tournament_gpp": 3,
        "tournament_max_exposure": 1.0,
    },
}


def load_config() -> dict:
    cfg = {section: dict(values) for section, values in DEFAULTS.items()}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            user = tomllib.load(f)
        for section, values in user.items():
            cfg.setdefault(section, {}).update(values)
    return cfg
