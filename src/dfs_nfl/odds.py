"""Current-week betting lines for live prediction.

Default source: nflverse schedules (lines refresh nightly during the
season). If ODDS_API_KEY is set, The Odds API overrides with fresher
closing-quality lines.

Usage: uv run python -m dfs_nfl.odds <season> <week>
"""

from __future__ import annotations

import os
import sys

import nflreadpy as nfl
import polars as pl
import requests

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

# The Odds API uses full team names; nflverse uses abbreviations.
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def schedule_lines(season: int, week: int) -> pl.DataFrame:
    """spread_line/total_line for one week, fresh from nflverse."""
    return (
        nfl.load_schedules()
        .filter((pl.col("season") == season) & (pl.col("week") == week))
        .select("game_id", "gameday", "gametime", "home_team", "away_team",
                "spread_line", "total_line", "roof")
    )


def odds_api_lines() -> pl.DataFrame | None:
    """Live spreads/totals from The Odds API, or None if no key set."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return None
    resp = requests.get(ODDS_API_URL, params={
        "apiKey": key, "regions": "us", "markets": "spreads,totals",
        "oddsFormat": "american",
    }, timeout=30)
    resp.raise_for_status()
    rows = []
    for game in resp.json():
        home = TEAM_ABBR.get(game["home_team"])
        away = TEAM_ABBR.get(game["away_team"])
        if not home or not away or not game.get("bookmakers"):
            continue
        spreads, totals = [], []
        for book in game["bookmakers"]:
            for market in book.get("markets", []):
                for out in market.get("outcomes", []):
                    if market["key"] == "spreads" and TEAM_ABBR.get(out["name"]) == home:
                        # nflverse convention: positive = home favored
                        spreads.append(-out["point"])
                    elif market["key"] == "totals" and out["name"] == "Over":
                        totals.append(out["point"])
        if spreads and totals:
            rows.append({
                "home_team": home, "away_team": away,
                "spread_line": sum(spreads) / len(spreads),
                "total_line": sum(totals) / len(totals),
            })
    return pl.DataFrame(rows) if rows else None


def current_lines(season: int, week: int) -> pl.DataFrame:
    lines = schedule_lines(season, week)
    live = odds_api_lines()
    if live is not None:
        lines = (
            lines.join(live, on=["home_team", "away_team"], how="left", suffix="_live")
            .with_columns(
                pl.coalesce("spread_line_live", "spread_line").alias("spread_line"),
                pl.coalesce("total_line_live", "total_line").alias("total_line"),
            )
            .drop("spread_line_live", "total_line_live")
        )
        print("lines: The Odds API (nflverse fallback for unmatched games)")
    else:
        print("lines: nflverse schedules (set ODDS_API_KEY for live odds)")
    return lines


if __name__ == "__main__":
    season, week = int(sys.argv[1]), int(sys.argv[2])
    print(current_lines(season, week))
