"""Kickoff weather forecasts for live prediction weeks (Open-Meteo, free).

Historical games get temp/wind from nflverse schedules; this module fills
the same two features for upcoming outdoor games from the forecast at the
home stadium's coordinates.

Usage: uv run python -m dfs_nfl.weather <season> <week>
"""

from __future__ import annotations

import sys

import polars as pl
import requests

from dfs_nfl.odds import schedule_lines

# Current home stadium coordinates (2025+ franchises).
STADIUM_COORDS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639),
    "JAX": (30.3240, -81.6373), "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389),
    "MIN": (44.9736, -93.2575), "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812), "NYG": (40.8128, -74.0742),
    "NYJ": (40.8128, -74.0742), "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158), "SEA": (47.5952, -122.3316),
    "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9077, -76.8645),
}

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def kickoff_forecast(lat: float, lon: float, gameday: str, gametime: str) -> dict:
    """Temp (F) and wind (mph) at the hour of kickoff (gametime is ET;
    close enough for weather at hourly resolution)."""
    resp = requests.get(FORECAST_URL, params={
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "America/New_York",
        "start_date": gameday, "end_date": gameday,
    }, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    hour = int(gametime.split(":")[0]) if gametime else 13
    idx = min(hour, len(hourly["time"]) - 1)
    return {
        "temp": hourly["temperature_2m"][idx],
        "wind": hourly["wind_speed_10m"][idx],
        "precip_prob": hourly["precipitation_probability"][idx],
    }


def week_weather(season: int, week: int) -> pl.DataFrame:
    """Forecast per game of the given week. Dome/closed games get the
    standard indoor values (70F, no wind)."""
    games = schedule_lines(season, week)
    rows = []
    for g in games.iter_rows(named=True):
        if g["roof"] in ("dome", "closed"):
            rows.append({"game_id": g["game_id"], "temp": 70.0, "wind": 0.0,
                         "precip_prob": 0.0})
            continue
        coords = STADIUM_COORDS.get(g["home_team"])
        if not coords:  # international/neutral site without coords
            rows.append({"game_id": g["game_id"], "temp": None, "wind": None,
                         "precip_prob": None})
            continue
        fc = kickoff_forecast(*coords, g["gameday"], g["gametime"])
        rows.append({"game_id": g["game_id"], **fc})
    return pl.DataFrame(rows)


if __name__ == "__main__":
    season, week = int(sys.argv[1]), int(sys.argv[2])
    print(week_weather(season, week))
