"""FanDuel NFL scoring (2024+ rules, verified vs fanduel.com/rules Sep 2026).

Scoring values live in config dicts so a future FanDuel rule change is a
one-line edit here, re-verified each September.
"""

from __future__ import annotations

import polars as pl

# --- Offensive players (half-PPR + 2024 yardage bonuses) ---
OFFENSE_SCORING = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "passing_interceptions": -1.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "receptions": 0.5,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "fumbles_lost": -2.0,
    "two_point_conversions": 2.0,
    "return_tds": 6.0,  # kick/punt return TD, own-fumble-recovery TD
}
PASS_YD_BONUS_THRESHOLD = 300
RUSH_YD_BONUS_THRESHOLD = 100
REC_YD_BONUS_THRESHOLD = 100
YARDAGE_BONUS = 3.0

# --- Team defense / special teams ---
DST_SCORING = {
    "sacks": 1.0,
    "interceptions": 2.0,
    "fumble_recoveries": 2.0,
    "safeties": 2.0,
    "blocked_kicks": 2.0,  # blocked punt, FG, or XP
    "tds": 6.0,  # defensive + return TDs
}

# FanDuel points-allowed brackets: (max points allowed inclusive, FD points).
# "Points allowed" counts only points scored against the defensive unit:
# 6*(rush TD + rec TD + own-fumble-recovery TD) + 2*(2pt conv) + XP + 3*FG.
DST_POINTS_ALLOWED_BRACKETS = [
    (0, 10.0),
    (6, 7.0),
    (13, 4.0),
    (20, 1.0),
    (27, 0.0),
    (34, -1.0),
    (10_000, -4.0),
]


def points_allowed_score(points_allowed: int) -> float:
    if points_allowed < 0:
        raise ValueError(f"negative points allowed? {points_allowed}")
    for ceiling, score in DST_POINTS_ALLOWED_BRACKETS:
        if points_allowed <= ceiling:
            return score
    raise ValueError(f"negative points allowed? {points_allowed}")


def offense_points(
    *,
    passing_yards: float = 0,
    passing_tds: int = 0,
    passing_interceptions: int = 0,
    rushing_yards: float = 0,
    rushing_tds: int = 0,
    receptions: int = 0,
    receiving_yards: float = 0,
    receiving_tds: int = 0,
    fumbles_lost: int = 0,
    two_point_conversions: int = 0,
    return_tds: int = 0,
) -> float:
    """Scalar FanDuel points for one offensive player-game."""
    s = OFFENSE_SCORING
    pts = (
        passing_yards * s["passing_yards"]
        + passing_tds * s["passing_tds"]
        + passing_interceptions * s["passing_interceptions"]
        + rushing_yards * s["rushing_yards"]
        + rushing_tds * s["rushing_tds"]
        + receptions * s["receptions"]
        + receiving_yards * s["receiving_yards"]
        + receiving_tds * s["receiving_tds"]
        + fumbles_lost * s["fumbles_lost"]
        + two_point_conversions * s["two_point_conversions"]
        + return_tds * s["return_tds"]
    )
    if passing_yards >= PASS_YD_BONUS_THRESHOLD:
        pts += YARDAGE_BONUS
    if rushing_yards >= RUSH_YD_BONUS_THRESHOLD:
        pts += YARDAGE_BONUS
    if receiving_yards >= REC_YD_BONUS_THRESHOLD:
        pts += YARDAGE_BONUS
    return round(pts, 2)


def dst_points(
    *,
    sacks: float = 0,
    interceptions: int = 0,
    fumble_recoveries: int = 0,
    safeties: int = 0,
    blocked_kicks: int = 0,
    tds: int = 0,
    points_allowed: int = 0,
) -> float:
    """Scalar FanDuel points for one team-defense game."""
    s = DST_SCORING
    pts = (
        sacks * s["sacks"]
        + interceptions * s["interceptions"]
        + fumble_recoveries * s["fumble_recoveries"]
        + safeties * s["safeties"]
        + blocked_kicks * s["blocked_kicks"]
        + tds * s["tds"]
    )
    return round(pts + points_allowed_score(points_allowed), 2)


def offense_points_expr() -> pl.Expr:
    """Polars expression computing FanDuel points from an nflverse
    player_stats frame (columns as in nflreadpy.load_player_stats)."""
    s = OFFENSE_SCORING

    def col(name: str) -> pl.Expr:
        return pl.col(name).fill_null(0)

    two_pt = (
        col("passing_2pt_conversions")
        + col("rushing_2pt_conversions")
        + col("receiving_2pt_conversions")
    )
    # special_teams_tds = kick/punt return TDs; fumble_recovery_tds = own
    # fumbles recovered in the end zone. Both are 6 on FanDuel.
    return_tds = col("special_teams_tds") + col("fumble_recovery_tds")
    base = (
        col("passing_yards") * s["passing_yards"]
        + col("passing_tds") * s["passing_tds"]
        + col("passing_interceptions") * s["passing_interceptions"]
        + col("rushing_yards") * s["rushing_yards"]
        + col("rushing_tds") * s["rushing_tds"]
        + col("receptions") * s["receptions"]
        + col("receiving_yards") * s["receiving_yards"]
        + col("receiving_tds") * s["receiving_tds"]
        + col("fumbles_lost_total") * s["fumbles_lost"]
        + two_pt * s["two_point_conversions"]
        + return_tds * s["return_tds"]
    )
    bonuses = (
        (col("passing_yards") >= PASS_YD_BONUS_THRESHOLD).cast(pl.Float64) * YARDAGE_BONUS
        + (col("rushing_yards") >= RUSH_YD_BONUS_THRESHOLD).cast(pl.Float64) * YARDAGE_BONUS
        + (col("receiving_yards") >= REC_YD_BONUS_THRESHOLD).cast(pl.Float64) * YARDAGE_BONUS
    )
    return (base + bonuses).round(2)


def dst_event_points_expr() -> pl.Expr:
    """Polars expression: DST points *excluding* points-allowed, from an
    nflverse team_stats frame (columns as in nflreadpy.load_team_stats)."""
    s = DST_SCORING

    def col(name: str) -> pl.Expr:
        return pl.col(name).fill_null(0)

    blocked = col("def_fg_blocks") + col("def_pat_blocks") + col("def_punt_blocks")
    tds = col("def_tds") + col("special_teams_tds")
    return (
        col("def_sacks") * s["sacks"]
        + col("def_interceptions") * s["interceptions"]
        + col("fumble_recovery_opp") * s["fumble_recoveries"]
        + col("def_safeties") * s["safeties"]
        + blocked * s["blocked_kicks"]
        + tds * s["tds"]
    ).round(2)


def points_allowed_score_expr(points_allowed_col: str = "points_allowed") -> pl.Expr:
    expr = pl.lit(DST_POINTS_ALLOWED_BRACKETS[-1][1])
    for ceiling, score in reversed(DST_POINTS_ALLOWED_BRACKETS[:-1]):
        expr = (
            pl.when(pl.col(points_allowed_col) <= ceiling)
            .then(pl.lit(score))
            .otherwise(expr)
        )
    return expr
