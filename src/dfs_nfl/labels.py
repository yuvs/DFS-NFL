"""Build curated label tables from data/raw/.

Outputs (data/curated/):
  player_week.parquet — one row per offensive player-game with fd_points
  dst_week.parquet    — one row per team-defense-game with fd_points

Usage: uv run python -m dfs_nfl.labels
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dfs_nfl import scoring

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CURATED_DIR = REPO_ROOT / "data" / "curated"

OFFENSE_POSITIONS = ("QB", "RB", "WR", "TE")

PLAYER_KEEP_COLS = [
    "player_id", "player_display_name", "position", "position_group",
    "season", "week", "season_type", "game_id", "team", "opponent_team",
    # raw stats worth carrying into the curated table for later features
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "passing_air_yards", "passing_2pt_conversions",
    "carries", "rushing_yards", "rushing_tds", "rushing_2pt_conversions",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_2pt_conversions",
    "target_share", "air_yards_share", "wopr",
    "fumbles_lost_total", "special_teams_tds", "fumble_recovery_tds",
    "fantasy_points", "fantasy_points_ppr",
]


def _read_all(name: str) -> pl.DataFrame:
    files = sorted((RAW_DIR / name).glob(f"{name}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw files for {name}; run dfs_nfl.ingest first")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def build_player_week() -> pl.DataFrame:
    stats = _read_all("player_stats")
    return (
        stats.filter(pl.col("position_group").is_in(OFFENSE_POSITIONS))
        .select(PLAYER_KEEP_COLS)
        .with_columns(scoring.offense_points_expr().alias("fd_points"))
        .sort(["season", "week", "team", "player_id"])
    )


def build_points_allowed() -> pl.DataFrame:
    """Per (game, defense): points scored against the defensive unit only.

    FanDuel: 6*(opp offensive TDs) + opp XP made + 3*opp FG made
    + 2*(opp offensive 2pt conversions). Return/defensive TDs by the
    opponent do NOT count against this defense (their XPs still do).
    """
    pbp = _read_all("pbp")
    off_td = (
        (pl.col("touchdown") == 1)
        & (pl.col("td_team") == pl.col("posteam"))
        & (pl.col("return_touchdown").fill_null(0) == 0)
    ).cast(pl.Int32)
    xp = (pl.col("extra_point_result") == "good").cast(pl.Int32).fill_null(0)
    fg = (pl.col("field_goal_result") == "made").cast(pl.Int32).fill_null(0)
    two_pt = (pl.col("two_point_conv_result") == "success").cast(pl.Int32).fill_null(0)

    return (
        pbp.filter(pl.col("defteam").is_not_null())
        .with_columns(
            (off_td * 6 + xp + fg * 3 + two_pt * 2).alias("pa_play"),
        )
        .group_by(["game_id", "season", "week", "season_type", "defteam"])
        .agg(pl.col("pa_play").sum().alias("points_allowed"))
        .rename({"defteam": "team"})
    )


def build_dst_week() -> pl.DataFrame:
    team_stats = _read_all("team_stats")
    pa = build_points_allowed()
    dst = (
        team_stats.select(
            "season", "week", "season_type", "game_id", "team", "opponent_team",
            "def_sacks", "def_interceptions", "fumble_recovery_opp",
            "def_safeties", "def_fg_blocks", "def_pat_blocks", "def_punt_blocks",
            "def_tds", "special_teams_tds",
        )
        .with_columns(scoring.dst_event_points_expr().alias("event_points"))
        .join(
            pa.select("game_id", "team", "points_allowed"),
            on=["game_id", "team"],
            how="left",
        )
    )
    missing = dst.filter(pl.col("points_allowed").is_null())
    if missing.height:
        raise ValueError(
            f"{missing.height} team-games missing points_allowed; "
            f"e.g. {missing.select('game_id', 'team').head(5).rows()}"
        )
    return dst.with_columns(
        (
            pl.col("event_points")
            + scoring.points_allowed_score_expr("points_allowed")
        )
        .round(2)
        .alias("fd_points")
    ).sort(["season", "week", "team"])


def main() -> None:
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    player_week = build_player_week()
    player_week.write_parquet(CURATED_DIR / "player_week.parquet")
    print(f"player_week: {player_week.height:,} rows, seasons "
          f"{player_week['season'].min()}–{player_week['season'].max()}")
    dst_week = build_dst_week()
    dst_week.write_parquet(CURATED_DIR / "dst_week.parquet")
    print(f"dst_week: {dst_week.height:,} rows")


if __name__ == "__main__":
    main()
