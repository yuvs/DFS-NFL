"""Join FanDuel slate salaries to projections -> value sheet.

Reads a FanDuel players-list export from data/raw/fanduel/ (most recent
file by default), matches players to the week's projections, and writes
outputs/value/{season}_w{week:02d}_slate{id}.csv with salary, value
(projected points per $1K), and ceiling value (p90 per $1K).

Matching: DSTs by team; players by normalized name + team + position,
falling back to name + position (handles team-data lags), then unique
last name + team + position. Unmatched slate players are reported.

Usage: uv run python -m dfs_nfl.value <season> <week> [slate_csv_path]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import polars as pl

from dfs_nfl.baselines import OUTPUTS_DIR
from dfs_nfl.features import RAW_DIR
from dfs_nfl.predict import PROJECTIONS_DIR

FANDUEL_DIR = RAW_DIR / "fanduel"
VALUE_DIR = OUTPUTS_DIR / "value"

# FanDuel team codes that differ from nflverse
TEAM_FIX = {"JAC": "JAX", "LAR": "LA"}


def latest_slate_file() -> Path:
    files = sorted(FANDUEL_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no slate CSVs in {FANDUEL_DIR}")
    return files[-1]


def slate_id(path: Path) -> str:
    m = re.search(r"(\d+)-players-list", path.name)
    return m.group(1) if m else "slate"


def name_key(col: str) -> pl.Expr:
    """Normalize a name for matching: lowercase alpha only, common
    suffixes stripped (jr/sr/ii-v)."""
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"\b(jr|sr|ii|iii|iv|v)\b\.?$", "")
        .str.replace_all(r"[^a-z]", "")
        .alias("name_key")
    )


def load_slate(path: Path) -> pl.DataFrame:
    raw = pl.read_csv(path)
    return raw.select(
        pl.col("Id").alias("fanduel_id"),
        pl.when(pl.col("Position") == "D").then(pl.lit("DST"))
        .otherwise(pl.col("Position")).alias("position"),
        pl.col("Nickname").alias("fd_name"),
        pl.col("Last Name").alias("last_name"),
        pl.col("Team").replace(TEAM_FIX).alias("team"),
        pl.col("Salary").cast(pl.Int64).alias("salary"),
        pl.col("Injury Indicator").alias("injury"),
        pl.col("Roster Position").alias("roster_position"),
    )


def match_slate(slate: pl.DataFrame, proj: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (matched rows with projection columns, unmatched slate rows)."""
    proj = proj.with_columns(name_key("name"))

    dst = (
        slate.filter(pl.col("position") == "DST")
        .join(proj.filter(pl.col("position") == "DST"),
              on=["team", "position"], how="left")
    )

    players = slate.filter(pl.col("position") != "DST").with_columns(
        name_key("fd_name")
    )
    pp = proj.filter(pl.col("position") != "DST")

    # pass 1: name + team + position
    m1 = players.join(pp, on=["name_key", "team", "position"], how="left")
    matched = m1.filter(pl.col("projection").is_not_null())
    rest = m1.filter(pl.col("projection").is_null()).select(players.columns)

    # pass 2: name + position (team moved / data lag)
    m2 = rest.join(pp.unique(subset=["name_key", "position"], keep="none"),
                   on=["name_key", "position"], how="left", suffix="_p")
    matched2 = m2.filter(pl.col("projection").is_not_null())
    rest = m2.filter(pl.col("projection").is_null()).select(players.columns)

    # pass 3: unique last name + team + position on both sides
    pp_last = pp.with_columns(
        pl.col("name").str.extract(r"(\S+)$").str.to_lowercase()
        .str.replace_all(r"[^a-z]", "").alias("lname")
    ).unique(subset=["lname", "team", "position"], keep="none")
    m3 = rest.with_columns(
        pl.col("last_name").str.to_lowercase().str.replace_all(r"[^a-z]", "")
        .alias("lname")
    ).join(pp_last, on=["lname", "team", "position"], how="left", suffix="_p")
    matched3 = m3.filter(pl.col("projection").is_not_null())
    unmatched = m3.filter(pl.col("projection").is_null()).select(players.columns)

    keep = ["fanduel_id", "fd_name", "position", "team", "salary", "injury",
            "roster_position", "player_id", "name", "opponent_team",
            "projection", "p10", "p50", "p90", "sd", "cv", "skew",
            "team_implied_total", "opp_implied_total", "confidence"]
    parts = [df.select([c for c in keep if c in df.columns])
             for df in (dst, matched, matched2, matched3)]
    return pl.concat(parts, how="diagonal_relaxed"), unmatched


def build_value_sheet(season: int, week: int, slate_path: Path | None = None) -> pl.DataFrame:
    slate_path = slate_path or latest_slate_file()
    proj_path = PROJECTIONS_DIR / f"{season}_w{week:02d}.parquet"
    if not proj_path.exists():
        raise FileNotFoundError(f"run dfs_nfl.predict first: {proj_path}")
    slate = load_slate(slate_path)
    proj = pl.read_parquet(proj_path)

    matched, unmatched = match_slate(slate, proj)
    sheet = (
        matched.filter(pl.col("projection").is_not_null())
        .with_columns(
            (pl.col("projection") / (pl.col("salary") / 1000)).round(3).alias("value"),
            (pl.col("p90") / (pl.col("salary") / 1000)).round(3).alias("value_p90"),
        )
        .sort("value", descending=True)
    )
    VALUE_DIR.mkdir(parents=True, exist_ok=True)
    out = VALUE_DIR / f"{season}_w{week:02d}_slate{slate_id(slate_path)}.csv"
    sheet.write_csv(out)
    print(f"wrote {out} ({sheet.height} of {slate.height} slate players matched)")

    if unmatched.height:
        notable = unmatched.filter(pl.col("salary") >= 5000).sort("salary", descending=True)
        print(f"unmatched: {unmatched.height} "
              f"(mostly deep bench; {notable.height} at salary >= $5k)")
        if notable.height:
            print(notable.select("fd_name", "position", "team", "salary").head(10))
    return sheet


def main() -> None:
    season, week = int(sys.argv[1]), int(sys.argv[2])
    path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    sheet = build_value_sheet(season, week, path)
    # null injury = healthy; is_in on null would be null and drop them
    playable = sheet.filter(
        pl.col("injury").is_null() | ~pl.col("injury").is_in(["O", "IR", "D"])
    )
    with pl.Config(tbl_rows=15, tbl_width_chars=160):
        print("\ntop value (projected pts per $1K, excluding O/IR/D):")
        print(playable.select("fd_name", "position", "team", "salary",
                              "projection", "value", "cv", "injury").head(15))


if __name__ == "__main__":
    main()
