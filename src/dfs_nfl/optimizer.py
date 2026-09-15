"""Optimal FanDuel lineup generation from a week's value sheet.

CP-SAT integer program over the matched slate (dfs_nfl.value output).

FanDuel classic roster: QB, 2 RB, 3 WR, TE, FLEX (RB/WR/TE), DST;
salary cap $60,000; max 4 players from one team; players from at least
3 different teams.

Modes (how the objective uses the calibrated quantiles):
  cash — floor-weighted:    maximize 0.5*projection + 0.5*p10
  gpp  — ceiling-weighted:  maximize 0.5*projection + 0.5*p90
  mean — projection only

Lineup range is reported as projection_total +/- 1.28 * sqrt(sum sd^2)
(independence approximation) — summing per-player p10/p90 would badly
overstate the spread of a 9-player total.

Usage:
  uv run python -m dfs_nfl.optimizer <season> <week> [options]
    --mode cash|gpp|mean   (default cash)
    --n N                  lineups to generate (default 1)
    --min-diff K           each lineup differs from every prior one by
                           at least K players (default 2)
    --lock "Name" ...      force players in (repeatable)
    --exclude "Name" ...   force players out (repeatable)
    --stack                require a WR/TE teammate with your QB (GPP staple)
    --allow-dst-conflict   permit offensive players facing your own DST
    --slate PATH           value-sheet CSV (default: newest for the week)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl
from ortools.sat.python import cp_model

from dfs_nfl.baselines import OUTPUTS_DIR
from dfs_nfl.value import VALUE_DIR

LINEUPS_DIR = OUTPUTS_DIR / "lineups"

SALARY_CAP = 60_000
ROSTER_SIZE = 9
SLOT_BOUNDS = {"QB": (1, 1), "RB": (2, 3), "WR": (3, 4), "TE": (1, 2), "DST": (1, 1)}
FLEX_POOL_SIZE = 7  # RB + WR + TE including the FLEX
MAX_PER_TEAM = 4
MIN_TEAMS = 3
OUT_TAGS = ["O", "IR", "D"]

MODE_WEIGHTS = {"cash": ("p10", 0.5), "gpp": ("p90", 0.5), "mean": (None, 0.0)}


def load_pool(season: int, week: int, slate_path: Path | None,
              safe_only: bool = False) -> pl.DataFrame:
    """safe_only additionally drops Q-tagged and cold-start players —
    the cash-mode default, since their floors are unknowable."""
    if slate_path is None:
        candidates = sorted(VALUE_DIR.glob(f"{season}_w{week:02d}_slate*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"no value sheet for {season} w{week} — run dfs_nfl.value first")
        slate_path = candidates[-1]
    pool = (
        pl.read_csv(slate_path)
        .filter(
            pl.col("projection").is_not_null()
            & pl.col("salary").is_not_null()
            & (pl.col("injury").is_null() | ~pl.col("injury").is_in(OUT_TAGS))
        )
        .with_columns(pl.col("sd").fill_null(0.0))
    )
    if safe_only:
        pool = pool.filter(
            (pl.col("injury").is_null() | (pl.col("injury") != "Q"))
            & (pl.col("confidence") != "low")
        )
    return pool


def _score_expr(mode: str) -> pl.Expr:
    quantile_col, w = MODE_WEIGHTS[mode]
    if quantile_col is None:
        return pl.col("projection")
    return (1 - w) * pl.col("projection") + w * pl.col(quantile_col)


def solve_lineup(
    pool: pl.DataFrame,
    mode: str = "cash",
    locks: list[str] | None = None,
    excludes: list[str] | None = None,
    stack: bool = False,
    allow_dst_conflict: bool = False,
    banned_sets: list[list[int]] | None = None,
    min_diff: int = 2,
) -> pl.DataFrame | None:
    """One optimal lineup, or None if infeasible. banned_sets are prior
    lineups (row indices) that the new lineup must differ from."""
    pool = pool.with_columns(_score_expr(mode).alias("score")).with_row_index("idx")
    rows = pool.to_dicts()
    model = cp_model.CpModel()
    x = {r["idx"]: model.new_bool_var(f"x{r['idx']}") for r in rows}

    def rows_where(pred):
        return [r for r in rows if pred(r)]

    model.add(sum(x.values()) == ROSTER_SIZE)
    model.add(sum(r["salary"] * x[r["idx"]] for r in rows) <= SALARY_CAP)
    for pos, (lo, hi) in SLOT_BOUNDS.items():
        at_pos = [x[r["idx"]] for r in rows_where(lambda r, p=pos: r["position"] == p)]
        model.add(sum(at_pos) >= lo)
        model.add(sum(at_pos) <= hi)
    flex_pool = [x[r["idx"]] for r in rows_where(lambda r: r["position"] in ("RB", "WR", "TE"))]
    model.add(sum(flex_pool) == FLEX_POOL_SIZE)

    teams = sorted({r["team"] for r in rows})
    team_used = {}
    for t in teams:
        members = [x[r["idx"]] for r in rows_where(lambda r, t=t: r["team"] == t)]
        model.add(sum(members) <= MAX_PER_TEAM)
        used = model.new_bool_var(f"team_{t}")
        model.add(sum(members) >= 1).only_enforce_if(used)
        model.add(sum(members) == 0).only_enforce_if(used.negated())
        team_used[t] = used
    model.add(sum(team_used.values()) >= MIN_TEAMS)

    def match_names(names):
        idxs = []
        for name in names or []:
            key = re.sub(r"[^a-z]", "", name.lower())
            hits = [r["idx"] for r in rows
                    if re.sub(r"[^a-z]", "", r["fd_name"].lower()) == key]
            if not hits:
                raise ValueError(f"player not in pool (out, unmatched, or typo): {name!r}")
            idxs.append(hits[0])
        return idxs

    for i in match_names(locks):
        model.add(x[i] == 1)
    for i in match_names(excludes):
        model.add(x[i] == 0)

    if stack:
        for qb in rows_where(lambda r: r["position"] == "QB"):
            mates = [x[r["idx"]] for r in rows_where(
                lambda r, t=qb["team"]: r["team"] == t and r["position"] in ("WR", "TE"))]
            model.add(sum(mates) >= 1).only_enforce_if(x[qb["idx"]])

    if not allow_dst_conflict:
        for dst in rows_where(lambda r: r["position"] == "DST"):
            facing = [x[r["idx"]] for r in rows_where(
                lambda r, opp=dst["opponent_team"]:
                r["team"] == opp and r["position"] != "DST")]
            if facing:
                model.add(sum(facing) == 0).only_enforce_if(x[dst["idx"]])

    for prev in banned_sets or []:
        model.add(sum(x[i] for i in prev if i in x) <= ROSTER_SIZE - min_diff)

    model.maximize(sum(round(r["score"] * 100) * x[r["idx"]] for r in rows))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_workers = 4
    solver.parameters.random_seed = 7
    if solver.solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return pool.filter(
        pl.col("idx").is_in([r["idx"] for r in rows if solver.value(x[r["idx"]])])
    )


def validate_lineup(lineup: pl.DataFrame) -> None:
    assert lineup.height == ROSTER_SIZE, "wrong roster size"
    assert lineup["salary"].sum() <= SALARY_CAP, "over the cap"
    counts = dict(lineup.group_by("position").len().rows())
    for pos, (lo, hi) in SLOT_BOUNDS.items():
        assert lo <= counts.get(pos, 0) <= hi, f"{pos} count {counts.get(pos, 0)}"
    assert sum(counts.get(p, 0) for p in ("RB", "WR", "TE")) == FLEX_POOL_SIZE
    assert lineup.group_by("team").len()["len"].max() <= MAX_PER_TEAM
    assert lineup["team"].n_unique() >= MIN_TEAMS


def lineup_summary(lineup: pl.DataFrame) -> dict:
    total_sd = float((lineup["sd"] ** 2).sum() ** 0.5)
    proj = float(lineup["projection"].sum())
    return {
        "salary": int(lineup["salary"].sum()),
        "projection": round(proj, 1),
        "p10_total": round(proj - 1.2816 * total_sd, 1),
        "p90_total": round(proj + 1.2816 * total_sd, 1),
    }


def generate(season: int, week: int, mode: str, n: int, min_diff: int,
             locks: list[str], excludes: list[str], stack: bool,
             allow_dst_conflict: bool, slate_path: Path | None,
             allow_risky: bool = False) -> list[pl.DataFrame]:
    # cash lineups are floor plays: Q-tagged and cold-start players are
    # out unless explicitly allowed (locks still work — lock = opt-in)
    safe_only = mode == "cash" and not allow_risky
    pool = load_pool(season, week, slate_path, safe_only=safe_only)
    lineups, banned = [], []
    for _ in range(n):
        lu = solve_lineup(pool, mode=mode, locks=locks, excludes=excludes,
                          stack=stack, allow_dst_conflict=allow_dst_conflict,
                          banned_sets=banned, min_diff=min_diff)
        if lu is None:
            break
        validate_lineup(lu)
        lineups.append(lu)
        banned.append(lu["idx"].to_list())
    return lineups


SLOT_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 5}

UPLOAD_HEADER = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF"]


def upload_row(lineup: pl.DataFrame) -> list[str]:
    """One FanDuel bulk-upload row: slate player IDs in slot order.
    The extra RB/WR/TE beyond the base counts fills FLEX."""
    by_pos = {
        pos: lineup.filter(pl.col("position") == pos)
        .sort("salary", descending=True)["fanduel_id"].to_list()
        for pos in ("QB", "RB", "WR", "TE", "DST")
    }
    flex = by_pos["RB"][2:] + by_pos["WR"][3:] + by_pos["TE"][1:]
    return (by_pos["QB"][:1] + by_pos["RB"][:2] + by_pos["WR"][:3]
            + by_pos["TE"][:1] + flex + by_pos["DST"][:1])


def write_lineups(lineups: list[pl.DataFrame], season: int, week: int,
                  mode: str) -> None:
    """Readable CSV + FanDuel bulk-upload CSV for a lineup set."""
    LINEUPS_DIR.mkdir(parents=True, exist_ok=True)
    readable = [
        lu.sort(pl.col("position").replace_strict(SLOT_ORDER), "salary",
                descending=[False, True])
        .drop("idx", "score", strict=False)
        .with_columns(pl.lit(i).alias("lineup"))
        for i, lu in enumerate(lineups, 1)
    ]
    pl.concat(readable).write_csv(
        LINEUPS_DIR / f"{season}_w{week:02d}_{mode}.csv")
    # plain text: the header repeats slot names (RB,RB,WR,WR,WR), which
    # a DataFrame can't represent
    upload_path = LINEUPS_DIR / f"{season}_w{week:02d}_{mode}_upload.csv"
    with open(upload_path, "w") as f:
        f.write(",".join(UPLOAD_HEADER) + "\n")
        for lu in lineups:
            row = upload_row(lu)
            if len(row) != len(UPLOAD_HEADER) or any(not v for v in row):
                raise ValueError(f"lineup missing slate IDs for upload: {row}")
            f.write(",".join(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int)
    ap.add_argument("week", type=int)
    ap.add_argument("--mode", choices=list(MODE_WEIGHTS), default="cash")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--min-diff", type=int, default=2)
    ap.add_argument("--lock", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--stack", action="store_true")
    ap.add_argument("--allow-dst-conflict", action="store_true")
    ap.add_argument("--allow-risky", action="store_true",
                    help="let cash mode roster Q-tagged / cold-start players")
    ap.add_argument("--slate", type=Path, default=None)
    args = ap.parse_args()

    lineups = generate(args.season, args.week, args.mode, args.n, args.min_diff,
                       args.lock, args.exclude, args.stack,
                       args.allow_dst_conflict, args.slate,
                       allow_risky=args.allow_risky)
    if not lineups:
        raise SystemExit("no feasible lineup — loosen constraints")

    for i, lu in enumerate(lineups, 1):
        s = lineup_summary(lu)
        print(f"\nlineup {i} [{args.mode}]  salary ${s['salary']:,}  "
              f"proj {s['projection']}  range {s['p10_total']}-{s['p90_total']}")
        show = lu.sort(
            pl.col("position").replace_strict(SLOT_ORDER), "salary",
            descending=[False, True],
        ).select("position", "fd_name", "team", "opponent_team", "salary",
                 "projection", "p10", "p90", "cv", "injury")
        with pl.Config(tbl_rows=9, tbl_width_chars=140, tbl_hide_dataframe_shape=True):
            print(show)

    write_lineups(lineups, args.season, args.week, args.mode)
    stem = LINEUPS_DIR / f"{args.season}_w{args.week:02d}_{args.mode}"
    print(f"\nwrote {stem}.csv and {stem}_upload.csv")


if __name__ == "__main__":
    main()
