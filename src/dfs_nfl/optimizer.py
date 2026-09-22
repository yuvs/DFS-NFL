"""Optimal FanDuel lineup generation from a week's value sheet.

CP-SAT integer program over the matched slate (dfs_nfl.value output).

FanDuel classic roster: QB, 2 RB, 3 WR, TE, FLEX (RB/WR/TE), DST;
salary cap $60,000; max 4 players from one team; players from at least
3 different teams.

Tournament (FanDuel single-game) format: 6 players — one MVP at 1.5x
points and 1.5x salary plus 5 FLEX of any position, same $60k cap, both
teams represented. Built from a single-game players-list export (that's
where the MVP salaries and slate IDs come from) and auto-detected from
its MVP roster positions; --format tournament|classic asserts the
expected format instead of trusting detection. Stack and DST-conflict
rules don't apply. Outputs get a _tournament_{AWAY-HOME} suffix so
Thursday and Monday builds in the same week don't overwrite each other.

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
    --format auto|classic|tournament
                           roster format; auto (default) detects it from
                           the value sheet, the others fail loudly on a
                           mismatch
    --max-exposure F       max share of the N lineups any player may be in
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
# FanDuel injury indicators that mean the player will not play
# (NA = "Not Active": declared inactive / coach's decision)
OUT_TAGS = ["O", "IR", "D", "NA"]

# FanDuel tournament / single-game: MVP (1.5x points, 1.5x salary) +
# 5 FLEX of any position, same $60k cap, both teams must be represented
TOURNAMENT_SIZE = 6
MVP_POINTS_MULT = 1.5

FORMATS = ("auto", "classic", "tournament")


def is_tournament(pool: pl.DataFrame) -> bool:
    """Single-game exports list an MVP roster position; classic ones don't."""
    return ("roster_position" in pool.columns
            and pool["roster_position"].str.contains("MVP").any())


def resolve_format(pool: pl.DataFrame, fmt: str = "auto") -> str:
    """'classic' or 'tournament' for this pool; raises if a requested
    format doesn't match what the value sheet supports."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}: {fmt!r}")
    detected = "tournament" if is_tournament(pool) else "classic"
    if fmt != "auto" and fmt != detected:
        hint = ("download the single-game players list from the FanDuel "
                "lobby — MVP salaries and slate IDs come from that export"
                if fmt == "tournament" else
                "this is a single-game export; classic needs the full-slate list")
        raise ValueError(f"value sheet is {detected}, not {fmt}: {hint}")
    return detected


MODE_WEIGHTS = {"cash": ("p10", 0.5), "gpp": ("p90", 0.5), "mean": (None, 0.0)}


def load_pool(season: int, week: int, slate_path: Path | None,
              safe_only: bool = False) -> pl.DataFrame:
    """safe_only additionally drops Q-tagged and cold-start players —
    the cash-mode default, since their floors are unknowable."""
    if slate_path is None:
        # newest sheet by mtime — slate IDs aren't chronological
        candidates = sorted(VALUE_DIR.glob(f"{season}_w{week:02d}_slate*.csv"),
                            key=lambda p: p.stat().st_mtime)
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
    if "mvp_salary" in pool.columns:
        pool = pool.with_columns(
            pl.col("mvp_salary")
            .fill_null((pl.col("salary") * 1.5).cast(pl.Int64))
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
    banned_idx: set[int] | None = None,
) -> pl.DataFrame | None:
    """One optimal lineup, or None if infeasible. banned_sets are prior
    lineups (row indices) that the new lineup must differ from;
    banned_idx are individual rows kept out (exposure caps)."""
    pool = pool.with_columns(_score_expr(mode).alias("score")).with_row_index("idx")
    rows = pool.to_dicts()
    model = cp_model.CpModel()
    x = {r["idx"]: model.new_bool_var(f"x{r['idx']}") for r in rows}

    def rows_where(pred):
        return [r for r in rows if pred(r)]

    single = is_tournament(pool)
    roster_size = TOURNAMENT_SIZE if single else ROSTER_SIZE
    teams = sorted({r["team"] for r in rows})
    mvp = None
    if single:
        # one MVP slot at 1.5x salary; every position is FLEX-eligible
        mvp = {r["idx"]: model.new_bool_var(f"m{r['idx']}") for r in rows}
        model.add(sum(x.values()) == TOURNAMENT_SIZE)
        model.add(sum(mvp.values()) == 1)
        for i, m in mvp.items():
            model.add(m <= x[i])
        model.add(sum(r["salary"] * x[r["idx"]]
                      + (r["mvp_salary"] - r["salary"]) * mvp[r["idx"]]
                      for r in rows) <= SALARY_CAP)
        for t in teams:  # both teams must be represented
            members = [x[r["idx"]] for r in rows_where(lambda r, t=t: r["team"] == t)]
            model.add(sum(members) >= 1)
    else:
        model.add(sum(x.values()) == ROSTER_SIZE)
        model.add(sum(r["salary"] * x[r["idx"]] for r in rows) <= SALARY_CAP)
        for pos, (lo, hi) in SLOT_BOUNDS.items():
            at_pos = [x[r["idx"]] for r in rows_where(lambda r, p=pos: r["position"] == p)]
            model.add(sum(at_pos) >= lo)
            model.add(sum(at_pos) <= hi)
        flex_pool = [x[r["idx"]] for r in rows_where(lambda r: r["position"] in ("RB", "WR", "TE"))]
        model.add(sum(flex_pool) == FLEX_POOL_SIZE)

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
    for i in banned_idx or ():
        if i in x:
            model.add(x[i] == 0)

    if stack and not single:
        for qb in rows_where(lambda r: r["position"] == "QB"):
            mates = [x[r["idx"]] for r in rows_where(
                lambda r, t=qb["team"]: r["team"] == t and r["position"] in ("WR", "TE"))]
            model.add(sum(mates) >= 1).only_enforce_if(x[qb["idx"]])

    # with only two teams a DST-conflict ban would force a one-sided
    # lineup, so it only applies to classic slates
    if not allow_dst_conflict and not single:
        for dst in rows_where(lambda r: r["position"] == "DST"):
            facing = [x[r["idx"]] for r in rows_where(
                lambda r, opp=dst["opponent_team"]:
                r["team"] == opp and r["position"] != "DST")]
            if facing:
                model.add(sum(facing) == 0).only_enforce_if(x[dst["idx"]])

    for prev in banned_sets or []:
        model.add(sum(x[i] for i in prev if i in x) <= roster_size - min_diff)

    obj = sum(round(r["score"] * 100) * x[r["idx"]] for r in rows)
    if single:
        obj += sum(round(r["score"] * 100 * (MVP_POINTS_MULT - 1)) * mvp[r["idx"]]
                   for r in rows)
    model.maximize(obj)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_workers = 4
    solver.parameters.random_seed = 7
    if solver.solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    out = pool.filter(
        pl.col("idx").is_in([r["idx"] for r in rows if solver.value(x[r["idx"]])])
    )
    if single:
        mvp_idx = next(i for i, m in mvp.items() if solver.value(m))
        out = out.with_columns((pl.col("idx") == mvp_idx).alias("is_mvp"))
    return out


def validate_lineup(lineup: pl.DataFrame) -> None:
    if "is_mvp" in lineup.columns:
        assert lineup.height == TOURNAMENT_SIZE, "wrong roster size"
        assert lineup["is_mvp"].sum() == 1, "need exactly one MVP"
        spend = sum(r["mvp_salary"] if r["is_mvp"] else r["salary"]
                    for r in lineup.to_dicts())
        assert spend <= SALARY_CAP, "over the cap"
        assert lineup["team"].n_unique() == 2, "both teams required"
        return
    assert lineup.height == ROSTER_SIZE, "wrong roster size"
    assert lineup["salary"].sum() <= SALARY_CAP, "over the cap"
    counts = dict(lineup.group_by("position").len().rows())
    for pos, (lo, hi) in SLOT_BOUNDS.items():
        assert lo <= counts.get(pos, 0) <= hi, f"{pos} count {counts.get(pos, 0)}"
    assert sum(counts.get(p, 0) for p in ("RB", "WR", "TE")) == FLEX_POOL_SIZE
    assert lineup.group_by("team").len()["len"].max() <= MAX_PER_TEAM
    assert lineup["team"].n_unique() >= MIN_TEAMS


def lineup_summary(lineup: pl.DataFrame) -> dict:
    single = "is_mvp" in lineup.columns
    rows = lineup.to_dicts()
    mult = {True: MVP_POINTS_MULT, False: 1.0}
    proj = sum(r["projection"] * mult[single and r["is_mvp"]] for r in rows)
    total_sd = sum((r["sd"] * mult[single and r["is_mvp"]]) ** 2
                   for r in rows) ** 0.5
    salary = sum(r["mvp_salary"] if single and r["is_mvp"] else r["salary"]
                 for r in rows)
    return {
        "salary": int(salary),
        "projection": round(proj, 1),
        "p10_total": round(proj - 1.2816 * total_sd, 1),
        "p90_total": round(proj + 1.2816 * total_sd, 1),
    }


def generate(season: int, week: int, mode: str, n: int, min_diff: int,
             locks: list[str], excludes: list[str], stack: bool,
             allow_dst_conflict: bool, slate_path: Path | None,
             allow_risky: bool = False,
             max_exposure: float = 1.0,
             fmt: str = "auto") -> list[pl.DataFrame]:
    """max_exposure caps how many of the n lineups any one player can
    appear in (fraction, 1.0 = no cap; locked players are exempt).
    Applied greedily: once a player hits the cap they're out of the
    remaining builds, so later lineups project slightly lower.
    fmt: 'auto' builds whatever format the value sheet supports;
    'classic' / 'tournament' raise if the sheet is the other kind."""
    # cash lineups are floor plays: Q-tagged and cold-start players are
    # out unless explicitly allowed (locks still work — lock = opt-in)
    safe_only = mode == "cash" and not allow_risky
    pool = load_pool(season, week, slate_path, safe_only=safe_only)
    resolve_format(pool, fmt)
    if not 0 < max_exposure <= 1:
        raise ValueError(f"max_exposure must be in (0, 1]: {max_exposure}")
    cap = max(1, int(max_exposure * n + 1e-9))
    lock_keys = {re.sub(r"[^a-z]", "", name.lower()) for name in locks}
    lineups, banned, capped = [], [], set()
    appearances: dict[int, int] = {}
    for _ in range(n):
        lu = solve_lineup(pool, mode=mode, locks=locks, excludes=excludes,
                          stack=stack, allow_dst_conflict=allow_dst_conflict,
                          banned_sets=banned, min_diff=min_diff,
                          banned_idx=capped)
        if lu is None:
            break
        validate_lineup(lu)
        lineups.append(lu)
        banned.append(lu["idx"].to_list())
        for idx, name in zip(lu["idx"], lu["fd_name"]):
            appearances[idx] = appearances.get(idx, 0) + 1
            locked = re.sub(r"[^a-z]", "", name.lower()) in lock_keys
            if appearances[idx] >= cap and not locked:
                capped.add(idx)
    return lineups


SLOT_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 4, "DST": 5}

UPLOAD_HEADER = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF"]
TOURNAMENT_UPLOAD_HEADER = ["MVP - 1.5X Points",
                            "AnyFLEX", "AnyFLEX", "AnyFLEX", "AnyFLEX", "AnyFLEX"]


def lineup_display_sort(lineup: pl.DataFrame) -> pl.DataFrame:
    """MVP first for tournament; slot order then salary for classic."""
    if "is_mvp" in lineup.columns:
        return lineup.sort(["is_mvp", "salary"], descending=[True, True])
    return lineup.sort(pl.col("position").replace_strict(SLOT_ORDER), "salary",
                       descending=[False, True])


def upload_row(lineup: pl.DataFrame) -> list[str]:
    """One FanDuel bulk-upload row: slate player IDs in slot order.
    The extra RB/WR/TE beyond the base counts fills FLEX."""
    if "is_mvp" in lineup.columns:
        return lineup_display_sort(lineup)["fanduel_id"].to_list()
    by_pos = {
        pos: lineup.filter(pl.col("position") == pos)
        .sort("salary", descending=True)["fanduel_id"].to_list()
        for pos in ("QB", "RB", "WR", "TE", "DST")
    }
    flex = by_pos["RB"][2:] + by_pos["WR"][3:] + by_pos["TE"][1:]
    return (by_pos["QB"][:1] + by_pos["RB"][:2] + by_pos["WR"][:3]
            + by_pos["TE"][:1] + flex + by_pos["DST"][:1])


def lineup_stem(lineups: list[pl.DataFrame], season: int, week: int,
                mode: str) -> Path:
    """Output path stem: {season}_w{week}_{mode} for classic, plus
    _tournament_{TEAM-TEAM} (alphabetical) for single-game builds."""
    stem = f"{season}_w{week:02d}_{mode}"
    if lineups and "is_mvp" in lineups[0].columns:
        stem += "_tournament_" + "-".join(sorted(set(lineups[0]["team"])))
    return LINEUPS_DIR / stem


def write_lineups(lineups: list[pl.DataFrame], season: int, week: int,
                  mode: str) -> Path:
    """Readable CSV + FanDuel bulk-upload CSV for a lineup set; returns
    the shared path stem."""
    LINEUPS_DIR.mkdir(parents=True, exist_ok=True)
    single = bool(lineups) and "is_mvp" in lineups[0].columns
    stem = lineup_stem(lineups, season, week, mode)
    readable = [
        lineup_display_sort(lu)
        .drop("idx", "score", strict=False)
        .with_columns(pl.lit(i).alias("lineup"))
        for i, lu in enumerate(lineups, 1)
    ]
    pl.concat(readable).write_csv(stem.with_name(stem.name + ".csv"))
    # plain text: the header repeats slot names (RB,RB,WR,WR,WR), which
    # a DataFrame can't represent
    header = TOURNAMENT_UPLOAD_HEADER if single else UPLOAD_HEADER
    upload_path = stem.with_name(stem.name + "_upload.csv")
    with open(upload_path, "w") as f:
        f.write(",".join(header) + "\n")
        for lu in lineups:
            row = upload_row(lu)
            if len(row) != len(header) or any(not v for v in row):
                raise ValueError(f"lineup missing slate IDs for upload: {row}")
            f.write(",".join(row) + "\n")
    return stem


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
    ap.add_argument("--max-exposure", type=float, default=1.0,
                    help="max share of the N lineups any player may appear in "
                         "(e.g. 0.5); locked players are exempt")
    ap.add_argument("--format", choices=FORMATS, default="auto",
                    help="classic (9-man) or tournament (MVP + 5 FLEX); "
                         "auto detects from the value sheet")
    args = ap.parse_args()

    lineups = generate(args.season, args.week, args.mode, args.n, args.min_diff,
                       args.lock, args.exclude, args.stack,
                       args.allow_dst_conflict, args.slate,
                       allow_risky=args.allow_risky,
                       max_exposure=args.max_exposure, fmt=args.format)
    if not lineups:
        raise SystemExit("no feasible lineup — loosen constraints")

    for i, lu in enumerate(lineups, 1):
        s = lineup_summary(lu)
        print(f"\nlineup {i} [{args.mode}]  salary ${s['salary']:,}  "
              f"proj {s['projection']}  range {s['p10_total']}-{s['p90_total']}")
        cols = ["position", "fd_name", "team", "opponent_team", "salary",
                "projection", "p10", "p90", "cv", "injury"]
        if "is_mvp" in lu.columns:
            cols.insert(0, "is_mvp")
        show = lineup_display_sort(lu).select(cols)
        with pl.Config(tbl_rows=9, tbl_width_chars=140, tbl_hide_dataframe_shape=True):
            print(show)

    stem = write_lineups(lineups, args.season, args.week, args.mode)
    print(f"\nwrote {stem}.csv and {stem}_upload.csv")


if __name__ == "__main__":
    main()
