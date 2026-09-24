# DFS-NFL

Weekly FanDuel fantasy-point projections for NFL offensive players and team
defenses. See `PLAN.md` for the full project plan.

## Quickstart

```sh
uv sync                              # install deps
uv run pytest                        # scoring-function tests
uv run python -m dfs_nfl.ingest      # download nflverse data (2016–2025)
uv run python -m dfs_nfl.labels      # build curated label tables
uv run python -m dfs_nfl.baselines   # walk-forward baseline backtest
uv run python -m dfs_nfl.features    # build feature store
uv run python -m dfs_nfl.models      # walk-forward LightGBM backtest
uv run python -m dfs_nfl.quantiles   # p10/p50/p90 floor-ceiling backtest
uv run python -m dfs_nfl.coldstart   # <2-game player prior backtest
uv run python -m dfs_nfl.odds 2026 1     # current-week Vegas lines
uv run python -m dfs_nfl.weather 2026 1  # kickoff weather forecasts
```

macOS note: LightGBM needs OpenMP (`brew install libomp`).

## Production (weekly)

```sh
uv run python -m dfs_nfl.weekly              # refresh -> scorecard -> project next week
uv run python -m dfs_nfl.predict 2026 3      # project a specific week
uv run python -m dfs_nfl.scorecard 2026 2    # score a completed week's projections
```

Projections land in `outputs/projections/{season}_w{week:02d}.csv`, one row
per player/DST. Columns:

- `projection` — expected FanDuel points; `p10`/`p50`/`p90` — calibrated
  floor / median / ceiling
- `sd` — spread of outcomes in points, derived from the quantile band;
  validated: realized error rises monotonically across sd quartiles
- `cv` — volatility relative to projection. **The roster-construction
  dial:** among players you like, low `cv` = stable (cash games /
  must-not-bust slots), high `cv` = volatile (GPP leverage). Comparable
  across salary tiers, unlike raw `sd`.
- `skew` — tail asymmetry in [-1, 1]; positive = more room above the
  median than below (boom/bust profile — the upside, when it hits, is
  disproportionately large)
- `team_implied_total` / `opp_implied_total` — team-level Vegas context,
  shared by every player on the team
- `confidence` — `low` = cold-start prior for <2-game players

Optimizing for upside: sort by `p90` (or filter high `cv` + positive
`skew`). Optimizing for stability: maximize `p10`, prefer low `cv`.
Scorecards append to `outputs/accuracy_log.csv`: per-position MAE,
RMSE, within-week Spearman, p10/p90 coverage, and `top5_hit` — the share
of the projected top 5 that finished in the actual top 10, the DFS
question the error metrics don't ask. The backtest reports the same
(`*_top5` columns in `outputs/model_backtest.csv`).

### Lineup scorecard

The same Tuesday job grades every lineup built for the week
(`outputs/lineups/`) on actual points and appends to
`outputs/lineup_log.csv`, with two references built from the same value
sheet under the same roster rules:

- `hindsight_max` — the lineup that maximizes *actual* points: the
  ceiling for that slate. `captured` = actual / hindsight_max.
- `fppg_lineup` — actual points of the lineup optimized on FanDuel's own
  FPPG column (their trailing average, shown in the lobby): the public
  baseline the model has to beat.

`players_out` counts rostered players with no stat line (inactive).
FanDuel doesn't publish salary history, so this is a live backtest that
accumulates from the saved slates; the classic cash lineup beat the
FPPG lineup by 54 and 39 points in weeks 1 and 2 of 2026, while the
week 2 Monday-night builds were the one clear miss (Dart left injured).

### What the models see

Per-position LightGBM over the feature store (`dfs_nfl.features`), every
input knowable before kickoff:

- **Trailing production** — EWMA (half-life 4 games) and last-4 means of
  FanDuel points and every box-score stat, plus snap share, target share,
  air-yards share and WOPR; volatility (8-game std) and recent max.
- **Opportunity** (`dfs_nfl.pbp_features`, from play-by-play) — red-zone
  and inside-10 targets, goal-line (≤5) and red-zone carries, the
  player's share of his team's red-zone targets / goal-line carries, and
  **expected touchdowns**: every target, carry and pass attempt credited
  the league TD rate for its yardline bucket (rates from prior seasons
  only). Trailing expected TDs are a steadier read on scoring chances
  than trailing actual TDs.
- **Team context** (play-by-play) — own offense's trailing plays per
  game, dropback rate, pass rate over expected in neutral script, and
  EPA per play; the opponent defense's trailing EPA allowed per dropback
  and per rush, and success rate allowed. For DST the roles flip (own
  defense, opponent offense). These rank in the top 15 features at every
  position.
- **Game context** — Vegas implied totals and edge, home/away, rest,
  divisional, dome, forecast temperature and wind.
- **Availability** — depth-chart rank, injury report status and practice
  participation, opportunity vacated by teammates ruled out, draft
  capital and rookie flag.
- **Stacked baselines** — the B1/B2 naive projections as inputs.

**Loss and blend.** The mean model trains with a huber objective
(delta 4 points): fantasy points are right-skewed and L2 spent the trees
chasing boom weeks. The shipped projection is 0.8 × model + 0.2 × B2.
Both were chosen on the 2024–25 walk-forward backtest
(`uv run python -m dfs_nfl.experiments [variant ...]` runs the grid —
residual-over-baseline target, recency-weighted rows, huber / tweedie
loss, early-stopped rounds — and writes `outputs/experiments.csv`;
`models.MODEL_SPEC` picks the production variant). Huber won at every
position; residual targets, recency weights and early stopping were
neutral or worse. Shipped-projection MAE / within-week Spearman, before
(L2, box-score features only) and after (huber, play-by-play features,
0.8 blend):

| | QB | RB | WR | TE | DST |
|---|---|---|---|---|---|
| before | 6.50 / 0.515 | 4.21 / 0.728 | 4.06 / 0.645 | 3.12 / 0.596 | 4.14 / 0.293 |
| after | 6.28 / 0.525 | 4.08 / 0.730 | 3.91 / 0.651 | 3.00 / 0.606 | 4.10 / 0.300 |

Quantile calibration is unchanged (p10 / p90 coverage within a point of
nominal at every position).

**Vegas anchor.** Vegas implied team totals are well calibrated
(2016–25: actual points ≈ 1.02 × implied, bias +0.2), and where the
model's team total disagrees with Vegas, actual scoring barely follows
the model. So after the blend, each week's offense projections are
summed per team and regressed on implied totals across that week's
teams. Every player on a team is then scaled to close 75%
(`models.VEGAS_ANCHOR`) of the team's gap to that line, with the factor
clipped to 0.75–1.25. Doubtful/Out players don't count toward team sums;
p10/p50/p90 move with the projection; DST is untouched. The factor is
written to the `anchor_factor` column. The line is fit on the same
week's projections, so it uses no outcomes. A full-week run saves it to
`outputs/projections/{season}_w{week:02d}_anchor.json`, and single-game
re-projections (too few teams to fit) reuse it or skip the anchor if
it's missing. 0.75 was picked on 2024 and held on 2025; walk-forward
2024–25, MAE / Spearman / top-5 hit (`blend_*` vs `final_*` in
`outputs/model_backtest.csv`):

| | QB | RB | WR | TE |
|---|---|---|---|---|
| blend | 6.28 / 0.525 / 0.506 | 4.08 / 0.730 / 0.483 | 3.91 / 0.651 / 0.328 | 3.00 / 0.606 / 0.478 |
| anchored | 6.20 / 0.540 / 0.506 | 4.06 / 0.736 / 0.500 | 3.90 / 0.656 / 0.350 | 2.99 / 0.609 / 0.483 |

The same study found no betting edge in the lines themselves: overs hit
49%, no spread segment reached the 52.4% needed at −110, and the
apparent primetime/high-wind unders flip season to season.

## Salary value sheets

Drop FanDuel "players list" CSV exports (lobby → download players list)
into `data/raw/fanduel/`, then:

```sh
uv run python -m dfs_nfl.value 2026 1   # match latest slate to projections
```

Writes `outputs/value/{season}_w{week:02d}_slate{id}.csv`: every matched
slate player with salary, `value` (projected points per $1K) and
`value_p90` (ceiling points per $1K — the GPP sort), plus the injury
indicator from the slate file. Filter out `O`/`IR`/`D`/`NA` (not active) before building
lineups; `Q` players carry extra risk the model can't see.

## Lineup optimizer

```sh
uv run python -m dfs_nfl.optimizer 2026 1 --mode cash
uv run python -m dfs_nfl.optimizer 2026 1 --mode gpp --n 5 --stack
uv run python -m dfs_nfl.optimizer 2026 1 --lock "Josh Allen" --exclude "Chase Brown"
uv run python -m dfs_nfl.optimizer 2026 2 --mode gpp --n 9 --format tournament
```

CP-SAT integer program over the value sheet. FanDuel classic rules are
built in: 9 slots (QB/2RB/3WR/TE/FLEX/DST), $60K cap, max 4 per team,
3+ teams. `--mode cash` maximizes 0.5·projection + 0.5·p10 and, by
default, excludes Q-tagged and cold-start players (`--allow-risky` to
override); `--mode gpp` maximizes 0.5·projection + 0.5·p90 (add
`--stack` to force a QB + pass-catcher stack). O/IR/D/NA players are always
excluded, and lineups never roster offense against their own DST
(`--allow-dst-conflict` to permit). Backup QBs are dropped too: per team
only the top depth-chart QB (the value sheet's `depth_rank` = 1, or the
highest-salaried QB when the chart is missing) is eligible, because the
model projects a benched QB off his earlier starts and he shows up as a
fake value play. `--allow-backup-qb` keeps them; a `--lock` on a backup
always wins. `--n N --min-diff K` generates N
lineups each differing by ≥K players. `--max-exposure 0.5` keeps any one
player out of more than half the lineups (locked players exempt). Lineup totals report a range of
proj ± 1.28·√(Σsd²) — per-player p10/p90 don't sum.

**Tournament (single-game) lineups.** FanDuel's single-game contests use
6 players under the same $60K cap: one MVP slot scoring 1.5x points at
1.5x salary, plus 5 FLEX slots open to any position (both teams must
be represented). Download that contest's own players list — its MVP
salaries and slate IDs differ from the full slate's — and the optimizer
detects the format from the export's MVP roster column. `--format
tournament` (or `classic`) makes the expectation explicit and fails
loudly if the newest value sheet is the other kind. Stacking and the
DST-conflict rule don't apply. The exposure cap bites hard here: with
two teams in the pool, a 50% cap runs out of starting QBs after a few
lineups, so leave it at 1.0 or use small counts.

Each run writes two files per mode to `outputs/lineups/`:
`{season}_w{week:02d}_{mode}.csv` (readable, one row per player) and
`..._upload.csv` — FanDuel's bulk-entry format (slot header
`QB,RB,RB,WR,WR,WR,TE,FLEX,DEF`, slate player IDs as cells). Tournament
builds get a `_tournament_{TEAM-TEAM}` suffix (e.g.
`2026_w02_gpp_tournament_LA-NYG.csv`) and the
`MVP - 1.5X Points,AnyFLEX,...` header, so a Thursday and a Monday
build in the same week coexist. On the contest entry screen choose
"Upload lineups from CSV" and hand it the upload file. IDs are
slate-specific, so regenerate after downloading a new players list.

`weekly` also auto-builds a value sheet and lineups for every FanDuel
slate CSV of the target week in `data/raw/fanduel/` (matched by the date
in the export's filename): full-slate exports get the classic cash/GPP
builds, single-game exports get the tournament builds. The only manual
step each week is downloading those CSVs from the FanDuel lobby.

Single-game slates are **re-projected at slate time**: the injury report
and depth chart are refreshed, the slate's own O/IR/D/NA tags are
applied as "Out" (FanDuel's game-day inactive tag is the only pre-kickoff
record of a Monday-night scratch; nflverse's report stops at Friday), and
just the two teams are projected into
`outputs/projections/{season}_w{week:02d}_slate{id}.csv`. The week's
main projection file is untouched, so the scorecard keeps grading the
Sunday-morning call. With Nacua tagged inactive for the week 2 Monday
game, Davante Adams moved from 9.6 to 13.0 projected.

How many lineups the automated run builds is set in `config.toml`
(`[lineups]`: `cash`, `gpp`, `gpp_stack`, `min_diff`, `max_exposure` for
classic slates; `tournament_cash`, `tournament_gpp`,
`tournament_max_exposure` for single-game slates; set a count to 0 to
skip that build). Missing file or keys fall back to defaults, and ad-hoc
`dfs_nfl.optimizer` runs use CLI flags instead.

Automation runs via launchd (survives sleep, unlike cron): agents
`com.dfsnfl.weekly.tue` (Tue 9:00) and `com.dfsnfl.weekly.sun`
(Sun 10:00, local time) run `dfs_nfl.weekly`, logging to
`outputs/weekly.log`. Reference plists live in `deploy/launchd/`; to
(re)install on a new machine:

```sh
cp deploy/launchd/*.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.dfsnfl.weekly.tue.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.dfsnfl.weekly.sun.plist
```

(`launchctl bootout gui/$(id -u)/com.dfsnfl.weekly.tue` etc. to remove;
`launchctl kickstart gui/$(id -u)/com.dfsnfl.weekly.sun` to run now.)

Optional: `export ODDS_API_KEY=...` (the-odds-api.com) for live line
overrides via `dfs_nfl.odds`; without it, nflverse schedule lines are used.

## Layout

- `src/dfs_nfl/scoring.py` — FanDuel scoring (2024+ rules incl. yardage
  bonuses), config-driven; validated against published FanDuel scores
- `src/dfs_nfl/ingest.py` — nflverse → `data/raw/` parquet, idempotent by season
- `src/dfs_nfl/labels.py` — `data/curated/player_week.parquet` and
  `dst_week.parquet` with `fd_points` labels (DST points-allowed reconstructed
  from play-by-play, offensive-unit points only)
- `src/dfs_nfl/baselines.py` — the two naive baselines any model must beat
  (trailing 4-game mean; EWMA × opponent-allowed × Vegas implied-total)
- `src/dfs_nfl/features.py` — pre-kickoff feature store (trailing usage/
  production EWMAs, snap share, Vegas lines, opponent-allowed, venue/rest,
  weather, official injury report status/practice participation, and
  vacated opportunity — trailing FD points of teammates ruled Out/Doubtful)
- `src/dfs_nfl/models.py` — per-position LightGBM (with stacked baselines
  as features) blended 80/20 with baseline B2, then anchored to Vegas
  team totals
- `src/dfs_nfl/quantiles.py` — p10/p50/p90 quantile models (calibrated
  within ~0.02 of target coverage on 2024–25)
- `src/dfs_nfl/coldstart.py` — prior model for players with <2 career
  games (depth chart, draft capital, Vegas); beats naive position mean
  by ~0.8–0.9 MAE
- `src/dfs_nfl/odds.py` / `weather.py` — live-week Vegas lines (nflverse,
  The Odds API if `ODDS_API_KEY` set) and Open-Meteo kickoff forecasts
- `src/dfs_nfl/predict.py` — production projections for an upcoming week
  (synthetic feature rows from depth-chart universe + trailing history)
- `src/dfs_nfl/scorecard.py` / `weekly.py` — accuracy tracking and the
  single-command weekly orchestration

## Backtest (2024–2025 REG, walk-forward, ≥2 prior games)

Final = 0.6·LightGBM + 0.4·B2. Beats both baselines at every position:

| Pos | n | Final MAE | B1 MAE | B2 MAE | Final Spearman | B2 Spearman |
|-----|------:|------:|------:|------:|------:|------:|
| QB  | 1,280 | **6.50** | 7.05 | 6.97 | **0.51** | 0.46 |
| RB  | 3,127 | **4.25** | 4.45 | 4.28 | **0.72** | 0.71 |
| WR  | 4,756 | **4.06** | 4.27 | 4.16 | **0.64** | 0.63 |
| TE  | 2,419 | **3.12** | 3.30 | 3.16 | **0.60** | 0.58 |
| DST | 1,088 | **4.14** | 4.69 | 4.28 | **0.29** | 0.26 |

Quantile coverage (target 0.10 / 0.50 / 0.90): p10 0.08–0.10,
p50 0.51–0.54, p90 0.89–0.90 across all positions.
