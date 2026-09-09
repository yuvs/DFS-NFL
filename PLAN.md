# NFL FanDuel Points Prediction Service — Project Plan

**Goal:** Every week, produce predicted FanDuel fantasy points for every NFL offensive player (QB, RB, WR, TE) and all 32 team defenses (D/ST), using models trained on historical data and features known before kickoff.

**Timing context:** The 2026 season kicks off this week. Train on ~2016–2024, validate on 2025, then go live during the 2026 season and score predictions against actuals each week.

---

## 1. The prediction target: FanDuel scoring

The first deliverable is an exact FanDuel scoring function — it defines the label for every model. FanDuel updated scoring in 2024 to add yardage bonuses, so any pre-2024 reference is stale.

### Offensive players (half-PPR)

| Event | Points |
|---|---|
| Passing yard | 0.04 |
| Passing TD | 4 |
| Interception thrown | -1 |
| 300+ passing yards bonus | 3 |
| Rushing yard | 0.1 |
| Rushing TD | 6 |
| 100+ rushing yards bonus | 3 |
| Reception | 0.5 |
| Receiving yard | 0.1 |
| Receiving TD | 6 |
| 100+ receiving yards bonus | 3 |
| Fumble lost | -2 |
| 2-point conversion (pass/rush/catch) | 2 |
| Kick/punt return TD, fumble recovery TD | 6 |

### Team defense / special teams

| Event | Points |
|---|---|
| Sack | 1 |
| Interception | 2 |
| Fumble recovery | 2 |
| Safety | 2 |
| Blocked punt/FG/XP | 2 |
| Defensive or return TD | 6 |
| Points allowed: 0 | 10 |
| Points allowed: 1–6 | 7 |
| Points allowed: 7–13 | 4 |
| Points allowed: 14–20 | 1 |
| Points allowed: 21–27 | 0 |
| Points allowed: 28–34 | -1 |
| Points allowed: 35+ | -4 |

Nuance: "points allowed" counts only points scored **against** the defense's unit (opponent offensive TDs, FGs, XPs, 2-pt conversions) — it excludes points the opponent scores on pick-sixes, punt returns, etc. against your own offense/ST. The scoring function must reconstruct this from play-by-play, not from the final scoreboard.

**Validation step:** Before modeling, verify the scoring function against actual FanDuel results for a few past slates (FanDuel shows player scores in completed contests) and against the current rules at fanduel.com/rules.

---

## 2. Phase 1 — Data gathering & pipeline

### Primary source: nflverse via `nflreadpy`

Free, community-maintained, updated nightly during the season. (`nfl_data_py` is deprecated; `nflreadpy` is its successor and returns Polars DataFrames.) Key loaders:

| Loader | Contents | Used for |
|---|---|---|
| `load_player_stats()` | Weekly box-score stats per player, 1999+ | Labels + trailing-performance features |
| `load_pbp()` | Full play-by-play with EPA, air yards, etc. | Red-zone/goal-line usage, D/ST points-allowed reconstruction, efficiency metrics |
| `load_schedules()` | Schedule + results, Vegas spread/total/moneyline, roof, surface | Game context, historical betting lines |
| `load_rosters_weekly()` / `load_depth_charts()` | Who's on the team and where on the depth chart, by week | Active-player universe, role changes |
| `load_injuries()` | Official injury reports & practice status | Availability features |
| `load_snap_counts()` | Offensive snap counts/percentages | Usage features |
| `load_participation()` / `load_ngs()` | Routes run, pass-rush data, Next Gen Stats | Route participation, advanced usage |
| `load_ff_opportunity()` | Expected fantasy points / expected yards models from ffverse | Strong ready-made opportunity features |
| `load_ff_playerids()` | Cross-platform ID map (includes FanDuel IDs) | Joining to FanDuel salaries/slates |

### Supplementary sources

- **Current-week betting odds** — historical lines come with `load_schedules()`, but for live prediction you need this week's spread/total *before* kickoff. Use The Odds API (free tier is enough for weekly pulls). Implied team total (`total/2 ∓ spread/2`) is one of the strongest single predictors of both offensive production and D/ST scoring.
- **Weather** — schedules give roof/surface; for outdoor stadiums pull wind/precip/temp forecasts from Open-Meteo (free, no key) keyed by stadium coordinates. Wind ≥ ~15 mph measurably suppresses passing.
- **FanDuel salaries & slates** (optional, later) — export CSV from the FanDuel lobby weekly. Not needed for point prediction, but needed if this evolves toward lineup optimization/value analysis.

### Storage & pipeline design

- **Storage:** Parquet files in a `data/` lake, queried with DuckDB. No database server needed; the full nflverse history is a few GB.
- **Ingestion:** idempotent Python scripts keyed by `(season, week)` — safe to re-run, always overwrite the current week. Layers: `data/raw/` (as-downloaded) → `data/curated/` (cleaned, joined, one row per player-week and team-week).
- **ID discipline:** standardize on nflverse `gsis_id` for players and team abbreviations everywhere; map to FanDuel IDs only at the output edge. Name-based joins are the #1 source of silent bugs in this domain.

**Phase 1 exit criteria:** one command rebuilds a player-week table (2016–present) with FanDuel points computed as the label, and the computed labels match spot-checked real FanDuel scores.

---

## 3. Phase 2 — Feature engineering

All features must be knowable **before kickoff** (see leakage rules in §5). Build a feature store as one wide table per position group.

**Opportunity / usage (the biggest signal for RB/WR/TE):**
- Snap share, route participation rate, target share, air-yards share (and WOPR = 1.5·target share + 0.7·air-yards share), carry share, red-zone and goal-line touch shares
- Each as exponentially weighted moving averages over trailing ~4 and ~8 games, plus season-to-date; decay tuned in backtesting

**Efficiency (regress heavily — it's noisy):**
- Yards per route run, YAC over expected, EPA/play, QB CPOE, TD rate vs. expected TD rate (from `load_ff_opportunity`)

**Game context:**
- Vegas implied team total, spread (favorites run more, underdogs pass more), game total, home/away, rest days, divisional game, roof/surface, weather forecast

**Opponent:**
- FanDuel points allowed by position (schedule-adjusted), pass/rush EPA allowed, pace (plays per game, seconds per snap), sack rate, blitz rate

**Availability & role:**
- Injury report status (Out/Doubtful/Questionable) and practice participation trend, depth chart rank, games missed recently
- **Vacated opportunity:** targets/carries freed by teammates ruled out — a major driver of week-to-week outliers

**D/ST features:**
- Opponent implied total (dominant predictor), opponent turnover and sack rates allowed, own pressure/takeaway rates, opponent QB status (backup starting = D/ST gold)

---

## 4. Phase 3 — Modeling

**Baselines first (the bar to beat):**
1. Trailing 4-game average FanDuel points
2. Season-to-date EWMA adjusted by opponent points-allowed and implied team total

If a model can't beat #2 out of sample, it isn't earning its complexity.

**Main models — gradient boosting (LightGBM), one per position group** (QB, RB, WR, TE, DST), on the player-week panel:
- **Mean model** for expected points (primary output)
- **Quantile models** (p10 / p50 / p90) for floor/ceiling — cheap to add and what actually matters for DFS decisions
- Position-specific models beat one pooled model because feature importance differs sharply (e.g., WOPR means nothing for QBs; implied total dominates for DST)

**Comparison track — component models:** predict volume (attempts, targets, carries) and efficiency (yards/attempt, TD rates) separately, then compose FanDuel points through the scoring function. More interpretable, robust to future scoring changes, and better at yardage-bonus probabilities; keep whichever backtests better, or blend.

**Cold-start handling:** rookies and role-changers lack trailing stats. Fall back on hierarchical priors: depth-chart-rank × position averages, draft capital, and preseason/depth-chart signals. Flag low-history predictions with a confidence indicator in the output.

---

## 5. Phase 4 — Backtesting & evaluation

**Protocol:** strict walk-forward. For each week W of each held-out season, train only on data through week W−1, predict week W. Never random cross-validation splits — they leak future information in time-series panels.

**Leakage rules (make these hard constraints in code):**
- Feature snapshot frozen at a simulated "Sunday 10am ET" — injury statuses, depth charts, and betting lines as they stood then, not as revised later
- No same-week stats in any feature; opponent-strength metrics computed only from prior weeks

**Metrics:**
- MAE and RMSE per position (expect MAE around ~5–6 for QB, ~4–5 for RB/WR on played games; treat these as the yardstick, not the goal)
- **Spearman rank correlation within position-week** — ranking players correctly matters more than absolute error for DFS
- Top-N hit rate (how often the model's top-5 at a position land in the actual top-10)
- Quantile calibration (does the actual fall below p90 ~90% of the time?)
- Everything compared against the two baselines; optionally scrape a public projection source for one season as an external benchmark

---

## 6. Phase 5 — Weekly production pipeline

In-season cadence, automated (cron locally, or GitHub Actions):

| When | Job |
|---|---|
| Tue AM | Ingest final stats for the completed week; compute actual FanDuel points; log model error vs. last week's predictions; refresh/retrain models |
| Wed–Sat daily | Pull rosters, depth charts, injury reports, current odds; regenerate provisional projections |
| Sun ~10am ET | Final feature build with latest injuries/lines/weather → final projections for every active offensive player + 32 DSTs |
| Sun (late slates) | Optional re-run for afternoon/night games with updated inactives |

**Outputs per run:** a versioned CSV/Parquet (`outputs/2026_w05_projections.csv`) with player, team, position, opponent, FanDuel ID, projected points, p10/p90, and confidence flag — plus a simple HTML report or dashboard. Keep every historical output file: that archive *is* the monitoring system (predicted vs. actual over time, error drift by position).

---

## 7. Tech stack & repo layout

Python 3.11+ · `nflreadpy` · Polars (+ pandas at edges) · DuckDB · LightGBM · scikit-learn · `uv` for env management · pytest for the scoring function and leakage guards.

```
DFS-NFL/
├── data/            # raw/ and curated/ parquet (gitignored)
├── src/
│   ├── scoring.py   # FanDuel scoring functions (offense + DST) — heavily unit-tested
│   ├── ingest/      # one module per source: nflverse, odds, weather, salaries
│   ├── features/    # feature store builders per position group
│   ├── models/      # train, predict, quantiles, cold-start priors
│   ├── backtest/    # walk-forward harness + metrics
│   └── pipeline/    # weekly orchestration entry points
├── outputs/         # versioned weekly projections + accuracy logs
├── notebooks/       # exploration only, nothing production depends on them
└── PLAN.md
```

---

## 8. Milestones

1. **Week 1 — Foundation:** repo scaffolding; nflverse ingestion; FanDuel scoring function validated against real FanDuel scores; player-week label table 2016–2025; both baselines backtested. *(Deliverable: "here's how hard this problem is" numbers.)*
2. **Week 2 — Models:** feature store; LightGBM mean models per position; walk-forward evaluation vs. baselines on 2024–2025.
3. **Week 3 — Completeness:** DST model; quantile models; cold-start priors; odds + weather ingestion for live weeks.
4. **Week 4 — Production:** weekly automated pipeline live for the 2026 season; output artifacts + accuracy tracking; first real-week scorecard the following Tuesday.
5. **Ongoing:** in-season error review; add features (vacated-opportunity refinement, NGS metrics); optional extensions — salary ingestion, value (points/$) rankings, lineup optimizer.

---

## 9. Risks & known gotchas

- **Irreducible noise:** week-level fantasy scoring is highly variant; even the best public projections show WR MAE near 5. Success = consistently beating naive baselines and ranking well, not pinpoint scores.
- **Leakage** is the most common way these projects produce fake-great backtests — enforce the snapshot rule in code, not by convention.
- **ID joins:** player names collide and change (Jr., team switches, "St. Brown" ×3). Always join on IDs.
- **Scoring drift:** FanDuel changed scoring in 2024 and can again — keep the scoring function config-driven and re-verify each September.
- **Data availability timing:** some nflverse assets (participation/routes) lag by a day or more or have coverage gaps in some seasons; features must degrade gracefully when a source is missing.
- **ToS note:** FanDuel salary/contest data is for personal-use export; avoid aggressive scraping of FanDuel itself.
