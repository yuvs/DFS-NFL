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

Projections land in `outputs/projections/{season}_w{week:02d}.csv` (one row
per player/DST: projection, p10/p50/p90, implied total, confidence flag —
`low` = cold-start prior for <2-game players). Scorecards append to
`outputs/accuracy_log.csv`.

Cron (times ET — Tue scorecard + early projections, Sun final pre-slate run):

```cron
0 9 * * 2 cd $HOME/DFS-NFL && $HOME/.local/bin/uv run python -m dfs_nfl.weekly >> outputs/weekly.log 2>&1
0 10 * * 0 cd $HOME/DFS-NFL && $HOME/.local/bin/uv run python -m dfs_nfl.weekly >> outputs/weekly.log 2>&1
```

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
  production EWMAs, snap share, Vegas lines, opponent-allowed, venue/rest)
- `src/dfs_nfl/models.py` — per-position LightGBM (with stacked baselines
  as features) blended 60/40 with baseline B2
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
