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
```

## Layout

- `src/dfs_nfl/scoring.py` — FanDuel scoring (2024+ rules incl. yardage
  bonuses), config-driven; validated against published FanDuel scores
- `src/dfs_nfl/ingest.py` — nflverse → `data/raw/` parquet, idempotent by season
- `src/dfs_nfl/labels.py` — `data/curated/player_week.parquet` and
  `dst_week.parquet` with `fd_points` labels (DST points-allowed reconstructed
  from play-by-play, offensive-unit points only)
- `src/dfs_nfl/baselines.py` — the two naive baselines any model must beat
  (trailing 4-game mean; EWMA × opponent-allowed × Vegas implied-total)

## Baseline results (2021–2025 REG, walk-forward)

| Pos | n | B1 MAE | B2 MAE | B2 Spearman |
|-----|------:|------:|------:|------:|
| QB  | 3,158 | 6.87 | 6.75 | 0.49 |
| RB  | 7,837 | 4.53 | 4.36 | 0.68 |
| WR  | 11,812 | 4.41 | 4.28 | 0.62 |
| TE  | 5,867 | 3.34 | 3.19 | 0.54 |
| DST | 2,718 | 4.68 | 4.32 | 0.22 |
