"""Produce FanDuel projections for an upcoming week (PLAN.md §6).

Builds synthetic feature rows for the target week (trailing stats through
the last completed game, Vegas lines and weather for the upcoming one),
trains on all history, and writes versioned projection files:

  outputs/projections/{season}_w{week:02d}.csv / .parquet

Universe: weekly depth charts (QB/RB/WR/TE) for teams playing, plus
anyone with a stat line for those teams in the last 4 completed weeks,
plus both team defenses per game. Players with <2 career games get the
cold-start prior and confidence="low".

Usage: uv run python -m dfs_nfl.predict <season> <week> [--no-refresh]
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from dfs_nfl import coldstart, labels, quantiles
from dfs_nfl.baselines import (
    CURATED_DIR,
    MIN_PRIOR_GAMES,
    OUTPUTS_DIR,
    build_predictions,
    load_panel,
)
from dfs_nfl.features import (
    RAW_DIR,
    build_dst_features,
    build_offense_features,
)
from dfs_nfl.models import (
    BLEND_WEIGHT,
    LGB_PARAMS,
    NUM_ROUNDS,
    POSITION_PARAMS,
    POSITIONS,
    feature_columns,
    to_matrix,
)

PROJECTIONS_DIR = OUTPUTS_DIR / "projections"
OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE"]


def refresh_data(season: int) -> None:
    """Re-ingest the current season (idempotent) and rebuild labels.
    Sources that don't exist yet (e.g. week 1 of a new season) are skipped."""
    from dfs_nfl import ingest

    for name, loader in ingest.SOURCES.items():
        try:
            out_dir = RAW_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            df = loader(season)
            if df.height:
                df.write_parquet(out_dir / f"{name}_{season}.parquet")
                print(f"  refreshed {name}_{season}: {df.height:,} rows")
        except Exception as e:  # noqa: BLE001 - source not yet published
            print(f"  {name} {season}: unavailable ({type(e).__name__}), skipped")
    ingest.ingest_schedules()
    ingest.ingest_playerids()
    ingest.ingest_players()
    labels.main()


def week_games(season: int, week: int) -> pl.DataFrame:
    """(game_id, team, opponent_team) both directions for the target week."""
    sched = (
        pl.read_parquet(RAW_DIR / "schedules.parquet")
        .filter((pl.col("season") == season) & (pl.col("week") == week)
                & (pl.col("game_type") == "REG"))
        .select("game_id", "home_team", "away_team")
    )
    if not sched.height:
        raise ValueError(f"no games found for {season} week {week}")
    return pl.concat([
        sched.select("game_id", pl.col("home_team").alias("team"),
                     pl.col("away_team").alias("opponent_team")),
        sched.select("game_id", pl.col("away_team").alias("team"),
                     pl.col("home_team").alias("opponent_team")),
    ])


def _depth_chart_universe(season: int, week: int) -> pl.DataFrame:
    """QB/RB/WR/TE from the freshest depth chart at or before the target
    week; falls back to the prior season's final chart if needed."""
    for yr in (season, season - 1):
        path = RAW_DIR / "depth_charts" / f"depth_charts_{yr}.parquet"
        if not path.exists():
            continue
        dc = pl.read_parquet(path).filter(
            pl.col("position").is_in(OFFENSE_POSITIONS)
        )
        usable = dc.filter(pl.col("week") <= week) if yr == season else dc
        if not usable.height:
            usable = dc
        if usable.height:
            latest = usable["week"].max()
            return (
                usable.filter(pl.col("week") == latest)
                .group_by(["team", "gsis_id", "position"])
                .agg(pl.col("depth_rank").min())
            )
    raise FileNotFoundError("no depth charts available for universe")


def offense_skeleton(season: int, week: int) -> pl.DataFrame:
    games = week_games(season, week)
    players = pl.read_parquet(RAW_DIR / "players.parquet").select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("display_name").alias("player_display_name"),
    )
    dc = (
        _depth_chart_universe(season, week)
        .rename({"gsis_id": "player_id"})
        .join(games, on="team", how="inner")  # only teams playing this week
        .select("player_id", "team", "position")
    )
    # anyone who logged a stat line for a playing team in the last 4
    # completed weeks (covers depth-chart gaps/lags)
    recent = (
        pl.read_parquet(CURATED_DIR / "player_week.parquet")
        .filter(
            (pl.col("season") == season)
            & (pl.col("week") >= week - 4) & (pl.col("week") < week)
        )
        .sort("week")
        .group_by("player_id")
        .agg(pl.col("team").last(), pl.col("position_group").last().alias("position"))
        .join(games.select("team"), on="team", how="inner")
        .select("player_id", "team", "position")
    )
    universe = pl.concat([dc, recent]).unique(subset="player_id", keep="first")
    return (
        universe.join(games, on="team", how="inner")
        .join(players, on="player_id", how="left")
        .with_columns(
            pl.col("player_display_name").fill_null("(unknown)"),
            pl.col("position").alias("position_group"),
            pl.lit(season, dtype=pl.Int32).alias("season"),
            pl.lit(week, dtype=pl.Int32).alias("week"),
            pl.lit("REG").alias("season_type"),
        )
    )


def dst_skeleton(season: int, week: int) -> pl.DataFrame:
    return week_games(season, week).with_columns(
        pl.lit(season, dtype=pl.Int32).alias("season"),
        pl.lit(week, dtype=pl.Int32).alias("week"),
        pl.lit("REG").alias("season_type"),
        pl.col("team").alias("player_id"),
        pl.lit("DST").alias("position"),
    )


def _skeleton_panel(off: pl.DataFrame, dst: pl.DataFrame) -> pl.DataFrame:
    cols = ["player_id", "position", "season", "week", "game_id", "team",
            "opponent_team"]
    return pl.concat([
        off.select(cols),
        dst.with_columns(
            pl.col("team").alias("player_id"), pl.lit("DST").alias("position")
        ).select(cols),
    ])


def _apply_weather(feats: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Fill the target week's null temp/wind from Open-Meteo forecasts."""
    try:
        from dfs_nfl.weather import week_weather

        wx = week_weather(season, week).select(
            "game_id",
            pl.col("temp").alias("temp_fc"), pl.col("wind").alias("wind_fc"),
        )
    except Exception as e:  # noqa: BLE001 - offline is fine, features are nullable
        print(f"  weather forecast unavailable ({type(e).__name__}), continuing")
        return feats
    return (
        feats.join(wx, on="game_id", how="left")
        .with_columns(
            pl.coalesce("temp", "temp_fc").alias("temp"),
            pl.coalesce("wind", "wind_fc").alias("wind"),
        )
        .drop("temp_fc", "wind_fc")
    )


def _train_and_predict(df: pl.DataFrame, pos: str, season: int, week: int) -> pl.DataFrame:
    cols = feature_columns(df)
    params = LGB_PARAMS | POSITION_PARAMS.get(pos, {})
    # strictly-before filter keeps historical re-runs (scorecard testing)
    # leakage-free; for a genuinely future week it changes nothing
    train = df.filter(
        pl.col("fd_points").is_not_null()
        & ((pl.col("season") < season)
           | ((pl.col("season") == season) & (pl.col("week") < week)))
    )
    test = df.filter((pl.col("season") == season) & (pl.col("week") == week))
    x_train, y_train = to_matrix(train, cols), train["fd_points"].to_numpy()
    x_test = to_matrix(test, cols)

    mean_pred = lgb.train(
        params, lgb.Dataset(x_train, label=y_train), num_boost_round=NUM_ROUNDS
    ).predict(x_test)

    qparams = params | {"objective": "quantile"}
    qhat = np.sort(np.column_stack([
        lgb.train(
            qparams | {"alpha": quantiles.TRAIN_ALPHA_P10_QB
                       if (a == 0.1 and pos == "QB")
                       else quantiles.TRAIN_ALPHA[a]},
            lgb.Dataset(x_train, label=y_train),
            num_boost_round=quantiles.NUM_ROUNDS,
        ).predict(x_test)
        for a in quantiles.ALPHAS
    ]), axis=1)

    return test.with_columns(
        pl.Series("pred_model", mean_pred),
        pl.Series("p10", qhat[:, 0].round(2)),
        pl.Series("p50", qhat[:, 1].round(2)),
        pl.Series("p90", qhat[:, 2].round(2)),
    ).with_columns(
        pl.when(pl.col("pred_b2").is_not_null())
        .then(BLEND_WEIGHT * pl.col("pred_model")
              + (1 - BLEND_WEIGHT) * pl.col("pred_b2"))
        .otherwise(pl.col("pred_model"))
        .alias("projection")
    )


def _cold_start_override(preds: pl.DataFrame, feats_off: pl.DataFrame,
                         season: int, week: int) -> pl.DataFrame:
    """Replace projections for <2-game offensive players with the pooled
    cold-start prior."""
    pool = feats_off.with_columns(
        pl.col("position").replace_strict(coldstart.POSITION_CODES,
                                          default=None).alias("position_code"),
        pl.col("draft_round").fill_null(8),
    )
    train = pool.filter(
        pl.col("fd_points").is_not_null()
        & (pl.col("prior_games") < MIN_PRIOR_GAMES)
        & ((pl.col("season") < season)
           | ((pl.col("season") == season) & (pl.col("week") < week)))
    )
    test = pool.filter((pl.col("season") == season) & (pl.col("week") == week)
                       & (pl.col("prior_games") < MIN_PRIOR_GAMES))
    if not test.height or train.height < 500:
        return preds
    booster = lgb.train(
        coldstart.LGB_PARAMS,
        lgb.Dataset(to_matrix(train, coldstart.FEATURES),
                    label=train["fd_points"].to_numpy()),
        num_boost_round=coldstart.NUM_ROUNDS,
    )
    cold = test.select("player_id", "game_id").with_columns(
        pl.Series("pred_cold", booster.predict(to_matrix(test, coldstart.FEATURES)))
    )
    return (
        preds.join(cold, on=["player_id", "game_id"], how="left")
        .with_columns(
            pl.coalesce("pred_cold", "projection").alias("projection"),
        )
        .drop("pred_cold")
    )


def predict_week(season: int, week: int, refresh: bool = True) -> pl.DataFrame:
    if refresh:
        print("refreshing data...")
        refresh_data(season)
    print("building prediction frames...")
    skel_off = offense_skeleton(season, week)
    skel_dst = dst_skeleton(season, week)
    skel_panel = _skeleton_panel(skel_off, skel_dst)
    base_preds = build_predictions(
        pl.concat(
            [load_panel().join(skel_panel.select("season", "week").unique(),
                               on=["season", "week"], how="anti"), skel_panel],
            how="diagonal_relaxed",
        )
    ).select("player_id", "game_id", "pred_b1", "pred_b2")

    feats_off = _apply_weather(build_offense_features(skeleton=skel_off), season, week)
    feats_dst = _apply_weather(build_dst_features(skeleton=skel_dst), season, week)

    print("training and predicting...")
    out = []
    for pos in POSITIONS:
        feats = feats_dst if pos == "DST" else feats_off.filter(pl.col("position") == pos)
        df = feats.join(base_preds, on=["player_id", "game_id"], how="left")
        preds = _train_and_predict(df, pos, season, week)
        if pos != "DST":
            preds = _cold_start_override(
                preds, feats_off.join(base_preds, on=["player_id", "game_id"], how="left"),
                season, week)
        name_expr = (
            (pl.col("team") + " D/ST").alias("name") if pos == "DST"
            else pl.col("player_display_name").alias("name")
        )
        out.append(preds.select(
            "season", "week", "game_id", "team", "opponent_team", "position",
            "player_id", name_expr,
            pl.col("projection").round(2),
            "p10", "p50", "p90",
            pl.col("implied_total").round(2),
            "prior_games",
            pl.when(pl.col("prior_games") < MIN_PRIOR_GAMES)
            .then(pl.lit("low")).otherwise(pl.lit("ok")).alias("confidence"),
        ))
    projections = pl.concat(out).sort("projection", descending=True)
    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    stem = PROJECTIONS_DIR / f"{season}_w{week:02d}"
    projections.write_csv(f"{stem}.csv")
    projections.write_parquet(f"{stem}.parquet")
    print(f"\nwrote {stem}.csv ({projections.height} rows)")
    return projections


def main() -> None:
    season, week = int(sys.argv[1]), int(sys.argv[2])
    refresh = "--no-refresh" not in sys.argv
    projections = predict_week(season, week, refresh=refresh)
    with pl.Config(tbl_rows=25, tbl_width_chars=160):
        print(projections.select("name", "position", "team", "opponent_team",
                                 "projection", "p10", "p90", "confidence").head(25))


if __name__ == "__main__":
    main()
