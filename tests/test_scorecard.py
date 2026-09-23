"""Top-N hit rate and the lineup scorecard on synthetic data."""

import polars as pl

from dfs_nfl import scorecard
from dfs_nfl.models import top_hit_rate
from test_optimizer import synthetic_pool, tournament_pool


def test_top_hit_rate_counts_favourites_that_landed():
    pred = list(range(20, 0, -1))            # 20 players, ranked 1..20
    assert top_hit_rate(pred, list(pred)) == 1.0          # perfect ranking
    assert top_hit_rate(pred, list(reversed(pred))) == 0.0  # top-5 finished bottom-5
    actual = list(pred)
    actual[3], actual[4] = -50, -60          # two favourites busted
    assert top_hit_rate(pred, actual) == 0.6  # 3 of the projected top-5 in the actual top-10
    assert top_hit_rate([1, 2, 3], [1, 2, 3]) is None   # too few to rank


def _sheet_with_ids(pool: pl.DataFrame) -> pl.DataFrame:
    return pool.with_columns(
        pl.format("77-{}", pl.int_range(pl.len())).alias("fanduel_id"),
        pl.format("p{}", pl.int_range(pl.len())).alias("player_id"),
        pl.lit("ok").alias("confidence"),
        (pl.col("projection") * 0.9).alias("fppg"),
    )


def test_score_lineups_grades_against_ceiling_and_fppg(monkeypatch, tmp_path):
    from dfs_nfl import optimizer, value

    lineups_dir, value_dir = tmp_path / "lineups", tmp_path / "value"
    lineups_dir.mkdir(); value_dir.mkdir()
    monkeypatch.setattr(optimizer, "LINEUPS_DIR", lineups_dir)
    monkeypatch.setattr(value, "VALUE_DIR", value_dir)
    monkeypatch.setattr(scorecard, "LINEUP_LOG_PATH", tmp_path / "lineup_log.csv")

    sheet = _sheet_with_ids(synthetic_pool())
    sheet.write_csv(value_dir / "2026_w01_slate77.csv")
    # actual = projection for everyone except one player who sat (null)
    act = sheet.select("player_id", pl.col("projection").alias("actual"))
    act = act.with_columns(
        pl.when(pl.col("player_id") == "p0").then(None).otherwise(pl.col("actual"))
        .alias("actual"))
    monkeypatch.setattr(scorecard, "actuals", lambda s, w: act)

    lu = optimizer.solve_lineup(sheet, mode="mean")
    optimizer.write_lineups([lu], 2026, 1, "cash")
    out = scorecard.score_lineups(2026, 1)

    row = out.row(0, named=True)
    assert row["file"] == "2026_w01_cash" and row["format"] == "classic"
    assert row["mode"] == "cash" and row["lineup"] == 1
    # with actual == projection the built lineup IS the hindsight optimum
    # unless it rostered the scratched player
    assert row["hindsight_max"] >= row["actual"]
    assert 0 < row["captured"] <= 1.0
    assert row["fppg_lineup"] is not None
    assert row["players_out"] == int("p0" in lu["player_id"].to_list())


def test_lineup_actual_applies_mvp_multiplier():
    lu = tournament_pool().head(6).with_columns(
        pl.Series("is_mvp", [True] + [False] * 5),
        pl.Series("actual", [10.0, 2.0, 2.0, 2.0, 2.0, None]),
    )
    assert scorecard._lineup_actual(lu) == 15.0 + 8.0
