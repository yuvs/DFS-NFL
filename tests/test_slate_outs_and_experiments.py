"""Slate inactives feeding the injury features, and the experiment
trainer's target transforms."""

import numpy as np
import polars as pl

from dfs_nfl import experiments as ex
from dfs_nfl import features


def test_extra_outs_become_out_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(features, "RAW_DIR", tmp_path)  # no injury files
    outs = pl.DataFrame({"season": [2026], "week": [2], "team": ["LA"],
                         "gsis_id": ["00-0039075"], "position": ["WR"]})
    inj = features._read_injuries(outs)
    assert inj.height == 1 and inj["report_status"][0] == "Out"
    status = features.injury_status_table(outs)
    assert status.row(0, named=True)["inj_status"] == features.STATUS_ORD["Out"]
    assert features._read_injuries(None).height == 0


def test_sample_weights_decay_by_season():
    w = ex.sample_weights(np.array([2020, 2023, 2026]), decay=3.0)
    assert np.allclose(w, [0.25, 0.5, 1.0])
    assert ex.sample_weights(np.array([2020]), None) is None


def test_variant_params():
    assert ex.variant_params("RB", {"objective": "huber"})["alpha"] == ex.HUBER_DELTA
    assert ex.variant_params("RB", {"objective": "tweedie"})["tweedie_variance_power"] == ex.TWEEDIE_POWER
    assert ex.variant_params("DST", {})["num_leaves"] == 15


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    b2 = 10 + 2 * x[:, 0]
    y = b2 + 3 * x[:, 1] + rng.normal(scale=0.5, size=n) - 6  # negatives exist
    return pl.DataFrame({"f0": x[:, 0], "f1": x[:, 1], "f2": x[:, 2],
                         "fd_points": y, "pred_b2": b2,
                         "season": np.where(np.arange(n) < n // 2, 2020, 2021)})


def test_residual_trainer_predicts_points_not_residuals():
    df = _toy()
    tr = ex.Trainer("RB", {"residual": True})
    tr.params |= {"num_threads": 1}
    yhat = tr.fit_predict(df, df, ["f0", "f1", "f2"], 2021)
    assert abs(float(np.mean(yhat - df["fd_points"].to_numpy()))) < 0.5


def test_tweedie_trainer_shifts_negative_labels_back():
    df = _toy()
    assert df["fd_points"].min() < 0
    tr = ex.Trainer("RB", {"objective": "tweedie"})
    tr.params |= {"num_threads": 1}
    yhat = tr.fit_predict(df, df, ["f0", "f1", "f2"], 2021)
    assert abs(float(np.mean(yhat)) - float(df["fd_points"].mean())) < 0.5


def test_early_stop_rounds_cached_per_season():
    df = pl.concat([_toy(1200, s) for s in range(3)]).with_columns(
        pl.Series("season", np.repeat([2019, 2020, 2021], 1200)))
    tr = ex.Trainer("RB", {"early_stop": True})
    tr.params |= {"num_threads": 1}
    r1 = tr._rounds(df, ["f0", "f1", "f2"], 2022)
    r2 = tr._rounds(df, ["f0", "f1", "f2"], 2022)
    assert r1 == r2 and 50 <= r1 <= ex.EARLY_MAX_ROUNDS
    assert tr.rounds_cache == {2022: r1}


def test_production_spec_is_huber_and_blend_retuned():
    from dfs_nfl import models
    assert models.model_spec("RB") == {"objective": "huber"}
    assert models.Trainer("RB", models.model_spec("RB")).params["objective"] == "huber"
    assert models.BLEND_WEIGHT == 0.8


def _panel():
    """Two seasons x three weeks, one row per (week, player)."""
    rows = []
    for season in (2025, 2026):
        for week in (1, 2, 3):
            for pid, prior in (("a", 0), ("b", 5)):
                rows.append({"season": season, "week": week, "player_id": pid,
                             "prior_games": prior, "fd_points": 1.0})
    return pl.DataFrame(rows)


def test_walk_forward_weeks_is_ordered_and_window_scoped():
    from dfs_nfl.models import walk_forward_weeks

    df = _panel()
    assert walk_forward_weeks(df, [2026]) == [(2026, 1), (2026, 2), (2026, 3)]
    assert walk_forward_weeks(df, [2025, 2026])[0] == (2025, 1)
    assert walk_forward_weeks(df, [2024]) == []


def test_split_week_trains_only_on_strictly_earlier_rows():
    from dfs_nfl.models import split_week

    train, test = split_week(_panel(), 2026, 2, min_prior=0)
    # nothing from the target week or later leaks into training
    assert train.filter((pl.col("season") == 2026) & (pl.col("week") >= 2)).height == 0
    assert train.filter(pl.col("season") == 2025).height == 6
    assert set(test["week"].to_list()) == {2}
    # min_prior drops players without enough history from the test set
    _, strict = split_week(_panel(), 2026, 2, min_prior=2)
    assert set(strict["player_id"].to_list()) == {"b"}


def test_eval_and_holdout_windows_are_disjoint():
    from dfs_nfl.models import EVAL_SEASONS, HOLDOUT_SEASONS

    assert not set(EVAL_SEASONS) & set(HOLDOUT_SEASONS)
    assert ex.WINDOWS == {"eval": EVAL_SEASONS, "holdout": HOLDOUT_SEASONS}


def test_train_alpha_routes_p10_by_position_and_leaves_p90_alone():
    from dfs_nfl import quantiles as q

    assert q.train_alpha("QB", 0.1) == q.TRAIN_ALPHA_P10_QB
    assert q.train_alpha("WR", 0.1) == q.TRAIN_ALPHA[0.1]
    assert q.train_alpha("QB", 0.9) == q.TRAIN_ALPHA[0.9]
    # the sweep must bracket whatever is shipped, or it can't re-find it
    assert min(q.P10_CANDIDATES) <= q.TRAIN_ALPHA_P10_QB <= max(q.P10_CANDIDATES)
    assert min(q.P10_CANDIDATES) <= q.TRAIN_ALPHA[0.1] <= max(q.P10_CANDIDATES)


def test_recommend_picks_smallest_worst_case_gap_not_the_average():
    from dfs_nfl.quantiles import recommend

    # 0.04 is near-perfect on eval but far off on holdout; 0.055 is
    # mediocre on both. Minimax must prefer 0.055 -- an n- or
    # mean-weighted rule would keep 0.04.
    r = pl.DataFrame({
        "window": ["eval", "eval", "holdout", "holdout"],
        "position": ["QB"] * 4,
        "train_alpha": [0.04, 0.055, 0.04, 0.055],
        "cov_p10": [0.100, 0.130, 0.050, 0.091],
    })
    pick = recommend(r)
    assert pick["pick"].to_list() == [0.055]
    assert pick["worst_gap"].to_list() == [0.030]
