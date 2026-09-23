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
