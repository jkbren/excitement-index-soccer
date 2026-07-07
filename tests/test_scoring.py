"""Structural contracts of the scorer, on synthetic feature matrices —
no event data or network required."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from excitement_index import load_config, score_matches


def synthetic_features(n=40, seed=7):
    """A plausible random feature matrix covering every taxonomy measure."""
    cfg = load_config()
    rng = np.random.default_rng(seed)
    measures = [m for feats in cfg["taxonomy"].values() for m in feats]
    fm = pd.DataFrame(rng.normal(1.0, 0.5, size=(n, len(measures))),
                      columns=measures,
                      index=pd.RangeIndex(1, n + 1, name="match_id"))
    fm["host_nation"] = rng.integers(0, 2, n).astype(float)
    fm["knockout"] = (fm.index > n * 0.75).astype(float)      # last quarter = knockouts
    fm["qualification_jeopardy"] = np.where(fm["knockout"] == 1, 1.0,
                                            rng.uniform(0, 1, n))
    fm["alive_until"] = rng.uniform(0, 1, n)
    fm["late_alive_30"] = rng.uniform(0, 1, n)
    return fm


def test_buckets_sum_to_raw():
    board = score_matches(synthetic_features())
    buckets = [c for c in board.columns if c.startswith("bucket_")]
    assert len(buckets) == 5
    np.testing.assert_allclose(board[buckets].sum(axis=1), board["raw"], atol=1e-9)


def test_taxes_are_never_bonuses_and_knockout_gated():
    fm = synthetic_features()
    board = score_matches(fm)
    assert (board["tax_dead_rubber"] <= 1e-12).all()
    assert (board["tax_aliveness"] <= 1e-12).all()
    group_ids = fm.index[fm["knockout"] == 0]
    np.testing.assert_allclose(board.loc[group_ids, "tax_aliveness"], 0.0, atol=1e-12)
    ko_ids = fm.index[fm["knockout"] == 1]
    np.testing.assert_allclose(board.loc[ko_ids, "tax_dead_rubber"], 0.0, atol=1e-12)


def test_aliveness_floor_binds():
    """The aliveness tax may drag a knockout to the pool median, never below it."""
    fm = synthetic_features()
    board = score_matches(fm)
    floor = board.attrs["floor"]
    taxed = board[board["tax_aliveness"] < -1e-9]
    assert (taxed["raw"] >= floor - 1e-9).all()


def test_nan_measures_drop_out():
    """A NaN measure drops out of its family mean instead of poisoning the score."""
    fm = synthetic_features()
    fm.loc[fm.index[:5], "gei"] = np.nan
    board = score_matches(fm)
    assert np.isfinite(board["raw"]).all()


def test_config_overrides_change_scores():
    fm = synthetic_features()
    base = score_matches(fm)
    heavier = score_matches(fm, config={"taxes": {"dead_rubber_k": 0.9}})
    group_ids = fm.index[(fm["knockout"] == 0) & (fm["qualification_jeopardy"] < 0.5)]
    assert (heavier.loc[group_ids, "tax_dead_rubber"]
            <= base.loc[group_ids, "tax_dead_rubber"] + 1e-12).all()
    assert (heavier.loc[group_ids, "tax_dead_rubber"]
            < base.loc[group_ids, "tax_dead_rubber"] - 1e-9).any()


def test_bad_config_fails_loudly():
    with pytest.raises(ValueError):
        score_matches(synthetic_features(), config={"weights": {"flow": 0.9}})
