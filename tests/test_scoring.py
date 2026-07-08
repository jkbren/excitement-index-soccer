"""Structural contracts of the scorer, checked on synthetic feature matrices.

These tests exercise score_matches() against random-but-well-formed inputs, so they need no
event data, network, or cache. They assert the scorer's invariants — bucket decomposition,
tax sign and gating, NaN handling, the aliveness floor, config-override monotonicity, and
config validation — rather than any specific numeric score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from excitement_index import load_config, score_matches


def synthetic_features(n=40, seed=7):
    """Build a plausible random feature matrix covering every taxonomy measure.

    Args:
        n (int): Number of synthetic matches (rows). Default 40.
        seed (int): Seed for the NumPy random generator, for reproducible test inputs.

    Returns:
        pandas.DataFrame: One row per synthetic match, indexed by a 1..n "match_id". Columns
        are every measure named in the config taxonomy (drawn N(1.0, 0.5)) plus the scoring
        side-inputs host_nation, knockout, qualification_jeopardy, alive_until, late_alive_30.

    The measure values are centered at 1.0 with spread 0.5 only to look like real standardized
    features; the tests never depend on the exact draws. The last quarter of matches (match_id
    > 0.75 * n) is flagged as knockout so both the group-only and knockout-only tax paths get
    exercised; group matches get a random jeopardy in [0, 1] while knockouts are pinned to 1.0.
    """
    cfg = load_config()
    rng = np.random.default_rng(seed)
    measures = [m for feats in cfg["taxonomy"].values() for m in feats]
    fm = pd.DataFrame(rng.normal(1.0, 0.5, size=(n, len(measures))),
                      columns=measures,
                      index=pd.RangeIndex(1, n + 1, name="match_id"))
    fm["host_nation"] = rng.integers(0, 2, n).astype(float)
    # Treat the final quarter of the schedule as knockout matches (0.75 = last 25% of rows).
    fm["knockout"] = (fm.index > n * 0.75).astype(float)
    # Knockouts always carry maximal jeopardy (1.0); group matches get a random stake in [0, 1].
    fm["qualification_jeopardy"] = np.where(fm["knockout"] == 1, 1.0,
                                            rng.uniform(0, 1, n))
    fm["alive_until"] = rng.uniform(0, 1, n)
    fm["late_alive_30"] = rng.uniform(0, 1, n)
    return fm


def test_buckets_sum_to_raw():
    """The five per-family bucket contributions must add up exactly to the raw score.

    The scorer decomposes each match's raw score into one column per taxonomy family
    (bucket_*); this checks there are exactly five and that they sum back to "raw" (atol
    1e-9 to absorb floating-point summation error, well below any editorially meaningful gap).
    """
    board = score_matches(synthetic_features())
    buckets = [c for c in board.columns if c.startswith("bucket_")]
    assert len(buckets) == 5
    np.testing.assert_allclose(board[buckets].sum(axis=1), board["raw"], atol=1e-9)


def test_taxes_are_never_bonuses_and_knockout_gated():
    """Both taxes are non-positive, and each applies only to the stage it is gated to.

    tax_dead_rubber and tax_aliveness are penalties, so neither may ever be positive
    (asserted <= 1e-12, i.e. zero up to float noise). Gating: the aliveness tax is
    knockout-only (exactly 0 on group matches) and the dead-rubber tax is group-only
    (exactly 0 on knockout matches). The 1e-12 tolerances stand in for exact zero.
    """
    fm = synthetic_features()
    board = score_matches(fm)
    # Neither tax may act as a bonus (a positive value); allow only float-noise above zero.
    assert (board["tax_dead_rubber"] <= 1e-12).all()
    assert (board["tax_aliveness"] <= 1e-12).all()
    # Aliveness tax is knockout-only: it must be exactly zero on every group-stage match.
    group_ids = fm.index[fm["knockout"] == 0]
    np.testing.assert_allclose(board.loc[group_ids, "tax_aliveness"], 0.0, atol=1e-12)
    # Dead-rubber tax is group-only: it must be exactly zero on every knockout match.
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
    """A heavier dead-rubber coefficient must monotonically deepen the dead-rubber tax.

    Raising the tax strength dead_rubber_k (0.9 here vs the config default) may only make the
    penalty more negative, never less. On the low-jeopardy group matches (< 0.5, where the tax
    actually bites) every value must be no larger than the baseline (allowing 1e-12 float
    noise), and at least one must be strictly smaller (1e-9), proving the override took effect.
    """
    fm = synthetic_features()
    base = score_matches(fm)
    heavier = score_matches(fm, config={"taxes": {"dead_rubber_k": 0.9}})
    # The dead-rubber tax only engages on group matches with low qualification stakes (< 0.5).
    group_ids = fm.index[(fm["knockout"] == 0) & (fm["qualification_jeopardy"] < 0.5)]
    # Heavier k: no match's tax may move upward (become less of a penalty).
    assert (heavier.loc[group_ids, "tax_dead_rubber"]
            <= base.loc[group_ids, "tax_dead_rubber"] + 1e-12).all()
    # ...and at least one must move strictly downward, so the override is not a no-op.
    assert (heavier.loc[group_ids, "tax_dead_rubber"]
            < base.loc[group_ids, "tax_dead_rubber"] - 1e-9).any()


def test_bad_config_fails_loudly():
    """An invalid config (family weights that do not satisfy the scorer's contract) must raise.

    Passing weights={"flow": 0.9} alone is not a valid weight set, so score_matches must reject
    it with a ValueError rather than silently scoring against a malformed configuration.
    """
    with pytest.raises(ValueError):
        score_matches(synthetic_features(), config={"weights": {"flow": 0.9}})
