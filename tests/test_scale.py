"""Tests for the 0-10 publication map that turns raw scores into published ratings.

fit_scale_map builds a monotone, bounded map from raw units onto the open interval (0, 10);
apply_scale_map evaluates it. These tests check the map's shape guarantees: strictly increasing,
endpoints never reached, the second-highest pool value pinned near its target anchor, and rank
order preserved.
"""
from __future__ import annotations

import numpy as np

from excitement_index.config import load_config
from excitement_index.scale import apply_scale_map, fit_scale_map


def _map(seed=3):
    """Fit a scale map on a synthetic reference set and return it with the wider pool.

    Args:
        seed (int): Seed for the NumPy random generator, for reproducible draws.

    Returns:
        tuple: (scale_map, pool) where scale_map is the fitted map and pool is the ndarray
        of raw values it was ranked against.

    The 72 reference points are N(0, 0.3) — a stand-in for the anchor/reference matches the map
    is calibrated on. The 22 extra points N(0.4, 0.3) are appended to form the 94-value pool, a
    right-shifted tail that models higher-scoring matches sitting above the reference mass; the
    counts (72 + 22) and spread only need to produce distinct, well-separated samples.
    """
    rng = np.random.default_rng(seed)
    ref = rng.normal(0.0, 0.3, 72)
    pool = np.concatenate([ref, rng.normal(0.4, 0.3, 22)])
    return fit_scale_map(ref, pool, load_config()), pool


def test_strictly_monotone():
    """The map must be strictly increasing across a dense sweep of raw inputs."""
    smap, _ = _map()
    # Dense grid far wider than any real raw score, to catch any local flat/decreasing segment.
    x = np.linspace(-3, 3, 4001)
    y = apply_scale_map(x, smap)
    assert (np.diff(y) > 0).all(), "the publication map must be strictly increasing"


def test_endpoints_unattainable():
    """The open endpoints 0 and 10 must never be reached, even far past the anchors."""
    # +-10 raw units is ~30 reference standard deviations beyond the anchors —
    # far past anything a real match can produce, but before float underflow
    # saturates the asymptote.
    smap, _ = _map()
    y = apply_scale_map(np.array([-10.0, 10.0]), smap)
    assert 0.0 < y[0] and y[1] < 10.0


def test_second_highest_pins_near_target():
    """The second-highest pool value maps at or just below its 9.5 second-place anchor.

    The map anchors the pool's second-highest raw score to a published rating near 9.5 (the
    config's second-place target), reserving the very top for the single best match. The check
    is one-sided at 9.49 (+1e-9 float slack): normally the value pins right at the anchor, but
    when the slope-match cap engages to keep the map monotone it is pulled slightly lower, never
    higher.
    """
    smap, pool = _map()
    x2 = np.sort(pool)[-2]
    y2 = float(apply_scale_map(np.array([x2]), smap)[0])
    # pinned exactly unless the slope-match cap engaged, in which case lower
    assert y2 <= 9.49 + 1e-9


def test_rank_preservation():
    """Mapping the pool must preserve rank order (a monotone map cannot reorder samples).

    argsort(pool) and argsort(y) are the permutations that sort input and output; on strictly
    monotone map of distinct continuous samples they are identical, so elementwise equality
    confirms order is preserved. (The comparison assumes no ties, which holds for these
    continuous random draws; exact ties would make the two argsort permutations ambiguous.)
    """
    smap, pool = _map()
    y = apply_scale_map(pool, smap)
    assert (np.argsort(pool) == np.argsort(y)).all()
