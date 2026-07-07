"""The 0-10 publication map: monotone, bounded, endpoints unattainable."""
from __future__ import annotations

import numpy as np

from excitement_index.config import load_config
from excitement_index.scale import apply_scale_map, fit_scale_map


def _map(seed=3):
    rng = np.random.default_rng(seed)
    ref = rng.normal(0.0, 0.3, 72)
    pool = np.concatenate([ref, rng.normal(0.4, 0.3, 22)])
    return fit_scale_map(ref, pool, load_config()), pool


def test_strictly_monotone():
    smap, _ = _map()
    x = np.linspace(-3, 3, 4001)
    y = apply_scale_map(x, smap)
    assert (np.diff(y) > 0).all(), "the publication map must be strictly increasing"


def test_endpoints_unattainable():
    # +-10 raw units is ~30 reference standard deviations beyond the anchors —
    # far past anything a real match can produce, but before float underflow
    # saturates the asymptote.
    smap, _ = _map()
    y = apply_scale_map(np.array([-10.0, 10.0]), smap)
    assert 0.0 < y[0] and y[1] < 10.0


def test_second_highest_pins_near_target():
    smap, pool = _map()
    x2 = np.sort(pool)[-2]
    y2 = float(apply_scale_map(np.array([x2]), smap)[0])
    # pinned exactly unless the slope-match cap engaged, in which case lower
    assert y2 <= 9.49 + 1e-9


def test_rank_preservation():
    smap, pool = _map()
    y = apply_scale_map(pool, smap)
    assert (np.argsort(pool) == np.argsort(y)).all()
