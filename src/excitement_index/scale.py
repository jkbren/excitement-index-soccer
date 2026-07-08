"""The 0-10 publication scale.

A strictly monotone map from raw composite scores onto a familiar 0-10 rating
scale — it calibrates the *scale*, never the *ordering* (a monotone map cannot
change any match's rank).

* **Interior:** seven quantile anchors. The x-anchors are quantiles of the
  reference pool's raw scores; the y-anchors are a target display distribution
  at the same quantiles (the shipped defaults reproduce the published index's
  fan-scale calibration). Linear interpolation between anchors.
* **Top tail:** above the last anchor, y = 10 − a·e^(−b·(x − x₇)) — an
  exponential approach to 10, with b pinned so the pool's second-highest raw
  score publishes exactly ``top_pin_rating`` (default 9.49), capped at the
  slope-matched bound for continuity.
* **Bottom tail:** below the first anchor, the slope-matched mirror
  y = y₁·e^(slope/y₁·(x − x₁)) decaying toward 0.

Both 0.0 and 10.0 are asymptotes: unattainable by design.
"""
from __future__ import annotations

import numpy as np


def fit_scale_map(reference_raws: np.ndarray, pool_raws: np.ndarray, cfg: dict) -> dict:
    """Fit the raw-to-0-10 map from reference-pool quantiles and a top-tail pin.

    Args:
        reference_raws: Raw composite scores of the reference pool; their quantiles
            at ``scale.anchor_percentiles`` become the interior x-anchors.
        pool_raws: Raw composite scores of the full pool being scored; the
            second-highest value pins the exponential top tail.
        cfg: Resolved config; reads the ``scale`` block (``anchor_percentiles``,
            ``anchor_display_values``, ``top_pin_rating``).

    Returns:
        A dict describing the fitted map:
            - ``anchors_x`` (list[float]): interior x-anchors (raw-score units).
            - ``anchors_y`` (list[float]): interior y-anchors (0-10 display units).
            - ``a`` (float): top-tail amplitude, ``10 - y_last``.
            - ``b`` (float): top-tail decay rate.

    The top tail is y = 10 - a*exp(-b*(x - x_last)); b is solved so the pool's
    second-highest raw score publishes exactly ``top_pin_rating``, then capped at
    the slope-matched bound so the tail is never steeper than the last interior
    segment.
    """
    s = cfg["scale"]
    p = np.asarray(s["anchor_percentiles"], float)
    xa = np.quantile(np.asarray(reference_raws, float), p)
    ya = np.asarray(s["anchor_display_values"], float).copy()
    # Enforce strict monotonicity so np.interp and the log/slope math stay well
    # defined. The two epsilons differ by scale of the axis they nudge: xa is in
    # raw-score units (spacing ~1e-2 or smaller near ties), so a 1e-9 bump is
    # enough; ya is in 0-10 display units (spacing ~1e-1), so it uses a coarser
    # 1e-6 bump. Both are far below the axis's resolution and only fire on exact
    # ties or inversions.
    for i in range(1, len(xa)):                       # enforce strict monotonicity
        if xa[i] <= xa[i - 1]:
            xa[i] = xa[i - 1] + 1e-9
        if ya[i] <= ya[i - 1]:
            ya[i] = ya[i - 1] + 1e-6
    a = 10.0 - ya[-1]
    allv = np.sort(np.asarray(pool_raws, float))
    # Pin on the second-highest raw score; with fewer than two points fall back to
    # a nominal 0.1 above the top so b stays finite.
    x2 = allv[-2] if len(allv) >= 2 else allv[-1] + 0.1
    pin = float(s.get("top_pin_rating", 9.49))
    # Solve 10 - a*exp(-b*(x2 - x_last)) = pin for b; max(..., 1e-9) guards a zero gap.
    b = float(-np.log((10.0 - pin) / a) / max(x2 - xa[-1], 1e-9))
    b = min(b, ((ya[-1] - ya[-2]) / (xa[-1] - xa[-2])) / a)   # never steeper than slope-matched
    return dict(anchors_x=xa.tolist(), anchors_y=ya.tolist(), a=float(a), b=float(b))


def apply_scale_map(raw, smap: dict) -> np.ndarray:
    """Raw scores -> 0-10 display values under a fitted map."""
    xa = np.asarray(smap["anchors_x"], float)
    ya = np.asarray(smap["anchors_y"], float)
    a, b = float(smap["a"]), float(smap["b"])
    slope_bot = (ya[1] - ya[0]) / (xa[1] - xa[0])
    x = np.asarray(raw, float)
    y = np.interp(x, xa, ya)
    hi = x > xa[-1]
    y[hi] = 10.0 - a * np.exp(-b * (x[hi] - xa[-1]))
    lo = x < xa[0]
    y[lo] = ya[0] * np.exp((slope_bot / ya[0]) * (x[lo] - xa[0]))
    return y
