"""Aggregation: measures -> standardized scores -> weighted families -> raw score.

The pipeline (matching the published method exactly):

1. **Standardize.** Every measure is z-scored against a *reference pool* of
   matches: z = (x − μ_ref) / (σ_ref + 1e-6), population σ, then clipped to
   [−3, +3]. Bounded 0/1 flags listed in ``config.fixed_scale`` skip the
   z-score and enter as raw flag × scale. Measures in ``negative_signs`` are
   multiplied by −1 (they count against excitement).
2. **Aggregate.** Family score = equal-weight mean of the family's available
   (non-nan) z-scores × the family weight; the raw core = sum over families
   (renormalized if whole families are unavailable).
3. **Deduct.** The dead-rubber tax, k·(1 − jeopardy)·max(core, 0), and the
   knockout-gated aliveness tax, min(δ·(1 − A), headroom-above-the-pool-median).
4. **Decompose.** Family contributions regroup into the five display buckets
   (both taxes fold into the *stakes* bucket); buckets sum exactly to raw.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import ConfigLike, load_config


def make_reference(features: pd.DataFrame, reference_ids: Sequence | None = None):
    """Compute the per-measure mean and standard deviation of the reference pool.

    The reference pool is the set of matches every measure is standardized against
    (the z-score baseline in :func:`score_matches`). The published index uses the
    tournament's group stage as this baseline.

    Args:
        features: Feature matrix, one row per match, indexed by match id. Only the
            numeric columns contribute to the returned statistics.
        reference_ids: Explicit match ids to use as the reference pool. When ``None``,
            defaults to every row with ``knockout == 0`` if that column exists, else
            all rows.

    Returns:
        A 3-tuple ``(mu, sigma, ref_ids)``:
            - ``mu`` (pd.Series): per-measure mean over the reference pool.
            - ``sigma`` (pd.Series): per-measure population standard deviation
              (``ddof=0``) over the reference pool.
            - ``ref_ids`` (list): the resolved reference match ids in pool order;
              the caller uses these to fit the display scale map on the same pool.

    Population sigma (``ddof=0``) is used because the pool is treated as the full
    population of benchmark matches, not a sample. An empty pool raises rather than
    producing all-NaN statistics downstream.
    """
    if reference_ids is not None:
        ref = features.loc[[i for i in features.index if i in set(reference_ids)]]
    elif "knockout" in features.columns:
        ref = features[features["knockout"] == 0]
    else:
        ref = features
    if ref.empty:
        raise ValueError("empty reference pool — pass reference_ids explicitly")
    num = ref.select_dtypes(include=[np.number])
    return num.mean(), num.std(ddof=0), list(ref.index)


def score_matches(features: pd.DataFrame, *, reference_ids: Sequence | None = None,
                  config: ConfigLike = None, floor: float | None = None) -> pd.DataFrame:
    """Score every match in a feature matrix into a raw composite and a 0-10 rating.

    This is the top-level entry point that runs the full pipeline described in the
    module docstring: standardize measures, aggregate into weighted families,
    subtract the two taxes, decompose into display buckets, and map to the 0-10
    scale.

    Args:
        features: Feature matrix, one row per match indexed by match id, as produced
            by :func:`excitement_index.build_feature_matrix`. May carry optional
            metadata columns (``slug``, ``home``, ``away``, ``stage``) and the
            columns the taxes read (``qualification_jeopardy``, ``knockout``,
            ``alive_until``, ``late_alive_30``).
        reference_ids: Match ids defining the z-score reference pool; passed through
            to :func:`make_reference`. ``None`` uses that function's default.
        config: Config source resolved by :func:`excitement_index.config.load_config`
            (``None`` -> shipped defaults, a path, or a mapping of overrides).
        floor: Override for the aliveness-tax floor, in raw-score units. When ``None``,
            the floor is the scored pool's median after-dead-rubber-tax raw score
            (the published recipe). Because this default is a median over the pool
            being scored, a match's rating depends on which other matches are scored
            alongside it; pass an explicit ``floor`` when scoring small or single-match
            batches where the median is unstable.

    Returns:
        A DataFrame indexed by match id, sorted best-first by ``raw``, with columns:
            - any present metadata columns (``slug``, ``home``, ``away``, ``stage``),
            - ``raw`` (float): the composite score after both taxes,
            - ``rating`` (float): the 0-10 display value, rounded to
              ``scale.round_decimals`` decimals,
            - ``bucket_<name>`` (float): the five display-bucket contributions, which
              sum exactly to ``raw``,
            - ``tax_dead_rubber`` / ``tax_aliveness`` (float, <= 0): the two tax line
              items as signed deductions (informational — already folded into the
              ``stakes`` bucket).
        ``out.attrs`` also carries the fitted ``scale_map`` and the resolved
        ``floor``.
    """
    cfg = load_config(config)
    taxonomy, weights = cfg["taxonomy"], cfg["weights"]
    # negative_signs measures count against excitement, so their sign flips to -1.
    signs = {m: -1.0 for m in cfg.get("negative_signs", [])}
    # fixed_scale measures are bounded 0/1 flags that bypass the z-score.
    fixed = cfg.get("fixed_scale", {}) or {}

    # Standardize: z = (x - mu_ref) / (sigma_ref + 1e-6), clipped to +/-3 std devs.
    # The 1e-6 in the denominator guards against a zero-variance measure; the +/-3
    # clip caps the influence of any single outlier match.
    mu, sd, ref_ids = make_reference(features, reference_ids)
    cols = [m for feats in taxonomy.values() for m in feats if m in features.columns]
    X = features[cols].astype(float)
    z = ((X - mu[cols]) / (sd[cols] + 1e-6)).clip(-3, 3)
    for m, scale in fixed.items():
        if m in z.columns:                      # bounded flags: raw value x scale, no clip
            z[m] = X[m] * float(scale)
    for m in z.columns:
        z[m] = z[m] * signs.get(m, 1.0)

    # Aggregate: each family score is the equal-weight mean of its available
    # z-scores times the family weight.
    fam = {}
    for g, feats in taxonomy.items():
        present = [m for m in feats if m in z.columns]
        if present:
            fam[g] = z[present].mean(axis=1) * weights[g]   # nan-skipping row mean
    F = pd.DataFrame(fam)
    # Pool-level renormalization: total_w is the sum of weights of the families
    # present in the whole pool (computed once). Dividing by it rescales the core
    # back to a full-weight basis when entire families are unavailable. Note the
    # per-row asymmetry this creates: F.sum(axis=1) below skips NaN, so a row whose
    # entire family is NaN contributes 0 for that family while total_w still counts
    # that family's weight, under-weighting that row. This is intended and
    # reference-faithful.
    total_w = sum(weights[g] for g in F.columns)
    F = F / total_w                                          # renormalize if families missing
    core = F.sum(axis=1)

    # Dead-rubber tax: k * (1 - jeopardy) * max(core, 0). Multiplicative in positive
    # quality (a dull match has little to lose) and inert without jeopardy data.
    k = float(cfg["taxes"]["dead_rubber_k"])
    jeo = (features["qualification_jeopardy"].astype(float)
           if "qualification_jeopardy" in features.columns
           else pd.Series(np.nan, index=features.index))
    tax = np.where(np.isfinite(jeo.to_numpy()),
                   k * (1.0 - jeo.to_numpy()) * core.clip(lower=0).to_numpy(), 0.0)
    after = core - tax

    # Aliveness tax: knockout-gated delta * (1 - A), capped by the headroom above
    # the pool floor so deadness makes a knockout mediocre, never historically bad.
    delta = float(cfg["taxes"]["aliveness_delta"])
    ko = (features["knockout"].astype(float)
          if "knockout" in features.columns else pd.Series(0.0, index=features.index))
    if {"alive_until", "late_alive_30"} <= set(features.columns):
        # Aliveness A is the mean of the two aliveness proxies (0.5 each).
        A = 0.5 * (features["alive_until"].astype(float) + features["late_alive_30"].astype(float))
    else:
        A = pd.Series(np.nan, index=features.index)
    A = A.fillna(1.0)                                        # unknown aliveness -> no tax
    # Floor is pool-dependent by default (median of the after-dead-rubber scores of
    # the pool being scored); see the `floor` arg note on small-batch instability.
    m_floor = float(after.median()) if floor is None else float(floor)
    ded = np.minimum(delta * ko.to_numpy() * (1.0 - A.to_numpy()),
                     (after - m_floor).clip(lower=0).to_numpy())
    raw = after - ded

    # Decompose: regroup family contributions into the five display buckets.
    parts = pd.DataFrame({b: F[[g for g in fams if g in F.columns]].sum(axis=1)
                          for b, fams in cfg["buckets"].items()})
    parts["stakes"] = parts["stakes"] - tax - ded            # both taxes live in Stakes

    # Fit the 0-10 map on the reference pool's raw scores, then apply it to all rows.
    from .scale import apply_scale_map, fit_scale_map
    smap = fit_scale_map(raw.loc[[i for i in ref_ids if i in raw.index]].to_numpy(float),
                         raw.to_numpy(float), cfg)
    out = pd.DataFrame(index=features.index)
    for c in ("slug", "home", "away", "stage"):
        if c in features.columns:
            out[c] = features[c]
    out["raw"] = raw
    out["rating"] = np.round(apply_scale_map(raw.to_numpy(float), smap),
                             int(cfg["scale"]["round_decimals"]))
    for b in parts.columns:
        out[f"bucket_{b}"] = parts[b]
    # Taxes are reported as signed deductions (<= 0) even though they already sit
    # inside the stakes bucket.
    out["tax_dead_rubber"] = -tax
    out["tax_aliveness"] = -ded
    out.attrs["scale_map"] = smap
    out.attrs["floor"] = m_floor
    return out.sort_values("raw", ascending=False)
