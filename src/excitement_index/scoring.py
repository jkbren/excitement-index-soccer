"""Aggregation: measures -> standardized scores -> weighted families -> raw score.

The pipeline (matching the published v1.4 method exactly):

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

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import ConfigLike, load_config


def make_reference(features: pd.DataFrame, reference_ids: Optional[Sequence] = None):
    """(μ, σ) per measure over the reference pool.

    ``reference_ids`` selects the benchmark matches (the published index uses
    the tournament's group stage). Defaults to every row with ``knockout == 0``
    when that column exists, else all rows."""
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


def score_matches(features: pd.DataFrame, *, reference_ids: Optional[Sequence] = None,
                  config: ConfigLike = None, floor: Optional[float] = None) -> pd.DataFrame:
    """Score every match in a feature matrix (one row per match, as produced by
    :func:`excitement_index.build_feature_matrix`).

    Returns a DataFrame sorted best-first with: ``raw`` (the composite),
    ``rating`` (the 0-10 display value), the five bucket contributions
    (summing exactly to ``raw``), and the two tax line items (informational —
    already contained inside the ``stakes`` bucket).

    ``floor`` overrides the aliveness-tax floor (default: the scored pool's
    median after-dead-rubber-tax raw score, the published recipe)."""
    cfg = load_config(config)
    taxonomy, weights = cfg["taxonomy"], cfg["weights"]
    signs = {m: -1.0 for m in cfg.get("negative_signs", [])}
    fixed = cfg.get("fixed_scale", {}) or {}

    mu, sd, ref_ids = make_reference(features, reference_ids)
    cols = [m for feats in taxonomy.values() for m in feats if m in features.columns]
    X = features[cols].astype(float)
    z = ((X - mu[cols]) / (sd[cols] + 1e-6)).clip(-3, 3)
    for m, scale in fixed.items():
        if m in z.columns:                      # bounded flags: raw value x scale, no clip
            z[m] = X[m] * float(scale)
    for m in z.columns:
        z[m] = z[m] * signs.get(m, 1.0)

    fam = {}
    for g, feats in taxonomy.items():
        present = [m for m in feats if m in z.columns]
        if present:
            fam[g] = z[present].mean(axis=1) * weights[g]   # nan-skipping row mean
    F = pd.DataFrame(fam)
    total_w = sum(weights[g] for g in F.columns)
    F = F / total_w                                          # renormalize if families missing
    core = F.sum(axis=1)

    # Dead-rubber tax: multiplicative in positive quality; inert without jeopardy.
    k = float(cfg["taxes"]["dead_rubber_k"])
    jeo = (features["qualification_jeopardy"].astype(float)
           if "qualification_jeopardy" in features.columns
           else pd.Series(np.nan, index=features.index))
    tax = np.where(np.isfinite(jeo.to_numpy()),
                   k * (1.0 - jeo.to_numpy()) * core.clip(lower=0).to_numpy(), 0.0)
    after = core - tax

    # Aliveness tax: knockout-gated, floored at the pool median (deadness makes
    # a knockout mediocre, never historically bad).
    delta = float(cfg["taxes"]["aliveness_delta"])
    ko = (features["knockout"].astype(float)
          if "knockout" in features.columns else pd.Series(0.0, index=features.index))
    if {"alive_until", "late_alive_30"} <= set(features.columns):
        A = 0.5 * (features["alive_until"].astype(float) + features["late_alive_30"].astype(float))
    else:
        A = pd.Series(np.nan, index=features.index)
    A = A.fillna(1.0)                                        # unknown aliveness -> no tax
    m_floor = float(after.median()) if floor is None else float(floor)
    ded = np.minimum(delta * ko.to_numpy() * (1.0 - A.to_numpy()),
                     (after - m_floor).clip(lower=0).to_numpy())
    raw = after - ded

    parts = pd.DataFrame({b: F[[g for g in fams if g in F.columns]].sum(axis=1)
                          for b, fams in cfg["buckets"].items()})
    parts["stakes"] = parts["stakes"] - tax - ded            # both taxes live in Stakes

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
    out["tax_dead_rubber"] = -tax
    out["tax_aliveness"] = -ded
    out.attrs["scale_map"] = smap
    out.attrs["floor"] = m_floor
    return out.sort_values("raw", ascending=False)
