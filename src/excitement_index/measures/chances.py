"""Chance measures — shot volume, quality, and ex-ante leverage.

Where shot quality lives in the index. The win-probability curve updates on
goals only (a validated calibration decision), so xG enters here instead: as
totals (``total_npxg``, ``big_chances``), as territory (``box_entries``), and
— the anticipation core — as **per-shot leverage**: each shot weighted by the
counterfactual probability swing it would have caused had it scored.

Conventions: ``big_chances`` includes penalty kicks; ``total_npxg`` excludes
them. All shot-derived measures exclude the shootout because events are
pre-filtered to periods 1-4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..clock import BOX_X, BOX_Y_HI, BOX_Y_LO, SOT_OUTCOMES, xy
from ..wp import per_shot_leverage
from .registry import MatchContext, measure


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _shot_xg(shots: pd.DataFrame) -> pd.Series:
    """Per-shot xG with missing values as 0 (empty series if no shots)."""
    if not len(shots) or "shot_statsbomb_xg" not in shots.columns:
        return pd.Series(dtype=float)
    return shots["shot_statsbomb_xg"].fillna(0.0)


def _np_mask(shots: pd.DataFrame) -> pd.Series:
    """True for non-penalty shots (all True when ``shot_type`` is absent)."""
    if len(shots) and "shot_type" in shots.columns:
        return shots["shot_type"] != "Penalty"
    return pd.Series(True, index=shots.index)


def _shot_leverage(ctx: MatchContext) -> pd.DataFrame:
    """The per-shot leverage table, computed once per match and cached in
    ``ctx.cache['shot_lev']`` so the timing/resolution measures reuse it."""
    lev = ctx.cache.get("shot_lev")
    if lev is None:
        lev = per_shot_leverage(ctx.ev, home=ctx.home, away=ctx.away, end=ctx.end,
                                prior_home=ctx.prior_home, prior_away=ctx.prior_away)
        ctx.cache["shot_lev"] = lev
    return lev


# ---------------------------------------------------------------------------
# Volume & quality
# ---------------------------------------------------------------------------
@measure("total_npxg", tier="core")
def total_npxg(ctx: MatchContext) -> float:
    """Total non-penalty xG, both teams."""
    shots = ctx.shots
    if not len(shots):
        return 0.0
    sx, np_mask = _shot_xg(shots), _np_mask(shots)
    npxg_h = float(sx[np_mask & (shots["team"] == ctx.home)].sum())
    npxg_a = float(sx[np_mask & (shots["team"] == ctx.away)].sum())
    return npxg_h + npxg_a


@measure("total_shots", tier="core")
def total_shots(ctx: MatchContext) -> float:
    """Shot count (regulation + extra time)."""
    return float(len(ctx.shots))


@measure("total_sot", tier="core")
def total_sot(ctx: MatchContext) -> float:
    """Shots on target (outcome Goal, Saved, or Saved To Post)."""
    shots = ctx.shots
    if not len(shots):
        return 0.0
    return float(shots["shot_outcome"].isin(SOT_OUTCOMES).sum())


@measure("big_chances", tier="core")
def big_chances(ctx: MatchContext) -> float:
    """Shots with xG >= 0.25 (penalties included)."""
    if not len(ctx.shots):
        return 0.0
    return float((_shot_xg(ctx.shots) >= 0.25).sum())


@measure("box_entries", tier="core")
def box_entries(ctx: MatchContext) -> float:
    """Completed passes and carries that end inside the penalty box having
    started outside it."""
    ev = ctx.ev

    def _enter(start: np.ndarray, end: np.ndarray) -> int:
        in_end = (end[:, 0] >= BOX_X) & (end[:, 1] >= BOX_Y_LO) & (end[:, 1] <= BOX_Y_HI)
        out_start = ~((start[:, 0] >= BOX_X) & (start[:, 1] >= BOX_Y_LO)
                      & (start[:, 1] <= BOX_Y_HI))
        return int((in_end & out_start).sum())

    n = 0
    pas = ev[(ev["type"] == "Pass") & (ev["pass_outcome"].isna())
             & ev["location"].notna() & ev["pass_end_location"].notna()]
    if len(pas):
        n += _enter(xy(pas["location"]), xy(pas["pass_end_location"]))
    car = ev[(ev["type"] == "Carry") & ev["location"].notna()
             & ev["carry_end_location"].notna()]
    if len(car):
        n += _enter(xy(car["location"]), xy(car["carry_end_location"]))
    return float(n)


# ---------------------------------------------------------------------------
# Ex-ante leverage (anticipation)
# ---------------------------------------------------------------------------
@measure("chance_leverage_total", tier="core")
def chance_leverage_total(ctx: MatchContext) -> float:
    """Sum over shots of xG x counterfactual WP swing — each shot weighted by
    how much scoring it would have moved the outcome probabilities at that
    moment."""
    lev = _shot_leverage(ctx)
    if lev.empty:
        return float(np.nan)
    return float(lev["leverage"].sum())


@measure("chance_leverage_p95", tier="core")
def chance_leverage_p95(ctx: MatchContext) -> float:
    """95th percentile of the per-shot leverage values — the match's
    near-biggest single moment of anticipation, robust to one outlier."""
    lev = _shot_leverage(ctx)
    if lev.empty:
        return float(np.nan)
    return float(np.percentile(lev["leverage"].to_numpy(float), 95))


@measure("shot_balance", tier="core")
def shot_balance(ctx: MatchContext) -> float:
    """min(shots_home, shots_away) / max(...): 1 for an even contest, 0 for a
    one-way barrage."""
    shots = ctx.shots
    sh_h = int((shots["team"] == ctx.home).sum())
    sh_a = int((shots["team"] == ctx.away).sum())
    return float(min(sh_h, sh_a) / max(sh_h, sh_a)) if max(sh_h, sh_a) else 0.0
