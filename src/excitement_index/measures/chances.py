"""Chance measures — shot volume, quality, and ex-ante leverage.

This module holds the shot-quality inputs to the excitement index. The
win-probability curve updates on goals only (a validated calibration
decision), so expected-goals (xG) information enters the pipeline here
instead: as totals (``total_npxg``, ``big_chances``), as territory
(``box_entries``), and as per-shot leverage — each shot weighted by the
counterfactual win-probability swing it would have caused had it scored.

Conventions: ``big_chances`` includes penalty kicks; ``total_npxg`` excludes
them. All shot-derived measures exclude the shootout because events are
pre-filtered to periods 1-4 upstream.
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
    """Per-shot StatsBomb xG with missing values filled to 0.

    Args:
        shots: The match's shot events (subset of the event feed).

    Returns:
        A float Series of per-shot xG aligned to ``shots``, or an empty
        Series if there are no shots or the ``shot_statsbomb_xg`` column is
        absent (an open-data feed may lack it).
    """
    if not len(shots) or "shot_statsbomb_xg" not in shots.columns:
        return pd.Series(dtype=float)
    return shots["shot_statsbomb_xg"].fillna(0.0)


def _np_mask(shots: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting non-penalty shots.

    Args:
        shots: The match's shot events.

    Returns:
        A boolean Series aligned to ``shots``, True for every non-penalty
        shot. When the ``shot_type`` column is absent, every shot is treated
        as non-penalty (all True).
    """
    if len(shots) and "shot_type" in shots.columns:
        return shots["shot_type"] != "Penalty"
    return pd.Series(True, index=shots.index)


def _shot_leverage(ctx: MatchContext) -> pd.DataFrame:
    """Per-shot leverage table for the match, computed once and cached.

    Args:
        ctx: The match context (events, team names, prior/end WP state).

    Returns:
        The per-shot leverage DataFrame from :func:`per_shot_leverage`, with a
        ``leverage`` column giving each shot's xG-weighted counterfactual WP
        swing. Cached under ``ctx.cache['shot_lev']`` so the several
        leverage-derived measures reuse the single computation.
    """
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
    """Total non-penalty xG summed across both teams.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        The pooled non-penalty xG as a float, or 0.0 when there are no shots.
    """
    shots = ctx.shots
    if not len(shots):
        return 0.0
    sx, np_mask = _shot_xg(shots), _np_mask(shots)
    # Sum non-penalty xG per team, then pool — penalties are excluded so the
    # figure reflects open-play/set-piece chance creation, not spot kicks.
    npxg_h = float(sx[np_mask & (shots["team"] == ctx.home)].sum())
    npxg_a = float(sx[np_mask & (shots["team"] == ctx.away)].sum())
    return npxg_h + npxg_a


@measure("total_shots", tier="core")
def total_shots(ctx: MatchContext) -> float:
    """Total shot count over regulation plus extra time.

    Args:
        ctx: The match context; ``ctx.shots`` is the shootout-excluded shots.

    Returns:
        The number of shots as a float.
    """
    return float(len(ctx.shots))


@measure("total_sot", tier="core")
def total_sot(ctx: MatchContext) -> float:
    """Count of shots on target.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        The number of shots whose outcome is in ``SOT_OUTCOMES`` (Goal,
        Saved, or Saved To Post) as a float, or 0.0 when there are no shots.

    Assumes any non-empty ``ctx.shots`` carries a ``shot_outcome`` column; a
    shots feed lacking that column would raise and be recorded as nan upstream.
    """
    shots = ctx.shots
    if not len(shots):
        return 0.0
    return float(shots["shot_outcome"].isin(SOT_OUTCOMES).sum())


@measure("big_chances", tier="core")
def big_chances(ctx: MatchContext) -> float:
    """Count of high-quality chances.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        The number of shots with xG >= 0.25 as a float, or 0.0 when there are
        no shots. Penalties are included (a penalty is itself a big chance).

    The 0.25 threshold is the index's fixed cutoff for a clear scoring chance.
    """
    if not len(ctx.shots):
        return 0.0
    return float((_shot_xg(ctx.shots) >= 0.25).sum())


@measure("box_entries", tier="core")
def box_entries(ctx: MatchContext) -> float:
    """Count of completed passes and carries that enter the penalty box.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed.

    Returns:
        The number of completed passes plus carries that end inside the box
        having started outside it, as a float.

    Load-bearing invariant: StatsBomb normalizes every possessing team to
    attack toward increasing x, so the attacking box is always the single
    right-hand box (x >= BOX_X). A non-normalized feed would miss entries into
    the left box entirely.
    """
    ev = ctx.ev

    def _enter(start: np.ndarray, end: np.ndarray) -> int:
        # A box entry: the end point is inside the right-hand box rectangle...
        in_end = (end[:, 0] >= BOX_X) & (end[:, 1] >= BOX_Y_LO) & (end[:, 1] <= BOX_Y_HI)
        # ...and the start point is outside it (so passes/carries wholly inside
        # the box are not counted as entries).
        out_start = ~((start[:, 0] >= BOX_X) & (start[:, 1] >= BOX_Y_LO)
                      & (start[:, 1] <= BOX_Y_HI))
        return int((in_end & out_start).sum())

    n = 0
    # Completed passes (no pass_outcome = complete) with both endpoints known.
    pas = ev[(ev["type"] == "Pass") & (ev["pass_outcome"].isna())
             & ev["location"].notna() & ev["pass_end_location"].notna()]
    if len(pas):
        n += _enter(xy(pas["location"]), xy(pas["pass_end_location"]))
    # Carries with both endpoints known (a carry has no outcome field).
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
    """Total ex-ante shot leverage over the match.

    Args:
        ctx: The match context; the leverage table is drawn from
            :func:`_shot_leverage`.

    Returns:
        The sum over shots of xG x counterfactual WP swing (each shot weighted
        by how much scoring it would have moved the outcome probabilities at
        that moment), as a float; nan when there are no shots.
    """
    lev = _shot_leverage(ctx)
    if lev.empty:
        return float(np.nan)
    return float(lev["leverage"].sum())


@measure("chance_leverage_p95", tier="core")
def chance_leverage_p95(ctx: MatchContext) -> float:
    """95th-percentile per-shot leverage — the match's near-biggest moment.

    Args:
        ctx: The match context; the leverage table is drawn from
            :func:`_shot_leverage`.

    Returns:
        The 95th percentile of the per-shot leverage values as a float; nan
        when there are no shots. The 95th percentile (rather than the max) is
        used so a single outlier shot does not define the statistic.
    """
    lev = _shot_leverage(ctx)
    if lev.empty:
        return float(np.nan)
    return float(np.percentile(lev["leverage"].to_numpy(float), 95))


@measure("shot_balance", tier="core")
def shot_balance(ctx: MatchContext) -> float:
    """Symmetry of shot volume between the two teams.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        min(shots_home, shots_away) / max(shots_home, shots_away): 1.0 for an
        even contest, near 0 for a one-way barrage. Returns 0.0 when either
        team took zero shots — note this 0.0 also covers the zero-shots match
        (max == 0 branch), so a value of 0.0 conflates 'totally lopsided' with
        'no shots at all'.
    """
    shots = ctx.shots
    sh_h = int((shots["team"] == ctx.home).sum())
    sh_a = int((shots["team"] == ctx.away).sum())
    return float(min(sh_h, sh_a) / max(sh_h, sh_a)) if max(sh_h, sh_a) else 0.0
