"""Brilliance measures — individual quality and attacking flair.

Flair-volume measures (take-ons, long carries, through balls, directness) plus
two moment-of-brilliance measures (screamer goals, individual takeover). All
compute on any StatsBomb feed: ``individual_takeover`` prefers on-ball value
(OBV) when the feed carries it but falls back to a common-core event recipe,
so it is never ``nan``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..clock import xy
from .registry import measure


@measure("take_ons", tier="core")
def take_ons(ctx) -> float:
    """Count of completed dribbles past an opponent.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed.

    Returns:
        The number of Dribble events with outcome *Complete* as a float, or
        0.0 when the ``dribble_outcome`` column is absent or there are none.
    """
    ev = ctx.ev
    drib = ev[ev["type"] == "Dribble"]
    if "dribble_outcome" in drib.columns and len(drib):
        return float((drib["dribble_outcome"] == "Complete").sum())
    return 0.0


@measure("long_carries", tier="core")
def long_carries(ctx) -> float:
    """Count of long ball carries.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed.

    Returns:
        The number of carries covering >= 15 pitch units (~15 m) in straight-
        line start-to-end distance, as a float; 0.0 when there are no carries
        with both endpoints known. The 15-unit threshold marks a carry that
        moved the ball a meaningful distance rather than a touch.
    """
    ev = ctx.ev
    car = ev[(ev["type"] == "Carry") & ev["location"].notna()
             & ev["carry_end_location"].notna()]
    if car.empty:
        return 0.0
    # Straight-line carry distance in pitch units (StatsBomb pitch ~ meters).
    a, b = xy(car["location"]), xy(car["carry_end_location"])
    return float((np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]) >= 15.0).sum())


@measure("line_breaking_passes", tier="core")
def line_breaking_passes(ctx) -> float:
    """Count of through balls.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed.

    Returns:
        The number of passes carrying StatsBomb's through-ball flag as a
        float, or 0.0 when the ``pass_through_ball`` column is absent.
    """
    ev = ctx.ev
    if "pass_through_ball" in ev.columns:
        return float((ev["pass_through_ball"] == True).sum())  # noqa: E712
    return 0.0


@measure("directness", tier="core")
def directness(ctx) -> float:
    """Share of completed passes that advance the ball toward goal.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed.

    Returns:
        The fraction of completed passes gaining >= 5 pitch units in x (toward
        goal, given StatsBomb's attack-toward-increasing-x normalization) as a
        float; nan when there are no completed passes. The 5-unit threshold
        marks a forward pass rather than a lateral/backward ball.
    """
    ev = ctx.ev
    pas = ev[(ev["type"] == "Pass") & (ev["pass_outcome"].isna())]
    if pas.empty:
        return float(np.nan)
    # Fast path: start points for rows that have a location.
    loc = pas.loc[pas["location"].notna(), "location"]
    s = xy(loc) if len(loc) else np.empty((0, 2))
    ends = pas.loc[pas["pass_end_location"].notna(), "pass_end_location"]
    # `s` and `e` must be positionally aligned to the SAME rows. The fast path
    # above is valid only when every pass has both endpoints (so start-array
    # and end-array both span all rows in order). If either endpoint is missing
    # on any row, that alignment breaks, so re-filter to rows that have BOTH
    # endpoints and rebuild both arrays from that identical row set.
    if len(ends) != len(pas) or len(s) != len(pas):
        m = pas["location"].notna() & pas["pass_end_location"].notna()
        pas = pas[m]
        if pas.empty:
            return float(np.nan)
        s = xy(pas["location"])
        e = xy(pas["pass_end_location"])
    else:
        e = xy(ends)
    # Forward gain of >= 5 pitch units in x.
    fwd = (e[:, 0] - s[:, 0]) >= 5.0
    return float(fwd.mean())


@measure("screamer_goals", tier="core")
def screamer_goals(ctx) -> float:
    """Improbability-weighted count of long-odds goals.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        The sum of 0.5 * (1 - xG) over non-penalty goals with xG < 0.08, as a
        float; 0.0 when there are no shots (or no ``shot_outcome`` column).
        The 0.08 xG cutoff isolates goals scored against long odds; the
        0.5 * (1 - xG) weight credits each by how improbable it was.
    """
    shots = ctx.shots
    if len(shots) and "shot_outcome" in shots.columns:
        sx = shots["shot_statsbomb_xg"].fillna(0.0)
        # Non-penalty mask (a screamer is an open-play strike, not a spot kick).
        if "shot_type" in shots.columns:
            np_mask = shots["shot_type"] != "Penalty"
        else:
            np_mask = pd.Series(True, index=shots.index)
        scr = (shots["shot_outcome"] == "Goal") & np_mask & (sx < 0.08)
        return float((0.5 * (1.0 - sx[scr])).sum())
    return 0.0


@measure("individual_takeover", tier="core")
def individual_takeover(ctx) -> float:
    """Largest single-player contribution — one player seizing the match.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed.

    Returns:
        The maximum over players of their positive on-ball value (OBV) total
        as a float, or 0.0 when there is no ``player`` column or no scoring
        events. Where OBV is unavailable (open data), an event-based fallback
        sums per player: 0.06 per completed take-on + 0.04 per shot-assist
        pass + sum over goals of (0.3 + 0.3 * (1 - xG)).

    The fallback coefficients are a bespoke scoring recipe calibrated so the
    common-core fallback lands on roughly the same scale as the OBV path;
    they are frozen and reproduced from the reference implementation (the
    golden-parity tests pin their exact values).
    """
    ev = ctx.ev
    if "player" not in ev.columns:
        return 0.0
    # Preferred path: subscription feeds carry per-event OBV. Sum each player's
    # positive OBV (negative actions clipped to 0) and take the match maximum.
    if "obv_total_net" in ev.columns and ev["obv_total_net"].notna().any():
        s = ev[ev["player"].notna() & ev["obv_total_net"].notna()]
        if s.empty:
            return 0.0
        pos = s["obv_total_net"].clip(lower=0).groupby(s["player"]).sum()
        return float(pos.max())
    # Fallback path (open data, no OBV): accumulate a per-player score.
    score = pd.Series(dtype=float)
    if "dribble_outcome" in ev.columns:
        d = ev[(ev["type"] == "Dribble") & (ev["dribble_outcome"] == "Complete")
               & ev["player"].notna()]
        if len(d):
            # 0.06 per completed take-on.
            score = score.add(d["player"].value_counts() * 0.06, fill_value=0.0)
    if "pass_shot_assist" in ev.columns:
        k = ev[(ev["pass_shot_assist"] == True) & ev["player"].notna()]  # noqa: E712
        if len(k):
            # 0.04 per shot-assist pass.
            score = score.add(k["player"].value_counts() * 0.04, fill_value=0.0)
    if "shot_outcome" in ev.columns:
        g = ev[(ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")
               & ev["player"].notna()]
        if len(g):
            xg = g["shot_statsbomb_xg"].fillna(0.0) if "shot_statsbomb_xg" in g.columns \
                else pd.Series(0.0, index=g.index)
            # Per goal: 0.3 base + 0.3 * (1 - xG), so harder goals score more.
            score = score.add((0.3 + 0.3 * (1.0 - xg)).groupby(g["player"]).sum(),
                              fill_value=0.0)
    return float(score.max()) if len(score) else 0.0
