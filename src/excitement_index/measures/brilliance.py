"""Brilliance measures — individual quality and attacking flair.

Flair volume (take-ons, long carries, through balls, directness) plus two
brilliance-of-the-moment measures (screamer goals, individual takeover). All
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
    """Completed dribbles past an opponent (dribble outcome *Complete*)."""
    ev = ctx.ev
    drib = ev[ev["type"] == "Dribble"]
    if "dribble_outcome" in drib.columns and len(drib):
        return float((drib["dribble_outcome"] == "Complete").sum())
    return 0.0


@measure("long_carries", tier="core")
def long_carries(ctx) -> float:
    """Ball carries covering >= 15 pitch units (~15 m) start to end."""
    ev = ctx.ev
    car = ev[(ev["type"] == "Carry") & ev["location"].notna()
             & ev["carry_end_location"].notna()]
    if car.empty:
        return 0.0
    a, b = xy(car["location"]), xy(car["carry_end_location"])
    return float((np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]) >= 15.0).sum())


@measure("line_breaking_passes", tier="core")
def line_breaking_passes(ctx) -> float:
    """Passes carrying StatsBomb's through-ball flag."""
    ev = ctx.ev
    if "pass_through_ball" in ev.columns:
        return float((ev["pass_through_ball"] == True).sum())  # noqa: E712
    return 0.0


@measure("directness", tier="core")
def directness(ctx) -> float:
    """Share of completed passes gaining >= 5 pitch units toward goal."""
    ev = ctx.ev
    pas = ev[(ev["type"] == "Pass") & (ev["pass_outcome"].isna())]
    if pas.empty:
        return float(np.nan)
    loc = pas.loc[pas["location"].notna(), "location"]
    s = xy(loc) if len(loc) else np.empty((0, 2))
    ends = pas.loc[pas["pass_end_location"].notna(), "pass_end_location"]
    if len(ends) != len(pas) or len(s) != len(pas):
        # align safely on rows with both endpoints
        m = pas["location"].notna() & pas["pass_end_location"].notna()
        pas = pas[m]
        if pas.empty:
            return float(np.nan)
        s = xy(pas["location"])
        e = xy(pas["pass_end_location"])
    else:
        e = xy(ends)
    fwd = (e[:, 0] - s[:, 0]) >= 5.0
    return float(fwd.mean())


@measure("screamer_goals", tier="core")
def screamer_goals(ctx) -> float:
    """Sum of 0.5 * (1 - xG) over non-penalty goals with xG < 0.08 —
    improbable goals, credited by their improbability."""
    shots = ctx.shots
    if len(shots) and "shot_outcome" in shots.columns:
        sx = shots["shot_statsbomb_xg"].fillna(0.0)
        if "shot_type" in shots.columns:
            np_mask = shots["shot_type"] != "Penalty"
        else:
            np_mask = pd.Series(True, index=shots.index)
        scr = (shots["shot_outcome"] == "Goal") & np_mask & (sx < 0.08)
        return float((0.5 * (1.0 - sx[scr])).sum())
    return 0.0


@measure("individual_takeover", tier="core")
def individual_takeover(ctx) -> float:
    """Largest single-player total of positive on-ball value (OBV) — one
    player seizing the match, measured fame-free. Where OBV is unavailable,
    an event-based fallback (take-ons, shot assists, and goal quality per
    player) is used: 0.06 per completed take-on + 0.04 per shot-assist pass
    + sum over goals of (0.3 + 0.3 * (1 - xG))."""
    ev = ctx.ev
    if "player" not in ev.columns:
        return 0.0
    if "obv_total_net" in ev.columns and ev["obv_total_net"].notna().any():
        s = ev[ev["player"].notna() & ev["obv_total_net"].notna()]
        if s.empty:
            return 0.0
        pos = s["obv_total_net"].clip(lower=0).groupby(s["player"]).sum()
        return float(pos.max())
    score = pd.Series(dtype=float)
    if "dribble_outcome" in ev.columns:
        d = ev[(ev["type"] == "Dribble") & (ev["dribble_outcome"] == "Complete")
               & ev["player"].notna()]
        if len(d):
            score = score.add(d["player"].value_counts() * 0.06, fill_value=0.0)
    if "pass_shot_assist" in ev.columns:
        k = ev[(ev["pass_shot_assist"] == True) & ev["player"].notna()]  # noqa: E712
        if len(k):
            score = score.add(k["player"].value_counts() * 0.04, fill_value=0.0)
    if "shot_outcome" in ev.columns:
        g = ev[(ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")
               & ev["player"].notna()]
        if len(g):
            xg = g["shot_statsbomb_xg"].fillna(0.0) if "shot_statsbomb_xg" in g.columns \
                else pd.Series(0.0, index=g.index)
            score = score.add((0.3 + 0.3 * (1.0 - xg)).groupby(g["player"]).sum(),
                              fill_value=0.0)
    return float(score.max()) if len(score) else 0.0
