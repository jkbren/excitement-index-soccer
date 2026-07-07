"""Endgame aliveness — the v1.4 aliveness-layer inputs.

Neither measure belongs to a scoring family; they feed the post-hoc aliveness
deduction, where A = ½(alive_until + late_alive_30). Both walk the score-margin
step function (:func:`excitement_index.clock.margin_walk`) — the exact frozen
goal convention shared with ``time_within_one_goal`` — and ask when, and for
how long near the end, the match was still a one-score game. Extra-time epics
and completed comebacks score 1.0 on both by construction. Unknown aliveness
(empty event feed) is ``nan`` and defaults to A = 1 downstream (no deduction).
"""
from __future__ import annotations

import numpy as np

from ..clock import margin_walk
from .registry import MatchContext, measure


def _aliveness(ctx: MatchContext) -> dict:
    """Compute both aliveness inputs once per match (shared via ctx.cache)."""
    if "aliveness" not in ctx.cache:
        ctx.cache["aliveness"] = _compute(ctx)
    return ctx.cache["aliveness"]


def _compute(ctx: MatchContext) -> dict:
    ev, end = ctx.ev, ctx.end
    if ev.empty or not end or end <= 0:
        return {"alive_until": float(np.nan), "late_alive_30": float(np.nan)}
    times, margins = margin_walk(ev, ctx.home, ctx.away, end)
    last_alive = 0.0
    for i in range(len(margins)):
        if abs(margins[i]) <= 1 and times[i + 1] > times[i]:
            last_alive = times[i + 1]
    lo = max(end - 30.0, 0.0)
    within = 0.0
    for i in range(len(margins)):
        s, e = max(times[i], lo), min(times[i + 1], end)
        if e > s and abs(margins[i]) <= 1:
            within += e - s
    return {"alive_until": float(last_alive / end),
            "late_alive_30": float(within / max(end - lo, 1e-9))}


@measure("alive_until", tier="core")
def alive_until(ctx: MatchContext) -> float:
    """alive_until ∈ [0, 1]. The fraction of the match elapsed before the goal
    margin moved beyond one for good (1 if the match ends a one-score game —
    hence every extra-time match and every completed comeback scores 1).
    'When did it die.'"""
    return _aliveness(ctx)["alive_until"]


@measure("late_alive_30", tier="core")
def late_alive_30(ctx: MatchContext) -> float:
    """late_alive_30 ∈ [0, 1]. The share of the final 30 minutes of playing
    time (the window [90', 120'] when extra time was played, else [60', 90'])
    spent with the margin within one. 'Did the ending matter.'"""
    return _aliveness(ctx)["late_alive_30"]
