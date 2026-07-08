"""Endgame aliveness — the aliveness-layer inputs.

Neither measure belongs to a scoring family; they feed the post-hoc aliveness
deduction, where A = 1/2 * (alive_until + late_alive_30). Both walk the
score-margin step function (:func:`excitement_index.clock.margin_walk`) — the
frozen goal convention shared with ``time_within_one_goal`` — and ask when, and
for how long near the end, the match was still a one-score game. Extra-time
matches and completed comebacks score 1.0 on both by construction. Unknown
aliveness (empty event feed) is ``nan`` and defaults to A = 1 downstream (no
deduction).
"""
from __future__ import annotations

import numpy as np

from ..clock import margin_walk
from .registry import MatchContext, measure


def _aliveness(ctx: MatchContext) -> dict:
    """Compute both aliveness inputs once per match, cached via ``ctx.cache``.

    Args:
        ctx: The match context; results are memoized under ``ctx.cache``.

    Returns:
        The dict from :func:`_compute` with keys ``alive_until`` and
        ``late_alive_30``.
    """
    if "aliveness" not in ctx.cache:
        ctx.cache["aliveness"] = _compute(ctx)
    return ctx.cache["aliveness"]


def _compute(ctx: MatchContext) -> dict:
    """Walk the score-margin step function to derive both aliveness inputs.

    Args:
        ctx: The match context; ``ctx.ev`` (events), ``ctx.home``/``ctx.away``
            (team names), and ``ctx.end`` (end minute of play) are used.

    Returns:
        A dict with ``alive_until`` (fraction of the match before the margin
        left one-goal range for good, in [0, 1]) and ``late_alive_30`` (share
        of the final 30 minutes spent within one goal, in [0, 1]). Both are nan
        when the event feed is empty or the end minute is non-positive.
    """
    ev, end = ctx.ev, ctx.end
    if ev.empty or not end or end <= 0:
        return {"alive_until": float(np.nan), "late_alive_30": float(np.nan)}
    # margin_walk returns len(times) == len(margins) + 1: `times` are the
    # segment boundaries and `margins[i]` is the score margin over the segment
    # [times[i], times[i+1]]. The loops rely on that off-by-one invariant to
    # index times[i + 1] safely — a change to margin_walk that broke it would
    # raise IndexError here.
    times, margins = margin_walk(ev, ctx.home, ctx.away, end)
    # Latest time at which the margin was still within one goal (a live game).
    last_alive = 0.0
    for i in range(len(margins)):
        if abs(margins[i]) <= 1 and times[i + 1] > times[i]:
            last_alive = times[i + 1]
    # Window covering the final 30 minutes of play (clamped at 0).
    lo = max(end - 30.0, 0.0)
    within = 0.0
    for i in range(len(margins)):
        # Intersect each margin segment with the late window and sum the time
        # spent within one goal there.
        s, e = max(times[i], lo), min(times[i + 1], end)
        if e > s and abs(margins[i]) <= 1:
            within += e - s
    # 1e-9 guards the divide when the late window has zero length (end <= 0 is
    # already handled above, so this only bites degenerate inputs).
    return {"alive_until": float(last_alive / end),
            "late_alive_30": float(within / max(end - lo, 1e-9))}


@measure("alive_until", tier="core")
def alive_until(ctx: MatchContext) -> float:
    """When the match stopped being a one-score game.

    Args:
        ctx: The match context.

    Returns:
        A value in [0, 1]: the fraction of the match elapsed before the goal
        margin moved beyond one for good (1.0 if the match ends a one-score
        game, so every extra-time match and every completed comeback scores 1).
    """
    return _aliveness(ctx)["alive_until"]


@measure("late_alive_30", tier="core")
def late_alive_30(ctx: MatchContext) -> float:
    """Whether the ending of the match was contested.

    Args:
        ctx: The match context.

    Returns:
        A value in [0, 1]: the share of the final 30 minutes of playing time
        (the window [90', 120'] when extra time was played, else [60', 90'])
        spent with the score margin within one goal.
    """
    return _aliveness(ctx)["late_alive_30"]
