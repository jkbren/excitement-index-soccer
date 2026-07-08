"""Back-and-forth measures — outcome dynamics read off the win-probability curve.

This family scores how the result moved over the course of a match: how far the
probability needle traveled (``gei``), how long the result stayed genuinely open
(``suspense``), the single biggest moment (``peak_tension``), scoreline and xG
lead flips, the largest fightback, and the share of the match spent as a
one-score game.

Most measures here are thin reductions of the shared goals-only Skellam
win-probability curve produced in :mod:`excitement_index.wp`; this module
forwards ``ctx.wp`` into the corresponding ``wp`` reducer, so their numeric
correctness lives in ``wp.py`` (the ``gei_tv``, ``entropy_area``, ``tv_steps``,
``lead_changes_from_curve``, ``comeback_magnitude_from_curve`` and
``xg_lead_changes_from_curve`` functions). Two measures do their own reduction
here: ``peak_tension`` takes the max over ``wp.tv_steps`` (with an empty-curve
guard), and ``time_within_one_goal`` is computed directly off the scoreline
margin walk from :mod:`excitement_index.clock`.
"""
from __future__ import annotations

from .. import wp as _wp
from ..clock import margin_walk
from .registry import measure


@measure("gei", tier="core")
def gei(ctx) -> float:
    """Game Excitement Index: the total distance the win-probability needle traveled.

    Args:
        ctx: The match context; only ``ctx.wp`` (the H/D/A win-probability curve
            DataFrame) is used.

    Returns:
        Total variation of the full 3-outcome (home, draw, away) win-probability
        vector summed over every curve step. Larger means a more volatile match.
    """
    return _wp.gei_tv(ctx.wp)


@measure("suspense", tier="core")
def suspense(ctx) -> float:
    """How long the result stayed genuinely uncertain, time-averaged.

    Args:
        ctx: The match context; only ``ctx.wp`` is used.

    Returns:
        Time-averaged normalized entropy of the (p_home, p_draw, p_away) curve,
        in [0, 1]. A knife-edge three-way toss-up scores near 1; a foregone
        conclusion near 0.
    """
    return _wp.entropy_area(ctx.wp)


@measure("peak_tension", tier="core")
def peak_tension(ctx) -> float:
    """The single biggest probability swing in the match.

    Args:
        ctx: The match context; only ``ctx.wp`` is used.

    Returns:
        The largest single-step win-probability movement (total variation of one
        curve step). In the goals-only curve this is effectively the biggest
        goal's probability swing. Returns 0.0 when the curve has no steps.
    """
    steps = _wp.tv_steps(ctx.wp)
    # Guard against an empty step array (a match with no curve movement).
    return float(steps.max()) if len(steps) else 0.0


@measure("lead_changes", tier="core")
def lead_changes(ctx) -> float:
    """How many times the scoreline leader changed.

    Args:
        ctx: The match context; only ``ctx.wp`` is used.

    Returns:
        Count of sign flips of the running goal margin. Passing through level
        (a margin of 0) does not count by itself; only crossing from one side
        leading to the other side leading is a lead change.
    """
    return float(_wp.lead_changes_from_curve(ctx.wp))


@measure("comeback_magnitude", tier="core")
def comeback_magnitude(ctx) -> float:
    """The largest fightback in the match.

    Args:
        ctx: The match context; only ``ctx.wp`` is used.

    Returns:
        The largest win-probability recovery achieved by a side while or after
        it was trailing on the scoreline. The recovery is gated on the team
        actually being behind on goals, which prevents phantom fightbacks driven
        by probability noise while the score is level.
    """
    return _wp.comeback_magnitude_from_curve(ctx.wp)


@measure("xg_lead_changes", tier="core")
def xg_lead_changes(ctx) -> float:
    """How many times "who deserved to lead" flipped.

    Args:
        ctx: The match context; only ``ctx.wp`` is used.

    Returns:
        Count of crossings of the cumulative non-penalty-xG race between the two
        teams — each crossing is a flip in which side had accumulated more
        expected goals up to that point.
    """
    return float(_wp.xg_lead_changes_from_curve(ctx.wp))


@measure("time_within_one_goal", tier="core")
def time_within_one_goal(ctx) -> float:
    """Fraction of the match played with the score within one goal.

    Args:
        ctx: The match context. Uses ``ctx.ev`` (playable events), ``ctx.home``
            / ``ctx.away`` (team names) and ``ctx.end`` (match end time, minutes).

    Returns:
        Fraction in [0, 1] of match minutes during which the running goal margin
        was at most one (a one-score game). Returns ``nan`` when there are no
        events, or when ``end`` is not positive (no measurable time span).

    ``margin_walk`` returns a step (segment) representation: ``times`` has one
    more element than ``margins``, so ``margins[i]`` is the margin held over the
    segment ``[times[i], times[i + 1]]`` and ``times[i + 1]`` is always in range.
    A future edit to ``margin_walk`` that broke that invariant would introduce a
    silent off-by-one here.
    """
    if ctx.ev.empty:
        return float("nan")
    end = ctx.end
    # times[i+1] is the end of the segment over which margins[i] held.
    times, margins = margin_walk(ctx.ev, ctx.home, ctx.away, end)
    # Sum segment durations (clamped at 0 for safety) where the margin was <= 1.
    within = sum(max(times[i + 1] - times[i], 0.0)
                 for i in range(len(margins)) if abs(margins[i]) <= 1)
    # Divide by total time; guard against a non-positive span, which returns nan.
    return float(within / end) if end > 0 else float("nan")
