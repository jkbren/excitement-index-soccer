"""Back-and-forth — the drama family read off the win-probability curve.

These are the index's outcome-dynamics measures: how far the probability
needle traveled (``gei``), how long the result stayed genuinely open
(``suspense``), the single biggest moment (``peak_tension``), scoreline and
xG lead flips, the largest fightback, and the share of the match spent a
one-score game. All but the last are thin reductions of the shared
goals-only Skellam curve in :mod:`excitement_index.wp`.
"""
from __future__ import annotations

from .. import wp as _wp
from ..clock import margin_walk
from .registry import measure


@measure("gei", tier="core")
def gei(ctx) -> float:
    """Game Excitement Index: total variation of the full 3-outcome WP vector
    summed over all curve steps — the total distance the needle traveled."""
    return _wp.gei_tv(ctx.wp)


@measure("suspense", tier="core")
def suspense(ctx) -> float:
    """Time-averaged normalized entropy of (p_home, p_draw, p_away): how long
    the result stayed genuinely uncertain. A knife-edge draw scores near 1; a
    foregone conclusion near 0."""
    return _wp.entropy_area(ctx.wp)


@measure("peak_tension", tier="core")
def peak_tension(ctx) -> float:
    """Largest single-step WP movement (in the goals-only curve, effectively
    the biggest goal's probability swing)."""
    steps = _wp.tv_steps(ctx.wp)
    return float(steps.max()) if len(steps) else 0.0


@measure("lead_changes", tier="core")
def lead_changes(ctx) -> float:
    """Number of times the scoreline leader changed (sign flips of the running
    goal margin; passing through level does not count by itself)."""
    return float(_wp.lead_changes_from_curve(ctx.wp))


@measure("comeback_magnitude", tier="core")
def comeback_magnitude(ctx) -> float:
    """Largest WP recovery achieved by a side while or after trailing on the
    scoreline; scoreline-gating prevents phantom fightbacks from probability
    noise."""
    return _wp.comeback_magnitude_from_curve(ctx.wp)


@measure("xg_lead_changes", tier="core")
def xg_lead_changes(ctx) -> float:
    """Crossings of the cumulative non-penalty-xG race between the teams
    ("who deserved to lead" flips)."""
    return float(_wp.xg_lead_changes_from_curve(ctx.wp))


@measure("time_within_one_goal", tier="core")
def time_within_one_goal(ctx) -> float:
    """Fraction of match minutes with goal margin <= 1."""
    if ctx.ev.empty:
        return float("nan")
    end = ctx.end
    times, margins = margin_walk(ctx.ev, ctx.home, ctx.away, end)
    within = sum(max(times[i + 1] - times[i], 0.0)
                 for i in range(len(margins)) if abs(margins[i]) <= 1)
    return float(within / end) if end > 0 else float("nan")
