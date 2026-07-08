"""Timing measures — when in the match the drama arrives.

A 3-3 whose goals cluster in the last ten minutes scores differently from a 3-3
settled by the hour mark. This family scores lateness (late goals, a late
winner), clustering (goal burstiness), the size of big chances missed while the
match still hung in the balance, and the shape of the win-probability trajectory
(its best 10-minute spell and its final-15-minute movement).

Shared machinery: the per-shot leverage frame (``wp.per_shot_leverage``) is
computed once per match and cached under ``ctx.cache["shot_lev"]`` so the timing,
chances and resolution families all reuse the same table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..clock import is_goal
from ..wp import best_window_tv, per_shot_leverage, tv_steps
from .registry import MatchContext, measure


def _goals(ctx: MatchContext) -> pd.DataFrame:
    """Goal rows (shot-goals plus own goals) among the playable events.

    Args:
        ctx: The match context; uses ``ctx.ev`` (playable events).

    Returns:
        The subset of ``ctx.ev`` rows that are goals, per ``clock.is_goal``.
    """
    return ctx.ev[is_goal(ctx.ev)]


def _shot_lev(ctx: MatchContext) -> pd.DataFrame:
    """The cached per-shot leverage frame (see ``wp.per_shot_leverage``).

    Args:
        ctx: The match context; uses ``ctx.ev``, team names, ``ctx.end`` and the
            pregame priors, and reads/writes ``ctx.cache["shot_lev"]``.

    Returns:
        The per-shot leverage table, computed once and memoized on the context so
        the timing and resolution families share a single computation per match.
    """
    if "shot_lev" not in ctx.cache:
        ctx.cache["shot_lev"] = per_shot_leverage(
            ctx.ev, home=ctx.home, away=ctx.away, end=ctx.end,
            prior_home=ctx.prior_home, prior_away=ctx.prior_away)
    return ctx.cache["shot_lev"]


@measure("late_goals", tier="core")
def late_goals(ctx: MatchContext) -> float:
    """Weighted count of goals scored late in the match.

    Args:
        ctx: The match context; uses ``ctx.ev``.

    Returns:
        The number of goals at minute >= 80, plus an extra 0.5 for each goal at
        minute >= 90 (which includes all extra-time goals). So a 92' goal counts
        1.5 and an 82' goal counts 1.0.
    """
    goals = _goals(ctx)
    # 80' is the "late" threshold; 90'+ (stoppage/extra time) earns the 0.5 bonus.
    late = goals[goals["_t"] >= 80]
    return float(len(late)) + 0.5 * float((late["_t"] >= 90).sum())


@measure("late_winner", tier="core")
def late_winner(ctx: MatchContext) -> float:
    """Whether the match ended on a late, lead-changing goal.

    Args:
        ctx: The match context; uses ``ctx.ev`` and team names ``ctx.home`` /
            ``ctx.away``.

    Returns:
        1.0 if the match's final goal came at minute >= 80 and changed who led
        (including breaking a deadlock), else 0.0.
    """
    goals = _goals(ctx)
    if goals.empty:
        return 0.0
    # Walk goals in time order, tracking the running home-minus-away margin d.
    seq, d = [], 0
    for _, g in goals.sort_values("_t").iterrows():
        # +1 for a home goal, -1 for an away goal, 0 for anything else (e.g. OG credited oddly).
        d += 1 if g["team"] == ctx.home else (-1 if g["team"] == ctx.away else 0)
        seq.append((float(g["_t"]), d))
    last_t, last_d = seq[-1]
    # Margin before the final goal (0 if it was the only goal).
    prev_d = seq[-2][1] if len(seq) >= 2 else 0
    # Late winner = final goal at >= 80' that flipped the sign of the lead.
    return 1.0 if (last_t >= 80 and np.sign(last_d) != np.sign(prev_d)) else 0.0


@measure("goal_burstiness", tier="core")
def goal_burstiness(ctx: MatchContext) -> float:
    """How tightly the goals were clustered in time.

    Args:
        ctx: The match context; uses ``ctx.ev``.

    Returns:
        The reciprocal of the mean gap (in minutes) between consecutive goals,
        so goals arriving in a tight burst score high and evenly spaced goals
        score low. Returns 0.0 for matches with fewer than two goals (no gap to
        measure).
    """
    gt = sorted(_goals(ctx)["_t"].tolist())
    if len(gt) < 2:
        return 0.0
    # The 1e-6 (minutes) is a divide-by-zero guard for the degenerate case where
    # every goal shares the same recorded minute, making the mean gap exactly 0.
    return float(1.0 / (np.mean(np.diff(gt)) + 1e-6))


@measure("big_chance_xg_missed", tier="core")
def big_chance_xg_missed(ctx: MatchContext) -> float:
    """Total xG of the big chances that were missed.

    Args:
        ctx: The match context; uses the cached per-shot leverage frame.

    Returns:
        Sum of xG over big chances (xG >= 0.25) that did not score. Penalty kicks
        are included. Returns ``nan`` when there are no shots. The 0.25 xG
        threshold is the project's definition of a "big chance".
    """
    lev = _shot_lev(ctx)
    if lev.empty:
        return float(np.nan)
    xg = lev["xg"].to_numpy(float)
    scored = lev["is_goal"].to_numpy(bool)
    # Big chances (xG >= 0.25) that did NOT result in a goal.
    return float(xg[(xg >= 0.25) & ~scored].sum())


@measure("leverage_missed_late", tier="core")
def leverage_missed_late(ctx: MatchContext) -> float:
    """Total xG of big chances missed late in a still-close match.

    Args:
        ctx: The match context; uses the cached per-shot leverage frame.

    Returns:
        Sum of xG over missed big chances (xG >= 0.25) at minute >= 75 with the
        score within one goal (``|d| <= 1``) — the late agony specifically.
        Returns ``nan`` when there are no shots. As in ``big_chance_xg_missed``,
        the same xG >= 0.25 filter is applied with no penalty exclusion, so
        penalty kicks are included here too.
    """
    lev = _shot_lev(ctx)
    if lev.empty:
        return float(np.nan)
    xg = lev["xg"].to_numpy(float)
    scored = lev["is_goal"].to_numpy(bool)
    t = lev["t"].to_numpy(float)
    # d is the home-minus-away goal margin at the time of each shot.
    d = lev["d"].to_numpy(int)
    # Missed big chance, >= 75', while the game was still a one-score contest.
    return float(xg[(xg >= 0.25) & ~scored & (t >= 75.0) & (np.abs(d) <= 1)].sum())


@measure("peak_window_tv", tier="core")
def peak_window_tv(ctx: MatchContext) -> float:
    """The most eventful 10-minute stretch of the match.

    Args:
        ctx: The match context; uses ``ctx.wp``.

    Returns:
        The maximum accumulated win-probability movement (sum of total-variation
        steps of the H/D/A curve) over any 10-minute window. The 10.0-minute
        window is fixed by convention for this measure.
    """
    return best_window_tv(ctx.wp, minutes=10.0)


@measure("late_window_tv", tier="core")
def late_window_tv(ctx: MatchContext) -> float:
    """Win-probability movement in the final 15 minutes of play.

    Args:
        ctx: The match context; uses ``ctx.wp`` and ``ctx.end`` (match end time,
            minutes).

    Returns:
        The sum of the curve's total-variation steps whose event time is at or
        after ``end - 15``. Returns 0.0 when the curve has no steps. The 15.0 is
        the "final quarter-hour" window in minutes.

    Alignment convention: ``tv_steps`` returns one value per row-to-row
    transition (``len == len(wp) - 1``), and step ``i`` is the movement into the
    later row, so it maps to the endpoint time ``ctx.wp["_t"][1:][i]``. That is
    why ``mid`` drops the first time and the ``>= end - 15`` filter selects the
    steps that land in the final 15 minutes.
    """
    steps = tv_steps(ctx.wp)
    if not len(steps):
        return 0.0
    # mid[i] is the time of the LATER row of transition i (steps has len(wp)-1 entries).
    mid = ctx.wp["_t"].to_numpy(float)[1:]
    return float(steps[mid >= ctx.end - 15.0].sum())
