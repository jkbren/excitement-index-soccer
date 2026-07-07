"""Timing measures — WHEN the drama arrives.

A 3-3 whose goals cluster in the last ten minutes is not the same match as a
3-3 settled by the hour mark. This family scores lateness (late goals, a late
winner), clustering (burstiness), the agony of big chances missed while the
match still hung in the balance, and the shape of the win-probability
trajectory (its best 10-minute spell and its final-15-minute movement).

Shared machinery: the per-shot leverage frame (``wp.per_shot_leverage``) is
computed once and cached under ``ctx.cache["shot_lev"]``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..clock import is_goal
from ..wp import best_window_tv, per_shot_leverage, tv_steps
from .registry import MatchContext, measure


def _goals(ctx: MatchContext) -> pd.DataFrame:
    """Goal rows (shot-goals plus own goals) among the playable events."""
    return ctx.ev[is_goal(ctx.ev)]


def _shot_lev(ctx: MatchContext) -> pd.DataFrame:
    """The cached per-shot leverage frame (see ``wp.per_shot_leverage``)."""
    if "shot_lev" not in ctx.cache:
        ctx.cache["shot_lev"] = per_shot_leverage(
            ctx.ev, home=ctx.home, away=ctx.away, end=ctx.end,
            prior_home=ctx.prior_home, prior_away=ctx.prior_away)
    return ctx.cache["shot_lev"]


@measure("late_goals", tier="core")
def late_goals(ctx: MatchContext) -> float:
    """Goals at >= 80', with an extra 0.5 weight for goals at >= 90' (including
    all extra-time goals)."""
    goals = _goals(ctx)
    late = goals[goals["_t"] >= 80]
    return float(len(late)) + 0.5 * float((late["_t"] >= 90).sum())


@measure("late_winner", tier="core")
def late_winner(ctx: MatchContext) -> float:
    """1 if the match's final goal came at >= 80' and changed who led (including
    breaking a deadlock)."""
    goals = _goals(ctx)
    if goals.empty:
        return 0.0
    seq, d = [], 0
    for _, g in goals.sort_values("_t").iterrows():
        d += 1 if g["team"] == ctx.home else (-1 if g["team"] == ctx.away else 0)
        seq.append((float(g["_t"]), d))
    last_t, last_d = seq[-1]
    prev_d = seq[-2][1] if len(seq) >= 2 else 0
    return 1.0 if (last_t >= 80 and np.sign(last_d) != np.sign(prev_d)) else 0.0


@measure("goal_burstiness", tier="core")
def goal_burstiness(ctx: MatchContext) -> float:
    """1 / mean gap (minutes) between consecutive goals, for matches with >= 2
    goals — goals in bursts."""
    gt = sorted(_goals(ctx)["_t"].tolist())
    if len(gt) < 2:
        return 0.0
    return float(1.0 / (np.mean(np.diff(gt)) + 1e-6))


@measure("big_chance_xg_missed", tier="core")
def big_chance_xg_missed(ctx: MatchContext) -> float:
    """Sum of xG over big chances (xG >= 0.25) that did not score. Penalty kicks
    are included."""
    lev = _shot_lev(ctx)
    if lev.empty:
        return float(np.nan)
    xg = lev["xg"].to_numpy(float)
    scored = lev["is_goal"].to_numpy(bool)
    return float(xg[(xg >= 0.25) & ~scored].sum())


@measure("leverage_missed_late", tier="core")
def leverage_missed_late(ctx: MatchContext) -> float:
    """Sum of xG over missed big chances (xG >= 0.25) at >= 75' with the score
    within one goal — the late agony specifically."""
    lev = _shot_lev(ctx)
    if lev.empty:
        return float(np.nan)
    xg = lev["xg"].to_numpy(float)
    scored = lev["is_goal"].to_numpy(bool)
    t = lev["t"].to_numpy(float)
    d = lev["d"].to_numpy(int)
    return float(xg[(xg >= 0.25) & ~scored & (t >= 75.0) & (np.abs(d) <= 1)].sum())


@measure("peak_window_tv", tier="core")
def peak_window_tv(ctx: MatchContext) -> float:
    """The best 10-minute window of accumulated WP movement (total-variation
    steps of the H/D/A curve)."""
    return best_window_tv(ctx.wp, minutes=10.0)


@measure("late_window_tv", tier="core")
def late_window_tv(ctx: MatchContext) -> float:
    """WP movement in the final 15 minutes of play: the sum of the curve's
    total-variation steps whose event time is at or after end − 15."""
    steps = tv_steps(ctx.wp)
    if not len(steps):
        return 0.0
    mid = ctx.wp["_t"].to_numpy(float)[1:]
    return float(steps[mid >= ctx.end - 15.0].sum())
