"""Goalkeeping & peril — saves, shot-stopping above expectation, one-on-ones,
and woodwork near-misses.

A great save is a goal that almost happened; a shot rattling the post is a
scream held mid-breath. These measures count the moments where the keeper (or
the frame of the goal) stood between the match and a different scoreline.

Conventions (shared with the whole index): ``keeper_saves`` counts outcome
*Saved* only — *Saved To Post* is counted under ``goal_line_peril``; and
``psxg_minus_goals`` uses StatsBomb's pre-shot xG on on-target shots (a
pre-shot approximation of post-shot save performance).
"""
from __future__ import annotations

from ..clock import SOT_OUTCOMES, is_goal
from .registry import MatchContext, measure


def _total_goals(ctx: MatchContext) -> int:
    """Total goals scored (both teams). Prefers the fixture-sheet row's final
    score (``score_home``/``score_away``, or ``home_score``/``away_score``);
    without a row, falls back to counting goal events by team."""
    row = ctx.row
    if row is not None:
        sh = int(row.get("score_home", row.get("home_score", 0)))
        sa = int(row.get("score_away", row.get("away_score", 0)))
        return sh + sa
    gl = ctx.ev[is_goal(ctx.ev)]
    sh = int((gl["team"] == ctx.home).sum())
    sa = int((gl["team"] == ctx.away).sum())
    return sh + sa


@measure("keeper_saves", tier="core")
def keeper_saves(ctx: MatchContext) -> float:
    """Shots with outcome *Saved*."""
    shots = ctx.shots
    if not len(shots):
        return 0.0
    return float((shots["shot_outcome"] == "Saved").sum())


@measure("psxg_minus_goals", tier="core")
def psxg_minus_goals(ctx: MatchContext) -> float:
    """On-target xG faced minus goals conceded (both teams pooled):
    shot-stopping above expectation.

    On-target shots are outcomes *Goal*, *Saved*, *Saved To Post*; goals
    conceded is the total final score. Uses StatsBomb's pre-shot xG (a
    pre-shot approximation of post-shot save performance)."""
    shots = ctx.shots
    ont = shots[shots["shot_outcome"].isin(SOT_OUTCOMES)] if len(shots) else shots
    if not len(ont):
        return 0.0
    return float(ont["shot_statsbomb_xg"].fillna(0).sum() - _total_goals(ctx))


@measure("great_saves", tier="core")
def great_saves(ctx: MatchContext) -> float:
    """Saves of shots with xG >= 0.3."""
    shots = ctx.shots
    if not len(shots):
        return 0.0
    sx = shots["shot_statsbomb_xg"].fillna(0.0)
    return float(((shots["shot_outcome"] == "Saved") & (sx >= 0.3)).sum())


@measure("one_on_ones", tier="core")
def one_on_ones(ctx: MatchContext) -> float:
    """Shots flagged one-on-one with the keeper."""
    shots = ctx.shots
    if "shot_one_on_one" not in shots.columns or not len(shots):
        return 0.0
    return float((shots["shot_one_on_one"] == True).sum())  # noqa: E712


@measure("goal_line_peril", tier="core")
def goal_line_peril(ctx: MatchContext) -> float:
    """Shots hitting the woodwork or saved onto the post (outcomes *Post*,
    *Saved To Post*)."""
    shots = ctx.shots
    if not len(shots):
        return 0.0
    return float(shots["shot_outcome"].isin({"Post", "Saved To Post"}).sum())
