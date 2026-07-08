"""Goalkeeping and peril — saves, shot-stopping above expectation,
one-on-ones, and woodwork near-misses.

These measures count the moments where the keeper (or the frame of the goal)
stood between the match and a different scoreline: saves, shots stopped above
expectation, one-on-one chances, and shots that hit the woodwork.

Conventions (shared with the whole index): ``keeper_saves`` counts outcome
*Saved* only — *Saved To Post* is counted under ``goal_line_peril``; and
``psxg_minus_goals`` uses StatsBomb's pre-shot xG on on-target shots (a
pre-shot approximation of post-shot save performance).
"""
from __future__ import annotations

from ..clock import SOT_OUTCOMES, is_goal
from .registry import MatchContext, measure


def _total_goals(ctx: MatchContext) -> int:
    """Total goals scored across both teams.

    Args:
        ctx: The match context; ``ctx.row`` is the fixture-sheet row (may be
            None) and ``ctx.ev`` is the event feed.

    Returns:
        The combined final score as an int. Prefers the fixture-sheet row's
        score (``score_home``/``score_away``, or ``home_score``/``away_score``);
        without a row, falls back to counting goal events by team.

    Notes:
        ``row.get(..., row.get(...))`` evaluates the nested ``get`` eagerly, so
        the ``home_score``/``away_score`` fallback is looked up whether or not
        ``score_home`` is present — a lookup, not a branch. A row that carries
        neither score pair yields 0 (not the event-count fallback, which only
        fires when there is no row at all). The score fields are assumed
        integer-valued; ``int(NaN)`` would raise, so a row with a NaN score is
        not supported here.
    """
    row = ctx.row
    if row is not None:
        sh = int(row.get("score_home", row.get("home_score", 0)))
        sa = int(row.get("score_away", row.get("away_score", 0)))
        return sh + sa
    # No fixture row: count goal events per side and pool.
    gl = ctx.ev[is_goal(ctx.ev)]
    sh = int((gl["team"] == ctx.home).sum())
    sa = int((gl["team"] == ctx.away).sum())
    return sh + sa


@measure("keeper_saves", tier="core")
def keeper_saves(ctx: MatchContext) -> float:
    """Count of shots with outcome *Saved*.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        The number of *Saved* shots as a float, or 0.0 when there are no
        shots. Assumes a non-empty shots feed carries a ``shot_outcome``
        column (a feed with shots but no ``shot_outcome`` would raise and be
        recorded as nan upstream).
    """
    shots = ctx.shots
    if not len(shots):
        return 0.0
    return float((shots["shot_outcome"] == "Saved").sum())


@measure("psxg_minus_goals", tier="core")
def psxg_minus_goals(ctx: MatchContext) -> float:
    """On-target xG faced minus goals conceded — shot-stopping above expectation.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        Summed xG of on-target shots minus the match's total goals, pooled
        across both teams, as a float; 0.0 when there are no on-target shots.
        On-target outcomes are *Goal*, *Saved*, *Saved To Post*
        (``SOT_OUTCOMES``); goals conceded is the total final score. Uses
        StatsBomb's pre-shot xG (a pre-shot approximation of post-shot save
        performance). Assumes a non-empty shots feed carries ``shot_outcome``.
    """
    shots = ctx.shots
    ont = shots[shots["shot_outcome"].isin(SOT_OUTCOMES)] if len(shots) else shots
    if not len(ont):
        return 0.0
    return float(ont["shot_statsbomb_xg"].fillna(0).sum() - _total_goals(ctx))


@measure("great_saves", tier="core")
def great_saves(ctx: MatchContext) -> float:
    """Count of saves of high-quality shots.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        The number of *Saved* shots with xG >= 0.3 as a float, or 0.0 when
        there are no shots. The 0.3 threshold marks a shot the keeper was more
        likely than not to concede. Assumes a non-empty shots feed carries
        ``shot_outcome``.
    """
    shots = ctx.shots
    if not len(shots):
        return 0.0
    sx = shots["shot_statsbomb_xg"].fillna(0.0)
    return float(((shots["shot_outcome"] == "Saved") & (sx >= 0.3)).sum())


@measure("one_on_ones", tier="core")
def one_on_ones(ctx: MatchContext) -> float:
    """Count of one-on-one chances against the keeper.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        The number of shots flagged one-on-one as a float, or 0.0 when the
        ``shot_one_on_one`` column is absent or there are no shots.
    """
    shots = ctx.shots
    if "shot_one_on_one" not in shots.columns or not len(shots):
        return 0.0
    return float((shots["shot_one_on_one"] == True).sum())  # noqa: E712


@measure("goal_line_peril", tier="core")
def goal_line_peril(ctx: MatchContext) -> float:
    """Count of woodwork near-misses.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        The number of shots hitting the woodwork or saved onto the post
        (outcomes *Post*, *Saved To Post*) as a float, or 0.0 when there are
        no shots. Assumes a non-empty shots feed carries ``shot_outcome``.
    """
    shots = ctx.shots
    if not len(shots):
        return 0.0
    return float(shots["shot_outcome"].isin({"Post", "Saved To Post"}).sum())
