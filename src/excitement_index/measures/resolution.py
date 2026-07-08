"""Resolution measures — how decisively and dramatically the match was settled.

Goals are not all weighted equally: each is scored by the win-probability swing
it caused, so an 89th-minute equalizer in a tied match counts heavily while a
sixth goal in a rout barely registers. The family also carries a plain
final-goals count and a kick-by-kick shootout drama score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..wp import per_shot_leverage
from .registry import MatchContext, measure


def _shot_lev(ctx: MatchContext) -> pd.DataFrame:
    """The per-shot leverage table, memoized on the match context.

    Args:
        ctx: The match context; uses ``ctx.ev``, team names, ``ctx.end`` and the
            pregame priors, and reads/writes ``ctx.cache["shot_lev"]``.

    Returns:
        The per-shot leverage table (see ``wp.per_shot_leverage``), computed once
        per match and shared with the chances and timing families through the
        context cache.
    """
    if "shot_lev" not in ctx.cache:
        ctx.cache["shot_lev"] = per_shot_leverage(
            ctx.ev, home=ctx.home, away=ctx.away, end=ctx.end,
            prior_home=ctx.prior_home, prior_away=ctx.prior_away)
    return ctx.cache["shot_lev"]


@measure("resolution_leverage")
def resolution_leverage(ctx: MatchContext) -> float:
    """Realized win-probability payoff of the goals that were actually scored.

    Args:
        ctx: The match context; uses the cached per-shot leverage frame.

    Returns:
        The sum, over shots that SCORED, of the pure counterfactual probability
        swing (the ``swing`` column — the same per-shot Skellam machinery as
        chance leverage, but WITHOUT the xG weight). An 89' equalizer contributes
        far more than a sixth goal in a rout. Own goals are excluded because they
        are not shot events. Returns ``nan`` when there are no shots.
    """
    lev = _shot_lev(ctx)
    if lev.empty:
        return float(np.nan)
    # Sum the WP swing only over shots that were goals.
    return float(lev.loc[lev["is_goal"], "swing"].sum())


@measure("total_goals")
def total_goals(ctx: MatchContext) -> float:
    """Combined final-score goals for both teams.

    Args:
        ctx: The match context; uses ``ctx.row`` (fixture-sheet row, may be None)
            and ``ctx.ev`` as a fallback.

    Returns:
        The full-time (or after-extra-time) goal total for both teams; shootout
        kicks are not counted as goals. Prefers the fixture-sheet score and falls
        back to counting goal events when no score is available.

    The ``row.get("score_home", row.get("home_score"))`` form is schema
    tolerance: the fixture sheet may name the column ``score_home`` or
    ``home_score``. Note the gotcha — if ``score_home`` is PRESENT but NaN,
    ``.get`` returns that NaN rather than trying the ``home_score`` fallback, and
    the ``pd.isna`` guard below then sends us to the event-count path. That is
    safe (we still return a goal total) but it does not recover a value from the
    alternate column in that particular case.
    """
    row = ctx.row
    if row is not None:
        # Accept either column-naming convention for the home/away score.
        sh = row.get("score_home", row.get("home_score"))
        sa = row.get("score_away", row.get("away_score"))
        # Only trust the sheet when both scores are present and non-NaN.
        if sh is not None and sa is not None and not (pd.isna(sh) or pd.isna(sa)):
            return float(int(sh) + int(sa))
    from ..clock import is_goal
    # No usable fixture score: count goal events (shot-goals plus own goals).
    return float(is_goal(ctx.ev).sum())


@measure("shootout_drama")
def shootout_drama(ctx: MatchContext) -> float:
    """Kick-by-kick drama of a penalty shootout.

    Args:
        ctx: The match context; uses ``ctx.events_all`` (all events, including the
            shootout, which is period 5).

    Returns:
        The number of shootout kicks that did NOT score, plus 0.5 for each kick
        beyond the standard ten (the sudden-death overrun). Returns 0.0 when the
        match had no shootout. This distinguishes a 12-kick, 5-miss epic from a
        routine 5-for-5.

    Convention: shootout kicks are the ``Shot`` events in period 5. The standard
    ten is the five kicks each side takes in the ordinary phase; anything past
    that is sudden death and earns the 0.5-per-kick overrun weight.
    """
    events = ctx.events_all
    # Without period/type columns we cannot identify shootout kicks at all.
    if "period" not in events.columns or "type" not in events.columns:
        return 0.0
    # Period 5 shot events are the shootout kicks.
    p5 = events[(events["period"] == 5) & (events["type"] == "Shot")]
    if p5.empty:
        return 0.0
    if "shot_outcome" in p5.columns:
        # A miss is any kick whose outcome is not a goal.
        misses = int((p5["shot_outcome"] != "Goal").sum())
    else:
        misses = 0
    # Kicks beyond the standard 10 (5 per side) are the sudden-death overrun.
    overrun = max(0, len(p5) - 10)
    return float(misses + 0.5 * overrun)
