"""Resolution — the thesis's own family: tension has to pay off.

Goals are not all equal here: each is weighted by the win-probability swing it
caused, so an 89th-minute equalizer in a tied match counts enormously while a
sixth goal in a rout barely registers. Shootouts are scored kick by kick."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..wp import per_shot_leverage
from .registry import MatchContext, measure


def _shot_lev(ctx: MatchContext) -> pd.DataFrame:
    """The per-shot leverage table, computed once per match and shared with the
    chances/timing families via the context cache."""
    if "shot_lev" not in ctx.cache:
        ctx.cache["shot_lev"] = per_shot_leverage(
            ctx.ev, home=ctx.home, away=ctx.away, end=ctx.end,
            prior_home=ctx.prior_home, prior_away=ctx.prior_away)
    return ctx.cache["shot_lev"]


@measure("resolution_leverage")
def resolution_leverage(ctx: MatchContext) -> float:
    """Realized win-probability payoff of the goals that landed: the sum, over
    shots that SCORED, of the pure counterfactual probability swing (the same
    per-shot Skellam machinery as chance leverage, WITHOUT the xG weight).
    An 89' equalizer >> a sixth in a rout. Own goals are excluded (not shots)."""
    lev = _shot_lev(ctx)
    if lev.empty:
        return float(np.nan)
    return float(lev.loc[lev["is_goal"], "swing"].sum())


@measure("total_goals")
def total_goals(ctx: MatchContext) -> float:
    """Final-score goals, both teams (full time or after extra time; shootout
    kicks are not goals). Prefers the fixture-sheet score; falls back to
    counting goal events when no score is available."""
    row = ctx.row
    if row is not None:
        sh = row.get("score_home", row.get("home_score"))
        sa = row.get("score_away", row.get("away_score"))
        if sh is not None and sa is not None and not (pd.isna(sh) or pd.isna(sa)):
            return float(int(sh) + int(sa))
    from ..clock import is_goal
    return float(is_goal(ctx.ev).sum())


@measure("shootout_drama")
def shootout_drama(ctx: MatchContext) -> float:
    """Kick-by-kick shootout drama from period-5 shot events: the number of
    kicks that did NOT score, plus 0.5 for each kick beyond the standard ten
    (the sudden-death overrun). 0.0 when the match had no shootout — a 12-kick,
    5-miss epic is not the same event as a routine 5-for-5."""
    events = ctx.events_all
    if "period" not in events.columns or "type" not in events.columns:
        return 0.0
    p5 = events[(events["period"] == 5) & (events["type"] == "Shot")]
    if p5.empty:
        return 0.0
    if "shot_outcome" in p5.columns:
        misses = int((p5["shot_outcome"] != "Goal").sum())
    else:
        misses = 0
    overrun = max(0, len(p5) - 10)
    return float(misses + 0.5 * overrun)
