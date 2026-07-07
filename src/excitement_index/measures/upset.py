"""Upset measures — the favourite failing, and how hard it was earned.

All three need an Elo table (tier ``context``): the pre-match favourite and
its win probability come from ``ctx.elo_ctx`` (the pregame Elo->Skellam
outcome probabilities); each returns ``nan`` when ratings are unavailable.
The final score is read from the fixture-sheet row (``ctx.row``), never from
the event feed, so shootouts don't change the "result" a la the private
implementation.
"""
from __future__ import annotations

import numpy as np

from .registry import measure


def _result(row) -> str:
    """"home"/"draw"/"away" from the fixture-sheet final score."""
    sh, sa = int(row["score_home"]), int(row["score_away"])
    return "home" if sh > sa else ("away" if sa > sh else "draw")


def _favourite(elo_ctx: dict) -> str:
    """The pre-match favourite: the higher of p_home / p_away (ties -> home)."""
    return "home" if elo_ctx["p_home"] >= elo_ctx["p_away"] else "away"


@measure("upset", tier="context")
def upset(ctx) -> float:
    """0 if the pre-match favourite won; otherwise the favourite's pre-match
    win probability (so a heavy favourite held to a draw registers a large
    upset)."""
    if ctx.elo_ctx is None or ctx.row is None:
        return float(np.nan)
    fav = _favourite(ctx.elo_ctx)
    return 0.0 if _result(ctx.row) == fav else float(ctx.elo_ctx["p_" + fav])


@measure("shock", tier="context")
def shock(ctx) -> float:
    """Time-averaged distance of the live WP state from the pre-match prior —
    sustained defiance of the script, not just the final result.

    Concretely: the total-variation distance 0.5 * ||p(t) - p0||_1 between the
    live (home, draw, away) curve and the pregame prior vector, averaged over
    match time (trapezoidal integral / time span). Buraimo et al.'s shock,
    operationalised on our curve.
    """
    if ctx.elo_ctx is None:
        return float(np.nan)
    curve = ctx.wp
    if curve is None or curve.empty:
        return 0.0
    p0 = np.asarray([ctx.elo_ctx["p_home"], ctx.elo_ctx["p_draw"],
                     ctx.elo_ctx["p_away"]], float)
    t = curve["_t"].to_numpy(float)
    p3 = curve[["p_home", "p_draw", "p_away"]].to_numpy(float)
    sh = 0.5 * np.abs(p3 - p0).sum(axis=1)
    if len(t) < 2:
        return float(sh.mean())
    span = float(t[-1] - t[0])
    return float(np.trapz(sh, t) / span) if span > 0 else float(sh.mean())


@measure("underdog_defiance", tier="context")
def underdog_defiance(ctx) -> float:
    """upset x the favourite's non-penalty xG: an underdog result earned while
    withstanding a barrage."""
    if ctx.elo_ctx is None or ctx.row is None:
        return float(np.nan)
    fav_team = ctx.home if _favourite(ctx.elo_ctx) == "home" else ctx.away
    shots = ctx.shots
    if len(shots):
        sx = shots["shot_statsbomb_xg"].fillna(0.0)
        if "shot_type" in shots.columns:
            np_mask = shots["shot_type"] != "Penalty"
        else:
            np_mask = sx == sx  # all True
        fav_npxg = float(sx[np_mask & (shots["team"] == fav_team)].sum())
    else:
        fav_npxg = 0.0
    return float(upset(ctx)) * fav_npxg
