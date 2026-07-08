"""Upset measures — the favourite failing, and how hard the result was earned.

All three measures need an Elo table and so sit in the ``context`` tier: the
pre-match favourite and its win probability come from ``ctx.elo_ctx`` (the
pregame Elo -> Skellam outcome probabilities), and each returns ``nan`` when
ratings are unavailable. The final result is read from the fixture-sheet row
(``ctx.row``), never from the event feed, so a shootout does not change the
recorded "result" — this matches the private reference implementation.
"""
from __future__ import annotations

import numpy as np

from ..wp import trapezoid
from .registry import measure


def _result(row) -> str:
    """The final result label from the fixture-sheet score.

    Args:
        row: The fixture-sheet row; uses integer ``score_home`` / ``score_away``.

    Returns:
        ``"home"``, ``"away"`` or ``"draw"`` from the full-time score (shootouts
        are not reflected here — the fixture score is the regulation/ET result).
    """
    sh, sa = int(row["score_home"]), int(row["score_away"])
    return "home" if sh > sa else ("away" if sa > sh else "draw")


def _favourite(elo_ctx: dict) -> str:
    """The pre-match favourite from the pregame Elo probabilities.

    Args:
        elo_ctx: The pregame Elo -> Skellam probability dict with ``p_home`` and
            ``p_away``.

    Returns:
        ``"home"`` or ``"away"`` — whichever has the higher pregame win
        probability. Exact ties are resolved to ``"home"`` by convention.
    """
    return "home" if elo_ctx["p_home"] >= elo_ctx["p_away"] else "away"


@measure("upset", tier="context")
def upset(ctx) -> float:
    """How large an upset the final result was, in favourite-probability units.

    Args:
        ctx: The match context; uses ``ctx.elo_ctx`` (pregame probabilities) and
            ``ctx.row`` (fixture-sheet score).

    Returns:
        0.0 if the pre-match favourite won; otherwise the favourite's pre-match
        win probability, so a heavier favourite failing to win registers a larger
        upset. A favourite held to a draw counts as an upset (a draw is not a
        favourite win). Returns ``nan`` when Elo context or the fixture row is
        missing.
    """
    if ctx.elo_ctx is None or ctx.row is None:
        return float(np.nan)
    fav = _favourite(ctx.elo_ctx)
    # Favourite won -> no upset; otherwise the size of the upset is p(favourite).
    return 0.0 if _result(ctx.row) == fav else float(ctx.elo_ctx["p_" + fav])


@measure("shock", tier="context")
def shock(ctx) -> float:
    """Time-averaged distance of the live WP state from the pre-match prior.

    This captures sustained defiance of the pregame script over the match, rather
    than just the final result.

    Args:
        ctx: The match context; uses ``ctx.elo_ctx`` (pregame prior) and
            ``ctx.wp`` (the live H/D/A curve).

    Returns:
        The total-variation distance ``0.5 * ||p(t) - p0||_1`` between the live
        (home, draw, away) curve and the pregame prior vector ``p0``, averaged
        over match time (trapezoidal integral divided by the time span). This is
        Buraimo et al.'s shock operationalized on our curve. Returns ``nan`` with
        no Elo context, 0.0 for an empty curve, and the plain mean when there is
        only one time point (no span to integrate over).
    """
    if ctx.elo_ctx is None:
        return float(np.nan)
    curve = ctx.wp
    if curve is None or curve.empty:
        return 0.0
    # p0 is the pregame prior over the three outcomes.
    p0 = np.asarray([ctx.elo_ctx["p_home"], ctx.elo_ctx["p_draw"],
                     ctx.elo_ctx["p_away"]], float)
    t = curve["_t"].to_numpy(float)
    p3 = curve[["p_home", "p_draw", "p_away"]].to_numpy(float)
    # 0.5 * L1 distance = total-variation distance between the live and prior distributions.
    sh = 0.5 * np.abs(p3 - p0).sum(axis=1)
    # With a single sample there is nothing to integrate; fall back to the mean.
    if len(t) < 2:
        return float(sh.mean())
    span = float(t[-1] - t[0])
    # Time-average via trapezoidal integral / span; guard a zero-length span.
    return float(trapezoid(sh, t) / span) if span > 0 else float(sh.mean())


@measure("underdog_defiance", tier="context")
def underdog_defiance(ctx) -> float:
    """An upset weighted by how much pressure the underdog withstood.

    Args:
        ctx: The match context; uses ``ctx.elo_ctx``, ``ctx.row``, team names and
            ``ctx.shots``.

    Returns:
        ``upset(ctx)`` multiplied by the favourite's non-penalty xG, so an
        underdog result earned while withstanding a barrage scores higher than
        one earned against a toothless favourite. Returns ``nan`` when Elo context
        or the fixture row is missing.
    """
    if ctx.elo_ctx is None or ctx.row is None:
        return float(np.nan)
    # Resolve which side ("home"/"away") the favourite is to its team name.
    fav_team = ctx.home if _favourite(ctx.elo_ctx) == "home" else ctx.away
    shots = ctx.shots
    if len(shots):
        # Fill missing xG with 0.0 first so the all-True fallback mask below is valid.
        sx = shots["shot_statsbomb_xg"].fillna(0.0)
        if "shot_type" in shots.columns:
            # Non-penalty shots only, so penalty xG does not inflate the barrage measure.
            np_mask = shots["shot_type"] != "Penalty"
        else:
            # No shot_type column: treat every shot as non-penalty. sx == sx is all True
            # only because sx was already fillna(0.0)'d (NaN == NaN would be False).
            np_mask = sx == sx  # all True
        # Favourite's non-penalty xG total.
        fav_npxg = float(sx[np_mask & (shots["team"] == fav_team)].sum())
    else:
        fav_npxg = 0.0
    return float(upset(ctx)) * fav_npxg
