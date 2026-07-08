"""Pre-match measures — how open the matchup looked on paper.

Both measures here need an Elo table (tier ``context``): they read the
pre-computed ``ctx.elo_ctx`` dict and return ``nan`` when ratings are
unavailable (``ctx.elo_ctx is None``). The dict is populated upstream by the
pipeline's Elo->Skellam step and carries the two keys these measures index:

    "elo_gap": signed Elo points, home minus away (raw rating units).
    "entropy": normalized Shannon entropy of the pregame (home, draw, away)
               outcome probabilities, in [0, 1].

Both keys are hard-coded below; a typo in either would raise a ``KeyError`` that
``compute_all`` swallows into ``nan``, so keep these names in sync with whoever
builds ``elo_ctx``.
"""
from __future__ import annotations

import numpy as np

from .registry import measure


@measure("neg_ranking_gap", tier="context")
def neg_ranking_gap(ctx) -> float:
    """Negated absolute Elo gap — closeness of the matchup on paper.

    Args:
        ctx: Match context; reads ``ctx.elo_ctx["elo_gap"]`` (signed Elo points).

    Returns:
        ``-|Elo_home - Elo_away|`` in raw Elo points, or ``nan`` when Elo ratings
        are unavailable. Negated so that a smaller gap (more evenly matched) is
        the larger, more-exciting value.
    """
    if ctx.elo_ctx is None:
        return float(np.nan)
    return float(-abs(float(ctx.elo_ctx["elo_gap"])))


@measure("prematch_openness", tier="context")
def prematch_openness(ctx) -> float:
    """Normalized entropy of the pregame outcome distribution.

    Args:
        ctx: Match context; reads ``ctx.elo_ctx["entropy"]``.

    Returns:
        The normalized Shannon entropy of the pre-match (home, draw, away)
        probabilities from the Elo-Skellam model, in [0, 1] where 1 is maximal
        pre-match uncertainty; ``nan`` when Elo ratings are unavailable.
    """
    if ctx.elo_ctx is None:
        return float(np.nan)
    return float(ctx.elo_ctx["entropy"])
