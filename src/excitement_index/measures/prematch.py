"""Pre-match measures — how open the matchup looked on paper.

Both need an Elo table (tier ``context``): they read the pre-computed
``ctx.elo_ctx`` dict (the pregame Elo->Skellam outcome probabilities plus the
raw Elo gap) and return ``nan`` when ratings are unavailable.
"""
from __future__ import annotations

import numpy as np

from .registry import measure


@measure("neg_ranking_gap", tier="context")
def neg_ranking_gap(ctx) -> float:
    """-|Elo_home - Elo_away| (raw Elo points): closeness of the matchup on
    paper."""
    if ctx.elo_ctx is None:
        return float(np.nan)
    return float(-abs(float(ctx.elo_ctx["elo_gap"])))


@measure("prematch_openness", tier="context")
def prematch_openness(ctx) -> float:
    """Normalized Shannon entropy of the pre-match (home, draw, away)
    probabilities from the Elo-Skellam model; 1 = maximal pre-match
    uncertainty."""
    if ctx.elo_ctx is None:
        return float(np.nan)
    return float(ctx.elo_ctx["entropy"])
