"""Controversy — cards, dismissals, and penalty kicks.

Discipline cuts both ways for a spectator: a booking-strewn slog drags a match
down (``cards`` and ``red_card`` enter the composite negatively), while a
penalty is a moment of pure concentrated jeopardy. Card counts prefer the
curated fixture-sheet numbers and fall back to the event feed (pooling
foul-committed and bad-behaviour cards) when the sheet has none — e.g. a
freshly-onboarded match.
"""
from __future__ import annotations

import pandas as pd

from .registry import MatchContext, measure


def _card_counts(ctx: MatchContext):
    """(yellows, reds) across both teams. Curated fixture-sheet counts when the
    row carries them (``yellows_home`` present and non-null); otherwise derived
    from the events by pooling ``foul_committed_card`` and
    ``bad_behaviour_card``, where a red is *Red Card* or *Second Yellow*."""
    if "card_counts" in ctx.cache:
        return ctx.cache["card_counts"]
    row = ctx.row
    if row is not None and pd.notna(row.get("yellows_home")):
        def ci(v):
            return int(v) if pd.notna(v) else 0          # curated counts, NaN-safe
        yel = ci(row.get("yellows_home")) + ci(row.get("yellows_away"))
        red = ci(row.get("reds_home")) + ci(row.get("reds_away"))
    else:
        ev = ctx.ev
        cc = ev["foul_committed_card"] if "foul_committed_card" in ev.columns else pd.Series(dtype=object)
        bc = ev["bad_behaviour_card"] if "bad_behaviour_card" in ev.columns else pd.Series(dtype=object)
        cards_pool = pd.concat([cc, bc])
        yel = int((cards_pool == "Yellow Card").sum())
        red = int(cards_pool.isin({"Red Card", "Second Yellow"}).sum())
    ctx.cache["card_counts"] = (yel, red)
    return yel, red


@measure("cards", tier="core")
def cards(ctx: MatchContext) -> float:
    """Yellow cards + 3 x red cards, both teams (curated match-sheet counts,
    with an event-derived fallback)."""
    yel, red = _card_counts(ctx)
    return float(yel + 3 * red)


@measure("red_card", tier="core")
def red_card(ctx: MatchContext) -> float:
    """1 if any red card (straight or second yellow) was shown."""
    _, red = _card_counts(ctx)
    return 1.0 if red > 0 else 0.0


@measure("penalties", tier="core")
def penalties(ctx: MatchContext) -> float:
    """In-match penalty kicks awarded (scored or missed; shootout kicks
    excluded — events are pre-filtered to playable periods)."""
    shots = ctx.shots
    if not len(shots) or "shot_type" not in shots.columns:
        return 0.0
    return float((shots["shot_type"] == "Penalty").sum())
