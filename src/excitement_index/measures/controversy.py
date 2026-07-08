"""Controversy — cards, dismissals, and penalty kicks.

Discipline cuts both ways for a spectator: a booking-strewn slog drags a match
down (``cards`` and ``red_card`` enter the composite negatively), while a
penalty is a moment of pure concentrated jeopardy. Card counts prefer the
curated fixture-sheet numbers and fall back to the event feed (pooling
foul-committed and bad-behaviour cards) when the sheet has none — e.g. a
freshly-onboarded match.

The event-feed fallback keys on the exact StatsBomb card vocabulary strings
``"Yellow Card"``, ``"Red Card"``, and ``"Second Yellow"``. If that upstream
vocabulary ever changes, the fallback counts silently drop to zero (and the
``nan``/zero would be masked by ``compute_all``), so these literals are the
schema contract for this module.
"""
from __future__ import annotations

import pandas as pd

from .registry import MatchContext, measure


def _card_counts(ctx: MatchContext):
    """Total (yellows, reds) across both teams for one match.

    Args:
        ctx: Match context; reads the fixture-sheet ``ctx.row`` card columns
            (``yellows_home/away``, ``reds_home/away``) when present, otherwise
            the event feed ``ctx.ev``. Memoized in ``ctx.cache["card_counts"]``.

    Returns:
        A ``(yellows, reds)`` tuple of ints summed over both teams.

    Curated counts are used when the row carries them (``yellows_home`` present
    and non-null); otherwise the counts are derived from the events by pooling
    the ``foul_committed_card`` and ``bad_behaviour_card`` columns. In the
    event-derived path a *Second Yellow* is counted as a red — and the booking
    that triggered it is a separate *Yellow Card* event — so a sent-off player
    contributes 1 yellow + 1 red.
    """
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
        # Cards can appear on either column; missing columns -> empty series so
        # the pooled count is a well-defined zero rather than a KeyError.
        cc = ev["foul_committed_card"] if "foul_committed_card" in ev.columns else pd.Series(dtype=object)
        bc = ev["bad_behaviour_card"] if "bad_behaviour_card" in ev.columns else pd.Series(dtype=object)
        cards_pool = pd.concat([cc, bc])
        # Exact StatsBomb card strings — see the module docstring schema note.
        yel = int((cards_pool == "Yellow Card").sum())
        red = int(cards_pool.isin({"Red Card", "Second Yellow"}).sum())
    ctx.cache["card_counts"] = (yel, red)
    return yel, red


@measure("cards", tier="core")
def cards(ctx: MatchContext) -> float:
    """Card load — yellows plus weighted reds, both teams.

    Args:
        ctx: Match context (see :func:`_card_counts` for the count source).

    Returns:
        ``yellows + 3 * reds`` as a float, using curated match-sheet counts with
        an event-derived fallback.

    A red is weighted 3x a yellow: the factor is a hand-set editorial weight
    (chosen so a dismissal dominates ordinary bookings, not fit from data) and
    is part of the frozen reference implementation. Because a second-yellow
    dismissal is counted as 1 yellow + 1 red (see :func:`_card_counts`), a
    sent-off player contributes 1 + 3 = 4 to this total.
    """
    yel, red = _card_counts(ctx)
    return float(yel + 3 * red)


@measure("red_card", tier="core")
def red_card(ctx: MatchContext) -> float:
    """Flag whether any player was sent off.

    Args:
        ctx: Match context (see :func:`_card_counts`).

    Returns:
        1.0 if at least one red card (straight or second yellow) was shown,
        else 0.0.
    """
    _, red = _card_counts(ctx)
    return 1.0 if red > 0 else 0.0


@measure("penalties", tier="core")
def penalties(ctx: MatchContext) -> float:
    """Count in-match penalty kicks awarded.

    Args:
        ctx: Match context; reads ``ctx.shots`` and its ``shot_type`` column.

    Returns:
        The number of in-match penalty kicks (scored or missed) as a float, or
        0.0 when there are no shots or no ``shot_type`` column. Shootout kicks
        are excluded because ``ctx.shots`` is pre-filtered to the playable
        periods; a penalty kick is a set piece, not open play.
    """
    shots = ctx.shots
    if not len(shots) or "shot_type" not in shots.columns:
        return 0.0
    return float((shots["shot_type"] == "Penalty").sum())
