"""The measure registry — how measures plug into the pipeline.

A *measure* is a function that reduces one match to one number. Registering it
is one decorator:

    from excitement_index.measures.registry import measure

    @measure("woodwork", tier="core")
    def woodwork(ctx) -> float:
        '''Shots that hit the post or bar.'''
        shots = ctx.shots
        return float(shots["shot_outcome"].isin({"Post"}).sum())

To *score* with it, list its name under a family in the config taxonomy — the
pipeline computes every registered measure and the scorer picks up whichever
ones the taxonomy names. Measures may return ``nan`` when their inputs are
unavailable (e.g. OBV columns on open data); a ``nan`` simply drops out of its
family's mean.

Each measure receives a :class:`MatchContext` — the precomputed shared inputs
(clocked events, shots, the win-probability curve, Elo context, ...) — so
individual measures stay tiny and never recompute expensive shared state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

#: name -> (function, tier). Tiers: "core" computes on any StatsBomb feed;
#: "context" needs Elo ratings; "extended" needs OBV columns.
_REGISTRY: dict[str, dict] = {}


@dataclass
class MatchContext:
    """Shared per-match inputs, computed once and handed to every measure.

    The pipeline builds one instance per match and passes it to every registered
    measure, so expensive shared state (the clocked event stream, the shots
    subset, the win-probability curve, the Elo context) is computed a single time
    rather than once per measure.

    Attributes:
        ev: Playable events (regulation + extra time, shootout removed) carrying
            the ``_t`` minute clock.
        home: Home team name as it appears in the event feed.
        away: Away team name as it appears in the event feed.
        end: Match end minute — 90 for a regulation game, 120 when extra time was
            played.
        shots: The ``ev[type == "Shot"]`` subset, precomputed for shot measures.
        wp: Goals-only Skellam home/draw/away win-probability curve over the
            match clock.
        events_all: Unfiltered events including the shootout, kept so measures
            such as ``shootout_drama`` can see kicks that ``ev`` drops.
        stage: Fixture-sheet stage string, lower-cased (e.g. ``"group"``,
            ``"round of 16"``). Defaults to ``"group"``.
        prior_home: Elo-derived per-minute scoring rate for the home team; None
            when Elo ratings are unavailable.
        prior_away: Elo-derived per-minute scoring rate for the away team; None
            when Elo ratings are unavailable.
        elo_ctx: Pregame Elo->Skellam context dict; None without Elo. When
            present it carries at least ``"elo_gap"`` (signed Elo points,
            home minus away) and ``"entropy"`` (normalized outcome entropy) —
            the keys ``prematch.py`` reads.
        row: The fixture-sheet row (final scores, curated card counts, host
            flags, ...); None for a match not on the sheet.
        cache: Scratch dict shared between measures within one match, so a
            derived quantity (e.g. card counts) is computed once and reused.
    """

    ev: pd.DataFrame                    # playable events with the _t clock
    home: str
    away: str
    end: float                          # match end minute (90 or 120)
    shots: pd.DataFrame                 # ev[type == Shot]
    wp: pd.DataFrame                    # goals-only Skellam H/D/A curve
    events_all: pd.DataFrame            # unfiltered events (incl. shootout, for shootout_drama)
    stage: str = "group"                # fixture-sheet stage string, lower-cased
    prior_home: float | None = None  # Elo-derived per-minute scoring rates
    prior_away: float | None = None
    elo_ctx: dict | None = None      # pregame p0/entropy/upset context; None without Elo
    row: pd.Series | None = None     # the fixture-sheet row (scores, cards, ...)
    cache: dict = field(default_factory=dict)   # scratch shared between measures


def measure(name: str, *, tier: str = "core") -> Callable:
    """Register a measure function under ``name``.

    Args:
        name: Registry key for the measure; must be unique.
        tier: Input requirement tier. ``"core"`` computes on any StatsBomb feed;
            ``"context"`` needs Elo ratings; ``"extended"`` needs OBV columns.

    Returns:
        The decorator that inserts the wrapped function into ``_REGISTRY`` and
        returns it unchanged. The wrapped function takes a :class:`MatchContext`
        and returns a float (``nan`` = inputs unavailable).

    Raises:
        ValueError: If ``name`` is already registered (duplicate keys would let
            one measure silently shadow another).
    """
    def deco(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"measure {name!r} is already registered")
        _REGISTRY[name] = {"fn": fn, "tier": tier}
        return fn
    return deco


def registered_measures() -> dict[str, dict]:
    """Return a shallow copy of the full registry.

    Returns:
        A ``name -> {"fn", "tier"}`` dict. Import ``excitement_index.measures``
        first so every family module has run its ``@measure`` decorators and the
        registry is fully populated.
    """
    return dict(_REGISTRY)


def compute_all(ctx: MatchContext) -> dict[str, float]:
    """Run every registered measure on one match.

    Args:
        ctx: The shared :class:`MatchContext` for the match.

    Returns:
        A ``name -> value`` dict with one entry per registered measure. A measure
        that raises is recorded as ``nan`` rather than aborting the whole match,
        and downstream a ``nan`` simply drops out of its family's mean.
    """
    out: dict[str, float] = {}
    for name, meta in _REGISTRY.items():
        try:
            out[name] = float(meta["fn"](ctx))
        # A measure typically raises because its inputs are absent for this feed
        # (missing OBV columns, no Elo table, an empty shots frame) — an expected
        # "unavailable" that must map to nan so the family mean skips it. The same
        # broad catch also swallows genuine bugs (a mistyped ctx key, a schema
        # change), which then surface silently as nan; when debugging a measure
        # that reads as unexpectedly nan, temporarily re-raise here to see the
        # traceback rather than trusting this fallback.
        except Exception:
            out[name] = float(np.nan)
    return out
