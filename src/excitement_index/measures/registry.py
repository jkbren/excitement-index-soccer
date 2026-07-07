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
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

#: name -> (function, tier). Tiers: "core" computes on any StatsBomb feed;
#: "context" needs Elo ratings; "extended" needs OBV columns.
_REGISTRY: Dict[str, dict] = {}


@dataclass
class MatchContext:
    """Shared per-match inputs, computed once and handed to every measure."""

    ev: pd.DataFrame                    # playable events with the _t clock
    home: str
    away: str
    end: float                          # match end minute (90 or 120)
    shots: pd.DataFrame                 # ev[type == Shot]
    wp: pd.DataFrame                    # goals-only Skellam H/D/A curve
    events_all: pd.DataFrame            # unfiltered events (incl. shootout, for shootout_drama)
    stage: str = "group"                # fixture-sheet stage string, lower-cased
    prior_home: Optional[float] = None  # Elo-derived per-minute scoring rates
    prior_away: Optional[float] = None
    elo_ctx: Optional[dict] = None      # pregame p0/entropy/upset context; None without Elo
    row: Optional[pd.Series] = None     # the fixture-sheet row (scores, cards, ...)
    cache: dict = field(default_factory=dict)   # scratch shared between measures


def measure(name: str, *, tier: str = "core") -> Callable:
    """Register a measure function under ``name``. The function takes a
    :class:`MatchContext` and returns a float (``nan`` = unavailable)."""
    def deco(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"measure {name!r} is already registered")
        _REGISTRY[name] = {"fn": fn, "tier": tier}
        return fn
    return deco


def registered_measures() -> Dict[str, dict]:
    """The full registry (import ``excitement_index.measures`` first so every
    family module has run its decorators)."""
    return dict(_REGISTRY)


def compute_all(ctx: MatchContext) -> Dict[str, float]:
    """Run every registered measure on one match. A measure that raises is
    recorded as ``nan`` rather than sinking the whole match."""
    out: Dict[str, float] = {}
    for name, meta in _REGISTRY.items():
        try:
            out[name] = float(meta["fn"](ctx))
        except Exception:
            out[name] = float(np.nan)
    return out
