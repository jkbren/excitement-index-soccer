"""excitement_index — a transparent excitement rating for soccer matches.

The published NetSI Sport / Northeastern Global News World Cup 2026 index
as an open, configurable pipeline:

    from excitement_index import opendata, build_feature_matrix, score_matches

    matches = opendata.load_matches("FIFA World Cup", "2022")
    features = build_feature_matrix(matches, opendata.load_events,
                                    elo=opendata.load_elo())
    board = score_matches(features)          # 0-10 ratings, best match first

Weights, taxonomy, and deduction parameters live in ``config/default.yaml``; pass
``config=`` overrides to experiment. New measures register with one decorator —
see ``excitement_index.measures.registry``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import measures  # noqa: F401  (imports run every @measure registration)
from .clock import match_end_minute, playable_events, resolve_team_name
from .config import load_config
from .jeopardy import add_qualification_jeopardy
from .measures.registry import MatchContext, compute_all
from .scoring import make_reference, score_matches
from .wp import pregame_outcome_probs, wp_curve

__version__ = "1.0.0"
__all__ = ["build_feature_matrix", "extract_features", "score_matches",
           "make_reference", "add_qualification_jeopardy", "load_config"]

# Elo-point home-field/host bump added to (or subtracted from) a side's rating
# before the win-probability prior. 50 Elo points is the project's standard
# host-advantage adjustment, applied only for tournament host nations.
HOST_ELO_EDGE = 50.0


def extract_features(events: pd.DataFrame, match_row: pd.Series,
                     elo: pd.DataFrame | None = None,
                     hosts: set | None = None) -> dict:
    """Compute all registered measures for one match.

    Args:
        events: The full event frame for the match (unfiltered).
        match_row: Fixture metadata. Needs ``home``/``away`` (fixture-sheet names,
            resolved onto the event feed's naming) and optionally ``stage`` and the
            final score.
        elo: Optional team-indexed frame with an ``elo`` column. Without it the
            context-tier measures are ``nan`` and the win-probability model falls
            back to a symmetric prior.
        hosts: Optional set of host-nation team names; a host gets the
            ``HOST_ELO_EDGE`` rating bump in the pre-match prior.

    Returns:
        dict: Measure name -> value for this match, as produced by ``compute_all``.
    """
    hosts = hosts or set()
    ev = playable_events(events)
    teams = set(ev["team"].dropna().unique())
    home = resolve_team_name(match_row.get("home"), teams)
    away = resolve_team_name(match_row.get("away"), teams)
    end = match_end_minute(ev)

    prior_home = prior_away = None
    elo_ctx = None
    if elo is not None and home in elo.index and away in elo.index:
        eh, ea = float(elo.loc[home, "elo"]), float(elo.loc[away, "elo"])
        # Give the host the edge; a host on either side flips the sign, else no bump.
        hfa = HOST_ELO_EDGE if home in hosts else (-HOST_ELO_EDGE if away in hosts else 0.0)
        pre = pregame_outcome_probs(eh, ea, hfa_elo=hfa)
        # pregame_outcome_probs returns expected goals per 90 minutes; the WP model
        # wants a per-minute scoring rate, so divide by the 90-minute match length.
        prior_home, prior_away = pre["mu_home"] / 90.0, pre["mu_away"] / 90.0
        elo_ctx = dict(pre, elo_home=eh, elo_away=ea, elo_gap=eh - ea)

    # Pass the raw, unfiltered `events` here (not `ev`): the WP curve needs every
    # event, including non-playable ones, to place the timeline. Everything else
    # in the context is built from the playable-filtered `ev`. This asymmetry is
    # intentional — do not "fix" it to `ev`.
    curve = wp_curve(events, home=home, away=away,
                     prior_home=prior_home, prior_away=prior_away, xg_update=False)
    # A stage cell that is present but NaN would stringify to "nan"; treat a missing OR NaN
    # stage as the group-stage default so the knockout/stage measures read it correctly.
    raw_stage = match_row.get("stage", "group")
    stage = "group" if pd.isna(raw_stage) else str(raw_stage).lower()
    ctx = MatchContext(ev=ev, home=home, away=away, end=end,
                       shots=ev[ev["type"] == "Shot"], wp=curve, events_all=events,
                       stage=stage,
                       prior_home=prior_home, prior_away=prior_away,
                       elo_ctx=elo_ctx, row=match_row)
    return compute_all(ctx)


def build_feature_matrix(matches: pd.DataFrame, load_events: Callable[[int], pd.DataFrame],
                         elo: pd.DataFrame | None = None,
                         config=None, jeopardy: bool = True) -> pd.DataFrame:
    """Build one feature row per match, indexed by ``match_id``.

    Args:
        matches: Match table. Needs ``match_id, home, away, stage`` (plus dates and
            scores for the jeopardy simulation).
        load_events: Callable mapping a match id to its event frame
            (e.g. :func:`excitement_index.opendata.load_events`).
        elo: Optional team-indexed Elo frame, forwarded to :func:`extract_features`.
        config: Optional config path/dict/override forwarded to ``load_config``.
        jeopardy: When True, add the qualification-jeopardy column via the
            simulation; when False, fill it as 1.0 for knockout matches and ``nan``
            otherwise.

    Returns:
        pd.DataFrame: Feature matrix indexed by ``match_id`` (empty frame if no
        match produced features). Matches whose events fail to load or come back
        empty are silently skipped (no error, no placeholder row).
    """
    cfg = load_config(config)
    hosts = set(cfg.get("host_nations", []))
    rows = []
    for _, r in matches.iterrows():
        # A single unreadable match must not sink the whole board, so swallow any
        # load error and move on; the match is simply absent from the output.
        try:
            ev = load_events(int(r["match_id"]))
        except Exception:
            continue
        if ev is None or ev.empty:
            continue
        feats = extract_features(ev, r, elo=elo, hosts=hosts)
        feats["match_id"] = int(r["match_id"])
        # Carry fixture identity columns through when present on the source row.
        for c in ("home", "away", "stage", "match_date"):
            if c in r.index:
                feats[c] = r[c]
        rows.append(feats)
    if not rows:
        return pd.DataFrame()
    fm = pd.DataFrame(rows).set_index("match_id")
    if jeopardy:
        fm = add_qualification_jeopardy(fm, matches, elo=elo, hosts=hosts)
    else:
        fm["qualification_jeopardy"] = np.where(fm.get("knockout", 0) == 1, 1.0, np.nan)
    return fm
