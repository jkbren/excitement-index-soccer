"""Qualification jeopardy — how much a group game's result could swing who advances.

**qualification_jeopardy ∈ [0, 1].** For knockout matches, exactly 1. For group
matches: the mean over both teams of max(P(advance | win) − P(advance | lose), 0),
estimated by Monte-Carlo simulation of the group's remaining fixtures as of
kickoff (same-day group games are treated as simultaneous and simulated, not
assumed known). Each remaining fixture's score is drawn from independent
Poissons with Elo-derived means; the focal match is conditioned on the win/lose
branch by rejection sampling; groups are ranked by points, goal difference, and
goals scored; the top two advance, and third place advances with a
points-conditional probability (from ~0.01 at ≤ 2 points to 0.95+ at 6+,
reflecting the 48-team format's best-thirds rule). 300 simulations per branch,
four branches per match, deterministically seeded by match id.

This is a *matrix-level* feature: it needs the full fixture sheet (every group
game's date and score), which per-match extraction never sees, so it is added
to a finished feature matrix by :func:`add_qualification_jeopardy` rather than
registered as a per-match measure. StatsBomb open data carries no group
letters, so :func:`infer_groups` first recovers them from the round-robin
structure — teams that meet in the group stage share a group.
"""
from __future__ import annotations

import string
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .wp import pregame_outcome_probs

#: Crude host-nation Elo edge fed into the simulated fixtures' goal means
#: (mirrors ``host_elo_edge`` in the config; World Cups are otherwise neutral).
HFA_ELO = 50.0

#: Elo assigned to a team missing from the ratings table.
DEFAULT_ELO = 1700.0

#: P(third place advances | final points) — the points-conditional prior that
#: stands in for the 48-team format's cross-group best-thirds comparison (a
#: 3-point third almost never survives the best-8 cut). Missing keys (8, 10+)
#: fall back to 0.99 via ``get(min(pts, 9), 0.99)``.
P3_BY_PTS = {0: 0.01, 1: 0.01, 2: 0.02, 3: 0.15, 4: 0.55, 5: 0.80, 6: 0.95,
             7: 0.98, 9: 0.99}


def _p3_advance(pts: int) -> float:
    return P3_BY_PTS.get(min(int(pts), 9), 0.99)


def _is_group_stage(stage) -> bool:
    """True for a group game. The private fixture sheet coded the stage as
    ``"group"`` (or left it blank); StatsBomb open data spells it
    ``"Group Stage"`` — accept both."""
    return str(stage).lower() in ("group", "group stage", "nan", "")


def _date_col(matches: pd.DataFrame) -> str:
    return "match_date" if "match_date" in matches.columns else "date"


def infer_groups(matches: pd.DataFrame) -> Dict[str, str]:
    """Recover group labels from the round-robin structure of the group stage.

    Open data has no group letters, but two teams meet in the group stage if
    and only if they share a group — so the groups are exactly the connected
    components of the "played each other in the group stage" graph. Returns
    ``{team: label}`` with deterministic labels (components ordered by their
    earliest kickoff, then alphabetically, and lettered ``A``, ``B``, ...).
    Empty when the sheet has no group-stage fixtures."""
    g = matches[matches["stage"].apply(_is_group_stage)] if "stage" in matches.columns \
        else matches
    if g.empty:
        return {}
    # Union-find over team names.
    parent: Dict[str, str] = {}

    def find(t: str) -> str:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for _, m in g.iterrows():
        h, a = str(m["home"]), str(m["away"])
        parent.setdefault(h, h)
        parent.setdefault(a, a)
        rh, ra = find(h), find(a)
        if rh != ra:
            parent[ra] = rh

    comps: Dict[str, set] = {}
    for t in parent:
        comps.setdefault(find(t), set()).add(t)

    dcol = _date_col(g)
    dates = g[dcol].astype(str) if dcol in g.columns else pd.Series("", index=g.index)

    def first_kickoff(teams: set) -> str:
        mask = g["home"].isin(teams) | g["away"].isin(teams)
        d = dates[mask]
        return str(d.min()) if len(d) else ""

    ordered = sorted(comps.values(), key=lambda ts: (first_kickoff(ts), tuple(sorted(ts))))
    out: Dict[str, str] = {}
    for i, teams in enumerate(ordered):
        label = string.ascii_uppercase[i] if i < 26 else "G%d" % (i + 1)
        for t in teams:
            out[t] = label
    return out


def _jeopardy_rank_probs(teams, base_pts, base_gd, base_gf, fixtures, focal,
                         focal_outcome, rng, n_sims=300):
    """One conditioned branch: Monte-Carlo the group's remaining fixtures with
    the focal match forced to ``focal_outcome`` ("H"/"A"/"D") by rejection
    sampling (60 tries, then a hard-coded 1-0 / 0-1 / 1-1 fallback). Ranks by
    points, goal difference, goals for, with seeded-rng jitter as the final
    tiebreak. Returns per-team P(top two) and P(third place advances)."""
    adv2 = {t: 0 for t in teams}
    third = {t: 0.0 for t in teams}
    for _ in range(n_sims):
        pts = dict(base_pts); gd = dict(base_gd); gf = dict(base_gf)

        def _apply(h, a, gh, ga):
            pts[h] += 3 if gh > ga else (1 if gh == ga else 0)
            pts[a] += 3 if ga > gh else (1 if gh == ga else 0)
            gd[h] += gh - ga; gd[a] += ga - gh
            gf[h] += gh; gf[a] += ga

        h, a, mh, ma = focal
        for _try in range(60):
            gh, ga = rng.poisson(mh), rng.poisson(ma)
            oc = "H" if gh > ga else ("A" if ga > gh else "D")
            if oc == focal_outcome:
                break
        else:
            gh, ga = (1, 0) if focal_outcome == "H" else ((0, 1) if focal_outcome == "A" else (1, 1))
        _apply(h, a, gh, ga)
        for (h2, a2, m2h, m2a) in fixtures:
            _apply(h2, a2, rng.poisson(m2h), rng.poisson(m2a))
        order = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True)
        for i, t in enumerate(order):
            if i < 2:
                adv2[t] += 1
            elif i == 2:
                third[t] += _p3_advance(pts[t])
    return ({t: adv2[t] / n_sims for t in teams}, {t: third[t] / n_sims for t in teams})


def _match_jeopardy(row: pd.Series, matches: pd.DataFrame, groups: Dict[str, str],
                    elo: Optional[pd.DataFrame], hosts: set, *,
                    n_sims: int = 300) -> float:
    """[0, 1] how much this match's result could swing qualification. Knockout
    matches score exactly 1.0; group matches run the four-branch simulation
    described in the module docstring. ``nan`` when the group/schedule/Elo
    context is unavailable."""
    if not _is_group_stage(row.get("stage", "group")):
        return 1.0
    home, away = row.get("home"), row.get("away")
    grp = groups.get(str(home))
    if grp is None or elo is None:
        return float(np.nan)
    dcol = _date_col(matches)
    date = str(row.get(dcol, ""))
    in_group = matches["home"].astype(str).map(groups) == grp
    gsched = matches[in_group & matches["stage"].apply(_is_group_stage)] \
        if "stage" in matches.columns else matches[in_group]
    if gsched.empty:
        return float(np.nan)

    # Base standings: same-group games strictly before the focal date, with scores.
    played = gsched[(gsched[dcol].astype(str) < date) & gsched["score_home"].notna()]
    teams = sorted(set(gsched["home"].astype(str)) | set(gsched["away"].astype(str)))
    base_pts = {t: 0 for t in teams}; base_gd = {t: 0 for t in teams}; base_gf = {t: 0 for t in teams}
    for _, m in played.iterrows():
        gh, ga = int(m["score_home"]), int(m["score_away"])
        h, a = str(m["home"]), str(m["away"])
        base_pts[h] += 3 if gh > ga else (1 if gh == ga else 0)
        base_pts[a] += 3 if ga > gh else (1 if gh == ga else 0)
        base_gd[h] += gh - ga; base_gd[a] += ga - gh
        base_gf[h] += gh; base_gf[a] += ga

    def _elo(t):
        return float(elo.loc[t, "elo"]) if t in elo.index else DEFAULT_ELO

    def _mus(h, a):
        hfa = HFA_ELO if h in hosts else (-HFA_ELO if a in hosts else 0.0)
        p = pregame_outcome_probs(_elo(h), _elo(a), hfa_elo=hfa)
        return p["mu_home"], p["mu_away"]

    # Remaining fixtures at kickoff: everything not already counted and not in
    # the past — same-date group games are FIFA-simultaneous, so they are
    # simulated, never assumed known.
    known_ids = set(played["match_id"].astype(int)) | {int(row["match_id"])}
    remaining = []
    for _, fx in gsched.iterrows():
        if int(fx["match_id"]) in known_ids or str(fx[dcol]) < date:
            continue
        mh, ma = _mus(str(fx["home"]), str(fx["away"]))
        remaining.append((str(fx["home"]), str(fx["away"]), mh, ma))
    mh0, ma0 = _mus(str(home), str(away))
    focal = (str(home), str(away), mh0, ma0)
    rng = np.random.default_rng(int(row.get("match_id", 0)) % (2 ** 31))
    vals = []
    for team, win_oc, lose_oc in ((str(home), "H", "A"), (str(away), "A", "H")):
        a2w, a3w = _jeopardy_rank_probs(teams, base_pts, base_gd, base_gf,
                                        remaining, focal, win_oc, rng, n_sims)
        a2l, a3l = _jeopardy_rank_probs(teams, base_pts, base_gd, base_gf,
                                        remaining, focal, lose_oc, rng, n_sims)
        vals.append(max((a2w[team] + a3w[team]) - (a2l[team] + a3l[team]), 0.0))
    return float(np.mean(vals))


def add_qualification_jeopardy(features: pd.DataFrame, matches: pd.DataFrame,
                               elo: Optional[pd.DataFrame] = None,
                               hosts: Optional[set] = None) -> pd.DataFrame:
    """Add the ``qualification_jeopardy`` column to a feature matrix.

    ``features`` is indexed by ``match_id`` (as built by
    :func:`excitement_index.build_feature_matrix`); ``matches`` is the full
    fixture sheet for the competition (``match_id, match_date, home, away,
    score_home, score_away, stage`` — e.g. from ``opendata.load_matches``);
    ``elo`` is a team-indexed frame with an ``elo`` column (teams missing from
    it get a 1700 default inside the simulation); ``hosts`` is the set of host
    nations granted the Elo home edge. Knockout matches get exactly 1.0; a
    group match whose group/schedule/Elo context is unavailable gets ``nan``."""
    hosts = set(hosts or ())
    try:
        groups = infer_groups(matches)
        by_id = matches.set_index(matches["match_id"].astype(int))
        vals = {}
        for mid in features.index:
            try:
                row = by_id.loc[int(mid)]
                if isinstance(row, pd.DataFrame):     # duplicate ids: take the first
                    row = row.iloc[0]
                vals[mid] = _match_jeopardy(row, matches, groups, elo, hosts)
            except Exception:
                vals[mid] = float(np.nan)
        features["qualification_jeopardy"] = pd.Series(vals)
    except Exception:
        features["qualification_jeopardy"] = np.nan
    return features
