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

import numpy as np
import pandas as pd

from .wp import pregame_outcome_probs

#: Crude host-nation Elo edge (points) fed into the simulated fixtures' goal
#: means (mirrors ``host_elo_edge`` in the config; World Cups are otherwise
#: neutral-venue, so only host games carry an edge).
HFA_ELO = 50.0

#: Elo assigned to a team missing from the ratings table. 1700 is a mid-table
#: international rating, so an unrated team simulates as roughly average.
DEFAULT_ELO = 1700.0

#: P(third place advances | final points) — the points-conditional prior that
#: stands in for the 48-team format's cross-group best-thirds comparison (a
#: 3-point third almost never survives the best-8 cut). Key 8 is intentionally
#: omitted: ``_p3_advance`` clamps points to at most 9 and looks them up with a
#: 0.99 default, so 8 (the only missing key <= 9) falls through to 0.99, while 9
#: and any higher value (which clamps to 9) hit the explicit 0.99 entry. Either
#: way a strong third place is treated as near-certain to advance. Do not "fill
#: the gap" at 8.
P3_BY_PTS = {0: 0.01, 1: 0.01, 2: 0.02, 3: 0.15, 4: 0.55, 5: 0.80, 6: 0.95,
             7: 0.98, 9: 0.99}


def _p3_advance(pts: int) -> float:
    """Prior probability that a third-placed team advances given its final points.

    Args:
        pts: Final group points for the third-placed team.

    Returns:
        P(third place advances) in [0, 1]: a table lookup in ``P3_BY_PTS`` with
        points clamped to at most 9. Key 8 is the only omitted key <= 9, so it
        falls through to the 0.99 default; 9 (and any higher value, which clamps
        to 9) hits the explicit 0.99 entry. Either path returns 0.99.
    """
    return P3_BY_PTS.get(min(int(pts), 9), 0.99)


def _is_group_stage(stage) -> bool:
    """True for a group game.

    Args:
        stage: The stage field of a fixture row (any type; stringified here).

    Returns:
        True when the stage denotes a group game. The private fixture sheet
        coded the stage as ``"group"`` (or left it blank, which stringifies to
        ``"nan"``/``""``); StatsBomb open data spells it ``"Group Stage"`` — all
        four spellings are accepted so a missing stage defaults to group.
    """
    return str(stage).lower() in ("group", "group stage", "nan", "")


def _date_col(matches: pd.DataFrame) -> str:
    """Name of the kickoff-date column in a fixture frame.

    Args:
        matches: A fixture frame.

    Returns:
        ``"match_date"`` when present (StatsBomb open-data schema), else
        ``"date"`` (the private fixture-sheet schema).
    """
    return "match_date" if "match_date" in matches.columns else "date"


def infer_groups(matches: pd.DataFrame) -> dict[str, str]:
    """Recover group labels from the round-robin structure of the group stage.

    Open data has no group letters, but two teams meet in the group stage if
    and only if they share a group — so the groups are exactly the connected
    components of the "played each other in the group stage" graph.

    Args:
        matches: The full fixture sheet. Uses ``home``/``away`` (team names),
            ``stage`` (to restrict to group games when present), and the kickoff
            date column for ordering.

    Returns:
        ``{team: label}`` with deterministic labels: components are ordered by
        their earliest kickoff, then alphabetically, and lettered ``A``, ``B``,
        ... (``G27``, ``G28``, ... past 26 groups). Empty when the sheet has no
        group-stage fixtures.
    """
    g = matches[matches["stage"].apply(_is_group_stage)] if "stage" in matches.columns \
        else matches
    if g.empty:
        return {}
    # Union-find over team names: each played group game unions its two teams,
    # so the resulting sets are the groups.
    parent: dict[str, str] = {}

    def find(t: str) -> str:
        # Path-halving find: point each node at its grandparent while walking up.
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

    # Collapse the forest into {representative: set(teams)}.
    comps: dict[str, set] = {}
    for t in parent:
        comps.setdefault(find(t), set()).add(t)

    dcol = _date_col(g)
    dates = g[dcol].astype(str) if dcol in g.columns else pd.Series("", index=g.index)

    def first_kickoff(teams: set) -> str:
        # Earliest kickoff string among the group's games; used only to order
        # the components so labels are stable. O(rows) per call — fine at
        # tournament scale (a few dozen components), but note it rescans ``g``.
        mask = g["home"].isin(teams) | g["away"].isin(teams)
        d = dates[mask]
        return str(d.min()) if len(d) else ""

    # Order components by (earliest kickoff, sorted team tuple) so lettering is
    # deterministic regardless of row order in the input sheet.
    ordered = sorted(comps.values(), key=lambda ts: (first_kickoff(ts), tuple(sorted(ts))))
    out: dict[str, str] = {}
    for i, teams in enumerate(ordered):
        label = string.ascii_uppercase[i] if i < 26 else f"G{i + 1}"
        for t in teams:
            out[t] = label
    return out


def _jeopardy_rank_probs(teams, base_pts, base_gd, base_gf, fixtures, focal,
                         focal_outcome, rng, n_sims=300):
    """Simulate one conditioned branch: the group's remaining fixtures with the
    focal match forced to a given outcome.

    Args:
        teams: All team names in the group.
        base_pts: ``{team: points}`` already accrued before the remaining
            fixtures (starting standings; copied per simulation, not mutated).
        base_gd: ``{team: goal difference}`` starting standings.
        base_gf: ``{team: goals for}`` starting standings.
        fixtures: Remaining non-focal fixtures as ``(home, away, mu_home,
            mu_away)`` tuples, where the ``mu``s are Poisson goal means.
        focal: The focal match as ``(home, away, mu_home, mu_away)``.
        focal_outcome: The outcome the focal match is conditioned on — ``"H"``
            (home win), ``"A"`` (away win), or ``"D"`` (draw).
        rng: Shared ``numpy`` Generator; consumed sequentially so results are
            reproducible (see ``_match_jeopardy`` for the threading contract).
        n_sims: Number of Monte-Carlo simulations for this branch.

    Returns:
        ``(adv2, third)`` where ``adv2`` is ``{team: P(finish top two)}`` and
        ``third`` is ``{team: P(third place advances)}`` — the latter weights a
        third-place finish by its points-conditional ``_p3_advance`` prior. Both
        are probabilities in [0, 1] over the ``n_sims`` draws.

    The focal match is conditioned by rejection sampling: draw scores from the
    two Poissons until the result matches ``focal_outcome``, capped at 60 tries
    to bound the loop. If no matching draw appears (rare, e.g. a lopsided mean
    with a low-probability target outcome), fall back to a minimal
    representative score — 1-0 / 0-1 / 1-1 — so the branch is never skipped.
    Ranking is by points, then goal difference, then goals for, with a
    ``rng.random()`` jitter as the final deterministic tiebreak.
    """
    adv2 = {t: 0 for t in teams}
    third = {t: 0.0 for t in teams}

    def _apply(pts, gd, gf, h, a, gh, ga):
        # Fold one played/simulated result (gh-ga) into the running standings.
        pts[h] += 3 if gh > ga else (1 if gh == ga else 0)
        pts[a] += 3 if ga > gh else (1 if gh == ga else 0)
        gd[h] += gh - ga
        gd[a] += ga - gh
        gf[h] += gh
        gf[a] += ga

    for _ in range(n_sims):
        # Fresh copy of the starting standings for this simulation.
        pts = dict(base_pts)
        gd = dict(base_gd)
        gf = dict(base_gf)

        h, a, mh, ma = focal
        # Rejection-sample the focal score until it matches the conditioned
        # outcome; 60 tries bounds the loop for low-probability target outcomes.
        for _try in range(60):
            gh, ga = rng.poisson(mh), rng.poisson(ma)
            oc = "H" if gh > ga else ("A" if ga > gh else "D")
            if oc == focal_outcome:
                break
        else:
            # No accepted draw in 60 tries: use a minimal representative score
            # consistent with the conditioned outcome so the branch still counts.
            gh, ga = (1, 0) if focal_outcome == "H" else ((0, 1) if focal_outcome == "A" else (1, 1))
        _apply(pts, gd, gf, h, a, gh, ga)
        # Simulate every remaining fixture from its Poisson means.
        for (h2, a2, m2h, m2a) in fixtures:
            _apply(pts, gd, gf, h2, a2, rng.poisson(m2h), rng.poisson(m2a))
        # Rank the final table; rng.random() breaks exact ties deterministically.
        order = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True)
        for i, t in enumerate(order):
            if i < 2:
                # Top two advance outright.
                adv2[t] += 1
            elif i == 2:
                # Third place advances only with its points-conditional prior.
                third[t] += _p3_advance(pts[t])
    return ({t: adv2[t] / n_sims for t in teams}, {t: third[t] / n_sims for t in teams})


def _match_jeopardy(row: pd.Series, matches: pd.DataFrame, groups: dict[str, str],
                    elo: pd.DataFrame | None, hosts: set, *,
                    n_sims: int = 300) -> float:
    """Score in [0, 1] for how much this match's result could swing qualification.

    Args:
        row: One fixture row (a ``matches`` row) — the focal match. Reads
            ``stage``, ``home``, ``away``, the kickoff date column, and
            ``match_id``.
        matches: The full fixture sheet for the competition.
        groups: ``{team: group label}`` from :func:`infer_groups`.
        elo: Team-indexed frame with an ``elo`` column, or ``None``. Teams
            missing from it fall back to ``DEFAULT_ELO`` inside the simulation.
        hosts: Set of host-nation team names granted the ``HFA_ELO`` edge.
        n_sims: Simulations per branch, passed through to
            :func:`_jeopardy_rank_probs`.

    Returns:
        1.0 for knockout matches; for group matches, the mean over both teams of
        ``max(P(advance | win) - P(advance | lose), 0)`` from the four-branch
        simulation described in the module docstring. ``nan`` when the group,
        schedule, or Elo context needed to simulate is unavailable.
    """
    if not _is_group_stage(row.get("stage", "group")):
        # Knockout: every game is do-or-die, so jeopardy is maximal by definition.
        return 1.0
    home, away = row.get("home"), row.get("away")
    grp = groups.get(str(home))
    if grp is None or elo is None:
        return float(np.nan)
    dcol = _date_col(matches)
    date = str(row.get(dcol, ""))
    # Restrict to the focal team's group's group-stage fixtures.
    in_group = matches["home"].astype(str).map(groups) == grp
    gsched = matches[in_group & matches["stage"].apply(_is_group_stage)] \
        if "stage" in matches.columns else matches[in_group]
    if gsched.empty:
        return float(np.nan)

    # Base standings: same-group games strictly before the focal date, with scores.
    played = gsched[(gsched[dcol].astype(str) < date) & gsched["score_home"].notna()]
    teams = sorted(set(gsched["home"].astype(str)) | set(gsched["away"].astype(str)))
    base_pts = {t: 0 for t in teams}
    base_gd = {t: 0 for t in teams}
    base_gf = {t: 0 for t in teams}
    for _, m in played.iterrows():
        gh, ga = int(m["score_home"]), int(m["score_away"])
        h, a = str(m["home"]), str(m["away"])
        base_pts[h] += 3 if gh > ga else (1 if gh == ga else 0)
        base_pts[a] += 3 if ga > gh else (1 if gh == ga else 0)
        base_gd[h] += gh - ga
        base_gd[a] += ga - gh
        base_gf[h] += gh
        base_gf[a] += ga

    def _elo(t):
        # Team Elo, or the mid-table default when the team is unrated.
        return float(elo.loc[t, "elo"]) if t in elo.index else DEFAULT_ELO

    def _mus(h, a):
        # Poisson goal means for a fixture, from the Elo gap plus the host edge
        # (positive when home is a host, negative when away is a host, else 0).
        hfa = HFA_ELO if h in hosts else (-HFA_ELO if a in hosts else 0.0)
        p = pregame_outcome_probs(_elo(h), _elo(a), hfa_elo=hfa)
        return p["mu_home"], p["mu_away"]

    # Remaining fixtures at kickoff. A game is skipped for one of two distinct
    # reasons, both folded into the single `continue`: it is already in the base
    # standings (its id is in `known_ids`, which also holds the focal id so the
    # focal match is not double-counted); or it is a past group game we could not
    # score (strictly before the focal date but absent from `played` for lack of
    # a score), which we drop rather than simulate. Everything else — including
    # same-date group games, which FIFA plays simultaneously — is simulated.
    known_ids = set(played["match_id"].astype(int)) | {int(row["match_id"])}
    remaining = []
    for _, fx in gsched.iterrows():
        if int(fx["match_id"]) in known_ids or str(fx[dcol]) < date:
            continue
        mh, ma = _mus(str(fx["home"]), str(fx["away"]))
        remaining.append((str(fx["home"]), str(fx["away"]), mh, ma))
    mh0, ma0 = _mus(str(home), str(away))
    focal = (str(home), str(away), mh0, ma0)
    # Determinism contract (load-bearing for golden parity): one rng, seeded from
    # the match id (masked to 31 bits so it fits numpy's non-negative seed range),
    # is threaded through all four branches in a fixed order — home team's
    # win-then-lose branch, then away team's — and consumed sequentially inside
    # each branch. That fixed ordering plus the single shared rng is what
    # reproduces the reference values; do not reorder the loop, split the rng, or
    # reseed per branch.
    rng = np.random.default_rng(int(row.get("match_id", 0)) % (2 ** 31))
    vals = []
    for team, win_oc, lose_oc in ((str(home), "H", "A"), (str(away), "A", "H")):
        # P(advance) for this team conditioned on it winning the focal match...
        a2w, a3w = _jeopardy_rank_probs(teams, base_pts, base_gd, base_gf,
                                        remaining, focal, win_oc, rng, n_sims)
        # ...and conditioned on it losing. The swing is the jeopardy signal.
        a2l, a3l = _jeopardy_rank_probs(teams, base_pts, base_gd, base_gf,
                                        remaining, focal, lose_oc, rng, n_sims)
        vals.append(max((a2w[team] + a3w[team]) - (a2l[team] + a3l[team]), 0.0))
    return float(np.mean(vals))


def add_qualification_jeopardy(features: pd.DataFrame, matches: pd.DataFrame,
                               elo: pd.DataFrame | None = None,
                               hosts: set | None = None) -> pd.DataFrame:
    """Add the ``qualification_jeopardy`` column to a feature matrix.

    ``features`` is indexed by ``match_id`` (as built by
    :func:`excitement_index.build_feature_matrix`); ``matches`` is the full
    fixture sheet for the competition (``match_id, match_date, home, away,
    score_home, score_away, stage`` — e.g. from ``opendata.load_matches``);
    ``elo`` is a team-indexed frame with an ``elo`` column (teams missing from
    it get a 1700 default inside the simulation); ``hosts`` is the set of host
    nations granted the Elo home edge. Knockout matches get exactly 1.0; a
    group match whose group/schedule/Elo context is unavailable gets ``nan``.

    Args:
        features: Feature matrix indexed by ``match_id`` (from
            :func:`excitement_index.build_feature_matrix`).
        matches: Full fixture sheet for the competition (``match_id,
            match_date, home, away, score_home, score_away, stage`` — e.g. from
            ``opendata.load_matches``).
        elo: Team-indexed frame with an ``elo`` column, or ``None`` (then every
            group match resolves to ``nan``, since Elo is required to simulate).
        hosts: Set of host-nation names, or ``None`` for no host edge.

    Returns:
        ``features`` with a ``qualification_jeopardy`` column added (in [0, 1]
        or ``nan``). The frame is modified in place and also returned.

    The two ``except Exception`` guards swallow any failure to ``nan`` (per-match
    inner guard, whole-column outer guard) so a schema surprise on one match
    never aborts building the rest of the matrix. The tradeoff is that a genuine
    bug — a mismatched schema, a bad cast — silently becomes ``nan`` with no
    signal, so a suddenly all-``nan`` column is the symptom to investigate here.
    """
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
                # One match failed to score (bad row, cast, or missing field):
                # leave it nan and keep going rather than losing the whole column.
                vals[mid] = float(np.nan)
        features["qualification_jeopardy"] = pd.Series(vals)
    except Exception:
        # Setup itself failed (e.g. no match_id column): whole column is nan.
        features["qualification_jeopardy"] = np.nan
    return features
