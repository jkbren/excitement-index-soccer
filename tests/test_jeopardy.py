"""Qualification-jeopardy invariants (Monte-Carlo, so we test properties, not
point values): determinism under the per-match seed, the knockout guarantee,
bounds, and the group-inference trick that recovers group labels from the
round-robin structure of open data."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("EXCITEMENT_INDEX_CACHE",
                      str(Path(__file__).parent.parent / ".opendata_cache"))


def _wc2022():
    """Load the 2022 World Cup match list, or skip the test if open data is unavailable.

    Returns:
        pandas.DataFrame: The match table from opendata.load_matches (one row per match).

    Skips (rather than fails) when running offline with a cold cache, matching the graceful-skip
    convention used across the parity/jeopardy tests.
    """
    from excitement_index import opendata
    try:
        return opendata.load_matches("FIFA World Cup", "2022")
    except Exception as e:
        pytest.skip(f"open data unavailable ({e})")


def test_infer_groups_recovers_eight_groups_of_four():
    """infer_groups must recover the 8 groups of 4 teams from the round-robin structure.

    Open data does not label group letters, so infer_groups reconstructs group membership from
    who played whom in the round-robin. For the 2022 World Cup that must yield exactly 8 groups,
    each with exactly 4 teams.
    """
    from excitement_index.jeopardy import infer_groups

    matches = _wc2022()
    groups = infer_groups(matches)
    # Invert the team -> group mapping into group -> set(teams) to count members per group.
    teams_per_group = {}
    for team, g in groups.items():
        teams_per_group.setdefault(g, set()).add(team)
    assert len(teams_per_group) == 8, f"expected 8 groups, got {len(teams_per_group)}"
    assert all(len(t) == 4 for t in teams_per_group.values()), \
        f"every group must have 4 teams: { {g: len(t) for g, t in teams_per_group.items()} }"


def test_jeopardy_bounds_knockouts_and_determinism():
    """Qualification jeopardy must be 1.0 on knockouts, in [0, 1] on group games, and deterministic.

    Builds a minimal feature frame (stage + knockout flag) for the 2022 World Cup and runs the
    Monte-Carlo jeopardy estimator twice. Contracts checked: knockout matches always score
    exactly 1.0 (elimination is certain); group-stage values are computable and lie in [0, 1];
    and two runs with the same inputs produce identical output, because the estimator seeds its
    randomness per match.
    """
    import pandas as pd

    from excitement_index import opendata
    from excitement_index.jeopardy import add_qualification_jeopardy

    matches = _wc2022()
    fm = pd.DataFrame(index=pd.Index(matches["match_id"], name="match_id"))
    fm["stage"] = matches.set_index("match_id")["stage"]
    # Everything past the group stage is a knockout (single-elimination) match.
    fm["knockout"] = (fm["stage"] != "group").astype(float)

    elo = opendata.load_elo()
    # Run twice on independent copies to check the per-match seeding makes the result reproducible.
    out1 = add_qualification_jeopardy(fm.copy(), matches, elo=elo)
    out2 = add_qualification_jeopardy(fm.copy(), matches, elo=elo)

    j1 = out1["qualification_jeopardy"]
    ko = fm.index[fm["knockout"] == 1]
    assert (j1.loc[ko] == 1.0).all(), "knockout matches must have jeopardy exactly 1.0"
    # Group-stage jeopardy (drop knockouts, drop any NaN) must exist and stay within [0, 1].
    grp = j1.drop(ko).dropna()
    assert len(grp) > 0, "group-stage jeopardy should be computable from open data"
    assert ((grp >= 0.0) & (grp <= 1.0)).all()
    pd.testing.assert_series_equal(j1, out2["qualification_jeopardy"],
                                   check_exact=True, obj="jeopardy must be deterministic")
