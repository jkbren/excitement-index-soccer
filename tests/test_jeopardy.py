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
    from excitement_index import opendata
    try:
        return opendata.load_matches("FIFA World Cup", "2022")
    except Exception as e:
        pytest.skip(f"open data unavailable ({e})")


def test_infer_groups_recovers_eight_groups_of_four():
    from excitement_index.jeopardy import infer_groups

    matches = _wc2022()
    groups = infer_groups(matches)
    group_sizes = groups.groupby(groups).size() if hasattr(groups, "groupby") else None
    teams_per_group = {}
    for team, g in groups.items():
        teams_per_group.setdefault(g, set()).add(team)
    assert len(teams_per_group) == 8, f"expected 8 groups, got {len(teams_per_group)}"
    assert all(len(t) == 4 for t in teams_per_group.values()), \
        f"every group must have 4 teams: { {g: len(t) for g, t in teams_per_group.items()} }"


def test_jeopardy_bounds_knockouts_and_determinism():
    import pandas as pd

    from excitement_index import opendata
    from excitement_index.jeopardy import add_qualification_jeopardy

    matches = _wc2022()
    fm = pd.DataFrame(index=pd.Index(matches["match_id"], name="match_id"))
    fm["stage"] = matches.set_index("match_id")["stage"]
    fm["knockout"] = (fm["stage"] != "group").astype(float)

    elo = opendata.load_elo()
    out1 = add_qualification_jeopardy(fm.copy(), matches, elo=elo)
    out2 = add_qualification_jeopardy(fm.copy(), matches, elo=elo)

    j1 = out1["qualification_jeopardy"]
    ko = fm.index[fm["knockout"] == 1]
    assert (j1.loc[ko] == 1.0).all(), "knockout matches must have jeopardy exactly 1.0"
    grp = j1.drop(ko).dropna()
    assert len(grp) > 0, "group-stage jeopardy should be computable from open data"
    assert ((grp >= 0.0) & (grp <= 1.0)).all()
    pd.testing.assert_series_equal(j1, out2["qualification_jeopardy"],
                                   check_exact=True, obj="jeopardy must be deterministic")
