"""StatsBomb open-data access.

Zero-dependency loader for the free StatsBomb open-data repository
(https://github.com/statsbomb/open-data): competitions, match lists, and
events, flattened into the wide event frame every measure expects. Files are
fetched over HTTPS on first use and cached locally (``EXCITEMENT_INDEX_CACHE``
env var, else ``.opendata_cache/`` in the working directory), so notebooks and
tests are fast and offline-friendly after the first run.

StatsBomb open data is released under its own user agreement — see the
open-data repository. This package does not redistribute it.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _cache_dir() -> Path:
    d = Path(os.environ.get("EXCITEMENT_INDEX_CACHE", ".opendata_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_json(rel_path: str):
    """Fetch ``data/<rel_path>`` from the open-data repo, with a local cache."""
    cached = _cache_dir() / rel_path
    if cached.is_file():
        with open(cached) as f:
            return json.load(f)
    url = f"{BASE_URL}/{rel_path}"
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https host
        payload = json.loads(resp.read().decode("utf-8"))
    cached.parent.mkdir(parents=True, exist_ok=True)
    with open(cached, "w") as f:
        json.dump(payload, f)
    return payload


def list_competitions() -> pd.DataFrame:
    """Every (competition, season) available in the open data."""
    return pd.DataFrame(_get_json("competitions.json"))


def load_matches(competition_name: str, season_name: str) -> pd.DataFrame:
    """The match list for one competition-season (e.g. ``"FIFA World Cup"``,
    ``"2022"``), flattened to the columns the pipeline uses: ``match_id,
    match_date, home, away, score_home, score_away, stage``."""
    comps = list_competitions()
    row = comps[(comps["competition_name"] == competition_name)
                & (comps["season_name"] == season_name)]
    if row.empty:
        available = comps[comps["competition_name"] == competition_name]["season_name"].tolist()
        raise ValueError(f"{competition_name} {season_name} not in open data; "
                         f"seasons available: {available}")
    comp_id = int(row.iloc[0]["competition_id"])
    season_id = int(row.iloc[0]["season_id"])
    raw = _get_json(f"matches/{comp_id}/{season_id}.json")
    out = pd.DataFrame([{
        "match_id": m["match_id"],
        "match_date": m.get("match_date"),
        "home": m["home_team"]["home_team_name"],
        "away": m["away_team"]["away_team_name"],
        "score_home": m.get("home_score"),
        "score_away": m.get("away_score"),
        "stage": _normalize_stage((m.get("competition_stage") or {}).get("name", "")),
    } for m in raw])
    return out.sort_values("match_date").reset_index(drop=True)


def _normalize_stage(stage: str) -> str:
    """Open-data stage names onto the index's conventions: lower-cased, with
    ``"Group Stage"`` mapped to ``"group"`` (the knockout flag tests equality
    with ``"group"``, so the spelling matters)."""
    s = str(stage).strip().lower()
    return "group" if s in {"group stage", "group"} else s


# Nested open-data fields -> flat event-frame columns. Each entry is
# (column_name, path). Missing fields become NaN.
_FIELDS = [
    ("id", ("id",)),
    ("index", ("index",)),
    ("period", ("period",)),
    ("minute", ("minute",)),
    ("second", ("second",)),
    ("type", ("type", "name")),
    ("team", ("team", "name")),
    ("player", ("player", "name")),
    ("possession", ("possession",)),
    ("play_pattern", ("play_pattern", "name")),
    ("location", ("location",)),
    ("counterpress", ("counterpress",)),
    ("pass_outcome", ("pass", "outcome", "name")),
    ("pass_end_location", ("pass", "end_location")),
    ("pass_through_ball", ("pass", "through_ball")),
    ("pass_shot_assist", ("pass", "shot_assist")),
    ("carry_end_location", ("carry", "end_location")),
    ("dribble_outcome", ("dribble", "outcome", "name")),
    ("shot_outcome", ("shot", "outcome", "name")),
    ("shot_statsbomb_xg", ("shot", "statsbomb_xg")),
    ("shot_type", ("shot", "type", "name")),
    ("shot_one_on_one", ("shot", "one_on_one")),
    ("foul_committed_card", ("foul_committed", "card", "name")),
    ("bad_behaviour_card", ("bad_behaviour", "card", "name")),
    # Timing / restart / stop-marker fields the ball-in-play estimator
    # (excitement_index.bip) differences to find dead time.
    ("timestamp", ("timestamp",)),
    ("duration", ("duration",)),
    ("out", ("out",)),
    ("pass_type", ("pass", "type", "name")),
    ("foul_committed_advantage", ("foul_committed", "advantage")),
    ("foul_won_advantage", ("foul_won", "advantage")),
    ("goalkeeper_outcome", ("goalkeeper", "outcome", "name")),
    ("duel_outcome", ("duel", "outcome", "name")),
    ("interception_outcome", ("interception", "outcome", "name")),
]


def _dig(obj, path):
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return np.nan
        obj = obj[key]
    return obj


def load_events(match_id: int) -> pd.DataFrame:
    """One match's events as the wide frame the measures expect. Open data has
    no on-ball-value columns, so the three OBV measures return ``nan`` and drop
    out of their family mean — the documented "extended tier" behavior."""
    raw = _get_json(f"events/{int(match_id)}.json")
    rows = [{col: _dig(e, path) for col, path in _FIELDS} for e in raw]
    return pd.DataFrame(rows)


def load_elo(path: str | None = None) -> pd.DataFrame:
    """The Elo table for pre-match strength (columns ``team, elo``), indexed by
    team name. Defaults to the snapshot shipped in ``data/elo.csv``
    (eloratings.net; regenerate with ``scripts/fetch_elo.py``)."""
    p = Path(path) if path else Path(__file__).resolve().parents[2] / "data" / "elo.csv"
    df = pd.read_csv(p)
    return df.set_index("team")
