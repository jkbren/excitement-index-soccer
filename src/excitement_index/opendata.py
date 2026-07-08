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
    """Resolve and create the local cache directory for fetched open-data files.

    Returns:
        Path: The cache root, taken from the ``EXCITEMENT_INDEX_CACHE`` env var
        when set, else ``.opendata_cache/`` under the current working directory.
        The directory (and any parents) is created if it does not yet exist.
    """
    d = Path(os.environ.get("EXCITEMENT_INDEX_CACHE", ".opendata_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_json(rel_path: str):
    """Fetch ``data/<rel_path>`` from the open-data repo, with a local cache.

    Performs live network I/O: on a cache miss this issues an HTTPS GET to the
    hardcoded ``BASE_URL`` host (StatsBomb open-data on raw.githubusercontent.com)
    and mirrors the response to the local cache before returning it. A cache hit
    reads from disk and does no network access.

    Args:
        rel_path: Path of the file relative to the repo's ``data/`` directory
            (e.g. ``"competitions.json"`` or ``"events/3869685.json"``).

    Returns:
        The parsed JSON payload (a list or dict, matching the file's contents).
    """
    cached = _cache_dir() / rel_path
    if cached.is_file():
        with open(cached) as f:
            return json.load(f)
    url = f"{BASE_URL}/{rel_path}"
    # The URL host is fixed (BASE_URL, https), so the S310 audit-for-untrusted-URL
    # lint does not apply; suppress it rather than route through a wrapper.
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https host
        payload = json.loads(resp.read().decode("utf-8"))
    cached.parent.mkdir(parents=True, exist_ok=True)
    # Non-atomic write: an interrupted or concurrent run can leave a truncated
    # JSON file that fails to parse on every later load. A temp-file-then-rename
    # would be more robust; delete the offending cache file to recover.
    with open(cached, "w") as f:
        json.dump(payload, f)
    return payload


def list_competitions() -> pd.DataFrame:
    """List every (competition, season) available in the open data.

    Returns:
        pd.DataFrame: The parsed ``competitions.json`` table, one row per
        (competition, season) with the open-data schema's columns (including
        ``competition_id``, ``season_id``, ``competition_name``, ``season_name``).
    """
    return pd.DataFrame(_get_json("competitions.json"))


def load_matches(competition_name: str, season_name: str) -> pd.DataFrame:
    """Load the match list for one competition-season, flattened for the pipeline.

    Args:
        competition_name: Open-data competition name (e.g. ``"FIFA World Cup"``).
        season_name: Open-data season name (e.g. ``"2022"``).

    Returns:
        pd.DataFrame: One row per match, sorted by ``match_date``, with columns
        ``match_id, match_date, home, away, score_home, score_away, stage``.

    Raises:
        ValueError: If the (competition, season) pair is not present in the open
            data; the message lists the season names available for that competition.
    """
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
        # home_team/away_team and their *_name keys are guaranteed by the
        # open-data match schema, so subscript directly; softer fields use .get().
        "home": m["home_team"]["home_team_name"],
        "away": m["away_team"]["away_team_name"],
        "score_home": m.get("home_score"),
        "score_away": m.get("away_score"),
        "stage": _normalize_stage((m.get("competition_stage") or {}).get("name", "")),
    } for m in raw])
    return out.sort_values("match_date").reset_index(drop=True)


def _normalize_stage(stage: str) -> str:
    """Map an open-data stage name onto the index's stage conventions.

    Args:
        stage: Raw ``competition_stage.name`` from the match record (may be empty).

    Returns:
        str: The lower-cased, stripped stage name, with ``"Group Stage"``/``"group"``
        collapsed to ``"group"``. The knockout flag tests equality against the
        literal ``"group"``, so this exact spelling is what downstream code keys on.
    """
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
    """Walk a tuple key path into nested dicts, returning ``np.nan`` if any level is missing.

    Args:
        obj: The root object (typically one event dict).
        path: Sequence of keys to follow in order (e.g. ``("pass", "outcome", "name")``).

    Returns:
        The value at the end of the path, or ``np.nan`` if any intermediate value
        is not a dict or any key along the way is absent.
    """
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return np.nan
        obj = obj[key]
    return obj


def load_events(match_id: int) -> pd.DataFrame:
    """Load one match's events as the wide frame the measures expect.

    Args:
        match_id: StatsBomb match id.

    Returns:
        pd.DataFrame: One row per event, with the flat columns declared in
        ``_FIELDS`` (nested source fields dug out via :func:`_dig`).

    Open data carries no on-ball-value columns, so the three OBV measures return
    ``nan`` and drop out of their family mean — the documented "extended tier"
    behavior.
    """
    raw = _get_json(f"events/{int(match_id)}.json")
    rows = [{col: _dig(e, path) for col, path in _FIELDS} for e in raw]
    return pd.DataFrame(rows)


def load_elo(path: str | None = None) -> pd.DataFrame:
    """Load the Elo table used for pre-match team strength.

    Args:
        path: Optional path to an Elo CSV. Defaults to the snapshot shipped at
            ``data/elo.csv`` (sourced from eloratings.net; regenerate with
            ``scripts/fetch_elo.py``).

    Returns:
        pd.DataFrame: The Elo table (columns ``team, elo``) indexed by team name,
        so callers can look up a rating with ``df.loc[team, "elo"]``.
    """
    p = Path(path) if path else Path(__file__).resolve().parents[2] / "data" / "elo.csv"
    df = pd.read_csv(p)
    return df.set_index("team")
