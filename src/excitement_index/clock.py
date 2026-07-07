"""Match-clock utilities and event-frame conventions.

Every measure in the index shares these conventions:

* **Playable periods.** Only periods 1-4 (regulation + extra time) enter any
  computation. Penalty shootouts (period 5) are excluded everywhere except the
  dedicated ``shootout_drama`` measure, which reads period 5 explicitly.
* **The clock.** ``_t`` is the absolute StatsBomb minute plus fractional
  seconds (the second half starts at 45, so stoppage-time minutes count).
  The match end is 120.0 if any extra-time event exists, else 90.0 — and never
  earlier than the last observed event.
* **Goals.** A goal is a shot with outcome ``Goal`` or an ``Own Goal For``
  event, attributed to the event's team.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

PLAYABLE = (1, 2, 3, 4)      # regulation + extra time; shootout (5) excluded
REG_END = 90.0
ET_END = 120.0

# Penalty box on StatsBomb's 120 x 80 pitch (attacking left -> right).
BOX_X, BOX_Y_LO, BOX_Y_HI = 102.0, 18.0, 62.0

# Shot outcomes counted as "on target".
SOT_OUTCOMES = {"Goal", "Saved", "Saved To Post"}


def playable_events(events: pd.DataFrame) -> pd.DataFrame:
    """Playable events (regulation + ET, shootout dropped) with an absolute-minute
    ``_t`` column, sorted in match order."""
    ev = events[events["period"].isin(PLAYABLE)].copy()
    if "index" in ev.columns:
        ev = ev.sort_values(["period", "index"])
    else:
        ev = ev.sort_values(["period", "minute", "second"])
    ev["_t"] = ev["minute"].astype(float) + ev["second"].fillna(0).astype(float) / 60.0
    return ev.reset_index(drop=True)


def match_end_minute(ev: pd.DataFrame) -> float:
    """End-of-play minute: 120 if extra time was played, else 90 — never earlier
    than the last event seen (so deep stoppage time extends the clock)."""
    if ev.empty:
        return REG_END
    hard = ET_END if ev["period"].isin((3, 4)).any() else REG_END
    return float(max(hard, ev["_t"].max()))


def is_goal(ev: pd.DataFrame) -> pd.Series:
    """Boolean mask: shot-goals plus own goals (``Own Goal For`` rows belong to
    the team the goal counts FOR)."""
    return ((ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")) | (ev["type"] == "Own Goal For")


def np_xg_row(ev: pd.DataFrame) -> np.ndarray:
    """Per-row non-penalty xG (0 for non-shots and penalty kicks)."""
    is_shot = (ev["type"] == "Shot").to_numpy()
    if "shot_type" in ev.columns:
        pen = (ev["shot_type"] == "Penalty").to_numpy()
    else:
        pen = np.zeros(len(ev), bool)
    xg = (ev["shot_statsbomb_xg"].fillna(0.0).to_numpy(float)
          if "shot_statsbomb_xg" in ev.columns else np.zeros(len(ev)))
    return np.where(is_shot & ~pen, xg, 0.0)


def xy(series: pd.Series) -> np.ndarray:
    """N x 2 array from a column of ``[x, y]`` location lists/arrays."""
    return np.array([[float(v[0]), float(v[1])] for v in series])


def ball_in_play_seconds(ev: pd.DataFrame, period: int, lo: float, hi: float,
                         *, dead_gap: float = 25.0) -> float:
    """Estimated ball-in-play seconds inside ``[lo, hi)`` of ``period``: the sum
    of inter-event gaps of at most ``dead_gap`` seconds (longer gaps are treated
    as dead-ball time)."""
    seg = ev[(ev["period"] == period) & (ev["_t"] >= lo) & (ev["_t"] < hi)]
    t = np.sort(seg["_t"].to_numpy(float)) * 60.0
    if len(t) < 2:
        return 0.0
    gaps = np.diff(t)
    return float(gaps[gaps <= dead_gap].sum())


def period_bounds(ev: pd.DataFrame, period: int):
    """(first, last) event minute observed in ``period``."""
    t = ev.loc[ev["period"] == period, "_t"]
    return (float(t.min()), float(t.max())) if len(t) else (np.nan, np.nan)


def resolve_team_name(name: str | None, ev_teams: Sequence[str]) -> str | None:
    """Map a fixture-sheet team name onto the name the event feed actually uses.

    An unresolved name silently corrupts every team-keyed measure (goals and xG
    attribute to nobody), so we try, in order: exact match, a small alias table,
    an accent-stripped case-insensitive match, and finally a fuzzy match."""
    if name is None or name in ev_teams:
        return name
    aliases = {"Türkiye": "Turkey", "Turkiye": "Turkey"}
    ali = aliases.get(str(name))
    if ali in ev_teams:
        return ali
    import difflib
    import unicodedata

    def strip(s: str) -> str:
        return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()

    for t in ev_teams:
        if strip(t) == strip(name):
            return t
    close = difflib.get_close_matches(str(name), [str(t) for t in ev_teams], n=1, cutoff=0.75)
    return close[0] if close else name


def goal_times(ev: pd.DataFrame, team: str, end: float | None = None) -> np.ndarray:
    """Sorted goal times (minutes) for ``team``, capped at ``end`` when given."""
    g = ev[is_goal(ev)]
    t = np.sort(g.loc[g["team"] == team, "_t"].to_numpy(float))
    return np.minimum(t, end) if end is not None else t


def margin_walk(ev: pd.DataFrame, home: str, away: str, end: float):
    """The score-margin step function: ``(times, margins)`` where ``margins[i]``
    is the home-minus-away goal margin on the interval ``[times[i], times[i+1])``.
    ``times`` starts at 0.0 and ends at ``end``; goal times are capped at ``end``."""
    goals = ev[is_goal(ev)].sort_values("_t")
    times, margins, d = [0.0], [0], 0
    for _, g in goals.iterrows():
        d += 1 if g["team"] == home else (-1 if g["team"] == away else 0)
        times.append(min(float(g["_t"]), end))
        margins.append(d)
    times.append(end)
    return times, margins
