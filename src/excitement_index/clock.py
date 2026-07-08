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

PLAYABLE = (1, 2, 3, 4)      # StatsBomb period ids for regulation + extra time; shootout (5) excluded
REG_END = 90.0               # end-of-regulation minute
ET_END = 120.0               # end-of-extra-time minute (both 15-min ET halves always played)

# Penalty box on StatsBomb's 120 x 80 pitch (attacking left -> right):
# x >= 102 is inside the 18-yard box; y in [18, 62] is its vertical span.
BOX_X, BOX_Y_LO, BOX_Y_HI = 102.0, 18.0, 62.0

# Shot outcomes counted as "on target" (forced the keeper or found the net).
SOT_OUTCOMES = {"Goal", "Saved", "Saved To Post"}


def playable_events(events: pd.DataFrame) -> pd.DataFrame:
    """Filter to playable-period events and attach the absolute-minute clock column.

    Args:
        events: Raw StatsBomb event frame with ``period``, ``minute``, ``second``,
            and (optionally) ``index`` columns.

    Returns:
        A copy holding only periods 1-4 (regulation + extra time, shootout dropped),
        sorted in match order, with a new float ``_t`` column (absolute minute plus
        fractional seconds) and a reset index.

    When present, ``index`` (StatsBomb's monotonic within-match event counter) is the
    primary sort key because it orders same-timestamp events correctly; otherwise the
    code falls back to sorting by (period, minute, second).
    """
    ev = events[events["period"].isin(PLAYABLE)].copy()
    if "index" in ev.columns:
        ev = ev.sort_values(["period", "index"])
    else:
        ev = ev.sort_values(["period", "minute", "second"])
    ev["_t"] = ev["minute"].astype(float) + ev["second"].fillna(0).astype(float) / 60.0
    return ev.reset_index(drop=True)


def match_end_minute(ev: pd.DataFrame) -> float:
    """Minute at which play ends, used as the horizon for minutes-left in the model.

    Args:
        ev: Playable-event frame (already carrying ``period`` and ``_t``).

    Returns:
        The end-of-play minute as a float: 120 if any extra-time event exists,
        else 90 — but never earlier than the last observed event, so deep
        stoppage time extends the clock.

    Any period-3 or period-4 event implies the full 120-minute clock: World Cup
    knockouts always play both 15-minute extra-time halves (no golden goal), so
    a single ET event fixes the horizon at 120 regardless of when play stopped.
    """
    if ev.empty:
        return REG_END
    hard = ET_END if ev["period"].isin((3, 4)).any() else REG_END
    return float(max(hard, ev["_t"].max()))


def is_goal(ev: pd.DataFrame) -> pd.Series:
    """Boolean mask marking rows that count as a goal.

    Args:
        ev: Event frame with ``type`` and ``shot_outcome`` columns.

    Returns:
        A boolean Series, True on shots with outcome ``Goal`` plus every
        ``Own Goal For`` row.

    StatsBomb records an own goal as two rows; ``Own Goal For`` is the one already
    credited to the team the goal counts FOR, so keying on it attributes correctly.
    """
    return ((ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")) | (ev["type"] == "Own Goal For")


def np_xg_row(ev: pd.DataFrame) -> np.ndarray:
    """Per-row non-penalty expected goals.

    Args:
        ev: Event frame; uses ``type``, and when present ``shot_type`` and
            ``shot_statsbomb_xg``.

    Returns:
        A float array, one entry per row: the shot's StatsBomb xG for open-play /
        set-piece shots, and 0.0 for non-shots and for penalty kicks.

    Penalties are excluded because their fixed ~0.79 xG would otherwise dominate the
    xG race and the xG-updating rate variant; the index scores run of play only.
    """
    is_shot = (ev["type"] == "Shot").to_numpy()
    if "shot_type" in ev.columns:
        pen = (ev["shot_type"] == "Penalty").to_numpy()
    else:
        pen = np.zeros(len(ev), bool)
    xg = (ev["shot_statsbomb_xg"].fillna(0.0).to_numpy(float)
          if "shot_statsbomb_xg" in ev.columns else np.zeros(len(ev)))
    return np.where(is_shot & ~pen, xg, 0.0)


def xy(series: pd.Series) -> np.ndarray:
    """Stack a column of ``[x, y]`` locations into a coordinate array.

    Args:
        series: Column whose entries are 2-element ``[x, y]`` lists or arrays
            (StatsBomb pitch coordinates on the 120 x 80 grid).

    Returns:
        An ``N x 2`` float array of the coordinates in row order.
    """
    return np.array([[float(v[0]), float(v[1])] for v in series])


def ball_in_play_seconds(ev: pd.DataFrame, period: int, lo: float, hi: float,
                         *, dead_gap: float = 25.0) -> float:
    """Estimate seconds the ball was in play within a minute window of one period.

    Args:
        ev: Playable-event frame carrying ``period`` and ``_t``.
        period: StatsBomb period id to restrict to.
        lo: Window start minute (inclusive).
        hi: Window end minute (exclusive).
        dead_gap: Maximum inter-event gap in seconds still counted as live play;
            longer gaps are treated as dead-ball time. Defaults to 25.0.

    Returns:
        The summed live-play time in seconds (0.0 if fewer than two events fall
        in the window).

    Events fire only while the ball is live, so consecutive-event gaps under the
    cutoff are live seconds and longer gaps span stoppages (throw-ins, fouls,
    subs). The 25 s cutoff is the empirical knee of the gap distribution: it sits
    above the routine few-second spacing of open play but below typical dead-ball
    stoppages, so it separates the two without clipping normal play.
    """
    seg = ev[(ev["period"] == period) & (ev["_t"] >= lo) & (ev["_t"] < hi)]
    t = np.sort(seg["_t"].to_numpy(float)) * 60.0  # convert absolute minutes to seconds
    if len(t) < 2:
        return 0.0
    gaps = np.diff(t)
    return float(gaps[gaps <= dead_gap].sum())


def period_bounds(ev: pd.DataFrame, period: int):
    """First and last observed event minute within a period.

    Args:
        ev: Playable-event frame carrying ``period`` and ``_t``.
        period: StatsBomb period id to inspect.

    Returns:
        A ``(first, last)`` tuple of event minutes, or ``(nan, nan)`` if the
        period has no events.
    """
    t = ev.loc[ev["period"] == period, "_t"]
    return (float(t.min()), float(t.max())) if len(t) else (np.nan, np.nan)


def resolve_team_name(name: str | None, ev_teams: Sequence[str]) -> str | None:
    """Map a fixture-sheet team name onto the name the event feed actually uses.

    Args:
        name: The team name as it appears on the fixture sheet (may be None).
        ev_teams: The distinct team names present in the event feed.

    Returns:
        The matching event-feed name, or None if ``name`` is None. On total match
        failure it returns the ORIGINAL unresolved ``name`` unchanged (see note).

    An unresolved name silently corrupts every team-keyed measure (goals and xG
    attribute to nobody), so resolution is attempted in order of decreasing
    confidence: exact match, a small alias table, an accent-stripped
    case-insensitive match, and finally a difflib fuzzy match. The 0.75 cutoff on
    the fuzzy pass is conservative — high enough to reject unrelated names, low
    enough to bridge spelling/diacritic drift. The final fallback returns ``name``
    itself (a silent passthrough): downstream code then keys on a name absent from
    the feed and quietly attributes nothing, so a caller that cannot tolerate that
    should validate the result against ``ev_teams``.
    """
    if name is None or name in ev_teams:
        return name
    aliases = {"Türkiye": "Turkey", "Turkiye": "Turkey"}
    ali = aliases.get(str(name))
    if ali in ev_teams:
        return ali
    # difflib/unicodedata are imported lazily here: this is a cold fallback path
    # (most names hit the exact match above), so we avoid paying the import at
    # module load for every caller.
    import difflib
    import unicodedata

    def strip(s: str) -> str:
        # NFKD-decompose then drop non-ASCII so accented letters compare equal to
        # their plain forms (e.g. "Côte d'Ivoire" vs "Cote d'Ivoire").
        return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()

    for t in ev_teams:
        if strip(t) == strip(name):
            return t
    # Last resort: nearest name by sequence-similarity ratio, accepted only at >=0.75.
    close = difflib.get_close_matches(str(name), [str(t) for t in ev_teams], n=1, cutoff=0.75)
    return close[0] if close else name


def goal_times(ev: pd.DataFrame, team: str, end: float | None = None) -> np.ndarray:
    """Sorted goal minutes for one team.

    Args:
        ev: Playable-event frame carrying ``team`` and ``_t``.
        team: Team name (as used in the event feed) to select goals for.
        end: Optional end-of-play minute; goal times are clamped to at most this.

    Returns:
        A sorted float array of goal minutes for ``team`` (clamped to ``end`` when
        given).
    """
    g = ev[is_goal(ev)]
    t = np.sort(g.loc[g["team"] == team, "_t"].to_numpy(float))
    return np.minimum(t, end) if end is not None else t


def margin_walk(ev: pd.DataFrame, home: str, away: str, end: float):
    """Build the running home-minus-away score margin as a step function.

    Args:
        ev: Playable-event frame carrying ``team`` and ``_t``.
        home: Home team name in the event feed.
        away: Away team name in the event feed.
        end: End-of-play minute; goal times are capped at this and it terminates
            the step function.

    Returns:
        A ``(times, margins)`` pair. ``times`` starts at 0.0 and ends at ``end``;
        ``margins[i]`` is the home-minus-away goal margin holding over the interval
        ``[times[i], times[i+1])``. Own goals credited to neither ``home`` nor
        ``away`` leave the margin unchanged.
    """
    goals = ev[is_goal(ev)].sort_values("_t")
    times, margins, d = [0.0], [0], 0
    for _, g in goals.iterrows():
        d += 1 if g["team"] == home else (-1 if g["team"] == away else 0)
        times.append(min(float(g["_t"]), end))
        margins.append(d)
    times.append(end)
    return times, margins
