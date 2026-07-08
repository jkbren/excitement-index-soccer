"""Ball-in-play timing: the ``statsbomb_bip`` v3 estimator, trimmed to one variant.

This is a faithful port of the exact call chain the private ``bip_share``
measure used — the standalone ``statsbomb_bip`` package's restart-anchored
``v3_duration_finalgap`` estimator (duration correction + final dead gap),
reduced to only what that variant's *number* needs. The full package also
carries three sibling variants, an independent stop-marker cross-check, and a
large QA-flag surface; none of those affect the v3 BIP percentage, so they are
deliberately not ported (the returned dict therefore has no ``range`` key).

How the estimate works, per period (regulation 1-2, plus extra time 3-4 when
present; the penalty shootout, period 5, is always excluded):

* Every *restart* event (a Referee Ball-Drop; a pass of type Corner / Free
  Kick / Goal Kick / Kick Off / Throw-in; a direct set-piece shot) closes a
  dead interval that began when the previous meaningful event *ended* —
  ``timestamp + duration`` where a positive duration exists (capped at the
  restart and the period end), else the event's start time. A pass with
  outcome ``Unknown`` (a foul called mid-flight) never trusts its duration.
  A period's opening restart is never counted as dead time; administrative
  and period-boundary events are skipped when pairing backward.
* If a period's last meaningful event is a *clear* stop marker (ball out,
  goal, offside, injury stoppage, own goal, foul without advantage, an
  out-like outcome) with no restart before the Half End, the trailing gap to
  the period end is dead time too (the "final dead gap").
* Ball-in-play = elapsed (sum of Half-End times) minus total dead time.

Durations are **decimal seconds** (never divided by 1000), and timestamps are
within-period ``HH:MM:SS.mmm`` strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

VARIANT = "v3_duration_finalgap"
PERIODS = (1, 2, 3, 4)          # shootout (5) excluded, extra time included

PERIOD_LABELS = {
    1: "First half",
    2: "Second half",
    3: "Extra time — first half",
    4: "Extra time — second half",
}

# Normalized (casefolded, hyphen -> space) StatsBomb names, per the licensed
# Event Data Specification the original package was grounded in.
_ADMIN = frozenset({"starting xi", "tactical shift", "substitution", "bad behaviour",
                    "player off", "player on", "camera on", "camera off"})
_RESTART_PASS = frozenset({"corner", "free kick", "goal kick", "kick off", "throw in"})
_RESTART_SHOT = frozenset({"corner", "free kick", "kick off", "penalty"})
_STOP_EVENTS = frozenset({"offside", "injury stoppage", "own goal against", "own goal for"})
_PASS_STOP = frozenset({"out", "injury clearance", "pass offside"})
# Out-like outcome tokens (normalized) that mark the ball leaving play. Each token
# comes from a different outcome column: plain "out" appears on goalkeeper_outcome /
# interception_outcome; "lost out" and "success out" are duel_outcome values (a duel
# lost/won that put the ball out); "punched out" is a goalkeeper_outcome value.
_GENERIC_OUT = frozenset({"out", "lost out", "success out", "punched out"})
# Non-pass/non-shot outcome fields checked for out-like values (the original
# normalizer carries exactly these three onto its internal event table).
_GENERIC_OUTCOME_COLS = ("goalkeeper_outcome", "duel_outcome", "interception_outcome")


def _isna(v: Any) -> bool:
    """True for None or a float NaN (``v != v``); used to gate the coercers below."""
    return v is None or (isinstance(v, float) and v != v)


def _norm(v: Any) -> str | None:
    """Casefold, hyphens -> spaces, collapse whitespace; None for missing."""
    if _isna(v):
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    out = " ".join(s.casefold().replace("-", " ").split())
    return out or None


def _num(v: Any) -> float | None:
    """Coerce a value to float, or None when missing or non-numeric.

    Args:
        v: any cell value (numeric, string, None, or NaN).

    Returns:
        The value as a float, or None if it is missing or cannot be parsed.
    """
    if _isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool(v: Any) -> bool | None:
    """Coerce a value to a tristate bool, or None when the truth value is unknown.

    Args:
        v: any cell value. Real bools pass through; strings are matched
            case-insensitively against a small true/false vocabulary
            ("true"/"1"/"yes"/"t" vs "false"/"0"/"no"/"f"/"").

    Returns:
        True/False when the value resolves cleanly, or None when it is missing
        or is a string outside the recognized vocabulary. The tristate matters:
        callers test ``is True`` so an unknown flag never counts as set.
    """
    if _isna(v):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().casefold()
        if s in ("true", "1", "yes", "t"):
            return True
        if s in ("false", "0", "no", "f", ""):
            return False
        return None
    try:
        return bool(v)
    except Exception:
        return None


def parse_timestamp(ts: Any) -> float:
    """Within-period StatsBomb timestamp ("HH:MM:SS.mmm") -> seconds (NaN if missing)."""
    if _isna(ts):
        return float("nan")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return float(ts)
    s = str(ts).strip()
    if not s or s.lower() == "nan":
        return float("nan")
    parts = s.split(":")
    if len(parts) == 3:
        h, m, rest = parts
    elif len(parts) == 2:
        h, m, rest = "0", parts[0], parts[1]
    else:
        h, m, rest = "0", "0", parts[0]
    return int(h) * 3600 + int(m) * 60 + float(rest)


@dataclass
class _View:
    """One pre-classified event (only the attributes the v3 number depends on)."""

    t: float
    dur: float | None
    is_restart: bool
    is_admin: bool
    is_boundary: bool
    is_half_end: bool
    is_ambiguous: bool          # pass with outcome Unknown (mid-flight stop)
    is_clear_final_stop: bool   # non-ambiguous stop marker, final-gap eligible


def _make_view(rec: dict[str, Any]) -> _View:
    """Pre-classify one raw event record into the flags the v3 estimate reads.

    Args:
        rec: one event row as a dict (a single record from the wide event frame),
            carrying at least ``timestamp``, ``duration``, ``type`` and the
            pass/shot/foul/outcome columns referenced below.

    Returns:
        A :class:`_View` whose flags drive dead-time pairing:

        * ``is_restart`` — the event puts the ball back in play (a Referee
          Ball-Drop; a pass of type Corner/Free Kick/Goal Kick/Kick Off/Throw-in;
          a direct set-piece shot). Restarts close the dead interval that
          preceded them.
        * ``is_ambiguous`` — a pass with outcome Unknown, i.e. a foul called
          mid-flight. Such a pass is a stop but its duration cannot be trusted,
          so it is excluded from ``is_clear_final_stop`` (see below) and its
          end time falls back to its start in :func:`_event_end`.
        * ``is_admin`` — a non-play bookkeeping event (lineup/tactics/subs/cards/
          camera) that is transparently skipped when pairing backward.
        * ``is_boundary`` — a Half Start / Half End marker, also skipped when
          pairing backward.
        * ``is_half_end`` — the Half End marker, used to fix the period end time.
        * ``is_clear_final_stop`` — a stop marker that is trustworthy as the
          period's final event (ball out, goal, offside/injury/own-goal event,
          a foul without advantage, an out-like pass or generic outcome). It
          excludes ambiguous Unknown passes precisely because their end time is
          untrustworthy, which would misplace the final dead gap.

    A foul only counts as a stop when the referee did NOT play advantage
    (``foul_committed_advantage`` / ``foul_won_advantage``), since an advantage
    means play continued.
    """
    etn = _norm(rec.get("type"))
    ptn = _norm(rec.get("pass_type"))
    pon = _norm(rec.get("pass_outcome"))
    stn = _norm(rec.get("shot_type"))
    son = _norm(rec.get("shot_outcome"))
    # A restart is any event that puts the ball back into play.
    is_restart = (etn == "referee ball drop"
                  or (etn == "pass" and ptn in _RESTART_PASS)
                  or (etn == "shot" and stn in _RESTART_SHOT))
    # Pass cut off mid-flight by a foul: a stop, but its duration is unreliable.
    is_ambiguous = etn == "pass" and pon == "unknown"
    # Referee waved play on: the foul did not actually stop the ball.
    advantage = (_bool(rec.get("foul_committed_advantage")) is True
                 or _bool(rec.get("foul_won_advantage")) is True)
    # Any signal that the ball went dead on this event.
    is_stop = (_bool(rec.get("out")) is True
               or (etn == "shot" and son == "goal")
               or etn in _STOP_EVENTS
               or (etn == "foul committed" and not advantage)
               or pon in _PASS_STOP
               or pon == "unknown"
               or any(_norm(rec.get(c)) in _GENERIC_OUT for c in _GENERIC_OUTCOME_COLS))
    return _View(
        t=parse_timestamp(rec.get("timestamp")),
        dur=_num(rec.get("duration")),
        is_restart=is_restart,
        is_admin=etn in _ADMIN,
        is_boundary=etn in ("half start", "half end"),
        is_half_end=etn == "half end",
        is_ambiguous=is_ambiguous,
        is_clear_final_stop=is_stop and not is_ambiguous,
    )


def _event_end(v: _View, next_t: float | None, period_end: float | None) -> float:
    """End time under v3's duration policy: ``t + duration`` when a positive,
    trustworthy duration exists (never for an ambiguous mid-flight pass),
    capped so it can't run past the next restart or the period end."""
    start = v.t
    dur = v.dur
    if v.is_ambiguous or dur is None or dur != dur or dur <= 0:
        end = start
    else:
        end = start + dur
    caps = [c for c in (next_t, period_end) if c is not None and c == c]
    if caps:
        cap = min(caps)
        if end > cap:
            end = max(start, cap)
    return end


def _period_end(views: list[_View]) -> float:
    """Half-End timestamp (max, if several) or the last finite event time."""
    he = [v.t for v in views if v.is_half_end and v.t == v.t]
    if he:
        return max(he)
    ts = [v.t for v in views if v.t == v.t]
    return max(ts) if ts else 0.0


def _dead_intervals(views: list[_View], period_end: float) -> list[float]:
    """Dead-time seconds for one period, one entry per counted dead interval.

    Args:
        views: the period's events, in play order, already classified by
            :func:`_make_view`.
        period_end: the period's end time in seconds (from :func:`_period_end`).

    Returns:
        A list of dead-interval lengths in seconds, each rounded to milliseconds
        (3 dp, matching how the original package records them). One entry per
        restart that closed a gap, plus at most one trailing "final dead gap".
    """
    out: list[float] = []
    restarts = [i for i, v in enumerate(views) if v.is_restart]
    for ri in restarts[1:]:              # a period's opening restart is never dead time
        restart = views[ri]
        # Walk back to the previous meaningful event, skipping only admin and
        # boundary rows. The walk intentionally stops at (and pairs against) a
        # preceding restart: two adjacent restarts are a valid meaningful pair,
        # so the gap between them is a counted dead interval, not a bug.
        j = ri - 1
        while j >= 0 and (views[j].is_admin or views[j].is_boundary):
            j -= 1
        if j < 0:
            continue
        dead_start = _event_end(views[j], next_t=restart.t, period_end=period_end)
        dead_end = restart.t
        if dead_end == dead_end and dead_start == dead_start:
            dead = 0.0 if dead_end < dead_start else dead_end - dead_start
        else:
            dead = 0.0
        out.append(round(dead, 3))
    # Final dead gap: a clear stop with no restart before the period end.
    if period_end == period_end:
        # Find the last meaningful event of the period. Skip NaN-timestamp,
        # admin and boundary rows; the ``+ 1e-9`` is float slack so an event
        # landing exactly on period_end is not skipped by rounding jitter.
        last = None
        for v in reversed(views):
            if v.t != v.t or v.t > period_end + 1e-9 or v.is_admin or v.is_boundary:
                continue
            last = v
            break
        if last is not None and last.is_clear_final_stop:
            final_start = _event_end(last, next_t=None, period_end=period_end)
            if period_end > final_start:
                out.append(round(period_end - final_start, 3))
    return out


def match_ball_in_play(events: pd.DataFrame) -> dict[str, Any] | None:
    """Per-match ball-in-play timing (total + per period) from an event frame.

    Args:
        events: the wide event frame (:func:`opendata.load_events`). It must
            carry ``timestamp``, ``duration``, ``type``, ``period`` and the
            restart / stop-marker columns for the estimate to be meaningful.
            Rows are re-sorted internally by (period, timestamp, index), so
            input order does not matter. Extra time is included when periods
            3/4 are present; the shootout (period 5) is always excluded.

    Returns:
        A dict, or None when no estimate can be produced (empty frame, no
        usable periods, or zero elapsed time). The dict has:

        * ``variant`` — the estimator id (:data:`VARIANT`).
        * ``total`` — ``{"bip_s", "elapsed_s", "bip_pct"}`` across all counted
          periods; ``bip_pct`` is ball-in-play seconds as a percent of elapsed.
        * ``per_period`` — one such block per period, plus ``period`` and a
          human ``label``; a period's ``bip_pct`` is None when its elapsed is 0.

    Seconds are decimal (never divided by 1000) and all outputs are rounded to
    milliseconds (3 dp). This function never raises: any failure returns None.
    """
    # Wrapped so the estimator degrades to "no estimate" (None) rather than
    # propagating. Trade-off: this also swallows genuine programming errors
    # (e.g. a schema/column regression), which then look like a benign None
    # instead of surfacing — worth knowing when a match unexpectedly has no BIP.
    try:
        if events is None or len(events) == 0:
            return None
        if ("id" in events.columns and events["id"].notna().any()
                and events["id"].duplicated().any()):
            events = events.drop_duplicates(subset="id", keep="first")
        recs = events.to_dict("records")
        keyed = []
        for rec in recs:
            per = _num(rec.get("period"))
            t = parse_timestamp(rec.get("timestamp"))
            idx = _num(rec.get("index"))
            keyed.append((
                per if per is not None else float("inf"),
                t if t == t else float("inf"),
                idx if idx is not None else float("inf"),
                rec,
            ))
        keyed.sort(key=lambda k: k[:3])          # stable: (period, t, index)
        by_period: dict[int, list[_View]] = {}
        for per, _t, _i, rec in keyed:
            if per == per and per != float("inf") and int(per) in PERIODS:
                by_period.setdefault(int(per), []).append(_make_view(rec))

        total_elapsed = 0.0
        period_ends: dict[int, float] = {}
        dead_by_period: dict[int, list[float]] = {}
        for period in PERIODS:
            views = by_period.get(period)
            if not views:
                continue
            pe = _period_end(views)
            period_ends[period] = pe
            if pe == pe:
                total_elapsed += pe
            dead_by_period[period] = _dead_intervals(views, pe)

        if not period_ends or not round(total_elapsed, 3):
            return None
        dead = round(sum(d for ds in dead_by_period.values() for d in ds), 3)
        bip = round(total_elapsed - dead, 3)
        bip_pct = round(100.0 * bip / total_elapsed, 3)

        per_period = []
        for period in sorted(period_ends):
            elapsed = round(period_ends[period], 3)
            pdead = round(sum(dead_by_period.get(period, [])), 3)
            pbip = round(elapsed - pdead, 3)
            per_period.append({
                "period": period,
                "label": PERIOD_LABELS.get(period, f"Period {period}"),
                "bip_s": pbip,
                "elapsed_s": elapsed,
                "bip_pct": round(100.0 * pbip / elapsed, 3) if elapsed > 0 else None,
            })
        return {
            "variant": VARIANT,
            "total": {"bip_s": bip, "elapsed_s": round(total_elapsed, 3), "bip_pct": bip_pct},
            "per_period": per_period,
        }
    except Exception:
        # See the note at the top of the try: any error is reported as "no estimate".
        return None
