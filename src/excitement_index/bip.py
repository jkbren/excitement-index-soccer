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
from typing import Any, Dict, List, Optional

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
_GENERIC_OUT = frozenset({"out", "lost out", "success out", "punched out"})
# Non-pass/non-shot outcome fields checked for out-like values (the original
# normalizer carries exactly these three onto its internal event table).
_GENERIC_OUTCOME_COLS = ("goalkeeper_outcome", "duel_outcome", "interception_outcome")


def _isna(v: Any) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _norm(v: Any) -> Optional[str]:
    """Casefold, hyphens -> spaces, collapse whitespace; None for missing."""
    if _isna(v):
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    out = " ".join(s.casefold().replace("-", " ").split())
    return out or None


def _num(v: Any) -> Optional[float]:
    if _isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool(v: Any) -> Optional[bool]:
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
    dur: Optional[float]
    is_restart: bool
    is_admin: bool
    is_boundary: bool
    is_half_end: bool
    is_ambiguous: bool          # pass with outcome Unknown (mid-flight stop)
    is_clear_final_stop: bool   # non-ambiguous stop marker, final-gap eligible


def _make_view(rec: "Dict[str, Any]") -> _View:
    etn = _norm(rec.get("type"))
    ptn = _norm(rec.get("pass_type"))
    pon = _norm(rec.get("pass_outcome"))
    stn = _norm(rec.get("shot_type"))
    son = _norm(rec.get("shot_outcome"))
    is_restart = (etn == "referee ball drop"
                  or (etn == "pass" and ptn in _RESTART_PASS)
                  or (etn == "shot" and stn in _RESTART_SHOT))
    is_ambiguous = etn == "pass" and pon == "unknown"
    advantage = (_bool(rec.get("foul_committed_advantage")) is True
                 or _bool(rec.get("foul_won_advantage")) is True)
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


def _event_end(v: _View, next_t: Optional[float], period_end: Optional[float]) -> float:
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


def _period_end(views: "List[_View]") -> float:
    """Half-End timestamp (max, if several) or the last finite event time."""
    he = [v.t for v in views if v.is_half_end and v.t == v.t]
    if he:
        return max(he)
    ts = [v.t for v in views if v.t == v.t]
    return max(ts) if ts else 0.0


def _dead_intervals(views: "List[_View]", period_end: float) -> "List[float]":
    """Per-interval dead seconds (each rounded to ms, as the original records them)."""
    out: "List[float]" = []
    restarts = [i for i, v in enumerate(views) if v.is_restart]
    for ri in restarts[1:]:              # a period's opening restart is never dead time
        restart = views[ri]
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


def match_ball_in_play(events: "pd.DataFrame") -> "Optional[Dict[str, Any]]":
    """Per-match ball-in-play timing (total + per period) from an event frame.

    ``events`` is the wide event frame (:func:`opendata.load_events`); it must
    carry ``timestamp``, ``duration``, ``type``, ``period`` and the restart /
    stop-marker columns for the estimate to be meaningful. Extra time is
    included when periods 3/4 are present; the shootout is always excluded.
    Returns ``None`` (never raises) when no estimate can be produced.
    """
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
        by_period: "Dict[int, List[_View]]" = {}
        for per, _t, _i, rec in keyed:
            if per == per and per != float("inf") and int(per) in PERIODS:
                by_period.setdefault(int(per), []).append(_make_view(rec))

        total_elapsed = 0.0
        period_ends: "Dict[int, float]" = {}
        dead_by_period: "Dict[int, List[float]]" = {}
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
                "label": PERIOD_LABELS.get(period, "Period %d" % period),
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
        return None
