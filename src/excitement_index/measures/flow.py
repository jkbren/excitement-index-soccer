"""Flow measures — intensity, transitions, tempo, and the threat stream.

The "did the game *move*" family: how fast the ball circulated, how much play
swung between the ends, how much of the clock the ball was actually alive, how
often turnovers turned into counters and pressing won the ball back — plus the
on-ball-value threat stream (extended tier) and the longest spark-free lull.
"""
from __future__ import annotations

import numpy as np

from .. import bip
from ..clock import ball_in_play_seconds, is_goal, match_end_minute, period_bounds
from ..wp import tv_steps
from .registry import measure


@measure("tempo", tier="core")
def tempo(ctx) -> float:
    """Completed passes per minute of ball-in-play, regulation time (in-play
    time from inter-event gaps, with gaps > 25 s counted as dead ball)."""
    et = ctx.ev[ctx.ev["period"].isin((1, 2))]
    if et.empty:
        return float(np.nan)
    bip_min = 0.0
    for p in (1, 2):
        lo, hi = period_bounds(et, p)
        if np.isfinite(hi):
            bip_min += ball_in_play_seconds(et, p, lo, hi) / 60.0
    if bip_min <= 0:
        return float(np.nan)
    completed = int(((et["type"] == "Pass") & (et["pass_outcome"].isna())).sum())
    return completed / bip_min


@measure("end_to_end", tier="core")
def end_to_end(ctx) -> float:
    """Standard deviation across possessions of each possession's mean field
    position — play swinging between ends."""
    ev = ctx.ev
    sub = ev[ev["location"].notna() & ev["possession"].notna()]
    if sub.empty:
        return float(np.nan)
    xs = sub["location"].apply(lambda v: float(v[0]))
    cent = xs.groupby(sub["possession"]).mean()
    return float(cent.std()) if len(cent) > 1 else float(np.nan)


@measure("bip_share", tier="core")
def bip_share(ctx) -> float:
    """Ball-in-play share of elapsed time (statsbomb_bip v3 estimator)."""
    try:
        r = bip.match_ball_in_play(ctx.events_all)
        return float(r["total"]["bip_pct"]) / 100.0 if r else float(np.nan)
    except Exception:
        return float(np.nan)


@measure("counter_shots", tier="core")
def counter_shots(ctx) -> float:
    """Shots arising from counter-attacks (play pattern *From Counter*)."""
    shots = ctx.shots
    if len(shots) and "play_pattern" in shots.columns:
        return float((shots["play_pattern"] == "From Counter").sum())
    return 0.0


@measure("counterpress", tier="core")
def counterpress(ctx) -> float:
    """Pressing actions within 5 s of a turnover (StatsBomb counterpress flag)."""
    ev = ctx.ev
    if "counterpress" not in ev.columns:
        return 0.0
    return float((ev["counterpress"] == True).sum())  # noqa: E712 - NaN-safe elementwise


@measure("ball_recoveries", tier="core")
def ball_recoveries(ctx) -> float:
    """Ball Recovery events."""
    return float((ctx.ev["type"] == "Ball Recovery").sum())


# ---------------------------------------------------------------------------
# The OBV threat stream (extended tier: needs the subscription-feed
# ``obv_for_net`` column; open data has none, so these return nan and drop
# out of their family mean). Shots are excluded so the stream doesn't
# double-count the xG/chance measures.
# ---------------------------------------------------------------------------
def _obv_threat(ctx) -> dict:
    """30-second-binned positive non-shot on-ball value, computed once per match."""
    if "obv_threat" in ctx.cache:
        return ctx.cache["obv_threat"]
    bin_seconds = 30.0
    out = dict(obv_end_to_end=np.nan, obv_peak=np.nan,
               obv_volatility=np.nan, obv_flat_share=np.nan)
    ev = ctx.ev
    if "obv_for_net" in ev.columns:
        ns = ev[(ev["type"] != "Shot") & ev["obv_for_net"].notna() & ev["team"].notna()]
        if not ns.empty:
            obv = ns["obv_for_net"].to_numpy(float)
            team = ns["team"].to_numpy(object)
            t = ns["_t"].to_numpy(float)
            pos_h = float(obv[(team == ctx.home) & (obv > 0)].sum())
            pos_a = float(obv[(team == ctx.away) & (obv > 0)].sum())
            end = match_end_minute(ev)
            nb = max(int(np.ceil(end / (bin_seconds / 60.0))), 1)
            binidx = np.clip((t / (bin_seconds / 60.0)).astype(int), 0, nb - 1)
            pos_bin = np.zeros(nb)
            np.add.at(pos_bin, binidx, np.where(obv > 0, obv, 0.0))
            out = dict(obv_end_to_end=2 * pos_h * pos_a / (pos_h + pos_a + 1e-9),
                       obv_peak=float(pos_bin.max()),
                       obv_volatility=float(pos_bin.std()),
                       obv_flat_share=float((pos_bin < 0.01).mean()))
    ctx.cache["obv_threat"] = out
    return out


@measure("obv_end_to_end", tier="extended")
def obv_end_to_end(ctx) -> float:
    """Harmonic mean of the two teams' total positive non-shot on-ball value —
    high only when *both* sides generated threat. Extended tier."""
    return float(_obv_threat(ctx)["obv_end_to_end"])


@measure("obv_peak", tier="extended")
def obv_peak(ctx) -> float:
    """Largest 30-second bin of positive non-shot OBV — the match's most
    dangerous passage. Extended tier."""
    return float(_obv_threat(ctx)["obv_peak"])


@measure("obv_flat_share", tier="extended")
def obv_flat_share(ctx) -> float:
    """Fraction of 30-second bins with negligible (< 0.01) OBV — sterile,
    threat-free time. Extended tier; signed negative in the composite."""
    return float(_obv_threat(ctx)["obv_flat_share"])


@measure("dead_air", tier="core")
def dead_air(ctx) -> float:
    """Longest stretch (minutes) with no "spark": no goal, no shot of
    xG >= 0.07, no completed take-on, no WP step >= 0.02. Signed negative in
    the composite — 'nothing happened for 25 minutes'."""
    ev, c3, shots, end = ctx.ev, ctx.wp, ctx.shots, ctx.end
    sparks = [0.0, float(end)]
    g = ev[is_goal(ev)]
    if len(g):
        sparks += [float(t) for t in g["_t"]]
    if len(shots) and "shot_statsbomb_xg" in shots.columns:
        sparks += [float(t) for t in
                   shots.loc[shots["shot_statsbomb_xg"].fillna(0) >= 0.07, "_t"]]
    if "dribble_outcome" in ev.columns:
        d = ev[(ev["type"] == "Dribble") & (ev["dribble_outcome"] == "Complete")]
        if len(d):
            sparks += [float(t) for t in d["_t"]]
    if len(c3) > 1:
        steps = tv_steps(c3)
        mid = c3["_t"].to_numpy(float)[1:]
        sparks += [float(t) for t in mid[steps >= 0.02]]
    s = np.sort(np.unique(np.asarray(sparks, float)))
    return float(np.diff(s).max()) if len(s) > 1 else float(end)
