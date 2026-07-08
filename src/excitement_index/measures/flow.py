"""Flow measures — intensity, transitions, tempo, and the threat stream.

The "did the game move" family: how fast the ball circulated (``tempo``), how
much play swung between the ends (``end_to_end``), how much of the clock the
ball was actually alive (``bip_share``), how often turnovers turned into
counters and pressing won the ball back (``counter_shots``, ``counterpress``,
``ball_recoveries``) — plus the on-ball-value threat stream (extended tier,
subscription feeds only) and the longest spark-free lull (``dead_air``).
"""
from __future__ import annotations

import numpy as np

from .. import bip
from ..clock import ball_in_play_seconds, is_goal, match_end_minute, period_bounds
from ..wp import tv_steps
from .registry import measure


@measure("tempo", tier="core")
def tempo(ctx) -> float:
    """Completed passes per minute of ball-in-play over regulation time.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed.

    Returns:
        Completed passes divided by ball-in-play minutes across periods 1-2,
        as a float; nan when regulation is empty or no in-play time is found.
        Ball-in-play time is reconstructed from inter-event gaps
        (:func:`ball_in_play_seconds`), where any gap > 25 s is treated as
        dead ball — the 25 s dead-gap threshold lives in that clock helper.
    """
    et = ctx.ev[ctx.ev["period"].isin((1, 2))]
    if et.empty:
        return float(np.nan)
    # Accumulate in-play minutes over each regulation half.
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
    """Spread of possession field positions — play swinging between the ends.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed.

    Returns:
        The standard deviation across possessions of each possession's mean x
        (field) position, as a float; nan when there are fewer than two
        located possessions. A large spread means possessions repeatedly moved
        the action from one end to the other.
    """
    ev = ctx.ev
    sub = ev[ev["location"].notna() & ev["possession"].notna()]
    if sub.empty:
        return float(np.nan)
    # Per-possession mean x, then the spread of those centroids.
    xs = sub["location"].apply(lambda v: float(v[0]))
    cent = xs.groupby(sub["possession"]).mean()
    return float(cent.std()) if len(cent) > 1 else float(np.nan)


@measure("bip_share", tier="core")
def bip_share(ctx) -> float:
    """Ball-in-play share of elapsed time (statsbomb_bip v3 estimator).

    Args:
        ctx: The match context; ``ctx.events_all`` is the full event feed
            (including the shootout) passed to the bip estimator.

    Returns:
        The ball-in-play fraction in [0, 1] as a float, or nan when the
        estimator returns nothing or raises.

    The blanket ``except Exception -> nan`` is intentional: on open-data feeds
    the bip estimator can legitimately lack the inputs it needs, and the
    pipeline treats a nan measure as "not available for this match" (it drops
    out of its family mean). The trade-off is that a genuine bug in
    ``bip.match_ball_in_play`` would also surface as nan rather than an error.
    """
    try:
        r = bip.match_ball_in_play(ctx.events_all)
        return float(r["total"]["bip_pct"]) / 100.0 if r else float(np.nan)
    except Exception:
        return float(np.nan)


@measure("counter_shots", tier="core")
def counter_shots(ctx) -> float:
    """Count of shots arising from counter-attacks.

    Args:
        ctx: The match context; ``ctx.shots`` supplies the shot events.

    Returns:
        The number of shots with play pattern *From Counter* as a float, or
        0.0 when there are no shots or no ``play_pattern`` column.
    """
    shots = ctx.shots
    if len(shots) and "play_pattern" in shots.columns:
        return float((shots["play_pattern"] == "From Counter").sum())
    return 0.0


@measure("counterpress", tier="core")
def counterpress(ctx) -> float:
    """Count of counterpressing actions.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed.

    Returns:
        The number of events flagged with StatsBomb's counterpress flag
        (pressing within 5 s of the team's own turnover) as a float, or 0.0
        when the ``counterpress`` column is absent.
    """
    ev = ctx.ev
    if "counterpress" not in ev.columns:
        return 0.0
    return float((ev["counterpress"] == True).sum())  # noqa: E712 - NaN-safe elementwise


@measure("ball_recoveries", tier="core")
def ball_recoveries(ctx) -> float:
    """Count of Ball Recovery events.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed.

    Returns:
        The number of Ball Recovery events as a float.
    """
    return float((ctx.ev["type"] == "Ball Recovery").sum())


# ---------------------------------------------------------------------------
# The OBV threat stream (extended tier: needs the subscription-feed
# ``obv_for_net`` column; open data has none, so these return nan and drop
# out of their family mean). Shots are excluded so the stream doesn't
# double-count the xG/chance measures.
# ---------------------------------------------------------------------------
def _obv_threat(ctx) -> dict:
    """30-second-binned positive non-shot on-ball value, computed once per match.

    Args:
        ctx: The match context; ``ctx.ev`` supplies the event feed and
            ``ctx.cache`` memoizes the result.

    Returns:
        A dict with keys ``obv_end_to_end``, ``obv_peak``, ``obv_volatility``,
        ``obv_flat_share`` (all float, all nan when the ``obv_for_net`` column
        is absent or no non-shot OBV events exist). Shots are excluded so this
        threat stream does not double-count the xG/chance measures.

    Note: ``obv_volatility`` (the per-bin standard deviation) is retained in the
    dict for parity with the reference implementation, but no ``@measure``
    exports it — it is a computed-but-unused field kept so the recipe matches
    byte-for-byte.
    """
    if "obv_threat" in ctx.cache:
        return ctx.cache["obv_threat"]
    # Bin width for the threat time series, in seconds.
    bin_seconds = 30.0
    out = dict(obv_end_to_end=np.nan, obv_peak=np.nan,
               obv_volatility=np.nan, obv_flat_share=np.nan)
    ev = ctx.ev
    if "obv_for_net" in ev.columns:
        # Non-shot on-ball-value events with a known team.
        ns = ev[(ev["type"] != "Shot") & ev["obv_for_net"].notna() & ev["team"].notna()]
        if not ns.empty:
            obv = ns["obv_for_net"].to_numpy(float)
            team = ns["team"].to_numpy(object)
            t = ns["_t"].to_numpy(float)
            # Total positive (threat-creating) OBV per team.
            pos_h = float(obv[(team == ctx.home) & (obv > 0)].sum())
            pos_a = float(obv[(team == ctx.away) & (obv > 0)].sum())
            end = match_end_minute(ev)
            # Number of 30 s bins spanning the match; bin_seconds/60 = bin width in minutes.
            nb = max(int(np.ceil(end / (bin_seconds / 60.0))), 1)
            binidx = np.clip((t / (bin_seconds / 60.0)).astype(int), 0, nb - 1)
            pos_bin = np.zeros(nb)
            np.add.at(pos_bin, binidx, np.where(obv > 0, obv, 0.0))
            out = dict(
                # Harmonic mean of the two teams' positive OBV; the 1e-9 guards
                # against a divide-by-zero when both totals are 0.
                obv_end_to_end=2 * pos_h * pos_a / (pos_h + pos_a + 1e-9),
                obv_peak=float(pos_bin.max()),
                obv_volatility=float(pos_bin.std()),
                # Share of bins below 0.01 OBV — effectively threat-free time.
                obv_flat_share=float((pos_bin < 0.01).mean()))
    ctx.cache["obv_threat"] = out
    return out


@measure("obv_end_to_end", tier="extended")
def obv_end_to_end(ctx) -> float:
    """Two-sided threat generation via the harmonic mean of team OBV.

    Args:
        ctx: The match context.

    Returns:
        The harmonic mean of the two teams' total positive non-shot on-ball
        value as a float — high only when both sides generated threat, nan on
        feeds without OBV. Extended tier.
    """
    return float(_obv_threat(ctx)["obv_end_to_end"])


@measure("obv_peak", tier="extended")
def obv_peak(ctx) -> float:
    """The match's single most dangerous 30-second passage.

    Args:
        ctx: The match context.

    Returns:
        The largest 30-second bin of positive non-shot OBV as a float, nan on
        feeds without OBV. Extended tier.
    """
    return float(_obv_threat(ctx)["obv_peak"])


@measure("obv_flat_share", tier="extended")
def obv_flat_share(ctx) -> float:
    """Share of the match that was threat-free.

    Args:
        ctx: The match context.

    Returns:
        The fraction of 30-second bins with negligible (< 0.01) OBV as a
        float, nan on feeds without OBV. Extended tier; signed negative in the
        composite (more sterile time lowers excitement).
    """
    return float(_obv_threat(ctx)["obv_flat_share"])


@measure("dead_air", tier="core")
def dead_air(ctx) -> float:
    """Longest spark-free stretch of the match, in minutes.

    Args:
        ctx: The match context; ``ctx.ev`` (events), ``ctx.wp`` (win-prob
            curve), ``ctx.shots`` (shots), and ``ctx.end`` (end minute) are used.

    Returns:
        The longest gap in minutes between consecutive "sparks" as a float,
        falling back to the full match length when no spark occurs. A spark is
        any of: a goal, a shot of xG >= 0.07, a completed take-on, or a
        win-probability step >= 0.02. Signed negative in the composite (a long
        lull lowers excitement).

    The thresholds isolate genuinely eventful moments: 0.07 xG is roughly a
    half-chance, and a 0.02 WP step is a perceptible swing in the outcome
    probabilities; smaller events do not reset the lull.
    """
    ev, c3, shots, end = ctx.ev, ctx.wp, ctx.shots, ctx.end
    # Seed spark times with the match boundaries so gaps are measured within [0, end].
    sparks = [0.0, float(end)]
    g = ev[is_goal(ev)]
    if len(g):
        sparks += [float(t) for t in g["_t"]]
    if len(shots) and "shot_statsbomb_xg" in shots.columns:
        # Shots worth at least a half-chance (xG >= 0.07).
        sparks += [float(t) for t in
                   shots.loc[shots["shot_statsbomb_xg"].fillna(0) >= 0.07, "_t"]]
    if "dribble_outcome" in ev.columns:
        d = ev[(ev["type"] == "Dribble") & (ev["dribble_outcome"] == "Complete")]
        if len(d):
            sparks += [float(t) for t in d["_t"]]
    if len(c3) > 1:
        # Win-probability steps between consecutive curve points; a step >= 0.02
        # is a perceptible momentum swing. `mid` are the times of the later
        # endpoint of each step (curve times excluding the first point).
        steps = tv_steps(c3)
        mid = c3["_t"].to_numpy(float)[1:]
        sparks += [float(t) for t in mid[steps >= 0.02]]
    # Sorted unique spark times; the largest gap between them is the dead air.
    s = np.sort(np.unique(np.asarray(sparks, float)))
    return float(np.diff(s).max()) if len(s) > 1 else float(end)
