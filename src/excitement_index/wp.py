"""The win-probability model and its derived quantities.

The index's drama measures read a per-match **win-probability curve**: at every
event, the probabilities (p_home, p_draw, p_away) of the final result. The model:

* The remaining-match goal margin is the difference of two Poisson processes
  (a **Skellam** distribution): each side scores at rate λ = r × minutes-left.
* Per-side rates come from a log-linear **Elo** model (μ = 1.3·e^(±adv/2) goals
  per 90, adv = Elo gap / 200, with a ±50-point host edge); without Elo both
  sides get the symmetric prior of 1.35 goals per 90.
* The curve updates **on goals only** (a validated design decision — updating
  on xG worsened outcome prediction in calibration, so shot quality lives in
  the chance measures instead). Red cards do not update the curve.
* A gentle **chase tilt** boosts a trailing side's rate late (and damps the
  leader's), scaled by lateness and margin.
* Only playable periods enter; the curve approaches certainty naturally as
  minutes-left → 0 (no artificial final-whistle jump).

Also here: the **per-shot leverage** machinery — the same Skellam evaluated at
each shot's moment (no chase tilt), giving each shot's counterfactual
probability swing had it scored. Several chance/timing/resolution measures are
sums over these per-shot quantities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .clock import is_goal, match_end_minute, np_xg_row, playable_events

PRIOR_GOAL_RATE = 1.35 / 90.0    # symmetric fallback prior: goals per team-minute
TAU_MINUTES = 30.0               # empirical-Bayes pseudo-minutes (xG-updating variant only)
MIN_ELAPSED = 1.0                # floor on elapsed minutes for live rates
CHASE = 0.25                     # max fractional λ tilt for a side trailing late

WP_COLS = ["_t", "period", "d", "p_home", "p_draw", "p_away", "cum_xg_home", "cum_xg_away"]

# numpy renamed trapz -> trapezoid in 2.0; support both (the fallback only
# evaluates on numpy < 2.0, where trapz still exists).
trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201


# ---------------------------------------------------------------------------
# Pre-match probabilities from Elo
# ---------------------------------------------------------------------------
def pregame_goal_rates(elo_home: float, elo_away: float, *, hfa_elo: float = 0.0,
                       total_goals: float = 2.6):
    """Per-side expected goals (μ_home, μ_away) from an Elo gap, log-linear:
    μ = (total/2)·e^(±adv/2) with adv = (Elo_home − Elo_away + hfa)/200, each
    floored at 0.02."""
    adv = (float(elo_home) - float(elo_away) + hfa_elo) / 200.0
    base = total_goals / 2.0
    mu_h = max(base * float(np.exp(adv / 2.0)), 0.02)
    mu_a = max(base * float(np.exp(-adv / 2.0)), 0.02)
    return mu_h, mu_a


def pregame_outcome_probs(elo_home: float, elo_away: float, *, hfa_elo: float = 0.0,
                          total_goals: float = 2.6) -> dict:
    """Pre-match H/D/A probabilities: Elo → per-side Poisson means → Skellam
    over the full 90 minutes. Returns ``p_home/p_draw/p_away``, the per-side
    means ``mu_home/mu_away``, and the normalized outcome entropy (0–1)."""
    mu_h, mu_a = pregame_goal_rates(elo_home, elo_away, hfa_elo=hfa_elo, total_goals=total_goals)
    p_home = float(1 - stats.skellam.cdf(0, mu_h, mu_a))
    p_draw = float(stats.skellam.pmf(0, mu_h, mu_a))
    p_away = float(stats.skellam.cdf(-1, mu_h, mu_a))
    p = np.clip(np.array([p_home, p_draw, p_away]), 1e-12, 1.0)
    entropy = float(-(p * np.log(p)).sum() / np.log(3))
    return dict(p_home=p_home, p_draw=p_draw, p_away=p_away,
                mu_home=mu_h, mu_away=mu_a, entropy=entropy)


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------
def wp_curve(events: pd.DataFrame, *, home: str, away: str,
             prior_home: float | None = None, prior_away: float | None = None,
             prior_rate: float = PRIOR_GOAL_RATE, chase: float = CHASE,
             xg_update: bool = False, tau: float = TAU_MINUTES) -> pd.DataFrame:
    """The 3-outcome win-probability curve, one row per playable event plus a
    synthetic kickoff row carrying the neutral pre-match prior.

    The index uses ``xg_update=False`` (goals-only). The xG-updating variant is
    kept for experimentation: it blends each side's prior rate toward its live
    np-xG-per-minute with weight t/(t+tau).
    """
    ev = playable_events(events)
    if ev.empty:
        return pd.DataFrame(columns=WP_COLS)
    end = match_end_minute(ev)
    t = ev["_t"].to_numpy(float)
    team = ev["team"].to_numpy(object)
    goal = is_goal(ev).to_numpy(bool)
    npxg = np_xg_row(ev)
    rh0 = prior_rate if prior_home is None else prior_home
    ra0 = prior_rate if prior_away is None else prior_away

    gh = np.cumsum(goal & (team == home))
    ga = np.cumsum(goal & (team == away))
    xh = np.cumsum(np.where(team == home, npxg, 0.0))
    xa = np.cumsum(np.where(team == away, npxg, 0.0))
    d = (gh - ga).astype(int)
    elapsed = np.maximum(t, MIN_ELAPSED)
    left = np.maximum(end - t, 0.0)
    if xg_update:
        w = t / (t + tau)
        rate_h = w * (xh / elapsed) + (1 - w) * rh0
        rate_a = w * (xa / elapsed) + (1 - w) * ra0
    else:
        rate_h = np.full_like(t, rh0, dtype=float)
        rate_a = np.full_like(t, ra0, dtype=float)
    lam_h = np.maximum(rate_h * left, 1e-6)
    lam_a = np.maximum(rate_a * left, 1e-6)

    # Chase tilt: trailing side's rate up, leader's down, growing with lateness.
    lateness = np.clip(1 - left / end, 0.0, 1.0)
    mag = np.minimum(np.abs(d), 2)
    boost = 1 + chase * lateness * mag
    damp = 1 - 0.5 * chase * lateness * mag
    behind_h, ahead_h = d < 0, d > 0
    lam_h = np.where(behind_h, lam_h * boost, np.where(ahead_h, lam_h * damp, lam_h))
    lam_a = np.where(ahead_h, lam_a * boost, np.where(behind_h, lam_a * damp, lam_a))

    # Degenerate point mass on the final scoreline once <0.05 minutes remain.
    p_home = np.where(d > 0, 1.0, 0.0)
    p_draw = np.where(d == 0, 1.0, 0.0)
    p_away = np.where(d < 0, 1.0, 0.0)
    live = left > 0.05
    if live.any():
        k = -d[live]
        cdf = stats.skellam.cdf(k, lam_h[live], lam_a[live])
        cdf1 = stats.skellam.cdf(k - 1, lam_h[live], lam_a[live])
        pmf = stats.skellam.pmf(k, lam_h[live], lam_a[live])
        p_home[live] = 1.0 - cdf
        p_draw[live] = pmf
        p_away[live] = cdf1

    out = pd.DataFrame({"_t": t, "period": ev["period"].to_numpy(), "d": d,
                        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
                        "cum_xg_home": xh, "cum_xg_away": xa})
    lam0 = max(rh0 * end, 1e-6), max(ra0 * end, 1e-6)
    base = pd.DataFrame({"_t": [0.0], "period": [1], "d": [0],
                         "p_home": [float(1 - stats.skellam.cdf(0, *lam0))],
                         "p_draw": [float(stats.skellam.pmf(0, *lam0))],
                         "p_away": [float(stats.skellam.cdf(-1, *lam0))],
                         "cum_xg_home": [0.0], "cum_xg_away": [0.0]})
    return pd.concat([base, out], ignore_index=True)


# ---------------------------------------------------------------------------
# Curve summaries used by the drama measures
# ---------------------------------------------------------------------------
def _p3(curve: pd.DataFrame) -> np.ndarray:
    return curve[["p_home", "p_draw", "p_away"]].to_numpy(float)


def tv_steps(curve: pd.DataFrame) -> np.ndarray:
    """Per-step total-variation movement of the (H, D, A) vector:
    0.5 · ‖Δp‖₁ for each consecutive pair of curve rows."""
    return 0.5 * np.abs(np.diff(_p3(curve), axis=0)).sum(axis=1) if len(curve) > 1 else np.array([])


def gei_tv(curve: pd.DataFrame) -> float:
    """Game Excitement Index: the total distance the probability needle traveled
    (sum of the per-step total-variation movements)."""
    if curve is None or curve.empty:
        return 0.0
    return float(tv_steps(curve).sum())


def entropy_area(curve: pd.DataFrame) -> float:
    """Time-averaged normalized H/D/A entropy (0-1): sustained outcome
    uncertainty. A uniform (1/3, 1/3, 1/3) scores 1; a near-certain draw scores
    low — which distinguishes a knife-edge 0-0 from a dead one."""
    if curve is None or curve.empty:
        return 0.0
    t = curve["_t"].to_numpy(float)
    P = np.clip(_p3(curve), 1e-12, 1.0)
    H = (-(P * np.log(P)).sum(axis=1)) / np.log(3)
    if len(t) < 2:
        return float(H.mean())
    span = float(t[-1] - t[0])
    return float(trapezoid(H, t) / span) if span > 0 else float(H.mean())


def best_window_tv(curve: pd.DataFrame, *, minutes: float) -> float:
    """The most gripping spell: the maximum accumulated total-variation movement
    inside any ``minutes``-long window (window anchored at each step midpoint)."""
    if curve is None or curve.empty:
        return 0.0
    mid = curve["_t"].to_numpy(float)[1:]
    s = tv_steps(curve)
    if not len(s):
        return 0.0
    best = 0.0
    for a in mid:
        best = max(best, float(s[(mid >= a) & (mid < a + minutes)].sum()))
    return best


def lead_changes_from_curve(curve: pd.DataFrame) -> int:
    """Times the scoreline leader changed: sign flips of the running goal margin
    (zeros dropped — going 1-0 → 1-1 → 1-2 is ONE lead change)."""
    sign = np.sign(curve["d"].to_numpy(int))
    nz = sign[sign != 0]
    return int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0


def xg_lead_changes_from_curve(curve: pd.DataFrame) -> int:
    """Crossings of the cumulative non-penalty-xG race ("who deserved to lead"
    flips)."""
    diff = np.round((curve["cum_xg_home"] - curve["cum_xg_away"]).to_numpy(float), 6)
    sign = np.sign(diff)
    nz = sign[sign != 0]
    return int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0


def comeback_magnitude_from_curve(curve: pd.DataFrame) -> float:
    """Largest win-probability recovery by a side that actually trailed on the
    SCORELINE (0 to ~0.5, on the 2-outcome view wp = p_home + p_draw/2). Gating
    on the scoreline keeps a probability blip in a blowout from reading as a
    phantom fightback. Symmetric and result-agnostic."""
    if curve is None or curve.empty:
        return 0.0
    wp_h = (curve["p_home"] + 0.5 * curve["p_draw"]).to_numpy(float)
    d = curve["d"].to_numpy(int)
    best = 0.0
    for side_wp, behind in ((wp_h, d < 0), (1.0 - wp_h, d > 0)):
        run_min = np.inf
        for i in range(len(side_wp)):
            if behind[i]:
                run_min = min(run_min, side_wp[i])
            if np.isfinite(run_min):
                best = max(best, side_wp[i] - run_min)
    return float(max(best, 0.0))


# ---------------------------------------------------------------------------
# Per-shot leverage (the anticipation/resolution machinery)
# ---------------------------------------------------------------------------
def per_shot_leverage(ev: pd.DataFrame, *, home: str, away: str, end: float,
                      prior_home: float | None = None,
                      prior_away: float | None = None) -> pd.DataFrame:
    """For every shot: the counterfactual probability swing had it scored.

    Uses the same Skellam with plain λ = rate × minutes-left (floored at 0.05
    minutes; NO chase tilt) and the score state at the shot's instant (a goal
    at exactly the same timestamp is not yet counted). Returns one row per shot:
    ``t, team, xg, is_goal, is_save, d`` (score state) ``, swing`` (0.5·‖Δp‖₁)
    ``, leverage`` (xg × swing). Empty frame if there are no shots or no xG."""
    shots = ev[ev["type"] == "Shot"]
    cols = ["t", "team", "xg", "is_goal", "is_save", "d", "swing", "leverage"]
    if shots.empty or "shot_statsbomb_xg" not in shots.columns:
        return pd.DataFrame(columns=cols)
    ph = prior_home if prior_home is not None else PRIOR_GOAL_RATE
    pa = prior_away if prior_away is not None else PRIOR_GOAL_RATE
    g = ev[is_goal(ev)]
    gh_t = np.sort(g.loc[g["team"] == home, "_t"].to_numpy(float))
    ga_t = np.sort(g.loc[g["team"] == away, "_t"].to_numpy(float))
    sx = shots["shot_statsbomb_xg"].fillna(0.0).to_numpy(float)
    st = shots["_t"].to_numpy(float)
    steam = shots["team"].to_numpy(object)
    sgoal = ((shots["shot_outcome"] == "Goal").to_numpy()
             if "shot_outcome" in shots.columns else np.zeros(len(shots), bool))
    ssave = ((shots["shot_outcome"].isin({"Saved", "Saved To Post"})).to_numpy()
             if "shot_outcome" in shots.columns else np.zeros(len(shots), bool))

    def wp3(dd, lh, la):
        return np.array([1 - stats.skellam.cdf(-dd, lh, la),
                         stats.skellam.pmf(-dd, lh, la),
                         stats.skellam.cdf(-dd - 1, lh, la)])

    swing = np.zeros(len(shots))
    d_arr = np.zeros(len(shots), int)
    for i in range(len(shots)):
        left = max(end - st[i], 0.05)
        lh, la = max(ph * left, 1e-6), max(pa * left, 1e-6)
        d = int(np.searchsorted(gh_t, st[i], "left") - np.searchsorted(ga_t, st[i], "left"))
        d_arr[i] = d
        pre = wp3(d, lh, la)
        post = wp3(d + 1, lh, la) if steam[i] == home else wp3(d - 1, lh, la)
        swing[i] = 0.5 * float(np.abs(post - pre).sum())
    return pd.DataFrame({"t": st, "team": steam, "xg": sx, "is_goal": sgoal,
                         "is_save": ssave, "d": d_arr, "swing": swing,
                         "leverage": sx * swing})
