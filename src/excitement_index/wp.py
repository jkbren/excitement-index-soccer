"""The win-probability model and its derived quantities.

The index's drama measures read a per-match **win-probability curve**: at every
event, the probabilities (p_home, p_draw, p_away) of the final result. The model:

* The remaining-match goal margin is the difference of two Poisson processes
  (a **Skellam** distribution): each side scores at rate λ = r × minutes-left.
* Per-side rates come from a log-linear **Elo** model (μ = 1.3·e^(±adv/2) goals
  per 90, adv = Elo gap / 200, with a ±50-point host edge); without Elo both
  sides get the symmetric prior of 1.35 goals per 90.

  Two "no-information" priors coexist on purpose and should not be unified. The
  Elo path centers on a 2.6-goal match (base = 1.3 per side) because it was
  calibrated against Elo-implied totals; the no-Elo curve prior uses 1.35 per side
  (2.7 total), the plain historical mean goals-per-team used when no rating is
  available. They answer different questions (rating-implied vs unconditional), so
  the small difference is expected, not a constant waiting to be merged.
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

# Symmetric no-Elo prior: 1.35 goals per team per 90 minutes, expressed per team-minute
# so it multiplies directly by minutes-left. The 2.7-goal total is the unconditional
# historical mean; the Elo path instead centers on 2.6 (see module docstring).
PRIOR_GOAL_RATE = 1.35 / 90.0
# Pseudo-minutes for the empirical-Bayes blend in the xG-updating variant: weight on the
# live np-xG rate is t/(t+TAU), so at t=30' prior and data are weighted equally. Unused by
# the shipped goals-only index.
TAU_MINUTES = 30.0
# Floor on elapsed minutes when forming live per-minute rates, so the first-minute divide
# does not blow up (1.0 = one minute).
MIN_ELAPSED = 1.0
# Maximum fractional tilt applied to a trailing side's scoring rate late in a match.
CHASE = 0.25

WP_COLS = ["_t", "period", "d", "p_home", "p_draw", "p_away", "cum_xg_home", "cum_xg_away"]

# numpy renamed trapz -> trapezoid in 2.0; support both (the fallback only
# evaluates on numpy < 2.0, where trapz still exists).
trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201


# ---------------------------------------------------------------------------
# Pre-match probabilities from Elo
# ---------------------------------------------------------------------------
def pregame_goal_rates(elo_home: float, elo_away: float, *, hfa_elo: float = 0.0,
                       total_goals: float = 2.6):
    """Per-side expected full-match goals from an Elo gap, log-linear.

    Args:
        elo_home: Home side's Elo rating (points).
        elo_away: Away side's Elo rating (points).
        hfa_elo: Home-field advantage added to the home rating, in Elo points
            (e.g. +50). Defaults to 0.0.
        total_goals: Expected combined goals for an even match; split evenly as the
            per-side base. Defaults to 2.6 (Elo-calibrated total).

    Returns:
        A ``(mu_home, mu_away)`` tuple of expected goals over the full 90, each
        floored at 0.02 so no side ever has a zero Poisson mean.

    The Elo gap is converted to a symmetric log advantage adv = (Elo_home −
    Elo_away + hfa)/200; the /200 scale sets how many goals a rating gap buys, and
    the ±adv/2 split moves the two sides in opposite directions around ``base``.
    """
    adv = (float(elo_home) - float(elo_away) + hfa_elo) / 200.0
    base = total_goals / 2.0  # even-match per-side mean (half the expected total)
    mu_h = max(base * float(np.exp(adv / 2.0)), 0.02)   # floor 0.02 keeps the Poisson mean positive
    mu_a = max(base * float(np.exp(-adv / 2.0)), 0.02)
    return mu_h, mu_a


def pregame_outcome_probs(elo_home: float, elo_away: float, *, hfa_elo: float = 0.0,
                          total_goals: float = 2.6) -> dict:
    """Pre-match home/draw/away probabilities from Elo via a Skellam over 90 minutes.

    Args:
        elo_home: Home side's Elo rating (points).
        elo_away: Away side's Elo rating (points).
        hfa_elo: Home-field advantage in Elo points added to the home side.
            Defaults to 0.0.
        total_goals: Expected combined goals for an even match. Defaults to 2.6.

    Returns:
        A dict with ``p_home/p_draw/p_away`` (result probabilities summing to 1),
        the per-side Poisson means ``mu_home/mu_away``, and ``entropy`` — the H/D/A
        Shannon entropy normalized to 0-1 (1 = maximally uncertain three-way).

    The full-match goal margin is home Poisson minus away Poisson, i.e. a Skellam;
    home wins on margin >= 1, draws on margin 0, away wins on margin <= -1.
    """
    mu_h, mu_a = pregame_goal_rates(elo_home, elo_away, hfa_elo=hfa_elo, total_goals=total_goals)
    p_home = float(1 - stats.skellam.cdf(0, mu_h, mu_a))   # P(margin >= 1)
    p_draw = float(stats.skellam.pmf(0, mu_h, mu_a))       # P(margin == 0)
    p_away = float(stats.skellam.cdf(-1, mu_h, mu_a))      # P(margin <= -1)
    # Clip before log so a near-zero probability cannot produce -inf in the entropy.
    p = np.clip(np.array([p_home, p_draw, p_away]), 1e-12, 1.0)
    entropy = float(-(p * np.log(p)).sum() / np.log(3))    # /log(3) normalizes to [0, 1]
    return dict(p_home=p_home, p_draw=p_draw, p_away=p_away,
                mu_home=mu_h, mu_away=mu_a, entropy=entropy)


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------
def wp_curve(events: pd.DataFrame, *, home: str, away: str,
             prior_home: float | None = None, prior_away: float | None = None,
             prior_rate: float = PRIOR_GOAL_RATE, chase: float = CHASE,
             xg_update: bool = False, tau: float = TAU_MINUTES) -> pd.DataFrame:
    """Build the per-event home/draw/away win-probability curve for one match.

    Args:
        events: Raw StatsBomb event frame for the match.
        home: Home team name in the event feed.
        away: Away team name in the event feed.
        prior_home: Home per-team-minute scoring rate; falls back to ``prior_rate``
            when None.
        prior_away: Away per-team-minute scoring rate; falls back to ``prior_rate``
            when None.
        prior_rate: Symmetric fallback rate for either side when its prior is None.
            Defaults to ``PRIOR_GOAL_RATE`` (1.35/90).
        chase: Maximum fractional rate tilt for a trailing side late in the match.
            Defaults to ``CHASE`` (0.25).
        xg_update: If True, blend each side's rate toward its live np-xG-per-minute
            (experimental variant). The shipped index uses False (goals-only).
        tau: Pseudo-minutes for that xG blend; only consulted when ``xg_update``.
            Defaults to ``TAU_MINUTES`` (30).

    Returns:
        A DataFrame with columns ``WP_COLS`` — one row per playable event, in match
        order, preceded by a synthetic kickoff row at ``_t = 0`` carrying the
        neutral pre-match prior. ``d`` is the running home-minus-away goal margin;
        ``cum_xg_home/away`` are cumulative non-penalty xG. Rows are in
        non-decreasing ``_t`` order (relied on by the curve-summary integrals).
        Empty frame if the match has no playable events.

    The remaining-match margin is modeled as a Skellam (difference of two Poissons,
    one per side); each side's Poisson mean is rate × minutes-left. Updates fire on
    goals only — the ``xg_update=True`` branch below is experimental and not
    exercised in production.
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

    # Running score state and xG totals up to and including each event.
    gh = np.cumsum(goal & (team == home))
    ga = np.cumsum(goal & (team == away))
    xh = np.cumsum(np.where(team == home, npxg, 0.0))
    xa = np.cumsum(np.where(team == away, npxg, 0.0))
    d = (gh - ga).astype(int)
    elapsed = np.maximum(t, MIN_ELAPSED)   # floored so the early xG-rate divide is stable
    left = np.maximum(end - t, 0.0)        # minutes remaining, never negative
    if xg_update:
        # Experimental (not shipped): shrink the live np-xG rate toward the prior,
        # with data weight growing as t/(t+tau).
        w = t / (t + tau)
        rate_h = w * (xh / elapsed) + (1 - w) * rh0
        rate_a = w * (xa / elapsed) + (1 - w) * ra0
    else:
        # Shipped path: constant prior rate, so only goals move the curve.
        rate_h = np.full_like(t, rh0, dtype=float)
        rate_a = np.full_like(t, ra0, dtype=float)
    # Poisson means for the remaining match; floored at 1e-6 so scipy always sees a
    # strictly positive rate even at the final whistle.
    lam_h = np.maximum(rate_h * left, 1e-6)
    lam_a = np.maximum(rate_a * left, 1e-6)

    # Chase tilt: trailing side's rate up, leader's down, growing with lateness.
    lateness = np.clip(1 - left / end, 0.0, 1.0)   # 0 at kickoff -> 1 at the final whistle
    mag = np.minimum(np.abs(d), 2)                 # margin drives the tilt, capped at 2 goals
    boost = 1 + chase * lateness * mag
    damp = 1 - 0.5 * chase * lateness * mag         # leader damped at half the trailer's boost
    behind_h, ahead_h = d < 0, d > 0
    lam_h = np.where(behind_h, lam_h * boost, np.where(ahead_h, lam_h * damp, lam_h))
    lam_a = np.where(ahead_h, lam_a * boost, np.where(behind_h, lam_a * damp, lam_a))

    # Default each row to a degenerate point mass on the current scoreline; this is the
    # value kept for the last <0.05 minutes (see `live` below), where the result is settled.
    p_home = np.where(d > 0, 1.0, 0.0)
    p_draw = np.where(d == 0, 1.0, 0.0)
    p_away = np.where(d < 0, 1.0, 0.0)
    # "Live" rows still have >0.05 minutes left; the same 0.05-minute cutoff is applied in
    # per_shot_leverage. Below it we treat the outcome as decided rather than evaluate a Skellam.
    live = left > 0.05
    if live.any():
        # k = -d = goals the home side currently trails by. With S = future home-minus-away
        # Skellam margin, home wins overall iff S > k, the match draws iff S == k.
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
    # Synthetic kickoff row at t=0: neutral 0-0 state with Poisson means over the whole match.
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
    """The (H, D, A) probability matrix of a curve as an ``N x 3`` float array."""
    return curve[["p_home", "p_draw", "p_away"]].to_numpy(float)


def tv_steps(curve: pd.DataFrame) -> np.ndarray:
    """Per-step total-variation movement of the outcome-probability vector.

    Args:
        curve: A win-probability curve (as returned by :func:`wp_curve`).

    Returns:
        A length ``N-1`` array; each entry is 0.5·‖Δp‖₁ between consecutive rows —
        the total-variation distance the (H, D, A) vector moved on that step. Empty
        array for a curve with fewer than two rows.
    """
    return 0.5 * np.abs(np.diff(_p3(curve), axis=0)).sum(axis=1) if len(curve) > 1 else np.array([])


def gei_tv(curve: pd.DataFrame) -> float:
    """Total probability-needle travel over the whole match (a Game Excitement Index).

    Args:
        curve: A win-probability curve.

    Returns:
        The summed per-step total-variation movement (0.0 for an empty curve).
    """
    if curve is None or curve.empty:
        return 0.0
    return float(tv_steps(curve).sum())


def entropy_area(curve: pd.DataFrame) -> float:
    """Time-averaged normalized outcome entropy: how sustained the uncertainty was.

    Args:
        curve: A win-probability curve; its ``_t`` column must be non-decreasing
            (as produced by :func:`wp_curve`) for the time integral to be valid.

    Returns:
        The mean over match time of the H/D/A entropy normalized to 0-1: a uniform
        (1/3, 1/3, 1/3) integrates to 1, a near-certain result to near 0. Falls back
        to the plain row mean when there is a single row or zero time span. 0.0 for
        an empty curve.

    Averaging over time (a trapezoid integral divided by the time span, not a row
    mean) is what distinguishes a knife-edge 0-0 held all match from a dead one.
    """
    if curve is None or curve.empty:
        return 0.0
    t = curve["_t"].to_numpy(float)
    P = np.clip(_p3(curve), 1e-12, 1.0)   # clip before log so a 0 probability is finite
    H = (-(P * np.log(P)).sum(axis=1)) / np.log(3)   # /log(3) normalizes three-way entropy to [0, 1]
    if len(t) < 2:
        return float(H.mean())
    span = float(t[-1] - t[0])
    return float(trapezoid(H, t) / span) if span > 0 else float(H.mean())


def best_window_tv(curve: pd.DataFrame, *, minutes: float) -> float:
    """Largest total-variation movement packed into any fixed-length time window.

    Args:
        curve: A win-probability curve; its ``_t`` column must be non-decreasing so
            each window is a contiguous run of steps.
        minutes: Window length in minutes.

    Returns:
        The maximum accumulated per-step total-variation movement inside any window
        of ``minutes`` starting at a step's timestamp (0.0 for an empty or
        movement-free curve).

    Each window is anchored at a step time and spans ``[a, a + minutes)``; scanning
    over every anchor finds the most eventful spell of that length.
    """
    if curve is None or curve.empty:
        return 0.0
    mid = curve["_t"].to_numpy(float)[1:]   # step times align with tv_steps (drop the kickoff row)
    s = tv_steps(curve)
    if not len(s):
        return 0.0
    best = 0.0
    for a in mid:
        best = max(best, float(s[(mid >= a) & (mid < a + minutes)].sum()))
    return best


def lead_changes_from_curve(curve: pd.DataFrame) -> int:
    """Number of times the scoreline leader changed hands.

    Args:
        curve: A win-probability curve carrying the running margin column ``d``.

    Returns:
        The count of sign flips of the goal margin, with zeros dropped first, so an
        equalizer between two leads is not counted (1-0 → 1-1 → 1-2 is ONE lead
        change, home-to-away). 0 for fewer than two nonzero-margin rows.
    """
    sign = np.sign(curve["d"].to_numpy(int))
    nz = sign[sign != 0]   # drop level-scoreline rows so equalizers are not lead changes
    return int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0


def xg_lead_changes_from_curve(curve: pd.DataFrame) -> int:
    """Number of times the cumulative non-penalty-xG leader changed ("deserved" lead).

    Args:
        curve: A win-probability curve carrying ``cum_xg_home`` and ``cum_xg_away``.

    Returns:
        The count of crossings of the home-minus-away cumulative xG difference
        (sign flips with zeros dropped). 0 for fewer than two nonzero-difference
        rows.
    """
    # Round to 6 dp so floating-point noise near equal xG totals does not read as a crossing.
    diff = np.round((curve["cum_xg_home"] - curve["cum_xg_away"]).to_numpy(float), 6)
    sign = np.sign(diff)
    nz = sign[sign != 0]
    return int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0


def comeback_magnitude_from_curve(curve: pd.DataFrame) -> float:
    """Largest win-probability recovery by a side that actually trailed on the scoreline.

    Args:
        curve: A win-probability curve carrying ``p_home``, ``p_draw`` and the
            running margin ``d``.

    Returns:
        The maximum rise in a side's two-outcome win probability (wp = p_home +
        p_draw/2 for the home view, its complement for the away view), measured only
        from moments when that side was behind on the scoreline. Ranges 0 to ~0.5.
        0.0 for an empty curve.

    Requiring the side to have actually trailed on the SCORELINE keeps a probability
    blip during a blowout from reading as a phantom fightback. The measure is
    symmetric (both sides checked) and result-agnostic (the trailing side need not
    have completed the comeback).
    """
    if curve is None or curve.empty:
        return 0.0
    wp_h = (curve["p_home"] + 0.5 * curve["p_draw"]).to_numpy(float)   # home 2-outcome win prob
    d = curve["d"].to_numpy(int)
    best = 0.0
    # Check both sides: (home wp while home trailed) and (away wp while away trailed).
    for side_wp, behind in ((wp_h, d < 0), (1.0 - wp_h, d > 0)):
        run_min = np.inf   # lowest wp seen so far while this side was behind
        for i in range(len(side_wp)):
            if behind[i]:
                run_min = min(run_min, side_wp[i])
            # Recovery = current wp minus the trough reached while trailing.
            if np.isfinite(run_min):
                best = max(best, side_wp[i] - run_min)
    return float(max(best, 0.0))


# ---------------------------------------------------------------------------
# Per-shot leverage (the anticipation/resolution machinery)
# ---------------------------------------------------------------------------
def per_shot_leverage(ev: pd.DataFrame, *, home: str, away: str, end: float,
                      prior_home: float | None = None,
                      prior_away: float | None = None) -> pd.DataFrame:
    """Counterfactual win-probability swing and leverage for every shot in a match.

    Args:
        ev: Playable-event frame for the match.
        home: Home team name in the event feed.
        away: Away team name in the event feed.
        end: End-of-play minute (horizon for minutes-left).
        prior_home: Home per-team-minute scoring rate; defaults to
            ``PRIOR_GOAL_RATE`` when None.
        prior_away: Away per-team-minute scoring rate; defaults to
            ``PRIOR_GOAL_RATE`` when None.

    Returns:
        One row per shot with columns ``t`` (minute), ``team``, ``xg`` (np-xG),
        ``is_goal``, ``is_save``, ``d`` (home-minus-away margin at the shot's
        instant), ``swing`` (0.5·‖Δp‖₁ of the H/D/A vector IF this shot had scored),
        and ``leverage`` (``xg × swing``). Empty frame if there are no shots or the
        xG column is absent.

    Uses the same Skellam as the curve but with plain λ = rate × minutes-left
    (floored at 0.05 minutes, the same live cutoff as :func:`wp_curve`) and NO chase
    tilt. ``swing`` is the POTENTIAL move — the probability shift a goal at this
    moment would have caused — not the realized move: it is computed the same way
    whether or not the shot actually scored, so it measures the leverage of the
    chance itself. Downstream sums should treat ``leverage`` as expected/potential
    movement, not as win probability that actually changed hands.
    """
    shots = ev[ev["type"] == "Shot"]
    cols = ["t", "team", "xg", "is_goal", "is_save", "d", "swing", "leverage"]
    if shots.empty or "shot_statsbomb_xg" not in shots.columns:
        return pd.DataFrame(columns=cols)
    ph = prior_home if prior_home is not None else PRIOR_GOAL_RATE
    pa = prior_away if prior_away is not None else PRIOR_GOAL_RATE
    # Sorted goal times per side, used to reconstruct the score state at each shot's instant.
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
        # (p_home, p_draw, p_away) given current margin dd and remaining Poisson means
        # (lh, la); -dd is the goals home trails by, matching the Skellam mapping in wp_curve.
        return np.array([1 - stats.skellam.cdf(-dd, lh, la),
                         stats.skellam.pmf(-dd, lh, la),
                         stats.skellam.cdf(-dd - 1, lh, la)])

    swing = np.zeros(len(shots))
    d_arr = np.zeros(len(shots), int)
    for i in range(len(shots)):
        left = max(end - st[i], 0.05)   # minutes-left, floored at the 0.05-min live cutoff
        lh, la = max(ph * left, 1e-6), max(pa * left, 1e-6)
        # Score margin just before this shot: goals each side scored strictly earlier
        # ("left" side of searchsorted excludes a goal at the shot's own timestamp).
        d = int(np.searchsorted(gh_t, st[i], "left") - np.searchsorted(ga_t, st[i], "left"))
        d_arr[i] = d
        pre = wp3(d, lh, la)
        # Counterfactual: margin moves +1 for a home shot, -1 for an away shot, had it scored.
        post = wp3(d + 1, lh, la) if steam[i] == home else wp3(d - 1, lh, la)
        swing[i] = 0.5 * float(np.abs(post - pre).sum())   # total-variation distance of the shift
    return pd.DataFrame({"t": st, "team": steam, "xg": sx, "is_goal": sgoal,
                         "is_save": ssave, "d": d_arr, "swing": swing,
                         "leverage": sx * swing})
