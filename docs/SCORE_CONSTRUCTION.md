# The NetSI match excitement rating: score construction
*(v1.4 · NetSI Sport for Northeastern Global News · July 2026)*

## Overview

The rating is a weighted linear index computed from match event data. For each match, the
pipeline (1) computes approximately fifty per-match measures from the event record,
(2) standardizes each measure against a fixed tournament benchmark, (3) combines the
standardized measures into a raw score by weighted averaging, (4) applies two rule-based
deductions for the match's competitive context, and (5) maps the raw score onto a 0–10
publication scale. No component of the score is fitted to fan ratings or any other
preference data. All weights, benchmarks, and parameters are frozen; the daily update
re-scores new matches but never re-fits.

## 1. Input data

The only input to the score is StatsBomb event data: a timestamped record of every on-ball
action in the match (approximately 3,500 events per game), including passes, carries,
shots with expected-goals (xG) values, saves, duels, and cards, each with pitch
coordinates. Pre-match team strength is taken from Elo world ratings. Fan ratings
(seriesgraph.com) and The Athletic's match ranking are used for validation comparisons
only and do not enter the computation.

## 2. Feature computation

Each match is reduced to ~50 scalar measures, organized into the five ingredient buckets
of the published decomposition — Stakes, Chances, Drama, Spectacle, and Payoff — each
built from smaller sub-families of related measures (eleven sub-families in total; the
grouping is listed in the technical appendix). Measures range from direct counts (shots
on target, completed dribbles, goals after the 80th minute) to model-based quantities:
several are derived from a win-probability curve — a Poisson/Skellam scoring model that
updates the home/draw/away probabilities at each goal, with team-strength priors from Elo
ratings — including the total variation of that curve over the match (the "game
excitement index"), the time the outcome remained uncertain, and the value of chances
weighted by the probability swing they would have caused had they scored. The full list
of measures, definitions, and thresholds is given in the technical appendix.

## 3. Standardization

Each measure is converted to a z-score against a fixed reference distribution: the 72
group-stage matches of this World Cup. For measure *i* with reference mean μᵢ and
standard deviation σᵢ, a match's value xᵢ becomes zᵢ = (xᵢ − μᵢ)/σᵢ, clipped to the
range [−3, +3]. The clip bounds the influence of any single extreme statistic.
Binary indicators without meaningful variance in the reference pool (e.g., the
host-nation flag) bypass the z-score and enter at a fixed bounded scale. A small number
of measures carry negative sign by construction (cards, red cards, sterile-possession
share, dead-air stretches), so that larger values reduce the score.

The reference distribution is frozen: every match in the tournament, including the
knockout rounds, is standardized against the same 72-game benchmark. This makes ratings
stable — a published score never changes when later matches are added.

## 4. Aggregation

The raw score is a weighted average computed in two stages. First, within each of the
eleven sub-families, member z-scores are averaged with equal weight — this prevents a
sub-family with many measures from outweighing one with few. Second, the sub-family means
are combined as a weighted sum with fixed weights, which roll up to the five published
buckets: **Stakes 24%** (matchup closeness .048 + competition stakes .157 + upset .039),
**Spectacle 23%** (individual brilliance .084 + game flow .115 + controversy .030),
**Chances 20%** (chance creation .139 + goalkeeping .066), **Drama 20%** (back-and-forth
.098 + timing .101), and **Payoff 12%** (resolution .124). Because the index is linear,
the decomposition is exact: the five bucket values published for each match sum to its
raw score.

## 5. Context deductions

Two deductions are applied to the raw composite. Both are deterministic functions with a
single parameter each, and both are reported as line items in the published decomposition.

**Dead-rubber deduction.** For each group-stage match we estimate qualification jeopardy
*j* ∈ [0, 1]: the difference between a team's probability of advancing if it wins versus
if it loses, computed by Monte-Carlo simulation of the remaining group fixtures (knockout
matches have *j* = 1 by definition). The deduction is multiplicative in match quality:
a match keeps a fraction 1 − 0.40·(1 − *j*) of its positive composite score. A match with
nothing at stake (*j* ≈ 0) therefore retains ~60% of its quality score; a knockout match
is untouched. Matches with negative composite scores are not adjusted.

**Aliveness deduction (knockout matches only).** We compute an endgame-aliveness value
*A* ∈ [0, 1] as the mean of two quantities: the fraction of the match elapsed before the
score margin moved beyond one goal for good, and the share of the final 30 minutes of
playing time spent within one goal. The deduction is 0.60·(1 − *A*), floored so that no
match is pushed below the median raw score of the evaluation pool. A knockout match that
remained within one goal throughout (including all extra-time matches, by construction)
has *A* = 1 and receives no deduction.

## 6. Publication scale

The adjusted raw score is mapped to 0–10 by a monotone piecewise function: seven quantile
anchors align the raw-score distribution of the reference pool with the corresponding
quantiles of the fan-rating distribution (this calibrates the *scale*, not the *ordering*
— a monotone map cannot change any match's rank), with linear interpolation between
anchors. Beyond the top anchor the map follows a slope-matched exponential approach to 10;
below the bottom anchor, a slope-matched exponential approach to 0. Both endpoints are
asymptotes: neither 0.0 nor 10.0 is attainable. Published ratings are rounded to two
decimals.

## 7. Weight selection and validation

The family weights were selected by constrained search rather than by fitting to a target.
20,016 candidate weight vectors were drawn and evaluated against a battery specified in
advance: eight hard face-validity constraints (e.g., a scoreless mutual-qualification
draw must rank in the bottom 15%; five consensus knockout classics must rank in the top
11%; a 19-save scoreless draw must rank above average; a 7–1 result must rank below
median); a requirement of loose agreement with crowd ratings (Spearman correlation with
group-stage fan ratings within [0.35, 0.75] — a plausibility band, not an optimization
target); and a robustness requirement (the constraints must continue to hold under ±10%
random perturbation of the weights; the selected vector survives 97% of perturbations).
As an out-of-sample check, the frozen index applied to the 2022 World Cup ranks the
Argentina–France final in the top three of 64 matches. The two deduction parameters
(0.40 and 0.60) were subsequently set as the smallest values satisfying additional
pre-registered rank constraints. All amendments to the index are versioned and documented;
the current version is v1.4.

## 8. Properties

- **Not a trained model.** No parameter is estimated by regression on fan ratings or any
  preference data. (Preliminary work established that models fitted to fan ratings load
  almost entirely on goal count; the index is intended as an independent definition of
  on-field tension.)
- **Deterministic and reproducible.** Given the event data and the published constants,
  every rating can be recomputed exactly.
- **Exactly decomposable.** Bucket contributions and both deductions sum to the raw score.
- **Stable.** Benchmarks, weights, parameters, and the publication map are frozen; new
  matches are scored without altering existing ratings.

*Data: StatsBomb (event data), eloratings.net (team strength). Fan ratings
(seriesgraph.com) and The Athletic's match ranking are comparison references only.
Analysis: NetSI Sport for Northeastern Global News.*
