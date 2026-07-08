# Quantifying "Exciting" Soccer Games (or trying to...)

A 0-10 excitement rating for soccer matches, computed from StatsBomb event data. This
repository contains the open implementation of the match excitement index built by NetSI
Sport (Northeastern University Network Science Institute) for Northeastern Global News'
coverage of the 2026 FIFA World Cup.

The rating is a weighted linear composite of approximately 57 per-match measures,
standardized against a tournament benchmark, with two rule-based deductions for the
match's competitive context. No parameter is fitted to fan ratings or other preference
data. Every published score decomposes exactly into five ingredient buckets.

```python
from excitement_index import opendata, build_feature_matrix, score_matches

matches  = opendata.load_matches("FIFA World Cup", "2022")   # StatsBomb open data
features = build_feature_matrix(matches, opendata.load_events, elo=opendata.load_elo())
board    = score_matches(features)                            # best match first
print(board[["home", "away", "rating"]].head(10))
```

Applied to the 2022 World Cup (freely available open data), the index rates the
Argentina-France final 9.59 and ranks it first of the 64 matches. The weights were
selected on 2026 data only; no 2022 match was used in their selection.

## Method

1. **Feature computation.** Each match's event record (~3,500 events) is reduced to ~57
   scalar measures, organized into eleven sub-families. Measures range from counts (shots
   on target, completed dribbles, goals after the 80th minute) to quantities derived from
   a win-probability model: a Poisson/Skellam process over the remaining goal margin,
   updated at each goal, with team-strength priors from Elo ratings. Derived measures
   include the total movement of the win-probability curve, the time-averaged outcome
   uncertainty, and each shot weighted by the probability swing it would have caused had
   it scored. The full catalog with definitions and thresholds is in
   [`docs/TECHNICAL_APPENDIX.md`](docs/TECHNICAL_APPENDIX.md).
2. **Standardization.** Each measure is z-scored against a fixed reference pool (the
   tournament's group stage): z = (x - mean) / (sd + 1e-6), clipped to [-3, +3]. Four
   measures carry negative sign (cards, red cards, sterile-possession share, dead-air
   stretches). Bounded 0/1 flags enter at a fixed scale instead of a z-score.
3. **Aggregation.** Z-scores are averaged with equal weight within each sub-family; the
   sub-family means are combined as a weighted sum. For presentation the sub-families are
   grouped into five buckets: Stakes (24%), Spectacle (23%), Chances (20%), Drama (20%),
   Payoff (12%). Because the composite is linear, the five bucket values sum exactly to
   the raw score.
4. **Context deductions.** (a) A dead-rubber deduction: for group matches, qualification
   jeopardy is estimated by Monte-Carlo simulation of the remaining group fixtures
   (P(advance if win) - P(advance if lose), averaged over both teams); a match retains
   1 - 0.40 x (1 - jeopardy) of its positive composite score. Knockout matches have
   jeopardy 1 and are not deducted. (b) An aliveness deduction, knockout matches only:
   0.60 x (1 - A), where A is the mean of the fraction of the match elapsed before the
   score margin moved beyond one goal for good and the share of the final 30 minutes
   spent within one goal. The deduction is capped so that no match falls below the
   reference pool's median raw score. A knockout match that stayed within one goal
   throughout (including all extra-time matches) has A = 1 and receives no deduction.
5. **Publication scale.** The raw score is mapped to 0-10 by a strictly monotone
   piecewise function: seven quantile anchors align the reference pool's raw-score
   distribution with a target display distribution, with linear interpolation between
   anchors and slope-matched exponential tails approaching 10 above the top anchor and 0
   below the bottom anchor. Both endpoints are asymptotes. Because the map is monotone,
   it does not affect any match's rank.

## Weight selection

The sub-family weights were selected by constrained search rather than by fitting to a
target. 20,016 candidate weight vectors were evaluated against a battery specified in
advance: eight face-validity constraints (for example, a scoreless draw between two
already-qualified teams must rank in the bottom 15%; five consensus knockout classics
must rank in the top 11%; a 19-save scoreless draw must rank above average; a 7-1 result
must rank below median), a requirement of loose agreement with crowd ratings (Spearman
correlation within [0.35, 0.75], used as a plausibility check rather than an optimization
target), and a robustness requirement (the constraints must continue to hold under +-10%
random perturbation of the weights; the selected vector satisfies this in 97% of draws).
Preference data was deliberately excluded from parameter fitting: in preliminary work,
models fitted to fan ratings loaded almost entirely on goal count.

## Configuration and extension

The taxonomy, weights, signs, deduction parameters, and display scale are defined in
[`config/v14.yaml`](config/v14.yaml). Overrides can be passed as a dict or a path:

```python
board = score_matches(features, config={"taxes": {"dead_rubber_k": 0.2}})
```

New measures are added by registering a function and listing it under a sub-family in
the config:

```python
from excitement_index.measures.registry import measure

@measure("woodwork")
def woodwork(ctx):
    """Shots that hit the post or crossbar."""
    return float((ctx.shots["shot_outcome"] == "Post").sum())
```

Three example notebooks cover the common cases: `01_score_a_tournament` (the 2022 World
Cup end to end), `02_adjust_the_weights`, and `03_add_a_measure`.

## Data

- The examples run on [StatsBomb open data](https://github.com/statsbomb/open-data),
  fetched on first use and cached locally. This repository does not redistribute
  StatsBomb event data. The 2026 World Cup event data is a paid feed and is not included
  in any form.
- `data/wc2026_board.csv` contains the published World Cup 2026 board (ratings, bucket
  decompositions, and deductions per match) as a derived output. The table is reproduced
  below.
- `data/elo.csv` is a snapshot of [eloratings.net](https://eloratings.net) world ratings
  (regenerate with `scripts/fetch_elo.py`). The default display-scale anchor values were
  calibrated against aggregate crowd-rating quantiles from seriesgraph.com.

## Validation

The test suite includes golden-parity fixtures generated from the reference
implementation: six open-data matches with every feature value checked to 1e-6 relative
tolerance (including which values are missing), and an end-to-end comparison of the full
2022 World Cup board (all 64 raw scores and the resulting ranking; run with
`RUN_SLOW=1 pytest`). Structural tests cover the decomposition identity, deduction
gating, missing-data behavior, and the monotonicity of the publication map.

## The World Cup 2026 board (v1.4, through the round of 16, July 7, 2026)

| # | Date | Match | Score | Stage | Rating |
|---|------|-------|-------|-------|--------|
| 1 | 2026-07-01 | Belgium vs Senegal | 3-2 aet | R32 | 9.65 |
| 2 | 2026-06-30 | Côte d'Ivoire vs Norway | 1-2 | R32 | 9.49 |
| 3 | 2026-06-29 | Germany vs Paraguay | 1-1 (3-4 p) | R32 | 9.47 |
| 4 | 2026-07-03 | Argentina vs Cape Verde Islands | 3-2 aet | R32 | 9.46 |
| 5 | 2026-07-05 | Brazil vs Norway | 1-2 | R16 | 9.43 |
| 6 | 2026-07-02 | Portugal vs Croatia | 2-1 | R32 | 9.41 |
| 7 | 2026-06-27 | Algeria vs Austria | 3-3 | group | 9.41 |
| 8 | 2026-06-29 | Brazil vs Japan | 2-1 | R32 | 9.41 |
| 9 | 2026-06-25 | Türkiye vs United States | 3-2 | group | 9.40 |
| 10 | 2026-06-29 | Netherlands vs Morocco | 1-1 (2-3 p) | R32 | 9.39 |
| 11 | 2026-07-01 | England vs Congo DR | 2-1 | R32 | 9.39 |
| 12 | 2026-07-03 | Australia vs Egypt | 1-1 (2-4 p) | R32 | 9.38 |
| 13 | 2026-07-05 | Mexico vs England | 2-3 | R16 | 9.16 |
| 14 | 2026-07-06 | Portugal vs Spain | 0-1 | R16 | 9.09 |
| 15 | 2026-06-28 | South Africa vs Canada | 0-1 | R32 | 8.97 |
| 16 | 2026-06-20 | Germany vs Côte d'Ivoire | 2-1 | group | 8.96 |
| 17 | 2026-06-26 | Egypt vs Iran | 1-1 | group | 8.84 |
| 18 | 2026-06-11 | South Korea vs Czech Republic | 2-1 | group | 8.80 |
| 19 | 2026-06-27 | Congo DR vs Uzbekistan | 3-1 | group | 8.76 |
| 20 | 2026-06-22 | Jordan vs Algeria | 1-2 | group | 8.67 |
| 21 | 2026-06-15 | Iran vs New Zealand | 2-2 | group | 8.59 |
| 22 | 2026-06-21 | New Zealand vs Egypt | 1-3 | group | 8.51 |
| 23 | 2026-06-17 | England vs Croatia | 4-2 | group | 8.48 |
| 24 | 2026-06-25 | Ecuador vs Germany | 2-1 | group | 8.42 |
| 25 | 2026-06-21 | Uruguay vs Cape Verde Islands | 2-2 | group | 8.30 |
| 26 | 2026-07-06 | United States vs Belgium | 1-4 | R16 | 8.30 |
| 27 | 2026-06-14 | Netherlands vs Japan | 2-2 | group | 8.30 |
| 28 | 2026-06-24 | Morocco vs Haiti | 4-2 | group | 8.29 |
| 29 | 2026-06-16 | Austria vs Jordan | 3-1 | group | 8.08 |
| 30 | 2026-06-14 | Côte d'Ivoire vs Ecuador | 1-0 | group | 8.06 |
| 31 | 2026-06-16 | France vs Senegal | 3-1 | group | 8.03 |
| 32 | 2026-06-22 | Norway vs Senegal | 3-2 | group | 7.99 |
| 33 | 2026-06-12 | Canada vs Bosnia-Herzegovina | 1-1 | group | 7.85 |
| 34 | 2026-06-24 | Scotland vs Brazil | 0-3 | group | 7.84 |
| 35 | 2026-06-13 | Brazil vs Morocco | 1-1 | group | 7.82 |
| 36 | 2026-06-20 | Ecuador vs Curaçao | 0-0 | group | 7.82 |
| 37 | 2026-06-13 | Australia vs Türkiye | 2-0 | group | 7.70 |
| 38 | 2026-06-26 | Norway vs France | 1-4 | group | 7.68 |
| 39 | 2026-06-26 | New Zealand vs Belgium | 1-5 | group | 7.64 |
| 40 | 2026-06-15 | Belgium vs Egypt | 1-1 | group | 7.63 |
| 41 | 2026-06-15 | Saudi Arabia vs Uruguay | 1-1 | group | 7.63 |
| 42 | 2026-06-25 | Japan vs Sweden | 1-1 | group | 7.63 |
| 43 | 2026-06-18 | Czech Republic vs South Africa | 1-1 | group | 7.62 |
| 44 | 2026-06-30 | France vs Sweden | 3-0 | R32 | 7.61 |
| 45 | 2026-07-02 | Switzerland vs Algeria | 2-0 | R32 | 7.61 |
| 46 | 2026-07-02 | Spain vs Austria | 3-0 | R32 | 7.61 |
| 47 | 2026-07-04 | Canada vs Morocco | 0-3 | R16 | 7.61 |
| 48 | 2026-06-17 | Ghana vs Panama | 1-0 | group | 7.60 |
| 49 | 2026-06-26 | Cape Verde Islands vs Saudi Arabia | 0-0 | group | 7.57 |
| 50 | 2026-06-13 | Qatar vs Switzerland | 1-1 | group | 7.55 |
| 51 | 2026-07-03 | Colombia vs Ghana | 1-0 | R32 | 7.53 |
| 52 | 2026-06-14 | Germany vs Curaçao | 7-1 | group | 7.49 |
| 53 | 2026-07-01 | United States vs Bosnia-Herzegovina | 2-0 | R32 | 7.45 |
| 54 | 2026-06-20 | Netherlands vs Sweden | 5-1 | group | 7.45 |
| 55 | 2026-06-30 | Mexico vs Ecuador | 2-0 | R32 | 7.42 |
| 56 | 2026-06-24 | Switzerland vs Canada | 2-1 | group | 7.41 |
| 57 | 2026-06-12 | United States vs Paraguay | 4-1 | group | 7.40 |
| 58 | 2026-06-24 | South Africa vs South Korea | 1-0 | group | 7.40 |
| 59 | 2026-06-18 | Switzerland vs Bosnia-Herzegovina | 4-1 | group | 7.36 |
| 60 | 2026-07-04 | Paraguay vs France | 0-1 | R16 | 7.35 |
| 61 | 2026-06-19 | Türkiye vs Paraguay | 0-1 | group | 7.33 |
| 62 | 2026-06-21 | Belgium vs Iran | 0-0 | group | 7.30 |
| 63 | 2026-06-18 | Mexico vs South Korea | 1-0 | group | 7.29 |
| 64 | 2026-06-27 | Colombia vs Portugal | 0-0 | group | 7.29 |
| 65 | 2026-06-16 | Iraq vs Norway | 1-4 | group | 7.24 |
| 66 | 2026-06-24 | Bosnia-Herzegovina vs Qatar | 3-1 | group | 7.22 |
| 67 | 2026-06-22 | Argentina vs Austria | 2-0 | group | 7.21 |
| 68 | 2026-06-15 | Spain vs Cape Verde Islands | 0-0 | group | 7.17 |
| 69 | 2026-06-27 | Panama vs England | 0-2 | group | 7.14 |
| 70 | 2026-06-27 | Croatia vs Ghana | 2-1 | group | 7.13 |
| 71 | 2026-06-23 | Panama vs Croatia | 0-1 | group | 7.13 |
| 72 | 2026-06-23 | Portugal vs Uzbekistan | 5-0 | group | 7.10 |
| 73 | 2026-06-13 | Haiti vs Scotland | 0-1 | group | 7.10 |
| 74 | 2026-06-24 | Czech Republic vs Mexico | 0-3 | group | 7.09 |
| 75 | 2026-06-17 | Portugal vs Congo DR | 1-1 | group | 7.06 |
| 76 | 2026-06-14 | Sweden vs Tunisia | 5-1 | group | 7.05 |
| 77 | 2026-06-18 | Canada vs Qatar | 6-0 | group | 7.03 |
| 78 | 2026-06-17 | Uzbekistan vs Colombia | 1-3 | group | 6.99 |
| 79 | 2026-06-19 | Scotland vs Morocco | 0-1 | group | 6.86 |
| 80 | 2026-06-26 | Senegal vs Iraq | 5-0 | group | 6.83 |
| 81 | 2026-06-22 | France vs Iraq | 3-0 | group | 6.52 |
| 82 | 2026-06-25 | Paraguay vs Australia | 0-0 | group | 6.52 |
| 83 | 2026-06-23 | Colombia vs Congo DR | 1-0 | group | 6.47 |
| 84 | 2026-06-26 | Uruguay vs Spain | 0-1 | group | 6.42 |
| 85 | 2026-06-27 | Jordan vs Argentina | 1-3 | group | 6.41 |
| 86 | 2026-06-19 | United States vs Australia | 2-0 | group | 6.36 |
| 87 | 2026-06-21 | Spain vs Saudi Arabia | 4-0 | group | 6.29 |
| 88 | 2026-06-25 | Tunisia vs Netherlands | 1-3 | group | 6.27 |
| 89 | 2026-06-25 | Curaçao vs Côte d'Ivoire | 0-2 | group | 6.02 |
| 90 | 2026-06-23 | England vs Ghana | 0-0 | group | 6.00 |
| 91 | 2026-06-16 | Argentina vs Algeria | 3-0 | group | 5.76 |
| 92 | 2026-06-19 | Brazil vs Haiti | 3-0 | group | 5.72 |
| 93 | 2026-06-20 | Tunisia vs Japan | 0-4 | group | 5.01 |
| 94 | 2026-06-11 | Mexico vs South Africa | 2-0 | group | 3.33 |

G1/G2/G3 denote group-stage matchdays. The five-bucket decomposition and both deduction
line items for every match are in `data/wc2026_board.csv`.

## Install and test

```bash
pip install -e ".[dev]"
pytest
```

## Citation

> NetSI Sport (2026). The NetSI match excitement index (v1.4). Network Science
> Institute, Northeastern University, for Northeastern Global News.
> https://github.com/jkbren/excitement-index-soccer

## License

MIT (code). StatsBomb open data, eloratings.net ratings, and seriesgraph.com aggregates
are subject to their own terms; see `LICENSE` for data notes.
