# Technical appendix: the NetSI match excitement rating
*(frozen July 7, 2026 · NetSI Sport for Northeastern Global News)*

This appendix documents every measure in the rating, the models behind them, and the exact
aggregation arithmetic. Together with the frozen constants listed in §8, it is sufficient
to recompute any published rating from the underlying event data.

## 1. Data and conventions

**Event data.** All measures are computed from StatsBomb event data: a timestamped record
of every on-ball action (~3,500 events per match) with pitch coordinates on a 120×80
grid, expected-goals (xG) values on shots, and outcome codes.

**Playable periods and clock.** Only periods 1–4 (regulation and extra time) enter the
computation; penalty shootouts (period 5) are excluded everywhere except the one shootout
measure noted below. The match clock is the absolute StatsBomb minute (second half starts
at 45', so stoppage time counts), and the match end is 120' if extra time was played,
90' otherwise.

**Goals.** A goal is a shot with outcome *Goal* or an *Own Goal For* event, attributed to
the event's team. Team names from the match sheet are alias-resolved onto the event feed's
naming before any team-keyed computation.

**Team strength.** Pre-match strength comes from Elo world ratings (eloratings.net). A
home-field adjustment of ±50 Elo points is applied when a host nation (United States,
Mexico, Canada) is involved.

## 2. The win-probability model

Several measures are derived from a per-match win-probability (WP) curve giving the
probabilities (p_home, p_draw, p_away) at every event.

- **Model.** The remaining-match goal margin is modeled as the difference of two Poisson
  processes (a Skellam distribution): each side scores at rate λ = r × minutes-remaining,
  and the three outcome probabilities follow from the current goal margin d plus the
  Skellam distribution of remaining goals.
- **Rates.** Per-side scoring rates r come from a log-linear Elo model: expected goals
  μ = 1.3·exp(±adv/2) per side (floored at 0.02), where adv = (Elo_home − Elo_away ±
  host adjustment)/200; r = μ/90. Without Elo coverage both sides receive the symmetric
  prior of 1.35 goals per 90 minutes.
- **Updates.** The curve updates on goals only. Calibration testing supports this choice:
  updating outcome probabilities on xG worsened prediction (ranked probability score)
  relative to goals-only. Shot quality therefore enters the index through the chance
  measures rather than the WP curve. Red cards do not update the curve.
- **Chase adjustment.** Late in a match, a trailing side's rate is tilted upward and the
  leader's downward: with lateness = 1 − minutes-left/end and margin magnitude m =
  min(|d|, 2), the trailing rate is multiplied by (1 + 0.25·lateness·m) and the leading
  rate by (1 − 0.125·lateness·m).
- **Endpoints.** A synthetic kickoff row carries the neutral pre-match prior; in the final
  ~3 seconds the curve collapses to certainty on the actual result, so there is no
  artificial final-whistle jump.
- **Per-shot leverage variant.** The chance-leverage and resolution measures reuse the
  same Skellam machinery evaluated at each shot's moment (current score state, minutes
  remaining floored at 0.05), without the chase adjustment.

## 3. Measure catalog

Fifty-seven measures across eleven families. Sign is + (raises the score) unless marked −.

### 3.1 Pre-match (family weight 0.048)

| Measure | Definition |
|---|---|
| neg_ranking_gap | −\|Elo_home − Elo_away\| (raw Elo points): closeness of the matchup on paper. |
| prematch_openness | Normalized Shannon entropy of the pre-match (home, draw, away) probabilities from the Elo–Skellam model; 1 = maximal pre-match uncertainty. |

### 3.2 Stakes (0.157)

| Measure | Definition |
|---|---|
| knockout | 1 if the match is any knockout stage, else 0. |
| elimination_stakes | Stage ordinal: group 0, round of 32 = 1, round of 16 = 2, quarterfinal 3, semifinal 4, third-place/final 5. |
| host_nation | 1 if the United States, Mexico, or Canada is playing, else 0. Enters at fixed scale (not z-scored; see §5). |

### 3.3 Back-and-forth (0.098)

| Measure | Definition |
|---|---|
| gei | Game Excitement Index: total variation of the full 3-outcome WP vector summed over all curve steps — the total distance the needle traveled. |
| suspense | Time-averaged normalized entropy of (p_home, p_draw, p_away): how long the result stayed genuinely uncertain. A knife-edge draw scores near 1; a foregone conclusion near 0. |
| peak_tension | Largest single-step WP movement (in the goals-only curve, effectively the biggest goal's probability swing). |
| lead_changes | Number of times the scoreline leader changed (sign flips of the running goal margin; passing through level does not count by itself). |
| comeback_magnitude | Largest WP recovery achieved by a side while or after trailing on the scoreline; scoreline-gating prevents phantom fightbacks from probability noise. |
| xg_lead_changes | Crossings of the cumulative non-penalty-xG race between the teams ("who deserved to lead" flips). |
| time_within_one_goal | Fraction of match minutes with goal margin ≤ 1. |

### 3.4 Brilliance (0.084)

| Measure | Definition |
|---|---|
| take_ons | Completed dribbles past an opponent (dribble outcome *Complete*). |
| long_carries | Ball carries covering ≥ 15 pitch units (~15 m) start to end. |
| line_breaking_passes | Passes carrying StatsBomb's through-ball flag. |
| directness | Share of completed passes gaining ≥ 5 pitch units toward goal. |
| screamer_goals | Σ 0.5·(1 − xG) over non-penalty goals with xG < 0.08 — improbable goals, credited by their improbability. |
| individual_takeover | Largest single-player total of positive on-ball value (OBV) — one player seizing the match, measured fame-free. Where OBV is unavailable, an event-based fallback (take-ons, shot assists, and goal quality per player) is used. |

### 3.5 Chances (0.139)

| Measure | Definition |
|---|---|
| total_npxg | Total non-penalty xG, both teams. |
| total_shots | Shot count (regulation + extra time). |
| total_sot | Shots on target (outcome Goal, Saved, or Saved To Post). |
| big_chances | Shots with xG ≥ 0.25 (penalties included). |
| box_entries | Completed passes and carries that end inside the penalty box having started outside it. |
| chance_leverage_total | Σ over shots of xG × counterfactual WP swing — each shot weighted by how much scoring it would have moved the outcome probabilities at that moment. |
| chance_leverage_p95 | 95th percentile of the per-shot leverage values — the match's near-biggest single moment of anticipation, robust to one outlier. |
| shot_balance | min(shots_home, shots_away) / max(...): 1 for an even contest, 0 for a one-way barrage. |

### 3.6 Goalkeeping (0.066)

| Measure | Definition |
|---|---|
| keeper_saves | Shots with outcome *Saved*. |
| psxg_minus_goals | On-target xG faced minus goals conceded (both teams pooled): shot-stopping above expectation. |
| great_saves | Saves of shots with xG ≥ 0.3. |
| one_on_ones | Shots flagged one-on-one with the keeper. |
| goal_line_peril | Shots hitting the woodwork or saved onto the post (outcomes *Post*, *Saved To Post*). |

### 3.7 Upset (0.039)

| Measure | Definition |
|---|---|
| upset | 0 if the pre-match favourite won; otherwise the favourite's pre-match win probability (so a heavy favourite held to a draw registers a large upset). |
| shock | Time-averaged distance of the live WP state from the pre-match prior — sustained defiance of the script, not just the final result. |
| underdog_defiance | upset × the favourite's non-penalty xG: an underdog result earned while withstanding a barrage. |

### 3.8 Timing (0.101)

| Measure | Definition |
|---|---|
| late_goals | Goals at ≥ 80', with an extra 0.5 weight for goals at ≥ 90' (including all extra-time goals). |
| late_winner | 1 if the match's final goal came at ≥ 80' and changed who led (including breaking a deadlock). |
| goal_burstiness | 1 / mean gap (minutes) between consecutive goals, for matches with ≥ 2 goals — goals in bursts. |
| big_chance_xg_missed | Σ xG over big chances (xG ≥ 0.25) that did not score. |
| leverage_missed_late | Σ xG over missed big chances at ≥ 75' with the score within one goal — the late agony specifically. |
| peak_window_tv | The best 10-minute window of accumulated WP movement. |
| late_window_tv | WP movement in the final 15 minutes of play. |

### 3.9 Resolution (0.124)

| Measure | Definition |
|---|---|
| resolution_leverage | Σ over scored goals of the WP swing they caused — goals weighted by the tension they resolved. An 89' equalizer in a tied match counts enormously; a sixth in a rout barely registers. |
| total_goals | Full-time (or after-extra-time) goals, both teams; shootout kicks excluded. |
| shootout_drama | For shootouts only: number of missed/saved kicks plus 0.5 × each kick beyond the standard ten (sudden-death overrun). Zero without a shootout. |

### 3.10 Flow (0.115)

| Measure | Definition |
|---|---|
| tempo | Completed passes per minute of ball-in-play, regulation time (in-play time from inter-event gaps, with gaps > 25 s counted as dead ball). |
| end_to_end | Standard deviation across possessions of each possession's mean field position — play swinging between ends. |
| bip_share | Ball-in-play share of elapsed time (statsbomb_bip v3 estimator). |
| counter_shots | Shots arising from counter-attacks (play pattern *From Counter*). |
| counterpress | Pressing actions within 5 s of a turnover (StatsBomb counterpress flag). |
| ball_recoveries | Ball Recovery events. |
| obv_end_to_end | Harmonic mean of the two teams' total positive non-shot on-ball value — high only when *both* sides generated threat. Extended tier (see §7). |
| obv_peak | Largest 30-second bin of positive non-shot OBV — the match's most dangerous passage. Extended tier. |
| obv_flat_share (−) | Fraction of 30-second bins with negligible (< 0.01) OBV — sterile, threat-free time. Extended tier. |
| dead_air (−) | Longest stretch (minutes) with no "spark": no goal, no shot of xG ≥ 0.07, no completed take-on, no WP step ≥ 0.02. |

### 3.11 Controversy (0.030)

| Measure | Definition |
|---|---|
| cards (−) | Yellow cards + 3 × red cards, both teams (curated match-sheet counts, with an event-derived fallback). |
| red_card (−) | 1 if any red card (straight or second yellow) was shown. |
| penalties | In-match penalty kicks awarded (scored or missed; shootout kicks excluded). |

## 4. Context-layer inputs (not part of the eleven families)

**qualification_jeopardy ∈ [0, 1].** For knockout matches, exactly 1. For group matches:
the mean over both teams of max(P(advance | win) − P(advance | lose), 0), estimated by
Monte-Carlo simulation of the group's remaining fixtures as of kickoff (same-day group
games are treated as simultaneous and simulated, not assumed known). Each remaining
fixture's score is drawn from independent Poissons with Elo-derived means; the focal match
is conditioned on the win/lose branch by rejection sampling; groups are ranked by points,
goal difference, and goals scored; the top two advance, and third place advances with a
points-conditional probability (from ~0.01 at ≤ 2 points to 0.95+ at 6+, reflecting the
48-team format's best-thirds rule). 300 simulations per branch, four branches per match,
deterministically seeded by match id.

**alive_until ∈ [0, 1].** The fraction of the match elapsed before the goal margin moved
beyond one for good (1 if the match ends a one-score game — hence every extra-time match
and every completed comeback scores 1).

**late_alive_30 ∈ [0, 1].** The share of the final 30 minutes of playing time (the window
[90', 120'] when extra time was played, else [60', 90']) spent with the margin within one.

## 5. Aggregation

For each match, every catalogued measure x is standardized against the frozen reference
pool — the 72 group-stage matches — as z = (x − μ_ref)/(σ_ref + 10⁻⁶), with σ the
population standard deviation, then clipped to [−3, +3] and multiplied by its sign. One
exception: host_nation is a 0/1 flag with near-zero reference variance, so it enters at
fixed scale (the raw flag) instead of a z-score.

Family score = the equal-weight mean of the family's available (non-missing) z-scores,
multiplied by the family weight. The frozen family weights (full precision) are: stakes
.15665, chances .13922, resolution .12398, flow .11512, timing .10059, back-and-forth
.09813, brilliance .08368, keeping .06605, pre-match .04752, upset .03882, controversy
.03022 (sum = 1). The core score is the sum of family scores.

Two deductions follow:

- **Dead-rubber deduction** = 0.40 × (1 − jeopardy) × max(core, 0). Multiplicative in
  positive quality (a fully dead rubber keeps 60% of its quality score); knockouts
  (jeopardy = 1) pay nothing; negative-quality matches are not adjusted; matches without
  jeopardy coverage (historical tournaments) pay nothing.
- **Aliveness deduction** (knockouts only) = min(0.60 × (1 − A), headroom), where
  A = ½(alive_until + late_alive_30) and headroom caps the deduction so no match falls
  below the frozen pool-median raw score (m = 0.0583). A dead knockout can drop to a
  mediocre score but cannot drop below that median. Unknown aliveness defaults to A = 1
  (no deduction).

raw = core − dead-rubber deduction − aliveness deduction.

For display, family contributions are regrouped into five buckets — Stakes = pre-match +
stakes + upset + both deductions (weight .243), Chances = chances + keeping (.205), Drama
= back-and-forth + timing (.199), Spectacle = brilliance + flow + controversy (.229),
Payoff = resolution (.124). The regrouping is arithmetic only; the five bucket values sum
exactly to the raw score, and both deductions are also published as separate line items.

## 6. Publication scale

The raw score maps to 0–10 through a frozen monotone function built from seven quantile
anchors (at the 2nd, 10th, 30th, 50th, 70th, 90th, and 98th percentiles): anchor x-values
are the raw-score quantiles of the 72-game reference pool; anchor y-values are the same
quantiles of the fan-rating distribution (matches with ≥ 100 votes), spanning 5.30 to
9.34. Between anchors the map is linear. Above the top anchor it follows y = 10 −
a·e^(−b·(x − x₇)) with a = 0.664 and b = 1.0148 (b set so the second-highest raw score in
the frozen pool publishes 9.49); below the bottom anchor it follows the slope-matched
mirror y = y₁·e^(slope/y₁·(x − x₁)) decaying toward 0. Both 0.0 and 10.0 are asymptotes
and unattainable. Published ratings are rounded to two decimals. Because the map is
strictly monotone, it cannot alter any match's rank.

## 7. Missing-data behavior

Three measure tiers exist. *Common-core* measures compute on any StatsBomb event feed.
*Context* measures (neg_ranking_gap, prematch_openness, upset, shock, underdog_defiance)
require the Elo table and are missing without it. *Extended* measures (the three OBV flow
measures) require on-ball-value data, absent from older open-data feeds. A missing measure
drops out of its family mean; a fully missing family drops out of the core with weight
renormalization. The jeopardy and aliveness layers are inert (zero deduction) where their
inputs are unavailable. Consequently the index scores any tournament with event data —
this is how the 2022 out-of-sample check is run.

## 8. Frozen artifacts and reproducibility

Four artifacts pin the index, each checksummed in a freeze manifest: the **normalization
reference** (the 72-game feature matrix fixing every μ and σ), the **scale map** (anchors,
tail constants, and the aliveness floor), the **evaluation pool** (the 94 match ids of
July 7, 2026, on which all pre-registered validation constraints are stated), and the
**manifest** itself (constants, checksums, amendment log). The family taxonomy, weights,
signs, and the two deduction parameters are code constants. The daily job re-scores
matches against these artifacts and cannot re-fit them; any change to the index is a
versioned amendment with a documented rationale (the version history and all amendment
records are available on request).

## 9. Conventions worth noting

- big_chances and big_chance_xg_missed include penalty kicks; total_npxg and
  screamer_goals exclude them.
- keeper_saves counts outcome *Saved* only; *Saved To Post* is counted under
  goal_line_peril.
- psxg_minus_goals uses StatsBomb's pre-shot xG on on-target shots (a pre-shot
  approximation of post-shot save performance).
- tempo is computed on regulation time only; all shot-derived measures exclude the
  shootout because events are pre-filtered to periods 1–4.
- The event feed carries additional computed columns (e.g., live-gated suspense variants,
  post-shot-xG goalkeeping) that are not part of the score; the scored set is exactly
  the 57 measures above plus the three context-layer inputs.

*Data: StatsBomb (events, xG, OBV); eloratings.net (team strength). Analysis: NetSI Sport
for Northeastern Global News.*
