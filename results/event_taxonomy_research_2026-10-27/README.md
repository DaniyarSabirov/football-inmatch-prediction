# Why only eight event types? A comparison of event taxonomies

Date: 2026-10-27. Research and decision support only. The production dataset, target, feature allowlist and
code were **not changed**, and no model was trained. Every number below comes from the 417 eligible matches
(1,551,841 raw events) via the scripts in `diagnostics/`. Section 7 separates measurements from judgement.

## 1. Executive summary

- **Origin.** The eight types appeared in the dataset builder's first commit (`f12931a`, 2026-07-30) as a bare
  constant, with no comment, document or test explaining the choice. Two earlier audits already recorded "no
  documented justification" (`docs/bug_and_risk_report_2026-10-13.md` F-6a; `docs/statsbomb_complete_category_audit.md`
  §6). No other selection was evaluated before it. **The choice was not scientifically justified; it was an
  undocumented preliminary simplification.**
- **It matches no published taxonomy.** The reference StatsBomb→SPADL converter, used by VAEP and related work,
  keeps several types the eight omit (Interception, tackle Duels, Foul Committed, Goal Keeper actions). It also
  drops two types the eight include (Ball Recovery, Dispossessed).
- **Including all raw types (Scenario B) is not a good idea.** About a quarter of the rows would be the
  mechanical step Pass → Ball Receipt\* (86%). Companion pairs become deterministic, for example
  Foul Committed → Foul Won (95%) and Dribbled Past → Dribble (100%). The target also gains administrative
  events (Half End, Player Off, …), and the imbalance ratio reaches 12,293:1. The model would partly learn
  StatsBomb's annotation conventions rather than football.
- **Scenario C** (all football actions, 19 types) has the same companion problem.
- **Scenario D is the recommended task.** It uses 15 types: the 8 current types plus Pressure, Duel,
  Interception, Block, Goal Keeper, Foul Committed and 50/50, chosen by explicit, measured criteria.
  - It adds the defensive and goalkeeper actions the current task cannot see.
  - It has **+182,568 input rows (+23.2%)**.
  - It has the lowest share of near-mechanical transitions, and the highest residual uncertainty given the
    current event type.
  - Its smallest class still has 1,504 rows.
- **Prediction becomes different, not merely harder.** Accuracy or macro-F1 cannot be compared across
  taxonomies, because they are different tasks.

## 2. Origin of the eight-type restriction (documented facts)

| Question | Answer | Evidence |
|---|---|---|
| When introduced? | 2026-07-30, commit `f12931a` "Add StatsBomb 360 next-event dataset builder" (author DaniyarSabirov) | `git log -S MEANINGFUL_EVENT_TYPES` |
| Earlier traces? | No. The only earlier commit (`2443029`) lists all unique event types for exploration and selects none | `git show 2443029` |
| Documented rationale? | None: no comment, commit message text, document, config or test explains it. The pinning test only freezes the set. | code, docs, tests |
| Based on 360 availability? | Not documented. Measured: it is **not** explained by 360 coverage. Ball Receipt\* (89.5%), Pressure (89.2%), Interception (92.5%) and Block (89.9%) have as much 360 data as the eight (84.9–96.8%). | `type_stats.csv` |
| Other types evaluated before exclusion? | No record of any evaluation | — |
| Inherited from an example or tutorial? | Not documented, and it does not match SPADL (§6) | — |
| Plausible reading (inference, not fact) | Roughly "actions of the ball carrier plus winning or losing the ball" | earlier audit §6, marked as inference there too |

## 3. Complete taxonomy of all 33 event types (`type_stats.csv`, `companions.csv`)

Columns:
- **% ev:** share of all events.
- **M:** matches with the type.
- **360:** share with a 360 record.
- **no 360:** occurrences without one.
- **loc:** has a location.
- **actor:** an actor point in its own snapshot.
- **cat:** the category from `event_type_categories.json`.
- **suitable input / target:** this research's assessment.

| Type | Count | % ev | M | 360 % | no 360 | loc % | actor | cat | Suitable input | Suitable target | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pass | 438,439 | 28.25 | 417 | 85.2 | 64,800 | 100 | 100% | on-ball | yes | yes | |
| Ball Receipt\* | 414,717 | 26.72 | 417 | 89.5 | 43,562 | 100 | 100% | companion | weak | **no** | 90.7% follow a Pass. It is the pass's arrival record; Pass → Ball Receipt\* is 86%. |
| Carry | 353,132 | 22.76 | 417 | 91.5 | 30,178 | 100 | 100% | on-ball | yes | yes | always starts at the timestamp of its predecessor |
| Pressure | 133,003 | 8.57 | 417 | 89.2 | 14,431 | 100 | 100% | defensive | yes | yes | concurrent with ball events |
| Ball Recovery | 40,649 | 2.62 | 417 | 89.7 | 4,185 | 100 | 100% | on-ball | yes | yes | |
| Duel | 26,980 | 1.74 | 417 | 87.8 | 3,292 | 100 | 100% | defensive | yes | yes (caveat) | Dispossessed → Duel 100% at the same time (36.5% of Duels) |
| Block | 16,432 | 1.06 | 417 | 89.9 | 1,667 | 100 | 100% | defensive | yes | yes | Shot → Block 27.5% |
| Clearance | 16,254 | 1.05 | 417 | 92.7 | 1,190 | 100 | 100% | on-ball | yes | yes | |
| Goal Keeper | 13,071 | 0.84 | 417 | **46.2** | 7,039 | 99.95 | 100% | goalkeeper | limited (half lack 360) | yes | Shot → Goal Keeper 71.9% |
| Dribble | 13,028 | 0.84 | 417 | 86.0 | 1,830 | 100 | 100% | on-ball | yes | yes | |
| Miscontrol | 11,291 | 0.73 | 417 | 87.9 | 1,366 | 100 | 100% | on-ball | yes | yes | |
| Shot | 10,831 | 0.70 | 417 | 96.8 | 348 | 100 | 100% | on-ball | yes | yes | |
| Foul Committed | 10,310 | 0.66 | 417 | 86.8 | 1,363 | 100 | 100% | foul | yes | yes | |
| Dispossessed | 9,852 | 0.63 | 417 | 84.9 | 1,492 | 100 | 99.99% | on-ball | yes | yes | |
| Foul Won | 9,794 | 0.63 | 417 | 86.6 | 1,312 | 100 | 100% | companion | weak | **no** | 99.9% at the same time right after Foul Committed |
| Interception | 9,701 | 0.63 | 417 | 92.5 | 726 | 100 | 100% | defensive | yes | yes | 90.5% recorded right after an incomplete Ball Receipt\* |
| Dribbled Past | 7,587 | 0.49 | 417 | 87.0 | 987 | 100 | 100% | companion | weak | **no** | always immediately followed by its Dribble (100%, same time) |
| Substitution | 3,610 | 0.23 | 417 | 0 | 3,610 | 0 | — | admin | no | no | |
| Injury Stoppage | 2,311 | 0.15 | 403 | 0 | 2,311 | 0 | — | stoppage | no | no | |
| Half Start | 1,824 | 0.12 | 417 | 0 | 1,824 | 0 | — | admin | no | no | |
| Half End | 1,824 | 0.12 | 417 | 0 | 1,824 | 0 | — | admin | no | no | |
| 50/50 | 1,685 | 0.11 | 301 | 89.4 | 179 | 100 | 99.9% | on-ball | yes | yes (rare) | recorded in pairs (50.3% same-time 50/50 predecessor) |
| Tactical Shift | 1,399 | 0.09 | 409 | 0 | 1,399 | 0 | — | admin | no | no | |
| Referee Ball-Drop | 996 | 0.06 | 300 | 2.2 | 974 | 100 | — | stoppage | no | no | no player |
| Starting XI | 834 | 0.05 | 417 | 0 | 834 | 0 | — | admin | no | no | |
| Shield | 543 | 0.03 | 299 | 67.2 | 178 | 100 | 100% | on-ball | yes | too rare | < 1,000 occurrences |
| Player Off | 494 | 0.03 | 253 | 0 | 494 | 0 | — | admin | no | no | |
| Player On | 491 | 0.03 | 252 | 0 | 491 | 0 | — | admin | no | no | |
| Bad Behaviour | 271 | 0.02 | 157 | 0 | 271 | 0 | — | discipline | no | no | card, no location |
| Error | 237 | 0.02 | 181 | 87.3 | 30 | 100 | 100% | retrospective | no | no | judged with hindsight (an error *leading to* a shot or goal) |
| Offside | 141 | 0.01 | 119 | 90.1 | 14 | 100 | 100% | discipline | no | no | referee decision |
| Own Goal For | 55 | 0.004 | 51 | 87.3 | 7 | 100 | 100% | companion | no | no | pair of Own Goal Against |
| Own Goal Against | 55 | 0.004 | 51 | 87.3 | 7 | 100 | 100% | retrospective | no | no | outcome marker |

**Checks on the existing taxonomy audit** (`docs/statsbomb_complete_category_audit.md`, 2026-10-20):
- its 33 types and its eligible event total (1,551,841) match the current raw population exactly;
- the per-type eligible counts agree.

The 360, location, actor and companion columns are new in this research.

**Actor coordinates.** Whenever an event has a non-empty 360 snapshot, the snapshot contains an actor point
(≥ 99.9% for every type). Actor positions become unavailable only through causal orientation (41 rows today),
not through event type.

## 4. Input, target and intermediate events

The current task is **not** "predict the immediate next raw event". It is "predict the next event of a
*selected* type". In Scenario A, **59.4% of rows skip at least one intermediate raw event** (mean 0.72 per row).
The events skipped most often:

| Skipped event | Times skipped |
|---|---|
| Ball Receipt\* | 354,383 |
| Pressure | 118,017 |
| Duel | 22,295 |
| Block | 14,696 |
| Goal Keeper | 12,263 |
| Foul Committed | 8,966 |
| Interception | 8,586 |

Example: Pass → Ball Receipt\* → Carry → Pressure → Shot yields the Scenario A rows Pass → Carry and
Carry → Shot. The receipt and the defender's pressure are invisible, so "Carry → Shot under pressure" and
"Carry → Shot unchallenged" become the same label.

| Treatment of a type | Consequence |
|---|---|
| Input only | More observation points, but the model is never asked to predict that event. Its occurrences sit between input and target and are skipped. |
| Target only | The model predicts it but never observes from its moment. This suits types that mostly lack 360 data (e.g. Goal Keeper). |
| Both | A consistent event chain; each step is both observed and predicted. |
| Excluded | Invisible: it is skipped as an intermediate event and changes what "next" means. |

Different input and target sets are legitimate, but they make the task harder to explain, and pairs of
near-identical rows can arise. For example, Ball Receipt\* and the Carry that starts at the same timestamp
usually share a snapshot (§8). Where the 360 data allow it, **identical input and target sets are the cleaner
design**. Section 9 discusses the one exception (Goal Keeper as input).

## 5. Quantitative comparison of the scenarios (`scenarios.json`, `scenario_targets.csv`)

Definitions:
- **A:** current 8 types.
- **B:** all raw types as targets; every event with a 360 record as input.
- **C:** 19 football actions. Excludes administrative, stoppage, referee-sanction and retrospective events.
  Includes the companions Ball Receipt\*, Foul Won and Dribbled Past.
- **D:** C minus the measured companions (≥ 90% of occurrences are the mechanical counterpart of one partner
  type) and minus types with fewer than 1,000 occurrences (Shield). That leaves 15 types.

All measured with the production target semantics: same period, event order, no time window. Inputs need
their own 360 record.

| Metric (measured) | A (8) | B (33) | C (19) | **D (15)** |
|---|---|---|---|---|
| Input rows | 787,331 | 1,357,626 | 1,356,505 | **969,899** |
| vs A | — | +72.4% | +72.3% | **+23.2%** |
| Matches | 417 | 417 | 417 | 417 |
| Target classes observed | 8 | 31 | 19 | 15 |
| Majority class (share) | Pass 48.6% | Pass 28.1% | Pass 28.3% | Pass 39.1% |
| Smallest class (rows) | Dispossessed 8,722 | Player Off 31 | Shield 418 | 50/50 1,504 |
| Imbalance ratio | 43.9 | 12,293 | 918 | **252** |
| Classes < 1% / < 0.1% | 0 / 0 | 23 / 13 | 11 / 1 | 4 / 0 |
| Classes < 1,000 rows | 0 | 11 | 1 | 0 |
| Raw events whose type is in the target set | 57.6% | 100% | 99.1% | 71.2% |
| Raw transitions between selected types (upper bound) | 22.8% | 100% | 98.7% | 43.1% |
| Rows skipping ≥ 1 intermediate event | 59.4% | 0% | 0.3% | 38.5% |
| Targets without their own 360 record | 8.2% | 6.0% | 5.8% | 7.8% |
| Zero-gap (simultaneous) rows | 5.1% | 32.6% | 32.6% | 7.3% |
| Target entropy (bits) | 1.66 | 2.70 | 2.66 | 2.39 |
| H(target \| current type) (bits) | 1.01 | 1.34 | 1.31 | **1.66** |
| Entropy explained by current type | 39.1% | 50.4% | 50.8% | **30.8%** |
| Rows in cells with P ≥ 0.9 | 0% | 1.7% | 1.7% | 0.9% |
| Rows in cells with P ≥ 0.7 | 76.3% | 66.4% | 66.6% | **29.9%** |
| Largest cell | Carry → Pass 36.7% (P = 0.90) | Pass → Ball Receipt\* 23.7% (0.86) | Pass → Ball Receipt\* 23.7% (0.86) | Pass → Carry 25.5% (0.66) |
| Estimated CSV size | 276 MB | 477 MB | 476 MB | 340 MB |

**Scenario D target classes:**

| Class | Rows | Share |
|---|---|---|
| Pass | 379,439 | 39.1% |
| Carry | 315,591 | 32.5% |
| Pressure | 117,672 | 12.1% |
| Ball Recovery | 35,262 | 3.6% |
| Duel | 22,280 | 2.3% |
| Block | 14,541 | 1.5% |
| Clearance | 13,891 | 1.4% |
| Goal Keeper | 12,366 | 1.3% |
| Dribble | 11,363 | 1.2% |
| Shot | 10,015 | 1.0% |
| Miscontrol | 9,916 | 1.0% |
| Foul Committed | 8,894 | 0.9% |
| Interception | 8,609 | 0.9% |
| Dispossessed | 8,556 | 0.9% |
| 50/50 | 1,504 | 0.16% |

**Selection effects:**
- In every scenario, inputs need their own 360 record. For most types 85–97% have one, but Goal Keeper only
  46%, so Goal Keeper inputs would be a selective subset.
- Targets come from the full stream and are unaffected.
- A excludes the defensive side of the game completely; D does not.
- **D's measured deterministic pair:** Dispossessed → Duel (8,356 rows, P = 1.0, the tackle recorded with the
  dispossession). It is 0.86% of rows (caveat in §9).

## 6. Literature

| Study | What it does with event types | Relation to us |
|---|---|---|
| Decroos, Bransen, Van Haaren & Davis, *Actions Speak Louder than Goals: Valuing Player Actions in Soccer*, KDD 2019, pp. 1851–1861 ([arXiv:1802.07127](https://arxiv.org/abs/1802.07127)) | Introduces SPADL, which keeps only *actions* ("a subset of vendor events that require a player to perform something"). Events such as the end of a game are not actions. | Supports excluding administrative events, and supports a selected action taxonomy. |
| socceraction, reference SPADL converter for StatsBomb ([source](https://github.com/ML-KULeuven/socceraction/blob/master/socceraction/spadl/statsbomb.py), [SPADL types](https://socceraction.readthedocs.io/en/latest/documentation/spadl/spadl.html)) | 22 action types. **Kept:** Pass, Dribble, Carry, Foul Committed, tackle Duels, Interception, Shot, Own Goal Against, some Goal Keeper subtypes, Clearance, Miscontrol. **Dropped as non-actions:** Ball Receipt\*, Pressure, Foul Won, Dribbled Past, Block, 50/50, other Duels, all administrative events (verified in the code). Ball Recovery and Dispossessed are also not mapped. | The closest published StatsBomb taxonomy. It treats Ball Receipt\*, Foul Won and Dribbled Past as non-actions, like D does, and it **includes** defensive and goalkeeper actions that our 8 exclude. |
| Simpson, Beal, Locke & Norman, *Seq2Event*, KDD 2022, pp. 3898–3908 ([eprint](https://eprints.soton.ac.uk/458099)) | Predicts the next match event (type and location) from previous events. | The same task family. Its exact class list could not be verified (the PDF returned 403), so it is cited only for the task. |
| Yeung, Sit & Fujii, *Transformer-Based Neural Marked Spatio Temporal Point Process Model for Football Match Events Analysis* ([arXiv:2302.09276](https://arxiv.org/abs/2302.09276); journal version in *Applied Intelligence*, [doi:10.1007/s10489-024-05996-9](https://link.springer.com/article/10.1007/s10489-024-05996-9)) | Predicts the next event's time, zone and action. Raw Wyscout types are **grouped into 5 classes** (pass, dribble, cross, shot, possession end); matches with own goals are dropped. | Shows that aggressive aggregation is common, and that "possession end" is sometimes a target. |
| Mendes-Neves, Meireles & Mendes-Moreira, *Forecasting Events in Soccer Matches Through Language* ([arXiv:2402.06820](https://arxiv.org/abs/2402.06820)); *Towards a foundation large events model for soccer*, Machine Learning 113 (2024) ([doi:10.1007/s10994-024-06606-y](https://link.springer.com/article/10.1007/s10994-024-06606-y)) | Predicts the next event type among **37 Wyscout action classes** (pass, shot, take-on, tackle, …) as part of the full event. | Shows that a broad, near-complete action vocabulary is also used. |
| Merhej, Beal, Ramchurn & Matthews, *What Happened Next? Using Deep Learning to Value Defensive Actions in Football Event-Data*, KDD 2021 ([arXiv:2106.01786](https://arxiv.org/abs/2106.01786)) | Values defensive actions (tackles, interceptions). According to the paper's PDF as quoted in search results, it uses StatsBomb data and SPADL. | Defensive actions carry analytic value that the current 8 omit. |

**What the literature shows:**
- There is no single standard taxonomy. Published work ranges from 5 aggregated classes to 37 actions.
- Administrative events are excluded throughout.
- The StatsBomb-specific reference (SPADL) excludes Ball Receipt\*, Pressure and the companion events.
- None of these sources was found to justify *our* exact 8 types.

No source is claimed to endorse D. D is our own criterion-based selection, consistent with the SPADL practice
of dropping companions and including defensive and goalkeeper actions. It differs from SPADL by keeping
Pressure, Block, Ball Recovery, Dispossessed and 50/50.

## 7. Consequences of including more types

**Measured:**
- **Dataset size:** rows rise by 72% (B and C) or 23% (D). Requiring a 360 record for inputs is not the
  binding constraint (most action types have 85–97% coverage).
- **Imbalance:** it explodes in B (12,293:1; 13 classes under 0.1%) and C (918:1). In D it is 252:1 with no
  class below 1,000 rows. Evaluation of 50/50 (1,504 rows, about 300 per test fold in a 5-fold split) is
  feasible but noisy.
- **Predictability:**
  - B and C contain a large, near-deterministic block: Pass → Ball Receipt\* alone is 23.7% of rows at 86%,
    and Foul Committed → Foul Won and Dribbled Past → Dribble are almost certain.
  - Overall accuracy would look high for reasons unrelated to football understanding.
  - In D the current type explains the least of the target (30.8%) and only 29.9% of rows fall in P ≥ 0.7
    cells (vs 76.3% in A). D is the *least* mechanical task.
- **Temporal structure:**
  - Simultaneous rows rise from 5.1% (A) to 32.6% (B, C), because StatsBomb records paired events at one
    timestamp and orders them by convention.
  - D stays at 7.3%.
  - Event-ordering conventions, such as Dribbled Past placed *before* its Dribble or Duel placed after
    Dispossessed, create artificial predictability.

**Judgement:**
- **Prediction becomes different, not simply harder.** Macro-F1, weighted-F1 and balanced accuracy of
  different taxonomies measure different tasks and must not be compared as one ranking.
- **Evaluation:**
  - Per-class precision and recall and confusion matrices should be reported.
  - Macro-F1 is sensitive to rare classes such as 50/50.
  - Weighted-F1 and accuracy are dominated by Pass and Carry.
  - A frequency baseline conditioned on the current type (computable from these tables) shows how much of any
    score comes from transition priors.
- **Football relevance:** D represents defending (Pressure, Duel, Interception, Block, Foul Committed) and
  goalkeeping. Those are the most tactically interesting reactions to the 360 configuration, which is the
  thesis's input.

**Computation (measured or extrapolated):**
- The dominant build cost, orientation and identity resolution, already runs over all 1,357,626 records for
  every scenario and is unchanged.
- Per-row cost scales with rows: CSV about 340 MB for D vs 276 MB now. A full build takes about 3.5 min today
  (7 workers), and about 10–25% more is expected for D.
- Model size changes only through the number of output classes (8 → 15).

## 8. Technical feasibility in the current pipeline

| Component | Status |
|---|---|
| Event-type selection | **Hard-coded in one place:** `MEANINGFUL_EVENT_TYPES` in `src/build_statsbomb_360_dataset.py` serves as both input filter and target filter (`find_next_meaningful_event`). Separate input and target sets would need two constants or a config file. |
| Event ordering, target generation | Type-agnostic: event order, same period, no time window. |
| 360 joins | Type-agnostic: by `event_uuid`. |
| Team identity, orientation | Type-agnostic: already resolved for **all** usable records, in both causal and offline versions. New input rows would reuse validated values. |
| Causal feature computation | Unchanged. Features stay relative to the *acting* team, which for defensive inputs (Pressure, Duel) is the defending team. `current_event_type` tells the model which. |
| Feature classification and allowlist | Column set unchanged; `current_event_type` gains values. Truncation and causality tests are type-agnostic. |
| Targets without their own 360 record | No missing-feature problem: features belong to the *input* row only. 5.8–8.2% of targets lack 360 in every scenario. |
| Inputs with 360 but no actor coordinates | Not type-dependent (an actor point exists in ≥ 99.9% of snapshots). The 41 causally unavailable orientations are flagged by `orientation_available`. |
| Tests | `test_meaningful_event_types_definition_is_exactly_as_documented` pins the 8 types and would change deliberately. `tests/test_ml_feature_safety.py` uses the constant for truncation labels and would follow. |
| Scripts | `audit_dataset.py`, `analyze_prediction_task_candidates.py`, `inspect_data_quality.py` and `audit_statsbomb_category_taxonomy.py` import the constant. These are analyses, not pipeline. |
| Visualization | Unaffected (it renders all event types). |
| Future temporal models | D's sequences have fewer gaps (61.5% of rows skip nothing vs 40.6% in A) and stay mostly non-simultaneous (7.3%). |

## 9. Methodological risks

1. **Mechanical companions (B, C).** These inflate scores and teach annotation conventions; D avoids them by
   a measured rule.
2. **Dispossessed → Duel in D** (8,356 rows, P = 1.0). It is the tackle recorded with the dispossession.
   Options:
   - (a) keep it and report it as a known deterministic transition;
   - (b) treat a same-timestamp Duel following Dispossessed as a companion and skip it (Duel stays a target
     elsewhere);
   - (c) drop Duel.
   Option (b) follows the same rule as the other companions. **This needs your decision.**
3. **Goal Keeper as input:** only 46% have 360 data, so they would be a selective subset of goalkeeper
   situations. Option: keep Goal Keeper as **target only**. Two consequences: input rows would be
   969,899 − about 6,000, and the input and target sets would then differ by this single, documented type.
4. **Pressure is a concurrent annotation.** It starts while ball events continue and is ordered by start time.
   It is a genuine defensive action, but its timing convention makes Carry → Pressure (20.8%) partly a timing
   artefact.
5. **Comparability.** A model trained on D cannot be ranked against results on A.
6. **50/50 is rare** (1,504 rows). It can be kept with per-class reporting, or merged into Duel if its
   per-class metrics are too unstable; the latter would be a decision at modelling time.

## 10. Recommendation

1. **Why exactly eight?** No documented reason. The set was hard-coded in the first builder commit and never
   evaluated.
2. **Justified?** No. It is not explained by 360 coverage, it matches no published taxonomy, and it hides
   defending and goalkeeping. 59% of rows skip events, mostly Ball Receipt\* and Pressure.
3. **All event types?** That gives +570,295 rows (B), but about a quarter of them are the mechanical
   Pass → Ball Receipt\* step. It adds administrative targets and deterministic companion pairs, and the
   imbalance reaches 12,293:1. **Not recommended.**
4. **Additional observations:** B +570,295; C +569,174; **D +182,568 (+23.2%)**.
5. **Harder or different?** Different. D carries more uncertainty per decision (1.66 vs 1.01 bits given the
   current type), and its scores are not comparable to A's.
6. **Football relevance:** clearly improved by D (defensive and goalkeeper actions).
7. **Exclude, and why:**
   - administrative and stoppage events: no 360 data and no location; not actions;
   - referee sanctions and retrospective annotations (Bad Behaviour, Offside, Error, Own Goal For/Against):
     judged by the referee or with hindsight;
   - companions (Ball Receipt\*, Foul Won, Dribbled Past): ≥ 90% mechanical counterparts;
   - Shield: 543 occurrences.
8. **Different input and target sets?** Same sets by default. The single defensible exception is Goal Keeper
   as target only (risk 3).
9. **Recommended scenario:** **D, with 15 types**, as the thesis's primary task. Keep A as a documented,
   clearly separated reference task (not a ranking).
10. **Required changes (after approval):**
    1. Add `configs/event_taxonomy.json` with explicit `input_event_types` and `target_event_types`, plus the
       companion rule.
    2. In the builder, replace `MEANINGFUL_EVENT_TYPES` with these two sets and generalise
       `find_next_meaningful_event` to the target set.
    3. Implement the chosen Dispossessed → Duel option.
    4. Update the pinned taxonomy test. Add tests that companions are skipped, administrative events are never
       targets, and Goal Keeper's input/target status is as decided.
    5. Rebuild with backup. Validate targets against an independent derivation and re-run the truncation and
       allowlist checks.
    6. Update the README, `docs/ml_dataset_identity_and_orientation.md` and the feature-classification lineage
       text.

## 11. Decisions requiring your approval

1. Adopt **Scenario D (15 types)** as the primary task, keep A, or choose another scenario.
2. **Dispossessed → Duel:** keep as it is (a), skip the same-timestamp companion (b, recommended), or drop
   Duel (c).
3. **Goal Keeper:** input and target, or target only (recommended, because only 46% have 360).
4. **50/50:** keep as its own class (recommended now), or merge into Duel later if its metrics are unstable.
5. **Penalty shootouts** (still open from the time-window audit): include, or exclude by period.

## Files

| Path | Content |
|---|---|
| `event_type_categories.json` | category of each type and the scenario rules |
| `type_stats.csv` | per type: counts, matches, 360, location, actor |
| `companions.csv`, `transitions.csv` | same-timestamp predecessors; full immediate-transition matrix |
| `raw_counts.json` | totals |
| `scenarios.json`, `scenario_targets.csv` | all scenario metrics and class distributions |
| `diagnostics/measure_event_taxonomy.py`, `diagnostics/compare_scenarios.py` | reproducible scripts (read-only on raw data) |

The bundled StatsBomb Events specification PDF could not be text-extracted in this environment, so no spec
wording is quoted. The companion classification rests on the measured pairings.
