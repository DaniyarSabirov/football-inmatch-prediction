# StatsBomb 360 coverage, event pairing and target taxonomy: a read-only investigation

Date: 2026-11-18.

**Scope:**
- the **304 Tier B development matches**, which are the experiment-2/4 folds;
- the final-test matches were never opened (asserted in every script);
- no existing file was modified, no model was trained, nothing was committed;
- this directory is the only output.

**Purpose.** Establish one justified definition of the next-event prediction task before
experiment 5.

**Evidence labels:**
- **[F]** a confirmed fact, measured on raw StatsBomb JSON;
- **[C]** a fact about the code, with a file and line reference;
- **[J]** scientific judgment or recommendation;
- **[A]** an assumption.

**Reproduce** (from the repository root; about 7 minutes in total):
```
.venv/bin/python results/investigation_360_taxonomy_2026-11-18/diagnostics/investigate.py
.venv/bin/python results/investigation_360_taxonomy_2026-11-18/diagnostics/pairs_extra.py
.venv/bin/python results/investigation_360_taxonomy_2026-11-18/diagnostics/actor_and_shared_frames.py
```

**Independence of the scripts.** They parse the raw `events/<id>.json` and
`three-sixty/<id>.json` themselves. Project code is imported only once:
`v3_taxonomy.companion_rules`, to compare it with an independent re-implementation. The
two agree on every record.

| Folder / file | Content |
|---|---|
| `tables/p1_*` | coverage by event type, missing-360 analysis, Ball Receipt\*, actor point, shared snapshots |
| `tables/p2_*` | protocols A/B/C: sizes, class distributions, transitions, A-vs-B shift |
| `tables/p3_*`, `p3b_*` | Goal Keeper / Duel / 50/50, residual pairs, same-timestamp transitions |
| `tables/p4_*` | scenarioD15 vs actions15 vs the 12-class alternative |
| `tables/p5_*` | the 15-class per-class table of experiment 4 |
| `examples/*.csv` | real sequences with event ids, timestamps, teams, players, coordinates and 360 flags |
| `summary*.json` | every headline number quoted below |

---

## 0. Summary for the supervisor (one page)

1. **Missing 360.** StatsBomb's own specification states that 360 frames are collected
   from broadcast video and that "not all events in the match will receive a 360 frame"
   (`data/raw/statsbomb/doc/Open Data 360 Frames v1.0.0.pdf`, p. 3).
   - **[F] Coverage.** In Tier B, 87.3 % of raw events have a 360 record. Among those
     records there are no empty freeze frames and only one without an actor.
   - **[F] Where 360 is missing:**
     - administrative records (substitutions, half start/end, 0 %);
     - events flagged `off_camera` (97 % lack 360);
     - events right after stoppages;
     - above all, the Goal Keeper annotation `Shot Faced`, with only 2.7 % coverage.
   - **[C] Our code never deletes or alters a source snapshot.** Orientation and identity
     corrections work on copies, no code writes to `data/raw`, and no raw file has been
     modified since the download (26 May 2026).
2. **[F] The 360 "actor" is not an independent position.** The actor point equals the
   event's annotated location in 93.8 % of 360 records (≥ 99.6 % for Pass, Carry,
   Pressure, Shot, Goal Keeper).
   - **Shared snapshots.** Paired records often carry the *identical* snapshot:
     Ball Receipt\* → Carry 91.6 %, Dispossessed → Duel 83.5 %.
   - **Consequence.** The ball position is not observed independently. The actor marker
     is the annotated event location.
3. **[F] Ball Receipt\* is redundant with the Pass record.** Every receipt is linked to its
   pass. The receiver is always the pass's `recipient`. The location equals the pass
   `end_location` for complete passes (median distance 0). 85.7 % of passes are followed
   immediately by a receipt.
   - **[J] Its exclusion is justified,** but only in retrospect. It began as an
     undocumented implementation choice (commit f12931a, 2026-07-30) and was justified
     later by the taxonomy research of 2026-10-27.
4. **[F] Training population (audit R1 reproduced).**
   - **Current state.** Experiment 4 trains on all transitions, but evaluates only rows
     whose current event has 360.
   - **Effect.** Training the Markov chain on the 360 rows only lowers the log loss by
     0.00152, in all 5 folds.
   - **[J] Recommendation: Protocol B.** Use the same rows for training and evaluation,
     with targets defined on the full event sequence.
   - **Protocol C rejected.** Dropping non-360 events *before* building targets changes
     the target for 5.7 % of rows and makes it depend on camera coverage.
5. **[F] Paired records.**
   - **Duel.** `Dispossessed → Duel` has P = 0.9996. All 11,135 Tackle duels are linked
     companions of a Dispossessed or an incomplete Dribble at the same timestamp.
   - **50/50.** 98.5 % of consecutive 50/50 pairs share the timestamp, involve opposite
     teams and are mutually linked: one contest recorded twice.
   - **Goal Keeper.** 53 % of GK records are `Shot Faced` and 9 % `Goal Conceded`. These
     are annotations, not keeper actions. Saves, claims, punches and sweeping (37 %) are
     genuine actions.
   - **Residual pairs.** `actions15` removes exactly these companions (K1–K3), but leaves
     residual two-record contests: every `Aerial Lost` duel is linked to the winner's
     header, recorded at the same timestamp.
6. **[F] Macro-F1.** Duel, Goal Keeper and 50/50 give 53.3 % of the summed per-class F1
   of experiment 4, but only 2.6 % of its correct predictions. Macro-F1 is 0.214 over 15
   classes and 0.125 over the other 12.
7. **[J] Recommended canonical task:**
   - `actions15` (deduplicate companion records; keep all 15 classes);
   - Protocol B;
   - the current-event 360 snapshot is required, the target-event snapshot is not;
   - event order by `index`;
   - no class exclusion (the 12-class alternative deletes 12,601 genuine actions);
   - versioned under a new task name, so that experiments 1–4 stay reproducible.

   This needs the supervisor's agreement (§7).

---

## 1. Part 1: why some events have no 360 snapshot

### 1.1 Data flow (code trace) [C]

| Step | Code | What happens to 360 |
|---|---|---|
| Raw read for the sequence models (experiments 1–4) | `src/sequence_models/raw_events.py:84, 104–105, 119` | joins frames by `event_uuid`; `has_360` = a record exists; `has_freeze_frame` is recorded but not used for eligibility |
| Cache | `raw_events.py:139–155` | derived pickle in `data/interim`, rebuilt if its match set differs; raw files only read |
| Task construction | `src/sequence_models/tasks.py:84` (drop period 5), `85–86` (keep the vocabulary types), `103–104` (`eval_eligible = prediction point & has_360`) | 360 decides only which rows are evaluated, never the sequence or the target |
| Experiment-4 training rows | `src/sequence_models/f0_experiment.py:284` (`y_all >= 0`: every transition) | training ignores 360 (Protocol A) |
| Production 8-class dataset | `src/build_statsbomb_360_dataset.py:377–381` (skip if no frame, or not a production type), `405–414` | orientation and team normalisation applied to **copies** |
| Orientation correction | `src/orientation_correction.py:364–377` ("returns a corrected COPY … never mutates the input") | copy |
| Team normalisation | `src/team_normalization.py:293–307` | copy; flags never touched |
| Renderer's usable set | `src/orientation_resolution.py:154–167` | a record with a non-empty `freeze_frame`; the others are not drawn, not deleted |

**Conclusion** [C/F]:
- **No code writes to raw.** No code path writes to `data/raw`; the write calls of
  `src/` and `scripts/` were grepped.
- **Raw files unchanged.** None of the 8,977 raw JSON files has a modification time after
  the download date.
- **Visualization, orientation correction and team identity** never delete or alter a
  source snapshot. They produce derived, corrected copies.
- **The sequence models** (experiments 1–4) use no corrected coordinates at all.

### 1.2 Coverage by event type [F]

Full table: `tables/p1_event_coverage_by_type.csv`. It has all 33 types, with the
mutually exclusive categories 1–4 and the input/target counts.

| Type | Raw | With 360 | % | Without | Period 5 | Not in taxonomy | Retained D15 | Train inputs (A) | Eval inputs | Targets (any) | Targets (eval) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pass | 317,269 | 269,764 | 85.0 | 47,505 | 0 | 0 | 317,269 | 316,894 | 269,476 | 316,627 | 274,245 |
| Ball Receipt\* | 299,981 | 268,286 | 89.4 | 31,695 | 0 | 299,981 | 0 | 0 | 0 | 0 | 0 |
| Carry | 255,369 | 233,270 | 91.3 | 22,099 | 0 | 0 | 255,369 | 255,369 | 233,270 | 255,369 | 227,729 |
| Pressure | 96,665 | 85,978 | 88.9 | 10,687 | 0 | 0 | 96,665 | 96,649 | 85,965 | 96,665 | 85,262 |
| Ball Recovery | 29,691 | 26,568 | 89.5 | 3,123 | 0 | 0 | 29,691 | 29,665 | 26,548 | 29,691 | 25,733 |
| Duel | 19,684 | 17,250 | 87.6 | 2,434 | 0 | 0 | 19,684 | 19,668 | 17,239 | 19,684 | 16,218 |
| Goal Keeper | 9,381 | 4,289 | **45.7** | 5,092 | 113 | 0 | 9,268 | 9,179 | 4,254 | 9,268 | 8,793 |
| Shot | 7,773 | 7,532 | 96.9 | 241 | 113 | 0 | 7,660 | 7,660 | 7,462 | 7,660 | 7,199 |
| 50/50 | 1,223 | 1,094 | 89.5 | 129 | 0 | 0 | 1,223 | 1,222 | 1,093 | 1,223 | 1,091 |
| Substitution | 2,622 | 0 | 0 | 2,622 | 0 | 2,622 | 0 | — | — | — | — |
| Half Start / End | 1,308 each | 0 | 0 | 1,308 | 24 | 1,284 | 0 | — | — | — | — |
| Starting XI, Tactical Shift, Player On/Off, Bad Behaviour, Injury Stoppage | 202–1,711 | 0 | 0 | all | ≤ 4 | all | 0 | — | — | — | — |
| Referee Ball-Drop | 750 | 12 | 1.6 | 738 | 0 | 750 | 0 | — | — | — | — |

**Totals.** 1,123,707 raw events; 981,431 360 records (87.3 %).
- Every 360 record matches an event; there are 0 orphan frames.
- 0 records have an empty `freeze_frame`, 1 has no actor (a 50/50), and 24 have more
  than one actor.
- Per match, the share of events with 360 ranges from 74.8 % to 95.3 % (median 87.3 %).

**The four categories.** These are mutually exclusive, per type, in the CSV:
1. **No 360 record in the original StatsBomb data.** 142,276 events. They are spread over
   all types, plus every administrative record.
2. **Has 360 but filtered by preprocessing.** Only the penalty-shoot-out period 5 (113
   Shots and 113 Goal Keeper records; their 360 status is in the CSV). No other
   preprocessing step removes 360 events from the sequence-model task.
3. **Has 360 but not in the 15-class taxonomy.** 268,286 Ball Receipt\*, 6,082 Foul Won,
   4,784 Dribbled Past, 272 Shield, and a few hundred others.
4. **Other reasons.** A retained event with 360 but no target (the last of its period);
   at most a few hundred per type. Empty freeze frames: 0.

**Why there is no 360** [F, with the interpretation marked [J]]:

| Situation | 360 coverage | Source |
|---|---|---|
| Administrative / referee records | 0 % | — |
| Events flagged `off_camera` | 97 % have no 360 (12,896 Passes) | — |
| Previous raw record Referee Ball-Drop | 40 % | `p1_no360_by_previous_raw_type_d15.csv` |
| Previous raw record Substitution | 46 % | same |
| Previous raw record Injury Stoppage | 50 % | same |
| Average | 87 % | — |
| Goal Keeper `Shot Faced` | 2.7 % (4,951 records) | `p3_goal_keeper_subtypes.csv` |
| Genuine keeper actions | 94–97 % | same |
| Incomplete Ball Receipt\* of a pass that went `Out` | 4.4 % | — |

**Missing-360 runs** (D15 events):
- 32 % of the 98,338 missing-360 D15 events are isolated single events;
- 15 % lie in runs of ≥ 5 consecutive missing events.

**[J] Interpretation.** Missing 360 reflects:
- the broadcast camera (off-camera events, and probably replays and camera cuts);
- stoppages;
- annotation records that are not associated with a camera frame.

It is *not* produced by our pipeline.

**[A] Unverified.** Replays and camera cuts are not observable in the data and were not
checked against video.

### 1.3 The 360 actor is the annotated event location, not an independent measurement [F]

`tables/p1b_actor_vs_event_location_by_type.csv`:

| Records | Share where the actor point equals the event's `location` (distance < 0.05) |
|---|---|
| All 360 records | 93.8 % |
| Pass, Carry, Pressure, Shot, Goal Keeper, Clearance, Block, Interception, Foul Committed, Ball Recovery | ≥ 99.6 % each |
| Dispossessed | 0.9 % |
| Foul Won | 0.2 % |
| Dribbled Past | 0.1 % |
| Duel | 55 % |
| 50/50 | 52 % |
| Dribble | 61 % |

- **Paired records.** The low-share records are those that share a snapshot with their
  partner (§1.4). There, the actor point belongs to the partner's location.
- **[J] Limitation, stated explicitly.**
  - StatsBomb 360 contains **no ball coordinates** and no continuous tracking.
  - The actor marker is placed at the event's annotated location.
  - For on-ball events, that location approximates the point of the ball action. This
    is an annotation, not a measured ball position.
  - For paired records, the actor marker may not even belong to the record's own player.
  - Neither the actor position nor the event location may be described as the ball
    position. Any "distance to ball" feature is a distance to the annotated event
    location.

### 1.4 Paired records frequently share one identical snapshot [F]

`tables/p1b_shared_snapshot_pairs.csv`. Measure: the share of consecutive raw pairs
(first → second) whose two freeze frames are identical.

| Pair | Share identical |
|---|---|
| Ball Receipt\* → Carry | 91.6 % (200,190) |
| Ball Recovery → Carry | 89.9 % |
| Dribbled Past → Dribble | 86.2 % |
| Dispossessed → Duel | 83.5 % |
| Foul Committed → Foul Won | 83.1 % |
| Ball Receipt\* → Pass | 42.1 % |

**Consequences for prediction:**
- The 360 snapshot of \(E_t\) can be literally the same picture as that of
  \(E_{t+1}\), for example a Ball Recovery and the Carry that starts from it.
- This is not leakage, because the snapshot belongs to the current moment. But it means
  that two records share one instant, so they are not two observations of the game
  state.

### 1.5 Ball Receipt\* [F]

`tables/p1_ball_receipt_vs_linked_pass.csv`, `p1_pass_next_raw_is_ball_receipt_by_pass_outcome.csv`,
`examples/ex_pass_ball_receipt.csv`.

**How a receipt relates to its pass:**
- **Count and coverage.** 299,981 receipts, 89.4 % of them with 360.
- **Link.** 100 % are linked in `related_events` to a Pass.
- **Receiver.** It is always the pass's `recipient` (100 %), on the same team (100 %).
- **Location.** For complete passes (260,399), the receipt location equals the pass
  `end_location` (median distance 0.0; 88 % within 1 unit overall).
- **Timing.** The median time from pass to receipt is 1.31 s: the arrival time.
- **Next action.** After a receipt, the next retained action is a Carry in 72.8 % of
  cases, and its snapshot is usually identical (§1.4).

**P(next raw record is Ball Receipt\* | Pass)**, by the pass's own outcome:

| Pass outcome | P |
|---|---|
| Complete | 0.908 |
| Incomplete | 0.626 |
| Out | 0.609 |
| Offside | 0.992 |

**Real examples** (`examples/ex_pass_ball_receipt.csv`; ids shortened to 8 characters).
Coordinates are in the acting team's direction; 360 is the record's own frame.

| Match | Index | Id | Time | Type | Team / player | Location → end | 360 | 360 actor point |
|---|---|---|---|---|---|---|---|---|
| (Tier B) | 131 | 258a49f9 | 00:01:53.515 | Pass (complete, short) | 217 / 5213 | (42.7, 50.8) → (48.1, 64.1) | yes | (42.7, 50.8) = the pass location |
| | 132 | abce4dd5 | 00:01:54.680 | Ball Receipt\* | 217 / 43728 | (48.1, 64.1) | yes | (48.1, 64.1) = the pass end |
| | 133 | 07d4d802 | 00:01:54.680 | Carry | 217 / 43728 | (48.1, 64.1) → (48.4, 64.8) | yes | same picture as the receipt |
| | 2939 | f7405a0f | 00:24:04.451 | Pass (complete, long) | 158 / 7440 | (84.1, 32.2) → (99.6, 71.0) | yes | at the pass location |
| | 2940 | ad572908 | 00:24:06.618 | Ball Receipt\* | 158 / 4260 | (99.6, 71.0) | yes | at the pass end |
| | 381 | 3ac3283a | 00:10:57.017 | Pass (Incomplete) | 1211 / 403287 | (52.0, 7.1) → (57.4, 5.7) | yes | — |
| | 382 | e1c0dc3e | 00:10:57.291 | Ball Receipt\* (Incomplete) | 1211 / 25447 | (66.3, 3.5) | yes | (62.7, 74.4) = the **interception's** location |
| | 383 | 8d7c918b | 00:10:57.291 | Interception | 2391 / 402655 | (62.7, 74.4) | yes | (62.7, 74.4) |
| | 118–120 | badeacaa / 19600bc7 / c24a4cf5 | 00:02:27.238 / 27.981 / 27.981 | Pass → Receipt → Pass | 217 | (96.0, 11.3) → (91.8, 17.2) | **no** on all three | — |

The full rows, with match ids and complete UUIDs, are in the CSV.

**[J] One action or two?**
- **Two moments.** A pass and its receipt are physically distinct moments, about 1.3 s
  apart, by two players: kick and control.
- **But no new information.** The receipt adds no information that the Pass record does
  not already contain: its recipient, end location and outcome. Its occurrence is
  determined by the pass outcome (P = 0.91 for complete passes).
- **Shared snapshot.** In the incomplete-pass example, the receipt's snapshot is the
  interceptor's picture. Its actor marker is the interceptor, not the receiver.

**[J] Is the exclusion justified?**
- **History.** Excluding Ball Receipt\* began as a historical implementation decision.
  `MEANINGFUL_EVENT_TYPES` (`src/build_statsbomb_360_dataset.py:74–83`) came without a
  rationale in the first builder commit (f12931a, 2026-07-30).
- **Later justification.** The taxonomy research
  (`results/event_taxonomy_research_2026-10-27/README.md`, lines 18, 60, 205) and the
  measurements above justify it:
  - **Domination by bookkeeping.** As a target, Ball Receipt\* would dominate the task
    with a near-deterministic bookkeeping step (Pass → Ball Receipt\* is 24 % of all
    rows at P = 0.86).
  - **Redundancy.** It duplicates information already in the pass record.
  - **Literature.** SPADL (socceraction) also treats it as a non-action.
- **Verdict.** The exclusion is scientifically defensible. The thesis should say that it
  was formalised retrospectively.
- **A different task.** Predicting whether a pass is *received* (pass completion) is a
  legitimate but different task.

**Caution for experiment 5** [J]. Several fields of the *current* record describe its
outcome:
- `pass.outcome`, `pass.end_location`, `pass.recipient`;
- `shot.outcome`;
- `duel.outcome`.

They are as informative about \(E_{t+1}\) as the receipt itself. For example, an
incomplete pass is followed by an Interception, Ball Recovery or Block. Whether they are
allowed as features depends on the defined *prediction time* (§6). The current allowlist
does not contain them.

---

## 2. Part 2: all transitions vs 360-only transitions (audit R1 reproduced)

### 2.1 Three concepts [F]

| Concept | Definition | Tier B count |
|---|---|---|
| A. Match with 360 coverage | a three-sixty file with ≥ 1 frame | 304 / 304 matches (per-match event coverage 74.8–95.3 %) |
| B. Event with a usable 360 snapshot | a 360 record with a non-empty `freeze_frame` (identical to "record exists" here: 0 empty frames) | 981,431 raw events; 701,613 retained D15 events in periods 1–4 |
| C. Prediction row with current-event 360 | a retained D15 event, with a target in its period, with a usable 360 snapshot; the target may lack 360 | **701,144 rows**, of which 645,590 (92.1 %) also have a target snapshot |

**[C] Experiment 4 filtering** — evaluation rows are C:
- `tasks.py:104`, `eval_eligible = is_prediction_point & has_360`;
- `f0_experiment.py:237–238`, the validation rows are restricted to `eval_eligible`.

Training rows are every transition (`f0_experiment.py:284`). This is Protocol A.

### 2.2 Protocols A, B and C

`tables/p2_protocol_sizes.csv`, `p2_class_distribution_by_protocol.csv`,
`p2_transitions_by_protocol.csv`, `p2_distribution_shift_A_vs_B.csv`.

| | A: all retained transitions | B: 360-only training rows | C: 360-only event sequence |
|---|---|---|---|
| Target definition | next retained D15 event (any 360) | **same as A** | next retained D15 event **that has 360** |
| Sequence observations | 799,951 | 799,951 | 701,613 |
| Training rows | 799,309 | 701,144 | 700,971 |
| Evaluation rows | 701,144 | 701,144 (identical rows and targets) | 700,971 (**different targets**) |
| Eval rows skipping ≥ 1 raw record | 38.5 % | 38.5 % | 43.8 % |
| Mean raw records skipped | 0.39 | 0.39 | 0.60 |
| Zero time gap / median gap / 90th percentile | 7.3 % / 1.09 s / 3.0 s | same | 7.3 % / 1.13 s / 3.7 s |

**Protocol C changes the task** [F]. Compared with A on the same current events:
- the target **event** changes for 55,554 rows;
- the target **class** changes for 40,195 rows (5.7 %);
- Goal Keeper falls from 1.25 % to 0.61 % of targets, because `Shot Faced` has no
  snapshot.

C therefore predicts "the next event that happened to be filmed". A and B predict "the
next event". **C is not comparable with A/B, and its target depends on broadcast
coverage.**

**A vs B distribution shift** (training rows, percentage points; B − A):

| Class | Target share A | B | Δ target | Δ current event |
|---|---|---|---|---|
| Pass | 39.61 % | 39.11 % | −0.50 | **−1.21** |
| Carry | 31.95 % | 32.48 % | +0.53 | **+1.32** |
| Duel | 2.46 % | 2.31 % | −0.15 | 0.00 |
| Goal Keeper | 1.16 % | 1.25 % | +0.10 | **−0.54** |
| Shot | 0.96 % | 1.03 % | +0.07 | +0.11 |
| Pressure | 12.09 % | 12.16 % | +0.07 | +0.17 |

**The 98,165 rows that only A contains** (`p2_training_rows_without_360_by_current_type.csv`):
- Pass 48 %, Carry 23 %, Pressure 11 %, Goal Keeper 5 %;
- that is, restarts, off-camera play and keeper annotations.

**Markov reproduction of audit R1** [F]. First-order chain, α = 1, experiment-2 folds,
identical evaluation rows; count-based only:

| Training population | Log loss |
|---|---|
| A | 1.151973 (= experiment 4) |
| B | 1.150457 |

Δ = **−0.001516**. B is better in all 5 folds (per fold −0.0019, −0.0009, −0.0014,
−0.0016, −0.0017; `summary.json → p2_markov1_alpha1_log_loss`).

**[J] Recommendation: Protocol B.**
- **The same population.** The thesis predicts the next event *from current-event 360
  geometry*. Every model must therefore be trained and evaluated on exactly the same
  population: rows whose current event has a usable snapshot.
- **The target stays as in A/B.** It is the next retained event, whether or not it was
  filmed.
- **Protocol A** can be kept as a labelled sensitivity analysis for non-spatial
  baselines. Its extra 12 % of rows come from a different situation mix, which is why
  B's log loss is better.
- **Protocol C is rejected** for the main task.

---

## 3. Part 3: paired records (Goal Keeper, Duel, 50/50)

**Sources:**
- `tables/p3_conditional_probabilities_both_directions.csv`;
- `tables/p3_goal_keeper_subtypes.csv`, `p3_shot_outcome_vs_next_goal_keeper_subtype.csv`;
- `tables/p3_duel_subtypes.csv`, `p3_duel_related_event_types.csv`;
- `tables/p3b_residual_pairs_actions15.csv`, `p3b_actions15_same_timestamp_pairs.csv`;
- the full statistics in `summary.json → p3`.

**Scope.** All neighbours are taken in the retained D15 sequence (periods 1–4).

### 3.1 Goal Keeper [F]

**Overview.**
- **Volume and coverage.** 9,268 records (periods 1–4): 4,285 with 360, 4,983 without.
- **Neighbours.**
  - Previous retained event: Shot 59.3 %, Block 23.4 % (blocked shots), Pass 14.1 %.
  - Next retained event: Pass 64.0 %, Ball Recovery 15.3 %.
- **The two requested probabilities.**
  - **P(next = Goal Keeper | current = Shot) = 0.717**
  - **P(previous = Shot | current = Goal Keeper) = 0.593**
- **Link to the previous event.** 74.2 % of GK records are linked to the previous retained
  record in `related_events`.
- **Timing.** The time from the previous record has a median of 0.75 s, and only 0.1 %
  are at the same timestamp. These are sequential records, **not** same-timestamp
  duplicates.

**Subtypes:**

| Subtype | n | Share | With 360 | Previous = Shot | Interpretation [J] |
|---|---|---|---|---|---|
| Shot Faced | 4,951 | 53.4 % | **2.7 %** | 58.9 % (the rest follow a Block) | annotation: the keeper faced a shot without acting (shot Off T 2,329, Wayward 397, Post 158, Blocked 30) |
| Shot Saved (+ to Post / Off Target) | 1,805 | 19.5 % | 97 % | 95 % | **genuine save** (follows a shot with outcome `Saved`) |
| Goal Conceded | 809 | 8.7 % | 97 % | 95 % | annotation (follows a shot with outcome `Goal`; outcome "No Touch" in the example) |
| Keeper Sweeper | 627 | 6.8 % | 94 % | 0 % | genuine action |
| Collected | 508 | 5.5 % | 95 % | 0 % | genuine claim |
| Punch | 433 | 4.7 % | 97 % | 0 % | genuine action |
| Penalty Conceded / Penalty Saved (+ to Post) | 72 / 23 | — | — | 100 % | annotation / genuine save |
| Smother, Save | 23, 17 | — | — | 0 % | genuine actions |

**[J] Verdict.** The paired-record hypothesis is **confirmed for `Shot Faced`, `Goal
Conceded` and `Penalty Conceded`** (62 %). These records describe the shot's
consequence, not a keeper action. It is **rejected for saves, claims, punches, sweeping
and smothers** (37 %). These are real keeper actions and legitimate targets, even though
a save is causally predictable from the shot.

### 3.2 Duel [F]

**Overview.**
- **Volume and coverage.** 19,684 records: 17,250 with 360.
- **Subtypes.** **Tackle 11,135** and **Aerial Lost 8,549**; no other duel types occur.
- **Neighbours.**
  - Previous retained event: Pass 38.3 %, Dispossessed 36.8 %, Dribble 19.8 %.
  - Next retained event: Pass 40.6 %, Ball Recovery 16.2 %, Carry 14.1 %, Clearance 13.0 %.

**Tackle.**
- 100 % of Tackles share the timestamp of the previous raw record **and** are linked to
  it.
- Their partner records: Dispossessed 7,232, Dribble 3,894 (`p3_duel_related_event_types.csv`).
- **P(next = Duel | current = Dispossessed) = 0.9996.**
- **P(previous = Dispossessed | current = Duel) = 0.368.**
- **[J] Verdict.** One contest, recorded twice: the dispossessed player's record plus the
  tackler's record.

**Aerial Lost.**
- All 8,549 are linked to an opponent record carrying `aerial_won` (Pass 5,259,
  Clearance 2,335, Shot 720, Miscontrol 235).
- That record has the same timestamp and is the adjacent raw record in 100 % of cases.
- The winner's record comes *after* the loser's in 86 % of cases.
- **[J] Verdict.** One aerial contest, recorded as the loser's Duel plus the winner's
  header. The header is a genuine on-ball action. The Aerial Lost record is the
  opponent's participation in the same contest.
- **Gap in actions15.** `actions15` does **not** deduplicate this pair (§4).

**Consecutive Duel → Duel.**
- Only 84 consecutive retained pairs (P = 0.004).
- **0 %** share a timestamp, and 89 % involve opposite teams.
- **[J] Verdict.** These are **separate duels** (e.g. two successive aerial challenges
  about 1.4 s apart in `examples/ex_goal_keeper_duel_5050.csv`), not repeated
  annotations.

### 3.3 50/50 [F]

**Overview.**
- **Volume and coverage.** 1,223 records: 1,094 with 360.
- **P(next = 50/50 | current = 50/50) = 0.504.**
- **Consecutive pairs.** 616 pairs:
  - **98.5 % at exactly the same timestamp**;
  - **99.5 % opposite teams**;
  - **98.5 % mutually linked**;
  - **98.5 % satisfy all three**, i.e. they appear to describe one physical contest.

**Outcome pairs** (complementary results of one contest):

| First record | Second record | n |
|---|---|---|
| Lost | Success To Team | 239 |
| Lost | Won | 230 |
| Lost | Success To Opposition | 72 |
| Success To Opposition | Lost | 66 |

**Singletons.** 9 records have no 50/50 partner.

**[J] Verdict.** One contest, recorded once per player.

### 3.4 Concrete examples

`examples/ex_goal_keeper_duel_5050.csv` holds 32 focus events, with ±2 neighbours, full
UUIDs, teams, players and 360 flags. A selection (ids shortened):

| Hypothesis | Match | Sequence (index: type [subtype/outcome], team, time) |
|---|---|---|
| supports (GK annotation) | 3998844 P2 | 1699 Shot [Blocked] 860, 08:08.823 → 1700 Block 862, 08:08.909 → **1701 GK [Shot Faced] 862, 08:09.131, no 360** |
| supports (GK annotation) | 3941021 P2 | 2993 Shot [Goal] 941, 47:42.173 → **2994 GK [Goal Conceded / No Touch] 905, 47:42.762** → 2995 Pass [Kick Off] |
| contradicts (genuine save) | 3893793 P2 | 2943 Shot [Saved] 1210, 27:49.953 → **2944 GK [Shot Saved / Success] 6045, 27:50.369, with 360** → 2945 Pass (keeper) |
| contradicts (GK action not after a shot) | 3835319 P1 | 148 Pass [Incomplete] 865 → 149 Ball Receipt\* [Incomplete] 865 → **150 GK [Keeper Sweeper / Claim] 859, 03:16.139** → 151 Carry (keeper) |
| supports (Tackle companion) | 3835336 P1 | **701 Dispossessed 852, 18:43.803 → 702 Duel [Tackle / Success In Play] 859, 18:43.803** (same timestamp, linked) |
| supports (Tackle companion of a dribble) | 3764661 P1 | **1325 Dribble [Incomplete] 217, 33:39.804 → 1326 Duel [Tackle / Lost In Play] 215, 33:39.804** |
| supports (aerial pair) | 3869219 P1 | **1132 Duel [Aerial Lost] 785, 29:26.161 → 1133 Clearance (aerial winner) 778, 29:26.161** |
| contradicts (separate duels) | 3794690 P1 | 1067 Duel [Aerial Lost] 912, 29:37.235 … 1069 Duel [Aerial Lost] 941, 29:38.668: two contests 1.4 s apart, with different players |
| supports (50/50 pair) | 3893791 P1 | **727 50/50 [Lost] 1832, 17:47.420 → 728 50/50 [Success To Team] 863, 17:47.420** |
| contradicts (single 50/50) | 3893805 P1 | 1537 Pass [Incomplete] 1832 → **1538 50/50 [Lost] 1210, 40:34.129, no partner** → 1539 Ball Recovery 1210 |

### 3.5 Same timestamp ≠ duplicate [F]

Same-timestamp transitions under actions15: 46,091, i.e. 5.9 % of transitions
(`p3b_actions15_same_timestamp_pairs.csv`).

**Linked pairs (93.8 %).** The largest group is **Ball Recovery → Carry, same player
(24,909)**. A Carry starts at the timestamp of the preceding on-ball record by StatsBomb
convention. Recovery and carry are two **separate actions**, not a duplicate. The same
holds for Dribble → Carry (5,020).

**Unlinked same-timestamp pairs** (e.g. Dispossessed → opponent's Carry, 1,698) are
separate records.

**Conclusion.** Deduplication must never rely on timestamp equality alone. The K1–K3 rules
additionally require a `related_events` link and a specific type pattern
(`v3_taxonomy.py:53–56`).

---

## 4. Part 4: scenarioD15 vs actions15 vs a 12-class alternative

### 4.1 Exact rules [C]

**scenarioD15.**
- **Vocabulary.** The 15 types of `configs/markov_hmm_cost_sensitive_v2.yaml:22–31`, in
  the canonical order of `src/sequence_models/event_classes.py`.
- **Filtered-event sequence.** Non-vocabulary records are skipped: `tasks.py:85–86`.
- **Excluded period.** Period 5 (yaml line 13; `tasks.py:84`).
- **No record-level rule.**

**actions15** (`src/sequence_models/v3_taxonomy.py`).
- **Classes.** The same 15 class names (line 40).
- **Companion records.** A record matching one of the rules below is relabelled
  `__companion__` (lines 72–79) and skipped like any non-vocabulary record.
- **Rule K1** (line 53): `Goal Keeper` with subtype `Shot Faced`, `Goal Conceded` or
  `Penalty Conceded` (line 41).
- **Rule K2** (lines 54–55): `Duel/Tackle` that meets all of:
  - the same timestamp as the previous raw record;
  - linked to it in `related_events`;
  - that previous record is `Dispossessed`, or a `Dribble` with outcome `Incomplete`.
- **Rule K3** (line 56): `50/50` whose previous raw record is a `50/50` with the same
  timestamp and a link.
- **Check.** My independent re-implementation equals the project's function on all
  1,123,707 records (`summary.json → p4_independent_rules_equal_project_rules = true`).
- **Rule counts, periods 1–4:** K1 5,832, K2 11,135, K3 607.

**alt12** (investigated only, not implemented in the project). Goal Keeper, Duel and
50/50 are removed from the vocabulary and skipped entirely.

### 4.2 Comparison [F]

`tables/p4_taxonomy_comparison.csv`, `p4_taxonomy_class_counts.csv`:

| | scenarioD15 | **actions15** | alt12 |
|---|---|---|---|
| Classes (canonical ids) | 15 (0–14) | 15 (0–14, same) | 12 (ids 0–8, 10, 11, 13 of D15) |
| Retained events | 799,951 | 782,377 | 769,776 |
| Removed vs D15 | — | 17,574 (2.2 %) | 30,175 (3.8 %) |
| Transitions | 799,309 | 781,735 | 769,134 |
| Eval rows (current 360) | 701,144 | 690,147 | 678,458 |
| D15 eval rows dropped as current event | — | 10,997 | 22,686 |
| D15 eval rows with a changed target event / class | — | 15,538 / 15,491 | 25,230 / 25,230 |
| Share of D15 eval rows affected | — | 3.8 % | 6.8 % |
| Match periods with ≥ 1 affected row | — | 100 % | 100 % |
| Eval rows skipping ≥ 1 raw record | 38.5 % | 41.3 % | 42.6 % |
| Zero-time-gap transitions | 7.3 % | 6.0 % | 4.8 % |
| Rows in cells with P(next \| current) ≥ 0.9 | 0.9 % (Dispossessed → Duel, 1.000) | **0 %** | 0 % |
| H(target) / H(target \| current) [bits] | 2.39 / 1.66 | 2.30 / 1.67 | 2.20 / 1.62 |

**Class counts.**

| Class | scenarioD15 | actions15 | alt12 |
|---|---|---|---|
| Duel | 19,684 | 8,549 (Aerial Lost only) | 0 |
| Goal Keeper | 9,268 | 3,436 (genuine keeper actions) | 0 |
| 50/50 | 1,223 | 616 (one record per contest) | 0 |

All other classes are unchanged.

**Genuine actions alt12 would delete** (`p4_records_removed_by_alt12_but_kept_by_actions15.csv`):
**12,601 records**:
- 8,549 Aerial Lost;
- 1,738 Shot Saved, 627 Keeper Sweeper, 508 Collected, 433 Punch, 130 other saves and
  smothers;
- 616 50/50 contests.

**[J] Rejection of alt12.** Excluding whole classes removes real football actions. It
re-routes Shot → GK save into "Shot → whatever comes after the save". It still leaves the
`Dispossessed`/`Dribble` side of each tackle pair. **Deduplicating companion records
(actions15) is preferable.**

**Before/after sequences** (`examples/ex_before_after_taxonomies.csv`; 6 sequences, 2 per
rule):

| Rule | Example (match, period, index: type …) | scenarioD15 | actions15 | alt12 |
|---|---|---|---|---|
| K1 | 3788765 P2: 2696 Shot [Blocked] → 2697 Block → **2698 GK [Shot Faced]** → 2699 Pass | Shot → Block → GK → Pass | Shot → Block → Pass | Shot → Block → Pass |
| K2 | 3788766 P1: 1183 Dispossessed 907 → **1184 Duel [Tackle, Won] 914** (same timestamp 30:31.686) → 1185 Carry 914 | Dispossessed → Duel → Carry | Dispossessed → Carry | Dispossessed → Carry |
| K3 | 3837785 P2: 3084 50/50 [Lost] 167 → **3085 50/50 [Won] 131** (20:04.350) → 3086 Carry | 50/50 → 50/50 → Carry | 50/50 → Carry | Miscontrol → Carry (50/50 gone) |

### 4.3 Residual pairs that actions15 keeps [F]

`p3b_residual_pairs_actions15.csv`:

| Pattern | n | P(second \| first) | Same timestamp | Linked | Median time gap | [J] Assessment |
|---|---|---|---|---|---|---|
| Aerial Lost → the winner's header (Pass / Clearance / Shot / Miscontrol) | 8,549 | 1.00 (linked partner); in the sequence, next = Pass 52 %, Clearance 28 %, Shot 8 % | 86 % (adjacent) | 86 % | 0 s | one contest, two players. The *class* of the next event remains informative (header type), but the pair inflates predictability. Candidate rule **K4**, needs a decision |
| Shot → keeper save | 1,736 | 1.00 given a `Saved` shot | 0 % | 100 % | 0.71 s | sequential genuine actions; keep |
| Blocked shot → Block | 2,002 | 0.99 | 0 % | 100 % | 0.15 s | the defender's genuine action; keep (predictable from the shot outcome) |
| Incomplete pass → Interception | 6,119 | 0.13 | 0 % | 100 % | 0.97 s | genuine; keep |

---

## 5. Part 5: interpreting macro-F1 (experiment 4, scenarioD15, Logistic F0 = Markov 1, argmax)

`tables/p5_per_class_experiment4_logistic_F0.csv`:

| Class | Support | Precision | Recall | F1 | Predicted | True positives |
|---|---|---|---|---|---|---|
| Pass | 274,245 | 0.634 | 0.868 | 0.733 | 375,845 | 238,163 |
| Carry | 227,729 | 0.668 | 0.911 | 0.771 | 310,612 | 207,561 |
| Ball Recovery | 25,733 | 0 | 0 | 0 | 0 | 0 |
| Clearance | 10,082 | 0 | 0 | 0 | 0 | 0 |
| Dribble | 8,216 | 0 | 0 | 0 | 0 | 0 |
| Shot | 7,199 | 0 | 0 | 0 | 0 | 0 |
| Miscontrol | 7,247 | 0 | 0 | 0 | 0 | 0 |
| Dispossessed | 6,279 | 0 | 0 | 0 | 0 | 0 |
| Pressure | 85,262 | 0 | 0 | 0 | 0 | 0 |
| **Duel** | 16,218 | 0.9995 | 0.378 | 0.548 | 6,132 | 6,129 |
| Interception | 6,210 | 0 | 0 | 0 | 0 | 0 |
| Block | 10,458 | 0 | 0 | 0 | 0 | 0 |
| **Goal Keeper** | 8,793 | 0.716 | 0.608 | 0.658 | 7,462 | 5,345 |
| Foul Committed | 6,382 | 0 | 0 | 0 | 0 | 0 |
| **50/50** | 1,091 | 0.507 | 0.508 | 0.507 | 1,093 | 554 |

1. **Macro-F1, 15 classes** = 3.217 / 15 = **0.2145**.
2. **Macro-F1, the other 12 classes** (denominator 12) = 1.504 / 12 = **0.1253**.
3. **Share of the sum of the 15 per-class F1 scores** attributable to Duel, Goal Keeper
   and 50/50 = 1.713 / 3.217 = **53.3 %**.
4. **Share of all correct predictions** attributable to the three = 12,028 / 457,752 =
   **2.6 %**.

**Why (3) and (4) differ.**
- **Macro-F1 weights every class equally,** whatever its size. A class with 1,091 rows
  counts as much as Pass with 274,245.
- **The three classes are tiny.** Together they are 3.7 % of the targets, but they are
  the only minority classes that argmax ever predicts. Each is predicted from a single,
  almost deterministic predecessor: Dispossessed, Shot, 50/50.
- **Correct predictions follow support.** Correct predictions count each row once, so
  they are dominated by Pass and Carry.
- **Consequence.** Most of macro-F1 here measures how well the model reproduces three
  recording conventions. It does not measure overall predictive skill.
- **No cross-taxonomy comparison.** Macro-F1 under actions15 or alt12 is a different task
  with different classes and targets. It must not be compared with these numbers as if
  it were an improvement or a deterioration.

---

## 6. Part 6: recommended canonical prediction task [J unless marked]

| Question | Recommendation | Reason / evidence |
|---|---|---|
| Current-event 360 required? | **Yes**, for training *and* evaluation rows (Protocol B) | the thesis studies prediction from current 360 geometry; A and B differ measurably (§2.2) |
| Target-event 360 required? | **No** | requiring it changes the target for 5.7 % of rows and ties it to camera coverage (Protocol C, §2.2) |
| May non-spatial baselines use extra transitions? | **Not in the main comparison.** Only as a labelled sensitivity analysis (Protocol A) | otherwise model and population effects are confounded (audit R1, Δ = −0.0015 in favour of B) |
| Ball Receipt\* | **Keep it out** of the vocabulary, as current and as target | redundant with the pass record (recipient, end location, outcome); near-deterministic after a pass; shares the next record's snapshot (§1.4–1.5). It may enter later *history* features only if causally prior |
| Goal Keeper, Duel, 50/50 | **Keep the classes**; deduplicate only the companion records (actions15 K1–K3) | annotation and companion records confirmed (§3); class exclusion deletes 12,601 genuine actions (§4.2) |
| Aerial Lost pairs | **Decide**: (a) keep, documented as a residual pair; or (b) a new rule K4 that drops the loser's record when it is linked, adjacent and shares a timestamp with an `aerial_won` record | 8,549 linked same-timestamp pairs (§4.3). (b) is more consistent with K2/K3, but needs a new taxonomy version |
| Exact-timestamp paired events | Remove **only** when (same timestamp) **and** (`related_events` link) **and** (a documented type pattern) | §3.5: same timestamp alone would wrongly merge 24,909 Ball Recovery → Carry pairs, which are separate actions |
| Separate actions with identical timestamps | Keep both; order by raw `index` | the Carry convention; unlinked pairs |
| Temporal order and leakage | Order by `index` (verified equal to the array order) within (match, period); never by timestamp. Features come only from \(E_t\), its own snapshot and earlier records. **Define the prediction time explicitly:** if it is the moment the action starts, outcome fields of the current record (`pass.outcome`, `end_location`, `recipient`, `shot.outcome`, `duel.outcome`) are future information and must be excluded | §1.5 caution; `configs/ml_feature_classification.json` |
| Taxonomy | **actions15**, as a new, versioned task (e.g. `actions15_v1`, with or without K4) | removes the only near-deterministic cell (Dispossessed → Duel, P = 1.000) and halves the double-counted contests, while keeping all genuine actions |
| Preserving experiments 1–4 | Do not edit `scenarioD15`, `configs/markov_hmm_cost_sensitive_v2.yaml`, `event_classes.py` or any `results/` directory. Introduce the new task under a new name, with a new config and result directories. Re-run the Markov/F0 baselines under the new task before experiment 5. Report old numbers only with their task label | the reproducibility guard of experiment 4 depends on `scenarioD15` staying unchanged |

**Advantages and disadvantages of the recommendation.**

**Advantages:**
- The target means "the next football action, each contest counted once".
- There are no deterministic bookkeeping cells.
- Training and evaluation populations match the spatial research question.
- No genuine action is deleted.

**Disadvantages:**
- The results are not numerically comparable with experiments 2–4; the baselines must be
  re-run.
- The rules depend on StatsBomb's `related_events` conventions.
- The Aerial Lost question remains a judgment call.
- 11–12 % fewer training rows for non-spatial models.

## 7. Prioritised decision list (before experiment 5)

1. **Training and evaluation population:** adopt Protocol B, and keep A only as a
   labelled sensitivity analysis. *Required.*
2. **Taxonomy:** `scenarioD15` (status quo) or `actions15` (recommended). *Required.*
   This determines every target and the meaning of Duel, Goal Keeper and 50/50.
3. **Aerial Lost (K4):** keep, or deduplicate. *Required if actions15 is chosen.*
4. **Prediction time:** define it formally (the start of the current action is
   recommended), and with it which current-record fields are admissible (outcome, end
   location, recipient). *Required before any F1/F2 feature beyond location.*
5. **Naming and versioning:** a new task name and config, and leave experiments 1–4
   untouched. *Required.*
6. **Ball Receipt\*:** confirm the exclusion and document its retrospective
   justification. *Low effort.*
7. **Reporting:** macro-F1 together with per-class F1 and AP. No macro-F1 comparison
   across taxonomies. Report the companion-class contribution. *Recommended.*
8. **Actor and ball limitation:** state in the thesis that the 360 actor point is the
   annotated event location and that no ball tracking exists. *Recommended.*

## 8. Limitations of this investigation

- **Population.** Tier B only (304 matches). The numbers may differ slightly in Tier A
  and in the final test, which was not inspected.
- **Documentation.** The StatsBomb *Events* specification PDF could not be text-extracted
  in this environment (no PDF tools), so the subtype semantics are inferred from the data
  (subtypes, outcomes, links). The *360* specification was read.
- **Off-camera interpretation.** Replays and camera cuts are not observable. The
  "broadcast coverage" interpretation of missing 360 is plausible and consistent with
  StatsBomb's note, but it was not verified against video.
- **"Identical snapshot".** Compared on rounded player coordinates (3 decimals), ignoring
  the flags.
- **Models.** No model was trained. The only model-like computation is the count-based
  Markov reproduction of audit R1 (§2.2), done on experiment 4's rows and folds.
