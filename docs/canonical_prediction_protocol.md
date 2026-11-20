# Canonical prediction protocol: `actions15_360current_v1`

**Status:** canonical. **Approved by the researcher:** 2026-11-19.

**Scope.** This protocol is the default for every experiment started after 2026-11-19.
It stays the default until a newer version is explicitly approved. It is independent of
model performance: no rule may be changed, or an earlier protocol restored, because it
gives better metrics.

| Item | Location |
|---|---|
| Implementation | `src/sequence_models/canonical_task.py` |
| Declarative configuration (validated by the loader) | `configs/actions15_360current_v1.yaml` |
| Dataset builder | `scripts/build_canonical_task.py` |
| Canonical baselines | `scripts/run_canonical_baselines.py` → `results/canonical_baselines_actions15_360current_v1_2026-11-19/` |
| Tests | `tests/test_canonical_task.py` |
| Supporting evidence | `results/investigation_360_taxonomy_2026-11-18/README.md`, `results/aerial_lost_k4_investigation_2026-11-19/README.md` |

## 1. Task definition

**Sequences.** For each (match, period) with period ≠ 5, the **canonical sequence** is
built as follows:
- **Records.** It contains the raw StatsBomb event records whose type is one of the
  15 classes of §2.
- **Order.** The records keep their original StatsBomb `index` order.
- **Companion records.** Records identified as companions (§3) are removed.

**Rows.** A **prediction row** is an observation \(E_t\) of the canonical sequence that
has both of these:
- **a target:** a next observation \(E_{t+1}\) in the same sequence;
- **a usable 360 snapshot of its own:** a StatsBomb three-sixty record whose `event_uuid`
  equals the event id and whose `freeze_frame` is non-empty.

**Target.**
\[
y_t = E_{t+1} \in \{1,\dots,K\},\qquad K = 15,
\]
\[
\text{models estimate}\quad P(E_{t+1}=c \mid \mathbf{x}_t),\quad c = 1,\dots,K .
\]
- \(t\) indexes positions in the canonical sequence, not elapsed time.
- \(\mathbf{x}_t\) contains only information available at the observation time of
  \(E_t\) (§5).
- In code, class ids are 0-based: \(c = \text{id} + 1\).

## 2. Taxonomy and class order (`actions15`)

| Id | Class | Id | Class | Id | Class |
|---|---|---|---|---|---|
| 0 | Pass | 5 | Shot | 10 | Interception |
| 1 | Carry | 6 | Miscontrol | 11 | Block |
| 2 | Ball Recovery | 7 | Dispossessed | 12 | Goal Keeper |
| 3 | Clearance | 8 | Pressure | 13 | Foul Committed |
| 4 | Dribble | 9 | Duel | 14 | 50/50 |

**Stability.** The order is identical to experiments 2–4. Tests check it against
`src/sequence_models/event_classes.py`, `v3_taxonomy.D15` and the configuration.

**Records outside the 15 types** are skipped, i.e. they are not observations:
- `Ball Receipt*`, `Foul Won`, `Dribbled Past`;
- substitutions and other administrative records;
- referee and stoppage records;
- `Shield` and a few rare types.

**Ball Receipt\* stays excluded.** Every receipt is linked to its pass. The receiver is
always the pass's `recipient`. The location equals the pass `end_location` for complete
passes. 86 % of passes are immediately followed by a receipt. It is a redundant
bookkeeping record of the pass outcome, and SPADL also treats it as a non-action
(investigation §1.5). This exclusion originated as an undocumented implementation choice
in July 2026 and was justified retrospectively in October–November 2026.

**Meaning of three classes under this taxonomy:**
- **Duel** contains only Aerial Lost records: the loser's record of an aerial contest.
  Tackle duels are companions (K2).
- **Goal Keeper** contains genuine keeper actions: saves, claims, punches, sweeping,
  smothers. Annotation records are companions (K1).
- **50/50** contains one record per contest (K3).

## 3. Companion records (rules K1–K3)

A companion is a second record of the same physical interaction, or an annotation without
an action of its own. Companions are removed **before** targets are built. They are
counted in `n_skipped_raw_events` like any skipped record. No class is removed.
Implementation: `src/sequence_models/v3_taxonomy.py::companion_rules`; the definitions
are at lines 41 and 53–56.

| Rule | Removed record | Evidence required |
|---|---|---|
| K1 | `Goal Keeper` with subtype `Shot Faced`, `Goal Conceded` or `Penalty Conceded` | subtype only. These are outcome annotations of the preceding shot (`Shot Faced`: 2.7 % have 360) |
| K2 | `Duel`/`Tackle` | the previous raw record is `Dispossessed`, or an incomplete `Dribble`, **and** it has the same timestamp **and** it is linked in `related_events` |
| K3 | the second `50/50` of a pair | the previous raw record is a `50/50`, **and** it has the same timestamp **and** it is linked |

**Development counts** (334 matches, periods 1–4): K1 6,441; K2 12,221; K3 659.

**Pairs with identical timestamps.** These are removed **only** under K1–K3. Timestamp
equality alone never identifies a duplicate. For example, `Ball Recovery → Carry` by the
same player (24,909 pairs in Tier B) are two separate actions: by StatsBomb convention, a
Carry starts at the timestamp of the preceding on-ball record. Both are kept, in `index`
order.

**K4 (Aerial Lost) is not active.** The rule was investigated and **not** approved
(`results/aerial_lost_k4_investigation_2026-11-19/`). All 8,549 Tier B Aerial Lost
records are linked, same-timestamp, opposite-team partners of the winner's header.
Removing them would empty the Duel class (15 → 14 classes).

## 4. Event eligibility (Protocol B)

| Rule | Value |
|---|---|
| Current event needs a usable 360 snapshot | **yes**, for training **and** evaluation rows (`row_eligible`) |
| Target event needs a 360 snapshot | **no**. In Tier B, 52,956 of 690,147 rows (7.7 %) have a target without one |
| Additional training rows for non-spatial models | **not allowed** in canonical experiments. All models in a comparison use the identical row set |
| Event order | StatsBomb `index` (verified equal to the raw file order); timestamps are never used to order events |
| Excluded periods | 5 (penalty shoot-out) |
| Sequence boundaries | targets never cross a period or a match |

**Why Protocol B.** Training on all transitions while evaluating on 360 rows (Protocol A,
experiments 2–4) mixes populations. A first-order Markov chain improves by 0.0015 log loss
when trained on the 360 rows only.

**Why the target is not filtered.** Filtering the event sequence by 360 *before* building
targets (Protocol C) changes the target for 5.7 % of rows and ties it to camera coverage.
Both points: investigation §2.

## 5. Feature availability and leakage restrictions

**Prediction time.** Prediction happens at the **observation time of the current event**.
A feature may depend only on:
- the class of \(E_t\);
- the 360 snapshot of \(E_t\);
- canonical or raw records **earlier** in the stream (e.g. \(E_{t-1}, E_{t-2}\), the time
  since the previous record).

**Forbidden as features.** `canonical_task.FORBIDDEN_RAW_FIELDS` lists them:
- **Outcome fields of the current record:** `pass.outcome`, `shot.outcome`,
  `duel.outcome`, etc. They state what happened next, e.g. an incomplete pass precedes an
  Interception or Recovery.
- **End locations and recipients:** `end_location`, `recipient`.
- **Links and durations:** `related_events` (it points to later records) and `duration`.
- **Target-derived fields:** `n_skipped_raw_events`, the time to the next event, and
  anything about \(E_{t+1}\).
- **StatsBomb `possession` ids:** they are credited before control (experiment 3).
- **Offline diagnostics:** the `*_offline` columns of the production dataset.

**Spatial interpretation.**
- **No ball data.** StatsBomb 360 has no ball coordinates and no continuous tracking.
- **The actor point.** It equals the annotated event location in 93.8 % of records. It is
  not an independent player or ball measurement.
- **Shared snapshots.** Paired records often share one identical snapshot.
- **Wording.** Features must be described accordingly, e.g. "distance to the annotated
  event location", not "distance to the ball".

**Approval of new features.** Every new feature needs:
- a feature card: availability, missingness, temporal validity, encoding;
- a truncation or leakage test before use.

## 6. Data splitting and evaluation

**Matches.**
- **Tier B:** the 304 development matches outside the 30-match pilot. They are defined by
  `results/markov_hmm_pilot/split_manifest.json`: final test 83, development 334, seed
  20261027.

**Folds.**
- **Outer folds:** the five match-grouped folds of experiment 2
  (`results/markov_hmm_cost_sensitive_v2/tiers/tier_b/scenarioD15/folds.json`; a partition
  of match ids that is independent of the taxonomy).
- **Isolation:** a match is never in training and validation of the same fold. This is
  enforced by `splits.verify_folds` and `LeakageError`.

**Hyperparameters.** They are chosen by nested leave-one-inner-fold-out over the outer
**training** folds. The outer validation fold never influences a choice. Grids are fixed
in the configuration before a run.

**Metrics.**
- **Primary:** log loss.
- **Also reported:**
  - Brier score and calibration;
  - accuracy, macro-F1 and weighted-F1, from argmax decisions and labelled as such;
  - per-class precision, recall, F1 and average precision;
  - Shot AP and Shot recall;
  - fold-by-fold values;
  - match-level (cluster) bootstrap CIs, paired between models, with per-fold signs.
- **Separate reporting:** cost-sensitive decisions are reported separately.

**Comparability.** Macro-F1 is never compared across taxonomies as if it were the same
task.

## 7. Final-test isolation

The 83 final-test matches are never loaded, inspected, trained on or evaluated:
- **Assertions.** Every loader asserts this: `splits.assert_no_final_test`, the
  development-only caches, and `canonical_task.load_development_raw`.
- **Final evaluation.** A final evaluation requires explicit researcher approval and
  frozen choices.

## 8. Versioning and reproducibility

**Version identity.** A task version is identified by:
- its name (`task_version`);
- its configuration file;
- the dataset hash.

**Rules for new versions.** A different rule set needs a **new version name and a new
configuration file**; existing ones are never edited. `load_task_spec` refuses a
configuration whose rules differ from the implementation.

**Dataset.** It is regenerable, and not committed: `data/interim/` is git-ignored.

```bash
python scripts/build_canonical_task.py          # writes data/interim/canonical_task/actions15_360current_v1/
python scripts/run_canonical_baselines.py       # writes the baseline result directory (refuses to overwrite)
python -m pytest tests/test_canonical_task.py
```

**Manifest.** `manifest.json` records:
- the task version and counts;
- `dataset_sha256_*`, the hash of the canonical table content;
- `raw_source_digest_development`, a digest over the sha256 of all 668 raw events and
  three-sixty files of the development matches.

**Result directories** record:
- configuration copies and a split manifest with source hashes;
- per-file raw source hashes;
- software versions, seeds and commands;
- counts and checksums.

**Expected Tier B counts:**
- 782,377 canonical observations;
- 781,735 transitions;
- **690,147 eligible rows** (current 360).

The real-data test checks these counts whenever the local caches exist.

**Tooling.** Reproduction needs only Python and the packages of `pyproject.toml`. No
AI-assistant tooling is required.

## 9. Historical and canonical experiments

| Experiment | Task | Status |
|---|---|---|
| 1, Markov/HMM pilot (`results/markov_hmm_pilot/`) | filtered8 and variants, Protocol A | historical |
| 2, cost-sensitive Markov/HMM (`results/markov_hmm_cost_sensitive_v2/`) | filtered8, scenarioD15, Protocol A | historical |
| 3, context-aware (`results/context_aware_event_prediction_v3/`) | scenarioD15 / actions15, Protocol A | historical |
| 4, logistic F0 (`results/logistic_regression_f0_2026-11-17/`) | scenarioD15, Protocol A | historical |
| Canonical baselines (`results/canonical_baselines_actions15_360current_v1_2026-11-19/`) | `actions15_360current_v1` | **canonical** |

**Historical experiments** stay unchanged:
- data, configurations, results and documentation are not modified;
- their numbers are quoted only with their task label.

**Interpreting differences.** Differences between historical and canonical results may
come from the changed taxonomy and training population, not from model quality.

## 10. References to the evidence

- `results/investigation_360_taxonomy_2026-11-18/README.md`:
  - 360 coverage by event type, and why snapshots are missing;
  - the actor point and shared snapshots;
  - Ball Receipt\*;
  - Protocols A/B/C;
  - paired Goal Keeper / Duel / 50/50 records;
  - scenarioD15 vs actions15 vs a 12-class alternative;
  - macro-F1 interpretation.
- `results/aerial_lost_k4_investigation_2026-11-19/README.md`: the evaluation of rule K4
  (not activated).
- `docs/thesis/02_mathematical_framework.md`: notation, Markov chains, softmax regression,
  and the equivalence of F0 logistic regression and the first-order Markov chain.
- StatsBomb Open Data 360 Frames specification v1.0.0 (`data/raw/statsbomb/doc/`), p. 3:
  360 data come from broadcast video, not every event receives a frame, and some frames
  lack an actor.
