# Repository analysis and the 15-class taxonomy (verified 2026-11-17)

Scope: the state of the repository at commit 7df070a ("Checkpoint: preserve sequence models
and thesis experiments"), with a clean working tree before this work. Every statement below
was checked against code or artifacts on 2026-11-17, unless marked as historical.

## 1. Relevant modules

| Area | Location | Role |
|---|---|---|
| Raw data access | `src/sequence_models/raw_events.py` | flat raw event table (one row per raw event, raw order); cached for development matches in `data/interim/sequence_models/event_table_development_v2.pkl` (334 matches, 1,241,255 events; no final-test match) |
| Exclusions | `configs/excluded_matches.json`, `src/match_exclusions.py` | 9 excluded matches, 417 usable |
| Production dataset (8 classes) | `src/build_statsbomb_360_dataset.py`, `scripts/build_full_dataset.py`, `data/processed/` | 417 matches, 787,331 rows; causal features + allowlist (`configs/ml_feature_classification.json`, `src/ml_features.py`). **Not** the 15-class dataset |
| Task construction | `src/sequence_models/tasks.py` (`build_task_table`) | sequences, targets, eligibility for any vocabulary; the 15-class task is built here |
| Taxonomy definitions | `configs/markov_hmm_cost_sensitive_v2.yaml` (`taxonomies.scenarioD15`), `src/sequence_models/v3_taxonomy.py` (`D15`), `results/event_taxonomy_research_2026-10-27/scenarios.json` | see §2 |
| Partitions | `results/markov_hmm_pilot/split_manifest.json` (final test 83 / development 334 / pilot 30; seed 20261027); `src/sequence_models/splits.py` | fixed hold-out; `assert_no_final_test`, `verify_folds` |
| Tier B folds | `results/markov_hmm_cost_sensitive_v2/tiers/tier_b/{scenarioD15,filtered8}/folds.json` | 5 match-grouped folds (StratifiedGroupKFold, seed 20261027) over the 304 Tier B matches; both files are identical (verified) |
| Markov models | `src/sequence_models/markov.py` (frequency baseline, 1st/2nd order, additive smoothing); `backoff_markov.py` (hierarchical Dirichlet backoff, zone contexts) | count-based baselines |
| HMMs | `src/sequence_models/hmm.py`, `categorical_hmm.py` | historical experiments 1–2 |
| Metrics | `src/sequence_models/metrics.py`, `evaluation_v2.py` (confusion-matrix metrics, calibration, match bootstrap from per-match sufficient statistics) | reused by experiment 4 |
| Cost-sensitive decisions | `src/sequence_models/cost_sensitive.py` | Bayes decision rules on unchanged probabilities |
| Experiment runners | `scripts/run_markov_hmm_pilot.py` (exp. 1), `scripts/run_markov_hmm_cost_sensitive_v2.py` (exp. 2), `scripts/run_context_aware_v3.py` (exp. 3) | each writes its own `results/…` directory, which is never overwritten |
| Existing logistic regression | `src/sequence_models/v3_models.py` (`fit_predict_lr`) | experiment 3's *rich-feature* LR (117 features). Not an event-only baseline; kept unchanged |
| Tests | `tests/` | 498 tests, all passing before this work (101 s) |

**New in this work:**
- **Code:**
  - `src/sequence_models/event_classes.py` (authoritative class mapping);
  - `src/sequence_models/feature_sets.py` (F0 and the planned F1–F6 registry);
  - `src/sequence_models/logistic.py` (multinomial LR);
  - `src/sequence_models/f0_experiment.py` (experiment 4 protocol).
- **Runner and config:** `scripts/run_logistic_f0.py`, `configs/logistic_regression_f0.yaml`.
- **Tests:** `tests/test_logistic_f0.py`.
- **Results:** `results/logistic_regression_f0_2026-11-17/`.

## 2. The 15 classes

| Id (code, 0-based) | \(c\) (text) | Class (exact spelling) |
|---|---|---|
| 0 | 1 | `Pass` |
| 1 | 2 | `Carry` |
| 2 | 3 | `Ball Recovery` |
| 3 | 4 | `Clearance` |
| 4 | 5 | `Dribble` |
| 5 | 6 | `Shot` |
| 6 | 7 | `Miscontrol` |
| 7 | 8 | `Dispossessed` |
| 8 | 9 | `Pressure` |
| 9 | 10 | `Duel` |
| 10 | 11 | `Interception` |
| 11 | 12 | `Block` |
| 12 | 13 | `Goal Keeper` |
| 13 | 14 | `Foul Committed` |
| 14 | 15 | `50/50` |

**Ordering and ids.**
- **Where ids are defined.** No earlier code defined explicit numeric ids. The id of a
  class was its position in the task vocabulary: `symbol_id` / `target_id` in
  `tasks.build_task_table`.
- **Consistency.** That position is the order above in:
  - experiment 2's config (`scenarioD15.event_types`);
  - experiment 3's `D15`;
  - every experiment-2/3 prediction file and per-class table (verified on
    `tiers/tier_b/scenarioD15/per_class_metrics.csv`).
- **Authoritative mapping.** The mapping is now explicit in
  `src/sequence_models/event_classes.py` (`SCENARIO_D15_CLASSES`, `CLASS_TO_ID`). Tests
  assert that it equals both earlier definitions
  (`tests/test_logistic_f0.py::test_taxonomy_agrees_with_every_existing_definition`).
- **Historical prefix.** The first 8 ids are the historical 8-class taxonomy (`filtered8`
  = production `MEANINGFUL_EVENT_TYPES`).

**Excluded and merged StatsBomb types.**
- **Merging.** No types are merged. Each class is exactly one raw StatsBomb type.
- **Skipped by definition** (the task is a filtered-event task):
  - derived companions: `Ball Receipt*`, `Foul Won`, `Dribbled Past`, `Own Goal For`;
  - administrative and stoppage records: `Substitution`, `Injury Stoppage`,
    `Tactical Shift`, `Referee Ball-Drop`, `Player Off`/`On`, `Starting XI`,
    `Half Start`/`End`, …;
  - discipline and annotation records: `Bad Behaviour`, `Error`, `Offside`,
    `Own Goal Against`;
  - `Shield`, with fewer than 1,000 occurrences.
- **Excluded periods.** The penalty shoot-out (period 5) is excluded entirely.
- **Measured counts.** `results/context_aware_event_prediction_v3/taxonomy/skipped_records.csv`
  and `event_mapping.csv`.

**Skipped intermediate events.**
- **Yes.** The target is the next of the 15 types, so intermediate non-vocabulary records
  are skipped.
- **Tier B evaluation rows** (`results/logistic_regression_f0_2026-11-17/leakage_checks.json`):
  38.5 % of rows skip at least one raw record, mean 0.39 per row. Most of them are
  `Ball Receipt*`.
- **Per-row count.** The number skipped is recorded as `n_skipped_raw_events`. It is
  target-derived, so it is never a feature.

**Class distribution.** Tier B, 701,144 evaluation rows, next event:

| Class | Share | Class | Share | Class | Share |
|---|---|---|---|---|---|
| Pass | 39.11 % | Duel | 2.31 % | Shot | 1.03 % |
| Carry | 32.48 % | Block | 1.49 % | Miscontrol | 1.03 % |
| Pressure | 12.16 % | Clearance | 1.44 % | Foul Committed | 0.91 % |
| Ball Recovery | 3.67 % | Goal Keeper | 1.25 % | Dispossessed | 0.90 % |
| Dribble | 1.17 % | Interception | 0.89 % | 50/50 | 0.16 % |

`Goal Keeper` is the next event of 8,793 evaluation rows, but the *current* event of only
4,254: many keeper records lack a 360 snapshot.

## 3. Inconsistencies found

| # | Finding | Impact | Action |
|---|---|---|---|
| 1 | `scenarios.json` lists the 15 types alphabetically; configs and code use the order of §2 | none so far, because no code takes ids from `scenarios.json` (experiment 2 compares *sets* only) | canonical order now explicit; test guards both |
| 2 | Class ids were implicit (vocabulary position) | risk for new code | `event_classes.py` + `assert_canonical_vocabulary` |
| 3 | Experiment 3 recommended `actions15`: the same 15 names, with StatsBomb companion records (K1–K3) skipped | under `scenarioD15`, `Dispossessed → Duel` has P = 1.0, and part of the predictability of `Duel`, `Goal Keeper` and `50/50` is bookkeeping | this work follows the instruction "scenarioD15 is the standard". **Open decision** for the supervisor; see `04_feature_engineering_roadmap.md` §5 |
| 4 | The 8-class production CSV and the 15-class task use different builders (`build_statsbomb_360_dataset.py` vs `tasks.py`) | the 15-class task has no on-disk dataset with the allowlisted 360 features; F4 will need one | to be built when F4 is implemented, with the same causal identity/orientation code |
| 5 | Experiment 3 reads Tier B folds from `filtered8/folds.json`, experiment 2 and this work from `scenarioD15/folds.json` | none: the files are identical (verified) | — |
| 6 | Historical "268 corrected flip-frames" | refers to orientation (rotation) flips in the visualization layer, not to team-identity corrections | clarified in `01_data_preparation_history.md` §5 |

**Historical numbers re-checked:**
- **Matches.** 426 → 417 usable; matches the exclusion registry.
- **Pilot.** 59,655 evaluation rows, 8-class second-order Markov log loss 0.672: both
  appear in `results/markov_hmm_pilot/README.md`.

  Those are 8-class pilot numbers and are not comparable with any 15-class result.

## 4. Leakage review of the 15-class task construction

| Check (Tier B) | Result |
|---|---|
| Target = symbol of the next observation of the same sequence | true |
| Target strictly later in the raw stream; recomputed independently from the raw event table: exactly one vocabulary record lies between \(E_t\) (exclusive) and \(E_{t+1}\) (inclusive) | true |
| Raw type of the target event equals the target class | true |
| Sequences crossing a period or match | 0 |
| Penalty-shoot-out rows | 0 |
| Final-test matches loaded | 0 |
| Train/validation overlap in any fold | 0 |

The values are in `results/logistic_regression_f0_2026-11-17/leakage_checks.json`.

**Remaining risks.**
- **Training rows.** Training uses all transitions, including the 98,165 whose current
  event has no 360 record. This is the same set as for the Markov chains and is not
  leakage.
- **Evaluation-row selection.** Selection by `has_360` of \(E_t\) uses only the current
  event.
