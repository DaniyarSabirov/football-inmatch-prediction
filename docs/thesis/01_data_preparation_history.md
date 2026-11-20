# Data preparation history (for the thesis "Data" chapter)

Status: 2026-11-17. A consolidated, chronological account of the data work done between
July and October 2026. Every number is taken from a report or artifact in this repository,
and the source is given.

**Evidence labels:**
- **[V]** re-verified on 2026-11-17 against the artifact named;
- **[R]** taken from the cited report, not recomputed today;
- **[L]** a known limitation.

The detailed reports remain the primary sources. This document does not replace them.

## 1. Data source and its limits

- **Source.** StatsBomb Open Data: event JSON files, StatsBomb 360 freeze-frame JSON files
  and match metadata, read from the local checkout `data/raw/statsbomb/data/` (immutable,
  not version-controlled).
- **What 360 provides.** One snapshot per event, with the positions of the *visible*
  players, plus flags: `teammate` (relative to the event's actor), `actor`, `keeper`.
  Each snapshot also has a `visible_area` polygon, the part of the pitch seen by the
  broadcast camera.
- **[L] What 360 does not provide.**
  - **No continuous tracking.** There is no continuous tracking of players or of the
    ball. Positions exist only at event-associated snapshots.
  - **Unobserved, not absent.** Players outside the visible area are unobserved, not
    absent.
  - **Ball position.** The ball is known only through the event's own location fields,
    i.e. its start and, for some types, end location.
- **Event timing.** Event timestamps are period-relative. Event order is the StatsBomb
  `index`, verified equal to the file's array order (`src/sequence_models/tasks.py`).
- **[R] Clock vs. order.** Timestamps tie for about 31 % of adjacent raw pairs and run
  backwards for 43 events. The event order, not the clock, is therefore used to define
  sequences.

## 2. Match selection and exclusions

| Step | Matches | Source |
|---|---|---|
| Matches with a 360 file in the checkout | 426 | `docs/final_statsbomb_360_orientation_research.md` [R] |
| Excluded: corrupted 360 JSON (one 16,384-byte block of NUL bytes), match 3845506 | −1 | `configs/excluded_matches.json` [V] |
| Excluded: 360 coverage too small to resolve the team reference, match 3923880 | −1 | same [V] |
| Excluded: 360 records whose event UUIDs do not match the event file (3837706, 3877115, 3877170, 3877194, 3895158, 3895266, 3895309) | −7 | same [V]; forensic analysis in `docs/excluded_matches_forensic_analysis_2026-10-18.md` |
| **Usable matches** | **417** | `data/processed/statsbomb_360_dataset_manifest.json` [V] |

- **Exclusion registry.** Every exclusion is recorded with its reason, evidence, date and
  report in a single registry (`configs/excluded_matches.json`, read only through
  `src/match_exclusions.py`).
- **Consumers.** The registry is used by the dataset builder, the audits and the modelling
  code.

**Modelling partition** [V] (`results/markov_hmm_pilot/split_manifest.json`, seed 20261027):
- **Final test.** 83 matches, drawn per (competition, season) stratum, never read by any
  experiment.
- **Development.** 334 matches, split into:
  - **Tier A:** the 30-match pilot, used for exploration and tuning;
  - **Tier B:** the other 304 matches, used for held-out development evaluation.

## 3. Exploration and event distributions

- **Category audit.** A complete audit of StatsBomb event types, subtypes and
  frequencies is in `docs/statsbomb_complete_category_audit.md` (2026-10-20).
- **Taxonomy research.** The research behind the event taxonomy is in
  `results/event_taxonomy_research_2026-10-27/README.md`. It compares four scenarios: the
  8 production types, all raw types, all football actions, and Scenario D (15 types).
- **Scenario D** [R]:
  - coverage: 71.2 % of raw events;
  - majority class: Pass (39.1 %);
  - smallest class: 50/50 (1,504 rows);
  - imbalance ratio: 252;
  - target entropy: 2.39 bits, of which the current event type explains 30.8 %.
- **Tier B class distribution of the 15-class task** [V]:
  `results/logistic_regression_f0_2026-11-17/class_distribution.csv`.
- **[R] Companion records.** Experiment 3 measured how often StatsBomb records one contest
  twice (`results/context_aware_event_prediction_v3/README.md`, §3):
  - Dispossessed → Duel (tackle) at the same timestamp in 100 % of cases;
  - 63 % of Goal Keeper records are outcome annotations of the preceding shot;
  - 50/50 pairs.

  These inflate the predictability of some Scenario D classes. This matters for the
  interpretation of per-class results (see `00_repository_and_taxonomy_audit.md`).

## 4. Freeze-frame coverage and timestamp alignment

- **Joining frames to events.** 360 records are joined to events by event UUID. A
  current event without a 360 record is not an evaluation row.
- **[R] Coverage.** Of 893,476 events of the 8 production types, 105,389 have no 360
  record (`results/next_event_time_window_audit_2026-10-27/README.md`).
- **[R] Snapshot intervals.** Time intervals between consecutive snapshots were analysed
  per match (`scripts/analyze_360_snapshot_intervals.py`; figures
  `results/statsbomb_360_interval_*.png`).
- **[R] Clock audit.** A full audit of the visualization clock and orientation is in
  `docs/full_visualization_orientation_and_clock_audit_2026-10-22.md`. The renderer shows
  only real joined snapshots: it never interpolates and never creates frames for
  artificial times (`README.md`, visualization section).
- **[R] Removed time window.** The original builder dropped rows whose next event came
  more than 10 s later. This rule had no documented rationale and removed 3.97 % of
  eligible rows, mostly restarts. It was removed on 2026-10-27
  (`results/next_event_time_window_audit_2026-10-27/README.md`; commit ae7b6ad).

## 5. Spatial normalisation and orientation correction

- **Problem.** 360 coordinates of a snapshot are given from the perspective of the
  event's team, so consecutive snapshots of different teams appear mirrored.
  Additionally, some snapshots are internally inconsistent.
- **Corrections.**
  - **Period-specific team normalisation** (`src/team_normalization.py`). The reference
    is the first 360 event in event order, i.e. the kick-off team.
  - **Snapshot orientation correction** (`src/orientation_correction.py`,
    `src/orientation_resolution.py`).
  - **Physical broadcast orientation.** It is inferred from the camera side of the
    `visible_area` polygon (`src/broadcast_orientation.py`), for display.
- **Investigations.**
  - Spain–Croatia physical validation: `docs/physical_pitch_orientation_validation.md`.
  - Russia–Belgium regression: `docs/russia_belgium_orientation_regression_2026-10-22.md`
    and `docs/belgium_russia_broadcast_orientation_forensic_audit.md`.
  - Feasibility for all matches: `docs/all_matches_physical_orientation_feasibility.md`.
  - Final report: `docs/final_statsbomb_360_orientation_research.md`.
- **[R] Final display-layer state** (`docs/visualization_outstanding_issues_2026-10-23.md`):
  - periods examined: 908;
  - usable frames: 1,357,626;
  - **confirmed incorrectly rotated frames: 268 → 0**;
  - copies of one snapshot drawn differently: 0 of 396,822;
  - externally verified matches: 2 (3788743, 3930160);
  - user-reviewed periods: 1.
- **Clarification of a historical number.** The figure "268 corrected flip-frames" refers
  to these **orientation** (rotation) flips in the visualization layer. It is not the
  number of team-identity corrections; that number is in §6.
- **[L] Verification status.** Orientation is *inferred* for 902 of 908 periods. Only 2
  matches are independently verified against video.
- **[R] Spatial checks used in modelling** (experiment 3,
  `results/context_aware_event_prediction_v3/taxonomy/orientation_audit.json`):
  - 99.85 % of 8,472 shots lie at x ≥ 60 in every period;
  - 99.3 % of keeper actions lie at x ≤ 30;
  - 100 % of kick-offs lie at 55 ≤ x ≤ 65.

  These confirm that StatsBomb *event* coordinates are expressed in the acting team's
  attacking direction.

## 6. Team identity problems

- **Problem.** Players appeared to change team between neighbouring snapshots, because of
  inconsistent `teammate` flags, mirrored copies and actor ambiguity.
- **Correction.** Evidence-based team identity (`src/team_identity.py`, commit 3423816),
  built from keeper goal side, view consistency, copy groups and simultaneous-snapshot
  pairing. It has a strictly **causal** mode for modelling and an offline mode for
  display.
- **[R] Final validation** (`results/team_identity_final_validation_2026-10-27/README.md`):
  - player pairs evaluated: 7,140,398;
  - colour transitions: 95,579 (1.339 %);
  - transitions confirmed incorrect by an independent anchor: **0**;
  - strongly suspected: 264;
  - the remainder are attributed mostly to player-pairing artefacts. The evidence: the
    rate rises about 10,000× with time gap and pairing ambiguity, and keepers transition
    at 0.145 % vs 1.373 % for other players.
- **[R] Points changed by the causal identity**
  (`results/team_identity_final_validation_2026-10-27/final_identity_counts.json`):
  - by a corrected view: 16,803;
  - by a corrected goalkeeper: 1,159;
  - left unresolved: 82,043 points.
- **[L] Residual error estimate.** About 2,000–4,000 genuine label disagreements may
  remain. This is an estimate under a stated assumption.

## 7. Data cleaning, validation and leakage investigations

| Investigation | Result | Source |
|---|---|---|
| Integrity and visualization audit; bug/risk report | issues ranked and fixed | `docs/data_integrity_audit_2026-10-13.md`, `docs/bug_and_risk_report_2026-10-13.md`, `docs/data_quality_*_2026-10-13.md` [R] |
| Prediction-task feasibility | 8-class next-event task chosen for the first dataset | `docs/prediction_task_feasibility_2026-10-14.md` [R] |
| Causal features | three look-ahead paths fixed (orientation from later copies, offline identity feeding causal features, timestamp ordering); feature allowlist of 11 causal columns | `results/ml_data_integrity_final_2026-10-27/README.md`, `configs/ml_feature_classification.json`, `src/ml_features.py` [R] |
| Truncation test | cut the stream after an event; features must not change: 155 cuts, 146,364 rows, **0 mismatches** | same [R]; `tests/test_ml_feature_safety.py` |
| Deterministic rebuild | the production dataset (417 matches, 787,331 rows) rebuilds bit-identically | `data/processed/statsbomb_360_dataset_manifest.json` [V], `results/ml_data_integrity_final_2026-10-27/dataset_sha256_build_*.txt` |
| Possession ids | StatsBomb `possession` ids are credited before the team controls the ball in 7.4 % of possessions (look-ahead), so they are not used | experiment 3 README [R] |
| Target alignment (15-class task) | target = next vocabulary event of the same period in raw order; independent recomputation from the raw stream | `results/logistic_regression_f0_2026-11-17/leakage_checks.json` [V] |

## 8. Visualization and its improvements

The renderer (`src/animate_statsbomb_360_match.py`) went through several stages:
1. **First version.** Raw snapshots for one match.
2. **Generalisation.** Extended to all available matches (commit 66cb015).
3. **Orientation overlays.** Overlays of the visible area and the camera.
4. **Corrections.** Team normalisation and evidence-based team colours (3423816).
5. **Final camera orientation.** Camera-side broadcast orientation with per-period
   reviews (`docs/final_statsbomb_360_orientation_research.md`).

Before/after GIFs of the main defects are in `results/` (e.g. `match_3857283_*_BEFORE_*`
vs `*_fixed*`, and the Spain–Croatia series). Random visual-check rounds:
`scripts/generate_random_visual_checks.py`. Review queue:
`results/physical_orientation/final_orientation_audit/review/`.

**[L] Purpose.** The visualization was the main *diagnostic* instrument of the data
work. Most defects above were first seen in renders, then quantified dataset-wide by
scripts. Rendering correctness is not proof of data correctness. Every claim above rests
on the dataset-wide audits, not on individual renders.

## 9. From the production dataset to the modelling tasks

- **Production dataset.** The 8-class production dataset (`data/processed/`) is preserved
  unchanged for reproducibility.
- **15-class task.** The modelling experiments do not read the production CSV. The task is
  rebuilt from the raw event table of the development matches:
  - event cache: `data/interim/sequence_models/event_table_development_v2.pkl`, which
    contains only the 334 development matches;
  - task builder: `src/sequence_models/tasks.py::build_task_table`, which can build both
    the 8-class and the 15-class task.
- **Production consistency check.** For the 8 classes, experiment 1 verified that its task
  rows equal the production rows: 59,655 pilot rows, 0 mismatches
  (`results/markov_hmm_pilot/README.md`).
