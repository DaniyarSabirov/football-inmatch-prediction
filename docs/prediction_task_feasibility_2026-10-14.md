# Prediction Task Feasibility — StatsBomb 360 Next-Event Dataset

Date: 2026-10-14. Prepared for discussion with the thesis supervisor before
any model training begins. No model was trained; no raw data was modified;
no existing processed CSV was overwritten.

**Evidence basis**: `scripts/analyze_prediction_task_candidates.py`, run
read-only, in-memory (never writing to `data/processed/`), across **417
non-excluded matches** — the 424 previously reported, minus **7 newly
discovered** unusable matches (see §0). Full machine-readable output:
`results/prediction_task_analysis/summary_2026-10-14_final.json`.
Manual-inspection sample: `results/prediction_task_analysis/manual_inspection_sample_2026-10-14.csv`.

---

## 0. New finding: 7 additional matches excluded today

While building the full next-event dataset in memory for this analysis
(something no prior audit had done across the *entire* dataset), 7 matches
raised `ValueError: Expected exactly two team ids in events, found [].`
from `resolve_reference_team_id`. Investigated directly:

- Both `events.json` and `three-sixty.json` are normal-sized, well-formed
  JSON for all 7 (e.g. match 3837706: 4,026 event records, 3,548
  three-sixty records — not corrupted, not sparse).
- **The two files share zero event ids in common.** Confirmed by set
  intersection on `event["id"]` vs `frame["event_uuid"]`: 0 overlap for
  all 7. The three-sixty file's `event_uuid`s do not appear in *any*
  `events.json` across the entire local dataset (scanned all 4,235
  files) — the three-sixty file does not describe this match's events at
  all.
- **This was invisible to the previous audit** (`scripts/audit_dataset.py`)
  because it only ever summed `matched_pairs` across the whole dataset; it
  never checked whether any single match's count was zero. **Fixed
  today**: the audit now reports `matches_with_zero_matched_pairs`
  explicitly, with a regression test.

All 7 (`3837706`, `3877115`, `3877170`, `3877194`, `3895158`, `3895266`,
`3895309`) were added to `configs/excluded_matches.json` under the reason
`three_sixty_event_uuid_mismatch`, following the same evidence-based
pattern as the 2 previously-excluded matches. **Total exclusions: 9.**
**Matches usable for modeling: 417** (of 426 with a three-sixty file).

---

## 1. Candidate prediction tasks compared

### A. Predict the next meaningful event type

- **Target**: `next_event_type` ∈ `{Pass, Carry, Dribble, Shot, Miscontrol, Ball Recovery, Clearance, Dispossessed}` (the existing, unmodified `MEANINGFUL_EVENT_TYPES` definition — not changed by this analysis).
- **Input features** (available before the target event, see §5): `current_event_type`, `current_team`, `actor_x`/`actor_y`, `visible_teammates`/`visible_opponents`/`visible_ambiguous`, `orientation`\*, `same_team`/`same_possession` are **not** inputs (they describe the target transition — see §5).
- **Usable samples**: 787,331 rows (417 matches, no time-gap filter) / 756,023 under the current production default (`max-time 10.0`).
- **Target distribution**: heavily imbalanced — `Pass` 48.6%, `Carry` 39.9%, `Ball Recovery` 4.6%, `Clearance` 1.8%, `Dribble` 1.5%, `Shot` 1.3%, `Miscontrol` 1.3%, `Dispossessed` 1.1%. Top two classes alone cover **88.5%** of all labels.
- **Missing-data limitations**: negligible — `actor_ambiguous=True` in 1/787,331 rows (0.0001%); every row has a defined `current_event_type`/`next_event_type` by construction.
- **Temporal leakage**: none in the core features (verified, see §5); one minor documented nuance (`orientation`/`orientation_confidence` can be informed by data up to 3s in the future for `UNCERTAIN` cases — see §5).
- **Metrics**: macro-F1 and per-class recall (accuracy is misleading given the 88.5% two-class concentration — see §2.6), confusion matrix, multiclass log-loss for calibration. **Must** report against a majority-class baseline (predict `Pass` always → 48.6% accuracy) and a **transition-frequency baseline** (predict the historically most common next type *given* `current_event_type` — e.g. predicting `Pass` after every `Carry` alone already reaches 89.5% accuracy *within that stratum*; see §2.3).
- **Implementation difficulty**: Low–Medium. Standard tabular/sequence multiclass classification; well-defined, already-materialized columns.
- **Scientific value**: Solid, standard sequence-prediction task with a large, clean sample and a natural, well-motivated baseline. Directly supports the controlled 360-vs-no-360 comparison in §3.

### B. Predict whether a shot occurs within the next 10 seconds

- **Target must be defined as**: "does a `Shot` event occur **anywhere** in the raw event stream (same period) within 10 seconds of the current event" — **not** "is the immediate next meaningful event a Shot." These give very different answers: the naive immediate-next-event version yields **1.19%** positive rate (9,347/787,331); the correct any-event-within-window version yields **6.09%** (48,033/788,087). Using the naive definition would silently discard ~5 out of 6 true positive windows and is flagged here explicitly as a definition that must **not** be used without this correction.
- **Input features**: same current-event feature set as A.
- **Usable samples**: 788,087 candidate (event, horizon) pairs (slightly more than A's 787,331 because this definition does not require a next *meaningful* event to exist — only a following `Shot` within the horizon, computed from the complete raw stream).
- **Target distribution**: **6.09% positive** — a genuine rare-event/imbalanced classification problem.
- **Missing-data limitations**: same as A for the input side; label construction itself requires scanning the raw (not `MEANINGFUL_EVENT_TYPES`-filtered) event stream, implemented and verified in `scripts/analyze_prediction_task_candidates.py::build_shot_within_horizon_labels`.
- **Temporal leakage risk**: higher than A by construction (the label is deliberately drawn from a future window) — the critical discipline is that **no input feature may be computed from events inside that same future window**. Verified: all current-event features are derived strictly from the current event's own 360 snapshot, never from anything after it.
- **Metrics**: PR-AUC / average precision and ROC-AUC (not accuracy — a trivial "never shot" classifier already scores 93.9% accuracy and is useless), recall at a fixed precision, F1 at a chosen operating threshold. Baseline: global base rate (6.09%) and a per-`current_event_type` base rate (e.g. shots are far more likely to follow a `Dribble` or `Carry` in the attacking third than a `Clearance`).
- **Implementation difficulty**: Medium — binary classification is simpler to model than A, but correct label construction (the any-event-in-window scan) is an easy place to introduce a silent bug, as the naive version above demonstrates.
- **Scientific value**: High applied/football relevance ("is this a dangerous moment"), and a good vehicle for demonstrating imbalanced-classification methodology rigor in the thesis. Recommended as a **secondary** task.

### C. Predict a future spatial state

**Not assumed feasible merely because freeze-frame coordinates exist — the limitation is structural**: StatsBomb 360 provides a **single discrete snapshot per covered event**, not continuous player tracking. There is no trajectory between two consecutive events, no velocity, and only the subset of players inside the camera's field of view at that instant (historically measured at ~85% of 22 players, per the first-round data-quality audit). "Future spatial state" can therefore only be defined at the granularity of the **next discrete 360-covered event**, not as a continuous forecast.

- **Feasible target constructions** (two realistic options, not a single obvious one):
  1. **Regression**: next meaningful event's actor `(x, y)` — requires the *next* event to also have 360 coverage.
  2. **Coarser classification**: which pitch zone (e.g. a 6×4 grid, or thirds × wide/central) the next actor will be in — less sensitive to the orientation/out-of-bounds edge cases already documented in the data-quality audits, more forgiving to evaluate.
- **Input features**: current `actor_x`/`actor_y`, `visible_teammates`/`visible_opponents` (team shape), `current_event_type`.
- **Usable samples**: bounded by **both** current-event 360 coverage (required to exist as a row at all) **and** next-event 360 coverage. Measured: **next_has_360 = 92.1%** of the 787,331 rows (723,031 rows) — still a large, usable sample, but a real, measured reduction, not assumed.
- **Missing-data limitations**: the same 1-row `actor_ambiguous` case as A; additionally, "visible" 360 data is a partial pitch observation by design (not all 22 players), which weakens any spatial feature (current or target) as a complete description of match state — a structural limitation of the source data, not fixable by preprocessing.
- **Temporal leakage**: regression/classification target is correctly drawn from the future event; input features are current-event-only (verified). One **sample-selection caveat** (not a leakage bug): `next_has_360` availability may itself correlate with `current_event_type` or match context, which could bias *which* rows are usable without leaking information into any individual row's features — worth checking empirically before relying on this subsample.
- **Metrics**: for regression — mean/median Euclidean error (pitch units) against a "no-movement" baseline (predict next location = current location) and a per-type mean-displacement baseline; for zone classification — macro-F1 against a majority-zone baseline.
- **Implementation difficulty**: Medium–High — more design decisions (coordinate vs. zone target, handling the ~8% next-event-without-360 rows), harder to evaluate and explain in football terms than A or B.
- **Scientific value**: High novelty (this is what makes 360 data distinctive versus plain event-stream data), but higher implementation/evaluation risk as a **standalone primary deliverable** within a Bachelor's thesis timeline. Better used as the 360-feature *half* of the controlled comparison in §3 than attempted as a full separate primary task.

---

## 2. Current next-event dataset — measured statistics

All figures below are computed **without** the production `--max-time`
filter (i.e. the true distribution), with the filtered subset reported
separately where relevant. Row-construction logic
(`build_rows_from_match_data`) was reused unmodified — no second
implementation.

### 2.1 Feature vs. target columns (current schema, verified against code)

| Category | Columns |
|---|---|
| Identifiers / metadata | `match_id`, `current_event_id`, `current_event_index`, `period`, `minute`, `timestamp`, `next_event_id` |
| Input features | `current_event_type`, `current_team`, `current_player`, `actor_x`, `actor_y`, `actor_ambiguous`, `orientation`\*, `orientation_confidence`\*, `team_normalization`, `visible_players`, `visible_teammates`, `visible_opponents`, `visible_ambiguous` |
| Prediction targets | `next_event_type`, `next_team`, `next_player`, `same_team`, `same_possession`, `time_to_next_event_seconds` |
| Evaluation-only | `next_event_id` (join key back to the full raw event for qualitative error review — never a model input) |

\* see §5 for the one documented, low-risk leakage nuance affecting these two columns.

**Note**: the on-disk CSVs in `data/processed/` predate the addition of
the `actor_ambiguous` column (confirmed by direct inspection — they were
generated before that fix) and the 7-match exclusion discovered today.
They were **not** regenerated as part of this analysis (per instructions);
all statistics in this report come from an in-memory rebuild using the
current code.

### 2.2 Next-event-type distribution

| Type | Count | % |
|---|---|---|
| Pass | 382,948 | 48.64% |
| Carry | 314,169 | 39.90% |
| Ball Recovery | 36,321 | 4.61% |
| Clearance | 13,829 | 1.76% |
| Dribble | 11,608 | 1.47% |
| Shot | 9,905 | 1.26% |
| Miscontrol | 9,829 | 1.25% |
| Dispossessed | 8,722 | 1.11% |

### 2.3 Transitions between consecutive meaningful event types (top entries)

| From → To | Count | % of "From" |
|---|---|---|
| Carry → Pass | 288,988 | **89.49%** |
| Pass → Carry | 273,093 | 73.18% |
| Pass → Pass | 66,156 | 17.73% |
| Ball Recovery → Carry | 30,899 | 84.82% |
| Clearance → Pass | 7,655 | 51.07% |
| Shot → Pass | 7,616 | 73.51% |
| Dribble → Carry | 7,298 | 65.25% |
| Clearance → Ball Recovery | 5,967 | 39.81% |

**This is why a majority-class baseline is insufficient and a
transition-frequency baseline is required**: simply knowing
`current_event_type` already predicts `next_event_type` correctly 89.5%
of the time for `Carry` and 73–85% for several other types. Any model
claiming skill on Candidate A must beat this conditional baseline, not
just the unconditional 48.6% majority-class rate.

### 2.4 Time-to-next-event distribution (seconds, unfiltered)

| Statistic | Value |
|---|---|
| n | 787,331 |
| mean | 2.785 |
| median | 1.336 |
| p10 | 0.281 |
| p25 | 0.861 |
| p75 | 2.099 |
| p90 | 3.698 |
| p99 | 36.855 |
| max | 503.411 |

### 2.5 Sample counts at temporal horizons

| Horizon (s) | Rows with gap ≤ horizon | % |
|---|---|---|
| 2 | 574,308 | 72.94% |
| 5 | 735,481 | 93.41% |
| 10 | 756,023 | 96.02% |
| 15 | 762,324 | 96.82% |
| 30 | 775,753 | 98.53% |
| 60 | 784,242 | 99.61% |

The current production default (`--max-time 10.0`) retains 96.0% of all
rows — a reasonable default, losing mainly the long tail (stoppages,
VAR checks, etc.), not a systematic subset.

### 2.6 360 availability, actor ambiguity

| Metric | Value |
|---|---|
| Rows with `actor_ambiguous=True` | 1 / 787,331 (0.0001%) |
| Rows where the **next** event also has 360 coverage | 723,031 / 787,331 (**91.83%**) |

### 2.7 Candidate B, both label definitions (full dataset)

| Definition | Positive | Total | Rate |
|---|---|---|---|
| Naive: immediate next meaningful event is a Shot, within 10s | 9,347 | 787,331 | 1.19% |
| Correct: any Shot anywhere in the stream within 10s | 48,033 | 788,087 | 6.09% |

---

## 3. Value of 360 data — proposed controlled comparison (not run)

Three feature sets, same target (Candidate A, next-event-type), same
train/evaluation rows:

1. **Event-history only**: `current_event_type`, one-hot or embedded;
   optionally a short window of prior event types (sequence features).
2. **Event-history + match context**: add `period`, `minute`,
   `same_team`/`same_possession` *of the previous transition* (not the
   current target — careful not to leak, see §5), score difference if
   available from a separate source.
3. **Event-history + 360 spatial features**: add `actor_x`/`actor_y`,
   `visible_teammates`, `visible_opponents`.

**Keeping samples comparable**: restrict ALL THREE feature sets to the
**same row population** — specifically, the 723,031 rows with 360
coverage on both the current *and* candidate evaluation needs (for A,
only current-event 360 is actually required, so the comparable population
is the full 787,331; for any 360-dependent target such as C, restrict to
the 723,031-row subset for all three feature sets so a performance
difference cannot be attributed to a different, easier sample). **Do not**
let feature-set (3) use rows that feature-sets (1)/(2) cannot — train and
evaluate all three on an identical row set and an identical train/test
split.

**Minimum experiment** (not run as part of this analysis): a single
stratified train/test split (e.g. by match, to avoid intra-match
leakage — see §5), Candidate A target, macro-F1 and per-class recall for
each of the three feature sets, reported against the transition-frequency
baseline (§2.3). A statistically meaningful difference between feature
sets (2) and (3) is the minimum evidence needed to claim 360 data adds
predictive value; (1) vs (2)/(3) establishes whether *any* additional
context helps at all.

---

## 4. Manual-inspection sample

**File**: `results/prediction_task_analysis/manual_inspection_sample_2026-10-14.csv`
(28 rows, all columns from the current schema plus `next_has_360`).

**Selection procedure** (reproducible, seeded, not hand-picked):
implemented in `select_manual_inspection_sample()`
(`scripts/analyze_prediction_task_candidates.py`). Rows are stratified by
`(current_event_type, time-gap bucket [short ≤2s / medium 2–10s / long
>10s], next_has_360, same_team)`, strata are shuffled with a fixed seed
(`--seed 42`), and one row is taken from each stratum in round-robin order
until 28 are collected — this guarantees coverage of short/long intervals,
attacks (`Carry`/`Dribble`→`Shot`-adjacent rows), possession changes
(`same_team=False`), and missing-360-coverage cases, rather than only the
most visually interesting examples.

**To inspect any row visually**: open the CSV and run, for the row's
`match_id`/`current_event_id`:

```bash
.venv/bin/python scripts/inspect_data_quality.py --match <match_id> --event <current_event_id>
```

This reuses the existing, already-tested inspection tool (built in the
prior data-quality session) — it prints the full event/360 report and
saves a side-by-side raw-vs-corrected pitch PNG to
`results/data_quality_inspections/`.

Reproduce the exact sample: `.venv/bin/python scripts/analyze_prediction_task_candidates.py --inspection-csv <path> --inspection-n 28 --seed 42` (full run, ~5 minutes across 417 matches).

---

## 5. Temporal leakage check

**Columns inspected directly in `src/build_statsbomb_360_dataset.py`**
(`build_rows_from_match_data`, lines computing each row's fields):

- `actor_x`, `actor_y`, `actor_ambiguous`, `visible_teammates`,
  `visible_opponents`, `visible_ambiguous`: all derived exclusively from
  `current_event`'s own (already orientation/team-normalization-corrected)
  `freeze_frame` — confirmed by code reading; never reads anything from
  `next_event`. Independently re-confirmed by the existing regression test
  `test_row_never_contains_next_event_spatial_fields`.
- `next_event_type`, `next_team`, `next_player`, `same_team`,
  `same_possession`, `time_to_next_event_seconds`: by definition describe
  the target transition — **must never be used as model input features**,
  only as labels or evaluation-only context.

**One documented, low-risk nuance** (not previously flagged in this exact
form): `orientation`/`orientation_confidence` for an `UNCERTAIN`-classified
current event can be resolved using a **temporal-neighbor** signal
(`resolve_uncertain_with_neighbors`, `src/orientation_correction.py`),
which looks up to `NEIGHBOR_ANCHOR_WINDOW=5` usable frames and
`NEIGHBOR_ANCHOR_MAX_DT=3.0` seconds in **either direction**, including
forward in time. This means a small number of `orientation_confidence`
values are technically informed by data a few seconds in the future. This
is a data-quality correction signal (which way a camera snapshot is
rotated), not football-semantic information, so it cannot meaningfully
leak "what happens next" — but a maximally conservative modeling choice
would exclude `orientation`/`orientation_confidence` from the input
feature set entirely (they are diagnostic columns, not intended as
predictive signal in the first place).

**Recommended column separation** (confirms §2.1's table is the
correct, final categorization — no redesign needed, since no concrete
leakage problem was found beyond the documented nuance above):

- **Identifiers/metadata**: never used as features or targets.
- **Input features**: current-event-derived only; `orientation`/`orientation_confidence` optional/excludable per the nuance above.
- **Targets**: `next_event_type` (A), the derived shot-within-horizon label (B), next-event actor position/zone (C).
- **Evaluation-only**: `next_event_id` for qualitative review.

**Cross-match leakage**: not checked in StatsBomb data terms (no player/
team appears to leak across matches via these columns), but **must** be
checked at the train/test-split level — splitting by individual *rows*
rather than by *match* would let a model see other events from the same
match (and implicitly the same players/teams/game state) in training while
being evaluated on a held-out row from that same match, which is a
well-known leakage pattern in event-sequence sports data. **Recommendation**:
split by `match_id`, not by row.

---

## 6. Decision

**Recommended primary task: Candidate A — predict the next meaningful
event type.** Largest clean sample (787,331 rows, 417 matches),
near-zero missing-data cost (1 ambiguous row), no confirmed leakage,
directly supports the 360-vs-no-360 controlled comparison needed to
answer the thesis's likely core question ("does 360 spatial data help"),
and has a natural, rigorous baseline already measured in this report
(§2.3's transition-frequency baseline).

**Recommended baseline**: per-`current_event_type` transition-frequency
baseline (§2.3), not the unconditional majority-class baseline — any
model must be shown to beat ~89.5%/73.2%/etc. conditional rates, not
just 48.6% overall.

**Minimum experiment**: the 3-feature-set comparison in §3, Candidate A
target, single match-level train/test split, macro-F1 + per-class recall,
on the identical row population for all three feature sets.

**Recommended secondary task**: Candidate B (shot within 10s, correct
"anywhere in stream" definition), to demonstrate imbalanced-classification
methodology.

**Candidate C**: recommended as a feature-set addition (§3's feature set
3) rather than a standalone primary deliverable, given the measured
92.1%-coverage sample reduction and the structural single-snapshot
limitation documented in §1.C.

### Unresolved methodological questions

1. Should `orientation`/`orientation_confidence` be excluded from model
   features entirely, given the minor forward-looking neighbor-resolution
   nuance in §5, even though it carries no football-semantic information?
2. For Candidate C, is coordinate regression or zone classification the
   more defensible target for a Bachelor's thesis, given the evaluation
   complexity difference noted in §1.C?
3. Should the existing `data/processed/` CSVs be regenerated now (to pick
   up `actor_ambiguous` and the 7 new exclusions) before any modeling
   work begins, or should modeling code simply rebuild from raw data
   directly going forward?

### Next implementation steps (checklist)

- [ ] Decide on Candidate A as primary vs. a combined A+B scope with the supervisor.
- [ ] Implement a match-level train/test split (not row-level).
- [ ] Implement the 3-feature-set minimum experiment (§3) and report macro-F1 against the transition-frequency baseline.
- [ ] Decide the `orientation`/`orientation_confidence` feature-inclusion question (§D.1 above) before finalizing the feature set.
- [ ] If Candidate B is pursued, implement label construction via the "anywhere in stream" definition (§1.B), not the naive immediate-next-event definition.
- [ ] Revisit whether to regenerate `data/processed/` CSVs with the current schema and the 9-match exclusion list before training.
