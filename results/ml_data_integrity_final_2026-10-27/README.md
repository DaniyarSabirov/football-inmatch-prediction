# Final ML data integrity audit

Date: 2026-10-27. **Verdict: PASS WITH LIMITATIONS.**

All features approved for modelling are now strictly causal. They are computed only from the current event, its
own 360 snapshot and records earlier in the event stream, and this is verified end to end. Three
look-ahead paths and one robustness defect were found and fixed. No model was trained.

## 1. The 55 orientation rows (`orientation_55_rows.csv`, `orientation_55_summary.json`)

**Where:** 55 rows in 46 matches, flagged `orientation_uses_later_records` in the build at 3423816. Event
types: Carry 42, Pass 8, Ball Recovery 2, Clearance 1, Dispossessed 1, Miscontrol 1.

**Why the resolver needed later records:** all 55 snapshots are UNCERTAIN on their own geometry.

| Route used | Rows | What it read |
|---|---|---|
| Duplicate snapshot | 45 | another copy of the same raw snapshot; in 24 rows a copy LATER in the stream |
| Reviewer decision | 9 | a human decision made while viewing the sequence (8 kick-offs) |
| Problem A neighbour vote | 1 | a neighbour that may lie after the record |

**What they influenced:**
- the snapshot orientation, hence `actor_x`/`actor_y` after the frame correction;
- through the shared geometry, the causal team identity: keeper goal side, continuity and simultaneous pairing
  all use orientation.

A causal flag alone would not have removed this.

**Causal result now:**
- **25** rows resolved from earlier information only, all agreeing with the offline value: 23 from an earlier
  copy, 1 by the kick-off rule, 1 by an earlier neighbour.
- **30** rows cannot be decided causally. Their `actor_x`/`actor_y` are now empty (`orientation_available =
  false`) and the offline values are kept in `actor_x_offline`/`actor_y_offline`.
- **0** contradictions with the offline value.

## 2. Leakage risks and safeguards

| Risk | Status |
|---|---|
| Orientation from later copies, two-sided neighbour votes, period camera side, reviewer decisions | **Fixed.** New `resolve_causal_orientations`: geometry, earlier-only neighbour vote, kick-off own-half rule (record only), earlier identical copy (the two own frames are mirror images exactly when teams differ). Otherwise UNCERTAIN and actor position unavailable. 41 rows dataset-wide are unavailable. |
| Causal identity fed with offline orientations | **Fixed.** It now receives the causal orientations and causal Problem A results. |
| Causal mode ordered by timestamp | **Fixed.** Availability follows event order (period, index); 33 adjacent event pairs in 31 matches have a later index but an earlier timestamp. |
| Normalization reference team taken from the first event by timestamp | **Fixed.** Now the first 360 event in event order (the kick-off team), the rule as documented. In 9 matches a late-indexed event with an early timestamp had set it, a one-bit look-ahead. Effect: those matches' coordinates are exactly mirrored (16,378 rows, verified); all other matches are identical. |
| Builder crashed on a partial match (one team's records) | **Fixed.** The offline diagnostics fall back to Problem A alone; the causal path is unaffected. |
| Offline values used as features | **Prevented** by an explicit allowlist (§3) with validation. |

Remaining by design:
- Row selection uses the target definition: a next meaningful event within 10 s.
- The `*_offline` diagnostics read later records and are excluded from training.

### Validation of causality

- `tests/test_ml_feature_safety.py` cuts the input right after an event. It keeps the event's label as bare
  event data, without its 360 record and without anything in between. Every column that is not TARGET or
  OFFLINE_DIAGNOSTIC must be unchanged for that row and all earlier rows. The test covers 5 real rows,
  including frame 505, a later-copy row, an earlier-copy row and the neighbour-vote row.
- `truncation_check.json` runs the same check on **all 55 rows plus 100 random rows**:
  - 155 cuts in 126 matches;
  - 146,364 rows and 3,659,100 cells compared;
  - **0 mismatches**.

## 3. Feature classification (`configs/ml_feature_classification.json`, `src/ml_features.py`)

All 42 columns are classified from their lineage in the builder, not from their names.

| Category | Columns |
|---|---|
| CAUSAL_FEATURE (15; 11 approved) | `current_event_type`, `period`, `minute`, `actor_x`, `actor_y`, `actor_ambiguous`, `orientation_available`, `visible_players`, `visible_teammates`, `visible_opponents`, `visible_ambiguous`; not approved: `identity_quality`, `identity_corrected_points` (quality flags for sensitivity analyses), `visible_teammates_raw`, `visible_opponents_raw` (uncorrected alternative) |
| TARGET (7) | `next_event_type` (label), `next_event_id`, `next_team`, `next_player`, `same_team`, `same_possession`, `time_to_next_event_seconds` |
| OFFLINE_DIAGNOSTIC (10) | `actor_x_offline`, `actor_y_offline`, `orientation_offline`, `orientation_offline_resolved_by`, `orientation_problem_a_offline`, `orientation_uses_later_records`, `visible_teammates_offline`, `visible_opponents_offline`, `visible_ambiguous_offline`, `identity_quality_offline` |
| METADATA (8) | `match_id` (grouped splits), `current_event_id`, `current_event_index`, `current_team`, `current_player`, `timestamp`, `team_normalization`, `orientation_resolved_by` |
| EXCLUDED (2) | `orientation`, `orientation_confidence` (causal, but processing artefacts) |

Training code must call `select_feature_matrix(df)`:
- it validates that the schema matches the classification exactly;
- it returns only the 11 approved columns.

`validate_feature_columns` rejects any target, offline, metadata, excluded or unknown column, and a
classification that approves a non-causal column fails to load. Tests:
`tests/test_ml_feature_safety.py` (schema coverage, exact allowlist, 10 unsafe columns rejected, unknown
columns rejected).

## 4. Law 8 kick-off diagnostic (diagnostic only; `scripts/diagnose_kickoff_law8.py`, `law8/`)

The diagnostic is independent of the builder and renderer (a test enforces this) and reads raw flags only. It
never recolours, reassigns, edits or excludes anything.

How it judges a kick-off:
- **Orientation:** only kick-offs whose frame is known independently of the flags count. The kick-off own-half
  rule is circular, because it uses flags plus Law 8.
- **Halfway line:** players within 2 yd of it and the actor are not judged.
- **Minor:** wrong-half players within 5 yd of the line are plausible movement, since the snapshot may be
  captured just after the kick.
- **Suspicious:** at least 3 players more than 5 yd into the wrong half.

| Verdict | Kick-offs (of 987 with a snapshot) |
|---|---|
| consistent | 455 |
| minor | 48 |
| **suspicious** | **1** |
| inconclusive (orientation not independently known) | 478 |
| inconclusive (fewer than 6 judged players) | 5 |

504 frames could be judged; 71 player observations were in the wrong half in those frames, 12 of them in the
suspicious frame.

**Suspicious:** match 3788749, period 2, 00:00:00.201, Poland kick-off. 12 of 12 judged players are in the
wrong half, 11 by more than 5 yd (up to 57 yd). A complete inversion fits a mirrored frame or a fully inverted
flag view equally well; the diagnostic does not decide which.

**Frame 505 (3930164):** inconclusive. Its orientation comes from a reviewer decision, not from flag-independent
evidence.

**Limitations:**
- Snapshot capture lag is unknown.
- Kick-offs after goals have no known whistle time.
- 478 kick-offs cannot be judged independently.

## 5. Git artifact classification

- **Version-controlled:**
  - source code, tests, configs, docs, scripts;
  - the reports in `results/**` (`.md`);
  - small tables and JSON, including the 1.1–1.5 MB audit tables that scripts read;
  - review pages (`.html`, `.txt`);
  - 230 evidence PNG panels (13.2 MB in total, each ≤ 1.2 MB) cited by reports and review pages.
- **Kept local, with specific `.gitignore` rules:**
  - regenerable renders: `results/**/*.gif` (73 files, 169 MB) and `results/**/*_test_frame.png` (62 files,
    7.4 MB);
  - per-row tables: `results/**/*.csv.gz` (9 files, 239 MB);
  - by exact path: `transition_classes.csv` (37 MB), `statsbomb_complete_category_audit_data.json` (4.7 MB),
    the taxonomy `.xlsx`;
  - `thesis_code_review/` and `football_orientation_review.zip`: local repository exports that contain copies
    of raw StatsBomb files.
- **Manual review (kept local, not published):**
  - `results/data_quality_inspections/match3845506_PARTIAL_recovered_three_sixty.json`, recovered raw 360
    content from an excluded match. It is forensic evidence, but raw data is never published.
  - The orientation review queue: its HTML is tracked, its image panels are tracked, and the GIFs stay local.
- **Unchanged:** `data/` stays ignored as before. Nothing was deleted, and no Git LFS is used.
- **Secrets:** a scan of all commit candidates for key, token and password patterns and credential-like
  filenames found nothing.
- **Already tracked:** files committed in 3423816 that now match an ignore rule (a few `context_test_frame.png`)
  stay tracked; ignore rules apply only to untracked files.

## 6. Production dataset (`dataset_validation.json`)

Rebuilt twice with `scripts/build_full_dataset.py`. The previous output (built at 3423816) is in
`data/interim/processed_backup_2026-10-27_built_at_3423816/`.

**Determinism:** both builds produce SHA-256 `2627582f47a7013060fdd71b9cb1858d9d58ef0498e556c8d088a4a5cff1f6c3`.

| Check | Result |
|---|---|
| Matches | 417 |
| Rows | **756,023** (per-match files and manifest agree); same events as before (0 added, 0 removed) |
| Duplicate event ids / rows | 0 / 0 |
| Targets | 0 invalid or missing; identical to the previous build |
| Class distribution | unchanged: Pass 46.64%, Carry 41.56%, Ball Recovery 4.76%, Clearance 1.83%, Dribble 1.53%, Miscontrol 1.30%, Shot 1.24%, Dispossessed 1.15% |
| Schema | classified exactly; approved feature matrix 756,023 × 11 |
| Count invariants | 0 violations (causal, offline, raw) |
| Causal orientation | NORMAL 744,121, ROTATED_180 11,861, UNCERTAIN 41 (`orientation_available = false`, actor empty, 0 inconsistencies) |
| Identity quality (causal) | clean 752,340 / corrected 1,696 / uncertain 1,987 |
| Offline columns | identical to the previous build in every identity count and orientation |

The 2 rows with an available orientation but an empty actor position have no actor point in the raw snapshot;
this predates this audit.

### Every difference from the previous build, explained

| Change | Rows | Cause |
|---|---|---|
| Actor position mirrored | 16,367 (9 matches) | Reference-team fix (§2); verified exact mirror |
| Actor position now unavailable | 41 | Causal orientation cannot be decided |
| Causal identity counts changed | 81 | Causal orientations now feed causal identity, plus event-order availability |
| Anything else | 0 | — |

## 7. Tests

**365 passed, 0 failed** (full suite).

New tests:
- `tests/test_ml_feature_safety.py`: truncation invariance of the whole pipeline (5 real cuts), causal
  orientation truncation, separation of causal and offline values, schema coverage, allowlist, unsafe-column
  rejection, approval guard.
- `tests/test_kickoff_law8_diagnostic.py`: verdicts, no input mutation, not used by `src/`.
- `tests/test_build_statsbomb_360_dataset.py`: actor unavailable when orientation is undecided; reference team
  from event order.

## 8. Remaining methodological limitations

1. Strict causality has a price. 41 rows have no actor position; snapshots copied from a later event keep
   their raw identity reading.
2. A row's own 360 snapshot can be a copy captured for a simultaneous later event (same millisecond). This is a
   property of the data, not of the pipeline.
3. Rows exist only when a next meaningful event follows within 10 s, so evaluation is conditional on that.
4. The causal identity has no two-neighbour detector, since it needs the next snapshot.
5. Law 8 can judge only about half of the kick-offs; frame 505 and 3788749 P2 remain as read.
6. Previously documented: about 2–4% of residual colour transitions are estimated genuine single-player
   disagreements (final validation report of 2026-10-27).

## Files

| Path | Content |
|---|---|
| `orientation_55_rows.csv`, `orientation_55_rows_as_built_at_3423816.csv`, `orientation_55_summary.json` | §1 |
| `truncation_check.json` | §2 |
| `law8/kickoffs.csv`, `law8/summary.json` | §4 |
| `dataset_validation.json`, `dataset_sha256_build_1.txt`, `dataset_sha256_build_2.txt`, `manifest_build_1.json` | §6 |
| `diagnostics/` | `audit_orientation_55.py`, `truncation_check.py`, `validate_dataset.py`, build logs |
