# Audit and removal of the 10-second next-event restriction

Date: 2026-10-27. **Decision: the restriction was not justified and has been removed.** No model was trained or
evaluated.

## 1. Original implementation and rationale

- **Where:** `src/build_statsbomb_360_dataset.py::build_rows_from_match_data`. After finding the target with
  `find_next_meaningful_event`, it dropped the row if
  - `time_difference > maximum_time_difference` (the 10-second rule), or
  - `time_difference < 0`.
- **Defaults:** `--max-time 10.0` in the builder CLI, repeated in `scripts/build_full_dataset.py`. Every caller
  and test passed `10.0`.
- **Origin:** the builder's first commit (`f12931a`, 2026-07-30), with no stated rationale.
- **Later documents:**
  - `docs/data_integrity_audit_2026-10-13.md` calls it "an additional cap on top of" the next-event framing.
  - `docs/prediction_task_feasibility_2026-10-14.md` calls it "a reasonable default … not a systematic subset".
  - No document gives a scientific or technical reason. **Confirmed:** it was an arbitrary preprocessing threshold.
- **What it affected:** row filtering only. The target, event sequencing and features were computed the same way
  before the filter.
- **Other time windows in the pipeline:** the identity and orientation rules use short windows (≤ 2–10 s) to find
  *evidence*. Those never select rows or targets, and they are unchanged.
- **Interaction with 360 data:** a current event needs a 360 record; the next event never did. Among the rows the
  rule excluded, only 47.6% of next events have a 360 record, vs 93.7% among retained rows. The excluded rows are
  mostly restarts after stoppages, which StatsBomb often captures without 360.

## 2. Quantitative impact (before any change; `funnel.json`, `breakdowns.json`, `tables/`)

Computed with the builder's own definitions. The "retained" count reproduces the production dataset exactly
(756,023).

| Stage | Events |
|---|---|
| Events in the 417 usable matches | 1,551,841 |
| … of a meaningful type | 893,476 |
| … without a 360 record (independent rule, unchanged) | 105,389 |
| **Eligible current events** (meaningful type + 360 record) | **788,087** |
| No meaningful successor in the same period (no target) | 756 |
| Excluded by the negative-gap rule | 0 |
| **Excluded only by the 10-second rule** | **31,308 (3.97% of eligible; 3.98% of those with a target)** |
| Retained by the old builder | 756,023 |

### Gap to the next event (787,331 events with a target)

| Statistic | Value |
|---|---|
| min | 0 s |
| P25 | 0.86 s |
| **median** | **1.34 s** |
| mean | 2.79 s |
| P75 | 2.10 s |
| **P90** | **3.70 s** |
| **P95** | **6.59 s** |
| **P99** | **36.86 s** |
| **max** | **503.41 s** |

Gaps of exactly 0 s (simultaneous events): 40,165.

| Gap range | Events |
|---|---|
| (10, 15] s | 6,301 |
| (15, 30] s | 13,429 |
| (30, 60] s | 8,489 |
| (60, 120] s | 2,566 |
| (120, 300] s | 514 |
| > 300 s | 9 |

### Exclusion was strongly selective

Share excluded, by current event type:

| Current event type | Excluded |
|---|---|
| **Shot** | **64.3%** (6,665 of 10,361) |
| Clearance | 19.0% |
| Miscontrol | 14.3% |
| Dispossessed | 13.5% |
| Dribble | 7.1% |
| Pass | 3.0% |
| Ball Recovery | 2.4% |
| Carry | 2.0% |

Share excluded, by next event type (the label):

| Next event type | Excluded |
|---|---|
| Pass | 7.9% |
| Shot | 5.6% (558 of 9,905) |
| Ball Recovery | 1.0% |
| Others | ≤ 0.14% |
| Carry | 0% |

Other cuts:
- **Transitions:** Shot→Pass 85% excluded (a restart after a shot), Clearance→Pass 37%, Pass→Pass 17%.
- **Periods:** 3.7% (P1), 4.2% (P2), 4.6% (P3), 5.1% (P4), and **100% of the penalty shootout (P5, 104 rows)**.
- **Matches:** 1.8–8.6% per match (median 3.9%), spread across all 417.

## 3. Assessment

| Consideration | Finding |
|---|---|
| Next-event classification vs a fixed horizon | The thesis task is next-event classification. A 10 s cap silently turns it into "the next event, *given* it happens within 10 s", a different, conditional task. |
| Selection bias | Confirmed. Rows after stoppages and restarts were removed (64% of current Shot rows, all shootout rows), so the old dataset under-represented exactly the situations where the next event is a restart. |
| Rare classes | Shot as a *target* lost 5.6%. Shot as a *current event* lost 64%: the model would barely have seen what follows a shot. |
| Match and period boundaries | Handled by event order within the period (unchanged); not related to a time cap. |
| Timestamp semantics | Timestamps are period-relative and can tie (40,165 zero gaps) or, rarely, run backwards; event order is the reliable sequence. The negative-gap rule never fired. |
| Long stoppages | Present (max 503 s; e.g. injuries or VAR). The next event is still well defined. |
| Technical need | None found. Features never depend on the gap; the gap is future information and was never a feature. |

**Decision:** remove the 10-second rule and the negative-gap rule, with no replacement threshold.

## 4. New target semantics (implemented)

For every event of a `MEANINGFUL_EVENT_TYPES` type that has a 360 record, `next_event_type` is the type of the
**first later event in the raw event stream** whose type is meaningful and that lies **in the same period**.

- **Event order:** the raw list order, which equals `index`. It decides ties and non-monotonic timestamps.
  Equal timestamps keep stream order.
- **Gap:** no time window. The gap is kept as `time_to_next_event_seconds`, classified TARGET (future
  information) and never in the feature allowlist. It can be 0, or negative for non-monotonic timestamps.
- **360 data:** the next event does not need it.
- **Terminal events:** the last meaningful event of a period has no target and is not a row. Targets never
  cross halftime or full time.
- **Unchanged rules:** the meaningful event types, the 360-record requirement for current events, and every
  feature rule.
- **Code:** `build_rows_from_match_data` and `build_match_dataset` no longer accept a time parameter, and the
  `--max-time` options are gone, so a horizon cannot be re-imposed by accident.

## 5. Old vs new dataset (`rebuild_validation.json`)

| | Old (021867b) | New |
|---|---|---|
| Matches | 417 | 417 |
| Rows | 756,023 | **787,331 (+31,308)** |
| Rows removed | — | 0 |

**Added rows:** exactly the 31,308 events excluded only by the 10-second rule (set equality verified).

**Old rows:** all 756,023 are **identical in all 41 compared columns** (0 changed cells), covering features,
causal identity, causal orientation, offline diagnostics and targets.

**Targets:** all 787,331 match an independent re-derivation from the raw StatsBomb event stream. Every eligible
event with a successor has a row.

**Determinism:** two independent builds have the same SHA-256 (`8f6ff118…`).

**Backup:** the previous production output is in `data/interim/processed_backup_2026-10-27_built_at_021867b/`.

### Target class distribution

| Next event | Old rows | New rows | Added | Old % | New % |
|---|---|---|---|---|---|
| Pass | 352,592 | 382,948 | +30,356 (+8.6%) | 46.64 | 48.64 |
| Carry | 314,169 | 314,169 | 0 | 41.56 | 39.90 |
| Ball Recovery | 35,959 | 36,321 | +362 | 4.76 | 4.61 |
| Clearance | 13,827 | 13,829 | +2 | 1.83 | 1.76 |
| Dribble | 11,597 | 11,608 | +11 | 1.53 | 1.47 |
| **Shot** | **9,347** | **9,905** | **+558 (+6.0%)** | 1.24 | 1.26 |
| Miscontrol | 9,822 | 9,829 | +7 | 1.30 | 1.25 |
| Dispossessed | 8,710 | 8,722 | +12 | 1.15 | 1.11 |

**Imbalance ratio** (largest / smallest class) rises from 40.48 to 43.91, because the added rows are mostly
restart Passes. The rare target classes keep nearly identical shares; Shot gains 558 labels.

**Current events:** Shot rows rise from 3,696 to 10,361. The model now sees what follows every shot, not only
shots followed by quick rebounds.

## 6. Leakage and integrity

- **Allowlist:** still enforced. The approved matrix is 787,331 × 11; `time_to_next_event_seconds` is not
  approved and is rejected by `validate_feature_columns`.
- **Truncation invariance:**
  - unit tests on real rows pass;
  - the 155-cut check from the integrity audit is unchanged (old rows are identical);
  - a new check on **60 added rows with gaps over 10 s** compared 1,412,550 cells with **0 mismatches**
    (`truncation_check_added_rows.json`).
- **Identity and orientation:** the 41 unavailable orientations and the identity values are unchanged for the old
  rows. The new rows use the same causal logic.
- **Invariants:** count invariants and the schema check show 0 violations.
- **Diagnostic flag:** `orientation_uses_later_records` is now 56. One added row's *offline* orientation read
  later records; its causal orientation is resolved. Diagnostic only.

## 7. Tests

**374 passed, 0 failed.** New regression tests in `tests/test_build_statsbomb_360_dataset.py`:
- next event after 2 s, exactly 10 s, 11 s, 30 s and 487.5 s (a long stoppage);
- no time parameter on the builder;
- simultaneous events;
- equal timestamps in a different stream order;
- non-monotonic timestamps (negative gap kept);
- last event of a period (no target, no crossing of halftime);
- last event of the match.

Also covered:
- a next event without 360 data (existing `test_next_event_does_not_require_360_frame`);
- causal feature independence (`tests/test_ml_feature_safety.py`, now without a window).

## 8. Remaining methodological limitations

1. **Established event set.** The target is the next event among the 8 meaningful types. Other events in
   between (e.g. Ball Receipt*, Pressure, Duel) are skipped by definition. This choice is earlier and
   documented, and it is not changed here.
2. **Assumption: shootouts are in scope.** The 104 penalty-shootout rows (period 5) are now included as a
   separate period, and the `period` feature identifies them. Whether the thesis analyses them is a scope
   decision; they can be filtered by period without any time rule.
3. **Current events need a 360 record.** This is an independent eligibility rule (105,389 meaningful events
   without one), unchanged.
4. **No terminal class.** The last meaningful event of each period (756) has no target; no "end of period"
   class is invented.
5. **Historical scripts not updated.** The report scripts under `results/team_identity_*` call the builder with
   the old `10.0` argument; they document the datasets of their commits and are not part of the pipeline.
6. **Higher imbalance.** It is now 43.9. This is the true distribution of the task and must be handled at
   modelling time.

**Confirmed findings:** sections 1, 2 and 5–7 are measured on the data and code. **Assumptions:** that the thesis
task is next-event classification without a horizon (as stated by the user), and limitation 2.

## Files

| Path | Content |
|---|---|
| `funnel.json`, `eligible_events.csv.gz` (local only) | §2 eligibility funnel and per-event outcomes |
| `breakdowns.json`, `tables/` | exclusion rates by type, match, period and transition; class distributions |
| `rebuild_validation.json`, `truncation_check_added_rows.json`, `dataset_sha256_build_{1,2}.txt` | §5–6 |
| `diagnostics/` | `quantify_time_window.py`, `breakdowns.py`, `validate_rebuild.py`, `truncation_check_added_rows.py`, build logs |
