# Data Quality Issues — Severity-Ranked (Second-Stage Investigation)

Date: 2026-10-13. Full evidence in `docs/data_quality_investigation_2026-10-13.md`.
Builds on `docs/bug_and_risk_report_2026-10-13.md` (first-stage findings
F-1…F-6, unchanged and not re-litigated here).

## Confirmed bugs (fixed in this investigation)

| ID | Severity | Status | Summary |
|---|---|---|---|
| F-7 | LOW | **FIXED** | `build_statsbomb_360_dataset.py` silently picked an arbitrary actor position via `next(...)` when a freeze_frame had 2+ `actor=True` points. In 26/28 real cases the points were effectively identical (safe either way); in exactly 2 (match 3802837) they were ~6.8 units apart — a genuine, previously-unflagged guess. Fixed with `resolve_actor_location()`: near-duplicates (≤1.0 unit) still resolve normally; genuine disagreements now report `actor_x=actor_y=None, actor_ambiguous=True` instead of guessing. |
| F-8 | LOW (tooling only) | **FIXED** | `scripts/audit_dataset.py`'s failure handling used a bare `except Exception: continue`, discarding the actual error for match 3845506 (only the id was recorded). Fixed: failures now capture `{match_id, file, error_type, error_message}`. No pipeline code was affected — only the audit script's own diagnostics. |

## Valid source-data characteristics (investigated, confirmed NOT bugs)

| Finding | Conclusion |
|---|---|
| **189,128 out-of-bounds freeze-frame points (0.88%)** | Genuine StatsBomb 360 tracking data. Dominated (87%) by touchline (Y-axis) overshoots of mostly small magnitude (62% ≤2 units) — consistent with real play near the sidelines. Affects 424/426 matches near-uniformly (not a localized processing artifact). Mathematically proven (isometry argument) and empirically verified (50-point real-data check, 0 mismatches) that orientation correction and team normalization can never create, remove, or hide an out-of-bounds point — they only reflect it, preserving the exact violation. The 89 extreme (>10 unit) outliers are all non-actor, non-keeper points, consistent with substitutes/staff visible in the camera frame, per StatsBomb 360's documented tracking scope. |
| **Match 3845506's three-sixty file is unparseable** | A single, isolated, 16,384-byte (exactly one storage block) run of NUL bytes overwrites real JSON content mid-array. Confirmed unique to this one file via a full NUL-byte scan of all 4,661 raw JSON files in the project (4,235 events + 426 three-sixty) — 0 other files affected. Storage/transfer-level corruption, not a schema or code issue; no safe recovery is possible without inventing lost data. Match remains correctly excluded and is now explicitly diagnosed (not silently dropped) by both the dataset builder (unhandled `JSONDecodeError`, propagates with file context) and the audit script (captured with full diagnostic detail). |
| **2 frames with zero actor points** | Both trace to real, correctly-associated, located events (50/50 and Dispossessed). The acting player was most plausibly outside the camera's tracked field of view at that instant — legitimate missing information per the 360 schema, not a code defect. Represented explicitly (`actor_x=actor_y=None, actor_ambiguous=False`), never invented. |
| **26/28 "multiple actor" frames** | Effectively duplicate points (≤0.5 unit apart) in the raw array — a StatsBomb-side data quirk present before any of our code runs, roughly evenly split between Problem-C-shared and non-shared snapshots, confirming it is unrelated to the shared-snapshot resolution logic. Not a defect in our pipeline; handled gracefully (picks the agreed-upon position). |
| **Orientation classification (93.90% NORMAL / 5.73% ROTATED_180 / 0.37% UNCERTAIN)** | Re-verified, not re-tuned. UNCERTAIN frames pass through unmodified (confirmed by code reading); zero/double-transformation ruled out (confirmed single, keyed-once application of each transform); the algorithm was not touched merely to change these numbers, per explicit instruction. |
| **13,303 events missing location / 7,072 missing player** | 100% concentrated in non-spatial administrative event types (Substitution, Half Start/End, Tactical Shift, Starting XI, Injury Stoppage, Referee Ball-Drop, etc.) per the StatsBomb schema's own design. **Zero** `MEANINGFUL_EVENT_TYPES` events are missing location or player info, dataset-wide — confirmed by direct recomputation. No impact on the dataset builder's actual input features or target labels. |
| **Shared-snapshot resolution (42,600 groups, 42,592 resolved, 8 ambiguous)** | Re-verified unchanged and correct: source-team resolution, actor-point attribution, and the dataset builder's/renderer's compatible use of the same `team_identity.py` logic were all re-confirmed by re-reading the code; no new issue found, no rewrite performed per "do not rewrite already-correct logic without evidence." |

## Unresolved / limitations

- The "substitutes/staff near the touchline" explanation for the 89
  extreme-overshoot points is the most evidence-consistent account
  available (actor/keeper flags, magnitude, category) but could not be
  independently confirmed against broadcast footage.
- No new automated test exists for `src/animate_statsbomb_360_match.py`
  directly (same gap as the first-stage audit); this investigation's
  visualization check was a manual spot-check plus cross-referencing the
  production CLI's own diagnostic output on one real match/window.

## Test suite impact

89 automated tests, all passing (73 from the first-stage audit + 1 new in
`test_orientation_correction.py` + 7 new in `test_build_statsbomb_360_dataset.py`
+ 8 new in `test_audit_dataset.py`). No test was weakened or skipped.
