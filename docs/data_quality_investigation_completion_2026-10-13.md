# Data Quality Investigation — Completion Report

Date: 2026-10-13 (continuation session). Companion to, and completes the
unfinished portions of:
`docs/data_integrity_audit_2026-10-13.md`, `docs/bug_and_risk_report_2026-10-13.md`,
`docs/data_quality_investigation_2026-10-13.md`, `docs/data_quality_issues_2026-10-13.md`.

This session resumed after the prior one was interrupted mid-task while
building `scripts/inspect_data_quality.py`. This report states, precisely,
what was already done, what was left unfinished, and what this session
completed — with evidence, not just a summary.

---

## 1. What the previous session had already completed

Verified by inspecting the repository on resume (git status, file
timestamps, and generated outputs in `results/data_quality_inspections/`):

- `scripts/inspect_data_quality.py` existed and was functional for all five
  modes (event inspection, out-of-bounds, actor-anomalies, orientation,
  corrupted-json), reusing `src/animate_statsbomb_360_match.py`'s pitch
  drawing, `src/orientation_correction.py`, `src/team_normalization.py`,
  and `src/build_statsbomb_360_dataset.py` directly (no duplicated
  transform logic).
- Generated and already on disk: a full event inspection for match
  3773477 event `49e3ba44`; out-of-bounds visualizations for match
  3788744 (whole-match and one event-scoped); actor-anomaly panels for
  matches 3773477 and 3802837; a corrupted-JSON diagnostic and a
  **9.4 MB partial-recovery file** for match 3845506 (2,761/2,762 raw
  360 records independently recovered via string-literal-aware brace
  matching).
- All findings from the FIRST completion round (F-7: actor-ambiguity
  guessing fixed; F-8: audit script's swallowed error diagnostics fixed)
  were already implemented in `src/build_statsbomb_360_dataset.py` and
  `scripts/audit_dataset.py`, with passing tests.

## 2. What was left unfinished when the previous session stopped

- **No test file existed for `scripts/inspect_data_quality.py`** — the
  single largest gap. The session was interrupted immediately after
  generating the match 3802837 actor-anomaly visualization, before
  writing any tests for the tool itself.
- The genuinely-ambiguous match 3802837 case had been *visualized* but not
  yet *written up* with full raw-JSON traceability (event metadata,
  `related_events`, exact candidate-to-`event.location` distances).
- The two zero-actor events had only been identified by UUID in the prior
  audit; neither had been traced to raw JSON, visualized, or had their
  root cause established in this session.
- Match 3845506's exclusion had not been re-confirmed against the
  *current* (post-fix) `build_statsbomb_360_dataset.py` entry point.
- No fresh, independent rerun of `scripts/audit_dataset.py` had been
  performed to confirm reproducibility against the existing baseline.
- The UUID-truncation usability risk (this tool's own reports print 8-char
  prefixes) had not been identified or addressed.

## 3. What this session investigated and completed

### 3.A — Match 3802837: genuinely ambiguous actor case (fully completed)

Traced both flagged events directly in raw JSON:

| Event | Type | `event.location` | `related_events` |
|---|---|---|---|
| `0a731280-...` | Ball Receipt* | `[36.0, 7.4]` | `47aad975-...` |
| `17380117-...` | Carry | `[36.0, 7.4]` | `0a731280-...`, `3efb9782-...`, `64908530-...` |

Both share the identical raw freeze_frame, with exactly two points flagged
`actor=True`: `[30.144737503288656, 10.794747684926325]` and
`[36.0, 7.400000095367432]` — **6.7682 pitch units apart** (recomputed
independently this session, matches the prior audit exactly).

**New finding**: the second candidate, `[36.0, 7.4]`, coincides with the
event's own `event.location` field to within floating-point rounding
(distance ≈ 1×10⁻⁷). This is suggestive corroborating evidence about which
candidate is more likely the true actor — but `resolve_actor_location()`
intentionally does **not** use `event.location` as a tie-breaker; it only
considers the freeze_frame's own internal evidence. This is a legitimate
methodological question for future improvement (§8), not something this
session implemented unilaterally, since it would be a deliberate design
change to resolution logic that currently has test coverage asserting the
opposite (no guessing).

**Verified**: `resolve_actor_location()` returns `(None, None, True)` for
both events — re-confirmed via the inspection tool's `--issue
actor-anomalies` and `--event` modes independently (both agree). The
existing regression test `test_resolve_actor_location_with_genuinely_different_actor_points_is_ambiguous`
(from the prior session) already uses these exact real coordinates — no
new test was needed for the core behavior; see §3.F for an adjacent
finding this traced to.

### 3.B — Both zero-actor events (fully completed, root cause established)

| Match | Event | Type | Team | `event.location` |
|---|---|---|---|---|
| 3893793 | `e43bc7e0-...` | 50/50 | Japan Women's | `[32.5, 44.2]` |
| 4018356 | `dfe42790-...` | Dispossessed | Switzerland Women's | `[45.1, 2.9]` |

**Root cause, established directly from raw data (not hypothesized)**:
in **both** cases, every single freeze_frame point has `teammate=False`
(14/14 and 16/16 respectively — zero teammates visible at all). Per
StatsBomb's own schema invariant (`actor=True` implies `teammate=True`),
zero teammates visible necessarily means zero actor points are possible —
this is a direct logical consequence of the camera's tracked field of view
at that instant happening to exclude the acting player's entire team, not
a flagging anomaly, an event/frame mismatch (both confirmed as exact,
correct `event_uuid` joins), or a shared-snapshot artifact (`detect_shared_cross_team_snapshots`
confirms `shared=False` for both). Both classify as `ROTATED_180` (0.940
and 0.851 confidence) using the classifier's `nearest_point`/`centroid`/
`visible_area` fallback votes, since the primary `actor` vote is
unavailable — confirming the classifier degrades gracefully when no actor
exists.

Generated focused two-panel (raw vs. current-pipeline) visualizations for
both events (`match3893793_event_e43bc7e0.png`, `match4018356_event_dfe42790.png`)
and confirmed visually: all-red (all-opponent) point clouds, no actor
ring, with the `visible_area` polygon tightly framing one zone of the
pitch — consistent with a corner/box-focused camera angle that happened
not to include the actor's own side.

**New regression test added** (`test_resolve_actor_location_real_zero_actor_case_all_opponents_visible`
in `tests/test_build_statsbomb_360_dataset.py`) using a representative
subset of match 3893793's real freeze_frame, pinning this exact scenario
down permanently.

### 3.C — Match 3845506 (re-verified against current code)

Re-confirmed, this session, against the current (fixed) entry point:

```
build_match_dataset(match_id=3845506, maximum_time_difference=10.0)
  -> raises json.decoder.JSONDecodeError:
     Expecting ',' delimiter: line 92794 column 3 (char 2637824)
```

The dataset-building pipeline **fails loudly**, not silently — confirmed
directly this session (not merely asserted from the prior report). The
audit script's `results/data_quality_audit_2026-10-13c.json` (fresh
rerun, see §5) explicitly lists this match under `failed_matches` with
full diagnostic detail (`match_id`, `file`, `error_type`, `error_message`),
never silently counted as processed.

Confirmed the corruption is a single, isolated 16,384-byte NUL-byte run
(re-verified this session: `find_nul_runs` on the raw bytes reports
exactly one run at `offset=2637824, length=16384`). The raw file's own
modification timestamp (`May 26 16:59`, unchanged since before this
investigation began) and MD5 checksum were checked this session and
confirm it has never been written to.

### 3.D — Orientation and coordinate transformations (re-verified + one new finding)

Fresh spot checks this session (not reused from the prior report) for
match 3773477, all three orientation classes:

- **UNCERTAIN** (`bbd36758-...`, confidence 0.425): visualized side-by-side;
  confirmed the raw and "current pipeline" panels show the *same* spatial
  configuration mirrored only by team normalization (Real Betis is the
  non-reference team) — the orientation (Problem A) mirror was correctly
  **not** applied, exactly matching the documented "never guess" policy.
- **NORMAL** (period 1, `195352ca-...`, confidence 0.935) and
  **ROTATED_180** (period 2, `aec2dae2-...`, confidence 0.873): both found
  and rendered without error via `--orientation`/`--period` filtering.
- **Distance/geometry preservation**: recomputed this session across 500
  real event/frame pairs from match 3773477 (not reused from the prior
  session's single-match proof) — **zero violations**: every pairwise
  distance between freeze_frame points is preserved to within 1×10⁻⁶ units
  through the full orientation + team-normalization pipeline.
- **Double-application safety**: confirmed `apply_team_normalization_to_frame`
  is **not inherently idempotent** — calling it twice on an already-mirrored,
  still-`MIRRORED`-labeled frame mirrors it a second time, landing back on
  the original raw coordinates (233/500 sampled frames would be affected if
  this ever happened). The pipeline is safe only because the calling code
  (`build_statsbomb_360_dataset.py:332`, `animate_statsbomb_360_match.py:619`)
  invokes this function **exactly once per frame**, confirmed by direct
  code reading of both call sites (single linear pass, no loops that could
  reprocess a frame). **New tests added**
  (`test_team_normalization_preserves_pairwise_distances`,
  `test_apply_team_normalization_is_not_idempotent_calling_code_must_apply_once`)
  pin both of these properties down permanently.

**New finding (latent, not previously documented)**: `orientation_correction.py`'s
`extract_geometry()` uses the same unguarded `next(...)` pattern to pick an
actor point that `build_statsbomb_360_dataset.py` had (before the earlier
fix) for its own actor resolution — i.e., when a freeze_frame has multiple
`actor=True` points, the orientation classifier's primary "actor" vote
(weight 0.40, the highest of any vote) is computed from an **arbitrarily
chosen** one of the candidates, not flagged as uncertain or averaged.

Tested empirically on all 3 cases where this could plausibly matter:
- The 2 genuinely-ambiguous match 3802837 events: classification stays
  `NORMAL` either way; confidence shifts from 0.874 to 0.942 depending on
  which candidate is used — comfortably on the same side of every
  threshold in both cases.
- 2 near-duplicate cases (match 3773477 `49e3ba44`, match 3773672
  `a9fb5567`): confidence changes by <0.003 when swapped, as expected
  given the candidates are ≤0.22 units apart.

**Conclusion**: this is a real, latent code-quality inconsistency (the
classifier does not use the same disambiguation-tolerance logic as the
dataset builder), but it has **zero confirmed effect on any real
classification outcome** across all 28 known multiple-actor frames in the
dataset (26 are near-duplicates where it mathematically cannot matter; the
2 genuinely-divergent ones were directly tested and do not flip).
Classified as a **LOW-severity, documented-but-not-fixed** finding —
changing `orientation_correction.py` (the project's single source of
truth for orientation, reused by both consumers) is a higher-risk change
than this investigation's evidence currently justifies; see §8 for a
recommendation.

### 3.E — Out-of-bounds coordinates (re-verified, not re-derived from scratch)

Pulled the already-computed distribution from `results/data_quality_audit_2026-10-13b.json`
and cross-checked it against an independent fresh rerun (`...c.json`,
§5) — **byte-for-byte identical across all 32 top-level fields**,
confirming full reproducibility:

- 189,128 / 21,561,945 points out of bounds (0.88%).
- Dominated by the Y-axis (touchline): `y>80` 83,206 + `y<0` 81,605 vs.
  `x>120` 14,201 + `x<0` 10,054 — consistent with real touchline/sideline
  positions, not goal-line positions (which would require standing behind
  a goal).
- 424/425 successfully-loaded matches affected (near-universal, small
  per-match rate — the signature of organic tracking behavior, not a
  localized processing bug).
- Magnitude: 62.6% of violations overshoot by ≤2.0 units; only 89 (0.05%)
  exceed 10 units.
- Re-confirmed this session, with the dedicated property test
  `test_mirror_preserves_out_of_bounds_status_exactly` (added in the first
  completion round, re-run and still passing): orientation correction and
  team normalization are isometries and **cannot** create, remove, or hide
  an out-of-bounds point — they only reflect it by the identical margin.
- Re-confirmed coordinate-convention consistency: `scripts/inspect_data_quality.py`
  and `scripts/audit_dataset.py` both use the identical
  `0.0 <= x <= 120.0 and 0.0 <= y <= 80.0` check as
  `src/animate_statsbomb_360_match.py::point_inside_pitch`, and the
  renderer's `FIXED_X_LIMITS`/`FIXED_Y_LIMITS` deliberately pad the
  viewport so near-boundary points remain visible rather than being
  clipped. No methodological option to "fix" these points was found or
  implemented — they are valid source data, left untouched.

### 3.F — Duplicate/multiple actor points, match 3773477 (re-verified + new finding)

Re-checked all 5 events from `actor_anomalies_match3773477.png` directly
against raw JSON: confirmed 4 are **exact** duplicates (`a5656e1f`,
`68022eee`, `284b4d1b`, `aa92e1df`, max pairwise distance 0.0000–0.0008)
and 1 is a sub-millimeter encoding variant (`49e3ba44`, 0.0008 units,
effectively a corner-flag position recorded twice with negligible
floating-point divergence). All resolve correctly via
`resolve_actor_location()` (`actor_ambiguous=False` in every case,
confirmed via the inspection tool's builder cross-check).

**New finding, quantified this session**: `build_rows_from_match_data`'s
`visible_players`/`visible_teammates` counts are **not deduplicated** —
for events `284b4d1b` and `aa92e1df` (both real CSV rows), `visible_players=15`
and `visible_teammates=8`, one count higher than the true number of
distinct physical players (14), because the literal duplicate freeze_frame
entry is counted twice. This is a genuine, confirmed, **LOW-severity**
downstream effect: given only 28 affected frames dataset-wide (0.002%),
this inflates spatial-density features by at most 1 for an extremely small
number of rows — not fixed (deduplicating the *entire* freeze_frame, not
just actor points, is a broader question outside this investigation's
scope and risks incorrectly merging two distinct nearby players), but now
explicitly **documented and pinned by a new regression test**
(`test_duplicate_actor_point_inflates_visible_player_count_by_one`).

## 4. Inspection tool reliability (task 3) — bugs found and fixed

Reviewed `scripts/inspect_data_quality.py` end-to-end this session:

- **Confirmed working correctly, unchanged**: exact match-id/event-UUID
  filtering (no fuzzy substitution anywhere); PNG/TXT/JSON report content
  agreement (spot-checked `match3773477_event_49e3ba44` — the printed
  `actor_x=120.0` matches the rendered star marker's position exactly);
  out-of-bounds counts/distances (cross-checked against an independent
  manual recomputation for match 3788744 — exact match); missing/corrupted
  file handling (clear, specific messages, never a bare traceback); no
  silent exception suppression anywhere in the tool.
- **Bug found and fixed**: a shortened/truncated UUID (e.g. the 8-char
  prefixes this tool's *own* reports print, like `event=49e3ba44`) was
  rejected with only a generic "not found" message — easy to misread as a
  real failure when the user likely just copy-pasted a truncated id from
  an earlier report. **Fixed**: added `describe_uuid_lookup_failure()`,
  which detects when the given value is a prefix of exactly one real id in
  the match and suggests the full UUID explicitly (and reports ambiguity
  rather than guessing if it matches 2–3 ids). Applied consistently to all
  three UUID-lookup call sites (event inspection, out-of-bounds-by-event,
  orientation-by-event). Verified: `--event 49e3ba44` now prints *"This
  looks like a truncated UUID. Did you mean the full UUID:
  49e3ba44-0b26-4a81-8721-dcc7f4f3831e ?"*
- **Output-overwrite behavior**: confirmed each inspection writes to a
  deterministic, case-specific filename inside its own dedicated directory
  (`results/data_quality_inspections/`); re-running the same inspection
  intentionally overwrites its own prior output (idempotent), while never
  touching any file outside that directory — verified via `git status`
  and directory-wide `find`/timestamp checks before and after this
  session's ~15 tool invocations.

## 5. Automated test suite — final results

```
.venv/bin/python -m pytest tests/ -v
...
118 passed in 0.92s
```

**118 passed, 0 failed, 0 skipped, 0 errors.** Breakdown by file (actually
run and counted this session, not estimated):

| File | Tests |
|---|---|
| `test_audit_dataset.py` | 8 |
| `test_build_statsbomb_360_dataset.py` | 31 (29 carried over + 2 new this session) |
| `test_inspect_data_quality.py` | 25 (**new this session** — did not exist before) |
| `test_orientation_correction.py` | 13 |
| `test_paths.py` | 1 |
| `test_team_identity.py` | 19 |
| `test_team_normalization.py` | 21 (19 carried over + 2 new this session) |

No existing test was modified, weakened, or skipped to make anything pass.

## 6. Dataset audit — reproducibility verification

Ran `scripts/audit_dataset.py --output results/data_quality_audit_2026-10-13c.json`
(fresh, full, ~2-minute run across all 426 matches) and diffed it
programmatically against the existing `...b.json` baseline:

```
Keys with ANY difference between the two independent full runs: []
Total keys compared: 32
```

**Byte-for-byte identical.** All headline figures reconfirmed stable:
425/426 matches loaded (1 failure: 3845506, with full diagnostic detail),
1,357,627 matched pairs, 224,631 events without 360 coverage, 0 duplicate
ids/UUIDs, 0 index-ordering violations, 189,128 out-of-bounds points, 2
zero-actor frames, 28 multiple-actor frames, orientation
93.90%/5.73%/0.37% NORMAL/ROTATED_180/UNCERTAIN, 42,600 shared groups
(42,592 resolved / 8 ambiguous). Also re-verified the `--compare-csv`
read-only cross-check against match 3902240's existing processed CSV
(2,013 rows ≤ 2,149 raw upper bound, as expected) without writing to it.

Three audit report files now coexist in `results/`, none overwritten:
`...2026-10-13.json` (original baseline), `...2026-10-13b.json`
(first completion round), `...2026-10-13c.json` (this session).

## 7. Files changed this session

- `src/build_statsbomb_360_dataset.py` — **unchanged this session** (all
  fixes from the first completion round already in place; re-verified
  only).
- `scripts/inspect_data_quality.py` — added `describe_uuid_lookup_failure()`
  and wired it into all three UUID-lookup sites (bug fix, §4).
- `tests/test_build_statsbomb_360_dataset.py` — 2 new tests (real
  zero-actor case, duplicate-point count inflation).
- `tests/test_team_normalization.py` — 2 new tests (distance preservation,
  double-application non-idempotence).
- `tests/test_inspect_data_quality.py` — **new file**, 25 tests covering
  every mode of the inspection tool.
- `docs/data_quality_investigation_completion_2026-10-13.md` — this report.
- New generated outputs in `results/data_quality_inspections/`:
  `match3893793_event_e43bc7e0.{png,txt}`, `match4018356_event_dfe42790.{png,txt}`,
  `match3802837_event_0a731280.{png,txt}`, plus several orientation-mode
  PNGs/TXTs for match 3773477.
- `results/data_quality_audit_2026-10-13c.json` — new, fresh full audit
  run (baselines `.json`/`b.json` preserved, untouched).

No raw file under `data/raw/` was modified (verified via `git status`,
MD5 checksum, and file-modification timestamps). No file under
`data/processed/` was touched this session.

## 8. Remaining unresolved issues and recommendations

| Issue | Severity | Status |
|---|---|---|
| `orientation_correction.py::extract_geometry` picks an arbitrary actor candidate when multiple exist (§3.D) | LOW | Documented, not fixed — zero confirmed real-world impact across all 28 known cases, but a latent inconsistency with the already-fixed dataset-builder logic. |
| `visible_players`/`visible_teammates` not deduplicated for literal duplicate freeze_frame points (§3.F) | LOW | Documented and tested, not fixed — affects ≤28 frames dataset-wide (0.002%), inflates counts by at most 1. |
| Match 3802837's `event.location` proximity to one actor candidate (§3.A) | Methodological | Not implemented — a deliberate, evidence-informed tie-breaking heuristic is a design decision requiring supervisor input, not a technical bug fix. |
| No automated test exists for `src/animate_statsbomb_360_match.py` directly | Disclosed gap | Unchanged from the first completion round; this session's new tests cover the inspection *tool*, not the production animation module itself. |

**Recommendations for preprocessing / thesis methodology** (facts above,
opinions here, kept separate as requested):

1. The dataset is reliable enough to proceed with model development for
   features derived from `MEANINGFUL_EVENT_TYPES` rows: zero of the
   confirmed data-quality findings (out-of-bounds points, zero/multi-actor
   frames, the one corrupted match) touch any row actually used for
   training, beyond the already-fixed, already-tested `visible_teammates`/
   `visible_opponents`/`actor_ambiguous` columns.
2. Document in the thesis methodology section: (a) match 3845506 is
   permanently excluded (1/426, 0.23% of matches) due to irrecoverable raw
   file corruption; (b) `actor_ambiguous=True` and `actor_x/actor_y=None`
   rows (≤30 known cases dataset-wide) should be filtered or imputed
   deliberately, not silently treated as zero; (c) `visible_players` is a
   near-exact but not perfectly deduplicated headcount.
3. Consider, as low-priority future work: aligning
   `orientation_correction.py`'s actor selection with the dataset
   builder's tolerance-based disambiguation for full internal consistency
   (not urgent given zero measured impact).

## 9. Questions requiring a supervisor's decision (not technical guesses)

1. Should `event.location` ever be used as corroborating evidence to
   resolve a genuinely ambiguous actor position (match 3802837-style
   cases), or should the stricter "never use a second data source to
   disambiguate" policy remain absolute?
2. Is a dedicated automated test suite for the visualization module
   (`src/animate_statsbomb_360_match.py`) itself required before the
   thesis is submitted, or is the existing manual/visual verification
   record (this report, the two prior audit reports, and the generated
   inspection PNGs) sufficient evidence of correctness?
3. Should `MEANINGFUL_EVENT_TYPES`'s exact 8-type definition be
   additionally justified/cited in the thesis methodology (carried over
   from the first completion round, still unresolved)?
