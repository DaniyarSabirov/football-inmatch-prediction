# Second-Stage Data Quality Investigation — StatsBomb 360 Pipeline

Date: 2026-10-13 (follow-up to `docs/data_integrity_audit_2026-10-13.md` and
`docs/bug_and_risk_report_2026-10-13.md`). This document investigates the
specific findings those reports flagged as needing deeper evidence, using
the raw StatsBomb JSON directly.

Every claim below states exactly what evidence supports it. No raw source
file was modified. No existing processed CSV was overwritten during this
investigation (one match's CSV was regenerated to `/tmp` only, for
comparison, in the first-stage audit — not repeated here).

---

## 1. The 189,128 out-of-bounds freeze-frame points (highest priority)

### 1.1 Recomputed, more detailed counts

Full-dataset rerun with a category-level breakdown (see
`results/data_quality_audit_2026-10-13b.json` for the complete machine-
readable output):

- **Total well-formed freeze-frame points**: 21,561,945
- **Out of bounds**: 189,128 (0.88%)
- **Category breakdown** (a point can match more than one category, e.g. a
  corner overshoot hits both an x- and a y-boundary):
  - `y>80`: 83,243
  - `y<0`: 81,630
  - `x>120`: 14,234
  - `x<0`: 10,083
- **Malformed or non-numeric points**: 0 (every point in the dataset has a
  well-formed `[x, y]` numeric pair — the violation is purely a *range*
  issue, never a *shape/type* issue).
- **Magnitude of the overshoot** (distance past the nearest violated
  boundary):

  | Overshoot | Count | % of OOB points |
  |---|---|---|
  | ≤0.5 | 40,328 | 21.32% |
  | ≤1.0 | 31,810 | 16.82% |
  | ≤2.0 | 46,187 | 24.42% |
  | ≤5.0 | 59,212 | 31.31% |
  | ≤10.0 | 11,502 | 6.08% |
  | >10.0 | 89 | 0.05% |

- **Distinct affected matches**: 424 / 426 (99.5%)
- **Distinct affected (match, event_uuid) pairs**: 159,238

### 1.2 Root-cause classification

**Dominant pattern (Y-axis, touchline, small magnitude)**: `y>80` and
`y<0` together account for 164,873 / ~189,500 category hits
(≈87%), vastly outnumbering the X-axis (goal-line) violations. This is
football-physically meaningful: players are tracked at or marginally past
the **touchline** far more often in real play (throw-ins, wide corners,
pressing out wide, overlapping full-backs) than past the **goal-line**
(which would require standing behind the goal — essentially only possible
for a goalkeeper on a goal-line clearance, or someone off-pitch near the
goal). ~62% of all OOB points overshoot by ≤2.0 pitch units — consistent
with a player's feet on the line but their tracked center-of-mass/marker
a fraction of a unit past it.

**424/426 matches affected, evenly**: a processing bug would typically
either affect *every* match identically and systematically, or cluster in
a specific subset tied to a particular condition (a stadium, a competition,
a code path triggered only sometimes). Instead this is spread almost
universally and in small per-match proportions — the signature of organic
measurement/tracking behavior at the pitch edges, not a systematic
transformation defect.

**The 89 extreme (>10 unit) outliers**: inspected individually. All 89 are
`actor=False, keeper=False` (never the acting player or goalkeeper), and
the overshoot is overwhelmingly in Y, with values as large as `y=117.3`
(37 units past the touchline) in match 3788744. These are consistent with
**non-participating people visible in the camera frame** — substitutes
warming up, technical-area staff, or ball boys near the touchline — which
StatsBomb 360's publicly documented scope explicitly includes (it tracks
all detected player-shaped objects visible in the broadcast camera's field
of view, not strictly players engaged in play). This is plausible, visible
source data, not a parsing or transformation defect.

**Representative traced examples** (raw JSON → corrected coordinates →
bounds status), reproduced via a dedicated script
(`/tmp/audit2/oob_investigation.py`, not committed — a one-off diagnostic;
the durable, reusable check now lives in
`scripts/audit_dataset.py`/`tests/test_orientation_correction.py`):

| Match | Event (prefix) | Raw location | Violation | Overshoot |
|---|---|---|---|---|
| 3764440 | 8e6876df | `[40.46, 80.21]` | y>80 | 0.21 |
| 3764440 | ca3058e2 | `[-0.84, 37.35]` | x<0 | 0.84 |
| 3764440 | 90926131 | `[120.19, 41.29]` | x>120 | 0.19 |
| 3788744 | dd2262ff | `[45.11, 117.30]` | y>80 | 37.30 |

### 1.3 Does orientation correction or team normalization create or hide violations?

**No — proven, not assumed.** `mirror_point((x,y)) = (120-x, 80-y)` is a
180° rotation about the pitch center, i.e. an **isometry** that maps the
rectangle `[0,120]×[0,80]` exactly onto itself. An isometry cannot move a
point from outside the rectangle to inside it, or vice versa — it can only
reflect an out-of-bounds point to the opposite side by the *same*
magnitude. `apply_team_normalization_to_frame` reuses this exact same
`apply_correction`/`mirror_point` machinery (confirmed by reading
`src/team_normalization.py:140-153` — it literally calls
`apply_correction(frame_360, ORIENTATION_ROTATED_180)`), so there is only
one mirror implementation in the whole codebase, used everywhere.

Empirically verified on 50 real points across 3 known-OOB events (match
3764440): ran the full pipeline (`classify_orientation` →
`apply_correction` → `apply_team_normalization_to_frame`) and compared
in/out-of-bounds status before and after. **Zero mismatches.** New
regression test: `tests/test_orientation_correction.py::test_mirror_preserves_out_of_bounds_status_exactly`
(exact-coordinate assertions on both out-of-bounds and in-bounds synthetic
fixtures, plus the exact overshoot-magnitude-preservation property).

### 1.4 Does the visualization use the same bounds assumptions as the audit script?

**Yes.** `src/animate_statsbomb_360_match.py:1623` uses the identical
`0.0 <= point[0] <= 120.0 and 0.0 <= point[1] <= 80.0` check (confirmed by
direct code reading) for its own `validate_selected_frames` /
`StructuralReport` diagnostic. Spot-checked on match 3764440,
period 1, 00:30–00:40: the production CLI's own printed structural report
said *"Player coordinates inside pitch: 117/118 (1 outside across 1
frames in raw data)"* — independently corroborating the audit script's
methodology and counts on a real, directly-rendered window.

Crucially, the renderer does **not** clip, delete, or hide out-of-bounds
points: it renders within a deliberately *padded* viewport
(`FIXED_X_LIMITS = (-3.0, 123.0)`, `FIXED_Y_LIMITS = (83.0, -3.0)` —
`src/animate_statsbomb_360_match.py:86-87`) specifically so near-boundary
players remain visible, while the strict `[0,120]×[0,80]` check is used
only for the separate structural-report *count*, never as a filter. This
is already the correct design.

### 1.5 Conclusion

**These 189,128 points are valid source data, not a bug** — in the audit
script, in the transformation pipeline, or in the visualization. The
category distribution (heavily touchline, rarely goal-line), the magnitude
distribution (mostly small), the near-universal match coverage, and the
isometry-preservation proof together rule out a processing-stage
explanation. No clipping, deletion, or "fix" was applied, consistent with
the project's standing policy to never silently alter unusual but
plausible records. **Confirmed bug count from this investigation: 0.**
One audit-script wording improvement was made (see §6) for transparency,
not because the prior count was wrong.

---

## 2. Match 3845506 — load failure

### 2.1 Root cause (definitively established)

```
$ .venv/bin/python3 -c "import json; json.load(open('data/raw/statsbomb/data/three-sixty/3845506.json'))"
json.decoder.JSONDecodeError: Expecting ',' delimiter: line 92794 column 3 (char 2637824)
```

- `data/raw/statsbomb/data/events/3845506.json` loads **successfully**
  (3,095 valid event records).
- `data/raw/statsbomb/data/three-sixty/3845506.json` fails with a
  `JSONDecodeError` at a specific byte offset.
- Inspecting the raw bytes at that offset: the text `"actor" : false,`
  is interrupted by **exactly 16,384 (2¹⁴) literal NUL (`\x00`) bytes**
  inserted mid-token, splitting it into `"actor" : fa` + 16,384 NULs +
  `lse,`. This is not a schema problem, not an encoding issue, and not a
  JSON-syntax choice by StatsBomb — it is a classic signature of a
  **single corrupted/zero-filled storage block** (16,384 bytes = one
  common filesystem/page block size), consistent with an interrupted
  write, disk-level corruption, or a sparse-file hole introduced when this
  file was produced or copied onto disk.

### 2.2 Is this isolated or systemic?

Scanned **every** raw JSON file in the project for embedded NUL bytes
(illegitimate in valid StatsBomb JSON/UTF-8 text):

- `data/raw/statsbomb/data/events/`: 4,235 files scanned, **0** contain
  NUL bytes.
- `data/raw/statsbomb/data/three-sixty/`: 426 files scanned, **exactly 1**
  (`3845506.json`, 16,384 NUL bytes) contains any.

**This corruption is completely isolated to one file.** It is not a
dataset-wide or systemic issue.

### 2.3 Is the loader's error handling adequate?

- `src/animate_statsbomb_360_match.py::load_json_records` already handles
  this well: it catches `JSONDecodeError`/`OSError` specifically and
  re-raises a `VisualizationError` naming the match's file path and label
  — not a bare `except Exception`.
- `src/build_statsbomb_360_dataset.py::load_json` has no
  try/except at all — a `JSONDecodeError` simply propagates with Python's
  own message (which includes the char offset but not, by itself, the
  match id or which of the two files failed, unless read from the
  surrounding traceback).
- **The one place that *did* swallow useful detail was my own**
  `scripts/audit_dataset.py` from the first-stage audit: it used a bare
  `except Exception: continue`, recording only the match id. **Fixed in
  this investigation** (§6): failures are now captured per-file with the
  exact exception type, message, and path.

### 2.4 Can it be fixed, and is it safe to recover?

**No safe recovery is possible.** The corrupted 16,384-byte block sits in
the *middle* of one `freeze_frame` array's player-location records,
overwriting an unknown number of original player positions with zero
bytes. There is no way to reconstruct the lost content without inventing
data, which violates this project's explicit "never invent" principle.
Truncating or skipping the corrupted region would require guessing array
boundaries with no guarantee of producing valid or meaningful JSON, and
would silently fabricate structure.

**Conclusion**: match 3845506's three-sixty file is genuinely
unparseable raw source data (a one-off, localized corruption, not a
dataset-wide problem, not a code defect). It **must remain excluded**
from all three-sixty-dependent processing. Its `events.json` alone is
intact and could be used for event-only (non-360) analysis if the project
ever needs that, but no such pipeline currently exists. The audit (both
versions) explicitly records this match's id, the failing file, and the
exact exception rather than silently dropping it — satisfying "never
silently skip failed matches."

---

## 3. Multiple-actor and zero-actor frames

### 3.1 Zero-actor frames (2 total, confirmed at full dataset scale)

Both traced to real, located, correctly-typed events:

| Match | Event | Type | Team | Location |
|---|---|---|---|---|
| 3893793 | e43bc7e0 | 50/50 | Japan Women's | `[32.5, 44.2]` |
| 4018356 | dfe42790 | Dispossessed | Switzerland Women's | `[45.1, 2.9]` |

Neither is an orphaned 360 record (both events exist in `events.json` with
full metadata) and neither is caused by an event/frame mismatch (the
`event_uuid` join is exact). The freeze_frame arrays for these two frames
simply contain no point flagged `actor=True`. Given StatsBomb 360 only
tracks *visible* players, the most plausible explanation is that the
acting player was outside the camera's tracked field of view at that
exact instant (a 50/50 and a Dispossessed event both involve a player who
may be partially obscured or at the edge of the frame). **This is valid,
expected "missing actor information," not a defect** — and the pipeline
now represents it as such explicitly (`actor_x=None, actor_y=None,
actor_ambiguous=False` — a distinct signal from "ambiguous").

A 40-match sample earlier in this investigation found 0 such cases,
illustrating why the full-dataset rerun (finding exactly 2) was necessary
rather than trusting a sample.

### 3.2 Multiple-actor frames (28 total, confirmed at full dataset scale)

Traced the pairwise distance between the actor-flagged points in all 28
cases:

| Max distance between actor points | Count |
|---|---|
| ≤0.01 (floating-point noise) | 21 |
| 0.01–0.5 | 5 |
| 0.5–1.0 | 0 |
| **~6.77 (one match, both its events)** | **2** |

**26 / 28 cases (92.9%) are effectively the same physical point** recorded
twice (byte-identical or sub-unit-apart coordinates) with `actor=True` on
both entries — a raw-data duplication quirk, not two distinct players.
Roughly half of these 28 cases belong to a Problem-C shared cross-team
snapshot group (confirmed via `detect_shared_cross_team_snapshots`), and
half do not — the duplicate-actor phenomenon is independent of, and not
caused by, the shared-snapshot phenomenon or its resolution logic; it is
present in the raw `freeze_frame` array itself, before any of our code
touches it.

**Exactly 2 cases (match 3802837, events `0a731280` and `17380117`, both
referencing the same instant)** have actor points **~6.8 pitch units
apart** — genuinely two different, non-trivially-separated positions. No
schema or event-association evidence resolves which one is the "real"
actor.

**Fix implemented** (§5): `build_statsbomb_360_dataset.py` previously used
`next(...)` to silently pick whichever actor-flagged point came first in
the array — for the 2 genuinely ambiguous cases, this was an unprincipled,
arbitrary choice presented as if it were certain. Now: points within
`ACTOR_DUPLICATE_TOLERANCE_UNITS = 1.0` of each other are treated as the
same position (safe, since they already agree to far less than a player's
own body size); points further apart are marked `actor_ambiguous=True`
with `actor_x=actor_y=None`, never guessed.

---

## 4. Orientation classification re-verification

No change to the orientation algorithm was made or is justified by this
investigation (re-confirmed, not merely re-asserted):

- **Three distinct, independently-verified concepts, never conflated**:
  (1) StatsBomb's original per-team-relative raw coordinates; (2) the
  classifier's per-event self-consistency estimate (NORMAL/ROTATED_180/
  UNCERTAIN); (3) the separate, period-independent team-coordinate-scale
  normalization (Problem B). Confirmed these remain architecturally
  distinct by re-reading `orientation_correction.py` and
  `team_normalization.py`'s own module docstrings and by the fact their
  test suites remain fully independent (`test_problem_a_classifier_and_correction_are_unaffected`,
  `test_problem_b_team_normalization_unaffected_by_problem_c`).
  Half-time attacking-direction changes are **not** a coordinate-system
  change requiring correction — this was independently re-verified with
  fresh period-stratified shot-location evidence in the first-stage audit
  and is not re-litigated here.
- **UNCERTAIN handling**: confirmed via code reading
  (`apply_correction`'s explicit `if orientation != ORIENTATION_ROTATED_180: return dict(frame_360)`)
  that UNCERTAIN frames are passed through **unchanged** by both the
  dataset builder and the renderer — never silently dropped, transformed,
  or guessed. `orientation_correction=UNCERTAIN` rows remain in the CSV
  output with their raw coordinates and an explicit `orientation` column
  value, so a downstream consumer can filter or weight them as needed.
- **Determinism**: `classify_orientation`/`apply_correction` are pure
  functions of their inputs (no randomness, no mutable global state);
  `test_build_rows_is_deterministic_for_identical_input` directly confirms
  the full pipeline (including orientation) is deterministic.
- **Zero or double transformation**: confirmed by code reading that
  orientation correction (Problem A) is applied exactly once per
  event/frame pair (keyed by `orientation_by_event_id`, computed once
  per match) and team normalization (Problem B) is applied exactly once,
  always in the fixed order A-then-B, both in the dataset builder and the
  renderer.
- **Event-location vs. freeze-frame consistency**: re-confirmed (not
  re-derived; established with a full 1,357,624-row mathematical proof in
  a prior session) that `event.location` deliberately does **not**
  receive the Problem A mirror — doing so would provably undo the
  correction for every ROTATED_180 case. No new evidence in this
  investigation changes that conclusion.

No change to orientation classification thresholds, votes, or decision
logic was made. The distribution (NORMAL 1,274,820 / ROTATED_180 77,816 /
UNCERTAIN 4,991, recomputed in this session, consistent with the
first-stage audit to the single-digit level after match 3845506's
exclusion is held constant) is **not** something this investigation sought
to reduce — per the explicit instruction not to tune the algorithm merely
to shrink the ROTATED_180/UNCERTAIN counts.

---

## 5. Code changes made in this investigation

### `src/build_statsbomb_360_dataset.py`
- Added `ACTOR_DUPLICATE_TOLERANCE_UNITS = 1.0` and a new function
  `resolve_actor_location(freeze_frame)` implementing the policy in §3.2:
  0 actor points → `(None, None, False)`; 1 → that point; 2+ within
  tolerance → the first (safe, same position); 2+ beyond tolerance →
  `(None, None, True)`, never guessed.
- Added a new CSV column, `actor_ambiguous` (bool), so a genuinely
  ambiguous actor position is explicitly distinguishable from a merely
  missing one — both now read `actor_x=actor_y=None`, but only the former
  sets `actor_ambiguous=True`.

### `scripts/audit_dataset.py`
See §6 for the full list of audit-logic improvements.

### `tests/test_orientation_correction.py`
- Added `test_mirror_preserves_out_of_bounds_status_exactly` (§1.3).

### `tests/test_build_statsbomb_360_dataset.py`
- Added 7 new tests for `resolve_actor_location` and its integration into
  `build_rows_from_match_data` (zero/one/near-duplicate/genuinely-ambiguous
  actor points; `actor_ambiguous` column correctness in both directions).

### `tests/test_audit_dataset.py` (new file)
- 8 new tests for the audit script's own logic: coordinate
  classification, overshoot magnitude/bucketing, diagnostic-preserving
  failure reporting, read-only guarantee (never writes into its input
  directories), and output-overwrite protection (refuses by default,
  honors `--overwrite`).

No change was made to `src/orientation_correction.py`,
`src/team_normalization.py`, `src/team_identity.py`, or
`src/animate_statsbomb_360_match.py` in this investigation — all three
were re-verified, not modified.

---

## 6. Audit script improvements (`scripts/audit_dataset.py`)

- **Diagnostic-preserving failure reporting**: `failed_matches` is now a
  list of `{match_id, file, error_type, error_message}` records instead of
  a bare id list, so a future corrupted file is immediately identifiable
  without re-running a diagnostic script by hand.
- **Missing vs. malformed vs. out-of-bounds, kept separate**:
  `classify_coordinate` returns one of `missing` / `malformed` /
  `non_numeric` / `in_bounds` / a specific combination of `x<0`/`x>120`/
  `y<0`/`y>80` — these are never folded into a single "invalid" bucket.
- **Category and magnitude breakdowns** for out-of-bounds points, plus
  unique affected match/event counts (not just a raw point count).
- **Missing-location/missing-player counts broken down by event type**, so
  it is immediately visible (without a follow-up script) that these are
  concentrated entirely in non-spatial administrative event types.
- **Bounded example lists** (`zero_actor_examples`, `multi_actor_examples`,
  up to 25 each) included directly in the JSON report.
- **Overwrite protection**: `main()` now refuses to overwrite an existing
  `--output` path unless `--overwrite` is passed.
- **Optional, explicit, read-only cross-check** (`--compare-csv` /
  `--compare-match-id`): recomputes a raw-derived upper bound on row count
  for one match and compares it against an existing processed CSV, never
  writing to the CSV.
- All of the above are covered by `tests/test_audit_dataset.py`.

---

## 7. Limitations of this investigation

- The visualization spot-check (§1.4) covered one match/window directly
  and cross-referenced the renderer's own structural-report output; it did
  not re-render every one of the 159,238 affected (match, event) pairs —
  infeasible, and unnecessary given the isometry proof in §1.3 establishes
  the general case independent of any specific example.
- No new automated test was added directly against
  `src/animate_statsbomb_360_match.py` in this investigation (same gap
  disclosed in the first-stage audit); the spot-check here is manual/
  visual plus a direct comparison against the production CLI's own
  diagnostic output, not a pytest regression test.
- The 89 extreme-overshoot points' "non-participant near the touchline"
  explanation is the most plausible account given the evidence (actor/
  keeper flags, magnitude, category) but could not be independently
  confirmed against broadcast footage or StatsBomb's own internal
  documentation — flagged as a plausible, evidence-backed conclusion, not
  an absolutely certain one.
