# StatsBomb 360 Data Quality — Summary for Supervisor Discussion

Date: 2026-10-13. Prepared from `scripts/audit_dataset.py`'s full-dataset
run (`results/data_quality_audit_2026-10-13d.json`, the current, final
report — see note on report versions below) and the three preceding
investigation documents in `docs/`.

This document summarizes the data-quality investigation of the StatsBomb
Open Data event + 360 freeze-frame pipeline underlying this thesis. It
validates **data processing correctness**, not predictive model
performance — no model has been trained or evaluated as part of this
investigation.

---

## A. Dataset overview

| Metric | Value |
|---|---|
| Matches discovered (have a three-sixty file) | 426 |
| Matches explicitly excluded (documented, see §B) | 2 |
| Matches processed successfully | 424 |
| Matches unexpectedly failed | 0 |
| Event records (raw, across the 424 processed matches) | 1,578,884 |
| Three-sixty (360 freeze-frame) records | 1,381,466 |
| Matched event/frame pairs | 1,357,626 |
| Events without 360 coverage | 221,258 (14.0%) |

**Coverage limitation**: StatsBomb 360 does not cover every event type by
design (e.g. administrative events such as Half Start/End, Substitution,
Starting XI never carry a 360 snapshot). The 14.0% uncovered figure is
expected, not a defect — confirmed by event-type breakdown: of the
13,267 events missing a `location` field entirely, 100% are non-spatial
administrative types (Substitution, Injury Stoppage, Half Start/End,
Tactical Shift, Starting XI, Player On/Off, Bad Behaviour). **Zero** of the
8 event types actually used as prediction targets (`Pass`, `Carry`,
`Dribble`, `Shot`, `Miscontrol`, `Ball Recovery`, `Clearance`,
`Dispossessed`) are missing location or player information anywhere in
the dataset.

### Note on report versions

Two audit runs exist in `results/`: `...2026-10-13c.json` (425 matches
"loaded OK", 1 "failed") predates the exclusion system; `...2026-10-13d.json`
(424 "loaded OK", 2 "excluded", 0 "failed", the figures used throughout
this document) was produced after implementing it today. The entire
numerical difference between the two runs is accounted for by match
3923880 moving from "counted normally" to "explicitly excluded" (it
previously contributed 3,374 event records, 1 three-sixty record, and 1
matched pair to the aggregate totals; `event_records` drops from
1,582,258 to 1,578,884 — exactly 3,374 — and the other deltas match
exactly). No other figure changed between the two runs.

---

## B. Data integrity and preprocessing

### Explicit match exclusions (new today)

A central, version-controlled exclusion manifest
(`configs/excluded_matches.json`) now documents exactly two matches as
permanently excluded from processing, each with independently-verifiable
evidence:

1. **Match 3845506** — the three-sixty JSON file is corrupted: a single,
   isolated 16,384-byte run of NUL bytes overwrites real content inside
   one freeze-frame record (`JSONDecodeError` at a specific byte offset).
   Confirmed unique to this one file across all 4,661 raw JSON files in
   the project. No safe recovery is possible without inventing the lost
   content.
2. **Match 3923880** — the three-sixty file is valid JSON but contains
   only 1 record for a 3,374-event match (0.03% coverage, vs. a typical
   match's ~85%), covering only one of the two teams. The project's team
   coordinate-normalization step requires at least one 360-covered event
   from each team to establish a reference; with only one team
   represented, this raises a reproducible `ValueError`. This is a
   genuine processing failure caused by sparse source data, confirmed by
   actually running the pipeline — **not** an exclusion based on file size
   alone (several other matches have smaller-than-typical but still
   usable 360 files and were kept).

Both the dataset builder (`src/build_statsbomb_360_dataset.py`) and the
audit script now check this manifest before touching either match's raw
files, and report exclusions in a category distinct from unexpected
failures. The manifest fails loudly (refuses to run) if it is missing,
malformed, or contains an invalid entry, rather than silently proceeding
as if there were no exclusions.

### Actor-position ambiguity and missing actor positions

- **28 freeze-frames (0.002% of 1.38M)** across 11 matches contain two
  points flagged as the event actor instead of one. 26 are effectively
  duplicate recordings of the same physical position (≤0.5 pitch units
  apart); exactly 2 (both in one match) are genuinely ~6.8 units apart
  with no way to determine the correct one from the data alone. The
  dataset builder resolves near-duplicates automatically and marks
  genuine ambiguity explicitly (`actor_ambiguous = True`, position left
  as missing) rather than guessing.
- **2 freeze-frames** have no actor point at all. Root cause verified
  directly from raw data: in both cases, literally every visible player
  in the snapshot belongs to the opposing team (0 teammates visible), so
  no actor can be flagged — a direct, logical consequence of StatsBomb's
  own schema, not a data error.

### Duplicate actor points and downstream effect

For the 2 matches with exact-duplicate actor points, the duplicate is
counted twice in the `visible_players`/`visible_teammates` columns,
inflating the visible-player count by exactly 1 for those specific rows.
Affects at most a handful of rows dataset-wide (bounded by the 28 known
cases); documented and covered by a regression test, not corrected,
since any deduplication rule risks incorrectly merging two distinct
nearby players in other (undetected) cases.

### Out-of-bounds coordinates

189,128 of 21,561,932 freeze-frame points (0.88%) fall outside the
nominal pitch rectangle `[0,120]×[0,80]`. Evidence indicates this is
genuine source data, not a processing artifact:
- 87% of violations are on the touchline (Y) axis rather than the
  goal-line (X) axis — consistent with real sideline/throw-in/corner
  play, not a coordinate-system error.
- 62% overshoot by ≤2 pitch units (negligible, consistent with
  measurement tolerance at the boundary).
- Affects 424/424 processed matches in small per-match proportions —
  the signature of organic tracking behavior, not a bug localized to
  specific matches.
- **Proven, not assumed**: the orientation-correction and
  team-normalization transforms are 180° rotations about the pitch
  center (mathematical isometries), so they cannot create, remove, or
  hide an out-of-bounds point — only reflect it by an identical margin.
  Verified both by direct proof and by an automated regression test.

**Methodological recommendation** (not yet decided): these points should
most likely be **retained**, not clipped or deleted — clipping would
silently distort real player positions near the boundary. If a model
requires strictly in-bounds input, clamping to the nearest valid
coordinate (not deletion) is the safer option, but this has not been
implemented and is listed as an open question (§D).

### Orientation correction and team normalization — validation

- 93.90% of matched event/frame pairs are self-consistent as recorded
  (NORMAL); 5.73% require a 180° correction (ROTATED_180), applied
  automatically; 0.37% have insufficient evidence and are deliberately
  left uncorrected (UNCERTAIN) rather than guessed.
- Verified this investigation, with exact-coordinate tests and a
  500-real-event empirical check: the correction pipeline **exactly**
  preserves all pairwise distances between players (0 violations) and is
  applied **exactly once** per event by the production code (confirmed by
  code inspection of both consumers), even though the underlying
  transform function is not itself idempotent.
- Half-time attacking-direction changes are already absorbed into
  StatsBomb's own raw coordinate convention and require no additional
  correction — independently re-verified across a 40-match sample with
  zero exceptions (prior investigation session).

### Test results and reproducibility

- **136 automated tests, 136 passing, 0 failing, 0 skipped** (synthetic,
  deterministic fixtures; no network access; runs in under 1.5 seconds),
  including new regression tests added today for the exclusion manifest
  (16 tests), the real zero-actor case, duplicate-point count inflation,
  and transform distance-preservation/non-idempotence.
- The full-dataset audit was run twice independently; **every one of the
  32 reported metrics was byte-for-byte identical** between the two runs
  (prior to today's exclusion-system change), confirming the audit itself
  is deterministic and reproducible.

---

## C. Methodological implications

**Issues that affect model features/labels directly** (require explicit
handling before training): `actor_ambiguous = True` rows (≤30 known
cases); rows derived from the 2 excluded matches (already removed from
any CSV built after today).

**Issues that are data-quality limitations, not feature/label
corruptions**: out-of-bounds coordinates (real positions, not errors);
the 221,258 events without 360 coverage (never enter the pipeline at all,
by construction); `UNCERTAIN`-orientation rows (left in raw, uncorrected
form, clearly labeled by the `orientation` column).

**Verified findings** (established directly from raw data/code, stated
as fact): the two excluded matches' root causes; the zero/multi-actor
counts and root causes; distance preservation; reproducibility of the
audit.

**Implemented fixes**: actor-ambiguity resolution with explicit
`actor_ambiguous` flag; the exclusion manifest and its pipeline
integration; several diagnostic/error-reporting improvements.

**Known residual limitations** (documented, not fixed): the orientation
classifier itself has a latent (empirically harmless on all 28 known
cases) inconsistency in how it picks an actor candidate when multiple
exist; duplicate-point count inflation (§B); no automated test suite
exists yet for the visualization module itself.

No claim is made about model accuracy, generalization, or predictive
validity — this work concerns only whether the input data is correctly
parsed, joined, transformed, and represented.

---

## D. Questions for the supervisor

1. Should `event.location` (a second, independently-recorded StatsBomb
   field) be used as a tie-breaker for the rare genuinely-ambiguous actor
   position cases (currently: left as missing, never guessed)?
2. How should the small number of rows with ambiguous or missing actor
   positions be handled in the modeling pipeline — excluded, imputed, or
   kept with an explicit missingness indicator feature?
3. Should out-of-bounds coordinates be retained as-is, clamped to the
   pitch boundary, or treated as a feature in their own right (e.g.
   "near-touchline" indicator) in the experimental pipeline?
