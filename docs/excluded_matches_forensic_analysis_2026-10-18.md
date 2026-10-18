# Forensic Analysis: 7 Matches Excluded for `three_sixty_event_uuid_mismatch`

Date: 2026-10-18. Investigation only — no raw data, processed datasets, or
`configs/excluded_matches.json` were modified. All findings below are
reproducible via `scripts/diagnose_excluded_match_mismatch.py` (new,
read-only) and the commands listed in each section.

---

## Executive summary

On 2026-10-14, building the full next-event dataset in memory (for the
prediction-task feasibility study) surfaced 7 matches that raised
`ValueError: Expected exactly two team ids in events, found [].` This
investigation traces the exact cause.

**Verified fact**: for all 7 matches, `events.json` and `three-sixty.json`
are both well-formed, normally-sized, internally self-consistent JSON
files — **but they share zero event identifiers in common.** This was
confirmed exhaustively, not merely re-stated: the orphan identifiers in
each match's `three-sixty.json` were searched against **all 4,235**
locally available `events.json` files (14,874,171 unique event ids) and
**all 426** locally available `three-sixty.json` files (1,381,467 unique
`event_uuid`s), including cross-checks among the 7 affected matches
themselves. **In every case, the search found nothing.**

**Verified fact**: this pattern affects exactly these 7 matches and *no
others*. Every one of the other 417 usable matches shows **exactly
100.00% overlap** between its two files — not 99%, not a partial
contamination, but a clean binary split (either entirely correct or
entirely mismatched).

**Root cause classification**: **Category A — source-data mismatch**,
with **high confidence** that the problem is **not** in this project's
code (the path-mapping logic is a trivial, universally-applied filename
convention, correct for 417/424 other matches) and **not** a corruption
of the bytes on disk (both files parse cleanly and contain realistic,
well-structured football data). The evidence narrows the origin to either
the original StatsBomb open-data release itself, or whatever process
populated this local copy of it — **this cannot be distinguished further
without access to the live upstream source, which this investigation did
not have** (no network access). This is stated as an explicit, unresolved
limitation, not a guess.

**Recovery status**: **Not currently recoverable** for all 7 — no
evidence-backed correspondence between the orphan `three-sixty.json`
content and any locally-available match could be established. Fabricating
one is explicitly out of scope per this investigation's constraints.

**All 7 exclusions are confirmed justified** by this independent
re-investigation; none should be reinstated.

---

## 1. The seven matches — identification and metadata

Source: `configs/excluded_matches.json` (exact list, not reconstructed
from memory) cross-referenced against
`data/raw/statsbomb/data/matches/<competition_id>/<season_id>.json`.

| Match ID | Home | Away | Competition | Season | Date | Metadata file |
|---|---|---|---|---|---|---|
| 3837706 | Paris Saint-Germain | Stade Brestois | Ligue 1 | 2022/2023 | 2022-09-10 | `matches/7/235.json` |
| 3877115 | Inter Miami | Toronto FC | Major League Soccer | 2023 | 2023-09-21 | `matches/44/107.json` |
| 3877170 | Inter Miami | Cincinnati | Major League Soccer | 2023 | 2023-10-08 | `matches/44/107.json` |
| 3877194 | Charlotte | Inter Miami | Major League Soccer | 2023 | 2023-10-22 | `matches/44/107.json` |
| 3895158 | Bayer Leverkusen | Borussia Dortmund | 1. Bundesliga | 2023/2024 | 2023-12-03 | `matches/9/281.json` |
| 3895266 | Bayer Leverkusen | Wolfsburg | 1. Bundesliga | 2023/2024 | 2024-03-10 | `matches/9/281.json` |
| 3895309 | Borussia Dortmund | Bayer Leverkusen | 1. Bundesliga | 2023/2024 | 2024-04-21 | `matches/9/281.json` |

File paths and status (all verified to exist and parse without error):

| Match ID | `events.json` path | `three-sixty.json` path | Both exist & parse | Excluded in manifest | Reason recorded |
|---|---|---|---|---|---|
| 3837706 | `data/raw/statsbomb/data/events/3837706.json` | `data/raw/statsbomb/data/three-sixty/3837706.json` | Yes | Yes | `three_sixty_event_uuid_mismatch` |
| 3877115 | `data/raw/statsbomb/data/events/3877115.json` | `data/raw/statsbomb/data/three-sixty/3877115.json` | Yes | Yes | `three_sixty_event_uuid_mismatch` |
| 3877170 | `data/raw/statsbomb/data/events/3877170.json` | `data/raw/statsbomb/data/three-sixty/3877170.json` | Yes | Yes | `three_sixty_event_uuid_mismatch` |
| 3877194 | `data/raw/statsbomb/data/events/3877194.json` | `data/raw/statsbomb/data/three-sixty/3877194.json` | Yes | Yes | `three_sixty_event_uuid_mismatch` |
| 3895158 | `data/raw/statsbomb/data/events/3895158.json` | `data/raw/statsbomb/data/three-sixty/3895158.json` | Yes | Yes | `three_sixty_event_uuid_mismatch` |
| 3895266 | `data/raw/statsbomb/data/events/3895266.json` | `data/raw/statsbomb/data/three-sixty/3895266.json` | Yes | Yes | `three_sixty_event_uuid_mismatch` |
| 3895309 | `data/raw/statsbomb/data/events/3895309.json` | `data/raw/statsbomb/data/three-sixty/3895309.json` | Yes | Yes | `three_sixty_event_uuid_mismatch` |

**Note**: competition clustering is visible — 1 Ligue 1 match, 3 MLS matches
(all involving Inter Miami's 2023 season, the Messi-signing season), 3
Bundesliga matches (all involving Bayer Leverkusen's 2023/24 unbeaten
season). This clustering is discussed in §4.

---

## 2. UUID overlap investigation (per match) — exact statistics

**What is compared, and why it is valid**: `events.json` is a flat array
of event objects, each with a top-level `id` field (a UUID string
identifying that specific event). `three-sixty.json` is a flat array of
freeze-frame *records*, each with a top-level `event_uuid` field that is
documented (and, across the other 417 working matches, empirically
confirmed) to reference the `id` of the event it was captured for. These
are the correct fields to compare — `event["id"]` against
`frame["event_uuid"]` — this is exactly what
`src/build_statsbomb_360_dataset.py` and `src/animate_statsbomb_360_match.py`
already do to join the two files, and what the first-stage data-quality
audit (2026-10-13) used. No event *count* is compared against a frame
*count* as a proxy for correctness anywhere in this analysis — only
identifier sets.

| Match ID | Event records | Frame records | Unique event ids | Unique frame uuids | Event id dupes | Frame uuid dupes | Null ids (either file) | Malformed UUID format | Overlap | Case-insensitive overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| 3837706 | 4,026 | 3,548 | 4,026 | 3,548 | 0 | 0 | 0 | 0 | **0** | 0 |
| 3877115 | 4,138 | 3,808 | 4,138 | 3,808 | 0 | 0 | 0 | 0 | **0** | 0 |
| 3877170 | 3,467 | 3,097 | 3,467 | 3,097 | 0 | 0 | 0 | 0 | **0** | 0 |
| 3877194 | 3,552 | 3,174 | 3,552 | 3,174 | 0 | 0 | 0 | 0 | **0** | 0 |
| 3895158 | 3,866 | 3,318 | 3,866 | 3,318 | 0 | 0 | 0 | 0 | **0** | 0 |
| 3895266 | 4,104 | 3,615 | 4,104 | 3,615 | 0 | 0 | 0 | 0 | **0** | 0 |
| 3895309 | 3,890 | 3,280 | 3,890 | 3,280 | 0 | 0 | 0 | 0 | **0** | 0 |

"Only in events" = 100% of `unique event ids` for every row (every single
event id is unmatched); "only in frames" = 100% of `unique frame uuids`
for every row (every single frame uuid is unmatched) — the overlap column
being exactly 0 means these two columns are simply the full counts, so
they are omitted from the table for readability.

**Format checks** (all 7 matches, identical result): 0 null UUIDs in
either file; 0 malformed UUID-format strings (all values match the
standard 8-4-4-4-12 hexadecimal UUID pattern); 0 leading/trailing
whitespace; checking overlap after lower-casing both sets changes nothing
(0 → 0) — this rules out a trivial case-sensitivity or whitespace
formatting bug as the explanation.

**Coverage-ratio sanity check** (frame records ÷ event records — informal,
only meaningful if the pairing were correct, but useful to characterize
whether the "orphan" data *looks like* genuine 360 coverage for some real
match):

| Match ID | Ratio |
|---|---|
| 3837706 | 88.1% |
| 3877115 | 92.0% |
| 3877170 | 89.3% |
| 3877194 | 89.4% |
| 3895158 | 85.8% |
| 3895266 | 88.1% |
| 3895309 | 84.3% |

All 7 ratios fall squarely within the normal range observed across the
417 working matches (typically ~85–92%), reinforcing that each
`three-sixty.json` file is very likely **complete, genuine 360 coverage
data for some real match** — not truncated, not garbled, not randomly
generated — just not *this* match.

Reproduce this table: `.venv/bin/python scripts/diagnose_excluded_match_mismatch.py`

---

## 3. Case studies — concrete evidence per match

Each case study below is representative of an identical pattern; sample
UUIDs are shown so the claim is independently checkable, not merely
asserted.

### 3837706 — Paris Saint-Germain vs Stade Brestois

- First 5 `events.json` `id` values: `8f7ce50b-03c9-4c07-952d-88ff2095824b`, `8bcf7056-2916-4b05-bdbb-80aa9e7e20cc`, `34e6e895-7af3-48fb-8077-18e8196de1d4`, `621b8a16-5fe8-46ee-9ae2-334953e147c8`, `2e1a33fc-0de6-446e-b989-a31ab27561b8`
- First 5 `three-sixty.json` `event_uuid` values: `988ae44f-1ad4-44b6-8ae7-f58a29787e4e`, `09c6fa28-b05d-4cec-ad6c-143d353ff3c2`, `aa141669-5a08-44f2-91cf-c2d8a45a444f`, `29acf834-08ec-47f2-ae6e-3a278d19ea35`, `6e279f94-76e4-497e-a58b-4b73f89b0473`
- **No correspondence could be established between these two lists, or
  between any of the 4,026 vs. 3,548 full identifier sets.** This is
  stated explicitly rather than presenting them as if they should align.
- `events.json` internal consistency check: the two team names appearing
  inside the event records themselves are exactly `{"Paris
  Saint-Germain", "Stade Brestois"}`, matching the official match
  metadata exactly, and the event stream begins with the expected
  `Starting XI` → `Starting XI` → `Half Start` sequence. This confirms
  `events.json` is almost certainly the **correct, undamaged** file for
  this match — the problem is isolated to `three-sixty.json`.
- `three-sixty.json` first record (field structure, for reference):
  ```json
  {
    "event_uuid": "988ae44f-1ad4-44b6-8ae7-f58a29787e4e",
    "visible_area": [78.6, 80.0, 41.2, 80.0, 14.0, 0.0, ...],
    "freeze_frame": [
      {"teammate": true, "actor": false, "keeper": false, "location": [36.4, 42.2]},
      ...
    ]
  }
  ```
  — structurally identical to a normal, working match's 360 record;
  coordinates fall within the expected pitch range; realistic
  teammate/opponent/keeper distribution.
- Diagnosis: **Category A (source-data mismatch)**. Confidence: **High**
  that `three-sixty.json`'s content belongs to a different match than
  `3837706`; **cannot determine which match** (see §5).
- Recovery status: **Not currently recoverable.**

### 3877115 / 3877170 / 3877194 — Inter Miami 2023 MLS trio

All three share the identical signature (0 overlap, well-formed data on
both sides, correct team names inside `events.json`):

- 3877115 (Inter Miami vs Toronto FC): sample event ids `a1004642-1b04-42c6-93ce-362b3f9dcc7f`, `c792ec58-0657-42f2-8c27-a391def7aef1`, `4ac5b5c0-26c7-4025-ae4c-f250e43deffa`; sample frame uuids `8f60c1f7-73ce-4127-89d4-5f7d7e222c9d`, `76c4e9f4-e38e-4f52-bc82-f66ad055041d`, `a4bbd323-95a9-467f-abaf-f198edeb9018`.
- 3877170 (Inter Miami vs Cincinnati): sample event ids `c26b5110-5e89-4697-b5bc-5025ec0bd314`, `97f05673-3389-4328-a9c5-d8927722cf44`; sample frame uuids `9df7c933-635e-4c90-b74f-037879bd7411`, `da87977e-51a5-4305-b2a2-07254ca2137c`.
- 3877194 (Charlotte vs Inter Miami): sample event ids `028bd27d-9bbe-423f-86a1-6f236a2d0e2a`, `feecd9b3-0468-4581-99ff-344ca138f30c`; sample frame uuids `e12a2894-11fa-4539-96af-1dc27e7bd95e`, `47488b53-1220-48e3-87b2-2730d9da9075`.
- **Explicitly tested and rejected hypothesis**: that one of these three
  matches' `three-sixty.json` actually belongs to one of the *other two*
  (a within-batch swap, since all three are part of the same MLS
  2023-season Inter Miami storyline and were likely downloaded/packaged
  together). A full pairwise cross-check among all 7 affected matches
  (49 combinations) found **zero overlap in every single combination**,
  including among these three. This specific, plausible swap hypothesis
  is **ruled out by direct evidence**, not merely assumed false.
- Diagnosis: **Category A**, same confidence level as 3837706.
- Recovery status: **Not currently recoverable**, all three.

### 3895158 / 3895266 / 3895309 — Bayer Leverkusen 2023/24 Bundesliga trio

- 3895158 (Bayer Leverkusen vs Borussia Dortmund): sample event ids `2d7cfccc-f05a-4ac4-b0c1-7c54cc85ed0c`, `2eb0f67a-fb3f-4a5d-813e-d6043315e72a`; sample frame uuids `122b21d1-8786-4031-922d-3d0c1fd6fed2`, `aa1f74f4-b5a3-4fa8-9032-9f4fad7fff75`.
- 3895266 (Bayer Leverkusen vs Wolfsburg): sample event ids `72380f9a-26d2-45c5-9143-2dc38214a4b9`, `d25395f2-1a79-48f9-bb53-67bca1187155`; sample frame uuids `298abfbd-4eec-48b1-8afc-c050a244b51d`, `4d70dff4-3078-4410-9b93-b575b44daee7`.
- 3895309 (Borussia Dortmund vs Bayer Leverkusen): sample event ids `3390c31e-effc-47fd-800d-bb0fb7353905`, `4c8c90f8-fbcc-4e6a-b15d-3ad9b958e3a6`; sample frame uuids `32b27a5f-fa6a-4b71-ba2a-8fd6661d035f`, `7ce7493e-d0d0-4a1e-97b7-7fc84c98149d`.
- **A second, independently-motivated hypothesis was tested and
  rejected**: the local dataset also contains three *older* matches with
  the **exact same team pairings** from an earlier Bundesliga package
  (season 2015/16): match `3890456` (Bayer Leverkusen vs Borussia
  Dortmund, 2016-02-21), `3890502` (Bayer Leverkusen vs Wolfsburg,
  2016-04-01), and `3890301` (Borussia Dortmund vs Bayer Leverkusen,
  2015-09-20). These three older matches have `events.json` files locally
  but **no** `three-sixty.json` (360 tracking did not exist for matches
  from that era) — making them superficially plausible "donors" if a
  season/fixture-based indexing bug had occurred upstream. **Directly
  tested**: compared each 2023/24 match's orphan `three-sixty.json` uuids
  against the corresponding same-fixture 2015/16 match's `events.json`
  ids. Result: **0/3,318, 0/3,615, and 0/3,280 overlap respectively —
  this hypothesis is also rejected by direct evidence.**
- Diagnosis: **Category A**, same confidence level.
- Recovery status: **Not currently recoverable**, all three.

---

## 4. Cross-match analysis — common root cause

All 7 matches share an **identical signature**:

1. Both `events.json` and `three-sixty.json` are valid, well-formed JSON
   of normal size for a match of this length (no truncation, no
   corruption markers — unlike the separately-diagnosed match 3845506,
   which has a literal byte-level NUL corruption; see
   `docs/data_quality_investigation_2026-10-13.md`).
2. `events.json` is internally self-consistent with the correct, official
   match metadata (team names, Starting XI sequence) in every case — the
   problem is isolated to the `three-sixty.json` side.
3. `three-sixty.json` contains structurally normal, realistic 360
   tracking data (valid coordinate ranges, plausible coverage ratio) —
   not garbage or synthetic noise.
4. The overlap between the two files' identifiers is **exactly zero**,
   with no partial/near-miss correspondence (no case-sensitivity,
   whitespace, or format issue found).
5. **File system timestamps are identical across all 7 affected matches
   and all 417 working matches** (`May 26 2026 16:59:55` for every single
   raw file checked, problem or not) — all files arrived in this local
   copy as part of one single bulk operation. There is **no time-based
   clustering** that would distinguish a "bad batch" from a "good batch"
   within this project's own acquisition history.
6. Two specific, independently plausible "swap" hypotheses were formed
   from the competition/season clustering pattern and **both were
   directly tested and rejected** (§3): no swap among the 3 MLS matches,
   no swap with the same-fixture 2015/16 Bundesliga matches.

**Conclusion**: this is a single, consistent root-cause pattern across
all 7 matches, not 7 unrelated incidents. The most defensible conclusion
given purely local evidence is that each affected match's
`three-sixty.json` file was populated, at some point before or during
this project's data acquisition, with 360 tracking data belonging to a
match that **is not present anywhere in this local copy of the StatsBomb
open-data repository** — neither as another match's `events.json` nor as
a duplicate `three-sixty.json`.

---

## 5. Could this affect any of the 417 currently usable matches?

**Checked directly, not assumed.** Computed the `events.json` ↔
`three-sixty.json` overlap ratio for all 417 non-excluded matches:

```
Checked 417 non-excluded matches.
10 LOWEST overlap ratios (as fraction of frame_uuids found in events.json):
  match=3764440 ratio=1.0000  match=3764661 ratio=1.0000  match=3773369 ratio=1.0000
  match=3773372 ratio=1.0000  match=3773377 ratio=1.0000  match=3773386 ratio=1.0000
  match=3773387 ratio=1.0000  match=3773403 ratio=1.0000  match=3773415 ratio=1.0000
  match=3773428 ratio=1.0000
Distribution summary:
  min=1.0000 p1=1.0000 p5=1.0000 median=1.0000 max=1.0000
  matches with ratio < 0.99: 0
  matches with ratio < 0.95: 0
```

**Every one of the 417 remaining matches has exactly 100.00% overlap —
not one exception, not even a single stray unmatched UUID.** There is no
partial or low-grade version of this problem hiding among the "clean"
matches. The 9 total exclusions (2 from the first-stage audit + 7 from
this investigation) are the complete, exhaustive set of affected matches
in the local dataset as of this investigation.

Reproduce: `.venv/bin/python scripts/diagnose_excluded_match_mismatch.py --check-all-ratios`

**Are the 7 exclusions individually justified, not just collectively?**
Yes — each was independently re-verified in this investigation (§2, §3)
against the same evidence standard, not merely carried over from the
2026-10-14 discovery. No match among the 7 shows any partial or
borderline evidence that would argue for reinstatement; all 7 show the
identical, complete 0%-overlap signature.

---

## 6. Recommended next steps (ranked)

1. **(High priority, external)** If/when internet access is available,
   check these 7 match IDs against the live, current
   `statsbomb/open-data` GitHub repository. If the public repository's
   current `three-sixty/{match_id}.json` content differs from this local
   copy (i.e., overlaps correctly with the corresponding `events.json`),
   that would prove the error was introduced during this project's local
   data acquisition and the files could be re-downloaded to resolve the
   issue cleanly. This is the single most likely path to actual
   recovery, and this investigation could not perform it.
2. **(Medium priority)** No further local recovery effort is justified —
   this investigation exhaustively searched the entire local dataset
   (14.87M event ids, 1.38M frame uuids, all pairwise combinations among
   the 7) and found no candidate correspondence. Continuing to search
   locally would not be a good use of time.
3. **(Low priority, documentation)** Keep the 9-match exclusion list as
   the single source of truth (already in place); no change needed as a
   result of this investigation.
4. **(Low priority, monitoring)** If this local StatsBomb data copy is
   ever refreshed/re-downloaded, re-run
   `scripts/diagnose_excluded_match_mismatch.py --check-all-ratios`
   (and `--full-search` if any new 0%-overlap match appears) to confirm
   whether the refresh resolves these 7 matches, and whether any new
   matches exhibit the same signature.

---

## 7. Final summary table

| Match ID | Fixture | Root cause category | Confidence | Origin (project vs. source data) | Recovery status |
|---|---|---|---|---|---|
| 3837706 | PSG vs Stade Brestois | A — source-data mismatch | High | Source data (likely upstream of this project; cannot fully exclude local acquisition) | Not currently recoverable |
| 3877115 | Inter Miami vs Toronto FC | A — source-data mismatch | High | Same as above | Not currently recoverable |
| 3877170 | Inter Miami vs Cincinnati | A — source-data mismatch | High | Same as above | Not currently recoverable |
| 3877194 | Charlotte vs Inter Miami | A — source-data mismatch | High | Same as above | Not currently recoverable |
| 3895158 | Bayer Leverkusen vs Borussia Dortmund | A — source-data mismatch | High | Same as above | Not currently recoverable |
| 3895266 | Bayer Leverkusen vs Wolfsburg | A — source-data mismatch | High | Same as above | Not currently recoverable |
| 3895309 | Borussia Dortmund vs Bayer Leverkusen | A — source-data mismatch | High | Same as above | Not currently recoverable |

**What would change "Not currently recoverable" to "Recoverable"**: a
verified, correctly-paired `three-sixty.json` for this exact match_id
obtained from an authoritative external source (the live StatsBomb
repository). No such verification was possible within this investigation.

---

## Reproducibility

```bash
# Per-match statistics and samples (fast, seconds):
.venv/bin/python scripts/diagnose_excluded_match_mismatch.py

# + overlap-ratio check across all 417 other matches (~1-2 minutes):
.venv/bin/python scripts/diagnose_excluded_match_mismatch.py --check-all-ratios

# + exhaustive full-dataset UUID search across all 4,235 events.json and
# 426 three-sixty.json files (~3-5 minutes):
.venv/bin/python scripts/diagnose_excluded_match_mismatch.py --full-search --check-all-ratios --output results/forensic_mismatch_full.json
```

All commands are read-only; none write to `data/raw/`, `data/processed/`,
or `configs/excluded_matches.json`.

**Test suite**: `.venv/bin/python -m pytest tests/ -q` → **138 passed, 0
failed, 0 skipped** (unchanged by this investigation; no production code
was modified, only a new standalone diagnostic script was added).
