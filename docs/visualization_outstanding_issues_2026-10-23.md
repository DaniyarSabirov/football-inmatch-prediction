# StatsBomb 360 visualization — outstanding issues (2026-10-23, updated after the fixes)

This report has two parts. Part I is the state **after** the approved fixes, recomputed over the full
dataset with the current code. Part II is the pre-fix audit that motivated them; its files are
kept in `results/physical_orientation/outstanding_issues/baseline_before_fix/`.

Reproduce:
- `.venv/bin/python scripts/audit_outstanding_issues.py` (~2 min; writes every file below and
  compares against the baseline).
- `.venv/bin/python scripts/audit_broadcast_orientation.py` (match-level categories).
- `.venv/bin/python scripts/build_orientation_review_queue.py` (review queue).
- `.venv/bin/python scripts/render_display_fix_evidence.py` (before/after images).

No raw data, ML data or Problem A/B/C logic was changed. All changes are in the display layer.

---

## Part I — state after the fixes

### 1. Headline counts (`outstanding_issues/summary.json`)

| Quantity | Value |
|---|---|
| Eligible matches | **417** |
| Periods examined | **908** (894 regular + 14 shootouts) |
| Usable 360 frames examined | **1,357,626** |
| Independently verified matches | **2** (3788743, 3930160 → 4 periods) |
| User-reviewed periods | **1** (3901833 P2) |
| Periods derived from a user-reviewed period by the halftime rule | **1** (3901833 P1) |
| High-confidence inferred periods | **902** |
| Periods requiring manual orientation review | **0** |
| Orientation conflicts (contradicted by independent evidence) | **0** |
| Fallback or legacy-derived periods | **0** |
| Confirmed incorrectly rotated frames, before → after | **268 → 0** (all 268 fixed) |
| Copies of one snapshot drawn differently (all 396,822 copies checked) | **0** |
| Previously correct frames with ground truth, still correct | **4,092 / 4,092** |
| Problem A UNCERTAIN frames | 4,974: **4,964 resolved for display**, **10 remaining** |
| Suspicious kick-off frames | 5 before → **2 remaining** (judged probably correct as drawn, §4) |
| Known visualization defects | **0** |
| Render exceptions (first/middle/last frame of every period) | **0** |
| Shot-direction cross-check | 716 agree, **0 disagree**, 192 unavailable |
| Issue rows | 49, all P3 (informational) |
| Manual tasks | 0 required; optional P2 validation only (§7) |

Period status reconciles: 4 + 1 + 1 + 902 = 908. UNCERTAIN frames reconcile: 4,443 duplicate +
470 kick-off rule + 48 lean+camera + 3 camera+continuity + 10 remaining = 4,974.

### 2. France Women's vs Morocco Women's (3901833), period 1: resolved

Anchor: your review of period 2 (PROPOSED: Morocco attacks left→right and defends the left goal;
France attacks right→left and defends the right goal). Under the end-switching rule, period 1 is the
reverse: **France attacks left→right and defends the left goal; Morocco attacks right→left and
defends the right goal.** That is what is displayed.

Recorded provenance:
- Period source `derived_from_user_reviewed_period_and_halftime_rule`.
- Plan provenance `USER_CONFIRMED_AND_DERIVED`.
- On-figure footer: "P1 derived from a user-reviewed period by the halftime rule – not
  broadcast-verified".
- Match category `user_reviewed_and_derived`.
- Not labelled as video-verified or user-reviewed. The registry's `period_reviews` still holds only
  the P2 decision.

Checks before accepting the derivation (none contradicts it):

| Evidence | Result |
|---|---|
| Camera geometry, checked separately from the halftime rule | P1's own footprints: 1,461 vs 183 support the camera at the bottom with France left→right, the same touchline as P2 (1,357 vs 101). France's own-frame camera side flips between halves (P1: 1,042 bottom vs 109 top; P2: 889 top vs 69 bottom), so the end switch is observed in the camera data. The opposite P1 orientation would put the fixed camera on the far touchline in P1 only. |
| The 183 minority footprints | All from confidently classified frames, ball inside the polygon in all 183; 111 behind-goal and 72 reverse-angle cameras (camera switches). |
| Goalkeepers (as displayed) | Keeper at the defended end 99.8% (P1) and 99.6% (P2). |
| Match structure | France kicks off P1, Morocco P2. |

The 88.9% agreement is below the 0.90 *decision* threshold, so the camera alone does not decide P1.
The derivation decides it, and the camera majority corroborates it. No manual review is needed. The
match left the review queue, which now holds 28 optional validation entries.

### 3. The 268 confirmed single-frame flips: all fixed

Implemented in `display_orientation_results` (renderer, display only, deterministic), in this order:

1. **Same-moment duplicates.** StatsBomb attaches one byte-identical snapshot to several events at
   the same instant, and all copies must be drawn identically. A copy is drawn rotated by
   D = c XOR r, where c is its Problem A correction and r its plan rotation, so an UNCERTAIN copy gets
   c = D XOR r from its confident copies. This is pure algebra with no distance thresholds.
   - If the confident copies disagree, D is the one under which the fixed camera appears where it
     must. If the camera is unreadable, the larger summed confidence wins; on a tie the group is left
     unchanged (this happened 0 times).
   - These copies are marked `display_duplicate_consensus`.
2. **Kick-off own-half rule** (Law 8), unless the camera contradicts it.
3. **Problem A lean AND camera side** (the earlier rule). The expected camera side now comes from the
   display plan, so it also works in plan-derived periods. This was mechanism A, the 9 frames of
   3901833 P1.
4. **Camera AND temporal continuity** with the nearest confident neighbouring snapshot (≤ 3 s).
   Validated on 1,012 frames with known correct display: "mirrored" never occurred.

Results (`summary.json → before_after`, `display_resolution_totals`):

| | Frames |
|---|---|
| Confirmed flipped before (sibling ground truth) | 268 |
| … fixed (now drawn exactly like their confident copy) | **268** |
| Additional duplicate flips fixed, missed before because the player layout was near-symmetric | 12 (old display was the exact mirror, distance 0.0; now identical) |
| Kick-off flips fixed (own-half rule + camera) | 3 |
| Flips fixed by camera + continuity (no duplicate) | 3 |
| Total UNCERTAIN frames whose drawing changed | **286** = 268 + 12 + 3 + 3 |
| Confidently classified copies re-aligned to their group (weak `Ball Receipt*` copies, confidence 0.47–0.67, against partners with 0.69–0.95; the camera agrees with the partners) | 25 |
| Previously correct frames that changed | **0** (4,092/4,092 unchanged) |
| Changed frames not explained by a previously detected or newly confirmed flip | 0 |

How newly introduced flips were ruled out:
- Every group of identical snapshots is drawn identically: 396,822 / 396,822 copies.
- No previously correct frame changed.
- None of the round-3 GIF intervals, and none of the five `final_orientation_audit/gifs` intervals,
  contains a changed frame, so those GIFs remain valid.

Of the 142 frames whose actor ring is still > 15 yd from the ball marker, 132 are drawn identically
to their confident same-moment copy, 4 are near-symmetric duplicates (identical by the rotation
algebra), 5 are the re-aligned weak copies and 1 is a continuity fix. This is a data property, not a
flip: in a snapshot borrowed from another event, StatsBomb's `actor` point is the source event's
player (for example the interceptor), not the failed receiver.

### 4. The 5 suspicious kick-off frames

| Frame | Own-half (Law 8) | Camera | Continuity | Outcome |
|---|---|---|---|---|
| 3773552 P1 00:01.454 | 0 of 12 in own half → rotated | rotated | unclear | **fixed** |
| 3788741 P1 00:00.878 | 0 of 7 → rotated | rotated | unclear | **fixed** |
| 3877072 P2 45:00.206 | 0 of 7 → rotated | rotated | unclear | **fixed** |
| 3773631 P1 00:00.992 | 5 of 5 own (below the 6-player minimum) | rotated | matches neighbour as drawn (2.7 vs 5.8 yd) | left as stored; probably correct |
| 3788741 P2 45:00.189 | 7 of 8 own → as drawn | rotated | unclear | left as stored; probably correct |

The own-half rule was validated on the 464 kick-offs whose camera side already agreed: 457 normal,
7 undecided, 0 contradictions.

For the last two frames, the rule-based signals point to the current drawing being correct, and only
the camera disagrees (a kick-off close-up is a plausible alternate angle). Nothing was forced. A
reviewer could confirm them by watching the kick-off; the alternatives are "players in the wrong
halves for one frame" or continuous. This is optional, one frame each.

### 5. Remaining UNCERTAIN frames: 10

| Class | Frames | Where |
|---|---|---|
| likely correct as drawn (camera at bottom, continuous) | 8 | 3788776 P1 00:00.080, 3837771 P2 45:00.590, 3857254 P2 45:01.404, 3857263 P2 45:00.388, 3893791 P1 27:50.417, 3930163 P1 45+1:15.525 (Ball Receipt* and Carry, one shared snapshot), 3930164 P1 00:00.505 |
| conflicting (see §4) | 2 | 3773631 P1 00:00.992, 3788741 P2 45:00.189 |

None is a known flip. They stay uncorrected because no rule has two agreeing independent signals.

### 6. Other checks (unchanged by the fixes)

- **Period transitions:** halftime swap in all regular matches; extra time measured per period (60
  periods; the 3→4 swap 30/30; P3 at P2's ends in 2 matches); all 14 shootouts decided in the shared
  frame.
- **Team identity and colours:** fixed per match; 270 points in 7 periods drawn "team unclear" (data).
- **Goalkeepers:** at the defended end in at least 94% of points in every period (minimum 0.94);
  28 frames with two keeper flags for one team (data).
- **visible_area alignment:** ball inside the polygon in 99.6% of snapshots; no missing or malformed
  polygons; 8,497 alternate-camera footprints (P3).
- **Clocks:** stoppage time shown as `45+…`/`90+…` (74,734 snapshots); shootout clock up to 139:05
  in 3902968 (data).
- **Coordinates:** 40,909 of 21.2 million player points (0.19%) lie outside the drawing viewport and
  are clipped (P3 limitation, coordinates unaltered); 652 sparse freeze-frames.

### 7. Action list

- **P0:** none.
- **P1:** none.
- **P2 (optional validation; `manual_review_priority.csv`, `review/review.html`):**
  - The extra-time exceptions 3869220 P3 and 4018357 P3.
  - The 26 other queued validation matches, to estimate the reliability of the camera inference.
  - Optionally the two conflicting kick-off frames (§4).
- **P3 (data limitations; `unresolved_matches.csv`, 49 rows):** 28 periods with frequent
  alternate-camera footprints, 10 with double keeper flags, 7 with unclear-team points, 1 long
  shootout clock, 2 periods with conflicting kick-off frames, and 1 informational row for the derived
  France–Morocco P1.

### 8. Visual evidence (`outstanding_issues/fix_evidence/`)

Each case is BEFORE (pre-fix logic) / AFTER (current) / REFERENCE (a confident copy or neighbour),
drawn with the production renderer in camera-overlay with players and visible_area:

| File | Case |
|---|---|
| `01_france_morocco_p1_ball_receipt.png` | France–Morocco P1 09:58.510: flipped before; identical to the same-moment Ball Recovery after |
| `02_ball_receipt_duplicate_fix.png` | Celta–Barcelona 13:22.642 Ball Receipt* |
| `03_previously_correct_confident_frame.png`, `04_previously_correct_resolved_frame.png` | no regression: identical before and after |
| `05_kickoff_own_half_fix.png` | Huesca–Barcelona kick-off: Barcelona in Huesca's half before, own half after |
| `06_duplicate_consensus_fix.png` | Alavés–Barcelona 52:30.149: weak Ball Receipt* copy re-aligned to the Carry |
| `07_camera_continuity_fix.png`, `continuity_3773631_neighbours.png` | Betis–Barcelona 25:39.675: continuous with the neighbouring frames after the fix |
| `after_*.gif` (+ `_frames.csv`) | the same windows rendered by the CLI with the standard flags |
| `contact_sheet.png` | all cases |

Tests: **281 passed**. The 29 tests in `tests/test_display_frame_resolution.py` use the real affected
frames (3901833, 3773466, 3773526, 3773552, 3788741, 3877072, 3773631, 3773587) and no-regression
frames. Passing tests show consistency; they are not broadcast verification.

### 9. Stability verdict

- **Match level:** stable. 2 verified, 414 inferred, 1 user-reviewed and derived; 0 conflicts.
- **Period level:** stable. All 908 periods are determined; 0 need manual review.
- **Frame level:** stable within the data. 0 known flips, every duplicate drawn consistently, and 10
  single frames left as stored because the evidence does not support changing them.
- What remains unverifiable without footage: the broadcast orientation of the 414 inferred matches.
  That is a verification gap, not a known error.

---

## Part II — pre-fix audit (baseline, for the record)

Before the fixes the same audit found the following (`baseline_before_fix/summary.json`):
- 767 uncorrected UNCERTAIN frames, of which **268 were confirmed flipped** in 180 matches and
  5 kick-offs were likely flipped.
- 3901833 P1 was ambiguous (camera agreement 0.889) and rule-derived.
- 2 defect mechanisms:
  - **A**: the lean+camera rule never ran in a plan-derived period (9 frames, 3901833 P1);
  - **B**: UNCERTAIN frames whose signals disagreed, or whose camera side was unreadable, were left as
    stored even when a confident same-moment copy showed the correct orientation (259 frames).
- The all-copies duplicate check was added after the baseline. It found 35 further mirrored copies
  between two confidently classified copies, the weak `Ball Receipt*` copies fixed in §3.
