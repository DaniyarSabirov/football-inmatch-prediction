# Final visualization validation — review package (2026-10-24)

Status: **prepared, awaiting your review.** Nothing in this package has been manually validated. The
automatic checks below test internal consistency, not agreement with real broadcast footage.
No orientation decision, registry entry, raw data or ML data was changed while building it.

Reproduce:
- Random GIFs: `.venv/bin/python scripts/sample_final_validation_gifs.py --seed 2734373985`
  (the selection alone: add `--dry-run`).
- Targeted evidence: `.venv/bin/python scripts/build_targeted_validation_evidence.py`.

Every GIF in this package was rendered by the production CLI
(`src/animate_statsbomb_360_match.py`) with `--apply-orientation-correction
--apply-team-normalization --apply-team-identity-resolution --view camera-overlay --timing-mode
real-time --display-orientation camera`. Each GIF has a per-frame CSV next to it. There is no
interpolation: one GIF frame per real 360 snapshot, with real-time holds.

## 1. Random GIFs (`random_gifs/`)

| GIF | Match | Teams | Period | Match clock | Min | 360 frames |
|---|---|---|---|---|---|---|
| random_01 | 3857269 | Brazil vs Switzerland (WC 2022) | 1 (early) | 03:37.882 – 07:01.058 | 3.6 | 144 |
| random_02 | 3837928 | Paris Saint-Germain vs Lyon (Ligue 1 22/23) | 1 (late) | 34:05.713 – 35:48.705 | 3.0 | 73 |
| random_03 | 3788761 | Sweden vs Slovakia (Euro 2020) | 2 (middle) | 64:14.861 – 66:19.628 | 3.9 | 131 |
| random_04 | 3895067 | Bayer Leverkusen vs Darmstadt 98 (BL 23/24) | 2 (late) | 84:30.253 – 86:25.359 | 2.0 | 89 |
| random_05 | 3901735 | England W. vs Nigeria W. (WWC 2023) | 4 (extra time, early) | 106:42.626 – 109:10.629 | 3.2 | 88 |

"Min" is the sampled interval length; the clock shows the first and last snapshot in it.

**Sampling.**
- Seed 2734373985, drawn fresh. Two dry runs gave identical selections.
- Five slots fix the coverage: P1 early, P1 late, P2 middle, P2 late, extra time. The match, the
  position inside the third and the length (120–240 s) are random.
- Excluded: 30 matches you have already seen in earlier samples or evidence (listed in
  `sampling_seed.json`).
- Rejection rules: fewer than 40 frames, or ball path under 150 yd or x-range under 20 yd. **0 draws
  were rejected** (`sampling_rejections.csv` is empty apart from its header).
- Overlap: 3901735 is also one of the 26 reliability-sample matches (shootout).

**Automatic checks** (`sampling_manifest.csv`). All five GIFs:
- were rendered with the required flags, with the camera layer active (provenance INFERRED, camera
  agreement 0.996–1.0);
- have GIF frames equal to the 360 frames, and players and visible_area present in every frame;
- show 0 camera-at-top snapshots, 0 suspected 180° flips and 0 uncorrected UNCERTAIN snapshots;
- have keepers at their defended end in every checked point;
- have 0 geometry and 0 colour mismatches between duplicate snapshots (159 copies in total).

**Warning, random_05:** the pixel check failed on frame 1 (106:42.626, Pass). StatsBomb's snapshot
there contains a single player (the actor), so only one dot plus the polygon is drawn. This is a
data property; frames 45 and 88 show 15 and 9 players.

**Per-GIF checklist** (mark each item yes/no):

| Item | What to look for |
|---|---|
| Orientation | The team named "attacks →" in the header goes toward the right goal throughout; the keepers (diamonds) stay at their own "defending goal" side |
| Team colours | Each team keeps one colour for the whole GIF; no dot switches colour between two consecutive frames showing the same moment |
| Camera polygon | Visible in every frame; usually narrower toward the bottom (camera side); the ball and most players lie inside it |
| Player positions | No dots far outside the pitch; plausible formation for the phase of play |
| Continuity | No frame where the whole picture jumps to the mirrored side and back; ball path plausible frame to frame |
| Clock | The header clock increases; stoppage shown as `45+`/`90+`; random_05 shows 106–109' |
| Anomalies | Anything else (frozen frames, empty frames, overlapping labels) |

## 2. Targeted evidence

### 2.1 Remaining UNCERTAIN frames (`uncertain_frames/`, 9 sheets = 10 frames)

Each sheet shows: PREVIOUS frame | THIS frame AS DRAWN | THIS frame rotated 180° | NEXT frame.

| Sheet | Match | Clock | Events | Automatic classification |
|---|---|---|---|---|
| 3773631_p1_000000_992 | Real Betis – Barcelona | 00:00.992 | Kick-off pass | **conflicting**: camera says rotated; continuity says as drawn |
| 3788741_p2_000000_189 | Turkey – Italy | 45:00.189 | Kick-off pass | **conflicting**: camera says rotated; 7 of 8 players in their own half as drawn |
| 3788776_p1_000000_080 | Sweden – Poland | 00:00.080 | Kick-off pass | likely correct |
| 3837771_p2_000000_590 | PSG – Troyes | 45:00.590 | Kick-off pass | likely correct |
| 3857254_p2_000001_404 | Denmark – Tunisia | 45:01.404 | Kick-off pass | likely correct |
| 3857263_p2_000000_388 | Spain – Germany | 45:00.388 | Kick-off pass | likely correct |
| 3893791_p1_002750_417 | Spain W. – Costa Rica W. | 27:50.417 | Kick-off pass (restart) | likely correct |
| 3930163_p1_004615_525 | England – Serbia | 45+1:15.525 | Ball Receipt* + Carry (one snapshot) | likely correct |
| 3930164_p1_000000_505 | Slovakia – Belgium | 00:00.505 | Kick-off pass | likely correct |

What needs your judgement: whether AS DRAWN or the ALTERNATIVE fits the neighbouring frames. At a
kick-off, every player must be in their own half. No frame has a same-moment duplicate, so there is no
ground truth. A kick-off as the first frame of a half has no "previous" panel. `index.csv` lists the
signals per frame.

### 2.2 Extra time (`extra_time/`, optional)

| Match | P2 end | P3 start | P4 start | Displayed left→right: P2 / P3 / P4 |
|---|---|---|---|---|
| 3869220 Morocco vs Spain | 90+3:02 – 90+5:01 (63 fr.) | 90:00 – 91:00 (55) | 105:00 – 105:38 (27) | Morocco / Morocco / Spain |
| 4018357 France W. vs Germany W. | 90+6:03 – 90+7:58 (46) | 90:00 – 91:00 (25) | 105:00 – 106:00 (43) | France W. / France W. / Germany W. |

Automatically validated: each period's own camera evidence is decided (agreement ≥ 0.98), and the
3→4 swap is observed. What is unusual and needs your eyes: extra time **starts at period 2's ends**.
The same team attacks the same goal from the end of P2 into P3, and the ends swap at P4. Check the
keepers' sides in each clip.

### 2.3 France W. vs Morocco W., 3901833 (`france_morocco/`)

| Clip | Window | France shot (as drawn) | Displayed left→right | Source |
|---|---|---|---|---|
| `3901833_p1_around_first_france_shot.gif` | 12:30 – 13:20 (21 fr.) | 13:10.561 at (95, 29), toward the right goal | France | derived from your P2 review + halftime rule |
| `3901833_p2_around_first_france_shot.gif` | 48:48 – 49:38 (44 fr.) | 49:27.948 at (17, 37), toward the left goal | Morocco | your P2 review (render, not video) |

Also included: `review_snapshot_p1_proposed.png` and `review_snapshot_p2_proposed.png`.

- Already reviewed by you: P2.
- Automatic for P1: P1's own camera majority (1,461 vs 183) agrees with the derivation, and keepers
  sit at their own end in 99.8% of points.
- Your judgement: whether the halftime change of attacking direction looks right across the two clips.

### 2.4 Corrected-frame examples (`corrected_frame_examples/`, copied, not re-rendered)

Before/after/reference sheets:
- `01` France–Morocco P1 Ball Receipt*.
- `02` Celta–Barcelona Ball Receipt* (duplicate fix).
- `03`, `04` previously correct frames, unchanged.
- `05` kick-off own-half fix.
- `06` duplicate consensus fix.
- `07` camera + continuity fix (with `continuity_3773631_neighbours.png`).
- `contact_sheet.png`.

Automatically validated: all 268 confirmed flips are fixed, and all 396,822 duplicate copies are drawn
identically (`docs/visualization_outstanding_issues_2026-10-23.md`). Optional for you: a visual
sanity check of these examples.

### 2.5 Reliability sample, 26 matches (optional)

`reliability_sample_index.csv` lists 12 shootouts, the 4 lowest-agreement matches (0.911–0.924) and
10 stratified matches. They are reviewed in `results/physical_orientation/final_orientation_audit/review/review.html`
(proposed vs 180° alternative; keys 1/2/3; Export), and decisions are imported with
`scripts/import_orientation_reviews.py <export> --write`. All 26 are automatically inferred, none is
ambiguous, and none has been reviewed by you.

## 3. Status overview

| Category | Automatically validated | Reviewed by you | Awaiting your review |
|---|---|---|---|
| 5 random GIFs | consistency checks pass (one sparse-frame warning) | — | **all 5** |
| 10 UNCERTAIN frames | classified (8 likely correct, 2 conflicting); not changed | — | **all 10** (9 sheets) |
| Extra time 3869220 / 4018357 | camera decided per period, swap 3→4 observed | — | optional |
| France–Morocco | P1 derivation corroborated by camera majority | P2 (renders) | P1/P2 relation (optional) |
| Corrected-frame examples | 268 → 0 flips; duplicates consistent | earlier ad-hoc inspection only | optional |
| Reliability sample (26) | all inferred with high confidence | — | optional |
| Registry-verified matches 3788743, 3930160 | camera agrees | yes (earlier) | — |

## 4. Automatic validation run (2026-10-24)

- `pytest`: **281 passed**.
- Flags: every GIF in this package uses all required flags, with the camera layer active.
- Polygon rendered: all 5 random GIFs have polygon-edge pixels in the sampled frames, and visible_area
  is present in 525 of 525 frames.
- Orientation: 0 suspected flips, 0 camera-at-top snapshots and 0 uncorrected UNCERTAIN snapshots in
  the random intervals.
- Duplicates: 0 geometry and 0 colour mismatches.
- Reproducibility: identical dry-run selections for seed 2734373985.
- Warnings: random_05 frame 1 has a single-player snapshot (data). No other failures.

## 5. Update 2026-10-24: your review of the nine uncertain-frame sheets

- All nine selections resolve to the CURRENT drawing; no orientation was changed. They are recorded in
  `configs/physical_orientation_calibration.json → frame_reviews`, as 9 snapshots / 10 events,
  evidence `user_review_of_renders` (not video). The decision file is
  `manual_decisions/uncertain_frames_decisions_2026-10-24.json`.
- Unresolved UNCERTAIN frames: 10 → 0 (all user-reviewed). Drawn output is unchanged
  (0 frames differ).
- Frame 505 (3930164): the orientation is confirmed, but there is a confirmed **team-colour error from
  raw StatsBomb `teammate` flags**. It is not corrected and stays an open issue. See
  `team_identity_investigation/README.md`.
- A separate pipeline error was found and fixed for display: 2 shared snapshots in 3857293 were
  coloured via the wrong event.
- Still awaiting your review: the 5 random GIFs, and the colour policy for raw-flag errors (§ "Not
  fixed" in the investigation README). The validation phase is **not** closed.
