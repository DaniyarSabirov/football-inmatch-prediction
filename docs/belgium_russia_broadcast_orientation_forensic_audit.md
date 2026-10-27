# Belgium vs Russia (3788743) — broadcast orientation forensic audit

Date: 2026-10-23. Supersedes `docs/russia_belgium_orientation_regression_2026-10-22.md`.
Reproduce: `.venv/bin/python scripts/belgium_russia_broadcast_audit.py`
Artefacts: `results/physical_orientation/belgium_russia_broadcast_audit/`
(`transformation_trace.json`, `variant_*.png`, `comparison_sheet.png`, `pitch_diagram.png`,
`lukaku_goal_camera_view.png`). The dataset-wide method this audit uses is described in
`docs/final_statsbomb_360_orientation_research.md`.

## Answers

| Question | Answer |
|---|---|
| Was the visualization you reported inconsistent with the broadcast? | **Yes.** It showed Belgium attacking left→right in the 2nd half with Russia defending the right goal. The broadcast shows the opposite. |
| Which transformation was wrong? | The **choice of shared frame** (Problem B's reference team). The legacy default made Russia the period-1 reference only because Russia had the first 360 event. That choice carries no physical meaning, and here it was wrong for both halves. No per-frame or per-point maths was wrong. |
| X orientation, Y orientation, 180° rotation, or camera side? | **A 180° rotation**, which fixes the left/right error and the near/far (camera-side) error together. A pure X or Y reflection would make a mirror image and is never correct (see §3). |
| Why does the polygon seem to face the opposite touchline? | Because it did. In the previous view the footprint's converging (camera) end pointed to the **top** touchline (apex at display y ≈ −39). The camera was therefore drawn on the far side, and you saw the stadium from behind the camera's opposite touchline. |
| Is the polygon mathematically aligned with the players? | **Yes, in every view.** The polygon, players, actor and ball always go through the same map (actor-to-ball distance 0.0; ball inside the polygon in 99.7–99.9% of frames). Mis-registration was never the problem; the absolute frame was. |
| Are the defended-goal labels and arrows semantically consistent? | **Yes.** "X defending goal" is drawn at the end X defends (x=0 when X attacks +x). In the previous view it labelled correctly for the wrong frame. Arrows are now derived exactly from the transform, cross-checked against shot origins (they agree). |
| What was fixed? | A display-only orientation layer (`src/broadcast_orientation.py`) infers each period's frame from the visible_area camera side, with no reference team. Goal labels moved off the header line. See the research report for the shootout and extra-time defects found along the way. |
| How was it validated independently? | §4: the event synchronized with your broadcast frame (Lukaku goal, 87:45, BEL 2–0 RUS) and its own freeze-frame keeper. The camera inference was computed **without** the registry entry and agrees with it. |
| What remains unknowable without video? | Whether this broadcaster mirrored its feed, and whether every snapshot is from the main camera. The 87:43 frame you described is consistent with the result, but I have not watched the footage myself. |
| Can other matches be affected? | Yes. The legacy default disagreed with the camera inference in **221 of 417** eligible matches; see the research report. |

## 1. Coordinate pipeline, raw JSON → pixels (Courtois Carry, P2 26:53.383 = 71:53)

Event `6d4668fb-2a8b-4f79-8d90-b7d87768a5f3`, Belgium, Carry, Thibaut Courtois.

| Stage | Event / Courtois (keeper, actor) | visible_area vertices | Camera side (apex) |
|---|---|---|---|
| 1. Raw StatsBomb (Belgium's own frame: Belgium attacks +x) | (12.9, 32.9) | (0,80) (0,38.16) (28.11,0) (70.28,0) (96.83,80) | LOW_Y, apex (57.2, −39.5) |
| 2. Problem A (`orientation_correction`) | NORMAL, conf 0.868, unchanged | unchanged | — |
| 3. Legacy display: P2 reference Belgium (kickoff baseline Russia, swapped) → **not rotated** | (12.9, 32.9), left goal | unchanged | LOW_Y = camera at **top** of screen |
| 4. New display: camera inference P2 reference Russia → Belgium **rotated 180°** | **(107.1, 47.1), right goal** | (120,0) (120,41.84) (91.89,80) (49.72,80) (23.17,0) | HIGH_Y, apex (62.8, 119.5) = camera at **bottom** |
| 5. Pixels | the display draws y=0 at the top (`FIXED_Y_LIMITS = (83, −3)`), x=0 at the left | | |

The raw freeze-frame (20 points: 11 teammates, including the keeper/actor, and 9 opponents) is listed in full in
`transformation_trace.json` at every stage. For this event the raw frame and the legacy display are
identical: the legacy display was literally Belgium's own attacking frame.

Reference-team decisions: automatic (legacy) = Russia for P1, Belgium for P2. Camera inference = **Belgium
P1, Russia P2** (P1: 1657 votes to 0; P2: 1479 votes to 0). Registry entry (verified, added 2026-10-23) =
Belgium P1. Halftime swap: observed in the data, not assumed.

Final directions (P2): Russia attacks +x (defends the **left** goal), Belgium attacks −x (defends the
**right** goal, where Courtois is). Belgium's arrow points left, toward Russia's goal, not toward its own.

## 2. Why a goalkeeper next to its own label proved nothing

Raw StatsBomb frames are team-relative: each team always attacks +x in its own events. Problem B
reflects one team so both share a frame, and either choice keeps every keeper next to its own label.
Within-team consistency is invariant under the 180° gauge choice. Only evidence tied to the physical
stadium can break the tie: the broadcast, or the camera footprint, which is recorded *in* the same
team-relative frame.

## 3. The four candidate transforms (applied to the previous view)

| Variant | Courtois | Camera side | Belgium attacks | Proper map? | Consistent with broadcast? |
|---|---|---|---|---|---|
| Identity (previous) | (12.9, 32.9) | top | → | yes | no (both left/right and camera side wrong) |
| **180° rotation** | (107.1, 47.1) | **bottom** | **←** | yes | **yes** |
| X reflection | (107.1, 32.9) | top | ← | **no** (mirror image) | left/right only; camera side and handedness wrong |
| Y reflection | (12.9, 47.1) | bottom | → | **no** (mirror image) | camera side only; left/right and handedness wrong |

Handedness was checked empirically. In their own team's frame, Left Backs average y ≈ 11–13 and Right
Backs y ≈ 69–72, so y=0 is the attacker's left. That matches StatsBomb's documented top-left origin and
the renderer's inverted y-axis. A reflection would put left-sided players on the right of their attack
direction. The 180° rotated view is identical to the new default (`variant_5_new_camera_view.png`).

## 4. Independent evidence

- **Synchronized broadcast moment.** Your frame shows clock 87:43 with BEL 2–0 RUS. The only Belgium goal
  at that score and minute is Lukaku's (P2, 87:45, event `b0e8f7c7…`), and it made the score 3–0. In the
  new view the shot comes from (17.5, 30.8), near the **left** goal. The shot's own freeze-frame keeper is
  an opponent (Russia) at (8.5, 35.4), in the left goal. The footprint covers the left third and converges
  toward the bottom (near) touchline. All 7 snapshots between 87:35 and 87:46 agree: camera at the bottom,
  mean event x = 39.2, polygon centroid x = 45.4. This is the arrangement in your frame: Russia's goal on
  the left, seen from the near side, with the far boards at the top.
- **Non-circularity.** The camera-only plan (no registry, no flags) is computed from visible_area
  geometry alone and reproduces the registry choice (Belgium = period-1 reference). The registry entry was
  set from your broadcast observation. The two sources are independent.
- **Not done.** I did not watch video. This environment cannot inspect footage, and the 87:43 frame was
  described and supplied by you.

## 5. Current status for this match

The registry entry (`verified`, horizontal and vertical) remains the anchor. The camera inference now
corroborates the vertical (camera-side) placement, which the entry had marked as unverified, because a
180° choice fixes both axes at once. I left the registry text unchanged. Updating
`vertical_orientation_verified` is your call.
