# StatsBomb 360 broadcast orientation — final research report

Date: 2026-10-23. Scope: all 426 StatsBomb 360 match files (417 eligible).
Companion: `docs/belgium_russia_broadcast_orientation_forensic_audit.md`.

## Answer to the research question

> Can we reconstruct the spectator-facing, broadcast-consistent orientation for all eligible matches
> without arbitrarily selecting a reference team?

**Yes, by inference, for 416 of 417 eligible matches. Verification is partial: 2 matches.** The
broadcast camera's position is recorded in the data. Every 360 snapshot's `visible_area` is the ground
footprint of the broadcast image, and its perspective convergence points at the touchline the camera
stands behind. Fixing the presentation convention "camera touchline at the bottom" then determines
every period's 180° choice. The reference team becomes an output, not an input.

What this does **not** give is verification. The inference assumes the polygons come from the main
broadcast camera and that the feed is not mirrored. Its agreement with the two independently verified
matches (2/2), and with every internal physical check, makes it high-confidence. Only external footage
turns it into "verified", and I could not inspect any video in this environment.

## 1. StatsBomb coordinate semantics (measured on the raw JSON)

| Quantity | Semantics |
|---|---|
| Event `location`, `*.end_location` | 120×80 yards, origin top-left. **Team-relative:** the event's own team always attacks toward x=120, in every period. |
| Handedness | y=0 is the attacker's **left** (own-frame mean y: Left Back 11–13, Left Wing 14–19, Right Back 69–72). Drawn with y downward, as this renderer does, the map is proper (not mirrored). |
| `freeze_frame[].location`, `visible_area` | The **same** team-relative frame as the owning event; the whole snapshot moves with its event. Exceptions are the ~5% of snapshots stored 180° rotated relative to their event (Problem A, `orientation_correction.py`). |
| `teammate` / `actor` / `keeper` | Relative to the event the snapshot was captured for; reused cross-team snapshots need Problem C (`team_identity.py`). |
| `visible_area` | Polygon clipped to the pitch rectangle: the broadcast camera's ground footprint. |
| Periods 1–4 | The two teams attack opposite physical goals, so their raw frames differ by a 180° rotation. |
| Period 5 (shootout) | **Both teams' raw frames coincide** (both attack the one goal used). Measured: 14/14 shootouts unanimous. |
| Halftime / extra time | Ends swap 1→2 and 3→4 (observed 416/416 and 30/30). Period 3 is **not** always at period 1's ends: in 2 of 30 matches it continues at period 2's ends. |

Required transformations for a physically coherent display: (A) the per-snapshot rotation fix; (B) a
180° rotation of one team per regular period, or of neither/both teams in a shootout; (D) the choice of
which, made by camera side. Reflections are never required.

## 2. Root cause of the recurring ambiguity

Team-relative coordinates leave one binary gauge freedom per period: which team stays unrotated. Event
data cannot fix it, because every choice is internally consistent: keepers sit by their own labels, and
shots go to the attacked goal. The project fixed it arbitrarily, with the team that has the first 360
event. That pick is a coin flip relative to the camera, and it disagreed with the camera-side inference in
**221/417 matches** (204 wrong in P1 and P2, 15 wrong in all four periods, 2 wrong only in extra time). Belgium–Russia was one of these
matches, and Spain–Croatia another.

## 3. The visible_area camera-side hypothesis

**Model.** For a pinhole camera without roll, the left and right image borders project to ground lines
that meet behind the camera's foot point, on the camera's side. The near and far borders project to
parallel lines. Clipping to the pitch only adds boundary edges and keeps the slopes of the real edges. The
estimator (`estimate_camera_side`) takes the two non-adjacent real edges with the largest angle between
them (≥ 8°) and intersects them. The y-offset of that apex from the centroid gives the camera touchline.
Footprints converging toward a goal line, or without a converging pair, yield UNKNOWN.

The naive "wide side" test was checked and rejected: it gave only ~70% per-period agreement and no
signal at all for Iceland–France. Oblique (panned) footprints and clipping defeat it.

**Empirical results (all 417 eligible matches, 1,352,652 snapshot votes):**

| Test | Result |
|---|---|
| Snapshots with a readable camera side | 99.4% |
| Per-period agreement (894 regular periods) | median 1.000, 5th percentile 0.957, minimum 0.889 |
| Teams in opposite frames within a regular period (prediction) | holds; otherwise agreement would be ~0.5 |
| 1→2 swap observed (not assumed) | 416/416 decided matches |
| 3→4 swap observed | 30/30 |
| Shootout: both teams same camera side | 14/14, unanimous |
| Registry-verified matches reproduced (Belgium–Russia, Spain–Croatia) | 2/2 |

A random or non-physical signal could not produce a halftime swap in 416/416 matches, so the camera side
carries real physical information.

**Identifiability**

| Quantity | From data alone? |
|---|---|
| Camera touchline in each event's frame | yes (apex) |
| Which team attacks toward the camera-left in each period | yes, given that touchline |
| Halftime / extra-time / shootout end relations | yes, measured |
| Broadcast left/right | yes, *if* the polygons come from the main broadcast camera and the feed is unmirrored. An assumption, checked only by external footage. |
| Real stadium side (main stand, compass direction, benches) | **no.** Nothing in StatsBomb encodes the stadium itself. |

Concrete counter-example to over-claiming: a broadcaster that mirrored its feed, or a match logged
mostly from a reverse-angle camera, would give the same polygons with the opposite real-world meaning.
Only video can rule this out.

## 4. Is a reference team necessary?

- **Required by StatsBomb's format?** No. The format only implies that the two teams' frames differ by a
  180° rotation (regular periods) or coincide (shootout).
- **Necessary to put both teams on one scale?** A gauge choice is necessary. A *team* is just one way to
  name it.
- **Can teams, attacking directions and the broadcast orientation be determined without choosing one?**
  Yes. Team identity comes from event data. With the camera convention, each period's unrotated team is
  derived, and attacking directions follow exactly (an unrotated team attacks +x).

Coordinate spaces, now kept separate:

| Space | Where | Defined by |
|---|---|---|
| A. Raw | JSON files (never modified) | StatsBomb, team-relative |
| B. Match-physical | Problem A + per-period relative transforms | measured frame relations (opposite or shared) |
| C. Broadcast/presentation | `src/broadcast_orientation.py` → `DisplayOrientationPlan` | camera touchline at the bottom; anchored by verified evidence when it exists |
| D. ML-normalized | `src/build_statsbomb_360_dataset.py` (unchanged) | one fixed reference team for the whole match: an analytical convention, internally consistent, deliberately not physical |

The ML dataset is untouched. Mirroring one team in a shootout is wrong *physically* but consistent with
space D's own team-relative convention, so it is not an ML defect. A test asserts that the builder does
not import the display layer.

## 5. Dataset-wide audit

Files: `results/physical_orientation/final_orientation_audit/per_match_audit.json` (full per-match
evidence), `per_match_audit.csv` (one row per match), and `audit_summary.json`. Reproduce:
`.venv/bin/python scripts/audit_broadcast_orientation.py` (~45 s, 8 workers).

Per match, the audit records metadata, teams, goalkeepers, periods, snapshot counts, Problem A counts,
the legacy reference, per-period camera votes, the end-swap checks, registry agreement, and the display
plan. It also runs **post-transform checks the inference never used**:

| Independent check (regular periods) | Result |
|---|---|
| Shot-origin direction vs transform | 716 agree, **0 disagree**, 177 too few shots |
| Keepers at defended end (Problem C identity) | median 0.998, min 0.940 |
| Transformed polygons with camera at bottom | median 1.000, min 0.911 |
| Ball/event inside visible_area | median 0.997, min 0.967 |
| Actor-to-ball median distance | 0.0 everywhere, so players, ball and polygon are transformed together |

**Classification**

| Category | Matches |
|---|---|
| Physically verified against independent evidence | 2 (3788743 Belgium–Russia, 3930160 Spain–Croatia) |
| High-confidence inferred, not independently verified | 414 |
| Geometrically consistent but physically ambiguous | 1 (3901833 France–Morocco WWC 2023: P1 agreement 0.889 < 0.90; P1 derived from P2 by the halftime rule, and the P1 majority agrees) |
| Conflicting or unreliable | 0 (a first pass flagged 6; all six were keeper `teammate` flags on reused snapshots, and resolved once the audit used Problem C) |
| Excluded due to insufficient data | 9 (the existing exclusion manifest: 7 UUID mismatches, 1 corrupted file, 1 with one snapshot) |

Competitions covered: UEFA Euro (102), FIFA World Cup (64), Women's World Cup (64), UEFA Women's Euro (61),
Ligue 1 (57), La Liga (35), Bundesliga (31), MLS (3).

## 6. External verification

- I cannot watch or inspect video in this environment, so I verified no match myself. I did not
  search for, or record, any highlight URLs as evidence.
- Verified matches are the two registry entries, both based on your observations:
  Spain–Croatia (screenshots plus your own video comparison) and Belgium–Russia (your broadcast frame at
  87:43, synchronized to Lukaku's 87:45 goal through clock and score). The camera inference was computed
  without them and agrees with both.
- Text match reports cannot establish broadcast left/right, so I did not use them.

## 7. Minimizing manual work

The automatic method resolves 416/417 matches. Manual review is needed only to (a) validate the method
on a sample, (b) settle the one ambiguous match, and (c) confirm the unusual cases. The queue holds **29
matches**: the ambiguous one, the 2 extra-time exceptions, 12 more shootouts, the 4 lowest-agreement
matches, and 1 random match per competition-season (10 strata, fixed seed).

- Builder: `.venv/bin/python scripts/build_orientation_review_queue.py`, which writes
  `final_orientation_audit/manual_review_queue.json` and `review/review.html` with real renders.
- `review.html`: open it locally, one match at a time. It shows the P1 and P2 snapshots in the PROPOSED and
  ALTERNATIVE (180°) views, with visible_area, all freeze-frame players and keeper markers. It also has
  a YouTube *search* link (labelled as unverified), keys 1/2/3, evidence fields (URL, video time,
  scoreboard clock, goal-side and camera-side evidence), browser-local saving, and **Export decisions**.
- Importer: `.venv/bin/python scripts/import_orientation_reviews.py decisions.json [--write]`. It
  creates `verified` only when a side is chosen with a source URL and a scoreboard clock;
  otherwise `partially_verified` or `ambiguous`, neither of which is auto-applied. It never overwrites an
  entry without `--replace` and does a dry run by default. You never edit the JSON by hand.

A sample of about 15 confirmed matches with no disagreement would bound the method's error rate on
this dataset tightly enough for thesis use. Any disagreement goes straight into the registry, which
then overrides the inference for that match.

## 8. Architecture and implementation changes

New: `src/broadcast_orientation.py` (Problem D, display only)
- `estimate_camera_side(visible_area)`: the apex estimator.
- `infer_broadcast_orientation(...)`: per-period votes with OPPOSITE (P1–4) and SHARED (P5) frame
  models, end-swap checks, and a status of INFERRED / AMBIGUOUS / CONFLICTING / NO_EVIDENCE.
- `build_display_plan(...)` → `DisplayOrientationPlan`: one `PeriodTransform` per period with its source
  (`anchor`, `camera`, `anchor+camera_end_relation`, `laws_rule`, `fallback`) and a provenance of
  VERIFIED / PREFERENCE / INFERRED / PARTIALLY_INFERRED / UNKNOWN. A verified anchor fixes the absolute
  frame. The camera then supplies only *relative* end relations, which hold whichever absolute side the
  anchor chose.

Renderer (`src/animate_statsbomb_360_match.py`)
- `--display-orientation camera` (new default) or `legacy` (the previous behaviour, kept as the
  migration path). All existing flags keep working unchanged.
- Precedence: `--physical-reference-team` > `--presentation-reference-team` > verified registry >
  camera inference > legacy fallback (marked UNKNOWN). A warning is printed if an anchor contradicts the
  camera.
- `apply_display_plan_to_match`: each event and its entire snapshot (players, actor, keeper, ball,
  end_location, polygon) are rotated together or not at all.
- Arrows are derived exactly from the transform, with the shot-origin estimate kept as a cross-check
  (a disagreement is printed). No arrows in shootouts.
- An on-figure provenance line ("View: camera touchline at bottom · inferred … not broadcast-verified").
- Goal labels moved into the pitch margin; previously they overlapped the secondary header.

`src/team_normalization.py`: extracted `rotate_event_180` (shared and unchanged in behaviour), and
documented the measured limits of the legacy `{2, 4}` rule.

Defects found and fixed (display only):
1. **Arbitrary absolute frame** in 221/417 matches; replaced by the camera inference.
2. **Penalty shootouts**: one team was mirrored, so half the kicks appeared at the wrong goal. Affected 14
   matches; now handled by the shared-frame model.
3. **Extra time**: the hard-coded "period 3 = period 1 ends" rule was wrong for 3869220 (Morocco–Spain)
   and 4018357 (France–Germany W). Periods 3 and 4 are now measured.

Not changed: raw data, the calibration registry, the ML dataset builder, and Problems A/B/C.

## 9. Regression tests and validation

`tests/test_broadcast_orientation.py` has 27 tests. Geometry: the real Courtois polygon, rotations, a
reflection test showing reflections are distinguishable, degenerate and malformed input. Inference and
plan: no reference team needed, halftime swap, conflict detection, no evidence → UNKNOWN fallback,
shootout shared frame, anchor priority with camera-relative end relations, rule-derived periods,
arrow/label semantics. Real data: Belgium–Russia (inference = broadcast; legacy was the exact
opposite; full Courtois trace; synchronized Lukaku goal at the left goal against Russia's keeper),
Spain–Croatia, Iceland–France, the Morocco–Spain extra-time exception, the Japan–Croatia shootout,
per-frame geometry for both halves, exact equivalence with Problem B's maths, and ML-pipeline isolation.

Full suite: **232 passed** (205 existing + 27 new). Passing tests show consistency and agreement with the
two verified matches; they are not proof of broadcast alignment for the other 414.

Visual outputs:
- `final_orientation_audit/gifs/`: Belgium–Russia around the Courtois event and at the 87' goal,
  Spain–Croatia 31', the Morocco–Spain extra-time P3, and the Japan–Croatia shootout. All use the
  standard flags `--apply-orientation-correction --apply-team-normalization
  --apply-team-identity-resolution --view camera-overlay --timing-mode real-time`.
- `final_orientation_audit/renders/`: the Courtois frame in the new view.
- `belgium_russia_broadcast_audit/`: the variant comparison sheet and pitch diagram.

## 10. Remaining unresolved

- **Verification gap (414 matches):** inferred only. This needs footage review, sampled through
  `review.html`.
- **3901833:** P1 camera evidence is split (89%). Its likely cause, a second camera angle in part of the
  half, is unconfirmed. Review it with footage.
- **Assumptions only video can test:** the polygons come from the main camera, and no broadcaster mirrored
  its feed.
- **Shootout snapshots** are few (3–19), and their footprint often shows the centre circle rather than the
  kick. The shared-frame result is unanimous but rests on little data per match.
- **Unknowable from StatsBomb:** the actual stadium stand and compass side of the camera.

## 11. Addendum (2026-10-23, later): period reviews, UNCERTAIN display frames, round 3

**Period-scoped review decisions.** The registry previously had no way to record "the reviewer
confirmed period 2's render" without claiming a whole-match physical calibration. A separate,
optional `period_reviews` list in `configs/physical_orientation_calibration.json` (loaded by
`load_period_reviews`) now holds one entry per (match, period), with `display_reference_team`
(the team attacking left→right on screen) and `evidence_type`. The evidence type is
`user_review_of_renders`, or `independent_video`, which requires a `source_url`. These entries
are never read as `calibrations` and never make a match "verified". In the display plan a review
fixes only its own period (source `user_confirmed`). A verified calibration or an explicit flag
still takes priority. The on-figure footer is period-specific.

First entry: **3901833 France W. vs Morocco W., period 2**: proposed render confirmed by the
project owner (Morocco attacks left→right and defends the left goal; France the reverse). Recorded
as `user_review_of_renders`, with no URL and no clock, so it is not independent video verification.
It agrees with the camera inference. Period 1 is unchanged: France attacks left→right, derived by
the halftime rule, and its majority camera reading (89%) agrees, but it has not been reviewed. The
match stays in category `geometrically_consistent_physically_ambiguous`, with the review noted in
the reason. A tooling flaw was also fixed: the audit and review queue chose snapshots only for
camera-decided periods, so the ambiguous period had no snapshot. P1 snapshots now exist.

**UNCERTAIN snapshots in the display.** Problem A leaves a snapshot uncorrected when neither
hypothesis is within 15 yd; roughly 11 per match. Where it was really stored rotated, the GIF
showed a one-frame 180° flip of the players and polygon. Round-3 validation found two (actor 80 yd
and 109 yd from the ball). Display-only rule (`resolve_uncertain_for_display`): rotate when Problem
A's vote lean says rotated **and** the fixed-camera side contradicts the snapshot's (period, team)
group; mark normal when both say normal; otherwise leave it as is. On 120 matches the two signals
agreed in 96% of UNCERTAIN snapshots; when the lean was "rotated", the camera contradicted the
event frame in 1085/1088. Problem A, the ML dataset and `--display-orientation legacy` are
unchanged. The renderer prints the resolved and uncorrected counts. The GIFs in
`final_orientation_audit/gifs/` were rendered before this rule existed.

**Random round 3:** `results/physical_orientation/random_visual_checks_round3/` (seed 3740371859,
`scripts/generate_random_visual_checks.py`; manifest.csv, selection_log.txt, per-GIF frame CSVs).
All 10 use the camera layer (INFERRED). GIF frames = 360 frames, every frame has players and a
polygon, 0 suspected flips, 0 snapshots left uncorrected. One match (3835328) has 10 footprints from
an alternate camera angle, with ball and player continuity intact.

Tests: 251 passed (16 period-review tests in `tests/test_period_reviews.py`, 3 UNCERTAIN-resolution tests).

## 12. Addendum (2026-10-23/24): frame-level fixes and France–Morocco P1

- 3901833 P1 is now derived from the user-reviewed P2 by the halftime rule (source
  `derived_from_user_reviewed_period_and_halftime_rule`, plan provenance
  `USER_CONFIRMED_AND_DERIVED`, match category `user_reviewed_and_derived`). Its own camera majority
  (1,461 vs 183) corroborates the derivation. It is not video-verified.
- The display-only UNCERTAIN resolution now uses, in order: same-moment duplicates (with camera-based
  consensus when confident copies disagree), the kick-off own-half rule, lean + camera with the
  plan-derived expected side, and camera + temporal continuity.
- Result: 268 → 0 confirmed flipped frames, and all 396,822 duplicate copies drawn identically.
  10 UNCERTAIN frames remain, none a known flip.
- Full details, counts and before/after evidence: `docs/visualization_outstanding_issues_2026-10-23.md`.
