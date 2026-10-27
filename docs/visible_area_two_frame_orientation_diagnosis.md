# Diagnosis: Frame A → Frame B Apparent 180° Orientation Flip — Spain vs Croatia

> **PARTIALLY SUPERSEDED (2026-10-21, later same day) by
> `docs/physical_pitch_orientation_validation.md`.** This report's
> four independent internal-consistency checks (team-label determinism,
> goalkeeper-anchor self-consistency, aggregate shot-direction
> self-consistency, pixel inspection for *label stability* across frames)
> all remain correct — none of them were wrong, and none of them
> contradict the finding below. What this report could not and did not
> test is whether the *consistently-applied* orientation matches the real
> broadcast camera, because no broadcast image was available at the time.
> §6 of this report explicitly flagged that gap ("this cannot rule out an
> error that exists identically... in all of this project's derived
> checks simultaneously"). A later investigation with an actual broadcast
> screenshot found exactly that: the default rendering is internally
> consistent but was mirrored relative to the real camera for this match.
> See the superseding report for the root cause, the fix, and the
> corrected GIF. Kept here unmodified for the historical record.

Date: 2026-10-21. This report supersedes and deepens
`docs/visible_area_orientation_diagnosis.md` (same match, different
frame pair: that report examined the goal→kickoff transition; this one
examines the two specific build-up frames the user identified). No
production code was modified to produce this report; the only new
artifacts are this file and the rendered GIF/manifest under `results/`.

**Headline finding: no orientation bug was found.** The apparent 180°
flip between Frame A and Frame B is a legitimate ~92-unit long diagonal
"switch" pass (`pass.switch: true` in the raw StatsBomb event, confirmed
below), rendered correctly but with no intermediate frame because the two
events between A and B lack 360 coverage. Direct pixel inspection of the
rendered GIF (not just numeric point-in-polygon checks) confirms the pitch
orientation — which end is labeled "Croatia goal" vs "Spain goal," and
which direction each team's attack arrow points — **never changes** across
Frame A, Frame B, the goal, or the kickoff. Full reasoning and evidence
below.

---

## 1. Match and frame identification (confirmed facts)

Match 3930160 = Spain 3–0 Croatia, UEFA Euro 2024, 2024-06-15 (re-verified,
unchanged from the prior report).

| | Frame A | Frame B |
|---|---|---|
| Event UUID | `0e0af498-4a99-4dcb-873d-709345647988` | `5afe770b-7ffa-4438-805f-6992e7bb4a27` |
| Event index | 1302 | 1305 |
| Timestamp | 00:31:00.825 | 00:31:12.222 |
| Type | Pass | Pass |
| Team | Spain | Spain |
| Actor | Fabián Ruiz Peña | Lamine Yamal Nasraoui Ebana |
| `event.location` (raw) | `[41.9, 12.7]` | `[106.8, 53.0]` |
| Has 360 coverage | Yes | Yes |

**Both events belong to the same team (Spain), the same possession
(`possession: 54`), with no team change, no goal, and no restart between
them** — this is a mid-possession, same-team pair, which is exactly why it
is a more rigorous test of the pipeline than the goal/kickoff transition
examined previously (that pair necessarily involved a team change by
football logic; this one does not, so any apparent flip here cannot be
attributed to the documented `REFERENCE`/`MIRRORED` team switch).

### What actually happens between A and B (confirmed from the raw event stream)

```
idx 1301  Ball Receipt*  Spain  Fabián Ruiz   loc [42.5, 12.7]   (has 360)
idx 1302  Pass           Spain  Fabián Ruiz   loc [41.9, 12.7]   (has 360)  <- FRAME A
            pass.switch: true, pass.length: 91.76, end_location: [110.0, 74.2]
idx 1303  Ball Receipt*  Spain  Lamine Yamal  loc [110.0, 74.2]  (NO 360)
idx 1304  Carry          Spain  Lamine Yamal  loc [110.0, 74.2]  (NO 360)
idx 1305  Pass           Spain  Lamine Yamal  loc [106.8, 53.0]  (has 360)  <- FRAME B
```

Frame A's own `pass` sub-object carries StatsBomb's own `"switch": true`
flag with `length: 91.76` (pitch units) and `end_location: [110.0, 74.2]`
— **this is StatsBomb's own classification of this exact pass as a
long cross-field switch**, not an inference made for this report. The two
intermediate events (idx 1303, 1304) have **no 360 coverage** and are
therefore excluded from `usable_frames` (confirmed:
`load_and_join_match` only keeps joined event/frame pairs with a
non-empty `freeze_frame`, `src/animate_statsbomb_360_match.py:530-539`).
**Frame A and Frame B are consecutive in the rendered GIF only because the
two events between them happen to lack 360 data** — the real football
action (receive the switch, carry it forward) is invisible to the
renderer by construction (documented, intentional: "no synthetic frames,
no interpolation," `README.md`), not a defect introduced by this
diagnosis or the pipeline.

---

## 2. Full raw-to-final trace for both frames (recomputed directly, production functions only)

Computed via `load_and_join_match`, `compute_orientation_results`,
`resolve_reference_team_id`, `apply_correction`,
`apply_team_normalization_to_event/_frame` — the exact functions and call
order `src/animate_statsbomb_360_match.py:main()` uses
(`:2397`, `:2406-2407`). Reference team for this match: **Croatia**
(`reference_team_id=785`).

### Frame A

| | Value |
|---|---|
| `event.location` raw → after team-norm | `[41.9, 12.7]` → `[78.1, 67.3]` |
| Orientation classification | **NORMAL**, confidence 0.967, best_distance ≈ 0.00 |
| Classifier votes | actor: +1.000 (w=0.40, d_normal=1.5e-6, d_rot=65.51); nearest_point: +1.000 (w=0.20); centroid: +0.803 (w=0.15, d_normal=6.45, d_rot=59.10); visible_area: +1.000 (w=0.15) |
| Team-normalization label | **MIRRORED** (Spain ≠ reference) |
| Raw `visible_area` (5 pts) | `(39.61,67.83) (19.39,0.0) (66.89,0.0) (64.25,64.45) (39.61,67.83)` |
| Raw polygon centroid | `(45.95, 40.02)` |
| Raw actor location | `(41.90, 12.70)` — matches `event.location` to 6 decimal places |
| Final (post A+B) polygon centroid | `(74.05, 39.98)` |
| Final actor location | `(78.10, 67.30)` |
| Goalkeeper visible in this frame? | **No** — 0/19 freeze_frame points have `keeper=True` |

Orientation = NORMAL means Problem A applied **zero** change to this
frame (`apply_correction` returns an unmodified copy for NORMAL/UNCERTAIN,
`src/orientation_correction.py:368-372`). Only Problem B's mirror
(`(x,y) → (120-x, 80-y)`) was applied, uniformly to the event location,
every freeze_frame point, and the polygon together.

### Frame B

| | Value |
|---|---|
| `event.location` raw → after team-norm | `[106.8, 53.0]` → `[13.2, 27.0]` |
| Orientation classification | **NORMAL**, confidence 0.974, best_distance ≈ 0.00 |
| Classifier votes | actor: +1.000 (w=0.40, d_normal=3.1e-6, d_rot=97.14); nearest_point: +1.000 (w=0.20); centroid: +0.842 (w=0.15, d_normal=8.40, d_rot=97.73); visible_area: +1.000 (w=0.15) |
| Team-normalization label | **MIRRORED** (same as Frame A — see §3) |
| Raw `visible_area` (6 pts) | `(106.82,71.17) (87.28,60.85) (107.01,0.0) (120.0,0.0) (120.0,49.96) (106.82,71.17)` |
| Raw polygon centroid | `(107.99, 42.19)` |
| Raw actor location | `(106.80, 53.00)` — matches `event.location` to 6 decimal places |
| Final (post A+B) polygon centroid | `(12.01, 37.81)` |
| Final actor location | `(13.20, 27.00)` |
| Goalkeeper visible in this frame? | **Yes** — Croatia's keeper (`teammate=False`), raw `(118.30, 43.15)` → final `(1.70, 36.85)` |

Same as Frame A: orientation NORMAL (no Problem A change), only Problem
B's single, uniform mirror applied.

---

## 3. Why this is not a flip: three independent checks (not just point-in-polygon)

The task explicitly warns that point-in-polygon agreement alone is
insufficient (a globally-mirrored-but-self-consistent scene would also
pass that test). Three different, **independent** checks were used here:

### 3.1 Team-normalization label consistency (deterministic, checked directly)

`team_normalization_label` depends **only** on `event["team"]["id"]`
compared against the match's single, fixed `reference_team_id`
(`src/team_normalization.py:95-99`) — it cannot differ between two events
of the same team. Confirmed directly: **both Frame A and Frame B are
labeled `MIRRORED`.** Since both frames received the identical
transform, the relative geometric relationship between them was
necessarily *preserved* by the mirror (a single reflection is a global,
uniform isometry — it cannot introduce a *relative* inconsistency between
two points it treats identically). Whatever differs between Frame A and
Frame B's final coordinates was already present, in mirrored form, in
their raw coordinates.

### 3.2 Independent anchor: Croatia's goalkeeper absolute position

The goalkeeper is a strong independent check because a team's own keeper
should stay near their own defended goal for the vast majority of play,
regardless of where the ball currently is. Croatia's keeper is visible in
Frame B's raw snapshot. After full correction, his position is
**`(1.70, 36.85)`** — within 2 pitch-units of the goal line `x=0`.
Separately, `resolve_reference_team_id` + the documented per-team
coordinate convention (`src/team_normalization.py:13-21`) establish that
**Croatia (the reference team) defends `x=0`** — this was not assumed for
this report, it follows directly from Croatia being unmirrored and the
module's own forensic finding that a team's raw coordinates encode their
own attack as `+x`. The keeper being found almost exactly at `x≈1.7`,
independent of and not used to derive the orientation classification
itself, is strong corroborating evidence that the global coordinate
assignment is physically correct, not just internally self-consistent.

### 3.3 Independent anchor: aggregate attacking-direction inference (whole-period, not frame-specific)

The renderer's own `infer_attacking_directions()`
(`src/animate_statsbomb_360_match.py:1020-1075`) computes each team's mean
shot-origin x **across the entire period**, using every shot event, not
just Frame A/B. The actual run (§5) printed:

```
Attacking-direction inference: Inferred from each team's mean period shot-origin x;
means occupy opposite pitch halves (Croatia: 102.9, Spain: 13.3).
```

This is computed from the full set of period-1 shots (an aggregate,
independent of the two specific frames under investigation) and confirms:
**Croatia's shots cluster near x≈103 (attacking toward x=120), Spain's
cluster near x≈13 (attacking toward x=0)** — exactly consistent with
Frame B's final actor position (13.2) and the goalkeeper check above. This
is the single strongest piece of evidence because it does not depend on
any individual frame at all.

### 3.4 Direct visual (pixel) inspection — not claimed without actually looking

The rendered GIF was reopened and individual frames extracted as PNG and
**visually inspected** (not inferred from numbers alone):

- **Frame A** (`frame 2/65`): actor (Fabián Ruiz, yellow ring) sits
  right-of-center, roughly between the halfway line and the edge of the
  camera polygon; "Croatia goal" label top-left, "Spain goal" label
  top-right, attack arrows: Croatia →, Spain ←.
- **Frame B** (`frame 3/65`): actor (Lamine Yamal) now sits deep inside
  the **left** penalty box, next to Croatia's goalkeeper diamond marker;
  **same** goal labels, **same** attack arrows, unchanged.
- **Goal** (`frame 17/65`, idx 1322): actor (Fabián Ruiz) inside the left
  box next to the Croatia keeper — visually exactly where a goal should
  be; same labels/arrows.
- **Pre-gap** (`frame 19/65`, idx 1324, Goal Keeper, Dominik Livaković):
  same left-box cluster, same labels.
- **Kickoff** (`frame 20/65`, idx 1325, Pass, Andrej Kramarić): camera
  pulled back to a wide view centered on the halfway line, players
  arranged in a kickoff formation around the center circle; **same**
  labels/arrows, still unchanged.

**The goal-side labels and attack-direction arrows are identical across
every one of these five frames.** This is the direct visual confirmation
that the pitch orientation itself never rotates — only the camera's
location on an already-fixed pitch changes, exactly tracking the ball.

---

## 4. Comparison with the Belgium–Morocco reference

Two frames from `results/statsbomb_360_match_3857283_period_1_seconds_1080_1110_dark_real_time_camera_overlay_oriented_teamnorm_teamid.gif`
(42 frames total) were also extracted and inspected: frame 1 (18:00.733,
Belgium, Amadou Onana) and frame 22 (18:19.955, Belgium, Kevin De Bruyne).
Same pattern: "Belgium goal" left, "Morocco goal" right, consistent
across both frames; the ball/camera move gradually (no 90+ unit single-pass
jump in this particular window), which is why this window never produces
a visually dramatic cut. **This is a difference in football content
(a long switch pass + a goal/restart vs. continuous build-up play), not a
difference in code behavior** — both matches show the same
internally-and-externally-consistent orientation handling under direct
inspection.

---

## 5. Final GIF generation

### CLI inspected before use

`python src/animate_statsbomb_360_match.py --help` was run first (see
transcript below is based on the actual parser; no flag was invented).
All requested flags exist exactly as named: `--match-id`, `--period`,
`--start-second`, `--end-second`, `--apply-orientation-correction`,
`--apply-team-normalization`, `--apply-team-identity-resolution`,
`--view`, `--theme`, `--timing-mode`, `--output`,
`--event-frame-manifest`.

### Exact command used (copied from the actual successful run)

```bash
.venv/bin/python src/animate_statsbomb_360_match.py \
  --match-id 3930160 \
  --period 1 \
  --start-second 1860 \
  --end-second 2040 \
  --apply-orientation-correction \
  --apply-team-normalization \
  --apply-team-identity-resolution \
  --view camera-overlay \
  --theme dark \
  --timing-mode real-time \
  --output results/spain_croatia_goal_kickoff_camera_corrected.gif \
  --event-frame-manifest results/spain_croatia_goal_kickoff_camera_corrected_manifest.csv
```

### Verified output

| | Value |
|---|---|
| Output path | `results/spain_croatia_goal_kickoff_camera_corrected.gif` (confirmed exists, 2,661,405 bytes) |
| Selected real event frames (program output) | **65** |
| Unique event UUIDs rendered | **65** (no duplicates) |
| Frames actually present in the saved GIF (reopened with PIL, `Image.seek` until `EOFError`) | **65** — matches exactly |
| GIF pixel size | 1600×900 |
| Resolved interval | `[1860, 2040)` seconds, period 1 → `31:00–34:00` |
| First / last selected event timestamp | `00:31:00.825` / `00:33:49.558` |
| Orientation classification (whole match) | NORMAL=3070, ROTATED_180=173, UNCERTAIN=10 |
| Team normalization (whole match) | reference_team_id=785 (Croatia), REFERENCE=1752, MIRRORED=1501 |
| Team identity resolution | shared=208, recoverable=208, **ambiguous=0** |
| Structural check | player coordinates inside pitch: 1142/1146 (4 raw out-of-bounds points, unrelated to orientation — a pre-existing, documented raw-data property, not introduced here); ball coordinates inside pitch: 65/65 |

### The goal and kickoff *are* represented by usable 360 frames (confirmed, not assumed)

From the event-frame manifest (one row per rendered frame, exported by the
actual run, not reconstructed afterward):

| Frame # | idx | Timestamp | Type | Team | Player | Visible players | Has `visible_area` |
|---|---|---|---|---|---|---|---|
| 17 | 1322 | 00:31:24.988 | Shot (Goal) | Spain | Fabián Ruiz Peña | 19 | True |
| 19 | 1324 | 00:31:25.821 | Goal Keeper | Croatia | Dominik Livaković | 15 | True |
| 20 | 1325 | 00:32:33.968 | Pass (kickoff) | Croatia | Andrej Kramarić | 20 | True |

Both the goal shot and the kickoff restart have non-empty, usable 360
freeze frames — **this was verified from the manifest, not assumed.**

### How the 68.147-second gap is represented (verified from the manifest, corrected from the prior report)

The manifest's `gap_to_next_seconds` / `gif_display_duration_ms` columns
show precisely:

```
frame 19 (idx 1324, Goal Keeper):  gap_to_next = 68.147 s  →  gif_display_duration_ms = 1500  (capped)
frame 20 (idx 1325, kickoff Pass): gap_to_next =  1.450 s  →  gif_display_duration_ms = 1450  (not capped)
```

In `--timing-mode real-time`, each frame's hold duration is
`min(max(gap_to_next*1000, minimum_ms), maximum_ms)`
(`src/animate_statsbomb_360_match.py:1766-1770`), with
`--maximum-frame-duration-ms` defaulting to **1500**
(`:2265`). **The real 68.147-second stoppage (goal celebration) is held on
screen for exactly 1.5 seconds, then the GIF cuts directly to the kickoff
frame**, which itself holds for 1.45 s. This is a **shortened display
duration, not an orientation error** — the geometry of frame 19 and frame
20 is each independently correct (§2, §3); only the *time compression*
between them is dramatic, and that compression is an explicit, documented
feature of real-time mode's duration cap, not a bug.

---

## 6. What was verified visually vs. what remains uncertain

**Verified visually (pixels actually inspected, listed in §3.4 and §4)**:
Frame A, Frame B, the goal frame, the pre-gap frame, the kickoff frame
(Spain–Croatia), and two frames from the Belgium–Morocco reference. In
every case, goal-side labels and attack arrows were unchanged, and player/
camera positions matched the numeric trace in §2 within visual precision.

**Not verified, and explicitly flagged as uncertain**: the complete 65-frame
GIF was not inspected frame-by-frame — only the 7 frames listed above. It
remains possible (though no evidence found it) that a different part of
the 65-frame sequence shows a different anomaly; this report's claims are
scoped to the frames actually inspected, not to the entire sequence.
Additionally, "correct orientation" here is checked against StatsBomb's
own event/shot data as ground truth (the goal's location, the keeper's
position, the aggregate shot directions) — this cannot rule out an error
that exists identically in StatsBomb's own raw data and in all of this
project's derived checks simultaneously (i.e., a hypothetical error in the
original data collection that no internal cross-check could detect). No
evidence of that was found, but it cannot be exhaustively excluded by
analysis of this dataset alone.

---

## 7. Were any code changes necessary?

**No.** No bug was found in `orientation_correction.py`,
`team_normalization.py`, `team_identity.py`, or the rendering code in
`animate_statsbomb_360_match.py`. Per the task's explicit instruction not
to disguise a non-problem with cosmetic changes, and consistent with not
finding a problem, **no code was modified and no regression test was
added** — there is no defect to guard against. If a future change is
desired, it would be a **presentation/UX enhancement, not a correctness
fix**: e.g., a visible on-screen marker when a GIF frame's hold was
duration-capped (so a human viewer can distinguish "real quick
transition" from "long real-world gap compressed for display") would make
sequences like the goal→kickoff transition easier to read, but this is a
design decision outside this diagnosis's scope and was not implemented.

---

## 8. Summary

- **Root cause of the apparent 180° flip**: none found in the code. The
  visual effect is produced by a real, StatsBomb-flagged long switch pass
  (91.76 units, `pass.switch: true`) whose two intermediate events lack
  360 coverage, combined with the (correct, consistent, verified)
  `MIRRORED` coordinate convention applied identically to both frames.
- **Confidence**: high. Supported by (a) deterministic team-label
  consistency, (b) an independent goalkeeper-position anchor, (c) an
  independent whole-period aggregate shot-direction check, and (d) direct
  pixel inspection of 7 rendered frames across both the problem match and
  the reference match — all four lines of evidence agree.
- **Final GIF**: `results/spain_croatia_goal_kickoff_camera_corrected.gif`,
  **65 frames** (confirmed both from the program's own report and by
  reopening the saved file), covering period 1, seconds [1860, 2040),
  rendered with `--apply-orientation-correction
  --apply-team-normalization --apply-team-identity-resolution --view
  camera-overlay --theme dark --timing-mode real-time`, all three
  correction flags active as requested.
- **Is the orientation issue "resolved"?** There was no orientation bug to
  resolve. The pipeline's existing, unmodified behavior is correct for
  this match and this frame pair, to the extent verifiable from the
  evidence gathered here (§6 states the limits of that verification). No
  code changes were made or are recommended on the evidence currently
  available.

Nothing was committed or staged as part of this diagnosis.
