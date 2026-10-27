# Diagnosis: Apparent 180° Camera/Orientation Jump — Spain vs Croatia (31:00–34:00)

> **PARTIALLY SUPERSEDED (2026-10-21, later same day) by
> `docs/physical_pitch_orientation_validation.md`.** This report's
> *internal-consistency* findings (point-in-polygon agreement, goal/
> kickoff geometry, timing-cap behavior) remain correct and unchanged.
> However, its broader claim of "no orientation bug" did not test
> against real broadcast footage, only internal self-consistency — a gap
> this report itself flagged in its own §6 ("what remains uncertain").
> A follow-up investigation using an actual broadcast screenshot found
> that the *default* team-normalization pick (`resolve_reference_team_id`'s
> automatic, broadcast-unverified rule) places Croatia's goal on the
> opposite side of the screen from the real camera for this match. See
> the superseding report for the root cause, the fix (an opt-in
> `--physical-reference-team` override), and the corrected GIF. Kept here
> unmodified for the historical record of what internal-consistency
> testing alone can and cannot establish.

Date: 2026-10-21. Diagnosis only — **no code, visualization, or correction
heuristic was modified**, and nothing was committed. Every number below was
recomputed directly from raw JSON using the project's existing, unmodified
functions (`src/orientation_correction.py`, `src/team_normalization.py`,
`src/team_identity.py`, `src/animate_statsbomb_360_match.py`) — none of the
three modules' logic was re-implemented for this diagnosis.

---

## 0. Match identity verification (confirmed fact)

The requested match ID, **3930160**, was checked against
`data/raw/statsbomb/data/matches/55/282.json` (loaded via the project's own
`load_match_metadata_index()`):

| Field | Value |
|---|---|
| Teams | Spain (home) vs Croatia (away) |
| Score | **3–0** |
| Competition / season | UEFA Euro 2024 / 2024 |
| Date | 2024-06-15 |

This is **not** the 2021 "Croatia vs Spain" Euro 2020 last-16 match
(`3794686`, also present locally) — the two are distinct fixtures and were
not confused. Both `data/raw/statsbomb/data/events/3930160.json` and
`data/raw/statsbomb/data/three-sixty/3930160.json` exist and loaded without
error via `load_and_join_match(3930160)` (3,253 usable joined frames, 0
load errors). **No rendered GIF for match 3930160 exists yet anywhere under
`results/`** — this diagnosis is based on direct recomputation from raw
records, not on inspecting an already-rendered image.

---

## 1. What is actually happening at 31:00–34:00 (confirmed fact)

Scanning `events/3930160.json`, period 1, seconds 1850–2060, directly:

| Time | idx | Team | Type | `location` | Has 360? |
|---|---|---|---|---|---|
| 00:31:24.988 | 1322 | Spain | **Shot** (`shot.outcome.name == "Goal"`) | `[103.9, 38.7]` | Yes |
| 00:31:25.047 | 1323 | Croatia | Block | `[14.6, 41.4]` | Yes |
| 00:31:25.821 | 1324 | Croatia | Goal Keeper (`Goal Conceded`) | `[2.5, 41.4]` | Yes |
| **00:32:33.968** | **1325** | **Croatia** | **Pass** (kickoff restart) | `[61.0, 40.1]` | Yes |

This is **Fabián Ruiz's goal for Spain** (confirmed: `shot.outcome.name ==
"Goal"`, scoring team = Spain), immediately followed by a **68.147-second
gap with zero recorded events** (goal celebration — a real, expected
StatsBomb recording gap, not missing data), then **Croatia's restart pass
from inside the center circle** (`[61.0, 40.1]` ≈ the center spot). This is
exactly the "Spain goal → kickoff" transition the task description refers
to, located and confirmed, not assumed.

---

## 2. Is there evidence of a problem in the raw data?

**No defect found. One genuine, large, verbatim raw-data discontinuity was
found and is the primary driver of the visual effect — this is real camera
behavior, not corrupted data.**

Raw `visible_area` (StatsBomb's own polygon, byte-for-byte, before any of
this project's code touches it) for the two frames straddling the 68-second
gap:

```
idx 1324 (Goal Keeper, pre-gap):  [0.0, 80.0, 0.0, 32.84, 10.69, 15.64, 28.95, 24.90, 10.48, 80.0, 0.0, 80.0]
idx 1325 (Pass/kickoff, post-gap): [23.70, 80.0, 44.90, 0.0, 76.57, 0.0, 97.45, 80.0, 23.70, 80.0]
```

Centroid: idx 1324 ≈ **(8.4, 52.2)** (deep in one defensive third, by the
goal). idx 1325 ≈ **(53.3, 48.0)** (center circle). **Neither frame is
transformed by any of this project's code**: both are classified `NORMAL`
by Problem A, and both events belong to Croatia, the match's
`resolve_reference_team_id` reference team, so Problem B's mirror is also a
no-op for both. The jump from x≈8 to x≈53 is present **verbatim in the raw
StatsBomb file**, confirmed by direct inspection — it is the broadcast
camera genuinely recentering on the restart after following the goal, not
an artifact introduced anywhere in this codebase.

---

## 3. Is there evidence of a problem in our interpretation or transformations?

**No bug found in `orientation_correction.py`, `team_normalization.py`, or
`team_identity.py` for this window, after direct numeric verification of
every usable event from idx 1310 to idx 1365 (34 events) and a full-match
consistency check.**

### 3.1 Direct check: is the event location inside the final camera polygon?

Using the real pipeline (`apply_correction` → `apply_team_normalization_to_event`
/ `apply_team_normalization_to_frame`, in the same order `main()` applies
them, `src/animate_statsbomb_360_match.py:2397-2407`), the point-in-polygon
test (`_point_in_polygon`, `src/orientation_correction.py:76-88`) was run on
every one of the 34 events in idx 1310–1365, including the goal (idx 1322)
and the restart (idx 1325):

**Result: `inside_polygon == True` for all 34/34 events in this window, with
no exceptions.** The full transcript (orientation, team-normalization
label, and polygon centroid at each stage) is reproducible via the commands
in §6.

### 3.2 Is this window exceptional, or is the pipeline this reliable everywhere?

Full-match point-in-polygon rates (same check, every usable frame):

| Match | Inside / Total | Rate |
|---|---|---|
| Spain vs Croatia (3930160) | 3,238 / 3,253 | **99.54%** |
| Belgium vs Morocco (3857283, the reference) | 3,075 / 3,089 | **99.55%** |

**These rates are statistically indistinguishable.** The Spain–Croatia
match is not measurably less reliable than the Belgium–Morocco match the
user confirmed looked correct — this is evidence *against* a
match-specific transformation defect.

### 3.3 Composite case: does a double transform (orientation `ROTATED_180` *and* team `MIRRORED`) ever land outside the polygon?

This is the one case worth checking explicitly, since the two corrections
compose in sequence (`src/animate_statsbomb_360_match.py:2397` then
`:2406-2407`): an event that is both `ROTATED_180` (Problem A mirrors its
frame once) **and** belongs to the non-reference team (Problem B mirrors
the *already-corrected* frame again) gets mirrored twice. Because
reflection through the pitch center is its own inverse
(`mirror_point(mirror_point(p)) == p`), this is mathematically expected to
remain consistent as long as Problem B also mirrors the event's own
`location` (it does — `apply_team_normalization_to_event`,
`src/team_normalization.py:115-137`), not just the frame.

Measured directly: **86/86** such composite cases in Spain–Croatia, and
**76/76** in Belgium–Morocco, land inside the final polygon. No exceptions
in either match. This confirms the two corrections compose correctly, not
just in this window but dataset-wide across both matches checked.

### 3.4 Is the classifier being used as intended?

Confirmed from code: `compute_orientation_results()`
(`src/animate_statsbomb_360_match.py:553-565`) calls
`classify_orientations_for_match()`
(`src/orientation_correction.py:321-343`), which runs the documented
two-pass pipeline (pass 1: `classify_orientation` geometry-only; pass 2:
`resolve_uncertain_with_neighbors` for still-`UNCERTAIN` cases only) — this
is the single shared implementation used by both the dataset builder and
the visualizer, exactly as the module's own docstring states. For Spain vs
Croatia: 3,070 `NORMAL`, 173 `ROTATED_180`, 10 `UNCERTAIN` (0 of the 10
rescued by pass 2's temporal-neighbor vote). **None of the 10 `UNCERTAIN`
frames fall within the idx 1310–1365 window** — the classifier is working
as designed and is not the source of the reported effect here.

### 3.5 Does team normalization "interact unexpectedly" with orientation correction here?

Yes, in one specific, **documented and intentional** sense: Spain's goal
(idx 1322, scoring team **Spain**) and Croatia's restart (idx 1325, acting
team **Croatia**) are on **opposite sides of the `REFERENCE`/`MIRRORED`
boundary** (`resolve_reference_team_id` → Croatia is the reference team for
this match). `team_normalization_label` therefore returns `MIRRORED` for
the goal event and `REFERENCE` for the restart event. By
`team_normalization.py`'s own documented design (lines 13-21: "a team's own
events are always expressed as if that team attacks toward increasing x...
two different teams' events are never on the same coordinate scale unless
one side is reflected"), **every single change of ball possession between
the two teams causes this same flip** — it is not unique to this window,
and it happens dozens of times per match (every time the acting team
changes). What makes this particular instance visually dramatic is not the
flip itself but that it **coincides with** the large, genuine raw-data jump
documented in §2 and the real-time hold-capping behavior in §3.6 — a
compounding of three independently-correct behaviors, not a fourth, hidden
bug.

### 3.6 A real rendering-amplifier, confirmed from code: the 68-second gap is compressed to 1.5 seconds on screen

`build_timing_plan()` (`src/animate_statsbomb_360_match.py:1725-1775`), in
`--timing-mode real-time`, computes each frame's GIF hold duration as:

```python
# src/animate_statsbomb_360_match.py:1766-1770
real_duration = 0.0 if gap_to is None else gap_to * 1000 / playback_speed
display_duration = round(
    min(max(real_duration, minimum_duration_ms), maximum_duration_ms)
)
```

with `--maximum-frame-duration-ms` defaulting to **1500** ms
(`src/animate_statsbomb_360_match.py:2265`). The real 68.147-second gap
between idx 1324 and idx 1325 is therefore **capped and displayed for only
1.5 seconds** before cutting straight to the restart frame. This turns a
real, continuous camera pan (which presumably happened over 68 real
seconds, captured by StatsBomb only at the two endpoints since no
in-between events exist) into an abrupt on-screen cut between two very
different camera positions **and** a team-perspective flip at the same
instant — a plausible, confirmed-from-code explanation for why this
specific transition looks like a sudden "jump to the opposite side," even
though every individual frame's geometry checks out (§3.1–3.3).

---

## 4. Comparison with the Belgium–Morocco reference window

| | Spain–Croatia, period 1, 31:00–34:00 | Belgium–Morocco, period 1, 18:00–18:30 |
|---|---|---|
| Content | Goal → **68 s event-free gap** → restart by the *other* team | Continuous back-and-forth play, no goal, no gap > ~2.5 s |
| Ball x-range (raw, own-team convention) | 103.9 (goal) → 2.5 (keeper) → 61.0 (restart) | 48–101, smooth, no team-restart discontinuity |
| Team-perspective flips (`REFERENCE`↔`MIRRORED`) | Present (as in every window) | Present (as in every window) |
| Point-in-polygon rate in the printed window | 34/34 (100%) | Not separately isolated, but full-match rate 99.55%, statistically equal to Spain–Croatia's 99.54% |
| Real-time hold capping relevant? | Yes — one 68 s gap capped to 1.5 s | No comparably large gap in this window |

**This does not prove the Belgium–Morocco window is bug-free and therefore
nothing is wrong anywhere** — it only shows that, by the one concrete,
reproducible test available (point-in-polygon consistency), the two windows
are not measurably different in correction quality. The *visual* difference
between them is best explained by a difference in football content (a goal
+ restart vs. continuous open play), not a difference in code behavior.

---

## 5. Answers to the five requested questions

1. **Evidence of a problem in the raw data?** No defect. The raw
   `visible_area` polygon genuinely jumps from the goal area to the center
   circle between idx 1324 and idx 1325 (§2) — confirmed verbatim in the
   source JSON, before any processing. This is real camera behavior
   (celebration → restart), not corruption.
2. **Evidence of a problem in our interpretation/transformations?** No bug
   found. Every event in the window (34/34) and every double-transform case
   dataset-wide (162/162 across both matches) is internally consistent
   (ball/actor location falls inside the final camera polygon). The
   `REFERENCE`/`MIRRORED` flip between the goal and the restart is
   intentional, documented behavior (§3.5), not a defect.
3. **Exact snapshots/event UUIDs demonstrating the issue**: `bcc2cfab-8318-4302-a2e3-910150d58e36`
   (idx 1324, Croatia Goal Keeper, pre-gap) and `26c5e9bc-9bda-47e4-b629-3aa9d2b09fee`
   (idx 1325, Croatia Pass, post-gap/kickoff) — the two frames bracketing
   the 68.147 s gap, both `NORMAL`/`REFERENCE` (untouched by any
   correction), whose raw `visible_area` polygons already differ exactly as
   described in §2.
4. **Most likely root cause, and confidence**: **high confidence** that the
   perceived "180° jump" is the compounding of three independently-correct
   behaviors, not a defect in any one of them: (a) a real, large raw-camera
   recentering already present in StatsBomb's own data (§2, confirmed
   fact); (b) the intentional `REFERENCE`/`MIRRORED` coordinate flip that
   occurs on every change of acting team (§3.5, confirmed fact, by design);
   (c) the real-time timing mode's 1.5-second display cap collapsing a real
   68-second gap into a single abrupt cut (§3.6, confirmed from code). No
   single-cause "bug" was identified; this is a **hypothesis about why it
   looks the way it does**, built on the confirmed facts above, not itself
   independently proven by watching the actual rendered GIF (see §6 for what
   would close that gap).
5. **Additional evidence needed to fully distinguish the causes**: this
   diagnosis never rendered the actual GIF/frames for match 3930160 — only
   recomputed the underlying geometry numerically. To confirm the visual
   impression directly, render the exact segment (`--match-id 3930160
   --period 1 --start-minute 31 --end-minute 34 --apply-orientation-correction
   --apply-team-normalization --apply-team-identity-resolution --view
   camera-overlay --theme dark --timing-mode real-time`) and inspect the
   frames at idx 1322, 1324, and 1325 directly (e.g. via
   `scripts/inspect_data_quality.py` or the GIF's own `_test_frame.png`
   output) to verify the rendered polygon matches the numeric centroids
   reported here. If the rendered frames match these numbers exactly (they
   should, since this diagnosis used the identical functions in the
   identical order), the remaining question is purely one of presentation
   (e.g. whether a visible on-screen indicator of the acting team / a
   slower minimum hold around large gaps would make this transition read as
   "the real game paused and restarted" rather than "the data flipped") —
   a design decision, not a data-correctness one, and explicitly out of
   scope for this diagnosis per the task's instructions.

---

## 6. Reproducing this diagnosis

```bash
# Confirm match identity:
.venv/bin/python -c "
import json
with open('data/raw/statsbomb/data/matches/55/282.json') as f:
    for m in json.load(f):
        if m['match_id'] == 3930160:
            print(m['home_team']['home_team_name'], m['home_score'], '-', m['away_score'], m['away_team']['away_team_name'])
"

# Recompute the orientation/team-normalization trace for idx 1310-1365
# (same code shown in this report) using the project's own functions:
.venv/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from src.animate_statsbomb_360_match import load_and_join_match, compute_orientation_results
from src.orientation_correction import apply_correction, extract_geometry
from src.team_normalization import resolve_reference_team_id, team_normalization_label, apply_team_normalization_to_frame, apply_team_normalization_to_event
match = load_and_join_match(3930160)
orientation_results = compute_orientation_results(match)
ref_id = resolve_reference_team_id([j.event for j in match.usable_frames])
for joined, orient in zip(match.usable_frames, orientation_results):
    ev = joined.event
    if ev['period'] == 1 and 1310 <= ev['index'] <= 1365:
        print(ev['index'], ev['team']['name'], ev['type']['name'], orient.orientation, orient.confidence)
"
```

No raw data, processed dataset, or production code was modified while
producing this report.
