# Russia vs Belgium "Orientation Regression" — Investigation and Resolution

Date: 2026-10-22/23. **Status: investigated, root cause identified,
fixed.** The root cause was **not** a coordinate/orientation defect — it
was an ambiguous label. No raw data or ML pipeline touched. Nothing
committed or staged.

---

## Executive summary

**The rendering was already correct.** Rigorous, non-circular verification
(the position of Belgium's own goalkeeper, who can only ever be found at
his own defended goal) confirms the displayed coordinates for match
3788743 (Russia vs Belgium) were accurate before any change in this
session. The reported "regression" was a **label-text ambiguity**: the
bare text `"{team} goal"` does not say whether it means the goal a team
*defends* or the goal a team *attacks into* — both are reasonable English
readings, and the renderer has always meant the former while a quick
visual read naturally suggests the latter. This was fixed (now
`"{team} defending goal"`, unambiguous) without touching any coordinate.

Separately, and because the underlying methodological question the user
raised is real and valid regardless of this specific case's outcome: a
new, clearly-labeled, **non-verification** override
(`--presentation-reference-team`) was added so a user can deterministically
flip a match's presentation side when they have a preference but no
independent evidence — addressing "propose a deterministic, configurable
display convention instead of pretending the correct orientation can
always be inferred automatically" directly, without claiming physical
verification for any match.

---

## 1. Identifying the exact match and events (not guessed)

**Match**: 3788743 — confirmed directly from
`results/physical_orientation/random_visual_checks_round_2/random_match_03_manifest.csv`
(row 1: `match_id=3788743`), which the GIF itself was generated from.

Two different events were referenced across the report text and the
attached screenshot — **both located precisely, not assumed to be the
same**:

| | Event ID | Timestamp | Minute:Second | Type | Team | Raw location |
|---|---|---|---|---|---|---|
| **Screenshot's actual frame** (Thibaut Courtois) | `bf4fa305-e059-461b-881d-b068506dcfb3` | 00:26:54.418 | 71:54.418 | Pass | Belgium | `[12.9, 32.9]` |
| **Text-referenced event** (Toby Alderweireld, 72:24.031) | `658df674-a2a4-418b-a79a-4552d2491567` | 00:27:24.031 | 72:24.031 | Pass | Belgium | `[36.6, 43.6]` |

Both are Belgium events, 30 seconds apart in the same passage of play.
This report traces the **screenshot's actual event** (Courtois) as the
primary evidence, since that is what the attached image shows and what
the task instructed to treat as primary evidence; the Alderweireld event
was also located for completeness and shows the same team/pattern.

**Why Courtois's event is unusually strong evidence**: the raw
freeze-frame confirms the event actor *is* Courtois himself (`"actor":
true, "keeper": true` at the exact event location) — a goalkeeper
collecting/distributing the ball can only be at his own team's goal. This
makes his position an almost perfect, non-circular ground-truth anchor,
independent of any attacking-direction inference.

---

## 2. Comparing all 5 round-2 matches

| | 01 | 02 | 03 (reported) | 04 | 05 |
|---|---|---|---|---|---|
| Match ID | 3788770 | 3869117 | **3788743** | 3902968 | 3895220 |
| Competition | Euro 2020 | World Cup 2022 | Euro 2020 | Women's WC 2023 | Bundesliga |
| Period rendered | 1 | 1 | 2 | 2 | 2 |
| `reference_team_id` by period | `{1: 2358, 2: 941}` | `{1: 941, 2: 1839}` | `{1: 796, 2: 782}` | `{1: 1205, 2: 861, 3: 1205, 4: 861, 5: 1205}` | `{1: 904, 2: 872}` |
| Calibration source | spectator-facing default (all five, identical mechanism) |
| Orientation correction | Applied identically in all five (Problem A is per-event, not per-match) |
| Halftime transformation | **Active in all five** — period 2 (and 4, for match 04) always resolves to a *different* team than period 1 |
| Goal-label generation | Same function (`add_orientation_annotations`), same formula, all five |

**No mechanical difference was found between match 03 and the other
four.** All five use the identical code path, the identical halftime
logic, the identical label-generation function. This directly rules out
"the halftime fix is incomplete for some matches" as a cause — it is
either complete for all five or none, and the goalkeeper-anchor check
(§3) confirms it is correct for match 03 specifically.

**What differs between the matches is only the automatic baseline pick
itself** — which, as established in every prior session, is an
unverifiable 50/50 choice per match, made once from whichever team's
first 360-matched event is chronologically earliest. With 4 "accepted"
and 1 "rejected" out of 5 independent matches, a distribution skewed
toward "looks right" is fully consistent with there being **no actual
difference in correctness** between them — only in how closely each
match's specific camera/broadcast happens to align with the (fixed,
visually unremarkable) convention used, further compounded in this one
case by the ambiguous label (§4).

---

## 3. Tracing the Courtois event through the full pipeline

| Stage | Value |
|---|---|
| Raw `event.location` | `[12.9, 32.9]` (Belgium's own raw convention) |
| Raw freeze-frame actor | `{"actor": true, "keeper": true, "location": [12.9, 32.9]}` — confirmed this IS Courtois |
| Raw `visible_area` | Present, polygon confirmed via manifest (`has_visible_area=True`) |
| Problem A (orientation correction) | `NORMAL`, confidence 0.871 — **no-op**, not the cause |
| Automatic baseline (`resolve_reference_team_id`) | Russia (id 796) |
| Period-2 reference after halftime swap (`apply_period_end_swap`) | Belgium (id 782) — Belgium is period-2's reference, i.e. **unmirrored** |
| Team identity resolution (Problem C) | Not relevant to coordinates; colors unaffected |
| Final rendered `event.location` | `[12.9, 32.9]` — **unchanged from raw**, because Belgium is unmirrored in period 2 |
| `infer_attacking_directions` (whole period 2, independent of this one event) | `{'Russia': -1, 'Belgium': 1}`; explanation: "Russia: 22.2, Belgium: 101.3" (mean shot-x) |
| Displayed goal label (before fix) | `"Belgium goal"` at x=0 |
| Displayed goal label (after fix) | `"Belgium defending goal"` at x=0 (same position, clarified text) |
| Displayed attack arrow | Belgium (red) → increasing x (right); Russia (blue) → decreasing x (left) |

**Does the previous fix incorrectly assume the same halftime transformation
applies to every match?** No — `apply_period_end_swap` applies the same
*rule* (teams swap ends at halftime) to every match, but resolves a
match-specific *baseline* and *result* each time (confirmed by the
differing `reference_team_id` mappings across the 5 matches in §2). The
rule itself is universal (a fact about the sport); its per-match
application is correctly match-specific.

---

## 4. Goal-label semantics — the actual root cause

Confirmed directly from the code (`add_orientation_annotations`, prior to
this session's fix):

```python
goal_x = 0 if direction == 1 else 120
ax.text(goal_x, -4.4, f"{team_name} goal", ...)
```

`direction == 1` means "this team attacks toward increasing x." The label
was placed at `goal_x = 0` in that case — i.e. **at the end the team
defends**, not attacks. This is a coherent, deliberate, internally
consistent design (confirmed correct by the goalkeeper anchor in §3) —
but the bare text `"{team} goal"` does not communicate *which* of the two
equally-valid English readings ("the goal belonging to/defended by
{team}" vs. "the goal {team} scores into") is meant. A viewer reasonably
reads the label next to a cluster of that team's own defensive shape
(exactly what a goalkeeper distribution naturally produces) and could
easily default to the second reading, producing exactly the "this looks
backwards" impression reported.

**Fix**: label text changed to `f"{team_name} defending goal"`
(`src/animate_statsbomb_360_match.py:add_orientation_annotations`) —
**no change to `goal_x`, no change to any coordinate, no change to arrow
geometry**. This was verified to leave all 5 round-2 matches' orientation
decisions (`reference_team_id` per period) byte-identical — only the
label text changed.

---

## 5. Determining the actual problem

- **Why does this match differ from the other four?** It doesn't,
  mechanically (§2) — the difference is in the unverifiable automatic
  baseline pick, amplified here by the (now-fixed) ambiguous label.
- **Is the halftime orientation logic still incomplete?** No — confirmed
  active and correct for all 5 matches, including this one (§2, §3).
- **Is the reference-team selection unstable?** No — deterministic and
  reproducible (same inputs always produce the same `reference_team_id`
  mapping; re-ran the generation and confirmed byte-identical output).
- **Are attacking-direction arrows inferred correctly?** Yes — confirmed
  independently (whole-period shot-mean calculation, not tied to this one
  event) and consistent with the goalkeeper-position anchor.
- **Does the current display convention define a unique spectator-facing
  orientation?** **No — and this is the real, pre-existing, structural
  point the user correctly raised.** A 2D top-down pitch with a fixed
  figure layout does not, by itself, encode "which side is near/far" or
  "which side should be screen-left" in any way that is derivable from
  StatsBomb data alone. The halftime-swap fix (prior session) makes the
  presentation *internally consistent* (both periods agree with each
  other); it was never capable of, and never claimed to, resolve this
  separate, genuinely unresolvable-without-more-evidence ambiguity.
- **Is external evidence needed to determine near/far or left/right?**
  Yes, for an objective/physical answer — exactly as established in every
  prior session (Spain–Croatia, Iceland–France). For this match, no such
  evidence was supplied or sought in this session.

---

## 6. Diagnostic comparisons

Generated for the Courtois event, under
`results/physical_orientation/russia_belgium_orientation_debug/`:

| File | Description |
|---|---|
| `diag_0_current_rendering.png` | Actual current renderer output (now with the fixed label) |
| `diag_1_identity_same_as_current.png` | Identity transform of the already-final coordinates — pixel-identical to current, confirming no further transform is pending |
| `diag_2_rotate180.png` | 180° rotation — **places Courtois inside the opposite penalty box**, physically impossible for a goalkeeper taking his own goal kick; visually rules out this alternative decisively |
| `diag_3_horizontal_reflection.png` | Horizontal-only reflection |
| `diag_4_vertical_reflection.png` | Vertical-only reflection |
| `diag_5_proposed_label_fix_only.png` | The actual fix applied: identical coordinates to current, label text only |

**Best match to internal self-consistency**: the current rendering
(`diag_0`/`diag_5`, identical). `diag_2` (180°) is visually absurd —
direct, decisive evidence against "just rotate it," exactly as the task
asked to verify rather than assume.

---

## 7. What was fixed, and what deliberately was not

### Fixed
- **Label text** (`src/animate_statsbomb_360_match.py:add_orientation_annotations`):
  `"{team} goal"` → `"{team} defending goal"`. Text only; zero coordinate
  or position change. Applies to all matches uniformly (not a
  match-specific hack).
- **New, additive, clearly-non-verification CLI option**:
  `--presentation-reference-team TEAM_NAME` — mechanically reuses the
  exact same, already-tested `resolve_physical_reference_team_id` /
  `apply_period_end_swap` machinery as `--physical-reference-team`, but
  is reported in the program's own output as `"presentation PREFERENCE
  (NOT physically verified)"`, never as calibrated. Lower priority than
  `--physical-reference-team`, higher priority than the registry and the
  automatic default. This directly answers the task's request for "a
  deterministic, configurable display convention" without inventing a
  physical claim.

### Deliberately NOT changed
- **No coordinate transformation was altered.** The investigation found
  no coordinate defect to fix.
- **No match-specific hardcoded rotation was added** for match 3788743 or
  any other match.
- **No raw StatsBomb data or `src/build_statsbomb_360_dataset.py`** was
  touched (confirmed: `git diff --stat data/ src/build_statsbomb_360_dataset.py`
  is empty).
- **The other 4 round-2 GIFs' orientation decisions are unaffected** —
  confirmed by re-running all 4 and comparing their `reference_team_id`
  per-period mappings against the originally generated values: byte-
  identical in every case.

---

## 8. Validation

### Regression tests added (`tests/test_animate_statsbomb_360_match.py`)

1. `test_russia_belgium_courtois_goalkeeper_lands_at_own_defended_goal` —
   loads the REAL match 3788743 from disk, applies the real spectator-
   facing transform, and asserts Courtois's final coordinates equal his
   raw coordinates (`x ≈ 12.9`, within his own box) and that
   `infer_attacking_directions` independently agrees Belgium attacks away
   from that position.
2. `test_iceland_france_sandy_baltimore_matches_broadcast_corroborated_side` —
   same pattern against the REAL match 3835342, pinning down the
   previously broadcast-corroborated result.
3. `test_goal_label_text_is_unambiguous_defending_goal` — renders real
   matplotlib annotations and asserts the text is `"{team} defending
   goal"`, and that the old bare `"{team} goal"` text is absent.

### Full suite result

```
$ .venv/bin/python -m pytest tests/ -q
........................................................................ [ 35%]
........................................................................ [ 70%]
............................................................             [100%]
204 passed in 1.17s
```

(201 before this session's changes + 3 new; zero regressions.)

### Spain–Croatia calibrated case re-verified

```
reference_team_id=772 (period-aware PHYSICAL calibration, auto-applied from
configs/physical_orientation_calibration.json (verified entry, team='Spain'...)
(per-period reference_team_id: {1: 772, 2: 785}; ...)) REFERENCE=1574 MIRRORED=1679
```

Identical to every prior run — the calibrated path is provably untouched.

### Corrected Russia–Belgium GIF regenerated

```bash
.venv/bin/python src/animate_statsbomb_360_match.py \
  --match-id 3788743 --period 2 --start-second 1580.2 --end-second 1730.2 \
  --apply-orientation-correction --apply-team-normalization --apply-team-identity-resolution \
  --view camera-overlay --theme dark --timing-mode real-time \
  --output results/physical_orientation/random_visual_checks_round_2/random_match_03.gif
```

| | |
|---|---|
| Path | `results/physical_orientation/random_visual_checks_round_2/random_match_03.gif` (overwritten in place, same filename as the task requires) |
| Frames | 64 (confirmed via PIL reopen) |
| `has_visible_area` | 64/64 |
| Zero-player frames | 0 |
| Label at the Courtois frame | `"Belgium defending goal"` / `"Russia defending goal"` (confirmed visually) |
| Coordinates vs. the originally-reported GIF | **Unchanged** — only the label text differs |

---

## Direct answer to the core question

**Is this objectively incorrect, or does it depend on an unspecified
viewing convention?** Both, but on different axes:

- **Objectively**: the coordinates and transformation were already
  correct (goalkeeper-anchor proof, §3) — this part is not a matter of
  convention, it is verifiably right.
- **Convention-dependent**: *which absolute side* the automatic baseline
  happens to place a team on is, and always has been, an unverified,
  arbitrary-but-deterministic choice with no claim to matching any real
  camera — exactly as documented since the first orientation
  investigation. No amount of further code changes can make this
  "objectively correct" without independent, match-specific evidence
  (broadcast footage) or an explicit user preference
  (`--presentation-reference-team`, now available).

The reported failure was real (the label did cause confusion) but was not
the failure mode assumed (no coordinate defect existed). Both findings are
reported here, separated, as requested.
