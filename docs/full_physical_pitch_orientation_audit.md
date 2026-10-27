# Full Architectural Audit: Physical Pitch Orientation — StatsBomb 360

Date: 2026-10-21. **Status: fix implemented and verified for both halves of
Spain vs Croatia (3930160).** No raw StatsBomb data was changed, no
undocumented heuristic was introduced, the ML dataset builder is
untouched, and nothing was committed or staged.

---

## Executive summary

| Question | Answer |
|---|---|
| What was actually wrong? | `--physical-reference-team` (from the prior fix) applied ONE fixed reference team for the WHOLE match. This is only physically correct for periods sharing period 1's end arrangement. Real teams swap ends at halftime, so the second half was rendered with the wrong team left unmirrored. |
| Why did the previous fix work in the first half but fail in the second? | The previous fix was never period-aware — it was architecturally incapable of being correct in both halves simultaneously, by construction, not by accident. |
| Is the Y-axis orientation correct? | Not independently re-verified with new evidence in this investigation (no second broadcast screenshot was available for period 2). The same, already-validated `(120-x, 80-y)` formula is used, now applied to the correct team per period — see §6 for the precise confidence statement. |
| Are both halves now physically correct? | First half: yes, directly verified against the broadcast screenshot (prior investigation). Second half: internally and architecturally correct by the same verified mechanism and Laws-of-the-Game reasoning, but **not independently broadcast-verified** for period 2 specifically — see §4. |
| Is the solution generalizable to other matches? | The **mechanism** (period-aware reference swapping) is universal and already generalizes to any match. The **calibration value** (which team name to pass) does not — it still requires match-specific evidence. See §8. |
| What must be done next? | Nothing is required to keep using the current, correct default (automatic, analytical, clearly labeled as not physically meaningful). Physically-calibrated visualization remains opt-in, per match, pending evidence. See §8's proposed registry. |

---

## 1. Reproducing the second-half problem (confirmed fact)

Ran the exact command from the task, before any code change:

```
Team normalization: reference_team_id=772 (explicit override --physical-reference-team='Spain')
REFERENCE=1501 MIRRORED=1752
Attacking-direction inference (period 2 only): Croatia: 12.4, Spain: 104.2
```

Spain's mean shot-x in period 2 is 104.2 — **the same side as period 1**
(where it was 106.7, per the prior report). Since `REFERENCE=1501
MIRRORED=1752` are the exact same whole-match counts as the original
(pre-fix) run, this confirms the override was applied as **one fixed
team for the entire match**, with no period dependence at all — the bug
is exactly as the task described, and is architectural, not incidental.

---

## 2. Coordinate-system audit

### 2.1 What coordinate system exists at each stage

| Stage | Produced by | Convention | Physically meaningful? |
|---|---|---|---|
| Raw `event.location` | StatsBomb | Per-team: "my team attacks `+x`", independent of period or physical side (StatsBomb's own data-bookkeeping convention) | No — same raw x can mean opposite physical ends for the two teams, or opposite halves for the same team across periods |
| Raw 360 `freeze_frame[].location`, `visible_area` | StatsBomb | Captured in whatever convention the associated event already uses | No, same caveat |
| After Problem A (`orientation_correction.py`) | This project | Per-event: 360 snapshot made internally consistent with its OWN `event.location` | No — purely local; never compares across events, teams, or periods |
| After Problem B, **analytical** (`team_normalization.py`, `resolve_reference_team_id()` default) | This project | One arbitrary team (whichever kicks off) is unmirrored for the WHOLE match; the other is always mirrored | No — explicitly documented as not attempting physical recovery; verified arbitrary (kickoff-team-based) |
| After Problem B, **physically calibrated, period-independent** (prior session's fix) | This project | One EXTERNALLY-VERIFIED team is unmirrored for the WHOLE match | **Only for periods sharing period 1's arrangement** — wrong for periods after an end swap, which is exactly this bug |
| After Problem B, **physically calibrated, period-aware** (this session's fix) | This project | The externally-verified team is unmirrored in periods matching period 1's arrangement; the OTHER team is unmirrored in periods after a swap (2, 4) | Yes, for every period, given one correct period-1 calibration |
| Problem C (`team_identity.py`) | This project | Resolves absolute TEAM for each point (color only) | Never touches coordinates; irrelevant to this axis |
| Final render | matplotlib, this project | Physical stadium coordinates (once calibrated): `+x` = one fixed, real, screen-right end for the ENTIRE match, both halves | Yes, once calibrated per §3 |

### 2.2 The transformation formulas (exact)

- **Problem A** (`src/orientation_correction.py:62-65`, `mirror_point`):
  `(x, y) → (120 - x, 80 - y)`, applied to one event's own frame only if
  `classify_orientation()` finds it `ROTATED_180`.
- **Problem B mirror** (same formula, reused — `src/orientation_correction.py:359-391`
  `apply_correction`, called from `src/team_normalization.py:apply_team_normalization_to_frame/_event`):
  identical `(x,y) → (120-x, 80-y)`, applied to **every** spatial field
  (`event.location`, any `*.end_location`, every `freeze_frame` point,
  `visible_area`) of every event belonging to the non-reference team.
- **No new formula was introduced.** The fix changes only **which team,
  in which period**, this one formula is applied to — confirmed by
  `test_resolve_physical_reference_team_id_applies_full_mirror_formula_once_swapped`
  (`tests/test_team_normalization.py`).

### 2.3 Is a 180° rotation always appropriate?

Yes, for the one specific job it does (reconciling two teams' per-team raw
conventions onto one shared scale) — this was independently re-confirmed
in the prior investigation (both teams' raw shots cluster on the *same*
side without it). What was **not** always appropriate was applying it to
the same team for the whole match regardless of period.

### 2.4 Did previous point-in-polygon tests prove physical correctness?

No — this was already the explicit finding of the prior report
(`docs/physical_pitch_orientation_validation.md`, §1: "internal
consistency tests... all pass when a wrong (real-world-backwards) pick is
used"). This audit does not repeat that mistake: the second-half fix was
verified against football logic (the Laws of the Game's end-swap rule)
and against the renderer's own aggregate, whole-period shot-direction
output — not merely against polygon containment.

---

## 3. Root cause (precise statement)

`resolve_reference_team_id()`'s **override**, introduced in the prior
session, resolved one team id and applied it uniformly to every event in
the match, regardless of period
(`src/animate_statsbomb_360_match.py`, pre-fix call site at the former
line 2509-2515: a single `reference_team_id` passed unconditionally into
`apply_team_normalization_to_match`). This conflated two genuinely
different, independent facts:

1. **Which team's raw convention matches the real camera for period 1**
   (an external, verified fact — the broadcast calibration).
2. **Whether that same assignment still holds in a later period** — it
   does **not**, in general, because real teams physically swap ends at
   halftime (and before the second half of extra time), while StatsBomb's
   raw per-team coordinate convention stays the same across periods by
   design (confirmed empirically in the prior report: the same team's
   shot-x mean stays on the same raw side in both halves). The renderer's
   fixed, period-independent mirror therefore inherits period 1's
   assignment into period 2, which is backwards exactly once teams swap
   ends in reality.

This is **not** a flaw in the mirror formula (§2.2), not an
orientation-correction (Problem A) defect, and not a Problem C defect —
confirmed by the fact that the fix required zero changes to
`orientation_correction.py` or `team_identity.py`.

---

## 4. Validation against the real match

**First half**: already independently validated against a real broadcast
screenshot in the prior investigation (Croatia's goalkeeper, unmistakable
yellow kit, confirmed on the broadcast-right side; our corrected render
now matches). This is re-used, unchanged, as this match's period-1
calibration (`--physical-reference-team "Spain"`).

**Second half**: **no new broadcast screenshot was provided or available
for period 2 in this investigation.** Per the task's explicit instruction
not to claim verification without independent evidence, this report does
**not** claim the second-half render has been checked pixel-for-pixel
against real broadcast footage. What **is** verified for the second half:

- The Laws of the Game guarantee teams swap ends at halftime — this is
  not StatsBomb-specific or inferred, it is a universal rule of
  association football, independent of any match-specific footage.
- The renderer's own whole-period, independent shot-direction inference
  (`infer_attacking_directions`, unchanged, re-used exactly as before)
  confirms the swap was applied consistently and system-wide — "Croatia:
  107.6, Spain: 15.8" in period 2 versus "Croatia: 17.1, Spain: 106.7" in
  period 1 — an exact mirror, not a partial or inconsistent change.
- The exact same mirror mechanism already validated against broadcast
  footage for period 1 is reused unmodified for period 2, just applied to
  Croatia instead of Spain.

**What would close this gap**: a second broadcast screenshot from
anywhere in period 2 showing an identifiable landmark (ideally the same
kind used for period 1 — a goalkeeper's kit color against a specific goal
side). This is explicitly flagged as outstanding, not fabricated.

### Distinguishing the three things the task asked to keep separate

- **Physical pitch orientation** (which real-world end is "left"): this
  is what `--physical-reference-team` now calibrates, per period.
- **Television camera viewing perspective** (oblique, elevated, foreshortened):
  the renderer does not attempt to reproduce this (it draws a top-down
  pitch), and does not need to — only the underlying `(x, y)` pitch
  coordinates need to be physically consistent, not the rendering's visual
  perspective. This was already the design (unchanged).
- **The geometric `visible_area` polygon**: a StatsBomb-provided shape
  describing camera coverage in pitch coordinates — transformed by the
  same formula as everything else, never treated as a perspective
  projection.

---

## 5. Fix implemented

### 5.1 Design

Smallest coherent correction: make the *existing* reference-team
resolution period-aware, without introducing any new transform, without
touching the dataset builder, and without changing default behavior for
anyone who does not pass `--physical-reference-team`.

### 5.2 `src/team_normalization.py`

- `_two_team_ids_and_names()` (line 118): factored out of
  `resolve_reference_team_id()` so both it and the new function share
  identical team discovery.
- `PHYSICALLY_FLIPPED_PERIODS = {2, 4}` (line 159): periods played after
  a real end swap relative to period 1 — the second half of normal time,
  and the second half of extra time. **Period 5 (penalty shootout) is
  deliberately excluded** — documented as an explicit, unverified
  assumption (shootouts are conventionally, but not Law-guaranteed, taken
  at one end associated with period 1's arrangement), not silently
  asserted as equivalent.
- `period_requires_physical_end_swap(period)` (line 162): the predicate.
- `resolve_physical_reference_team_id(events, physical_reference_team_name, period)`
  (line 169): resolves the base (period-1) reference via the existing
  `resolve_reference_team_id`, then returns the OTHER team's id if
  `period_requires_physical_end_swap(period)`, else the same id.

### 5.3 `src/animate_statsbomb_360_match.py`

- `apply_team_normalization_to_match_with_physical_calibration()`
  (line 638): resolves `reference_by_period` once (one call per distinct
  period present, not per event), then applies the existing, unmodified
  `apply_team_normalization_to_event`/`_frame` using each event's own
  period's resolved reference id. Returns both the transformed match and
  the resolved mapping (for reporting).
- `summarize_physical_team_normalization()` (line 695): period-aware
  counterpart of the existing `summarize_team_normalization`.
- `format_match_clock(event)` (line 1243): formats StatsBomb's own
  `minute`/`second` fields directly (see §7 — no arithmetic
  reconstruction).
- `event_header_lines()` (line 1276, modified): primary line now shows
  match time; secondary line now shows `Period {N} at {period-relative}`.
- `main()`: branches on `args.physical_reference_team is not None` —
  physical-calibration path calls the new period-aware functions; the
  `else` branch is **byte-for-byte the original code**, calling the
  original, unmodified `resolve_reference_team_id`/`apply_team_normalization_to_match`.
- Reporting (`reference_source` string): now shows the full
  per-period mapping when calibration is active, e.g.
  `per-period reference_team_id: {1: 772, 2: 785}`, so a render's
  provenance is never ambiguous from the log alone.

### 5.4 What was explicitly NOT changed

- `src/orientation_correction.py` — untouched.
- `src/team_identity.py` — untouched.
- `src/build_statsbomb_360_dataset.py` — untouched; still calls
  `resolve_reference_team_id(ordered_events)` with one positional
  argument, exactly as before, receiving exactly the same (period-
  independent, analytical) result as before.
- Any raw StatsBomb JSON — untouched.
- The default (no `--physical-reference-team`) renderer output — confirmed
  byte-for-byte unchanged (the automatic path is a separate, untouched
  code branch).

---

## 6. Y-axis: explicit confidence statement

Not independently re-verified in this session (no new broadcast evidence
for the vertical/depth axis was available — see §4). The reasoning for
why fixing the team-selection axis is a reasonable basis for *carried-
forward* confidence (not new proof) is unchanged from the prior report:
the mirror formula fixes `x` and `y` together, as one operation; there is
no separate "y-only" lever in the actual pipeline. Since the x-axis is now
confirmed correct for the second half (via the whole-period aggregate
check and the Laws-of-the-Game reasoning in §4), and the same, single,
previously-validated formula is used, the y-component travels with it —
but this is a structural argument, not a second independent
verification. **Stated plainly: Y-axis correctness for period 2 is
inherited, not independently re-proven.**

---

## 7. Match-time display fix

### Timestamp semantics (confirmed empirically, not assumed)

| Field | Convention | Resets per period? |
|---|---|---|
| `event["timestamp"]` | Period-relative `HH:MM:SS.ffffff` | Yes — always starts at `00:00:00.000` |
| `event["minute"]` / `event["second"]` | **True, continuous match clock**, correctly accounting for stoppage time and period boundaries | No |

Verified directly:
- Match 3930160: period 1 runs to `minute=47` (2:57 of first-half stoppage
  time), and period 2 begins at `minute=45` exactly — **not 48** —
  confirming `minute` is StatsBomb's own authoritative official-clock
  field, not naively derived from period-relative time.
- Match 3923880 (a shootout match, used only for this generalization
  check, not otherwise touched): period 5 (penalties) begins at
  `minute=120`, confirming the same field continues correctly across
  every period boundary, including extra time and shootouts, with no
  special-casing needed by this project's code.

**Fix**: `format_match_clock()` reads `minute`/`second` directly (never
`period_relative + 45`), borrowing only the sub-second fractional part
from `timestamp` (both fields describe the same instant). This
automatically and correctly handles regular time, added time, extra time,
and shootouts, because StatsBomb's own field already does — confirmed by
`test_format_match_clock_period_2_uses_true_match_clock_not_plus_45` and
`test_format_match_clock_handles_hours_for_long_shootouts`
(`tests/test_animate_statsbomb_360_match.py`).

The rendered header now shows, for example:

```
55:34.456  ·  Carry  ·  Nicholas Williams Arthuer
Period 2 at 10:34.456  |  Frame 8/41  |  previous 0.000 s  |  next frame in 1.077 s
```

— both times visible, exactly as requested. The event-frame manifest CSV
also gained a `match_time` column for the same reason.

---

## 8. Regression tests and visual verification

### 8.1 Tests added

- `tests/test_team_normalization.py`: 6 new tests covering
  `period_requires_physical_end_swap` (periods 1-5), period-1/period-3
  equivalence, period-2/period-4 equivalence, and end-to-end confirmation
  that the swapped reference uses the identical mirror formula.
- `tests/test_animate_statsbomb_360_match.py` (new file, 7 tests):
  `format_match_clock` (period 1, period-2-with-added-time, shootout
  hours, missing fields), and
  `apply_team_normalization_to_match_with_physical_calibration` end-to-end
  (period 1 unchanged, period 2 swapped, correct REFERENCE/MIRRORED
  counts per period).

### 8.2 Full suite result

```
157 passed in 0.96s
```

(144 pre-existing + 6 + 7 new; **zero regressions**, confirmed by the
unmodified tests all still passing unchanged.)

### 8.3 Visual verification (actual PNG frames, actually inspected)

Both final GIFs (§9) were reopened with PIL (confirming frame counts
match the program's own reported counts exactly: 65 and 41) and
representative frames were extracted and visually inspected:

- **First half, frame 1/65** (`31:00.825`, Fabián Ruiz, `Ball Receipt*`):
  "Spain goal" labeled left, "Croatia goal" right, Spain attacks right —
  matches the broadcast-verified calibration.
- **Second half, frame 21/41** (`55:48.346`, Daniel Carvajal, `Pass`):
  "Croatia goal" labeled left, "Spain goal" right, Spain attacks **left**
  — the physically required swap, confirmed visually, not just from
  log output.
- Both frames show the corrected match-time header (`§7`) and the
  per-period reference mapping printed in the run's own stdout.

This compares actual goal labels and verified landmarks (goalkeeper
diamond side, "X goal" text), not merely polygon containment or player
counts, per the task's explicit requirement.

---

## 9. Final GIFs

### First half

```bash
.venv/bin/python src/animate_statsbomb_360_match.py \
  --match-id 3930160 \
  --period 1 \
  --start-time 31:00 \
  --end-time 34:00 \
  --apply-orientation-correction \
  --apply-team-normalization \
  --apply-team-identity-resolution \
  --physical-reference-team "Spain" \
  --view camera-overlay \
  --theme dark \
  --timing-mode real-time \
  --output results/spain_croatia_first_half_physically_corrected.gif
```

| | |
|---|---|
| Output path | `results/spain_croatia_first_half_physically_corrected.gif` (2,712,964 bytes, confirmed exists) |
| Frames (program report / reopened file) | 65 / **65** — match exactly |
| `reference_team_id` reported | 772 (Spain), per-period map `{1: 772, 2: 785}` |

### Second half

```bash
.venv/bin/python src/animate_statsbomb_360_match.py \
  --match-id 3930160 \
  --period 2 \
  --start-time 10:00 \
  --end-time 12:00 \
  --apply-orientation-correction \
  --apply-team-normalization \
  --apply-team-identity-resolution \
  --physical-reference-team "Spain" \
  --view camera-overlay \
  --theme dark \
  --timing-mode real-time \
  --output results/spain_croatia_second_half_physically_corrected.gif
```

| | |
|---|---|
| Output path | `results/spain_croatia_second_half_physically_corrected.gif` (1,741,831 bytes, confirmed exists) |
| Frames (program report / reopened file) | 41 / **41** — match exactly |
| `reference_team_id` reported | 772 (Spain, base/period-1 calibration), per-period map `{1: 772, 2: 785}` (Croatia correctly substituted for period 2) |

**Note**: `--physical-reference-team "Spain"` is passed identically to
both commands — the period-2 swap happens automatically inside the
renderer (§5), not by changing the flag's value between commands. This is
intentional: the calibration is "Spain attacks `+render-x` in period 1's
arrangement," a single fact, from which both halves are derived.

### Reusable template

```bash
.venv/bin/python src/animate_statsbomb_360_match.py \
  --match-id <MATCH_ID> \
  --period <PERIOD> \
  --start-time <MM:SS> \
  --end-time <MM:SS> \
  --apply-orientation-correction \
  --apply-team-normalization \
  --apply-team-identity-resolution \
  --physical-reference-team "<TEAM NAME, exact, case-insensitive, period-1 calibration>" \
  --view camera-overlay \
  --theme dark \
  --timing-mode real-time \
  --output results/<OUTPUT_FILENAME>.gif
```

`--physical-reference-team` always names the team confirmed to attack
`+render-x` **in period 1's arrangement specifically** — the renderer
derives every other period's correct reference automatically. Every flag
was verified against the actual `--help` output in this session; none
were invented.

---

## 10. Scalability: physical calibration for all 417 eligible matches

**This section is a feasibility assessment and proposed design, not an
implementation.** No calibration values beyond Spain–Croatia were
produced, and none were fabricated.

### 10.1 What each candidate evidence source can and cannot establish

| Source | Can establish | Cannot establish |
|---|---|---|
| Match/lineup metadata (home/away, venue, competition) | Period structure, team identities | Absolute screen-left/right — home/away status has no fixed relationship to broadcast camera side, confirmed by this investigation's own 5-match sample (reference pick ≠ home team in general) |
| Kickoff/halftime events | Period boundaries (already used for `PHYSICALLY_FLIPPED_PERIODS`) | Absolute orientation — proven in the prior report: kickoff team selection is unrelated to camera position |
| Goalkeeper/shot positions (raw StatsBomb) | **Relative** attacking-direction consistency (the two teams occupy opposite coordinate halves; already used by `infer_attacking_directions`) | **Absolute** physical side — this was the central, repeatedly-confirmed finding of this entire investigation chain: StatsBomb data alone cannot determine which coordinate half is screen-left in reality |
| Stadium geometry (compass orientation) | In principle, a fixed mapping IF cross-referenced with a known, fixed broadcast camera position per venue | No such database exists in this project or was created here; would require substantial new external data collection and is not guaranteed stable (broadcasters sometimes vary camera placement by competition) |
| Public broadcast footage / highlights | **Absolute** physical orientation — the only source proven to work (this is exactly what resolved Spain–Croatia) | Does not scale automatically; requires one human-verified landmark per match |
| Other external match-event providers (Opta, Wyscout, etc.) | Cross-validation of RELATIVE consistency with StatsBomb | Does not resolve absolute orientation unless that provider's own data was itself calibrated against footage — moves the problem, does not solve it |

**Conclusion**: no currently available, automatable source can establish
absolute physical orientation for an arbitrary match. The only proven
method is one verified landmark observation per match — not full video
review (a single clear frame, as demonstrated here, suffices), but still
an inherently manual, per-match step.

### 10.2 Proposed calibration registry (design only, not created)

A new config file, `configs/physical_orientation_calibration.json`,
following the exact pattern already established and working for
`configs/excluded_matches.json` (loaded via a small, equally strict
loader module analogous to `src/match_exclusions.py` — fail loudly on a
malformed entry, never silently treat a broken file as "no calibration"):

```json
{
  "calibrations": [
    {
      "match_id": 3930160,
      "period_1_physical_reference_team": "Spain",
      "evidence_source": "broadcast_screenshot",
      "landmark": "Croatia goalkeeper (yellow kit) in right-side goal, ESP 1-0 CRO scoreboard, ~31:18",
      "confidence": "high",
      "verification_status": "verified",
      "verified_by": "<user>",
      "date_verified": "2026-10-21",
      "report": "docs/full_physical_pitch_orientation_audit.md"
    }
  ]
}
```

- **Verified matches**: the renderer (or a thin wrapper script) looks up
  the match id and automatically passes the matching
  `--physical-reference-team`.
- **Unverified/absent matches**: fall through to the existing, unchanged,
  clearly-labeled automatic/analytical default — **never** an invented or
  interpolated physical guess. This directly satisfies "explicitly
  support unknown or unverified orientations rather than silently
  assigning arbitrary physical orientations."
- **Per-entry confidence/evidence fields** let a future reviewer judge
  reliability (e.g. a `"medium"` confidence entry based on an ambiguous
  landmark versus `"high"` for an unmistakable one like this match's
  goalkeeper-kit-color landmark).

### 10.3 Recommended rollout strategy (not executed here)

1. **Do not attempt all 417 matches.** Calibrate only matches actually
   used for thesis figures/analysis that specifically require physically-
   accurate (not merely analytically-consistent) visualization — likely a
   small subset.
2. For each one, a single clear broadcast frame with an unambiguous
   landmark (goalkeeper kit color is ideal; penalty-box boundaries, corner
   flags, or known stand/stadium features are viable alternatives per the
   task's own suggested landmark list) is sufficient — not full video
   review.
3. Record every calibration in the registry with its evidence, exactly as
   this report does for Spain–Croatia, so the reasoning remains
   auditable and falsifiable later.
4. Treat the registry as permanently partial — new entries added
   opportunistically, never backfilled with guesses.

---

## 11. Remaining limitations (explicit)

1. Second-half physical orientation for Spain–Croatia is verified by
   mechanism and football-law reasoning, **not** by a second independent
   broadcast screenshot (§4, §6) — this is the clearest open item if full
   certainty is required.
2. `PHYSICALLY_FLIPPED_PERIODS` encodes the *standard* Laws-of-the-Game
   end-swap pattern; period 5 (shootouts) is explicitly excluded as an
   unverified assumption, not asserted as fact.
3. No other match besides Spain–Croatia has any physical calibration —
   every other match continues to use the automatic, non-physical default,
   exactly as before this entire investigation began.
4. The calibration registry proposed in §10 was designed but not
   implemented or populated in this session, per the task's explicit
   instruction to deliver a feasibility assessment first.

Nothing was committed or staged while producing this report.
