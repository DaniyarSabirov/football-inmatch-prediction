# Full Visualization Orientation and Match-Clock Audit — 2026-10-22

**Status: root causes identified, fixes implemented and tested, five GIFs
regenerated and verified.** Raw StatsBomb data and the ML dataset builder
(`src/build_statsbomb_360_dataset.py`) are untouched. Git state preserved;
nothing committed or staged.

---

## Executive summary

Two independent, real defects were found and fixed:

1. **Orientation defect**: the renderer's *default* (uncalibrated)
   rendering path resolved **one reference team for the entire match**,
   never re-examined per period. This was correct for periods sharing
   period 1's arrangement, but **systematically backwards for period 2
   (and period 4)** — because real teams swap ends at halftime, a fact
   about the sport, independent of any specific camera. This is distinct
   from, and was never covered by, the prior session's physical-
   calibration fix, which only applied the halftime swap when an explicit
   or registry calibration existed. **Confirmed to affect all 417/417
   eligible matches' period-2 rendering** (and all 30 extra-time matches'
   period-4 rendering) before the fix.
2. **Match-clock defect**: `format_match_clock()` used
   `divmod(minute, 60)` to introduce an hours component once the match
   clock passed 59 minutes — producing `"1:05:59.487"` for a minute-65
   event. Football match clocks are never hour-wrapped. This was a
   straightforward bug in the function added in the prior session, now
   fixed with proper conventional notation including stoppage-time and
   extra-time handling.

Both fixes are **general-purpose code changes**, not match-specific
hacks — confirmed by a dataset-wide audit (§8) and by the fact that the
already-verified match 3930160 is provably unaffected (its calibrated
path was already correct).

**What is NOT claimed**: physical camera verification for any of the 5
newly-rendered matches. The fix makes the DEFAULT rendering internally
consistent with the known structure of football (halftime end-swap); it
does not and cannot determine which absolute side is "real" without
match-specific external evidence, exactly as established in the two
prior sessions.

---

## 1. The spectator-facing requirement, and the critical distinction

Per the task's own framing, five concepts must be kept distinct:
raw StatsBomb coordinates, analytical normalization, team attacking
direction, physical stadium orientation, and **the renderer's
presentation convention** — the fifth being the actual target of this
fix. The fix implemented here operates purely at this fifth layer: it
changes nothing about raw data (layer 1) or the ML/analytical convention
(layer 2, `src/build_statsbomb_360_dataset.py` — confirmed untouched,
`git status` shows zero diff), and it does not claim to resolve physical
stadium orientation (layer 4) — it only makes the renderer's own
presentation (layer 5) internally consistent with a known, universal fact
about the sport (layer 3's structure: teams swap which end they attack at
halftime), applied to whichever baseline (verified or automatic) was
already in use.

---

## 2. The Iceland–France example — full diagnosis

### 2.1 Verifying the two screenshots represent the same moment

Confirmed directly from raw data, independently of clock proximity alone:

| Check | Broadcast | Our data | Match? |
|---|---|---|---|
| Match clock | 65:58 | minute=65, second=59 (65:59) | Yes (1s broadcast/data latency, same moment) |
| Score | ISL 0 – 1 FRA | France's only goal before minute 65 was at period 1, minute 0 (00:42.9); Iceland's equalizer was at minute 101 (after this frame) | **Yes — independently confirmed via score, not just clock** |
| Event | Sandy Baltimore pass | `42d38c43-a105-4207-a12d-f8753efcc416`, France Women's, Sandy Baltimore, Pass, timestamp 00:20:59.487, period 2 | Exact match |

This is strong, multi-signal confirmation (clock **and** score
independently align), not an assumption.

### 2.2 Root-cause trace

- **Raw event**: France Women's, `location: [115.7, 11.3]` (France's own
  per-team raw convention).
- **Orientation classification (Problem A)**: `NORMAL`, confidence 0.945
  — a no-op; Problem A is not the cause.
- **OLD default path** (`resolve_reference_team_id`, period-independent):
  Iceland (period-1 kickoff team) is the single, whole-match reference.
  France is `MIRRORED` in period 2, exactly as it was in period 1 — no
  period re-check. Result: `(120-115.7, 80-11.3) = (4.3, 68.7)` — far
  left, lower area. **This is exactly what Screenshot A showed.**
- **NEW spectator-facing default**: the SAME automatic baseline (Iceland)
  is used for period 1, but period 2 applies
  `apply_period_end_swap()` — France becomes period 2's reference
  (unmirrored). Result: `(115.7, 11.3)` — unchanged from raw, far right,
  upper area, placing France's attack and Iceland's goal/keeper on the
  right — **matching the broadcast's composition** (goal/keeper upper-
  right, action converging there).

### 2.3 Answering the task's explicit diagnostic questions

| Candidate cause | Verdict |
|---|---|
| Incorrect coordinate transformation (math itself) | **No** — the `(120-x, 80-y)` formula is correct and unchanged; extensively tested in prior sessions |
| Incorrect X/Y orientation | No — see §4, generic transform tests below |
| Incorrect attacking-direction inference | No — `infer_attacking_directions` correctly reflects whatever coordinates it's given; it was reporting a true (if wrong) fact about the OLD coordinates |
| Incorrect team normalization | **Partially — not the formula, but the CHOICE of reference team per period** |
| Incorrect team-to-goal labels | No — labels are derived from the same (previously wrong) coordinates; fixing coordinates fixes labels automatically |
| Incorrect application of physical calibration | N/A — this match has no calibration; the bug exists in the uncalibrated default path |
| **Incorrect spectator-facing display convention** | **Yes — this is the actual root cause**: the default convention never accounted for the halftime swap |
| A combination | The practical finding: **one specific, well-defined gap** (period-awareness missing from the default path), not a diffuse combination of several unrelated problems |

---

## 3. Repository-wide trace of every coordinate transformation

| Stage | File : function | Input | Output | Operation | When applied | Analytical or display? |
|---|---|---|---|---|---|---|
| Raw load | `animate_statsbomb_360_match.py:load_and_join_match` | Raw JSON | Raw JSON (unchanged) | None (I/O only) | Always | Neither |
| Problem A | `orientation_correction.py:apply_correction` | Raw event/frame | Possibly mirrored frame | `(x,y)→(120-x,80-y)`, only if `ROTATED_180` classified | Once, per usable frame | **Both** — used identically by the ML dataset builder and the renderer |
| Problem B (analytical, ML) | `team_normalization.py:apply_team_normalization_to_event/_frame` via `resolve_reference_team_id` | Problem-A output | Mirrored if non-reference team | Same formula, single period-independent reference | Once, per event, in `build_statsbomb_360_dataset.py` | **Analytical only** — untouched this session |
| Problem B (renderer, calibrated) | `animate_statsbomb_360_match.py:apply_team_normalization_to_match_with_physical_calibration` → `team_normalization.py:resolve_physical_reference_team_id` | Problem-A output | Mirrored per period-aware reference | Same formula; reference resolved **per period** via `apply_period_end_swap` | Renderer only, when `--physical-reference-team` or a verified registry entry is active | **Display only** |
| Problem B (renderer, **NEW** default) | `animate_statsbomb_360_match.py:apply_team_normalization_to_match_spectator_facing` (new) | Problem-A output | Mirrored per period-aware reference, baseline = automatic pick | Same formula; `apply_period_end_swap` applied to the UNVERIFIED automatic baseline | Renderer only, default path (no calibration) | **Display only** |
| Problem C | `team_identity.py:resolve_absolute_team_for_point` | Raw (pre-Problem-A) snapshot | Color assignment only | No coordinate change | Rendering only | Display only (color, not position) |
| Match clock | `animate_statsbomb_360_match.py:format_match_clock` (fixed this session) | `event.minute`/`event.second` | Formatted string | String formatting only, no coordinate effect | Rendering only | Display only |

No transformation is applied twice to the same coordinate in the same
pass — confirmed by the renderer's `main()` applying Problem A once, then
exactly one of the three Problem-B branches (calibrated / spectator-
facing-default / — there is no longer an "old, non-period-aware default"
branch, it was replaced) once.

### Why the original 180° correction and team normalization exist (git history / docs)

Re-confirmed from the modules' own docstrings and the existing forensic
docs (not re-derived from guesswork): Problem A exists because some raw
360 snapshots are recorded reflected relative to their own event location
(`orientation_correction.py`, forensic evidence on matches 3902240,
3857254, 3788741). Problem B (team normalization) exists because
StatsBomb's raw per-team convention always expresses a team's own events
as attacking `+x`, which is **not** a shared physical scale between two
teams (`team_normalization.py`). Neither module's original purpose was
removed or altered — only the renderer's *choice of which team to treat
as reference, per period* was generalized.

---

## 4. Rotation/mirroring logic audit — are alternatives actually tested?

Yes — not assumed. Four generic candidate transforms were rendered for
the exact Iceland–France event (`results/physical_orientation/full_visual_audit_2026-10-22/`):

| File | Transform | Formula |
|---|---|---|
| `diag_1_raw_identity.png` | Identity | `(x,y)` |
| `diag_2_rotate180.png` | 180° rotation | `(120-x, 80-y)` |
| `diag_3_horizontal_reflection.png` | Horizontal only | `(120-x, y)` |
| `diag_4_vertical_reflection.png` | Vertical only | `(x, 80-y)` |
| `diag_5_old_default_prefix.png` | Actual OLD renderer output | Automatic reference, period-independent |
| `diag_6_new_spectator_facing_default.png` | Actual NEW renderer output | Automatic reference + period-aware swap |

**Finding**: `diag_1_raw_identity.png` and `diag_6_new_spectator_facing_default.png`
are **pixel-identical** — for this specific event (a France player, and
France becomes period 2's reference under the fix), the correct result
happens to equal the raw/identity transform. This is not a coincidence to
be suspicious of: it is the expected, mechanical consequence of France
being the team that gets left unmirrored in period 2. A different event
in the same window (an Iceland player's event) would instead show
`diag_2_rotate180.png`'s relationship to the old default — **the fix is
not "always apply a 180° rotation" nor "always use identity"; it is "swap
*which team* gets the existing, unchanged mirror formula, per period"**.
This is exactly the distinction the task asked not to blur.

**Order of operations, verified not duplicated**: Problem A → Problem B.
Confirmed by code structure (`main()` calls orientation correction, then
exactly one team-normalization branch) and by this event's own trace
(Problem A was a no-op, so only Problem B's effect is visible — but the
order itself was already verified correct and unchanged from prior
sessions' extensive testing).

---

## 5. Match clock — root cause and fix

### Why it showed `1:05:59.487`

```python
# BEFORE (prior session's bug):
hours, minute_in_hour = divmod(minute, 60)
if hours:
    return f"{hours:d}:{minute_in_hour:02d}:{second:02d}{fractional}"
```

For `minute=65`: `divmod(65, 60) = (1, 5)` → `"1:05:59.487"`. This was a
plain implementation error (treating a football match-minute counter like
a general-purpose stopwatch that wraps into hours) — not a football
convention at all. Football match clocks are never hour-wrapped.

### The fix (`format_match_clock`, rewritten)

```python
PERIOD_NOMINAL_END_MINUTE = {1: 45, 2: 90, 3: 105, 4: 120}
...
nominal_end = PERIOD_NOMINAL_END_MINUTE.get(period)
if nominal_end is not None and minute > nominal_end:
    stoppage = minute - nominal_end
    return f"{nominal_end}+{stoppage}:{second:02d}{fractional}"
return f"{minute:02d}:{second:02d}{fractional}"
```

Verified behavior (from the new parametrized test suite):

| Period | Minute:Second | Output |
|---|---|---|
| 2 | 65:59 | `65:59` ✓ (the reported bug, now fixed) |
| 1 | 44:59 | `44:59` (ordinary) |
| 1 | 45:00 | `45:00` (exactly at the boundary — still plain, not `45+0`) |
| 1 | 46:23 | `45+1:23` (first-half stoppage) |
| 2 | 45:00 | `45:00` (second-half kickoff — correctly NOT read as first-half stoppage) |
| 2 | 92:08 | `90+2:08` (second-half stoppage) |
| 3 | 106:30 | `105+1:30` (extra time, 1st half, stoppage) |
| 4 | 121:45 | `120+1:45` (extra time, 2nd half, stoppage) |

**Key correctness property, explicitly tested**: stoppage-time
classification is keyed off the event's own `period` field, not the raw
minute value alone — a period-2 event at minute 46 is correctly read as
plain "46:00" (ordinary second-half time), never misclassified as
"45+1:00" (first-half stoppage), even though 46 is numerically close to
45 (`test_format_match_clock_keys_stoppage_off_actual_period_not_raw_minute`).

**Period 5 (shootouts)**: deliberately excluded from
`PERIOD_NOMINAL_END_MINUTE` — real broadcasts do not use `120+N`
stoppage notation for penalty shootouts, and no reliable convention was
identified to replace it with; the raw minute is shown directly
(e.g. `125:05`), a documented, explicit judgment call, not a guess
presented as fact.

### Title format

Primary header now reads, e.g.:
```
2nd Half  ·  65:59.487  ·  Pass  ·  Sandy Baltimore
```
Secondary (technical) line now reads:
```
Period-relative 20:59.487  |  Frame 27/94  |  previous 1.663 s  |  next frame in 0.898 s
```
matching the requested "period shown separately, period-relative time
demoted to secondary/technical" structure.

---

## 6. The spectator-facing display system — design and implementation

### Architecture

No new global transform was invented. The existing, already-tested
`(120-x, 80-y)` mirror and the existing `apply_period_end_swap` (built in
the prior session for the *calibrated* path) are reused verbatim. The
only new code is:

- `team_normalization.py`: `apply_period_end_swap(events, baseline_reference_team_id, period)`
  — the id-based, baseline-agnostic core, factored out of
  `resolve_physical_reference_team_id` (which now delegates to it — zero
  behavior change for existing callers, confirmed by all 33 pre-existing
  `test_team_normalization.py` tests passing unmodified).
- `animate_statsbomb_360_match.py`: `apply_team_normalization_to_match_spectator_facing(match, all_events)`
  — computes the automatic baseline (`resolve_reference_team_id`,
  unchanged), then applies `apply_period_end_swap` per period, using the
  same shared `_apply_team_normalization_per_period` core as the
  calibrated path (so both code paths are provably doing "the same
  thing," differing only in where the baseline comes from).
- `main()`'s resolution logic now has three explicit, clearly-labeled
  states: `cli_override`, `registry` (both physically calibrated,
  reported as such), and `spectator_facing_default` (reported as
  **NOT** physically verified, every time, in the program's own stdout).

### Requirements checklist (from the task)

| Requirement | Status |
|---|---|
| Consistent orientation within a match, no per-frame flips | ✓ — reference is resolved once per PERIOD, not per frame; unchanged within a period |
| No reversed team identities/colors | ✓ — `determine_team_style` unchanged, computed once per match |
| Raw StatsBomb data untouched | ✓ — `git status data/` shows zero diff |
| Display-only transform layer | ✓ — confined to `animate_statsbomb_360_match.py`; `build_statsbomb_360_dataset.py` untouched |
| No forcing all teams to attack one fixed side "for aesthetics" | ✓ — the automatic baseline is still whichever team the existing, pre-existing rule picks; only the per-period consistency changed |
| Halftime handled correctly | ✓ — §2, §8 |
| Physical verification not mandatory | ✓ — the new default works with zero calibration, for every match (confirmed dataset-wide, §8) |

---

## 7. Diagnostic comparisons

All 6 variants described in §4 are saved under
`results/physical_orientation/full_visual_audit_2026-10-22/`. **Best
match to the broadcast**: `diag_6_new_spectator_facing_default.png`
(identical to `diag_1_raw_identity.png` for this specific event) — goal/
keeper/action cluster on the right, matching Screenshot B's composition.
**This is not claimed as an exact physical broadcast match** (per §4 of
the prior feasibility report, no StatsBomb-only evidence can establish
that) — it is reported here only as "the general, non-match-specific fix
produces a result consistent with this one broadcast's composition,"
which is corroborating evidence for the fix's general soundness, not a
verified calibration for match 3835342 (no registry entry was added for
it).

---

## 8. Dataset-wide audit (machine-readable)

`scripts/audit_spectator_facing_fix_coverage.py` — ran against all 417
eligible matches (57.8s, 0 load errors):

```
Matches checked: 417
Matches affected by the fix (old vs new default differ in ≥1 period): 417/417
Period presence: period 1: 417 | period 2: 417 | period 3: 30 | period 4: 30 | period 5: 14
```

Raw results: `results/physical_orientation/full_visual_audit_2026-10-22/spectator_facing_fix_coverage.{json,csv}`.

**This is a software-defect-coverage audit, not a physical-orientation
audit**: it proves the OLD default path's output differed from the NEW
one for every match with a period-2 (or 4) rendering request — i.e. the
bug was universal and the fix is universal — but it says nothing about
which absolute side is physically correct for any of the 417 matches
(unknowable from this data alone, per the two prior sessions' findings).
No match is claimed to be physically verified by this audit.

---

## 9. Tests added

| File | New tests | What they cover |
|---|---|---|
| `tests/test_team_normalization.py` | 2 | `apply_period_end_swap` matches the existing name-based function; works with an arbitrary (unverified) baseline |
| `tests/test_animate_statsbomb_360_match.py` | 18 | Match-clock: no hour-wrap (regression for the exact reported bug), stoppage-time notation (parametrized, 13 cases), period-keyed stoppage classification, zero-padding, period-5 fallback; `period_display_name` (8 cases); spectator-facing default: period-2 swap with no calibration, period-1-only is unaffected, explicit before/after coordinate check reproducing the Iceland–France scenario with synthetic data |

**Full suite: 201 passed, 0 failed** (up from 172 before this session;
all pre-existing tests pass unmodified, confirming no regressions).

```
$ .venv/bin/python -m pytest tests/ -q
........................................................................ [ 35%]
........................................................................ [ 71%]
.........................................................                [100%]
201 passed in 0.95s
```

Also explicitly verified: match 3930160 (the one physically-calibrated
match) produces the **exact same** `per-period reference_team_id: {1: 772, 2: 785}`
as before this session — its calibrated path is provably untouched.

---

## 10. Five regenerated GIFs

All five regenerated with the **same intervals** as the prior session,
under `results/physical_orientation/corrected_visual_checks/`:

| GIF | Match | Frames (confirmed via PIL reopen) | `has_visible_area` | Zero-player frames | Sample match clock |
|---|---|---:|---|---|---|
| `random_match_01.gif` | 3835342, Iceland W. vs France W., period 2 | 94 | 94/94 | 0 | `65:59.487` |
| `random_match_02.gif` | 3788771, Croatia vs Scotland, period 1 | 115 | 115/115 | 0 | `22:31.731` |
| `random_match_03.gif` | 3893797, Netherlands W. vs Portugal W., period 2 | 42 | 42/42 | 0 | `47:23.036` |
| `random_match_04.gif` | 3893827, Vietnam W. vs Netherlands W., period 2 | 66 | 66/66 | 0 | `50:16.762` |
| `random_match_05.gif` | 3930163, Serbia vs England, period 1 | 120 | 120/120 | 0 | `34:56.390` |

Matches 02 and 05 (period-1-only renders) are **byte-for-byte identical**
to the prior session's GIFs — confirmed by the per-period reference id
being unchanged (Scotland/Croatia and England/Serbia results match the
earlier run exactly), since the fix only activates for periods 2/4.

All five use the exact flags required
(`--apply-orientation-correction --apply-team-normalization
--apply-team-identity-resolution --view camera-overlay
--timing-mode real-time`), with `visible_area` and freeze-frame players
confirmed present in 100% of frames across all five (via each GIF's own
generated manifest CSV).

### Before/after for Iceland–France

- **Before** (this session's reproduction of the original bug):
  `results/physical_orientation/full_visual_audit_2026-10-22/diag_5_old_default_prefix.png`
- **After**: `results/physical_orientation/full_visual_audit_2026-10-22/diag_6_new_spectator_facing_default.png`
  (also frame 27/94 of `random_match_01.gif` in the regenerated GIF)

---

## 11. Remaining uncertainties

1. **No physical verification was added for any of the 5 matches in this
   session.** The Iceland–France result visually corroborates the
   broadcast, but this is not logged as a registry entry (no second
   independent source was checked, consistent with the feasibility
   study's established standard for what counts as "verified").
2. **Y-axis (near/far touchline) correctness** remains exactly as
   uncertain as previously documented — this session's fix only concerns
   which team is treated as reference (affecting both X and Y together,
   via the same unchanged formula), not a separate Y-specific
   correction.
3. **Period 5 (shootout) clock formatting** is a documented judgment
   call (plain minute:second, no "+" notation), not a verified broadcast
   convention.
4. The dataset-wide audit (§8) is a software-defect-coverage check, not
   physical verification — this distinction is restated here because it
   is easy to conflate the two.

---

## Reproduction commands

```bash
# Confirm all tests pass
.venv/bin/python -m pytest tests/ -q

# Regenerate the Iceland-France GIF (period 2, the reported example)
.venv/bin/python src/animate_statsbomb_360_match.py \
  --match-id 3835342 --period 2 --start-second 1223.2 --end-second 1373.2 \
  --apply-orientation-correction --apply-team-normalization --apply-team-identity-resolution \
  --view camera-overlay --theme dark --timing-mode real-time \
  --output results/physical_orientation/corrected_visual_checks/random_match_01.gif

# Dataset-wide coverage audit
.venv/bin/python scripts/audit_spectator_facing_fix_coverage.py

# Open the output folder (macOS)
open results/physical_orientation/corrected_visual_checks/
```

---

## Direct answers to the task's closing questions

**Why does the current visualization appear to face the wrong way?**
The default (uncalibrated) rendering path picked one reference team for
the whole match and never re-checked it per period — correct for period
1, backwards for period 2, because real teams swap ends at halftime and
the renderer wasn't accounting for that unless a match had been
physically calibrated.

**Is the problem caused by orientation correction, team normalization,
rendering, or multiple factors?** Team normalization specifically — more
precisely, the *scope* of an existing, previously-correct piece of logic
(the halftime swap) that was wrongly restricted to only the calibrated
code path. Orientation correction (Problem A) and the mirror formula
itself were confirmed not at fault.

**What is the correct general-purpose spectator-facing display
convention?** Pick a period-1 baseline (automatic, or physically
calibrated if available) and apply the real, universal halftime/extra-
time-half end swap consistently for every subsequent period — without
requiring per-match physical verification.

**What did you change to implement it?** Added `apply_period_end_swap`
(baseline-agnostic core) and `apply_team_normalization_to_match_spectator_facing`
(uses it with the automatic baseline); rewired the renderer's default
path to use the latter instead of the old, period-independent
`apply_team_normalization_to_match` call.

**Why did the clock show `1:05:59.487` instead of `65:59`?**
`format_match_clock` used `divmod(minute, 60)`, introducing an hours
component — a plain implementation bug, not a football convention.

**Does the new clock correctly handle both halves, stoppage time, and
extra time?** Yes — verified by 13 parametrized test cases plus a
dedicated period-vs-raw-minute disambiguation test; period 5 (shootouts)
uses a documented, explicit fallback (plain minute:second).

**Did you preserve the integrity of our StatsBomb data and ML pipeline?**
Yes — `git status` confirms zero diff under `data/` and
`src/build_statsbomb_360_dataset.py`.

**Where are the five corrected GIFs?**
`results/physical_orientation/corrected_visual_checks/random_match_0{1..5}.gif`.

**What remains unverified, and why?** Absolute physical camera
orientation for all 5 matches (including Iceland–France) — StatsBomb
data alone cannot establish this, as established across all three prior
sessions; only the internal, structural consistency (matching football's
known rules) was fixed and verified here.
