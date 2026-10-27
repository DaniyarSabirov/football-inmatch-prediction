# Physical Pitch Orientation Validation — Spain vs Croatia

**STATUS: COMPLETE.** The two broadcast/visualization screenshots were
received and inspected directly. A real mismatch was confirmed, root-caused,
fixed with the smallest justified change, covered by regression tests, and
the corrected GIF was regenerated and re-verified against the broadcast
image. No raw StatsBomb data was changed; nothing was committed or staged.

---

## 1. Executive summary

- **Confirmed**: the renderer's default team-normalization pick placed
  Croatia's goal/goalkeeper on the opposite side of the screen from the
  real broadcast camera, for this match.
- **Root cause**: `resolve_reference_team_id()` picks a team by an
  arbitrary, broadcast-irrelevant rule (whichever team's first
  360-matched event is chronologically earliest — in practice, whichever
  team kicks off). This rule has no connection to real stadium/camera
  geometry, and for this match it picked the team whose raw convention
  happens to be backwards relative to the real broadcast camera.
- **Not the cause**: the mirror formula itself (`(x,y) → (120-x, 80-y)`)
  is correct, and orientation correction (Problem A) is unrelated — it
  was a no-op for the frame examined.
- **Fix implemented**: an opt-in `physical_reference_team_name` override
  on `resolve_reference_team_id()` (`src/team_normalization.py`) and a
  matching `--physical-reference-team` CLI flag
  (`src/animate_statsbomb_360_match.py`), defaulting to `None` /
  unset, which preserves the existing automatic (analytical) behavior
  exactly. No change to `src/build_statsbomb_360_dataset.py` or any raw
  data. 6 new regression tests added; all 144 tests pass.
- **Verified**: the corrected render (`--physical-reference-team Spain`)
  places Croatia's goal/goalkeeper on the right, matching the broadcast,
  for the exact frame the user screenshotted and for the goal/kickoff
  sequence.

---

## 2. Broadcast-vs-render comparison (the decisive evidence)

**Broadcast screenshot**: Spain vs Croatia, scoreboard reads "ESP 1 | 0
CRO" at 31:18 — consistent with this being the passage of play
immediately before Fabián Ruiz's goal at 31:24.988 (which made it 2-0,
confirmed separately from raw event data in the prior report). Croatia's
goalkeeper, wearing an unmistakable bright **yellow** kit, is standing in
the goal mouth on the **right** side of the frame, with several Spanish
(red/dark kit) and Croatian (white/checkered kit) players clustered in
front of that box. A Spanish player is receiving/controlling the ball in
the center-right of the frame, approaching the box.

**Our original render** (`--apply-team-normalization`, no override,
frame 7/65, same event: `Ball Receipt*`, Pedro González López,
00:31:19.808): "Croatia goal" labeled top-**left**, goalkeeper diamond
marker on the **left**, player cluster concentrated in the **left** box.

**This is a direct, unambiguous contradiction** on the one landmark that
cannot be mistaken for anything else — Croatia's goalkeeper's kit color
and which goalmouth he stands in. It is not a subtle or inferred
mismatch.

### The four-transform test (as required: horizontal ≠ 180° rotation)

Applied to the **raw** (untransformed) frame for this exact event,
rendered with the actual `render_event_frame` production function:

| Transform | Formula | Resulting Croatia GK x | Matches broadcast (GK on right, high x)? |
|---|---|---|---|
| Identity | `(x,y)` | 117.48 (raw) | **Yes** |
| Horizontal only | `(120-x, y)` | 2.52 | No |
| Vertical only | `(x, 80-y)` | 117.48 (unchanged by y-flip) | **Yes** |
| 180° rotation (= our mirror formula, = original production output) | `(120-x, 80-y)` | 2.52 | No |

**Key finding, exactly addressing the task's warning not to conflate
horizontal mirroring with 180° rotation**: in this dataset, a "180°
rotation" and a "horizontal-only" flip have the **identical effect on
the x-coordinate** (both compute `120-x`) — they only differ in whether y
is also flipped. Since the one unambiguous broadcast landmark (which side
the goal is on) is purely an x-axis fact, this single landmark is
sufficient to show that **both** "horizontal only" and "180° rotation"
(i.e. our actual mirror formula) are wrong for this event, while
**identity and vertical-only are both right** on x. Because our pipeline
only exposes one binary knob — *which team* gets the (full, unchanged)
`(120-x, 80-y)` mirror — and not a separate x-only/y-only toggle, the
only real-world fix available (short of inventing a new, unevidenced
transform) is to change **which team** receives that mirror. See §3.

### Y-axis: explicitly flagged as less certain

The broadcast camera is an oblique, elevated, single-side perspective
shot, not a top-down view — mapping its vertical/depth axis precisely to
pitch-width (`y`) coordinates is inherently less precise than the x-axis
check (which rests on an unambiguous, unmistakable landmark: goalkeeper
kit color and goalmouth side). This report does **not** independently
verify the y-axis from the broadcast image with the same confidence as
the x-axis. However: the only corrective action actually available in the
code is swapping which team receives the *existing, already-proven*
`(120-x, 80-y)` mirror — not a new, separately-tunable y-only transform.
Since that existing formula has already been extensively validated for
internal self-consistency (point-in-polygon, pairwise-distance
preservation, goalkeeper-near-own-goal checks — see the prior two reports
and `tests/test_team_normalization.py`), and the x-axis check after the
swap is now confirmed correct (§4), there is good reason to expect y is
also correctly handled by the swap — but this is **not independently,
separately proven** and is stated here as the one remaining point of
residual uncertainty, exactly as the task requires.

---

## 3. Root cause, with the code citation requested

`resolve_reference_team_id()` (`src/team_normalization.py`, pre-fix
docstring, lines 63-77) explicitly documented — **before this
investigation, as a known, accepted limitation, not a bug that was
hidden**:

> "Which of the two teams ends up 'the reference' is not itself
> meaningful (Problem B does not attempt to recover true physical sides)
> — only that exactly one is picked and used consistently for the whole
> match."

This investigation adds the missing piece: *what* does determine the
pick, concretely? Confirmed by direct inspection across a 5-match sample
(`3857283`, `3930160`, `3788741`, `3857254`, `3902240`): in **every**
case, the reference team is exactly whichever team's event is first in
`match.usable_frames` (chronologically sorted) — which is always that
team's kickoff pass. **Kickoff order has no relationship to stadium
geometry or broadcast camera position.** For Spain–Croatia, Croatia
happened to kick off (confirmed: `events[4]` in the raw file, index 5, is
"Croatia Pass"), making Croatia the reference team — and, as the
broadcast comparison in §2 shows, that happened to be the team whose raw
convention is backwards relative to this specific broadcast camera.

This is **not** a formula defect (the mirror math is correct, and
extensively tested) and **not** an orientation-correction (Problem A)
defect (confirmed a no-op for the examined frame). It is a **team
*selection*** defect relative to the goal of physically-accurate
visualization — a goal the code never previously claimed to meet. The
existing analytical (ML-feature) use case is unaffected: nothing shown
here demonstrates incorrectness for cross-team *relative* consistency,
only for *absolute* (real-camera) left/right in visualization.

---

## 4. Fix implemented

### Design decision

Per the task's explicit instruction: preserve the analytical coordinate
system; do not touch the dataset builder; do not auto-remove team
normalization or replace it with an arbitrary flip. The chosen fix is the
smallest one that satisfies all of these: an **opt-in override** for
*which* team is picked as reference, used only when a caller has
independent, verified physical-orientation evidence.

### `src/team_normalization.py`

```python
def resolve_reference_team_id(
    events: list[dict[str, Any]],
    physical_reference_team_name: str | None = None,
) -> int:
```

- `physical_reference_team_name=None` (the default): **byte-for-byte
  identical behavior to before** — same automatic pick, same return
  value, verified by the existing 20 tests continuing to pass unmodified
  plus a new explicit test (`test_override_none_preserves_automatic_behavior`).
- When given, matched case-insensitively against the two teams actually
  present in `events`; raises `ValueError` if it does not match exactly
  one of them, or if there are not exactly two teams (checked first, so
  the error message is never misleading about which problem occurred).
- `src/build_statsbomb_360_dataset.py` calls this function with a single
  positional argument and is **completely unaffected** — confirmed by
  reading the call site (unchanged) and by the full test suite passing.

### `src/animate_statsbomb_360_match.py`

- New CLI flag `--physical-reference-team TEAM_NAME`, default `None`,
  fully documented in `--help` (reproduced in §2 of this doc's companion
  `--help` output), explicitly stating it only affects
  `--apply-team-normalization` and does not touch the dataset builder.
- Wired into the existing `resolve_reference_team_id(...)` call site
  (`main()`), passed through only when supplied.
- The program's own stdout report now states explicitly whether the
  reference team came from the automatic rule or an explicit override,
  so a render's provenance is never ambiguous from the log alone.

### Regression tests added (`tests/test_team_normalization.py`)

6 new tests, all passing:
1. `test_override_none_preserves_automatic_behavior`
2. `test_override_can_select_the_team_that_would_not_be_picked_automatically`
3. `test_override_matches_case_insensitively`
4. `test_override_with_non_matching_name_raises`
5. `test_override_still_requires_exactly_two_teams_first`
6. `test_override_flips_which_events_team_normalization_treats_as_mirrored`
   (end-to-end: confirms swapping the reference team swaps which team's
   coordinates get mirrored, using the *same* already-tested formula)

**Full suite**: 144/144 passed (138 pre-existing + 6 new), confirming no
regression anywhere else in the project.

---

## 5. Final corrected GIF — generated and verified

### Command used (exact, copy-pasted from the actual successful run)

```bash
.venv/bin/python src/animate_statsbomb_360_match.py \
  --match-id 3930160 \
  --period 1 \
  --start-minute 31 \
  --end-minute 34 \
  --apply-orientation-correction \
  --apply-team-normalization \
  --apply-team-identity-resolution \
  --physical-reference-team "Spain" \
  --view camera-overlay \
  --theme dark \
  --timing-mode real-time \
  --output results/spain_croatia_physically_corrected.gif \
  --event-frame-manifest results/spain_croatia_physically_corrected_manifest.csv
```

### Verified output

| | Value |
|---|---|
| Output path | `results/spain_croatia_physically_corrected.gif` (confirmed exists, 2,655,877 bytes) |
| Frames (program output) | **65** selected, 65 unique event UUIDs |
| Frames actually in the saved file (reopened with PIL) | **65** — matches exactly |
| `reference_team_id` | 772 (Spain) — **explicit override** `--physical-reference-team='Spain'` (confirmed in the run's own stdout) |
| `REFERENCE` / `MIRRORED` counts | 1501 / 1752 — exactly swapped from the original run's 1752/1501, confirming the override inverted the whole-match assignment as intended |
| Attacking-direction inference (whole period 1, all shots) | "Croatia: 17.1, Spain: 106.7" — exactly mirrored from the original run's "Croatia: 102.9, Spain: 13.3", confirming the flip is global and consistent, not just local to one frame |

### Visual re-verification against the broadcast (actually inspected, not assumed)

Three frames extracted from the corrected GIF and visually compared
against the broadcast screenshot and football logic:

- **Frame 7/65** (Pedro González, `Ball Receipt*`, 31:19.808 — the exact
  frame the user screenshotted): "Croatia goal" now labeled **right**,
  goalkeeper diamond on the **right**, player cluster in the right box —
  **matches the broadcast's composition**: goal/keeper on the right,
  players converging on that box.
- **Frame 17/65** (Fabián Ruiz, `Shot`/Goal, 31:24.988): ball and actor
  inside the right-hand box, next to Croatia's keeper — correct, a goal
  must be scored right at the goalmouth.
- **Frame 20/65** (Andrej Kramarić, kickoff `Pass`, 32:33.968): camera
  pulled back to a wide, centered view, players arranged in a kickoff
  formation around the center circle — unchanged in character from the
  original render (as expected — a kickoff is symmetric and not
  diagnostic of left/right), still consistent with "Croatia goal" now
  correctly labeled on the right throughout.

**Conclusion**: the corrected rendering now matches the broadcast on the
one unambiguous landmark (Croatia goalkeeper/goal side) at the exact
frame in question, and remains internally and globally consistent
(whole-match `REFERENCE`/`MIRRORED` swap, whole-period shot-direction
inference swap) rather than being a one-frame coincidence.

---

## 6. Reusable commands

### A. Exact command for this Spain–Croatia GIF (copy-paste into PyCharm terminal)

```bash
.venv/bin/python src/animate_statsbomb_360_match.py --match-id 3930160 --period 1 --start-minute 31 --end-minute 34 --apply-orientation-correction --apply-team-normalization --apply-team-identity-resolution --physical-reference-team "Spain" --view camera-overlay --theme dark --timing-mode real-time --output results/spain_croatia_physically_corrected.gif
```

### B. Reusable template for other matches

```bash
.venv/bin/python src/animate_statsbomb_360_match.py \
  --match-id <MATCH_ID> \
  --period <PERIOD> \
  --start-time <MM:SS> \
  --end-time <MM:SS> \
  --apply-orientation-correction \
  --apply-team-normalization \
  --apply-team-identity-resolution \
  --physical-reference-team "<TEAM NAME, exact, case-insensitive>" \
  --view camera-overlay \
  --theme dark \
  --timing-mode real-time \
  --output results/<OUTPUT_FILENAME>.gif
```

Notes on this template:
- `--start-time`/`--end-time` (period-relative `MM:SS`) is used here
  instead of `--start-minute`/`--end-minute` or
  `--start-second`/`--end-second` because it is the least error-prone to
  retype by hand for a new match; all three are real, verified flags (see
  `--help`) and interchangeable per `resolve_interval_seconds`'s
  documented priority order.
- **`--physical-reference-team` must be re-derived per match** —
  confirmed in §3 that the automatic pick has no general rule, so there
  is no default value that is safe to reuse across matches. Omit the
  flag entirely to fall back to the automatic (analytical, not
  broadcast-verified) pick if no broadcast evidence is available for a
  given match — do **not** guess a team name without evidence, since a
  wrong guess is worse than using the (at least clearly-labeled)
  automatic default.
- Every flag above was verified against the actual `--help` output in
  this session; none were invented.

### C. Open the generated GIF on macOS

```bash
open results/spain_croatia_physically_corrected.gif
```

---

## 7. Does this affect other matches?

Yes, potentially all of them, since the mechanism (§3) is universal, not
Spain–Croatia-specific. **This investigation does not claim to have
checked or corrected any other match** — Belgium–Morocco (3857283) and
every other match in the dataset still use the automatic, broadcast-
unverified pick by default, exactly as before (the fix is opt-in and
changes nothing unless `--physical-reference-team` is explicitly passed).
Determining the correct reference team for any other specific match would
require the same kind of broadcast-landmark evidence gathered here for
Spain–Croatia — there is no way to derive it from StatsBomb data alone
(confirmed in the prior report, §2/§3: raw coordinates alone cannot even
establish which end either team is defending, only that mirroring one of
them is necessary).

---

## 8. Remaining uncertainty (stated explicitly, not glossed over)

1. **Y-axis (near/far touchline) correctness** is not independently
   verified from the broadcast image with the same confidence as the
   x-axis (§2) — only inferred from the existing, separately-validated
   mirror formula's internal consistency properties.
2. **Only one event/frame pair and its immediate neighbors (goal,
   kickoff) were checked against the broadcast.** The remaining 62 frames
   in the corrected GIF were not individually re-verified against
   broadcast footage (none exists for most of them in this session).
3. **No other match was checked or corrected.** The fix is general-
   purpose, but applying it correctly to any other match requires new,
   match-specific broadcast evidence.
4. **The fix assumes exactly two teams and relies on exact (case-
   insensitive) team-name matching** against whatever name StatsBomb
   recorded for that match — this is consistent with how team names are
   used elsewhere in the codebase, but a typo in `--physical-reference-team`
   will raise a clear `ValueError` rather than silently doing nothing
   (verified by `test_override_with_non_matching_name_raises`).

Nothing was committed or staged while producing this report.
