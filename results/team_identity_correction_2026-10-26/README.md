# Team identity correction and orientation synchronization — implementation and impact

Date: 2026-10-26/27. Status: **code implemented and tested; nothing committed; production ML dataset NOT
regenerated** (both await approval). Raw StatsBomb files, the earlier investigation
(`results/team_identity_hypothesis_investigation/`) and earlier GIF evidence are untouched. All numbers
below come from the scripts in `diagnostics/` run on all 417 usable matches.

## 1. What was implemented

| Where | What |
|---|---|
| `src/team_identity.py` | `resolve_match_team_identity` — the new shared identity resolution (below). The legacy functions remain for comparisons. |
| `src/orientation_resolution.py` (new) | The renderer's validated UNCERTAIN-orientation resolution (`display_orientation_results`, display plan, frame reviews) **moved unchanged** out of the renderer, plus `resolve_match_orientation`. The renderer re-exports the names. |
| `src/build_statsbomb_360_dataset.py` | Uses both shared functions. New columns: `orientation_problem_a`, `orientation_resolved_by`, `visible_teammates_raw`, `visible_opponents_raw` (spec reading of the raw flags, preserved), `identity_corrected_points`, `identity_quality` (clean / corrected / uncertain). `visible_ambiguous` now counts every unresolved player; unresolved players are counted on neither side. |
| `src/animate_statsbomb_360_match.py` | Draws from the same `FrameIdentity`: corrected players get a dashed ring ("Team corrected on evidence"), unresolved players get the neutral "Team unclear" fill. Only those players are affected, never the whole frame unless every player in it is unresolved. The display-only refinement is superseded. |
| `scripts/audit_team_identity_continuity.py` | Detector made crowding-safe (unambiguous pairing, common offset removed); it now writes to `continuity_audit/` here, so the old outputs are kept. |
| `tests/test_team_identity_evidence.py` (new) | 26 tests (below). Full suite: **327 passed**. |

### Identity rules (shared by renderer and builder)

1. **Spec reading is the default:** `teammate` relative to the record's own event team.
2. **Structural links:** two records of the same period are the same scene when their raw coordinates are
   identical, or identical up to one constant offset, as stored or after a 180° mirror (residual < 0.05 yd,
   ≥ 3 players). Equal flags mean the same flag view; inverted flags mean opposite views. This finds
   byte-identical copies, copies that differ only in actor/keeper/visible_area (case_01, case_05) and
   mirrored or shifted derived records (case_02).
3. **A group whose view constraints contradict its members' own teams** gets its view from evidence, never
   from event order or the coordinate frame alone:
   - (a) **continuity:** unambiguous pairs with the nearest unlinked snapshots (≤ 2 s); ≥ 4 players,
     ≤ max(1, 10%) disagree; no neighbour may support or lean towards the other view; a contradicting
     goalkeeper vetoes unless the coordinate owner sides with continuity.
   - (b) **goalkeeper goal side AND coordinate owner** agree.
   - (c) the **legacy Problem C** source, if nothing objects.
   - (d) otherwise **UNRESOLVED**.
4. **Crowding-safe contradiction detector:** a player is unresolved only if *both* neighbouring snapshots
   pair him unambiguously and both carry the same other team. The whole record is unresolved only if
   ≥ 4 and ≥ 50% of its evaluable players are contradicted.
5. **Goalkeeper-only rule (as you specified).** A keeper whose colour contradicts the goal he stands at
   (≤ 30 yd from the goal line) keeps the identity of the nearest earlier keeper observation, but only
   when all of these hold:
   - it is the same keeper: within 2 + 1.5 yd/s (max 10 yd), at the same goal end, ≤ 10 s earlier;
   - that earlier identity is verified by two independent signals (its own raw flag AND its goal side),
     not inherited, and was never corrected itself;
   - no Substitution / Player Off / Player On / Tactical Shift / half boundary lies in between;
   - it is the same period, and not a shootout;
   - the rest of the frame is consistent.

   Otherwise the keeper is UNRESOLVED. Only the keeper point changes.

Snapshots whose orientation is still unknown give and receive no geometric evidence, so orientation and
identity stay separate.

## 2. Scale of the problem (all 417 matches)

| Unit | Total |
|---|---|
| Matches | 417 |
| Events (all types) | 1,551,841 |
| Events with a 360 record = usable snapshots | 1,357,626 |
| Player observations (freeze-frame points) | 21,193,635 |
| ML rows (10 s horizon) | 756,023 (same rows before and after; 0 lost) |

**Player observations** (denominator 21,193,635):

| Category | Count | % |
|---|---|---|
| Confirmed correct (positive evidence: unambiguous neighbour agreement or keeper goal side) | 13,154,864 | 62.07 |
| Unverified (no evidence either way; kept as read) | 8,029,092 | 37.88 |
| **Confirmed errors of the original pipeline, now corrected** | **2,822** | **0.0133** |
|   of which: goalkeeper rule | 1,175 | 0.0055 |
|   of which: view of copied/derived records | 1,647 | 0.0078 |
| Originally "team unclear", now resolved | 127 | 0.0006 |
| **Unresolved after correction (all)** | **9,679** | **0.0457** |
|   proven structural conflict, direction undecided | 3,670 | 0.0173 |
|   suspected label error: contradicted by both neighbours | 3,455 | 0.0163 |
|   suspected: copy flags disagree on one player | 1,134 | 0.0054 |
|   suspected: goalkeeper at the wrong goal, no verified predecessor | 900 | 0.0042 |
|   whole record contradicted (raw scramble) | 520 | 0.0025 |

**Snapshots** (denominator 1,357,626):

| Category | Count | % |
|---|---|---|
| Structurally inconsistent (flag view contradicts own team; mostly byte-identical copies already handled by legacy Problem C) | 101,041 | 7.44 |
| Changed vs the original pipeline | 1,287 | 0.095 |
| ≥ 1 unresolved player | 5,423 | 0.40 |
| Entirely unresolved | 270 | 0.020 |
| Raw labels scrambled (≥ 1 player contradicted by both neighbours) | 3,237 | 0.24 |

How the 101,041 structurally inconsistent snapshots were decided:

| Decision | Snapshots |
|---|---|
| Continuity | 67,779 |
| Legacy source | 17,082 |
| Keeper + owner | 15,944 |
| Undecided | 236 |

**Goalkeepers** (denominator: 488,292 keeper observations ≤ 30 yd from a goal):

- The original pipeline coloured **2,280 (0.47%)** of them wrongly by goal side. **2,122 (0.43%)** are
  goalkeeper-only errors, i.e. the rest of the record was unchanged and resolved.
- Goalkeeper-only outcomes: **1,154 corrected** by the continuity rule; **968 left unresolved**
  (884 without a verified predecessor, 84 where the copies disagree).
- **Held-out check of the rule:** the next observation of the same keeper (not used by the rule) agrees in
  **826 of 826** cases where one exists; 349 corrections have no forward observation.
- **About the earlier figure of 957:** it could not be reproduced exactly from the saved files. The same
  definition reproduces 1,000 records. Of these, 551 are now corrected, 447 unresolved and 6 unchanged.
- "0 wrongly coloured keepers after correction" is **true by construction** (inconsistent keepers are
  either corrected or marked unclear), so it is not evidence.

**Independent checks of the group decisions** (`tables/group_decision_validation.csv`):

- Continuity decisions agree with the coordinate owner, which they did not use, in **99.87%** of 67,499
  snapshots, and with the goalkeeper in **99.53%** of 28,691.
- 90 continuity decisions overturn the legacy choice. 14 of them have a goalkeeper, and **all 14** agree
  with the new view.

**Detector before/after** (`continuity_audit/`):

- Flagged frames: **102 strong + 2,094 weak → 1 strong + 199 weak**.
- Paired players changing colour between consecutive snapshots: 4.9% → 1.35%. The remaining 1.35% are
  one-sided changes, where only one neighbour disagrees. They are not attributable to the record and are
  left as an open limitation.

**Kick-off 505 (3930164):** it has no structural partner and no previous snapshot, so the two-neighbour rule
cannot evaluate it, and it keeps its raw colours. This is a known limitation, documented by a test.

## 3. Orientation synchronization (the 535 rows)

- All 535 rows that the ML builder used to leave UNCERTAIN are now resolved with the renderer's rules:
  469 kick-off rule, 45 same-moment duplicate, 12 camera side, 9 user frame review.
- **485** resolve to NORMAL, so their coordinates are unchanged and only the label changes.
- **50** resolve to ROTATED_180, so actor_x/actor_y are mirrored (**47 rows** actually change; the other 3
  have no actor point).
- **0** confidently classified rows changed.
- Orientation is applied once: only the resolved result replaces Problem A, before the unchanged team
  normalization.
- Actor and players are rotated together. Identity is never derived from orientation.
- A test asserts that builder and renderer orientations are identical per event.

## 4. ML impact (denominator 756,023 ML rows)

| Measure | Rows | % |
|---|---|---|
| Teammate/opponent counts changed vs original | 3,610 | 0.48 |
|   of which certain (corrected, no unresolved player) | 667 | 0.088 |
| ≥ 1 unresolved player (`identity_quality = uncertain`) | 2,949 | 0.39 |
| Entire freeze frame unresolved | 93 | 0.012 |
| Would be excluded if unresolved rows are dropped | 2,949 | 0.39 |

- **Size of the change:** typically one player (median unresolved players per uncertain row = 1; the most
  common count change is −1).
- **Feature distributions:** mean visible_teammates moves 7.717 → 7.714 and visible_opponents
  7.950 → 7.948 (KS ≤ 0.0007). actor_x/actor_y KS ≈ 1e-5.
- **Concentration:**
  - 406 of 417 matches have at least one uncertain row; the per-match share is 0–2.9% (median 0.29%).
    The top 10 matches hold 12.9% of uncertain rows.
  - The current event type is statistically but weakly associated with uncertainty (χ² p < 1e-16,
    Cramér's V 0.011). Shot rows are uncertain at 1.00% vs 0.39% overall (37 of 3,696); Clearance
    0.68%, Miscontrol 0.64%.
  - Period: no association (p = 0.15). Stoppage time (90'+): 0.57%.
- **Class balance if uncertain rows are excluded:** every class loses 0.35–0.72% of its rows (Shot: 56,
  0.60%). The relative class shares change by at most −0.33% (Clearance), and the majority/minority
  ratio moves from 40.48 to 40.54. Excluded rows have fewer visible teammates and opponents by
  construction (KS 0.29), so exclusion is mildly non-random but tiny in volume.

### Controlled sensitivity probe (`model_probe/`)

There is **no existing baseline** in the repository, so this is a *probe*, not a thesis result. Setup:

- Models: HistGradientBoosting and logistic regression, fixed settings.
- Target: next_event_type (8 classes).
- Split: GroupKFold(5) by match, identical for every variant.

| HGB, pooled over 5 folds | macro-F1 | bal. acc | MCC | F1 Shot |
|---|---|---|---|---|
| V0 original features | 0.3025 | 0.2920 | 0.6772 | 0.132 |
| V1 corrected | 0.3027 | 0.2919 | 0.6773 | 0.134 |
| V2 exclude uncertain rows | 0.3034 | 0.2924 | 0.6773 | 0.135 |
| V3 keep + mask counts + flag | 0.3042 | 0.2932 | 0.6774 | 0.144 |
| V4 no team-count features | 0.2810 | 0.2792 | 0.6762 | 0.000 |

Paired per-fold macro-F1 differences vs V1 (fold-to-fold SD of V1 is 0.0015):

| Comparison | Mean ± SD |
|---|---|
| V0 | −0.0001 ± 0.0013 |
| V2 | +0.0007 ± 0.0020 |
| V3 | +0.0015 ± 0.0019 |
| V4 | **−0.0217 ± 0.0028** |

Logistic regression: all variant differences are |Δ| ≤ 0.0002 (V4: +0.002).

**Reading:**

- The team-count features carry real signal: removing them costs 0.022 macro-F1, and Shot F1 collapses.
- Correcting, excluding or masking the identity problems changes nothing beyond fold noise.
- Caveat: these are weak models (macro-F1 ≈ 0.30) and the Shot F1 differences rest on ~9,300 rows.
  A stronger model should be re-checked with the same script.

## 5. Recommendation

1. Rebuild the dataset with the corrected pipeline (after your approval). It removes the confirmed errors
   (2,822 players in 3,610 changed rows) and aligns orientation with the renderer (47 rows).
2. Keep the uncertain rows and record the uncertainty (`identity_quality`, `visible_ambiguous`).
   - In the main model, use V1 (unresolved players counted on neither side), or V3 masking.
   - Report exclusion (V2) as a **sensitivity analysis**, not as the main dataset.
   - Never encode "unclear" as a third team.
3. In the thesis, report the numbers in §2–§4 as a data-quality limitation. 0.39% of rows are uncertain;
   corrections affect 0.48% of rows; the measured effect is within cross-validation noise.

Exclusion is defensible too: it removes 0.39% of rows and barely changes the classes. But it is mildly
non-random (Shot as current event 2.6× over-represented among the excluded rows), and it buys nothing
measurable.

## 6. Files

| Path | Content |
|---|---|
| `records.csv.gz` | per snapshot: statuses, reasons, decision and evidence, changes vs legacy and vs spec, orientation route |
| `keepers.csv.gz`, `goalkeeper_rule.csv` | keeper observations; every goalkeeper correction with its forward validation |
| `ml_rows.csv.gz` | original and corrected features per ML row (in memory, not `data/processed`) |
| `summary.json`, `tables/` | all aggregates; concentration by match, event type, period, class |
| `model_probe/` | metrics, per-fold metrics, paired differences |
| `case_visualizations/case_NN/` | `before_after.png` (previous / reviewed / next; before vs after), `context.gif` (production CLI, established flags, camera polygon), `points.csv` |
| `continuity_audit/` | corrected detector output |
| `diagnostics/` | scripts, in order: `audit_identity_correction.py`, `analyze_identity_impact.py`, `validate_group_decisions.py`, `model_sensitivity_probe.py`, `render_corrected_cases.py`; `legacy_builder_head.py` = `git show HEAD:src/build_statsbomb_360_dataset.py` |

Five reviewed cases:

| Case | Decision | Result |
|---|---|---|
| case_01 | keeper + owner | 18/18 recoloured |
| case_02 | continuity | 16/16 recoloured |
| case_03, case_04 | — | unchanged (crowding artefacts) |
| case_05 | continuity | 15/15 recoloured |

The 3857293 groups keep the Ghana keeper colouring.
