# Final validation of the team-identity and orientation corrections

Date: 2026-10-27. **Verdict: PASS WITH LIMITATIONS.**

Three implementation defects were found and fixed, each with regression tests:
- tiny byte-identical copies were not linked;
- simultaneous contradictions were left coloured;
- the default ML features used offline evidence from after the event.

No material correctness issue remains. Raw StatsBomb data, earlier reports and earlier visualizations were
not modified. Every number here is reproducible with the scripts in `diagnostics/`, run on all 417 usable
matches. The earlier report (`results/team_identity_correction_2026-10-26/README.md`) is kept unchanged; where
its numbers differ from this report, this report reflects the final code (§6).

## 1. Residual colour transitions (`diagnostics/transition_audit.py`, `classify_transitions.py`)

### Population

Same population as `scripts/audit_team_identity_continuity.py`:
- consecutive distinct snapshots of one period, ≤ 2 s apart, in drawn coordinates;
- players paired by unambiguous optimal assignment (≤ 3 yd, next-best candidate ≥ 1 yd further);
- unresolved players skipped.

| Unit | Count |
|---|---|
| Player pairs evaluated | **7,140,398** |
| Pairs whose team colour changes (transitions) | **95,579 (1.339%)** |
| Matches affected | 417 of 417 |
| Snapshot links with ≥ 1 transition | 80,725 |
| Distinct snapshots involved | 154,366 |
| Distinct player observations involved | 191,034 |

Per match: 140–416 transitions, median 222. The top 10 matches hold 3.7%, so transitions are spread evenly,
not concentrated.

### What they are: mostly the pairing, not the identities

Transition rate (%) by time gap and pairing margin:

| time gap \ margin | ≤ 1.5 yd | ≤ 2 yd | ≤ 3 yd | ≤ 5 yd | > 5 yd |
|---|---|---|---|---|---|
| same ms | 0.214 | 0.109 | 0.008 | 0.005 | 0.000 |
| ≤ 0.25 s | 0.494 | 0.262 | 0.013 | 0.006 | 0.001 |
| ≤ 0.5 s | 2.043 | 0.771 | 0.287 | 0.135 | 0.070 |
| ≤ 1 s | 5.509 | 3.107 | 1.351 | 0.372 | 0.093 |
| ≤ 2 s | 11.761 | 8.893 | 5.752 | 2.586 | 0.488 |

- A genuine label error does not depend on how far players moved between snapshots, or on how close the
  next pairing candidate is. The observed rate rises **~10,000×** from the near-certain corner to the
  uncertain corner. That is the signature of pairing errors: two players crossing, or fast movement over
  1–2 s.
- **Goalkeepers**, who are isolated and therefore paired reliably, transition at 0.145% vs 1.373% for
  everyone else.
- **Change of the event's team:** holding the time gap fixed, a change between the two snapshots does not
  raise the rate (≤ 0.25 s: 0.052% different team vs 0.057% same team). This gives no evidence of a
  systematic perspective error.
- **Structural links:** linked vs unlinked records show 1.348% vs 1.276%.
- **Orientation route:** non-geometry routes show 1.612% on only 45k pairs.
- **Corrected points:** at margin > 5 yd and gap ≤ 1 s they transition about as often as uncorrected ones
  (0.10% vs 0.07%, and 0.11% vs 0.09%). The excess at longer gaps comes from actor points of copied
  records (contested-ball players).
  - A wrong view decision would flip almost every player of a record.
  - Only 11 of 8,161 links touching a corrected record have ≥ 4 flips, and all but one are ≥ 0.69 s apart
    with tight margins.
  - **No systematic error in the identity resolution was found.**

### Classification of the 95,579 transitions

| Class | Definition | Transitions |
|---|---|---|
| Confirmed incorrect | an independent anchor (keeper goal side; actor semantics in an unlinked record) contradicts one side | **0** |
| Strongly suspected | pairing near-certain (≤ 0.25 s, margin > 2 yd) or two-sided in the sequence (blip) | **264** |
| Ambiguous | ≤ 1 s and margin > 3 yd, no anchor | **3,296** |
| Likely pairing artefact | > 1 s or margin ≤ 3 yd | **92,019** |

Patterns, looking one snapshot further back and forward:

| Pattern | Transitions |
|---|---|
| unanchored | 82,006 |
| step | 13,343 (12,787 of them in the pairing-artefact class) |
| blip | 212 |
| alternating | 18 |

**Estimate of genuine label disagreements.** In the near-certain stratum (≤ 0.25 s, margin > 5 yd), the
first run of this audit, before the simultaneous rule, showed 0.029–0.059% transitions. Applied to all
pairs, that gives roughly **2,000–4,000 genuine disagreements, about 2–4% of the transitions**. The rest are
attributed to pairing. This is an estimate under the assumption that genuine disagreements occur at the
same rate in every stratum.

The post-fix floor (0.0009%) cannot be used for this estimate, because the new simultaneous rule removes
exactly that stratum.

### Two genuine gaps found and fixed (`view_conflicts.csv`)

51 near-simultaneous links had ≥ 4 players flipping together.

1. **29 same-scene copies with partly different flags.** Coordinates are identical but, for example,
   13/19 flags are equal. The code had deliberately skipped them. Identical coordinates at one instant
   prove that the differing players carry contradictory labels.
2. **22 near-copies of opposing teams whose flags were not inverted.** They are mirrored, 0.1–0.2 s apart,
   with ≥ 90% of flags equal. The spec requires inverted flags here, so one record carries the wrong view.

Both are raw-data contradictions visible on raw coordinates, independent of our orientation processing, and
in most of them the wider context cannot tell which record is right.

**Fix:** the *simultaneous-contradiction rule*. A player paired with margin ≥ 2 yd in two unlinked
snapshots ≤ 250 ms apart, but given different teams, becomes UNRESOLVED in both offline (in causal mode, in
the later record only). Nothing is recoloured.

**Effect:**
- 3,478 points marked (0.016%);
- near-certain transitions drop from 0.03–0.06% to ≈ 0.001%;
- whole-view conflicts at ≤ 0.25 s are gone from the transition population.

### Third defect

A byte-identical cross-team copy with fewer than 3 players was never linked, so it silently fell back to the
spec reading and undid the legacy handling (5 groups, 14 records). Byte-identical records are now linked
regardless of size; all 14 again match the original colours.

### Independent check at kick-offs (Law 8, `kickoff_law8.csv`)

Only the 509 kick-offs whose orientation did not come from flags count: the kick-off route itself and
reviewer decisions are excluded as circular.

- **74 of 6,675 evaluated players (1.11%)** stand in the wrong half.
- The count is identical under the raw, original, offline and causal identities, so the corrections neither
  add nor remove such errors.
- **Depth:** 52 of the 74 are within 5 yd of the halfway line, consistent with movement after the whistle.
  12 of the deeper ones are in one scrambled kick-off record (3788749), similar to frame 505.

## 2. The five before/after cases (`case_visualizations/`, `five_cases_checks.csv`)

All renders use the final code and the established flags, with `--theme dark` passed explicitly. All 83 GIF
frames have the visible_area polygon, and identity resolution never moves a coordinate.

| Case | Snapshot | Original problem | After correction | Independent checks |
|---|---|---|---|---|
| case_01 | 3893791 Spain–Costa Rica, P1 18:25.761, Costa Rica Duel | verbatim copy of Spain's Ball Receipt* (only actor differs), read in Costa Rica's view | Spain's view, 18/18 recoloured (decision keeper + owner) | Goalkeeper wrong side before, correct after. Disagreements with the two same-ms partners: 12/12 and 10/10 before, 0/12 and 0/10 after. Actor point 1.8 yd from the duel spot, now drawn Spain: the copied record's actor flag marks the Spain dueller. |
| case_02 | 3895107 Köln–Leverkusen, P2 82:13.060, Köln Block | derived from Leverkusen's Block 82 ms earlier (mirror + 2.53 yd), flags not inverted | Leverkusen's view, 16/16 (continuity) | Disagreements 16/16 and 10/10 → 0/16 and 0/10; no keeper visible |
| case_03 | 3857256 Switzerland–Serbia, P2 90:23.300, Serbia Pass | goal-mouth crowding flagged by the old detector | unchanged (correct) | 1/8 and 0/4 disagreements, identical before/after |
| case_04 | 3844386 Sweden–Belgium, P1 31:58.616, Sweden Pressure | crowding | unchanged (correct) | Keeper at Belgium's goal, consistent; 0/6 and 1/12 disagreements |
| case_05 | 3773386 Barcelona–Alavés, P1 27:13.282, Barcelona Ball Receipt* | verbatim copy of Alavés's Block (actor differs), read in Barcelona's view | Alavés's view, 15/15 (continuity) | Disagreements 9/10 and 13/13 → 1/10 and 0/13 |

**No new inconsistency was introduced.**

**Visual caveat (data, not pipeline), case_05:** the "Event actor" ring of Barcelona's Ball Receipt* sits on
a player now drawn in Alavés colours. The actor flag in this copied record is **26.1 yd from the event's own
location**, so it does not mark the receiving Barcelona player. The ring faithfully shows a misattributed raw
`actor` flag.

## 3. Frame 505 (match 3930164) — `frame_505/`

**The record:**
- Event index 5: P1 00:00:00.505, Slovakia **kick-off Pass**, Lukáš Haraslín → Stanislav Lobotka (backward,
  ending at 48.3, 39.2).
- 20 freeze-frame points, 10 flagged teammate, **no goalkeeper flag**.
- The only earlier events are Starting XI ×2 and Half Start ×2, so there is no earlier snapshot.
- Problem A: UNCERTAIN (the event location is the centre spot, where both readings coincide). The
  orientation comes from your **frame review** (NORMAL).

**Independent evidence:**

| Reading | Law 8 violations | Disagreements with the snapshot 1.1 s later |
|---|---|---|
| As drawn | 6 Slovakia players beyond halfway + 2 Belgium players in Slovakia's half = **8**, several 20+ yd deep | **10 of 14** |
| Frame mirrored | 1 + 4 | 4 of 5 |
| Flags inverted | 4 + 1 | 4 of 14 |

No single orientation or view correction makes the record consistent. At least the 8 ringed players are very
likely mis-labelled, but the record is scrambled rather than mis-processed.

**Decision:** retained unchanged, as instructed. The two-neighbour detector cannot evaluate it, and no
existing rule applies.

**Optional rule for your decision, not implemented:** mark players ≥ 5 yd on the wrong side at kick-offs as
unresolved. Among non-circular kick-offs this would affect about 22 players. It would add Law 8 as a new
evidence source.

## 4. ML features and leakage (`ml/`)

**Correctness:** checked on all 756,023 rows, in memory (`ml_validation.py`) and again on the rebuilt files
(`rebuilt_dataset_validation.json`).

- 0 violations of teammates + opponents + ambiguous = visible players (causal and offline);
- 0 violations of raw teammates + raw opponents = visible players;
- 0 inconsistent `identity_quality` values;
- 0 rows lost against the original builder.

**Applied exactly once:**
- Orientation is applied once, with the same per-event result as the renderer (test).
- The builder's offline counts equal the renderer's colours (test).
- Identity never touches coordinates.

### Leakage

The offline identity resolution uses records up to 2 s **after** the event:
- continuity "after" neighbours;
- structural partners;
- the two-neighbour detector;
- simultaneous partners.

| Measure | Value |
|---|---|
| Rows with the predicted event's snapshot within 2 s | 72.9% |
| Rows where offline features would differ from causal ones | 3,801 (0.50%) |
| … of which the predicted event's snapshot is in the window | 3,255 |

Measured information:
- Every identity flag carries detectable but tiny target information beyond the current event type
  (conditional MI 0.0002–0.0003 bits, above the 99th permutation percentile). That is about 1.5% of a coarse
  actor-position feature (0.017 bits).
- The causal flag carries no less than the offline one, so this reflects real situations such as crowding
  and copies, not future access.

**Fix:** the dataset's feature columns now use a new **causal** resolution. Each record is resolved only
from itself and earlier records in (period, timestamp, index) order. The offline values stay in clearly
named `*_offline` columns for data-quality analysis.

A property test on real data proves causality: truncating a match after any record leaves every earlier
causal result unchanged.

**Price:** a snapshot copied from a *later* event cannot be recognised as a copy at its own time.
Dataset-wide, 567,057 points (2.7%) keep their raw reading in causal mode. They are mostly Ball Receipt*
records, rarely dataset rows.

### Predictive check (`ml/model_probe/`)

HistGradientBoosting, GroupKFold(5) by match, macro-F1 pooled; paired per-fold difference vs causal V1.
The fold SD of V1 is 0.00085.

| Variant | Macro-F1 | Δ vs V1 |
|---|---|---|
| V0 original | 0.3025 | −0.0002 ± 0.0006 |
| V1o offline | 0.3025 | −0.0003 ± 0.0006 |
| **V1 causal** | **0.3028** | — |
| V2 exclude uncertain | 0.3021 | −0.0007 ± 0.0017 |
| V3 mask | 0.3029 | +0.0002 ± 0.0008 |
| V4 no team counts | 0.2810 | −0.0218 ± 0.0024 |

Future-dependent cleaning gives no measurable advantage. The team-count features themselves matter.

### Orientation

- 55 rows (0.007%) were oriented by a route that can read later records: duplicates, camera continuity,
  reviewer decisions, and one Problem A temporal-neighbour case that already existed at HEAD.
- 39 of them have changed actor coordinates.
- They are flagged by the new column `orientation_uses_later_records`.
- 481 rows use the period's camera side, a stadium property estimated from the whole period.

## 5. Production dataset (rebuilt in Stage B)

`scripts/build_full_dataset.py` wrote 417 per-match CSVs, a combined CSV and a manifest to `data/processed/`
(gitignored):
- **756,023 rows**, schema identical across matches, 0 duplicate events;
- 0 real mismatches against the independent in-memory build (the 2 apparent ones are NaN = NaN actor
  positions);
- class shares unchanged: Pass 46.64%, Carry 41.56%, Ball Recovery 4.76%, Clearance 1.83%, Dribble 1.53%,
  Miscontrol 1.30%, Shot 1.24%, Dispossessed 1.15%;
- `identity_quality`: clean 752,253 / corrected 1,751 / uncertain 2,019;
- the 3 previous files were backed up to `data/interim/processed_backup_2026-10-27/`.

## 6. Final identity totals (`final_identity_counts.json`, offline)

All counts below are out of 21,193,635 player observations.

- **Changed vs the original pipeline: 2,819 (0.0133%)**
  - goalkeeper rule: 1,175
  - copy/derived view: 998
  - legacy groups whose source was overturned by evidence: 646
- **Unresolved: 12,291 (0.058%)**, in 6,960 snapshots

| Reason for unresolved | Points |
|---|---|
| copy group undecided | 3,670 |
| simultaneous contradiction | 3,478 |
| contradicted by both neighbours | 2,883 |
| copy flag deviation | 1,134 |
| keeper goal side unverified | 896 |
| record contradicted | 230 |

These replace the 2,822 / 9,679 of the 10-26 report: the simultaneous rule and the small-copy fix are new.

## 7. Limitations for the thesis

1. ~2–4% of the 1.34% residual transitions are estimated to be genuine single-player label disagreements
   between snapshots more than 0.25 s apart. They cannot be attributed to a record and are left as read.
2. Records without a previous snapshot, such as the first kick-off of a period, cannot be checked by the
   two-neighbour rule. Frame 505 and 3788749 are scrambled kick-offs that remain as read.
3. In copied records the actor flag can mark the source event's player (case_05). The actor ring shows the
   raw flag.
4. Causal features keep the raw reading for snapshots borrowed from later events (0.50% of rows differ from
   offline).
5. 55 rows' orientation used later records; they are flagged.
6. The model probe is not a thesis baseline. Re-run `diagnostics/model_probe.py` with the final model.

## Files

| Path | Content |
|---|---|
| `transitions.csv.gz`, `pair_strata.csv`, `transition_classes.csv`, `transition_summary.json`, `view_conflicts.csv` | §1 |
| `kickoff_law8.csv` | Law 8 check |
| `case_visualizations/`, `five_cases_checks.csv` | §2 |
| `frame_505/` | §3 |
| `ml/rows.csv.gz`, `ml/invariants.json`, `ml/leakage.json`, `ml/model_probe/` | §4 |
| `rebuilt_dataset_validation.json`, `dataset_sha256_pre_commit.txt` | §5 |
| `final_identity_counts.json` | §6 |
| `diagnostics/` | every script, runnable from the project root |
