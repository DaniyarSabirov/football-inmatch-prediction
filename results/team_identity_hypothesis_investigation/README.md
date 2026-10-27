# Team identity: Hypothesis A vs Hypothesis B — investigation report

Status: investigation only. No production code, raw data, ML data, GIFs, orientation decisions or earlier
evidence were changed. All numbers come from the diagnostic scripts in `diagnostics/`, which run on the
current pipeline.

## Files

| File | Content |
|---|---|
| `hypothesis_comparison.csv` | the 8 known cases: events, team ids, raw flag counts, Problem A, Problem C source, near-simultaneous relations, evidence, verdict |
| `temporal_threshold_analysis.csv` | per threshold (0, 0.1, 0.3, 1, 10, 100 ms): pairs, identical / near coordinates, conflicts, spatial safety, flagged frames touched, keeper-verified effect of propagation |
| `rule_outcomes_by_threshold.csv` | keeper-verified fixed/broken counts for each candidate rule and threshold |
| `full_dataset_audit.csv` | root-cause class of each of the 102 strong + 2,094 weak flagged frames |
| `ml_impact_analysis.md` | ML feature dependence, rows affected, visualization vs ML differences |
| `pairs.csv.gz` | 560,769 near-simultaneous record pairs (≤ 100 ms) with raw and drawn relations |
| `record_perspective.csv.gz` | 1,357,626 records: empirical flag view from temporal neighbours |
| `keeper_validation.csv.gz` | 488,292 keeper observations: current, own-team and opposing readings vs goal side |
| `flagged_frame_audit.csv` | per flagged frame: candidate corrections evaluated on unambiguous pairs |
| `ml_rows_impact.csv.gz`, `ml_impact_summary.json` | in-memory ML rows with exposure flags |

Diagnostics (`diagnostics/`), run in this order to reproduce everything:

1. `pair_audit.py`
2. `record_perspective.py`
3. `keeper_validation.py`
4. `flagged_frame_audit.py`
5. `classify_flagged.py`
6. `rules_evaluation.py`
7. `ml_impact.py`
8. `known_cases.py`

`snapshot_relations.py` is a shared helper. Each script documents its definitions.

## 1. What the data format actually says

The official spec shipped with the data (`data/raw/statsbomb/doc/Open Data 360 Frames v1.0.0`) defines:

- **`location`**: "coordinates oriented in the same direction as the linked event (i.e. the actor's team
  attacking 0 to 120)".
- **`teammate`**: "the player plays on the same team as the 'actor' in this event".
- **`actor`**: the player performing the event.
- **No identities:** freeze frames "will not contain player identification, beyond their team".
- **Known quirk:** "The 'keeper' attribute will, in some rare cases, refer to the keeper on the same team
  as the 'actor', without being marked as a 'teammate'."

So each record is fully event-relative. The same physical player **legitimately** has opposite
`teammate` values in two opposing teams' records of the same moment, and only spatial correspondence can
link players across records.

## 2. Raw JSON → rendered colour (current pipeline)

1. The raw record is joined to its event by `event_uuid`. Each record belongs to exactly one event; the
   association itself was never found to be wrong.
2. Problem A (`orientation_correction.py`) rotates the snapshot when its coordinates are in the other
   team's frame; the display layer resolves UNCERTAIN snapshots.
3. The display plan (`broadcast_orientation.py`) rotates per period and team. Coordinates only.
4. Problem C (`team_identity.py`) works out each point's team:
   - not shared → `event team if teammate else opponent` (the spec reading);
   - byte-identical copy across teams → `source team if teammate else opponent`, where the source is the
     Problem-A-confident member.
5. The display-only refinement (`refine_shared_sources_by_continuity`) can switch a shared group's source
   when both neighbouring snapshots demand it; it does so in 2 groups.
6. Colour = team colour of the resolved team.

There is no "reference team" in the colour path. Team normalization changes coordinates only, so the
fixed-reference vs event-relative question does not affect colours.

## 3. Hypothesis A: interpretation errors

| Question | Answer | Evidence |
|---|---|---|
| Is `teammate` relative to the current event's team? | **Yes** (spec), and the pipeline reads it so | Unshared records: the own-team reading puts the keeper at his own goal in **99.7%** of 280,339 keeper observations |
| Is our event-relative reading wrong in general? | **No** | as above |
| Can the same player have opposite `teammate` values in two events? | **Yes, legitimately**, for opposing teams' records | 48,273 (identical coordinates) + 27,544 (mirrored with constant offset) cross-team pairs within 100 ms have inverted flags, i.e. are spec-consistent views of one moment |
| Is the event-to-snapshot association wrong? | **No evidence** | `event_uuid` joins are 1:1 |
| Does reference-team normalization affect colours? | **No** | coordinates only |
| Does shared-snapshot detection miss copies that aren't byte-identical? | **Yes** | 56,458 cross-team same-scene pairs at 0 ms (mostly identical coordinates) have *equal* flags, i.e. one team's view copied. Some differ only in `actor` (case_01, case_05) or in visible_area, so Problem C misses them. |
| Does Problem C pick the right view for byte-identical copies? | **Mostly; not always** | Shared records: 98.1% keeper-consistent where the flags are the own view, and the opposing view is correctly applied in 97.5% (n = 9,395). Problem C couples the flag view to the coordinate owner, and 3857293 breaks that: coordinates in Ghana's frame, flags in Uruguay's. |
| Are some records' flags not re-expressed even though the coordinates are? | **Yes** (raw spec violation, but structured) | case_02: the Köln Block is mirrored with a constant 2.53 yd offset from Leverkusen's Block 82 ms earlier, but its flags are still Leverkusen's (EQUAL) |

**Conclusion:** Hypothesis A is **partly supported**.
- The pipeline reads `teammate` correctly per the spec.
- Its weakness is structural: copy detection requires byte identity, and a copy's flag view is tied to its
  coordinate owner.
- The raw data contains copied or derived records whose flags weren't re-expressed (structured, detectable
  spec violations), plus some genuinely scrambled labels (505).

## 4. Hypothesis B: transfer identities between simultaneous snapshots

- **Spatially feasible:** at 0 ms, 448,389 pairs have identical coordinates. Among same-scene pairs whose
  drawn colours conflict, 93 of 177 pair every player unambiguously (margin ≥ 1 yd); the rest involve
  crowding.
- **Direction is the problem.** Measured against keeper ground truth (`rule_outcomes_by_threshold.csv`):

| Rule (0 ms) | Changed (keeper evidence) | Fixed | Broken |
|---|---|---|---|
| R4: earlier → later (your proposal, literal) | 81 | 11 | **17** |
| R4 at 100 ms | 313 | 27 | **40** |
| R2: coordinate-owner view (Problem C's logic extended to near-copies) | 9 | 5 | 4 |
| R3: majority of the same-moment cluster | 3,587 | 45 | **3,542** |

Event order does not say which record is right (case_05: the *earlier* Barcelona record is the wrong
one). The coordinate owner is not reliable either (3857293).

**Conclusion:** Hypothesis B is technically feasible *as spatial matching*. As a timestamp- or order-based
transfer rule it is **unsafe**: it breaks more correct frames than it fixes.

## 5. Temporal thresholds (`temporal_threshold_analysis.csv`)

| Threshold | Pairs | Identical coordinates | Near (incl. shifted / mirrored) | Conflicting same-scene pairs | of which spatially unambiguous | Strong frames in a conflicting pair |
|---|---|---|---|---|---|---|
| 0 / 0.1 / 0.3 ms | 471,553 | 448,389 | 19,602 | 177 | 93 | 7 |
| 1 ms | 472,082 | 448,448 | 20,027 | 179 | 95 | 7 |
| 10 ms | 476,731 | 449,219 | 23,449 | 194 | 99 | 7 |
| 100 ms | 560,745 | 467,090 | 76,259 | 740 | 368 | 15 |

**No timestamp in the dataset has sub-millisecond digits** (checked over all records). So 0.1 ms and
0.3 ms are mathematically identical to 0 ms. Timestamps were parsed to integer milliseconds.

**Exact equality (0 ms) is the best threshold.** It captures essentially all verbatim copies (identical
coordinates). Tolerances up to 10 ms add only 5,178 pairs and 17 conflicts. 100 ms adds 89,192 pairs,
mostly genuinely different snapshots, and doubles the harm of naive propagation (40 broken). Derived
records such as case_02 (82 ms, constant offset) need *structural* detection (mirror/shift + equal flags),
not a wider time window.

## 6. The 102 strong and 2,094 weak flagged frames (`full_dataset_audit.csv`)

Classification uses unambiguous pairs only, plus structure. The keeper check is independent of both.

| Root cause | Strong | Weak | Keeper check (strong): consistent / inconsistent / none |
|---|---|---|---|
| crowding artefact (no unambiguous colour change) | 47 | 644 | 39 / 0 / 8 |
| disagrees with one neighbour only (that neighbour is the likelier culprit) | 5 | 0 | 3 / 0 / 2 |
| copy missed by Problem C (verbatim cross-team copy, not byte-identical) | 3 | 7 | 0 / 2 / 1 |
| derived record, flags not re-expressed | 8 | 11 | 0 / 1 / 7 |
| shared group, wrong view | 5 | 161 | 1 / 1 / 3 |
| record-level inversion (no partner; inversion fits both neighbours) | 5 | 250 | 0 / 0 / 5 |
| raw label scramble (505 type) | 16 | 224 | 4 / 1 / 11 |
| insufficient evidence | 13 | 797 | 7 / 0 / 6 |

- **The continuity detector overstates the problem.** 52 of 102 strong frames show no reliable colour
  change. Across all 1,600 flagged frames with keeper evidence, only 35 (2.2%) have a wrongly coloured
  keeper.
- **Weak frames:** single-link verdicts are unreliable. 109 of 112 keeper-evidenced weak "inversions" have
  a correctly coloured keeper.
- **Structurally explained strong frames:** 21 (copy / derived / shared / inversion). For **10** of them a
  correction resolves the frame on its own neighbour evidence. For the other 11 the structure is clear but
  the neighbour evidence is not decisive.
- **Genuinely inconsistent raw labels:** 16 strong scrambles (505 type), and 224 weak ones whose direction
  is not determinable.

Beyond the flagged set, keeper truth over 488,287 records: the current colours are correct for **99.53%**
and wrong for **2,272**. 957 of those are keeper-only flag errors (the documented StatsBomb quirk).

## 7. The five reviewed GIFs: why players change colour

| Case | Mechanism | Pipeline or data? |
|---|---|---|
| case_01 (3893791, Costa Rica Duel, 18:25.761) | Verbatim copy of Spain's Ball Receipt* record at the same ms: identical coordinates, 18/18 equal flags, only `actor` differs. Problem A rotated the coordinates; the flags were still read as Costa Rica's. | **Pipeline**: copy detection too strict |
| case_02 (3895107, Köln Block, 82:13.060) | Record derived from Leverkusen's Block 82 ms earlier: mirrored with a constant 2.53 yd offset, flags not inverted (16/16 equal). Inverted against Köln's own Pass, equal to Leverkusen's records. | **Data** spec violation, structured and detectable |
| case_03 (3857256, Serbia Pass, 90:23.300) | Goal-mouth crowd; only 2 unambiguously paired players change colour against each neighbour | Mostly a **detection artefact**; at most 2 raw labels |
| case_04 (3844386, Sweden Pressure, 31:58.616) | 0/4 and 1/2 unambiguous disagreements; keeper coloured correctly | **Not an error** (crowding) |
| case_05 (3773386, Barcelona Ball Receipt*, 27:13.282) | Verbatim copy with Alavés's Block at the same ms (actor differs). The flags are Alavés's view; the Barcelona record is read in Barcelona's (6/6 unambiguous wrong). Reading both in Alavés's view makes both consistent. | **Pipeline**: copy detection too strict |
| 505 (3930164, kick-off) | No copy; 10/17 unambiguously paired players inconsistent with the next snapshot; Law 8 violated by 8 points | **Data**: scrambled raw labels |
| 3857293 ×2 (Uruguay–Ghana) | Byte-identical copy, coordinates in Ghana's frame, flags in Uruguay's view; Problem C chose Ghana | **Pipeline** assumption (flag view = coordinate owner); fixed in the display |

## 8. Recommended correction strategy (not implemented; awaiting approval)

1. **Keep the spec reading as the default.** It is validated at 99.7% for unshared records.
2. **Fix the detector before fixing colours.** Pair players unambiguously (optimal assignment, margin ≥ 1
   yd), so crowding stops producing false flags. This removes 52 of 102 strong and 644 of 2,094 weak
   flags without changing a single colour.
3. **Generalize copy detection:**
   - Treat as one group any opposing-team records in the same period at the **same millisecond (0 ms)**
     whose raw coordinates are identical (max pair distance < 0.01 yd) and whose flags are equal.
   - Add **derived records** (mirror or constant-offset relation, equal flags) as a structural class, with
     no time-based tolerance.
4. **Choose a group's flag view from evidence, never from order or coordinate owner.**
   - Use the existing, already-validated rule: colour continuity with non-group snapshots on both sides,
     unambiguous pairs, ≥ 50% vs ≤ 10%.
   - Where that is undecided (for example case_01, a same-ms cluster with no other neighbour within 2 s),
     do not guess. Either keep the current colours or draw the group's points in the neutral "team
     unclear" colour; **your choice**.
5. **Isolated raw scrambles (505 type):** no correction. Optionally flag the frame or use the neutral
   colour; no recolouring by kick-off assumptions.
6. **Expected effect:**
   - the 10 strong frames with decisive evidence are fixed, along with the copy and derived cases among
     the weak ones;
   - case_01, case_02 and case_05 become consistent (case_01 only if the cluster rule's undecided policy
     allows it);
   - no change to the 99.5% of keeper-consistent records outside such groups.
   - This will be re-verified with the same keeper and continuity audits before and after.
7. **ML:** move this group logic into `team_identity.py` so the renderer and the builder share it. Then
   rebuild the ML dataset only after your approval, with a before/after diff of the affected rows: about
   264 rows in error classes plus the keeper-quirk rows. Separately, consider applying the display's
   UNCERTAIN-orientation resolution in the builder (535 rows, actor coordinates).

## 9. What remains ambiguous

- 13 strong / 797 weak frames with insufficient neighbour evidence.
- 11 structurally explained strong frames without decisive evidence.
- 16 strong / 224 weak raw scrambles.
- The 2,272 keeper-verified errors that are not in any structural class (mostly the documented keeper
  quirk).

For these, the data does not determine the correct colours.
