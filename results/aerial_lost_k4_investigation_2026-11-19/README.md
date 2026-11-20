# Aerial Lost and the proposed companion rule K4: quantitative investigation

Date: 2026-11-19. Read-only.

**Scope:**
- **Matches.** The 304 Tier B development matches, periods 1–4. No final-test match was
  opened.
- **Rule status.** K4 is **not** active in any task version. It is implemented only
  inside `diagnostics/k4_investigation.py`, for counting.

**Reproduce** (about 20 s):

```bash
.venv/bin/python results/aerial_lost_k4_investigation_2026-11-19/diagnostics/k4_investigation.py
```

**Outputs:**
- `summary.json`;
- `tables/aerial_lost_records_with_k4_evidence.csv` (one row per Aerial Lost record, with
  every criterion);
- `tables/variant_comparison.csv`;
- `tables/class_distribution_v1_vs_k4.csv`;
- `examples/aerial_lost_examples.csv` (12 contests with context rows).

**Evidence labels.** [F] measured; [J] judgment.

## 1. Candidate rule K4

A `Duel` with subtype `Aerial Lost` is a companion record when **all** of the following
hold:
- (a) its `related_events` contain a record with `aerial_won = true` (an explicit
  relationship);
- (b) that record has the same timestamp;
- (c) that record belongs to the opposite team;
- (d) it is the adjacent raw record (StatsBomb `index` ± 1), a documented pattern;
- (e) it is a retained canonical action.

This mirrors the structure of K2 (Tackle) and K3 (50/50). Timestamp equality alone is
never sufficient.

## 2. Results [F]

| Quantity | Count | % of Aerial Lost |
|---|---|---|
| Aerial Lost records | 8,549 | 100 |
| (a) linked to a record with `aerial_won` (the winner's header) | 8,549 | 100.0 |
| Link reciprocated: the winner's `related_events` contains the Aerial Lost record | 8,549 | 100.0 |
| (b) same timestamp | 8,549 | 100.0 |
| (c) opposite teams | 8,549 | 100.0 |
| (d) adjacent raw records | 8,549 | 100.0 |
| (e) winner record is a canonical action | 8,549 | 100.0 |
| **All criteria: one physical contest, confirmed companion** | **8,549** | **100.0** |
| Uncertain cases (any criterion fails) | 0 | 0 |
| Same player on both records | 0 | 0 |
| Winner recorded **before** the loser | 1,185 | 13.9 |
| Aerial Lost with a usable 360 snapshot | 7,842 | 91.7 |

**The winner's record.**
- **Type.** Pass 5,259 (61.5 %), Clearance 2,335 (27.3 %), Shot 720 (8.4 %),
  Miscontrol 235 (2.7 %).
- **360 coverage.** It has a snapshot in 99.95 % of the cases where the loser has one.

**Transitions in the canonical v1 sequence** (K4 off):
- **Aerial Lost → its own winner record:** 7,364.
- **Winner record → its own Aerial Lost:** 1,185.
- Together these are all 8,549 contests: every Aerial Lost is adjacent to its partner in
  the sequence.
- **Next after Aerial Lost:** Pass 52.4 %, Clearance 28.3 %, Shot 8.4 %.
- **Previous before Aerial Lost:** Pass 88.3 %. This is usually the long ball that
  creates the contest.

**The Duel class under v1.**
- **Composition.** All 8,549 canonical Duel observations are Aerial Lost records.
- **Why.** K2 already removes every linked Tackle duel, and no other duel subtype occurs.

**Effect of activating K4** (`variant_comparison.csv`):

| | actions15_360current_v1 (K4 off) | hypothetical +K4 |
|---|---|---|
| Observations | 782,377 | 773,828 (−8,549) |
| Transitions | 781,735 | 773,186 |
| Eligible rows (current 360) | 690,147 | 682,305 (−7,842 rows whose current event is Aerial Lost) |
| v1 rows whose target event/class changes | — | 6,791 |
| **Classes with any observation** | 15 | **14 (Duel empty)** |
| Duel as target of eligible rows | 6,851 (0.99 %) | 0 |
| Rows in cells with P ≥ 0.9 | 0 | 0 |
| Largest cell probability (n ≥ 50) | 0.855 | 0.855 |
| H(target) / H(target \| current) [bits] | 2.303 / 1.671 | 2.243 / 1.640 |

**Class shares of the other classes under K4.** These change by ≤ 0.4 percentage points
(`class_distribution_v1_vs_k4.csv`). Clearance, Shot and Miscontrol targets decrease
slightly. The rows that targeted them were the contests in which the winner's record came
first.

**Would K4 remove a meaningful independent action?**
- **No contest is lost.** In 0 cases would a contest disappear from the sequence, because
  the winner's header is always retained.
- **But information is lost.** In all 8,549 cases, the only *event-type* evidence that
  an aerial contest took place would disappear. Under v1, the winner's record is an
  ordinary Pass, Clearance, Shot or Miscontrol class. The fact that it was a headed duel
  survives only in the `aerial_won` attribute, which is not part of the canonical
  sequence.
- **Who is removed.** The removed record is the losing player's participation, a
  different player from the winner (0 shared players).

## 3. Examples (`examples/aerial_lost_examples.csv`; ids shortened)

| # | Match, period | Index | Aerial Lost (team, player) | Winner record (team, player) | Timestamp | Order | 360 |
|---|---|---|---|---|---|---|---|
| 1 | 3788748 P2 | 3368 / 3369 | d5de850c (942, 3581) | 6e53e6dd Pass (Recovery) (912, 16570) | 00:47:22.712 | loser first | none |
| 2 | 3835324 P2 | 3470 / 3471 | cde62017 (851, 10645) | 7eaaa510 Pass (858, 10220) | 00:47:57.163 | loser first | both |
| 3 | 3835334 P1 | 528 / 529 | 91ba4a9c (855, 25544) | 440fb243 Clearance (862, 10120) | 00:18:20.155 | loser first | both |
| 4 | 3835338 P1 | 694 / 695 | b400978c (920, 13713) | 8c910285 Clearance (857, 15178) | 00:18:57.432 | loser first | both |
| 5 | 3835332 P2 | 1663 / 1664 | 5c94d1c8 (851, 10647) | 073ba409 Shot (864, 10159) | 00:01:06.494 | loser first | both |
| 6 | 3835324 P2 | 3330 / 3331 | ca5e5697 (851, 10645) | 1fe5fe41 Shot (858, 4633) | 00:41:44.984 | loser first | both |
| 7 | 3835334 P2 | 1875 / 1876 | f0f9db93 (855, 25770) | bc5906c7 Miscontrol (862, 32460) | 00:17:38.646 | loser first | both |
| 8 | 3837844 P2 | 2931 / 2932 | f1f561a7 (131, 13621) | 10bc68d3 Miscontrol (139, 356762) | 00:28:17.951 | loser first | both |
| 9 | 3895194 P1 | 1200 / 1199 | f4bddb4e (172, 25118) | 79c3b324 Pass (Recovery) (904, 32712) | 00:27:08.542 | winner first | both |
| 10 | 3930183 P2 | 2242 / 2241 | 48351349 (905, 16344) | cbf613c4 Pass (908, 8967) | 00:21:37.025 | winner first | both |
| 11 | 3998859 P2 | 2201 / 2200 | 68fc18bd (851, 10658) | 70a4dfd4 Pass (Recovery) (861, 31633) | 00:30:15.076 | winner first | both |
| 12 | 3902240 P1 | 1371 / 1370 | bc58a9a0 (851, 10649) | 72bad48b Pass (Recovery) (863, 32694) | 00:39:44.671 | winner first | both |

**Details in the CSV.**
- **Links.** In every example, each record's `related_events` contains the other's UUID.
  The CSV gives the full UUIDs, `related_events` and two context records on each side.
- **Uncertain cases.** None exist in Tier B, so no uncertain example can be shown.
  Contradicting cases (separate aerial duels by different players about 1.4 s apart)
  are in `results/investigation_360_taxonomy_2026-11-18/examples/ex_goal_keeper_duel_5050.csv`.
  K4 would not merge those: they are not linked and have no shared timestamp.

## 4. Assessment and recommendation [J]

**Confirmed companion.** By every criterion, each Aerial Lost record is the losing side
of one physical aerial contest, recorded in addition to the winner's on-ball action. This
is the same structure as the Tackle (K2) and 50/50 (K3) companions.

**Activating K4 is nonetheless not recommended for the canonical task.** The reasons:
1. **It removes the class.** K4 removes the entire Duel class from the canonical
   sequence (15 → 14 classes with any observation). The approved rules require 15
   classes and forbid removing entire event classes. A K4 task would therefore need a
   new version with K = 14, which is a different task, not a refinement.
2. **It erases the contest from the type sequence.** Under v1 the Aerial Lost record is
   the only event-type marker of an aerial contest. 88 % of them follow a Pass, typically
   a long ball. Predicting "an aerial contest happens next" (Duel as target; 6,851
   eligible rows) and "which header follows a contest" (Duel as current event) are
   football-relevant questions. K4 would erase them, unless an `aerial_won` attribute
   were added to the winner's record later.
3. **It is not deterministic bookkeeping like K2/K3.**
   - *The cell.* Under v1 the pair produces no near-deterministic cell: the largest
     P(next | Duel) is 0.52 (Pass).
   - *How K2 differed.* Dispossessed → Duel had P = 1.000 before K2.
   - *Predictability.* The next class after Aerial Lost is informative, not mechanical.
4. **The change is small.** It touches 1.0 % of eligible rows (7,842 removed as current
   events, 6,791 targets changed). The decision does not affect the other 14 classes
   materially (≤ 0.4 percentage points).

**Recommendation:**
- **Keep K4 inactive** in `actions15_360current_v1`.
- **Document the Duel class** as "aerial contest, recorded by the losing player's
  record". It is a known residual pair with its winner's action.
- **If the researcher prefers strict one-record-per-contest consistency,** define a new
  version (e.g. `actions14_360current_v2`). In it, K4 is active and Duel is no longer a
  class. Preferably the aerial information is retained as a causal feature of the
  current record: whether *its* header was an aerial duel, known at observation time.
- **Approval.** That change requires explicit researcher approval and new baselines.
