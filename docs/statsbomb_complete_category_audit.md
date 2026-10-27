# Complete StatsBomb Data Taxonomy and Frequency Audit

Date: 2026-10-20. Prepared as a standalone data audit for the thesis's
prediction-task design. All numbers in this report are read directly from
`results/data_taxonomy/statsbomb_complete_category_audit_data.json`, which
is produced deterministically by `scripts/audit_statsbomb_category_taxonomy.py`
and also drives every sheet in
`results/data_taxonomy/statsbomb_complete_category_audit.xlsx`. No existing
raw data, processed dataset, or prediction-pipeline code was modified to
produce this report.

---

## 0. Scope and population definitions

The local StatsBomb open-data checkout (`data/raw/statsbomb/data/`) contains
**4,235** total match event files, but only **426** of those matches also
have a local `three-sixty/*.json` file. Because the thesis combines event
data with 360 freeze-frame data, this audit's **raw population is the 426
matches with a local 360 file** — the same population
`scripts/audit_dataset.py` and `src/animate_statsbomb_360_match.py`'s
`available_360_match_ids()` already draw from. The other ~3,809 event-only
matches are out of scope and are not individually audited; they are
reported only as a context count (see §8).

**Eligible population** = raw minus the 9 matches in
`configs/excluded_matches.json` (see `src/match_exclusions.py`) = **417
matches**. This reconciles exactly with the previously reported "417 usable
matches" figure from `docs/prediction_task_feasibility_2026-10-14.md` and
`docs/excluded_matches_forensic_analysis_2026-10-18.md` — **verified
current, not assumed**.

One of the 9 excluded matches (`3845506`) has a corrupted `three-sixty.json`
(documented NUL-byte corruption, see `docs/data_quality_investigation_2026-10-13.md`)
and could not even be loaded by this audit's own JSON parser — it is this
audit's one and only malformed-file encounter, logged by the script rather
than silently skipped:

```
match 3845506: JSONDecodeError: Expecting ',' delimiter: line 92794 column 3 (char 2637824)
file: data/raw/statsbomb/data/three-sixty/3845506.json
```

This leaves **425/426** raw matches actually readable and counted below;
the 426th (3845506) contributes 0 to every raw total but is still counted
in "raw population size" and listed with `load_status=excluded` in the
Match Statistics sheet.

Every statistic below is reported for **both** the raw population (425
loaded matches) and the eligible population (417 matches), with the
denominator stated explicitly.

---

## 1. Executive summary

| Metric | Value |
|---|---|
| Raw population (matches with local 360 JSON) | 426 (425 successfully loaded) |
| Eligible population (raw minus documented exclusions) | 417 |
| Excluded matches | 9 |
| Total local StatsBomb event files (incl. matches without 360) | 4,235 |
| Malformed/unreadable files encountered | 1 (`3845506`, already a documented exclusion) |
| Total raw events | 1,582,258 |
| Total eligible events | 1,551,841 |
| Unique primary event types (`type.name`) found in the raw population | **33** |
| Categorical/boolean fields discovered | **99** |
| Distinct categorical values discovered (sum across all fields) | **15,315** |
| Total 360 snapshots (raw / eligible) | 1,381,467 / 1,357,626 |
| Total 360 freeze-frame player entries (raw / eligible) | 21,561,945 / 21,193,635 |
| Goals via `shot.outcome.name == "Goal"` (raw / eligible) | 1,303 / 1,279 |
| Own goals (`Own Goal Against` events, raw / eligible) | 55 / 55 |
| Total goals, shot + own (raw / eligible) | 1,358 / 1,334 |
| Matches reconciled exactly against recorded score | 406 / 425 checked (19 explained — see §5) |
| Current 8-class target implemented as documented | Yes, exactly `MEANINGFUL_EVENT_TYPES` |
| Can the current target distinguish a goal from a non-goal shot? | **No** |

**Headline finding**: the current 8-class target set (`Pass`, `Carry`,
`Ball Recovery`, `Clearance`, `Dispossessed`, `Miscontrol`, `Shot`,
`Dribble`) excludes **`Ball Receipt*`** (423,059 raw occurrences, **26.74%
of all events** — the *second most frequent event type in the entire
dataset*, larger than `Carry`) and **`Pressure`** (135,493 occurrences,
8.56%). Excluding the single largest pass-complement event type from the
"meaningful" set is the most significant omission found by this audit (see
§7).

---

## 2. Complete list of primary event types (raw population)

All 33 types found in the 425 successfully-loaded raw matches. `%` is of
all raw events (1,582,258); eligible `%` is of all eligible events
(1,551,841). "Matches with" uses the raw population size (425) as
denominator unless noted.

| Event Type | Raw Count | Raw % | Eligible Count | Eligible % | Matches With (raw) | In 8 Classes |
|---|---:|---:|---:|---:|---:|---|
| Pass | 447,122 | 28.26% | 438,439 | 28.25% | 425/425 | **Yes** |
| Ball Receipt* | 423,059 | 26.74% | 414,717 | 26.72% | 425/425 | No |
| Carry | 360,203 | 22.77% | 353,132 | 22.76% | 425/425 | **Yes** |
| Pressure | 135,493 | 8.56% | 133,003 | 8.57% | 425/425 | No |
| Ball Recovery | 41,389 | 2.62% | 40,649 | 2.62% | 425/425 | **Yes** |
| Duel | 27,405 | 1.73% | 26,980 | 1.74% | 425/425 | No |
| Block | 16,729 | 1.06% | 16,432 | 1.06% | 425/425 | No |
| Clearance | 16,507 | 1.04% | 16,254 | 1.05% | 425/425 | **Yes** |
| Goal Keeper | 13,322 | 0.84% | 13,071 | 0.84% | 425/425 | No |
| Dribble | 13,248 | 0.84% | 13,028 | 0.84% | 425/425 | **Yes** |
| Miscontrol | 11,501 | 0.73% | 11,291 | 0.73% | 425/425 | **Yes** |
| Shot | 11,033 | 0.70% | 10,831 | 0.70% | 425/425 | **Yes** |
| Foul Committed | 10,518 | 0.66% | 10,310 | 0.66% | 425/425 | No |
| Dispossessed | 10,029 | 0.63% | 9,852 | 0.63% | 425/425 | **Yes** |
| Foul Won | 9,994 | 0.63% | 9,794 | 0.63% | 425/425 | No |
| Interception | 9,895 | 0.63% | 9,701 | 0.63% | 425/425 | No |
| Dribbled Past | 7,722 | 0.49% | 7,587 | 0.49% | 425/425 | No |
| Substitution | 3,678 | 0.23% | 3,610 | 0.23% | 425/425 | No |
| Injury Stoppage | 2,339 | 0.15% | 2,311 | 0.15% | 410/425 | No |
| Half End | 1,858 | 0.12% | 1,824 | 0.12% | 425/425 | No |
| Half Start | 1,858 | 0.12% | 1,824 | 0.12% | 425/425 | No |
| 50/50 | 1,731 | 0.11% | 1,685 | 0.11% | 309/425 | No |
| Tactical Shift | 1,435 | 0.09% | 1,399 | 0.09% | 417/425 | No |
| Referee Ball-Drop | 1,016 | 0.06% | 996 | 0.06% | 306/425 | No |
| Starting XI | 850 | 0.05% | 834 | 0.05% | 425/425 | No |
| Shield | 548 | 0.03% | 543 | 0.03% | 304/425 | No |
| Player Off | 501 | 0.03% | 494 | 0.03% | 257/425 | No |
| Player On | 498 | 0.03% | 491 | 0.03% | 256/425 | No |
| Bad Behaviour | 279 | 0.02% | 271 | 0.02% | 162/425 | No |
| Error | 244 | 0.02% | 237 | 0.02% | 185/425 | No |
| Offside | 144 | 0.01% | 141 | 0.01% | 121/425 | No |
| Own Goal Against | 55 | 0.00% | 55 | 0.00% | 51/425 | No |
| Own Goal For | 55 | 0.00% | 55 | 0.00% | 51/425 | No |

Full per-type statistics (including per-match mean/median/min/max/std,
`type.id`, and a thesis-relevance note for every type) are in the **Event
Types** sheet of the workbook — every type is included, none dropped for
rarity (`Own Goal For`/`Against` at 55 occurrences each are the rarest, and
are both still fully reported).

**Discrepancy vs. the task's illustrative field list, discovered and
documented, not assumed**: an ad-hoc scan of an arbitrary sample of the
broader ~4,235-match corpus (outside the 360-having population) also
surfaced `Camera On` and `Camera off` event types. Neither occurs in any of
the 425 raw-population matches actually audited here — they appear to be
specific to certain non-360 competition feeds. They are therefore absent
from the Event Types sheet (33 types, not 35) and are noted here rather
than silently omitted.

---

## 3. Nested categorical fields — summary

The audit script recursively discovers **every** field path in the event
JSON schema (no hardcoded field list), classifying each leaf as
`categorical`, `boolean`, `numeric`, `identifier`, `timestamp`, or
`enum_id` (a small integer code that mirrors a sibling `.name` field, e.g.
`type.id` mirrors `type.name` — reported separately so integer codes are
never mistaken for meaningful numeric distributions). Totals:

| Data type | Field count |
|---|---:|
| boolean | 61 |
| categorical (string) | 38 |
| enum_id (integer code mirroring a `.name` field) | 37 |
| numeric | 10 |
| list_of_scalars (coordinates, UUID lists — not categorical) | 9 |
| identifier (`id`/`uuid`/`*_id`) | 4 |
| list_of_objects (`tactics.lineup`, `freeze_frame`, `shot.freeze_frame`) | 3 |
| timestamp | 1 |
| **Total discovered fields** | **163** |

99 of these (boolean + categorical) carry enumerable category values,
together producing **15,315 distinct (field, value) combinations** —
635 in the main **Category Values** sheet, 14,680 in **Category Values
(High Card)** for the five fields with more than 300 distinct raw values
(`pass.recipient.name`: 3,271 unique; `player.name`; `tactics.lineup[].player.name`;
`shot.freeze_frame[].player.name`; `substitution.replacement.name`) — all
of these are person names, moved to a separate sheet purely for
readability, **nothing is truncated or capped**.

### 3.1 Two important "missing vs. implicit" conventions found in the raw data

1. **Boolean flags are omitted, not written as `false`, when they don't apply.**
   `under_pressure`, `counterpress`, `out`, `off_camera`, `shot.first_time`,
   `shot.one_on_one`, every `block.*` boolean, `clearance.right_foot`-style
   per-body-part booleans, etc. only ever take the literal value `True` in
   the raw JSON (confirmed: `unique_values_raw == 1` for every one of them).
   "Missing" for these fields means **implicitly False**, not "unknown" —
   e.g. `under_pressure` is present on 294,629/1,582,258 raw events
   (18.62%); the other 81.38% were simply not under pressure, not
   unrecorded.
2. **`pass.outcome` is only written when a pass did *not* complete.**
   Of 447,122 raw `Pass` events, only 79,354 (17.75%) carry a
   `pass.outcome` sub-object at all; its 5 values are `Incomplete`
   (86.28% of those 79,354), `Out` (9.10%), **`Unknown`** (2.54% — a
   genuine *explicit* category, not a missing-value placeholder, correctly
   kept distinct from the 82.25% of passes with no outcome field at all),
   `Pass Offside` (1.70%), `Injury Clearance` (0.39%).

### 3.2 Illustrative field-list correction, discovered from the data

The task's illustrative field list includes `player.position.name`. The
actual StatsBomb schema places `position` as a **top-level sibling of
`player`** on the event object, not nested inside it:
`event.position.name`, confirmed present on 1,575,186/1,582,258 raw events
(99.55%) with 25 distinct position values, while `player.position.name`
occurs **zero** times (verified by direct query, not assumed from
documentation). Reported exactly as discovered in the Data Quality sheet.

### 3.3 A data-driven field the task's list did not anticipate

`shot.freeze_frame` — a per-shot array of nearby players (teammate/
opponent/position/location) **embedded directly in the event JSON**,
present on 10,737/11,033 raw `Shot` events (97.3%). This is **not** the 360
product (`three-sixty.json`); it predates/exists independently of 360
coverage and must not be confused with it (see §4). It was discovered
purely by walking the actual data, confirming the audit's "discover, don't
assume" approach surfaces real structure the illustrative spec missed.

### 3.4 Data-quality finding: inconsistent player/team name spellings

Checking whether every `<container>.id` maps to exactly one
`<container>.name` (and vice versa) surfaced **7 containers with real
inconsistencies** — the same StatsBomb numeric ID is spelled two different
ways across different events/matches:

| Container | id | Names found |
|---|---|---|
| `team` / `possession_team` | 147 | "Marseille", "Olympique de Marseille" |
| `player` (and its 4 mirrors: `tactics.lineup[].player`, `pass.recipient`, `shot.freeze_frame[].player`, `substitution.replacement`) | 9914 | "Daniel Ward", "Danny Ward" |
| | 4354 | "Phil Foden", "Philip Foden" |
| | 18617 | "Mykola Matvienko", "Mykola Matviyenko" |
| | 3961 | "N''Golo Kanté", "N'Golo Kanté" (double vs. single apostrophe) |
| | 401635 | "Clare Hunt", "Clare Josephine Hunt" |
| | 4951 | "Quinn Quinn", "Rebecca Quinn" |

**Implication for modeling**: any feature or join keyed on player/team
*name* (rather than the numeric id) will silently split these 6
players'/1 team's events into two different categories. Always join on
`id`, never on `name`, for exactly this reason — now verified, not
hypothetical.

---

## 4. 360 freeze-frame data inventory

| Metric | Raw | Eligible |
|---|---:|---:|
| Matches with 360 data | 426 (425 loaded) | 417 |
| Total 360 snapshots (`three-sixty.json` records) | 1,381,467 | 1,357,626 |
| Total freeze-frame **player entries** (sum of `freeze_frame` array lengths) | 21,561,945 | 21,193,635 |
| Snapshots per match: mean / median / min / max / std | 3,250.5 / 3,238 / 1 / 4,511 / 449.5 | 3,255.7 / 3,236 / 2,268 / 4,511 / 423.3 |
| Player entries per match: mean / median / min / max / std | 50,734.0 / 51,419 / 13 / 80,635 / 10,014.0 | 50,824.1 / 51,316 / 30,411 / 80,635 / 9,759.3 |
| Snapshots with an empty `freeze_frame` array | 0 | 0 |

A snapshot (one `three-sixty.json` record = one `event_uuid` +
`visible_area` + `freeze_frame` array) is **never** the same thing as a
freeze-frame player entry (one element of that array, ~15.6 players per
snapshot on average) — the two are tracked and reported separately
throughout this audit, per the task's explicit requirement.

**Raw 360 freeze-frame categorical fields** (every player entry has
exactly these four keys — no additional categorical fields were
discovered in the raw 360 product):

| Field | Present | Unique values | Breakdown |
|---|---:|---:|---|
| `freeze_frame[].teammate` | 21,561,945 (100%) | 2 | boolean, both True/False actually occur |
| `freeze_frame[].actor` | 21,561,945 (100%) | 2 | boolean |
| `freeze_frame[].keeper` | 21,561,945 (100%) | 2 | boolean |
| `freeze_frame[].location` | 21,561,945 (100%) | — | coordinate pair, not categorical |
| `visible_area` | 1,381,467 (100%, one per snapshot) | — | 10-float polygon, not categorical |

No player identity (no `id`/`name`) is present in the raw 360 product —
only `teammate`/`opponent` (via `teammate=False`), `keeper`, and `actor`
flags plus location. This is a structural property of the StatsBomb 360
product, not a data-quality defect.

**Raw vs. processed 360 schema difference**: `src/build_statsbomb_360_dataset.py`
derives `visible_teammates`/`visible_opponents`/`visible_ambiguous` (not
simply `teammate`/`not teammate`) because of two corrections not present in
the raw product: orientation correction (`src/orientation_correction.py`)
and cross-team snapshot-sharing resolution (`src/team_identity.py`,
"Problem C"). The processed CSV schema is therefore a genuinely different,
derived categorical space from the raw 360 JSON, not just a renamed copy —
documented in the existing `docs/bug_and_risk_report_2026-10-13.md` (F-3)
and reconfirmed here by direct inspection of
`src/build_statsbomb_360_dataset.py:356-371`.

---

## 5. Goals and own goals — full analysis

### 5.1 How goals are represented (combination, not one single field)

Goals appear in **two structurally different ways** in the raw event
stream, and must both be counted, without double-counting either:

1. **A `Shot` event with `shot.outcome.name == "Goal"`.** This is the
   primary representation for a goal scored directly by a shot (open play,
   free kick, penalty, or **penalty-shootout kick** — see §5.3).
   **Raw: 1,303. Eligible: 1,279.**
2. **A paired `Own Goal Against` / `Own Goal For` event.** Verified by
   direct inspection (`data/raw/statsbomb/data/events/*.json`, e.g. match
   with `Washington Spirit` vs `San Diego Wave`): every own goal produces
   **exactly one** `Own Goal Against` event (team = the team that
   conceded; carries the offending player and their location) and **exactly
   one** `Own Goal For` event (team = the beneficiary team; no player
   field; mirrored location) in the *same* `related_events` pair, at the
   same `period`/`timestamp`. **Raw: 55 of each, eligible: 55 of each** —
   confirmed perfectly balanced (`own_goal_for_raw == own_goal_against_raw`),
   which is itself a data-quality check that passed.

**Total goals** (raw / eligible) = Shot-Goal + Own-Goal-Against =
**1,358 / 1,334.** Own-Goal-*For* is used only to attribute scoring credit
to the correct team (§5.2); it is never added a second time to the total,
which would double-count every own goal.

A `Shot` event's own `shot.outcome.name` is never `"Own Goal"` — own goals
are represented purely by the separate `Own Goal Against`/`Own Goal For`
event pair, confirmed by inspecting all 1,303 raw Shot-Goal events (none
carry an own-goal flag) and all 55 Own-Goal-Against events (none have a
`type.name == "Shot"`). **The two representations are mutually exclusive
by construction** — no double-counting risk between them.

### 5.2 Cross-check against match metadata (home_score / away_score)

For each of the 425 raw-population matches with usable score metadata
(`data/raw/statsbomb/data/matches/<competition>/<season>.json`), this audit
computed each team's goals as (Shot-Goal events credited to that team) +
(Own-Goal-For events credited to that team) and compared against the
recorded `home_score`/`away_score`:

| | Count |
|---|---:|
| Matches checked | 425 |
| Matches matching exactly | **406** |
| Matches mismatching | **19** |

### 5.3 Explaining all 19 mismatches: penalty shootouts

All 19 mismatches were individually re-verified: **every single one** is a
knockout match that went to a penalty shootout (`period == 5` events
present — StatsBomb's period 5 is the shootout). Example, directly
re-computed from `data/raw/statsbomb/data/events/3794691.json` (France vs.
Switzerland, EURO 2020 Round of 16):

| Period | Scoring events |
|---|---|
| 1 (minute 14) | Switzerland (Seferović) |
| 2 (minutes 56, 58, 74, 80, 89) | France x3 (Benzema x2, Pogba), Switzerland x2 (Seferović, Gavranović) |
| **5 (shootout, minutes 120-127)** | **9 further `Shot`/`Goal` events** (both teams) |

Regulation+ET score from periods 1-2 only: **France 3 – Switzerland 3**,
which is exactly the recorded `home_score`/`away_score`. The mismatch is
caused entirely by the **9 shootout kicks also being recorded as
`shot.outcome.name == "Goal"`**, which match metadata's score field does
**not** include.

**Documented limitation**: `shot.outcome.name == "Goal"` alone cannot
distinguish a regulation/extra-time goal from a penalty-shootout conversion
— a model or feature built on this field must additionally filter
`period != 5` (or check `competition_stage`) if "goal" is meant to reflect
the official match outcome. This is now a verified, explained finding, not
a guess — see the **Goals Audit Mismatches** sheet for the full list of 19.

### 5.4 Is the current pipeline/target able to use this?

- **Preprocessing retains goals.** `shot.outcome.name` and the
  `Own Goal For`/`Against` events are present in the raw JSON that
  `src/build_statsbomb_360_dataset.py` reads; nothing in the pipeline
  deletes them from the source.
- **But the current 8-class next-event label cannot distinguish a goal
  from a non-goal shot.** `MEANINGFUL_EVENT_TYPES` collapses every `Shot`
  — regardless of `shot.outcome.name` — into one label, `"Shot"`
  (`src/build_statsbomb_360_dataset.py:52-61`). The processed CSV's
  `next_event_type` column can equal `"Shot"` but never carries outcome
  information. `Own Goal For`/`Against` are not in `MEANINGFUL_EVENT_TYPES`
  at all, so they can never become a `current_event_type` or
  `next_event_type` — an own goal is **silently invisible** to the current
  next-event target (it is simply skipped over by
  `find_next_meaningful_event`, see §6).
- **Consequence**: today's 8-class target answers "was the next meaningful
  action a shot?", not "did a goal happen next?" — a materially different,
  football-irrelevant-for-outcome question. See §7, Alternative D.

---

## 6. The current 8-class target — traced through the actual source code

Verified directly in `src/build_statsbomb_360_dataset.py` (not inferred
from documentation):

```python
# src/build_statsbomb_360_dataset.py:52-61
MEANINGFUL_EVENT_TYPES = {
    "Pass",
    "Carry",
    "Dribble",
    "Shot",
    "Miscontrol",
    "Ball Recovery",
    "Clearance",
    "Dispossessed",
}
```

This is **exactly** the 8 classes named in the task (`Pass`, `Carry`,
`Ball Recovery`, `Clearance`, `Dispossessed`, `Miscontrol`, `Shot`,
`Dribble`) — confirmed character-for-character against the live code, and
independently pinned by the project's own regression test
`tests/test_build_statsbomb_360_dataset.py::test_meaningful_event_types_definition_is_exactly_as_documented`.
**No label is renamed, merged, or re-filtered afterward** — the raw
`type.name` string is written straight into `current_event_type` /
`next_event_type` (`src/build_statsbomb_360_dataset.py:388-407`).

**Why these 8 were selected**: **no documented justification exists
anywhere in this repository.** This is not this audit's own speculation —
an earlier, independent audit already reached the same conclusion and
recorded it explicitly:

> "`MEANINGFUL_EVENT_TYPES = {Pass, Carry, Dribble, Shot, Miscontrol, Ball
> Recovery, Clearance, Dispossessed}` is the implemented 'meaningful event'
> definition, excluding Pressure/Duel/Ball Receipt\*/Interception/Block/Foul
> Won etc. **No documented justification found in the repository for this
> exact set.**"
> — `docs/bug_and_risk_report_2026-10-13.md`, finding F-6a

A plausible (but unverified — never claimed as fact by either audit)
rationale is that these 8 types are exactly StatsBomb's "on-ball possession
actions with a clear single actor and clear outcome," which would
explain the exclusion of `Ball Receipt*` (automatically generated,
redundant with the preceding `Pass`) and `Pressure` (concurrent annotation
on a non-possessing player, not itself a ball-state change) — but this
reasoning is **this audit's own inference**, not a documented fact, and is
explicitly flagged as such.

**Excluded primary event types**: all 25 of the other 33 discovered types
— `Ball Receipt*`, `Pressure`, `Duel`, `Block`, `Goal Keeper`,
`Foul Committed`, `Foul Won`, `Interception`, `Dribbled Past`,
`Substitution`, `Injury Stoppage`, `Half Start`/`End`, `50/50`,
`Tactical Shift`, `Referee Ball-Drop`, `Starting XI`, `Shield`,
`Player On`/`Off`, `Bad Behaviour`, `Error`, `Offside`,
`Own Goal For`/`Against` (§2 has full counts for each).

**How excluded types are treated when constructing the next-event label**
— traced directly:

```python
# src/build_statsbomb_360_dataset.py:150-161
def find_next_meaningful_event(events, current_position, current_period):
    for candidate in events[current_position + 1:]:
        if candidate["period"] != current_period:
            return None
        if candidate["type"]["name"] in MEANINGFUL_EVENT_TYPES:
            return candidate
    return None
```

```python
# src/build_statsbomb_360_dataset.py:296-304 (current-event filter)
if current_event_id not in frames_by_event_id:   # must have 360 coverage
    continue
if current_event_type not in MEANINGFUL_EVENT_TYPES:
    continue
```

This settles every sub-question the task raised about the label
definition:

- **"Next event" means**: the next **raw** event (scanning the complete,
  unfiltered event stream — not only 360-covered events) whose
  `type.name` is in the 8-class set, within the **same period**. If the
  period ends before such an event occurs, there is no label
  (`next_event is None`, row dropped).
- **Intervening non-meaningful events are skipped, not treated as steps.**
  A `Pass` immediately followed by `Pressure`, then `Ball Receipt*`, then
  `Carry` yields the label `Pass -> Carry`, with the `Pressure` and
  `Ball Receipt*` events **completely invisible** to the dataset — neither
  logged, counted, nor representable as an intermediate state.
- **The *next* event does not need 360 coverage** — only the *current*
  event does (line 300 checks only `current_event_id`). Measured
  separately in `docs/prediction_task_feasibility_2026-10-14.md` §2.6:
  91.83% of rows have a 360-covered next event; this audit's role is only
  to confirm the code path, not to re-run that measurement.
- **Methodological consequence**: because `Ball Receipt*` (26.74% of all
  events) and `Pressure` (8.56%) are skipped, the label `Pass -> Carry`
  can represent very different real-world sequences — e.g. an uncontested
  receive-and-run vs. a receive-under-heavy-pressure-and-run — collapsed
  into the same transition label. This is a genuine information loss, not
  a bug, and is exactly why Alternative C (§7) is worth considering.

---

## 7. Alternative target definitions — comparison

| | A. Current 8-class | B. All 33 observed types | C. Broader on-ball set (documented rule) | D. Hierarchical: type, then outcome |
|---|---|---|---|---|
| **Definition** | `MEANINGFUL_EVENT_TYPES` (unchanged) | Every `type.name` found in §2, no filtering | 8 current + `Ball Receipt*`, `Duel`, `Interception`, `Block`, `50/50`, `Goal Keeper` (6 more on-ball actions; administrative/stoppage types excluded by an explicit rule: "represents a discrete, attributable on-ball or save/claim action by one player") | A's 8 classes for the primary head; a second head predicting the outcome of type-specific sub-cases (e.g. `Shot` -> `Goal`/`No Goal`) |
| **Number of classes** | 8 | 33 | 14 | 8 (+ small outcome heads, e.g. 2 for Shot) |
| **Class imbalance** | High (Pass 47.4% + Carry 41.0% = 88.4% of pipeline current-events, eligible) | Very high (`Half Start`/`Starting XI`/`Own Goal*` all <0.1%) | Moderate-high (new classes mostly 0.5-8.6% each; `Ball Receipt*` is actually comparable in size to `Carry`) | Primary head unchanged from A; Shot->Goal head is itself highly imbalanced (1,279/10,831 eligible Shot-Goal rate ≈ 11.8%, consistent with `shot.outcome.name` distribution in §3) |
| **Rare-class risk** | Low (smallest of the 8 is `Dispossessed`/`Miscontrol` at ~1.1-1.3%, still tens of thousands of rows eligible) | Severe (`Own Goal For`/`Against` at 55 raw occurrences each; several administrative types under 500) | Low-moderate (`50/50` at 1,685 eligible occurrences is the rarest addition, still usable) | Low for the primary head; the outcome head is naturally small/binary by football logic, not an artifact of class design |
| **Interpretability** | High, already the thesis baseline (`docs/prediction_task_feasibility_2026-10-14.md`) | Low — mixes football actions with stoppages/administration | High — stays within "things one player does to/with the ball" | High — separates "what kind of event" from "how did it turn out," mirroring how football analysts actually talk about shots |
| **Data quality / label reliability** | Already verified (§2.1-2.6 of the feasibility report; zero missing `current_event_type`/`next_event_type` by construction) | Administrative types (`Half Start`, `Starting XI`, ...) are reliably labeled but **meaningless as a "next ball action"** — their presence would reward a model for predicting period boundaries, not football | Same reliability as A for the 8 existing classes; the 6 additions have the same StatsBomb labeling process, no new reliability concerns found | Reuses A's reliability; the Shot outcome head additionally needs the period-5/shootout caveat from §5.3 if "Goal" is meant to reflect the official scoreline |
| **Thesis suitability** | **Already the right scope for a Bachelor's thesis** — manageable, well-documented, directly supports the 360-vs-no-360 comparison already planned | Not suitable as the primary target; several classes have single-digit raw counts across the *entire* eligible dataset | Suitable as a secondary/extended experiment if time permits — meaningfully closes the `Ball Receipt*`/`Pressure` gap identified in §2 without the sparsity risk of B | Suitable as a complementary analysis alongside A, directly resolving the "cannot distinguish Goal" limitation from §5.4 without redesigning the primary task |

### Recommendation

- **Keep Candidate A (the current 8-class target) as the thesis's primary
  task.** It is already well-understood, already has a measured baseline
  (the transition-frequency baseline in
  `docs/prediction_task_feasibility_2026-10-14.md` §2.3), and this audit
  found no data-quality defect that would invalidate it.
- **Do not switch to Candidate B** (all 33 types) — the administrative/
  stoppage types (`Half Start`, `Starting XI`, `Referee Ball-Drop`, ...)
  are not football actions and several have raw counts in the hundreds
  across the entire 425-match corpus; including them would mostly reward a
  model for recognizing period boundaries and substitution bookkeeping.
- **Recommend Candidate C as a documented secondary/extended experiment**,
  not a replacement: adding `Ball Receipt*` in particular directly
  addresses this audit's single largest finding (§2) — a near-`Carry`-sized
  event type currently invisible to the target — at a bounded cost (6 new
  classes, all with thousands of eligible occurrences, see §2).
- **Recommend Candidate D as a complementary analysis**, run alongside A,
  specifically to give the thesis a documented answer to "can this
  pipeline tell a goal from a near-miss?" (today: no — see §5.4), without
  touching the primary 8-class scope.
- This is a methodological judgment informed by, but going beyond, the raw
  frequency data above; the frequency counts themselves (class sizes,
  rarity, matches-with-coverage) are what this audit measured directly and
  are reproducible from the workbook. The suitability/recommendation
  columns are this report's own reasoning, clearly separated here from the
  measured counts.

---

## 8. Data quality findings

1. **1 malformed file**: `data/raw/statsbomb/data/three-sixty/3845506.json`
   — already a documented exclusion (§0); this audit's own JSON parser
   independently reproduces the same `JSONDecodeError` reported in
   `docs/data_quality_investigation_2026-10-13.md`.
2. **7 enum id/name consistency violations** — real player/team identity
   spelling inconsistencies in the raw data, detailed in §3.4. Zero
   containers had an id mapping to the wrong *number* of names beyond
   this; zero names mapped to multiple ids (the inconsistency is strictly
   one-directional: a few ids have two spellings, no spelling is shared by
   two different ids).
3. **19 matches with a goal-count/score-metadata mismatch**, fully
   explained as penalty shootouts (§5.3) — not a defect, but an important
   documented caveat for any score-reconciliation feature.
4. **`Own Goal For`/`Against` pair counts are perfectly balanced** (55/55
   raw) — a data-integrity check that *passed*, included here as positive
   evidence, not only negative findings.
5. **4,235 − 426 = 3,809 local event files have no 360 data at all** and
   are out of scope for this audit by design (§0) — flagged explicitly so
   the scope boundary is never mistaken for an oversight.
6. **Illustrative field-list correction**: `position.name` is a top-level
   sibling of `player`, not `player.position.name` (§3.2) — confirmed from
   live data, not assumed from the task's own field list.
7. **Two "missing = implicit" conventions** (boolean flags omitted when
   false; `pass.outcome` omitted when complete) must not be treated as
   genuine missing-data problems — documented in §3.1 specifically so
   future feature engineering does not misinterpret them as data loss.
8. **`shot.freeze_frame` is a distinct, pre-existing per-shot nearby-player
   array embedded in the event JSON itself**, structurally different from
   the `three-sixty.json` 360 product, present on 97.3% of raw `Shot`
   events (§3.3) — flagged so it is never conflated with 360 coverage when
   reasoning about which rows have "spatial data."
9. **0 duplicate `event.id` or `event_uuid` collisions** were introduced by
   this audit's own aggregation (verified structurally: every `id`/`uuid`-
   named leaf is tracked as an `identifier` field, and the per-match,
   per-file loading never merges two different matches' id spaces).

---

## 9. Recommended next steps for the thesis

1. Adopt Candidate A as the primary target (no change needed to
   `src/build_statsbomb_360_dataset.py`), but **document the exclusion of
   `Ball Receipt*` and `Pressure` explicitly in the thesis methodology
   section** — this audit found no prior documentation of that choice
   anywhere in the repository (§6), and the thesis should not repeat that
   gap.
2. If time permits, implement Candidate C (§7) as a secondary experiment
   specifically to quantify how much the `Ball Receipt*`/`Pressure`
   omission affects next-event predictability — this audit provides the
   exact class sizes needed to scope that experiment (§2).
3. If the thesis discusses shots at all, add the Candidate D outcome head
   (Shot -> Goal/No Goal) and explicitly filter `period != 5` (or
   equivalent) wherever "goal" is meant to reflect the official scoreline,
   per the shootout caveat in §5.3.
4. Treat `docs/bug_and_risk_report_2026-10-13.md` finding F-6a (no
   documented rationale for the 8-class set) as still open; this audit
   independently reconfirms it rather than resolving it.
5. When engineering any player/team-identity feature, join on numeric
   `id`, never on `name` (§3.4) — now backed by 7 concrete counterexamples
   in this exact dataset.

---

## 10. Reproducing this audit

```bash
# Full audit (all 426 raw-population matches, ~4 minutes):
.venv/bin/python scripts/audit_statsbomb_category_taxonomy.py

# Fast iteration on a random subset:
.venv/bin/python scripts/audit_statsbomb_category_taxonomy.py --sample 40 --seed 0
```

Outputs (both overwritten deterministically on each run, given unchanged
underlying data — verified by running twice and diffing the JSON output
with the timestamp field excluded):

- `results/data_taxonomy/statsbomb_complete_category_audit_data.json` —
  full machine-readable results (also what this report's numbers are read
  from).
- `results/data_taxonomy/statsbomb_complete_category_audit.xlsx` — the
  13-sheet workbook (10 required sheets + 3 documented splits: a
  high-cardinality overflow for Category Values, and a dedicated Goals
  Audit Mismatches sheet).

Run the existing test suite to confirm this audit made no changes to
production code:

```bash
.venv/bin/python -m pytest tests/ -q   # 138 passed, unchanged by this audit
```
