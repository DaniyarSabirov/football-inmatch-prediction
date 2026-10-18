# Bug and Risk Report — StatsBomb 360 Pipeline

Date: 2026-10-13. Companion to `docs/data_integrity_audit_2026-10-13.md`
(full evidence there); this document is the severity-ranked punch list.

## Confirmed defects

| ID | Severity | Status | Summary |
|---|---|---|---|
| F-3 | **MODERATE** | **FIXED this audit** | `build_statsbomb_360_dataset.py`'s `visible_teammates`/`visible_opponents` used the naive per-event `teammate` reading, silently wrong for 1.48% of `MEANINGFUL_EVENT_TYPES` rows (11,688/788,088) whose 360 snapshot is shared with a different team's event. A further 9 rows were genuinely ambiguous. Fixed by wiring the existing Problem C resolution (`src/team_identity.py`) into the dataset builder; added a `visible_ambiguous` column. Verified on real match 3902240: exactly 1/2,013 rows changed, split flipped from (8,9) to (9,8), 0 unexpected side effects. |
| F-1 | HIGH | Fixed (prior session) | Shared-snapshot source resolution treated multiple *agreeing* confident same-team sources as ambiguous (grey players in the visualization). Fixed in `src/team_identity.py`. |
| F-2 | HIGH | Fixed (prior session) | Shared-snapshot actor point was forced to the currently-displayed event's own team, flipping the same physical point's color depending on which event was on screen. Fixed in `src/team_identity.py`. |
| F-4 | LOW | Documented only | Borrower-row `actor_x`/`actor_y` is a real, correctly-transformed coordinate, but may physically belong to the snapshot's true source event rather than the row's own team. Not a coordinate error; a labeling-precision nuance. |
| F-5 | LOW | Documented only | 28/1,381,467 frames (0.002%, 11/426 matches) carry two `actor:true` points in the same freeze_frame — a rare raw StatsBomb data quirk. The builder's `next()`-based selection is a defensible, low-impact degradation (duplicates are at identical/near-identical coordinates). |

## Methodological risks (not code bugs — require a decision)

| ID | Summary |
|---|---|
| F-6a | `MEANINGFUL_EVENT_TYPES = {Pass, Carry, Dribble, Shot, Miscontrol, Ball Recovery, Clearance, Dispossessed}` is the implemented "meaningful event" definition, excluding Pressure/Duel/Ball Receipt*/Interception/Block/Foul Won etc. No documented justification found in the repository for this exact set. |
| F-6b | The CSV bundles input-eligible current-event fields and next-event **label** fields (`next_event_*`, `same_team`, `same_possession`, `time_to_next_event_seconds`) into one row with no feature/label type annotation — a future consumer could accidentally leak the label into model inputs. Not a bug in the generator; a usability/documentation gap. |
| F-6c | No canonical "attacking direction" feature currently exists in the CSV output (`orientation`/`team_normalization` are diagnostic columns, not a reusable canonical-direction feature). If the thesis needs one, it must be designed. |

## Explicitly ruled out (investigated, not a bug)

- **Half-time orientation handling**: StatsBomb's raw per-team coordinate
  convention is already period-independent (re-verified fresh in this
  audit across match 3857283, a 40-match shot-direction sample — 69/69
  team-period pairs consistent, zero exceptions — and match 3902240's
  extra-time periods 1–4). No code-level "flip at half-time" logic is
  needed or missing.
- **Event-to-frame matching**: a plain, exact `event_uuid` dict join; no
  nearest-timestamp fallback exists anywhere. 0 duplicate ids, 0 duplicate
  `event_uuid`s, 0 index-ordering violations across the full dataset.
- **`event.location` orientation correction** (investigated in a prior
  session): mathematically and empirically disproven across 1,357,624
  events — applying orientation correction to `event.location` would make
  things strictly worse for 5.73% of events with zero benefit anywhere.
  Current behavior (no correction applied to `event.location`) is correct.
- **"Disappearing" visible-area polygon** (reported during development):
  was a `--view` CLI flag selection matter, not a code regression; the
  default `analysis` view has always intentionally omitted the polygon.
- **"Ball / event location" marker separation from the actor** (reported
  during development): confirmed to be the correct, documented
  `event.location` field — not a missing or broken ball-tracking feature.

## Disclosed process incident

Running the dataset-builder CLI directly during verification overwrote
`data/processed/statsbomb_360_match_3902240_next_events.csv` in place
(no `--output` flag exists on that script). The file is untracked/
gitignored derived output, not raw data; no other processed file was
touched; a valid before/after comparison was reconstructed via a temporary
code revert. Full disclosure in §12 of the main audit report.

## Test suite impact

73 automated tests, all passing (51 pre-existing + 22 new in
`tests/test_build_statsbomb_360_dataset.py`, including a dedicated
regression test for F-3). No tests were weakened or skipped to make this
pass.

## Outstanding gap

`src/animate_statsbomb_360_match.py` has no dedicated automated test file;
its correctness has been established via manual/visual verification only
across this project's history. Recommended follow-up, not completed in
this audit.
