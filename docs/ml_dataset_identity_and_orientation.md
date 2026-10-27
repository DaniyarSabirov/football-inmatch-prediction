# Next-event dataset: causal features, identity and orientation

Status: 2026-10-27 (updated after the ML data integrity audit). Built by `scripts/build_full_dataset.py` (all 417
usable matches) or `src/build_statsbomb_360_dataset.py --match-id N` into `data/processed/` (gitignored).
Validation: `results/ml_data_integrity_final_2026-10-27/README.md`.

## Prediction setting

A row describes the **current event** and its own 360 snapshot; the label is the **next meaningful event** in the
same period, by event order, however many seconds later it happens.
Model inputs may use only:
- the current event;
- its snapshot;
- records that come **earlier in the event stream** (period, index).

They may never use later events, later snapshots or labels. Event order, not timestamps, decides what is
earlier: 33 adjacent event pairs dataset-wide have a later index but an earlier timestamp.

## Selecting model inputs: use the allowlist

`configs/ml_feature_classification.json` classifies all 42 columns (CAUSAL_FEATURE / TARGET / OFFLINE_DIAGNOSTIC
/ METADATA / EXCLUDED) and marks the approved inputs explicitly. Training code must use
`src.ml_features.select_feature_matrix(df)`:
- it checks that the dataset schema matches the classification;
- it returns only the approved columns;
- `validate_feature_columns` rejects anything else, including any new, unclassified column.

Approved causal features (11): `current_event_type`, `period`, `minute`, `actor_x`, `actor_y`,
`actor_ambiguous`, `orientation_available`, `visible_players`, `visible_teammates`, `visible_opponents`,
`visible_ambiguous`. Target: `next_event_type`.

## Causal vs offline columns

| Quantity | Causal (feature columns) | Offline (diagnostics only) |
|---|---|---|
| Snapshot orientation | `orientation`, `orientation_resolved_by`, `orientation_available` | `orientation_offline`, `orientation_offline_resolved_by`, `orientation_problem_a_offline`, `orientation_uses_later_records` |
| Actor position | `actor_x`, `actor_y` (empty when `orientation_available` is false) | `actor_x_offline`, `actor_y_offline` |
| Team identity | `visible_teammates`, `visible_opponents`, `visible_ambiguous`, `identity_quality`, `identity_corrected_points` | `*_offline` |
| Raw flags | `visible_teammates_raw`, `visible_opponents_raw` (no correction) | — |

### Causal orientation

Implemented by `src/orientation_resolution.resolve_causal_orientations`, using the snapshot itself and earlier
records only:

1. Problem A geometry on the record alone.
2. Problem A's neighbour vote with **earlier** anchors only.
3. The kick-off own-half rule (record only, no camera veto).
4. An **earlier** copy of the same raw snapshot: two copies' own frames are mirror images exactly when their
   teams differ.

Otherwise the orientation is UNCERTAIN and the actor position is **unavailable**. It is neither guessed nor
taken from a look-ahead resolution.

### Offline orientation

This is the renderer's resolution, for retrospective visualization. It may use:
- copies in any order;
- a camera side estimated from the whole period;
- reviewer decisions made while watching sequences;
- Problem A neighbours on both sides.

A match- or period-level convention inferred from the whole match is acceptable for visualization. It is
**not** a predictive feature unless a prediction setting explicitly allows whole-match information.

### Causal identity

Computed by `src/team_identity.resolve_match_team_identity(..., causal=True)` with the causal orientations.
Every rule uses only earlier records; the two-neighbour detector needs the next snapshot and is therefore not
run.

**Price of strict causality:** a snapshot copied from a *later* event cannot be recognised as a copy at its
own time, so it keeps its raw reading.

## Guarantees and checks

- **Truncation invariance:** cutting the input right after any event, and keeping only that event's label,
  leaves every non-target, non-offline column of that row and of all earlier rows unchanged. This is checked
  in `tests/test_ml_feature_safety.py` and, on 155 real cuts (3.66M cells), in
  `results/ml_data_integrity_final_2026-10-27/truncation_check.json`.
- **Count invariants:** `visible_teammates + visible_opponents + visible_ambiguous == visible_players`, for both
  causal and offline columns.
- **Row selection:** every meaningful current event with a 360 record that has a meaningful successor in the
  same period (event order). There is no time window: the 10-second cap was removed on 2026-10-27 because it
  excluded 3.97% of rows selectively (64% of current Shot rows, all penalty-shootout rows); see
  `results/next_event_time_window_audit_2026-10-27/README.md`. The last meaningful event of each period has no
  target and is not a row.
