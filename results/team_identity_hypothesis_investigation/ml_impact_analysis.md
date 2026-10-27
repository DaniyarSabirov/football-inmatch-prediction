# ML impact of the team-identity findings

Investigation only. The production dataset in `data/processed/` was not rebuilt or changed (file dates are
unchanged). Rows were rebuilt **in memory** with the production builder for all 417 eligible matches,
using `build_rows_from_match_data` with its CLI default of a 10 s horizon
(`diagnostics/ml_impact.py`; per-row flags in `ml_rows_impact.csv.gz`, totals in
`ml_impact_summary.json`).

## 1. Which features depend on team identity

`src/build_statsbomb_360_dataset.py::build_rows_from_match_data`:

| Feature | Depends on | Identity rule |
|---|---|---|
| `visible_teammates`, `visible_opponents`, `visible_ambiguous` | every freeze-frame point's team | `resolve_absolute_team_for_point` (Problem C) |
| `visible_players` | number of points | none |
| `actor_x`, `actor_y` (`resolve_actor_location`) | the `actor` point, after Problem A and team normalization | none (orientation only) |
| `same_team`, `same_possession`, next-event fields | event data | none (no 360 identity) |

The builder computes no per-team spatial distributions or distances. A team-identity error therefore
changes only the teammate/opponent counts of the affected row:
- a whole-record perspective error **swaps** `visible_teammates` and `visible_opponents`;
- a single-point error (for example the keeper quirk below) moves one point between them.

## 2. Visualization vs ML: are the rules the same?

| Aspect | Renderer | ML builder | Rows where they differ |
|---|---|---|---|
| Problem C (byte-identical shared snapshots) | yes | yes | — |
| Display-only shared-source continuity refinement (2 groups, 3857293) | yes | no | **0 rows**: Ball Receipt* and Block are not `MEANINGFUL_EVENT_TYPES` |
| Display-only UNCERTAIN orientation resolution (duplicates, kick-off rule, camera+continuity, user reviews) | yes | no | **535 rows** on Problem A UNCERTAIN frames: `actor_x/actor_y` may be mirrored in ML. Team counts are unaffected (orientation does not change identity). |
| Team normalization | per-period display plan | one fixed reference team (analytical frame) | by design, documented earlier |

## 3. Rows potentially affected

| Exposure | Rows | Share |
|---|---|---|
| All rows | 756,023 | — |
| Rows with keeper evidence | 261,551 | 34.6% |
| Keeper-verified wrong colour assignment (current rule; the keeper is put in the wrong team) | **1,187** | 0.45% of keeper-evidenced rows |
| Rows on byte-identical shared snapshots (Problem C reading used; validated 98% by keepers) | 35,447 | 4.7% |
| Rows on frames in the structural or raw error classes of `full_dataset_audit.csv` (copy missed, derived record, shared wrong view, record-level inversion, raw scramble) | 264 | 0.035% |
| Rows on frames flagged by the continuity detector (any) / strong | 1,170 / 43 | most are detector artefacts (see README §6) |
| Rows whose `actor_x/actor_y` come from an UNCERTAIN orientation | 535 | 0.07% |

Median |teammates − opponents| in the affected rows is 1. A full swap would therefore usually move about
one count between the two features.

Caveat on the 1,187 keeper-verified rows: StatsBomb's own 360 spec says the `keeper` attribute "will, in
some rare cases, refer to the keeper on the same team as the 'actor', without being marked as a
'teammate'". In 957 of the 2,272 keeper-verified display errors, the rest of the record agrees with its
neighbours. Those are keeper-only flag errors, which shift exactly one point between teammates and
opponents.

## 4. Assessment for the thesis

- The identity errors that are demonstrably wrong affect well under 1% of rows, and mostly by a single
  count. They are unlikely to change model conclusions, but they are not zero. Report them as a known
  data-quality limitation with these numbers.
- The more important consistency point is §2: ML and display use different orientation resolution for
  535 rows. That affects actor coordinates, not team identity.
- A unified correction would live in **one shared module** used by both consumers, with the same
  evidence-directed rule (README §8). It should be applied to the ML builder only after you approve it,
  with a before/after diff of the affected rows.

## 5. Not done (by instruction)

The ML dataset was not modified, the builder was not changed, and `src/team_identity.py` was not changed.
