# Next-event dataset: team identity and orientation columns

Status: 2026-10-27. Built by `scripts/build_full_dataset.py` (all 417 usable matches) or
`src/build_statsbomb_360_dataset.py --match-id N` (one match) into `data/processed/` (gitignored). Validation:
`results/team_identity_final_validation_2026-10-27/README.md`.

## Which columns to use as model features

| Column | Meaning | Use as a feature? |
|---|---|---|
| `visible_teammates`, `visible_opponents` | Players of the acting team / the other team in the current snapshot, by the **causal** team-identity resolution: decided only from this snapshot and earlier records, never from the events that follow (which include the event being predicted). Players whose team is not established are counted on neither side. | **yes** |
| `visible_ambiguous` | Players whose team is not established (UNRESOLVED). A data-quality count, not a third team. | yes, as a quality flag, or for masking |
| `identity_quality` | `uncertain` (≥ 1 unresolved player), `corrected` (≥ 1 player read differently from the raw flag, causal), `clean` | quality flag / masking / sensitivity analysis |
| `identity_corrected_points` | Players read differently from their raw `teammate` flag (causal) | analysis |
| `visible_teammates_raw`, `visible_opponents_raw` | The untouched StatsBomb `teammate` flags, read relative to the event's own team | analysis / baseline comparisons |
| `visible_teammates_offline`, `visible_opponents_offline`, `visible_ambiguous_offline`, `identity_quality_offline` | **Offline** resolution: evidence from both directions in time, up to 2 s after the event. This is what the renderer draws. | **no**: it uses records after the event |
| `actor_x`, `actor_y` | Actor position after the shared orientation resolution (`src/orientation_resolution.py`) and team normalization (one fixed reference team per match) | yes |
| `orientation`, `orientation_problem_a`, `orientation_resolved_by` | Final snapshot correction, the geometry classifier's result, and the route that decided it | analysis |
| `orientation_uses_later_records` | True when the orientation route could read records after the event (55 rows of 756,023) | exclude these rows in a strict sensitivity check |

Invariants (checked on the rebuilt dataset):

- `visible_teammates + visible_opponents + visible_ambiguous == visible_players`, for both causal and offline;
- `visible_teammates_raw + visible_opponents_raw == visible_players`.

## Why causal features

- **The risk:** 72.9% of rows have the snapshot of the event to be predicted within 2 s. The offline
  resolution may use exactly that snapshot.
- **How much it changes:** offline and causal features differ in 3,801 rows (0.50%).
- **Measured effect:** in a match-grouped 5-fold probe, the difference was −0.0003 ± 0.0006 macro-F1, below
  fold noise. The causal default therefore costs nothing measurable and removes the methodological objection.

**Price of strict causality:** a snapshot copied from a *later* event cannot be recognised as a copy when its
own event happens. It keeps its raw reading in the causal columns. This is mostly Ball Receipt* records,
which are rarely dataset rows.

## Rules in brief

Team identity (`src/team_identity.py`):

- spec reading by default;
- structural copies/derived records resolved by evidence (continuity > keeper + owner > legacy source);
- crowding-safe two-neighbour detector (offline only);
- narrow goalkeeper continuity rule;
- simultaneous contradictions (≤ 250 ms, margin ≥ 2 yd) unresolved.

See the module's section comment for details.

Orientation: the renderer's validated resolution of UNCERTAIN snapshots (duplicates, kick-off rule, camera
side, camera + continuity, reviewer decisions), applied once, before team normalization.
