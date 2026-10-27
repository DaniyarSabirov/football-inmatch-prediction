# Football In-Match Event Prediction

Research code for a bachelor's thesis on machine learning methods for predicting
events during football matches.

The repository contains the data pipeline for StatsBomb event and 360 data
(validation, orientation and team-identity correction, a next-event dataset
with causal features). No prediction model has been trained yet.

## Research scope

The project is intended to support a transparent workflow for:

1. acquiring and documenting football event data;
2. validating and transforming raw observations;
3. constructing temporally valid in-match features;
4. training and evaluating predictive models;
5. reporting metrics, tables, and figures used in the thesis.

## Prediction task and dataset

**Task:** given the information available at the current event, predict the
type of the next event in the same period of the match.

- **Rows:** every event of one of 8 types (`MEANINGFUL_EVENT_TYPES`: Pass,
  Carry, Dribble, Shot, Miscontrol, Ball Recovery, Clearance, Dispossessed)
  that has a StatsBomb 360 record, in the 417 usable matches.
- **Target (`next_event_type`):** the type of the first LATER event in the raw
  event stream (event order, i.e. `index`) whose type is one of the 8 types
  and that lies in the same period. The next event does not need 360 data.
  There is **no time window**: the next event counts however many seconds
  later it happens (the earlier 10-second cap was removed on 2026-10-27, see
  `results/next_event_time_window_audit_2026-10-27/README.md`). Events with
  no such successor in their period have no target and are not rows.
- **Features:** only the 11 approved causal columns in
  `configs/ml_feature_classification.json`, selected with
  `src.ml_features.select_feature_matrix`. They use the current event, its
  own 360 snapshot and earlier records only. The gap to the next event
  (`time_to_next_event_seconds`) is future information and never a feature.
- **Build:** `python scripts/build_full_dataset.py` writes per-match CSVs, a
  combined CSV and a manifest to `data/processed/` (not version-controlled).

Details: `docs/ml_dataset_identity_and_orientation.md`.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` only if local environment variables become
necessary. Never commit credentials.

## Project layout

```text
.
├── configs/                 Experiment and pipeline configuration
├── data/
│   ├── raw/                 Immutable source data
│   ├── interim/             Intermediate transformations
│   ├── processed/           Analysis-ready datasets
│   └── external/            Third-party reference data
├── docs/                    Research and technical documentation
├── models/                  Serialized model artifacts
├── notebooks/               Numbered exploratory and reporting notebooks
├── results/
│   ├── figures/             Generated plots
│   ├── tables/              Generated tables
│   └── metrics/             Machine-readable evaluation results
├── src/
│   ├── data/                Data ingestion and validation code
│   ├── features/            Feature engineering code
│   ├── models/              Training and inference code
│   ├── visualization/       Reusable plotting code
│   └── utils/               Shared project utilities
└── tests/                   Automated tests
```

## Reproducibility conventions

- Treat files in `data/raw/` as immutable.
- Keep generated data, model artifacts, and results out of version control.
- Put reusable logic in `src/`; notebooks should orchestrate and explain it.
- Name notebooks in execution order, for example `01_data_exploration.ipynb`.
- Record random seeds, data versions, configuration, and evaluation results for
  every experiment.
- Use paths from `src.utils.paths` instead of working-directory-dependent paths.

## StatsBomb 360 event visualization

`src/animate_statsbomb_360_match.py` visualizes the event-based StatsBomb 360
freeze frames available in the local open-data checkout. Every logical animation
frame corresponds to exactly one real `three-sixty` record joined to its real
event by UUID. Time arguments only filter those existing joined events: the tool
does not create one frame per second, choose nearest events for artificial times,
duplicate observations, or interpolate player positions.

List matches that have local 360 data, optionally filtering metadata:

```bash
python src/animate_statsbomb_360_match.py \
  --list-available-matches \
  --team-filter "Inter"
```

The listing includes match IDs, date, competition, season, teams, score, total
360 records, and records with non-empty freeze frames. Metadata is optional;
matches remain renderable by ID when their event and 360 JSON files exist.

After choosing a match ID, render every usable real event between 20:00 and
30:00 of period 1:

```bash
./scripts/run_360_preview.sh \
  --match-id MATCH_ID \
  --period 1 \
  --start 20:00 \
  --end 30:00 \
  --theme dark
```

Render all usable events in one period:

```bash
./scripts/run_360_preview.sh \
  --match-id MATCH_ID \
  --period 1 \
  --all-period-events \
  --theme dark
```

Render all usable events across every period:

```bash
./scripts/run_360_preview.sh \
  --match-id MATCH_ID \
  --all-events \
  --theme dark
```

Use `--maximum-render-frames N` to stop before a selection larger than an
explicit safety limit. A value of `0` means no limit; the tool never silently
samples or truncates events.

Frame selection and display timing are independent. In `fixed` timing mode,
every selected real event is shown for the same `--frame-duration-ms`. In
`real-time` mode, each selected event is held according to its real gap to the
next selected event, subject to the configured minimum and maximum duration.
Neither timing mode creates additional logical frames.

The wrapper generates a GIF, a static test-frame PNG, and an event-frame
manifest containing exactly one row per rendered real event. For bounded or
whole-period selections it also generates a temporal-density plot. The Python
CLI additionally supports the backward-compatible timing CSV export with
`--timing-csv` and direct manifest export with `--event-frame-manifest`.

## Testing

```bash
pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
