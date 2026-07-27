# Football In-Match Event Prediction

Research code for a bachelor's thesis on machine learning methods for predicting
events during football matches.

The repository currently provides the reproducible project foundation only.
It intentionally contains no datasets, feature pipelines, or machine-learning
models yet.

## Research scope

The project is intended to support a transparent workflow for:

1. acquiring and documenting football event data;
2. validating and transforming raw observations;
3. constructing temporally valid in-match features;
4. training and evaluating predictive models;
5. reporting metrics, tables, and figures used in the thesis.

The precise prediction target, observation window, datasets, and evaluation
protocol should be documented before modeling begins.

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

## Testing

```bash
pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
