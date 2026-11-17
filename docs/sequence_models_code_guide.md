# Sequence models: code guide

A walk-through of the code behind the Markov chain, HMM and cost-sensitive experiments, written for
presenting the thesis code. Line numbers refer to the files as of 2026-10-28; if they drift, search for the
function name.

## 1. The big picture

```text
raw StatsBomb JSON ──► event table ──► task table (sequences + targets) ──► model ──► probabilities ──► decision
data/raw/statsbomb    raw_events.py     tasks.py                          markov / backoff_markov /    cost_sensitive.py
                                                                          hmm / categorical_hmm
```

1. **Read the raw events** of a match into one flat table: one row per raw event, in StatsBomb order.
2. **Turn that table into a prediction task.**
   - Keep the event types of a taxonomy (e.g. the 8 production types or the 15 Scenario D types).
   - Cut the stream into sequences, one per (match, period).
   - The **target** of each event is the next kept event in the same period.
3. **Fit a model on training matches.** It outputs, for every event, a probability for every possible next event.
4. **Turn probabilities into a predicted class**: either the most likely class, or the class with the lowest *expected cost* (cost-sensitive decision).
5. **Evaluate on validation matches** that the model never saw, grouped by match.

## 2. Where each piece lives

### 2.1 Data and the prediction task

| What | File | Function / class | Line |
|---|---|---|---|
| Read one match's raw events (+ whether each has a 360 record) | `src/sequence_models/raw_events.py` | `load_match_event_records` | 110 |
| Flatten events into table rows | same | `event_records` | 78 |
| Event table for many matches (parallel, cached) | same | `build_event_table`, `load_or_build_event_table` | 128, 139 |
| Taxonomy / task definition (vocabulary, excluded periods) | `src/sequence_models/config.py` | `TaskSpec` | 22 |
| **Sequences and targets** | `src/sequence_models/tasks.py` | `build_task_table` | 72 |
| Check that the raw order is valid (index = file order, no duplicates) | same | `validate_event_order` | 48 |
| Row positions of each sequence | same | `sequences` | 119 |
| Pitch third of an event (0–40, 40–80, 80–120) | `src/sequence_models/zones.py` | `zone_of_x` | 43 |

**How `build_task_table` builds the training targets** (`tasks.py` lines 72–116):

- It drops period 5 (the penalty shootout) and keeps only the taxonomy's event types.
- Inside each (match, period) it shifts the event column by one: `grouped["symbol"].shift(-1)`. So the target of event *t* is event *t+1* of the same period, and a sequence can never cross halftime or a match boundary.
- `n_skipped_raw_events` counts how many raw StatsBomb records lie between an event and its target.
- `eval_eligible` marks the rows that are evaluated: the current event has its own 360 record and a target exists.

### 2.2 Models

| Model | File | Class | Line | Key methods |
|---|---|---|---|---|
| Frequency baseline (predicts class frequencies) | `src/sequence_models/markov.py` | `FrequencyBaseline` | 54 | `fit` 63, `predict_proba` 70 |
| 1st-order Markov (experiment 1) | same | `MarkovChain` | 85 | `fit` 95, `predict_proba` 108 |
| 2nd-order Markov (experiment 1, hard backoff) | same | `SecondOrderMarkovChain` | 131 | `fit` 144, `predict_proba` 154 |
| **Markov models with contexts and smooth backoff** (`mc1`, `mc2`, `mc1_zone`, `mc2_zone`) | `src/sequence_models/backoff_markov.py` | `BackoffMarkov` | 66 | `fit` 95, `predict_proba` 108 |
| Context features (previous event, current event, zone) | same | `build_contexts` | 48 | |
| Model definitions (which contexts each model uses) | same | `MODEL_LEVELS` | 36 | |
| HMM, categorical × Gaussian (experiment 1) | `src/sequence_models/hmm.py` | `HiddenMarkovModel` | 95 | `fit` 192, `filter` 250, `predict_proba` 257 |
| HMM with event / zone emissions (experiment 2) | `src/sequence_models/categorical_hmm.py` | `CategoricalHMM` | 83 | `fit` 212, `filter` 289, `predict_proba` 296 |
| Cost-sensitive decisions | `src/sequence_models/cost_sensitive.py` | `cost_matrix`, `bayes_decisions` | 73, 96 | |

### 2.3 How transition probabilities are estimated and smoothed

**1st-order Markov (`markov.py`, `transition_counts` line 33, `normalise_rows` line 43).**
- Count how often event *i* is followed by event *j* inside the same sequence: $C_{ij}$.
- With Laplace smoothing $\alpha = 1$: $P(j \mid i) = (C_{ij}+1) / (\sum_k C_{ik} + V)$.
- No transition ever gets probability 0.

**Backoff Markov (`backoff_markov.py`).** This is the experiment-2 family; `mc2_zone` is the best model so far.
- A model is a *chain of contexts*, from most to least specific. For `mc2_zone`:
  $(\text{previous}, \text{current}, \text{zone}) \rightarrow (\text{current}, \text{zone}) \rightarrow (\text{current}) \rightarrow \text{overall frequency}$.
- `predict_proba` (line 108) starts with the overall frequency, then refines it level by level:
  $$p_\ell(k\mid c) = \frac{C_\ell(c,k) + \beta\,p_{\ell-1}(k)}{N_\ell(c) + \beta}$$
- A context seen thousands of times is dominated by its own counts. A rare or unseen context stays close to the shorter context above it.
- $\beta$ (`concentration`) is chosen per fold by inner cross-validation: `run_markov` in `experiment_v2.py`, line 173.
- `scripts/demo_explain_prediction.py` prints exactly these steps for one real event (§6).

**HMM (`categorical_hmm.py`).**
- Training is Baum–Welch / EM: `fit` (line 212) alternates `_forward` (line 168) and `_backward` (line 188) with the M-step `_m_step_emissions` (line 260).
- Prediction is **causal**: `filter` returns $p(\text{state}_t \mid \text{events}_{1..t})$. `predict_proba` multiplies by the transition matrix and the emission matrix to get $p(\text{event}_{t+1} \mid \text{events}_{1..t})$.
- Future events are never used.

### 2.4 Training, validation and prediction

| Step | Where |
|---|---|
| Final test hold-out, development pool, pilot (experiment 1) | `src/sequence_models/splits.py` `final_test_split` 65, `pilot_subset` 78; saved in `results/markov_hmm_pilot/split_manifest.json` |
| Match-grouped folds (StratifiedGroupKFold) | `splits.py` `grouped_folds` 95, `verify_folds` 110 |
| Refusing final-test matches | `splits.py` `assert_no_final_test` 124; `experiment_v2.py` `load_development_events` 96 |
| Tier A / Tier B partition (experiment 2) | `experiment_v2.py` `load_partition` 82 |
| Fit each Markov model on the training folds, predict the validation fold | `experiment_v2.py` `run_markov` 173 (uses `_markov_fit_predict` 159) |
| Fit the HMMs in parallel, with a resumable cache | `experiment_v2.py` `run_hmm` 263, `_hmm_job` 232 |
| Cost-sensitive decisions per fold | `experiment_v2.py` `fold_priors` 370, `decide` 376 |
| All metrics, bootstrap intervals, rare-event analysis | `experiment_v2.py` `evaluate_task` 423; `evaluation_v2.py` (`metrics_from_confusion` 37, `calibration` 87, `bootstrap_summary` 144) |
| Orchestration (stages) | `scripts/run_markov_hmm_cost_sensitive_v2.py` (`stage_tier` 106, `stage_repro` 240) |

### 2.5 Cost-sensitive decisions in two sentences

- `cost_matrix` (`cost_sensitive.py` line 73) gives missing a rare class $k$ the cost $m_k = \min(\text{cap}, (\pi_{\max}/\pi_k)^\gamma)$, where $\pi$ are the class shares in the *training* folds.
- `bayes_decisions` (line 96) then picks the class with the lowest expected cost $\sum_k C(a,k)\,p_k$. The probabilities themselves are never changed.

## 3. Where results are stored

| Content | Experiment 1 | Experiment 2 |
|---|---|---|
| Configuration | `configs/markov_hmm_pilot.yaml` | `configs/markov_hmm_cost_sensitive_v2.yaml` |
| Split / fold manifest | `results/markov_hmm_pilot/split_manifest.json` | `results/markov_hmm_cost_sensitive_v2/split_manifest.json` |
| Fitted model parameters | `results/markov_hmm_pilot/models/*.json`, `hmm_parameters.json` | `results/markov_hmm_cost_sensitive_v2/models/<tier>_<task>.json`: key `"markov"` → `"<model>|fold<k>"` (counts, smoothing); key `"hmm_selected"` → HMM matrices |
| Out-of-fold predictions | `results/markov_hmm_pilot/predictions.csv` | `results/markov_hmm_cost_sensitive_v2/predictions.parquet`: one row per evaluated event, columns `p__<model>__<class>` and `decision__<model>__<scheme>` |
| Cost matrices per fold | — | `results/markov_hmm_cost_sensitive_v2/cost_matrix_definitions.json` |
| Metrics | `metrics_summary.csv` | `model_comparison.csv`, `cost_sensitive_comparison.csv`, `rare_event_analysis.csv`, … |
| Development event cache (no final-test matches) | — | `data/interim/sequence_models/event_table_development_v2.pkl` |

## 4. How to change things

| Change | Where | Notes |
|---|---|---|
| Markov order / contexts | `backoff_markov.py` `MODEL_LEVELS` (line 36) and the `markov.models` list in the YAML | A new entry is a tuple of context levels, most specific first; every level must contain the next one (checked in `__post_init__`). New context features must be added to `build_contexts` and `context_cardinalities`. |
| Spatial feature | `zones.py` (`BOUNDARIES`) | e.g. `(20, 40, 60, 80, 100)` for 6 bands; `context_cardinalities` uses `N_ZONES` from `ZONE_NAMES` |
| Smoothing | YAML `markov.concentration_grid` (β candidates) and `markov.base_alpha` | |
| Cost multipliers | YAML `cost.schemes` (`gamma`, `cap`, `false_alarm`) | Only the decisions change; log loss and Brier do not. |
| Taxonomy | YAML `taxonomies.<name>.event_types` | Log loss of different taxonomies is not comparable. |

Experiment 3 adds its own configuration and modules rather than editing these (see
`results/context_aware_event_prediction_v3/README.md`).

## 5. Commands that are safe to run (PyCharm terminal, project root)

| Command | Time | Effect |
|---|---|---|
| `.venv/bin/python scripts/demo_explain_prediction.py` | ~3 s | explains one prediction; writes nothing |
| `.venv/bin/python scripts/demo_explain_prediction.py --model mc1 --task filtered8` | ~3 s | same for another model / taxonomy |
| `.venv/bin/python -m pytest tests/test_v2_spatial_models.py -q` | ~3 s | hand-checked Markov / HMM / zone tests |
| `.venv/bin/python -m pytest tests/test_v2_cost_sensitive.py -q` | ~2 s | cost-sensitive decision tests |
| `.venv/bin/python -m pytest -q` | ~3 min | whole test suite |
| `.venv/bin/python scripts/verify_markov_hmm_cost_sensitive_v2.py` | ~1 min | independent re-check of experiment 2; rewrites only `independent_verification.json` |

**Do not run during a demo:**
- `run_markov_hmm_cost_sensitive_v2.py tier_a` / `tier_b`: these take hours (HMM fits) and would overwrite results.
- Any stage with `--overwrite`.

## 6. Inspecting a single prediction

`scripts/demo_explain_prediction.py` proceeds in four steps:

1. Picks a stored evaluation row (default: the first Tier B fold-0 event whose true next event is a Shot), or the event given by `--event-id`.
2. Rebuilds that match's sequence from the raw JSON. It refuses final-test matches.
3. Loads the model fitted on the other folds and prints each backoff level: context, number of times seen in training, and the top probabilities.
4. Prints all class probabilities, checks them against `predictions.parquet`, and shows the uniform / moderate / strong decisions with their expected costs.

Example output (shortened):

```text
current event  : Carry at x = 104.8 -> Attacking third     previous event: Pass     true next event: Shot
Step 1  context (current=Carry): seen 204,615 times   -> Pass 0.729, Pressure 0.206, ...
Step 3  context (previous=Pass, current=Carry, zone=Attacking third): seen 37,134 times -> Pass 0.666, Pressure 0.250, Miscontrol 0.032, Shot 0.029
```

The model gives Shot only 2.9% here, although the Carry ends inside the penalty box. The attacking third (x ≥ 80) is
too coarse to separate a carry at the edge of the box from one near the halfway line. This is one of the questions of
experiment 3.

## 7. A small example without the full dataset

```python
import numpy as np
from src.sequence_models.backoff_markov import BackoffMarkov, build_contexts, context_cardinalities, targets_of

seqs = [np.array([0, 1, 0, 1, 1]), np.array([1, 0])]          # two short sequences of event ids
zones = [np.array([0, 0, 2, 2, 1]), np.array([1, 1])]          # pitch third of every event
ctx = build_contexts(seqs, zones, n_symbols=2, n_zones=3)
m = BackoffMarkov("mc2_zone", 2, context_cardinalities(2, 3), concentration=2.0).fit(ctx, targets_of(seqs))
print(m.predict_proba(ctx))                                    # one probability row per event
```

`tests/test_v2_spatial_models.py::test_first_order_hierarchical_smoothing_by_hand` checks the same formula against
numbers computed by hand.

## 8. What to show in a live demonstration

1. `src/sequence_models/tasks.py` `build_task_table`: how sequences and targets are made, and why no target crosses halftime.
2. `src/sequence_models/backoff_markov.py` `predict_proba` (about 10 lines): the whole smoothing formula.
3. `scripts/demo_explain_prediction.py`, run live: one real prediction, step by step.
4. `src/sequence_models/cost_sensitive.py` `bayes_decisions`: decisions change, probabilities do not.
5. `tests/test_v2_spatial_models.py` and `tests/test_sequence_hmm.py`: hand-computed checks (HMM likelihood 0.10893).
6. `results/markov_hmm_cost_sensitive_v2/README.md` §6: the model comparison with uncertainty intervals.
