# Experiment 6A: MLR-H5 (multinomial logistic regression, five-event history) on `actions15_360current_v1`

Date: 2026-11-23. Method and interpretation: `docs/thesis/07_logistic_regression_h5.md`.

**Research question.** If a multinomial logistic regression receives exactly the
five-event history of XGB-H5, does it reach comparable predictive performance?

**Answer (primary metric, pooled out-of-fold log loss).** No.
- **vs. XGB-H5.** MLR-H5 has log loss **1.10297**, against XGB-H5's 1.05185. The paired
  difference is **+0.0511** (95 % match-bootstrap CI [0.0498, 0.0525]), and XGB-H5 is
  better in all 5 folds.
- **vs. Markov 2.** MLR-H5 is also worse than the second-order Markov chain, by +0.0349
  (CI [0.0338, 0.0360]; 5/5 folds). Markov 2 sees only \((e_{t-1}, e_t)\), but it models
  their interaction.
- **vs. logistic F0.** MLR-H5 is better than logistic F0, by −0.0555
  (CI [−0.0571, −0.0541]; 5/5 folds).

## Reproduce

```bash
python scripts/run_logistic_h5_experiment6a.py --smoke    # fold-0 inner data only: timing, memory, convergence
python scripts/run_logistic_h5_experiment6a.py            # writes this directory (refuses to overwrite); resumes committed folds
python scripts/verify_logistic_h5_experiment6a.py         # independent post-run audit -> post_run_verification.json
python -m pytest tests/test_logistic_h5_experiment6a.py
```

- **Execution.** The run used `nohup` with `caffeinate -i -w <pid>`. The wall time was
  1,057 s on a MacBook Air (8 GB RAM, 8 threads); the smoke test measured a peak RSS of
  about 1.0 GB.
- **Source provenance.**
  - **At run time.** The run started from commit `a2343e9` with the experiment files not yet
    committed, so that commit alone does not identify the executed code.
  - **Identification.** The executed code is identified by the per-file SHA-256 in
    `environment.json` (`source_files_sha256`).
  - **Verification.** The final audit (`results/audit_experiments6a_6b_2026-11-24/`) verified
    that the published files are byte-identical to the executed ones.
- **Scope of the CIs.** The CIs resample evaluation matches for fixed out-of-fold predictions.
  They do not include:
  - refitting;
  - the overlap of training sets between folds;
  - hyperparameter selection;
  - generalisation to other competitions.
- **Software.** Python 3.11.1, scikit-learn 1.9.0, NumPy 2.4.6, SciPy 1.17.1,
  pandas 3.0.5.

## 1. Setup

| Item | Value |
|---|---|
| Rows | 690,147 Protocol-B rows of Tier B, from 304 matches. Event ids, order, targets and folds are identical row by row to experiments 5 and 5B (verified element-wise and by hash) |
| Final test | 83 matches, never loaded (`final_test_loaded: 0`, `data_access_audit.csv`) |
| Folds | The 5 match-grouped outer folds of experiment 2. Nested leave-one-inner-fold-out (4 inner folds) is used for C |
| Features | \((e_{t-4},\dots,e_t)\) from `event_history` (the XGB-H5 history, not re-implemented). Position-specific one-hot: lags 4..1 over 16 categories (15 classes + `<START>`), \(e_t\) over 15; 79 sparse binary columns. Names and order are identical to XGB-H5 |
| Model | Softmax regression with an L2 penalty. scikit-learn `LogisticRegression(solver="newton-cholesky", l1_ratio=0)`, intercepts unpenalised, no class weights |
| Grid | C ∈ {0.01, 0.1, 1, 10, 100}, with a pre-declared edge-extension rule. The rule was never triggered |
| Fitting | Count-weighted unique (features, target) cells. This is the identical objective; verified by an unaggregated refit of fold 0: max \|Δp\| = 2.1e-11 |
| Bootstrap | 1,000 match-level replicates, seed 7 (the same resampled matches as the baselines and experiments 5/5B) |

**Solver choice.** The choice was made before the full run, on optimisation evidence only:
- lbfgs needed 13,205 iterations at C = 100, because rare one-hot columns make the
  problem ill-conditioned.
- newton-cholesky converges in 10–39 iterations.

Every one of the 105 fits converged without warnings. Its max \|gradient\|, recomputed
independently in NumPy, is ≤ 7.5e-11.

## 2. Results (pooled out-of-fold, same 690,147 rows)

| Model | Log loss [95 % CI] | Brier | Accuracy | Macro-F1 | Weighted-F1 | Shot AP | Shot recall | Shot precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic F0 | 1.15850 [1.1445, 1.1738] | 0.52249 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 / 7,212 | – (0 predicted) |
| Markov 1 | 1.15850 [1.1445, 1.1738] | 0.52249 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 / 7,212 | – |
| Markov 2 | 1.06807 [1.0554, 1.0819] | 0.49375 | 0.6659 | 0.1204 | 0.5731 | 0.0527 | 0 / 7,212 | – |
| XGB-F0 | 1.15850 [1.1445, 1.1738] | 0.52249 | 0.6505 | 0.1004 | 0.5497 | 0.0337 | 0 / 7,212 | – |
| XGB-H2 | 1.06761 [1.0549, 1.0815] | 0.49365 | 0.6659 | 0.1208 | 0.5731 | 0.0529 | 0 / 7,212 | – |
| XGB-H5 | 1.05185 [1.0394, 1.0654] | 0.48798 | 0.6668 | 0.1325 | 0.5778 | 0.0769 | 12 / 7,212 (0.0017) | 0.400 (12/30) |
| **MLR-H5** | **1.10297 [1.0898, 1.1174]** | **0.50569** | **0.6491** | **0.1219** | **0.5579** | **0.0585** | 1 / 7,212 (0.0001) | 0.167 (1/6) |

- **Prior models.** Their metrics were recomputed from their saved predictions and equal
  experiment 5B's stored tables, CIs included, to ≤ 3.1e-13 (`baseline_recomputation_check.csv`).
- **Argmax metrics.** Accuracy, F1 and Shot recall/precision are argmax decisions.
- **Shot recall.** It is near zero for every model and is not a meaningful discriminator.

### Paired differences (MLR-H5 − reference; match-bootstrap 95 % CI; folds where MLR-H5 is better)

| Reference | Δ log loss | rel. | CI | MLR-H5 better in | Δ accuracy | Δ macro-F1 | Δ Shot AP |
|---|---:|---:|---|---:|---:|---:|---:|
| XGB-H5 (primary) | +0.05112 | +4.9 % | [+0.0498, +0.0525] | 0 / 5 | −0.0178 | −0.0107 | −0.0185 |
| Markov 2 | +0.03490 | +3.3 % | [+0.0338, +0.0360] | 0 / 5 | −0.0168 | +0.0014 [+0.0002, +0.0028] | +0.0057 |
| XGB-H2 | +0.03536 | +3.3 % | [+0.0343, +0.0364] | 0 / 5 | −0.0168 | +0.0010 [−0.0002, +0.0023] | +0.0056 |
| Logistic F0 | −0.05553 | −4.8 % | [−0.0571, −0.0541] | 5 / 5 | −0.0014 | +0.0214 | +0.0248 |

The XGB-F0 and Markov 1 rows equal the logistic F0 row up to the third decimal. All CIs
are in `paired_differences.csv`.

**Independent check.** The independent verifier recomputed the primary difference as the
mean of \(d_i\) exactly. Its own bootstrap (2,000 reps, seed 20261123) gives
[0.0497, 0.0525].

### Per fold (log loss)

| Fold | C (inner CV) | MLR-H5 | XGB-H5 | Markov 2 | Logistic F0 |
|---|---:|---:|---:|---:|---:|
| 0 | 10 | 1.10450 | 1.05345 | 1.06980 | 1.16019 |
| 1 | 1 | 1.10029 | 1.04873 | 1.06511 | 1.15523 |
| 2 | 1 | 1.10522 | 1.05418 | 1.07074 | 1.16226 |
| 3 | 1 | 1.10338 | 1.05279 | 1.06857 | 1.15792 |
| 4 | 10 | 1.10145 | 1.05009 | 1.06614 | 1.15689 |

**Flat inner-CV curve.** The curve is flat between C = 1 and C = 10: they differ by
≤ 2.2e-5 (`inner_cv_scores.csv`). The two choices are practically equivalent.

## 3. Where the gap arises (descriptive, `nll_difference_by_stratum.csv`)

- **Rows with full history.** 687,771 of the 690,147 rows have a full history (t ≥ 4).
  They carry essentially the whole gap.
- **Rows after a Pressure.** Rows with current event Pressure are 12 % of rows but
  contribute +0.029 of the +0.051 gap to XGB-H5 and +0.026 of the +0.035 gap to Markov 2.
  Their mean per-row NLL differences are +0.23 and +0.21.
- **Inference (not proven by these numbers).** The next event after a Pressure depends
  jointly on the pressure and on what preceded it. An additive model cannot express that
  the effect of \(e_{t-1}\) changes with \(e_t\). Markov 2 and the trees can.
- **Short-history rows.** At t ≤ 2, MLR-H5 beats Markov 2. Markov 2 uses a first-order
  fallback at t = 0, whereas MLR-H5 has `<START>` columns.

## 4. Coefficients (`coefficient_*.csv`, `top_coefficients.csv`, `position_strength.csv`)

- **Source.** The coefficients come from the 5 outer-fold models, each fitted on
  outer-training rows only. They are deviation-coded: Σ over classes = 0, and Σ over
  each position block = 0 (verified in `identifiability.csv`).
- **Strength by position.** The mean range of coefficients over categories is 6.77 for
  \(e_t\), 2.71 for \(e_{t-1}\), 1.52 for \(e_{t-2}\), 1.07 for \(e_{t-3}\) and 1.02 for
  \(e_{t-4}\).
- **Held-out permutation importance.** Permuting each position raises log loss by:
  - \(e_t\): +2.25;
  - \(e_{t-1}\): +0.160;
  - \(e_{t-2}\): +0.021;
  - \(e_{t-3}\): +0.008;
  - \(e_{t-4}\): +0.003;
  - \(e_{t-4..t-2}\) jointly: +0.026.
- **Stability.** Among features with ≥ 200 training rows in every fold, 77 % have the same
  sign in all 5 folds.

## 5. Files

| File | Content |
|---|---|
| `model_comparison.csv`, `metrics_by_fold.csv`, `per_class_metrics.csv`, `confusion_matrices.csv`, `predicted_class_distribution.csv` | metrics of all 8 models on the same rows |
| `paired_differences.csv`, `paired_differences_by_fold.csv` | paired comparisons with match-bootstrap CIs and fold signs |
| `nll_difference_by_stratum.csv` | decomposition of the log-loss gaps |
| `hyperparameter_choices.csv`, `inner_cv_scores.csv`, `fit_log.csv`, `timings.csv`, `fold_counts.csv` | nested CV, every fit (iterations, warnings, gradient), timings, class counts |
| `coefficients_by_fold.csv`, `coefficient_summary.csv`, `top_coefficients.csv`, `position_strength.csv`, `intercepts.csv`, `identifiability.csv`, `permutation_importance_mlr_h5.csv` | interpretation |
| `feature_verification.json`, `raw_feature_verification.csv` | feature checks; raw-JSON rebuild of 707 rows in 11 strata |
| `fit_verification.csv` | unaggregated row-level refit of fold 0 |
| `baseline_recomputation_check.csv` | prior models recomputed from saved predictions |
| `integrity_audit.csv` | the run's 40 checks: 39 PASS, 1 INFO |
| `post_run_verification.json` | the independent verifier: 31 checks, PASS |
| `data_access_audit.csv`, `split_manifest.json`, `task_manifest.json`, `environment.json`, `experiment_config.yaml`, `task_config.yaml` | provenance |
| `checksums.json` | the run's manifest; this README is excluded |
| `checksums_supplementary.json` | files written after the run |

**Git-ignored artefacts.** Hashes are in `environment.json`.
- `data/interim/logistic_h5_experiment6a/oof_predictions_mlr_h5.parquet`
- `fold_models_mlr_h5.npz`
- `checkpoints/`
- `run.log`
