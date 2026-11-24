# Final independent audit: experiments 6A (MLR-H5) and 6B (MLP-H5)

Date: 2026-11-24. Script: `scripts/audit_experiments6a_6b.py`. It runs as
`python scripts/audit_experiments6a_6b.py` and took 430 s. All 69 checks passed
(`audit_checks.csv`).

## Verdicts

| Experiment | Verdict |
|---|---|
| 6A — MLR-H5 | **PASS WITH LIMITATIONS** |
| 6B — MLP-H5 | **PASS WITH LIMITATIONS** |

- **What was not found.** No critical finding and no correctness finding of major severity.
  No data leakage and no final-test access.
- **Limitations (§4).** They restrict the interpretation, not the correctness, of the reported
  numbers.

## 1. Independence of the checks

The script recomputes every audited quantity with code written in the script itself. It
works from raw StatsBomb JSON, the canonical table and the stored artefacts.

**Not used for verification.** The production history builder (`event_history`), the 6A/6B
model and encoding modules, `raw_history_check`, `canonical_task`, `evaluation_v2` and
`xgboost_experiment`.

**Production code imported only as the object under test.**
- `HistoryRows`, to hash the exact training input and compare it with the run manifests.
- The 6B `MLP` class, for the dense-reference and finite-difference gradient checks.
- The 6B `train_fit`, to reproduce one outer refit.

## 2. What was verified

| Area | Evidence |
|---|---|
| Raw reconstruction | 40 complete Tier B matches (seed 20261124) were rebuilt from events and three-sixty JSON with an own implementation of K1–K3. Canonical observations (ids, order, period, index, class, usable 360, eligibility) are identical. All **90,674** eligible rows have identical 5-position histories and targets (class and target event id). Strata: 79 period starts, 320 rows with `<START>`, 72 period ends, 40 match starts, 5,377 repeated timestamps, 6,892 rows with a companion inside the window, rare classes (50/50, Goal Keeper, Duel, Shot, Pressure, Block, Foul Committed) |
| Full history | A pandas `shift` within (match, period) on all 690,147 rows equals the training input. Its hash equals the 6A and 6B manifests |
| Population | Tier B = development − pilot = 304 matches; 690,147 rows; the 83 final-test ids are disjoint from the data and from every fold. The folds partition the matches without train/validation overlap |
| Row identity | The stored predictions of all 9 models (experiments 5, 5B, 6A, 6B) have event ids, matches, targets and folds identical element-wise to the canonical rows. The 6A/6B manifest hashes recompute |
| Probabilities | 15 columns p0..p14 in canonical order; finite, ≥ 0, row sums within 8.9e-16. Column means match class frequencies |
| 6A model | Probabilities from the stored W, b (own encoding, NumPy softmax): max \|Δp\| = 0. The stored coefficients satisfy the optimality conditions of the **row-level** objective \(\frac1N\sum -\log p + \lVert W\rVert^2/(2CN)\) on every fold's outer-training rows (max \|∇\| ≤ 1.0e-13). Deviation coding holds |
| 6A selection | An independent re-run of fold 2's complete nested CV (raw scikit-learn, own cell aggregation) gives identical pooled inner scores, C, and predictions (Δ = 0). Selected C = argmin of the stored scores in all folds, all interior. All 105 fits converged without warnings |
| 6A aggregation and solver | Cells keep all 551,095 training rows (36,704 cells). An unaggregated row-level refit of fold 4 matches the stored predictions (\|Δp\| = 1.9e-11). lbfgs and newton-cholesky reach the same objective (Δ = 7e-12) |
| 6B model | NumPy float64 forward pass of the stored weights equals the stored probabilities (max \|Δp\| = 5.5e-7, float32 network) for all 3 seeds |
| 6B first layer | The embedding-sum first layer equals an explicit dense one-hot network in forward values and **all** gradients (≤ 1.4e-17, float64) |
| 6B gradients and loss | Finite differences on the production module agree to 5.8e-10. NumPy log-sum-exp NLL = `F.cross_entropy` on logits = stored fold-0 log loss |
| 6B training records | For all 175 fits: best epoch = first argmin of the stopping curve (epoch 0 included); stored best loss = the curve minimum; epochs = min(best + 5, 60). Total epochs 5,363. No fit has best epoch 0 |
| 6B selection | Recomputed from per-fit inner scores, it equals the stored choice and the stored weights' configuration (F, F, F, F, E). The 8 frozen configurations were used |
| 6B stopping subsets | Disjoint from the fit matches, inside the outer-training matches, identical to the data-access audit |
| 6B reproduction | Fold 3's primary refit was reproduced from scratch: identical stopping curve, best epoch 17, bitwise identical weights and predictions (CPU, 1 thread) |
| 6B checkpoints | Identifiers equal the run's. Fold records hold probabilities, states, choices and trace. No in-progress files are left. Supervisor status: completed, exit codes 0 / 0 |
| 6B seeds | Seed-level log losses recompute (1.05267 / 1.05215 / 1.05258). The primary replicate 0 is not the best seed |
| Metrics | Pooled metrics, per-class precision/recall/F1/support and confusion matrices of all 9 models were recomputed with own NumPy code and with scikit-learn. They equal the stored tables (≤ 2.2e-16), pooled and per fold |
| Paired bootstrap | Own implementation of the seed-7, 1,000-replicate match bootstrap. All six reported differences, relative differences, CIs and fold signs reproduce exactly |
| Pressure | 12.46 % of rows. Row-weighted contributions reproduce the stored strata and sum to the totals |
| Provenance | Every executed source file (6A: 9, 6B: 11) is byte-identical to the published file. The git-ignored artefacts match their run-time hashes. All historical and 6A/6B checksums verify |

### Recomputed pooled results

| Model | Log loss | Accuracy | Macro-F1 | Weighted-F1 | Shot AP | Shot TP / predicted |
|---|---:|---:|---:|---:|---:|---:|
| Logistic F0 | 1.15850 | 0.65047 | 0.10043 | 0.54974 | 0.03364 | 0 / 0 |
| Markov 2 | 1.06807 | 0.66591 | 0.12043 | 0.57310 | 0.05275 | 0 / 0 |
| XGB-H2 | 1.06761 | 0.66590 | 0.12085 | 0.57311 | 0.05290 | 0 / 0 |
| XGB-H5 | 1.05185 | 0.66683 | 0.13253 | 0.57785 | 0.07694 | 12 / 30 |
| MLR-H5 | 1.10297 | 0.64908 | 0.12186 | 0.55786 | 0.05849 | 1 / 6 |
| MLP-H5 | 1.05267 | 0.66679 | 0.13015 | 0.57616 | 0.07567 | 13 / 51 |

**Shot support.** 7,212 Shot rows for every model.

### Recomputed paired log-loss differences

\(d_i\) is model − reference per-row NLL. The data are 690,147 rows from 304 matches. The
bootstrap uses 1,000 match-level replicates with seed 7.

| Comparison | Mean \(d_i\) | Relative | 95 % CI | Folds model better |
|---|---:|---:|---|---:|
| MLR-H5 − XGB-H5 | +0.05112 | +4.86 % | [+0.04975, +0.05253] | 0 / 5 |
| MLR-H5 − Markov 2 | +0.03490 | +3.27 % | [+0.03380, +0.03595] | 0 / 5 |
| MLR-H5 − Logistic F0 | −0.05554 | −4.79 % | [−0.05706, −0.05406] | 5 / 5 |
| MLP-H5 − XGB-H5 | +0.00082 | +0.08 % | [+0.00052, +0.00115] | 0 / 5 |
| MLP-H5 − MLR-H5 | −0.05030 | −4.56 % | [−0.05173, −0.04895] | 5 / 5 |
| MLP-H5 − Markov 2 | −0.01540 | −1.44 % | [−0.01617, −0.01475] | 5 / 5 |

## 3. Findings

| # | Severity | Location | Finding | Impact | Resolution |
|---|---|---|---|---|---|
| F1 | MAJOR (interpretation, not correctness) | 6B design: `src/sequence_models/mlp_h5_experiment.py` `run_nested_cv`; config `training.stopping_fraction` | The MLP outer refits update parameters on 89.8–90.5 % of the outer-training rows (24 matches per fold reserved for early stopping). XGB-H5 used 100 % (`training_population_6b_vs_xgb_h5.csv`). The MLP-H5 vs XGB-H5 comparison (+0.0008) is a comparison of complete training procedures, not controlled for training-data quantity | The reported numbers are correct. "XGB-H5 is better than MLP-H5" cannot be attributed to the model family alone | Documented prominently (6B README, `08_mlp_h5.md` §9 with per-fold table, conclusion). Predictions unchanged; no retraining. A full-population refit is proposed as a future, not yet authorised experiment |
| F2 | MINOR | `docs/thesis/07*.md`, `08*.md`, both result READMEs | The CIs were not described as excluding refitting / CV-overlap / selection / dataset uncertainty. There was no explicit statement that "close to XGB-H5" is not formal equivalence. A claim that MPS is "less deterministic" was untested. There was no caveat that permutation importance uses unrealistic sequences. "Source commit a2343e9" could be read as the executed code | Over-interpretation risk only | Corrected in the documentation |
| F3 | MINOR | `tests/test_logistic_h5_experiment6a.py:10,22` | Unused imports (pyflakes) | none | Removed |
| F4 | MINOR | `scripts/run_mlp_h5_experiment6b.py` (integrity rows "all fits finished with finite losses", "history identical …") and `scripts/run_logistic_h5_experiment6a.py` ("MLR-H5 ids identical …") | These audit rows are recorded as PASS literals. They document guards that raise earlier in the run, rather than being checks evaluated at that point | Could overstate the number of independent checks | Reported here. The underlying properties were verified independently by this audit. Run artefacts are not rewritten |
| F5 | MINOR | `tests/test_mlp_h5_experiment6b.py` | No test covered row-weighted epoch-loss aggregation with an unequal final batch. The code was correct | Coverage gap | Regression test added (`test_epoch_train_loss_is_row_weighted_with_unequal_final_batch`) |
| F6 | INFORMATIONAL | `mlp_h5_experiment.py` `train_fit` | Epoch-0 bug (initial model not the incumbent best) was found by a unit test **before** the full run and fixed. The executed file is byte-identical to the published one, and all 175 fits satisfy the corrected rule | none on results | Verified |
| F7 | INFORMATIONAL | 6B `fit_log.csv` | The 60-epoch cap bound for 14 inner fits, all of candidate H (lr 3e-4), which was never selected. H's inner score may be slightly pessimistic | Selection could only change in favour of H; the margin to F is 0.0013–0.0025 in inner log loss | Documented |
| F8 | INFORMATIONAL | 6B | The seed-to-seed spread of the pooled log loss (0.0005) is of the order of the MLP-H5 − XGB-H5 gap. All 3 seeds have the same sign | The CI does not include optimisation randomness | Documented |
| F9 | INFORMATIONAL | 6B `train_fit` | The reported `train_loss` is the running row-weighted mean over the epoch in training mode (dropout on), not an end-of-epoch evaluation. Model selection uses only the stopping and held losses (evaluation mode) | none | Documented here |
| F10 | INFORMATIONAL | `pyproject.toml` | `torch>=2.15,<3` was added as a core dependency, following the xgboost convention. It was tested with 2.15.0. It makes `pip install -e .` heavier. There is no lock file; exact versions are in each `environment.json` | Reproducibility relies on the recorded versions | Accepted |
| F11 | INFORMATIONAL | `scripts/supervise_mlp_h5_experiment6b.py` | If the supervisor itself crashed between stages, `run_status.json` would remain "running". `--status` then reports `supervisor_alive: false`. The test for the supervisor is string-based; the lifecycle was verified by a live smoke run and the full run | none on results | Documented |
| F12 | INFORMATIONAL | both | Argmax metrics for rare classes are near zero for every model (Shot: ≤ 13 TP of 7,212); AP and log loss are the informative measures | Interpretation | Already documented |

**Audit-tool defect (not a project finding).** The first audit run stopped in section 5 because
the per-fold table has no `shot_precision` column. The script was corrected and re-run
completely; this directory holds the complete re-run.

## 4. Limitations that remain

**Both experiments.**
- **CI scope.** Out-of-fold estimates on the 304 development matches only; the final test is
  untouched. The CIs describe match resampling of fixed predictions.
- **Mechanism.** The interaction interpretation of the MLR-H5 deficit, which is concentrated
  on Pressure rows, is supported by MLP-H5 and XGB-H5 closing it. It is not a proven
  mechanism.
- **Feature scope.** Event-type history only: five canonical events, not five seconds, and no
  spatial or temporal features.

**6A.**
- C is chosen per fold (1 or 10, a flat optimum).
- Only main effects; no interaction features.

**6B.**
- F1 (training population).
- A small search space (8 configurations).
- Bitwise reproducibility verified on this machine, thread count and software only.

## 5. Files

| File | Content |
|---|---|
| `audit_checks.csv` | all 69 checks with values |
| `audit_summary.json` | totals |
| `audit_provenance.json` | script hash, versions, seeds |
| `metrics_recomputed.csv` | pooled metrics of the 9 models; includes the pooled minus unweighted fold-mean log loss, showing pooled ≠ fold average |
| `paired_differences_recomputed.csv` | the six paired comparisons |
| `pressure_decomposition.csv` | Pressure contributions |
| `training_population_6b_vs_xgb_h5.csv` | the F1 numbers |
