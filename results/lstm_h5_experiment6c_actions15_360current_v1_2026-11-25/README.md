# Experiment 6C — LSTM-H5 on `actions15_360current_v1`

**Run.** `exp6c-20261125T195623-158a9af7`, 2026-11-25 19:56 → 2026-11-26 01:06 (18,600 s
experiment + 23 s independent audit).

**Method note.** `docs/thesis/09_lstm_h5.md`. **Configuration** (frozen before the run):
`experiment_config.yaml` = `configs/lstm_h5_experiment6c_v1.yaml`
(sha256 `3741f739…ac63184a`).

**Verdict: PASS WITH LIMITATIONS.** Confirmed by the final independent audit, `results/audit_experiment6c_2026-11-26/`.

| Item | Value |
|---|---|
| Run integrity audit | `integrity_audit.csv`: 70 checks, 66 PASS, 4 INFO, 0 FAIL |
| Independent audit | `post_run_verification.json`: 59 checks, 0 FAIL |
| Exit codes (supervisor) | experiment 0, independent audit 0; `COMPLETED` written |

## 1. Setup (identical to experiments 5–6B)

- **Rows.** 690,147 Protocol-B rows of Tier B (304 matches), 15 classes in canonical order.
- **Row identity.** Event ids, order, targets, folds and the five-position history are
  identical element-wise to experiments 5, 5B, 6A and 6B (verified, and checked against the
  manifest hashes).
- **Final test.** The 83 final-test matches were never loaded.
- **Input.** \(X_t = (E_{t-4},\dots,E_t)\) as 5 one-hot vectors over 16 symbols (15 classes +
  `<START>`), oldest first; tensor (B, 5, 16).
- **Model.** Unidirectional, stateless `torch.nn.LSTM` (\(h_0 = c_0 = 0\) for every window);
  logits \(W_\text{out}\,\mathrm{drop}(h_5) + b_\text{out}\) of the last layer; softmax only
  at inference.
- **Training protocol of MLP-H5, unchanged.** AdamW (β = 0.9/0.999, ε = 1e-8, decoupled weight
  decay on weight matrices), batch 512, ≤ 60 epochs, patience 5, no gradient clipping.
- **Nested CV.** 5 outer × 4 inner match-grouped folds, early stopping on a 10 % match-level
  stopping subset of the fit's training matches.
- **Same data sets as MLP-H5.** All 75 distinct inner/outer training, stopping and validation
  match sets, and their 525 uses, are **identical to experiment 6B** (`data_access_audit.csv`
  vs. 6B's; `stopping_subset_manifest.json`).
  - The seed-derivation convention is also the same. Equal seed integers do not imply equal
    random draws for different architectures.
  - Architecture, parameter counts, the candidate lists and the selected configurations differ
    between the two experiments.
- **Search.** 8 prespecified configurations (`search_candidates.csv`); 175 fits, 6,085 epochs.
- **Execution.**
  - CPU, 1 thread per process, deterministic algorithms; the 5 outer folds ran in separate
    worker processes, at most 4 in parallel.
  - The fold-0 primary refit was retrained in the main process and is **bitwise identical**
    (max |Δp| = 0).

## 2. Selected configurations

| Fold | Selected | H × layers | Inner log loss | Refit best epoch / epochs | Outer-val. log loss |
|---:|---|---|---:|---|---:|
| 0 | E | 32 × 2 | 1.051947 | 31 / 36 | 1.052802 |
| 1 | F | 64 × 2 | 1.052799 | 17 / 22 | 1.048671 |
| 2 | F | 64 × 2 | 1.051708 | 22 / 27 | 1.054362 |
| 3 | F | 64 × 2 | 1.052439 | 17 / 22 | 1.052201 |
| 4 | E | 32 × 2 | 1.052979 | 28 / 33 | 1.049275 |

- **Inner-CV ranking.** A two-layer network (E or F) was selected in every fold, and E and F
  ranked first and second in four folds. In fold 2, C was second and E third.
  - G (lr 3e-4) was worst in four folds and second-worst in fold 4.
  - G hit the 60-epoch cap in 11 of its 20 inner fits and H in 2. Their inner scores may
    therefore be slightly pessimistic. Their inner log losses were above the selected
    configuration's by 0.0009–0.0015 (H) and 0.0017–0.0029 (G).
- **Gradients.** The largest per-batch global gradient norm was 2.59. 718 of 4,541,996 batches
  exceeded 1, none exceeded 5. No non-finite value or warning occurred.

## 3. Pooled out-of-fold results (690,147 rows, 304 matches)

| Model | Log loss | Brier | Accuracy | Macro-F1 | Weighted-F1 | Shot AP | Shot TP / predicted (recall) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic F0 | 1.15850 | 0.52249 | 0.65047 | 0.10043 | 0.54974 | 0.03364 | 0 / 0 (0) |
| Markov 2 | 1.06807 | 0.49375 | 0.66591 | 0.12043 | 0.57310 | 0.05275 | 0 / 0 (0) |
| XGB-H2 | 1.06761 | 0.49365 | 0.66590 | 0.12085 | 0.57311 | 0.05290 | 0 / 0 (0) |
| XGB-H5 | 1.05185 | 0.48798 | 0.66683 | **0.13253** | **0.57785** | 0.07694 | 12 / 30 (0.0017) |
| MLR-H5 | 1.10297 | 0.50569 | 0.64908 | 0.12186 | 0.55786 | 0.05849 | 1 / 6 (0.0001) |
| MLP-H5 | 1.05267 | 0.48838 | 0.66679 | 0.13015 | 0.57616 | 0.07567 | 13 / 51 (0.0018) |
| **LSTM-H5** | **1.05147** | 0.48803 | **0.66688** | 0.12982 | 0.57667 | **0.08084** | **28 / 75 (0.0039)** |

**Shot.** 28 correct argmax Shot predictions (against 12 for XGB-H5 and 13 for MLP-H5) are still
only 0.388 % of the 7,212 Shot targets. This is not practical Shot detection. Shot AP, a
threshold-free ranking measure, is the informative quantity; argmax recall reflects a single
decision rule.

- **Prior models.** All metrics and CIs were recomputed from their saved predictions and equal
  6B's stored table (≤ 1e-12, `baseline_recomputation_check.csv`).
- **Shot support.** 7,212.
- **Argmax metrics for rare classes.** They remain near zero for every model. LSTM-H5 never
  predicts Dribble, Miscontrol, Dispossessed, Interception, Foul Committed or 50/50 as the argmax
  (`per_class_metrics.csv`).

## 4. Paired differences (LSTM-H5 − reference)

**Bootstrap.** Match-level, 1,000 replicates, seed 7. Negative log loss favours LSTM-H5.

| Reference | Δ log loss | rel. | 95 % CI | Folds LSTM better |
|---|---:|---:|---|---:|
| **MLP-H5** | −0.00120 | −0.114 % | [−0.00153, −0.00090] | 5 / 5 |
| **XGB-H5** | −0.00038 | −0.036 % | [−0.00069, −0.00009] | 4 / 5 |
| **MLR-H5** | −0.05150 | −4.67 % | [−0.05300, −0.05011] | 5 / 5 |
| Markov 2 | −0.01661 | −1.56 % | [−0.01740, −0.01589] | 5 / 5 |
| XGB-H2 | −0.01614 | −1.51 % | [−0.01693, −0.01543] | 5 / 5 |
| Logistic F0 | −0.10704 | −9.24 % | [−0.10969, −0.10470] | 5 / 5 |

- **Per fold vs XGB-H5.** −0.00065, −0.00006, +0.00018, −0.00059, −0.00082.
- **Per fold vs MLP-H5.** −0.00098, −0.00053, −0.00178, −0.00083, −0.00190.

**Secondary metrics.**
- **vs MLP-H5.**
  - Accuracy +0.00009 [−0.00020, +0.00037] and macro-F1 −0.00034 [−0.0014, +0.0007]: neither
    CI excludes 0.
  - Brier −0.00036 [−0.00049, −0.00024].
  - Shot AP +0.0052 [+0.0025, +0.0081].
- **vs XGB-H5.**
  - Accuracy +0.00005 [−0.00025, +0.00032] and Brier +0.00004 [−0.00007, +0.00016]: neither
    CI excludes 0.
  - Macro-F1 −0.0027 [−0.0037, −0.0018], i.e. lower.
  - Shot AP +0.0039 [+0.0013, +0.0066].

## 5. Seeds (prespecified; replicate 0 is primary and was not chosen by result)

| Replicate | Log loss | − MLP-H5 (same replicate) | − XGB-H5 |
|---:|---:|---:|---:|
| 0 (primary) | 1.051465 | −0.00120 | −0.00038 |
| 1 | 1.051218 | −0.00094 | −0.00063 |
| 2 | 1.051344 | −0.00124 | −0.00050 |

- **Spread.** The LSTM seed spread is 0.00025.
- **Against MLP-H5.** All three LSTM seeds are better than all three MLP-H5 seeds
  (1.05215–1.05267).
- **Against XGB-H5.** All three are better than XGB-H5. The primary is the *least* favourable of
  the three.

## 6. Where the differences come from (`nll_difference_by_stratum.csv`, descriptive)

| Reference | Total | Pressure rows (12.5 %) | Pass | Carry | All other current classes |
|---|---:|---:|---:|---:|---:|
| MLP-H5 | −0.00120 | −0.00044 | −0.00039 | −0.00027 | −0.00010 |
| XGB-H5 | −0.00038 | −0.00080 | −0.00005 | +0.00014 | +0.00033 |
| MLR-H5 | −0.05150 | −0.02959 | −0.01286 | −0.00278 | −0.00627 |

- **vs XGB-H5.** The 85,965 Pressure rows (12.46 % of the rows; conditional mean −0.00642)
  contribute −0.00080 to the pooled −0.00038. The other 14 current classes together contribute
  +0.00042; Carry, Clearance, Goal Keeper and Ball Recovery are among those with positive
  contributions. Full table: `results/audit_experiment6c_2026-11-26/decomposition_by_current_class.csv`.
- **vs MLP-H5.** The advantage is spread over Pressure, Pass and Carry.
- **Status.** These are descriptive decompositions, not causal explanations.

**Permutation importance** (held-out, descriptive; the permuted histories are artificial). Mean
log-loss increase: \(E_t\) +2.83, \(E_{t-1}\) +0.35, \(E_{t-2}\) +0.048, \(E_{t-3}\) +0.012,
\(E_{t-4}\) +0.004.

## 7. Verification highlights

| Check | Result |
|---|---|
| Raw JSON | 707 stratified rows (6A sample) and 12 complete matches (27,265 rows; audited K1–K3 code) rebuilt: histories and targets identical, before the run and again in the independent audit |
| NumPy LSTM reference vs nn.LSTM | Pre-run, all 8 architectures: float32 ≤ 5.2e-8 and float64 ≤ 6.9e-17 in logits |
| Saved weights | Probabilities of all rows and seeds: ≤ 5.6e-7 (runner); an independent re-implementation in the audit is also < 1e-5 |
| Gate level (float64, every fold's primary model) | h_s and c_s at all 5 positions, the recovered output gate and the logits ≤ 5.3e-15 |
| Finite differences | Of the NumPy loss vs autograd: 3.0e-10 |
| Early stopping | Every fit: best epoch = first argmin of its stopping curve, best loss = curve minimum, epochs = min(best + 5, 60); no fit had best epoch 0 |
| Selection | Recomputed from the per-fit inner scores in the independent audit: equal to the stored choice and the stored weights' configuration |
| Paired differences and CIs | Reproduced exactly (1e-12) by an own seed-7 bootstrap; an independent 2,000-replicate bootstrap (seed 20261125) agrees on every CI's exclusion of 0 |
| Historical results | Checksums of the baselines, experiments 5, 5B, 6A and 6B and the 6A/6B audit unchanged; their predictions, models and the canonical dataset unchanged |

## 8. Limitations

- **Training population vs XGB-H5.**
  - LSTM-H5, like MLP-H5, updates parameters on 89.8–90.5 % of the outer-training rows
    (`training_population.csv`); XGB-H5 used 100 %.
  - The −0.00038 vs XGB-H5 is therefore a comparison of complete procedures. This establishes
    a difference in training-data availability. The magnitude and direction of its effect on
    performance are unknown without a controlled refit.
  - LSTM-H5 and MLP-H5 share the training/stopping partitions, the evaluation rows, the
    protocol and the seed convention. They differ in architecture, parameter count, search space
    and selected configuration, so the comparison does not isolate the recurrence alone.
- **Effect size.** The gains over MLP-H5 (approximately 0.11 %; 0.114 %) and XGB-H5
  (approximately 0.04 %; 0.036 %) are statistically detectable but very small in practical
  terms. They are of the order of the seed spread of MLP-H5 (0.0005).
- **Scope of support.** The results support a small predictive advantage of the selected
  LSTM-H5 procedure on this development population. They do not support general superiority of
  recurrent networks.
- **No equivalence claim.** No equivalence margin was prespecified.
- **CI scope.** The CIs cover the resampling of evaluation matches for fixed predictions. They do
  not cover refitting, the overlap of cross-validation training sets, hyperparameter selection,
  initialisation (reported via 3 seeds) or the choice of competitions.
- **Input limits.** Five event positions and event types only, so no claim about long-term
  football dependencies is possible. No spatial or temporal features.
- **Search.** A small, prespecified search (8 configurations); G and H were partly capped by
  the epoch limit.
- **Reproducibility.** Bitwise reproducibility was verified on this machine, thread count and
  software only (Python 3.11.1, torch 2.15.0, NumPy 2.4.6, Apple M1, CPU).

## 9. Files

**Main tables.** `model_comparison.csv`, `metrics_by_fold.csv`, `per_class_metrics.csv`,
`confusion_matrices.csv`, `predicted_class_distribution.csv`, `paired_differences.csv`,
`paired_differences_by_fold.csv`, `shot_summary.csv`, `seed_robustness.csv`,
`nll_difference_by_stratum.csv`, `log_loss_by_current_class.csv`,
`permutation_importance_lstm_h5.csv`.

**Training.** `search_candidates.csv` (with parameter counts), `parameter_counts.csv` (LSTM-H5
and MLP-H5), `inner_cv_scores.csv`, `hyperparameter_choices.csv`, `fit_log.csv` (including
gradient-norm statistics), `training_histories.csv`, `timings.csv`, `training_population.csv`.

**Integrity.** `integrity_audit.csv`, `post_run_verification.json`, `gate_level_verification.csv`,
`mathematical_verification_pre_run.json`, `feature_verification.json`,
`raw_feature_verification.csv`, `raw_reconstruction.csv`, `data_access_audit.csv`,
`baseline_recomputation_check.csv`.

**Provenance.** `environment.json` (versions, hardware, seeds, source hashes, worker PIDs and
memory, historical snapshots), `split_manifest.json`, `stopping_subset_manifest.json`,
`task_manifest.json`, `task_config.yaml`, `checksums.json` (written by the run; excludes this
README), `checksums_supplementary.json` (files created after the run).

**Large artefacts** (git-ignored, hashed in `environment.json`), in `data/interim/lstm_h5_experiment6c/`:
- `oof_predictions_lstm_h5.parquet`: primary and 2 seeds;
- `fold_models_lstm_h5.npz` and `.pt`: all weights;
- `checkpoints/`;
- logs, `run_status.json`, `COMPLETED`.

## 10. Reproduction

```bash
python scripts/run_lstm_h5_experiment6c.py --smoke            # fold-0 inner data only: timing, memory, worker plumbing
python scripts/supervise_lstm_h5_experiment6c.py --detach     # runner + independent audit; caffeinate -d -i -w <supervisor>
python scripts/supervise_lstm_h5_experiment6c.py --status
python scripts/verify_lstm_h5_experiment6c.py                 # independent audit only
python -m pytest tests/test_lstm_h5_experiment6c.py
```

**Seeds.** Training base 20261123 (= 6B), folds 20261027, bootstrap 7, permutation 11, raw
reconstruction 20261125.

**Source provenance.** The run started from commit `46d0e2d` with the 6C files uncommitted. The
executed sources are identified by `environment.json → source_files_sha256` and were verified
byte-identical to the current files after the run.
