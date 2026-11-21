# Experiment 5B: XGB-H2 vs. Markov 2 on `actions15_360current_v1`

Date: 2026-11-21. This is the controlled comparison of model families. XGB-H2 receives
exactly the canonical Markov-2 context \((E_{t-1}, E_t)\).

Method: `docs/thesis/06_xgboost_h2_controlled_comparison.md`. Experiment 5:
`results/xgboost_experiment5_actions15_360current_v1_2026-11-20/`, which is unchanged and
checksum-verified before and after this run.

**Reproduce:**

```bash
python scripts/run_xgboost_experiment5b.py --smoke             # timing (fold-0 inner sets only)
caffeinate -i python scripts/run_xgboost_experiment5b.py       # this directory; resumes committed folds
python scripts/analyze_xgboost_experiment5b.py                 # post-run decomposition (§4)
python -m pytest tests/test_xgboost_experiment5b.py
```

**Inputs.** The run reads, without changing them:
- the canonical dataset `data/interim/canonical_task/actions15_360current_v1/`;
- Experiment 5's out-of-fold predictions `data/interim/xgboost_experiment5/oof_predictions.parquet`.

Both are git-ignored and hash-verified.

## 1. Setup

| Item | Value |
|---|---|
| Rows | 690,147 Protocol-B rows, Tier B, 304 matches. Row IDs, order, targets and folds are identical to experiment 5 and the canonical baselines |
| Final test | 83 matches, never loaded (`final_test_loaded: 0`) |
| Input | \(\mathbf{x}_t = (E_{t-1}, E_t)\): 31 one-hot columns. 568 rows have \(E_{t-1}\) = `<START>`. 191 of the 240 possible contexts occur (`feature_verification.json`, schema hash `5ebd61eb…`) |
| Target | \(y_t = E_{t+1}\), 15 classes in canonical order |
| Procedure | Identical to experiment 5. The runner asserts identical `xgboost`/`search` configurations: 20 seeded candidates, lockstep pooled inner curve, patience 50, cap 2,000, same tie rule, refit with \(M\) rounds. Tuned independently: nothing was copied from XGB-H5 |
| Training representation | Count-weighted cells, as in experiment 5. The audit found no defect (§5) |
| Comparators | Stored out-of-fold predictions of experiment 5: frequency, Markov 1/2, logistic F0, XGB-F0, XGB-H5. They reproduce the stored metrics to ≤ 2·10⁻¹⁶ |
| Uncertainty | Paired match-level bootstrap: 1,000 replicates, seed 7 (same replicates as before), **percentile** 95 % CI; per-fold signs |
| Run | Completed normally: all 5 folds computed, none resumed. 405 fits, 681,415 boosting rounds. 9,384 s wall-clock, of which search took 8,748 s; Mac kept awake by `caffeinate -i -w <pid>` |
| Software | XGBoost 3.1.3, Python 3.11.1, numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0 (`environment.json`) |

## 2. Results (pooled out-of-fold, 690,147 rows)

| Model | Information | Log loss [95 % CI] | Brier | Accuracy | Macro-F1 | Weighted-F1 | Shot AP | Shot recall |
|---|---|---|---|---|---|---|---|---|
| Markov 1 | \(E_t\) | 1.15850 | 0.5225 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 |
| logistic F0 | \(E_t\) | 1.15850 | 0.5225 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 |
| XGB-F0 | \(E_t\) | 1.15850 | 0.5225 | 0.6505 | 0.1004 | 0.5497 | 0.0337 | 0 |
| Markov 2 | \(E_{t-1}, E_t\) | 1.06807 [1.0554, 1.0819] | 0.4937 | 0.6659 | 0.1204 | 0.5731 | 0.0527 | 0 |
| **XGB-H2** | \(E_{t-1}, E_t\) | **1.06761** [1.0549, 1.0815] | 0.4937 | 0.6659 | 0.1208 | 0.5731 | 0.0529 | **0** |
| XGB-H5 | \(E_{t-4}, \dots, E_t\) | 1.05185 [1.0394, 1.0654] | 0.4880 | 0.6668 | 0.1325 | 0.5778 | 0.0769 | 0.0017 |

The CIs are from `model_comparison.csv`. Every value was re-derived independently from the
saved predictions with scikit-learn: maximum difference 2·10⁻¹⁶.

**Fold log loss** (`metrics_by_fold.csv`):

| Fold | Markov 2 | XGB-H2 | XGB-H5 |
|---|---|---|---|
| 0 | 1.06980 | 1.06930 | 1.05345 |
| 1 | 1.06511 | 1.06474 | 1.04873 |
| 2 | 1.07074 | 1.07024 | 1.05418 |
| 3 | 1.06857 | 1.06804 | 1.05279 |
| 4 | 1.06614 | 1.06570 | 1.05009 |
| mean ± SD | 1.0681 ± 0.0024 | 1.0676 ± 0.0023 | 1.0519 ± 0.0023 |

## 3. Controlled comparisons (`paired_differences.csv`, `paired_differences_by_fold.csv`)

### Primary: XGB-H2 − Markov 2 (same information)

| Metric | Difference [95 % CI] | Relative | Folds better / worse / equal |
|---|---|---|---|
| Log loss | **−0.00047** [−0.00056, −0.00038] | −0.044 % | 5 / 0 / 0 |
| Brier | −0.00010 [−0.00012, −0.00008] | −0.02 % | 5 / 0 / 0 |
| Accuracy | −0.000007 [−0.00006, +0.00004] | — | 2 / 3 / 0 |
| Macro-F1 | +0.00042 [+0.00017, +0.00069] | +0.35 % | 4 / 1 / 0 |
| Weighted-F1 | +0.00001 [−0.00007, +0.00008] | — | 4 / 1 / 0 |
| Shot AP | +0.00015 [−0.00023, +0.00050] | +0.28 % | 4 / 1 / 0 |
| Shot recall | 0 (neither model ever predicts Shot) | — | 0 / 0 / 5 |

**Per-fold Δ log loss:** −0.00049, −0.00037, −0.00050, −0.00053, −0.00045.

**Predicted classes** (`predicted_class_distribution_h2_vs_markov2.csv`). XGB-H2 predicts
9 classes; Markov 2 predicts 6. The change in argmax counts is tiny:

| Class | Change in argmax predictions |
|---|---|
| Clearance | +12 |
| Goal Keeper | +20 |
| Dribble | +5 |
| Pressure | +25 |
| Ball Recovery | −73 |

### Secondary: XGB-H5 − XGB-H2 (same model family, adding \(E_{t-2..t-4}\))

| Metric | Difference [95 % CI] | Folds better |
|---|---|---|
| Log loss | **−0.01576** [−0.01645, −0.01511] (−1.48 %) | 5 / 5 |
| Accuracy | +0.00093 [+0.00060, +0.00124] | 5 / 5 |
| Macro-F1 | +0.01169 [+0.01070, +0.01281] | 5 / 5 |
| Shot AP | +0.02405 [+0.02076, +0.02769] | 5 / 5 |
| Shot recall | +0.00166 [+0.00069, +0.00281] | 5 / 5 |

**Per-fold Δ log loss:** −0.01585, −0.01601, −0.01606, −0.01525, −0.01560.

**Other comparisons:**
- **XGB-H2 − XGB-F0:** −0.0909 [−0.0930, −0.0888]. This is almost exactly
  Markov 2 − Markov 1 (−0.0904).
- **Arithmetic decomposition of experiment 5's XGB-H5 − Markov 2 (−0.01622).** Pooled log
  losses are means over the same rows, so the split is exact:
  - XGB-H5 − XGB-H2 = −0.01576 (97.1 %). This is associated with the longer history, within
    the XGBoost estimator.
  - XGB-H2 − Markov 2 = −0.00047 (2.9 %). This reflects the estimator together with the
    different \(t = 0\) rule, with identical information (§4).

  This is a decomposition of observed differences along one path (via XGB-H2), not a
  causal attribution. Another path, for example via a fifth-order Markov chain (not
  fitted), need not split the same way. `post_run_verification.json` checks the
  arithmetic.

## 4. Where XGB-H2 differs from Markov 2 (`h2_vs_markov2_decomposition.csv`)

The analysis is post-run and reads only stored predictions
(`scripts/analyze_xgboost_experiment5b.py`). Under the canonical Markov 2, a row **backs
off** to Markov 1 when \(t = 0\) or when its context has fewer than 20 training rows
(`markov2_backoff_rates.csv`: 0.13–0.15 % of validation rows per fold).

| Subset | Rows | Mean Δ log loss (H2 − M2) [95 % CI] | Share of the total gain |
|---|---|---|---|
| Backoff rows (all) | 989 | −0.189 [−0.231, −0.149] | **58 %** |
| — \(t = 0\), period start | 568 | −0.259 [−0.290, −0.223] | 46 % |
| — rare context | 421 | −0.095 [−0.179, −0.013] | 12 % |
| Full-context rows | 689,158 | −0.00020 [−0.00027, −0.00012] | 42 % |

**Interpretation.**
- **Most of the gain comes from missing-history handling.** Most of XGB-H2's small
  advantage comes from 0.14 % of rows. On those rows the two models deliberately treat
  missing or rare history differently: Markov 2 replaces the context with Markov 1, while
  XGB-H2 learns the category `<START>` and pools rare contexts.
- **Where both models see the full context, they almost coincide.** The mean |Δp| is
  1·10⁻⁴, and the log-loss gain is 0.0002.
- **The difference is about the estimator and the \(t = 0\) rule, not extra
  information.** It is a difference in smoothing and backoff policy. It is not evidence
  that boosted trees capture structure a two-event table cannot express.
- **Same information, not the same model.** XGB-H2 can represent any table over the 240
  contexts \((E_{t-1} \text{ or <START>}, E_t)\). The canonical Markov 2 has no
  \(t = 0\) context of its own and uses Markov 1 there.

## 5. Weighted-cell audit (inner data of outer fold 0; no outer-validation data)

**Exactly the same in both forms** (`weighted_cell_audit_checks.csv`,
`weighted_cell_audit_round1_structure.csv`):
- Cell weights sum to the training-row count (414,899).
- 953 cells share features with another cell but have a different target; they are kept
  separate.
- Weights are not normalised by XGBoost: doubling them changes the fit when λ > 0.
- The intercept is identical.
- Round-1 trees are identical, with leaf differences ≤ 1.2·10⁻⁷.
- Repeated fits are bit-identical. A row-level fit is invariant to row order.

**Divergence** (`weighted_cell_audit_divergence.csv`). Divergence begins when the first
near-tied split flips on float32 accumulation: at round 134 for candidate 0 and round 364
for candidate 2. Effects:
- |Δ log loss| ≤ 1.2·10⁻⁴;
- maximum |Δp| ≤ 0.068;
- argmax agreement ≥ 99.6 %.

**Round selection** (`weighted_cell_audit_selection.csv`). Cells and rows can select
different numbers of rounds on a flat curve:
- candidate 0: 531 vs. 298 rounds, with minimum inner log loss 1.067543 vs. 1.067551;
- candidate 2: both 2,000.

**Fold-0 refit on individual rows** (`fit_verification.csv`): |Δ log loss| = 1.3·10⁻¹⁰,
maximum |Δp| = 0.0047, argmax agreement 99.992 %.

**Verdict.** There is no correctness defect. Weighted cells are a numerically different
realisation of the same objective, not a bit-identical one. This is a limitation, but small
relative to every comparison above except XGB-H2 − Markov 2 (−0.00047). That comparison is
only about four times the largest cell/row log-loss divergence measured for a single long
fit (1.2·10⁻⁴ in the audit; 1.3·10⁻¹⁰ for the actual fold-0 model). It should therefore be
read as "XGB-H2 ≈ Markov 2, slightly better through missing-history handling".

## 6. Diagnostics

**Hyperparameters** (`hyperparameter_choices.csv`). The selected models are:
- depth 8, η 0.2, λ 100: candidate 8 (`min_child_weight` 5, colsample 0.5, folds 0, 2, 3,
  \(M\) = 1,798–1,991);
- candidate 5 (`min_child_weight` 1, colsample 1.0, folds 1, 4, \(M\) = 621–737).

None is at the cap. However, 71 of the 100 searches reached 2,000 rounds, and in folds 2 and
3 \(M\) lies within 16 rounds of the cap. The curves there are flat, so the cap may bind
weakly.

**Over- or underfitting.**
- **Overfitting: no sign.** Inner vs. outer log loss per fold is 1.0675/1.0693,
  1.0685/1.0647, 1.0672/1.0702, 1.0678/1.0680 and 1.0684/1.0657. The gap is
  non-systematic in sign and comparable to between-fold variation.
- **Underfitting: no sign beyond the shared cap.**

**Permutation importance** (`permutation_importance_h2.csv`): \(E_t\) 2.34, \(E_{t-1}\) 0.285
(log-loss increase). These describe how the fitted model uses its inputs; they are not causal.

## 7. Integrity audit (`integrity_audit.csv`: 32 checks, 0 FAIL)

| Check | Result |
|---|---|
| Experiment 5 predictions match its recorded row/target/fold hashes and reproduce all 6 stored log losses | PASS |
| Row IDs, order, targets and folds identical to experiment 5; 690,147 rows, 304 matches | PASS |
| All 5 outer folds completed; each row predicted exactly once; no duplicate row IDs | PASS |
| Probabilities finite, > 0, 15 columns in canonical order; row sums within 7·10⁻¹⁶ of 1 | PASS |
| Fold and pooled metrics reproduce from saved predictions; independently via scikit-learn (≤ 2·10⁻¹⁶) | PASS |
| No outer-validation match in any of the 45 tuning/training data sets (`data_access_audit.csv`) | PASS |
| No final-test match anywhere (83 IDs checked) | PASS |
| Refit of fold 0 bit-identical to the checkpointed predictions | PASS |
| Raw-JSON rebuild of 600 rows (300 random, 150 period start, 150 period end) | PASS (600 / 600) |
| Raw source digest equals the canonical manifest | PASS |
| Experiment 5 results unchanged (23 checksummed files + README, before and after) | PASS |
| Full test suite | PASS (576 tests) |
| Weighted-cell vs. row-level fitting | LIMITATION (§5) |
| Shared round cap reached by 71 / 100 searches | LIMITATION (§6) |

## 8. Checkpointing

**Location:** `data/interim/xgboost_experiment5b/checkpoints/` (git-ignored).

**Contents:** one committed set per outer fold:
- `fold<k>.npz`: validation row positions and probabilities;
- `fold<k>.ubj`: the model;
- `fold<k>.json`: the commit marker, written last. It holds the choice, the inner curves,
  the timings, the fold metrics, the data-access trace, the hashes of the two other files,
  and the identifiers.

**Atomic writes:** temp file, `fsync`, `os.replace`.

**Resume rule.** A fold is resumed only if every identifier matches:
- configuration and search sha256;
- dataset hash;
- folds file and split manifest;
- row-ID and feature-matrix hashes;
- source digest;
- XGBoost version.

A mismatch or a corrupted file raises `CheckpointMismatch`, so runs are never mixed. This
run resumed no fold.

## 9. Files

**Tables:**
- `model_comparison.csv`, `metrics_by_fold.csv`;
- `paired_differences.csv`, `paired_differences_by_fold.csv`;
- `per_class_metrics.csv`, `confusion_matrices.csv`;
- `predicted_class_distribution.csv`, `predicted_class_distribution_h2_vs_markov2.csv`.

**Search:**
- `hyperparameter_choices.csv`, `search_candidates.csv`;
- `inner_cv_curves.csv`, `timings.csv`, `fold_metrics_from_checkpoints.csv`.

**Verification:**
- `feature_verification.json`, `feature_class_counts.csv`, `raw_feature_verification.csv`;
- `data_access_audit.csv`, `fit_verification.csv`, `integrity_audit.csv`;
- `weighted_cell_audit_*.csv`, `markov2_backoff_rates.csv`.

**Post-run files** (created after the run; not in the run's original `checksums.json`,
which is left unchanged):
- `h2_vs_markov2_decomposition.csv` (`scripts/analyze_xgboost_experiment5b.py`);
- `post_run_verification.json`: 22 independent checks of experiments 5 and 5B from stored
  artefacts (`scripts/verify_xgboost_experiments5.py`).

Their sha256 are listed in `checksums_supplementary.json`, which the verification script
writes.

**Provenance of code vs. publication.** The run was executed from commit `c61939d` plus the
then-uncommitted experiment-5/5B files (`environment.json`). The sha256 of every source file
the run used is recorded in `environment.json` → `source_files_sha256`. The published files
are byte-identical to them. The commit that publishes these artefacts is later than the
execution commit; it does not change the executed code.

**Provenance:**
- `experiment_config.yaml`, `task_config.yaml`;
- `task_manifest.json`, `split_manifest.json`;
- `environment.json` (git `c61939d` plus the uncommitted experiment-5/5B files; per-file
  source hashes; checkpoint identifiers);
- `checksums.json`.
