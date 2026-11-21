# Experiment 5: XGBoost on `actions15_360current_v1`

Date: 2026-11-20/21. Task: canonical `actions15_360current_v1`
(`docs/canonical_prediction_protocol.md`), used unchanged. Mathematical description:
`docs/thesis/05_xgboost_gradient_boosting.md`.

**Data scope:**
- **Development data only.** The 83 final-test matches were never loaded
  (`split_manifest.json`: `final_test_loaded: 0`).
- **Nothing earlier was modified.** No earlier dataset, result or experiment changed. The
  canonical dataset was read, not rebuilt, and its hashes were verified.

**Reproduce:**

```bash
python scripts/build_canonical_task.py                 # only if data/interim/canonical_task/ is missing
python scripts/run_xgboost_experiment5.py --smoke      # timing check on fold-0 inner sets only
python scripts/run_xgboost_experiment5.py --overwrite  # this directory (~5 h of compute on an M1)
python -m pytest tests/test_xgboost_experiment5.py
```

**Requirements.** XGBoost 3.1.3 is pinned and asserted at run time. On macOS it needs the
OpenMP runtime: `brew install libomp`.

## 1. Setup

| Item | Value |
|---|---|
| Rows | Protocol B, Tier B: **690,147 rows, 304 matches**. The identical row IDs, targets and fold of every row are used for all 6 models (`task_manifest.json` → `row_identity`) |
| Outer folds | Experiment 2's 5 match-grouped folds, checked to be identical to the canonical baseline run |
| XGB-F0 | \(\mathbf{x}_t = (E_t)\), one-hot, 15 columns |
| XGB-H5 | \(\mathbf{x}_t = (E_{t-4},\dots,E_t)\), 79 columns. Missing history is the token `<START>`. Padded rows: \(E_{t-1}\) 568, \(E_{t-2}\) 1,174, \(E_{t-3}\) 1,771, \(E_{t-4}\) 2,376; all of them are kept |
| Target | \(y_t = E_{t+1}\), the next canonical event |
| Model | `multi:softprob`, K = 15, `hist`, `nthread` 8, seed 20261120; `subsample` 1 (see §5) |
| Search | 20 seeded candidates over `max_depth`, `learning_rate`, `min_child_weight`, `reg_lambda`, `colsample_bytree` (`search_candidates.csv`). Leave-one-inner-fold-out within the outer training folds. \(M\) (`n_estimators`) is chosen from the pooled inner curve: patience 50, cap 2,000 |
| Workload | 810 fits (405 per feature set) and 1,181,898 boosting rounds. Compute ≈ 5.5 h (search 5.2 h); wall-clock 15.0 h, because the Mac slept during the run |
| Baselines | frequency, Markov 1, Markov 2 and logistic F0, recomputed in memory. They match the stored canonical results exactly (`baseline_recomputation_check.csv`: max \(\lvert\Delta\rvert\) = 0) |
| Uncertainty | Match-level bootstrap: 1,000 replicates, seed 7, paired. Shot AP is bootstrapped with match weights |

## 2. Results (pooled out-of-fold, 690,147 rows)

| Model | Log loss [95 % CI] | Brier | Accuracy | Macro-F1 | Weighted-F1 | Shot AP | Shot recall | Classes predicted |
|---|---|---|---|---|---|---|---|---|
| frequency | 1.59618 [1.5844, 1.6090] | 0.7116 | 0.4019 | 0.0382 | 0.2304 | 0.0103 | 0 | 1 |
| Markov 1 | 1.15850 [1.1445, 1.1738] | 0.5225 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 | 2 |
| logistic F0 | 1.15850 [1.1445, 1.1738] | 0.5225 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 | 2 |
| **XGB-F0** | 1.15850 [1.1445, 1.1738] | 0.5225 | 0.6505 | 0.1004 | 0.5497 | 0.0337 | 0 | 2 |
| Markov 2 | 1.06807 [1.0554, 1.0819] | 0.4937 | 0.6659 | 0.1204 | 0.5731 | 0.0527 | 0 | 6 |
| **XGB-H5** | **1.05185** [1.0394, 1.0654] | **0.4880** | **0.6668** | **0.1325** | **0.5778** | **0.0769** | 0.0017 | 11 |

**Fold means ± SD** (`metrics_by_fold.csv`):
- **XGB-F0:** log loss 1.1585 ± 0.0028, the same as Markov 1.
- **XGB-H5:** log loss 1.0519 ± 0.0023, Shot AP 0.077 ± 0.003.
- **Markov 2:** log loss 1.0681 ± 0.0024, Shot AP 0.051 ± 0.002.

**Paired differences** (`paired_differences.csv`). Negative log-loss differences are better.
The bootstrap CI is at the match level. "Folds" gives the number of folds better / worse.

| Comparison | Δ log loss [95 % CI] (relative) | Folds | Δ accuracy | Δ macro-F1 | Δ Shot AP [CI] | Δ Shot recall [CI] |
|---|---|---|---|---|---|---|
| XGB-F0 − Markov 1 | −0.000001 [−0.000008, +0.000007] | 2 / 3 | 0 | 0 | +0.00002 [−0.0001, +0.0001] | 0 |
| XGB-F0 − logistic F0 | −0.000003 [−0.000015, +0.000006] | 2 / 3 | 0 | 0 | +0.00002 | 0 |
| XGB-H5 − XGB-F0 | **−0.1067** [−0.1093, −0.1043] (−9.2 %) | 5 / 0 | +0.0164 | +0.0321 | +0.0433 [+0.039, +0.048] | +0.0017 [+0.0007, +0.0028] |
| XGB-H5 − Markov 2 | **−0.0162** [−0.0169, −0.0156] (−1.5 %) | 5 / 0 | +0.0009 [+0.0006, +0.0012] | +0.0121 | +0.0242 [+0.021, +0.028] | +0.0017 [+0.0007, +0.0028] |
| Markov 2 − Markov 1 (reference) | −0.0904 | 5 / 0 | +0.0154 | +0.0200 | +0.0191 | 0 |

**Per-fold Δ log loss, XGB-H5 − Markov 2:** −0.0164, −0.0164, −0.0166, −0.0158, −0.0161
(`paired_differences_by_fold.csv`).

**Shot.** Two metrics behave differently:
- **Ranking (Shot AP).** XGB-H5 ranks Shot targets better: AP 0.077 vs. 0.053 for
  Markov 2, at a prevalence of 1.05 %.
- **Argmax decisions (Shot recall).** XGB-H5 predicts Shot 30 times in 690,147 rows;
  12 are correct (recall 0.0017, precision 0.40). Every other model never predicts Shot.
  The recall change is therefore real (CI excludes 0), but practically negligible.

**Per class** (`per_class_metrics.csv`, `confusion_matrices.csv`,
`predicted_class_distribution.csv`):
- **Pass and Carry still dominate.** XGB-H5 predicts 11 classes, against 6 for Markov 2.
- **Recall gains:** Ball Recovery 0.112 → 0.154, Block 0.029 → 0.046, Clearance 0 → 0.050.
- **AP rises for 15 of 15 classes vs. Markov 2.** The largest absolute gains are Shot
  +0.024, Dribble +0.030, Pass +0.026, Carry +0.024, Pressure +0.026 and
  Dispossessed +0.020.
- **Never correctly predicted under argmax:** 50/50, Dispossessed, Dribble, Duel,
  Foul Committed and Interception.

**Calibration.** Classwise ECE is 0.0003 for XGB-H5 and 0.0001 for Markov 2. Both are
small (`model_comparison.csv`).

## 3. History positions (XGB-H5)

**Held-out permutation importance** (`permutation_importance_h5.csv`). The permutation is
done on each outer-validation fold, with that fold's final model, 10 repeats. Values are
fold means [min–max over folds]:

| Permuted | Δ log loss | Δ accuracy |
|---|---|---|
| \(E_t\) | 2.310 [2.293–2.322] | −0.329 |
| \(E_{t-1}\) | 0.254 [0.241–0.258] | −0.043 |
| \(E_{t-2}\) | 0.0281 [0.0275–0.0291] | −0.0023 |
| \(E_{t-3}\) | 0.0073 [0.0068–0.0079] | −0.0006 |
| \(E_{t-4}\) | 0.0031 [0.0029–0.0033] | −0.0002 |
| \(E_{t-4..t-2}\) jointly (beyond the Markov-2 context) | 0.0372 [0.0360–0.0383] | −0.0031 |
| \(E_{t-4..t-1}\) jointly | 0.282 [0.270–0.288] | −0.043 |

- **Every position is used, and the ordering is the same in every fold.** All values are
  positive in every fold, with SD ≤ 0.009 over repeats. The importance decays
  monotonically with lag.
- **Training-set split gain** (`gain_importance_h5.csv`) gives the same order: \(E_t\)
  82 %, \(E_{t-1}\) 14 %, \(E_{t-2}\) 2.5 %, \(E_{t-3}\) 0.8 %, \(E_{t-4}\) 0.5 %.
- **Caveats:**
  - Permuting one position breaks its dependence on the neighbouring positions, so the
    values describe how the fitted model uses its inputs.
  - Neither measure shows a causal effect.

## 4. Interpretation

1. **XGB-F0 = Markov 1 = logistic F0.** All three agree in log loss to within
   \(3\cdot10^{-6}\), and their argmax decisions are identical. With one categorical input,
   all three represent the same \(15\times15\) table
   (`docs/thesis/05_xgboost_gradient_boosting.md` §4). Boosting adds neither bias nor
   variance here. **XGBoost does not outperform the first-order baselines when it has the
   same information.**
2. **History is what helps.** Adding \(E_{t-4..t-1}\) improves XGBoost by 0.107 log loss
   (−9.2 %).
3. **XGB-H5 beats Markov 2, but most of that comes from \(E_{t-1}\), which both use.** The
   margin over Markov 2 is −0.016 (−1.5 %), consistent in 5/5 folds, with a CI far from
   0. XGB-H5 sees three more positions than Markov 2, so this is not a pure
   model-family comparison. Its sources are:
   - \(E_{t-2..t-4}\): permuting them jointly costs 0.037 log loss inside XGB-H5;
   - less coarse use of the \((E_{t-1}, E_t)\) context than Markov 2's count-and-backoff
     table.

   This experiment does not separate the two. That would need an XGB-H2 ablation, which
   is outside the approved scope.
4. **Practical meaning.**
   - Accuracy improves by +0.0009 (+0.09 percentage points), and Pass/Carry still
     dominate the argmax decisions.
   - The gains are in the probabilities: log loss, AP for all classes, and Shot AP
     ×1.46 over Markov 2.
   - Cost: XGB-H5 uses \(15 \times\) ~2,000 depth-4 trees, against a table of at most
     \(15^3\) cells for Markov 2. Fitting takes minutes instead of milliseconds.

## 5. Verification and caveats

**Checks:**
- **Raw-JSON verification.** 500 sampled rows (100 of them with padding) were
  reconstructed from the raw StatsBomb JSON by independent code, including K1–K3. All 500
  feature vectors and targets match (`raw_feature_verification.csv`).
- **Outer-validation isolation.** `data_access_audit.csv` lists the 100 data sets handed to
  XGBoost. No outer-validation match appears in any inner or outer training set; outer
  validation is used only for the 10 final predictions.
- **Reproducibility.** Refitting the fold-0 models gives bit-identical predictions
  (`fit_verification.csv`).
- **Cells vs. rows** (`fit_verification.csv`). The objective is identical, but float32
  accumulation differs:

  | | Δ log loss | Max Δp | Argmax agreement |
  |---|---|---|---|
  | XGB-F0 | 1e-7 | 0.00006 | 100 % |
  | XGB-H5, 1,895 rounds | 1.6e-4 (the cell fit is lower) | 0.21 | 99.4 % |

  For long H5 fits the cell fit and the row fit are numerically different realisations of
  the same objective. The effect is about 1 % of the XGB-H5 − Markov 2 gap.

**Caveats:**
- **The round cap binds for XGB-H5.** The selected candidate is the same in all folds
  (#2: depth 4, η 0.1, λ 100, `min_child_weight` 1, `colsample` 0.75), and its \(M\) is
  1,895, 1,992, 2,000, 2,000 and 2,000. Its inner curve still fell by about 1e-4 over the
  last 200 rounds, so XGB-H5 is probably slightly under-fitted. The cap was fixed in
  advance and was not changed. A higher cap would likely gain only around 1e-4 in log loss,
  about 1 % of the margin over Markov 2.
- **Cap hits for XGB-F0.** 80 of 100 F0 searches reached the cap on a flat curve
  (≈1e-7 per round). The F0 selections themselves stopped early (190–545 rounds).
- **`subsample` was fixed at 1, not tuned.** On count-weighted cells it would sample
  cells, not rows.
- **Hyperparameter ties.** The inner log losses of the 20 candidates are close: F0 within
  about 3e-4, H5 within about 1e-2 (`inner_cv_curves.csv`).

## 6. Files

**Tables:**
- `model_comparison.csv`, `metrics_by_fold.csv`;
- `paired_differences.csv`, `paired_differences_by_fold.csv`;
- `per_class_metrics.csv`, `confusion_matrices.csv` (15 × 15 for each model),
  `predicted_class_distribution.csv`, `fold_counts.csv`;
- `hyperparameter_choices.csv`, `search_candidates.csv`, `inner_cv_curves.csv` (the full
  pooled curves), `timings.csv`;
- `permutation_importance_h5.csv`, `gain_importance_h5.csv`.

**Verification:**
- `baseline_recomputation_check.csv`, `raw_feature_verification.csv`;
- `data_access_audit.csv`, `fit_verification.csv`.

**Provenance:**
- `experiment_config.yaml`, `task_config.yaml`;
- `task_manifest.json` (dataset hashes, recomputed raw source digest, row identity);
- `split_manifest.json`;
- `environment.json` (git `c61939d` plus the then-uncommitted Experiment 5 files, package
  versions, seeds, workload, commands);
- `checksums.json`.

**Not committed** (`data/interim/xgboost_experiment5/`, git-ignored): the 10 fitted
boosters (`.ubj`) and the out-of-fold probabilities of all 6 models
(`oof_predictions.parquet`).
