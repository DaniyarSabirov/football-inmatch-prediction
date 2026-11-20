# Canonical baselines: `actions15_360current_v1`

Date: 2026-11-19. This is the first experiment on the canonical task
(`docs/canonical_prediction_protocol.md`).

**Data scope:**
- **Development data only.** The 83 final-test matches were never loaded.
- **Earlier work untouched.** No earlier dataset or result was modified.

**Reproduce:**

```bash
python scripts/build_canonical_task.py              # canonical dataset (data/interim/, git-ignored)
python scripts/run_canonical_baselines.py --overwrite   # this directory, about 1.5 min on an M1
python -m pytest tests/test_canonical_task.py
```

## 1. Setup

| Item | Value |
|---|---|
| Task | `actions15_360current_v1` (`task_config.yaml`): actions15, companions K1–K3 removed, K4 inactive |
| Rows | Protocol B: the current event has a usable 360 snapshot; the same rule for training and evaluation |
| Target | next canonical event (52,956 eligible rows, 7.7 %, have a target without 360) |
| Matches | Tier B, 304 development matches; folds: experiment 2's 5 match-grouped folds (`split_manifest.json`) |
| Rows per fold | train 551,095–552,724; validation 137,423–139,052 (`fold_counts.csv`); 690,147 validation rows in total |
| Hyperparameters | nested leave-one-inner-fold-out on the outer training folds; grids fixed in `experiment_config.yaml` before the run |
| Uncertainty | match-level bootstrap, 1,000 replicates, seed 7; paired differences; per-fold signs |
| Dataset hash | Tier B table `c4a71d1f…53b1ef` (`task_manifest.json`) |
| Raw source digest | over the 668 development raw files: `79821dfe…ff86c70` (`raw_source_hashes.csv` lists every file) |
| Software | Python 3.11.1, numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0 (`environment.json`) |

**Models.**
- **frequency.** α = 1.
- **markov1.** Additive α; nested CV chose **0.3** in all folds.
- **markov2.** Context \((E_{t-1}, E_t)\), backing off below 20 training rows; α = **0.1**
  in 4 folds and 0.3 in 1.
- **logistic_F0.** One-hot \(E_t\); L2, lbfgs, `tol = 1e-8`; C = **10** in all folds,
  inside the extended grid 1e-4…1e3. It converged in all folds after 3,130–3,596
  iterations. It was fitted on the count-weighted (current, next) cells, which is exactly
  the row-level objective.
- **No model selection by outer folds.** Nothing was chosen on the outer validation
  folds.

## 2. Results (pooled out-of-fold, 690,147 rows)

| Model | Log loss [95 % CI] | Brier | Accuracy | Macro-F1 | Weighted-F1 | Shot AP | Shot recall | Classes predicted |
|---|---|---|---|---|---|---|---|---|
| frequency | 1.5962 [1.5844, 1.6090] | 0.7116 | 0.4019 | 0.0382 | 0.2304 | 0.0103 | 0 | 1 |
| markov1 | 1.1585 [1.1445, 1.1738] | 0.5225 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 | 2 |
| **markov2** | **1.0681** [1.0554, 1.0819] | **0.4937** | **0.6659** | **0.1204** | **0.5731** | **0.0527** | 0 | 6 |
| logistic_F0 | 1.1585 [1.1445, 1.1738] | 0.5225 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 | 2 |

**Metric types.**
- **From probabilities:** log loss, Brier and AP.
- **From argmax decisions:** accuracy, F1, recall and the number of classes predicted.

**Paired log-loss differences** (`paired_differences.csv`):

| Comparison | Δ log loss [95 % CI] | Folds better / worse |
|---|---|---|
| markov1 − frequency | −0.4377 [−0.4422, −0.4330] | 5 / 0 |
| markov2 − markov1 | −0.0904 [−0.0925, −0.0883] | 5 / 0 |
| logistic_F0 − markov1 | +0.000002 [−0.000005, +0.000012] | 3 / 2 |

**Fold level** (`metrics_by_fold.csv`): log loss ranges, folds 0–4:

| Model | Range |
|---|---|
| frequency | 1.592–1.600 |
| markov1 = logistic_F0 | 1.155–1.162 |
| markov2 | 1.065–1.071 |

The ranking is the same in every fold.

**Per class** (`per_class_metrics.csv`, which has precision, recall, F1, AP, support,
predicted counts and true positives for every model and class):
- **First-order models.** Under argmax they predict **only Pass and Carry**: F1 0.732 and
  0.774, recall 0.866 and 0.916.
- **markov2.** It also predicts Ball Recovery (recall 0.11), Pressure, Block and
  Miscontrol.
- **Shot.** No model predicts it: recall 0.
- **Ranking (AP).** Shot AP is 0.034 for markov1, 0.053 for markov2, against a
  prevalence of 1.05 %.
- **The highest minority-class APs (markov1):**
  - Goal Keeper 0.134 (prevalence 0.46 %);
  - Pressure 0.191;
  - Ball Recovery 0.197.

**Validation class distribution** (all folds): Pass 277,340, Carry 227,728,
Pressure 85,509, Ball Recovery 26,560, Block 10,585, Clearance 10,294, Dribble 8,216,
Miscontrol 7,247, Shot 7,212, Duel 6,851, Foul Committed 6,407, Dispossessed 6,280,
Interception 6,210, Goal Keeper 3,149, 50/50 559.

## 3. Interpretation

1. **Logistic F0 equals the first-order Markov chain.** The predictions coincide, as the
   theory predicts (`docs/thesis/02_mathematical_framework.md` §6). The log-loss
   difference is 2·10⁻⁶ with a CI including 0, and the argmax decisions are identical.
   The equivalence found in experiment 4 therefore also holds under the canonical task.
2. **History matters more than model form.** The second-order chain improves log loss by
   0.090 (−7.8 %) and raises Shot AP from 0.034 to 0.053 (×1.6) over the first-order
   models.
3. **No deterministic "free" classes any more.** Under the canonical task, first-order
   macro-F1 is 0.100, compared with 0.214 for experiment 4. This is not a loss of model
   quality: the companion records that had made Duel, Goal Keeper and 50/50 trivially
   predictable (`results/investigation_360_taxonomy_2026-11-18/` §5) no longer exist.

## 4. Historical reference points (different tasks — not a model comparison)

| | Experiment 4 (historical): scenarioD15, Protocol A, 701,144 rows | **Canonical**: actions15_360current_v1, Protocol B, 690,147 rows |
|---|---|---|
| frequency, log loss | 1.6592 | 1.5962 |
| Markov 1, log loss | 1.1520 (α = 1) | 1.1585 (α = 0.3) |
| Markov 2, log loss | 1.0497 (α = 1) | 1.0681 (α nested) |
| logistic F0, log loss | 1.1519 | 1.1585 |
| Markov 1 / logistic F0, macro-F1 | 0.2145 | 0.1004 |
| Markov 1 / logistic F0, accuracy | 0.6529 | 0.6505 |

**These differences come from the changed task, not from model quality:**
- the taxonomy (companion records removed, which changes targets and current events);
- the training population (360-only rows);
- the evaluation population (10,997 fewer rows).

None of these numbers was used to choose or reject the canonical protocol, which was
fixed in advance on scientific grounds.

## 5. Files

**Tables:**
- `model_comparison.csv`, `paired_differences.csv`, `metrics_by_fold.csv`;
- `per_class_metrics.csv`, `confusion_matrices.csv`;
- `fold_counts.csv` (training and validation rows and class counts per fold);
- `hyperparameter_choices.csv` (inner-CV curves, grid-edge flags), `logistic_fits.csv`,
  `timings.csv`.

**Reproducibility:**
- configuration copies: `experiment_config.yaml`, `task_config.yaml`;
- `task_manifest.json` (counts, dataset and source hashes);
- `split_manifest.json` (folds, and the hashes of the split manifest and the folds file);
- `raw_source_hashes.csv`, `environment.json` (versions, seeds, commands),
  `checksums.json`.

**Version note.** `environment.json` records the git commit the run was made from
(`7df070a`), together with the then-uncommitted files. After this work is committed,
rerunning with `--overwrite` records the new commit. The numbers do not change, because
the code is deterministic.
