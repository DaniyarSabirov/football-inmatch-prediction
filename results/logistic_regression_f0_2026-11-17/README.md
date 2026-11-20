# Experiment 4: multinomial logistic regression with the current event only (F0), K = 15

Date: 2026-11-17. **Development data only.**
- **Final test.** The 83 final-test matches were never read (`leakage_checks.json`:
  `final_test_matches_loaded = 0`).
- **Earlier work.** Earlier experiments, the production dataset and the visualization were
  not modified.
- **Not committed.** No commit was made.

**Reproduce** (about 51 min on the M1; 5 × ~8.5 min of row-level L-BFGS):
```bash
.venv/bin/python scripts/run_logistic_f0.py --overwrite
```

**Generated files.**
- **Configuration and summary.** `experiment_config.yaml`, `summary.md`.
- **Tables.** All tables are CSV.
- **Figures.** `figures/`.
- **Predictions.** `predictions_f0.parquet` is git-ignored.
- **Code.** Model `src/sequence_models/logistic.py`; features
  `src/sequence_models/feature_sets.py`; protocol `src/sequence_models/f0_experiment.py`;
  classes `src/sequence_models/event_classes.py`.
- **Theory.** `docs/thesis/02_mathematical_framework.md` §6–7.

## 1. Question and design

**Question.** With exactly the information of a first-order Markov chain, the current
event \(E_t\), does softmax regression estimate \(P(E_{t+1}=c\mid E_t)\) differently, or
better?

**Theory.** Without a penalty, F0 logistic regression *is* the Markov maximum-likelihood
estimate. With an L2 penalty, it shrinks rows toward \(\operatorname{softmax}(\mathbf{b})\)
instead of toward the uniform distribution. Expected answer: practically identical.

| Item | Setting |
|---|---|
| Taxonomy | `scenarioD15`, K = 15, canonical order (`class_mapping.json`) |
| Matches | Tier B: 304 development matches |
| Folds | experiment 2's 5 match-grouped folds (`…/tiers/tier_b/scenarioD15/folds.json`), re-verified |
| Training rows | all within-period transitions of the training matches (≈ 640k per fold), the same set the Markov chain counts |
| Evaluation rows | 701,144: events of validation matches with a 360 record |
| Features (F0) | \(\mathbf{x}_t=\operatorname{OneHot}(E_t)\), d = 15; the only input column is `symbol_id` |
| Model | scikit-learn 1.9.0 `LogisticRegression(solver="lbfgs", l1_ratio=0, C)`, multinomial softmax, intercept unpenalised, `tol = 1e-8`, `max_iter = 5000`; deterministic |
| C selection | nested leave-one-inner-fold-out on the 4 outer training folds; grid 1e-6 … 10; pooled log loss |
| Baselines | frequency baseline; first-order Markov chain with α = 1 (primary reference) |
| Supplementary | Markov with α chosen by the same nested CV; LR with C = ∞; second-order Markov (min. context 20) |
| Uncertainty | match bootstrap, 1,000 replicates, seed 7; paired differences |

**Exactness check.** The runner recomputed experiment 2's published Tier B values and
would have stopped on any mismatch. Frequency baseline, first-order and second-order
Markov log loss all match **exactly** (difference 0.0), on 701,144 = 701,144 rows. This
proves that the rows, targets and folds are experiment 2's.

## 2. Results: pooled out-of-fold, argmax decisions

| Model | Role | Log loss [95 % CI] | Brier | Accuracy | Macro-F1 | Weighted-F1 | Bal. acc. | Shot AP | Classes predicted |
|---|---|---|---|---|---|---|---|---|---|
| frequency baseline | primary | 1.6592 [1.6473, 1.6729] | 0.7237 | 0.3911 | 0.0375 | 0.2199 | 0.0667 | 0.0101 | 1 |
| first-order Markov, α = 1 | primary | 1.1520 [1.1386, 1.1665] | 0.5190 | 0.6529 | 0.2145 | 0.5588 | 0.2182 | 0.0257 | 5 |
| **logistic F0** (C = 10) | primary | **1.1519** [1.1385, 1.1664] | 0.5190 | 0.6529 | 0.2145 | 0.5588 | 0.2182 | 0.0257 | 5 |
| logistic F0, C = ∞ | suppl. | 1.1521 [1.1387, 1.1665] | 0.5190 | 0.6529 | 0.2145 | 0.5588 | 0.2182 | 0.0257 | 5 |
| first-order Markov, nested α (= 0.1) | suppl. | 1.1520 [1.1386, 1.1664] | 0.5190 | 0.6529 | 0.2145 | 0.5588 | 0.2182 | 0.0257 | 5 |
| second-order Markov | suppl. | 1.0497 [1.0376, 1.0626] | 0.4866 | 0.6689 | 0.2498 | 0.5809 | 0.2427 | 0.0525 | 8 |

- **Metric types.** Log loss, Brier and Shot AP are computed from the probability
  distributions. Accuracy, F1, balanced accuracy and "classes predicted" use argmax
  decisions.
- **Other outputs.** Fold-level values are in `metrics_by_fold.csv`. Calibration
  (classwise and top-label ECE) is in `model_comparison.csv`; it is the same for LR and
  Markov.

**Paired differences** (`paired_differences.csv`; same matches resampled):

| Comparison | Δ log loss | 95 % CI | Δ accuracy / macro-F1 |
|---|---|---|---|
| logistic F0 − Markov α = 1 | −0.000033 | [−0.000054, −0.000009] | 0 / 0 (identical decisions) |
| logistic F0 − Markov nested α | −0.000015 | [−0.000026, −0.000004] | 0 / 0 |
| logistic F0 − logistic C = ∞ | −0.000130 | [−0.000203, −0.000068] | 0 / 0 |
| Markov α = 1 − frequency baseline | −0.5073 | [−0.5115, −0.5029] | +0.262 / +0.177 |
| second-order − first-order Markov | −0.1023 | [−0.1046, −0.1001] | +0.016 / +0.035 |

**Fold level.** Logistic F0 and the Markov chain differ by at most 0.00008 in log loss in
every fold, and have identical accuracy and macro-F1 in every fold.

## 3. Interpretation

1. **[V] Theory confirmed.** The F0 logistic regression reproduces the first-order Markov
   chain:
   - *Same decisions.* The argmax decisions, confusion matrix and per-class precision,
     recall and F1 are identical.
   - *Probabilities.* On the training folds, the fitted transition tables differ from the
     empirical (MLE) table by at most 6.5·10⁻⁴ (the rare `50/50` row) and by ≤ 2·10⁻⁵ for
     frequent rows (`theory_markov_equivalence.csv`).
   - *Calibration in the large.* The intercept identity
     \(\sum_a N_a p_{a,c} = \sum_a C_{a,c}\) holds to ≤ 1.7·10⁻⁷ in every fold
     (`reproducibility_checks.json`).
2. **[V] Statistically detectable, practically irrelevant.** The log-loss advantage of
   logistic F0 over Laplace smoothing is 3·10⁻⁵ nats, 0.003 %. Its source is the form of
   shrinkage:
   - *Zero cells.* 43 of the 225 transition cells are empty in Tier B (e.g. 7 of 15 for
     `50/50`, 13 for `Dispossessed`). α = 1 gives each empty cell about 10⁻⁴–10⁻³
     probability, while mild L2 shrinkage toward \(\operatorname{softmax}(\mathbf{b})\)
     gives about 10⁻⁷–10⁻⁶.
   - *Why it helps slightly.* Those cells are almost always empty in validation too.
   - *Same direction for the Markov chain.* Nested CV likewise chose a smaller α (0.1 in
     all 5 folds) than the default.
3. **[V] The unpenalised fit is slightly worse than mild L2.** C = ∞ is worse by 1.3·10⁻⁴.
   - *No finite MLE.* With empty cells the MLE does not exist; L-BFGS stopped at
     `max_iter` (5,000) with diverging logits, and the convergence flag is correctly
     `False`.
   - *Too confident.* The resulting probabilities are too confident on cells that do
     occur in validation.
4. **[V] The choice of C does not matter once C ≥ 0.1.**
   - *Selection.* Nested CV chose C = 10 in all folds; this is the upper edge of the grid.
   - *Flatness.* The inner log loss changes by < 10⁻⁴ between C = 1 and 10.
   - *Beyond the grid.* The next point beyond the grid, C = ∞, is worse, so the optimum
     lies in [10, ∞) and is practically irrelevant.
   - *Strong penalties.* These are harmful: C = 10⁻³ gives 1.209, and C = 10⁻⁶ gives
     1.629, close to the frequency baseline (`figures/regularisation_path.png`). The
     reason is the size of the training set. The penalty is \(\lVert W\rVert^2/(2C)\)
     against a data term summed over ≈ 640k transitions, so only very small C
     regularise.
5. **[V] Shot.** No model predicts Shot under argmax: recall 0 for every first-order model
   (Shot support 7,199).
   - *Ranking.* Shot AP is 0.026, 2.5× its prevalence of 1.03 %. The current event alone
     barely ranks shots: \(P(\text{Shot}\mid\text{Carry})\) = 0.009 and
     \(P(\text{Shot}\mid\text{Pass})\) = 0.005, both below Shot's base rate.
   - *What is needed.* Spatial and history features are required. The second-order chain
     doubles Shot AP to 0.053; experiment 3 reached 0.30 with a 6 × 2 location grid and
     0.36 with a rich-feature logistic regression.
6. **[V, H] The non-trivial argmax "successes" are mostly StatsBomb bookkeeping**
   (`per_class_metrics.csv`, logistic F0):

   | Class | Precision | Recall | Source |
   |---|---|---|---|
   | `Duel` | 0.9995 | 0.378 | predicted only after `Dispossessed` (all 6,132 cases); the Dispossessed → Duel/Tackle companion (P = 1.0) |
   | `Goal Keeper` | 0.716 | 0.608 | predicted only after `Shot` (all 7,462 cases; the keeper's "Shot Faced" annotation) |
   | `50/50` | 0.507 | 0.508 | predicted only after `50/50` (paired records) |

   The argmax map is: Pass → Carry; Carry, Pressure, Duel, Block, Clearance, Miscontrol,
   Foul Committed and Goal Keeper → Pass; Ball Recovery, Dribble and Interception → Carry;
   plus the three cells above. This was verified from `predictions_f0.parquet`.

   These cells inflate macro-F1 under `scenarioD15`. This is the reason experiment 3
   recommended `actions15` (§5 of the audit document).
7. **Cost.** For F0 the logistic regression is about 50,000× more expensive to train:
   ~508 s per fold on the rows, against 0.01 s for counting. Fitting on the 225
   count-weighted cells is mathematically the same objective and takes 0.1–0.5 s. The
   row-level and cell-level fits agree to ≤ 3.7·10⁻⁶ in probability.

## 4. Quality checks

| Check | Result | File |
|---|---|---|
| Experiment 2 values reproduced exactly | yes (Δ = 0.0) | `reproducibility_checks.json` |
| Final-test matches loaded | 0 | `leakage_checks.json` |
| Train/validation match overlap | 0 in every fold | same |
| Target = next vocabulary record of the same period; independent raw-stream recomputation | true | same |
| Raw type of the target event equals the target class | true | same |
| Sequences crossing a period or match; shoot-out rows | 0; 0 | same |
| F0 reads only `symbol_id`; forbidden columns declared | none | same; `tests/test_logistic_f0.py` |
| Determinism (fold 0 refit) | max \|Δp\| = 0.0 | `reproducibility_checks.json` |
| Probabilities valid (finite, ≥ 0, rows sum to 1 ± 1e-9) | yes (runner refuses otherwise) | — |
| Convergence of the reported model | converged in all folds (3,324–3,597 iterations) | `logistic_fits.csv` |

## 5. Limitations

- **Tier B use.** Tier B was already inspected in experiments 2 and 3. These are held-out
  *development* estimates, not final-test results.
- **Taxonomy.** `scenarioD15` includes StatsBomb companion records (§3, item 6). The
  equivalence result does not depend on this; the per-class decision metrics do.
- **Edge of the grid.** C = 10 lies at the edge of the grid; see §3, item 4. It is
  harmless here, but the grid for F1+ should extend upward (e.g. to 10³).
- **Timing comparability.** Timings are for one M1 machine, running nothing else. They
  measure the implementation, not intrinsic model cost.
- **Shot.** Shot recall is 0 under argmax for all first-order models. Cost-sensitive
  rules were not evaluated here. If they are used, report them separately, as in
  experiment 2.

## 6. Files

**Tables:**
- `model_comparison.csv`, `paired_differences.csv`, `metrics_by_fold.csv`;
- `per_class_metrics.csv` (precision, recall, F1, AP per class and model);
- `confusion_matrices.csv` (rows = true class);
- `class_distribution.csv`.

**Logistic-regression details:**
- `hyperparameter_choices.csv` (inner-CV curves);
- `logistic_fits.csv` (solver, C, iterations, convergence, time);
- `logistic_coefficients.csv` (W and b per fold, canonical order);
- `regularisation_path.csv` (descriptive C path on outer folds, not used for selection);
- `theory_markov_equivalence.csv`;
- `timings.csv`.

**Checks and environment:**
- `leakage_checks.json`, `reproducibility_checks.json`;
- `environment.json` (versions, git commit, uncommitted files);
- `checksums.json`.
