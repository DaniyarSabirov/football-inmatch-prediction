# MLP-H5: a feed-forward neural network on the five-event history (experiment 6B)

**Status.** Methodology and results note, 2026-11-23.

**Task.** `actions15_360current_v1` (`docs/canonical_prediction_protocol.md`). The notation
follows `02_mathematical_framework.md` and `07_logistic_regression_h5.md`.

| Item | Location |
|---|---|
| Implementation | `src/sequence_models/mlp_h5_experiment.py`, `scripts/run_mlp_h5_experiment6b.py` |
| Lifecycle supervisor | `scripts/supervise_mlp_h5_experiment6b.py` |
| Independent audit | `scripts/verify_mlp_h5_experiment6b.py` |
| Configuration (frozen before the run) | `configs/mlp_h5_experiment6b_v1.yaml` |
| Tests | `tests/test_mlp_h5_experiment6b.py` |
| Results | `results/mlp_h5_experiment6b_actions15_360current_v1_2026-11-23/README.md` |

## 1. Research question and motivation

**Question.** Using the five most recent canonical event types, can an MLP outperform MLR-H5
and approach or exceed XGB-H5 under identical data and evaluation conditions?

**Background.** Experiment 6A found that MLR-H5 is 0.051 log loss worse than XGB-H5. MLR-H5 is
a softmax regression on the position-specific one-hot history and is additive over positions.
It is also worse than the two-position Markov chain.
- **The 6A interpretation.** Interactions between positions matter. This was an inference.
- **What this experiment tests.** It tests that inference with a model that receives exactly
  the same input but can represent interactions: a feed-forward network with ReLU hidden
  layers.
- **What is controlled.** Information, rows, folds and metrics are identical across the
  models. Only the function class and its estimation procedure differ.

## 2. Data and protocol (unchanged)

- **Rows.** 690,147 Protocol-B rows of Tier B (304 matches). Ids, order, targets, folds and
  the 5-position history are identical row by row to experiments 5, 5B and 6A. This is
  verified element-wise and against the stored manifest hashes. All comparisons are paired at
  the observation level.
- **Canonical rules.** All apply: 15 classes in canonical order, K1–K3 companions removed (K4
  inactive), StatsBomb `index` order, period 5 excluded, and no history or target across
  periods or matches. Rows at period starts are kept and receive `<START>`.
- **Final test.** The 83 final-test matches were never loaded.

## 3. Model

**Input and target.**
\[
X_t = (e_{t-4}, e_{t-3}, e_{t-2}, e_{t-1}, e_t), \qquad Y_t = e_{t+1},
\]
- The history is the XGB-H5 / MLR-H5 history (`event_history.HistoryRows`, not
  re-implemented).
- \(\phi(X_t) \in \{0,1\}^{79}\) is the position-specific one-hot vector of experiment 6A:
  lags 4..1 have 16 categories (15 classes + `<START>`), \(e_t\) has 15.
- It was verified to equal 6A's encoding on all rows.
- No interaction features are constructed by hand.

**Network** (L hidden layers, ReLU, dropout \(p\) on hidden activations only):
\[
h_1 = \mathrm{ReLU}(W_1\phi(X_t) + b_1),\qquad
h_\ell = \mathrm{ReLU}(W_\ell\,\mathrm{drop}_p(h_{\ell-1}) + b_\ell),\qquad
z = W_{L+1}\,\mathrm{drop}_p(h_L) + b_{L+1} \in \mathbb{R}^{15},
\]
\[
P(Y_t = k \mid X_t) = \mathrm{softmax}(z)_k = \frac{\exp z_k}{\sum_{j=1}^{15}\exp z_j}.
\]
- **First layer.** \(\phi\) has exactly five ones, so \(W_1\phi(X_t) = \sum_{j=1}^{5} W_1[:,c_j(X_t)]\).
  It is implemented as this sum (`nn.Embedding`), an exact identity tested against a dense
  linear layer.
- **Initialisation.** As `nn.Linear(79, H_1)`: \(U(\pm 1/\sqrt{79})\). The other layers use
  PyTorch's default initialisation.
- **Dropout.** It is active only in training mode; predictions use evaluation mode (tested).

**Relation to MLR-H5.** With no hidden layer the model reduces to MLR-H5 without its L2
penalty. Through the hidden ReLUs, the effect of \(e_{t-1}\) on the logits can depend on
\(e_t\) and on the other positions. Section 9 shows this on a synthetic pure-interaction
process.

## 4. Loss, optimisation, regularisation

| Component | Choice |
|---|---|
| Data loss | \(L = -\frac{1}{N}\sum_i \log P(Y_i = y_i \mid X_i)\), computed by `F.cross_entropy` on logits (log-softmax inside; no softmax before the loss). No class weights, resampling or decision thresholds |
| Optimiser | AdamW, \(\beta = (0.9, 0.999)\), \(\epsilon = 10^{-8}\), mini-batches of 512 rows in a reproducible per-epoch permutation |
| Weight decay | Decoupled from the loss: each step applies \(w \leftarrow w - \eta\lambda w\) to weight matrices (biases excluded), followed by the Adam update. It is not an L2 term added to \(L\), so \(\lambda\) is not comparable to 6A's C |
| Dropout | \(p \in \{0, 0.1\}\) on hidden activations |
| Early stopping | See §5. The returned network is the state of the best epoch on the stopping subset; the initial state (epoch 0) is the incumbent |
| Precision | Training in float32. Probabilities are the float64 softmax of the float32 logits, so every row sums to 1 within 1e-15 |

**Hyperparameter search.** It was frozen before the run: 8 prespecified configurations, a
structured subset of the proposed grid.

| | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| hidden | 32 | 64 | 64-32 | 128-64 | 64-32 | 128-64 | 128-64 | 64 |
| weight decay | 0 | 0 | 0 | 0 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| dropout | 0 | 0 | 0 | 0 | 0.1 | 0.1 | 0 | 0.1 |
| learning rate | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 3e-4 | 3e-4 |

The search space was not expanded after the run.

## 5. Nested cross-validation and early stopping

For each of the 5 outer folds:
1. **Inner CV.** Leave-one-inner-fold-out over the other 4 folds. For every configuration and
   held inner fold \(h\):
   - **Data.** The training matches are the other 3 inner folds. A seeded 10 % of those
     matches is split off as the **stopping subset**.
   - **Training.** The network is trained on the remaining 90 % and stopped on the stopping
     subset (patience 5, at most 60 epochs).
   - **Scoring.** Only then is it scored on fold \(h\).
2. **Selection.** The configuration with the smallest pooled log loss on the held inner folds
   is selected. The pooled loss is the sum of NLL over the 4 held folds divided by their rows;
   ties go to the earlier configuration.
3. **Refit.** The selected configuration is refitted on the outer-training matches minus a
   seeded 10 % stopping subset, with the same stopping rule. The refit predicts the untouched
   outer-validation rows once.

**Why the held fold is not used for stopping.** A held inner fold that chose the stopping
epoch and also scored the configuration would be used for two selections. Here it does only
the scoring, so the selection score is not biased by the epoch choice.
- **Outer training duration.** It is determined from training matches only; outer-validation
  labels are never used.
- **Data-access audit.** `data_access_audit.csv` records every data set; none of the
  training, stopping or inner sets touches the outer-validation or final-test matches.
- **Adversarial test.** It changes all outer-validation labels of a fold. It confirms that
  the fold's selection, weights and predictions stay bit-identical, while the other folds,
  for which those labels are training data, change.

**Cost of this choice.** The refit sees 90 % of the outer-training matches, about 219 of 243.

## 6. Reproducibility controls

- **Device.** The device was chosen by benchmark: CPU took 0.6 s per epoch, MPS 1.7–4.3 s.
  MPS determinism was not tested; CPU determinism was verified (see below). Settings: CPU,
  1 thread, `torch.use_deterministic_algorithms(True)`.
  The device was never switched during the run.
- **Seeds.** Every random quantity derives from `numpy.random.SeedSequence`.
  - Each fit uses (base 20261123, outer fold, inner fold, configuration, replicate).
  - Each epoch is re-seeded from (fit seed, stream, epoch), with separate streams for the
    permutation, dropout, NumPy and Python RNGs.
  - The stopping subsets use (base, outer, inner).
  - There are no DataLoader workers; batches are index slices of an in-memory tensor.
- **Checkpoints.** One checkpoint is written per epoch boundary. It contains the model,
  optimiser state, epoch, best state and metric, the early-stopping counter, the RNG states
  and the identifiers (configuration, dataset, folds, rows, features, model sources, software,
  device, threads).
  - Completed fits and folds are committed atomically.
  - A resume refuses different identifiers, different hyperparameters or corrupted files.
- **Tests.**
  - An interrupted run resumed from its checkpoint gives bit-identical weights and loss
    curves to an uninterrupted run, with dropout active.
  - The optimiser and RNG states are restored exactly.
- **Levels of reproducibility.**
  - **Protocol:** configuration and seeds are fixed.
  - **Saved predictions:** the stored probabilities are recomputed from the stored weights
    by an independent NumPy forward pass (max \(|\Delta p|\) = 5.5e-7, float32 vs. float64).
  - **Bitwise training:** a fresh retraining of fold 0's refit gave bit-identical weights and
    predictions on this machine. This is not claimed across machines, thread counts or
    PyTorch versions.

## 7. Training behaviour

| Item | Value |
|---|---|
| Fits | 175 (160 inner, 5 primary refits, 10 robustness refits), 5,363 epochs |
| Fitting time | 3,782 s; the whole run took 4,017 s wall time |
| Inner fits | Stopped after 13–60 epochs (median 29) |
| Best epochs | Median 24 for inner fits; 8–38 for the primary refits |
| Max-epoch cap | It bound in 14 inner fits, all of configuration H (lr 3e-4, one layer of 64). H's inner scores may therefore be slightly pessimistic; H was not selected |
| Selection | F ([128, 64], weight decay 1e-4, dropout 0.1, lr 1e-3) in folds 0–3; E ([64, 32], same regularisation) in fold 4, where E and F differ by 2e-6 |
| Inner-CV ranking | Larger networks and regularisation were better. Fold 0: F < E < G < D < H < C < B < A; the other folds are similar (`inner_cv_scores.csv`) |
| Stability | No warning was raised; every batch and epoch passed the finiteness checks of loss, gradients and parameters |
| Memory | Peak RSS ≈ 1.4 GB (smoke test) on an 8 GB MacBook Air |

## 8. Results

**Pooled out-of-fold results on 690,147 rows from 304 matches.**

| Model | Log loss | Accuracy | Macro-F1 | Weighted-F1 | Shot AP | Shot recall |
|---|---:|---:|---:|---:|---:|---:|
| Logistic F0 | 1.15850 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 / 7,212 |
| Markov 2 | 1.06807 | 0.6659 | 0.1204 | 0.5731 | 0.0527 | 0 / 7,212 |
| XGB-H2 | 1.06761 | 0.6659 | 0.1208 | 0.5731 | 0.0529 | 0 / 7,212 |
| XGB-H5 | **1.05185** | 0.6668 | **0.1325** | **0.5778** | **0.0769** | 12 / 7,212 |
| MLR-H5 | 1.10297 | 0.6491 | 0.1219 | 0.5579 | 0.0585 | 1 / 7,212 |
| **MLP-H5** | 1.05267 | **0.6668** | 0.1302 | 0.5762 | 0.0757 | **13 / 7,212** |

**Paired log-loss differences.** Here \(d_i = -\log\hat p^{\text{MLP}}_{i,y_i} + \log\hat p^{\text{ref}}_{i,y_i}\);
a negative value means MLP-H5 is better. The match-level cluster bootstrap uses 1,000
replicates with seed 7, the same resampled matches as all canonical experiments.

| Reference | \(\bar d\) | rel. | 95 % CI | Folds MLP-H5 better |
|---|---:|---:|---|---:|
| XGB-H5 | +0.00082 | +0.08 % | [+0.00052, +0.00114] | 0 / 5 |
| MLR-H5 | −0.05030 | −4.6 % | [−0.0517, −0.0489] | 5 / 5 |
| Markov 2 | −0.01540 | −1.4 % | [−0.0162, −0.0147] | 5 / 5 |

**Per fold.** Per-fold MLP-H5 log loss is 1.0492–1.0561. The per-fold gap to XGB-H5 is
+0.0002 to +0.0020.

**Secondary metrics against XGB-H5.**
- **Accuracy.** The difference is −0.00004, CI [−0.00033, +0.00027]: no detectable
  difference.
- **Macro-F1.** It is lower by 0.0024, CI [−0.0032, −0.0015].
- **Shot AP.** It is lower by 0.0013; the CI [−0.0033, +0.0006] includes 0.

**Seed robustness.** It was prespecified: replicate 0 is primary and never chosen by result.
The three refit seeds give pooled log loss 1.05267, 1.05215 and 1.05258.
- All three are worse than XGB-H5, by +0.0003 to +0.0008.
- All three are better than MLR-H5, by about −0.050.
- **Size of the seed effect.** The seed-to-seed spread (0.0005) is of the same order as the
  gap to XGB-H5. The primary CI covers row and match sampling for fixed fitted models.
- **Uncertainty the CIs do not include.** None of the CIs include:
  - optimisation randomness;
  - refitting;
  - the overlap of training sets between cross-validation folds;
  - hyperparameter selection;
  - the choice of competitions and seasons in the dataset.

  They describe this development population, not generalisation to other competitions.

**Rare classes.**
- **Supports.** The rarest classes have 559 (50/50) to 7,212 (Shot) evaluation rows.
- **Argmax decisions.** They almost never predict these classes, for any model. MLP-H5
  predicts Shot 51 times, 13 of them correct.
- **AP.** Per-class AP of MLP-H5 is close to XGB-H5's for every class. It is clearly above
  MLR-H5's for Ball Recovery, Carry, Clearance, Foul Committed and Shot.

## 9. Interpretation

**MLP-H5 vs. MLR-H5.** Nonlinear modelling of the same history improves log loss
substantially and consistently, by 0.050 with all five folds agreeing.
- **Where the gain is.** Rows with current event Pressure account for 0.029 of the 0.050. On
  those rows MLR-H5 was weakest in experiment 6A (−0.234 per row here).
- **What it supports.** This supports, but does not prove, the 6A interpretation: the
  additive model's deficit comes from interactions between history positions that a model
  with hidden layers can represent.
- **Synthetic illustration.** On a synthetic pure interaction \(y = (e_{t-1}+e_t) \bmod 15\),
  MLR stays at \(\approx \log 15\) while the MLP fits it (unit tests).

**MLP-H5 vs. XGB-H5.** The MLP comes very close to XGB-H5 but does not reach it.
- **The gap.** +0.0008 log loss, with a CI excluding 0. XGB-H5 is better in every fold and
  for every robustness seed.
- **Its size.** The gap is small in absolute terms: 0.08 %, about 1/60 of the MLR-H5 gap.
  It is of the same order as the variation between training seeds.
- **Reading.** The two nonlinear model families are close in practical terms on this input,
  with a small but consistent edge for gradient-boosted trees.
- **No equivalence claim.** No equivalence margin or equivalence test was specified in
  advance. This is therefore not a claim of formal equivalence.
- **Scope of the CI.** The CI covers only the resampling of evaluation matches for fixed
  predictions.

**Not established by these numbers.**
- That the MLP architecture is inferior in general.
- That a different search, longer training or other architectures could not close the gap.

**Differences in estimation that remain.**

**Training population: a material, documented limitation.** The comparison is valid as a
comparison of two complete training *procedures* on identical evaluation rows. It is **not**
controlled for the quantity of training data.
- **XGB-H5.** Its outer refit used all outer-training rows. Its number of boosting rounds came
  from inner CV.
- **MLP-H5.** Its outer refit updated its parameters on about 90 % of the outer-training
  rows. The remaining 24 matches per fold were used only for early stopping.

Per-fold numbers (`results/audit_experiments6a_6b_2026-11-24/training_population_6b_vs_xgb_h5.csv`):

| Fold | Outer-training matches / rows | MLP parameter-update matches / rows | MLP stopping matches / rows | XGB-H5 training rows |
|---:|---|---|---|---:|
| 0 | 244 / 552,724 | 220 / 500,172 (90.5 %) | 24 / 52,552 | 552,724 |
| 1 | 243 / 552,322 | 219 / 497,462 (90.1 %) | 24 / 54,860 | 552,322 |
| 2 | 243 / 551,095 | 219 / 496,378 (90.1 %) | 24 / 54,717 | 551,095 |
| 3 | 243 / 552,457 | 219 / 495,863 (89.8 %) | 24 / 56,594 | 552,457 |
| 4 | 243 / 551,990 | 219 / 498,034 (90.2 %) | 24 / 53,956 | 551,990 |

- **What this means for the result.** Whether the +0.0008 gap would persist with a
  full-population MLP refit is **unknown**.
- **Possible follow-up.** For example, refit on all outer-training matches for a number of
  epochs fixed from the inner fits. This would be a separate, not yet authorised controlled
  experiment.

| | XGB-H5 | MLP-H5 |
|---|---|---|
| Training data | All outer-training rows (count-weighted cells) | 90 % of them (10 % stopping subset) |
| Regularisation | Tree regularisation; boosting rounds from inner-CV curves | Weight decay / dropout; epochs from early stopping |
| Search | 20 random configurations | 8 prespecified configurations |

Any of these can contribute to a gap of this size.

**Order of history positions.** Held-out permutation importance confirms the decay over lags
seen in 6A: \(e_t\) +3.96, \(e_{t-1}\) +0.47, \(e_{t-2}\) +0.048, \(e_{t-3}\) +0.007,
\(e_{t-4}\) +0.003. The MLP relies more strongly on \(e_{t-1}\) than MLR-H5 does (+0.47 vs.
+0.16), as expected when its effect is conditioned on \(e_t\).
- **Unrealistic histories.** Each permutation shuffles one position across the rows of a fold,
  so the permuted histories can contain combinations of events that rarely or never occur.
- **What the importances measure.** They describe how much the fitted model relies on a
  position under such artificial inputs. They are neither additive nor an attribution of
  information.

## 10. Limitations

- **Small search space.** The search covered 8 configurations; configuration H was capped at
  60 epochs in 14 inner fits.
- **Optimisation randomness not in the CI.** It is reported separately (3 seeds), not
  integrated into the CI.
- **Bitwise reproducibility is local.** It was verified only for this machine, thread count
  and software.
- **Training data and regularisation differ from XGB-H5** (see §9).
- **Argmax metrics for rare classes** are near zero for every model; no cost-sensitive
  decisions were used, by design.
- **Mechanism.** The decompositions by stratum are descriptive; mechanisms are inferred, not
  proven.
- **Development data only.** The results are out-of-fold estimates on the development
  matches; the final test is untouched.

## 11. Reproduction

```bash
python scripts/run_mlp_h5_experiment6b.py --smoke
python scripts/supervise_mlp_h5_experiment6b.py --detach      # runner + verifier; caffeinate -d -i -w <supervisor>
python scripts/supervise_mlp_h5_experiment6b.py --status
python -m pytest tests/test_mlp_h5_experiment6b.py
```

**Run environment.** Python 3.11.1, torch 2.15.0, NumPy 2.4.6, scikit-learn 1.9.0.

**Source provenance.**
- **At run time.** The run started from commit `a2343e9` with the experiment files not yet
  committed, so `a2343e9` alone does not identify the executed code.
- **Identification.** The executed sources are identified by the SHA-256 of every source file
  in `environment.json` (`source_files_sha256`).
- **Verification.** The final audit (`results/audit_experiments6a_6b_2026-11-24/`) verified
  that the committed files are byte-identical to the executed ones.

**Seeds.** Training base seed 20261123, folds 20261027, bootstrap 7, permutation 11.

## 12. Integrity verdict and conclusion

**Integrity.**
- **Run audit.** 51 checks (48 PASS, 3 INFO).
- **Independent verifier.** 38 checks, PASS.
- **Exit codes.** The experiment and the audit both exited 0, as recorded by the supervisor.
- **Verdict.** **PASS WITH LIMITATIONS**, the limitations being those of §10.

**Conclusion.** Given exactly the same five-event history:
- **vs. MLR-H5.** A small ReLU MLP is clearly better than the additive logistic regression,
  by −0.050 log loss in all folds.
- **vs. XGB-H5.** It essentially matches XGB-H5 but remains slightly and consistently worse,
  by +0.0008 (CI [+0.0005, +0.0011]).
- **Implication.** Nonlinear modelling of the event history accounts for most of the
  improvement over the additive model. Between boosted trees and this feed-forward network,
  the difference on this input is small, with a consistent edge for XGB-H5.
- **Caveat on that edge.** The edge was obtained with the MLP trained on about 90 % of the
  outer-training rows (§9). It is not a training-data-controlled comparison, and it is not a
  claim of formal equivalence.
