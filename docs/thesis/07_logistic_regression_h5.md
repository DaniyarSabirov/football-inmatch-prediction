# MLR-H5: multinomial logistic regression with a five-event history (experiment 6A)

Status: methodology and results note, 2026-11-23. Task: `actions15_360current_v1`
(`docs/canonical_prediction_protocol.md`). Notation as in `02_mathematical_framework.md` and
`05_xgboost_gradient_boosting.md`.

| Item | Location |
|---|---|
| Implementation | `src/sequence_models/logistic_h5_experiment.py`, `scripts/run_logistic_h5_experiment6a.py` |
| Independent audit | `scripts/verify_logistic_h5_experiment6a.py` |
| Configuration | `configs/logistic_h5_experiment6a_v1.yaml` |
| Tests | `tests/test_logistic_h5_experiment6a.py` |
| Results | `results/logistic_h5_experiment6a_actions15_360current_v1_2026-11-23/README.md` |

## 1. Research question and motivation

**Question.** Does a multinomial logistic regression that uses the five most recent event
types reach predictive performance comparable to XGB-H5, when both models receive exactly
the same event-history information?

**Why it matters.** Experiment 5B showed that ~97 % of XGB-H5's gain over Markov 2 comes
from the longer history, not from the estimator. That leaves open whether a *linear* model
can exploit the same history, which would make it the simpler, interpretable choice. The two
models differ only in the function class:

| | Information | Function of the history |
|---|---|---|
| XGB-H5 | \((e_{t-4},\dots,e_t)\) | trees: may combine several positions (interactions) |
| MLR-H5 | \((e_{t-4},\dots,e_t)\) | additive over positions on the logit scale |

The primary metric is the multiclass log loss; everything else is secondary. A negative
result was admissible in advance.

## 2. Data and protocol (unchanged)

- **Rows.** The 690,147 Protocol-B rows of Tier B (304 matches): every canonical observation
  \(e_t\) that has its own usable 360 snapshot and a next observation \(e_{t+1}\) in the same
  (match, period). The target needs no snapshot.
- **Identity with experiments 5 and 5B.** Event ids, order, targets and fold assignments are
  identical row by row. This is verified element-wise and by the hashes in experiment 5's
  manifest, so every comparison is paired at the observation level.
- **Canonical rules.** All canonical rules apply unchanged:
  - the 15 classes in canonical order;
  - companions K1–K3 removed (K4 inactive);
  - StatsBomb `index` order;
  - period 5 excluded;
  - no targets across periods or matches.
- **Folds and final test.** The 5 match-grouped outer folds of experiment 2. The 83
  final-test matches were never loaded.

## 3. Mathematical formulation

Let \(e_1,\dots,e_T\) be the canonical sequence of one (match, period). The target is
\(Y_t = e_{t+1}\) and the input is

\[
X_t = (e_{t-4}, e_{t-3}, e_{t-2}, e_{t-1}, e_t), \qquad e_{t-\ell} := \texttt{<START>} \text{ if } t-\ell < 1 .
\]

The model is

\[
P(Y_t = k \mid X_t) = \frac{\exp\!\big(w_k^\top \phi(X_t) + b_k\big)}{\sum_{j=1}^{15} \exp\!\big(w_j^\top \phi(X_t) + b_j\big)},
\qquad k = 1,\dots,15 .
\]

**Representation \(\phi\) (position-specific one-hot).**
\[
\phi(X_t) = \big(\mathbf{1}[e_{t-4}=c]\big)_{c \in \mathcal{C}^+} \oplus \dots \oplus
\big(\mathbf{1}[e_{t-1}=c]\big)_{c \in \mathcal{C}^+} \oplus \big(\mathbf{1}[e_t=c]\big)_{c \in \mathcal{C}},
\]
where \(\mathcal{C}\) holds the 15 classes and \(\mathcal{C}^+ = \mathcal{C} \cup \{\texttt{<START>}\}\).
This gives \(4 \cdot 16 + 15 = 79\) binary columns, with exactly 5 active per row.
- **Position-specific.** "Pass at lag 0" and "Pass at lag 1" are different columns. No
  bag-of-events or frequency representation is used.
- **Same encoding as XGB-H5.** Column order and names equal XGB-H5's
  (`event_history.encode`).
- **`<START>`.** It marks history positions before the period start. History is never
  borrowed from a previous period or match, and early rows are kept, not dropped.
- **Additive logit.** Because \(\phi\) is a direct sum of blocks, the logit is additive over
  positions:
  \[
  z_k(X_t) = b_k + \sum_{\ell=0}^{4} \beta_{k,\ell}(e_{t-\ell}),
  \]
  with \(\beta_{k,\ell}(c) = w_{k,(\ell,c)}\).

**No interaction terms.** The log-odds contribution of \(e_{t-1}\) is the same whatever
\(e_t\) is. MLR-H5 is therefore **not** equivalent to a fifth-order Markov chain, which has
one free distribution per 5-context (up to \(16^4 \cdot 15\) contexts).
- **Special case.** It is not even equivalent to the second-order chain on
  \((e_{t-1}, e_t)\), except in the special case of no interaction.
- **F0.** Only for one position (F0) does the one-hot softmax coincide with the first-order
  Markov chain (`02_mathematical_framework.md` §6).

**Objective (scikit-learn ≥ 1.8 parameterisation).**

\[
\min_{W, b}\; \frac{1}{N}\sum_{i=1}^{N} -\log P(Y_i = y_i \mid X_i) \;+\; \frac{1}{2CN}\,\lVert W\rVert_F^2,
\]

- The intercepts \(b\) are unpenalised; there is no class weighting.
- \(C\) is the inverse regularisation strength.
- The penalty does not grow with \(N\), so a fixed C regularises less on more data.

**Fitting on cells.** All inputs are discrete, so the training rows are collapsed to unique
(features, target) cells weighted by their counts.
- **Same objective.** The weighted data term and the penalty (Σ weights = N) are exactly the
  row-level objective.
- **Same optimum.** The problem is strictly convex in \(W\), so the optimum is the same. An
  unaggregated refit of fold 0 (552,724 rows vs. 36,507 cells) gives max \(|\Delta p|\) =
  2.1e-11.

**Solver.** `newton-cholesky` minimises the same symmetric objective as lbfgs. This was
checked in scikit-learn 1.9.0's `_glm/_newton_solver.py`: all K weight vectors are
penalised, and only the unpenalised intercept redundancy is pinned and then re-centred.
- **Why it was chosen.** It was chosen before the full run, from optimisation evidence on
  outer fold 0's inner data. lbfgs at tol 1e-8 needed 13,205 iterations at C = 100, because
  rare one-hot columns make the problem ill-conditioned. Newton-Cholesky reached the same
  optimum in 17–58 iterations.
- **Every fit checked.** After each of the 105 fits, the max \(|\nabla|\) of the objective
  is recomputed independently in NumPy. The run fails above 1e-6; the observed maximum is
  7.5e-11. No fit raised a warning.

**Identifiability.** Two directions leave the predictions unchanged:
- within each position block the one-hot columns sum to 1, so a constant added to a block
  can be moved into the intercept;
- the softmax is invariant to adding a common vector to all classes.

The unique L2 optimum therefore satisfies \(\sum_k w_{k,f} = 0\) and
\(\sum_{c} w_{k,(\ell,c)} = 0\) (KKT conditions of the unpenalised intercepts; verified in
`identifiability.csv`). The coefficients are thus **deviation-coded**.
\(w_{k,(\ell,c)} - w_{k,(\ell,c')}\) is the change of the centred log-probability
\(\log P_k - \frac{1}{15}\sum_j \log P_j\) when \(e_{t-\ell}\) changes from \(c'\) to \(c\),
with the other positions fixed. The change of \(\log(P_k/P_j)\) is the double contrast
\((w_{k,(\ell,c)} - w_{k,(\ell,c')}) - (w_{j,(\ell,c)} - w_{j,(\ell,c')})\).

**Limits of interpretation.**
- A coefficient is not a change in probability.
- Under L2 shrinkage, coefficients of rare categories are pulled towards 0.
- Coefficients describe the fitted model, not causal effects of football actions.

## 4. Hyperparameter tuning and nested cross-validation

For each outer fold \(f\):
1. **Inner CV.** Leave-one-inner-fold-out over the other 4 folds (4 inner fits per C). The
   pooled inner-validation log loss is the sum of NLL over the 4 held-out inner folds,
   divided by their row count.
2. **Choice of C.** C is chosen from {0.01, 0.1, 1, 10, 100}. Ties go to the smaller C.
   - **Edge rule.** It was pre-declared and uses training data only: an argmin at a grid
     edge would add 1000 / 10⁴ (or 10⁻³ / 10⁻⁴).
   - **Never triggered.** The rule was not triggered in any fold.
3. **Refit and prediction.** The model is refitted on all outer-training rows with that C
   and predicts the untouched outer-validation rows once. The full 15-class distribution is
   saved.

**Isolation.** The outer-validation matches never enter an inner set or the refit
(`data_access_audit.csv`). A test changes all outer-validation labels of fold 0. It
verifies that fold 0's C, inner scores, coefficients and predictions are bit-identical.

**Checkpoints.** Each completed fold is committed atomically: arrays first, then a JSON
carrying their hash. A resume is refused unless the configuration, dataset, folds, rows,
feature matrix, model sources and library versions are identical.

**Selected C.** C = 10 in folds 0 and 4, C = 1 in folds 1–3. The inner-CV curve is flat in
between: \(|\Delta|\) ≤ 2.2e-5 between C = 1 and C = 10.

## 5. Evaluation

- **Same definitions.** The metric definitions and code are those of the canonical
  experiments (`xgboost_experiment.evaluate_models`):
  - log loss, multiclass Brier score;
  - argmax accuracy, macro-/weighted-F1, per-class precision/recall/F1/support,
    confusion matrices;
  - Shot AP and Shot recall/precision.
- **Pooled and per-fold metrics.** Pooled metrics are computed over all out-of-fold
  predictions. Fold metrics are reported separately, never averaged into the pooled value.
- **Uncertainty.** A paired match-level (cluster) bootstrap, 1,000 replicates, seed 7, with
  the same resampled matches as the earlier experiments.
  - Each metric is recomputed on every bootstrap sample, so non-linear metrics (macro-F1,
    AP) are handled correctly.
  - For log loss the paired difference is the mean of
    \(d_i = -\log \hat p^{\text{MLR-H5}}_{i,y_i} + \log \hat p^{\text{ref}}_{i,y_i}\);
    negative means MLR-H5 is better.
- **Scope of the confidence intervals.** The bootstrap resamples matches for **fixed**
  out-of-fold predictions. It quantifies the uncertainty from which matches are evaluated.
  It does not include the variability from:
  - refitting the models;
  - the overlap of training sets between cross-validation folds;
  - hyperparameter selection;
  - the choice of competitions and seasons in the dataset.

  The intervals therefore describe this development population, not generalisation to
  other competitions. A statistically clear difference is not automatically practically
  relevant.

## 6. Results

Pooled over 690,147 rows (full tables: the result README):

| Model | Log loss | Accuracy | Macro-F1 | Weighted-F1 | Shot AP | Shot recall |
|---|---:|---:|---:|---:|---:|---:|
| Logistic F0 | 1.15850 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0.0000 |
| Markov 2 | 1.06807 | 0.6659 | 0.1204 | 0.5731 | 0.0527 | 0.0000 |
| XGB-H2 | 1.06761 | 0.6659 | 0.1208 | 0.5731 | 0.0529 | 0.0000 |
| XGB-H5 | 1.05185 | 0.6668 | 0.1325 | 0.5778 | 0.0769 | 0.0017 |
| **MLR-H5** | **1.10297** | **0.6491** | **0.1219** | **0.5579** | **0.0585** | **0.0001** |

**Paired log-loss differences.** In the table, d̄ is the mean per-row NLL difference,
MLR-H5 − reference.

| Comparison | d̄ | 95 % CI | Folds MLR-H5 better |
|---|---:|---|---:|
| MLR-H5 − XGB-H5 | +0.0511 (+4.9 %) | [+0.0498, +0.0525] | 0 / 5 |
| MLR-H5 − Markov 2 | +0.0349 (+3.3 %) | [+0.0338, +0.0360] | 0 / 5 |
| MLR-H5 − Logistic F0 | −0.0555 (−4.8 %) | [−0.0571, −0.0541] | 5 / 5 |

Per-fold MLR-H5 log loss lies between 1.1003 and 1.1052. The fold-wise gap to XGB-H5 lies
between 0.0506 and 0.0516.

**Rare classes.**
- **Supports.** In the evaluation rows: 50/50 559; Goal Keeper 3,149; Interception 6,210;
  Dispossessed 6,280; Foul Committed 6,407; Duel 6,851; Shot 7,212.
- **Argmax recall.** For these classes it is 0 or below 0.02 for all models. For example,
  MLR-H5 predicts Shot 6 times, 1 of them correct. XGB-H5 predicts it 30 times, 12 of them
  correct.
- **Use the probabilistic metrics.** Small rounded values of recall/F1 must be read with
  the supports. AP and log loss are the informative metrics for these classes.
- **Per-class AP.** MLR-H5's AP lies between Markov 2's and XGB-H5's for most classes.

## 7. Interpretation

**Primary comparison.** With identical information, MLR-H5 is clearly and consistently
worse than XGB-H5. The gap of 0.051 is about three times XGB-H5's advantage over Markov 2
(0.016), and the CI excludes zero by a wide margin.

**Comparison with Markov 2.** The five-position additive model is also worse than the
two-position Markov chain. Markov 2 models the joint context \((e_{t-1}, e_t)\) without
restriction.
- **What this suggests (inference).** For this task, interactions between positions carry
  more predictive information than additional, additively combined history positions.
- **Not proof of mechanism.** This is a reading of the comparison, not a proof of the
  mechanism.

**Where the gap is concentrated (descriptive).** Rows whose current event is Pressure
(12 % of rows) account for 0.029 of the 0.051 gap to XGB-H5 and 0.026 of the 0.035 gap to
Markov 2.
- **A plausible explanation (not tested here).** After a Pressure, the next event depends
  jointly on the pressure and on the action under pressure, for example the preceding Pass
  or Carry. The additive model must use the same \(e_{t-1}\) effect regardless of \(e_t\).

**Comparison with logistic F0.** MLR-H5 is clearly better than logistic F0, by 0.056: the
additional positions do carry information that a linear model can use. Held-out
permutation importance decreases with lag:
- \(e_t\): +2.25 log loss;
- \(e_{t-1}\): +0.160;
- \(e_{t-2}\): +0.021;
- \(e_{t-3}\): +0.008;
- \(e_{t-4}\): +0.003.

**Reading the permutation importances.**
- **How they are computed.** Each permutation shuffles one position across the rows of a
  fold.
- **Unrealistic histories.** The permuted histories can contain combinations of events that
  rarely or never occur in real sequences.
- **What they measure.** The importances measure how much the fitted model relies on a
  position under such artificial inputs. They are not an attribution of predictive
  information.
- **Not additive.** The position values are not additive, and the joint permutation of
  several positions is reported separately.

The mean coefficient range by position is 6.8, 2.7, 1.5, 1.1 and 1.0 for lags 0–4.

**Coefficients (deviation-coded, mean over the 5 fold models; features with ≥ 200 training
rows).** These are descriptions of the model, not causal statements.
- **Empirical check of the short descriptions.** The descriptions in the table were checked
  against the empirical next-event rates of the 690,147 rows:

  | Transition | Rate |
  |---|---:|
  | P(next = Pass \| Foul Committed) | 0.85 |
  | P(next = Pass \| Carry) | 0.73 |
  | P(next = Carry \| Ball Recovery) | 0.85 |
  | P(next = Carry \| Carry) | 0.000 |
  | P(next = Shot \| Foul Committed) | 0.054 (base rate 0.010) |

- **Limits of the check.** These marginal rates describe the data; they do not establish
  why the coefficients take these values.

| Target | Largest positive | Largest negative |
|---|---|---|
| **Pass** | \(e_t\)=Foul Committed (+2.41) and \(e_t\)=Carry (+1.79): a restart and a carry are usually followed by a pass | \(e_t\)=Pass (−1.42) |
| **Carry** | \(e_t\)=Ball Recovery (+5.09), \(e_t\)=Dribble (+4.40), \(e_t\)=Pass (+3.53): a carry follows receiving or winning the ball | \(e_t\)=Carry (−6.64): consecutive carries are rare in this event model |
| **Shot** | \(e_t\)=Foul Committed (+4.40), \(e_t\)=Duel (+3.72), \(e_t\)=Ball Recovery (+3.17) | \(e_t\)=Miscontrol (−2.79), \(e_t\)=Goal Keeper (−2.58) |
| **Ball Recovery** | \(e_t\)=Dispossessed (+2.08), \(e_t\)=Miscontrol (+1.94) | \(e_t\)=Carry (−4.22), \(e_{t-1}\)=Carry (−1.65) |
| **Pressure** | \(e_t\)=Carry (+1.94), \(e_t\)=Miscontrol (+1.43) | \(e_t\)=Duel (−1.59), \(e_t\)=Pressure (−1.39) |
| **Duel** | \(e_t\)=Pass (+2.41), \(e_t\)=Clearance (+2.21), \(e_{t-1}\)=Shot (+1.23) | \(e_t\)=Carry (−3.60), \(e_t\)=Pressure (−2.64) |

- **Recent vs. older positions.** For every class the strongest effects are at \(e_t\); older
  positions mostly adjust these. `<START>` columns appear among the larger coefficients for
  some classes. They act as period-start indicators and rest on few rows: 568 evaluation rows have
  `<START>` at lag 1, and 2,376 at lag 4.
- **Stability.** Among features with ≥ 200 training rows in every fold, 77 % have the same
  sign in all folds.

## 8. Computational cost

- **Fits.** 105 Newton-Cholesky fits on 31–37 thousand weighted cells: 10–39 iterations and
  4–18 s each.
- **Nested selection.** About 145 s per outer fold.
- **Whole run.** The run took 1,057 s wall time, including feature checks, the raw-JSON
  verification, an unaggregated row-level refit (113 s) and a 1,000-replicate bootstrap
  for 8 models.
- **Memory.** The smoke test measured a peak RSS of about 1.0 GB on an 8 GB MacBook Air.
  The sparse inner-train matrix takes 27 MB.

## 9. Verification

- **Before training.** The independent raw-JSON rebuild covered 707 rows in 11 strata:
  - random rows;
  - t = 0, t = 1, t = 2–3;
  - period end;
  - first and last row of a match;
  - a K1–K3 companion inside the history window;
  - a repeated timestamp;
  - excluded raw records between \(e_{t-1}\) and \(e_t\);
  - rare classes.

  `raw_history_check` shares no code with the table builder or `event_history`. For each
  row it checked the history, the one-hot column names, the `<START>` padding, the target
  and target event id, and that every history record precedes the target.
- **The run's own audit.** 40 checks: 39 PASS, 1 INFO. They include:
  - row-level identity with experiments 5/5B;
  - prior models' metrics recomputed equal to the stored tables (≤ 3.1e-13);
  - NumPy softmax of the saved \(W, b\) equal to scikit-learn's probabilities;
  - fold-0 refit reproducibility;
  - historical results unchanged.
- **Independent post-run verifier.** 31 checks, PASS, exit code 0.
  - The history is rebuilt with pandas `shift`, and the probabilities are recomputed from
    the stored coefficients (max difference 0).
  - The metrics are recomputed with scikit-learn.
  - An independent bootstrap and the checksums are also checked.

## 10. Limitations

- **Only one linear specification.** Only position-specific main effects were tested.
  Logistic models with pairwise interaction features (e.g. \(e_{t-1} \times e_t\)) would
  close part of the gap by construction. They are a different model and were not run.
- **Mechanism not established.** The Pressure decomposition and the "interaction"
  explanation are descriptive and inferential. Model performance alone does not prove why
  the gap exists.
- **Fold-dependent C.** C was chosen per outer fold (1 or 10, practically equivalent).
  The coefficients of the five fold models therefore differ slightly in shrinkage.
- **Argmax metrics.** Argmax metrics for rare classes are near zero for every model; no
  cost-sensitive decision rule was applied (by design of the primary experiment).
- **Development data only.** The results are out-of-fold estimates on the 304 Tier B
  development matches. The final test remains untouched.

## 11. Reproducibility

```bash
python scripts/run_logistic_h5_experiment6a.py --smoke
caffeinate -i python scripts/run_logistic_h5_experiment6a.py
python scripts/verify_logistic_h5_experiment6a.py
python -m pytest tests/test_logistic_h5_experiment6a.py
```

- **Run environment.** Python 3.11.1, scikit-learn 1.9.0, NumPy 2.4.6, SciPy 1.17.1,
  pandas 3.0.5.
- **Source provenance.**
  - **At run time.** The run started from commit `a2343e9` with the experiment files not
    yet committed, so `a2343e9` alone does not identify the executed code.
  - **Identification.** The executed sources are identified by the SHA-256 of every source
    file, recorded in `environment.json` (`source_files_sha256`).
  - **Verification.** The final audit (`results/audit_experiments6a_6b_2026-11-24/`)
    verified that the committed files are byte-identical to the executed ones.
- **Seeds.** Folds 20261027, bootstrap 7, permutation 11, raw-verification sample 61. The
  solver is deterministic.

## 12. Conclusion

- **Primary result.** Under identical information and identical rows, folds and metrics,
  the multinomial logistic regression with position-specific one-hot history (MLR-H5) does
  **not** reach XGB-H5's performance. Its log loss is 0.051 higher (95 % CI
  [0.050, 0.053]), consistently in all five folds.
- **Against Markov 2.** It is also worse than the second-order Markov chain.
- **Against F0.** It is substantially better than current-event-only models.
- **Reading of the evidence.** The evidence suggests that the order-dependent *combination*
  of recent events matters for next-event prediction. Additively accumulating more history
  positions does not substitute for modelling it.
- **Practical implication for the thesis.** A linear softmax on this representation is an
  interpretable but clearly weaker baseline. Its main use is in describing how the
  influence of the history decays with lag.
