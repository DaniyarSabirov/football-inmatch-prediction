# XGB-H2: a controlled comparison with the second-order Markov chain (experiment 5B)

Status: methodology note, 2026-11-21. Task: `actions15_360current_v1`. Notation as in
`02_mathematical_framework.md` and `05_xgboost_gradient_boosting.md`.
- **Implementation:** `src/sequence_models/xgboost_h2_experiment.py` and
  `scripts/run_xgboost_experiment5b.py`.
- **Configuration:** `configs/xgboost_experiment5b_h2_v1.yaml`.
- **Results:** `results/xgboost_experiment5b_h2_actions15_360current_v1_2026-11-21/README.md`.

## 1. Why this experiment is needed

Experiment 5 found XGB-H5 better than Markov 2 by 0.016 in log loss. The two models,
however, differ in two ways at once:

| | Information | Model family |
|---|---|---|
| Markov 2 | \((E_{t-1}, E_t)\) | smoothed count table with backoff |
| XGB-H5 | \((E_{t-4}, \dots, E_t)\) | boosted trees |

The difference can therefore not be attributed to either factor. XGB-H2 receives **exactly**
Markov 2's information, so the comparisons separate the two factors:

| Comparison | Information | Model family | What it isolates |
|---|---|---|---|
| XGB-H2 vs. Markov 2 | equal | differs | the estimator (boosted trees vs. smoothed counts with backoff), including the \(t = 0\) rule (§3) |
| XGB-H5 vs. XGB-H2 | \(E_{t-2}, E_{t-3}, E_{t-4}\) added | equal | the contribution of the longer history |
| XGB-H2 vs. XGB-F0 | \(E_{t-1}\) added | equal | the contribution of \(E_{t-1}\) within boosting |

## 2. Definitions

| Object | Definition |
|---|---|
| Sequence position \(t\) (`seq_pos`) | position in the canonical sequence of one (match, period); 0-based |
| StatsBomb `index` | position of a raw record in the match file. It orders records, but is not \(t\) |
| `timestamp` | elapsed time in the period. Never used for ordering, and not a feature |
| Target | \(y_t = E_{t+1}\), the next canonical event of the same sequence |
| Input | \(\mathbf{x}_t = (E_{t-1}, E_t)\). Where \(t = 0\), \(E_{t-1}\) is the token `<START>` |
| Output | \(\hat{\mathbf{p}}_t \in \Delta^{14}\) with \(\hat p_{t,c} = \hat P(E_{t+1} = c \mid E_{t-1}, E_t)\), \(c = 1,\dots,15\); column \(c-1\) = canonical class id |

In the brief's notation, \(X_t = \mathbf{x}_t\), \(Y_t = y_t\), and
\(\hat p_{t+1,k} = \hat p_{t,c}\) with \(k = c\).

**Encoding.**
- \(E_{t-1}\) is one-hot over 16 categories: 15 classes, then `<START>`.
- \(E_t\) is one-hot over the 15 classes.

This gives 31 binary columns, the same encoding as experiment 5. No other information
enters, not even the position of the event within the match.

**Rows.** The rows are the 690,147 Protocol-B rows of Tier B, identical in order to those
of experiment 5. 568 of them have \(t = 0\).

## 3. How XGBoost differs from the second-order Markov chain

**Hypothesis space: almost, but not exactly, the same.**
- **XGB-H2** can represent any table \(P(E_{t+1} \mid E_{t-1}, E_t)\) over the
  \(16 \times 15 = 240\) contexts, including the 15 contexts \((\text{<START>}, E_t)\).
- **The canonical Markov 2** estimates the \(15 \times 15 = 225\) contexts with a real
  \(E_{t-1}\). At \(t = 0\) it has no context of its own and uses the first-order
  distribution \(P(E_{t+1} \mid E_t)\).

The two models therefore receive the same information, but they do not form the same
model: they differ in the \(t = 0\) rule and in how they estimate and regularise. Below,
\(u = E_{t-1}\), \(a = E_t\), and \(C_{u,a,c}\) is the training count of the transition
\((u, a) \to c\), with \(N_{u,a} = \sum_c C_{u,a,c}\). The vector \(\mathbf{b}\) is XGBoost's
intercept.

| | Markov 2 (canonical) | XGB-H2 |
|---|---|---|
| Estimate of a context's distribution | \((C_{u,a,c} + \alpha)/(N_{u,a} + K\alpha)\) | \(\operatorname{softmax}(\mathbf{b} + \eta \sum_m \text{leaf weights})\) |
| Shrinkage | toward uniform \(1/K\), strength \(\alpha\) (nested CV) | toward the intercept, through \(\lambda\), \(\eta\), finite \(M\), depth and `min_child_weight` |
| Rare contexts (< 20 training rows) | replaced by Markov 1, a hard backoff | pooled with similar contexts by shared tree splits, a soft backoff |
| \(t = 0\) (no \(E_{t-1}\)) | Markov 1 | the learned category `<START>` |
| Tuning | \(\alpha\) on a 7-value grid | 20 candidates × \(M\) |

**Interpretation rule.** A difference between XGB-H2 and Markov 2 is therefore a
difference of **estimators**: smoothing, backoff and regularisation. It is not evidence that
trees "see" structure the table cannot express. The mechanism can be inspected through the
contexts in which the predictions differ, and through Markov 2's backoff rate
(`markov2_backoff_rates.csv`).

## 4. Procedure

**Training procedure (identical to experiment 5):**
- the outer folds;
- leave-one-inner-fold-out within the outer training folds;
- the 20 seeded candidates;
- the lockstep pooled inner curve, with patience 50 and a 2,000-round cap;
- the tie rule;
- the refit on all outer-training rows with exactly \(M\) rounds;
- one evaluation per outer fold.

**Tuning.** XGB-H2 is tuned on its own inner folds. Nothing is copied from XGB-H5.

**Representation.** Training uses count-weighted cells, as in experiment 5 (see §5).

**Checkpoints.**
- **What is saved:** every completed outer fold is committed atomically, with its choice,
  inner curves, predictions, model and data-access trace.
- **When it is reused:** only if a hash identifies the same configuration, dataset, folds,
  split manifest, row order, feature matrix, source code and XGBoost version. Otherwise
  the run stops; it never mixes runs.
- **Effect on selection:** none. A checkpoint carries its own fold's choice.

**Comparisons.**
- Paired, on identical rows.
- The other models' predictions are the stored out-of-fold predictions of experiment 5,
  verified to reproduce its stored metrics.
- 1,000 match-level bootstrap replicates, seed 7. These are the same replicates as for the
  baselines and experiment 5.
- **Percentile** 95 % intervals of the paired difference, with per-fold signs.
- **Reading a CI.** A CI that excludes 0 means that the paired difference keeps its sign
  under resampling of matches. Nothing beyond that is claimed: no p-values and no
  multiple-comparison correction.

## 5. Weighted-cell training: what is and is not identical

**Exactly the same with weighted cells:**
- the objective;
- gradient and hessian sums;
- the intercept;
- `min_child_weight` and \(\lambda\) (weights are not normalised);
- the early-stopping metric.

**The difference** is float32 accumulation of \(w \cdot g\) vs. \(\sum g\). Near-tied splits
can flip, and the differences then compound over rounds.

The audit (`weighted_cell_audit_*.csv`) measures this on inner-split data. Outer-validation
data are never used. Results: see the results README.

## 6. Limitations (design)

- **Only event classes.** No spatial or temporal context is used.
- **Shared round cap and search space.** They are taken from experiment 5 for control. If
  they bind for H2, they do so equally by design.
- **Markov 2 is unchanged.** Its \(\alpha\) grid, its backoff threshold of 20 and its
  \(t = 0\) rule are the canonical ones. A better-smoothed Markov chain (for example
  interpolated or Kneser–Ney) was not evaluated.

## 7. Results (added after the run, 2026-11-21)

These figures are copied from
`results/xgboost_experiment5b_h2_actions15_360current_v1_2026-11-21/README.md`.
- **Data:** 690,147 rows and 304 matches.
- **Intervals:** paired match-level percentile bootstrap, 95 %.

| Comparison | Δ log loss [95 % CI] | Folds better |
|---|---|---|
| XGB-H2 − Markov 2 (same information) | −0.00047 [−0.00056, −0.00038] | 5 / 5 |
| XGB-H5 − XGB-H2 (adds \(E_{t-2..t-4}\)) | −0.01576 [−0.01645, −0.01511] | 5 / 5 |
| XGB-H2 − XGB-F0 (adds \(E_{t-1}\)) | −0.0909 [−0.0930, −0.0888] | 5 / 5 |

**Findings:**
- **Same information, nearly the same model.** With identical information, XGB-H2 and
  Markov 2 are practically equivalent.
  - **Decisions:** the accuracy difference is −0.000007, with a CI that includes 0. Neither
    model ever predicts Shot.
  - **Ranking:** Shot AP is +0.00015, with a CI that includes 0.
- **Most of the small log-loss gain comes from handling missing or rare history.**
  - **Share:** 58 % of the gain comes from the 0.14 % of rows where the canonical Markov 2
    backs off to Markov 1. The period starts alone contribute 46 %.
  - **On full-context rows** the gain is 0.0002.
- **What this is evidence of.** It is a difference in smoothing and backoff policy, not
  evidence that trees capture structure beyond the two-event table.
- **Decomposition of experiment 5's advantage.** This is an arithmetic decomposition of
  observed differences, not a causal attribution. The pooled log loss is a mean over the
  same rows, so the XGB-H5 advantage over Markov 2 splits exactly into two observed
  differences:
  \[
  \underbrace{-0.016224}_{\text{XGB-H5} - \text{Markov 2}}
  = \underbrace{-0.015756}_{\text{XGB-H5} - \text{XGB-H2}\ (97.1\,\%)}
  + \underbrace{-0.000468}_{\text{XGB-H2} - \text{Markov 2}\ (2.9\,\%)} .
  \]
  - **The 97 %** is the difference between two boosted-tree models that differ only in
    \(E_{t-2..t-4}\). It is therefore associated with the longer history, given the XGBoost
    estimator.
  - **The 3 %** includes the different \(t = 0\) rule as well as smoothing.
  - **Order matters.** Other decomposition paths need not split the same way, for example
    via a fifth-order Markov chain, which was not fitted.
  - **No significance claim.** The split is not a significance statement; the uncertainty
    of each term is given by its own paired CI.
