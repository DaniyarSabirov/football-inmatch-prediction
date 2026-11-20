# Mathematical framework for next-event prediction (K = 15)

Status: methodology note, 2026-11-17, written after the supervisor meeting of 2026-11-16.
It fixes the notation for every new document, code comment and report. Where a statement
depends on the implementation, the code location is given. Experimental evidence lives in
`results/logistic_regression_f0_2026-11-17/README.md`.

## 1. Notation

| Symbol | Meaning |
|---|---|
| \(t\) | position in the **event sequence** of one match period (an index, not elapsed time) |
| \(E_t\) | current event (its class) |
| \(E_{t+1}\) | next event (its class): the prediction target |
| \(c\) | a target class, \(c \in \{1,\dots,K\}\) |
| \(K = 15\) | number of classes (`scenarioD15`, `src/sequence_models/event_classes.py`) |
| \(\mathbf{x}_t \in \mathbb{R}^d\) | feature vector available at the current event |
| \(d\) | number of input features |
| \(W \in \mathbb{R}^{K\times d}\) | weight matrix; row \(c\) is \(\mathbf{w}_c^T\) |
| \(b_c\) | class-specific bias (intercept) |
| \(\alpha\) | additive smoothing parameter of the Markov chain |
| \(C_{a,c}\) | number of training transitions from class \(a\) to class \(c\) |
| \(N_a = \sum_{r} C_{a,r}\) | number of training transitions leaving class \(a\) |
| \(N\) | number of training examples (transitions) |
| \(y_i\) | observed class of the next event of training example \(i\) |
| \(C\) | inverse L2-penalty strength of the logistic regression (scikit-learn's `C`) |

Two different "C"s appear: \(C_{a,c}\) (a count, always with two subscripts) and \(C\) (the
regularisation hyperparameter, never subscripted). In code, class ids are 0-based
(`Pass` = 0, …, `50/50` = 14); in the text, \(c = \text{id} + 1\).

**Sequence index vs. row index.** \(t\) indexes events inside one sequence (one match
period). A training set pools the transitions of many sequences; we index those pooled
examples by \(i = 1,\dots,N\), and write \(\mathbf{x}_{(i)}\), \(y_i\) for the feature vector
and next-event class of example \(i\). Example \(i\) is one pair \((E_t, E_{t+1})\) of some
sequence.

**Example.** The sequence Carry → Pass → Shot gives \(E_{t-1} = \text{Carry}\),
\(E_t = \text{Pass}\), \(E_{t+1} = \text{Shot}\), and two training examples
(Carry → Pass) and (Pass → Shot).

## 2. The prediction task

Given the information available at the current event, estimate the categorical
distribution of the next event:
\[
P(E_{t+1} = c \mid \mathbf{x}_t), \qquad c = 1,\dots,K,
\qquad \sum_{c=1}^{K} P(E_{t+1} = c \mid \mathbf{x}_t) = 1 .
\]
The event-only special case is \(P(E_{t+1} = c \mid E_t)\).

**Event sequence.** The data define a *filtered-event* task
(`src/sequence_models/tasks.py`). A sequence is one (match, period); the penalty shoot-out
(period 5) is excluded. The observations are the raw StatsBomb records whose type is one of
the 15 classes, in raw event order (StatsBomb `index`, verified equal to file order).
Timestamps are not used to order events: they tie for about 31 % of adjacent raw pairs.
Records of other types are skipped by definition, mostly `Ball Receipt*`, `Foul Won` and
`Dribbled Past` companions. So \(E_{t+1}\) is the next *recorded football action of one of
the 15 types* in the same period, however many skipped records or seconds lie in between.
A sequence never crosses a period or match boundary. The last event of a period has no
target.

**Evaluation rows.** Predictions are scored on events \(E_t\) that have a StatsBomb 360
record, because later feature sets need the freeze frame. Training uses every transition
of the training matches. On Tier B, 98,165 of 799,309 transitions have no 360 record at
\(E_t\).

## 3. First-order Markov chain

**Assumption.** Given the current event, the next event is conditionally independent of the
earlier events:
\(P(E_{t+1}\mid E_t, E_{t-1}, \dots, E_1) = P(E_{t+1}\mid E_t)\). By the chain rule,
\[
P(E_1,\dots,E_T) = P(E_1)\prod_{t=1}^{T-1} P(E_{t+1}\mid E_t).
\]

**Estimation.** The parameters are the \(K\times K\) transition probabilities
\(P(E_{t+1}=c\mid E_t=a)\). Their maximum-likelihood estimate is \(C_{a,c}/N_a\). With
additive (Lidstone; \(\alpha = 1\): Laplace) smoothing,
\[
\hat P(E_{t+1}=c\mid E_t=a) = \frac{C_{a,c}+\alpha}{\sum_{r=1}^{K} C_{a,r} + K\alpha}.
\]
- **Bayesian reading.** Additive smoothing is the posterior mean under a symmetric
  Dirichlet(\(\alpha\)) prior on each row, and the MAP estimate under
  Dirichlet(\(\alpha+1\)).
- **Effect.** Every probability becomes positive, so the log loss is finite. Each row
  shrinks toward the **uniform** distribution \(1/K\), with weight
  \(K\alpha/(N_a+K\alpha)\), so rarely observed current events are shrunk most.
- **Implementation.** `src/sequence_models/markov.py`, `MarkovChain`. A current class
  never seen in training falls back to the smoothed marginal of the next event
  (`FrequencyBaseline`).

**On cross-validation (meeting of 2026-11-16).** A transition matrix is computed in closed
form from counts; no iterative optimisation and no cross-validation are needed to obtain it.
Cross-validation is nevertheless meaningful for the Markov chain, for two reasons:
- the transition probabilities are parameters estimated from data, so an honest estimate
  of predictive performance must use matches that were not counted;
- \(\alpha\) (and, for higher-order or backoff chains, the order and the concentration) is
  a hyperparameter that can only be chosen on held-out data.

The earlier experiments used match-grouped cross-validation for exactly these purposes, and
that remains correct. Cross-validation is a procedure for **evaluation and model
selection**; it is not part of fitting.

## 4. Multinomial logistic regression (softmax regression)

**Binary case.** For two classes, \(z = \mathbf{w}^T\mathbf{x}+b\) and
\(P(y=1\mid\mathbf{x}) = \sigma(z) = 1/(1+e^{-z})\).

**Multiclass case.** For \(K = 15\) classes, each class has its own linear score (logit)
\[
z_{t,c} = \mathbf{w}_c^T\mathbf{x}_t + b_c ,
\]
and the softmax function maps the \(K\) logits to a distribution:
\[
\boxed{P(E_{t+1}=c\mid\mathbf{x}_t) = \frac{\exp(z_{t,c})}{\sum_{r=1}^{K}\exp(z_{t,r})}}
\]

Properties:
1. Every probability is in (0, 1) and the probabilities sum to one.
2. Adding the same constant to all logits changes nothing (shift invariance), so only
   differences of logits are identified. With \(K = 2\), softmax reduces to the sigmoid of
   \(z_1 - z_2\).
3. The log-odds of two classes are linear in the features:
   \(\log\frac{P(c\mid\mathbf{x})}{P(r\mid\mathbf{x})} = (\mathbf{w}_c-\mathbf{w}_r)^T\mathbf{x} + b_c - b_r\).
4. A numerically stable evaluation subtracts \(\max_r z_{t,r}\) before exponentiating;
   scikit-learn does this internally.

**Softmax is not a substitute for cross-validation.** Softmax is the *link function* that
turns scores into a categorical distribution; it is part of the model. Cross-validation is
a *procedure* that fits the whole model on some matches and scores it on others, to estimate
performance and choose hyperparameters such as \(C\). Every softmax model still needs an
evaluation protocol, and every protocol needs a model.

### Why softmax suits the 15-class task (Question B)
- **One label per event.** The next event has exactly one of 15 mutually exclusive
  classes, so the target is categorical. Softmax regression is the generalised linear
  model for a categorical response: it is the canonical link of the multinomial
  distribution.
- **Full distribution.** It outputs a complete distribution, not just a label. The
  thesis evaluates probabilities with proper scoring rules (log loss, Brier score), and
  argmax or cost-sensitive decisions can be derived from them afterwards.
- **Matching training objective.** Its log-likelihood is the cross-entropy, i.e. exactly
  the log loss used for evaluation.
- **Consistency.** All 15 probabilities are estimated jointly, so they are mutually
  consistent. 15 independent one-vs-rest sigmoids would not sum to one.

## 5. Training objective

**Cross-entropy.** For one-hot targets, the average negative log-likelihood over the
training examples is
\[
\mathcal{L}(W,\mathbf{b}) = -\frac{1}{N}\sum_{i=1}^{N}\log P(E_{t+1}=y_i\mid\mathbf{x}_{(i)}).
\]
It is convex in \((W,\mathbf{b})\), because log-sum-exp is convex and the logits are
linear, so every local minimum is global.

**Regularised objective (implemented).** `src/sequence_models/logistic.py` wraps
scikit-learn 1.9.0 `LogisticRegression(solver="lbfgs", l1_ratio=0.0, C=…)`. According to
the installed source (`sklearn/linear_model/_logistic.py`: `l2_reg_strength = 1/(C·Σ
sample_weight)`; `_linear_loss.py`: the intercept is not penalised), it minimises
\[
\mathcal{L}_C(W,\mathbf{b}) = \mathcal{L}(W,\mathbf{b}) + \frac{1}{2CN}\lVert W\rVert_F^2
\quad\Longleftrightarrow\quad
\sum_{i=1}^{N} -\log P(\cdot) + \frac{1}{2C}\lVert W\rVert_F^2 .
\]
- **Penalty strength.** The penalty is fixed while the data term grows with \(N\). A given
  \(C\) therefore regularises less on more data: with \(N \approx 6.4\times10^5\), \(C \ge
  10^{-2}\) is almost unpenalised.
- **No penalty.** \(C=\infty\) gives the unpenalised maximum-likelihood fit. This is the
  non-deprecated spelling: `penalty` is deprecated since scikit-learn 1.8, and the
  removed `multi_class` parameter is not used.
- **Solver.** L-BFGS is a quasi-Newton method. It is deterministic, so no random seed is
  involved. Convergence is recorded from `n_iter_` and scikit-learn's
  `ConvergenceWarning`.

## 6. Markov chain vs. logistic regression on the same information (Question A)

**F0 features.** In F0, \(\mathbf{x}_t = \operatorname{OneHot}(E_t) = \mathbf{e}_a\) when
\(E_t = a\). Then \(z_{t,c} = W_{c,a} + b_c =: \theta_{a,c}\), so the model has one free
logit vector \(\boldsymbol\theta_a\) per current class, and
\[
P(E_{t+1}=c\mid E_t=a) = \operatorname{softmax}(\boldsymbol\theta_a)_c .
\]

**Likelihood.** The log-likelihood depends on the data only through the counts:
\[
\log\mathcal{\ell} = \sum_{a=1}^{K}\sum_{c=1}^{K} C_{a,c}\,\log \operatorname{softmax}(\boldsymbol\theta_a)_c ,
\]
and it separates into \(K\) independent problems, one per current class \(a\).

**Claim 1 (no penalty: identical to the Markov MLE).** Let \(N_a > 0\) and \(C_{a,c} > 0\)
for all \(c\). Each problem maximises \(\sum_c C_{a,c}\log q_c\) over the probability simplex.
By Gibbs' inequality, the unique maximiser is \(q_c = C_{a,c}/N_a\). Every interior point of
the simplex is a softmax of some \(\boldsymbol\theta_a\). Therefore the unpenalised F0 logistic
regression returns exactly the empirical transition probabilities: the first-order Markov
MLE, i.e. the Markov chain with \(\alpha = 0\).
- **Over-parameterisation.** The model has \(K^2 + K = 240\) parameters for
  \(K(K-1) = 210\) free probabilities. The parameters are not unique (shift invariance per
  row, plus a trade-off between \(W\) and \(\mathbf{b}\)); the probabilities are.
- **Zero cells.** If some \(C_{a,c} = 0\), the supremum is reached only as
  \(\theta_{a,c}\to-\infty\), so no finite MLE exists (the multinomial analogue of
  separation). A numerical optimiser stops at a large negative logit and returns a tiny
  probability. In Tier B, 43 of the 225 cells are empty.

**Claim 2 (L2 penalty: shrinkage toward a data-driven base distribution).** Set the
gradient of \(\sum_i -\log P + \lVert W\rVert^2/(2C)\) to zero. Writing
\(f_{a,c} = C_{a,c}/N_a\) and \(p_{a,c}\) for the fitted probability:
\[
\frac{\partial}{\partial W_{c,a}}:\quad W_{c,a} = C\,N_a\,(f_{a,c} - p_{a,c}),
\qquad
\frac{\partial}{\partial b_c}:\quad \sum_{a} N_a\,p_{a,c} = \sum_a C_{a,c}.
\]
- **Shrinkage scale.** The fitted row deviates from the empirical row by
  \(W_{\cdot,a}/(C N_a)\). Rarely observed current events (small \(N_a\)) are shrunk most,
  as with additive smoothing.
- **Shrinkage target.** The limit is different. As \(W\to 0\), every row tends to
  \(\operatorname{softmax}(\mathbf{b})\), a data-driven base distribution. Additive
  smoothing shrinks toward the uniform \(1/K\) instead. L2-penalised F0 regression is
  therefore closer to a Dirichlet prior centred on the marginal, like experiment 2's
  hierarchical backoff chain `mc1`, than to Laplace smoothing.
- **Calibration in the large.** The second condition holds for every \(C\): the fitted
  model reproduces the empirical next-event frequencies on the training data exactly,
  because the intercept is unpenalised.
- **Tested.** `tests/test_logistic_f0.py` checks both conditions numerically:
  `test_l2_stationarity_condition_and_rare_state_shrinkage` and
  `test_intercept_reproduces_training_marginal_for_any_penalty`.

**Consequence for the thesis.** With exactly the information of the Markov chain, the F0
regression cannot be expected to beat it. Any difference comes from:
- (i) the form of smoothing or shrinkage;
- (ii) cells with zero or very small counts;
- (iii) optimisation tolerance.

All three are small when every \(N_a\) is large. Logistic regression becomes interesting
once \(\mathbf{x}_t\) contains information that a count table cannot hold efficiently:
continuous coordinates, many interacting categorical variables, or 360 geometry.
- **Count tables.** They grow multiplicatively with each added variable, so their cells
  become sparse.
- **Linear-softmax model.** Its parameters grow additively: one weight per feature and
  class.

This makes F0 the reference point for the controlled feature expansion of
`04_feature_engineering_roadmap.md`.

## 7. Regularisation and rare classes (Question C)

**Rare current events (rows).** By Claim 2, the deviation from the empirical row scales
with \(1/(C N_a)\): small-\(N_a\) rows are pulled toward \(\operatorname{softmax}(\mathbf{b})\).
This is desirable when \(N_a\) is genuinely small, because it reduces variance. It is
harmful if the rare event really has a distinctive successor distribution, such as
`Dispossessed → Duel`.

**Rare target classes (columns).** The unpenalised intercept guarantees that the average
predicted probability of every class equals its training frequency. The penalty does not
remove rare classes. It shrinks their class-specific *variation* across inputs, i.e. the
\(\mathbf{w}_c\) of rare classes. A rare class \(c\) then receives a nearly constant small
probability, and under argmax it is almost never predicted. Its ranking quality (average
precision) degrades before its log loss does.

**Feature scaling.** With standardised continuous features (F2 onward), the same \(C\)
penalises all features alike. One-hot columns are not standardised in F0. This is
harmless, because F0 has no continuous feature competing for the penalty.

**What regularisation does not fix.**
- **Argmax and imbalance.** Shot recall under argmax is limited mostly by class imbalance
  (Shot ≈ 1 % of next events) and by the argmax rule itself, not by the penalty.
- **Better tools.** Decisions for rare classes are better addressed separately, by
  threshold or cost-sensitive Bayes rules on the *unchanged* probabilities (experiment 2,
  `cost_sensitive.py`), reported apart from argmax metrics.
- **Class weights.** These change the estimated probabilities (they are no longer
  calibrated) and were rejected in experiment 2 (Approach B).

## 8. Evaluation

**Probability metrics.** These use the full distribution:
- multiclass log loss \(-\frac1n\sum\log\hat p_{y}\) (natural log; the primary metric, a
  strictly proper scoring rule);
- multiclass Brier score;
- calibration (classwise and top-label ECE);
- one-vs-rest average precision, e.g. for Shot (a ranking metric).

**Decision metrics.** These use a decision rule; the standard rule is argmax:
- accuracy, macro-F1, weighted-F1, balanced accuracy;
- per-class precision, recall and F1;
- the confusion matrix.

Cost-sensitive decision rules are always reported separately and labelled as such.

**Protocol.** The splits are match-grouped, because events within a match are dependent:
- a fixed final-test hold-out of 83 matches (never used for development);
- Tier B: 304 development matches with 5 match folds;
- hyperparameters chosen by nested leave-one-inner-fold-out cross-validation inside each
  outer training split.

**Uncertainty.** Uncertainty comes from a match-level (cluster) bootstrap. Paired model
differences resample the same matches for both models.

## 9. Feature engineering strategy (summary)

Model complexity and information are varied **separately**: the feature set (F0…F6) and the
model family (logistic regression, gradient boosting, MLP, recurrent networks) are crossed
in a factorial design. The details are in `04_feature_engineering_roadmap.md`.

## References

Cited from bibliographic knowledge. They were not re-accessed in this session, so check
the edition and section numbers before submission. The scikit-learn statements in §5 were
verified against the installed source code of scikit-learn 1.9.0.

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. §4.3.4
  (multiclass logistic regression), §2.2 (multinomial and Dirichlet distributions).
- Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*
  (2nd ed.). Springer. §4.4 (logistic regression), §7.10 (cross-validation).
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
  Ch. 8 (logistic regression), §3.4 (Dirichlet-multinomial, add-one smoothing).
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. §6.2.2.3
  (softmax units), §4.1 (numerical stability of softmax).
- McCullagh, P., Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman &
  Hall. Multinomial response models.
- Norris, J. R. (1997). *Markov Chains*. Cambridge University Press.
- Manning, C. D., Schütze, H. (1999). *Foundations of Statistical Natural Language
  Processing*. MIT Press. §6.2 (Laplace and Lidstone smoothing).
- Chen, S. F., Goodman, J. (1999). An empirical study of smoothing techniques for language
  modeling. *Computer Speech & Language*, 13(4), 359–394.
- Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
  estimation. *JASA*, 102(477), 359–378.
- Albert, A., Anderson, J. A. (1984). On the existence of maximum likelihood estimates in
  logistic regression models. *Biometrika*, 71(1), 1–10.
- Liu, D. C., Nocedal, J. (1989). On the limited memory BFGS method for large scale
  optimization. *Mathematical Programming*, 45, 503–528.
- Varma, S., Simon, R. (2006). Bias in error estimation when using cross-validation for
  model selection. *BMC Bioinformatics*, 7, 91.
- scikit-learn developers. `sklearn.linear_model.LogisticRegression`, version 1.9.0
  documentation and source code.
