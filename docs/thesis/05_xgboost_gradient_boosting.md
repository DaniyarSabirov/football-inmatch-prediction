# Gradient-boosted trees (XGBoost) for next-event prediction (experiment 5)

Status: methodology note, 2026-11-20. Task: `actions15_360current_v1`
(`docs/canonical_prediction_protocol.md`). Notation follows `02_mathematical_framework.md`.
Implementation: `src/sequence_models/event_history.py` (features),
`src/sequence_models/xgboost_experiment.py` (model, nested CV, evaluation),
`scripts/run_xgboost_experiment5.py`. Results and their interpretation:
`results/xgboost_experiment5_actions15_360current_v1_2026-11-20/README.md`.

## 1. The prediction problem

**Sequences.** For each match and each period \(p \in \{1,2,3,4\}\) (the penalty shoot-out,
period 5, is excluded), the canonical sequence \(E_0, E_1, \dots, E_{T-1}\) is the list of
raw StatsBomb records of that period that
- have one of the \(K = 15\) `actions15` types, and
- are not companion records (rules K1–K3; K4 is not active),

ordered by the raw StatsBomb `index`.

**Three different "indices".** These must not be confused:

| Symbol / column | Meaning |
|---|---|
| \(t\) (`seq_pos`, 0-based) | position in the **canonical sequence** of one (match, period). Consecutive observations have consecutive \(t\) |
| StatsBomb `index` | position of a **raw record** in the match's event file (all types, all periods). Between \(E_t\) and \(E_{t+1}\) there can be any number of skipped raw records (Ball Receipt\*, companions, substitutions, …), so their `index` values are usually not consecutive |
| `timestamp` / `t_seconds` | elapsed time within the period. Never used for ordering (ties for ~31 % of adjacent raw pairs) |

**Target.** For an observation \(E_t\) with a successor in the same sequence,
\[
y_t = E_{t+1} \in \{1, \dots, K\}, \qquad
\hat p_{t,c} = \hat P(E_{t+1} = c \mid \mathbf{x}_t), \quad c = 1,\dots,K,
\]
with \(\sum_c \hat p_{t,c} = 1\). In code, class ids are 0-based (\(c = \text{id} + 1\);
`Pass` = 0, …, `50/50` = 14).

**Information set.** \(\mathbf{x}_t\) is a function of the causally available history
\(\mathcal{H}_t\): the observations \(E_0, \dots, E_t\) of the same sequence. In the
notation of the experiment brief, \(X_t = \mathbf{x}_t\) and \(Y_t = y_t = E_{t+1}\), so
\(\hat p_{t+1,k} = P(E_{t+1} = k \mid \mathcal{H}_t)\) is \(\hat p_{t,c}\) here, with
\(k = c\). The repository writes the target index as \(t\) (the row), not \(t+1\).

**Rows (Protocol B).** A row exists for every \(E_t\) that has a successor \(E_{t+1}\) in the
same sequence **and** its own usable 360 snapshot. The successor needs no snapshot. The same
rule applies to training and evaluation, and to every model: on Tier B there are
690,147 rows in 304 matches.

**Boundaries.** A sequence starts at the first canonical observation of a period and ends
at its last one. The last observation has no target. No target and no history position is
ever taken from another period or match.

## 2. Feature configurations

| Name | \(\mathbf{x}_t\) | Encoding | Dimension |
|---|---|---|---|
| XGB-F0 | \((E_t)\) | one-hot over 15 classes | 15 |
| XGB-H5 | \((E_{t-4}, E_{t-3}, E_{t-2}, E_{t-1}, E_t)\) | \(E_t\): one-hot over 15; each \(E_{t-j}\): one-hot over 16 = 15 classes + `<START>` | 79 |

- **Padding.** If \(t - j < 0\), position \(E_{t-j}\) takes the token `<START>` (id 15). It
  is not one of the 15 classes, so it cannot be mistaken for an event. Padding is
  monotone: if \(E_{t-j}\) is `<START>`, so are all earlier positions. Rows near the start
  of a period are kept, not dropped.
- **History and 360.** The history positions are earlier observations of the sequence,
  whatever their 360 status. They are available at prediction time. This is the same rule
  as the context \((E_{t-1}, E_t)\) of the canonical second-order Markov chain.
- **Nothing is fitted** in the encoding. The vocabulary is fixed, so the encoding is
  identical in every fold.
- **Not used:** coordinates, timestamps, team or player identities, outcomes, end
  locations, recipients, `related_events`, durations, possession ids, match/event ids, and
  anything derived from \(E_{t+1}\).
- **Names.** "XGB-F0"/"XGB-H5" avoid the planned feature-set names F1–F6
  (`src/sequence_models/feature_sets.py`), where F5 means "previous event classes" with an
  unspecified length.

## 3. The XGBoost model

**Additive ensemble of trees.** For each class \(c\), XGBoost's `multi:softprob` objective
fits a score
\[
z_c(\mathbf{x}) = b_c + \sum_{m=1}^{M} f_{m,c}(\mathbf{x}), \qquad f_{m,c} \in \mathcal{F},
\]
- \(M\) is the number of boosting rounds (`n_estimators`); every round adds one regression
  tree per class, i.e. \(15M\) trees in total;
- \(b_c\) is a constant intercept (`base_score`), estimated from the training labels;
- \(\mathcal{F}\) is the set of regression trees: a tree \(f\) partitions the feature space
  into \(T\) leaves by axis-aligned splits and assigns a weight \(w_\ell\) to each leaf:
  \(f(\mathbf{x}) = w_{q(\mathbf{x})}\), where \(q\) maps \(\mathbf{x}\) to its leaf.

**Softmax.** The scores are mapped to a distribution:
\[
\boxed{P(E_{t+1} = c \mid \mathbf{x}_t) = \frac{\exp z_c(\mathbf{x}_t)}{\sum_{j=1}^{K} \exp z_j(\mathbf{x}_t)}}
\]
This is the same link function as in softmax regression (`02_mathematical_framework.md`,
§4). Only the score function differs:
- **softmax regression:** \(z_c\) is **linear** in \(\mathbf{x}\);
- **XGBoost:** \(z_c\) is a **sum of trees**. It is piecewise constant, and it can
  represent interactions between inputs up to the tree depth.

**Regularised objective.** With \(\ell(y, \mathbf{z}) = -\log \operatorname{softmax}(\mathbf{z})_y\)
(the multiclass log loss, natural log),
\[
\mathcal{L} = \sum_{i=1}^{N} \ell\bigl(y_i, \mathbf{z}(\mathbf{x}_{(i)})\bigr)
 + \sum_{m,c} \Omega(f_{m,c}), \qquad
\Omega(f) = \gamma T + \tfrac{\lambda}{2} \lVert \mathbf{w} \rVert_2^2 + \alpha_1 \lVert \mathbf{w} \rVert_1 .
\]
In experiment 5: \(\gamma = 0\), \(\alpha_1 = 0\); \(\lambda\) is `reg_lambda` (tuned).

**Boosting step (second order).** In round \(m\), the existing scores are fixed and the new
trees are fitted to a second-order Taylor expansion of the loss. For class \(c\) and
example \(i\),
\[
g_{i,c} = \frac{\partial \ell}{\partial z_c} = p_{i,c} - \mathbb{1}[y_i = c], \qquad
h_{i,c} = 2\,p_{i,c}(1 - p_{i,c}) .
\]
The exact diagonal of the Hessian of \(\ell\) is \(p_{i,c}(1-p_{i,c})\). XGBoost's
`multi:softprob` implementation uses this diagonal (no off-diagonal terms) multiplied by 2.
This was verified for XGBoost 3.1.3: a single unsplittable round with \(\lambda = 0\) and
\(\eta = 1\) gives leaf values \(-G/H\) with exactly this \(H\). The factor rescales every
leaf weight, and therefore interacts with \(\eta\), \(\lambda\) and `min_child_weight`; it is
part of the implemented algorithm and applies equally to all runs.
For a fixed tree structure with leaves \(I_\ell\),
- the optimal leaf weight is \(w_\ell^\ast = -G_\ell / (H_\ell + \lambda)\), with
  \(G_\ell = \sum_{i \in I_\ell} g_{i,c}\) and \(H_\ell = \sum_{i \in I_\ell} h_{i,c}\);
- a split of a leaf into left/right children has the gain
  \[
  \tfrac12\Bigl[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\Bigr] - \gamma .
  \]

Trees are grown greedily by gain, subject to:
- `max_depth`, the maximum tree depth;
- `min_child_weight`, the minimum hessian sum \(H\) per leaf (a sample-size floor);
- `colsample_bytree`, the fraction of feature columns offered to each tree.

**Shrinkage.** Each new tree is multiplied by the learning rate \(\eta\) (`learning_rate`)
before it is added. A smaller \(\eta\) needs more rounds \(M\). \(M\) is chosen by inner
cross-validation (§6).

**Probabilities in this implementation.** The code takes the raw margins \(z_c\) from
XGBoost and applies the softmax itself, in float64 with the max-subtraction of the
framework (§4, property 4). Every row is therefore strictly positive and sums to 1. Column \(c\) is
canonical class id \(c\):
- labels are passed as canonical ids, without a label encoder;
- `num_class = 15` keeps all 15 outputs, even for a class absent from a training fold.

## 4. Relationship to Markov chains and logistic regression

**XGB-F0 on one categorical input.** With \(\mathbf{x}_t = \operatorname{OneHot}(E_t)\),
every tree is a function of \(E_t\) alone, and so is \(z_c\). The model can therefore only
represent a table \(P(E_{t+1} = c \mid E_t = a)\): the same hypothesis space as the
first-order Markov chain and F0 logistic regression
(`02_mathematical_framework.md` §6). The three differ only in how the table is estimated
and regularised:

| Model | Estimate of row \(a\) | Shrinkage toward |
|---|---|---|
| Markov 1 | \((C_{a,c} + \alpha)/(N_a + K\alpha)\) | uniform \(1/K\) |
| Logistic F0 | softmax of \(\theta_a\), L2 penalty on \(W\) | intercept \(b\) (the marginal) |
| XGB-F0 | softmax of \(b + \eta \sum_m (\text{leaf weights})\) | intercept \(b\), through \(\lambda\), \(\eta\), finite \(M\), `min_child_weight` |

With ~690,000 rows and only 15 contexts, every reasonable estimator approaches the
empirical table. **XGB-F0 is therefore expected to match Markov 1, not to beat it.** It is a
control: it shows whether the boosting machinery and the nested selection add bias or
variance when they have no extra information.

**XGB-H5 vs. higher-order Markov chains.**
- **A fifth-order Markov table** has \(15 \cdot 16^4 \approx 10^6\) possible contexts. On
  Tier B, 17,280 of them occur, most only a few times. Counting alone would overfit; a
  table needs backoff or smoothing.
- **XGBoost shares statistical strength** across contexts. A tree splits on whichever
  positions and classes reduce the loss most, so rare contexts inherit the leaf of similar
  frequent contexts. Interactions between positions are limited by the depth. The additive
  sum over trees is additive in log-odds.
- **The canonical Markov 2** uses only \((E_{t-1}, E_t)\), with backoff to Markov 1 below 20
  training rows. **XGB-H5 sees three more history positions.** A comparison of XGB-H5 with
  Markov 2 is therefore a comparison of feature configurations **and** model families. It
  is not an architecture-only comparison.

**Logistic regression on the H5 one-hot features** (not fitted in this experiment) would
be additive in log-odds over positions, without interactions. XGBoost adds interactions,
for example "Pressure at \(t-1\) and Pass at \(t\)".

## 5. Estimation on count-weighted cells

The H5 inputs are discrete. The 690,147 rows contain only 41,032 distinct
\((\mathbf{x}, y)\) cells, and XGB-F0 has at most 225. Each training set is therefore
collapsed to its distinct cells \(j\), with weight \(n_j\) equal to the number of rows in the
cell.

**Claim.** With row subsampling switched off (`subsample = 1`), the weighted fit has the
same objective and, in exact arithmetic, the same boosting steps as the row-level fit.

**Proof sketch.**
- Rows in one cell have identical \(\mathbf{x}\) and \(y\). They therefore have identical
  scores and gradients at every round.
- XGBoost multiplies \(g\) and \(h\) by the instance weight. Every sum \(G_\ell\) and
  \(H_\ell\) above is thus the same over cells (weighted) as over rows.
- So are the gains, the leaf weights, the `min_child_weight` constraint and the
  log-loss-based intercept.
- Binary features have the same two histogram bins in both forms.

**Precedent.** This is the same argument as the count-weighted logistic F0 fit of the
canonical baselines.

**Numerical difference.** What differs is the float32 summation order. Near-tied splits can
therefore flip, and these differences compound over many rounds. Measured effects (outer
fold 0, refit on individual rows; `fit_verification.csv`):

| Fit | Rounds | \(\lvert\Delta\) log loss\(\rvert\) | Max \(\lvert\Delta p\rvert\) | Argmax agreement |
|---|---|---|---|---|
| short check, H5 features | 50 | \(2 \cdot 10^{-7}\) | 0.0006 | — |
| XGB-F0 | 376 | \(1 \cdot 10^{-7}\) | 0.00006 | 100 % |
| XGB-H5 | 1,895 | \(1.6 \cdot 10^{-4}\) (cell fit lower) | 0.21 | 99.38 % |

For long H5 fits the two forms are therefore two numerically different realisations of the
same objective, not bit-identical models. The log-loss difference is about 1 % of the
XGB-H5 vs. Markov 2 gap (0.016), so no conclusion depends on it.

**Subsample is fixed at 1.** Row subsampling (`subsample` < 1) would sample cells rather
than rows. That is a different randomisation, so `subsample` is not tuned and stays
at 1.

## 6. Nested cross-validation and the number of rounds

**Folds.**
- **Outer folds.** The 5 match-grouped outer folds of experiment 2, identical to the
  canonical baselines.
- **Inner folds.** For outer fold \(k\), the inner folds are the other four outer folds,
  used leave-one-out: each inner model trains on 3 of them and is validated on the
  fourth.

**Candidates.** 20 hyperparameter configurations are drawn once, with a fixed seed, from a
fixed space:

| Hyperparameter | Values |
|---|---|
| `max_depth` | {2, 3, 4, 6, 8} |
| `learning_rate` | {0.05, 0.1, 0.2} |
| `min_child_weight` | {1, 5, 20, 100} |
| `reg_lambda` | {0.1, 1, 10, 100} |
| `colsample_bytree` | {0.5, 0.75, 1} |

The same 20 candidates are used for both feature sets and all folds.

**Number of rounds \(M\).**
- **Lockstep training.** For each candidate, the 4 inner boosters are trained in lockstep.
  After round \(r\) the pooled inner-validation log loss is
  \[
  \bar\ell(r) = \frac{\sum_{\text{inner } v} \sum_{i \in v} \ell_i(r)}{\sum_v |v|} .
  \]
- **Stopping.** Training stops after 50 rounds without improvement, or at 2,000 rounds.
- **Selection.** The selected pair is \((\text{candidate}, M) = \arg\min \bar\ell\). Ties go
  to the earlier candidate, then to fewer rounds.

**Observed cap behaviour.** The cap was reached by 80 of the 100 XGB-F0 searches and 39 of
the 100 XGB-H5 searches. For XGB-F0 the curves were flat, improving by about \(10^{-7}\)
per round. In XGB-H5 the selected candidate's \(M\) equals the cap in 3 of 5 folds; its
pooled inner curve still improved by about \(10^{-4}\) over the last 200 rounds. The cap
was fixed in advance and was not changed after inspection.

**Refit.** The selected model is refitted on all outer-training rows with exactly \(M\)
rounds, without rescaling \(M\) for the larger training set. It is then scored once on the
outer-validation rows.

**Isolation.** The outer-validation fold influences nothing:
- not the candidate, \(M\) or early stopping;
- not the encoding, which has no fitted parameters;
- not any threshold (decisions are argmax).

`data_access_audit.csv` lists every data set handed to XGBoost.

## 7. Evaluation and interpretation

**Metrics.** The definitions are those of the canonical baselines:
- **Primary:** log loss.
- **Probability metrics:** Brier score and calibration.
- **Decision metrics (argmax):** accuracy, macro-F1 and weighted-F1.
- **Per class:** precision, recall, F1 and AP.
- **Shot:** Shot AP and Shot recall.

**Ranking vs. decisions.** Shot AP is a **ranking** metric of \(\hat p_{t,\text{Shot}}\).
Shot recall is an **argmax decision** metric. An improvement in one does not imply an
improvement in the other.

**Uncertainty.**
- **Bootstrap.** Match-level (cluster) bootstrap with 1,000 replicates and seed 7: the same
  resampled matches as the baselines, paired between models.
- **Folds.** Per-fold signs are reported as well.
- **Rows are not independent.** Event rows within a match are dependent, so rows are never
  resampled individually.

**Permutation importance (XGB-H5).** For each outer fold, the fold's final model is scored on
its outer-validation rows with one history position (or a block of positions) permuted
across rows. The increase in log loss is that position's importance. Two caveats:
- **Correlated positions.** Permuting \(E_{t-1}\) independently of \(E_t\) creates
  combinations that rarely occur, for example "Shot followed by Shot". The values describe
  how the **fitted model** uses an input. They do not describe how football works.
- **Not causal.** Neither permutation importance nor tree split gain is evidence of a
  causal effect.
