# Thesis draft: prediction framework, baselines and the move to discriminative models

Draft text for the methodology chapter, 2026-11-17. The notation follows
`02_mathematical_framework.md`. Sections 1–5 describe methods. Section 6 reports the only
*completed* experiment of the new direction (experiment 4), with numbers taken from
`results/logistic_regression_f0_2026-11-17/`. Section 7 is explicitly future work.
Bracketed notes `[…]` are for the author and should be removed before submission.

---

## 1. Prediction framework

We study next-event prediction in football. A match period is represented as an ordered
sequence of on-ball and defensive actions \(E_1, E_2, \dots, E_T\), where each \(E_t\) belongs
to one of \(K = 15\) event classes: Pass, Carry, Ball Recovery, Clearance, Dribble, Shot,
Miscontrol, Dispossessed, Pressure, Duel, Interception, Block, Goal Keeper, Foul Committed
and 50/50. The index \(t\) denotes the position of an event within the sequence, not elapsed
time. Sequences are formed from StatsBomb event data in the recorded event order. Records
that do not describe one of the 15 actions, predominantly ball receipts, fouls won and
administrative records, are not part of the sequence. A sequence never extends across a
period or a match.

At every event we predict the probability distribution of the class of the next event,
\[
P(E_{t+1} = c \mid \mathbf{x}_t), \qquad c \in \{1,\dots,K\},
\]
where \(\mathbf{x}_t\) collects the information available at the time of \(E_t\): the event
itself, its location, its timing, the preceding events and, through StatsBomb 360, the
positions of the visible players at that moment. Every feature is computed from \(E_t\), its
own freeze frame and earlier records only, so a prediction never uses information from the
future. Predictions are evaluated on events that have a 360 freeze frame.

We treat the task as probabilistic classification. The primary quantity of interest is
the predicted distribution, not a single predicted label. Football is highly stochastic:
even with perfect information, the next action is uncertain. A model that assigns a
calibrated 8 % probability to a shot is more useful, both analytically and for downstream
decision rules, than one that always predicts the majority class.

## 2. Statistical baselines

**Empirical frequency baseline.** It predicts the marginal distribution of next-event
classes in the training data, independently of \(\mathbf{x}_t\). It defines the
no-information reference.

**First-order Markov chain.** It assumes that, given the current event, the next event is
conditionally independent of the earlier history:
\[
P(E_1,\dots,E_T) = P(E_1)\prod_{t=1}^{T-1}P(E_{t+1}\mid E_t).
\]
Its parameters are the \(K\times K\) transition probabilities, estimated from transition
counts \(C_{a,c}\) with additive smoothing,
\[
\hat P(E_{t+1}=c\mid E_t=a) = \frac{C_{a,c}+\alpha}{\sum_{r=1}^{K}C_{a,r}+K\alpha},
\]
which corresponds to a symmetric Dirichlet prior and guarantees non-zero probabilities.
Earlier stages of the project also evaluated second-order chains, zone-conditioned chains
with hierarchical backoff, and hidden Markov models. These remain reference models. Their
main role in the present chapter is to define what can be achieved with event history and
coarse spatial context alone.

Although the transition matrix is obtained in closed form, its entries are parameters
estimated from data. Out-of-sample evaluation therefore still requires that the matches
used for counting differ from those used for scoring, and the smoothing parameter
\(\alpha\) can only be chosen on held-out data. Cross-validation is thus used for
*evaluation and model selection*, not to fit the chain.

## 3. From generative sequence models to discriminative classification

Markov chains model the event sequence generatively and condition on a small discrete
state. Extending the state with further variables multiplies the number of cells: with 15
event classes, 12 pitch zones and 15 previous-event classes, a single table has 2,700
contexts × 15 outcomes. Most cells become sparse, and continuous quantities such as
distances or angles must be discretised.

Discriminative classifiers instead model \(P(E_{t+1}\mid\mathbf{x}_t)\) directly as a
function of a feature vector whose dimension grows only additively with each added
variable. The simplest such model, and the natural bridge from the Markov baseline, is
multinomial logistic regression.

## 4. Multinomial logistic regression

For each class \(c\), a linear score (logit) is computed,
\(z_{t,c} = \mathbf{w}_c^T\mathbf{x}_t + b_c\), and the softmax function converts the \(K\)
logits into a categorical distribution:
\[
P(E_{t+1}=c\mid\mathbf{x}_t) = \frac{\exp(z_{t,c})}{\sum_{r=1}^{K}\exp(z_{t,r})}.
\]
For two classes, this reduces to the familiar sigmoid \(\sigma(z) = 1/(1+e^{-z})\). The
softmax yields probabilities that are positive and sum to one. The log-odds between any
two classes are linear in the features, so every coefficient has a direct interpretation
as a change in a log-odds ratio.

The weights \(W\) and biases \(\mathbf{b}\) are estimated by minimising the cross-entropy,
i.e. the average negative log-likelihood of the observed next events, with an \(L_2\)
penalty on \(W\):
\[
\mathcal{L}_C = -\frac{1}{N}\sum_{i=1}^{N}\log P(E_{t+1}=y_i\mid\mathbf{x}_{(i)}) + \frac{1}{2CN}\lVert W\rVert_F^2 .
\]
The objective is convex. We use the L-BFGS implementation of scikit-learn 1.9.0, in which
the intercepts are not penalised, and select the inverse penalty strength \(C\) by nested
cross-validation.

The softmax function and cross-validation serve different purposes. The softmax is the
link function that turns scores into probabilities and is part of the model.
Cross-validation is a procedure that estimates how well the whole fitted model
generalises to unseen matches, and it is used to choose hyperparameters such as \(C\).

**Relation to the Markov chain.** If the feature vector contains only the current event,
\(\mathbf{x}_t = \operatorname{OneHot}(E_t)\), the model has a free logit vector for every
current class. Without a penalty, its maximum-likelihood solution reproduces the empirical
transition frequencies \(C_{a,c}/N_a\) exactly: it *is* the first-order Markov chain, with
\(\alpha = 0\). With an \(L_2\) penalty, the rows are shrunk toward a common, data-driven
base distribution \(\operatorname{softmax}(\mathbf{b})\) rather than toward the uniform
distribution used by additive smoothing. The shrinkage is strongest for rarely observed
current events. Consequently, a logistic regression that receives the same information as
the Markov chain should not be expected to outperform it. Its value lies in incorporating
richer information in a controlled way.

## 5. Evaluation strategy

**Data partitioning.** All partitions are match-grouped, because events within a match
are strongly dependent.
- **Final test.** 83 of the 417 usable matches were drawn per competition and season
  with a fixed seed and set aside as a final test set. They have not been used in any
  development step.
- **Development set.** The remaining 334 matches are split into:
  - a 30-match pilot set, used for exploratory analyses;
  - 304 matches (Tier B), used for held-out development evaluation with five match-grouped
    folds.
- **Hyperparameters.** They are selected by nested cross-validation within the training
  folds of each outer fold, so the validation matches of a fold never influence its own
  model choice.

**Training and evaluation rows.** All models are trained on every within-period
transition of the training matches and evaluated on identical rows: the events of the
validation matches that have a 360 freeze frame (701,144 events).

**Metrics.**
- **Primary metric.** The multiclass log loss, a strictly proper scoring rule for the
  predicted distribution.
- **Probability metrics.** The Brier score and calibration error.
- **Ranking metric.** The one-vs-rest average precision for Shot and the other minority
  classes.
- **Decision metrics.** We report accuracy, macro- and weighted-F1, and per-class
  precision, recall and F1 separately, and label them as metrics of the argmax decision
  rule. Cost-sensitive decision rules, when used, are reported separately again.

**Uncertainty.** It is quantified by a match-level bootstrap with 1,000 replicates;
differences between models are bootstrapped in a paired manner.

## 6. Completed experiment: event-only logistic regression (experiment 4)

[Numbers from `results/logistic_regression_f0_2026-11-17/README.md`. All of them are Tier B development estimates, not final-test results.]

As a first discriminative model, we fitted a multinomial logistic regression whose only
input is the one-hot encoded current event (feature set F0, \(d = 15\)). It was compared
with the empirical frequency baseline and the first-order Markov chain (\(\alpha = 1\)) on
identical training transitions, evaluation rows (701,144 events of 304 matches) and
match-grouped folds. The inverse penalty strength was selected by nested cross-validation
from \(C \in \{10^{-6},\dots,10\}\); \(C = 10\) was chosen in all five folds.

The logistic regression and the Markov chain produced practically identical predictions:
- **Log loss.** 1.1519 vs 1.1520 (95 % CI of both ≈ ±0.014), against 1.6592 for the
  frequency baseline. The paired difference in log loss is \(-3.3\times10^{-5}\) (95 % CI
  \([-5.4, -0.9]\times10^{-5}\)). It is statistically detectable because the comparison is
  paired over 701,144 events, but its magnitude (0.003 %) is irrelevant in practice.
- **Decisions.** The argmax decisions coincide for every evaluation event (accuracy
  0.653, macro-F1 0.215).
- **Training-fold tables.** On the training folds, the fitted conditional distributions
  differ from the empirical transition frequencies by at most \(6.5\times10^{-4}\) (for
  the rarest current class, 50/50).
- **Source of the difference.** The remaining difference stems from the treatment of
  transitions never observed in training: 43 of the 225 cells. Additive smoothing assigns
  them a probability of order \(10^{-3}\)–\(10^{-4}\), whereas the mildly penalised
  regression shrinks them toward a data-driven base distribution.
- **Unpenalised fit.** Without any penalty, no finite maximum-likelihood estimate exists
  for these cells, and the unpenalised fit is slightly worse
  (\(+1.3\times10^{-4}\) in log loss).

These results confirm empirically the theoretical equivalence of the two models under
identical information. The choice between a count-based and a softmax model therefore
does not, by itself, improve prediction. Improvements must come from the information
contained in \(\mathbf{x}_t\). The analysis also exposes the limits of the event-only
view:
- **Shot under argmax.** No first-order model ever predicts a Shot under the argmax rule.
- **Shot ranking.** The current event alone ranks shots only weakly (average precision
  0.026 at a prevalence of 0.010).
- **Recording conventions.** The few minority classes that are predicted (Duel after
  Dispossessed, Goal Keeper after Shot, 50/50 after 50/50) are largely companion records
  of the data-collection protocol rather than distinct football actions.

## 7. Planned work: incremental feature representation

[Planned, not yet carried out.]

Building on the event-only baseline, information is added in controlled steps, each
evaluated on the same rows and folds:
- F1: pitch zone of the current event;
- F2: continuous location, distance and angle to goal;
- F5: previous event classes;
- F3: temporal context;
- F4: geometric descriptors of the 360 freeze frame, such as distances to the nearest
  opponents and local player density.

Each step is compared with the previous one by the paired difference in log loss. To
separate the effect of additional information from that of increased model flexibility,
the feature sets are crossed with model families: multinomial logistic regression versus
gradient-boosted trees, and possibly a multilayer perceptron. Recurrent sequence models
are considered only if longer event histories continue to improve the prediction. Before
use, each feature is audited for availability, missingness, temporal validity and
leakage.
