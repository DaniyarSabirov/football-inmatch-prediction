# Feature engineering roadmap and model plan (K = 15)

Status: plan, 2026-11-17. Nothing in sections 2–5 has been run yet unless marked
**[done]**. The implemented code is the F0 feature set (`src/sequence_models/feature_sets.py`)
and the multinomial logistic regression (`src/sequence_models/logistic.py`). Notation follows
`02_mathematical_framework.md`.

## 1. Principle: vary information and model complexity separately

The thesis asks two questions that are easily confounded:

1. **Information.** How much does each kind of information (zone, coordinates, time, 360
   geometry, history) improve \(P(E_{t+1}\mid\mathbf{x}_t)\)?
2. **Complexity.** How much does a more flexible model improve it *for the same
   information*?

**Answer to Question F.** A factorial design over feature sets × model families, evaluated
on identical rows and folds (§5).
- **Feature gain.** The difference between two feature sets *within* one model family.
- **Complexity gain.** The difference between two model families *within* one feature
  set.
- **Interaction.** A feature that helps only a flexible model, because the needed
  nonlinearity is otherwise absent, shows up as an interaction.
- **F0 anchor.** Theory predicts that every model family ties on F0, because all of them
  can represent the 15 × 15 table, and experiment 4 confirms this for the first two
  (`results/logistic_regression_f0_2026-11-17/`). Any F0 gap is therefore a bug or a
  smoothing effect, not a feature effect.

## 2. Rules every feature must satisfy before use

For each feature, these are recorded in a feature card (one row of `feature_audit.csv` per
experiment):

| Item | Requirement |
|---|---|
| Availability | which raw field; share of evaluation rows where it exists |
| Missingness | encoded explicitly: an "unknown" category or 0 + missing indicator; the column set never depends on the fold |
| Temporal validity | computed from \(E_t\), its own 360 snapshot and earlier records only (`configs/ml_feature_classification.json`) |
| Leakage | not derived from \(E_{t+1}\): `n_skipped_raw_events`, `time_to_next_event_seconds`, `next_*` and StatsBomb `possession` ids are forbidden (`FORBIDDEN_COLUMNS`); verified by a truncation test (cut the stream after \(E_t\); the feature must not change) |
| Encoding / normalisation | one-hot for categories; continuous features standardised with training-fold statistics only |
| Effect | paired Δ log loss vs. the previous feature set, 95 % match-bootstrap CI; Shot AP; per-class recall |

## 3. Feature groups (Question D)

Ranked by expected value per unit of effort and risk. The evidence column cites earlier
results (experiments 2 and 3, same development data).

| Set | Content | Evidence so far | Risks |
|---|---|---|---|
| **F0** | one-hot \(E_t\), d = 15 | **[done]** experiment 4 | — |
| **F1** | pitch third / 6 × 2 grid of the \(E_t\) location (one-hot, plus unknown) | exp. 2: thirds −0.03 log loss for Markov; exp. 3: 6 × 2 grid best, Shot AP 0.16 → 0.30 [Tier B] | location missing for some types; zone boundaries arbitrary |
| **F2** | F1 + continuous \(x\), \(y\), distance and angle to the attacked goal; interactions \(E_t\) × distance | exp. 3 LR with geometry: best Shot AP (0.36) | the angle must be the opening angle of the goal mouth, defined in the attacking direction; StatsBomb event coordinates are already in the acting team's direction (verified: 99.85 % of shots at x ≥ 60); defending-team events (Pressure, Duel) sit at the far end, so add a "possession frame" or same-team indicator |
| **F3** | period (one-hot), minute (scaled), \(\log\) time since the previous event | exp. 3: \(\Delta t\) small gain | only the time since the *previous* event; the time to the next event is the target's |
| **F5** | \(E_{t-1}\), \(E_{t-2}\) one-hot (+ start-of-period category); same-team-as-previous flag | exp. 2/3: order 2 > order 1 (−9 % log loss), order 3 > 2; same-team flag −0.9 % | history must be shifted *within* the sequence; the team of \(E_t\) is known at \(E_t\) |
| **F4** | 360 geometry of the current snapshot: distance from the actor to the nearest opponent, opponents within 5 m / 10 m, teammates ahead of the ball, opponents between the ball and the goal, keeper distance, visible-area share | not yet measured | team identity must be the **causal** one; orientation unavailable in 41 rows (missing indicator); the visible area is partial (players outside it are unobserved, not absent), so counts must be normalised by, or conditioned on, the visible area; snapshots are event-triggered, not continuous tracking; actor ambiguity |
| **F6** | score state (from earlier goals), home/away; team strength | not measured | team identity features let models memorise teams; team strength needs earlier-season data only (optional future work) |

**Recommended order.**
- **Measure first:** F1 → F2 → F5 → F3. These give cheap, well-understood gains that
  experiment 3 already observed for count models.
- **Then F4.** This is the novel 360 contribution of the thesis. It is worth doing after
  F2/F5, so that its gain is measured *on top of* location and history and is not
  confused with them.

**Missingness of 360 features.** Evaluation rows require a 360 record by construction, but
a freeze frame can still lack the actor (actor ambiguity) or the orientation. Each F4
feature therefore gets a missing indicator.

**Training rows.** The F0 experiment trains on all transitions, including 98,165
current events without 360 data, exactly as the Markov chain does. From F4 on, training
rows must be restricted to rows that *have* the features. The rule: whenever a feature set
changes the training population, refit the reference model on the same restricted
population, so the comparison stays paired.

## 4. Model families (Question E)

| Model | Strengths for this task | Limitations for this task |
|---|---|---|
| **Multinomial logistic regression** | convex, deterministic, fast (seconds on 640k rows); calibrated in the large by construction; coefficients are interpretable log-odds; reproduces the Markov chain on F0 (theory + exp. 4); scales additively with features | linear in the features: interactions (event × location, distance × angle) must be engineered; rare classes get flat probabilities under L2 |
| **Gradient-boosted trees (XGBoost / LightGBM)** | learns interactions and thresholds automatically; robust to scale; native missing values; strong on tabular data of this size | many hyperparameters; multiclass probabilities need calibration checks; 15 classes → 15 trees per round; feature importances are not causal |
| **Multilayer perceptron** | smooth nonlinear interactions; embeddings for categorical inputs; a softmax output gives the same probabilistic interface | needs scaling, early stopping and seeds (non-deterministic); usually no better than boosting on tabular data; harder to explain in a thesis |
| **LSTM / GRU (sequence model)** | learns the history representation instead of hand-picked \(E_{t-1}, E_{t-2}\); handles variable-length context | needs many sequences (we have ~1,200 Tier B periods); expensive tuning; it is unclear what it learns beyond order-3 Markov context; risk of reading across periods if batching is wrong |

**Plan.** Logistic regression on F0–F5 first, with gradient boosting as the main
"complexity" comparison. An MLP is optional. An LSTM only if F5 shows that longer history
keeps helping: the order-3 > order-2 result of experiment 3 suggests it might.

## 5. Experimental design (Question F)

- **Rows and folds.** Tier B, 5 match folds (experiment 2's `folds.json`); the same
  evaluation rows for every cell of the design.
- **Hyperparameters.** Nested inner cross-validation on the outer training folds; Tier A
  (30 matches) only for exploratory checks. The final-test set stays untouched until an
  explicitly approved final run.
- **Design.** Feature sets {F0, F1, F2, F2+F5, F2+F5+F3, +F4} × models {LR, GBM, (MLP)}.
- **Primary metric.** Log loss, with paired match-bootstrap CIs of Δ between neighbouring
  cells.
- **Secondary metrics.** Brier, ECE, Shot AP and per-class AP (ranking), and argmax
  accuracy / macro-F1 / per-class recall (decisions). Cost-sensitive decisions are
  reported separately.
- **Interpretation.**
  - Δ along rows = information gain.
  - Δ along columns = complexity gain.
  - Δ(GBM − LR) growing with richer feature sets = interaction or nonlinearity in the
    features.
- **Size-controlled check.** To separate "more parameters" from "more information", add a
  *placebo* feature: a random permutation of the new feature within the training fold.
  Its gain must be ≈ 0.
- **Taxonomy caveat.** `scenarioD15` keeps StatsBomb's companion records (e.g.
  `Dispossessed → Duel` with P = 1.0). Experiment 3 recommended `actions15` (the same 15
  class names, with companion records skipped) as the cleaner target. This document
  follows the 2026-11-17 decision to use `scenarioD15`. The choice should be confirmed
  with the supervisor, because it changes what "Duel", "Goal Keeper" and "50/50" mean.

## 6. Next experiment (proposal)

**Experiment 5: F1 and F2 with logistic regression,** on the same rows, folds and protocol
as experiment 4.
- **Feature sets.** F1 = one-hot \(E_t\) + one-hot 6 × 2 zone (+ unknown); F2 = F1 +
  standardised \(x\), \(y\), distance, goal-mouth angle, \(E_t\) × distance and
  \(E_t\) × angle interactions.
- **Comparisons.** F1 vs. the zone-conditioned Markov chain (`mc1_zone`, same information)
  is again an equivalence check. F2 vs. F1 measures what continuous geometry adds beyond
  a grid.
- **Estimated cost.** Under one hour of compute and about a day of implementation, plus
  feature cards and leakage tests for each new column.

## References

Search-verified on 2026-11-17 (bibliographic record and abstract only, full text not read):
- Simpson, I., Beal, R. J., Locke, D., Norman, T. J. (2022). Seq2Event: Learning the
  language of soccer using transformer-based match event prediction. *KDD '22*,
  3898–3908. doi:10.1145/3534678.3539138.
- Decroos, T., Bransen, L., Van Haaren, J., Davis, J. (2019). Actions speak louder than
  goals: Valuing player actions in soccer. *KDD '19*, 1851–1861.
- Mendes-Neves, T., Meireles, L., Mendes-Moreira, J. (2024). Forecasting events in soccer
  matches through language. arXiv:2402.06820.

Cited from bibliographic knowledge (not re-accessed; verify before submission):
- Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*,
  785–794.
- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
  *NeurIPS 30*.
- Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*,
  9(8), 1735–1780.
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still
  outperform deep learning on typical tabular data? *NeurIPS Datasets and Benchmarks*.
- Fernández, J., Bornn, L. (2018). Wide open spaces: A statistical technique for measuring
  space creation in professional soccer. *MIT Sloan Sports Analytics Conference*.
- Rudd, S. (2011). A framework for tactical analysis and individual offensive production
  assessment in soccer using Markov chains. *New England Symposium on Statistics in
  Sports*.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural
  networks. *ICML*.
