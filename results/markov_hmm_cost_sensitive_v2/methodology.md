# Experiment 2 — methodology

Companion to `README.md`. Code: `src/sequence_models/` (`cost_sensitive.py`, `backoff_markov.py`,
`categorical_hmm.py`, `zones.py`, `evaluation_v2.py`, `experiment_v2.py`, `taxonomy_audit.py`, `report_v2.py`),
`scripts/run_markov_hmm_cost_sensitive_v2.py`, `scripts/verify_markov_hmm_cost_sensitive_v2.py`,
`configs/markov_hmm_cost_sensitive_v2.yaml`.

## 1. Prediction tasks

Both taxonomies define a **filtered-event task**, not a next-raw-event task.

- A sequence is one (match, period) in raw StatsBomb event order (`index`). Periods 1–4 are included; period 5 (the penalty shootout) is removed entirely. Ordinary penalties are kept.
- The sequence keeps only events of the taxonomy. Every other raw event is skipped **by definition**, and the number skipped is recorded per row (`n_skipped_raw_events`).
- **Target:** the next observation of the same sequence.
- **Evaluation rows:** observations of the taxonomy with their own 360 record and a successor in the period.

| Taxonomy | Classes | Source | Notes |
|---|---|---|---|
| `filtered8` | the 8 production types | `MEANINGFUL_EVENT_TYPES` | identical to the production target (verified row by row) |
| `scenarioD15` | 8 + Pressure, Duel, Interception, Block, Goal Keeper, Foul Committed, 50/50 | `results/event_taxonomy_research_2026-10-27` Scenario D | input set = target set. The research's open options are kept at the measured defaults: Dispossessed → Duel not skipped, Goal Keeper as input and target, 50/50 a class of its own. Verified against the research script on Tier A (73,152 rows, identical targets). |

Log loss, F1 and similar metrics of the two taxonomies measure **different tasks** and are never ranked against each other.

## 2. Data partitions

| Set | Matches | Role |
|---|---|---|
| Final test | 83 | experiment 1's hold-out. **Never read** by experiment 2: every loader checks match lists against it, and experiment 1's all-match cache is refused. |
| Tier A | 30 | experiment 1's pilot matches and its 5 match folds. Exploration and **tuning**: HMM K, HMM initialisation type, cost exponent of the tuned scheme, rare-class list. |
| Tier B | 304 | the remaining development matches; 5 new match-grouped folds (StratifiedGroupKFold, seed 20261027). **Held-out development evaluation**: every tuned choice is frozen from Tier A. |

Why two tiers? Fully nested CV for the HMM grid (8 values of K × 3 emissions × 6 initialisations × 5 folds × 2 taxonomies)
would multiply about 1,400 fits by roughly 4, and Tier B fits take about 3 minutes each on this machine. The two tiers
separate tuning and evaluation groups instead.

The cheap Markov models are tuned by genuinely **nested** CV. Inside each outer training split, the Dirichlet concentration
is chosen by leave-one-inner-fold-out log loss.

Within Tier A:
- **K per fold:** chosen by a cross-fold rule. K for fold *f* minimises the mean validation log loss of the *other* folds. This is not fully nested: fold *f*'s matches are in the training data of the other folds' models.
- **Initialisation per (fold, K):** chosen by the training objective.

## 3. Models

### 3.1 Markov chains with nested contexts (`backoff_markov.py`)

Contexts are known right after event *t*:
- `cur` — the type of event *t*;
- `prev` — the type of event *t−1*, or START;
- `zone` — the pitch third of event *t*, or UNKNOWN.

| Model | Context chain (most specific first) |
|---|---|
| `mc1` | (cur) → marginal |
| `mc2` | (prev, cur) → (cur) → marginal |
| `mc1_zone` | (cur, zone) → (cur) → marginal |
| `mc2_zone` | (prev, cur, zone) → (cur, zone) → (cur) → marginal |

Hierarchical Dirichlet (interpolated backoff) smoothing (MacKay & Peto 1995):

$$p_0(k)=\frac{N_k+\alpha_0}{N+V\alpha_0},\qquad p_\ell(k\mid c)=\frac{C_\ell(c,k)+\beta\,p_{\ell-1}(k\mid\pi(c))}{N_\ell(c)+\beta}$$

- $\alpha_0 = 1$.
- $\beta \in \{1, 4, 16, 64, 256\}$, chosen by nested inner CV.
- A rare or unseen context falls back to its parent, so spatial contexts cannot create unsupported probabilities.

The experiment-1 chains (Laplace 1st order; 2nd order with hard backoff below 20 contexts) are re-run as anchors
(`exp1_markov_order1/2`). On Tier A they reproduce experiment 1's log loss exactly.

### 3.2 Pitch thirds (`zones.py`)

- **Zones:** relative to the acting team's attacking direction: defensive 0 ≤ x < 40, middle 40 ≤ x < 80, attacking 80 ≤ x ≤ 120.
- **Source:** the raw event `location`. The StatsBomb spec defines it in the acting team's attacking frame. For defensive actions (Pressure, Duel, …) the acting team is the defending team.
- **Validation:** agreement with the zone of the approved causal 360 actor point (experiment 1 transform) is **99.97%** on 628,526 development production rows (`audit_checks.json`).
- **Why this source:** it needs no orientation inference, it is causal (the current event's own attribute), and it exists for the 7 Scenario D types that are absent from the production dataset.
- **Edge cases:** two shots at x = 120.1 / 120.5 lie within a 1-unit tolerance and are assigned to the attacking third. Missing location → UNKNOWN (0.001% of rows).
- **Approval status:** `zone` is a new derived input. Like experiment 1's spatial features, it is pending your approval for the allowlist.
- **Interpretation:** zones are spatial bins, not tactical phases.

### 3.3 Redesigned spatial HMM (`categorical_hmm.py`)

**Why experiment 1's spatial HMM failed.**
- It combined a categorical event probability with Gaussian **densities** over continuous coordinates. A density's scale depends on the units: two standardised dimensions contributed about −3.0 nats per observation against −0.7 for the event part.
- EM maximises the joint likelihood, so the states were spent on explaining location, and the predictive event distribution collapsed to the marginal.
- This is a property of the observation model, not a bug (experiment 1, §8.5).

**Redesign: every observation component is a probability of a discrete outcome.**

| Emission | $p(o_t \mid z_t=k)$ | Assumption | Free emission parameters |
|---|---|---|---|
| `event` | $B^E_{k,e}$ | — | $K(V-1)$ |
| `factorized` | $B^E_{k,e}B^Z_{k,g}$ | event ⟂ zone given state | $K(V-1)+K(Z-1)$ |
| `joint` | $B^{EZ}_{k,e,g}$ | none | $K(VZ-1)$ |

- A missing zone is marginalised exactly. The joint M-step distributes expected counts over zones in proportion to $B^{EZ}$ (EM for missing data).
- No likelihood term is re-weighted; these are ordinary generative HMMs.
- A conditional (input-output) HMM (Bengio & Frasconi 1995) was considered but not implemented. The zone-conditioned Markov chain already measures the zone's conditional value, and an IOHMM adds K × 3 transition matrices.

**Causal next-event prediction** (the zone at *t+1* is unknown and summed out):

$$p(e_{t+1}=v\mid o_{1:t})=\sum_j\Big(\sum_i\hat\alpha_t(i)A_{ij}\Big)M_{jv},\qquad M=B^E\ \text{or}\ \textstyle\sum_g B^{EZ}_{\cdot,\cdot,g}$$

with the filtered posterior $\hat\alpha_t=p(z_t\mid o_{1:t})$ (scaled forward recursion; Rabiner 1989).

**Training and selection.**
- Baum–Welch on independent, padded and masked sequences (never concatenated).
- Dirichlet MAP pseudo-count 1.
- Initialisations: 3 seeds × 2 initialisation types: `init_self_transition` 0.5 (experiment 1, persistence-biased) and 0.0 (unbiased).
- Stopping: per-observation improvement < 1e-6, at most 1,000 iterations.
- With `event` emissions and the 0.5 initialisation, the model is identical to experiment 1's HMM (tested to 0.0 difference).
- K ∈ {2, 3, 4, 5, 6, 8, 10, 12} on Tier A.
- Tier B uses the K with the lowest mean Tier A validation log loss, and the initialisation type that most often had the best **training** objective on Tier A, with all 3 seeds.

## 4. Cost-sensitive learning

### 4.1 Approach A — Bayes-optimal decisions (implemented)

The model's predictive distribution $p_k=P(Y=k\mid x)$ is **unchanged**. The decision minimises the expected cost
(Elkan 2001; Duda, Hart & Stork 2001, ch. 2):

$$\hat y(x)=\arg\min_a R(a\mid x),\qquad R(a\mid x)=\sum_k C(a,k)\,p_k$$

Ties within $10^{-12}$ go to the lowest class index (vocabulary order: Pass first).

**Cost family.** For $a\neq k$, $C(a,k)=m_k+f_a$, with $C(k,k)=0$ and

$$m_k=\min\big(\text{cap},(\pi_{\max}/\pi_k)^{\gamma}\big)$$

- $\pi$ is the Laplace-smoothed class prior of the fold's **training** evaluation rows.
- $m_k$ is the cost of *missing* a true $k$ (false negative). $f_a$ is the extra cost of a *false alarm* for $a$ (false positive). $f = 0$ in all schemes here.
- **Unit:** one cost unit is the cost of missing the most frequent class (Pass).
- With $f = 0$ the rule is $\arg\max_a m_a p_a$: a rare class $r$ beats a frequent class $j$ iff $p_r/p_j > m_j/m_r$. Mathematically this is a prior-ratio threshold. It is derived here from an explicit cost matrix, not searched as a free threshold.

**Schemes.**

| Scheme | γ | cap | Meaning |
|---|---|---|---|
| uniform | 0 | 1 | 0–1 loss, i.e. argmax (reference) |
| moderate | 0.5 | 5 | square-root inverse frequency, at most 5× |
| strong | 1.0 | 20 | inverse frequency (the balanced-error Bayes rule), bounded at 20× |
| tuned | γ* ∈ {0, .25, .5, .75, 1} | 20 | γ* maximises macro-F1 on **Tier A** out-of-fold predictions, applied unchanged to Tier B |

**No real-world utility.** There is no external utility for misclassifying football events (no betting, coaching or scouting
loss function was specified). The predefined schemes are therefore a **sensitivity analysis**, not a claim of optimal costs.
The tuned scheme makes its objective explicit: macro-F1 on the tuning tier.

### 4.2 Approach B — cost-sensitive training (investigated, not adopted)

**Markov chain.**
- Training on class-weighted counts $w_kC_{ik}$ (weights also on the pseudo-counts) gives the prior-shifted estimator
  $$q(k\mid i)=\frac{w_k\,p(k\mid i)}{\sum_j w_j\,p(j\mid i)}$$
  (Elkan 2001, Theorem 1; Saerens, Latinne & Decaestecker 2002).
- Its argmax equals Approach A's decision with $m=w$, $f=0$, exactly (tested). So the decisions gain nothing, while $q$ is miscalibrated for the real class frequencies.
- Measured in `approach_b_check.csv`: decision agreement 100%, while log loss and calibration-in-the-large deteriorate.

**HMM.** Weighting the emission likelihood of observations by class has no generative interpretation:
- the weighted "likelihood" is not a probability model of the sequence;
- the E-step posteriors would be computed under one model and the M-step would optimise another;
- the weighting would distort the latent dynamics, i.e. the transition matrix that carries the sequential information.

**Conclusion:** cost sensitivity belongs in the decision layer for these generative models.

### 4.3 What changes and what does not

| Changes with the cost scheme | Unchanged by the cost scheme |
|---|---|
| decisions, confusion matrices, precision/recall/F1, macro-F1, balanced accuracy, realised cost | probabilities, log loss, Brier score, all calibration diagnostics |

Both columns are reported for every scheme, and the invariance is tested.

## 5. Metrics

- **Probabilities (primary):** multiclass log loss (natural log) and multiclass Brier score $\frac1n\sum_i\sum_c(p_{ic}-1[y_i=c])^2$, both on the original, unbalanced validation distribution.
- **Calibration** (diagnostic; nothing is recalibrated):
  - classwise ECE (Kull et al. 2019): 10 equal-width bins per class, averaged over classes;
  - top-label ECE (Guo et al. 2017);
  - per-class calibration-in-the-large $\bar p_k-\bar y_k$;
  - reliability tables.
- **Decisions:**
  - accuracy, macro- and weighted-F1, balanced accuracy (mean recall over classes present);
  - per class: precision, recall, F1, support, number predicted, TP/FP/FN, lift = precision / prevalence;
  - confusion matrices;
  - predicted-class frequencies.
- **Costs:**
  - realised cost $\frac1n\sum_i C(\hat y_i,y_i)$ under the scheme's own matrix and under every other scheme's matrix (`cross_cost.csv`);
  - model-implied expected cost.
- **Aggregation:**
  - pooled out-of-fold over evaluation rows;
  - fold-level values in `metrics_by_fold.csv`.
- **Uncertainty:**
  - match-level (cluster) bootstrap, 1,000 replicates, computed from per-match confusion matrices and log-loss/Brier sums (identical to recomputing on resampled rows; tested);
  - paired differences use the same resampled matches.
- **Rare events:**
  - a class is "rare" if its Tier A share is below 5%: 6 classes in `filtered8`, 12 in `scenarioD15`;
  - for each rare class: recall/precision with CIs, recall gain over the uniform rule with CI, extra false positives, and the number of matches (with ≥ 3 occurrences) in which recall improved or worsened.

## 6. References

- Bengio, Y. & Frasconi, P. (1995). An input output HMM architecture. *NIPS 7*.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1).
- Decroos, T., Bransen, L., Van Haaren, J. & Davis, J. (2019). Actions speak louder than goals: valuing player actions in soccer. *KDD*.
- Duda, R. O., Hart, P. E. & Stork, D. G. (2001). *Pattern Classification* (2nd ed.), ch. 2 (Bayesian decision theory). Wiley.
- Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*.
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, H. & Flach, P. (2019). Beyond temperature scaling: obtaining well-calibrated multiclass probabilities with Dirichlet calibration. *NeurIPS*.
- MacKay, D. J. C. & Peto, L. C. B. (1995). A hierarchical Dirichlet language model. *Natural Language Engineering* 1(3).
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proc. IEEE* 77(2).
- Saerens, M., Latinne, P. & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities. *Neural Computation* 14(1).
- Field, C. A. & Welsh, A. H. (2007). Bootstrapping clustered data. *JRSS B* 69(3).
