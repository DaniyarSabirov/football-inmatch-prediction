# Experiment 2 — cost-sensitive decisions, Scenario D, pitch thirds and a redesigned spatial HMM

Date: 2026-10-28. **Development data only.**
- The 83-match final test set was not read.
- Experiment 1 (`results/markov_hmm_pilot/`) is unchanged: its 58 files were re-verified against `checksums.json`.
- The production dataset, taxonomy constant and feature allowlist were not modified.
- No commit was made.
- Mathematical definitions and references: `methodology.md`.
- Every number below is copied from a file in this directory.

**Labels:** [V] verified, [D] decision, [H] hypothesis or interpretation, [U] unresolved, [E] exploratory (Tier A),
[C] held-out development result (Tier B).

## 1. Summary

| Question | Answer |
|---|---|
| Data ready? | **Yes, with limitations.** Scenario D is reproduced exactly from the research definition. Pitch thirds agree with the causal 360 actor point in 99.97% of rows. |
| Does cost-sensitive decision-making recognise rare events? | **Partly, and at a price.** On Tier B, the moderate scheme raises macro-F1 from 0.211 to 0.282 (8 classes) and from 0.275 to 0.330 (15 classes) for the best model (`mc2_zone`); all CIs exclude 0. But false positives multiply, accuracy drops, and several classes stay unrecognised under every scheme. |
| Are probabilities preserved? | **Yes, by construction and verified.** Log loss, Brier score and calibration are identical across cost schemes (max difference 0.0). Cost-sensitive *training* (Approach B) yields exactly the same decisions but doubles the log loss, so it was not adopted. |
| Scenario D vs 8 classes | **Scenario D is scientifically preferable** (§3): 38.5% vs 59.3% of rows skip raw events, and the defensive and goalkeeper actions are visible. Its scores are not comparable with the 8-class task. |
| Do pitch thirds help? | **Yes, in Markov chains.** Log loss improves by 0.027–0.033 nats on Tier B in all four (taxonomy × order) ablations; every CI excludes 0 and every fold agrees. |
| Does the redesigned spatial HMM help? | **No.** The categorical zone HMMs no longer collapse like experiment 1's Gaussian HMM, but they are worse than the event-only HMM: +0.009 to +0.012 log loss (8 classes), +0.146 to +0.150 (15 classes). |
| Best model | **`mc2_zone`** (2nd-order chain conditioned on the pitch third) on both tiers and both taxonomies. Tier B log loss 0.664 (8 classes), 1.019 (15 classes). |
| Hidden states = tactical phases? | **No.** Event-only states are event-type proxies with self-transition ≈ 0. Zone HMM states are (event type × third) cells. |

## 2. Setup

- **Tiers** ([D]; `split_manifest.json`):
  - **Tier A:** experiment 1's 30 pilot matches and folds. Used for exploration and for choosing the HMM K, the HMM initialisation type, the cost exponent γ of the tuned scheme, and the rare-class list.
  - **Tier B:** the other 304 development matches, with 5 new match-grouped folds. Every Tier A choice is frozen.
  - Markov smoothing is chosen by nested inner CV in both tiers.
  - Fully nested CV of the HMM grid was infeasible: about 1,400 fits × 4, at up to ~3 minutes per Tier B fit.
- **Within one (tier, taxonomy):** identical evaluation rows and folds for every model and cost scheme.
- **Tasks:** both taxonomies are filtered-event tasks (§1 of `methodology.md`). They predict the next event *of the taxonomy*, not the next raw event.
- **Evaluation rows:**

  | | filtered8 | scenarioD15 |
  |---|---|---|
  | Tier A | 59,655 | 73,130 |
  | Tier B | 568,908 | 701,144 |

- **filtered8 integrity:** its rows and targets equal the production dataset on both tiers (`leakage_checks.json`).

## 3. Taxonomy: 8 vs 15 types (`taxonomy_comparison.csv`, `class_distribution.csv`; 334 development matches)

| | filtered8 | scenarioD15 |
|---|---|---|
| Evaluation rows | 628,563 | 774,274 |
| Raw events in the taxonomy | 57.6% | 71.2% |
| Rows skipping ≥ 1 raw event | 59.3% | **38.5%** |
| Mean skipped raw events per row | 0.72 | 0.39 |
| Observations with 360 / targets without own 360 | 88.0% / 8.3% | 87.7% / 7.9% |
| Majority class | Pass 48.7% | Pass 39.2% |
| Smallest class | Dispossessed 7,007 | 50/50 1,179 (in 239 of 334 matches) |
| Imbalance ratio | 43.7 | 257.1 |
| Classes < 5% / < 1% | 6 / 0 | 12 / 4 |
| H(target) → H(target \| current) → H(target \| current, zone) → H(target \| previous, current), bits | 1.662 → 1.011 → 0.971 → 0.991 | 2.391 → 1.655 → 1.607 → 1.502 |
| Rows in near-deterministic cells (P ≥ 0.9) | 0% | 0.9% (Dispossessed → Duel, P = 0.9996, 6,721 rows) |
| Parameters MC1 / MC2 / MC2+zone | 56 / 504 / 2,016 | 210 / 3,360 / 13,440 |

**Implementation check [V]:** on Tier A, my Scenario D rows and targets equal the research script's own routine (73,152 rows incl. period 5).

**Recommendation [D]:** Scenario D as the thesis's primary taxonomy.
- It removes 21 percentage points of skipped events.
- It represents defending (Pressure 12%, Duel, Block, Interception, Foul Committed) and goalkeeping.
- It is the less mechanical task (higher conditional entropy).
- **Caveats:**
  - Duel, Goal Keeper and 50/50 are partly *mechanically* predictable: Dispossessed → Duel, Shot → Goal Keeper, paired 50/50s. Duel precision is 0.94 under argmax.
  - Their scores partly measure StatsBomb's recording conventions.
  - The research's option (b) for Dispossessed → Duel is still open [U].

## 4. Cost-sensitive learning

### 4.1 Method (details: `methodology.md` §4)

- **Approach A — implemented.** The probabilities are unchanged; the prediction is $\hat y=\arg\min_a\sum_k C(a,k)p_k$.
- **Costs:** $C(a,k)=m_k$ for $a\neq k$, with $m_k=\min(\text{cap},(\pi_{\max}/\pi_k)^\gamma)$ from training-fold priors (`cost_matrix_definitions.json`, `figures/cost_matrices.png`).
- **Unit:** one unit = the cost of missing a Pass.
- **Schemes:**
  - uniform (argmax);
  - moderate (γ = 0.5, cap 5);
  - strong (γ = 1, cap 20);
  - tuned: γ chosen per model on Tier A by macro-F1, frozen for Tier B. For `mc2_zone`: 0.75 (8 classes), 0.5 (15 classes).
- **No real-world utility:** the predefined schemes are a **sensitivity analysis**, not optimal costs.
- **Approach B — investigated, rejected** (`approach_b_check.csv`).
  - Class-weighted training of the Markov chain gives **100%** identical decisions to Approach A on all 4 (tier × taxonomy) cells.
  - But log loss rises from 0.704 to 1.331 (Tier B, 8 classes) and from 1.152 to 1.781 (15 classes).
  - Calibration-in-the-large worsens from about 0.015 to about 0.3.
  - For the HMM, weighted likelihoods have no generative interpretation.

### 4.2 Results: Tier B [C], `mc2_zone` (`cost_sensitive_comparison.csv`; 95% match-bootstrap CIs)

| Taxonomy | Scheme | Accuracy | Macro-F1 [CI] | Balanced acc. [CI] | Classes predicted | Log loss | Brier |
|---|---|---|---|---|---|---|---|
| 8 | uniform | 0.790 | 0.211 [0.210, 0.212] | 0.226 [0.225, 0.227] | 3 | 0.664 | 0.327 |
| 8 | moderate | 0.784 | **0.282** [0.280, 0.284] | 0.282 | 5 | 0.664 | 0.327 |
| 8 | strong | 0.559 | 0.262 | **0.379** | 8 | 0.664 | 0.327 |
| 8 | tuned (γ = 0.75) | 0.681 | **0.293** [0.291, 0.295] | 0.353 | 7 | 0.664 | 0.327 |
| 15 | uniform | 0.672 | 0.275 [0.274, 0.277] | 0.260 | 13 | 1.019 | 0.479 |
| 15 | moderate | 0.640 | **0.330** [0.328, 0.332] | 0.342 | 13 | 1.019 | 0.479 |
| 15 | strong | 0.453 | 0.298 | **0.398** | 15 | 1.019 | 0.479 |
| 15 | tuned (γ = 0.5) | 0.638 | 0.321 | 0.344 | 13 | 1.019 | 0.479 |

- **Probabilities are untouched:** log loss and Brier are identical in every row of a model, as the invariant requires.
- **Macro-F1 gains vs uniform** (paired, same matches) are positive with CIs excluding 0 for every Markov model and scheme. For example, `mc2_zone` moderate: +0.071 [0.069, 0.073] (8 classes) and +0.055 [0.053, 0.057] (15 classes).
- **Not universal:** for the event-only HMM on 15 classes, the strong scheme *lowers* macro-F1 (−0.014 [−0.015, −0.013]).
- **A degenerate case:** the frequency baseline under the strong scheme predicts one class for every row (Duel; accuracy 0.023). With p ≈ π, the rule is driven by tiny differences between two prior estimates. Cost-sensitive decisions need informative probabilities.
- **Bayes optimality holds empirically [V]** (`cross_cost.csv`). Under each scheme's own cost matrix, that scheme's decisions have the lowest realised cost among all rules. For example, 8 classes under the strong matrix: strong 1.577, tuned 1.614, moderate 1.763, uniform 1.951. This is consistent with well-calibrated probabilities.

### 4.3 Rare events (`rare_event_analysis.csv`, `figures/rare_event_recall.png`, `figures/precision_recall_tradeoff.png`; Tier B, `mc2_zone`)

**Recall** (95% CIs in the CSV). Values are uniform → moderate → strong, with the extra false positives under the strong scheme in parentheses.

| Class | 8 classes: recall | 8 classes: strong precision | 15 classes: recall | 15 classes: strong precision |
|---|---|---|---|---|
| Ball Recovery | 0.006 → 0.360 → 0.599 (+70,823 FP) | 0.18 | 0.122 → 0.583 → 0.575 | 0.27 |
| Clearance | 0 → 0.057 → 0.526 (+58,608 FP) | 0.08 | 0.178 → 0.322 → 0.590 | 0.10 |
| Shot | 0 → 0.086 → 0.178 | 0.14 | 0.040 → 0.177 → 0.399 | 0.12 |
| Dribble | 0 → 0 → 0.554 (+69,138 FP) | 0.06 | 0 → 0.296 → 0.569 | 0.09 |
| Dispossessed | 0 → 0 → 0.010 | 0.07 | 0 → 0.103 → 0.111 | 0.11 |
| Miscontrol | 0 → 0 → 0.000 | — | 0.000 → 0.006 → 0.009 | 0.15 |
| Duel | — | — | 0.396 → 0.581 → 0.722 | 0.21 (0.94 under uniform) |
| Block | — | — | 0 → 0.000 → 0.257 | 0.06 |
| Interception | — | — | 0 → 0 → 0.055 | 0.06 |
| Foul Committed | — | — | 0 → 0 → 0.125 | 0.08 |
| Goal Keeper | — | — | 0.839 → 0.840 → 0.841 | 0.77 |
| 50/50 | — | — | 0.496 → 0.496 → 0.497 | 0.84 |

- **Become predictable [C]:**
  - Ball Recovery, Clearance, Shot and Dribble reach meaningful recall under the moderate or strong schemes.
  - Under moderate costs, precision stays 8–17× above prevalence (8 classes) and 3–35× (15 classes, excluding the mechanical Goal Keeper and 50/50) (`lift_over_prevalence`).
  - The recall gains are consistent across matches. Ball Recovery recall improved in 304/304 matches; Shot (15 classes, moderate) in 276/304; no match got worse.
- **Remain unrecognised:**
  - Miscontrol (recall ≤ 0.01 everywhere), Interception, Foul Committed and Dispossessed (≤ 0.17).
  - Given only event type, previous event and pitch third, the model has no signal that separates them. Raising their cost buys recall only with precision near 0.06–0.10.
- **Trade-off:**
  - Under the strong scheme, rare-class predictions become largely false alarms: precision 0.06–0.18 for most rare classes.
  - The predictions are still better than chance (lift 4–14), but they are rarely right in absolute terms.
  - The moderate scheme is the more useful operating point under realistic class frequencies.
- **Not genuine recognition:** Goal Keeper and 50/50 are recognised under every scheme. Their recall reflects recording conventions (Shot → GK, paired 50/50s), not anticipation.

### 4.4 Calibration (`calibration_metrics.csv`, `figures/calibration_*.png`) [C]

- Tier B, `mc2_zone`: classwise ECE 0.0027 (8 classes) / 0.0012 (15 classes); top-label ECE 0.008 / 0.004; max per-class calibration-in-the-large 0.010 / 0.007.
- On Tier B, all Markov and HMM models have classwise ECE ≤ 0.0063.
- Reliability curves lie close to the diagonal for frequent classes. The high-probability Shot bins are sparse and noisy.
- No recalibration was fitted.

## 5. Spatial models

### 5.1 Pitch-third ablations (`spatial_ablation.csv`, `figures/spatial_model_comparison.png`)

Paired log-loss difference (negative = the zone model is better), 95% match-bootstrap CI:

| Comparison | Tier A 8 [E] | Tier A 15 [E] | Tier B 8 [C] | Tier B 15 [C] |
|---|---|---|---|---|
| mc1_zone − mc1 | −0.024 [−0.026, −0.022] | −0.028 [−0.031, −0.026] | **−0.028 [−0.028, −0.027]** | **−0.033 [−0.034, −0.032]** |
| mc2_zone − mc2 | −0.023 [−0.025, −0.021] | −0.020 [−0.023, −0.018] | **−0.027 [−0.028, −0.027]** | **−0.030 [−0.030, −0.029]** |
| hmm_factorized − hmm_event | +0.009 | +0.145 | +0.012 [0.010, 0.013] | +0.146 [0.144, 0.149] |
| hmm_joint − hmm_event | +0.002 (CI incl. 0) | +0.150 | +0.009 [0.008, 0.010] | +0.150 [0.147, 0.153] |

- **Statistically robust [C]:**
  - The zone gain in the Markov chains holds in every tier, taxonomy and order.
  - All 5 Tier B folds agree (`metrics_by_fold.csv`): `mc2_zone` 0.663–0.665 vs `mc2` 0.691–0.692 (8 classes).
  - It is consistent with the descriptive entropy reduction in §3.
  - The effect is small, about 4% of the log loss.
- **Effect on argmax decisions:** with 8 classes the zone barely moves them (uniform macro-F1 +0.002). With 15 classes it raises uniform macro-F1 by +0.016 (1st order) and +0.027 (2nd order). Its main value is in the probabilities and in better cost-sensitive decisions.
- **Previous event vs zone:** the previous event matters more than the zone for Scenario D (mc2 − mc1 = −0.103) but less for the 8-class task (−0.013).

### 5.2 Redesigned spatial HMM

- **Experiment 1's failure (documented in `methodology.md` §3.3):** Gaussian densities dominated the likelihood (−3.0 vs −0.7 nats per observation), so EM spent the states on location.
- **The categorical redesign removes the scale problem:** no collapse to the marginal (e.g. 8 classes, joint: 0.699 vs 1.153 for the frequency baseline).
- **But it does not help:**

| Tier B log loss | hmm_event | hmm_factorized | hmm_joint | mc2_zone |
|---|---|---|---|---|
| filtered8 | 0.690 | 0.702 | 0.699 | **0.664** |
| scenarioD15 | 1.104 | 1.251 | 1.255 | **1.019** |

- **Why [H, supported by the state analysis].**
  - Zone is highly persistent along a sequence, while event types alternate (Pass ⇄ Carry).
  - With K = 12 states, the zone HMMs spend the states on (event type × third) cells (`hidden_state_analysis.csv`). For example, the 8-class factorised state 2 is "Carry 0.99, 100% attacking third".
  - That leaves too few states for the event dynamics, which K = 12 cannot fully represent even without zones for 15 classes. In Scenario D, the event-only HMM (1.104) is worse than the 2nd-order chain (1.049).
  - A conditional (input-output) HMM, where the zone conditions the transitions instead of being emitted, would avoid this. It was not implemented (`methodology.md`).
- **K selection [U]:**
  - K = 12, the upper end of the requested grid, was best for all HMMs on Tier A, and validation loss was still decreasing (`figures/hmm_k_sensitivity.png`). The optimum may lie above 12.
  - The factorised and joint curves are non-monotone at small K, with seed spreads of 0.19–0.23 nats: strong local optima.
- **Convergence (Tier B):**
  - Factorised and joint fits all converged (≤ 296 iterations).
  - 1 of the 15 event-only fits per taxonomy and 1 of the 15 joint fits (8 classes) reached the 1,000-iteration cap without converging.
  - All EM objectives were monotone.
  - The unbiased initialisation (`init_self_transition` 0) won on training objective for every emission and taxonomy on Tier A.

### 5.3 Hidden-state interpretation (`hidden_state_analysis.csv`, `figures/hmm_state_pitch_thirds.png`, `figures/hmm_transitions_*.png`)

Filtered posteriors on Tier B fold-0 validation matches. Descriptive.

- **Event-only HMM:**
  - States are event-type proxies with near-zero self-transitions, e.g. "Carry 1.00", "Pass 1.00", "Shot 0.97 with 98% in the attacking third", "Pressure 0.99 with 90% team change".
  - There are no persistent, phase-like states.
  - [H] A few states group turnover situations: Ball Recovery or Clearance with 74–81% team change.
- **Zone HMMs:** states are (event type × third) cells.
- **Labelling:** no state is labelled defence, build-up or attack. Interpreting states as tactical phases is **not supported**.

## 6. Model comparison, Tier B [C] (`model_comparison.csv`; uniform decisions)

| Model | 8: macro-F1 | 8: bal. acc. | 8: log loss [CI] | 8: Brier | 15: macro-F1 | 15: bal. acc. | 15: log loss [CI] | 15: Brier |
|---|---|---|---|---|---|---|---|---|
| frequency baseline | 0.082 | 0.125 | 1.153 | 0.602 | 0.038 | 0.067 | 1.659 | 0.724 |
| mc1 (= exp. 1 MC1) | 0.210 | 0.226 | 0.704 [0.694, 0.716] | 0.337 | 0.215 | 0.218 | 1.152 [1.139, 1.167] | 0.519 |
| mc2 | 0.210 | 0.225 | 0.692 | 0.332 | 0.248 | 0.242 | 1.049 | 0.486 |
| mc1_zone | 0.211 | 0.226 | 0.677 | 0.332 | 0.231 | 0.243 | 1.119 | 0.513 |
| **mc2_zone** | 0.211 | 0.226 | **0.664 [0.653, 0.676]** | **0.327** | **0.275** | **0.260** | **1.019 [1.007, 1.032]** | **0.479** |
| hmm_event (K = 12) | 0.210 | 0.225 | 0.690 | 0.331 | 0.203 | 0.210 | 1.104 | 0.501 |
| hmm_factorized (K = 12) | 0.207 | 0.223 | 0.702 | 0.341 | 0.128 | 0.138 | 1.251 | 0.542 |
| hmm_joint (K = 12) | 0.209 | 0.225 | 0.699 | 0.335 | 0.098 | 0.117 | 1.255 | 0.552 |

- **Reproduction of experiment 1:** on Tier A, `exp1_markov_order1/2` reproduce experiment 1 exactly (0.6831 / 0.6719). The new `mc1` equals it (difference < 0.0001).
- **Per-class results:** `per_class_metrics.csv`. Pass and Carry F1 ≈ 0.83–0.85 under uniform decisions.
- **Confusion matrices:** `confusion_matrices.csv`, `figures/confusion_*.png`.
- **Comparability:** values of the two taxonomies are different tasks.

## 7. Additional features (`feature_assessment.csv`)

| Feature | Measurement | Decision |
|---|---|---|
| Distance to goal | location availability 99.999% | discretised as pitch thirds (a continuous density would reintroduce the experiment-1 problem) |
| Time since previous event | 6.9% zero gaps | deferred |
| Possession-change indicator | binary, causal; but mechanical in D (defensive actions by the other team) | recommended as the next Markov context ablation |
| Possession duration | relies on annotator possession ids, which are non-monotone | not used |
| Previous event types | — | used (2nd order) |
| Visible players / nearest opponent | only for the 8 production types in the dataset; censored by `visible_area` | not built (needs a new causal feature build) |

No 360 freeze frame is visualised, so no visible-area polygon applies.

## 8. Verification

| Check | Result |
|---|---|
| New tests (37): cost decisions, ties, expected cost, metrics vs scikit-learn, decision-invariance of probability metrics, calibration, bootstrap = direct recomputation, zone boundaries, backoff smoothing by hand, zone-HMM likelihood by enumeration incl. missing zones, future invariance, parity with the experiment-1 HMM, taxonomy mapping, shootout exclusion, test-set refusal, nested-CV and cost-prior leakage, determinism | **37 passed** |
| Whole project suite | **452 passed, 0 failed** (415 earlier + 37); no existing test changed |
| Independent verification (`scripts/verify_markov_hmm_cost_sensitive_v2.py`, no import of the pipeline) | metrics recomputed from `predictions.parquet`: max deviation 3.1e-10; probability metrics across schemes: deviation 0.0; loop-based Bayes decisions agree 100%; Laplace MC1 recounted from raw JSON: 2.9e-8 (float32 storage) → `independent_verification.json` **passed** |
| Leakage (`leakage_checks.json`) | no final-test match in any tier; tiers disjoint; folds match-grouped; cost priors training-only; tuning on Tier A only; filtered8 targets = production; Scenario D = research script |
| Reproducibility (`reproducibility_checks.json`) | Tier B Markov models of both taxonomies re-run from raw data: log loss identical (max deviation 2.2e-16). All 6 Tier B fold-0 selected HMMs re-fitted from scratch: parameter differences **0.0**, identical iteration counts (e.g. 843 / 843). **Passed** |
| Experiment 1 integrity | 58/58 files unchanged |

**Known incident (disclosed):**
- On 2026-10-28, before any experiment-2 code existed, one exploratory check loaded experiment 1's all-match event cache, which includes final-test matches, and immediately filtered it to the 30 pilot matches.
- No statistic of a final-test match was computed or viewed.
- Experiment-2 code now refuses that cache (tested).

## 9. Limitations

1. **Tier design.** Tier A tuning is not fully nested, because of the cross-fold K rule and the γ selection on the same 30 matches. Tier B is the unbiased development estimate for the tuned choices.
2. **HMM.** K is at the grid edge (12) [U]. Local optima are severe (seed spreads up to 0.23 nats). Tier B used 3 seeds of one initialisation type for compute reasons. 3 of the 90 Tier B fits stopped at 1,000 iterations.
3. **Costs.** There is no external utility, so the schemes are a sensitivity analysis. The tuned scheme optimises macro-F1, which is an explicit choice of objective.
4. **Mechanical classes in Scenario D** (Duel after Dispossessed, Goal Keeper after Shot, 50/50 pairs) inflate those classes' scores.
5. **Zone.** It is a new derived input from the raw event location. It is not yet in the approved allowlist [U].
6. **Uncertainty.** CIs resample matches within each tier. They do not cover the choice of development matches or the final test set.
7. **Storage.** `predictions.parquet` is 238 MB (float32 probabilities; recomputation agrees to 3e-10). It should not be Git-tracked; consider adding it to `.gitignore`.

## 10. Recommendation

- **Primary model for the thesis baseline:** `mc2_zone` on Scenario D, evaluated with log loss and Brier as primary metrics.
- **Cost-sensitive decisions:** report them as a separate decision layer (moderate scheme) with per-class precision and recall. They should never replace the probability metrics.
- **HMM:** not justified as superior, in any variant tested.
- **Next steps (each needs your approval):**
  1. decide Scenario D's open options (Dispossessed → Duel, Goal Keeper);
  2. approve `zone` for the allowlist;
  3. add the possession-change context as the next ablation;
  4. for discriminative models (LR, GRU, LSTM, Transformer), use the same tiers, folds, rows and cost layer for a fair comparison.
- **Final test:** not run; it needs your explicit authorisation.

## Files

| Path | Content |
|---|---|
| `README.md`, `methodology.md` | this report; definitions and references |
| `experiment_config.yaml`, `split_manifest.json` | exact configuration; final-test / Tier A / Tier B ids and folds |
| `taxonomy_comparison.csv`, `class_distribution.csv`, `feature_assessment.csv`, `audit_checks.json` | taxonomy audit and feature assessment |
| `cost_matrix_definitions.json`, `cost_sensitive_comparison.csv`, `cross_cost.csv`, `approach_b_check.csv` | cost schemes, per-fold matrices, tuned γ; scheme comparison; Bayes-optimality check; Approach B |
| `model_comparison.csv`, `metrics_by_fold.csv`, `per_class_metrics.csv`, `confusion_matrices.csv`, `paired_differences.csv` | evaluation (both tiers) |
| `rare_event_analysis.csv`, `spatial_ablation.csv`, `calibration_metrics.csv`, `reliability_table.csv` | rare events, zone ablations, calibration |
| `hmm_training_log.csv`, `markov_concentration_choices.csv`, `hidden_state_analysis.csv` | every HMM fit; nested smoothing choices; state analysis |
| `predictions.parquet` | wide format: one row per evaluation observation; `p__<model>__<class>` (float32) and `decision__<model>__<scheme>` |
| `models/` | per tier and taxonomy: per-fold Markov counts and selected HMM parameters |
| `leakage_checks.json`, `reproducibility_checks.json`, `independent_verification.json`, `environment.json`, `checksums.json` | verification and provenance |
| `figures/` | class distribution, cost matrices, rare-event recall, precision/recall trade-off, confusion matrices, calibration, Markov and HMM transitions, hidden-state pitch thirds, K sensitivity, spatial and model comparisons, cost-sensitive macro-F1 / balanced accuracy |
| `tiers/` | raw per-tier outputs (and the HMM fit cache) from which the top-level files are assembled |

**Reproduce** (about 1 h for Tier A and about 3.3 h for Tier B on this 8 GB machine; HMM fits are cached and resumable):

```bash
.venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py audit
.venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py tier_a
.venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py tier_b --task filtered8
.venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py tier_b --task scenarioD15
.venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py repro
.venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py report
.venv/bin/python scripts/verify_markov_hmm_cost_sensitive_v2.py
```
