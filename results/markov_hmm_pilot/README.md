# Markov chain and hidden Markov model pilot — StatsBomb 360 next-event prediction

Date: 2026-10-27. **Pilot only**: 30 development matches, 5-fold match-grouped cross-validation. The final
test set (83 matches) was **not loaded or evaluated**. Full-data training was **not started**; it needs your
approval. Nothing in the production dataset builder, the event taxonomy, the feature allowlist or earlier
results was changed. No commit was made.

Every number below is copied from a file in this directory. Status labels used throughout:

- **[V]** verified fact (measured, reproducible script);
- **[D]** modelling decision;
- **[H]** hypothesis / interpretation;
- **[U]** unresolved.

## 1. Summary

| Question | Answer |
|---|---|
| Is the data ready for sequence modelling? | **Yes, with limitations.** The raw order is verified. The production targets are reproduced exactly, and period 5 is verifiably the shootout. The binding limitation is scientific: the approved 8-type taxonomy hides 59% of intermediate events, and that taxonomy decision is still open. |
| Primary target for the pilot | `filtered8`: the next event among the 8 production types, explicitly defined as a filtered-event task. It is identical to the production target with period 5 removed. |
| Best pilot models | 1st-order Markov chain, 2nd-order Markov chain and event-only HMM (K=5) are practically tied on the primary task (macro-F1 0.212, balanced accuracy 0.227). The 2nd-order chain has the lowest log loss (0.672 vs 0.683; paired match-bootstrap difference −0.011, 95% CI [−0.014, −0.009]). |
| Is the HMM better than the Markov chain? | **No.** Event-only HMM vs 1st-order chain, log-loss difference −0.0001 [−0.0026, 0.0023]. Its states reproduce the event types themselves (§8). |
| Does spatial information help the HMM? | **Not in this observation model.** The categorical × Gaussian HMM spends its states on pitch zones, and its event predictions collapse to about the marginal distribution (log loss 1.119, frequency baseline 1.139). |
| Leakage / reproducibility | All automated checks pass. A full re-run reproduced all metrics, parameters and predictions byte-for-byte, and an independent re-derivation from raw JSON agrees to 1e-15 (§10). |
| Ready for full-scale training? | **Not yet.** Four issues must be decided or fixed first (§12). |

## 2. Phase 1 — data readiness audit (`data_audit.md`, `data_audit.json`, `exclusion_report.csv`)

The audit reads all 417 eligible matches *before* the split. It is descriptive only: nothing computed there
fits or selects a model.

**Verified facts [V]:**

**Matches**
- 426 matches have 360 files.
- 9 are excluded by `configs/excluded_matches.json`: corrupted JSON, a single 360 record, or 360 UUIDs that do not join to the events.
- **417 are eligible.** 387 end in period 2, 12 in period 4 (extra time) and 18 in period 5 (shootout).
- There are no duplicate matches: identical content hashes 0, identical (date, home, away) 0, event ids shared across matches 0.

**Event stream**
- 1,551,841 raw events of 33 types.
- `index` equals the raw array order in every match, event ids are unique, and periods never decrease.
- **The clock is not a reliable order.**
  - 487,523 adjacent raw pairs share a timestamp.
  - 43 have a negative gap. 41 of these are Ball Receipt\* records near the end of a period, stamped about 00:00:01, i.e. corrupted timestamps.
  - Sequences are therefore ordered by `index` only. Time-based features would need an explicit policy (§5).
- Every period starts with Starting XI / Half Start and ends with Half End (912 periods).
- The possession id decreases 9 times (annotation interleaving around referee ball-drops). Possession ids are not used.

**Penalties**
- Period 5 contains only penalty shots (183), Goal Keeper (183), Half Start/End, 4 Bad Behaviour and 1 Tactical Shift. So period 5 is the shootout, and it is excluded entirely.
- The 142 ordinary penalty kicks in periods 1–4 are kept.
- The production dataset contains 104 period-5 rows. The pilot drops them by task definition; the production file is untouched.

**Production dataset**
- SHA-256 `8f6ff118…259c`, 787,331 rows, 417 matches.
- An independent re-derivation of every row and target from the raw stream gives 0 missing rows, 0 extra rows, 0 type mismatches and 0 target-id mismatches.

**Taxonomy gaps**
- 59.3% of filtered rows skip at least one raw event (mean 0.72).
- The most frequently skipped events: Ball Receipt\* 354,383; Pressure 118,017; Duel 22,295; Block 14,696; Goal Keeper 12,159; Foul Committed 8,966; Interception 8,586.

**360 coverage and missingness**
- 105,389 of 893,476 events of the 8 types (11.8%) have no 360 record.
- Causal orientation is unavailable for 41 rows, and 43 rows have no actor_x.
- The dataset stores visible-player counts but neither the `visible_area` polygon nor player coordinates.

**Spatial frame**
- `actor_x/actor_y` are on the reference team's scale.
- Mapped into the acting team's attacking frame via `team_normalization`, they lie within 5 units of the raw event `location` for **99.978%** of rows (within 1 unit: 98.99%).
- Without the transform the figure is 52.4%, so the transform is externally validated against the StatsBomb location convention.

**Which corrections apply where**
- Causal orientation and causal team identity feed the dataset feature columns.
- The offline orientation, camera-side display orientation and reviewer decisions are used for visualization or `*_offline` diagnostics only, never as model inputs.

**Can the existing dataset support valid Markov/HMM training without reconstruction?**
- Partly.
  - Its rows and targets are correct, and they are the evaluation population.
  - But it only contains events that have a 360 record and a successor, so it is not a sequence. Consecutive rows are not always consecutive filtered events, because 11.8% of filtered events lack 360 and have no row.
- The pilot therefore rebuilds the *sequences* from the raw stream (`src/sequence_models/tasks.py`) and joins the production rows for evaluation and spatial features.
- This is not a competing dataset: the rebuilt evaluation rows are verified identical to the production rows (§10).

## 3. Phase 2 — task specification

### 3.1 Candidate targets compared (all 417 matches, periods 1–4, evaluation rows; `data_audit.json`)

| | **filtered8** (primary) | **fullstream9** (comparison) |
|---|---|---|
| Definition | next event of one of the 8 production types in the same period, by event order; other raw events are skipped *by definition* | the immediate next raw event in the same period; its type if one of the 8, else the explicit class `Other` (all 25 remaining types, including administrative events) |
| Evaluation rows | 787,227 | 787,971 |
| Majority class | Pass 48.6% | **Other 59.4%** |
| Imbalance (max/min) | 43.9 | 230.9 |
| H(target) / H(target \| current type), bits | 1.661 / 1.011 | 1.462 / 0.688 |
| Rows skipping ≥ 1 raw event | 59.3% | 0% |
| Largest transition cell | Carry → Pass (36.7% of rows) | Pass → Other (46.0%; almost all Ball Receipt\*) |

**Differences.**
- `filtered8` answers *"which of these 8 actions comes next, whatever happens in between"*. Its transitions are transitions of a filtered process, not raw adjacency. That is legitimate only because it is declared as such, and the number of skipped events is recorded per row (`n_skipped_raw_events`).
- `fullstream9` preserves raw adjacency, but its target is dominated by `Other`, and the largest `Other` cell is the mechanical Pass → Ball Receipt\* annotation pair. Its lower conditional entropy reflects annotation convention more than football. The earlier taxonomy research reached the same conclusion for full-stream targets.

**Recommendation [D].**
- `filtered8` is the primary target for this pilot because:
  - it is the approved production target;
  - all later models (logistic regression, LSTM, GRU, Transformer) will be evaluated on it;
  - its filtering is explicit and documented.
- `fullstream9` is a comparison that shows how the conclusions change when raw adjacency is preserved.
- **For the thesis**, the primary target should be re-decided together with the open taxonomy decision (`results/event_taxonomy_research_2026-10-27/README.md`, Scenario D). The pipeline is vocabulary-driven (`configs/markov_hmm_pilot.yaml`), so a different taxonomy is a configuration change.

### 3.2 Formal specification (`src/sequence_models/tasks.py`)

| Item | Definition |
|---|---|
| Sequence | one (match, period), periods 1–4. Never crosses a period or match boundary. Order = raw `index`. |
| Observation unit | an event of the sequence: its vocabulary symbol and, for the spatial HMM, its causal spatial features |
| Prediction timestamp | immediately after event *t* (`prediction_time` = its period-relative timestamp, `prediction_event_index` = its index) |
| Target | symbol of observation *t+1* of the same sequence |
| Allowed inputs | symbols of observations 1..*t* (events with or without 360 data) and, for the spatial HMM, spatial features of observations 1..*t*. Nothing from *t+1* onward. |
| Vocabulary | filtered8: Pass, Carry, Ball Recovery, Clearance, Dribble, Shot, Miscontrol, Dispossessed (`event_vocabulary.json`). fullstream9: the same + Other. |
| Prediction points | observations whose raw type is one of the 8 and that have a target |
| Evaluation population | prediction points whose current event has a 360 record (the production dataset population). Identical across models within a task. |
| Exclusions | period 5 (whole period removed, so no transition is created); last observation of a period (no target); prediction points without a 360 record (still observed and still predicted, not evaluated; `current_event_without_360_record`) |
| Extra time | periods 3 and 4 included as separate sequences (5 pilot matches) |
| Possession changes | not a boundary. Turnovers are predictable events. |
| Simultaneous / non-monotonic timestamps | keep raw order; timestamps never reorder or split |
| Removed events | nothing is deleted and then reconnected. Skips exist only in `filtered8`, by definition, and are counted. |
| Unknown categories | `UnknownEventTypeError` (filtered8) / mapped to `Other` (fullstream9); unknown symbol indices raise in every model |
| Missing coordinates | NaN, marginalised out of the spatial emission (§5); never imputed, never zero |

## 4. Phase 6 — splits (`split_manifest.json`)

- **Final test hold-out:**
  - 83 of 417 matches (19.9%), drawn per (competition, season) stratum with seed 20261027 (numpy `default_rng`).
  - It uses match metadata only; no test-match event file is opened by the split or pilot stages.
  - Status: **untouched**.
- **Development pool:** 334 matches.
- **Pilot:** 30 development matches.
  - 2 were drawn from extra-time matches first; the rest were drawn proportionally over the 11 competition-season strata (at least 1 each).
  - The pilot contains 5 extra-time matches.
- **Cross-validation:**
  - `StratifiedGroupKFold(5, shuffle=True, random_state=20261027)` over evaluation rows, grouped by match, stratified by the filtered8 target.
  - The same match folds are used for both tasks. Each pilot match is validated exactly once and never trained and validated in one fold.
  - Pooled out-of-fold metrics therefore use one prediction per evaluation row.
- **Model selection:**
  - The HMM seed per (fold, K) is chosen by the highest **training** penalised log-likelihood.
  - K is chosen by the lowest mean **validation** log loss over folds.
  - Both choices use development data only.

## 5. Phase 3 — spatial features (`src/sequence_models/spatial.py`)

| Candidate | Decision | Reason |
|---|---|---|
| x in the acting team's attacking frame (`x_attacking`, 0 = own goal line, 120 = opponent goal line) | **used** | causal (approved `actor_x` + validated transform); 99.98% agreement with event location |
| \|y − 40\| (`lateral_distance_from_centre`) | **used** | same source. \|·\| makes it invariant to the y mirror and avoids a bimodal Gaussian. |
| distance to goal | not used | a deterministic function of the two features above |
| time since previous event | deferred | 487,523 raw ties and 41 corrupted timestamps; a non-Gaussian, heavy-tailed variable that needs its own emission family |
| visible teammates / opponents | not used in the HMM | counts of *visible* players depend on camera framing; unobserved ≠ absent |
| nearest-opponent distance, pressure, goalkeeper position | not built | need freeze-frame coordinates and `visible_area` handling, which the dataset does not carry; a new feature build would need approval |
| possession / team-change indicator | descriptive only | used for state interpretation, not as an emission |

Notes:
- `team_normalization` is METADATA in `configs/ml_feature_classification.json`. Here it is used **only** as a coordinate transform, never as an input.
- **The two derived features are not in the approved allowlist [U]**, so the spatial HMM is provisional until you approve them.
- Missingness:
  - In the evaluation rows only 0.003% lack spatial features.
  - Across all sequence observations 12.1% do, mostly filtered events without a 360 record, which are not evaluated.
  - Missing features are marginalised. This assumes missing-at-random [H], and the assumption is questionable because restarts lack 360 more often.
- The scaler (mean/SD) is fitted on each fold's training sequences only (verified: §10).

## 6. Phase 4 — Markov chains and baseline (`src/sequence_models/markov.py`)

With counts $C_{ij}$ of within-sequence transitions $i \to j$ in the training folds, $V$ classes and $\alpha = 1$:

$$\hat P(o_{t+1}=j \mid o_t=i) = \frac{C_{ij} + \alpha}{\sum_k C_{ik} + V\alpha}$$

- **Frequency baseline:** $\hat P(j) = (N_j + \alpha) / (N + V\alpha)$ over training targets.
- **Unseen source state:** falls back to the frequency baseline, and this is flagged.
- **Second order:** $\hat P(o_{t+1} \mid o_{t-1}, o_t)$ with the same smoothing. It backs off to first order when the context has fewer than 20 training occurrences, and at the first observation of a sequence.
- **Outputs:**
  - descriptive transition counts and probabilities on all 30 pilot matches: `transition_counts.csv`, `transition_probabilities.csv`, `figures/markov_transition_heatmap.png`;
  - per-fold parameters: `models/cv_fold_models_*.json`.

## 7. Phase 5 — hidden Markov model (`src/sequence_models/hmm.py`)

**Observation model.** The event symbol and the spatial features are conditionally independent given the state:

$$p(o_t \mid z_t=k) = B_{k,e_t}\prod_{d\in\text{obs}(t)}\mathcal N(x_{t,d};\mu_{kd},\sigma^2_{kd})$$

- **Event-only:** categorical emissions (a categorical HMM).
- **Spatial:** categorical × diagonal Gaussian.
- Event symbols are never treated as numbers. A GaussianHMM on integer codes was rejected.
- hmmlearn was not used because it cannot marginalise missing continuous dimensions and offers smoothed, not filtered, posteriors.
- The implementation is tested against brute-force path enumeration (§10).

**Training.**
- Algorithm: Baum–Welch with scaled forward–backward over **independent, padded and masked sequences** (70 per task, never concatenated).
- Prior: Dirichlet MAP pseudo-count 1 on start, transition and emission rows; variance floor 1e-3 on standardised features.
- Initialisation:
  - start probabilities ~ Dir(1);
  - transitions = 0.5·I + 0.5·Dir(1);
  - emission rows ~ Dir(20 × training marginal);
  - Gaussian means = K random complete training observations; variances = training variance.
- Grid: K ∈ {2, 3, 4, 5} × seeds {0, …, 4} × 5 folds.
- Stopping: per-observation improvement of the penalised objective < 1e-6, or 1,000 iterations.
- Pre-run decision: `max_iter` was raised from 300 to 1,000 after a timing check on *training* data showed an unconverged K=5 fit at 300 iterations. No validation metric existed at that point.
- Logged per fit (`hmm_training_log.csv`): seed, initialisation method, iterations, convergence, monotonicity, training log-likelihood, validation metrics. Parameters: `hmm_parameters.json`, `models/`.

**Causal prediction.** With the scaled forward variable $\hat\alpha_t = p(z_t \mid o_{1:t})$:

$$p(e_{t+1}=v\mid o_{1:t}) = \sum_j\Big(\sum_i \hat\alpha_t(i)A_{ij}\Big)B_{jv}$$

- The next step's spatial features are not available at prediction time. They integrate out of the joint emission, so the formula is exact for the mixed model as well.
- Smoothed posteriors and Viterbi paths are used only in the descriptive timeline figure.

## 8. Phase 7–8 — pilot results

Pilot population:
- filtered8: 59,655 evaluation rows (8,120 not evaluated for lacking 360; 70 last-of-period).
- fullstream9: 59,713 evaluation rows.
- Classes (filtered8): Pass 29,091; Carry 24,003; Ball Recovery 2,627; Clearance 1,058; Dribble 811; Shot 732; Miscontrol 725; Dispossessed 608.
- All 8 classes are present in every fold.

### 8.1 Model comparison, filtered8 (pooled out-of-fold; 95% match-level bootstrap CI, 1,000 replicates)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced acc. | Log loss | Brier |
|---|---|---|---|---|---|---|
| Frequency baseline | 0.488 | 0.082 [0.081, 0.083] | 0.320 | 0.125 | 1.139 [1.111, 1.172] | 0.598 |
| Markov 1st order | 0.801 | 0.212 [0.209, 0.215] | 0.754 | 0.227 [0.225, 0.229] | 0.683 [0.648, 0.721] | 0.325 |
| Markov 2nd order | 0.800 | 0.212 [0.208, 0.215] | 0.753 | 0.227 [0.224, 0.229] | **0.672** [0.637, 0.710] | **0.320** |
| HMM event-only, K=5 | 0.801 | 0.212 [0.209, 0.215] | 0.754 | 0.227 [0.225, 0.229] | 0.683 [0.648, 0.722] | 0.322 |
| HMM event + spatial, K=5 | 0.488 | 0.082 | 0.320 | 0.125 | 1.119 [1.090, 1.153] | 0.595 |

**Paired differences vs the 1st-order chain** (same rows; match bootstrap):
- 2nd order: log loss −0.011 [−0.014, −0.009], Brier −0.005, accuracy −0.0015 [−0.0019, −0.0011].
- Event HMM: log loss −0.0001 [−0.0026, 0.0023], Brier −0.0026 [−0.0043, −0.0012], accuracy −0.0001.

**Fold-level log loss** (`metrics_by_fold.csv`):
- Markov 1st order: 0.648–0.706.
- Event HMM: 0.652–0.704.
- Fold SD ≈ 0.025 for both, larger than every between-model difference except the baselines.

### 8.2 Per-class results (`per_class_metrics.csv`, `figures/per_class_f1.png`)

- Every non-baseline model predicts **only Pass and Carry** under argmax.
  - Markov 1st order: Pass P 0.861 / R 0.824 / F1 0.842; Carry P 0.749 / R 0.992 / F1 0.853.
  - The six rare classes have recall 0. Macro-F1 ≈ 0.21 is therefore exactly "two classes predicted well, six never".
- This is a property of argmax over imbalanced, weakly conditioned distributions: no rare class is ever the most probable next event given the current type alone.
- It is not a failure to *represent* rare classes. Their probabilities are positive and calibrated in aggregate: for every class, the mean predicted probability is within 0.015 of the class share (e.g. Shot 0.0122 vs 0.0123 for the 1st-order chain; `mean_predicted_probability` in `per_class_metrics.csv`).
- The frequency baseline and the spatial HMM predict Pass for every row.

### 8.3 Comparison task, fullstream9

| Model | Accuracy | Macro-F1 | Balanced acc. | Log loss |
|---|---|---|---|---|
| Frequency baseline | 0.593 | 0.083 | 0.111 | 1.152 |
| Markov 1st order | 0.847 | 0.288 | 0.310 | 0.459 [0.427, 0.496] |
| Markov 2nd order | 0.849 | 0.293 | 0.312 | **0.451** |
| HMM event-only, K=5 | 0.846 | 0.287 | 0.301 | 0.510 (worse than Markov: +0.051 [0.042, 0.060]) |

These scores are higher than filtered8's because Pass → Other (Ball Receipt\*) is near-deterministic. Different tasks must not be ranked against each other.

### 8.4 Sensitivity to K and convergence (`hmm_training_log.csv`, `figures/hmm_k_sensitivity*.png`, `figures/hmm_seed_stability*.png`)

| Task / variant | K=2 | K=3 | K=4 | K=5 |
|---|---|---|---|---|
| filtered8 event-only, mean val. log loss | 1.124 | 0.757 | 0.703 | **0.683** |
| filtered8 spatial | 1.144 | 1.136 | 1.126 | **1.119** |
| fullstream9 event-only | 1.147 | 0.540 | 0.534 | **0.510** |

- **All 300 cross-validation fits converged** (≤ 651 iterations) and every EM objective was monotone.
- **The selected K=5 is the edge of the grid [U].** Validation loss is still decreasing, so the optimum may lie at K ≥ 6.
- **Local optima are severe.** The mean within-fold spread of validation log loss across seeds is 0.14 (filtered8, K=3) and 0.61 (fullstream9, K=3).
  - At K=2 all seeds converge to an almost uninformative solution (training log-likelihood ≈ marginal entropy).
  - [H] The initial transition matrix favours self-persistence (0.5·I), while the data alternate (Pass → Carry → Pass). Five seeds per configuration are not enough.

### 8.5 Hidden-state interpretation (`hidden_state_analysis.csv`, figures `hmm_*_emissions.png`, `hmm_*_transition_heatmap.png`, `hmm_spatial_state_pitch.png`, `hmm_*_state_timeline_*.png`)

Statistics are computed with **filtered** (causal) posteriors on all 30 pilot matches. They are descriptive, not a performance estimate.

**Event-only HMM (K=5): states are event-type proxies, not tactical phases.**

| State | Main emissions | Typical next state |
|---|---|---|
| 1 | Pass 0.97 | State 2 (0.89) |
| 2 | Carry 0.99 | State 1 (0.83) |
| 0 | Pass 0.61, Miscontrol 0.14, Dispossessed 0.11, Shot 0.09 | — |
| 3 | Pass 0.65, Clearance 0.24; team changed since the previous event in 53% | — |
| 4 | Ball Recovery 0.51, Pass 0.49; team changed in 62% | — |

- States 1 and 2 form a Pass ⇄ Carry alternation.
- Self-transitions are ≤ 0.42, so states do not persist like "phases"; mean actor x is 51–68 for every state.
- [H] States 3 and 4 look like "after a turnover". That is the only phase-like structure, and it is weak.
- No state is labelled defence, build-up or attack: **states are reported as State 0–4**.

**Spatial HMM (K=5): states are pitch zones.**

| State | Mean x | Mean \|y − 40\| | Region |
|---|---|---|---|
| 0 | 19 | 13 | own third |
| 3 | 52 | 32 | wide |
| 4 | 51 | 13 | central midfield |
| 1 | 78 | 20 | attacking half |
| 2 | 100 | 22 | final third |

- The states are persistent (self-transitions 0.67–0.81).
- Their event emissions are close to the marginal, with some spatially plausible shifts: Clearance 0.09 in state 0, Shot 0.07 in state 2.
- [V] The Gaussian part dominates the likelihood (training log-likelihood about −3.0 vs −0.7 nats per observation for event-only), so EM optimises location, not event sequencing.
- [H] Zones are interpretable as "where the ball is", but under this observation model they carry almost no information about the next event *type*. A model that keeps sequential event dependence and location separate is needed (§12).

## 9. Prediction output contract (`predictions.csv`, `predictions_fullstream9.csv`)

One row per (model, observation whose raw type is one of the 8 production types, in a validation fold).
Coverage:
- selected configurations in these files;
- other K values in `predictions_*_other_hmm_configurations.csv.gz`.

| Column | Meaning |
|---|---|
| task | `filtered8` or `fullstream9` |
| model_name | `frequency_baseline`, `markov_order1`, `markov_order2`, `hmm_event_K5`, `hmm_spatial_K5` |
| fold_id | CV fold whose validation set contains the match (model trained on the other 4 folds) |
| match_id, event_id | StatsBomb ids of the current event (the prediction point) |
| sequence_id, sequence_position | `<match>_P<period>` and 0-based position in the task's sequence |
| prediction_period, prediction_time, prediction_event_index | period, period-relative timestamp (HH:MM:SS.fff) and raw `index` of the current event: the prediction is made right after it |
| current_event | current event's vocabulary symbol (an input) |
| true_next_event, true_next_event_id, true_next_raw_type | target symbol, its event id and raw type (evaluation only, never an input; empty if no target) |
| n_skipped_raw_events | raw events between current and target (filtered8 only; 0 in fullstream9) |
| predicted_next_event | argmax of the probabilities |
| p_\<class\> | predicted probability of each class (sums to 1, all > 0) |
| eval_eligible | True if the row is in the evaluation population |
| exclusion_reason | `current_event_without_360_record`, `no_target_in_period`, or empty |

Metrics can be recomputed from this file alone (`scripts/verify_markov_hmm_pilot.py` does so).
- Pooled metrics are **micro-aggregated over evaluation rows of all validation folds**.
- Fold metrics are per validation fold.
- Intervals resample **matches**, never events.

## 10. Verification

| Check | Result | Where |
|---|---|---|
| New unit tests (41) | **41 passed** | `tests/test_sequence_hmm.py`, `tests/test_sequence_markov_metrics.py`, `tests/test_sequence_tasks_pipeline.py` |
| Whole project suite | **415 passed, 0 failed** (374 pre-existing + 41 new); no existing test changed | `pytest` |
| HMM likelihood and predictive probability | hand-computed value (P = 0.10893; P(e₂=0 \| o₁) = 0.662903) and brute-force enumeration incl. Gaussian and missing features | `test_sequence_hmm.py` |
| Future-data leakage | changing observations after *t* never changes predictions ≤ *t* (synthetic tests; real sequence 3773586_P1: max change before cut 0.0) | tests + `leakage_and_reproducibility_checks.json` |
| Target leakage through the whole CV pipeline | changing every validation sequence's last observation leaves all other fold predictions identical, for all models | `test_cross_validation_has_no_target_or_future_leakage` |
| Match grouping | no train/validation overlap; each pilot match validated once | `leakage_and_reproducibility_checks.json` |
| Final test untouched | pilot ∩ test = ∅; no test match in loaded data or predictions; CV refuses test matches (test) | same + `independent_verification.json` |
| Preprocessing leakage | fold-0 scaler refit from training sequences equals the stored scaler | same |
| Targets | pilot evaluation rows and targets identical to the production rows (59,655, 0 mismatches) | same |
| Independent re-derivation | Markov probabilities recounted from raw JSON without the pipeline: max deviation 1.1e-16; metrics recomputed with scikit-learn: ≤ 4.4e-16 | `scripts/verify_markov_hmm_pilot.py` → `independent_verification.json` |
| Reproducibility | refitting a stored model: parameter difference 0.0; full re-run: all metric files, `hmm_parameters.json` and `predictions.csv` byte-identical | `leakage_and_reproducibility_checks.json` |

## 11. Limitations

1. **Pilot size.** 30 matches give about 60k evaluation rows but only 608–2,627 rows per rare class, and fold-to-fold variation (SD ≈ 0.025 log loss) exceeds most model differences. Per-class conclusions for rare classes are weak.
2. **Filtered task.** Transition probabilities describe the filtered 8-type process. They are not raw event adjacency, and defensive and goalkeeper actions are invisible (taxonomy decision still open).
3. **Argmax metrics.** Macro-F1 and balanced accuracy are capped by argmax on imbalanced classes. Log loss and Brier are the informative comparisons for probabilistic sequence models.
4. **HMM optimisation.** Strong local optima, a persistence-biased initialisation and K at the grid edge mean the HMM results are a lower bound for this model family, not its best achievable result.
5. **Spatial model.** It uses two hand-chosen features, a mixed emission whose continuous part dominates, a missing-at-random assumption, and two derived features that are not yet approved.
6. **Evaluation population.** Only events with a 360 record are evaluated (the production convention). 11.8% of filtered events are predicted but not scored.
7. **Uncertainty.** The match bootstrap covers between-match variation within the pilot, not variation of the pilot draw itself.

## 12. Readiness for full-scale training — **READY WITH LIMITATIONS** for the pipeline, **not yet** for the full run

Must be decided or fixed before full-data training:

1. **Taxonomy and primary target.** Keep `filtered8`, or adopt Scenario D (and the Dispossessed → Duel / Goal Keeper options). This changes the vocabulary in `configs/markov_hmm_pilot.yaml` only.
2. **HMM search.**
   - Extend K beyond 5 (e.g. up to 12, since states ≈ event types).
   - Use ≥ 10 seeds.
   - Add a non-persistent initialisation (e.g. `init_self_transition` 0), selected by training likelihood.
3. **Spatial model.**
   - Approve (or reject) `x_attacking` and `lateral_distance_from_centre` for the allowlist.
   - Replace the dominated mixed emission with a model where location cannot swamp the event sequence. Options:
     - discretised pitch zones as a second categorical emission with tempered weight;
     - an input-output HMM conditioning transitions on location;
     - leaving spatial context to the discriminative models.
4. **Primary metric.** Fix log loss (with Brier) as the primary model-selection metric before seeing full results; report argmax metrics as secondary.

Primary candidate for the thesis baseline: the **2nd-order Markov chain** (lowest log loss, simplest, no optimisation risk), with the 1st-order chain as the reference. The HMM is not justified as superior by this evidence.

## Files

| Path | Content |
|---|---|
| `data_audit.md`, `data_audit.json`, `exclusion_report.csv` | Phase 1 audit; event-level accounting per task and reason |
| `experiment_config.yaml` | exact configuration (copy of `configs/markov_hmm_pilot.yaml`) |
| `split_manifest.json` | final-test, development, pilot ids, strata, pilot folds |
| `dataset_manifest.json`, `event_vocabulary.json` | pilot data counts, vocabularies, target check |
| `metrics_summary.csv`, `metrics_by_fold.csv`, `per_class_metrics.csv`, `confusion_matrices.csv`, `bootstrap_intervals.csv` | evaluation |
| `predictions.csv`, `predictions_fullstream9.csv`, `predictions_*_other_hmm_configurations.csv.gz` | out-of-fold predictions (§9). `predictions*.csv` are 113 MB each: above GitHub's 100 MB limit if ever committed. |
| `transition_counts.csv`, `transition_probabilities.csv` | 1st-order chain on all pilot matches (descriptive) |
| `hmm_parameters.json`, `hmm_training_log.csv`, `hmm_model_selection.json`, `hidden_state_analysis.csv` | HMM parameters, every fit, selection, state analysis |
| `models/` | per-fold models (`cv_fold_models_*.json`) and all-pilot descriptive models |
| `figures/` | heatmaps, emissions, confusion matrices, per-class F1, model comparison, K sensitivity, seed stability, pitch states, timelines |
| `leakage_and_reproducibility_checks.json`, `independent_verification.json`, `environment.json`, `checksums.json` | verification and provenance |

Code: `src/sequence_models/` (`raw_events`, `data_audit`, `tasks`, `spatial`, `splits`, `markov`, `hmm`, `metrics`,
`experiment`, `figures`, `config`), `scripts/run_markov_hmm_pilot.py` (stages `audit`, `split`, `pilot`),
`scripts/verify_markov_hmm_pilot.py`, `configs/markov_hmm_pilot.yaml`.

Reproduce:

```bash
.venv/bin/python scripts/run_markov_hmm_pilot.py audit
.venv/bin/python scripts/run_markov_hmm_pilot.py split
.venv/bin/python scripts/run_markov_hmm_pilot.py pilot
.venv/bin/python scripts/verify_markov_hmm_pilot.py
```

The run takes about 21 minutes on 8 workers.
