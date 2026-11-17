# Experiment 3 — context-aware next-event prediction, taxonomy audit and rare-event diagnostics

Date: 2026-10-28. **Development data only:** the 83 final-test matches were never read.
- Experiments 1 and 2 are unchanged (58/58 and 60/60 files re-verified against their checksums).
- The production dataset and the visualization pipeline were not modified. No commit was made.
- Definitions: `methodology.md`. Code guide: `docs/sequence_models_code_guide.md` (see also `code_navigation.md`).

**Evidence labels:**
- **[A]** Tier A: 30 matches; exploratory, and used for every choice.
- **[B]** Tier B: 304 matches; frozen plan only.
- **[V]** verified.
- **[H]** interpretation.

**Tier B caveat:** Tier B was already inspected in experiment 2. Its results are held-out **development** estimates, not a pristine final test.

## 1. Executive summary

| Finding | Evidence |
|---|---|
| **Shot is ranked moderately well, but argmax can never choose it.** p(Shot) is 15.8× better than chance at ranking Shots, yet argmax picks it for only 4% of true Shots because Pass is almost always more likely. | [B] `mc2_zone`: Shot AP 0.162 vs prevalence 0.010, ROC AUC 0.92 |
| **Better information helps Shot ranking far more than overall log loss.** | [B] Shot AP: `mc2_zone` 0.162 → `g12` grid 0.300 (+85%) → logistic regression 0.360 (+122%) |
| **Best probability model: the frozen combination** `[cur, b6, prev1, same_team, prev2, dt]`. | [B] log loss 0.9851 vs 1.0193 (**−3.36%**, CI [−0.0353, −0.0332]); Brier −3.38%; macro-F1 0.303 vs 0.275 |
| Every single context ablation helps a little. | [B] −0.9% to −1.6% log loss; all CIs exclude 0 |
| **Shot recall at a fixed false-positive budget** (thresholds fixed on Tier A) | [B] at 10 false Shots / 1,000 events: `mc2_zone` 0.28 → combination 0.34 → `g12` 0.43 → logistic regression 0.45. At 20 / 1,000: up to 0.57–0.59 |
| **Scenario D's best-scored rare classes are partly bookkeeping.** Without StatsBomb's annotation companions (`actions15`), their AP collapses. | [A] Goal Keeper 0.771 → 0.162; Duel 0.622 → 0.079; 50/50 0.53 → 0.01 |
| **StatsBomb possession ids leak future information** and are not used. | [V] 7.4% of StatsBomb possessions are credited before the team controls the ball |
| Everything ran in about 20 minutes on the M1. No HMM was fitted. | `runtime_report.csv` |

## 2. Why Shots are missed (`diagnostics/`)

Tier A diagnostics were used to form hypotheses; Tier B values were computed after freezing. All rows below are `mc2_zone` on Scenario D, Tier B (`diagnostics/tier_b/rare_event_class_diagnostics.csv`).

| Class | Prevalence | AP | AP / prevalence | ROC AUC | Argmax recall | Moderate recall / precision | False positives / 1,000 (moderate) |
|---|---|---|---|---|---|---|---|
| Shot | 1.0% | 0.162 | 15.8 | 0.923 | 0.040 | 0.177 / 0.356 | 3.3 |
| Interception | 0.9% | 0.030 | 3.3 | 0.821 | 0 | 0 / 0 | 0 |
| Miscontrol | 1.0% | 0.050 | 4.8 | 0.802 | 0 | 0.006 / 0.31 | 0.1 |
| Foul Committed | 0.9% | 0.053 | 5.8 | 0.857 | 0 | 0 / 0 | 0 |
| Dispossessed | 0.9% | 0.094 | 10.5 | 0.942 | 0 | 0.103 / 0.108 | 7.6 |
| Clearance | 1.4% | 0.254 | 17.7 | 0.917 | 0.178 | 0.322 / 0.252 | 13.7 |
| Dribble | 1.2% | 0.101 | 8.6 | 0.926 | 0 | 0.296 / 0.126 | 24.0 |
| Block | 1.5% | 0.134 | 9.0 | 0.860 | 0 | 0 / 0.04 | 0.1 |

**Ranking vs decision [B].**
- For Shot, the 90th percentile of p(Shot) on true Shots is 0.38, and the maximum is 0.56.
- Pass is almost always more probable, so argmax rarely selects Shot.
- True Shots are predicted as Pass (62%) or Carry (38%) [A].
- The main problem is therefore **both**:
  - the multiclass rule hides a useful ranking: a threshold of p ≥ 0.10 already gives recall 0.23 at precision 0.25 [A];
  - the ranking itself is only moderate: AP 0.16 means that a high recall always comes with many false positives.

**Interception, Miscontrol and Foul Committed** are barely rankable (AP 0.03–0.05, lift 3–6). Event type, recent history and zone carry almost no signal for them.

**Shot by context [A]** (`diagnostics/tier_a/shot_breakdowns.csv`).
- Shot rate falls steeply with distance to goal: 7.3% of next events within 12 units, 0.1% at 50–70.
- **38% of Shots follow a defender's record** (Pressure 171, aerial Duel 84, Foul Committed 24, …). Their acting-team location is far from the attacked goal. This motivated the possession-frame and grid candidates.
- Per match, moderate-rule Shot recall varies from 0 to 0.50 (mean 0.18, SD 0.09; 304 matches [B]).

## 3. Taxonomy (`taxonomy/`; 334 development matches)

| | filtered8 | scenarioD15 | **actions15** | raw33 (diagnostic) |
|---|---|---|---|---|
| Raw-event coverage | 57.6% | 71.2% | 69.6% | 100% |
| Rows skipping ≥ 1 raw record | 60.4% | 39.4% | 42.4% | 0% |
| Same-timestamp (zero-gap) rows | 5.0% | 7.2% | 5.9% | 31.5% |
| Rows in near-deterministic cells (P ≥ 0.9) | 0% | 0.9% (Dispossessed → Duel, P = 1.000) | **0%** | 1.8% |
| Smallest class | Dispossessed 7,944 | 50/50 1,327 | 50/50 668 | Own Goal For 41 |
| H(target \| current), bits | 1.05 | 1.69 | 1.71 | 1.36 |

**What `skipped_records.csv` shows.**
- What Scenario D skips is mostly companions: Ball Receipt\* 345k, Foul Won, Dribbled Past.
- The rest is administrative or stoppage records: 4.4k + 2.7k.
- No on-ball action is skipped except Shield (438).

**Same-timestamp records are recording conventions, not simultaneity** (`same_timestamp_audit.csv`).
- Every Carry starts at its predecessor's timestamp (Ball Receipt → Carry 100%; Ball Recovery, Dribble and Duel → Carry 100%).
- Companion pairs share a timestamp: Dispossessed → Duel 100%, Foul Committed → Foul Won 99.9%, Dribbled Past → Dribble 100%.
- **Event order, not the clock, is the reliable sequence.**

**Dispossessed → Duel.**
- 100% of Dispossessed records are followed by a Duel/Tackle at the same timestamp, linked in `related_events`. It is one contest recorded twice.
- Every one of the 12,221 Tackle duels is such a companion: of a Dispossessed, or of an incomplete Dribble.
- So "Duel" in `actions15` means aerial duels only.

**Goal Keeper.**
- 63% of its records (Shot Faced, Goal Conceded, Penalty Conceded) are outcome annotations of the preceding shot, without a keeper action.
- 37% are genuine keeper actions: saves, claims, punches, sweeping, smothers.

**Option A vs B/C** [A] (same reference model; log loss not compared):

| Class | AP in Scenario D | AP in `actions15` | Argmax recall in Scenario D |
|---|---|---|---|
| Goal Keeper | 0.771 | 0.162 | 0.84 |
| Duel | 0.622 | 0.079 | 0.40 |
| 50/50 | 0.53 | 0.01 | 0.51 |

These classes do contain independent football information (keeper saves, aerial contests), but most of their Scenario D score came from predicting the companion record.

**Recommendation:** `actions15` as the primary taxonomy for the thesis (mapping: `taxonomy/event_mapping.csv`).
- **Target meaning:** "the next football action, where an action is one contest between players, recorded once".
- **Inclusion:** the 15 Scenario D action types.
- **Exclusion:**
  - administrative, stoppage, referee and retrospective records;
  - the companions Ball Receipt\*, Foul Won and Dribbled Past;
  - the K1–K3 annotation records;
  - Shield (< 1,000 occurrences).
- **The controlled comparison** stayed on Scenario D (it is valid, only partly mechanical). `actions15` was confirmed as a separate task: the combination improves its log loss by −3.06% [B] (1.0083 vs 1.0402; CI [−0.0327, −0.0310]).

## 4. Spatial context (`spatial_ablation.csv`)

| Representation (with order 2) | Log loss [A] | Δ [A] | Δ [B] | Shot AP [A] | Shot AP [B] |
|---|---|---|---|---|---|
| thirds `z3` (reference) | 1.0092 | — | — | 0.145 | 0.162 |
| 6 bands `b6` | 1.0025 | −0.67% | −0.92% | 0.209 | 0.226 |
| **6 × 2 grid `g12`** | 1.0028 | −0.64% | **−1.60%** | **0.268** | **0.300** |
| distance bins `d7` | 1.0062 | −0.30% | — | 0.178 | — |
| thirds + danger area `z3d` | 1.0045 | −0.48% | — | 0.171 | — |
| distance bins, possession frame | 1.0066 | −0.27% | — | 0.248 | — |
| thirds + danger, possession frame | 1.0064 | −0.29% | — | 0.176 | — |

- **[B] the grid wins.** The 6 × 2 grid (longitudinal bands × central/wide) is the best spatial representation on Tier B, for log loss and especially for Shot ranking.
- **[A] distance bins** in the possession frame strongly help Shot ranking (AP 0.248), but help log loss least.
- **Continuous geometry** enters only through the logistic regression (§7).
- **Orientation verified [V]:** 99.85% of 8,472 shots lie at x ≥ 60 in every period (including extra time); 99.3% of keeper actions lie at x ≤ 30, 100% of kick-offs at 55 ≤ x ≤ 65, and 92% of clearances at x ≤ 50 (`taxonomy/orientation_audit.json`).

## 5. Possession and temporal context (`possession_temporal_ablation.csv`)

| Candidate (added to `mc2_zone`) | Δ log loss [A] | Δ [B] |
|---|---|---|
| same team as previous record | **−0.91%** | −1.26% |
| duration × progression (16 cells) | −0.58% | — |
| possession duration | −0.43% | — |
| records in possession | −0.39% | — |
| recent possession change | −0.34% | — |
| progression | −0.27% | — |
| time since previous record | −0.57% | −1.02% |
| tempo (records in last 10 s) | +0.01% (CI incl. 0) | — |
| match minute | **+0.62% (worse)** | — |

**Is a long possession alone a Shot signal? No.**
- Duration adds 0.43%; duration × progression adds 0.58% [A].
- In the attacking third, P(Shot next) *falls* slightly with duration: 2.7% at < 5 s, 2.1% at > 30 s [A].
- It *rises* with duration (0.6% → 3.1%) only for records in the acting team's defensive third. Those are mostly defenders under sustained attack (`figures/shot_rate_zone_x_possession_duration_tier_a.png`).
- Possession duration matters only together with field position and with who is acting [H].

## 6. Markov order (`markov_order_comparison.csv`)

| Order | Δ log loss [A] | Δ [B] | Mean observed contexts (most specific level) [A] |
|---|---|---|---|
| 2 (reference) | — | — | 340 |
| 3 | **−0.71%** | **−1.13%** | 1,418 |
| 4 | −0.52% | — | 3,986 |
| 3, reset at possession change | −0.57% | — | 1,118 |
| 4, reset | −0.38% | — | 2,508 |

- **Order 3 beats order 2** on both tiers.
- **Order 4 is worse than order 3.** Contexts become sparse and the backoff dilutes them.
- **Resetting history** at possession changes does not help.

## 7. Model comparison, Tier B [B] (`model_comparison.csv`; 701,144 rows; CIs from the match bootstrap)

| Model | Log loss [95% CI] | Δ vs `mc2_zone` | Brier | Macro-F1 | Balanced acc. | Shot AP | Shot ROC AUC |
|---|---|---|---|---|---|---|---|
| `mc1` (experiment 1/2 baseline) | 1.1520 | +13.0% | 0.519 | 0.215 | 0.218 | 0.026 | 0.70 |
| **`mc2_zone`** (reference) | 1.0193 [1.0072, 1.0321] | — | 0.4792 | 0.275 | 0.260 | 0.162 | 0.923 |
| `b6` | 1.0098 | −0.92% | 0.4753 | 0.280 | 0.263 | 0.226 | 0.940 |
| order 3 | 1.0077 | −1.13% | 0.4751 | 0.279 | 0.263 | 0.187 | 0.925 |
| time since previous | 1.0089 | −1.02% | 0.4734 | 0.290 | 0.274 | 0.174 | 0.924 |
| same team | 1.0064 | −1.26% | 0.4737 | 0.288 | 0.269 | 0.164 | 0.923 |
| `g12` | 1.0030 | −1.60% | 0.4722 | 0.293 | 0.275 | 0.300 | 0.955 |
| **combination** | **0.9851** [0.9733, 0.9974] | **−3.36%** | **0.4630** | 0.303 | 0.282 | 0.249 | 0.941 |
| logistic regression (C = 0.1) | 1.0249 | +0.55% (worse) | 0.4811 | 0.303 | 0.279 | **0.360** | **0.959** |

- **Calibration:** good for every context Markov model (classwise ECE ≤ 0.0012, top-label ≤ 0.004; `mc1` 0.0019 / 0.012). The logistic regression is less calibrated (top-label ECE 0.018).
- **Per class:** the combination improves AP for all 12 rare classes (50/50 essentially unchanged, 0.537 → 0.538), e.g. Clearance 0.254 → 0.355, Duel 0.639 → 0.724, Interception 0.030 → 0.059 (`tier_b/rare_event_class_diagnostics_scenarioD15.csv`).
- **Decision quality (macro-F1) improves less than probability quality.**

## 8. Shot-specific cost sensitivity [B] (`cost_sensitive_shot_comparison.csv`, `figures/shot_*`)

Moderate scheme with the Shot miss cost × λ (probabilities, log loss and calibration unchanged):

| Model | λ | Shot recall | Shot precision | Shot F1 | False Shots / 1,000 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| `mc2_zone` | 1.0 | 0.177 | 0.356 | 0.236 | 3.3 | 0.330 | 0.640 |
| `mc2_zone` | 1.5 (λ\* from Tier A) | 0.270 | 0.224 | 0.245 | 9.6 | 0.330 | 0.636 |
| `mc2_zone` | 2.0 | 0.331 | 0.163 | 0.219 | 17.5 | 0.326 | 0.634 |
| combination | 1.0 | 0.287 | 0.294 | 0.290 | 7.1 | 0.360 | 0.646 |
| `g12` | 1.0 | 0.471 | 0.253 | 0.329 | 14.3 | 0.336 | 0.645 |
| logistic regression | 1.0 | 0.461 | 0.299 | 0.363 | 11.1 | 0.354 | 0.629 |

Raising λ trades Shot precision for Shot recall. Shot F1 peaks at λ ≈ 1–1.5 for every model, and macro-F1 is flat or falls slightly (≤ 0.007).

**Best attainable Shot recall under false-positive budgets** (τ fixed on Tier A, realised on Tier B; `shot_threshold_rules.csv`):

| Budget (false Shots / 1,000) | `mc2_zone` | combination | `g12` | logistic regression |
|---|---|---|---|---|
| 5 | 0.179 (prec 0.35) | 0.266 | 0.240 | **0.351** (prec 0.42) |
| 10 | 0.276 | 0.337 | 0.425 | **0.451** (prec 0.31) |
| 20 | 0.333 | 0.441 | 0.568 | **0.587** (prec 0.23) |

The realised false-positive rates stay close to the budgets (e.g. logistic regression at 10: 10.1).

**New information vs decision rule (Shot).**
- At the *same* false-positive budget (10 / 1,000), information raises recall from 0.28 (`mc2_zone`) to 0.43–0.45 (`g12` / logistic regression).
- The decision rule alone moves `mc2_zone` from 0.04 (argmax) to 0.28.
- **Macro-F1:**

  | Step | Macro-F1 | Gain |
  |---|---|---|
  | `mc2_zone` argmax | 0.275 | — |
  | `mc2_zone` + moderate costs (decision) | 0.330 | +0.055 |
  | combination argmax (information) | 0.303 | +0.027 |
  | combination + moderate costs (both) | 0.360 | +0.085 |

## 9. Verification, reproducibility, leakage

| Check | Result |
|---|---|
| New tests (17) | **17 passed**: geometry by hand, danger-area unit conversion, zone bins, causal possession by hand, truncation invariance, history tokens, companion rules, reference = experiment-2 `mc2_zone` (1e-15), nested-CV leakage, LR leakage and column stability, Shot cost matrix, threshold rule, checkpoints |
| Whole project suite | **469 passed, 0 failed** (452 earlier + 17 new); no existing test changed |
| Independent verification (`scripts/verify_context_aware_v3.py`, no pipeline import) | **passed**: 0 final-test matches in feature tables; targets re-derived from raw JSON (5 matches, 0 mismatches); geometry deviation 1.4e-14; all metrics ≤ 2e-16, except one documented case (below); threshold rules 3.6e-15 |
| Reproducibility (`reproducibility_checks.json`) | reference and combination re-fitted without checkpoints: **bit-identical**; reference vs experiment 2's stored `mc2_zone`: 3.0e-8 (float32 storage) |
| Leakage (`leakage_checks.json`) | tiers disjoint; folds match-grouped; plan hash verified (`frozen_plan.sha256`); thresholds and λ\* from Tier A only; StatsBomb possession ids, outcomes and later records never used as inputs |

**The one documented verification case.** For the combination, scikit-learn's log loss differs by 1.6e-6 because it clips probabilities at machine epsilon. One Tier B row has a true-class probability of 7.3e-17 (`independent_verification.json`). This is a genuine tail risk of large, well-supported contexts.

## 10. Limitations

1. **Tier B is not pristine** (inspected in experiment 2). Gains of 1–3% are small and could partly reflect development feedback, even though the plan was frozen.
2. **Two documented amendments** (possession-frame candidates; the `g12` secondary hypothesis) were added after Tier A results, before Tier B.
3. **The combination's feature order** (by Tier A gain) is one of many possible backoff orders. It was not tuned.
4. **The logistic regression has main effects plus two interaction blocks only.** Its log loss is worse, but its Shot ranking is the best. A richer classical model may combine both.
5. **Zones and possession features are new derived inputs**, not yet in the approved allowlist. Possession reconstruction is rule-based: for example, Clearance counts as control.
6. **Cost multipliers have no real-world utility.** They are a sensitivity analysis.
7. **`actions15`** has 50/50 at only 668 rows and a narrower Duel class (aerial only).
8. **Near-zero probabilities in sparse contexts** (minimum 7e-17) could become a problem on new data. A probability floor or a stronger concentration would guard against it.

## 11. Answers to the questions

1. **Why does the model miss Shots?**
   - The probability mass at a Shot moment is spread over the frequent continuations (Pass, Carry, Pressure). Argmax therefore never picks Shot.
   - The reference's zones are too coarse ("attacking third" = x ≥ 80): in the demo event, a Carry *inside the box* gets p(Shot) = 0.029.
   - A third of Shots follow defenders' records whose acting-team location is misleading.
2. **Is Shot ranking better than top-1 recall suggests?** Yes. AP is 15.8× the prevalence and ROC AUC 0.92 for `mc2_zone`, rising to AP 0.30 (`g12`) and 0.36 (logistic regression), while argmax recall is 0.04.
3. **Best Shot recall under reasonable false-positive constraints?** 0.35 / 0.45 / 0.59 at 5 / 10 / 20 false Shots per 1,000 events (logistic regression; Markov `g12`: 0.24 / 0.43 / 0.57), with thresholds fixed on Tier A.
4. **Best spatial representation?** The 6 × 2 grid (longitudinal bands × central/wide). Distance-to-goal bins in the possession frame are a close second for Shot ranking.
5. **Do possession duration and progression help?** Only slightly (−0.4% to −0.6% log loss on Tier A), and only jointly with field position. The most useful possession signal is whether the current record is by the same team as the previous one (−1.26% [B]).
6. **Do orders 3 or 4 beat order 2?** Order 3 does (−1.13% [B]). Order 4 does not improve on order 3 (sparsity).
7. **Is the next-meaningful-action taxonomy better?** Yes, conceptually and empirically. It removes every near-deterministic transition and exposes how much of Scenario D's Goal Keeper, Duel and 50/50 performance was bookkeeping. Recommended: `actions15`.
8. **Information vs cost-sensitive decisions?**
   - For Shot recall at a fixed false-positive budget: information adds +0.15–0.18; the decision rule (argmax → threshold) adds +0.24.
   - For macro-F1: decisions add +0.055, information +0.027, and both together +0.085.
   - Only information improves the probabilities: log loss −3.4%.
9. **What is worthwhile on this machine?**
   - Count-based Markov models: seconds on Tier A, about 20–60 s per model on Tier B.
   - Logistic regression: about 2 min per Tier B fold.
   - The full study took about 20 minutes. HMMs (hours, worse results) are not worthwhile.
10. **More Markov, or another classical model?**
    - The Markov family is close to saturation. The best combination is only 3.4% better, deeper histories and finer cells get sparse, and continuous geometry cannot enter it without binning.
    - The logistic regression already ranks Shots best despite worse overall log loss.
    - **Next:** a regularised classical model that combines the Markov context with continuous geometry. Options: logistic regression with the Markov log-probabilities as an input (stacking), or gradient-boosted trees on the same causal features. Use the same tiers, folds and frozen-plan procedure, on `actions15`.

## Files

| Path | Content |
|---|---|
| `experiment_config.yaml`, `frozen_plan.json`, `frozen_plan.sha256` | preregistered configuration; frozen Tier B plan |
| `taxonomy/` | `taxonomy_audit.csv`, `event_mapping.csv`, `raw_types_and_subtypes.csv`, `annotation_pairs.csv`, `skipped_records.csv`, `same_timestamp_audit.csv`, `class_distribution.csv`, `orientation_audit.json`, `possession_audit.json` |
| `diagnostics/tier_a`, `diagnostics/tier_b` | rare-event class diagnostics, "predicted instead", thresholds, Shot breakdowns, per-match recall, PR curves of experiment-2 models |
| `tier_a/`, `tier_b/` | summaries, per-class metrics, paired differences, calibration, Shot thresholds, smoothing choices, LR fits, PR curves |
| `model_comparison.csv`, `paired_differences.csv`, `per_class_metrics.csv`, `calibration_metrics.csv` | model comparison with CIs and % change |
| `spatial_ablation.csv`, `possession_temporal_ablation.csv`, `markov_order_comparison.csv` | ablations |
| `cost_sensitive_shot_comparison.csv`, `shot_threshold_rules.csv` | Shot cost multipliers and false-positive budgets |
| `leakage_checks.json`, `reproducibility_checks.json`, `independent_verification.json`, `runtime_report.csv`, `environment.json`, `checksums.json` | verification, provenance, runtime |
| `figures/` | Shot PR / false-positive curves, log-loss change vs reference (both tiers), ranking vs decision, Shot cost sweeps, taxonomy class shares, Shot rate by zone × duration |

**Reproduce** (about 20 min; every configuration cached and resumable):

```bash
.venv/bin/python scripts/run_context_aware_v3.py taxonomy
.venv/bin/python scripts/run_context_aware_v3.py features
.venv/bin/python scripts/run_context_aware_v3.py diagnose --tier tier_a
.venv/bin/python scripts/run_context_aware_v3.py screen
.venv/bin/python scripts/run_context_aware_v3.py freeze      # refuses to overwrite an existing plan
.venv/bin/python scripts/run_context_aware_v3.py confirm
.venv/bin/python scripts/run_context_aware_v3.py diagnose --tier tier_b
.venv/bin/python scripts/run_context_aware_v3.py report
.venv/bin/python scripts/verify_context_aware_v3.py
```
