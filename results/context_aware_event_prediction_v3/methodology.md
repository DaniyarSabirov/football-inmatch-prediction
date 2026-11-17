# Experiment 3 — methodology

Code:
- modules: `src/sequence_models/v3_events.py`, `v3_taxonomy.py`, `v3_features.py`, `v3_models.py`, `v3_experiment.py`, `v3_diagnostics.py`, `v3_report.py`;
- scripts: `scripts/run_context_aware_v3.py`, `scripts/verify_context_aware_v3.py`, `scripts/demo_explain_prediction.py`;
- configuration: `configs/context_aware_event_prediction_v3.yaml`;
- tests: `tests/test_v3_features_taxonomy.py`, `tests/test_v3_models.py`.

## 1. Data and partitions

- **Development matches only.** 334 development matches: experiment 1's 30 pilot matches = **Tier A**, the other 304 = **Tier B**.
- **Final test.** The 83 final-test matches are refused by every loader; experiment 1's all-match cache is refused too.
- **Folds.** Tier A uses experiment 1's pilot folds; Tier B uses experiment 2's Tier B folds. Both are match-grouped, 5 folds.
- **Tier B is not pristine.** It was inspected in experiment 2. It is a held-out *development* set: repeated feedback limits its independence, and it is not a final test.

## 2. Procedure (preregistered)

1. **Write the configuration before any result.** All candidates and selection rules were written in the YAML before the Tier A screen.
   - **First amendment** (two possession-frame spatial candidates): added after the Tier A *diagnostics*, before any screening result.
   - **Second amendment** (a secondary Shot-ranking hypothesis): added after the Tier A screen, before any Tier B run.
   - Both amendments are recorded in the YAML.
2. **Tier A diagnostics** on experiment 2's stored out-of-fold predictions, to generate hypotheses.
3. **Tier A screen** of every candidate, with nested-CV smoothing and out-of-fold probabilities.
4. **Freeze.** `freeze` applies the rules mechanically and writes `frozen_plan.json` with its SHA-256. `confirm` refuses a modified plan.
5. **Tier B confirmation** of the frozen plan only. Shot thresholds come from Tier A out-of-fold predictions. Models without a Tier A screening run (the rule-defined combination, both `actions15` models) are run on Tier A first to obtain their thresholds.
6. **Tier B diagnostics**, only after freezing.

## 3. Taxonomies

| Name | Rule |
|---|---|
| `filtered8` | 8 production types; next event of these types in the same period |
| `scenarioD15` | Scenario D, 15 types (experiment 2); **main task** |
| `actions15` | the same 15 class names. Records that annotate the *same contest* as their predecessor are skipped (option C). Composite labelling (option B) gives the identical sequence. |
| `raw33` | every raw type, immediate next record; diagnostic only |

The `actions15` companion rules (measured; `taxonomy/annotation_pairs.csv`):

- **K1 — Goal Keeper records without a keeper action** (Shot Faced, Goal Conceded, Penalty Conceded).
  - Shot Faced has no outcome; these records follow the shot and are linked to it in 99.9% of cases.
  - Saves, claims, punches, sweeping and smothers stay "Goal Keeper".
- **K2 — Duel/Tackle companions.** A Duel/Tackle immediately after a Dispossessed (100% of Dispossessed) or after an incomplete Dribble (4,280 of 4,830), at the same timestamp and linked in `related_events`.
- **K3 — the second record of a 50/50 pair**, at the same timestamp and linked.

All taxonomies are **filtered-event tasks**: they predict the next event *of the taxonomy*, not the next raw record.
Log loss is never compared across taxonomies.

## 4. Features (all causal: event *t* and earlier records of the same period)

**Coordinates.**
- The raw event location is in the acting team's attacking frame, per the StatsBomb spec and experiments 1–2.
- Independent check (`taxonomy/orientation_audit.json`): 99.85% of 8,472 shots have x ≥ 60, in every period. The 13 exceptions are listed in that file.
- 360 freeze frames are not used here.
- Missing location: 0.001%; it is its own category.

**Geometry** (120 × 80 units, nominally yards; 1 m = 1/0.9144 units):

| Feature | Definition |
|---|---|
| distance to goal | $\sqrt{(120-x)^2+(40-y)^2}$ |
| shooting angle | $|\mathrm{atan2}(44-y,120-x)-\mathrm{atan2}(36-y,120-x)|$ (posts at y = 36, 44) |
| danger area | penalty area (x ≥ 102, 18 ≤ y ≤ 62) grown by 5 m = 5.468 units on its three open sides |
| zones | `z3` thirds · `b6` 6 bands of 20 · `g12` = `b6` × central (\|y−40\| ≤ 22) / wide · `d7` distance bins (12, 18, 25, 35, 50, 70) · `z3d` thirds with the danger area split off |
| possession frame (`pf_*`) | the location mirrored into the frame of the team causally in possession, so a defender's record is seen from the attacking side |

**Causal possession** (`v3_features.causal_raw_features`).
- The controlling team after record *t* is the team of the latest ball-control record: Pass, completed Ball Receipt, Carry, Dribble, Shot, Ball Recovery, Miscontrol, Clearance, Shield, keeper actions, won Interceptions.
- From it come: duration so far, records so far, progression in the possession frame, same team as the previous record, a recent possession change, time since the previous record, records in the last 10 s, and match-minute bins.
- Backwards-running timestamps (corrupted) become unknown, never negative.
- **StatsBomb `possession` ids are not used**, because they show look-ahead. 7.4% of StatsBomb possessions are credited to a team whose first record is not that team's ball control (e.g. 3,126 start with the opponent's Duel). StatsBomb also merges brief opponent touches (59k boundaries vs 91k causal ones).

**History.** `prev1..prev3` within the period (START before the period start). A reset variant replaces records from before the current causal possession with NEWPOSS.

## 5. Models

**Context Markov** (`v3_models`).
- An ordered feature list; the backoff drops features from the end. `[cur, z3, prev1]` reproduces experiment 2's `mc2_zone` exactly.
- Smoothing: hierarchical Dirichlet (experiment 2), $p_\ell(k\mid c)=\frac{C_\ell(c,k)+\beta p_{\ell-1}(k)}{N_\ell(c)+\beta}$, with β from {1, 4, 16, 64, 256} chosen by nested inner CV.

**Logistic regression** (classical comparison).
- Multinomial with an L2 penalty, C ∈ {0.1, 1} on Tier A.
- Inputs: one-hot cur / prev1 / prev2; 17 continuous causal features with missing indicators; cur × distance and cur × angle; period.
- The standardisation is fitted on the training fold only.

**Not run:** HMMs. Experiment 2's HMMs (K = 12) stay as baselines.

## 6. Decisions (the probabilities are never changed)

| Rule | Definition |
|---|---|
| uniform | argmax |
| moderate + Shot multiplier | Bayes rule with $C(a,k)=m_k$, $m_k=\min(5,(\pi_{\max}/\pi_k)^{0.5})$, and $m_{\text{Shot}}$ × λ, λ ∈ {1, 1.25, 1.5, 2} |
| Shot threshold | predict Shot iff p(Shot) ≥ τ, with τ fixed on Tier A for budgets of 5 / 10 / 20 false Shots per 1,000 events |

The training-fold priors π come from training rows only. These are sensitivity analyses: no economic utility is claimed.

## 7. Metrics and uncertainty

- **Probability quality:** log loss (primary), Brier score, classwise / top-label ECE.
- **Decision quality:** accuracy; macro-F1 (the mean of per-class F1, **not** accuracy); balanced accuracy; per-class precision / recall / F1.
- **Ranking:** per-class average precision (AP) and ROC AUC.
- **Shot:** recall, precision, F1, false positives per 1,000 events.
- **Uncertainty:** paired match-level bootstrap (1,000 replicates) for differences against the reference, on identical rows and folds.
- **Relative change:** 100·(model − reference)/reference.

## 8. Computational notes

| Stage | Wall time | Notes |
|---|---|---|
| Feature build | 28 s | peak 2.1 GB |
| Tier A screen | 2.6 min (23 configurations) | |
| Tier B confirmation | 16 min | peak below 2 GB, single process |

Every configuration is checkpointed (`data/interim/sequence_models/v3/checkpoints/`, keyed by a hash of specification, folds and data size) and logged with an ETA. No stage exceeded 30 minutes.
