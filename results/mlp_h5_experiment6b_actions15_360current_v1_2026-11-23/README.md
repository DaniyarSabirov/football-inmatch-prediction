# Experiment 6B: MLP-H5 (feed-forward network, five-event history) on `actions15_360current_v1`

Date: 2026-11-23. Method and interpretation are in `docs/thesis/08_mlp_h5.md`.

**Question.** With exactly the five-event history of XGB-H5 and MLR-H5, does a nonlinear MLP
beat MLR-H5 and approach or exceed XGB-H5?

**Answer (primary metric: pooled out-of-fold log loss).**

| Comparison | Pooled log loss | Paired difference (MLP-H5 − reference) | 95 % match-bootstrap CI | Folds |
|---|---|---:|---|---|
| vs. MLR-H5 | 1.05267 vs. 1.10297 | **−0.0503** | [−0.0517, −0.0489] | MLP-H5 better in 5/5 |
| vs. XGB-H5 | 1.05267 vs. 1.05185 | **+0.00082** | [+0.00052, +0.00114] | XGB-H5 better in 5/5 |

- **Against MLR-H5.** The MLP is much better, so the nonlinear model removes almost the whole
  gap that MLR-H5 has to XGB-H5.
- **Against XGB-H5.** XGB-H5 stays very slightly better, by 0.08 % relative. This holds for all
  three prespecified refit seeds as well (+0.0003 to +0.0008).

**Important limitations.**
- **Training data.** The comparison with XGB-H5 is a comparison of complete training
  procedures. It is not controlled for training-data quantity.
  - The MLP refits updated their parameters on 89.8–90.5 % of the outer-training rows: 219
    or 220 of 243–244 matches. The remaining 24 matches per fold served only for early
    stopping.
  - XGB-H5 trained on all outer-training rows.
  - Per-fold numbers are in `results/audit_experiments6a_6b_2026-11-24/training_population_6b_vs_xgb_h5.csv`.
- **Scope of the CIs.** The CIs resample evaluation matches for fixed predictions. They do not
  include:
  - refitting or optimisation randomness (seed spread 0.0005);
  - hyperparameter selection;
  - generalisation to other competitions.
- **No equivalence claim.** No equivalence margin was defined, so "close to XGB-H5" is not a
  claim of formal equivalence.
- **Source provenance.**
  - **At run time.** The run started from commit `a2343e9` with uncommitted experiment files.
  - **Identification.** The executed code is identified by the per-file SHA-256 in
    `environment.json`.
  - **Verification.** The final audit verified that the published files are byte-identical to
    the executed ones.

## Reproduce

```bash
python scripts/run_mlp_h5_experiment6b.py --smoke                 # fold-0 inner data only: timing, memory
python scripts/supervise_mlp_h5_experiment6b.py --detach          # full run + independent audit, detached, caffeinate -d -i
python scripts/supervise_mlp_h5_experiment6b.py --status          # inspect a run (never starts anything)
python scripts/verify_mlp_h5_experiment6b.py                      # independent audit only
python -m pytest tests/test_mlp_h5_experiment6b.py
```

**Run.** `exp6b-20261123T190641-0ebe66f2`.

| Item | Value |
|---|---|
| Supervisor PID | 39374, in its own session |
| Runner PID | 39377 |
| Audit PID | 40480 |
| `caffeinate -d -i -w 39374` PID | 39376 |
| Exit codes | experiment 0, independent audit 0 (`data/interim/mlp_h5_experiment6b/run_status.json`, `COMPLETED`) |
| Wall time | 4,017 s for the experiment + 16 s for the audit |
| Hardware | Apple M1, 8 GB, CPU, 1 thread |

**Software.** The source commit is `a2343e9`. The experiment files were uncommitted at run time;
they are listed in `environment.json`. Versions: Python 3.11.1, torch 2.15.0, NumPy 2.4.6,
scikit-learn 1.9.0, pandas 3.0.5.

## 1. Setup

| Item | Value |
|---|---|
| Rows | 690,147 Protocol-B rows of Tier B (304 matches). Ids, order, targets, folds and history are identical row by row to experiments 5 / 5B / 6A (element-wise and by manifest hashes) |
| Final test | 83 matches, never loaded |
| Input | \((e_{t-4},\dots,e_t)\) from `event_history`, unchanged; the 79-column position-specific one-hot of 6A (verified equal on all rows) |
| Network | ReLU MLP, 15 logits, softmax. First layer \(W_1\phi(x)\) = sum of the 5 active columns of \(W_1\) |
| Loss | Mean cross-entropy on logits (`F.cross_entropy`). No class weights or resampling |
| Optimiser | AdamW, β = (0.9, 0.999), batch size 512. Decoupled weight decay on weight matrices only |
| Search | 8 prespecified configurations, A–H (`search_candidates.csv`), frozen before the run |
| Early stopping | On a 10 % match-grouped stopping subset of each fit's OWN training matches. Patience 5, max 60 epochs, best-epoch weights returned |
| Selection | Pooled log loss on the held inner fold. That fold never influences stopping |
| Refit | Outer-training matches minus a 10 % stopping subset. Same stopping rule. Predicts the outer-validation rows once |
| Determinism | CPU, 1 thread, `torch.use_deterministic_algorithms(True)`. All seeds derive from `SeedSequence`, re-seeded per epoch |

**Reproducibility.** A fresh retraining of fold 0's refit reproduced weights and predictions
bit-identically (max |Δp| = 0).

## 2. Results (pooled out-of-fold, same 690,147 rows)

| Model | Log loss [95 % CI] | Brier | Accuracy | Macro-F1 | Weighted-F1 | Shot AP | Shot recall (argmax) | Shot precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic F0 | 1.15850 [1.1445, 1.1738] | 0.52249 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 / 7,212 | – |
| Markov 2 | 1.06807 [1.0554, 1.0819] | 0.49375 | 0.6659 | 0.1204 | 0.5731 | 0.0527 | 0 / 7,212 | – |
| XGB-H2 | 1.06761 [1.0549, 1.0815] | 0.49365 | 0.6659 | 0.1208 | 0.5731 | 0.0529 | 0 / 7,212 | – |
| XGB-H5 | 1.05185 [1.0394, 1.0654] | 0.48798 | 0.6668 | 0.1325 | 0.5778 | 0.0769 | 12 / 7,212 | 0.400 |
| MLR-H5 | 1.10297 [1.0898, 1.1174] | 0.50569 | 0.6491 | 0.1219 | 0.5579 | 0.0585 | 1 / 7,212 | 0.167 |
| **MLP-H5** | **1.05267 [1.0403, 1.0662]** | **0.48838** | **0.6668** | **0.1302** | **0.5762** | **0.0757** | **13 / 7,212** | **0.255** |

- **Prior models.** They were recomputed from their saved predictions. The recomputed values
  equal experiment 6A's tables, CIs included, within 2.5e-13.
- **Argmax metrics.** Accuracy, F1 and Shot recall are argmax decisions.

### Paired differences (MLP-H5 − reference). Match-bootstrap 95 % CI (1,000 replicates, seed 7)

| Reference | Δ log loss | rel. | CI | MLP-H5 better in | Δ accuracy [CI] | Δ macro-F1 [CI] | Δ Shot AP [CI] |
|---|---:|---:|---|---:|---|---|---|
| **XGB-H5** | +0.00082 | +0.08 % | [+0.00052, +0.00114] | 0 / 5 | −0.00004 [−0.00033, +0.00027] | −0.0024 [−0.0032, −0.0015] | −0.0013 [−0.0033, +0.0006] |
| **MLR-H5** | −0.05030 | −4.6 % | [−0.0517, −0.0489] | 5 / 5 | +0.0177 [+0.0168, +0.0187] | +0.0083 [+0.0070, +0.0095] | +0.0172 [+0.0146, +0.0197] |
| Markov 2 | −0.01540 | −1.4 % | [−0.0162, −0.0147] | 5 / 5 | +0.0009 | +0.0097 | +0.0229 |
| XGB-H2 | −0.01494 | −1.4 % | [−0.0157, −0.0143] | 5 / 5 | +0.0009 | +0.0093 | +0.0228 |
| Logistic F0 | −0.10583 | −9.1 % | [−0.1084, −0.1035] | 5 / 5 | +0.0163 | +0.0297 | +0.0420 |

**Independent bootstrap.** The independent verifier ran its own bootstrap (2,000 replicates,
seed 20261124). Against XGB-H5 it gives [+0.00050, +0.00114].

### Per fold (log loss)

| Fold | Selected | MLP-H5 | XGB-H5 | MLR-H5 | Markov 2 |
|---|---|---:|---:|---:|---:|
| 0 | F | 1.05378 | 1.05345 | 1.10450 | 1.06980 |
| 1 | F | 1.04920 | 1.04873 | 1.10029 | 1.06511 |
| 2 | F | 1.05614 | 1.05418 | 1.10522 | 1.07074 |
| 3 | F | 1.05303 | 1.05279 | 1.10338 | 1.06857 |
| 4 | E | 1.05118 | 1.05009 | 1.10145 | 1.06614 |

- **F** = [128, 64], weight decay 1e-4, dropout 0.1, lr 1e-3.
- **E** = [64, 32], with the same regularisation. In fold 4 their inner-CV scores differ by only 2e-6 (E 1.054277 vs. F 1.054279), so the choice is practically a tie.

### Seed robustness (prespecified; replicate 0 is primary, never chosen by result)

| Replicate | Pooled log loss | Δ vs XGB-H5 | Δ vs MLR-H5 |
|---|---:|---:|---:|
| 0 (primary) | 1.05267 | +0.00082 | −0.0503 |
| 1 | 1.05215 | +0.00030 | −0.0508 |
| 2 | 1.05258 | +0.00073 | −0.0504 |

**Size of the seed effect.** The spread caused by the random seed is 0.0005, which is
comparable to the gap to XGB-H5. The sign of that gap is the same for all three seeds.

## 3. Where the differences arise (descriptive, `nll_difference_by_stratum.csv`)

- **Pressure rows vs. MLR-H5.** For rows whose current event is a Pressure, MLP-H5 is better
  than MLR-H5 by 0.234 NLL per row. This is 0.029 of the 0.050 total gap. In experiment 6A
  these rows carried most of MLR-H5's deficit.
- **Interpretation (an inference).** An input-identical nonlinear model closes this gap.
  This is consistent with the 6A interpretation that interactions between positions matter.
- **Gap to XGB-H5.** It is spread over the main current classes: Pass, Carry and Ball Recovery
  are slightly worse for MLP-H5, Pressure is slightly better. There is no single dominant
  stratum.
- **Held-out permutation importance.** The increase in log loss is:
  - \(e_t\): +3.96;
  - \(e_{t-1}\): +0.466;
  - \(e_{t-2}\): +0.048;
  - \(e_{t-3}\): +0.007;
  - \(e_{t-4}\): +0.003;
  - \(e_{t-4..t-2}\) jointly: +0.049.

## 4. Training

| Item | Value |
|---|---|
| Fits | 175 (160 inner, 5 primary refits, 10 robustness refits), 5,363 epochs in total, 3,782 s of fitting |
| Epochs | Inner fits stopped after 13–60 epochs (median 29). The best epoch had a median of 24 |
| Max-epoch cap | It bound for 14 inner fits, all of candidate **H** (lr 3e-4, never selected) |
| Refits | Best epochs 8–38 |
| Warnings | None. Loss, gradient and parameter checks after every batch and epoch passed |

## 5. Integrity

| Check | Result |
|---|---|
| Run's own audit (`integrity_audit.csv`) | 51 checks: 48 PASS, 3 INFO |
| Independent verifier (`post_run_verification.json`) | 38 checks, PASS |
| NumPy forward pass of the stored weights on an independently rebuilt history | Equals the stored probabilities (max \|Δp\| 5.5e-7; float32 network vs. float64 check) |
| Raw-JSON rebuild | 707 rows in 11 strata |
| Historical result directories (4) and inputs | Unchanged before and after |

The audit also covers row identity and folds, the data-access audit and the metrics recomputed
with scikit-learn.

## 6. Files

| File | Content |
|---|---|
| `model_comparison.csv`, `metrics_by_fold.csv`, `per_class_metrics.csv`, `confusion_matrices.csv`, `predicted_class_distribution.csv` | metrics of all 9 models on the same rows |
| `paired_differences.csv`, `paired_differences_by_fold.csv` | paired comparisons with CIs and fold signs |
| `seed_robustness.csv` | the three refit seeds |
| `nll_difference_by_stratum.csv`, `permutation_importance_mlp_h5.csv` | descriptive analyses |
| `search_candidates.csv`, `inner_cv_scores.csv`, `hyperparameter_choices.csv`, `fit_log.csv`, `training_histories.csv`, `timings.csv`, `fold_counts.csv` | search, every fit, per-epoch curves |
| `feature_verification.json`, `raw_feature_verification.csv` | input checks |
| `baseline_recomputation_check.csv` | prior models recomputed |
| `integrity_audit.csv`, `post_run_verification.json` | audits |
| `data_access_audit.csv`, `split_manifest.json`, `task_manifest.json`, `environment.json`, `experiment_config.yaml`, `task_config.yaml` | provenance |
| `checksums.json` | run manifest; this README is excluded |
| `checksums_supplementary.json` | files written after the run |

**Git-ignored artefacts.** They are in `data/interim/mlp_h5_experiment6b/`; their hashes are in
`environment.json`.
- `oof_predictions_mlp_h5.parquet`: primary and seed replicates.
- `fold_models_mlp_h5.npz` and `fold_models_mlp_h5.pt`: weights.
- `checkpoints/`: per-epoch and per-fit checkpoints.
- Logs, `run_status.json` and `COMPLETED`.
