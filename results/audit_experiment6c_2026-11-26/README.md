# Final independent audit: experiment 6C (LSTM-H5)

Date: 2026-11-26. Script: `scripts/audit_experiment6c.py`. It runs as
`python scripts/audit_experiment6c.py` and took 116 s, under a process-bound
`caffeinate -d -i -w <audit pid>`. All 66 checks passed: 65 independently executed and 1 by
inspection (`audit_checks.csv`, column `evidence`).

## Verdict

**Experiment 6C (LSTM-H5): PASS WITH LIMITATIONS.**
- **No defects in results.** No critical or major finding, no data leakage, no final-test access.
- **Corrections made.** Overstated wording in the result README and the thesis chapter was
  corrected (F1). No prediction, metric or model was changed, and nothing was retrained.

## 1. Independence of the checks

**Own implementations.** The script recomputes every audited quantity with code written in the
script itself, from:
- raw StatsBomb JSON;
- the canonical table;
- the stored artefacts: predictions, weights in the `.npz`/`.pt` files, the 175 per-fit
  checkpoints (including each fit's stored best-epoch weights), fit logs, histories and manifests.

**Not used for verification:**
- `event_history`, `canonical_task`, `evaluation_v2`, `xgboost_experiment`;
- the 6C module's NumPy reference;
- the 6C verifier;
- the 6A/6B audit's raw reconstruction. The earlier 6C checks used it; this audit adds a third,
  separately written K1–K3 implementation.

**Production code imported only as the object under test:**
- the 6C model class: PyTorch hidden/cell states, autograd gradients, statelessness, dropout;
- `make_optimizer`: the AdamW step;
- `FitResult`: needed to unpickle the fit checkpoints.

| Evidence type | Checks |
|---|---:|
| Independently executed in this audit | 65 |
| Inspection of cited source lines | 1 |
| Existing tests re-run (full suite, §6) | 664 tests |

## 2. What was verified

| Area | Evidence |
|---|---|
| Provenance | Checked against `environment.json` and the run identifiers: all 36 recorded result checksums, the 3 large artefacts, the configuration hash and the 14 executed sources (byte-identical). Run `exp6c-20261125T195623-158a9af7`: experiment and audit exit 0, one attempt each, `COMPLETED` present. Workers: 5, each exit 0, computed only its own fold, 1 thread, deterministic, no warnings. Checkpoints: 175 completed fits (35 per fold directory), 5 fold records, no in-progress file. Weights are identical in fold records, `.pt` and `.npz` |
| Raw JSON | 60 complete Tier B matches (seed 20261126; not the 12 of the run) rebuilt with own code from events and three-sixty JSON: K1–K3, K4 inactive, period 5 dropped (67 records), `index` order, usable 360 = non-empty `freeze_frame`. Canonical observations (ids, order, class, period, index, usable 360, eligibility) are identical. All **138,830** eligible rows have identical 5-position histories and targets (class and target event id). Strata: 115 period starts, 480 rows with `<START>`, 108 period ends, 7,987 repeated timestamps, 10,824 rows with a companion in the window, 36,397 with a history event lacking 360, 3,419 companions removed, 1,618 Aerial Lost kept, all rare classes present. Per-match summary: `raw_reconstruction_summary_by_match.csv` |
| Training input | An own history construction (index arithmetic within sequence) on all 690,147 rows has the hash of the run's training input. Targets need no 360 snapshot |
| Future events | Adversarial mutation of every event at sequence position ≥ 200 leaves all 128,400 earlier rows' histories unchanged |
| Population | Tier B = development − pilot = 304 matches; 690,147 rows; the 83 final-test ids are disjoint from data and folds (ids only, no final-test content read). Folds partition the matches |
| Row identity | The stored predictions of experiments 5, 5B, 6A, 6B and 6C have event id, match, target and fold identical element-wise to the canonical rows. The 6C file has 15 primary + 30 seed columns in canonical order |
| Model reconstruction | Own float64 NumPy LSTM of all 15 stored models (E in folds 0 and 4, F in folds 1–3; 3 seeds each): probabilities of **all 690,147 rows × 3 seeds** reproduced, max \|Δp\| = 5.6e-7 (float32 network). Shapes: 2 layers, layer-2 input H, 4H gate blocks, 15 logits. Parameter counts: E 15,343, F 55,247 |
| Convention sensitivity | Exchanging gate blocks (i,g,f,o or f,i,g,o), dropping `bias_hh` or reversing the positions changes probabilities by 0.29–0.99, so the equality above is informative |
| Gate level (float64, all 15 models, 2,000 windows each) | Layer-1 and layer-2 \(h_s, c_s\) for s = 1..5 and the logits agree with PyTorch to ≤ 7.1e-15; the recovered output gate to 5.0e-13 |
| Gradients | Central differences (h = 1e-6) of the own NumPy loss vs autograd, for every parameter tensor of the trained E and F models (10 tensors each): max error 2.3e-10 |
| Loss | Own log-sum-exp NLL = `F.cross_entropy(logits)` = stored fold-0 log loss (1.052802…); a double softmax would give 2.28 |
| AdamW | Two steps of the production optimiser equal the closed form \(\theta(1-\eta\lambda) - \eta\hat m/(\sqrt{\hat v}+\epsilon)\) with bias correction (max dev 1.0e-8, float32); decay only on `weight_ih`, `weight_hh`, `out.weight` |
| Statelessness (trained F model) | Each window gives the same output alone, in a batch, after permutation or reversal of the batch, and when all other windows change. Repeated calls are identical |
| Dropout | `nn.LSTM.dropout = 0`, output dropout 0.1; train mode stochastic, eval mode deterministic |
| Nested CV, all 175 fits | Seeds equal SeedSequence(20261123, outer, inner+1 \| 0, candidate, replicate). Training/stopping row counts equal an own match-level stopping split. **Every fit's stored best-epoch weights, re-scored with the own LSTM, reproduce its recorded held-out log loss and best stopping loss (≤ 7.4e-9).** Best epoch = first argmin of the stopping curve (epoch 0 included), epochs = min(best + 5, 60); 0 fits chose epoch 0 and 13 hit the cap (all G/H) |
| Selection | Recomputed from own held-fold scores: E, F, F, F, E. Margins to the runner-up are 4e-5 to 6e-4 |
| Data sets | Own reconstruction: 75 distinct match sets, 525 uses, identical to the 6C **and** the 6B data-access audits |
| Training population | Refits update on 495,863–500,172 rows (89.8–90.5 %, 219–220 matches) and stop on 24 matches (52,552–56,594 rows), disjoint, never outer validation (`training_population.csv`) |
| Seeds | Primary = replicate 0 (prespecified), which is not the best seed. Seed-matched LSTM − MLP: −0.00120, −0.00094, −0.00124, all CIs below 0 (`seed_comparisons.csv`) |
| Metrics | Own NumPy, scikit-learn-checked, for all 10 models (≤ 2.2e-16): pooled, per fold, per class, confusion matrices, Brier. Pooled ≠ mean of folds. All 35 values of the reference table reproduced after rounding |
| Shot | Support 7,212. LSTM-H5: 28 correct of 75 Shot predictions, so recall 0.388 % and precision 37.3 %. XGB-H5 12, MLP-H5 13 correct |
| Bootstrap | Own seed-7 implementation (1,000 match resamples; ratio of resampled sums, so varying match sizes are handled). All 8 paired log-loss differences, relative differences, CIs and fold signs reproduce exactly (1e-12) |
| Decomposition | By current class, for MLP-H5, XGB-H5, MLR-H5 and Markov 2 references: equals the stored strata and sums to the totals (`decomposition_by_current_class.csv`) |
| History | Checksums of the baselines, 5, 5B, 6A, 6B and the 6A/6B audit are unchanged; their predictions, models and the canonical dataset are unchanged since before the run |

### Verified key numbers (LSTM-H5 − reference, log loss)

| Reference | Δ | rel. | 95 % CI | Folds LSTM better |
|---|---:|---:|---|---:|
| MLP-H5 | −0.001205 | −0.114 % | [−0.001531, −0.000896] | 5 / 5 |
| XGB-H5 | −0.000384 | −0.036 % | [−0.000690, −0.000088] | 4 / 5 |
| MLR-H5 | −0.051504 | −4.67 % | [−0.053004, −0.050110] | 5 / 5 |
| Markov 2 | −0.016608 | −1.56 % | [−0.017402, −0.015890] | 5 / 5 |
| XGB-H2 | −0.016140 | −1.51 % | [−0.016932, −0.015434] | 5 / 5 |
| Logistic F0 | −0.107039 | −9.24 % | [−0.109688, −0.104703] | 5 / 5 |

**Pressure decomposition** (85,965 rows, 12.46 %):

| Reference | Conditional mean on Pressure rows | Pressure contribution | Other 14 classes | Total |
|---|---:|---:|---:|---:|
| MLP-H5 | −0.00355 | −0.00044 | −0.00076 | −0.00120 |
| XGB-H5 | −0.00642 | −0.00080 | +0.00042 | −0.00038 |

This is descriptive, not causal.

## 3. Findings

No finding changes trained weights, predictions, metrics, data eligibility or experimental
validity, and none requires retraining. The executed sources were **not** modified, which keeps
their byte-identity with the run's provenance hashes.

| # | Severity | Location | Finding | Impact | Resolution |
|---|---|---|---|---|---|
| F1 | MINOR (interpretation) | Result README §1, §6, §8; `docs/thesis/09_lstm_h5.md` §1, §9, §16, §17, §19 | Several claims were overstated, listed below the table | Over-interpretation only | Corrected in both documents |
| F2 | MINOR (provenance comment) | `configs/lstm_h5_experiment6c_v1.yaml:61` | The comment "MPS 0.8-1.4x of CPU time" is inaccurate. The benchmark gives MPS/CPU time ratios of 0.72–1.56 (A 1.56, B 0.97, D 0.80, E 0.84, F 0.72) | None. CPU was chosen for verified determinism, and extra threads were slower | Not edited: the configuration hash is bound to every checkpoint. Corrected here. The benchmark, previously only in a session scratch directory, is archived in `execution_evidence/` |
| F3 | MINOR (documentation of recovery) | `scripts/supervise_lstm_h5_experiment6c.py:14-18, 157`; `scripts/run_lstm_h5_experiment6c.py:549-551` | The supervisor restarts the runner only if the *runner* is killed by a signal. If a *worker* is killed (e.g. out of memory), the runner exits 10 and no automatic restart happens, although the docstring mentions out-of-memory kills | None. Not triggered: all workers exited 0 | Manual relaunch resumes safely from the atomic checkpoints. Documented here |
| F4 | INFORMATIONAL | `scripts/run_lstm_h5_experiment6c.py:314` | Per-worker log file handles are opened and never explicitly closed | Negligible (≤ 5 descriptors) | Documented |
| F5 | INFORMATIONAL | `run_lstm_h5_experiment6c.py:899` | A failed integrity audit leaves `<output>.partial` without its marker; a rerun then refuses (exit 9) | Intended: preserves evidence; requires deliberate intervention | Documented |
| F6 | INFORMATIONAL | `09_lstm_h5.md` §5 | A cross-reference pointed to §13 instead of §15. The thesis uses the project notation \(E_t\), \(K\); the notes use \(e_t\), \(V\) | None | Reference corrected; notation mapping added |
| F7 | INFORMATIONAL | `inner_cv_scores.csv` | Selection margins are small (4e-5 to 6e-4); E vs F is close in every fold | Both are 2-layer LSTMs; validity unaffected | Documented |
| F8 | INFORMATIONAL | `fit_log.csv` | 13 inner fits (G: 11, H: 2; lr 3e-4) hit the 60-epoch cap | Their inner scores may be pessimistic; they were 0.0009–0.0029 above the selected configuration | Documented |
| F9 | INFORMATIONAL (test coverage) | `tests/test_lstm_h5_experiment6c.py` | AdamW was tested only with zero gradients (decay term) | Code correct (audit §2) | Regression test `test_adamw_update_with_nonzero_gradients_equals_closed_form` added |
| F10 | INFORMATIONAL | `pyproject.toml:46-50` | Ruff is configured (line length 88) but not installed. All experiment modules since experiment 5 use longer lines | None | pyflakes run on all 6C files: clean. Ruff not added (no new dependency) |
| F11 | INFORMATIONAL | `mlp_h5_experiment.py:181` (`load_checked`, reused by 6C) | Checkpoints are loaded with `torch.load(weights_only=False)` (pickle) | Safe for locally produced files only | Documented |

**The claims corrected under F1:**
- **"Controlled" comparison with MLP-H5.** It was described as controlled without qualification,
  or with the remaining difference attributed to the function class. In fact only the data
  partitions, the training protocol and the seed convention are shared. The architecture,
  parameter counts, candidate lists and selected configurations differ, and equal seed integers
  do not mean equal random draws.
- **"The unequal data favour XGB-H5".** This is unproven: only the difference in data
  availability is established, not its effect.
- **"Within 0.1 %".** The LSTM vs MLP difference is 0.114 %.
- **Pressure.** "More than fully / entirely" lacked counts and the offsetting contributions.
- **Shot.** The practical weakness of 0.388 % recall was not stated.

**Audit-tool defect (not a project finding).** The first audit run stopped in section 7. The
stored strata also contain a Markov 2 reference, which the script did not recompute. The script
was corrected (Markov 2 added, set equality checked) and re-run completely; this directory holds
only the complete re-run.

## 4. Limitations that remain

- **Training data vs XGB-H5.** LSTM-H5 (and MLP-H5) update on about 90 % of the outer-training
  rows; XGB-H5 used 100 %. The effect of this on the LSTM-H5 vs XGB-H5 difference is unknown
  without a controlled full-population refit, which is not authorised or run.
- **LSTM-H5 vs MLP-H5.** The two share partitions and protocol but not the architecture, search
  space or parameter count. The comparison is not an isolated test of recurrence.
- **Effect sizes.** About 0.11 % (vs MLP-H5) and 0.04 % (vs XGB-H5): statistically detectable,
  practically very small. The seed spread (LSTM 0.00025, MLP 0.0005) is of the order of the
  XGB-H5 gap.
- **CI scope.** The CIs cover match resampling of fixed out-of-fold predictions. They do not cover
  refitting, the overlap of cross-validation training sets, hyperparameter selection, seed
  variation or the competition mix. No equivalence claim is made.
- **Input.** Five events, event types only. No claim about long-term football dependencies.
- **Search.** A restricted search of 8 configurations, partly capped at 60 epochs (G, H).
- **Reproducibility.** Bitwise reproducibility was verified on this machine only (Apple M1, CPU,
  1 thread, Python 3.11.1, torch 2.15.0, NumPy 2.4.6). Bitwise re-training was shown by the run
  for one refit (fold 0). This audit re-scored stored weights and did not retrain.

## 5. Files

| File | Content |
|---|---|
| `audit_checks.csv` | All 66 checks with evidence type and values |
| `audit_summary.json` | Totals |
| `audit_provenance.json` | Script hash, versions, seeds, PIDs, first-attempt note, hash of the large reconstruction file (git-ignored) |
| `fit_rescoring.csv` | All 175 fits re-scored from their stored best-epoch weights |
| `training_population.csv` | Refit populations per fold and replicate |
| `metrics_recomputed.csv`, `paired_differences_recomputed.csv`, `seed_comparisons.csv`, `decomposition_by_current_class.csv` | Recomputed statistics |
| `raw_reconstruction_summary_by_match.csv` | Per-match strata of the 60-match rebuild; full rows in `data/interim/lstm_h5_experiment6c/final_audit/` |
| `execution_evidence/` | The pre-run device benchmark and smoke test (copied unchanged) |

## 6. Test suite

`python -m pytest` (2026-11-26, after the audit): **664 passed**, 0 failed, 0 skipped, no warnings reported. That is the 663 earlier tests plus the F9 regression test. No test was removed, weakened or skipped. pyflakes is clean on all 6C files and on the audit script.
