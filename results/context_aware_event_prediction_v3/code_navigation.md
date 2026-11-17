# Code navigation (experiment 3)

The full guide to the sequence-model code (experiments 1–2: data, Markov chains, HMMs, cost-sensitive decisions,
demo script) is in `docs/sequence_models_code_guide.md`. Experiment 3 adds:

| File | Main functions | Purpose |
|---|---|---|
| `src/sequence_models/v3_events.py` | `build_event_details` | raw subtypes, outcomes and `related_events` of development matches (cached) |
| `src/sequence_models/v3_taxonomy.py` | `companion_rules`, `taxonomy_table`, `audit`, `annotation_pairs`, `event_mapping` | the four taxonomies, the `actions15` K1–K3 rules, the taxonomy audit |
| `src/sequence_models/v3_features.py` | `spatial_features`, `causal_raw_features`, `possession_bins`, `history_tokens`, `orientation_audit`, `possession_audit` | causal geometry, zones, possession and time features, history tokens |
| `src/sequence_models/v3_models.py` | `MarkovSpec`, `nested_markov`, `fit_predict_markov`, `lr_design`, `cv_lr`, `Checkpoint`, `Progress` | context Markov from a feature priority list; logistic regression; resumable runs |
| `src/sequence_models/v3_experiment.py` | `build_feature_table`, `shot_cost_matrix`, `threshold_for_budget`, `evaluate` | feature tables, cost layer, evaluation with paired bootstrap |
| `src/sequence_models/v3_diagnostics.py` | `class_diagnostics`, `breakdown`, `per_match` | rare-event ranking-vs-decision diagnostics |
| `src/sequence_models/v3_report.py` | `stage_report`, `reproducibility` | final tables, figures, re-runs, checksums |
| `scripts/run_context_aware_v3.py` | stages `taxonomy` … `report` | orchestration; `freeze` writes the hash-locked plan |
| `scripts/verify_context_aware_v3.py` | | independent verification (no pipeline import) |
| `scripts/demo_explain_prediction.py` | | explains one saved prediction step by step (about 3 s) |

**Changing a model in experiment 3.**
- Add an entry under `candidates:` in `configs/context_aware_event_prediction_v3.yaml`, e.g. `[cur, g12, prev1, prev2]`.
- The backoff drops features from the end of the list.
- New context features must exist in the feature table and in `v3_features.CONTEXT_CARDINALITY`.
- Tier B runs only what `frozen_plan.json` lists.
