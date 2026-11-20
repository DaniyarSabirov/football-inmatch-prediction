# Thesis documentation (since the supervisor meeting of 2026-11-16)

> **Canonical task since 2026-11-19:** `actions15_360current_v1`. See
> `docs/canonical_prediction_protocol.md`. The documents below describe experiment 4,
> which used the historical `scenarioD15` task with all-transition training (Protocol A);
> its numbers are historical reference points. The mathematical framework (02) applies
> unchanged to the canonical task.

| File | Content |
|---|---|
| `00_repository_and_taxonomy_audit.md` | repository analysis; the verified 15-class taxonomy (names, ids, order, sources); inconsistencies; leakage review |
| `01_data_preparation_history.md` | data source, match exclusions, coverage, alignment, orientation, team identity, cleaning, leakage audits, visualization, with sources and evidence labels |
| `02_mathematical_framework.md` | notation; Markov chain; softmax regression; cross-entropy; Markov ↔ logistic equivalence (Questions A–C); evaluation |
| `03_thesis_draft_methodology.md` | academic draft text for the methodology chapter; completed experiment 4 vs planned work |
| `04_feature_engineering_roadmap.md` | feature groups F1–F6, feature audit rules, model families (Question E), factorial design (Question F), next experiment |

**Experiment results:**
- **Experiment 4 (this phase).** `results/logistic_regression_f0_2026-11-17/README.md`.
- **Historical.**
  - Experiment 1: `results/markov_hmm_pilot/`, 8 classes;
  - Experiment 2: `results/markov_hmm_cost_sensitive_v2/`;
  - Experiment 3: `results/context_aware_event_prediction_v3/`.

  These are preserved unchanged.
