"""Sequence models (empirical frequency, Markov chains, hidden Markov models)
for StatsBomb next-event prediction.

Module layout (data preparation, training, inference and evaluation are kept
separate):

- ``raw_events``  : read the raw StatsBomb event stream into one flat table
- ``data_audit``  : readiness checks on the raw stream and the production dataset
- ``tasks``       : formal prediction-task definitions (sequences, targets, rows)
- ``spatial``     : causal spatial observations for the spatial HMM
- ``splits``      : match-grouped final-test hold-out, pilot subset, CV folds
- ``markov``      : empirical-frequency baseline, first- and second-order Markov chains
- ``hmm``         : hidden Markov model (categorical / categorical x Gaussian
                    emissions), EM training and causal filtering
- ``metrics``     : probabilistic and classification metrics, grouped bootstrap
- ``experiment``  : cross-validated pilot runner and artefact writer
- ``figures``     : static result figures
- ``config``      : validated experiment configuration

Nothing here modifies the production dataset builder, the event taxonomy or
the feature allowlist; see results/markov_hmm_pilot/README.md.
"""
