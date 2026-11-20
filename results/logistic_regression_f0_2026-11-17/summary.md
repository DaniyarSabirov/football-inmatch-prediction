# Experiment 4 — generated summary (do not edit; see README.md)

## Pooled out-of-fold metrics (argmax decisions; 95% match-bootstrap CI for log loss)

| model | role | log_loss | log_loss_ci_low | log_loss_ci_high | brier | accuracy | macro_f1 | weighted_f1 | balanced_accuracy | shot_average_precision | n_classes_predicted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frequency_baseline | primary | 1.6592 | 1.6473 | 1.6729 | 0.7237 | 0.3911 | 0.0375 | 0.2199 | 0.0667 | 0.0101 | 1 |
| markov_order1 | primary | 1.1520 | 1.1386 | 1.1665 | 0.5190 | 0.6529 | 0.2145 | 0.5588 | 0.2182 | 0.0257 | 5 |
| logistic_F0 | primary | 1.1519 | 1.1385 | 1.1664 | 0.5190 | 0.6529 | 0.2145 | 0.5588 | 0.2182 | 0.0257 | 5 |
| logistic_F0_unpenalised | supplementary | 1.1521 | 1.1387 | 1.1665 | 0.5190 | 0.6529 | 0.2145 | 0.5588 | 0.2182 | 0.0257 | 5 |
| markov_order1_nested_alpha | supplementary | 1.1520 | 1.1386 | 1.1664 | 0.5190 | 0.6529 | 0.2145 | 0.5588 | 0.2182 | 0.0257 | 5 |
| markov_order2 | supplementary | 1.0497 | 1.0376 | 1.0626 | 0.4866 | 0.6689 | 0.2498 | 0.5809 | 0.2427 | 0.0525 | 8 |

## Shot

| model | support | n_predicted | precision | recall | f1 | average_precision_one_vs_rest |
|---|---|---|---|---|---|---|
| frequency_baseline | 7199 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0101 |
| markov_order1 | 7199 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0257 |
| logistic_F0 | 7199 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0257 |
| logistic_F0_unpenalised | 7199 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0257 |
| markov_order1_nested_alpha | 7199 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0257 |
| markov_order2 | 7199 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0525 |

## Hyperparameter choices (nested inner CV)

| fold_id | model | chosen |
|---|---|---|
| 0 | markov_order1_nested_alpha | 0.1000 |
| 0 | logistic_F0 | 10.0000 |
| 1 | markov_order1_nested_alpha | 0.1000 |
| 1 | logistic_F0 | 10.0000 |
| 2 | markov_order1_nested_alpha | 0.1000 |
| 2 | logistic_F0 | 10.0000 |
| 3 | markov_order1_nested_alpha | 0.1000 |
| 3 | logistic_F0 | 10.0000 |
| 4 | markov_order1_nested_alpha | 0.1000 |
| 4 | logistic_F0 | 10.0000 |

## Logistic fits

| fold_id | model | C | n_iter | converged | fit_seconds | n_train_rows |
|---|---|---|---|---|---|---|
| 0 | logistic_F0 | 10.0000 | 3592 | True | 547.3517 | 6.4e+05 |
| 0 | logistic_F0_unpenalised | inf | 5000 | False | 0.4600 | 6.4e+05 |
| 1 | logistic_F0 | 10.0000 | 3566 | True | 506.2781 | 6.402e+05 |
| 1 | logistic_F0_unpenalised | inf | 5000 | False | 0.4644 | 6.402e+05 |
| 2 | logistic_F0 | 10.0000 | 3581 | True | 503.6945 | 6.384e+05 |
| 2 | logistic_F0_unpenalised | inf | 5000 | False | 0.4656 | 6.384e+05 |
| 3 | logistic_F0 | 10.0000 | 3597 | True | 507.5987 | 6.393e+05 |
| 3 | logistic_F0_unpenalised | inf | 5000 | False | 0.4665 | 6.393e+05 |
| 4 | logistic_F0 | 10.0000 | 3324 | True | 473.1633 | 6.393e+05 |
| 4 | logistic_F0_unpenalised | inf | 5000 | False | 0.4612 | 6.393e+05 |

## Mean timings per fold (seconds)

| model | fit_seconds | predict_seconds |
|---|---|---|
| frequency_baseline | 0.0102 | 0.0126 |
| logistic_F0 | 507.6173 | 0.0363 |
| logistic_F0_unpenalised | 0.4635 | 0.0317 |
| markov_order1 | 0.0102 | 0.0111 |
| markov_order2 | 0.0179 | 0.0245 |

## Reproduction of experiment 2 (same rows and folds)

```json
{
 "n_eval_rows": {
  "expected": 701144,
  "got": 701144
 },
 "frequency_baseline": {
  "expected_log_loss": 1.6592461673936363,
  "got": 1.6592461673936363,
  "abs_difference": 0.0
 },
 "markov_order1": {
  "expected_log_loss": 1.151972941501348,
  "got": 1.151972941501348,
  "abs_difference": 0.0
 },
 "markov_order2": {
  "expected_log_loss": 1.049693324374873,
  "got": 1.049693324374873,
  "abs_difference": 0.0
 },
 "all_match": true
}
```
