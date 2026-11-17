r"""Evaluation metrics for probabilistic multiclass next-event predictions.

All functions take integer labels ``y`` (n,) in 0..V-1 and a probability
matrix ``p`` (n, V). Predicted class = argmax (ties -> lowest index).

- Log loss: :math:`-\frac1n \sum_i \log p_{i, y_i}` (natural log), computed
  exactly; a zero probability on the true class yields ``inf`` and is never
  clipped away silently.
- Multiclass Brier score: :math:`\frac1n \sum_i \sum_c (p_{ic} - 1[y_i = c])^2`
  (range 0..2).
- Accuracy, macro-/weighted-F1, balanced accuracy, per-class precision /
  recall / F1 / support, confusion matrix: scikit-learn with ``labels``
  fixed to the full vocabulary and ``zero_division=0`` (a class never
  predicted has precision 0 and F1 0).

"Pooled" metrics are micro-aggregated over all events of all validation
folds; fold-level metrics are computed per validation fold. Uncertainty uses
a match-level (cluster) bootstrap: matches are resampled with replacement,
never individual events, because events within a match are dependent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


class ProbabilityError(ValueError):
    """Predicted probabilities are not a valid distribution."""


def validate_probabilities(p: np.ndarray, tolerance: float = 1e-9) -> None:
    if p.ndim != 2:
        raise ProbabilityError("expected an (n, V) array")
    if not np.all(np.isfinite(p)):
        raise ProbabilityError("non-finite probabilities")
    if np.any(p < 0):
        raise ProbabilityError("negative probabilities")
    if np.any(np.abs(p.sum(axis=1) - 1) > tolerance):
        raise ProbabilityError(f"rows do not sum to one (max deviation {np.abs(p.sum(axis=1) - 1).max():.3g})")


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    true_p = p[np.arange(len(y)), y]
    with np.errstate(divide="ignore"):
        return float(-np.mean(np.log(true_p)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def predicted_class(p: np.ndarray) -> np.ndarray:
    return p.argmax(axis=1)


def summary_metrics(y: np.ndarray, p: np.ndarray, n_classes: int) -> dict[str, float]:
    labels = np.arange(n_classes)
    pred = predicted_class(p)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": balanced_accuracy(y, pred, n_classes),
        "log_loss": log_loss(y, p),
        "brier": brier_score(y, p),
        "n_classes_present": int(len(np.unique(y))),
        "n_classes_predicted": int(len(np.unique(pred))),
    }


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Mean recall over the classes PRESENT in ``y`` (scikit-learn's definition)."""
    return float(balanced_accuracy_score(y, pred))


def per_class_metrics(y: np.ndarray, p: np.ndarray, vocabulary: tuple[str, ...]) -> pd.DataFrame:
    labels = np.arange(len(vocabulary))
    pred = predicted_class(p)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return pd.DataFrame({
        "class": list(vocabulary), "precision": precision, "recall": recall, "f1": f1,
        "support": support, "n_predicted": np.bincount(pred, minlength=len(vocabulary)),
        "mean_predicted_probability": p.mean(axis=0),
    })


def confusion(y: np.ndarray, p: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    return confusion_matrix(y, predicted_class(p), labels=np.arange(n_classes))


def match_bootstrap(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_classes: int, n_replicates: int, seed: int,
    p_reference: np.ndarray | None = None,
) -> pd.DataFrame:
    """Match-level bootstrap of summary metrics.

    With ``p_reference`` (another model's predictions on the same rows) the
    replicates are of the paired difference metric(p) - metric(p_reference).
    Returns one row per metric: estimate, 2.5% and 97.5% percentiles.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    keys = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "log_loss", "brier")

    def stat(idx: np.ndarray) -> np.ndarray:
        m = summary_metrics(y[idx], p[idx], n_classes)
        value = np.array([m[k] for k in keys])
        if p_reference is not None:
            r = summary_metrics(y[idx], p_reference[idx], n_classes)
            value = value - np.array([r[k] for k in keys])
        return value

    estimate = stat(np.arange(len(y)))
    replicates = np.empty((n_replicates, len(keys)))
    for b in range(n_replicates):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        replicates[b] = stat(np.concatenate([rows_by_group[g] for g in chosen]))
    return pd.DataFrame({
        "metric": keys, "estimate": estimate,
        "ci_low": np.percentile(replicates, 2.5, axis=0), "ci_high": np.percentile(replicates, 97.5, axis=0),
    })
