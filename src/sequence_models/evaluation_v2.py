r"""Evaluation for experiment 2: decisions and probabilities kept separate.

- Decision metrics (from a decision vector ``d``, e.g. argmax or a Bayes
  cost-sensitive rule): accuracy, macro-/weighted-F1, balanced accuracy,
  per-class precision / recall / F1 / support / predicted count. They are
  computed from the confusion matrix (rows = true, columns = predicted), with
  0 for undefined precision/F1, as in scikit-learn ``zero_division=0``. Balanced
  accuracy averages recall over classes present in ``y``.
- Probability metrics (from ``p`` only, never from ``d``): multiclass log loss
  (natural log), multiclass Brier score, calibration diagnostics.
- Calibration (no recalibration is fitted, so there is nothing to leak):
  classwise ECE (Kull et al. 2019): for each class k, 10 equal-width bins of
  p_k, :math:`\sum_b \frac{n_b}{n}|\bar p_{k,b} - \bar y_{k,b}|`, averaged over
  classes; top-label ECE (Guo et al. 2017) on the argmax probability;
  calibration-in-the-large per class: :math:`\bar p_k - \bar y_k`.
- Uncertainty: match-level (cluster) bootstrap from per-match sufficient
  statistics (confusion matrices and log-loss / Brier sums). Resampling
  matches with replacement gives the same replicates as recomputing on
  concatenated rows (tested), but cheaply. Paired differences use the same
  resampled matches for both models.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

N_BINS = 10


def confusion_counts(y: np.ndarray, d: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(y * n + d, minlength=n * n).reshape(n, n).astype(float)


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    total = cm.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    present = support > 0
    return {
        "accuracy": float(tp.sum() / total),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / total),
        "balanced_accuracy": float(recall[present].mean()),
    }


def per_class_from_confusion(cm: np.ndarray, vocabulary: tuple[str, ...] | list[str]) -> pd.DataFrame:
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    prevalence = support / cm.sum()
    return pd.DataFrame({
        "class": list(vocabulary), "support": support.astype(int), "n_predicted": predicted.astype(int),
        "true_positives": tp.astype(int), "false_positives": (predicted - tp).astype(int),
        "false_negatives": (support - tp).astype(int), "precision": precision, "recall": recall, "f1": f1,
        "prevalence": prevalence, "lift": np.divide(precision, prevalence, out=np.zeros_like(precision), where=prevalence > 0),
        "predicted_share": predicted / cm.sum(),
    })


def log_loss_sum(y: np.ndarray, p: np.ndarray) -> float:
    with np.errstate(divide="ignore"):
        return float(-np.log(p[np.arange(len(y)), y]).sum())


def brier_sum(y: np.ndarray, p: np.ndarray) -> float:
    sq = (p ** 2).sum(axis=1) - 2 * p[np.arange(len(y)), y] + 1.0
    return float(sq.sum())


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"log_loss": log_loss_sum(y, p) / len(y), "brier": brier_sum(y, p) / len(y)}


def calibration(y: np.ndarray, p: np.ndarray, vocabulary) -> tuple[dict[str, float], pd.DataFrame]:
    """(summary, reliability table). Equal-width bins on [0, 1]."""
    n, v = p.shape
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    edges = np.linspace(0, 1, N_BINS + 1)
    rows, class_ece = [], []
    for k in range(v):
        bins = np.clip(np.digitize(p[:, k], edges[1:-1], right=False), 0, N_BINS - 1)
        count = np.bincount(bins, minlength=N_BINS)
        mean_p = np.bincount(bins, weights=p[:, k], minlength=N_BINS)
        freq = np.bincount(bins, weights=onehot[:, k], minlength=N_BINS)
        has = count > 0
        mp, fr = np.where(has, mean_p / np.maximum(count, 1), np.nan), np.where(has, freq / np.maximum(count, 1), np.nan)
        class_ece.append(float(np.nansum(count / n * np.abs(mp - fr))))
        for b in range(N_BINS):
            rows.append({"class": vocabulary[k], "bin": b, "bin_low": edges[b], "bin_high": edges[b + 1],
                         "n": int(count[b]), "mean_predicted": mp[b], "observed_frequency": fr[b]})
    top = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    tb = np.clip(np.digitize(top, edges[1:-1]), 0, N_BINS - 1)
    tc = np.bincount(tb, minlength=N_BINS)
    tp_ = np.bincount(tb, weights=top, minlength=N_BINS)
    ta = np.bincount(tb, weights=correct, minlength=N_BINS)
    top_ece = float(np.sum(np.abs(tp_ - ta)) / n)
    summary = {"classwise_ece": float(np.mean(class_ece)), "top_label_ece": top_ece,
               "max_abs_calibration_in_the_large": float(np.abs(p.mean(axis=0) - onehot.mean(axis=0)).max())}
    for k in range(v):
        summary[f"citl_{vocabulary[k]}"] = float(p[:, k].mean() - onehot[:, k].mean())
        summary[f"ece_{vocabulary[k]}"] = class_ece[k]
    return summary, pd.DataFrame(rows)


# ------------------------------------------------------------------ match bootstrap
def per_match_stats(y: np.ndarray, p: np.ndarray, decisions: dict[str, np.ndarray], groups: np.ndarray,
                    n_classes: int) -> dict:
    """Sufficient statistics per match: confusion matrix per decision rule, LL and Brier sums."""
    matches = np.unique(groups)
    index = np.searchsorted(matches, groups)
    ll = np.bincount(index, weights=-np.log(p[np.arange(len(y)), y]), minlength=len(matches))
    sq = (p ** 2).sum(axis=1) - 2 * p[np.arange(len(y)), y] + 1.0
    br = np.bincount(index, weights=sq, minlength=len(matches))
    count = np.bincount(index, minlength=len(matches)).astype(float)
    cms = {}
    for rule, d in decisions.items():
        flat = (index * n_classes + y) * n_classes + d
        cms[rule] = np.bincount(flat, minlength=len(matches) * n_classes * n_classes).reshape(
            len(matches), n_classes, n_classes).astype(float)
    return {"matches": matches, "ll": ll, "brier": br, "n": count, "cm": cms}


def bootstrap_multiplicities(n_matches: int, n_replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_matches, n_matches), minlength=n_matches)
                     for _ in range(n_replicates)]).astype(float)


def bootstrap_summary(stats: dict, rule: str, mult: np.ndarray) -> np.ndarray:
    """Replicates x [accuracy, macro_f1, weighted_f1, balanced_accuracy, log_loss, brier]."""
    n = mult @ stats["n"]
    ll = (mult @ stats["ll"]) / n
    br = (mult @ stats["brier"]) / n
    cm = np.tensordot(mult, stats["cm"][rule], axes=(1, 0))
    out = np.empty((len(mult), 6))
    for b in range(len(mult)):
        m = metrics_from_confusion(cm[b])
        out[b] = (m["accuracy"], m["macro_f1"], m["weighted_f1"], m["balanced_accuracy"], ll[b], br[b])
    return out


def bootstrap_recall(stats: dict, rule: str, mult: np.ndarray) -> np.ndarray:
    """Replicates x classes: recall (NaN if the class is absent in a replicate)."""
    cm = np.tensordot(mult, stats["cm"][rule], axes=(1, 0))
    support = cm.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.diagonal(cm, axis1=1, axis2=2) / support


def bootstrap_precision(stats: dict, rule: str, mult: np.ndarray) -> np.ndarray:
    cm = np.tensordot(mult, stats["cm"][rule], axes=(1, 0))
    predicted = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(predicted > 0, np.diagonal(cm, axis1=1, axis2=2) / np.maximum(predicted, 1), np.nan)


SUMMARY_KEYS = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "log_loss", "brier")


def interval(replicates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2.5% / 97.5% percentiles ignoring undefined replicates (e.g. precision of a class
    never predicted); NaN where every replicate is undefined."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(replicates, 2.5, axis=0), np.nanpercentile(replicates, 97.5, axis=0)
