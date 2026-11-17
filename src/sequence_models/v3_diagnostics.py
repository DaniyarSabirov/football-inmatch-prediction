"""Experiment 3: why are rare events (especially Shot) missed? Diagnostics on stored predictions.

Inputs: out-of-fold probabilities (experiment 2's ``predictions.parquet`` or experiment 3's own),
joined by event_id to an experiment-3 feature table (zone, distance, previous event, causal
possession duration, minute). Everything is descriptive; nothing is fitted.

Two separate questions per class k:
- ranking:  does p_k put true k events above the others? -> average precision (AP) vs prevalence,
  ROC AUC, PR curve;
- decision: does the multiclass rule ever choose k? -> top-1 / cost-sensitive recall and precision,
  and recall at one-vs-rest thresholds on p_k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def class_diagnostics(y: np.ndarray, p: np.ndarray, decisions: dict[str, np.ndarray], vocab: list[str],
                      classes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per-class summary, confusion rows 'predicted instead', threshold table)."""
    n = len(y)
    summary, instead, thr_rows = [], [], []
    for c in classes:
        k = vocab.index(c)
        t = y == k
        row = {"class": c, "support": int(t.sum()), "prevalence": float(t.mean()),
               "average_precision": float(average_precision_score(t, p[:, k])) if t.any() else np.nan,
               "roc_auc": float(roc_auc_score(t, p[:, k])) if 0 < t.sum() < n else np.nan,
               "p_true_median_tp_or_fn": float(np.median(p[t, k])) if t.any() else np.nan,
               "p_k_median_when_not_k": float(np.median(p[~t, k])),
               "p_k_p90_when_not_k": float(np.quantile(p[~t, k], 0.9)),
               "p_k_p90_when_k": float(np.quantile(p[t, k], 0.9)) if t.any() else np.nan,
               "max_p_k": float(p[:, k].max()), "share_rows_k_is_argmax": float((p.argmax(axis=1) == k).mean())}
        row["ap_lift_over_prevalence"] = row["average_precision"] / row["prevalence"] if row["prevalence"] else np.nan
        for rule, d in decisions.items():
            pred = d == k
            tp, fp = int((pred & t).sum()), int((pred & ~t).sum())
            row[f"{rule}_recall"] = tp / max(t.sum(), 1)
            row[f"{rule}_precision"] = tp / pred.sum() if pred.any() else 0.0
            r, pr = row[f"{rule}_recall"], row[f"{rule}_precision"]
            row[f"{rule}_f1"] = 2 * r * pr / (r + pr) if r + pr > 0 else 0.0
            row[f"{rule}_fp_per_1000"] = 1000 * fp / n
            # p_k distributions for TP / FN / FP under this rule
            for label, mask in (("tp", pred & t), ("fn", ~pred & t), ("fp", pred & ~t)):
                row[f"{rule}_median_p_{label}"] = float(np.median(p[mask, k])) if mask.any() else np.nan
            if rule == "uniform":
                pi = pd.Series(np.array(vocab)[d[t]]).value_counts(normalize=True)
                for other, share in pi.items():
                    instead.append({"class": c, "predicted_as": other, "share_of_true": float(share)})
        for tau in THRESHOLDS:
            pred = p[:, k] >= tau
            tp, fp = int((pred & t).sum()), int((pred & ~t).sum())
            thr_rows.append({"class": c, "threshold": tau, "recall": tp / max(t.sum(), 1),
                             "precision": tp / pred.sum() if pred.any() else np.nan, "fp_per_1000": 1000 * fp / n,
                             "flagged_per_1000": 1000 * pred.sum() / n})
        summary.append(row)
    return pd.DataFrame(summary), pd.DataFrame(instead), pd.DataFrame(thr_rows)


def breakdown(y: np.ndarray, p: np.ndarray, d: np.ndarray, frame: pd.DataFrame, k: int, by: str,
              min_rows: int = 200) -> pd.DataFrame:
    """Class k by one context column: rows, prevalence, AP within the stratum, top-1 recall."""
    rows = []
    for value, idx in frame.groupby(by, sort=True).indices.items():
        if len(idx) < min_rows:
            continue
        t = y[idx] == k
        rows.append({"by": by, "value": value, "rows": len(idx), "k_events": int(t.sum()), "prevalence": float(t.mean()),
                     "mean_p_k": float(p[idx, k].mean()),
                     "average_precision": float(average_precision_score(t, p[idx, k])) if 0 < t.sum() < len(idx) else np.nan,
                     "decision_recall": float((d[idx][t] == k).mean()) if t.any() else np.nan,
                     "decision_fp_per_1000": 1000 * float(((d[idx] == k) & ~t).sum()) / len(idx)})
    return pd.DataFrame(rows)


def per_match(y: np.ndarray, p: np.ndarray, d: np.ndarray, groups: np.ndarray, k: int, min_events: int = 3) -> pd.DataFrame:
    rows = []
    for g in np.unique(groups):
        idx = groups == g
        t = y[idx] == k
        if t.sum() < min_events:
            continue
        rows.append({"match_id": int(g), "k_events": int(t.sum()), "recall": float((d[idx][t] == k).mean()),
                     "average_precision": float(average_precision_score(t, p[idx, k])) if t.sum() < idx.sum() else np.nan,
                     "prevalence": float(t.mean())})
    return pd.DataFrame(rows)
