r"""Canonical baselines on ``actions15_360current_v1`` (frequency, Markov 1/2, logistic F0).

Rows. Training and evaluation rows are the canonical ``row_eligible`` rows (Protocol B):
every model of every fold is fitted on the eligible rows of the fold's training matches and
scored on the eligible rows of its validation matches. Nothing else is counted.

Models (all predict :math:`P(E_{t+1} = c \mid \cdot)` over K = 15 classes):

- ``frequency``: :math:`(n_c + \alpha) / (N + K\alpha)`, :math:`\alpha = 1` (no tuning needed).
- ``markov1``: :math:`(C_{a,c} + \alpha) / (N_a + K\alpha)`; a current class never seen in
  training falls back to ``frequency``. :math:`\alpha` chosen by nested CV.
- ``markov2``: context :math:`(E_{t-1}, E_t)` where :math:`E_{t-1}` is the previous
  observation of the canonical sequence (any 360 status; it is earlier, hence available).
  Contexts with fewer than ``min_context_count`` training rows, and the first observation of
  a sequence, back off to ``markov1`` with the same :math:`\alpha`. :math:`\alpha` nested.
- ``logistic_F0``: multinomial logistic regression on :math:`\operatorname{OneHot}(E_t)`
  (scikit-learn lbfgs, L2, intercept unpenalised), fitted on the count-weighted
  (current, next) cells. This is exactly the row-level objective (the data term and
  scikit-learn's penalty scale ``1/(C * sum(weights))`` are identical); see
  ``docs/thesis/02_mathematical_framework.md`` section 6. C chosen by nested CV.

Nested CV: for each outer fold, leave-one-inner-fold-out over the other 4 folds (pooled log
loss on their eligible rows); the outer validation fold never influences a choice. Ties go to
the earlier grid value.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .canonical_task import CLASSES, K, NO_TARGET
from .evaluation_v2 import (
    SUMMARY_KEYS,
    bootstrap_multiplicities,
    bootstrap_summary,
    calibration,
    confusion_counts,
    interval,
    metrics_from_confusion,
    per_class_from_confusion,
    per_match_stats,
    probability_metrics,
)
from .logistic import MultinomialLogisticModel
from .splits import FoldAssignment, LeakageError

MODELS = ("frequency", "markov1", "markov2", "logistic_F0")
SHOT = CLASSES.index("Shot")


@dataclass
class Rows:
    """Eligible rows as arrays (one entry per canonical row)."""

    match_id: np.ndarray
    cur: np.ndarray
    prev: np.ndarray     # previous observation of the canonical sequence, -1 at sequence start
    y: np.ndarray

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> "Rows":
        prev = table.groupby("sequence_id", sort=False)["symbol_id"].shift(1).fillna(-1).astype(np.int64)
        e = table["row_eligible"].to_numpy()
        if (table.loc[e, "target_id"].to_numpy() == NO_TARGET).any():
            raise ValueError("eligible row without target")
        return cls(table["match_id"].to_numpy()[e], table["symbol_id"].to_numpy()[e].astype(np.int64),
                   prev.to_numpy()[e], table["target_id"].to_numpy()[e].astype(np.int64))

    def mask(self, match_ids) -> np.ndarray:
        return np.isin(self.match_id, list(match_ids))


# ---------------------------------------------------------------------------- count models
def counts1(r: Rows, m: np.ndarray) -> np.ndarray:
    c = np.zeros((K, K))
    np.add.at(c, (r.cur[m], r.y[m]), 1)
    return c


def counts2(r: Rows, m: np.ndarray) -> np.ndarray:
    c = np.zeros((K, K, K))
    ok = m & (r.prev >= 0)
    np.add.at(c, (r.prev[ok], r.cur[ok], r.y[ok]), 1)
    return c


def frequency_table(c1: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    n = c1.sum(axis=0)
    return (n + alpha) / (n.sum() + K * alpha)


def markov1_table(c1: np.ndarray, alpha: float) -> np.ndarray:
    tab = (c1 + alpha) / (c1.sum(axis=1, keepdims=True) + K * alpha)
    tab[c1.sum(axis=1) == 0] = frequency_table(c1, alpha)
    return tab


def markov2_predict(c1, c2, alpha, min_count, cur, prev) -> np.ndarray:
    p = markov1_table(c1, alpha)[cur]
    ctx = c2.sum(axis=2)
    use = (prev >= 0) & (ctx[np.maximum(prev, 0), cur] >= min_count)
    p[use] = (c2[prev[use], cur[use]] + alpha) / (ctx[prev[use], cur[use]][:, None] + K * alpha)
    return p


def logistic_table(c1: np.ndarray, c_value: float, lr: dict) -> tuple[np.ndarray, dict]:
    a, c = np.nonzero(c1)
    model = MultinomialLogisticModel(K, c_value, int(lr["max_iter"]), float(lr["tol"]), float(lr["absent_class_floor"]))
    model.fit(np.eye(K)[a], c, sample_weight=c1[a, c])
    info = {k: model.fit_info[k] for k in ("n_iter", "converged", "fit_seconds", "absent_classes")}
    info["n_train_rows"] = int(c1.sum())
    return model.predict_proba(np.eye(K)), info


def log_loss_sum(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(p[np.arange(len(y)), y]).sum())


# ---------------------------------------------------------------------------- CV
@dataclass
class CVOutput:
    proba: dict[str, np.ndarray]
    fold_of_row: np.ndarray
    choices: list[dict] = field(default_factory=list)
    fits: list[dict] = field(default_factory=list)
    timings: list[dict] = field(default_factory=list)
    counts: list[dict] = field(default_factory=list)


def predict(model: str, r: Rows, tr: np.ndarray, rows: np.ndarray, value, cfg: dict):
    """Fit ``model`` with hyperparameter ``value`` on rows ``tr``; predict rows ``rows``."""
    c1 = counts1(r, tr)
    if model == "frequency":
        return np.tile(frequency_table(c1, 1.0), (rows.sum(), 1)), {}
    if model == "markov1":
        return markov1_table(c1, value)[r.cur[rows]], {}
    if model == "markov2":
        return markov2_predict(c1, counts2(r, tr), value, int(cfg["markov2"]["min_context_count"]),
                               r.cur[rows], r.prev[rows]), {}
    if model == "logistic_F0":
        tab, info = logistic_table(c1, value, cfg["logistic"])
        return tab[r.cur[rows]], info
    raise ValueError(model)


def grid_of(model: str, cfg: dict) -> list:
    return {"frequency": [None], "markov1": list(cfg["markov1"]["alpha_grid"]),
            "markov2": list(cfg["markov2"]["alpha_grid"]), "logistic_F0": list(cfg["logistic"]["C_grid"])}[model]


def run_cv(r: Rows, folds: list[FoldAssignment], cfg: dict) -> CVOutput:
    n = len(r.y)
    fold_of_row = np.full(n, -1)
    for f in folds:
        fold_of_row[r.mask(f.validation_match_ids)] = f.fold_id
    if (fold_of_row < 0).any():
        raise LeakageError("an eligible row belongs to no validation fold")
    out = CVOutput({m: np.full((n, K), np.nan) for m in MODELS}, fold_of_row)
    for f in folds:
        if set(f.train_match_ids) & set(f.validation_match_ids):
            raise LeakageError(f"fold {f.fold_id}: match in training and validation")
        tr, va = r.mask(f.train_match_ids), r.mask(f.validation_match_ids)
        inner = [g for g in folds if g.fold_id != f.fold_id]
        out.counts.append({"fold_id": f.fold_id, "train_matches": len(f.train_match_ids),
                           "validation_matches": len(f.validation_match_ids), "train_rows": int(tr.sum()),
                           "validation_rows": int(va.sum()),
                           **{f"train_n_{c}": int((r.y[tr] == i).sum()) for i, c in enumerate(CLASSES)},
                           **{f"validation_n_{c}": int((r.y[va] == i).sum()) for i, c in enumerate(CLASSES)}})
        for model in MODELS:
            grid = grid_of(model, cfg)
            start = time.perf_counter()
            if len(grid) > 1:
                scores = {}
                for v in grid:
                    ll, cnt = 0.0, 0
                    for held in inner:
                        itr = r.mask([m for g in inner if g is not held for m in g.validation_match_ids])
                        iva = r.mask(held.validation_match_ids)
                        p, _ = predict(model, r, itr, iva, v, cfg)
                        ll += log_loss_sum(p, r.y[iva])
                        cnt += int(iva.sum())
                    scores[v] = ll / cnt
                best = min(grid, key=lambda v: (round(scores[v], 12), grid.index(v)))
                out.choices.append({"fold_id": f.fold_id, "model": model, "chosen": best,
                                    "at_grid_edge": best in (grid[0], grid[-1]),
                                    **{f"inner_log_loss_{v:g}": s for v, s in scores.items()}})
            else:
                best = grid[0]
            search = time.perf_counter() - start
            start = time.perf_counter()
            p, info = predict(model, r, tr, va, best, cfg)
            fit_predict = time.perf_counter() - start
            out.proba[model][va] = p
            out.timings.append({"fold_id": f.fold_id, "model": model, "selection_seconds": search,
                                "final_fit_predict_seconds": fit_predict})
            if info:
                out.fits.append({"fold_id": f.fold_id, "model": model, "C": best, **info})
    for m, p in out.proba.items():
        if np.isnan(p).any():
            raise RuntimeError(f"{m}: rows without prediction")
    return out


# ---------------------------------------------------------------------------- evaluation
def evaluate(r: Rows, cv: CVOutput, cfg: dict) -> dict[str, pd.DataFrame]:
    y, groups = r.y, r.match_id
    ecfg = cfg["evaluation"]
    mult = bootstrap_multiplicities(len(np.unique(groups)), int(ecfg["bootstrap_replicates"]), int(ecfg["bootstrap_seed"]))
    summary, by_fold, per_class, confusion, stats = [], [], [], [], {}
    for name, p in cv.proba.items():
        if (p < 0).any() or not np.isfinite(p).all() or np.abs(p.sum(1) - 1).max() > float(ecfg["probability_tolerance"]):
            raise ValueError(f"{name}: invalid probabilities")
        d = p.argmax(axis=1)
        cm = confusion_counts(y, d, K)
        pc = per_class_from_confusion(cm, CLASSES)
        ap = [float(average_precision_score(y == c, p[:, c])) if (y == c).any() else np.nan for c in range(K)]
        cal, _ = calibration(y, p, CLASSES)
        row = {"model": name, "n_rows": len(y), **probability_metrics(y, p), **metrics_from_confusion(cm),
               "shot_average_precision": ap[SHOT], "shot_recall": float(pc.at[SHOT, "recall"]),
               "shot_precision": float(pc.at[SHOT, "precision"]), "n_classes_predicted": int((cm.sum(0) > 0).sum()),
               "classwise_ece": cal["classwise_ece"], "top_label_ece": cal["top_label_ece"],
               "max_abs_calibration_in_the_large": cal["max_abs_calibration_in_the_large"]}
        s = per_match_stats(y, p, {"argmax": d}, groups, K)
        stats[name] = s
        lo, hi = interval(bootstrap_summary(s, "argmax", mult))
        for i, key in enumerate(SUMMARY_KEYS):
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = float(lo[i]), float(hi[i])
        summary.append(row)
        pc.insert(0, "class_id", np.arange(K))
        pc.insert(0, "model", name)
        pc["average_precision"] = ap
        pc["mean_predicted_probability"] = p.mean(axis=0)
        per_class.append(pc)
        for f in sorted(set(cv.fold_of_row)):
            sel = cv.fold_of_row == f
            fcm = confusion_counts(y[sel], d[sel], K)
            by_fold.append({"model": name, "fold_id": int(f), "n_rows": int(sel.sum()), **probability_metrics(y[sel], p[sel]),
                            **metrics_from_confusion(fcm),
                            "shot_average_precision": float(average_precision_score(y[sel] == SHOT, p[sel, SHOT]))
                            if (y[sel] == SHOT).any() else np.nan,
                            "shot_recall": float(per_class_from_confusion(fcm, CLASSES).at[SHOT, "recall"])})
        cmf = pd.DataFrame(cm.astype(int), index=list(CLASSES), columns=list(CLASSES)).rename_axis("true_class")
        cmf = cmf.reset_index().melt(id_vars="true_class", var_name="predicted_class", value_name="count")
        cmf.insert(0, "model", name)
        confusion.append(cmf)
    paired = []
    for model, ref in (("markov1", "frequency"), ("markov2", "markov1"), ("logistic_F0", "markov1"),
                       ("logistic_F0", "frequency")):
        diff = bootstrap_summary(stats[model], "argmax", mult) - bootstrap_summary(stats[ref], "argmax", mult)
        lo, hi = interval(diff)
        mv = next(x for x in summary if x["model"] == model)
        rv = next(x for x in summary if x["model"] == ref)
        for i, key in enumerate(SUMMARY_KEYS):
            per_fold = [b for b in by_fold if b["model"] == model]
            ref_fold = {b["fold_id"]: b for b in by_fold if b["model"] == ref}
            signs = [np.sign(b[key] - ref_fold[b["fold_id"]][key]) for b in per_fold]
            paired.append({"model": model, "reference": ref, "metric": key, "model_value": mv[key],
                           "reference_value": rv[key], "difference": mv[key] - rv[key], "ci_low": float(lo[i]),
                           "ci_high": float(hi[i]), "ci_excludes_zero": bool(lo[i] > 0 or hi[i] < 0),
                           "folds_negative": int(sum(s < 0 for s in signs)), "folds_positive": int(sum(s > 0 for s in signs))})
    return {"model_comparison": pd.DataFrame(summary), "metrics_by_fold": pd.DataFrame(by_fold),
            "per_class_metrics": pd.concat(per_class, ignore_index=True),
            "confusion_matrices": pd.concat(confusion, ignore_index=True), "paired_differences": pd.DataFrame(paired)}
