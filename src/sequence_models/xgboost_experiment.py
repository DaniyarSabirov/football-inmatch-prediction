r"""Experiment 5: XGBoost (gradient-boosted trees) on ``actions15_360current_v1``.

Model. For K = 15 classes XGBoost's ``multi:softprob`` objective fits K additive tree
ensembles :math:`z_c(\mathbf{x}) = b_c + \sum_{m=1}^{M} f_{m,c}(\mathbf{x})` (one regression
tree per class per boosting round, :math:`b_c` the intercept ``base_score``) and predicts

.. math:: P(E_{t+1} = c \mid \mathbf{x}_t) = \exp z_c(\mathbf{x}_t) / \sum_{j=1}^{K} \exp z_j(\mathbf{x}_t).

Training minimises the multiclass log loss plus
:math:`\sum_{m,c} \gamma T_{m,c} + \tfrac{\lambda}{2}\lVert w_{m,c}\rVert^2` by second-order
(Newton) boosting. Probabilities are computed here in float64 from XGBoost's raw margins, so
every row is a strictly positive distribution that sums to 1. Column c is class id c of
``canonical_task.CLASSES``: labels are passed as canonical ids, no label encoder is used, and
``num_class = 15`` keeps all 15 outputs even when a class is absent from a training fold.

Fitting on cells. Features are discrete, so the rows of a training set are collapsed to their
unique (features, target) cells with weights = row counts. Gradient and hessian sums - and
hence split gains, leaf values, ``min_child_weight`` and the intercept - are the same weighted
sums as in the row-level fit. Only the float32 accumulation order differs. Near-tied splits can
flip, and the differences compound over rounds: measured held-out |delta log loss| is 1e-7
(XGB-F0, 376 rounds) and 1.6e-4 (XGB-H5, 1,895 rounds), in ``fit_verification.csv``.
Row subsampling would sample cells instead of rows, so ``subsample`` is fixed to 1.

Nested CV (as in ``canonical_baselines``): for each outer fold, leave-one-inner-fold-out over
the 4 other folds. A candidate's 4 inner boosters are trained in lockstep; the pooled
inner-validation log loss is recorded after every round; training stops after ``patience``
rounds without improvement. The chosen (candidate, rounds) minimises that pooled curve (ties:
earlier candidate, then fewer rounds). The model is refitted on all outer-training rows with
exactly that number of rounds and scored once on the outer-validation rows. Outer-validation
matches never enter any inner set (``trace`` records every data set handed to XGBoost).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from .canonical_task import CLASSES, K
from .evaluation_v2 import (
    SUMMARY_KEYS,
    bootstrap_multiplicities,
    bootstrap_recall,
    bootstrap_summary,
    calibration,
    confusion_counts,
    interval,
    metrics_from_confusion,
    per_class_from_confusion,
    per_match_stats,
    probability_metrics,
)
from .event_history import FEATURE_SETS, HistoryRows, aggregate_cells, encode, feature_position
from .splits import FoldAssignment, LeakageError

SHOT = CLASSES.index("Shot")


# ============================================================================ model
def softmax64(margins: np.ndarray) -> np.ndarray:
    z = np.asarray(margins, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def booster_params(cfg: dict, candidate: dict) -> dict:
    x = cfg["xgboost"]
    p = {"objective": x["objective"], "num_class": int(x["num_class"]), "tree_method": x["tree_method"],
         "eval_metric": x["eval_metric"], "nthread": int(x["nthread"]), "seed": int(x["seed"]),
         "verbosity": 0, **x["fixed_params"], **candidate}
    if p["num_class"] != K or p["objective"] != "multi:softprob":
        raise ValueError("experiment 5 requires multi:softprob over the 15 canonical classes")
    return p


def cell_dmatrix(ids: np.ndarray, y: np.ndarray) -> tuple[xgb.DMatrix, int]:
    """Weighted DMatrix of the unique (features, target) cells of rows (ids, y)."""
    cx, cy, w, _ = aggregate_cells(ids, y)
    x, names = encode(cx)
    return xgb.DMatrix(x, label=cy, weight=w, feature_names=[_safe(n) for n in names]), int(w.sum())


def _safe(name: str) -> str:
    # XGBoost forbids '[', ']' and '<' in feature names
    return name.replace("<", "(").replace(">", ")").replace("[", "(").replace("]", ")")


@dataclass
class XGBNextEventModel:
    """XGBoost next-event model on canonical event ids (one feature configuration)."""

    params: dict
    n_rounds: int
    booster: xgb.Booster | None = None

    def fit(self, ids: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None) -> "XGBNextEventModel":
        if weight is None:
            dm, _ = cell_dmatrix(ids, y)
        else:   # explicit rows/weights (used by the row-level equivalence check and tests)
            x, names = encode(ids)
            dm = xgb.DMatrix(x, label=y, weight=weight, feature_names=[_safe(n) for n in names])
        self.booster = xgb.train(self.params, dm, num_boost_round=int(self.n_rounds))
        return self

    def fit_rows(self, ids: np.ndarray, y: np.ndarray) -> "XGBNextEventModel":
        """Row-level fit (every row once, weight 1), for verification only."""
        return self.fit(ids, y, np.ones(len(y)))

    def predict_proba(self, ids: np.ndarray) -> np.ndarray:
        """(n, 15) float64 probabilities; column c = canonical class id c."""
        if self.booster is None:
            raise RuntimeError("model not fitted")
        cells, _, _, inverse = aggregate_cells(ids)
        x, names = encode(cells)
        z = self.booster.predict(xgb.DMatrix(x, feature_names=[_safe(n) for n in names]), output_margin=True)
        z = np.asarray(z).reshape(len(cells), -1)
        if z.shape[1] != K:
            raise ValueError(f"booster returned {z.shape[1]} margins, expected {K}")
        return softmax64(z)[inverse]


# ============================================================================ search space
def candidate_list(cfg: dict) -> list[dict]:
    """``n_candidates`` distinct configurations drawn once from the fixed space (seeded)."""
    s = cfg["search"]
    rng = np.random.default_rng(int(s["candidate_seed"]))
    keys = list(s["space"])
    total = int(np.prod([len(s["space"][k]) for k in keys]))
    n = int(s["n_candidates"])
    if n > total:
        raise ValueError("more candidates than configurations")
    seen, out = set(), []
    while len(out) < n:
        cand = {k: s["space"][k][int(rng.integers(len(s["space"][k])))] for k in keys}
        key = tuple(cand[k] for k in keys)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


# ============================================================================ nested CV
@dataclass
class NestedOutput:
    feature_set: str
    proba: np.ndarray
    fold_of_row: np.ndarray
    models: dict[int, XGBNextEventModel] = field(default_factory=dict)
    choices: list[dict] = field(default_factory=list)
    curves: list[dict] = field(default_factory=list)
    timings: list[dict] = field(default_factory=list)
    n_fits: int = 0
    n_boosting_rounds: int = 0


def lockstep_curve(params: dict, inner: list[tuple[xgb.DMatrix, xgb.DMatrix, int]], max_rounds: int,
                   patience: int) -> np.ndarray:
    """Pooled inner-validation log loss after rounds 1, 2, ...; stops after ``patience``
    rounds without improvement. ``inner``: (train cells, validation cells, validation rows)."""
    boosters = [xgb.Booster(params, [dtr, dva]) for dtr, dva, _ in inner]
    n_total = sum(n for _, _, n in inner)
    curve, best, since = [], np.inf, 0
    for it in range(max_rounds):
        pooled = 0.0
        for b, (dtr, dva, n) in zip(boosters, inner):
            b.update(dtr, it)
            pooled += float(b.eval(dva, "v", it).rsplit(":", 1)[1]) * n   # weighted mean x rows = sum
        curve.append(pooled / n_total)
        if curve[-1] < best - 1e-12:
            best, since = curve[-1], 0
        else:
            since += 1
            if since >= patience:
                break
    return np.asarray(curve)


def run_nested_cv(rows: HistoryRows, folds: list[FoldAssignment], cfg: dict, feature_set: str,
                  trace: list | None = None, log=None) -> NestedOutput:
    ids_all = rows.features(feature_set)
    n = len(rows.y)
    fold_of_row = np.full(n, -1)
    for f in folds:
        fold_of_row[rows.mask(f.validation_match_ids)] = f.fold_id
    if (fold_of_row < 0).any():
        raise LeakageError("an eligible row belongs to no validation fold")
    out = NestedOutput(feature_set, np.full((n, K), np.nan), fold_of_row)
    cands = candidate_list(cfg)
    s = cfg["search"]

    def dm(match_ids, purpose, fold_id):
        m = rows.mask(match_ids)
        if trace is not None:
            trace.append({"feature_set": feature_set, "outer_fold": fold_id, "purpose": purpose,
                          "match_ids": frozenset(int(x) for x in np.unique(rows.match_id[m]))})
        return (*cell_dmatrix(ids_all[m], rows.y[m]), m)

    for f in folds:
        outer_val = set(f.validation_match_ids)
        if set(f.train_match_ids) & outer_val:
            raise LeakageError(f"fold {f.fold_id}: match in training and validation")
        inner_folds = [g for g in folds if g.fold_id != f.fold_id]
        start = time.perf_counter()
        inner = []
        for held in inner_folds:
            tr_ids = [m for g in inner_folds if g is not held for m in g.validation_match_ids]
            if (set(tr_ids) | set(held.validation_match_ids)) & outer_val:
                raise LeakageError(f"fold {f.fold_id}: outer-validation match in an inner set")
            dtr, _, _ = dm(tr_ids, "inner_train", f.fold_id)
            dva, n_va, _ = dm(held.validation_match_ids, "inner_validation", f.fold_id)
            inner.append((dtr, dva, n_va))
        scores = []
        for ci, cand in enumerate(cands):
            t0 = time.perf_counter()
            curve = lockstep_curve(booster_params(cfg, cand), inner, int(s["max_rounds"]), int(s["patience"]))
            r = int(np.argmin(curve)) + 1
            scores.append((round(float(curve[r - 1]), 12), ci, r))
            out.n_fits += len(inner)
            out.n_boosting_rounds += len(curve) * len(inner)
            out.curves.append({"feature_set": feature_set, "fold_id": f.fold_id, "candidate": ci, **cand,
                               "best_rounds": r, "rounds_trained": len(curve), "inner_log_loss": float(curve[r - 1]),
                               "hit_max_rounds": len(curve) == int(s["max_rounds"]),
                               "seconds": time.perf_counter() - t0, "curve": json.dumps([round(v, 8) for v in curve])})
            if log:
                log(f"{feature_set} fold {f.fold_id} cand {ci} {cand} -> rounds {r}/{len(curve)} "
                    f"inner LL {curve[r - 1]:.6f} ({time.perf_counter() - t0:.0f}s)")
        best_score, best_ci, best_r = min(scores)
        selection = time.perf_counter() - start
        t0 = time.perf_counter()
        dtr, _, _ = dm(f.train_match_ids, "outer_train", f.fold_id)
        model = XGBNextEventModel(booster_params(cfg, cands[best_ci]), best_r)
        tr = rows.mask(f.train_match_ids)
        model.fit(ids_all[tr], rows.y[tr])
        out.n_fits += 1
        out.n_boosting_rounds += best_r
        va = rows.mask(f.validation_match_ids)
        if trace is not None:
            trace.append({"feature_set": feature_set, "outer_fold": f.fold_id, "purpose": "outer_validation_predict",
                          "match_ids": frozenset(int(x) for x in np.unique(rows.match_id[va]))})
        out.proba[va] = model.predict_proba(ids_all[va])
        out.models[f.fold_id] = model
        out.choices.append({"feature_set": feature_set, "fold_id": f.fold_id, "candidate": best_ci,
                            **cands[best_ci], "n_estimators": best_r, "inner_log_loss": best_score,
                            "rounds_at_max": best_r == int(s["max_rounds"]),
                            "train_rows": int(tr.sum()), "validation_rows": int(va.sum())})
        out.timings.append({"feature_set": feature_set, "fold_id": f.fold_id, "selection_seconds": selection,
                            "final_fit_predict_seconds": time.perf_counter() - t0})
    if np.isnan(out.proba).any():
        raise RuntimeError(f"{feature_set}: rows without prediction")
    return out


# ============================================================================ metrics
def check_probabilities(p: np.ndarray, tol: float) -> None:
    if p.ndim != 2 or p.shape[1] != K:
        raise ValueError(f"probability matrix has shape {p.shape}, expected (n, {K})")
    if not np.isfinite(p).all() or (p < 0).any() or np.abs(p.sum(axis=1) - 1).max() > tol:
        raise ValueError("invalid probabilities")


def weighted_average_precision(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None,
                               order: np.ndarray | None = None) -> float:
    """Average precision, identical to sklearn's (ties grouped by score), with optional
    weights. ``order`` (descending-score argsort) may be precomputed for repeated calls."""
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=np.float64)
    if order is None:
        order = np.argsort(-score, kind="stable")
    s, yy, ww = score[order], y[order], w[order]
    tp = np.cumsum(ww * yy)
    fp = np.cumsum(ww * (1 - yy))
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp, fp = tp[last], fp[last]
    if tp[-1] <= 0:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate_models(y: np.ndarray, groups: np.ndarray, fold_of_row: np.ndarray, proba: dict[str, np.ndarray],
                    pairs: list[tuple[str, str]], cfg: dict) -> dict[str, pd.DataFrame]:
    """Same metric definitions as ``canonical_baselines.evaluate``, for any set of models,
    plus paired Shot AP / Shot recall bootstrap and fold-level paired differences."""
    ecfg = cfg["evaluation"]
    matches, match_index = np.unique(groups, return_inverse=True)
    mult = bootstrap_multiplicities(len(matches), int(ecfg["bootstrap_replicates"]), int(ecfg["bootstrap_seed"]))
    summary, by_fold, per_class, confusion, predicted_dist = [], [], [], [], []
    stats, boot, shot_ap_boot, shot_rec_boot = {}, {}, {}, {}
    for name, p in proba.items():
        check_probabilities(p, float(ecfg["probability_tolerance"]))
        d = p.argmax(axis=1)
        cm = confusion_counts(y, d, K)
        pc = per_class_from_confusion(cm, CLASSES)
        ap = [weighted_average_precision(y == c, p[:, c]) if (y == c).any() else np.nan for c in range(K)]
        cal, _ = calibration(y, p, CLASSES)
        row = {"model": name, "n_rows": len(y), "n_matches": len(matches), **probability_metrics(y, p),
               **metrics_from_confusion(cm), "shot_average_precision": ap[SHOT],
               "shot_recall": float(pc.at[SHOT, "recall"]), "shot_precision": float(pc.at[SHOT, "precision"]),
               "n_classes_predicted": int((cm.sum(0) > 0).sum()), "classwise_ece": cal["classwise_ece"],
               "top_label_ece": cal["top_label_ece"],
               "max_abs_calibration_in_the_large": cal["max_abs_calibration_in_the_large"]}
        s = per_match_stats(y, p, {"argmax": d}, groups, K)
        stats[name] = s
        boot[name] = bootstrap_summary(s, "argmax", mult)
        lo, hi = interval(boot[name])
        for i, key in enumerate(SUMMARY_KEYS):
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = float(lo[i]), float(hi[i])
        order = np.argsort(-p[:, SHOT], kind="stable")
        shot_ap_boot[name] = np.array([weighted_average_precision(y == SHOT, p[:, SHOT], m[match_index], order)
                                       for m in mult])
        shot_rec_boot[name] = bootstrap_recall(s, "argmax", mult)[:, SHOT]
        row["shot_average_precision_ci_low"], row["shot_average_precision_ci_high"] = (
            float(np.nanpercentile(shot_ap_boot[name], 2.5)), float(np.nanpercentile(shot_ap_boot[name], 97.5)))
        summary.append(row)
        pc.insert(0, "class_id", np.arange(K))
        pc.insert(0, "model", name)
        pc["average_precision"] = ap
        pc["mean_predicted_probability"] = p.mean(axis=0)
        per_class.append(pc)
        for c in range(K):
            predicted_dist.append({"model": name, "class_id": c, "class": CLASSES[c],
                                   "true_count": int((y == c).sum()), "argmax_predicted_count": int((d == c).sum()),
                                   "mean_predicted_probability": float(p[:, c].mean()),
                                   "expected_count": float(p[:, c].sum())})
        for f in sorted(set(fold_of_row)):
            sel = fold_of_row == f
            fcm = confusion_counts(y[sel], d[sel], K)
            by_fold.append({"model": name, "fold_id": int(f), "n_rows": int(sel.sum()),
                            "n_matches": int(len(np.unique(groups[sel]))), **probability_metrics(y[sel], p[sel]),
                            **metrics_from_confusion(fcm),
                            "shot_average_precision": weighted_average_precision(y[sel] == SHOT, p[sel, SHOT]),
                            "shot_recall": float(per_class_from_confusion(fcm, CLASSES).at[SHOT, "recall"]),
                            "n_classes_predicted": int((fcm.sum(0) > 0).sum())})
        cmf = pd.DataFrame(cm.astype(int), index=list(CLASSES), columns=list(CLASSES)).rename_axis("true_class")
        cmf = cmf.reset_index().melt(id_vars="true_class", var_name="predicted_class", value_name="count")
        cmf.insert(0, "model", name)
        confusion.append(cmf)

    sm = {r["model"]: r for r in summary}
    bf = pd.DataFrame(by_fold)
    paired, fold_paired = [], []
    keys = list(SUMMARY_KEYS) + ["shot_average_precision", "shot_recall"]
    for model, ref in pairs:
        reps = {k: boot[model][:, i] - boot[ref][:, i] for i, k in enumerate(SUMMARY_KEYS)}
        reps["shot_average_precision"] = shot_ap_boot[model] - shot_ap_boot[ref]
        reps["shot_recall"] = shot_rec_boot[model] - shot_rec_boot[ref]
        mf, rf = bf[bf.model == model].set_index("fold_id"), bf[bf.model == ref].set_index("fold_id")
        for key in keys:
            diff = sm[model][key] - sm[ref][key]
            lo, hi = np.nanpercentile(reps[key], 2.5), np.nanpercentile(reps[key], 97.5)
            fd = (mf[key] - rf[key]).to_numpy()
            paired.append({"model": model, "reference": ref, "metric": key, "model_value": sm[model][key],
                           "reference_value": sm[ref][key], "difference": diff,
                           "relative_difference": diff / sm[ref][key] if sm[ref][key] else np.nan,
                           "ci_low": float(lo), "ci_high": float(hi), "ci_excludes_zero": bool(lo > 0 or hi < 0),
                           "folds_negative": int((fd < 0).sum()), "folds_positive": int((fd > 0).sum()),
                           "folds_zero": int((fd == 0).sum())})
            for fid, v in zip(mf.index, fd):
                fold_paired.append({"model": model, "reference": ref, "metric": key, "fold_id": int(fid),
                                    "model_value": float(mf.at[fid, key]), "reference_value": float(rf.at[fid, key]),
                                    "difference": float(v)})
    return {"model_comparison": pd.DataFrame(summary), "metrics_by_fold": bf,
            "per_class_metrics": pd.concat(per_class, ignore_index=True),
            "confusion_matrices": pd.concat(confusion, ignore_index=True),
            "predicted_class_distribution": pd.DataFrame(predicted_dist),
            "paired_differences": pd.DataFrame(paired), "paired_differences_by_fold": pd.DataFrame(fold_paired)}


# ============================================================================ interpretability
def permutation_importance(rows: HistoryRows, nested: NestedOutput, folds: list[FoldAssignment], cfg: dict
                           ) -> pd.DataFrame:
    """Held-out permutation importance of history positions (XGB-H5), per outer fold.

    For each outer fold, the fold's FINAL model is scored on its outer-validation rows with the
    columns of one position (or a block of positions, jointly) permuted across those rows.
    Reported: increase of log loss and decrease of accuracy over the unpermuted predictions.
    Computed after model selection; nothing here feeds back into any choice."""
    ecfg = cfg["evaluation"]
    ids_all = rows.features(nested.feature_set)
    width = ids_all.shape[1]
    groups = {f"E_{{t-{width - 1 - j}}}" if j < width - 1 else "E_t": [j] for j in range(width)}
    if width == 5:
        groups["E_{t-4..t-1} (all history)"] = [0, 1, 2, 3]
        groups["E_{t-4..t-2} (beyond Markov-2 context)"] = [0, 1, 2]
    rng = np.random.default_rng(int(ecfg["permutation_seed"]))
    out = []
    for f in folds:
        va = rows.mask(f.validation_match_ids)
        x, y = ids_all[va], rows.y[va]
        model = nested.models[f.fold_id]
        base = model.predict_proba(x)
        base_ll = -np.log(base[np.arange(len(y)), y]).mean()
        base_acc = (base.argmax(1) == y).mean()
        for gname, cols in groups.items():
            dll, dacc = [], []
            for _ in range(int(ecfg["permutation_repeats"])):
                xp = x.copy()
                xp[:, cols] = x[rng.permutation(len(x))][:, cols]
                pp = model.predict_proba(xp)
                dll.append(-np.log(pp[np.arange(len(y)), y]).mean() - base_ll)
                dacc.append(base_acc - (pp.argmax(1) == y).mean())
            out.append({"fold_id": f.fold_id, "position": gname, "n_rows": int(va.sum()), "base_log_loss": base_ll,
                        "log_loss_increase_mean": float(np.mean(dll)), "log_loss_increase_sd": float(np.std(dll, ddof=1)),
                        "accuracy_decrease_mean": float(np.mean(dacc)), "accuracy_decrease_sd": float(np.std(dacc, ddof=1))})
    return pd.DataFrame(out)


def gain_importance(nested: NestedOutput) -> pd.DataFrame:
    """Total split gain per history position (training-set quantity; descriptive only)."""
    out = []
    for fid, model in nested.models.items():
        score = model.booster.get_score(importance_type="total_gain")
        names = model.booster.feature_names
        pos = dict(zip(names, feature_position(names)))
        total = sum(score.values())
        agg: dict[str, float] = {}
        for feat, g in score.items():
            agg[pos[feat]] = agg.get(pos[feat], 0.0) + g
        for p_, g in agg.items():
            out.append({"fold_id": fid, "position": p_, "total_gain": g, "share_of_total_gain": g / total})
    return pd.DataFrame(out)


__all__ = ["FEATURE_SETS", "XGBNextEventModel", "candidate_list", "run_nested_cv", "evaluate_models",
           "permutation_importance", "gain_importance", "weighted_average_precision", "softmax64", "booster_params"]
