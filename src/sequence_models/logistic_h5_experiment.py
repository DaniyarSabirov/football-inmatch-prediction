r"""Experiment 6A: multinomial logistic regression with a five-event history (MLR-H5).

Task ``actions15_360current_v1`` (``docs/canonical_prediction_protocol.md``), unchanged.

Features. For an eligible row at canonical position :math:`t` the input is the XGB-H5 history
of experiment 5, :math:`X_t = (e_{t-4}, e_{t-3}, e_{t-2}, e_{t-1}, e_t)`, built by
``event_history.HistoryRows`` (not re-implemented here): previous observations of the SAME
canonical sequence (match, period), any 360 status; positions before the sequence start hold
``PAD`` = ``<START>``. The target is :math:`Y_t = e_{t+1}` (``target_id``).

Encoding :math:`\phi(X_t)` (fixed vocabulary, nothing fitted, identical in every fold): a
position-specific one-hot vector - one block per history position, so ``Pass`` at lag 0 and
``Pass`` at lag 1 are different columns. Lag blocks 4..1 have 16 categories (15 classes +
``<START>``), the :math:`e_t` block has 15; 79 binary columns in the column order and with the
names of ``event_history.encode`` (XGB-H5). Stored as a sparse CSR matrix (5 non-zeros per row).

Model. :math:`P(Y_t = k \mid X_t) = \exp(w_k^\top\phi(X_t) + b_k) / \sum_j \exp(w_j^\top\phi(X_t) + b_j)`,
fitted with scikit-learn ``LogisticRegression(solver="newton-cholesky", l1_ratio=0)``, which
minimises (``l2_reg_strength = 1 / (C * sum(sample_weight))``, intercepts unpenalised)

.. math:: \frac{1}{N}\sum_{i=1}^{N} -\log P(Y_i = y_i \mid X_i) + \frac{1}{2CN}\lVert W\rVert_F^2 .

The model is additive over history positions on the logit scale: it has no interaction terms
between positions, unlike a fifth-order Markov chain (one free distribution per 5-context) or
trees (which can split on several positions jointly).

Fitting on cells. All inputs are discrete, so the rows of a training set are collapsed to their
unique (features, target) cells with weights = row counts (``event_history.aggregate_cells``).
The weighted data term and the penalty (``sum(weights) = N``) are exactly the row-level
objective; the problem is strictly convex in W, so the optimum is the same (verified by an
unaggregated refit of one fold).

Identifiability. Per position block the one-hot columns sum to 1 (collinear with the
intercept) and the softmax is invariant to adding a common vector to all classes. Neither
direction is penalised away by the data, so the unique L2 optimum satisfies
:math:`\sum_k w_{k,f} = 0` for every feature :math:`f` and :math:`\sum_{c} w_{k,(\ell,c)} = 0`
for every class :math:`k` and position block :math:`\ell` (KKT conditions of the intercepts).
Coefficients are therefore deviation-coded: a difference :math:`w_{k,(\ell,c)} - w_{k,(\ell,c')}`
is the change of the centred log-probability :math:`\log P_k - \tfrac1K\sum_j \log P_j` when
:math:`e_{t-\ell}` changes from :math:`c'` to :math:`c` with the other positions fixed.
"""

from __future__ import annotations

import io
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from .canonical_task import CLASSES, K
from .event_history import N_CATEGORIES, PAD, PAD_NAME, HistoryError, aggregate_cells, position_label
from .logistic import MultinomialLogisticModel
from .splits import FoldAssignment, LeakageError
from .xgboost_experiment import softmax64
from .xgboost_h2_experiment import CheckpointMismatch, atomic_write_bytes, sha256_bytes

FEATURE_SET = "MLR-H5"
N_PREVIOUS = 4
N_COLUMNS = N_PREVIOUS * N_CATEGORIES + K   # 79


# ============================================================================ encoding
def block_widths(n_previous: int = N_PREVIOUS) -> list[int]:
    """Width of each column block of history ids E_{t-n}, ..., E_{t-1}, E_t."""
    return [N_CATEGORIES] * n_previous + [K]


def feature_names(n_previous: int = N_PREVIOUS) -> list[str]:
    """Column names, identical to ``event_history.encode`` (e.g. 'E_{t-2}=Pass', 'E_t=Shot')."""
    names = []
    for col in range(n_previous + 1):
        lag = n_previous - col
        names += [f"{position_label(lag)}={c}" for c in list(CLASSES) + ([PAD_NAME] if lag else [])]
    return names


def encode_sparse(ids: np.ndarray) -> sp.csr_matrix:
    """Position-specific one-hot CSR matrix (float64) of history ids from ``history_ids``."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 2:
        raise HistoryError("history ids must be a 2-d array")
    n, width = ids.shape
    widths = block_widths(width - 1)
    if (ids[:, -1] == PAD).any():
        raise HistoryError("the current event E_t can never be padding")
    for col, w in enumerate(widths):
        if (ids[:, col] < 0).any() or (ids[:, col] >= w).any():
            raise HistoryError(f"category outside the vocabulary at {position_label(width - 1 - col)}")
    offsets = np.r_[0, np.cumsum(widths)[:-1]]
    cols = (ids + offsets).ravel()
    rows = np.repeat(np.arange(n), width)
    return sp.csr_matrix((np.ones(n * width), (rows, cols)), shape=(n, int(sum(widths))))


def feature_table(n_previous: int = N_PREVIOUS) -> pd.DataFrame:
    """One row per column: name, position label, lag, category id and name."""
    out = []
    for col in range(n_previous + 1):
        lag = n_previous - col
        cats = list(CLASSES) + ([PAD_NAME] if lag else [])
        for cid, c in enumerate(cats):
            out.append({"feature": f"{position_label(lag)}={c}", "position": position_label(lag), "lag": lag,
                        "category_id": cid, "category": c})
    return pd.DataFrame(out)


# ============================================================================ objective check
def objective_and_gradient(coef: np.ndarray, intercept: np.ndarray, x: sp.csr_matrix, y_index: np.ndarray,
                           w: np.ndarray, C: float) -> tuple[float, float]:
    """(objective, max |gradient|) of (1/N) sum w_i (-log p_i) + ||W||^2 / (2 C N), N = sum w.

    Recomputed in NumPy from the fitted parameters (independent of the solver). ``coef``
    (K', d) and ``intercept`` (K',) for the K' classes present; ``y_index`` indexes them."""
    z = np.asarray(x @ coef.T) + intercept
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    r = p.copy()
    r[np.arange(len(y_index)), y_index] -= 1.0
    r *= w[:, None]
    n = float(w.sum())
    g_w = np.asarray(x.T @ r).T / n + coef / (C * n)
    g_b = r.sum(axis=0) / n
    obj = float(-(w * np.log(p[np.arange(len(y_index)), y_index])).sum() / n + (coef ** 2).sum() / (2 * C * n))
    return obj, float(max(np.abs(g_w).max(), np.abs(g_b).max()))


# ============================================================================ model
@dataclass
class SolverLogisticModel(MultinomialLogisticModel):
    """``logistic.MultinomialLogisticModel`` (class mapping, absent-class floor, coefficient
    access) with a configurable solver and full warning capture: EVERY warning raised during
    the fit is recorded (not only ConvergenceWarning), e.g. newton-cholesky's fallback to lbfgs."""

    solver: str = "newton-cholesky"

    def _estimator(self) -> LogisticRegression:
        return LogisticRegression(C=self.C, l1_ratio=0.0, solver=self.solver, max_iter=self.max_iter, tol=self.tol,
                                  fit_intercept=True)

    def fit(self, x, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "SolverLogisticModel":
        y = np.asarray(y, dtype=np.int64)
        if y.size == 0:
            raise ValueError("no training rows")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(f"labels must lie in 0..{self.n_classes - 1}")
        if len(np.unique(y)) < 2:
            raise ValueError("at least two target classes are needed")
        self.estimator = self._estimator()
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.estimator.fit(x, y, sample_weight=sample_weight)
        n_iter = int(np.max(self.estimator.n_iter_))
        msgs = [f"{w.category.__name__}: {str(w.message)[:200]}" for w in caught]
        self.fit_info = {
            "C": self.C, "solver": self.solver, "penalty": "l2 (l1_ratio=0)", "tol": self.tol,
            "max_iter": self.max_iter, "n_iter": n_iter, "warnings": msgs,
            "converged": bool(n_iter < self.max_iter and not msgs),
            "n_train_rows": float(len(y) if sample_weight is None else np.sum(sample_weight)),
            "n_train_cells": int(len(y)), "n_features": int(x.shape[1]),
            "absent_classes": [int(c) for c in self.absent_classes], "fit_seconds": time.perf_counter() - start,
        }
        return self

    def predict_proba(self, x) -> np.ndarray:
        """As the parent, for sparse ``x`` (the parent sizes the output with ``len(x)``)."""
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        p = np.zeros((x.shape[0], self.n_classes))
        p[:, self.classes_present] = self.estimator.predict_proba(x)
        if len(self.absent_classes):
            p[:, self.absent_classes] = self.absent_class_floor
            p /= p.sum(axis=1, keepdims=True)
        return p


@dataclass
class MLRH5Model:
    """MLR-H5 on canonical history ids (n, 5) -> (n, 15) probabilities in canonical order."""

    C: float
    lr: dict
    model: SolverLogisticModel | None = None
    fit_info: dict = field(default_factory=dict)
    support: np.ndarray | None = None   # weighted training count of every feature column

    def fit(self, ids: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None) -> "MLRH5Model":
        """Fit on count-weighted unique (features, target) cells; with ``weight``, on the given
        rows with those weights (``fit_rows`` = unaggregated verification fit)."""
        if weight is None:
            cx, cy, w, _ = aggregate_cells(np.asarray(ids), np.asarray(y))
        else:
            cx, cy, w = np.asarray(ids), np.asarray(y), np.asarray(weight, dtype=np.float64)
        x = encode_sparse(cx)
        self.model = SolverLogisticModel(K, float(self.C), int(self.lr["max_iter"]), float(self.lr["tol"]),
                                         float(self.lr["absent_class_floor"]), name=FEATURE_SET,
                                         solver=str(self.lr["solver"])).fit(x, cy, sample_weight=w)
        est = self.model.estimator
        y_index = np.searchsorted(est.classes_, cy)
        obj, grad = objective_and_gradient(est.coef_, est.intercept_, x, y_index, w, float(self.C))
        self.support = np.asarray(x.T @ w).ravel()
        self.fit_info = {**self.model.fit_info, "objective": obj, "max_abs_gradient": grad,
                         "gradient_ok": bool(grad <= float(self.lr["gradient_check_tolerance"]))}
        return self

    def fit_rows(self, ids: np.ndarray, y: np.ndarray) -> "MLRH5Model":
        return self.fit(ids, y, np.ones(len(y)))

    def predict_proba(self, ids: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted")
        cells, _, _, inverse = aggregate_cells(np.asarray(ids))
        return self.model.predict_proba(encode_sparse(cells))[inverse]

    def frozen(self) -> "FrozenMLR":
        return FrozenMLR(self.model.coefficients(), self.model.intercepts(), float(self.lr["absent_class_floor"]))


@dataclass
class FrozenMLR:
    """Fitted parameters only (W (15, 79) in canonical class order, NaN rows for absent classes;
    b (15,)). Probabilities are recomputed in NumPy float64 - an implementation-level check of
    scikit-learn's ``predict_proba`` and the form used after a checkpoint resume."""

    coef: np.ndarray
    intercept: np.ndarray
    absent_class_floor: float = 1e-9

    def predict_proba(self, ids: np.ndarray) -> np.ndarray:
        cells, _, _, inverse = aggregate_cells(np.asarray(ids))
        present = ~np.isnan(self.intercept)
        p = np.zeros((len(cells), K))
        p[:, present] = softmax64(np.asarray(encode_sparse(cells) @ self.coef[present].T) + self.intercept[present])
        if not present.all():
            p[:, ~present] = self.absent_class_floor
            p /= p.sum(axis=1, keepdims=True)
        return p[inverse]


# ============================================================================ hyperparameter selection
def select_C(score, grid: list[float], extension: dict) -> tuple[float, dict[float, float], list[float]]:
    """Argmin of ``score(C)`` (pooled inner-validation log loss) over ``grid``, extended by the
    pre-declared edge rule. Ties go to the smaller C (stronger penalty). Returns
    (chosen C, {C: score}, grid actually evaluated in ascending order)."""
    grid = sorted(float(c) for c in grid)
    upper = sorted(float(c) for c in extension.get("upper", []))
    lower = sorted((float(c) for c in extension.get("lower", [])), reverse=True)
    if (upper and upper[0] <= grid[-1]) or (lower and lower[0] >= grid[0]):
        raise ValueError("edge-extension values must lie outside the grid")
    scores = {c: float(score(c)) for c in grid}
    while True:
        order = sorted(scores)
        best = min(order, key=lambda c: (round(scores[c], 12), c))
        if best == order[-1] and upper:
            c = upper.pop(0)
            scores[c] = float(score(c))
        elif best == order[0] and lower:
            c = lower.pop(0)
            scores[c] = float(score(c))
        else:
            return best, scores, order


# ============================================================================ checkpoints
def save_fold_checkpoint(ckpt_dir: Path, fold_id: int, identifiers: dict, record: dict, arrays: dict) -> None:
    """Arrays first, JSON (holding their hash) last = commit marker. Atomic per file."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    atomic_write_bytes(ckpt_dir / f"fold{fold_id}.npz", data)
    meta = {"identifiers": identifiers, "fold_id": fold_id, "npz_sha256": sha256_bytes(data), **record}
    atomic_write_bytes(ckpt_dir / f"fold{fold_id}.json", json.dumps(meta, indent=1, default=_json_default).encode())


def load_fold_checkpoint(ckpt_dir: Path, fold_id: int, identifiers: dict):
    """None if no committed checkpoint; CheckpointMismatch if incompatible or corrupt."""
    jpath = ckpt_dir / f"fold{fold_id}.json"
    if not jpath.exists():
        return None
    meta = json.loads(jpath.read_text(encoding="utf-8"))
    if meta.get("identifiers") != identifiers:
        diff = sorted(k for k in set(identifiers) | set(meta.get("identifiers", {}))
                      if identifiers.get(k) != meta.get("identifiers", {}).get(k))
        raise CheckpointMismatch(f"fold {fold_id}: checkpoint identifiers differ ({diff}); refusing to mix runs")
    data = (ckpt_dir / f"fold{fold_id}.npz").read_bytes()
    if sha256_bytes(data) != meta["npz_sha256"]:
        raise CheckpointMismatch(f"fold {fold_id}: checkpoint arrays do not match their recorded hash")
    z = np.load(io.BytesIO(data))
    return meta, {k: z[k] for k in z.files}


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


# ============================================================================ nested CV
@dataclass
class NestedResult:
    proba: np.ndarray
    fold_of_row: np.ndarray
    models: dict[int, FrozenMLR] = field(default_factory=dict)
    support: dict[int, np.ndarray] = field(default_factory=dict)
    choices: list[dict] = field(default_factory=list)
    inner_scores: list[dict] = field(default_factory=list)
    fits: list[dict] = field(default_factory=list)
    timings: list[dict] = field(default_factory=list)
    fold_metrics: list[dict] = field(default_factory=list)
    resumed_folds: list[int] = field(default_factory=list)
    computed_folds: list[int] = field(default_factory=list)


def check_fit(info: dict, where: str) -> None:
    """Fail loudly on any optimisation problem (never accept it silently)."""
    if not info["converged"] or not info["gradient_ok"]:
        raise RuntimeError(f"{where}: optimisation problem (n_iter {info['n_iter']}, warnings {info['warnings']}, "
                           f"max |gradient| {info['max_abs_gradient']:.3g})")


def run_nested_cv(ids_all: np.ndarray, match_id: np.ndarray, y: np.ndarray, folds: list[FoldAssignment], lr: dict,
                  ckpt_dir: Path | None = None, identifiers: dict | None = None, trace: list | None = None,
                  log=None) -> NestedResult:
    """Nested CV of MLR-H5 (protocol of ``canonical_baselines`` / experiment 5).

    For each outer fold: leave-one-inner-fold-out over the other 4 folds; for every C the 4
    inner fits are scored on their held inner fold; the pooled inner-validation log loss (sum
    of NLL / number of inner-validation rows) selects C (``select_C``). The model is refitted on
    all outer-training rows with that C and predicts the outer-validation rows once. The outer
    validation matches never enter an inner set or the refit (checked, and recorded in
    ``trace``). With ``ckpt_dir`` each completed fold is committed atomically and resumed only
    when ``identifiers`` match exactly."""
    n = len(y)
    fold_of_row = np.full(n, -1)
    for f in folds:
        fold_of_row[np.isin(match_id, f.validation_match_ids)] = f.fold_id
    if (fold_of_row < 0).any():
        raise LeakageError("an eligible row belongs to no validation fold")
    res = NestedResult(np.full((n, K), np.nan), fold_of_row)

    def mask(match_ids, purpose, fold_id):
        m = np.isin(match_id, list(match_ids))
        if trace is not None:
            trace.append({"outer_fold": fold_id, "purpose": purpose,
                          "match_ids": frozenset(int(x) for x in np.unique(match_id[m]))})
        return m

    for f in folds:
        outer_val = set(f.validation_match_ids)
        if set(f.train_match_ids) & outer_val:
            raise LeakageError(f"fold {f.fold_id}: match in training and validation")
        if ckpt_dir is not None:
            loaded = load_fold_checkpoint(ckpt_dir, f.fold_id, identifiers)
            if loaded is not None:
                meta, arr = loaded
                va = np.isin(match_id, f.validation_match_ids)
                if not np.array_equal(arr["val_index"], np.flatnonzero(va)):
                    raise CheckpointMismatch(f"fold {f.fold_id}: validation rows differ from the checkpoint")
                res.proba[va] = arr["proba"]
                res.models[f.fold_id] = FrozenMLR(arr["coef"], arr["intercept"], float(lr["absent_class_floor"]))
                res.support[f.fold_id] = arr["support"]
                for key in ("choices", "inner_scores", "fits", "timings", "fold_metrics"):
                    getattr(res, key).extend(meta[key])
                if trace is not None:
                    trace.extend({**t, "match_ids": frozenset(t["match_ids"])} for t in meta["trace"])
                res.resumed_folds.append(f.fold_id)
                if log:
                    log(f"fold {f.fold_id}: resumed from checkpoint (C = {meta['choices'][0]['C']})")
                continue
        n_trace = len(trace) if trace is not None else 0
        t_fold = time.perf_counter()
        inner_folds = [g for g in folds if g.fold_id != f.fold_id]
        splits = []
        for held in inner_folds:
            tr_ids = [m for g in inner_folds if g is not held for m in g.validation_match_ids]
            if (set(tr_ids) | set(held.validation_match_ids)) & outer_val:
                raise LeakageError(f"fold {f.fold_id}: outer-validation match in an inner set")
            itr = mask(tr_ids, "inner_train", f.fold_id)
            iva = mask(held.validation_match_ids, "inner_validation", f.fold_id)
            splits.append((held.fold_id, itr, iva))
        fold_fits, fold_scores = [], []

        def score(C: float) -> float:
            t0 = time.perf_counter()
            nll, cnt = 0.0, 0
            for held_id, itr, iva in splits:
                m = MLRH5Model(C, lr).fit(ids_all[itr], y[itr])
                check_fit(m.fit_info, f"fold {f.fold_id} inner {held_id} C={C:g}")
                p = m.predict_proba(ids_all[iva])
                part = float(-np.log(p[np.arange(int(iva.sum())), y[iva]]).sum())
                nll += part
                cnt += int(iva.sum())
                fold_fits.append({"fold_id": f.fold_id, "stage": "inner", "held_inner_fold": held_id,
                                  **{k: v for k, v in m.fit_info.items() if k != "penalty"},
                                  "inner_validation_rows": int(iva.sum()), "inner_validation_log_loss": part / iva.sum()})
            fold_scores.append({"fold_id": f.fold_id, "C": C, "pooled_inner_log_loss": nll / cnt,
                                "inner_validation_rows": cnt, "seconds": time.perf_counter() - t0})
            if log:
                log(f"fold {f.fold_id} C={C:g}: pooled inner log loss {nll / cnt:.8f} ({time.perf_counter() - t0:.0f}s)")
            return nll / cnt

        best, scores, evaluated = select_C(score, list(lr["C_grid"]), lr.get("C_edge_extension", {}))
        selection_seconds = time.perf_counter() - t_fold
        t0 = time.perf_counter()
        tr = mask(f.train_match_ids, "outer_train", f.fold_id)
        model = MLRH5Model(best, lr).fit(ids_all[tr], y[tr])
        check_fit(model.fit_info, f"fold {f.fold_id} outer refit C={best:g}")
        va = mask(f.validation_match_ids, "outer_validation_predict", f.fold_id)
        p = model.predict_proba(ids_all[va])
        res.proba[va] = p
        frozen = model.frozen()
        res.models[f.fold_id] = frozen
        res.support[f.fold_id] = model.support
        ll = float(-np.log(p[np.arange(int(va.sum())), y[va]]).mean())
        choice = {"fold_id": f.fold_id, "C": best, "inner_log_loss": scores[best],
                  "grid_evaluated": json.dumps(evaluated), "at_evaluated_grid_edge": best in (evaluated[0], evaluated[-1]),
                  "edge_extension_used": len(evaluated) > len(lr["C_grid"]),
                  "train_matches": len(f.train_match_ids), "validation_matches": len(f.validation_match_ids),
                  "train_rows": int(tr.sum()), "validation_rows": int(va.sum()),
                  "absent_training_classes": json.dumps(model.fit_info["absent_classes"])}
        fold_fits.append({"fold_id": f.fold_id, "stage": "outer_refit", "held_inner_fold": -1,
                          **{k: v for k, v in model.fit_info.items() if k != "penalty"}})
        timing = {"fold_id": f.fold_id, "selection_seconds": selection_seconds,
                  "final_fit_predict_seconds": time.perf_counter() - t0}
        fm = {"fold_id": f.fold_id, "log_loss": ll, "n_rows": int(va.sum())}
        res.choices.append(choice)
        res.inner_scores.extend(fold_scores)
        res.fits.extend(fold_fits)
        res.timings.append(timing)
        res.fold_metrics.append(fm)
        res.computed_folds.append(f.fold_id)
        if log:
            log(f"fold {f.fold_id}: C = {best:g} (evaluated {evaluated}); outer-validation log loss {ll:.6f}")
        if ckpt_dir is not None:
            fold_trace = [{**t, "match_ids": sorted(t["match_ids"])} for t in trace[n_trace:]] if trace is not None else []
            save_fold_checkpoint(ckpt_dir, f.fold_id, identifiers,
                                 {"choices": [choice], "inner_scores": fold_scores, "fits": fold_fits,
                                  "timings": [timing], "fold_metrics": [fm], "trace": fold_trace},
                                 {"val_index": np.flatnonzero(va).astype(np.int64), "proba": p.astype(np.float64),
                                  "coef": frozen.coef, "intercept": frozen.intercept, "support": model.support})
    if np.isnan(res.proba).any():
        raise RuntimeError("rows without prediction")
    return res


# ============================================================================ interpretation
def coefficient_tables(models: dict[int, FrozenMLR], support: dict[int, np.ndarray], min_support: float
                       ) -> dict[str, pd.DataFrame]:
    """Coefficients of the 5 outer-fold models (each fitted on its outer-training rows only).

    - ``coefficients_by_fold``: long table (fold, class, feature, coefficient, training support).
    - ``coefficient_summary``: mean / sd / min / max over folds, sign agreement over folds.
    - ``top_coefficients``: per target class, the 10 largest and 10 smallest mean coefficients
      among features with training support >= ``min_support`` rows in every fold.
    - ``position_strength``: per target class and history position, the range (max - min) and
      root mean square of the mean coefficients over the position's categories with support.
    - ``identifiability``: max |sum over classes| and max |sum over a position block|.
    - ``intercepts``: per fold.
    """
    ft = feature_table()
    long, ident, inter = [], [], []
    for fid, m in sorted(models.items()):
        w = m.coef
        for k in range(K):
            long.append(pd.DataFrame({"fold_id": fid, "class_id": k, "class": CLASSES[k], **ft.to_dict("list"),
                                      "coefficient": w[k], "train_support": support[fid]}))
            inter.append({"fold_id": fid, "class_id": k, "class": CLASSES[k], "intercept": m.intercept[k]})
        offs = np.r_[0, np.cumsum(block_widths())]
        block = max(float(np.nanmax(np.abs(w[:, a:b].sum(axis=1)))) for a, b in zip(offs[:-1], offs[1:]))
        ident.append({"fold_id": fid, "max_abs_sum_over_classes": float(np.nanmax(np.abs(np.nansum(w, axis=0)))),
                      "max_abs_sum_over_position_block": block, "max_abs_coefficient": float(np.nanmax(np.abs(w)))})
    long = pd.concat(long, ignore_index=True)
    g = long.groupby(["class_id", "class", "feature", "position", "lag", "category"], sort=False)
    summ = g["coefficient"].agg(mean="mean", sd="std", min="min", max="max").reset_index()
    summ["min_train_support"] = g["train_support"].min().to_numpy()
    summ["sign_agreement_over_folds"] = g["coefficient"].apply(lambda s: bool((s > 0).all() or (s < 0).all())).to_numpy()
    eligible = summ[summ["min_train_support"] >= min_support]
    top = []
    for k in range(K):
        sk = eligible[eligible["class_id"] == k]
        top.append(sk.nlargest(10, "mean").assign(direction="positive"))
        top.append(sk.nsmallest(10, "mean").assign(direction="negative"))
    top = pd.concat(top, ignore_index=True)
    pos = []
    supported = summ[summ["min_train_support"] > 0]
    for (k, cname, p, lag), s in supported.groupby(["class_id", "class", "position", "lag"], sort=False):
        pos.append({"class_id": k, "class": cname, "position": p, "lag": lag, "n_categories": len(s),
                    "range": float(s["mean"].max() - s["mean"].min()), "rms": float(np.sqrt((s["mean"] ** 2).mean()))})
    pos = pd.DataFrame(pos)
    overall = pos.groupby(["position", "lag"], as_index=False).agg(range=("range", "mean"), rms=("rms", "mean"))
    overall.insert(0, "class", "mean over classes")
    overall.insert(0, "class_id", -1)
    pos = pd.concat([pos, overall], ignore_index=True)
    return {"coefficients_by_fold": long, "coefficient_summary": summ, "top_coefficients": top,
            "position_strength": pos, "identifiability": pd.DataFrame(ident), "intercepts": pd.DataFrame(inter)}


def nll_difference_strata(y: np.ndarray, p_model: np.ndarray, p_ref: np.ndarray, seq_pos: np.ndarray,
                          current: np.ndarray) -> pd.DataFrame:
    r"""Descriptive decomposition of the pooled log-loss difference
    :math:`\bar d = \frac1n\sum_i d_i`, :math:`d_i = -\log p^{model}_{i,y_i} + \log p^{ref}_{i,y_i}`,
    by history length (t < 4 rows contain <START>), current class and target class.
    ``contribution`` = sum of d_i in the stratum / n (the strata contributions add to d-bar)."""
    n = len(y)
    d = -np.log(p_model[np.arange(n), y]) + np.log(p_ref[np.arange(n), y])
    out = [{"stratum": "all", "level": "all", "n_rows": n, "mean_difference": float(d.mean()),
            "contribution": float(d.sum() / n), "share_model_better": float((d < 0).mean())}]
    hist = np.where(seq_pos >= 4, "t >= 4 (full history)", np.char.add("t = ", seq_pos.astype(str)))
    for name, labels in (("history_length", hist), ("current_class", np.asarray(CLASSES)[current]),
                         ("target_class", np.asarray(CLASSES)[y])):
        for lev in pd.unique(labels):
            s = labels == lev
            out.append({"stratum": name, "level": str(lev), "n_rows": int(s.sum()), "mean_difference": float(d[s].mean()),
                        "contribution": float(d[s].sum() / n), "share_model_better": float((d[s] < 0).mean())})
    return pd.DataFrame(out)


__all__ = ["FEATURE_SET", "N_COLUMNS", "encode_sparse", "feature_names", "feature_table", "objective_and_gradient",
           "MLRH5Model", "FrozenMLR", "SolverLogisticModel", "select_C", "run_nested_cv", "coefficient_tables",
           "nll_difference_strata", "save_fold_checkpoint", "load_fold_checkpoint", "CheckpointMismatch"]
