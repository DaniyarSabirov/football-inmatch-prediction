r"""Multinomial (softmax) logistic regression for next-event prediction.

Model. For a feature vector :math:`\mathbf{x}_t \in \mathbb{R}^d` and each class
:math:`c \in \{1, \dots, K\}`, the logit is :math:`z_{t,c} = \mathbf{w}_c^T \mathbf{x}_t + b_c` and

.. math:: P(E_{t+1} = c \mid \mathbf{x}_t) = \frac{\exp(z_{t,c})}{\sum_{r=1}^{K} \exp(z_{t,r})}.

Estimation (scikit-learn ``LogisticRegression``, ``solver="lbfgs"``, ``l1_ratio=0``, i.e. an
L2 penalty). scikit-learn >= 1.8 minimises, over :math:`W` and :math:`\mathbf{b}`,

.. math:: \frac{1}{N}\sum_{i=1}^{N} -\log P(E_{t_i+1} = y_i \mid \mathbf{x}_{t_i})
          + \frac{1}{2\,C\,N}\lVert W \rVert_F^2,

(``l2_reg_strength = 1 / (C * sum(sample_weight))`` in ``sklearn/linear_model/_logistic.py``;
the intercepts are not penalised, ``_linear_loss.py``). Equivalently
:math:`\sum_i -\log P(\cdot) + \lVert W\rVert_F^2 / (2C)`: the penalty is fixed while the
data term grows with N, so a given C regularises less on more data. ``C = inf`` is the
unpenalised maximum-likelihood fit (the non-deprecated way to request no penalty). With
more than two classes and lbfgs, scikit-learn always fits the multinomial (softmax) model;
the removed ``multi_class`` parameter is not used. lbfgs is deterministic, so no random seed
is involved.

Class handling. Probabilities are always returned as an (n, K) array in canonical class
order (``event_classes.SCENARIO_D15_CLASSES``), whatever classes occur in training:

- a target class ABSENT from the training labels cannot be learnt by scikit-learn (it is not
  in ``classes_``); it receives probability ``absent_class_floor`` and the row is
  renormalised, so the log loss stays finite. ``absent_classes`` reports them.
- an input category never seen in training has an all-zero feature column; its weights get
  no data gradient, so the L2 penalty keeps them at 0 and the prediction is the
  intercept-only distribution softmax(b) (a data-driven fallback, analogous to the Markov
  chain's frequency fallback).

Convergence is recorded from ``n_iter_`` and from scikit-learn's ``ConvergenceWarning``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .event_classes import K
from .feature_sets import one_hot


@dataclass
class MultinomialLogisticModel:
    """Thin, explicit wrapper around scikit-learn's multinomial logistic regression."""

    n_classes: int = K
    C: float = 1.0
    max_iter: int = 2000
    tol: float = 1e-6
    absent_class_floor: float = 1e-9
    name: str = "logistic_regression"
    estimator: LogisticRegression | None = None
    fit_info: dict = field(default_factory=dict)

    def _estimator(self) -> LogisticRegression:
        return LogisticRegression(C=self.C, l1_ratio=0.0, solver="lbfgs", max_iter=self.max_iter, tol=self.tol,
                                  fit_intercept=True)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "MultinomialLogisticModel":
        y = np.asarray(y, dtype=np.int64)
        if y.size == 0:
            raise ValueError("no training rows")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(f"labels must lie in 0..{self.n_classes - 1}")
        present = np.unique(y if sample_weight is None else y[np.asarray(sample_weight) > 0])
        if len(present) < 2:
            raise ValueError("at least two target classes are needed to fit a logistic regression")
        self.estimator = self._estimator()
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.estimator.fit(x, y, sample_weight=sample_weight)
        seconds = time.perf_counter() - start
        n_iter = int(np.max(self.estimator.n_iter_))
        warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        self.fit_info = {
            "C": self.C, "solver": "lbfgs", "penalty": "l2 (l1_ratio=0)" if np.isfinite(self.C) else "none (C=inf)",
            "tol": self.tol, "max_iter": self.max_iter, "n_iter": n_iter,
            "converged": bool(n_iter < self.max_iter and not warned), "convergence_warning": warned,
            "n_train_rows": int(len(y)) if sample_weight is None else float(np.sum(sample_weight)),
            "n_unique_train_rows": int(len(y)), "n_features": int(x.shape[1]),
            "absent_classes": [int(c) for c in self.absent_classes], "fit_seconds": seconds,
        }
        return self

    @property
    def classes_present(self) -> np.ndarray:
        return self.estimator.classes_.astype(np.int64)

    @property
    def absent_classes(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.n_classes), self.classes_present)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        p = np.zeros((len(x), self.n_classes))
        p[:, self.classes_present] = self.estimator.predict_proba(x)
        if len(self.absent_classes):
            p[:, self.absent_classes] = self.absent_class_floor
            p /= p.sum(axis=1, keepdims=True)
        return p

    def coefficients(self) -> np.ndarray:
        """W as a (K, d) array in canonical class order (rows of absent classes are NaN)."""
        w = np.full((self.n_classes, self.estimator.coef_.shape[1]), np.nan)
        w[self.classes_present] = self.estimator.coef_
        return w

    def intercepts(self) -> np.ndarray:
        b = np.full(self.n_classes, np.nan)
        b[self.classes_present] = self.estimator.intercept_
        return b

    def logits(self, x: np.ndarray) -> np.ndarray:
        """z (n, K) = X W^T + b; NaN columns for absent classes."""
        return x @ self.coefficients().T + self.intercepts()


def aggregate_one_hot(current: np.ndarray, target: np.ndarray, n_classes: int = K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Sufficient statistics of a one-hot (F0) design: the unique (current, target) pairs.

    Returns (X_unique, y_unique, counts). With ``sample_weight = counts`` the weighted
    objective equals the row-level objective exactly (same data term and, because
    scikit-learn scales the penalty by sum(sample_weight) = N, the same penalty), so the fit
    is identical up to solver tolerance. Used for fast hyperparameter search; reported
    models are fitted on the rows.
    """
    current = np.asarray(current, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    counts = np.bincount(current * n_classes + target, minlength=n_classes * n_classes)
    cells = np.flatnonzero(counts)
    return one_hot(cells // n_classes, n_classes), cells % n_classes, counts[cells].astype(float)


@dataclass
class OneHotLogisticChain:
    """F0 logistic regression with the Markov models' sequence interface.

    ``fit(sequences)`` / ``predict_proba(sequences)`` as in ``markov.MarkovChain``: transitions
    are taken within each sequence only; output row t is P(E_{t+1} | E_t) over K classes."""

    n_symbols: int = K
    C: float = 1.0
    max_iter: int = 2000
    tol: float = 1e-6
    aggregate: bool = False
    name: str = "logistic_F0"
    model: MultinomialLogisticModel | None = None

    def fit(self, sequences: list[np.ndarray]) -> "OneHotLogisticChain":
        cur = np.concatenate([s[:-1] for s in sequences if len(s) > 1]).astype(np.int64)
        nxt = np.concatenate([s[1:] for s in sequences if len(s) > 1]).astype(np.int64)
        self.model = MultinomialLogisticModel(self.n_symbols, self.C, self.max_iter, self.tol, name=self.name)
        if self.aggregate:
            x, y, w = aggregate_one_hot(cur, nxt, self.n_symbols)
            self.model.fit(x, y, sample_weight=w)
        else:
            self.model.fit(one_hot(cur, self.n_symbols), nxt)
        return self

    def transition_matrix(self) -> np.ndarray:
        """(K, K): row a = P(E_{t+1} = . | E_t = a)."""
        return self.model.predict_proba(np.eye(self.n_symbols))

    def predict_proba(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        table = self.transition_matrix()
        out = []
        for seq in sequences:
            seq = np.asarray(seq, dtype=np.int64)
            if len(seq) and (seq.min() < 0 or seq.max() >= self.n_symbols):
                raise ValueError(f"symbol outside 0..{self.n_symbols - 1}: unknown category")
            out.append(table[seq])
        return out
