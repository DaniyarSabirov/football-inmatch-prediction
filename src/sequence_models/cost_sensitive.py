r"""Cost-sensitive decisions on top of unchanged probabilistic predictions.

**Approach A (implemented): Bayes-optimal decisions.** The model's predictive
distribution :math:`p_k = P(Y = k \mid x)` is left untouched. The predicted
class minimises the expected misclassification cost (Elkan 2001):

.. math::

    \hat y(x) = \arg\min_a R(a \mid x),\qquad R(a \mid x) = \sum_k C(a, k)\, p_k

with cost matrix :math:`C(a, k)` = cost of predicting :math:`a` when the true
class is :math:`k`. Ties (within ``tie_tolerance``) go to the lowest class
index, i.e. the vocabulary order. Probability metrics (log loss, Brier,
calibration) are computed from :math:`p` and so are unchanged by the decision
rule.

**Cost family.** For :math:`a \ne k`

.. math::

    C(a, k) = m_k + f_a, \qquad C(k, k) = 0,\qquad
    m_k = \min\big(\text{cap},\ (\pi_{\max} / \pi_k)^{\gamma}\big)

- :math:`m_k`: cost of MISSING a true :math:`k` (false negative for :math:`k`);
- :math:`f_a`: extra cost of a FALSE ALARM for :math:`a` (false positive);
- :math:`\pi_k`: class prior estimated from TRAINING-fold targets only
  (Laplace-smoothed);
- :math:`\gamma = 0` gives the uniform 0-1 cost, i.e. argmax.

Unit: one cost unit is the cost of missing the most frequent class
(:math:`m_{\max\pi} = 1`). With :math:`f = 0` the rule reduces to
:math:`\arg\max_a m_a p_a`, so class :math:`r` is preferred to :math:`j` iff
:math:`p_r / p_j > m_j / m_r`.

**Approach B (cost-sensitive training) is not used.** Training a Markov chain
on class-weighted counts :math:`w_k C_{ik}` estimates the prior-shifted
distribution :math:`q_k \propto w_k p_k` (``prior_shifted``). Its argmax is
exactly Approach A's decision with :math:`m = w, f = 0`, but :math:`q` is
miscalibrated for the real class frequencies. Weighting the HMM likelihood
has no generative interpretation. See methodology.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostScheme:
    name: str
    gamma: float
    cap: float
    false_alarm: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.gamma < 0 or self.cap < 1 or self.false_alarm < 0:
            raise ValueError(f"{self.name}: need gamma >= 0, cap >= 1, false_alarm >= 0")


def training_priors(train_targets: np.ndarray, n_classes: int) -> np.ndarray:
    """Laplace-smoothed class priors from TRAINING targets only."""
    counts = np.bincount(train_targets, minlength=n_classes).astype(float)
    return (counts + 1.0) / (counts.sum() + n_classes)


def miss_costs(priors: np.ndarray, scheme: CostScheme) -> np.ndarray:
    return np.minimum(scheme.cap, (priors.max() / priors) ** scheme.gamma)


def cost_matrix(priors: np.ndarray, scheme: CostScheme) -> np.ndarray:
    """C[a, k] (rows: predicted class a, columns: true class k)."""
    m = miss_costs(priors, scheme)
    v = len(priors)
    c = m[None, :] + scheme.false_alarm * np.ones((v, 1))
    np.fill_diagonal(c, 0.0)
    return c


def validate_cost_matrix(c: np.ndarray) -> None:
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cost matrix must be square")
    if np.any(c < 0) or not np.all(np.isfinite(c)):
        raise ValueError("costs must be finite and non-negative")
    if np.any(np.diag(c) != 0):
        raise ValueError("correct predictions must cost 0")


def expected_costs(p: np.ndarray, c: np.ndarray) -> np.ndarray:
    """R[i, a] = sum_k C[a, k] p[i, k]."""
    return p @ c.T


def bayes_decisions(p: np.ndarray, c: np.ndarray, tie_tolerance: float = 1e-12) -> np.ndarray:
    """argmin_a R(a); ties within tolerance -> lowest class index (deterministic)."""
    validate_cost_matrix(c)
    r = expected_costs(p, c)
    best = r.min(axis=1, keepdims=True)
    return np.argmax(r <= best + tie_tolerance * np.maximum(1.0, np.abs(best)), axis=1)


def realised_cost(y: np.ndarray, decisions: np.ndarray, c: np.ndarray) -> float:
    return float(c[decisions, y].mean())


def expected_cost_of_decisions(p: np.ndarray, decisions: np.ndarray, c: np.ndarray) -> float:
    """Model-implied expected cost of the chosen decisions (uses p, not labels)."""
    return float(expected_costs(p, c)[np.arange(len(p)), decisions].mean())


def prior_shifted(p: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """q_k = w_k p_k / sum_j w_j p_j: what class-weighted training estimates (Approach B)."""
    q = p * weights[None, :]
    return q / q.sum(axis=1, keepdims=True)
