r"""Hidden Markov model for event sequences, with causal next-event prediction.

Observation model (state-conditional independence of the event symbol and
the spatial features):

.. math::

    p(o_t \mid z_t = k) = B_{k, e_t} \prod_{d \in \mathrm{obs}(t)} \mathcal{N}(x_{t,d}; \mu_{k,d}, \sigma^2_{k,d})

- ``n_features = 0``: a categorical HMM (event symbols only).
- ``n_features > 0``: categorical x diagonal-Gaussian emissions. A missing
  feature (NaN: no 360 record, undecided orientation, ...) is marginalised
  out, i.e. it contributes the factor 1. This is exact under the model and
  assumes missingness carries no information about the state beyond what is
  modelled (missing-at-random); the report discusses that assumption.

Event symbols are categories; they are **never** fed to a Gaussian.

Training: EM (Baum-Welch) on independent sequences (one per match period;
sequences are padded and masked, never concatenated), with Dirichlet MAP
pseudo-counts on start, transition and emission rows and a variance floor.
The scaled forward recursion

.. math::

    \hat\alpha_1 \propto \pi \odot b_1, \qquad
    \hat\alpha_t \propto (\hat\alpha_{t-1} A) \odot b_t, \qquad
    c_t = \textstyle\sum_k [\,\cdot\,]_k, \qquad
    \log p(o_{1:T}) = \textstyle\sum_t \log c_t

gives the **filtered** posterior :math:`\hat\alpha_t = p(z_t \mid o_{1:t})`.
Causal one-step-ahead prediction of the next event symbol:

.. math::

    p(e_{t+1} = v \mid o_{1:t}) = \sum_j \Big(\sum_i \hat\alpha_t(i) A_{ij}\Big) B_{j v}

The next step's spatial features are not available at prediction time; they
integrate out of the joint emission (Gaussian densities integrate to one), so
the categorical marginal above is exact for the mixed model too. Smoothed
posteriors and Viterbi paths use future observations and are provided only
for retrospective description (``viterbi``), never for evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOG_2PI = float(np.log(2 * np.pi))


class HMMError(ValueError):
    """Invalid HMM input or parameters."""


@dataclass
class TrainingRecord:
    seed: int
    init_method: str
    n_iter: int = 0
    converged: bool = False
    log_likelihood: float = float("nan")
    penalised_objective: float = float("nan")
    n_observations: int = 0
    monotonic: bool = True
    history: list[float] = field(default_factory=list)


def _pad(symbols: list[np.ndarray], features: list[np.ndarray] | None, n_features: int):
    if not symbols:
        raise HMMError("no sequences")
    lengths = np.array([len(s) for s in symbols])
    if (lengths == 0).any():
        raise HMMError("empty sequence")
    n_seq, t_max = len(symbols), int(lengths.max())
    sym = np.zeros((n_seq, t_max), dtype=np.int64)
    x = np.full((n_seq, t_max, max(n_features, 1)), np.nan)
    mask = np.zeros((n_seq, t_max), dtype=bool)
    for i, s in enumerate(symbols):
        sym[i, : len(s)] = s
        mask[i, : len(s)] = True
        if n_features:
            f = features[i]
            if f.shape != (len(s), n_features):
                raise HMMError(f"feature array {i} has shape {f.shape}, expected {(len(s), n_features)}")
            x[i, : len(s)] = f
    return sym, x[..., :n_features], mask, lengths


@dataclass
class HiddenMarkovModel:
    n_states: int
    n_symbols: int
    n_features: int = 0
    pseudo_count: float = 1.0
    variance_floor: float = 1e-3
    max_iter: int = 300
    tol: float = 1e-6
    seed: int = 0
    init_method: str = "dirichlet_around_marginal"
    init_concentration: float = 20.0
    init_self_transition: float = 0.5
    startprob: np.ndarray | None = None
    transmat: np.ndarray | None = None
    emissionprob: np.ndarray | None = None
    means: np.ndarray | None = None
    variances: np.ndarray | None = None
    record: TrainingRecord | None = None

    # ------------------------------------------------------------------ setup
    def _check_inputs(self, symbols: list[np.ndarray], features: list[np.ndarray] | None) -> None:
        for s in symbols:
            if len(s) and (s.min() < 0 or s.max() >= self.n_symbols):
                raise HMMError(f"symbol outside 0..{self.n_symbols - 1}: unknown category")
        if self.n_features and (features is None or len(features) != len(symbols)):
            raise HMMError("features required: one (T, n_features) array per sequence")
        if not self.n_features and features is not None:
            raise HMMError("this model has no spatial features")

    def initialise(self, symbols: list[np.ndarray], features: list[np.ndarray] | None) -> None:
        if self.init_method != "dirichlet_around_marginal":
            raise HMMError(f"unknown init_method {self.init_method!r}")
        rng = np.random.default_rng(self.seed)
        k, v = self.n_states, self.n_symbols
        counts = np.bincount(np.concatenate(symbols), minlength=v)
        marginal = (counts + 1.0) / (counts.sum() + v)
        self.startprob = rng.dirichlet(np.ones(k))
        w = self.init_self_transition
        self.transmat = w * np.eye(k) + (1 - w) * rng.dirichlet(np.ones(k), size=k)
        self.emissionprob = rng.dirichlet(self.init_concentration * marginal, size=k)
        self.emissionprob = np.clip(self.emissionprob, 1e-12, None)
        self.emissionprob /= self.emissionprob.sum(axis=1, keepdims=True)
        if self.n_features:
            stacked = np.concatenate(features)
            complete = stacked[~np.isnan(stacked).any(axis=1)]
            if len(complete) < k:
                raise HMMError("too few complete spatial observations to initialise the means")
            self.means = complete[rng.choice(len(complete), size=k, replace=False)].copy()
            self.variances = np.tile(np.maximum(np.nanvar(stacked, axis=0), self.variance_floor), (k, 1))

    # ------------------------------------------------------------- likelihood
    def _log_emission(self, sym: np.ndarray, x: np.ndarray) -> np.ndarray:
        """log b_k(o_t) of shape (S, T, K); padded positions are arbitrary."""
        log_b = np.log(self.emissionprob.T)[sym]  # (S, T, K)
        for d in range(self.n_features):
            xd = x[..., d][..., None]
            observed = ~np.isnan(xd)
            ll = -0.5 * (LOG_2PI + np.log(self.variances[:, d]) + (xd - self.means[:, d]) ** 2 / self.variances[:, d])
            log_b = log_b + np.where(observed, ll, 0.0)
        return log_b

    def _forward(self, log_b: np.ndarray, mask: np.ndarray):
        """Scaled forward pass. Returns (alpha_hat, scale c, b scaled, per-sequence log-likelihood)."""
        offsets = log_b.max(axis=2)
        b = np.exp(log_b - offsets[..., None])
        n_seq, t_max, k = b.shape
        alpha = np.empty((n_seq, t_max, k))
        scale = np.ones((n_seq, t_max))
        a = self.startprob * b[:, 0]
        scale[:, 0] = a.sum(axis=1)
        alpha[:, 0] = a / scale[:, 0, None]
        for t in range(1, t_max):
            a = (alpha[:, t - 1] @ self.transmat) * b[:, t]
            valid = mask[:, t]
            s = np.where(valid, a.sum(axis=1), 1.0)
            scale[:, t] = s
            alpha[:, t] = np.where(valid[:, None], a / s[:, None], alpha[:, t - 1])
        if np.any(scale[mask] <= 0) or not np.all(np.isfinite(scale)):
            raise HMMError("numerical underflow in the forward pass")
        loglik = (np.where(mask, np.log(scale) + offsets, 0.0)).sum(axis=1)
        return alpha, scale, b, loglik

    def _backward(self, b: np.ndarray, scale: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n_seq, t_max, k = b.shape
        beta = np.ones((n_seq, t_max, k))
        for t in range(t_max - 2, -1, -1):
            nxt = b[:, t + 1] * beta[:, t + 1] / scale[:, t + 1, None]
            beta[:, t] = np.where(mask[:, t + 1, None], nxt @ self.transmat.T, 1.0)
        return beta

    # --------------------------------------------------------------- training
    def _log_prior(self) -> float:
        pc = self.pseudo_count
        if pc == 0:
            return 0.0
        return pc * float(np.log(self.startprob).sum() + np.log(self.transmat).sum() + np.log(self.emissionprob).sum())

    def fit(self, symbols: list[np.ndarray], features: list[np.ndarray] | None = None) -> "HiddenMarkovModel":
        self._check_inputs(symbols, features)
        self.initialise(symbols, features)
        sym, x, mask, _ = _pad(symbols, features, self.n_features)
        n_obs = int(mask.sum())
        onehot = np.eye(self.n_symbols, dtype=float)[sym] * mask[..., None]  # (S, T, V)
        record = TrainingRecord(seed=self.seed, init_method=self.init_method, n_observations=n_obs)
        previous = -np.inf
        pc = self.pseudo_count
        for iteration in range(1, self.max_iter + 1):
            alpha, scale, b, loglik = self._forward(self._log_emission(sym, x), mask)
            beta = self._backward(b, scale, mask)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=2, keepdims=True)
            gamma *= mask[..., None]
            objective = float(loglik.sum()) + self._log_prior()
            record.history.append(objective)
            if objective < previous - 1e-8 * abs(previous):
                record.monotonic = False
            improvement = (objective - previous) / n_obs
            # M-step
            right = b[:, 1:] * beta[:, 1:] / scale[:, 1:, None] * mask[:, 1:, None]
            xi = np.einsum("stk,stl->kl", alpha[:, :-1], right) * self.transmat
            start = gamma[:, 0].sum(axis=0) + pc
            trans = xi + pc
            emis = np.einsum("stk,stv->kv", gamma, onehot) + pc
            self.startprob = start / start.sum()
            self.transmat = trans / trans.sum(axis=1, keepdims=True)
            self.emissionprob = emis / emis.sum(axis=1, keepdims=True)
            for d in range(self.n_features):
                xd = x[..., d]
                observed = (~np.isnan(xd)) & mask
                w = gamma * observed[..., None]
                total = w.sum(axis=(0, 1))
                xd0 = np.where(observed, xd, 0.0)[..., None]
                mean = np.where(total > 0, (w * xd0).sum(axis=(0, 1)) / np.maximum(total, 1e-300), self.means[:, d])
                var = np.where(total > 0, (w * (xd0 - mean) ** 2).sum(axis=(0, 1)) / np.maximum(total, 1e-300),
                               self.variances[:, d])
                self.means[:, d] = mean
                self.variances[:, d] = np.maximum(var, self.variance_floor)
            record.n_iter = iteration
            if iteration > 1 and abs(improvement) < self.tol:
                record.converged = True
                break
            previous = objective
        # final likelihood under the final parameters
        _, _, _, loglik = self._forward(self._log_emission(sym, x), mask)
        record.log_likelihood = float(loglik.sum())
        record.penalised_objective = record.log_likelihood + self._log_prior()
        self.record = record
        return self

    # ------------------------------------------------------------- inference
    def score(self, symbols: list[np.ndarray], features: list[np.ndarray] | None = None) -> float:
        self._check_inputs(symbols, features)
        sym, x, mask, _ = _pad(symbols, features, self.n_features)
        return float(self._forward(self._log_emission(sym, x), mask)[3].sum())

    def filter(self, symbols: list[np.ndarray], features: list[np.ndarray] | None = None) -> list[np.ndarray]:
        """Filtered state posteriors p(z_t | o_1..t), one (T, K) array per sequence."""
        self._check_inputs(symbols, features)
        sym, x, mask, lengths = _pad(symbols, features, self.n_features)
        alpha = self._forward(self._log_emission(sym, x), mask)[0]
        return [alpha[i, :n] for i, n in enumerate(lengths)]

    def predict_proba(self, symbols: list[np.ndarray], features: list[np.ndarray] | None = None) -> list[np.ndarray]:
        """p(e_{t+1} = v | o_1..t) for every t: one (T, V) array per sequence."""
        return [(f @ self.transmat) @ self.emissionprob for f in self.filter(symbols, features)]

    def viterbi(self, symbols: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        """Most likely state path of ONE whole sequence (uses future observations:
        retrospective description only, never for evaluation)."""
        sym, x, mask, _ = _pad([symbols], None if features is None else [features], self.n_features)
        log_b = self._log_emission(sym, x)[0]
        log_a = np.log(self.transmat)
        t_max = len(symbols)
        delta = np.log(self.startprob) + log_b[0]
        back = np.zeros((t_max, self.n_states), dtype=np.int64)
        for t in range(1, t_max):
            cand = delta[:, None] + log_a
            back[t] = cand.argmax(axis=0)
            delta = cand.max(axis=0) + log_b[t]
        path = np.zeros(t_max, dtype=np.int64)
        path[-1] = int(delta.argmax())
        for t in range(t_max - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        def arr(a):
            return None if a is None else np.asarray(a).tolist()
        return {
            "n_states": self.n_states, "n_symbols": self.n_symbols, "n_features": self.n_features,
            "pseudo_count": self.pseudo_count, "variance_floor": self.variance_floor, "max_iter": self.max_iter,
            "tol": self.tol, "seed": self.seed, "init_method": self.init_method,
            "init_concentration": self.init_concentration, "init_self_transition": self.init_self_transition,
            "startprob": arr(self.startprob), "transmat": arr(self.transmat), "emissionprob": arr(self.emissionprob),
            "means": arr(self.means), "variances": arr(self.variances),
            "record": None if self.record is None else self.record.__dict__,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HiddenMarkovModel":
        def arr(a):
            return None if a is None else np.array(a, dtype=float)
        model = cls(**{k: d[k] for k in (
            "n_states", "n_symbols", "n_features", "pseudo_count", "variance_floor", "max_iter", "tol", "seed",
            "init_method", "init_concentration", "init_self_transition")})
        model.startprob, model.transmat, model.emissionprob = arr(d["startprob"]), arr(d["transmat"]), arr(d["emissionprob"])
        model.means, model.variances = arr(d["means"]), arr(d["variances"])
        model.record = None if d.get("record") is None else TrainingRecord(**d["record"])
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HiddenMarkovModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
