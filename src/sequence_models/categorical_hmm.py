r"""Categorical HMM with an optional categorical pitch-zone observation.

Why this redesign. The first experiment's spatial HMM combined a categorical
event emission with Gaussian densities over continuous coordinates. A density
is not a probability: its scale depends on the units, and the two continuous
dimensions contributed about -3.0 nats per observation against -0.7 for the
event part, so EM spent the hidden states on location and the next-event
predictions collapsed to the marginal. Here every observation component is a
*probability* of a discrete outcome, and the model is an ordinary generative
HMM. No term is re-weighted.

Emission structures (``emission``):

- ``"event"``      : :math:`p(o_t \mid z_t=k) = B^E_{k,e_t}`
- ``"factorized"`` : :math:`B^E_{k,e_t}\,B^Z_{k,g_t}`. Event and zone are
  conditionally independent given the state (an explicit assumption).
- ``"joint"``      : :math:`B^{EZ}_{k,e_t,g_t}`, one categorical over
  (event, zone) pairs (no independence assumption, K(VZ-1) parameters).

A missing zone (``-1``) is marginalised exactly: the factorized model drops
:math:`B^Z`; the joint model uses :math:`\sum_g B^{EZ}_{k,e,g}`, and its M-step
distributes the expected counts over zones in proportion to :math:`B^{EZ}`.

Causal prediction of the next EVENT (the zone at t+1 is unknown and
marginalised):

.. math::

    p(e_{t+1}=v \mid o_{1:t}) = \sum_j \Big(\sum_i \hat\alpha_t(i)\,A_{ij}\Big)\,M_{jv},\quad
    M = B^E \text{ or } \textstyle\sum_g B^{EZ}_{\cdot,\cdot,g}

with the filtered posterior :math:`\hat\alpha_t = p(z_t \mid o_{1:t})`.

Training: Baum-Welch on independent padded/masked sequences (never
concatenated), Dirichlet MAP pseudo-counts on every row, two documented
initialisations (``init_self_transition`` 0.5 = persistence-biased as in
experiment 1, 0.0 = no persistence bias). With ``emission="event"`` and
``init_self_transition=0.5`` the initialisation draws are identical to
``hmm.HiddenMarkovModel`` (tested), so experiment 1 is a special case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EMISSIONS = ("event", "factorized", "joint")


class CategoricalHMMError(ValueError):
    """Invalid input or parameters."""


@dataclass
class FitRecord:
    seed: int
    init_self_transition: float
    n_iter: int = 0
    converged: bool = False
    log_likelihood: float = float("nan")
    penalised_objective: float = float("nan")
    n_observations: int = 0
    monotonic: bool = True
    history: list[float] = field(default_factory=list)


def _pad(arrays: list[np.ndarray], fill: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.array([len(a) for a in arrays])
    if len(arrays) == 0 or (lengths == 0).any():
        raise CategoricalHMMError("empty input or empty sequence")
    out = np.full((len(arrays), int(lengths.max())), fill, dtype=np.int64)
    mask = np.zeros(out.shape, dtype=bool)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
        mask[i, : len(a)] = True
    return out, mask, lengths


@dataclass
class CategoricalHMM:
    n_states: int
    n_events: int
    emission: str = "event"
    n_zones: int = 0
    pseudo_count: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-6
    seed: int = 0
    init_concentration: float = 20.0
    init_self_transition: float = 0.5
    startprob: np.ndarray | None = None
    transmat: np.ndarray | None = None
    event_emission: np.ndarray | None = None   # (K, V)  event / factorized
    zone_emission: np.ndarray | None = None    # (K, Z)  factorized
    joint_emission: np.ndarray | None = None   # (K, V, Z) joint
    record: FitRecord | None = None

    def __post_init__(self) -> None:
        if self.emission not in EMISSIONS:
            raise CategoricalHMMError(f"emission must be one of {EMISSIONS}")
        if self.emission != "event" and self.n_zones < 2:
            raise CategoricalHMMError("zone emissions need n_zones >= 2")

    # ----------------------------------------------------------------- checks
    def _check(self, events: list[np.ndarray], zones: list[np.ndarray] | None) -> None:
        for e in events:
            if len(e) and (e.min() < 0 or e.max() >= self.n_events):
                raise CategoricalHMMError(f"event symbol outside 0..{self.n_events - 1}: unknown category")
        if self.emission == "event":
            if zones is not None:
                raise CategoricalHMMError("event-only model takes no zones")
            return
        if zones is None or len(zones) != len(events):
            raise CategoricalHMMError("zones required: one array per sequence")
        for e, z in zip(events, zones):
            if len(z) != len(e):
                raise CategoricalHMMError("zone and event arrays differ in length")
            if len(z) and (z.min() < -1 or z.max() >= self.n_zones):
                raise CategoricalHMMError("zone outside -1..n_zones-1")

    def event_marginal_emission(self) -> np.ndarray:
        """M[k, v] = P(event = v | state k), zone marginalised."""
        return self.joint_emission.sum(axis=2) if self.emission == "joint" else self.event_emission

    # ---------------------------------------------------------- initialise
    def initialise(self, events: list[np.ndarray], zones: list[np.ndarray] | None) -> None:
        rng = np.random.default_rng(self.seed)
        k, v = self.n_states, self.n_events
        ev = np.concatenate(events)
        marginal = (np.bincount(ev, minlength=v) + 1.0) / (len(ev) + v)
        self.startprob = rng.dirichlet(np.ones(k))
        w = self.init_self_transition
        self.transmat = w * np.eye(k) + (1 - w) * rng.dirichlet(np.ones(k), size=k)
        if self.emission in ("event", "factorized"):
            b = np.clip(rng.dirichlet(self.init_concentration * marginal, size=k), 1e-12, None)
            self.event_emission = b / b.sum(axis=1, keepdims=True)
        if self.emission == "factorized":
            z = np.concatenate(zones)
            z = z[z >= 0]
            zm = (np.bincount(z, minlength=self.n_zones) + 1.0) / (len(z) + self.n_zones)
            b = np.clip(rng.dirichlet(self.init_concentration * zm, size=k), 1e-12, None)
            self.zone_emission = b / b.sum(axis=1, keepdims=True)
        if self.emission == "joint":
            z = np.concatenate(zones)
            obs = z >= 0
            pairs = np.bincount(ev[obs] * self.n_zones + z[obs], minlength=v * self.n_zones)
            jm = (pairs + 1.0) / (pairs.sum() + v * self.n_zones)
            b = np.clip(rng.dirichlet(self.init_concentration * jm, size=k), 1e-12, None)
            b /= b.sum(axis=1, keepdims=True)
            self.joint_emission = b.reshape(k, v, self.n_zones)

    # ---------------------------------------------------------- likelihood
    def _log_emission(self, ev: np.ndarray, zn: np.ndarray | None) -> np.ndarray:
        if self.emission == "event":
            return np.log(self.event_emission.T)[ev]
        if self.emission == "factorized":
            log_b = np.log(self.event_emission.T)[ev]
            log_z = np.log(self.zone_emission.T)[np.maximum(zn, 0)]
            return log_b + np.where((zn >= 0)[..., None], log_z, 0.0)
        log_joint = np.log(self.joint_emission)                         # (K, V, Z)
        log_marg = np.log(self.joint_emission.sum(axis=2))               # (K, V)
        observed = np.moveaxis(log_joint, 0, -1)[ev, np.maximum(zn, 0)]  # (S, T, K)
        return np.where((zn >= 0)[..., None], observed, log_marg.T[ev])

    def _forward(self, log_b: np.ndarray, mask: np.ndarray):
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
            raise CategoricalHMMError("numerical underflow in the forward pass")
        loglik = np.where(mask, np.log(scale) + offsets, 0.0).sum(axis=1)
        return alpha, scale, b, loglik

    def _backward(self, b: np.ndarray, scale: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n_seq, t_max, k = b.shape
        beta = np.ones((n_seq, t_max, k))
        for t in range(t_max - 2, -1, -1):
            nxt = b[:, t + 1] * beta[:, t + 1] / scale[:, t + 1, None]
            beta[:, t] = np.where(mask[:, t + 1, None], nxt @ self.transmat.T, 1.0)
        return beta

    def _log_prior(self) -> float:
        pc = self.pseudo_count
        if pc == 0:
            return 0.0
        total = np.log(self.startprob).sum() + np.log(self.transmat).sum()
        for b in (self.event_emission, self.zone_emission, self.joint_emission):
            if b is not None:
                total += np.log(b).sum()
        return pc * float(total)

    def _inputs(self, events, zones):
        ev, mask, lengths = _pad(events, 0)
        zn = _pad(zones, -1)[0] if self.emission != "event" else None
        return ev, zn, mask, lengths

    # ---------------------------------------------------------------- EM
    def fit(self, events: list[np.ndarray], zones: list[np.ndarray] | None = None) -> "CategoricalHMM":
        self._check(events, zones)
        self.initialise(events, zones)
        ev, zn, mask, _ = self._inputs(events, zones)
        flat = mask.ravel()
        ev_flat = ev.ravel()[flat]
        zn_flat = zn.ravel()[flat] if zn is not None else None
        n_obs = int(flat.sum())
        k, v, pc = self.n_states, self.n_events, self.pseudo_count
        record = FitRecord(seed=self.seed, init_self_transition=self.init_self_transition, n_observations=n_obs)
        previous = -np.inf
        for iteration in range(1, self.max_iter + 1):
            alpha, scale, b, loglik = self._forward(self._log_emission(ev, zn), mask)
            beta = self._backward(b, scale, mask)
            objective = float(loglik.sum()) + self._log_prior()
            record.history.append(objective)
            if objective < previous - 1e-8 * abs(previous):
                record.monotonic = False
            improvement = (objective - previous) / n_obs
            right = b[:, 1:] * beta[:, 1:] / scale[:, 1:, None] * mask[:, 1:, None]
            xi = np.einsum("stk,stl->kl", alpha[:, :-1], right) * self.transmat
            del right
            gamma = alpha * beta
            del alpha, beta, b
            gamma /= gamma.sum(axis=2, keepdims=True)
            g = gamma.reshape(-1, k)[flat]
            del gamma
            start = self._first_gamma(g, mask) + pc
            self.startprob = start / start.sum()
            trans = xi + pc
            self.transmat = trans / trans.sum(axis=1, keepdims=True)
            self._m_step_emissions(g, ev_flat, zn_flat, pc)
            record.n_iter = iteration
            if iteration > 1 and abs(improvement) < self.tol:
                record.converged = True
                break
            previous = objective
        record.log_likelihood = float(self._forward(self._log_emission(ev, zn), mask)[3].sum())
        record.penalised_objective = record.log_likelihood + self._log_prior()
        self.record = record
        return self

    @staticmethod
    def _first_gamma(g: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Sum of the posteriors of each sequence's first observation."""
        starts = np.concatenate([[0], np.cumsum(mask.sum(axis=1))[:-1]])
        return g[starts].sum(axis=0)

    def _m_step_emissions(self, g: np.ndarray, ev: np.ndarray, zn: np.ndarray | None, pc: float) -> None:
        k, v = self.n_states, self.n_events
        if self.emission in ("event", "factorized"):
            counts = np.stack([np.bincount(ev, weights=g[:, s], minlength=v) for s in range(k)]) + pc
            self.event_emission = counts / counts.sum(axis=1, keepdims=True)
        if self.emission == "factorized":
            obs = zn >= 0
            counts = np.stack([np.bincount(zn[obs], weights=g[obs, s], minlength=self.n_zones) for s in range(k)]) + pc
            self.zone_emission = counts / counts.sum(axis=1, keepdims=True)
        if self.emission == "joint":
            z_count = self.n_zones
            obs = zn >= 0
            counts = np.stack([np.bincount(ev[obs] * z_count + zn[obs], weights=g[obs, s], minlength=v * z_count)
                               for s in range(k)]).reshape(k, v, z_count)
            if (~obs).any():  # missing zone: expected counts in proportion to B(e, g | k)
                share = self.joint_emission / self.joint_emission.sum(axis=2, keepdims=True)  # (K, V, Z)
                miss_e, miss_g = ev[~obs], g[~obs]
                for s in range(k):
                    per_event = np.bincount(miss_e, weights=miss_g[:, s], minlength=v)
                    counts[s] += per_event[:, None] * share[s]
            counts = counts + pc
            self.joint_emission = counts / counts.sum(axis=(1, 2), keepdims=True)

    # ------------------------------------------------------------ inference
    def score(self, events: list[np.ndarray], zones: list[np.ndarray] | None = None) -> float:
        self._check(events, zones)
        ev, zn, mask, _ = self._inputs(events, zones)
        return float(self._forward(self._log_emission(ev, zn), mask)[3].sum())

    def filter(self, events: list[np.ndarray], zones: list[np.ndarray] | None = None) -> list[np.ndarray]:
        """Filtered posteriors p(z_t | o_1..t), one (T, K) array per sequence."""
        self._check(events, zones)
        ev, zn, mask, lengths = self._inputs(events, zones)
        alpha = self._forward(self._log_emission(ev, zn), mask)[0]
        return [alpha[i, :n] for i, n in enumerate(lengths)]

    def predict_proba(self, events: list[np.ndarray], zones: list[np.ndarray] | None = None) -> list[np.ndarray]:
        """p(e_{t+1} = v | o_1..t): one (T, V) array per sequence."""
        m = self.event_marginal_emission()
        return [(f @ self.transmat) @ m for f in self.filter(events, zones)]

    def viterbi(self, events: np.ndarray, zones: np.ndarray | None = None) -> np.ndarray:
        """Retrospective most likely path of ONE sequence (description only)."""
        ev, zn, mask, _ = self._inputs([events], None if zones is None else [zones])
        log_b = self._log_emission(ev, zn)[0]
        log_a = np.log(self.transmat)
        delta = np.log(self.startprob) + log_b[0]
        back = np.zeros((len(events), self.n_states), dtype=np.int64)
        for t in range(1, len(events)):
            cand = delta[:, None] + log_a
            back[t] = cand.argmax(axis=0)
            delta = cand.max(axis=0) + log_b[t]
        path = np.zeros(len(events), dtype=np.int64)
        path[-1] = int(delta.argmax())
        for t in range(len(events) - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def n_free_parameters(self) -> int:
        k, v, z = self.n_states, self.n_events, self.n_zones
        n = (k - 1) + k * (k - 1)
        if self.emission == "event":
            return n + k * (v - 1)
        if self.emission == "factorized":
            return n + k * (v - 1) + k * (z - 1)
        return n + k * (v * z - 1)

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        def arr(a):
            return None if a is None else np.asarray(a).tolist()
        return {
            "n_states": self.n_states, "n_events": self.n_events, "emission": self.emission, "n_zones": self.n_zones,
            "pseudo_count": self.pseudo_count, "max_iter": self.max_iter, "tol": self.tol, "seed": self.seed,
            "init_concentration": self.init_concentration, "init_self_transition": self.init_self_transition,
            "startprob": arr(self.startprob), "transmat": arr(self.transmat),
            "event_emission": arr(self.event_emission), "zone_emission": arr(self.zone_emission),
            "joint_emission": arr(self.joint_emission),
            "record": None if self.record is None else self.record.__dict__,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CategoricalHMM":
        def arr(a):
            return None if a is None else np.array(a, dtype=float)
        model = cls(**{k: d[k] for k in ("n_states", "n_events", "emission", "n_zones", "pseudo_count", "max_iter",
                                         "tol", "seed", "init_concentration", "init_self_transition")})
        model.startprob, model.transmat = arr(d["startprob"]), arr(d["transmat"])
        model.event_emission, model.zone_emission = arr(d["event_emission"]), arr(d["zone_emission"])
        model.joint_emission = arr(d["joint_emission"])
        model.record = None if d.get("record") is None else FitRecord(**d["record"])
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CategoricalHMM":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
