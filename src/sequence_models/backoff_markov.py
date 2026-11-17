r"""Discrete Markov models with nested contexts and hierarchical Dirichlet smoothing.

A model is a chain of nested context levels, most specific first, for example
the pitch-third-conditioned second-order chain::

    (prev, cur, zone) -> (cur, zone) -> (cur,) -> marginal

Each level's distribution is a Dirichlet-smoothed estimate centred on its
parent level (interpolated backoff; MacKay & Peto 1995):

.. math::

    p_0(k) = \frac{N_k + \alpha_0}{N + V\alpha_0},\qquad
    p_\ell(k \mid c) = \frac{C_\ell(c, k) + \beta\, p_{\ell-1}(k \mid \pi(c))}{N_\ell(c) + \beta}

where :math:`\pi(c)` drops the extra context feature. A context seen often
is dominated by its own counts. A rare or unseen context falls back smoothly
to its parent, so sparse combinations never get unsupported probabilities.
The concentration :math:`\beta` is chosen by nested cross-validation (log
loss on inner folds of the training data only).

Context features (all known at prediction time, i.e. right after event t):

- ``cur``  : symbol of event t (0..V-1)
- ``prev`` : symbol of event t-1 in the same sequence, or START (= V) at the
  first event of a period
- ``zone`` : pitch third of event t (0..2), or UNKNOWN (= 3) without a location
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MODEL_LEVELS = {
    "mc1": (("cur",),),
    "mc2": (("prev", "cur"), ("cur",)),
    "mc1_zone": (("cur", "zone"), ("cur",)),
    "mc2_zone": (("prev", "cur", "zone"), ("cur", "zone"), ("cur",)),
}


def context_cardinalities(n_symbols: int, n_zones: int) -> dict[str, int]:
    return {"cur": n_symbols, "prev": n_symbols + 1, "zone": n_zones + 1}


def build_contexts(symbols: list[np.ndarray], zones: list[np.ndarray] | None, n_symbols: int,
                   n_zones: int) -> dict[str, np.ndarray]:
    """Context features for every observation of every sequence (concatenated)."""
    cur = np.concatenate(symbols)
    prev = np.concatenate([np.concatenate([[n_symbols], s[:-1]]) for s in symbols])
    out = {"cur": cur, "prev": prev}
    if zones is not None:
        z = np.concatenate(zones)
        out["zone"] = np.where(z < 0, n_zones, z)
    return out


def targets_of(symbols: list[np.ndarray]) -> np.ndarray:
    """Next symbol in the same sequence; -1 at the last observation."""
    return np.concatenate([np.concatenate([s[1:], [-1]]) for s in symbols])


@dataclass
class BackoffMarkov:
    name: str
    n_symbols: int
    cardinalities: dict[str, int]
    concentration: float = 10.0
    base_alpha: float = 1.0
    levels: tuple[tuple[str, ...], ...] = ()
    base_counts: np.ndarray | None = None
    level_counts: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.levels:
            self.levels = MODEL_LEVELS[self.name]
        for more, less in zip(self.levels, self.levels[1:]):
            if not set(less) < set(more):
                raise ValueError("context levels must be nested, most specific first")

    def _key(self, level: tuple[str, ...], contexts: dict[str, np.ndarray]) -> np.ndarray:
        key = np.zeros(len(contexts["cur"]), dtype=np.int64)
        for feature in level:
            values = contexts[feature]
            if values.min() < 0 or values.max() >= self.cardinalities[feature]:
                raise ValueError(f"context feature {feature} outside its range: unknown category")
            key = key * self.cardinalities[feature] + values
        return key

    def _size(self, level: tuple[str, ...]) -> int:
        return int(np.prod([self.cardinalities[f] for f in level]))

    def fit(self, contexts: dict[str, np.ndarray], targets: np.ndarray) -> "BackoffMarkov":
        has = targets >= 0
        if targets[has].max(initial=0) >= self.n_symbols:
            raise ValueError("target outside the vocabulary")
        v = self.n_symbols
        self.base_counts = np.bincount(targets[has], minlength=v).astype(float)
        self.level_counts = []
        for level in self.levels:
            key = self._key(level, {f: c[has] for f, c in contexts.items()})
            counts = np.bincount(key * v + targets[has], minlength=self._size(level) * v).astype(float)
            self.level_counts.append(counts.reshape(self._size(level), v))
        return self

    def predict_proba(self, contexts: dict[str, np.ndarray]) -> np.ndarray:
        v, beta = self.n_symbols, self.concentration
        base = (self.base_counts + self.base_alpha) / (self.base_counts.sum() + v * self.base_alpha)
        p = np.tile(base, (len(contexts["cur"]), 1))
        for level, counts in zip(reversed(self.levels), reversed(self.level_counts)):
            c = counts[self._key(level, contexts)]
            p = (c + beta * p) / (c.sum(axis=1, keepdims=True) + beta)
        return p

    def context_support(self, contexts: dict[str, np.ndarray]) -> np.ndarray:
        """Training count of the most specific context of each row."""
        return self.level_counts[0][self._key(self.levels[0], contexts)].sum(axis=1)

    def to_dict(self) -> dict:
        return {"name": self.name, "n_symbols": self.n_symbols, "cardinalities": self.cardinalities,
                "concentration": self.concentration, "base_alpha": self.base_alpha,
                "levels": [list(level) for level in self.levels], "base_counts": self.base_counts.tolist(),
                "level_counts": [c.tolist() for c in self.level_counts]}

    @classmethod
    def from_dict(cls, d: dict) -> "BackoffMarkov":
        model = cls(name=d["name"], n_symbols=d["n_symbols"], cardinalities=d["cardinalities"],
                    concentration=d["concentration"], base_alpha=d["base_alpha"],
                    levels=tuple(tuple(level) for level in d["levels"]))
        model.base_counts = np.array(d["base_counts"])
        model.level_counts = [np.array(c) for c in d["level_counts"]]
        return model
