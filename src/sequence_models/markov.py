"""Empirical-frequency baseline and first-/second-order discrete Markov chains.

All models share one interface:

- ``fit(symbols_per_sequence)`` with a list of int arrays (one per sequence;
  transitions are counted only *within* a sequence, never across);
- ``predict_proba(symbols_per_sequence)`` returning, for every observation t,
  the full distribution over the V target classes for observation t+1 given
  observations up to t (an array of shape (n_t, V), rows summing to one).

Smoothing is additive: P(j | i) = (C[i, j] + alpha) / (sum_j C[i, j] + V * alpha).
With alpha > 0 every probability is strictly positive, so the log loss is
finite. A state never seen as a transition source in training ("unseen
state") is predicted with the frequency baseline and reported via
``unseen_state_mask``. Symbols outside 0..V-1 raise ``ValueError``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _check_symbols(sequences: list[np.ndarray], n_symbols: int) -> None:
    for seq in sequences:
        if len(seq) and (seq.min() < 0 or seq.max() >= n_symbols):
            raise ValueError(f"symbol outside 0..{n_symbols - 1}: unknown category")


def transition_counts(sequences: list[np.ndarray], n_symbols: int) -> np.ndarray:
    """C[i, j] = number of within-sequence transitions i -> j."""
    _check_symbols(sequences, n_symbols)
    counts = np.zeros((n_symbols, n_symbols), dtype=np.int64)
    for seq in sequences:
        if len(seq) > 1:
            np.add.at(counts, (seq[:-1], seq[1:]), 1)
    return counts


def normalise_rows(counts: np.ndarray, alpha: float) -> np.ndarray:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    smoothed = counts.astype(float) + alpha
    totals = smoothed.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = smoothed / totals
    return probabilities  # rows with zero total (alpha = 0, unseen) are NaN by design


@dataclass
class FrequencyBaseline:
    """P(next = j) = smoothed training frequency of j as a transition *target*."""

    n_symbols: int
    alpha: float = 1.0
    counts: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    name: str = "frequency_baseline"

    def fit(self, sequences: list[np.ndarray]) -> "FrequencyBaseline":
        _check_symbols(sequences, self.n_symbols)
        targets = [seq[1:] for seq in sequences]
        self.counts = np.bincount(np.concatenate(targets) if targets else np.array([], int), minlength=self.n_symbols)
        self.probabilities = (self.counts + self.alpha) / (self.counts.sum() + self.n_symbols * self.alpha)
        return self

    def predict_proba(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        _check_symbols(sequences, self.n_symbols)
        return [np.tile(self.probabilities, (len(seq), 1)) for seq in sequences]

    def to_dict(self) -> dict:
        return {"name": self.name, "n_symbols": self.n_symbols, "alpha": self.alpha,
                "counts": self.counts.tolist(), "probabilities": self.probabilities.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FrequencyBaseline":
        return cls(n_symbols=d["n_symbols"], alpha=d["alpha"], counts=np.array(d["counts"]),
                   probabilities=np.array(d["probabilities"]))


@dataclass
class MarkovChain:
    """First-order Markov chain P(o_{t+1} | o_t)."""

    n_symbols: int
    alpha: float = 1.0
    counts: np.ndarray | None = None
    transition_matrix: np.ndarray | None = None
    fallback: FrequencyBaseline | None = None
    name: str = "markov_order1"

    def fit(self, sequences: list[np.ndarray]) -> "MarkovChain":
        self.counts = transition_counts(sequences, self.n_symbols)
        self.transition_matrix = normalise_rows(self.counts, self.alpha)
        self.fallback = FrequencyBaseline(self.n_symbols, self.alpha).fit(sequences)
        return self

    def unseen_states(self) -> np.ndarray:
        return self.counts.sum(axis=1) == 0

    def unseen_state_mask(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        unseen = self.unseen_states()
        return [unseen[seq] for seq in sequences]

    def predict_proba(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        _check_symbols(sequences, self.n_symbols)
        unseen = self.unseen_states()
        out = []
        for seq in sequences:
            probabilities = self.transition_matrix[seq].copy()
            probabilities[unseen[seq]] = self.fallback.probabilities
            out.append(probabilities)
        return out

    def to_dict(self) -> dict:
        return {"name": self.name, "n_symbols": self.n_symbols, "alpha": self.alpha,
                "counts": self.counts.tolist(), "transition_matrix": self.transition_matrix.tolist(),
                "fallback": self.fallback.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "MarkovChain":
        return cls(n_symbols=d["n_symbols"], alpha=d["alpha"], counts=np.array(d["counts"]),
                   transition_matrix=np.array(d["transition_matrix"]),
                   fallback=FrequencyBaseline.from_dict(d["fallback"]))


@dataclass
class SecondOrderMarkovChain:
    """P(o_{t+1} | o_{t-1}, o_t), backing off to first order when the context
    (o_{t-1}, o_t) has fewer than ``min_context_count`` training occurrences,
    and at the first observation of a sequence (no o_{t-1})."""

    n_symbols: int
    alpha: float = 1.0
    min_context_count: int = 20
    counts: np.ndarray | None = None
    first_order: MarkovChain | None = None
    name: str = "markov_order2"
    backoff_rate: dict = field(default_factory=dict)

    def fit(self, sequences: list[np.ndarray]) -> "SecondOrderMarkovChain":
        _check_symbols(sequences, self.n_symbols)
        v = self.n_symbols
        self.counts = np.zeros((v, v, v), dtype=np.int64)
        for seq in sequences:
            if len(seq) > 2:
                np.add.at(self.counts, (seq[:-2], seq[1:-1], seq[2:]), 1)
        self.first_order = MarkovChain(v, self.alpha).fit(sequences)
        return self

    def predict_proba(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        _check_symbols(sequences, self.n_symbols)
        context_totals = self.counts.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):  # unseen contexts (alpha = 0) are never used: backoff
            smoothed = (self.counts + self.alpha) / (context_totals[..., None] + self.n_symbols * self.alpha)
        first = self.first_order.predict_proba(sequences)
        out, backed_off, total = [], 0, 0
        for seq, p1 in zip(sequences, first):
            probabilities = p1.copy()
            if len(seq) > 1:
                prev, cur = seq[:-1], seq[1:]
                usable = context_totals[prev, cur] >= self.min_context_count
                rows = np.flatnonzero(usable) + 1
                probabilities[rows] = smoothed[prev[usable], cur[usable]]
                backed_off += int((~usable).sum()) + 1
            else:
                backed_off += len(seq)
            total += len(seq)
            out.append(probabilities)
        self.backoff_rate = {"backed_off": backed_off, "total": total}
        return out

    def to_dict(self) -> dict:
        return {"name": self.name, "n_symbols": self.n_symbols, "alpha": self.alpha,
                "min_context_count": self.min_context_count, "counts": self.counts.tolist(),
                "first_order": self.first_order.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "SecondOrderMarkovChain":
        return cls(n_symbols=d["n_symbols"], alpha=d["alpha"], min_context_count=d["min_context_count"],
                   counts=np.array(d["counts"]), first_order=MarkovChain.from_dict(d["first_order"]))


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"class": type(model).__name__, "model": model.to_dict()}), encoding="utf-8")


def load_model(path: Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = {c.__name__: c for c in (FrequencyBaseline, MarkovChain, SecondOrderMarkovChain)}
    return classes[payload["class"]].from_dict(payload["model"])
