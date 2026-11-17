r"""Experiment 3 models: context Markov chains from a feature priority list, and a small
regularised multinomial logistic regression (classical comparison).

**Context Markov.** A model is an ordered list of context features starting with ``cur``. The
backoff chain drops features from the END of the list:

    [cur, z3, prev1]           -> (cur, z3, prev1) -> (cur, z3) -> (cur) -> marginal   (= experiment 2 mc2_zone)
    [cur, z3, prev1, prev2]    -> order 3 with the same backoff
    [cur, z3, prev1, poss_dur4]-> mc2_zone, refined by possession duration only where data support it

Smoothing is experiment 2's hierarchical Dirichlet backoff (``BackoffMarkov``); the concentration is
chosen by nested inner CV on the outer training folds. UNKNOWN context values (-1) are their own
category, never imputed.

**Logistic regression.** Same causal inputs, continuous geometry and possession features
(standardised on the training fold only), one-hot event history, L2 penalty. It shows what continuous
features add when they do not have to be binned.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .backoff_markov import BackoffMarkov
from .v3_features import CONTEXT_CARDINALITY

LOGGER = logging.getLogger(__name__)
HISTORY = ("prev1", "prev2", "prev3")


@dataclass(frozen=True)
class MarkovSpec:
    name: str
    features: tuple[str, ...]
    reset_history_at_possession: bool = False
    group: str = ""

    def __post_init__(self) -> None:
        if not self.features or self.features[0] != "cur":
            raise ValueError("a context model starts with 'cur'")

    def levels(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(self.features[:n]) for n in range(len(self.features), 0, -1))

    def to_dict(self) -> dict:
        return {"name": self.name, "features": list(self.features), "reset": self.reset_history_at_possession,
                "group": self.group}


def cardinalities(n_symbols: int) -> dict[str, int]:
    card = {"cur": n_symbols, **{h: n_symbols + 2 for h in HISTORY}}
    card.update({k: v + 1 for k, v in CONTEXT_CARDINALITY.items()})  # + UNKNOWN
    return card


def context_arrays(frame: pd.DataFrame, spec: MarkovSpec, n_symbols: int) -> dict[str, np.ndarray]:
    card = cardinalities(n_symbols)
    out = {}
    for f in spec.features:
        col = f if f == "cur" else (f"{f}_reset" if f in HISTORY and spec.reset_history_at_possession else f)
        values = frame[col].to_numpy().astype(np.int64)
        if f not in ("cur",) + HISTORY:
            values = np.where(values < 0, card[f] - 1, values)
        out[f] = values
    return out


def fit_predict_markov(spec: MarkovSpec, train: pd.DataFrame, pred: pd.DataFrame, n_symbols: int,
                       concentration: float, base_alpha: float = 1.0) -> tuple[BackoffMarkov, np.ndarray]:
    """``train``: observations of training sequences (rows with target_id = -1 are not counted)."""
    model = BackoffMarkov(spec.name, n_symbols, cardinalities(n_symbols), concentration, base_alpha, levels=spec.levels())
    model.fit(context_arrays(train, spec, n_symbols), train["target_id"].to_numpy())
    return model, model.predict_proba(context_arrays(pred, spec, n_symbols))


def nested_markov(spec: MarkovSpec, table: pd.DataFrame, folds, n_symbols: int, grid: list[float],
                  base_alpha: float = 1.0) -> tuple[np.ndarray, list[dict], dict]:
    """Out-of-fold probabilities for the EVALUATION rows of ``table`` (eval_eligible), in table order.

    For each outer fold the concentration is chosen by leave-one-inner-fold-out log loss over
    the outer TRAINING folds only."""
    elig = table["eval_eligible"].to_numpy()
    out = np.full((int(elig.sum()), n_symbols), np.nan)
    eval_pos = np.cumsum(elig) - 1
    match = table["match_id"].to_numpy()
    y = table["target_id"].to_numpy()
    choices, models = [], {}
    for fold in folds:
        inner = [f for f in folds if f.fold_id != fold.fold_id]
        scores = {}
        for beta in grid:
            ll, n = 0.0, 0
            for held in inner:
                tr = np.isin(match, [m for f in inner if f is not held for m in f.validation_match_ids])
                va = np.isin(match, held.validation_match_ids) & elig
                _, p = fit_predict_markov(spec, table[tr], table[va], n_symbols, beta, base_alpha)
                ll += float(-np.log(p[np.arange(va.sum()), y[va]]).sum())
                n += int(va.sum())
            scores[beta] = ll / n
        best = min(scores, key=lambda b: (scores[b], b))
        tr = np.isin(match, fold.train_match_ids)
        va = np.isin(match, fold.validation_match_ids) & elig
        model, p = fit_predict_markov(spec, table[tr], table[va], n_symbols, best, base_alpha)
        out[eval_pos[va]] = p
        models[fold.fold_id] = model.to_dict()
        choices.append({"model": spec.name, "fold_id": fold.fold_id, "chosen_concentration": best,
                        "n_contexts_most_specific": int((model.level_counts[0].sum(axis=1) > 0).sum()),
                        **{f"inner_log_loss_{b}": s for b, s in scores.items()}})
    return out, choices, models


def context_support(spec: MarkovSpec, model_dict: dict, frame: pd.DataFrame, n_symbols: int) -> np.ndarray:
    """Training count of the most specific context of each row (0 = unseen)."""
    model = BackoffMarkov.from_dict(model_dict)
    return model.context_support(context_arrays(frame, spec, n_symbols))


# --------------------------------------------------------------------------- logistic regression
LR_CONTINUOUS = ("x", "lateral", "distance_to_goal", "shooting_angle", "danger_area", "penalty_area",
                 "pf_distance_to_goal", "pf_shooting_angle", "pf_danger_area",
                 "log_possession_duration", "log_possession_events", "possession_progression", "same_team_as_previous",
                 "recent_possession_change", "log_time_since_previous", "events_last_10s", "minute_scaled")


def lr_design(frame: pd.DataFrame, n_symbols: int) -> tuple[np.ndarray, list[str]]:
    """Design matrix (unscaled): one-hot cur/prev1/prev2, continuous features each with a missing
    indicator (missing -> 0 + indicator; the column set never depends on the data), cur x distance and cur x angle interactions, period one-hot."""
    cols, names = [], []
    for f, k in (("cur", n_symbols), ("prev1", n_symbols + 2), ("prev2", n_symbols + 2)):
        oh = np.eye(k)[frame[f].to_numpy().astype(int)]
        cols.append(oh)
        names += [f"{f}={i}" for i in range(k)]
    for c in LR_CONTINUOUS:
        v = frame[c].to_numpy(dtype=float)
        miss = np.isnan(v)
        cols.append(np.where(miss, 0.0, v)[:, None])
        names.append(c)
        cols.append(miss[:, None].astype(float))  # always present: identical columns in every split
        names.append(f"{c}_missing")
    cur = np.eye(n_symbols)[frame["cur"].to_numpy().astype(int)]
    for c in ("distance_to_goal", "shooting_angle"):
        v = np.nan_to_num(frame[c].to_numpy(dtype=float))
        cols.append(cur * v[:, None])
        names += [f"cur={i}*{c}" for i in range(n_symbols)]
    cols.append(np.eye(4)[np.clip(frame["period"].to_numpy().astype(int) - 1, 0, 3)])
    names += [f"period={p}" for p in range(1, 5)]
    return np.hstack(cols), names


def fit_predict_lr(train: pd.DataFrame, pred: pd.DataFrame, n_symbols: int, c_value: float, max_iter: int = 300):
    from sklearn.linear_model import LogisticRegression

    tr = train[train["target_id"] >= 0]
    xtr, names = lr_design(tr, n_symbols)
    xpr, _ = lr_design(pred, n_symbols)
    mean, std = xtr.mean(axis=0), xtr.std(axis=0)
    std[std == 0] = 1.0                       # scaler fitted on the training fold only
    model = LogisticRegression(C=c_value, max_iter=max_iter, tol=1e-4)
    model.fit((xtr - mean) / std, tr["target_id"].to_numpy())
    p = np.zeros((len(pred), n_symbols))
    p[:, model.classes_] = model.predict_proba((xpr - mean) / std)
    if len(model.classes_) < n_symbols:      # a class absent from training: tiny floor, renormalised
        p = (p + 1e-9) / (1 + n_symbols * 1e-9)
    return model, p, {"n_iter": int(np.max(model.n_iter_)), "converged": bool(np.max(model.n_iter_) < max_iter),
                      "n_features": len(names)}


def cv_lr(table: pd.DataFrame, folds, n_symbols: int, c_value: float) -> tuple[np.ndarray, list[dict]]:
    elig = table["eval_eligible"].to_numpy()
    out = np.full((int(elig.sum()), n_symbols), np.nan)
    eval_pos = np.cumsum(elig) - 1
    match = table["match_id"].to_numpy()
    info = []
    for fold in folds:
        start = time.perf_counter()
        tr = np.isin(match, fold.train_match_ids)
        va = np.isin(match, fold.validation_match_ids) & elig
        _, p, meta = fit_predict_lr(table[tr], table[va], n_symbols, c_value)
        out[eval_pos[va]] = p
        info.append({"fold_id": fold.fold_id, "C": c_value, **meta, "seconds": time.perf_counter() - start})
    return out, info


# --------------------------------------------------------------------------- checkpoints
def spec_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Checkpoint:
    """One .npz (probabilities) + .json (metadata) per completed configuration; resumable."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str, key: str) -> Path:
        return self.root / f"{name}__{key}"

    def load(self, name: str, key: str):
        p = self.path(name, key)
        if (p.with_suffix(".npz")).exists() and (p.with_suffix(".json")).exists():
            return np.load(p.with_suffix(".npz"))["proba"], json.loads(p.with_suffix(".json").read_text())
        return None

    def save(self, name: str, key: str, proba: np.ndarray, meta: dict) -> None:
        p = self.path(name, key)
        np.savez_compressed(p.with_suffix(".npz"), proba=proba)
        p.with_suffix(".json").write_text(json.dumps(meta, default=str))


class Progress:
    def __init__(self, total: int, label: str) -> None:
        self.total, self.done, self.label, self.start = total, 0, label, time.time()

    def step(self, what: str) -> None:
        self.done += 1
        elapsed = time.time() - self.start
        eta = elapsed / self.done * (self.total - self.done)
        LOGGER.info("[%s] %d/%d done (%s) elapsed %.0fs, ~%.0fs remaining", self.label, self.done, self.total, what,
                    elapsed, eta)
