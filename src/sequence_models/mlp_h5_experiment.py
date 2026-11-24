r"""Experiment 6B: multilayer perceptron with a five-event history (MLP-H5).

Task ``actions15_360current_v1`` (``docs/canonical_prediction_protocol.md``), unchanged.

Input. The XGB-H5 / MLR-H5 history :math:`X_t = (e_{t-4}, \dots, e_t)` from
``event_history.HistoryRows`` (not re-implemented): positions before the start of the
(match, period) sequence hold ``<START>``. Representation: the position-specific one-hot
vector :math:`\phi(X_t) \in \{0,1\}^{79}` of experiment 6A (lags 4..1: 16 categories, e_t: 15).

Model (for hidden sizes :math:`H_1, \dots, H_L`):

.. math::
   h_1 = \mathrm{ReLU}(W_1 \phi(X_t) + b_1),\quad h_\ell = \mathrm{ReLU}(W_\ell\, \mathrm{drop}(h_{\ell-1}) + b_\ell),\quad
   z = W_{L+1}\, \mathrm{drop}(h_L) + b_{L+1} \in \mathbb{R}^{15},\quad P(Y_t = k \mid X_t) = \mathrm{softmax}(z)_k .

Because :math:`\phi` has exactly 5 ones, :math:`W_1 \phi(X_t) = \sum_{j=1}^{5} W_1[:, c_j(X_t)]` where
:math:`c_j` is the active column of block :math:`j`; the first layer is implemented as that sum
(``nn.Embedding``), an exact identity, initialised like ``nn.Linear(79, H_1)``. Unlike MLR-H5
the network can represent interactions between history positions (through the hidden ReLUs).

Training: mean cross-entropy on logits (``F.cross_entropy``), AdamW with decoupled weight decay
on weight matrices only, mini-batches of rows in a reproducible per-epoch permutation, early
stopping on a match-grouped stopping subset of the fit's OWN training matches; the returned
model is the best-epoch state. All randomness (initialisation, permutation, dropout) is derived
from ``numpy.random.SeedSequence`` of (base seed, fold, inner fold, configuration, replicate,
epoch), re-seeded at every epoch boundary, so a fit resumed from an epoch checkpoint continues
bit-identically (CPU, 1 thread, deterministic algorithms).

Nested CV (protocol of experiments 5 / 6A): for each outer fold, leave-one-inner-fold-out over
the other 4 folds. An inner fit trains on 3 inner folds minus a 10 % stopping subset of their
matches, stops on that subset, and is SCORED on the held inner fold, which never influences its
stopping epoch. The configuration with the smallest pooled inner-validation log loss is refitted
on the outer-training matches (again minus a 10 % stopping subset) and predicts the
outer-validation rows once.
"""

from __future__ import annotations

import io
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .canonical_task import K
from .event_history import N_CATEGORIES, PAD, HistoryError, aggregate_cells, position_label
from .splits import FoldAssignment, LeakageError
from .xgboost_experiment import softmax64
from .xgboost_h2_experiment import CheckpointMismatch, atomic_write_bytes, sha256_bytes

FEATURE_SET = "MLP-H5"
N_PREVIOUS = 4
WIDTHS = [N_CATEGORIES] * N_PREVIOUS + [K]
OFFSETS = np.r_[0, np.cumsum(WIDTHS)[:-1]].astype(np.int64)    # [0, 16, 32, 48, 64]
N_INPUT = int(sum(WIDTHS))                                     # 79


class NumericalError(RuntimeError):
    """Non-finite loss, gradient or parameter during training."""


class Interrupted(RuntimeError):
    """Raised by the test hook ``interrupt_after`` to simulate a killed process."""


# ============================================================================ encoding
def column_indices(ids: np.ndarray) -> np.ndarray:
    """(n, 5) active column of each position block in the 79-column one-hot layout of 6A."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 2 or ids.shape[1] != N_PREVIOUS + 1:
        raise HistoryError("expected (n, 5) history ids E_{t-4}..E_t")
    if (ids[:, -1] == PAD).any():
        raise HistoryError("the current event E_t can never be padding")
    for j, w in enumerate(WIDTHS):
        if (ids[:, j] < 0).any() or (ids[:, j] >= w).any():
            raise HistoryError(f"category outside the vocabulary at {position_label(N_PREVIOUS - j)}")
    return ids + OFFSETS


# ============================================================================ model
class MLP(nn.Module):
    """Feed-forward ReLU network on the 79-column one-hot history; 15 output logits."""

    def __init__(self, hidden: list[int], dropout: float = 0.0):
        super().__init__()
        if not hidden:
            raise ValueError("at least one hidden layer")
        self.hidden, self.dropout_p = [int(h) for h in hidden], float(dropout)
        self.first = nn.Embedding(N_INPUT, self.hidden[0])          # columns of W_1 (as rows)
        self.first_bias = nn.Parameter(torch.empty(self.hidden[0]))
        bound = 1.0 / np.sqrt(N_INPUT)                              # = nn.Linear(79, H_1) default
        nn.init.uniform_(self.first.weight, -bound, bound)
        nn.init.uniform_(self.first_bias, -bound, bound)
        layers: list[nn.Module] = []
        sizes = self.hidden + [K]
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.ReLU(), nn.Dropout(self.dropout_p), nn.Linear(a, b)]
        self.body = nn.Sequential(*layers)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.body(self.first(idx).sum(dim=1) + self.first_bias)

    def weight_matrices(self) -> list[np.ndarray]:
        """[W_1 (H_1, 79), W_2, ..., W_{L+1} (15, H_L)] and biases, for independent checks."""
        ws = [self.first.weight.detach().T.numpy().astype(np.float64)]
        bs = [self.first_bias.detach().numpy().astype(np.float64)]
        for m in self.body:
            if isinstance(m, nn.Linear):
                ws.append(m.weight.detach().numpy().astype(np.float64))
                bs.append(m.bias.detach().numpy().astype(np.float64))
        return ws, bs


def configure_torch(tcfg: dict) -> None:
    if tcfg["device"] != "cpu":
        raise ValueError("experiment 6B is frozen to CPU")
    torch.set_num_threads(int(tcfg["num_threads"]))
    torch.use_deterministic_algorithms(bool(tcfg["deterministic_algorithms"]))


def derive_seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1, dtype=np.uint32)[0])


def build_model(candidate: dict, seed: int) -> MLP:
    torch.manual_seed(seed)
    return MLP(candidate["hidden"], candidate["dropout"])


def make_optimizer(model: MLP, candidate: dict, tcfg: dict) -> torch.optim.Optimizer:
    decay = [p for n, p in model.named_parameters() if n.endswith("weight")]
    no_decay = [p for n, p in model.named_parameters() if not n.endswith("weight")]
    return torch.optim.AdamW([{"params": decay, "weight_decay": float(candidate["weight_decay"])},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=float(candidate["lr"]), betas=tuple(tcfg["betas"]), eps=float(tcfg["eps"]))


@torch.no_grad()
def logits_of(model: MLP, idx: torch.Tensor, batch: int = 65536) -> torch.Tensor:
    model.eval()
    return torch.cat([model(idx[i:i + batch]) for i in range(0, len(idx), batch)]) if len(idx) else torch.empty(0, K)


def mean_nll(model: MLP, idx: torch.Tensor, y: torch.Tensor) -> float:
    z = logits_of(model, idx).double()
    return float(-z.log_softmax(dim=1)[torch.arange(len(y)), y].mean())


def predict_proba(model: MLP, ids: np.ndarray) -> np.ndarray:
    """(n, 15) float64 probabilities; evaluated once per unique history, eval mode."""
    cells, _, _, inverse = aggregate_cells(np.asarray(ids))
    z = logits_of(model, torch.from_numpy(column_indices(cells))).double().numpy()
    return softmax64(z)[inverse]


def state_copy(model: MLP) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ============================================================================ checkpoint I/O
def _torch_bytes(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def save_atomic(path: Path, obj) -> str:
    data = _torch_bytes(obj)
    atomic_write_bytes(path, data)
    return sha256_bytes(data)


def load_checked(path: Path, identifiers: dict, what: str):
    try:
        obj = torch.load(io.BytesIO(path.read_bytes()), weights_only=False)
    except Exception as exc:  # corrupted or truncated file
        raise CheckpointMismatch(f"{what}: unreadable checkpoint {path.name} ({exc.__class__.__name__})") from exc
    if obj.get("identifiers") != identifiers:
        diff = sorted(k for k in set(identifiers) | set(obj.get("identifiers", {}))
                      if identifiers.get(k) != obj.get("identifiers", {}).get(k))
        raise CheckpointMismatch(f"{what}: checkpoint identifiers differ ({diff}); refusing to mix runs")
    return obj


# ============================================================================ one fit
@dataclass
class FitResult:
    key: str
    candidate: dict
    seed: int
    best_state: dict
    best_epoch: int
    epochs_trained: int
    stop_reason: str
    best_stop_log_loss: float
    history: list[dict]
    n_train_rows: int
    n_stop_rows: int
    seconds: float
    resumed_from_epoch: int = 0


def train_fit(key: str, X: torch.Tensor, y: torch.Tensor, train_rows: np.ndarray, stop_rows: np.ndarray, candidate: dict,
              tcfg: dict, seed: int, ckpt_path: Path | None = None, identifiers: dict | None = None,
              interrupt_after: int | None = None, log=None) -> FitResult:
    """Train one MLP with early stopping; epoch-boundary checkpoints in ``ckpt_path`` (resumed
    if present and compatible). ``interrupt_after`` (tests): raise after that many epochs."""
    t0 = time.perf_counter()
    xtr, ytr = X[train_rows], y[train_rows]
    xst, yst = X[stop_rows], y[stop_rows]
    if len(ytr) == 0 or len(yst) == 0:
        raise ValueError(f"{key}: empty training or stopping set")
    ident = {**(identifiers or {}), "fit_key": key, "candidate": json.dumps(candidate, sort_keys=True), "seed": seed,
             "train_rows_sha256": sha256_bytes(np.ascontiguousarray(train_rows, dtype=np.int64).tobytes()),
             "stop_rows_sha256": sha256_bytes(np.ascontiguousarray(stop_rows, dtype=np.int64).tobytes())}
    model = build_model(candidate, seed)
    opt = make_optimizer(model, candidate, tcfg)
    start_epoch, best, best_epoch, bad, history = 0, float("inf"), 0, 0, []
    best_state = state_copy(model)
    resumed = 0
    if ckpt_path is not None and ckpt_path.exists():
        ck = load_checked(ckpt_path, ident, key)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch, best, best_epoch, bad = ck["epoch"], ck["best"], ck["best_epoch"], ck["bad"]
        history, best_state = ck["history"], ck["best_state"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        resumed = start_epoch
        if log:
            log(f"{key}: resumed at epoch boundary {start_epoch}")
    if start_epoch == 0:
        history.append({"epoch": 0, "train_loss": None, "stop_log_loss": mean_nll(model, xst, yst), "seconds": 0.0})
        best = history[0]["stop_log_loss"]            # the initial model is the incumbent (best_epoch = 0)
    bs, n = int(tcfg["batch_size"]), len(ytr)
    reason = "max_epochs"
    if bad >= int(tcfg["patience"]):
        reason = "patience"
    for epoch in range(start_epoch, int(tcfg["max_epochs"]) if reason == "max_epochs" else start_epoch):
        te = time.perf_counter()
        es = derive_seed(seed, 1, epoch)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(es))
        torch.manual_seed(derive_seed(seed, 2, epoch))                   # dropout masks of this epoch
        np.random.seed(derive_seed(seed, 3, epoch))
        random.seed(derive_seed(seed, 4, epoch))
        model.train()
        loss_sum = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(model(xtr[b]), ytr[b])
            if not torch.isfinite(loss):
                raise NumericalError(f"{key}: non-finite loss at epoch {epoch + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise NumericalError(f"{key}: non-finite gradient at epoch {epoch + 1}")
            opt.step()
            loss_sum += loss.detach().item() * len(b)
        stop_ll = mean_nll(model, xst, yst)
        if not np.isfinite(stop_ll) or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise NumericalError(f"{key}: non-finite parameters or stopping loss at epoch {epoch + 1}")
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / n, "stop_log_loss": stop_ll,
                        "seconds": time.perf_counter() - te})
        if stop_ll < best - float(tcfg["min_delta"]):
            best, best_epoch, bad, best_state = stop_ll, epoch + 1, 0, state_copy(model)
        else:
            bad += 1
        if ckpt_path is not None:   # epoch boundary: everything needed to continue exactly
            save_atomic(ckpt_path, {"identifiers": ident, "epoch": epoch + 1, "model": model.state_dict(),
                                    "optimizer": opt.state_dict(), "best": best, "best_epoch": best_epoch, "bad": bad,
                                    "best_state": best_state, "history": history, "torch_rng": torch.get_rng_state(),
                                    "numpy_rng": np.random.get_state(), "python_rng": random.getstate()})
        if interrupt_after is not None and epoch + 1 == interrupt_after:
            raise Interrupted(f"{key}: simulated interruption after epoch {epoch + 1}")
        if bad >= int(tcfg["patience"]):
            reason = "patience"
            break
    if best_epoch == 0:
        best = history[0]["stop_log_loss"]
    return FitResult(key, candidate, seed, best_state, best_epoch, len(history) - 1, reason, best, history,
                     int(n), int(len(yst)), time.perf_counter() - t0, resumed)


def model_from_state(candidate: dict, state: dict) -> MLP:
    m = MLP(candidate["hidden"], candidate["dropout"])
    m.load_state_dict(state)
    m.eval()
    return m


# ============================================================================ nested CV
def stopping_split(match_ids: list[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic match-grouped split of training matches: (fit matches, stopping matches)."""
    ms = sorted(int(m) for m in match_ids)
    n_stop = max(1, int(round(fraction * len(ms))))
    stop = sorted(int(m) for m in np.random.default_rng(seed).choice(ms, n_stop, replace=False))
    s = set(stop)
    return [m for m in ms if m not in s], stop


@dataclass
class NestedResult:
    proba: np.ndarray                                  # primary (replicate 0)
    proba_replicates: dict[int, np.ndarray]            # robustness refits (replicates 1..R)
    fold_of_row: np.ndarray
    states: dict[tuple[int, int], dict] = field(default_factory=dict)     # (fold, replicate) -> state
    selected: dict[int, dict] = field(default_factory=dict)              # fold -> candidate
    choices: list[dict] = field(default_factory=list)
    inner_scores: list[dict] = field(default_factory=list)
    fits: list[dict] = field(default_factory=list)
    histories: list[dict] = field(default_factory=list)
    timings: list[dict] = field(default_factory=list)
    fold_metrics: list[dict] = field(default_factory=list)
    resumed_folds: list[int] = field(default_factory=list)
    computed_folds: list[int] = field(default_factory=list)
    resumed_fits: int = 0


def run_nested_cv(ids_all: np.ndarray, match_id: np.ndarray, y: np.ndarray, folds: list[FoldAssignment], cfg: dict,
                  ckpt_dir: Path | None = None, identifiers: dict | None = None, trace: list | None = None,
                  log=None) -> NestedResult:
    tcfg, cands = cfg["training"], cfg["mlp"]["candidates"]
    configure_torch(tcfg)
    base = int(tcfg["seed"])
    n_rep = 1 + int(tcfg["robustness_seeds"])
    ident = identifiers or {}
    X = torch.from_numpy(column_indices(ids_all))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    n = len(y)
    fold_of_row = np.full(n, -1)
    for f in folds:
        fold_of_row[np.isin(match_id, f.validation_match_ids)] = f.fold_id
    if (fold_of_row < 0).any():
        raise LeakageError("an eligible row belongs to no validation fold")
    res = NestedResult(np.full((n, K), np.nan), {r: np.full((n, K), np.nan) for r in range(1, n_rep)}, fold_of_row)

    def rows_of(match_ids, purpose, fold_id):
        m = np.isin(match_id, list(match_ids))
        if trace is not None:
            trace.append({"outer_fold": fold_id, "purpose": purpose,
                          "match_ids": frozenset(int(x) for x in np.unique(match_id[m]))})
        return np.flatnonzero(m)

    def fit(key, fold_id, train_m, stop_m, cand, seed, purpose):
        fdir = ckpt_dir / f"fold{fold_id}" if ckpt_dir is not None else None
        done = fdir / f"{key}.done.pt" if fdir is not None else None
        tr = rows_of(train_m, f"{purpose}_train", fold_id)
        st = rows_of(stop_m, f"{purpose}_stop", fold_id)
        if done is not None and done.exists():
            rec = load_checked(done, {**ident, "fit_key": key}, key)
            r = rec["result"]
            if r.seed != seed or r.candidate != cand or r.n_train_rows != len(tr) or r.n_stop_rows != len(st):
                raise CheckpointMismatch(f"{key}: completed fit does not match its specification")
            res.resumed_fits += 1
            return r
        r = train_fit(key, X, yt, tr, st, cand, tcfg, seed, fdir / f"{key}.inprogress.pt" if fdir else None, ident, log=log)
        if done is not None:
            save_atomic(done, {"identifiers": {**ident, "fit_key": key}, "result": r})
            (fdir / f"{key}.inprogress.pt").unlink(missing_ok=True)
        return r

    def record(r: FitResult, fold_id, stage, held, extra=None):
        res.fits.append({"fold_id": fold_id, "stage": stage, "held_inner_fold": held, "candidate": r.candidate["name"],
                         "seed": r.seed, "best_epoch": r.best_epoch, "epochs_trained": r.epochs_trained,
                         "stop_reason": r.stop_reason, "hit_max_epochs": r.epochs_trained == int(tcfg["max_epochs"]),
                         "best_stop_log_loss": r.best_stop_log_loss,
                         "final_train_loss": r.history[-1]["train_loss"], "n_train_rows": r.n_train_rows,
                         "n_stop_rows": r.n_stop_rows, "seconds": r.seconds, "resumed_from_epoch": r.resumed_from_epoch,
                         **(extra or {})})
        res.histories.append({"fold_id": fold_id, "stage": stage, "held_inner_fold": held, "candidate": r.candidate["name"],
                              "seed": r.seed, "history": json.dumps(r.history)})

    for f in folds:
        outer_val = set(f.validation_match_ids)
        if set(f.train_match_ids) & outer_val:
            raise LeakageError(f"fold {f.fold_id}: match in training and validation")
        va = np.flatnonzero(np.isin(match_id, f.validation_match_ids))
        fold_rec = ckpt_dir / f"fold{f.fold_id}" / "fold_complete.pt" if ckpt_dir is not None else None
        if fold_rec is not None and fold_rec.exists():
            rec = load_checked(fold_rec, ident, f"fold {f.fold_id}")
            if not np.array_equal(rec["val_index"], va):
                raise CheckpointMismatch(f"fold {f.fold_id}: validation rows differ from the checkpoint")
            res.proba[va] = rec["proba"][0]
            for r in range(1, n_rep):
                res.proba_replicates[r][va] = rec["proba"][r]
            for r in range(n_rep):
                res.states[(f.fold_id, r)] = rec["states"][r]
            res.selected[f.fold_id] = rec["selected"]
            for key in ("choices", "inner_scores", "fits", "histories", "timings", "fold_metrics"):
                getattr(res, key).extend(rec[key])
            if trace is not None:
                trace.extend({**t, "match_ids": frozenset(t["match_ids"])} for t in rec["trace"])
            res.resumed_folds.append(f.fold_id)
            if log:
                log(f"fold {f.fold_id}: resumed from completed-fold checkpoint ({rec['selected']['name']})")
            continue
        n_trace, n_fits, n_hist, n_scores = (len(trace) if trace is not None else 0), len(res.fits), len(res.histories), len(res.inner_scores)
        t_fold = time.perf_counter()
        inner = [g for g in folds if g.fold_id != f.fold_id]
        splits = []
        for held in inner:
            itr = [m for g in inner if g is not held for m in g.validation_match_ids]
            if (set(itr) | set(held.validation_match_ids)) & outer_val:
                raise LeakageError(f"fold {f.fold_id}: outer-validation match in an inner set")
            fit_m, stop_m = stopping_split(itr, float(tcfg["stopping_fraction"]), derive_seed(base, 7, f.fold_id, held.fold_id + 1))
            splits.append((held, fit_m, stop_m))
        scores = []
        for ci, cand in enumerate(cands):
            tc = time.perf_counter()
            nll, cnt = 0.0, 0
            for held, fit_m, stop_m in splits:
                key = f"f{f.fold_id}_inner{held.fold_id}_{cand['name']}_r0"
                r = fit(key, f.fold_id, fit_m, stop_m, cand, derive_seed(base, f.fold_id, held.fold_id + 1, ci, 0), "inner")
                hv = rows_of(held.validation_match_ids, "inner_validation", f.fold_id)
                p = predict_proba(model_from_state(cand, r.best_state), ids_all[hv])
                part = float(-np.log(p[np.arange(len(hv)), y[hv]]).sum())
                nll += part
                cnt += len(hv)
                record(r, f.fold_id, "inner", held.fold_id, {"inner_validation_rows": len(hv),
                                                             "inner_validation_log_loss": part / len(hv)})
            scores.append((round(nll / cnt, 12), ci))
            res.inner_scores.append({"fold_id": f.fold_id, "candidate": cand["name"], **{k: v for k, v in cand.items() if k != "name"},
                                     "hidden": json.dumps(cand["hidden"]), "pooled_inner_log_loss": nll / cnt,
                                     "inner_validation_rows": cnt, "seconds": time.perf_counter() - tc})
            if log:
                log(f"fold {f.fold_id} {cand['name']} {cand}: pooled inner log loss {nll / cnt:.6f} ({time.perf_counter() - tc:.0f}s)")
        best_score, best_ci = min(scores)
        cand = cands[best_ci]
        selection_seconds = time.perf_counter() - t_fold
        t0 = time.perf_counter()
        fit_m, stop_m = stopping_split(list(f.train_match_ids), float(tcfg["stopping_fraction"]), derive_seed(base, 7, f.fold_id, 0))
        probas, states = [], []
        for rep in range(n_rep):
            key = f"f{f.fold_id}_outer_{cand['name']}_r{rep}"
            r = fit(key, f.fold_id, fit_m, stop_m, cand, derive_seed(base, f.fold_id, 0, best_ci, rep), "outer")
            vrows = rows_of(f.validation_match_ids, "outer_validation_predict", f.fold_id)
            if not np.array_equal(vrows, va):
                raise RuntimeError("validation rows changed")
            p = predict_proba(model_from_state(cand, r.best_state), ids_all[va])
            probas.append(p)
            states.append(r.best_state)
            ll = float(-np.log(p[np.arange(len(va)), y[va]]).mean())
            record(r, f.fold_id, "outer_refit" if rep == 0 else "robustness_refit", -1,
                   {"replicate": rep, "outer_validation_log_loss": ll})
            if log:
                log(f"fold {f.fold_id} refit {cand['name']} replicate {rep}: best epoch {r.best_epoch}/{r.epochs_trained}, "
                    f"outer-validation log loss {ll:.6f}")
        res.proba[va] = probas[0]
        for rep in range(1, n_rep):
            res.proba_replicates[rep][va] = probas[rep]
        for rep in range(n_rep):
            res.states[(f.fold_id, rep)] = states[rep]
        res.selected[f.fold_id] = cand
        choice = {"fold_id": f.fold_id, "candidate": cand["name"], "candidate_index": best_ci, "inner_log_loss": best_score,
                  **{k: v for k, v in cand.items() if k != "name"}, "hidden": json.dumps(cand["hidden"]),
                  "train_matches": len(f.train_match_ids), "refit_fit_matches": len(fit_m), "refit_stop_matches": len(stop_m),
                  "validation_matches": len(f.validation_match_ids), "validation_rows": int(len(va))}
        timing = {"fold_id": f.fold_id, "selection_seconds": selection_seconds, "refit_predict_seconds": time.perf_counter() - t0}
        fm = {"fold_id": f.fold_id, "log_loss": float(-np.log(probas[0][np.arange(len(va)), y[va]]).mean()), "n_rows": int(len(va))}
        res.choices.append(choice)
        res.timings.append(timing)
        res.fold_metrics.append(fm)
        res.computed_folds.append(f.fold_id)
        if log:
            log(f"fold {f.fold_id}: selected {cand['name']}; outer-validation log loss {fm['log_loss']:.6f}")
        if ckpt_dir is not None:
            ft = [{**t, "match_ids": sorted(t["match_ids"])} for t in trace[n_trace:]] if trace is not None else []
            save_atomic(fold_rec, {"identifiers": ident, "val_index": va, "proba": probas, "states": states, "selected": cand,
                                   "choices": [choice], "inner_scores": res.inner_scores[n_scores:], "fits": res.fits[n_fits:],
                                   "histories": res.histories[n_hist:], "timings": [timing], "fold_metrics": [fm], "trace": ft})
    if np.isnan(res.proba).any() or any(np.isnan(p).any() for p in res.proba_replicates.values()):
        raise RuntimeError("rows without prediction")
    return res


def numpy_forward(ws: list[np.ndarray], bs: list[np.ndarray], onehot: np.ndarray) -> np.ndarray:
    """Independent float64 forward pass on a dense one-hot matrix (eval mode: no dropout)."""
    h = onehot
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w.T + b
        if i < len(ws) - 1:
            h = np.maximum(h, 0.0)
    return softmax64(h)


__all__ = ["FEATURE_SET", "N_INPUT", "OFFSETS", "column_indices", "MLP", "configure_torch", "derive_seed", "build_model",
           "make_optimizer", "train_fit", "FitResult", "predict_proba", "model_from_state", "stopping_split",
           "run_nested_cv", "NestedResult", "numpy_forward", "save_atomic", "load_checked", "Interrupted",
           "NumericalError", "CheckpointMismatch"]
