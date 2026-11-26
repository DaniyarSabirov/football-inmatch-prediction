r"""Experiment 6C: long short-term memory network with a five-event history (LSTM-H5).

Task ``actions15_360current_v1`` (``docs/canonical_prediction_protocol.md``), unchanged.

Input. The XGB-H5 / MLR-H5 / MLP-H5 history :math:`X_t = (E_{t-4}, \dots, E_t)` from
``event_history.HistoryRows`` (not re-implemented); positions before the start of the
(match, period) sequence hold ``<START>`` (id 15). Each position :math:`s = 1, \dots, 5`
(:math:`s = 1` is :math:`E_{t-4}`, :math:`s = 5` is :math:`E_t`) is the one-hot vector
:math:`x_s \in \{0,1\}^{16}` over the shared vocabulary (15 classes + ``<START>``), so the
input tensor of a batch of :math:`B` windows has shape ``(B, 5, 16)``, oldest position first.

Model (one recurrent layer; ``torch.nn.LSTM``, gate order i, f, g, o as in PyTorch):

.. math::
   i_s = \sigma(W_i x_s + U_i h_{s-1} + b_i),\quad f_s = \sigma(W_f x_s + U_f h_{s-1} + b_f),\quad
   g_s = \tanh(W_g x_s + U_g h_{s-1} + b_g),\quad o_s = \sigma(W_o x_s + U_o h_{s-1} + b_o),

   c_s = f_s \odot c_{s-1} + i_s \odot g_s,\qquad h_s = o_s \odot \tanh(c_s),\qquad h_0 = c_0 = 0,

   z = W_{\text{out}}\,\mathrm{drop}_p(h_5^{(L)}) + b_{\text{out}} \in \mathbb{R}^{15},\qquad
   P(Y_t = c \mid X_t) = \mathrm{softmax}(z)_c .

PyTorch stores ``weight_ih = [W_i; W_f; W_g; W_o]`` (4H x 16), ``weight_hh = [U_i; ...; U_o]``
(4H x H) and two bias vectors; the effective gate bias is ``bias_ih + bias_hh``. With
:math:`L = 2` layers, layer 2 receives :math:`h_s^{(1)}` (dimension H) as its input; the
built-in inter-layer dropout of ``nn.LSTM`` is disabled (0), the only dropout is on
:math:`h_5^{(L)}` before the output layer.

Stateless windows. ``nn.LSTM`` is called without an initial state, so :math:`h_0 = c_0 = 0`
for every window in every call; no state is carried between windows, batches, matches, folds
or between training and evaluation. Each 5-position window is an independent example.

Training, nested CV, stopping subsets, seeds and checkpointing follow experiment 6B
(``mlp_h5_experiment``) exactly; its seed derivation, stopping split and checkpoint I/O are
reused unchanged, so with the same base seed the inner and outer stopping subsets are the
SAME matches as in 6B. Additionally the global L2 norm of the gradient of every mini-batch
is recorded (no clipping is applied; the norm doubles as the finiteness check).
"""

from __future__ import annotations

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
from .event_history import N_CATEGORIES, PAD, HistoryError, aggregate_cells
from .mlp_h5_experiment import (Interrupted, NumericalError, derive_seed, load_checked, save_atomic, state_copy,
                                stopping_split)
from .splits import FoldAssignment, LeakageError
from .xgboost_experiment import softmax64
from .xgboost_h2_experiment import CheckpointMismatch, sha256_bytes

FEATURE_SET = "LSTM-H5"
N_POSITIONS = 5                 # s = 1..5  <->  E_{t-4}, ..., E_t
N_INPUT = N_CATEGORIES          # 16 = 15 classes + <START>, shared by all positions
GATES = ("input", "forget", "candidate", "output")      # PyTorch block order in weight_ih / weight_hh
GRAD_NORM_THRESHOLDS = (1.0, 5.0, 10.0)                  # diagnostic counts only; no clipping


# ============================================================================ encoding
def validate_ids(ids: np.ndarray) -> np.ndarray:
    """(n, 5) int64 history ids E_{t-4}..E_t, chronological; checked against the vocabulary."""
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.shape[1] != N_POSITIONS:
        raise HistoryError("expected (n, 5) history ids E_{t-4}..E_t")
    if not np.issubdtype(ids.dtype, np.integer):
        raise HistoryError("history ids must be integers")
    ids = ids.astype(np.int64)
    if (ids < 0).any() or (ids >= N_INPUT).any():
        raise HistoryError("category outside the vocabulary 0..15")
    if (ids[:, -1] == PAD).any():
        raise HistoryError("the current event E_t (last position) can never be <START>")
    return ids


def one_hot_sequence(ids: np.ndarray) -> np.ndarray:
    """Explicit (n, 5, 16) float64 one-hot tensor (used for independent checks)."""
    ids = validate_ids(ids)
    x = np.zeros((len(ids), N_POSITIONS, N_INPUT))
    x[np.arange(len(ids))[:, None], np.arange(N_POSITIONS)[None, :], ids] = 1.0
    return x


# ============================================================================ model
class LSTMH5(nn.Module):
    """Unidirectional, stateless LSTM over the 5 one-hot positions; 15 output logits from the
    final hidden state of the last layer."""

    def __init__(self, hidden: int, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        hidden, layers, dropout = int(hidden), int(layers), float(dropout)
        if hidden < 1 or layers not in (1, 2) or not 0.0 <= dropout < 1.0:
            raise ValueError(f"invalid LSTM configuration: hidden={hidden}, layers={layers}, dropout={dropout}")
        self.hidden, self.layers, self.dropout_p = hidden, layers, dropout
        # built-in dropout (between stacked layers) deliberately 0: see the module docstring
        self.lstm = nn.LSTM(input_size=N_INPUT, hidden_size=hidden, num_layers=layers, bias=True,
                            batch_first=True, dropout=0.0, bidirectional=False)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, K)

    def encode(self, idx: torch.Tensor) -> torch.Tensor:
        return F.one_hot(idx, N_INPUT).to(self.out.weight.dtype)       # (B, 5, 16)

    def final_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """h_5 of the last layer, (B, H). No initial state is passed: h_0 = c_0 = 0 per window."""
        _, (h_n, _) = self.lstm(self.encode(idx))
        return h_n[-1]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(self.final_hidden(idx)))

    def export(self) -> dict[str, np.ndarray]:
        """float64 copies of all parameters under their PyTorch names (independent checks)."""
        return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in self.state_dict().items()}


def parameter_count_formula(hidden: int, layers: int, n_input: int = N_INPUT, n_out: int = K) -> dict[str, int]:
    """P_LSTM(layer) = 4 H m + 4 H^2 + 8 H with m the layer's input dimension; P_out = 15 H + 15."""
    per_layer = []
    m = n_input
    for _ in range(layers):
        per_layer.append(4 * hidden * m + 4 * hidden * hidden + 8 * hidden)
        m = hidden
    out = n_out * hidden + n_out
    return {"lstm_layers": per_layer, "output": out, "total": sum(per_layer) + out}


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def configure_torch(tcfg: dict) -> None:
    if tcfg["device"] != "cpu":
        raise ValueError("experiment 6C is frozen to CPU (chosen by benchmark before the run)")
    torch.set_num_threads(int(tcfg["num_threads"]))
    torch.use_deterministic_algorithms(bool(tcfg["deterministic_algorithms"]))


def build_model(candidate: dict, seed: int) -> LSTMH5:
    torch.manual_seed(seed)
    return LSTMH5(candidate["hidden"], candidate["layers"], candidate["dropout"])


def make_optimizer(model: LSTMH5, candidate: dict, tcfg: dict) -> torch.optim.Optimizer:
    """AdamW; decoupled weight decay on weight matrices (weight_ih_l*, weight_hh_l*, out.weight)
    only, never on biases (the 6B convention)."""
    decay = [p for n, p in model.named_parameters() if n.rsplit(".", 1)[-1].startswith("weight")]
    no_decay = [p for n, p in model.named_parameters() if not n.rsplit(".", 1)[-1].startswith("weight")]
    if len(decay) + len(no_decay) != len(list(model.parameters())) or not decay or not no_decay:
        raise RuntimeError("parameter grouping for weight decay failed")
    return torch.optim.AdamW([{"params": decay, "weight_decay": float(candidate["weight_decay"])},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=float(candidate["lr"]), betas=tuple(tcfg["betas"]), eps=float(tcfg["eps"]))


@torch.no_grad()
def logits_of(model: LSTMH5, idx: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    """Eval-mode logits in fixed chunks of ``batch`` windows (memory bound; windows are independent)."""
    model.eval()
    if not len(idx):
        return torch.empty(0, K)
    return torch.cat([model(idx[i:i + batch]) for i in range(0, len(idx), batch)])


def mean_nll(model: LSTMH5, idx: torch.Tensor, y: torch.Tensor) -> float:
    z = logits_of(model, idx).double()
    return float(-z.log_softmax(dim=1)[torch.arange(len(y)), y].mean())


def predict_proba(model: LSTMH5, ids: np.ndarray) -> np.ndarray:
    """(n, 15) float64 probabilities; evaluated once per unique history window, eval mode
    (exact: a stateless window's output depends only on its own 5 ids)."""
    cells, _, _, inverse = aggregate_cells(validate_ids(ids))
    z = logits_of(model, torch.from_numpy(cells)).double().numpy()
    return softmax64(z)[inverse]


def model_from_state(candidate: dict, state: dict) -> LSTMH5:
    m = LSTMH5(candidate["hidden"], candidate["layers"], candidate["dropout"])
    m.load_state_dict(state)
    m.eval()
    return m


def global_grad_norm(model: nn.Module) -> float:
    """||g||_2 over all parameters (non-finite iff some gradient entry is non-finite)."""
    norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms), 2))


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
    """Train one LSTM with early stopping (same rule as 6B); epoch-boundary checkpoints in
    ``ckpt_path`` (resumed if present and compatible). ``interrupt_after`` (tests): raise after
    that many epochs. ``X``: (n, 5) int64 history ids."""
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
        history.append({"epoch": 0, "train_loss": None, "stop_log_loss": mean_nll(model, xst, yst), "seconds": 0.0,
                        "grad_norm_max": None, "grad_norm_mean": None})
        best = history[0]["stop_log_loss"]            # the initial model is the incumbent (best_epoch = 0)
    bs, n = int(tcfg["batch_size"]), len(ytr)
    reason = "max_epochs"
    if bad >= int(tcfg["patience"]):
        reason = "patience"
    for epoch in range(start_epoch, int(tcfg["max_epochs"]) if reason == "max_epochs" else start_epoch):
        te = time.perf_counter()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(derive_seed(seed, 1, epoch)))
        torch.manual_seed(derive_seed(seed, 2, epoch))                   # dropout masks of this epoch
        np.random.seed(derive_seed(seed, 3, epoch))
        random.seed(derive_seed(seed, 4, epoch))
        model.train()
        loss_sum, gmax, gsum, nb = 0.0, 0.0, 0.0, 0
        above = [0] * len(GRAD_NORM_THRESHOLDS)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(model(xtr[b]), ytr[b])                # logits in; log-softmax inside
            if not torch.isfinite(loss):
                raise NumericalError(f"{key}: non-finite loss at epoch {epoch + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g = global_grad_norm(model)
            if not np.isfinite(g):
                raise NumericalError(f"{key}: non-finite gradient at epoch {epoch + 1}")
            opt.step()
            loss_sum += loss.detach().item() * len(b)                    # row-weighted (final batch smaller)
            gmax, gsum, nb = max(gmax, g), gsum + g, nb + 1
            for j, thr in enumerate(GRAD_NORM_THRESHOLDS):
                above[j] += int(g > thr)
        stop_ll = mean_nll(model, xst, yst)
        if not np.isfinite(stop_ll) or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise NumericalError(f"{key}: non-finite parameters or stopping loss at epoch {epoch + 1}")
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / n, "stop_log_loss": stop_ll,
                        "seconds": time.perf_counter() - te, "grad_norm_max": gmax, "grad_norm_mean": gsum / nb,
                        "n_batches": nb, **{f"n_grad_norm_above_{t:g}": a for t, a in zip(GRAD_NORM_THRESHOLDS, above)}})
        if stop_ll < best - float(tcfg["min_delta"]):
            best, best_epoch, bad, best_state = stop_ll, epoch + 1, 0, state_copy(model)
        else:
            bad += 1
        if ckpt_path is not None:   # epoch boundary: everything needed to continue exactly
            save_atomic(ckpt_path, {"identifiers": ident, "epoch": epoch + 1, "model": model.state_dict(),
                                    "optimizer": opt.state_dict(), "best": best, "best_epoch": best_epoch, "bad": bad,
                                    "best_state": best_state, "history": history, "torch_rng": torch.get_rng_state(),
                                    "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                                    "batch_order": "torch.randperm(n, Generator.manual_seed(derive_seed(seed, 1, epoch)))"})
        if interrupt_after is not None and epoch + 1 == interrupt_after:
            raise Interrupted(f"{key}: simulated interruption after epoch {epoch + 1}")
        if bad >= int(tcfg["patience"]):
            reason = "patience"
            break
    if best_epoch == 0:
        best = history[0]["stop_log_loss"]
    return FitResult(key, candidate, seed, best_state, best_epoch, len(history) - 1, reason, best, history,
                     int(n), int(len(yst)), time.perf_counter() - t0, resumed)


# ============================================================================ nested CV
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
                  log=None, only_folds: list[int] | None = None) -> NestedResult:
    """Nested match-grouped CV of experiment 6B with the LSTM candidates of ``cfg['lstm']``.
    Seeds: fit (base, outer, inner + 1 | 0 for refits, candidate index, replicate); stopping
    subsets (base, 7, outer, inner + 1 | 0) -- identical derivation to 6B.

    ``only_folds`` (worker processes): compute only these outer folds (each still uses all
    other folds as its inner folds) and commit them as ``fold_complete.pt``; the other rows
    stay without prediction. Every outer fold is a deterministic function of (data,
    configuration, seeds), so computing folds in separate processes gives the same result
    as computing them sequentially."""
    tcfg, cands = cfg["training"], cfg["lstm"]["candidates"]
    configure_torch(tcfg)
    base = int(tcfg["seed"])
    n_rep = 1 + int(tcfg["robustness_seeds"])
    ident = identifiers or {}
    X = torch.from_numpy(validate_ids(ids_all))
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
        if set(train_m) & set(stop_m):
            raise LeakageError(f"{key}: a match is both a training and a stopping match")
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
        if log:
            log(f"{key}: best epoch {r.best_epoch}/{r.epochs_trained} ({r.stop_reason}), stop LL {r.best_stop_log_loss:.6f}, "
                f"{r.seconds:.0f}s")
        return r

    def record(r: FitResult, fold_id, stage, held, extra=None):
        trained = r.history[1:]
        res.fits.append({"fold_id": fold_id, "stage": stage, "held_inner_fold": held, "candidate": r.candidate["name"],
                         "seed": r.seed, "best_epoch": r.best_epoch, "epochs_trained": r.epochs_trained,
                         "stop_reason": r.stop_reason, "hit_max_epochs": r.epochs_trained == int(tcfg["max_epochs"]),
                         "best_stop_log_loss": r.best_stop_log_loss,
                         "final_train_loss": r.history[-1]["train_loss"], "n_train_rows": r.n_train_rows,
                         "n_stop_rows": r.n_stop_rows, "seconds": r.seconds, "resumed_from_epoch": r.resumed_from_epoch,
                         "grad_norm_max": max((h["grad_norm_max"] for h in trained), default=None),
                         "n_batches": sum(h["n_batches"] for h in trained),
                         **{f"n_grad_norm_above_{t:g}": sum(h[f"n_grad_norm_above_{t:g}"] for h in trained)
                            for t in GRAD_NORM_THRESHOLDS},
                         **(extra or {})})
        res.histories.append({"fold_id": fold_id, "stage": stage, "held_inner_fold": held, "candidate": r.candidate["name"],
                              "seed": r.seed, "history": json.dumps(r.history)})

    if only_folds is not None and not set(only_folds) <= {f.fold_id for f in folds}:
        raise ValueError(f"unknown outer folds {only_folds}")
    for f in folds:
        outer_val = set(f.validation_match_ids)
        if set(f.train_match_ids) & outer_val:
            raise LeakageError(f"fold {f.fold_id}: match in training and validation")
        if only_folds is not None and f.fold_id not in only_folds:
            continue
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
                                     "pooled_inner_log_loss": nll / cnt, "inner_validation_rows": cnt,
                                     "seconds": time.perf_counter() - tc})
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
                  **{k: v for k, v in cand.items() if k != "name"},
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
    if only_folds is None and (np.isnan(res.proba).any() or any(np.isnan(p).any() for p in res.proba_replicates.values())):
        raise RuntimeError("rows without prediction")
    return res


# ============================================================================ independent NumPy reference
def _sigmoid(a: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (no overflow for large |a|)."""
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    e = np.exp(a[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def numpy_lstm_forward(params: dict[str, np.ndarray], x: np.ndarray, layers: int, return_states: bool = False):
    r"""Independent float64 implementation of the LSTM equations (never calls ``nn.LSTM``).

    ``params``: PyTorch-named arrays (``lstm.weight_ih_l{k}``, ``lstm.weight_hh_l{k}``,
    ``lstm.bias_ih_l{k}``, ``lstm.bias_hh_l{k}``, ``out.weight``, ``out.bias``); ``x``:
    (n, 5, m) input. Gate blocks are split in PyTorch order (i, f, g, o) and the effective
    bias is ``bias_ih + bias_hh``. Returns (logits, probabilities[, per-layer states])."""
    n, S, _ = x.shape
    inp = x
    states = []
    for k in range(layers):
        Wih, Whh = params[f"lstm.weight_ih_l{k}"], params[f"lstm.weight_hh_l{k}"]
        b = params[f"lstm.bias_ih_l{k}"] + params[f"lstm.bias_hh_l{k}"]
        H = Whh.shape[1]
        W = dict(zip(GATES, np.split(Wih, 4, axis=0)))
        U = dict(zip(GATES, np.split(Whh, 4, axis=0)))
        bb = dict(zip(GATES, np.split(b, 4)))
        h, c = np.zeros((n, H)), np.zeros((n, H))
        rec = {g: np.zeros((n, S, H)) for g in ("i", "f", "g", "o", "c", "h")}
        for s in range(S):
            xs = inp[:, s, :]
            i = _sigmoid(xs @ W["input"].T + h @ U["input"].T + bb["input"])
            f = _sigmoid(xs @ W["forget"].T + h @ U["forget"].T + bb["forget"])
            g = np.tanh(xs @ W["candidate"].T + h @ U["candidate"].T + bb["candidate"])
            o = _sigmoid(xs @ W["output"].T + h @ U["output"].T + bb["output"])
            c = f * c + i * g
            h = o * np.tanh(c)
            for name, v in (("i", i), ("f", f), ("g", g), ("o", o), ("c", c), ("h", h)):
                rec[name][:, s, :] = v
        states.append(rec)
        inp = rec["h"]
    z = inp[:, -1, :] @ params["out.weight"].T + params["out.bias"]
    p = softmax64(z)
    return (z, p, states) if return_states else (z, p)


def numpy_nll(z: np.ndarray, y: np.ndarray) -> float:
    """Mean of -z_y + logsumexp(z), stable."""
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return float((lse - z[np.arange(len(y)), y]).mean())


__all__ = ["FEATURE_SET", "N_INPUT", "N_POSITIONS", "GATES", "validate_ids", "one_hot_sequence", "LSTMH5",
           "parameter_count_formula", "count_parameters", "configure_torch", "derive_seed", "build_model", "make_optimizer",
           "logits_of", "mean_nll", "predict_proba", "model_from_state", "global_grad_norm", "train_fit", "FitResult",
           "stopping_split", "run_nested_cv", "NestedResult", "numpy_lstm_forward", "numpy_nll", "save_atomic",
           "load_checked", "Interrupted", "NumericalError", "CheckpointMismatch"]
