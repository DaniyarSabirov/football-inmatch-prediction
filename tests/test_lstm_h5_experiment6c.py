"""Tests for experiment 6C (LSTM-H5): input encoding, LSTM equations against an independent NumPy
reference, gradients, statelessness, training / early stopping, checkpoint resume, nested-CV
isolation, adversarial leakage checks and saved results.

Synthetic data throughout (an implementation check, never evidence about football); real-data
checks run only when local caches / results exist."""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import log_loss as sk_log_loss

from src.sequence_models import lstm_h5_experiment as lm
from src.sequence_models import mlp_h5_experiment as mh
from src.sequence_models import xgboost_experiment as xe
from src.sequence_models.canonical_task import CLASSES, build_canonical_table
from src.sequence_models.event_history import PAD, HistoryError, HistoryRows, history_ids
from src.sequence_models.splits import FoldAssignment, LeakageError, assert_no_final_test
from src.sequence_models.xgboost_h2_experiment import CheckpointMismatch
from src.utils.paths import PROJECT_ROOT

CONFIG = PROJECT_ROOT / "configs" / "lstm_h5_experiment6c_v1.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
TCFG = {**CFG["training"], "max_epochs": 12, "patience": 3, "batch_size": 128, "robustness_seeds": 1}
SMALL = {**CFG, "training": TCFG, "lstm": {**CFG["lstm"], "candidates": [
    {"name": "s", "hidden": 8, "layers": 1, "dropout": 0.0, "weight_decay": 0.0, "lr": 1e-2},
    {"name": "m", "hidden": 8, "layers": 2, "dropout": 0.1, "weight_decay": 1e-4, "lr": 1e-2}]}}
DATASET = PROJECT_ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
RESULTS = PROJECT_ROOT / CFG["output_dir"]
C = {c: i for i, c in enumerate(CLASSES)}
lm.configure_torch(CFG["training"])
CANONICAL_ORDER = ["Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed", "Pressure",
                   "Duel", "Interception", "Block", "Goal Keeper", "Foul Committed", "50/50"]


def raw_records(match_id, types, period=1, start=0, h360=None):
    out = []
    for j, typ in enumerate(types):
        i = start + j
        out.append({"match_id": match_id, "list_position": i, "index": i + 1, "event_id": f"{match_id}-{i}", "period": period,
                    "timestamp": f"00:00:{j:06.3f}", "t_seconds": float(j), "type": typ,
                    "has_360": True if h360 is None else h360[j], "has_freeze_frame": True if h360 is None else h360[j],
                    "subtype": "Shot Saved" if typ == "Goal Keeper" else None, "outcome": None, "related_events": ""})
    return out


def synthetic(n_matches=10, n_events=400, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for m in range(n_matches):
        s = [int(rng.integers(0, 15)), int(rng.integers(0, 15))]
        for _ in range(n_events - 2):
            s.append((s[-1] + s[-2]) % 15 if rng.random() < 0.7 else int(rng.integers(0, 15)))
        recs += raw_records(m, [CLASSES[c] for c in s], h360=list(rng.random(n_events) > 0.1))
    t = build_canonical_table(pd.DataFrame(recs))
    rows = HistoryRows.from_table(t)
    ids = sorted(set(rows.match_id))
    folds = [FoldAssignment(f, tuple(m for m in ids if m % 5 != f), tuple(m for m in ids if m % 5 == f)) for f in range(5)]
    return t, rows, folds


@pytest.fixture(scope="module")
def synth():
    return synthetic()


@pytest.fixture(scope="module")
def nested(synth):
    _, rows, folds = synth
    trace: list = []
    return lm.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, trace=trace), trace


def tiny(n=600, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 15, (n, 5))
    ids[:, :2] = np.where(rng.random((n, 2)) < 0.1, PAD, ids[:, :2])
    return ids, (ids[:, 3] + ids[:, 4]) % 15


def model64(hidden=6, layers=1, seed=0, dropout=0.0):
    torch.manual_seed(seed)
    m = lm.LSTMH5(hidden, layers, dropout).double().eval()
    with torch.no_grad():               # larger than default weights so that every gate is far from trivial
        for p in m.parameters():
            p.uniform_(-0.8, 0.8)
    return m


# ----------------------------------------------------------------------------- configuration / vocabulary / encoding
def test_frozen_configuration():
    assert CFG["feature_set"] == {"name": "LSTM-H5", "n_positions": 5, "input_dim": 16, "representation": "one-hot"}
    cands = CFG["lstm"]["candidates"]
    assert len(cands) == 8 and [c["name"] for c in cands] == list("ABCDEFGH")
    assert all(c["layers"] in (1, 2) and c["hidden"] in (32, 64, 128) for c in cands)
    assert CFG["lstm"]["bidirectional"] is False and CFG["lstm"]["builtin_interlayer_dropout"] == 0.0
    t, t6b = CFG["training"], yaml.safe_load((PROJECT_ROOT / CFG["experiment6b_config"]).read_text())["training"]
    for key in ("optimizer", "betas", "eps", "batch_size", "max_epochs", "patience", "min_delta", "stopping_fraction", "seed",
                "robustness_seeds", "deterministic_algorithms", "device"):
        assert t[key] == t6b[key], key                                  # identical training protocol to MLP-H5
    assert t["gradient_clipping"] is None
    assert CFG["evaluation"]["bootstrap_seed"] == 7 and CFG["evaluation"]["bootstrap_replicates"] == 1000
    assert "class_weight" not in json.dumps(CFG["lstm"]) and "label_smoothing" not in json.dumps(CFG)


def test_vocabulary_and_output_class_order():
    assert list(CLASSES) == CANONICAL_ORDER and len(CLASSES) == 15
    assert PAD == 15 and lm.N_INPUT == 16 and lm.N_POSITIONS == 5
    m = lm.LSTMH5(8)
    assert m.out.out_features == 15 and m.lstm.input_size == 16
    assert m(torch.tensor([[PAD, PAD, PAD, PAD, 0]])).shape == (1, 15)          # <START> is input only, never an output


def test_input_tensor_is_chronological_one_hot_with_start():
    ids = np.array([[PAD, PAD, PAD, PAD, C["Pass"]], [PAD, PAD, PAD, C["Pass"], C["Carry"]],
                    [C["Shot"], C["Pass"], C["Carry"], C["Pressure"], C["Duel"]]])
    m = lm.LSTMH5(4)
    x = m.encode(torch.from_numpy(ids))
    assert x.shape == (3, 5, 16) and x.dtype == torch.float32
    assert torch.equal(x.sum(-1), torch.ones(3, 5))                              # exactly one active symbol per position
    assert x.argmax(-1).tolist() == ids.tolist()                                 # position s carries E_{t-5+s}, oldest first
    assert x[0, :4, 15].tolist() == [1, 1, 1, 1] and x[0, 4, 0] == 1             # <START> = column 15
    assert np.array_equal(lm.one_hot_sequence(ids), x.numpy().astype(float))
    for bad in (np.array([[0, 0, 0, 0, PAD]]), np.array([[16, 0, 0, 0, 0]]), np.array([[0, 0, 0, 0]]),
                np.array([[-1, 0, 0, 0, 0]]), np.array([[0.0, 0, 0, 0, 0]])):
        with pytest.raises(HistoryError):
            lm.validate_ids(bad)


def test_parameter_counts_formula_equals_pytorch():
    for c in CFG["lstm"]["candidates"]:
        m = lm.build_model(c, 0)
        f = lm.parameter_count_formula(c["hidden"], c["layers"])
        assert lm.count_parameters(m) == f["total"]
        assert sum(p.numel() for n, p in m.named_parameters() if n.startswith("lstm")) == sum(f["lstm_layers"])
    assert lm.parameter_count_formula(64, 1)["total"] == 4 * 64 * 16 + 4 * 64 ** 2 + 8 * 64 + 15 * 64 + 15 == 21967
    assert lm.parameter_count_formula(32, 2)["lstm_layers"] == [4 * 32 * 16 + 4 * 32 ** 2 + 8 * 32, 4 * 32 * 32 + 4 * 32 ** 2 + 8 * 32]


def test_invalid_configuration_rejected():
    for args in ((0, 1, 0.0), (8, 3, 0.0), (8, 1, 1.0), (8, 1, -0.1)):
        with pytest.raises(ValueError):
            lm.LSTMH5(*args)
    with pytest.raises(ValueError):
        lm.configure_torch({**CFG["training"], "device": "mps"})
    lm.configure_torch(CFG["training"])


# ----------------------------------------------------------------------------- equations vs independent NumPy reference
@pytest.mark.parametrize("layers", [1, 2])
def test_numpy_reference_equals_pytorch_lstm_states_logits_probabilities(layers):
    m = model64(hidden=7, layers=layers, seed=layers)
    ids, _ = tiny(64, 3)
    idx = torch.from_numpy(ids)
    params = m.export()
    z_np, p_np, st = lm.numpy_lstm_forward(params, lm.one_hot_sequence(ids), layers, return_states=True)
    with torch.no_grad():
        out, (h_n, c_n) = m.lstm(m.encode(idx))
        z = m(idx)
    assert np.abs(z.numpy() - z_np).max() < 1e-12
    assert np.abs(torch.softmax(z, 1).numpy() - p_np).max() < 1e-12
    assert np.abs(out.numpy() - st[-1]["h"]).max() < 1e-12                      # h_s of the last layer, every s
    for k in range(layers):
        assert np.abs(h_n[k].numpy() - st[k]["h"][:, -1]).max() < 1e-12
        assert np.abs(c_n[k].numpy() - st[k]["c"][:, -1]).max() < 1e-12
    for s in range(1, 6):                                                        # c_s via prefixes of length s
        with torch.no_grad():
            _, (_, c_pref) = m.lstm(m.encode(idx[:, :s]))
        assert np.abs(c_pref[-1].numpy() - st[-1]["c"][:, s - 1]).max() < 1e-12
    o_rec = out.numpy() / np.tanh(st[-1]["c"])                                   # o_s = h_s / tanh(c_s) from PyTorch's outputs
    ok = np.abs(np.tanh(st[-1]["c"])) > 1e-3
    assert np.abs(o_rec[ok] - st[-1]["o"][ok]).max() < 1e-9


def test_gate_order_and_effective_bias_are_detectable():
    """The reference is sensitive to the gate order and to the bias convention, so the
    equality above would fail if the blocks were exchanged or only one bias were used."""
    m = model64(hidden=5, seed=4)
    ids, _ = tiny(50, 5)
    x = lm.one_hot_sequence(ids)
    with torch.no_grad():
        z = m(torch.from_numpy(ids)).numpy()
    p = m.export()
    for perm in ([1, 0, 2, 3], [0, 1, 3, 2], [3, 1, 2, 0], [0, 2, 1, 3]):
        q = dict(p)
        for name in ("lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0"):
            q[name] = np.concatenate([np.split(p[name], 4)[j] for j in perm])
        assert np.abs(lm.numpy_lstm_forward(q, x, 1)[0] - z).max() > 1e-3
    q = {**p, "lstm.bias_hh_l0": np.zeros_like(p["lstm.bias_hh_l0"])}            # bias_ih only -> different
    assert np.abs(lm.numpy_lstm_forward(q, x, 1)[0] - z).max() > 1e-3
    assert np.abs(lm.numpy_lstm_forward(p, x, 1)[0] - z).max() < 1e-12


def test_reference_cell_gates_equal_numpy_gates():
    """Explicit reference cell written with torch primitives (sigmoid / tanh), same parameters:
    gates i, f, g, o and states agree with the NumPy reference, and its h_s / c_s agree with nn.LSTM."""
    m = model64(hidden=6, seed=7)
    ids, _ = tiny(40, 9)
    x = m.encode(torch.from_numpy(ids))
    Wi, Wf, Wg, Wo = m.lstm.weight_ih_l0.chunk(4, 0)
    Ui, Uf, Ug, Uo = m.lstm.weight_hh_l0.chunk(4, 0)
    bi, bf, bg, bo = (m.lstm.bias_ih_l0 + m.lstm.bias_hh_l0).chunk(4, 0)
    _, _, st = lm.numpy_lstm_forward(m.export(), lm.one_hot_sequence(ids), 1, return_states=True)
    h = torch.zeros(40, 6, dtype=torch.float64)
    c = torch.zeros(40, 6, dtype=torch.float64)
    with torch.no_grad():
        out, _ = m.lstm(x)
        for s in range(5):
            xs = x[:, s]
            i = torch.sigmoid(xs @ Wi.T + h @ Ui.T + bi)
            f = torch.sigmoid(xs @ Wf.T + h @ Uf.T + bf)
            g = torch.tanh(xs @ Wg.T + h @ Ug.T + bg)
            o = torch.sigmoid(xs @ Wo.T + h @ Uo.T + bo)
            c = f * c + i * g
            h = o * torch.tanh(c)
            for name, v in (("i", i), ("f", f), ("g", g), ("o", o), ("c", c), ("h", h)):
                assert np.abs(v.numpy() - st[0][name][:, s]).max() < 1e-13, (name, s)
            assert torch.allclose(h, out[:, s], atol=1e-13, rtol=0)


def test_final_hidden_state_extraction():
    for layers in (1, 2):
        m = model64(hidden=5, layers=layers, seed=layers)
        idx = torch.from_numpy(tiny(30)[0])
        with torch.no_grad():
            out, (h_n, _) = m.lstm(m.encode(idx))
            assert torch.equal(m.final_hidden(idx), out[:, -1]) and torch.equal(m.final_hidden(idx), h_n[-1])
            assert torch.allclose(m(idx), m.out(out[:, -1]))
        if layers == 2:
            assert not torch.allclose(h_n[0], h_n[1])                            # layer 1 state is not the output


def test_stable_sigmoid_and_nll_reference():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = lm._sigmoid(np.array([-1000.0, -30.0, 0.0, 30.0, 1000.0]))
    assert s[0] == 0.0 and s[-1] == 1.0 and s[2] == 0.5 and np.all(np.isfinite(s))
    z = np.random.default_rng(0).normal(size=(50, 15)) * 30
    y = np.random.default_rng(1).integers(0, 15, 50)
    assert abs(lm.numpy_nll(z, y) - float(F.cross_entropy(torch.from_numpy(z), torch.from_numpy(y)))) < 1e-12


# ----------------------------------------------------------------------------- gradients / BPTT
@pytest.mark.parametrize("layers", [1, 2])
def test_autograd_equals_finite_differences_of_numpy_reference(layers):
    """Central differences (h = 1e-6, float64) of the INDEPENDENT NumPy loss against PyTorch's
    autograd gradient of F.cross_entropy, for every parameter tensor. Truncation error is
    O(h^2) ~ 1e-12 and rounding error ~ 1e-16 / h ~ 1e-10, hence the tolerance 1e-7."""
    m = model64(hidden=4, layers=layers, seed=11 + layers)
    ids, y = tiny(40, 13)
    x = lm.one_hot_sequence(ids)
    m.zero_grad()
    F.cross_entropy(m(torch.from_numpy(ids)), torch.from_numpy(y)).backward()
    grads = {n: p.grad.numpy().copy() for n, p in m.named_parameters()}
    params = m.export()
    rng = np.random.default_rng(layers)
    err, checked = 0.0, set()
    for name in params:
        flat = params[name].reshape(-1)
        for j in rng.choice(flat.size, min(6, flat.size), replace=False):
            old = flat[j]
            flat[j] = old + 1e-6
            lp = lm.numpy_nll(lm.numpy_lstm_forward(params, x, layers)[0], y)
            flat[j] = old - 1e-6
            lmn = lm.numpy_nll(lm.numpy_lstm_forward(params, x, layers)[0], y)
            flat[j] = old
            err = max(err, abs((lp - lmn) / 2e-6 - grads[name].reshape(-1)[j]))
        checked.add(name.split(".")[-1].rstrip("0123456789").rstrip("_l") if name.startswith("lstm") else name)
    assert err < 1e-7
    assert {"weight_ih", "weight_hh", "bias_ih", "bias_hh", "out.weight", "out.bias"} <= checked


def test_backpropagation_reaches_every_history_position():
    """The gradient of the loss with respect to the input reaches all five positions, including
    the oldest one (no detach / truncation inside the window)."""
    for layers in (1, 2):
        m = model64(hidden=6, layers=layers, seed=3)
        ids, y = tiny(30, 2)
        x = m.encode(torch.from_numpy(ids)).requires_grad_(True)
        _, (h_n, _) = m.lstm(x)
        F.cross_entropy(m.out(h_n[-1]), torch.from_numpy(y)).backward()
        assert all(float(x.grad[:, s].abs().sum()) > 1e-8 for s in range(5))
        assert "detach" not in lm.LSTMH5.forward.__code__.co_names + lm.LSTMH5.final_hidden.__code__.co_names


def test_cross_entropy_on_logits_and_single_softmax():
    m = lm.build_model({"hidden": 8, "layers": 1, "dropout": 0.0}, 0)
    ids, y = tiny(64)
    idx, yt = torch.from_numpy(ids), torch.from_numpy(y)
    z = m(idx)
    loss = F.cross_entropy(z, yt)
    assert torch.allclose(loss, -(z - z.logsumexp(1, keepdim=True))[torch.arange(64), yt].mean(), atol=1e-6)
    assert not torch.allclose(loss, F.cross_entropy(torch.softmax(z, 1), yt), atol=1e-3)   # double softmax would differ
    p = lm.predict_proba(m, ids)
    assert np.abs(p - xe.softmax64(lm.logits_of(m, idx).double().numpy())).max() < 1e-15   # softmax applied exactly once
    assert np.abs(p.sum(1) - 1).max() < 1e-12 and (p > 0).all() and np.isfinite(p).all()
    assert not np.allclose(p, xe.softmax64(p))


def test_optimizer_groups_and_decoupled_weight_decay():
    m = lm.build_model({"hidden": 8, "layers": 2, "dropout": 0.0}, 0)
    opt = lm.make_optimizer(m, {"lr": 1e-2, "weight_decay": 0.1}, CFG["training"])
    named = dict(m.named_parameters())
    decay = {n for n, p in named.items() if any(p is q for q in opt.param_groups[0]["params"])}
    assert decay == {"lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.weight_ih_l1", "lstm.weight_hh_l1", "out.weight"}
    assert opt.param_groups[1]["weight_decay"] == 0.0 and len(opt.param_groups[1]["params"]) == 5   # 4 LSTM biases + out.bias
    before = {n: p.detach().clone() for n, p in named.items()}
    for p in m.parameters():                       # zero gradient: Adam term vanishes, only w <- w - lr * wd * w remains
        p.grad = torch.zeros_like(p)
    opt.step()
    for n, p in named.items():
        expected = before[n] * (1 - 1e-2 * 0.1) if n in decay else before[n]
        assert torch.allclose(p.detach(), expected, atol=1e-7, rtol=0), n
    ids, y = tiny(64)
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(m(torch.from_numpy(ids)), torch.from_numpy(y)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    snap = [p.detach().clone() for p in m.parameters()]
    opt.step()
    assert all(not torch.equal(a, b) for a, b in zip(snap, m.parameters()))


def test_adamw_update_with_nonzero_gradients_equals_closed_form():
    """Regression test (final audit): the zero-gradient test above checks only the decay term.
    Here two real AdamW steps are compared with the closed form
    theta <- theta (1 - eta lambda) - eta m_hat / (sqrt(v_hat) + eps), with bias correction at
    q = 1 and q = 2, and lambda = 0 for biases."""
    cand = {"name": "c", "hidden": 8, "layers": 2, "dropout": 0.0, "weight_decay": 0.05, "lr": 1e-2}
    m = lm.build_model(cand, 4)
    opt = lm.make_optimizer(m, cand, CFG["training"])
    (b1, b2), eps, lr, wd = CFG["training"]["betas"], CFG["training"]["eps"], cand["lr"], cand["weight_decay"]
    ids, y = tiny(128, 6)
    mom = {n: 0.0 for n, _ in m.named_parameters()}
    vel = dict(mom)
    for q in (1, 2):
        theta = {n: p.detach().double().clone() for n, p in m.named_parameters()}
        opt.zero_grad()
        F.cross_entropy(m(torch.from_numpy(ids[64 * (q - 1):64 * q])), torch.from_numpy(y[64 * (q - 1):64 * q])).backward()
        grads = {n: p.grad.detach().double().clone() for n, p in m.named_parameters()}
        opt.step()
        for n, p in m.named_parameters():
            mom[n] = b1 * mom[n] + (1 - b1) * grads[n]
            vel[n] = b2 * vel[n] + (1 - b2) * grads[n] ** 2
            lam = wd if n.rsplit(".", 1)[-1].startswith("weight") else 0.0
            expected = theta[n] * (1 - lr * lam) - lr * (mom[n] / (1 - b1 ** q)) / ((vel[n] / (1 - b2 ** q)).sqrt() + eps)
            assert torch.allclose(p.detach().double(), expected, atol=1e-6, rtol=0), (n, q)


def test_dropout_semantics():
    m = lm.build_model({"hidden": 64, "layers": 2, "dropout": 0.5}, 1)
    assert m.lstm.dropout == 0.0 and m.drop.p == 0.5 and m.lstm.bidirectional is False and m.lstm.batch_first
    idx = torch.from_numpy(tiny(20)[0])
    m.train()
    assert not torch.equal(m(idx), m(idx))                                       # output dropout active in training
    assert torch.equal(m.final_hidden(idx), m.final_hidden(idx))                 # no hidden dropout inside the recurrence
    m.eval()
    assert torch.equal(m(idx), m(idx))
    p = lm.predict_proba(m, tiny(20)[0])
    m.train()
    assert np.array_equal(lm.predict_proba(m, tiny(20)[0]), p)                   # predict_proba switches to eval itself
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                           # no "dropout has no effect" warning
        lm.LSTMH5(8, 1, 0.1)


# ----------------------------------------------------------------------------- statelessness
def test_stateless_windows_batch_independence_and_order():
    m = model64(hidden=8, layers=2, seed=5)
    ids, _ = tiny(200, 8)
    idx = torch.from_numpy(ids)
    with torch.no_grad():
        full = m(idx)
        alone = torch.cat([m(idx[i:i + 1]) for i in range(200)])
        perm = torch.randperm(200, generator=torch.Generator().manual_seed(0))
        permuted = m(idx[perm])
        again = m(idx)                                                           # no state carried between calls
        other = idx.clone()
        other[1:] = torch.from_numpy(tiny(199, 99)[0])                           # all OTHER windows changed
        changed = m(other)
    assert torch.allclose(full, alone, atol=1e-12, rtol=0)
    assert torch.allclose(full[perm], permuted, atol=1e-12, rtol=0)
    assert torch.equal(full, again)
    assert torch.allclose(changed[0], full[0], atol=1e-12, rtol=0)
    m32 = lm.build_model({"hidden": 16, "layers": 1, "dropout": 0.0}, 3)
    p1 = lm.predict_proba(m32, ids)
    p2 = np.vstack([lm.predict_proba(m32, ids[i:i + 13]) for i in range(0, 200, 13)])
    z = lm.logits_of(m32, idx, batch=7).double().numpy()
    assert np.abs(p1 - p2).max() < 1e-6 and np.abs(p1 - xe.softmax64(z)).max() < 1e-6   # unique-cell evaluation is exact


def test_history_has_no_future_or_cross_boundary_information(synth):
    t, rows, _ = synth
    full = history_ids(t, 4)
    later = t.copy()
    later.loc[later.index[700:], "symbol_id"] = 0                                # mutate future events
    assert np.array_equal(history_ids(later, 4)[:700], full[:700])
    assert np.array_equal(history_ids(t.assign(target_id=0, event_id="x", match_id=-1), 4), full)
    p1 = raw_records(5, ["Pass", "Carry", "Shot"])
    p2 = raw_records(5, ["Clearance", "Pass"], period=2, start=3)
    p3 = raw_records(6, ["Duel", "Pass", "Carry"])
    r = HistoryRows.from_table(build_canonical_table(pd.DataFrame(p1 + p2 + p3)))
    starts = r.ctx[r.seq_pos == 0]
    assert starts.tolist() == [[PAD] * 4 + [C["Pass"]], [PAD] * 4 + [C["Clearance"]], [PAD] * 4 + [C["Duel"]]]
    assert r.ctx[r.seq_pos == 1].tolist()[0] == [PAD, PAD, PAD, C["Pass"], C["Carry"]]
    assert (r.ctx[:, -1] != PAD).all()                                           # current event always at the last position
    m = lm.build_model({"hidden": 8, "layers": 1, "dropout": 0.0}, 0)
    lm.predict_proba(m, r.ctx)                                                   # every canonical window is a valid input
    with pytest.raises(LeakageError):
        assert_no_final_test([1, 2], [2])


# ----------------------------------------------------------------------------- synthetic learning checks
def test_tiny_dataset_is_overfit():
    """Implementation sanity check (not a scientific result): a pure interaction is learnt."""
    ids, y = tiny(400)
    X, yt = torch.from_numpy(ids), torch.from_numpy(y)
    r = lm.train_fit("overfit", X, yt, np.arange(400), np.arange(400), {"name": "o", "hidden": 64, "layers": 1,
                     "weight_decay": 0.0, "dropout": 0.0, "lr": 1e-2}, {**TCFG, "max_epochs": 150, "patience": 150, "batch_size": 64}, 0)
    p = lm.predict_proba(lm.model_from_state(r.candidate, r.best_state), ids)
    assert -np.log(p[np.arange(400), y]).mean() < 0.05 and (p.argmax(1) == y).mean() == 1.0


def test_order_dependent_target_is_learnt():
    """Every window is a permutation of the multiset {Pass, Carry, z, z, z}; the label is 1 if
    Pass precedes Carry, else 0. Any order-blind (bag-of-events) model is at chance (log 2); the LSTM
    must resolve the order. A second task copies the OLDEST position (information has to pass
    through four recurrent steps)."""
    rng = np.random.default_rng(0)
    n = 1200
    ids = np.empty((n, 5), dtype=np.int64)
    y = np.empty(n, dtype=np.int64)
    a, b = C["Pass"], C["Carry"]
    for i in range(n):
        z = int(rng.integers(2, 15))
        w = np.array([a, b, z, z, z])[rng.permutation(5)]
        ids[i] = w
        y[i] = int(np.flatnonzero(w == a)[0] < np.flatnonzero(w == b)[0])
    bag = np.zeros((n, 16))
    np.add.at(bag, (np.repeat(np.arange(n), 5), ids.ravel()), 1)
    assert len({(tuple(bag[i]), y[i]) for i in range(n)}) > len({tuple(b) for b in bag})   # same bag, both labels occur
    cand = {"name": "o", "hidden": 32, "layers": 1, "weight_decay": 0.0, "dropout": 0.0, "lr": 1e-2}
    tc = {**TCFG, "max_epochs": 80, "patience": 80, "batch_size": 64}
    r = lm.train_fit("order", torch.from_numpy(ids), torch.from_numpy(y), np.arange(1000), np.arange(1000, n), cand, tc, 0)
    p = lm.predict_proba(lm.model_from_state(cand, r.best_state), ids[1000:])
    assert (p.argmax(1) == y[1000:]).mean() > 0.9                                # held-out windows
    ids2 = rng.integers(0, 15, (1200, 5))
    y2 = ids2[:, 0].copy()                                                       # target = E_{t-4}
    r2 = lm.train_fit("copy", torch.from_numpy(ids2), torch.from_numpy(y2), np.arange(1000), np.arange(1000, 1200), cand, tc, 0)
    p2 = lm.predict_proba(lm.model_from_state(cand, r2.best_state), ids2[1000:])
    assert (p2.argmax(1) == y2[1000:]).mean() > 0.9


# ----------------------------------------------------------------------------- early stopping / determinism / resume
def test_epoch_train_loss_is_row_weighted_with_unequal_final_batch():
    ids, y = tiny(50, 7)
    X, yt = torch.from_numpy(ids), torch.from_numpy(y)
    cand = {"name": "z", "hidden": 8, "layers": 1, "weight_decay": 0.0, "dropout": 0.0, "lr": 0.0}
    r = lm.train_fit("w", X, yt, np.arange(50), np.arange(50), cand, {**TCFG, "batch_size": 7, "max_epochs": 1}, 3)
    m = lm.model_from_state(cand, r.best_state)
    z = lm.logits_of(m, X).double()
    row_mean = float(-z.log_softmax(1)[torch.arange(50), yt].mean())
    batch_means = [float(-z[i:i + 7].log_softmax(1)[torch.arange(len(z[i:i + 7])), yt[i:i + 7]].mean()) for i in range(0, 50, 7)]
    assert abs(r.history[1]["train_loss"] - row_mean) < 1e-6
    assert abs(r.history[1]["stop_log_loss"] - row_mean) < 1e-12
    assert abs(np.mean(batch_means) - row_mean) > 1e-4
    assert r.history[1]["n_batches"] == 8


def test_early_stopping_keeps_best_epoch_state():
    ids, y = tiny(800, 3)
    y = np.where(np.random.default_rng(1).random(800) < 0.5, y, np.random.default_rng(2).integers(0, 15, 800))
    X, yt = torch.from_numpy(ids), torch.from_numpy(y)
    cand = {"name": "big", "hidden": 128, "layers": 1, "weight_decay": 0.0, "dropout": 0.0, "lr": 3e-2}
    r = lm.train_fit("es", X, yt, np.arange(500), np.arange(500, 800), cand, {**TCFG, "max_epochs": 80, "patience": 3}, 0)
    stop = [h["stop_log_loss"] for h in r.history]
    assert r.stop_reason == "patience" and r.epochs_trained == r.best_epoch + 3
    assert r.best_stop_log_loss == min(stop) == stop[r.best_epoch] and r.best_epoch == int(np.argmin(stop))
    m = lm.model_from_state(cand, r.best_state)
    assert abs(lm.mean_nll(m, X[500:], yt[500:]) - r.best_stop_log_loss) < 1e-12   # stored loss <-> stored weights


def test_epoch_zero_model_is_the_incumbent():
    """Training labels 0..6, stopping labels 7..14: every update worsens the stopping loss, so
    the initial model (epoch 0) must be returned."""
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 15, (600, 5))
    y = np.r_[rng.integers(0, 7, 400), rng.integers(7, 15, 200)]
    cand = {"name": "e0", "hidden": 16, "layers": 1, "weight_decay": 0.0, "dropout": 0.0, "lr": 1e-2}
    r = lm.train_fit("e0", torch.from_numpy(ids), torch.from_numpy(y), np.arange(400), np.arange(400, 600), cand,
                     {**TCFG, "max_epochs": 20, "patience": 3}, 5)
    init = lm.build_model(cand, 5).state_dict()
    assert r.best_epoch == 0 and r.epochs_trained == 3 and r.stop_reason == "patience"
    assert all(torch.equal(init[k], r.best_state[k]) for k in init)
    assert r.best_stop_log_loss == r.history[0]["stop_log_loss"] < min(h["stop_log_loss"] for h in r.history[1:])


def test_training_is_bitwise_deterministic():
    ids, y = tiny(500)
    X, yt = torch.from_numpy(ids), torch.from_numpy(y)
    cand = SMALL["lstm"]["candidates"][1]
    a = lm.train_fit("d", X, yt, np.arange(400), np.arange(400, 500), cand, TCFG, 9)
    b = lm.train_fit("d", X, yt, np.arange(400), np.arange(400, 500), cand, TCFG, 9)
    assert all(torch.equal(a.best_state[k], b.best_state[k]) for k in a.best_state)
    strip = [[(h["epoch"], h["train_loss"], h["stop_log_loss"], h["grad_norm_max"]) for h in r.history] for r in (a, b)]
    assert strip[0] == strip[1] and a.best_epoch == b.best_epoch
    c = lm.train_fit("d", X, yt, np.arange(400), np.arange(400, 500), cand, TCFG, 10)
    assert not all(torch.equal(a.best_state[k], c.best_state[k]) for k in a.best_state)


def test_interrupted_and_resumed_fit_equals_uninterrupted_fit(tmp_path):
    ids, y = tiny(700, 4)
    X, yt = torch.from_numpy(ids), torch.from_numpy(y)
    cand = {"name": "r", "hidden": 16, "layers": 2, "weight_decay": 1e-4, "dropout": 0.1, "lr": 1e-2}   # dropout: RNG matters
    cfg = {**TCFG, "max_epochs": 8, "patience": 8}
    ident = {"run": "x"}
    full = lm.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), cand, cfg, 5)
    ck = tmp_path / "k.inprogress.pt"
    with pytest.raises(lm.Interrupted):
        lm.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), cand, cfg, 5, ck, ident, interrupt_after=3)
    saved = torch.load(ck, weights_only=False)
    assert saved["epoch"] == 3 and len(saved["history"]) == 4
    assert {"model", "optimizer", "torch_rng", "numpy_rng", "python_rng", "best_state", "best", "bad", "identifiers"} <= set(saved)
    resumed = lm.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), cand, cfg, 5, ck, ident)
    assert resumed.resumed_from_epoch == 3 and resumed.epochs_trained == full.epochs_trained
    assert all(torch.equal(full.best_state[k], resumed.best_state[k]) for k in full.best_state)
    strip = lambda r: [(h["train_loss"], h["stop_log_loss"], h["grad_norm_max"]) for h in r.history]
    assert strip(full) == strip(resumed)
    with pytest.raises(CheckpointMismatch):                           # different hyperparameters -> refused
        lm.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), {**cand, "lr": 2e-2}, cfg, 5, ck, ident)
    with pytest.raises(CheckpointMismatch):                           # different identifiers -> refused
        lm.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), cand, cfg, 5, ck, {"run": "y"})
    with pytest.raises(CheckpointMismatch):                           # different rows -> refused
        lm.train_fit("k", X, yt, np.arange(590), np.arange(600, 700), cand, cfg, 5, ck, ident)
    ck.write_bytes(ck.read_bytes()[:200])                              # corrupted -> refused
    with pytest.raises(CheckpointMismatch):
        lm.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), cand, cfg, 5, ck, ident)


def test_nonfinite_values_are_detected(monkeypatch):
    ids, y = tiny(100)
    cand = {"name": "n", "hidden": 8, "layers": 1, "weight_decay": 0.0, "dropout": 0.0, "lr": 1e-2}
    real = lm.build_model

    def poisoned(c, s):
        m = real(c, s)
        with torch.no_grad():
            m.out.bias[0] = float("nan")
        return m
    monkeypatch.setattr(lm, "build_model", poisoned)
    with pytest.raises(lm.NumericalError):
        lm.train_fit("nan", torch.from_numpy(ids), torch.from_numpy(y), np.arange(80), np.arange(80, 100), cand, TCFG, 0)

    def nan_gradient(c, s):                                            # finite loss, non-finite gradient
        m = real(c, s)
        m.lstm.weight_hh_l0.register_hook(lambda g: g * float("nan"))
        return m
    monkeypatch.setattr(lm, "build_model", nan_gradient)
    with pytest.raises(lm.NumericalError, match="gradient"):
        lm.train_fit("nan_g", torch.from_numpy(ids), torch.from_numpy(y), np.arange(80), np.arange(80, 100), cand, TCFG, 0)


# ----------------------------------------------------------------------------- nested CV isolation
def test_outer_inner_and_stopping_isolation(synth, nested):
    _, rows, folds = synth
    res, trace = nested
    for f in folds:
        val = set(f.validation_match_ids)
        ts = [t for t in trace if t["outer_fold"] == f.fold_id]
        assert {t["purpose"] for t in ts} == {"inner_train", "inner_stop", "inner_validation", "outer_train", "outer_stop",
                                               "outer_validation_predict"}
        for t in ts:
            if t["purpose"] != "outer_validation_predict":
                assert not (t["match_ids"] & val)
            else:
                assert t["match_ids"] == val
        trains = [t for t in ts if t["purpose"] == "inner_train"]
        stops = [t for t in ts if t["purpose"] == "inner_stop"]
        held = [t for t in ts if t["purpose"] == "inner_validation"]
        for a, b, h in zip(trains, stops, held):
            assert not (a["match_ids"] & b["match_ids"]) and not ((a["match_ids"] | b["match_ids"]) & h["match_ids"])
        assert set(rows.match_id[res.fold_of_row == f.fold_id]) == val
    bad = [FoldAssignment(0, (0, 1), (1, 2))]
    with pytest.raises(LeakageError):
        lm.run_nested_cv(rows.ctx[:20], np.tile([0, 1, 2, 3], 5), rows.y[:20], bad, SMALL)


def test_stopping_subsets_and_seeds_identical_to_experiment_6b(synth):
    """Same seed derivation as MLP-H5: identical match-level stopping subsets per (outer, inner)."""
    _, rows, folds = synth
    base = int(CFG["training"]["seed"])
    for f in folds:
        trm = list(f.train_match_ids)
        assert lm.stopping_split(trm, 0.1, lm.derive_seed(base, 7, f.fold_id, 0)) == \
            mh.stopping_split(trm, 0.1, mh.derive_seed(base, 7, f.fold_id, 0))
    assert lm.derive_seed is mh.derive_seed and lm.stopping_split is mh.stopping_split


def test_adversarial_outer_validation_labels_change_nothing_in_their_fold(synth, nested):
    _, rows, folds = synth
    res, _ = nested
    y2 = rows.y.copy()
    va = res.fold_of_row == 0
    y2[va] = (y2[va] + 5) % 15
    out = lm.run_nested_cv(rows.ctx, rows.match_id, y2, folds, SMALL)
    a, b = res.choices[0], out.choices[0]
    assert a["candidate"] == b["candidate"] and a["inner_log_loss"] == b["inner_log_loss"]
    assert all(torch.equal(res.states[(0, r)][k], out.states[(0, r)][k]) for r in (0, 1) for k in res.states[(0, 0)])
    assert np.array_equal(res.proba[va], out.proba[va])
    strip = lambda r: [(h["stage"], h["held_inner_fold"], h["candidate"], h["best_epoch"], h["best_stop_log_loss"])
                       for h in r.fits if h["fold_id"] == 0]
    assert strip(res) == strip(out)                                    # stopping epochs and losses unaffected
    assert any(not torch.equal(res.states[(f, 0)]["out.weight"], out.states[(f, 0)]["out.weight"]) for f in (1, 2, 3, 4))


def test_selection_score_is_the_pooled_held_inner_log_loss(synth, nested):
    _, rows, folds = synth
    res, _ = nested
    sc = pd.DataFrame(res.inner_scores)
    for f in range(5):
        s = sc[sc.fold_id == f]
        assert res.choices[f]["candidate"] == s.loc[s.pooled_inner_log_loss.idxmin(), "candidate"]
    fl = pd.DataFrame(res.fits)
    inner = fl[(fl.fold_id == 0) & (fl.stage == "inner") & (fl.candidate == "s")]
    pooled = (inner.inner_validation_log_loss * inner.inner_validation_rows).sum() / inner.inner_validation_rows.sum()
    assert abs(pooled - sc[(sc.fold_id == 0) & (sc.candidate == "s")].pooled_inner_log_loss.iat[0]) < 1e-12
    assert set(fl.stage) == {"inner", "outer_refit", "robustness_refit"}
    assert (fl.n_batches > 0).all() and (fl.grad_norm_max > 0).all()


def test_nested_output_probabilities_and_metrics(synth, nested):
    _, rows, _ = synth
    res, _ = nested
    for p in [res.proba, *res.proba_replicates.values()]:
        xe.check_probabilities(p, 1e-9)
        assert (p > 0).all() and p.shape == (len(rows.y), 15)
    cfg = {"evaluation": {"bootstrap_replicates": 40, "bootstrap_seed": 7, "probability_tolerance": 1e-9}}
    uniform = np.full_like(res.proba, 1 / 15)
    t = xe.evaluate_models(rows.y, rows.match_id, res.fold_of_row, {"LSTM-H5": res.proba, "u": uniform}, [("LSTM-H5", "u")], cfg)
    mc = t["model_comparison"].set_index("model")
    assert abs(mc.at["LSTM-H5", "log_loss"] - sk_log_loss(rows.y, res.proba, labels=list(range(15)))) < 1e-12
    d = -np.log(res.proba[np.arange(len(rows.y)), rows.y]) + np.log(1 / 15)
    pdiff = t["paired_differences"].set_index("metric").loc["log_loss"]
    assert abs(pdiff["difference"] - d.mean()) < 1e-12 and pdiff["ci_high"] < 0      # learns the synthetic structure


def test_completed_fits_and_folds_resume_without_retraining(tmp_path, synth, nested):
    _, rows, folds = synth
    res, _ = nested
    ident = {"run": "r1"}
    a = lm.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, tmp_path, ident)
    assert np.array_equal(a.proba, res.proba)                          # checkpointing does not change results
    (tmp_path / "fold4" / "fold_complete.pt").unlink()                 # interrupted inside fold 4 after its fits
    b = lm.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, tmp_path, ident)
    assert b.resumed_folds == [0, 1, 2, 3] and b.computed_folds == [4] and b.resumed_fits > 0
    assert np.array_equal(a.proba, b.proba) and a.choices == b.choices
    with pytest.raises(CheckpointMismatch):
        lm.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, tmp_path, {"run": "r2"})


def test_folds_computed_in_separate_runs_equal_the_sequential_run(tmp_path, synth, nested):
    """Worker mode: each outer fold computed alone (only_folds) and committed; the assembled
    result equals the sequential computation bit for bit."""
    _, rows, folds = synth
    res, trace = nested
    ident = {"run": "par"}
    for k in (3, 0, 4, 1, 2):                                          # arbitrary order, as independent processes would
        part = lm.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, tmp_path, ident, trace=[], only_folds=[k])
        assert part.computed_folds == [k] and np.isnan(part.proba[part.fold_of_row != k]).all()
    tr2: list = []
    full = lm.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, tmp_path, ident, trace=tr2)
    assert full.resumed_folds == [0, 1, 2, 3, 4] and not full.computed_folds
    assert np.array_equal(full.proba, res.proba) and all(np.array_equal(full.proba_replicates[r], res.proba_replicates[r]) for r in (1,))
    assert [c["candidate"] for c in full.choices] == [c["candidate"] for c in res.choices]
    assert all(torch.equal(full.states[k][n], res.states[k][n]) for k in res.states for n in res.states[k])
    assert sorted((t["outer_fold"], t["purpose"], tuple(sorted(t["match_ids"]))) for t in tr2) == \
        sorted((t["outer_fold"], t["purpose"], tuple(sorted(t["match_ids"]))) for t in trace)
    with pytest.raises(ValueError):
        lm.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, only_folds=[7])


def test_supervisor_and_runner_guards_exist():
    sup = (PROJECT_ROOT / "scripts/supervise_lstm_h5_experiment6c.py").read_text()
    run = (PROJECT_ROOT / "scripts/run_lstm_h5_experiment6c.py").read_text()
    assert "start_new_session=True" in sup and '"caffeinate", "-d", "-i", "-w"' in sup and "LOCK_NB" in sup
    assert "COMPLETED" in sup and "exit_code" in sup
    assert "LOCK_NB" in run and ".partial" in run and "staging.rename(out)" in run and "load_inputs(cfg)" in run


# ----------------------------------------------------------------------------- real data
needs_data = pytest.mark.skipif(not DATASET.exists(), reason="local canonical dataset not available")
needs_results = pytest.mark.skipif(not (RESULTS / "integrity_audit.csv").exists(), reason="experiment 6C not run")
needs_6b = pytest.mark.skipif(not (PROJECT_ROOT / "results/mlp_h5_experiment6b_actions15_360current_v1_2026-11-23/"
                                   "data_access_audit.csv").exists(), reason="experiment 6B results not available")


@needs_data
def test_historical_results_unchanged():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_lstm_h5_experiment6c import historical_snapshot
    snap = historical_snapshot(CFG)
    assert all(not e["mismatches"] for e in snap["directories"].values()) and len(snap["directories"]) == 5


@needs_6b
def test_refit_stopping_subsets_equal_the_6b_access_audit():
    import hashlib
    da = pd.read_csv(PROJECT_ROOT / "results/mlp_h5_experiment6b_actions15_360current_v1_2026-11-23/data_access_audit.csv")
    folds = json.loads((PROJECT_ROOT / "results/markov_hmm_cost_sensitive_v2/tiers/tier_b/scenarioD15/folds.json").read_text())["scenarioD15"]
    h = lambda ms: hashlib.sha256(",".join(map(str, sorted(ms))).encode()).hexdigest()
    base = int(CFG["training"]["seed"])
    for f in folds:
        fit_m, stop_m = lm.stopping_split(f["train_match_ids"], 0.10, lm.derive_seed(base, 7, f["fold_id"], 0))
        rec = da[da.outer_fold == f["fold_id"]]
        assert h(fit_m) in set(rec[rec.purpose == "outer_train"].match_ids_sha256)
        assert h(stop_m) in set(rec[rec.purpose == "outer_stop"].match_ids_sha256)


@needs_results
def test_saved_experiment6c_results_are_consistent():
    from src.sequence_models.experiment import sha256_file
    audit = pd.read_csv(RESULTS / "integrity_audit.csv")
    assert not (audit["result"] == "FAIL").any()
    checks = json.loads((RESULTS / "checksums.json").read_text())
    assert all(sha256_file(RESULTS / k) == v for k, v in checks.items())
    pred = pd.read_parquet(PROJECT_ROOT / CFG["artefact_dir"] / "oof_predictions_lstm_h5.parquet")
    p = pred[[f"LSTM-H5__p{c}" for c in range(15)]].to_numpy()
    mc = pd.read_csv(RESULTS / "model_comparison.csv").set_index("model")
    assert len(pred) == 690_147 and pred["event_id"].is_unique
    assert abs(sk_log_loss(pred["y"], p, labels=list(range(15))) - mc.at["LSTM-H5", "log_loss"]) < 1e-9
