"""Tests for experiment 6B (MLP-H5): encoding, network, training, early stopping, checkpoint /
resume, nested-CV isolation, adversarial leakage checks, metrics and saved results.

Synthetic data throughout; real-data checks run only when local caches / results exist."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import log_loss as sk_log_loss

from src.sequence_models import logistic_h5_experiment as lh
from src.sequence_models import mlp_h5_experiment as mh
from src.sequence_models import xgboost_experiment as xe
from src.sequence_models.canonical_task import CLASSES, build_canonical_table
from src.sequence_models.event_history import PAD, HistoryError, HistoryRows, history_ids
from src.sequence_models.splits import FoldAssignment, LeakageError, assert_no_final_test
from src.sequence_models.xgboost_h2_experiment import CheckpointMismatch
from src.utils.paths import PROJECT_ROOT

CONFIG = PROJECT_ROOT / "configs" / "mlp_h5_experiment6b_v1.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
TCFG = {**CFG["training"], "max_epochs": 30, "patience": 4, "batch_size": 64, "robustness_seeds": 1}
SMALL = {**CFG, "training": TCFG, "mlp": {**CFG["mlp"], "candidates": [
    {"name": "s", "hidden": [8], "weight_decay": 0.0, "dropout": 0.0, "lr": 1e-2},
    {"name": "m", "hidden": [16, 8], "weight_decay": 1e-4, "dropout": 0.1, "lr": 1e-2}]}}
DATASET = PROJECT_ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
RESULTS = PROJECT_ROOT / CFG["output_dir"]
C = {c: i for i, c in enumerate(CLASSES)}
mh.configure_torch(CFG["training"])


def raw_records(match_id, types, period=1, start=0, h360=None):
    out = []
    for j, typ in enumerate(types):
        i = start + j
        out.append({"match_id": match_id, "list_position": i, "index": i + 1, "event_id": f"{match_id}-{i}", "period": period,
                    "timestamp": f"00:00:{j:06.3f}", "t_seconds": float(j), "type": typ,
                    "has_360": True if h360 is None else h360[j], "has_freeze_frame": True if h360 is None else h360[j],
                    "subtype": "Shot Saved" if typ == "Goal Keeper" else None, "outcome": None, "related_events": ""})
    return out


def synthetic(n_matches=15, n_events=600, seed=0):
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
    return mh.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, trace=trace), trace


def tiny(n=600, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 15, (n, 5))
    ids[:, :2] = np.where(rng.random((n, 2)) < 0.1, PAD, ids[:, :2])
    return ids, (ids[:, 3] + ids[:, 4]) % 15


# ----------------------------------------------------------------------------- configuration / encoding
def test_frozen_configuration():
    assert CFG["feature_set"] == {"name": "MLP-H5", "n_previous": 4, "n_columns": 79}
    assert len(CFG["mlp"]["candidates"]) == 8 and len({c["name"] for c in CFG["mlp"]["candidates"]}) == 8
    t = CFG["training"]
    assert t["device"] == "cpu" and t["optimizer"] == "AdamW" and t["deterministic_algorithms"] is True
    assert 0 < t["stopping_fraction"] < 0.5 and t["max_epochs"] >= 20 and t["patience"] >= 1
    cfg6a = yaml.safe_load((PROJECT_ROOT / CFG["experiment6a_config"]).read_text())
    assert CFG["evaluation"]["bootstrap_seed"] == cfg6a["evaluation"]["bootstrap_seed"] == 7
    assert "class_weight" not in json.dumps(CFG["mlp"])


def test_column_indices_equal_6a_one_hot_and_are_position_specific():
    ids = np.array([[PAD, PAD, C["Pass"], C["Carry"], C["Shot"]], [C["Pass"]] * 5])
    cols = mh.column_indices(ids)
    assert cols.tolist() == [[15, 31, 32, 49, 69], [0, 16, 32, 48, 64]]           # Pass at 5 lags -> 5 columns
    assert np.array_equal(np.sort(cols, 1), lh.encode_sparse(ids).indices.reshape(-1, 5))
    names = lh.feature_names()
    assert [names[c] for c in cols[0]] == ["E_{t-4}=<START>", "E_{t-3}=<START>", "E_{t-2}=Pass", "E_{t-1}=Carry", "E_t=Shot"]
    for bad in (np.array([[0, 0, 0, 0, PAD]]), np.array([[16, 0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])):
        with pytest.raises(HistoryError):
            mh.column_indices(bad)


# ----------------------------------------------------------------------------- network
def test_first_layer_equals_linear_on_one_hot_and_shapes():
    m = mh.build_model({"hidden": [16, 8], "dropout": 0.0}, 3).eval()
    ids, _ = tiny(50)
    idx = torch.from_numpy(mh.column_indices(ids))
    onehot = torch.zeros(50, 79)
    onehot[torch.arange(50).repeat_interleave(5), idx.reshape(-1)] = 1
    lin = onehot @ m.first.weight + m.first_bias                       # W_1 phi(x) + b_1
    assert torch.allclose(m.first(idx).sum(1) + m.first_bias, lin, atol=1e-6)
    z = m(idx)
    assert z.shape == (50, 15) and torch.isfinite(z).all()
    bound = 1 / np.sqrt(79)
    assert m.first.weight.abs().max() <= bound and m.first_bias.abs().max() <= bound
    ws, bs = m.weight_matrices()
    assert [w.shape for w in ws] == [(16, 79), (8, 16), (15, 8)] and [b.shape for b in bs] == [(16,), (8,), (15,)]
    p = mh.numpy_forward(ws, bs, onehot.numpy().astype(float))
    assert np.abs(p - torch.softmax(z.double(), 1).detach().numpy()).max() < 1e-6


def test_cross_entropy_softmax_gradients_and_optimizer_update():
    m = mh.build_model({"hidden": [8], "dropout": 0.0}, 0)
    ids, y = tiny(64)
    idx, yt = torch.from_numpy(mh.column_indices(ids)), torch.from_numpy(y)
    z = m(idx)
    loss = F.cross_entropy(z, yt)
    manual = -(z - z.logsumexp(1, keepdim=True))[torch.arange(64), yt].mean()
    assert torch.allclose(loss, manual, atol=1e-6)                     # loss on logits = mean NLL of softmax
    p = torch.softmax(z.double(), 1)
    assert torch.allclose(p.sum(1), torch.ones(64, dtype=torch.float64)) and (p >= 0).all()
    opt = mh.make_optimizer(m, {"lr": 1e-2, "weight_decay": 1e-4}, CFG["training"])
    before = [q.detach().clone() for q in m.parameters()]
    opt.zero_grad()
    loss.backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in m.parameters())
    opt.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
    groups = {g["weight_decay"]: sum(q.numel() for q in g["params"]) for g in opt.param_groups}
    assert groups[0.0] == 8 + 15                                       # biases are not decayed
    assert groups[1e-4] == 79 * 8 + 8 * 15


def test_deterministic_initialisation_and_batch_size_independent_inference():
    a = mh.build_model({"hidden": [16, 8], "dropout": 0.1}, 42)
    b = mh.build_model({"hidden": [16, 8], "dropout": 0.1}, 42)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    ids, _ = tiny(300)
    idx = torch.from_numpy(mh.column_indices(ids))
    full = mh.logits_of(a, idx, batch=300)
    assert torch.allclose(full, mh.logits_of(a, idx, batch=7), atol=1e-6)
    p1 = mh.predict_proba(a, ids)
    p2 = np.vstack([mh.predict_proba(a, ids[i:i + 13]) for i in range(0, 300, 13)])
    assert np.abs(p1 - p2).max() < 1e-6 and np.allclose(p1.sum(1), 1, atol=1e-12)


def test_dropout_active_in_train_mode_and_disabled_in_eval():
    m = mh.build_model({"hidden": [64, 64], "dropout": 0.5}, 1)
    idx = torch.from_numpy(mh.column_indices(tiny(20)[0]))
    m.train()
    assert not torch.equal(m(idx), m(idx))                             # stochastic in training
    m.eval()
    assert torch.equal(m(idx), m(idx))
    p = mh.predict_proba(m, tiny(20)[0])                               # predict_proba switches to eval itself
    m.train()
    assert np.array_equal(mh.predict_proba(m, tiny(20)[0]), p)


def test_tiny_dataset_is_overfit():
    """Implementation sanity check (not a scientific result): a pure interaction is learnt."""
    ids, y = tiny(400)
    X, yt = torch.from_numpy(mh.column_indices(ids)), torch.from_numpy(y)
    r = mh.train_fit("overfit", X, yt, np.arange(400), np.arange(400), {"name": "o", "hidden": [64, 64], "weight_decay": 0.0,
                     "dropout": 0.0, "lr": 1e-2}, {**TCFG, "max_epochs": 150, "patience": 150, "batch_size": 64}, 0)
    p = mh.predict_proba(mh.model_from_state(r.candidate, r.best_state), ids)
    assert -np.log(p[np.arange(400), y]).mean() < 0.05 and (p.argmax(1) == y).mean() == 1.0


# ----------------------------------------------------------------------------- early stopping / determinism / resume
def test_epoch_train_loss_is_row_weighted_with_unequal_final_batch():
    """Regression test (final audit): with lr = 0 the parameters never change, so the epoch's
    reported training loss must equal the row-level mean NLL, although the last batch
    (50 = 7 x 7 + 1 rows) is much smaller than the others. An unweighted mean of batch
    losses would differ."""
    ids, y = tiny(50, 7)
    X, yt = torch.from_numpy(mh.column_indices(ids)), torch.from_numpy(y)
    cand = {"name": "z", "hidden": [8], "weight_decay": 0.0, "dropout": 0.0, "lr": 0.0}
    r = mh.train_fit("w", X, yt, np.arange(50), np.arange(50), cand, {**TCFG, "batch_size": 7, "max_epochs": 1}, 3)
    m = mh.model_from_state(cand, r.best_state)
    z = mh.logits_of(m, X).double()
    row_mean = float(-z.log_softmax(1)[torch.arange(50), yt].mean())
    batch_means = [float(-z[i:i + 7].log_softmax(1)[torch.arange(len(z[i:i + 7])), yt[i:i + 7]].mean()) for i in range(0, 50, 7)]
    assert abs(r.history[1]["train_loss"] - row_mean) < 1e-6
    assert abs(r.history[1]["stop_log_loss"] - row_mean) < 1e-12        # evaluation: float64, all rows, eval mode
    assert abs(np.mean(batch_means) - row_mean) > 1e-4                  # the test can detect unweighted averaging


def test_early_stopping_keeps_best_epoch_state():
    ids, y = tiny(800, 3)
    y = np.where(np.random.default_rng(1).random(800) < 0.5, y, np.random.default_rng(2).integers(0, 15, 800))
    X, yt = torch.from_numpy(mh.column_indices(ids)), torch.from_numpy(y)
    cand = {"name": "big", "hidden": [128, 128], "weight_decay": 0.0, "dropout": 0.0, "lr": 3e-2}
    r = mh.train_fit("es", X, yt, np.arange(500), np.arange(500, 800), cand, {**TCFG, "max_epochs": 80, "patience": 3}, 0)
    stop = [h["stop_log_loss"] for h in r.history]
    assert r.stop_reason == "patience" and r.epochs_trained == r.best_epoch + 3
    assert r.best_stop_log_loss == min(stop) == stop[r.best_epoch]
    m = mh.model_from_state(cand, r.best_state)
    assert abs(mh.mean_nll(m, X[500:], yt[500:]) - r.best_stop_log_loss) < 1e-12


def test_training_is_bitwise_deterministic():
    ids, y = tiny(500)
    X, yt = torch.from_numpy(mh.column_indices(ids)), torch.from_numpy(y)
    cand = SMALL["mlp"]["candidates"][1]
    a = mh.train_fit("d", X, yt, np.arange(400), np.arange(400, 500), cand, TCFG, 9)
    b = mh.train_fit("d", X, yt, np.arange(400), np.arange(400, 500), cand, TCFG, 9)
    assert all(torch.equal(a.best_state[k], b.best_state[k]) for k in a.best_state)          # bitwise identical weights
    strip = [[(h["epoch"], h["train_loss"], h["stop_log_loss"]) for h in r.history] for r in (a, b)]
    assert strip[0] == strip[1] and a.best_epoch == b.best_epoch       # identical training trajectory (timings excluded)
    c = mh.train_fit("d", X, yt, np.arange(400), np.arange(400, 500), cand, TCFG, 10)
    assert not all(torch.equal(a.best_state[k], c.best_state[k]) for k in a.best_state)


def test_interrupted_and_resumed_fit_equals_uninterrupted_fit(tmp_path):
    ids, y = tiny(700, 4)
    X, yt = torch.from_numpy(mh.column_indices(ids)), torch.from_numpy(y)
    cand = {"name": "r", "hidden": [16, 8], "weight_decay": 1e-4, "dropout": 0.1, "lr": 1e-2}   # dropout: RNG matters
    cfg = {**TCFG, "max_epochs": 8, "patience": 8}
    ident = {"run": "x"}
    full = mh.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), cand, cfg, 5)
    ck = tmp_path / "k.inprogress.pt"
    with pytest.raises(mh.Interrupted):
        mh.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), cand, cfg, 5, ck, ident, interrupt_after=3)
    saved = torch.load(ck, weights_only=False)
    assert saved["epoch"] == 3 and "optimizer" in saved and "torch_rng" in saved and len(saved["history"]) == 4
    resumed = mh.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), cand, cfg, 5, ck, ident)
    assert resumed.resumed_from_epoch == 3 and resumed.epochs_trained == full.epochs_trained
    assert all(torch.equal(full.best_state[k], resumed.best_state[k]) for k in full.best_state)
    assert [h["stop_log_loss"] for h in full.history] == [h["stop_log_loss"] for h in resumed.history]
    # optimizer state restored exactly: one more manual step from both states agrees
    with pytest.raises(CheckpointMismatch):                           # different hyperparameters -> refused
        mh.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), {**cand, "lr": 2e-2}, cfg, 5, ck, ident)
    with pytest.raises(CheckpointMismatch):                           # different data / identifiers -> refused
        mh.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), cand, cfg, 5, ck, {"run": "y"})
    ck.write_bytes(ck.read_bytes()[:200])                              # corrupted -> refused
    with pytest.raises(CheckpointMismatch):
        mh.train_fit("k", X, yt, np.arange(600), np.arange(600, 700), cand, cfg, 5, ck, ident)


def test_checkpoint_restores_optimizer_and_rng_state(tmp_path):
    m = mh.build_model({"hidden": [8], "dropout": 0.0}, 0)
    opt = mh.make_optimizer(m, {"lr": 1e-2, "weight_decay": 0.0}, CFG["training"])
    idx, yt = torch.from_numpy(mh.column_indices(tiny(64)[0])), torch.from_numpy(tiny(64)[1])
    for _ in range(3):
        opt.zero_grad()
        F.cross_entropy(m(idx), yt).backward()
        opt.step()
    torch.manual_seed(123)
    mh.save_atomic(tmp_path / "s.pt", {"identifiers": {"a": 1}, "model": m.state_dict(), "optimizer": opt.state_dict(),
                                       "rng": torch.get_rng_state()})
    expected_draw = torch.rand(3)
    ck = mh.load_checked(tmp_path / "s.pt", {"a": 1}, "t")
    m2 = mh.build_model({"hidden": [8], "dropout": 0.0}, 99)
    m2.load_state_dict(ck["model"])
    opt2 = mh.make_optimizer(m2, {"lr": 1e-2, "weight_decay": 0.0}, CFG["training"])
    opt2.load_state_dict(ck["optimizer"])
    for mm, oo in ((m, opt), (m2, opt2)):
        oo.zero_grad()
        F.cross_entropy(mm(idx), yt).backward()
        oo.step()
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), m2.parameters()))
    torch.set_rng_state(ck["rng"])
    assert torch.equal(torch.rand(3), expected_draw)
    with pytest.raises(CheckpointMismatch):
        mh.load_checked(tmp_path / "s.pt", {"a": 2}, "t")
    assert not list(tmp_path.glob(".*tmp*"))


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
        mh.run_nested_cv(rows.ctx[:20], np.tile([0, 1, 2, 3], 5), rows.y[:20], bad, SMALL)


def test_adversarial_outer_validation_labels_change_nothing_in_their_fold(synth, nested):
    _, rows, folds = synth
    res, _ = nested
    y2 = rows.y.copy()
    va = res.fold_of_row == 0
    y2[va] = (y2[va] + 5) % 15
    out = mh.run_nested_cv(rows.ctx, rows.match_id, y2, folds, SMALL)
    a, b = res.choices[0], out.choices[0]
    assert a["candidate"] == b["candidate"] and a["inner_log_loss"] == b["inner_log_loss"]
    assert all(torch.equal(res.states[(0, 0)][k], out.states[(0, 0)][k]) for k in res.states[(0, 0)])
    assert np.array_equal(res.proba[va], out.proba[va])
    # sanity: the test can detect label use - fold 0's labels are training data for other folds
    assert any(not torch.equal(res.states[(f, 0)]["first.weight"], out.states[(f, 0)]["first.weight"]) for f in (1, 2, 3, 4))


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


def test_nested_output_probabilities_and_metrics(synth, nested):
    _, rows, _ = synth
    res, _ = nested
    for p in [res.proba, *res.proba_replicates.values()]:
        xe.check_probabilities(p, 1e-9)
        assert (p > 0).all() and p.shape == (len(rows.y), 15)
    cfg = {"evaluation": {"bootstrap_replicates": 40, "bootstrap_seed": 7, "probability_tolerance": 1e-9}}
    uniform = np.full_like(res.proba, 1 / 15)
    t = xe.evaluate_models(rows.y, rows.match_id, res.fold_of_row, {"MLP-H5": res.proba, "u": uniform}, [("MLP-H5", "u")], cfg)
    mc = t["model_comparison"].set_index("model")
    assert abs(mc.at["MLP-H5", "log_loss"] - sk_log_loss(rows.y, res.proba, labels=list(range(15)))) < 1e-12
    d = -np.log(res.proba[np.arange(len(rows.y)), rows.y]) + np.log(1 / 15)
    pdiff = t["paired_differences"].set_index("metric").loc["log_loss"]
    assert abs(pdiff["difference"] - d.mean()) < 1e-12 and pdiff["ci_high"] < 0      # learns the interaction


def test_completed_fits_and_folds_resume_without_retraining(tmp_path, synth, nested):
    _, rows, folds = synth
    res, _ = nested
    ident = {"run": "r1"}
    a = mh.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, tmp_path, ident)
    assert np.array_equal(a.proba, res.proba)                          # checkpointing does not change results
    (tmp_path / "fold4" / "fold_complete.pt").unlink()                 # interrupted inside fold 4 after its fits
    b = mh.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, tmp_path, ident)
    assert b.resumed_folds == [0, 1, 2, 3] and b.computed_folds == [4] and b.resumed_fits > 0
    assert np.array_equal(a.proba, b.proba) and a.choices == b.choices
    with pytest.raises(CheckpointMismatch):
        mh.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL, tmp_path, {"run": "r2"})


# ----------------------------------------------------------------------------- leakage / boundaries
def test_history_has_no_future_or_cross_period_information(synth):
    t, rows, _ = synth
    full = history_ids(t, 4)
    later = t.copy()
    later.loc[later.index[700:], "symbol_id"] = 0
    assert np.array_equal(history_ids(later, 4)[:700], full[:700])
    assert np.array_equal(history_ids(t.assign(target_id=0, event_id="x", match_id=-1), 4), full)
    p1 = raw_records(5, ["Pass", "Carry", "Shot"])
    p2 = raw_records(5, ["Clearance", "Pass"], period=2, start=3)
    r = HistoryRows.from_table(build_canonical_table(pd.DataFrame(p1 + p2)))
    assert r.ctx[r.seq_pos == 0][1].tolist() == [PAD] * 4 + [C["Clearance"]]
    with pytest.raises(LeakageError):
        assert_no_final_test([1, 2], [2])


def test_supervisor_and_runner_guards_exist():
    sup = (PROJECT_ROOT / "scripts/supervise_mlp_h5_experiment6b.py").read_text()
    run = (PROJECT_ROOT / "scripts/run_mlp_h5_experiment6b.py").read_text()
    assert "start_new_session=True" in sup and '"caffeinate", "-d", "-i", "-w"' in sup and "LOCK_NB" in sup
    assert "LOCK_NB" in run and ".partial" in run and "staging.rename(out)" in run and "load_inputs(cfg)" in run


# ----------------------------------------------------------------------------- real data
needs_data = pytest.mark.skipif(not DATASET.exists(), reason="local canonical dataset not available")
needs_results = pytest.mark.skipif(not (RESULTS / "integrity_audit.csv").exists(), reason="experiment 6B not run")


@needs_data
def test_historical_results_unchanged():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_mlp_h5_experiment6b import historical_snapshot
    snap = historical_snapshot(CFG)
    assert all(not e["mismatches"] for e in snap["directories"].values()) and len(snap["directories"]) == 4


@needs_results
def test_saved_experiment6b_results_are_consistent():
    from src.sequence_models.experiment import sha256_file
    audit = pd.read_csv(RESULTS / "integrity_audit.csv")
    assert not (audit["result"] == "FAIL").any()
    checks = json.loads((RESULTS / "checksums.json").read_text())
    assert all(sha256_file(RESULTS / k) == v for k, v in checks.items())
    pred = pd.read_parquet(PROJECT_ROOT / CFG["artefact_dir"] / "oof_predictions_mlp_h5.parquet")
    p = pred[[f"MLP-H5__p{c}" for c in range(15)]].to_numpy()
    mc = pd.read_csv(RESULTS / "model_comparison.csv").set_index("model")
    assert len(pred) == 690_147 and pred["event_id"].is_unique
    assert abs(sk_log_loss(pred["y"], p, labels=list(range(15))) - mc.at["MLP-H5", "log_loss"]) < 1e-9
