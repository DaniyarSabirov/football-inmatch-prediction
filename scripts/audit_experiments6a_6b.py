"""Independent final audit of experiments 6A (MLR-H5) and 6B (MLP-H5).

Recomputes the critical quantities with implementations written here, from raw StatsBomb JSON
and stored artefacts. For verification it does NOT use the production history builder
(``event_history``), the 6A / 6B model or encoding code, ``raw_history_check``,
``canonical_task``, ``evaluation_v2`` or ``xgboost_experiment``. Production code is imported
only as the object under test: the 6B network class (dense-reference and finite-difference
gradient checks; reproduction of one outer refit) and ``HistoryRows`` (hash of the exact
training input, compared with the run manifests).

Sections (results in ``results/audit_experiments6a_6b_2026-11-24/``):
  1  raw-JSON reconstruction of complete matches (canonical sequences, eligibility, histories, targets)
  2  row identity of all stored predictions; folds; final-test exclusion
  3  6A: independent probabilities from coefficients; optimality gradients at the stored
     coefficients; independent nested CV of one outer fold; weighted-cell vs row fit; lbfgs objective
  4  6B: independent forward pass; dense reference and finite-difference gradients; loss;
     early-stopping bookkeeping of all fits; selection; training population; reproduction of one refit
  5  metrics of all 9 models recomputed (own NumPy implementation + scikit-learn), pooled and per fold
  6  paired match-level bootstrap recomputed (own implementation of the seed-7 convention)
  7  Pressure decomposition
  8  checksums of all historical and 6A / 6B result directories

Usage:
    .venv/bin/python scripts/audit_experiments6a_6b.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "audit_experiments6a_6b_2026-11-24"
RAW = ROOT / "data" / "raw" / "statsbomb" / "data"
DATASET = ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
FOLDS = ROOT / "results/markov_hmm_cost_sensitive_v2/tiers/tier_b/scenarioD15/folds.json"
SPLIT = ROOT / "results/markov_hmm_pilot/split_manifest.json"
R6A = ROOT / "results/logistic_h5_experiment6a_actions15_360current_v1_2026-11-23"
R6B = ROOT / "results/mlp_h5_experiment6b_actions15_360current_v1_2026-11-23"
A6A, A6B = ROOT / "data/interim/logistic_h5_experiment6a", ROOT / "data/interim/mlp_h5_experiment6b"
PRED = {"exp5": ROOT / "data/interim/xgboost_experiment5/oof_predictions.parquet",
        "exp5b": ROOT / "data/interim/xgboost_experiment5b/oof_predictions_h2.parquet",
        "exp6a": A6A / "oof_predictions_mlr_h5.parquet", "exp6b": A6B / "oof_predictions_mlp_h5.parquet"}
MODELS = {"frequency": "exp5", "markov1": "exp5", "markov2": "exp5", "logistic_F0": "exp5", "XGB-F0": "exp5",
          "XGB-H5": "exp5", "XGB-H2": "exp5b", "MLR-H5": "exp6a", "MLP-H5": "exp6b"}
# canonical order, written out here (not imported)
CLASSES = ["Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed", "Pressure",
           "Duel", "Interception", "Block", "Goal Keeper", "Foul Committed", "50/50"]
CID = {c: i for i, c in enumerate(CLASSES)}
K, START, SHOT, PRESSURE = 15, 15, 5, 8
OFF = np.array([0, 16, 32, 48, 64])
CHECKS: list[dict] = []


def check(section, name, ok, value=None):
    CHECKS.append({"section": section, "check": name, "result": "PASS" if ok else "FAIL",
                   "value": json.dumps(value, default=float) if not isinstance(value, (str, type(None))) else value})
    print(f"[{section}] {'PASS' if ok else 'FAIL'}  {name}  {'' if value is None else value}", flush=True)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_arr(a: np.ndarray) -> str:
    a = np.asarray(a)
    return hashlib.sha256("\n".join(map(str, a)).encode() if a.dtype == object else np.ascontiguousarray(a).tobytes()).hexdigest()


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ============================================================================ 1 raw reconstruction
def nm(d, *ks):
    for k in ks:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def companion(ev, prev):
    """K1-K3 exactly as written in docs/canonical_prediction_protocol.md section 3."""
    t = nm(ev, "type", "name")
    if t == "Goal Keeper" and nm(ev, "goalkeeper", "type", "name") in ("Shot Faced", "Goal Conceded", "Penalty Conceded"):
        return "K1"
    if prev is None:
        return ""
    linked = prev["id"] in (ev.get("related_events") or [])
    same = prev.get("timestamp") == ev.get("timestamp")
    pt = nm(prev, "type", "name")
    if t == "Duel" and nm(ev, "duel", "type", "name") == "Tackle" and linked and same and (
            pt == "Dispossessed" or (pt == "Dribble" and nm(prev, "dribble", "outcome", "name") == "Incomplete")):
        return "K2"
    if t == "50/50" and pt == "50/50" and linked and same:
        return "K3"
    return ""


def rebuild_match(mid: int) -> tuple[pd.DataFrame, dict]:
    events = json.loads((RAW / "events" / f"{mid}.json").read_text())
    p360 = RAW / "three-sixty" / f"{mid}.json"
    frames = json.loads(p360.read_text()) if p360.exists() else []
    usable = {f["event_uuid"] for f in frames if f.get("freeze_frame")}
    rows, stats = [], {"companions": 0, "raw_records": len(events)}
    for period in sorted({e["period"] for e in events if e["period"] != 5}):
        raw = sorted((e for e in events if e["period"] == period), key=lambda e: e["index"])
        stream, prev, comp_idx = [], None, []
        for e in raw:
            c = companion(e, prev)
            if c:
                comp_idx.append(e["index"])
                stats["companions"] += 1
            elif nm(e, "type", "name") in CID:
                stream.append(e)
            prev = e
        comp_idx = np.array(comp_idx)
        for p, e in enumerate(stream):
            hist = [CID[stream[p - l]["type"]["name"]] if p - l >= 0 else START for l in (4, 3, 2, 1)]
            nxt = stream[p + 1] if p + 1 < len(stream) else None
            lo = stream[max(p - 4, 0)]["index"]
            rows.append({"match_id": mid, "period": period, "seq_pos": p, "event_id": e["id"], "index": e["index"],
                         "cur": CID[e["type"]["name"]], "h4": hist[0], "h3": hist[1], "h2": hist[2], "h1": hist[3],
                         "target": CID[nxt["type"]["name"]] if nxt else -1, "target_event_id": nxt["id"] if nxt else None,
                         "usable_360": e["id"] in usable, "eligible": nxt is not None and e["id"] in usable,
                         "period_end": p == len(stream) - 2,
                         "repeated_timestamp": p >= 1 and stream[p - 1]["timestamp"] == e["timestamp"],
                         "companion_in_window": bool(len(comp_idx) and ((comp_idx > lo) & (comp_idx < e["index"])).any()),
                         "history_without_360": any(stream[p - l]["id"] not in usable for l in (1, 2, 3, 4) if p - l >= 0),
                         "skipped_raw_before": p >= 1 and e["index"] - stream[p - 1]["index"] > 1})
    return pd.DataFrame(rows), stats


def section1(tb: pd.DataFrame, ctx: np.ndarray, final: set) -> pd.DataFrame:
    rng = np.random.default_rng(20261124)
    tier_b = np.unique(tb["match_id"])
    sample = sorted(int(m) for m in rng.choice(tier_b, 40, replace=False))
    assert not set(sample) & final
    elig = tb[tb["row_eligible"]].reset_index(drop=True)
    pos_of = {e: i for i, e in enumerate(elig["event_id"])}
    recs, n_obs_ok, n_comp = [], True, 0
    for mid in sample:
        r, st = rebuild_match(mid)
        n_comp += st["companions"]
        prod = tb[tb["match_id"] == mid]
        n_obs_ok &= len(r) == len(prod) and list(r["event_id"]) == list(prod["event_id"]) \
            and list(r["cur"]) == list(prod["symbol_id"]) and list(r["period"]) == list(prod["period"]) \
            and list(r["index"]) == list(prod["index"]) and list(r["usable_360"]) == list(prod["usable_360"]) \
            and list(r["eligible"]) == list(prod["row_eligible"])
        recs.append(r[r["eligible"]])
    rec = pd.concat(recs, ignore_index=True)
    check(1, "40 complete Tier B matches rebuilt from raw JSON: canonical observations (ids, order, period, index, class, "
             "usable 360, eligibility) identical to the production table", n_obs_ok, f"{sum(len(x) for x in recs)} eligible rows")
    i = rec["event_id"].map(pos_of).to_numpy()
    hist_ok = np.array_equal(rec[["h4", "h3", "h2", "h1", "cur"]].to_numpy(), ctx[i])
    tgt_ok = np.array_equal(rec["target"].to_numpy(), elig["target_id"].to_numpy()[i]) and \
        list(rec["target_event_id"]) == list(elig["target_event_id"].to_numpy()[i])
    check(1, "independently rebuilt 5-position histories equal the training input (HistoryRows) row by row", hist_ok, len(rec))
    check(1, "independently rebuilt targets (class and target event id) equal the canonical targets", tgt_ok, len(rec))
    strata = {"rows": len(rec), "seq_pos_0": int((rec.seq_pos == 0).sum()), "with_<START>": int((rec.h4 == START).sum()),
              "period_end": int(rec.period_end.sum()), "match_first_row": len(sample),
              "repeated_timestamp": int(rec.repeated_timestamp.sum()), "companion_in_window": int(rec.companion_in_window.sum()),
              "history_event_without_360": int(rec.history_without_360.sum()), "skipped_raw_before": int(rec.skipped_raw_before.sum()),
              **{f"current_or_target_{c}": int(((rec.cur == CID[c]) | (rec.target == CID[c])).sum())
                 for c in ("Pressure", "Shot", "Goal Keeper", "Duel", "50/50", "Block", "Foul Committed")},
              "companions_removed_in_sample": n_comp}
    check(1, "reconstruction covers the required edge cases (counts per stratum)", all(v > 0 for v in strata.values()), strata)
    return rec


# ============================================================================ 2 row identity
def section2(tb, elig, final, folds):
    split = json.loads(SPLIT.read_text())
    tier_b = set(split["development_match_ids"]) - set(split["pilot_match_ids"])
    check(2, "Tier B = development - pilot = 304 matches = dataset matches; 83 final-test ids disjoint",
          tier_b == set(tb["match_id"]) and len(tier_b) == 304 and len(final) == 83 and not (tier_b & final)
          and not (set(pd.read_parquet(DATASET, columns=["match_id"])["match_id"]) & final))
    check(2, "690,147 eligible rows", len(elig) == 690_147, len(elig))
    fom = {int(m): f["fold_id"] for f in folds for m in f["validation_match_ids"]}
    fold = elig["match_id"].map(fom).to_numpy()
    check(2, "outer folds partition the 304 matches; no train/validation overlap; no final-test match",
          len(fom) == 304 and all(not set(f["train_match_ids"]) & set(f["validation_match_ids"]) and
                                  set(f["train_match_ids"]) | set(f["validation_match_ids"]) == tier_b for f in folds))
    proba = {}
    for key, path in PRED.items():
        o = pd.read_parquet(path)
        ok = (np.array_equal(o["event_id"].to_numpy(), elig["event_id"].to_numpy()) and
              np.array_equal(o["match_id"].to_numpy(), elig["match_id"].to_numpy()) and
              np.array_equal(o["y"].to_numpy(), elig["target_id"].to_numpy()) and np.array_equal(o["fold_id"].to_numpy(), fold)
              and o["event_id"].is_unique and len(o) == 690_147)
        check(2, f"{key} predictions: event id, match, target, fold identical element-wise to the canonical rows", ok)
        for m, k in MODELS.items():
            if k == key:
                cols = [f"{m}__p{c}" for c in range(K)]
                p = o[cols].to_numpy()
                proba[m] = p
                check(2, f"{m}: 15 columns p0..p14, finite, >= 0, rows sum to 1 (1e-9)",
                      list(o.columns[o.columns.str.startswith(m + "__p")]) == cols and bool(np.isfinite(p).all() and (p >= 0).all())
                      and float(np.abs(p.sum(1) - 1).max()) <= 1e-9, float(np.abs(p.sum(1) - 1).max()))
    freq = np.bincount(elig["target_id"], minlength=K) / len(elig)
    order_ok = all(abs(proba[m].mean(0) - freq).max() < 0.02 for m in MODELS if m != "frequency")
    check(2, "class order: mean predicted probability of every column matches its class frequency (|diff| < 0.02)", order_ok)
    for m6, rdir in (("MLR-H5", R6A), ("MLP-H5", R6B)):
        ri = json.loads((rdir / "task_manifest.json").read_text())["row_identity"]
        check(2, f"{m6} manifest row identity hashes equal independently computed hashes of the canonical rows",
              ri["event_id_sha256"] == sha_arr(elig["event_id"].to_numpy()) and ri["target_sha256"] == sha_arr(elig["target_id"].to_numpy().astype(np.int64))
              and ri["fold_of_row_sha256"] == sha_arr(fold.astype(np.int64)) and ri["n_rows"] == 690_147)
    return proba, fold


# ============================================================================ 3 experiment 6A
def onehot(cols: np.ndarray) -> sp.csr_matrix:
    n = len(cols)
    return sp.csr_matrix((np.ones(n * 5), (np.repeat(np.arange(n), 5), cols.ravel())), shape=(n, 79))


def cells(cols, y):
    key = np.column_stack([cols, y])
    u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return u[:, :5], u[:, 5], cnt.astype(float)


def nll_mean(p, y):
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def fit_lr(cols, y, C, solver="newton-cholesky", **kw):
    cx, cy, w = cells(cols, y)
    m = LogisticRegression(C=C, l1_ratio=0.0, solver=solver, tol=kw.get("tol", 1e-10), max_iter=kw.get("max_iter", 500))
    m.fit(onehot(cx), cy, sample_weight=w)
    return m, (cx, cy, w)


def lr_proba(m, cols):
    p = np.zeros((len(cols), K))
    p[:, m.classes_] = m.predict_proba(onehot(cols))
    return p


def objective_grad(W, b, cols, y, C):
    """Row-level (unaggregated) objective and max |gradient|."""
    z = np.zeros((len(y), K)) + b
    for j in range(5):
        z += W[:, cols[:, j]].T
    p = softmax(z)
    r = p.copy()
    r[np.arange(len(y)), y] -= 1
    n = len(y)
    gW = (onehot(cols).T @ r).T / n + W / (C * n)
    return float(-np.log(p[np.arange(n), y]).mean() + (W ** 2).sum() / (2 * C * n)), float(max(np.abs(gW).max(), np.abs(r.sum(0) / n).max()))


def section3(cols, y, match, fold, folds, proba):
    npz = np.load(A6A / "fold_models_mlr_h5.npz")
    hc = pd.read_csv(R6A / "hyperparameter_choices.csv").set_index("fold_id")
    rec = np.empty((len(y), K))
    grads = {}
    for f in range(5):
        W, b = npz[f"coef_fold{f}"], npz[f"intercept_fold{f}"]
        sel = fold == f
        rec[sel] = softmax(sum(W[:, cols[sel, j]].T for j in range(5)) + b)
        tr = np.isin(match, folds[f]["train_match_ids"])
        _, g = objective_grad(W, b, cols[tr], y[tr], float(hc.at[f, "C"]))
        grads[f] = g
    err = np.abs(rec - proba["MLR-H5"])
    check(3, "6A probabilities recomputed from stored W, b (own encoding, NumPy softmax) equal stored predictions",
          err.max() < 1e-10, {"max_abs": float(err.max()), "mean_abs": float(err.mean())})
    check(3, "6A stored coefficients are optimal for the ROW-LEVEL objective on each fold's outer-training rows (max |grad| < 1e-7)",
          max(grads.values()) < 1e-7, grads)
    W0 = npz["coef_fold0"]
    check(3, "6A deviation coding: sum over classes = 0 and sum over each position block = 0 (|.| < 1e-4)",
          float(np.abs(W0.sum(0)).max()) < 1e-6 and max(float(np.abs(W0[:, a:a + w].sum(1)).max()) for a, w in zip(OFF, [16, 16, 16, 16, 15])) < 1e-4)
    # independent nested CV of outer fold 2
    t0 = time.perf_counter()
    f = 2
    inner = [g for g in folds if g["fold_id"] != f]
    grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    scores = {}
    for C in grid:
        s, n = 0.0, 0
        for h in inner:
            tr = np.isin(match, [m for g in inner if g is not h for m in g["validation_match_ids"]])
            va = np.isin(match, h["validation_match_ids"])
            assert not np.isin(match[tr | va], folds[f]["validation_match_ids"]).any()
            m, _ = fit_lr(cols[tr], y[tr], C)
            s += -np.log(lr_proba(m, cols[va])[np.arange(va.sum()), y[va]]).sum()
            n += va.sum()
        scores[C] = s / n
    best = min(grid, key=lambda c: (round(scores[c], 12), c))
    stored = pd.read_csv(R6A / "inner_cv_scores.csv").query("fold_id == @f").set_index("C")["pooled_inner_log_loss"]
    tr = np.isin(match, folds[f]["train_match_ids"])
    va = fold == f
    m, (cx, cy, w) = fit_lr(cols[tr], y[tr], best)
    pd_ = np.abs(lr_proba(m, cols[va]) - proba["MLR-H5"][va]).max()
    check(3, "6A fold 2 nested CV re-run independently (raw sklearn, own cells): same pooled inner scores, same C, same predictions",
          best == hc.at[f, "C"] and max(abs(scores[c] - stored[c]) for c in grid) < 1e-9 and pd_ < 1e-9,
          {"C": best, "max_score_diff": max(abs(scores[c] - stored[c]) for c in grid), "max_prob_diff": float(pd_),
           "seconds": round(time.perf_counter() - t0)})
    check(3, "6A cell aggregation: cells keep every training row (sum of weights = rows) and only training matches",
          int(w.sum()) == int(tr.sum()), {"rows": int(tr.sum()), "cells": len(cy)})
    # weighted cells vs unaggregated rows, outer fold 4
    f = 4
    tr, va = np.isin(match, folds[f]["train_match_ids"]), fold == f
    C = float(hc.at[f, "C"])
    mr = LogisticRegression(C=C, l1_ratio=0.0, solver="newton-cholesky", tol=1e-10, max_iter=500).fit(onehot(cols[tr]), y[tr])
    dp = np.abs(lr_proba(mr, cols[va]) - proba["MLR-H5"][va]).max()
    check(3, "6A fold 4: unaggregated row-level fit reproduces the stored (weighted-cell) predictions (|dp| < 1e-8)", dp < 1e-8, float(dp))
    # lbfgs reaches the same optimum (same objective) on fold 2's outer-training cells
    f = 2
    tr = np.isin(match, folds[f]["train_match_ids"])
    C = float(hc.at[f, "C"])
    lb, _ = fit_lr(cols[tr], y[tr], C, solver="lbfgs", tol=1e-9, max_iter=30000)
    nw, _ = fit_lr(cols[tr], y[tr], C)
    o_lb, g_lb = objective_grad(lb.coef_, lb.intercept_, cols[tr], y[tr], C)
    o_nw, g_nw = objective_grad(nw.coef_, nw.intercept_, cols[tr], y[tr], C)
    check(3, "6A solver: lbfgs and newton-cholesky reach the same objective value (|diff| < 1e-8); newton is at least as low",
          abs(o_lb - o_nw) < 1e-8 and o_nw <= o_lb + 1e-12, {"lbfgs": o_lb, "newton": o_nw, "lbfgs_iter": int(lb.n_iter_[0]), "grad_lbfgs": g_lb, "grad_newton": g_nw})
    fl = pd.read_csv(R6A / "fit_log.csv")
    check(3, "6A fit log: 105 fits = 5 x (4 inner x 5 C + 1 refit), all converged, no warnings, gradient <= 1e-6",
          len(fl) == 105 and fl.converged.all() and (fl.warnings == "[]").all() and fl.max_abs_gradient.max() <= 1e-6
          and (fl.stage == "inner").sum() == 100, float(fl.max_abs_gradient.max()))
    sc = pd.read_csv(R6A / "inner_cv_scores.csv")
    rec_sel = {f: sc[sc.fold_id == f].sort_values(["pooled_inner_log_loss", "C"]).iloc[0]["C"] for f in range(5)}
    check(3, "6A selected C per fold = argmin of the stored pooled inner scores; all interior", all(rec_sel[f] == hc.at[f, "C"] for f in range(5))
          and all(c not in (0.01, 100.0) for c in rec_sel.values()), rec_sel)


# ============================================================================ 4 experiment 6B
def section4(cols, y, match, fold, folds, proba, ctx):
    import torch
    import torch.nn.functional as F

    from src.sequence_models import mlp_h5_experiment as mh     # object under test only
    cfg = yaml.safe_load((ROOT / "configs/mlp_h5_experiment6b_v1.yaml").read_text())
    mh.configure_torch(cfg["training"])
    npz = np.load(A6B / "fold_models_mlp_h5.npz")
    sel_cfg = {int(k): v for k, v in json.loads(str(npz["selected"])).items()}
    pred = pd.read_parquet(A6B / "oof_predictions_mlp_h5.parquet")
    worst = 0.0
    for f in range(5):
        s = fold == f
        x = onehot(cols[s])
        for r in range(3):
            nl = sum(1 for k in npz.files if k.startswith(f"fold{f}_rep{r}_W"))
            h = x
            for i in range(1, nl + 1):
                h = np.asarray(h @ npz[f"fold{f}_rep{r}_W{i}"].T) + npz[f"fold{f}_rep{r}_b{i}"]
                if i < nl:
                    h = np.maximum(h, 0)
            col = [f"MLP-H5__p{c}" if r == 0 else f"MLP-H5_seed{r}__p{c}" for c in range(K)]
            worst = max(worst, float(np.abs(softmax(h) - pred.loc[s, col].to_numpy()).max()))
    check(4, "6B probabilities: NumPy float64 forward pass of stored weights (own sparse one-hot) equal stored (|dp| < 1e-5), 3 seeds",
          worst < 1e-5, worst)
    # dense reference: forward and gradients (float64)
    m = mh.MLP([16, 8], 0.0).double()
    xb = torch.from_numpy(cols[:300])
    yb = torch.from_numpy(y[:300])
    dense = torch.zeros(300, 79, dtype=torch.float64)
    dense[torch.arange(300).repeat_interleave(5), xb.reshape(-1)] = 1.0
    W1 = m.first.weight.detach().clone().T.requires_grad_(True)       # (16, 79) dense W_1
    b1 = m.first_bias.detach().clone().requires_grad_(True)
    rest = [p.detach().clone().requires_grad_(True) for p in m.body.parameters()]

    def dense_forward(xd):
        h = torch.relu(xd @ W1.T + b1)
        h = torch.relu(h @ rest[0].T + rest[1])
        return h @ rest[2].T + rest[3]
    z_ref, z = dense_forward(dense), m(xb)
    F.cross_entropy(z_ref, yb).backward()
    F.cross_entropy(z, yb).backward()
    fwd = float((z - z_ref).abs().max())
    g1 = float((m.first.weight.grad.T - W1.grad).abs().max())
    gb = float((m.first_bias.grad - b1.grad).abs().max())
    gr = max(float((p.grad - q.grad).abs().max()) for p, q in zip(m.body.parameters(), rest))
    check(4, "6B first layer: embedding-sum forward and ALL gradients equal an explicit dense one-hot network (float64, < 1e-12)",
          max(fwd, g1, gb, gr) < 1e-12, {"forward": fwd, "grad_W1": g1, "grad_b1": gb, "grad_other": gr})
    # finite differences (float64) on the production module
    m.zero_grad()
    loss = F.cross_entropy(m(xb), yb)
    loss.backward()
    rng = np.random.default_rng(0)
    fd_err = 0.0
    for p in m.parameters():
        flat = p.data.view(-1)
        for idx in rng.choice(flat.numel(), min(8, flat.numel()), replace=False):
            old = flat[idx].item()
            flat[idx] = old + 1e-6
            lp = F.cross_entropy(m(xb), yb).item()
            flat[idx] = old - 1e-6
            lm = F.cross_entropy(m(xb), yb).item()
            flat[idx] = old
            fd_err = max(fd_err, abs((lp - lm) / 2e-6 - p.grad.view(-1)[idx].item()))
    check(4, "6B finite-difference gradient check on the production module (float64, 48 parameters, |err| < 1e-7)", fd_err < 1e-7, fd_err)
    # loss: NumPy log-sum-exp vs F.cross_entropy on a real fold with a stored model
    st = torch.load(A6B / "fold_models_mlp_h5.pt", weights_only=False)
    model = mh.model_from_state(st["selected"][0], st["states"][(0, 0)])
    s = np.flatnonzero(fold == 0)
    with torch.no_grad():
        zz = model(torch.from_numpy(cols[s])).double().numpy()
    lse = zz.max(1) + np.log(np.exp(zz - zz.max(1, keepdims=True)).sum(1))
    np_loss = float((lse - zz[np.arange(len(s)), y[s]]).mean())
    with torch.no_grad():
        t_loss = float(F.cross_entropy(torch.from_numpy(zz), torch.from_numpy(y[s])))
    check(4, "6B loss: NumPy stable log-sum-exp NLL = F.cross_entropy on logits = stored fold-0 log loss",
          abs(np_loss - t_loss) < 1e-12 and abs(np_loss - nll_mean(proba["MLP-H5"][s], y[s])) < 1e-6,
          {"numpy": np_loss, "torch": t_loss, "stored": nll_mean(proba["MLP-H5"][s], y[s])})
    # early-stopping bookkeeping of all 175 fits
    fl = pd.read_csv(R6B / "fit_log.csv")
    th = pd.read_csv(R6B / "training_histories.csv")
    bad = []
    tot_epochs = 0
    for a, b in zip(fl.itertuples(), th.itertuples()):
        hist = json.loads(b.history)
        stop = [hh["stop_log_loss"] for hh in hist]
        be = int(np.argmin(stop))           # first minimum (strict improvement rule)
        n_ep = len(hist) - 1
        tot_epochs += n_ep
        exp_n = min(be + int(cfg["training"]["patience"]), int(cfg["training"]["max_epochs"]))
        if not (a.candidate == b.candidate and a.best_epoch == be and abs(a.best_stop_log_loss - stop[be]) < 1e-15
                and n_ep == a.epochs_trained == exp_n and hist[0]["epoch"] == 0 and [h["epoch"] for h in hist] == list(range(n_ep + 1))):
            bad.append((a.fold_id, a.stage, a.candidate, a.held_inner_fold))
    check(4, "6B all 175 fits: best epoch = first argmin of the stopping curve (epoch 0 included), stored best loss = curve minimum, "
             "epochs = min(best + patience, 60)", not bad and len(fl) == 175, {"violations": bad[:5], "epoch0_best": int((fl.best_epoch == 0).sum())})
    check(4, "6B epoch count: 5,363 training epochs in fit log and in the stored curves; 175 = 5 x (4 x 8 + 1 + 2) fits",
          tot_epochs == int(fl.epochs_trained.sum()) == 5363 and (fl.stage == "inner").sum() == 160, tot_epochs)
    pooled = fl[fl.stage == "inner"].assign(s=lambda d: d.inner_validation_log_loss * d.inner_validation_rows).groupby(
        ["fold_id", "candidate"]).apply(lambda d: d.s.sum() / d.inner_validation_rows.sum(), include_groups=False)
    order = [c["name"] for c in cfg["mlp"]["candidates"]]
    hc = pd.read_csv(R6B / "hyperparameter_choices.csv").set_index("fold_id")
    chosen = {f: min(order, key=lambda c: (round(pooled[(f, c)], 12), order.index(c))) for f in range(5)}
    check(4, "6B selection recomputed from per-fit inner scores = stored choice and stored stopping weights' configuration",
          all(chosen[f] == hc.at[f, "candidate"] == sel_cfg[f]["name"] for f in range(5)), chosen)
    sc = pd.read_csv(R6B / "inner_cv_scores.csv")
    check(4, "6B search space in the run = 8 frozen configurations of the configuration file", sorted(sc.candidate.unique()) == sorted(order)
          and sc.groupby("fold_id").size().eq(8).all() and len(pd.read_csv(R6B / "search_candidates.csv")) == 8)
    # training population, stopping subsets
    da = pd.read_csv(R6B / "data_access_audit.csv")
    pop = []
    x5 = pd.read_csv(ROOT / "results/xgboost_experiment5_actions15_360current_v1_2026-11-20/hyperparameter_choices.csv")
    for f in range(5):
        trm = folds[f]["train_match_ids"]
        fit_m, stop_m = mh.stopping_split(trm, 0.10, mh.derive_seed(int(cfg["training"]["seed"]), 7, f, 0))
        h = lambda ms: hashlib.sha256(",".join(map(str, sorted(ms))).encode()).hexdigest()
        rec = da[(da.outer_fold == f)]
        ok = (h(fit_m) in set(rec[rec.purpose == "outer_train"].match_ids_sha256) and h(stop_m) in set(rec[rec.purpose == "outer_stop"].match_ids_sha256)
              and not set(fit_m) & set(stop_m) and set(fit_m) | set(stop_m) == set(trm) and not set(trm) & set(folds[f]["validation_match_ids"]))
        r = fl[(fl.fold_id == f) & (fl.stage == "outer_refit")].iloc[0]
        xg = x5[(x5.fold_id == f) & (x5.feature_set == "XGB-H5")]
        pop.append({"fold_id": f, "outer_train_matches": len(trm), "outer_train_rows": int(np.isin(match, trm).sum()),
                    "mlp_fit_matches": len(fit_m), "mlp_fit_rows": int(r.n_train_rows), "mlp_stop_matches": len(stop_m),
                    "mlp_stop_rows": int(r.n_stop_rows), "mlp_share_of_rows": r.n_train_rows / np.isin(match, trm).sum(),
                    "xgb_h5_train_rows": int(xg.train_rows.iat[0]) if len(xg) else None, "sets_match_access_audit": bool(ok)})
    pop = pd.DataFrame(pop)
    pop.to_csv(OUT / "training_population_6b_vs_xgb_h5.csv", index=False)
    check(4, "6B refit stopping subsets: disjoint from fit matches, inside outer-training matches, identical to the access audit",
          bool(pop.sets_match_access_audit.all()))
    check(4, "6B refit trained on fewer rows than XGB-H5 (documented limitation): share of outer-training rows",
          True, {f"fold{r.fold_id}": round(r.mlp_share_of_rows, 4) for r in pop.itertuples()})
    # reproduce outer fold 3's primary refit from scratch
    f = 3
    cand = sel_cfg[f]
    ci = order.index(cand["name"])
    fit_m, stop_m = mh.stopping_split(folds[f]["train_match_ids"], 0.10, mh.derive_seed(int(cfg["training"]["seed"]), 7, f, 0))
    t0 = time.perf_counter()
    r = mh.train_fit("audit_f3", torch.from_numpy(cols), torch.from_numpy(y), np.flatnonzero(np.isin(match, fit_m)),
                     np.flatnonzero(np.isin(match, stop_m)), cand, cfg["training"], mh.derive_seed(int(cfg["training"]["seed"]), f, 0, ci, 0))
    same_w = all(torch.equal(r.best_state[k], st["states"][(f, 0)][k]) for k in r.best_state)
    hist_stored = json.loads(th[(th.fold_id == f) & (th.stage == "outer_refit")].history.iat[0])
    same_h = [(h["train_loss"], h["stop_log_loss"]) for h in r.history] == [(h["train_loss"], h["stop_log_loss"]) for h in hist_stored]
    s = fold == f
    pa = mh.predict_proba(mh.model_from_state(cand, r.best_state), ctx[s])
    check(4, "6B fold 3 primary refit reproduced from scratch: identical stopping curve, best epoch, weights (bitwise) and predictions",
          same_w and same_h and np.array_equal(pa, proba["MLP-H5"][s]),
          {"best_epoch": r.best_epoch, "max_dp": float(np.abs(pa - proba["MLP-H5"][s]).max()), "seconds": round(time.perf_counter() - t0)})
    # checkpoint contents and provenance
    env = json.loads((R6B / "environment.json").read_text())
    fc = torch.load(A6B / "checkpoints/fold2/fold_complete.pt", weights_only=False)
    done = torch.load(next((A6B / "checkpoints/fold2").glob("*inner1_A_r0.done.pt")), weights_only=False)
    left = list((A6B / "checkpoints").rglob("*.inprogress.pt"))
    check(4, "6B checkpoints: identifiers equal the run's; fold record holds probabilities/states/choices/trace; no in-progress files left",
          fc["identifiers"] == env["checkpoint_identifiers"] and {"val_index", "proba", "states", "selected", "trace"} <= set(fc)
          and done["identifiers"]["fit_key"].endswith("A_r0") and not left)
    status = json.loads((A6B / "run_status.json").read_text())
    check(4, "6B supervisor status: state completed, experiment and audit exit codes 0, COMPLETED marker present",
          status["state"] == "completed" and [s_["exit_code"] for s_ in status["stages"]] == [0, 0] and (A6B / "COMPLETED").exists(),
          {s_["stage"]: s_["exit_code"] for s_ in status["stages"]})
    rob = pd.read_csv(R6B / "seed_robustness.csv").set_index("replicate")
    rr = {r_: nll_mean(pred[[f"MLP-H5__p{c}" if r_ == 0 else f"MLP-H5_seed{r_}__p{c}" for c in range(K)]].to_numpy(), y) for r_ in range(3)}
    check(4, "6B seed-level log losses recomputed; primary = replicate 0 (not the best seed)",
          all(abs(rr[r_] - rob.at[r_, "pooled_log_loss"]) < 1e-12 for r_ in rr) and min(rr, key=rr.get) != 0, rr)


# ============================================================================ 5 metrics
def metrics(y, p):
    n = len(y)
    d = p.argmax(1)
    cm = np.bincount(y * K + d, minlength=K * K).reshape(K, K)
    tp, sup, prd = np.diag(cm).astype(float), cm.sum(1).astype(float), cm.sum(0).astype(float)
    prec = np.divide(tp, prd, out=np.zeros(K), where=prd > 0)
    rec = np.divide(tp, sup, out=np.zeros(K), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    oh = np.zeros_like(p)
    oh[np.arange(n), y] = 1

    def ap(score, pos):
        o = np.argsort(-score, kind="stable")
        s, t = score[o], pos[o].astype(float)
        tpc, fpc = np.cumsum(t), np.cumsum(1 - t)
        last = np.r_[np.flatnonzero(np.diff(s) != 0), n - 1]
        tpc, fpc = tpc[last], fpc[last]
        prec_, rec_ = tpc / (tpc + fpc), tpc / tpc[-1]
        return float(np.sum(np.diff(np.r_[0, rec_]) * prec_))
    return {"log_loss": float(-np.log(p[np.arange(n), y]).mean()), "brier": float(((p - oh) ** 2).sum(1).mean()),
            "accuracy": float(tp.sum() / n), "macro_f1": float(f1.mean()), "weighted_f1": float((f1 * sup).sum() / n),
            "shot_average_precision": ap(p[:, SHOT], y == SHOT), "shot_recall": float(rec[SHOT]), "shot_precision": float(prec[SHOT]),
            "shot_tp": int(tp[SHOT]), "shot_predicted": int(prd[SHOT])}, cm, prec, rec, f1, sup


def section5(y, fold, proba):
    mc = pd.read_csv(R6B / "model_comparison.csv").set_index("model")
    bf = pd.read_csv(R6B / "metrics_by_fold.csv").set_index(["model", "fold_id"])
    pc = pd.read_csv(R6B / "per_class_metrics.csv").set_index(["model", "class_id"])
    cmt = pd.read_csv(R6B / "confusion_matrices.csv")
    rows, worst, worst_f, sk = [], 0.0, 0.0, 0.0
    keys = ["log_loss", "brier", "accuracy", "macro_f1", "weighted_f1", "shot_average_precision", "shot_recall", "shot_precision"]
    for m, p in proba.items():
        mt, cm, prec, rec, f1, sup = metrics(y, p)
        rows.append({"model": m, **mt})
        worst = max(worst, max(abs(mt[k] - mc.at[m, k]) for k in keys))
        sk = max(sk, abs(mt["log_loss"] - log_loss(y, p, labels=list(range(K)))),
                 abs(mt["macro_f1"] - f1_score(y, p.argmax(1), labels=list(range(K)), average="macro", zero_division=0)),
                 abs(mt["shot_average_precision"] - average_precision_score(y == SHOT, p[:, SHOT])))
        pcm = pc.loc[m]
        worst = max(worst, float(np.abs(pcm["precision"].to_numpy() - prec).max()), float(np.abs(pcm["recall"].to_numpy() - rec).max()),
                    float(np.abs(pcm["f1"].to_numpy() - f1).max()), float(np.abs(pcm["support"].to_numpy() - sup).max()))
        c = cmt[cmt.model == m].pivot(index="true_class", columns="predicted_class", values="count").loc[CLASSES, CLASSES].to_numpy()
        worst = max(worst, float(np.abs(c - cm).max()))
        for f in range(5):
            s = fold == f
            mf = metrics(y[s], p[s])[0]
            worst_f = max(worst_f, max(abs(mf[k] - bf.loc[(m, f), k]) for k in keys if k != "shot_precision"))  # not stored per fold
        pooled_vs_mean = abs(mt["log_loss"] - np.mean([bf.loc[(m, f), "log_loss"] for f in range(5)]))
        rows[-1]["pooled_minus_unweighted_fold_mean_log_loss"] = pooled_vs_mean
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "metrics_recomputed.csv", index=False)
    check(5, "pooled metrics, per-class precision/recall/F1/support and confusion matrices of all 9 models recomputed (own NumPy) "
             "equal the stored 6B tables (< 1e-9)", worst < 1e-9, worst)
    check(5, "per-fold metrics recomputed equal the stored tables (< 1e-9)", worst_f < 1e-9, worst_f)
    check(5, "own implementations agree with scikit-learn (log loss, macro-F1, average precision; < 1e-9)", sk < 1e-9, sk)
    mc6a = pd.read_csv(R6A / "model_comparison.csv").set_index("model")
    check(5, "6A stored pooled table agrees with the recomputation for its 8 models (< 1e-9)",
          max(abs(t.set_index("model").at[m, k] - mc6a.at[m, k]) for m in mc6a.index for k in keys) < 1e-9)
    print(t[["model", "log_loss", "accuracy", "macro_f1", "weighted_f1", "shot_average_precision", "shot_tp", "shot_predicted"]].round(5).to_string())
    return t


# ============================================================================ 6 bootstrap
def section6(y, match, fold, proba):
    mids, inv = np.unique(match, return_inverse=True)
    rng = np.random.default_rng(7)                     # convention of evaluation_v2.bootstrap_multiplicities, re-implemented
    mult = np.stack([np.bincount(rng.integers(0, len(mids), len(mids)), minlength=len(mids)) for _ in range(1000)]).astype(float)
    n_g = np.bincount(inv).astype(float)
    nll = {m: -np.log(p[np.arange(len(y)), y]) for m, p in proba.items()}
    out = []
    for rdir, pairs in ((R6A, [("MLR-H5", "XGB-H5"), ("MLR-H5", "markov2"), ("MLR-H5", "logistic_F0")]),
                        (R6B, [("MLP-H5", "XGB-H5"), ("MLP-H5", "MLR-H5"), ("MLP-H5", "markov2")])):
        st = pd.read_csv(rdir / "paired_differences.csv").query("metric == 'log_loss'").set_index(["model", "reference"])
        for m, r in pairs:
            d = nll[m] - nll[r]
            boot = mult @ np.bincount(inv, weights=d) / (mult @ n_g)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            fs = [float(d[fold == f].mean()) for f in range(5)]
            row = st.loc[(m, r)]
            ok = (abs(d.mean() - row.difference) < 1e-12 and abs(lo - row.ci_low) < 1e-12 and abs(hi - row.ci_high) < 1e-12
                  and sum(v < 0 for v in fs) == row.folds_negative and sum(v > 0 for v in fs) == row.folds_positive
                  and abs(d.mean() / nll[r].mean() - row.relative_difference) < 1e-12)
            out.append({"experiment": rdir.name[:22], "model": m, "reference": r, "difference": d.mean(), "relative": d.mean() / nll[r].mean(),
                        "ci_low": lo, "ci_high": hi, "folds_model_better": sum(v < 0 for v in fs), "fold_differences": json.dumps(fs),
                        "n_rows": len(y), "n_matches": len(mids), "matches_stored": ok})
            check(6, f"{m} - {r}: mean d_i, relative difference, 95% cluster-bootstrap CI and fold signs reproduce the stored values",
                  ok, {"diff": round(d.mean(), 6), "ci": [round(lo, 6), round(hi, 6)], "folds_better": sum(v < 0 for v in fs)})
    pd.DataFrame(out).to_csv(OUT / "paired_differences_recomputed.csv", index=False)


# ============================================================================ 7 Pressure
def section7(y, cur, proba):
    n = len(y)
    pr = cur == PRESSURE
    out = []
    for m, r in (("MLR-H5", "XGB-H5"), ("MLR-H5", "markov2"), ("MLP-H5", "MLR-H5"), ("MLP-H5", "XGB-H5")):
        d = -np.log(proba[m][np.arange(n), y]) + np.log(proba[r][np.arange(n), y])
        out.append({"model": m, "reference": r, "pressure_share_of_rows": pr.mean(), "total_difference": d.mean(),
                    "pressure_contribution": d[pr].sum() / n, "pressure_conditional_mean": d[pr].mean(),
                    "other_contribution": d[~pr].sum() / n})
    t = pd.DataFrame(out)
    t.to_csv(OUT / "pressure_decomposition.csv", index=False)
    s6a = pd.read_csv(R6A / "nll_difference_by_stratum.csv").query("stratum == 'current_class' and level == 'Pressure'").set_index("reference")
    s6b = pd.read_csv(R6B / "nll_difference_by_stratum.csv").query("stratum == 'current_class' and level == 'Pressure'").set_index("reference")
    ok = (abs(t.iloc[0].pressure_contribution - s6a.at["XGB-H5", "contribution"]) < 1e-12 and
          abs(t.iloc[2].pressure_contribution - s6b.at["MLR-H5", "contribution"]) < 1e-12 and
          all(abs(r_.pressure_contribution + r_.other_contribution - r_.total_difference) < 1e-12 for r_ in t.itertuples()))
    check(7, "Pressure share, contribution (row-weighted, sums to the total) and conditional means recomputed = stored strata",
          ok, t.round(5).drop(columns="other_contribution").to_dict("records"))


# ============================================================================ 8 checksums
def section8():
    dirs = [ROOT / "results" / d for d in ("canonical_baselines_actions15_360current_v1_2026-11-19",
                                           "xgboost_experiment5_actions15_360current_v1_2026-11-20",
                                           "xgboost_experiment5b_h2_actions15_360current_v1_2026-11-21")] + [R6A, R6B]
    for d in dirs:
        bad, n = [], 0
        for man in ("checksums.json", "checksums_supplementary.json"):
            if (d / man).exists():
                c = json.loads((d / man).read_text())
                c = c.get("files", c) if "supplementary" in man else c
                bad += [k for k, v in c.items() if sha_file(d / k) != v]
                n += len(c)
        check(8, f"checksums of {d.name}", not bad, bad or n)
    for d, keyfiles in ((R6A, ["oof_predictions_mlr_h5.parquet", "fold_models_mlr_h5.npz"]),
                        (R6B, ["oof_predictions_mlp_h5.parquet", "fold_models_mlp_h5.npz", "fold_models_mlp_h5.pt"])):
        env = json.loads((d / "environment.json").read_text())
        check(8, f"{d.name[:24]}: git-ignored artefacts match the hashes recorded at run time",
              all(sha_file(ROOT / rel) == h for rel, h in env["artefacts_sha256"].items()), len(env["artefacts_sha256"]))
        cur = {}
        for name in env["source_files_sha256"]:
            p = next(iter(list((ROOT / "src/sequence_models").glob(name)) + list((ROOT / "scripts").glob(name))))
            cur[name] = sha_file(p) == env["source_files_sha256"][name]
        check(8, f"{d.name[:24]}: every executed source file is byte-identical to the current file (provenance)", all(cur.values()),
              {k: v for k, v in cur.items() if not v} or len(cur))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from src.sequence_models.event_history import HistoryRows     # object under test (the exact training input)
    table = pd.read_parquet(DATASET)
    final = set(int(m) for m in json.loads(SPLIT.read_text())["final_test_match_ids"])
    tb = table[table["tier"] == "tier_b"].reset_index(drop=True)
    del table
    folds = json.loads(FOLDS.read_text())["scenarioD15"]
    ctx = HistoryRows.from_table(tb).ctx
    for d in (R6A, R6B):
        check(0, f"training-input hash in {d.name[:24]} manifest equals HistoryRows of the canonical table",
              json.loads((d / "task_manifest.json").read_text())["row_identity"]["feature_matrix_sha256"] == sha_arr(ctx))
    elig = tb[tb["row_eligible"]].reset_index(drop=True)
    # independent full-table history (pandas shift within sequence) vs training input
    hist = np.column_stack([tb.groupby("sequence_id")["symbol_id"].shift(l).fillna(START).astype(int).to_numpy()[tb["row_eligible"].to_numpy()]
                            for l in (4, 3, 2, 1)] + [elig["symbol_id"].to_numpy()])
    check(0, "full-table history rebuilt with pandas shift within (match, period) equals the training input on all 690,147 rows",
          np.array_equal(hist, ctx))
    section1(tb, ctx, final)
    proba, fold = section2(tb, elig, final, folds)
    y, match = elig["target_id"].to_numpy().astype(np.int64), elig["match_id"].to_numpy()
    cols = hist + OFF
    section3(cols, y, match, fold, folds, proba)
    section4(cols, y, match, fold, folds, proba, ctx)
    section5(y, fold, proba)
    section6(y, match, fold, proba)
    section7(y, elig["symbol_id"].to_numpy(), proba)
    section8()
    n_fail = sum(c["result"] == "FAIL" for c in CHECKS)
    pd.DataFrame(CHECKS).to_csv(OUT / "audit_checks.csv", index=False)
    (OUT / "audit_summary.json").write_text(json.dumps({"verdict_of_checks": "PASS" if not n_fail else "FAIL", "n_checks": len(CHECKS),
                                                       "n_fail": n_fail, "runtime_seconds": round(time.time() - t0)}, indent=1))
    print(f"\n{len(CHECKS)} checks, {n_fail} FAIL, {time.time() - t0:.0f} s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
