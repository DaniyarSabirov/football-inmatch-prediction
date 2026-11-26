"""Independent post-run verification of experiment 6C (LSTM-H5) from its stored artefacts.

Trains nothing, imports neither PyTorch nor the experiment module, and does not trust the
runner's summary:

- result checksums (``checksums.json`` as written by the run) and artefact hashes; the result
  directory is complete and the run's own integrity audit has no FAIL;
- row identity of the stored predictions against the canonical dataset, the folds file and the
  stored predictions of experiments 5, 5B, 6A and 6B (paired row by row);
- history semantics: the 5-position history is rebuilt with pandas ``groupby().shift()``
  (not ``event_history``); probabilities are recomputed by an LSTM forward pass written HERE in
  NumPy (gate blocks i, f, g, o; effective bias b_ih + b_hh; h_0 = c_0 = 0 per window) from the
  stored weights, for the primary and every robustness replicate, on every row;
- the stored weights' shapes equal the selected architecture; parameter counts equal
  4Hm + 4H^2 + 8H per layer + 15H + 15;
- early-stopping bookkeeping of every fit; selection recomputed from the per-fit inner scores;
  the stopping subsets are the same matches as in experiment 6B;
- probabilities valid; metrics recomputed with scikit-learn (pooled and per fold);
- paired differences against MLP-H5, XGB-H5, MLR-H5, Markov 2, XGB-H2, logistic F0: the stored
  values and CIs reproduced exactly with an own implementation of the seed-7 match bootstrap,
  and an independent bootstrap with a different seed;
- data access (outer validation only predicted, final test never touched), historical results
  and inputs unchanged, raw-JSON re-verification of the stored stratified sample, and an
  independent raw-JSON rebuild of complete matches (>= 1,000 rows) with the audited K1-K3 code.

Writes ``post_run_verification.json`` and ``checksums_supplementary.json``.

Usage:
    .venv/bin/python scripts/verify_lstm_h5_experiment6c.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_experiments6a_6b as aud  # noqa: E402  (independently audited raw reconstruction, K1-K3)

from src.sequence_models.raw_events import EVENTS_DIR  # noqa: E402
from src.sequence_models.raw_history_check import load_events, reconstruct  # noqa: E402

CONFIG = PROJECT_ROOT / "configs/lstm_h5_experiment6c_v1.yaml"
DATASET = PROJECT_ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
FOLDS = PROJECT_ROOT / "results/markov_hmm_cost_sensitive_v2/tiers/tier_b/scenarioD15/folds.json"
SPLIT = PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json"
AUDIT_6AB = PROJECT_ROOT / "results/audit_experiments6a_6b_2026-11-24"
POST_RUN_FILES = ["post_run_verification.json"]
CLASSES = ["Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed", "Pressure",
           "Duel", "Interception", "Block", "Goal Keeper", "Foul Committed", "50/50"]       # written out, not imported
K, START, SHOT, PRESSURE, NAME = 15, 15, 5, 8, "LSTM-H5"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(a):
    return np.where(a >= 0, 1.0 / (1.0 + np.exp(-np.abs(a))), np.exp(-np.abs(a)) / (1.0 + np.exp(-np.abs(a))))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def lstm_forward(w: dict, ids: np.ndarray, layers: int) -> np.ndarray:
    """Own float64 LSTM: x_s one-hot(16); gates from PyTorch blocks [i; f; g; o]."""
    n = len(ids)
    x = np.zeros((n, 5, 16))
    x[np.arange(n)[:, None], np.arange(5)[None, :], ids] = 1.0
    seq = x
    for k in range(layers):
        Wih, Whh = w[f"lstm.weight_ih_l{k}"].astype(float), w[f"lstm.weight_hh_l{k}"].astype(float)
        b = w[f"lstm.bias_ih_l{k}"].astype(float) + w[f"lstm.bias_hh_l{k}"].astype(float)
        H = Whh.shape[1]
        h, c = np.zeros((n, H)), np.zeros((n, H))
        outs = []
        for s in range(5):
            a = seq[:, s] @ Wih.T + h @ Whh.T + b
            i, f, g, o = sigmoid(a[:, :H]), sigmoid(a[:, H:2 * H]), np.tanh(a[:, 2 * H:3 * H]), sigmoid(a[:, 3 * H:])
            c = f * c + i * g
            h = o * np.tanh(c)
            outs.append(h)
        seq = np.stack(outs, axis=1)
    return softmax(seq[:, -1] @ w["out.weight"].astype(float).T + w["out.bias"].astype(float))


def main() -> int:
    t_start = time.time()
    config = Path(sys.argv[sys.argv.index("--config") + 1]).resolve() if "--config" in sys.argv else CONFIG   # dry runs only
    cfg = yaml.safe_load(config.read_text())
    out, art = PROJECT_ROOT / cfg["output_dir"], PROJECT_ROOT / cfg["artefact_dir"]
    checks: list[dict] = []

    def check(name, ok, value=None):
        checks.append({"check": name, "result": "PASS" if ok else "FAIL",
                       "value": value.item() if isinstance(value, (np.floating, np.integer)) else value})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {'' if value is None else value}", flush=True)

    check("result directory complete (renamed from staging; no .partial left)",
          out.is_dir() and not out.with_name(out.name + ".partial").exists() and not (out / ".staging_experiment6c").exists())
    cs = json.loads((out / "checksums.json").read_text())
    bad = [k for k, v in cs.items() if sha(out / k) != v]
    check(f"result checksums ({len(cs)} files)", not bad, bad or len(cs))
    env = json.loads((out / "environment.json").read_text())
    for rel, h in env["artefacts_sha256"].items():
        check(f"artefact hash {rel}", sha(PROJECT_ROOT / rel) == h)
    audit = pd.read_csv(out / "integrity_audit.csv")
    check("run's own integrity audit has no FAIL", not (audit["result"] == "FAIL").any(), int((audit["result"] == "PASS").sum()))

    # ------------------------------------------------------------------ rows
    final = set(int(m) for m in json.loads(SPLIT.read_text())["final_test_match_ids"])
    table = pd.read_parquet(DATASET)
    check("canonical dataset contains no final-test match; 83 final-test ids", not (set(table["match_id"]) & final) and len(final) == 83)
    tb = table[table["tier"] == "tier_b"].reset_index(drop=True)
    del table
    elig = tb[tb["row_eligible"]].reset_index(drop=True)
    pred = pd.read_parquet(art / "oof_predictions_lstm_h5.parquet")
    y = pred["y"].to_numpy()
    fold = pred["fold_id"].to_numpy()
    match = pred["match_id"].to_numpy()
    check("690,147 rows, 304 matches, unique row ids", len(pred) == 690_147 and pred["match_id"].nunique() == 304
          and bool(pred["event_id"].is_unique), len(pred))
    check("row ids, order, matches and targets equal the canonical eligible rows",
          np.array_equal(pred["event_id"].to_numpy(), elig["event_id"].to_numpy())
          and np.array_equal(match, elig["match_id"].to_numpy()) and np.array_equal(y, elig["target_id"].to_numpy()))
    folds = json.loads(FOLDS.read_text())["scenarioD15"]
    fom = {int(m): int(f["fold_id"]) for f in folds for m in f["validation_match_ids"]}
    check("fold of every row = outer folds file; no train/validation match overlap; no final-test match in folds",
          np.array_equal(fold, pred["match_id"].map(fom).to_numpy()) and len(fom) == 304
          and all(not (set(f["train_match_ids"]) & set(f["validation_match_ids"])) for f in folds)
          and not any((set(f["train_match_ids"]) | set(f["validation_match_ids"])) & final for f in folds))
    prior = {}
    for label, key, models in (("experiment 5", "experiment5_predictions", ["XGB-H5", "markov2", "logistic_F0"]),
                               ("experiment 5B", "experiment5b_predictions", ["XGB-H2"]),
                               ("experiment 6A", "experiment6a_predictions", ["MLR-H5"]),
                               ("experiment 6B", "experiment6b_predictions", ["MLP-H5", "MLP-H5_seed1", "MLP-H5_seed2"])):
        o = pd.read_parquet(PROJECT_ROOT / cfg[key])
        check(f"paired with {label}: identical event_id / match / fold / target row by row",
              all(np.array_equal(o[c].to_numpy(), pred[c].to_numpy()) for c in ("event_id", "match_id", "fold_id", "y")))
        for m in models:
            prior[m] = o[[f"{m}__p{c}" for c in range(K)]].to_numpy()

    # ------------------------------------------------------------------ independent history + NumPy LSTM
    hist = pd.DataFrame({lag: tb.groupby("sequence_id")["symbol_id"].shift(lag) for lag in (4, 3, 2, 1)}).fillna(START).astype(int)
    hist = hist[tb["row_eligible"].to_numpy()].reset_index(drop=True)
    cur = elig["symbol_id"].to_numpy()
    ids = np.column_stack([hist[lag].to_numpy() for lag in (4, 3, 2, 1)] + [cur])
    pos = elig["seq_pos"].to_numpy()
    check("<START> at lag l exactly when seq_pos < l (no cross-period history); current event never <START> and last",
          all(np.array_equal(hist[lag].to_numpy() == START, pos < lag) for lag in (1, 2, 3, 4)) and bool((cur != START).all()))
    tm = json.loads((out / "task_manifest.json").read_text())["row_identity"]
    check("history hash in the run manifest equals the independently built history",
          tm["feature_matrix_sha256"] == hashlib.sha256(np.ascontiguousarray(ids.astype(np.int64)).tobytes()).hexdigest())
    npz = np.load(art / "fold_models_lstm_h5.npz")
    selected = {int(k): v for k, v in json.loads(str(npz["selected"])).items()}
    hc = pd.read_csv(out / "hyperparameter_choices.csv").set_index("fold_id")
    check("checkpoint provenance: stored configuration per fold = selected configuration",
          all(selected[f]["name"] == hc.at[f, "candidate"] and selected[f]["hidden"] == hc.at[f, "hidden"]
              and selected[f]["layers"] == hc.at[f, "layers"] for f in range(5)))
    reps = sorted({int(k.split("_rep")[1].split("__")[0]) for k in npz.files if k.startswith("fold0_rep")})
    pcols = {r: [f"{NAME}__p{c}" if r == 0 else f"{NAME}_seed{r}__p{c}" for c in range(K)] for r in reps}
    worst, shapes_ok, counts_ok = 0.0, True, True
    for f in range(5):
        sel = fold == f
        uniq, inv = np.unique(ids[sel], axis=0, return_inverse=True)
        H, L = int(selected[f]["hidden"]), int(selected[f]["layers"])
        for r in reps:
            w = {k.split("__", 1)[1]: npz[k] for k in npz.files if k.startswith(f"fold{f}_rep{r}__")}
            shapes_ok &= (w["lstm.weight_ih_l0"].shape == (4 * H, 16) and w["out.weight"].shape == (15, H)
                          and all(w[f"lstm.weight_hh_l{k}"].shape == (4 * H, H) for k in range(L)) and f"lstm.weight_ih_l{L}" not in w
                          and (L == 1 or w["lstm.weight_ih_l1"].shape == (4 * H, H)))
            n_par = sum(v.size for v in w.values())
            counts_ok &= n_par == sum(4 * H * m + 4 * H * H + 8 * H for m in [16] + [H] * (L - 1)) + 15 * H + 15
            p = lstm_forward(w, uniq, L)[inv.ravel()]
            worst = max(worst, float(np.abs(p - pred.loc[sel, pcols[r]].to_numpy()).max()))
    check("stored weights have the selected architecture (16 inputs, H, layers, 4 gate blocks, 15 logits)", bool(shapes_ok))
    check("stored parameter counts = 4Hm + 4H^2 + 8H per layer + 15H + 15", bool(counts_ok))
    check(f"probabilities recomputed by an own NumPy LSTM of the stored weights on an independently built history equal the stored "
          f"ones on all rows (|dp| < 1e-5; {len(reps)} replicates)", worst < 1e-5, worst)
    p = pred[pcols[0]].to_numpy()
    check("15 probability columns in canonical order; finite, positive; rows sum to 1 (1e-9)",
          bool(np.isfinite(p).all() and (p > 0).all()) and float(np.abs(p.sum(1) - 1).max()) <= 1e-9
          and abs(p.mean(0) - np.bincount(y, minlength=K) / len(y)).max() < 0.02, float(np.abs(p.sum(1) - 1).max()))

    # ------------------------------------------------------------------ training records
    fl = pd.read_csv(out / "fit_log.csv")
    th = pd.read_csv(out / "training_histories.csv")
    tcfg = cfg["training"]
    viol = []
    for a, b in zip(fl.itertuples(), th.itertuples()):
        hh = json.loads(b.history)
        stop = [e["stop_log_loss"] for e in hh]
        be = int(np.argmin(stop))
        n_ep = len(hh) - 1
        if not (a.candidate == b.candidate and a.best_epoch == be and abs(a.best_stop_log_loss - stop[be]) < 1e-15
                and n_ep == a.epochs_trained == min(be + int(tcfg["patience"]), int(tcfg["max_epochs"]))
                and [e["epoch"] for e in hh] == list(range(n_ep + 1))
                and all(e["n_batches"] == -(-a.n_train_rows // int(tcfg["batch_size"])) for e in hh[1:])):
            viol.append((a.fold_id, a.stage, a.candidate, a.held_inner_fold))
    n_fits = 5 * (4 * len(cfg["lstm"]["candidates"]) + 1 + int(tcfg["robustness_seeds"]))
    check(f"all {n_fits} fits: best epoch = first argmin of the stopping curve (epoch 0 included), stored best loss = curve "
          f"minimum, epochs = min(best + patience, max), batches per epoch = ceil(rows / 512)", not viol and len(fl) == n_fits,
          {"violations": viol[:5], "epoch0_best": int((fl.best_epoch == 0).sum()), "total_epochs": int(fl.epochs_trained.sum())})
    order = [c["name"] for c in cfg["lstm"]["candidates"]]
    inner = fl[fl.stage == "inner"].assign(s=lambda d: d.inner_validation_log_loss * d.inner_validation_rows)
    pooled = inner.groupby(["fold_id", "candidate"]).apply(lambda d: d.s.sum() / d.inner_validation_rows.sum(), include_groups=False)
    chosen = {f: min(order, key=lambda c: (round(pooled[(f, c)], 12), order.index(c))) for f in range(5)}
    check("selection recomputed from per-fit inner scores = stored choice = stored weights' configuration",
          all(chosen[f] == hc.at[f, "candidate"] == selected[f]["name"] for f in range(5)), chosen)
    sc = pd.read_csv(out / "inner_cv_scores.csv")
    nc = len(order)
    check(f"search space in the run = the {nc} frozen configurations", sorted(sc.candidate.unique()) == sorted(order)
          and sc.groupby("fold_id").size().eq(nc).all() and len(pd.read_csv(out / "search_candidates.csv")) == nc)
    check("gradient-norm statistics recorded for every fit (no clipping)", bool(fl.grad_norm_max.notna().all() and (fl.n_batches > 0).all()),
          {"max": float(fl.grad_norm_max.max()), "batches_above_1": int(fl.n_grad_norm_above_1.sum()), "batches": int(fl.n_batches.sum())})
    da = pd.read_csv(out / "data_access_audit.csv")
    da6 = pd.read_csv(PROJECT_ROOT / cfg["experiment6b_results_dir"] / "data_access_audit.csv")
    key = lambda d: sorted(zip(d.outer_fold.astype(int), d.purpose, d.match_ids_sha256))   # noqa: E731
    check("all inner/outer training, stopping and validation match sets are identical to experiment 6B (MLP-H5)", key(da) == key(da6), len(da))
    pop = []
    for f in folds:
        r0 = fl[(fl.fold_id == f["fold_id"]) & (fl.stage == "outer_refit")].iloc[0]
        trr = int(np.isin(match, f["train_match_ids"]).sum())
        pop.append(r0.n_train_rows / trr)
    check("refits update parameters on ~90 % of the outer-training rows (as MLP-H5; documented limitation vs XGB-H5)",
          all(0.85 < s < 0.95 for s in pop), [round(s, 4) for s in pop])

    # ------------------------------------------------------------------ metrics (scikit-learn)
    mc = pd.read_csv(out / "model_comparison.csv").set_index("model")
    d = p.argmax(1)
    mine = {"log_loss": log_loss(y, p, labels=list(range(K))), "brier": float(((p - np.eye(K)[y]) ** 2).sum(1).mean()),
            "accuracy": accuracy_score(y, d), "macro_f1": f1_score(y, d, labels=list(range(K)), average="macro", zero_division=0),
            "weighted_f1": f1_score(y, d, labels=list(range(K)), average="weighted", zero_division=0),
            "shot_average_precision": average_precision_score(y == SHOT, p[:, SHOT]),
            "shot_recall": recall_score(y == SHOT, d == SHOT, zero_division=0),
            "shot_precision": precision_score(y == SHOT, d == SHOT, zero_division=0)}
    diffs = {k: abs(v - mc.at[NAME, k]) for k, v in mine.items()}
    check("pooled LSTM-H5 metrics recomputed with scikit-learn equal the stored table (1e-9)", max(diffs.values()) < 1e-9,
          {k: float(v) for k, v in diffs.items()})
    for m in ("MLP-H5", "XGB-H5", "MLR-H5", "markov2"):
        check(f"{m}: stored pooled log loss in this run's table = scikit-learn on its saved predictions (1e-9)",
              abs(log_loss(y, prior[m], labels=list(range(K))) - mc.at[m, "log_loss"]) < 1e-9)
    bf = pd.read_csv(out / "metrics_by_fold.csv").query("model == @NAME").set_index("fold_id")
    w = max(max(abs(log_loss(y[fold == f], p[fold == f], labels=list(range(K))) - bf.at[f, "log_loss"]),
                abs(f1_score(y[fold == f], d[fold == f], labels=list(range(K)), average="macro", zero_division=0) - bf.at[f, "macro_f1"]))
            for f in range(5))
    check("per-fold log loss and macro-F1 recomputed equal the stored table (1e-9)", w < 1e-9, w)
    pc = pd.read_csv(out / "per_class_metrics.csv").query("model == @NAME").set_index("class_id")
    check("per-class support equals the target counts; Shot support 7,212",
          np.array_equal(pc["support"].to_numpy(), np.bincount(y, minlength=K)) and int(pc.at[SHOT, "support"]) == 7212)
    rob = pd.read_csv(out / "seed_robustness.csv").set_index("replicate")
    check("robustness-seed log losses recompute from stored predictions (1e-12)",
          all(abs(log_loss(y, pred[pcols[r]].to_numpy(), labels=list(range(K))) - rob.at[r, "pooled_log_loss"]) < 1e-12 for r in reps))

    # ------------------------------------------------------------------ paired comparisons
    pdf = pd.read_csv(out / "paired_differences.csv")
    mids, inv = np.unique(match, return_inverse=True)
    s_n = np.bincount(inv).astype(float)
    rng7 = np.random.default_rng(7)                     # re-implementation of the seed-7 convention (evaluation_v2)
    mult7 = np.stack([np.bincount(rng7.integers(0, len(mids), len(mids)), minlength=len(mids)) for _ in range(1000)]).astype(float)
    rng = np.random.default_rng(20261125)
    mult = np.stack([np.bincount(rng.integers(0, len(mids), len(mids)), minlength=len(mids)) for _ in range(2000)]).astype(float)
    nll = -np.log(p[np.arange(len(y)), y])
    for refm in ("MLP-H5", "XGB-H5", "MLR-H5", "markov2", "XGB-H2", "logistic_F0"):
        ref_nll = -np.log(prior[refm][np.arange(len(y)), y])
        di = nll - ref_nll
        row = pdf.query("model == @NAME and reference == @refm and metric == 'log_loss'").iloc[0]
        boot7 = (mult7 @ np.bincount(inv, weights=di)) / (mult7 @ s_n)
        lo7, hi7 = np.percentile(boot7, [2.5, 97.5])
        fs = [float(di[fold == f].mean()) for f in range(5)]
        check(f"{NAME} - {refm}: mean d_i, relative difference, seed-7 CI and fold signs reproduce the stored values (1e-12)",
              abs(di.mean() - row["difference"]) < 1e-12 and abs(di.mean() / ref_nll.mean() - row["relative_difference"]) < 1e-12
              and abs(lo7 - row["ci_low"]) < 1e-12 and abs(hi7 - row["ci_high"]) < 1e-12
              and sum(v < 0 for v in fs) == int(row["folds_negative"]) and sum(v > 0 for v in fs) == int(row["folds_positive"]),
              {"difference": float(di.mean()), "ci": [float(lo7), float(hi7)], "folds_better": sum(v < 0 for v in fs)})
        boot = (mult @ np.bincount(inv, weights=di)) / (mult @ s_n)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        check(f"independent match bootstrap vs {refm} (2,000 reps, seed 20261125): CI excludes zero iff the stored CI does",
              bool((lo > 0 or hi < 0) == bool(row["ci_excludes_zero"])), {"independent_ci": [float(lo), float(hi)]})
    st = pd.read_csv(out / "nll_difference_by_stratum.csv")
    pr = cur == PRESSURE
    ok = True
    for refm in ("MLP-H5", "XGB-H5", "MLR-H5"):
        di = nll + np.log(prior[refm][np.arange(len(y)), y])
        srow = st.query("reference == @refm and stratum == 'current_class' and level == 'Pressure'").iloc[0]
        ok &= abs(di[pr].sum() / len(y) - srow["contribution"]) < 1e-12 and int(srow["n_rows"]) == int(pr.sum())
    check("Pressure contribution (row-weighted) to the differences vs MLP-H5 / XGB-H5 / MLR-H5 recomputed = stored strata", bool(ok))

    # ------------------------------------------------------------------ access, history, inputs, raw JSON
    check("data-access audit: outer validation only predicted; stopping/training sets never touch it; final test never touched",
          not da.query("purpose != 'outer_validation_predict'")["touches_outer_validation"].any() and not da["touches_final_test"].any()
          and {"inner_train", "inner_stop", "inner_validation", "outer_train", "outer_stop"} <= set(da["purpose"]), len(da))
    for k in ("baseline_results_dir", "experiment5_results_dir", "experiment5b_results_dir", "experiment6a_results_dir",
              "experiment6b_results_dir"):
        dd = PROJECT_ROOT / cfg[k]
        c = json.loads((dd / "checksums.json").read_text())
        if (dd / "checksums_supplementary.json").exists():
            c = {**c, **json.loads((dd / "checksums_supplementary.json").read_text())["files"]}
        b = [x for x, v in c.items() if sha(dd / x) != v]
        check(f"historical results unchanged: {cfg[k]}", not b, b or len(c))
    c = json.loads((AUDIT_6AB / "checksums.json").read_text())
    check("6A/6B final audit results unchanged", all(sha(AUDIT_6AB / x) == v for x, v in c.items()), len(c))
    before = env["historical_verification"]["before"]["files"]
    check("stored predictions / models of experiments 5-6B and the canonical dataset unchanged since the run",
          all(sha(PROJECT_ROOT / rel) == h for rel, h in before.items()), len(before))
    rv = pd.read_csv(out / "raw_feature_verification.csv")
    okc, cache = 0, {}
    index_of = {e: i for i, e in enumerate(pred["event_id"].to_numpy())}
    for r in rv.itertuples():
        if r.match_id not in cache:
            cache = {r.match_id: load_events(EVENTS_DIR, int(r.match_id))}
        h, t, _ = reconstruct(cache[r.match_id], r.event_id, 4)
        i = index_of[r.event_id]
        okc += int(h == ids[i].tolist() and t == int(y[i]))
    check(f"raw-JSON rebuild of the {len(rv)} stored stratified rows equals the independently built history and target",
          okc == len(rv) and len(rv) >= 500, okc)
    rc = cfg["evaluation"]["raw_reconstruction"]
    sample = sorted(int(m) for m in np.random.default_rng(int(rc["seed"])).choice(np.unique(tb["match_id"]), int(rc["n_matches"]), replace=False))
    recs = [aud.rebuild_match(m)[0] for m in sample]
    rec = pd.concat([r[r["eligible"]] for r in recs], ignore_index=True)
    ii = rec["event_id"].map(index_of).to_numpy()
    same = bool(np.array_equal(rec[["h4", "h3", "h2", "h1", "cur"]].to_numpy(), ids[ii]) and np.array_equal(rec["target"].to_numpy(), y[ii]))
    check(f"independent raw-JSON reconstruction of {len(sample)} complete matches ({len(rec)} rows; periods starts/ends, <START>, "
          f"companions, repeated timestamps, missing 360) equals the independently built history and targets",
          same and len(rec) >= 1000 and not set(sample) & final,
          {"rows": len(rec), "seq_pos_0": int((rec.seq_pos == 0).sum()), "period_end": int(rec.period_end.sum()),
           "companion_in_window": int(rec.companion_in_window.sum()), "repeated_timestamp": int(rec.repeated_timestamp.sum()),
           "history_without_360": int(rec.history_without_360.sum()),
           "rare_classes": {c_: int(((rec.cur == CLASSES.index(c_)) | (rec.target == CLASSES.index(c_))).sum())
                            for c_ in ("50/50", "Goal Keeper", "Duel", "Shot", "Block")}})
    status = art / "run_status.json"
    if status.exists():
        s = json.loads(status.read_text())
        exp = [x for x in s["stages"] if x["stage"] == "experiment"]
        check("supervisor: the experiment stage exited 0 before this audit started; caffeinate attached to the supervisor",
              bool(exp) and exp[-1]["exit_code"] == 0 and s["caffeinate_command"] == f"caffeinate -d -i -w {s['supervisor_pid']}",
              {"run_id": s["run_id"], "experiment_exit_code": exp[-1]["exit_code"] if exp else None})

    n_fail = sum(c["result"] == "FAIL" for c in checks)
    report = {"verdict": "PASS" if n_fail == 0 else "FAIL", "n_checks": len(checks), "n_fail": n_fail,
              "runtime_seconds": round(time.time() - t_start), "checks": checks}
    (out / "post_run_verification.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "checksums_supplementary.json").write_text(json.dumps({
        "description": "sha256 of files created AFTER the 6C run; checksums.json is the run's original manifest and is not rewritten",
        "files": {f: sha(out / f) for f in POST_RUN_FILES},
        "generated_by": {f: "scripts/verify_lstm_h5_experiment6c.py" for f in POST_RUN_FILES}}, indent=1))
    print(f"\nverdict: {report['verdict']} ({len(checks)} checks, {n_fail} FAIL)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
