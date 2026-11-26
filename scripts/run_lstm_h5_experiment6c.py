"""Experiment 6C: long short-term memory network with a five-event history (LSTM-H5) on the
canonical task actions15_360current_v1.

Same rows (690,147), outer folds, inner folds, history and stopping subsets as experiment 6B
(MLP-H5). Final-test matches are never loaded. The canonical dataset, the stored predictions of
experiments 5 / 5B / 6A / 6B and all historical result directories are read only and
hash-verified before and after the run.

Execution. After all input verification the runner writes a compact, hash-checked worker input
(history ids, targets, matches, folds; Tier B rows only) and computes the five outer folds in
separate single-threaded worker processes (``--worker K``). Each fold is a deterministic
function of (data, configuration, seeds), so this equals a sequential run; this is verified by
retraining fold 0's primary refit in the main process (bitwise). Every fit is checkpointed at
epoch boundaries and every completed fit / outer fold is committed atomically; a restarted run
resumes only from compatible checkpoints.

Results are written to ``<output_dir>.partial`` and renamed to ``<output_dir>`` only after the
integrity audit and the checksums are complete. Lock files prevent duplicate runners / workers.

Usage (normally started by scripts/supervise_lstm_h5_experiment6c.py):
    .venv/bin/python scripts/run_lstm_h5_experiment6c.py --smoke   # fold-0 inner data only: timing / memory / worker plumbing
    .venv/bin/python scripts/run_lstm_h5_experiment6c.py           # full run
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import logging
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.sequence_models import lstm_h5_experiment as lm  # noqa: E402
from src.sequence_models.canonical_task import CLASSES, K, TASK_VERSION  # noqa: E402
from src.sequence_models.experiment import environment_info, sha256_file, write_json  # noqa: E402
from src.sequence_models.splits import FoldAssignment, LeakageError  # noqa: E402
from src.sequence_models.xgboost_h2_experiment import sha256_array, source_digest  # noqa: E402

LOGGER = logging.getLogger("lstm_h5_experiment6c")
CONFIG = PROJECT_ROOT / "configs" / "lstm_h5_experiment6c_v1.yaml"
NAME = lm.FEATURE_SET
EXP5_MODELS = ["frequency", "markov1", "markov2", "logistic_F0", "XGB-F0", "XGB-H5"]
PRIOR_MODELS = EXP5_MODELS + ["XGB-H2", "MLR-H5", "MLP-H5"]
PAIRS = [(NAME, "MLP-H5"), (NAME, "XGB-H5"), (NAME, "MLR-H5"), (NAME, "markov2"), (NAME, "XGB-H2"), (NAME, "logistic_F0"),
         (NAME, "markov1"), (NAME, "XGB-F0")]
HIST_DIRS = ("baseline_results_dir", "experiment5_results_dir", "experiment5b_results_dir", "experiment6a_results_dir",
             "experiment6b_results_dir")
HIST_FILES = ("experiment5_predictions", "experiment5b_predictions", "experiment6a_predictions", "experiment6b_predictions")
MODEL_SOURCES = [PROJECT_ROOT / p for p in ("src/sequence_models/lstm_h5_experiment.py", "src/sequence_models/mlp_h5_experiment.py",
                                            "src/sequence_models/event_history.py", "src/sequence_models/canonical_task.py")]
ALL_SOURCES = MODEL_SOURCES + [PROJECT_ROOT / p for p in (
    "src/sequence_models/xgboost_experiment.py", "src/sequence_models/evaluation_v2.py",
    "src/sequence_models/logistic_h5_experiment.py", "src/sequence_models/raw_history_check.py",
    "scripts/run_lstm_h5_experiment6c.py", "scripts/run_logistic_h5_experiment6a.py", "scripts/run_xgboost_experiment5.py",
    "scripts/supervise_lstm_h5_experiment6c.py", "scripts/verify_lstm_h5_experiment6c.py", "scripts/audit_experiments6a_6b.py")]


def maxrss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6          # bytes on macOS


# ============================================================================ historical artefacts
def historical_snapshot(cfg: dict) -> dict:
    out = {"directories": {}, "files": {}}
    for key in HIST_DIRS:
        d = PROJECT_ROOT / cfg[key]
        entry = {"mismatches": []}
        for manifest in ("checksums.json", "checksums_supplementary.json"):
            if (d / manifest).exists():
                c = json.loads((d / manifest).read_text())
                c = c.get("files", c) if manifest == "checksums_supplementary.json" else c
                entry["mismatches"] += [k for k, v in c.items() if sha256_file(d / k) != v]
                entry[f"{manifest}_sha256"] = sha256_file(d / manifest)
                entry[f"{manifest}_files"] = len(c)
        entry["readme_sha256"] = sha256_file(d / "README.md")
        entry["n_files"] = sum(1 for p in d.rglob("*") if p.is_file())
        out["directories"][cfg[key]] = entry
    for key in HIST_FILES:
        out["files"][cfg[key]] = sha256_file(PROJECT_ROOT / cfg[key])
    for rel in ("data/interim/canonical_task/actions15_360current_v1/observations.parquet",
                "data/interim/canonical_task/actions15_360current_v1/manifest.json",
                "data/interim/logistic_h5_experiment6a/fold_models_mlr_h5.npz",
                "data/interim/mlp_h5_experiment6b/fold_models_mlp_h5.npz",
                "data/interim/mlp_h5_experiment6b/fold_models_mlp_h5.pt",
                "results/audit_experiments6a_6b_2026-11-24/checksums.json"):
        out["files"][rel] = sha256_file(PROJECT_ROOT / rel)
    return out


def load_prior_predictions(cfg: dict, rows, fold_of_row: np.ndarray) -> tuple[dict, list[dict]]:
    """Stored out-of-fold predictions of experiments 5, 5B, 6A and 6B, aligned row by row."""
    from src.sequence_models import xgboost_experiment as xe
    checks = []
    manifests = {"experiment 5": cfg["experiment5_results_dir"], "experiment 6A": cfg["experiment6a_results_dir"],
                 "experiment 6B": cfg["experiment6b_results_dir"]}
    for label, d in manifests.items():
        r = json.loads((PROJECT_ROOT / d / "task_manifest.json").read_text())["row_identity"]
        checks.append((f"{label} manifest: event-id hash equals this run's rows (n = {r['n_rows']})",
                       r["event_id_sha256"] == sha256_array(rows.event_id) and r["n_rows"] == len(rows.y), r["n_rows"]))
        checks.append((f"{label} manifest: target hash equals this run's targets", r["target_sha256"] == sha256_array(rows.y), None))
        checks.append((f"{label} manifest: fold-of-row hash equals this run's folds", r["fold_of_row_sha256"] == sha256_array(fold_of_row), None))
        if "feature_matrix_sha256" in r:
            checks.append((f"{label} manifest: history (feature-matrix) hash equals this run's five-event history",
                           r["feature_matrix_sha256"] == sha256_array(rows.ctx), None))
    proba = {}
    for label, key, models in (("experiment 5", "experiment5_predictions", EXP5_MODELS),
                               ("experiment 5B", "experiment5b_predictions", ["XGB-H2"]),
                               ("experiment 6A", "experiment6a_predictions", ["MLR-H5"]),
                               ("experiment 6B", "experiment6b_predictions", ["MLP-H5", "MLP-H5_seed1", "MLP-H5_seed2"])):
        o = pd.read_parquet(PROJECT_ROOT / cfg[key])
        for col, ref in (("event_id", rows.event_id), ("match_id", rows.match_id), ("y", rows.y), ("fold_id", fold_of_row)):
            checks.append((f"{label} predictions: {col} identical row by row", np.array_equal(o[col].to_numpy(), ref), None))
        checks.append((f"{label} predictions: unique row ids", bool(o["event_id"].is_unique), len(o)))
        for m in models:
            proba[m] = o[[f"{m}__p{c}" for c in range(K)]].to_numpy()
            xe.check_probabilities(proba[m], float(cfg["evaluation"]["probability_tolerance"]))
    return proba, [{"check": c, "result": "PASS" if ok else "FAIL", "value": v} for c, ok, v in checks]


# ============================================================================ independent raw reconstruction
def raw_reconstruction(cfg: dict, tb: pd.DataFrame, ids: np.ndarray, final_test: set) -> tuple[pd.DataFrame, dict]:
    """Complete matches rebuilt from raw StatsBomb JSON with the audited K1-K3 implementation of
    the 6A/6B final audit (shares no code with canonical_task / event_history)."""
    import audit_experiments6a_6b as aud
    rc = cfg["evaluation"]["raw_reconstruction"]
    rng = np.random.default_rng(int(rc["seed"]))
    sample = sorted(int(m) for m in rng.choice(np.unique(tb["match_id"]), int(rc["n_matches"]), replace=False))
    if set(sample) & final_test:
        raise LeakageError("raw reconstruction sample contains a final-test match")
    elig = tb[tb["row_eligible"]].reset_index(drop=True)
    pos = {e: i for i, e in enumerate(elig["event_id"])}
    recs, obs_ok = [], True
    for mid in sample:
        r, _ = aud.rebuild_match(mid)
        prod = tb[tb["match_id"] == mid]
        obs_ok &= (len(r) == len(prod) and list(r["event_id"]) == list(prod["event_id"]) and list(r["cur"]) == list(prod["symbol_id"])
                   and list(r["eligible"]) == list(prod["row_eligible"]) and list(r["index"]) == list(prod["index"]))
        recs.append(r[r["eligible"]])
    rec = pd.concat(recs, ignore_index=True)
    i = rec["event_id"].map(pos).to_numpy()
    rec["history_equal"] = (rec[["h4", "h3", "h2", "h1", "cur"]].to_numpy() == ids[i]).all(axis=1)
    rec["target_equal"] = (rec["target"].to_numpy() == elig["target_id"].to_numpy()[i]) & \
        (rec["target_event_id"].to_numpy() == elig["target_event_id"].to_numpy()[i])
    strata = {"rows": len(rec), "matches": len(sample), "seq_pos_0": int((rec.seq_pos == 0).sum()),
              "with_<START>": int((rec.h4 == 15).sum()), "period_end": int(rec.period_end.sum()),
              "repeated_timestamp": int(rec.repeated_timestamp.sum()), "companion_in_window": int(rec.companion_in_window.sum()),
              "history_event_without_360": int(rec.history_without_360.sum()),
              **{f"current_or_target_{c}": int(((rec.cur == CLASSES.index(c)) | (rec.target == CLASSES.index(c))).sum())
                 for c in ("Shot", "Goal Keeper", "Duel", "50/50", "Block", "Foul Committed")}}
    return rec, {"observations_identical": bool(obs_ok), "strata": strata}


# ============================================================================ mathematical verification on real windows
def mathematical_checks(cfg: dict, ids: np.ndarray, y: np.ndarray) -> dict:
    """Before training: for every candidate architecture (fresh initialisation) the independent
    NumPy reference equals nn.LSTM on 2,000 real windows; parameter counts; float64 finite
    differences of the NumPy loss vs autograd. Tolerances: float32 network vs float64 reference
    |dz| < 1e-5 (float32 unit roundoff 6e-8 accumulated over <= 128-term dot products and 5
    recurrent steps), float64 vs float64 < 1e-12."""
    rng = np.random.default_rng(0)
    sel = rng.choice(len(y), 2000, replace=False)
    xi, yi = ids[sel], y[sel]
    x = lm.one_hot_sequence(xi)
    out = {"candidates": [], "tolerances": {"float32_logits": 1e-5, "float64": 1e-12, "finite_difference": 1e-7}}
    for c in cfg["lstm"]["candidates"]:
        m = lm.build_model(c, 1).eval()
        with torch.no_grad():
            z32 = m(torch.from_numpy(xi)).double().numpy()
        z_np, p_np, st = lm.numpy_lstm_forward(m.export(), x, c["layers"], return_states=True)
        m64 = lm.model_from_state(c, m.state_dict()).double().eval()
        with torch.no_grad():
            seq, (h_n, c_n) = m64.lstm(m64.encode(torch.from_numpy(xi)))
            z64 = m64(torch.from_numpy(xi)).numpy()
        f = lm.parameter_count_formula(c["hidden"], c["layers"])
        out["candidates"].append({
            "name": c["name"], "parameters_formula": f["total"], "parameters_pytorch": lm.count_parameters(m),
            "max_abs_logit_float32_vs_numpy": float(np.abs(z32 - z_np).max()),
            "max_abs_logit_float64_vs_numpy": float(np.abs(z64 - z_np).max()),
            "max_abs_hidden_states_float64": float(np.abs(seq.numpy() - st[-1]["h"]).max()),
            "max_abs_final_cell_states_float64": float(max(np.abs(c_n[k].numpy() - st[k]["c"][:, -1]).max() for k in range(c["layers"]))),
            "max_abs_probability_float64": float(np.abs(lm.softmax64(z64) - p_np).max())})
    m = lm.LSTMH5(4, 2).double()
    with torch.no_grad():
        for p in m.parameters():
            p.uniform_(-0.8, 0.8)
    xs, ys = xi[:64], yi[:64]
    m.zero_grad()
    F.cross_entropy(m(torch.from_numpy(xs)), torch.from_numpy(ys)).backward()
    params, x64 = m.export(), lm.one_hot_sequence(xs)
    err, n = 0.0, 0
    for name, p in m.named_parameters():
        flat = params[name].reshape(-1)
        for j in rng.choice(flat.size, min(10, flat.size), replace=False):
            old = flat[j]
            flat[j] = old + 1e-6
            lp = lm.numpy_nll(lm.numpy_lstm_forward(params, x64, 2)[0], ys)
            flat[j] = old - 1e-6
            lq = lm.numpy_nll(lm.numpy_lstm_forward(params, x64, 2)[0], ys)
            flat[j] = old
            err, n = max(err, abs((lp - lq) / 2e-6 - p.grad.reshape(-1)[j].item())), n + 1
    out["finite_difference"] = {"architecture": "H=4, 2 layers, float64, 64 real windows", "entries_checked": n,
                                "max_abs_error": err}
    t = out["tolerances"]
    out["pass"] = bool(all(r["parameters_formula"] == r["parameters_pytorch"] and r["max_abs_logit_float32_vs_numpy"] < t["float32_logits"]
                           and max(r["max_abs_logit_float64_vs_numpy"], r["max_abs_hidden_states_float64"],
                                   r["max_abs_final_cell_states_float64"], r["max_abs_probability_float64"]) < t["float64"]
                           for r in out["candidates"]) and err < t["finite_difference"])
    return out


# ============================================================================ worker processes
def write_worker_input(path: Path, ids, match_id, y, folds, identifiers) -> str:
    tmp = path.with_name("." + path.name + ".tmp.npz")
    np.savez(tmp, ids=ids.astype(np.int64), match_id=np.asarray(match_id, dtype=np.int64), y=np.asarray(y, dtype=np.int64),
             folds=json.dumps([{"fold_id": f.fold_id, "train_match_ids": list(f.train_match_ids),
                                "validation_match_ids": list(f.validation_match_ids)} for f in folds]),
             identifiers=json.dumps(identifiers, sort_keys=True))
    os.replace(tmp, path)
    return sha256_file(path)


def worker(fold: int, path: Path, digest: str, check_only: bool) -> int:
    """One outer fold in its own process; reads only the hash-checked worker input."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    art = PROJECT_ROOT / cfg["artefact_dir"]
    if sha256_file(path) != digest:
        LOGGER.error("worker input hash differs from the runner's; refusing")
        return 11
    lock = open(art / f"worker_fold{fold}{'_check' if check_only else ''}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOGGER.error("another worker for fold %d holds the lock; refusing to start a duplicate", fold)
        return 3
    d = np.load(path, allow_pickle=False)
    ids, match_id, y = d["ids"], d["match_id"], d["y"]
    ident = json.loads(str(d["identifiers"]))
    folds = [FoldAssignment(int(f["fold_id"]), tuple(f["train_match_ids"]), tuple(f["validation_match_ids"]))
             for f in json.loads(str(d["folds"]))]
    if (sha256_array(ids) != ident["feature_matrix_sha256"] or sha256_array(y) != ident["target_sha256"]
            or sha256_file(CONFIG) != ident["config_sha256"] or len(y) != 690_147 or len(np.unique(match_id)) != 304):
        LOGGER.error("worker input does not match the run identifiers / frozen configuration")
        return 12
    status_path = art / f"worker_fold{fold}{'_check' if check_only else ''}_status.json"
    st = {"fold": fold, "pid": os.getpid(), "parent_pid": os.getppid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "state": "check_only" if check_only else "running"}
    write_json(status_path, st)
    if check_only:
        LOGGER.info("worker %d: input verified (check only)", fold)
        return 0
    parent = os.getppid()

    def watchdog():                      # the runner died: stop (checkpoints are atomic; a restart resumes)
        while True:
            time.sleep(20)
            if os.getppid() != parent:
                LOGGER.error("worker %d: parent runner %d is gone; exiting", fold, parent)
                os._exit(13)
    threading.Thread(target=watchdog, daemon=True).start()
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = lm.run_nested_cv(ids, match_id, y, folds, cfg, PROJECT_ROOT / cfg["checkpoint_dir"], ident, trace=[],
                               log=LOGGER.info, only_folds=[fold])
    st.update({"state": "completed", "ended": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": time.time() - t0,
               "computed_folds": res.computed_folds, "resumed_folds": res.resumed_folds, "resumed_fits": res.resumed_fits,
               "maxrss_mb": maxrss_mb(), "warnings": [f"{w.category.__name__}: {str(w.message)[:300]}" for w in caught],
               "torch_threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled()})
    write_json(status_path, st)
    LOGGER.info("worker %d done in %.0f s (maxrss %.0f MB, %d warnings)", fold, st["seconds"], st["maxrss_mb"], len(st["warnings"]))
    return 0


def run_workers(folds_todo: list[int], art: Path, wpath: Path, digest: str, parallel: int, check_only: bool = False) -> dict:
    """Start one worker per fold (at most ``parallel`` at a time); returns {fold: exit code}."""
    procs: dict[int, subprocess.Popen] = {}
    codes: dict[int, int] = {}
    queue = list(folds_todo)

    def terminate(signum, _frame):
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        raise SystemExit(128 + signum)
    old = signal.signal(signal.SIGTERM, terminate)
    reg = art / ("workers_check.json" if check_only else "workers.json")
    try:
        while queue or procs:
            while queue and len(procs) < parallel:
                k = queue.pop(0)
                logf = open(art / f"worker_fold{k}{'_check' if check_only else ''}.log", "a")
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(k), "--input", str(wpath), "--sha256", digest,
                       "--config", str(CONFIG)]
                procs[k] = subprocess.Popen(cmd + (["--check-only"] if check_only else []), cwd=PROJECT_ROOT,
                                            stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT)
                LOGGER.info("worker for fold %d started: pid %d", k, procs[k].pid)
                write_json(reg, {"runner_pid": os.getpid(), "workers": {str(f): {"pid": p.pid} for f, p in procs.items()},
                                 "exit_codes": {str(f): c for f, c in codes.items()}})
            time.sleep(1 if check_only else 15)
            for k, p in list(procs.items()):
                if p.poll() is not None:
                    codes[k] = p.returncode
                    LOGGER.info("worker for fold %d (pid %d) exited with %d", k, p.pid, p.returncode)
                    del procs[k]
                    write_json(reg, {"runner_pid": os.getpid(), "workers": {str(f): {"pid": q.pid} for f, q in procs.items()},
                                     "exit_codes": {str(f): c for f, c in codes.items()}})
    finally:
        signal.signal(signal.SIGTERM, old)
    return codes


# ============================================================================ smoke test
def smoke(cfg: dict, rows, folds, ids: np.ndarray, art: Path, identifiers: dict) -> None:
    """Fold-0 INNER data only (outer-validation rows of fold 0 never touched): one inner split,
    the cheapest and the most expensive candidate, 3 epochs each; timing, memory; worker
    plumbing check (input hash, locks, exit codes) without training."""
    tcfg = {**cfg["training"], "max_epochs": 3}
    lm.configure_torch(tcfg)
    f = folds[0]
    inner = [g for g in folds if g.fold_id != f.fold_id]
    held = inner[0]
    itr = [m for g in inner if g is not held for m in g.validation_match_ids]
    fit_m, stop_m = lm.stopping_split(itr, float(tcfg["stopping_fraction"]), 1)
    if (set(itr) | set(held.validation_match_ids)) & set(f.validation_match_ids):
        raise LeakageError("smoke test touched outer-validation matches")
    X = torch.from_numpy(ids)
    y = torch.from_numpy(rows.y)
    tr, st = np.flatnonzero(rows.mask(fit_m)), np.flatnonzero(rows.mask(stop_m))
    report = []
    cands = {c["name"]: c for c in cfg["lstm"]["candidates"]}
    for cand in (cands["A"], cands["D"], cands["F"]):
        r = lm.train_fit(f"smoke_{cand['name']}", X, y, tr, st, cand, tcfg, 1)
        hv = np.flatnonzero(rows.mask(held.validation_match_ids))
        p = lm.predict_proba(lm.model_from_state(cand, r.best_state), ids[hv])
        ep = [h["seconds"] for h in r.history[1:]]
        report.append({"candidate": cand, "train_rows": len(tr), "stop_rows": len(st), "seconds_per_epoch": float(np.mean(ep)),
                       "stop_log_loss_by_epoch": [h["stop_log_loss"] for h in r.history],
                       "grad_norm_max_by_epoch": [h["grad_norm_max"] for h in r.history[1:]],
                       "max_rss_mb": maxrss_mb(), "proba_shape": list(p.shape),
                       "proba_row_sum_max_dev": float(np.abs(p.sum(1) - 1).max())})
        LOGGER.info("smoke %s: %.2f s/epoch, stop LL %s, grad-norm max %s, maxrss %.0f MB", cand["name"],
                    report[-1]["seconds_per_epoch"], [round(v, 4) for v in report[-1]["stop_log_loss_by_epoch"]],
                    [round(v, 3) for v in report[-1]["grad_norm_max_by_epoch"]], report[-1]["max_rss_mb"])
    wpath = art / "worker_input_smoke.npz"
    digest = write_worker_input(wpath, ids, rows.match_id, rows.y, folds, identifiers)
    codes = run_workers([0, 1], art, wpath, digest, 2, check_only=True)
    bad = run_workers([2], art, wpath, "0" * 64, 1, check_only=True)
    wpath.unlink()
    LOGGER.info("worker plumbing: valid input -> exit codes %s; wrong hash -> %s (expected 11)", codes, bad)
    n_fits = 5 * (4 * len(cfg["lstm"]["candidates"]) + 1 + int(cfg["training"]["robustness_seeds"]))
    write_json(art / "smoke_test.json", {"note": "technical smoke test; inner data of outer fold 0 only; not a result",
                                         "fits": report, "n_fits_full_run": n_fits,
                                         "worker_check_exit_codes": codes, "worker_wrong_hash_exit_codes": bad})
    if codes != {0: 0, 1: 0} or bad != {2: 11}:
        raise RuntimeError("worker plumbing check failed")


# ============================================================================ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--sha256", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--config", type=Path, default=None,
                        help="technical dry runs only (miniature configuration, outputs outside the repository)")
    args = parser.parse_args()
    global CONFIG
    if args.config is not None:
        CONFIG = args.config.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.captureWarnings(True)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.worker is not None:
        lm.configure_torch(cfg["training"])
        return worker(args.worker, args.input, args.sha256, args.check_only)
    started = time.time()
    cfg5 = yaml.safe_load((PROJECT_ROOT / cfg["experiment5_config"]).read_text(encoding="utf-8"))
    for key in ("bootstrap_replicates", "bootstrap_seed", "probability_tolerance", "permutation_repeats", "permutation_seed"):
        if cfg["evaluation"][key] != cfg5["evaluation"][key]:
            raise SystemExit(f"evaluation.{key} differs from experiment 5")
    t6b = yaml.safe_load((PROJECT_ROOT / cfg["experiment6b_config"]).read_text(encoding="utf-8"))["training"]
    for key in ("optimizer", "betas", "eps", "batch_size", "max_epochs", "patience", "min_delta", "stopping_fraction", "seed",
                "robustness_seeds", "deterministic_algorithms", "device"):
        if cfg["training"][key] != t6b[key] and not (args.config is not None and key in ("max_epochs", "patience")):
            raise SystemExit(f"training.{key} differs from experiment 6B")
    if cfg["feature_set"] != {"name": NAME, "n_positions": lm.N_POSITIONS, "input_dim": lm.N_INPUT, "representation": "one-hot"}:
        raise SystemExit("feature_set in the configuration differs from the implementation")
    lm.configure_torch(cfg["training"])
    art = PROJECT_ROOT / cfg["artefact_dir"]
    art.mkdir(parents=True, exist_ok=True)
    lock = open(art / ("runner_smoke.lock" if args.smoke else "runner.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOGGER.error("another experiment-6C runner holds the lock; refusing to start a duplicate")
        return 3
    out = PROJECT_ROOT / cfg["output_dir"]
    staging = out.with_name(out.name + ".partial")
    ckpt = PROJECT_ROOT / cfg["checkpoint_dir"]
    if not args.smoke and out.exists():
        LOGGER.error("%s exists: the experiment is complete; refusing to overwrite results", cfg["output_dir"])
        return 4

    # ------------------------------------------------------------------ inputs (verified)
    from build_canonical_task import raw_source_hashes
    from run_logistic_h5_experiment6a import raw_verification
    from run_xgboost_experiment5 import load_inputs

    from src.sequence_models import canonical_baselines as cb
    from src.sequence_models import logistic_h5_experiment as lh
    from src.sequence_models import xgboost_experiment as xe
    from src.sequence_models.event_history import PAD, HistoryRows, history_ids, position_label
    hist_before = historical_snapshot(cfg)
    bad = {d: e["mismatches"] for d, e in hist_before["directories"].items() if e["mismatches"]}
    if bad:
        LOGGER.error("historical results changed before the run: %s", bad)
        return 5
    inp = load_inputs(cfg)        # canonical hashes vs manifests, Tier B set, folds == baseline folds, no final test
    tb, folds, part = inp["tb"], inp["folds"], inp["part"]
    final_test = set(part.final_test)
    dataset_matches = set(inp["table"]["match_id"])
    del inp["table"]
    rows = HistoryRows.from_table(tb)
    ids = rows.ctx
    same_xgb = bool(np.array_equal(ids, rows.features("XGB-H5")))
    same_path = bool(np.array_equal(ids, history_ids(tb, 4)[tb["row_eligible"].to_numpy()]))
    brows = cb.Rows.from_table(tb)
    same_base = bool(np.array_equal(ids[:, 4], brows.cur) and np.array_equal(rows.y, brows.y)
                     and np.array_equal(rows.match_id, brows.match_id))
    if not (same_xgb and same_path and same_base):
        raise RuntimeError("LSTM-H5 history differs from the XGB-H5 history / second construction path / baseline rows")
    if len(rows.y) != inp["manifest"]["counts_tier_b"]["eligible_rows"] or len(rows.y) != 690_147 or len(set(rows.match_id)) != 304:
        LOGGER.error("row count %d / matches %d differ from the protocol", len(rows.y), len(set(rows.match_id)))
        return 6
    LOGGER.info("%s: %d rows, %d matches, history ids %s; torch %s, device cpu, %d thread(s) per process", TASK_VERSION,
                len(rows.y), len(set(rows.match_id)), ids.shape, torch.__version__, cfg["training"]["num_threads"])
    identifiers = {
        "experiment": cfg["experiment_name"], "task_version": TASK_VERSION, "config_sha256": sha256_file(CONFIG),
        "dataset_sha256_tier_b": inp["manifest"]["dataset_sha256_tier_b"],
        "folds_file_sha256": sha256_file(PROJECT_ROOT / inp["spec"].raw["partition"]["folds_file"]),
        "split_manifest_sha256": sha256_file(PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json"),
        "row_event_id_sha256": sha256_array(rows.event_id), "feature_matrix_sha256": sha256_array(ids),
        "target_sha256": sha256_array(rows.y), "model_source_digest": source_digest(MODEL_SOURCES)["digest"],
        "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "device": cfg["training"]["device"], "num_threads": int(cfg["training"]["num_threads"]),
    }
    if args.smoke:
        smoke(cfg, rows, folds, ids, art, identifiers)
        return 0

    integrity: list[dict] = []

    def check(name, ok, value=None, kind="PASS/FAIL"):
        integrity.append({"check": name, "result": ("PASS" if ok else "FAIL") if kind == "PASS/FAIL" else kind, "value": value})

    check("history identical to XGB-H5 and to a second construction path; (E_t, y, match) = canonical baseline rows",
          same_xgb and same_path and same_base)
    fold_of_row = np.full(len(rows.y), -1)
    for f in folds:
        fold_of_row[rows.mask(f.validation_match_ids)] = f.fold_id
    prior, prior_checks = load_prior_predictions(cfg, rows, fold_of_row)
    integrity += prior_checks
    if any(c["result"] == "FAIL" for c in prior_checks):
        pd.DataFrame(integrity).to_csv(art / "alignment_failure.csv", index=False)
        LOGGER.error("row-level alignment with experiments 5 / 5B / 6A / 6B failed; not training")
        return 7

    # ------------------------------------------------------------------ input / data verification (before training)
    x_check = lm.one_hot_sequence(ids[:5000])
    xs = lh.encode_sparse(ids)
    fver = {"input_shape_per_row": [lm.N_POSITIONS, lm.N_INPUT],
            "one_hot_recovers_ids_sample_5000": bool(np.array_equal(x_check.argmax(-1), ids[:5000]) and (x_check.sum(-1) == 1).all()),
            "position_specific_79_column_encoding_is_a_bijection_of_the_same_ids": bool(np.array_equal(
                np.sort(ids + np.array([0, 16, 32, 48, 64]), axis=1), xs.indices.reshape(-1, 5))),
            "e_t_never_start": bool((ids[:, 4] != PAD).all()),
            "start_count_by_lag": {position_label(lag): int((ids[:, 4 - lag] == PAD).sum()) for lag in range(1, 5)},
            "rows_with_seq_pos_below_lag": {position_label(lag): int((rows.seq_pos < lag).sum()) for lag in range(1, 5)},
            "n_unique_windows": int(len(np.unique(ids, axis=0)))}
    fver["start_padding_iff_before_sequence_start"] = fver["start_count_by_lag"] == fver["rows_with_seq_pos_below_lag"]
    del xs
    check("one-hot (5 x 16) input carries exactly the ids of the 79-column 6A / 6B encoding (same information)",
          fver["one_hot_recovers_ids_sample_5000"] and fver["position_specific_79_column_encoding_is_a_bijection_of_the_same_ids"])
    check("<START> at lag l exactly when seq_pos < l; E_t (last position) never <START>",
          fver["start_padding_iff_before_sequence_start"] and fver["e_t_never_start"])
    rv_cfg = yaml.safe_load((PROJECT_ROOT / cfg["experiment6a_config"]).read_text())["evaluation"]["raw_verification"]
    rv = raw_verification(rows, tb, rv_cfg)
    rv_cols = ["features_match", "one_hot_names_match", "start_padding_correct", "target_match", "history_strictly_before_target"]
    rv_ok = bool(rv[rv_cols].all().all()) and len(rv) >= 500
    rv.to_csv(art / "raw_feature_verification_pre_run.csv", index=False)
    check(f"raw-JSON rebuild of {len(rv)} stratified rows (6A sample: history, <START>, target, causality)", rv_ok,
          json.dumps(rv["strata"].str.split(";").explode().value_counts().to_dict()))
    rec, rinfo = raw_reconstruction(cfg, tb, ids, final_test)
    rec_ok = rinfo["observations_identical"] and bool(rec.history_equal.all() and rec.target_equal.all()) and len(rec) >= 1000 \
        and all(v > 0 for v in rinfo["strata"].values())
    rec.to_csv(art / "raw_reconstruction_pre_run.csv", index=False)
    check(f"independent raw-JSON reconstruction of {rinfo['strata']['matches']} complete matches ({len(rec)} eligible rows; "
          f"audited K1-K3 code): observations, 5-position histories and targets identical", rec_ok, json.dumps(rinfo["strata"]))
    if not (rv_ok and rec_ok):
        LOGGER.error("independent raw-JSON verification failed")
        return 8
    math_pre = mathematical_checks(cfg, ids, rows.y)
    write_json(art / "mathematical_verification_pre_run.json", math_pre)
    check("pre-run: NumPy LSTM reference = nn.LSTM for all 8 architectures on 2,000 real windows (float32 < 1e-5, float64 < 1e-12); "
          "parameter counts = 4Hm + 4H^2 + 8H per layer + 15H + 15; finite differences < 1e-7", math_pre["pass"],
          json.dumps({"worst_float32": max(r["max_abs_logit_float32_vs_numpy"] for r in math_pre["candidates"]),
                      "worst_float64": max(r["max_abs_logit_float64_vs_numpy"] for r in math_pre["candidates"]),
                      "finite_difference": math_pre["finite_difference"]["max_abs_error"]}))
    if not math_pre["pass"]:
        LOGGER.error("mathematical verification failed before training")
        return 8
    LOGGER.info("input, raw (%d + %d rows) and mathematical verification passed", len(rv), len(rec))
    pre_maxrss = maxrss_mb()
    del tb, brows                    # the canonical table is not needed after verification: free memory for the workers
    inp["tb"] = None
    gc.collect()

    # ------------------------------------------------------------------ nested CV: workers, then assembly (checkpointed)
    wpath = art / "worker_input.npz"
    digest = write_worker_input(wpath, ids, rows.match_id, rows.y, folds, identifiers)
    todo = [f.fold_id for f in folds if not (ckpt / f"fold{f.fold_id}" / "fold_complete.pt").exists()]
    LOGGER.info("folds to compute in workers: %s (already complete: %s)", todo, sorted(set(range(5)) - set(todo)))
    t_workers = time.time()
    codes = run_workers(todo, art, wpath, digest, int(cfg["training"]["parallel_fold_workers"])) if todo else {}
    if any(c != 0 for c in codes.values()):
        LOGGER.error("worker failure: %s", codes)
        return 10
    worker_seconds = time.time() - t_workers
    trace: list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = lm.run_nested_cv(ids, rows.match_id, rows.y, folds, cfg, ckpt, identifiers, trace, LOGGER.info)
    worker_status = {f.fold_id: json.loads((art / f"worker_fold{f.fold_id}_status.json").read_text()) for f in folds
                     if (art / f"worker_fold{f.fold_id}_status.json").exists()}
    train_warnings = [f"{w.category.__name__}: {str(w.message)[:300]}" for w in caught]
    train_warnings += [w for s in worker_status.values() for w in s.get("warnings", [])]
    for w in train_warnings:
        LOGGER.warning("warning during training: %s", w)
    LOGGER.info("nested CV assembled: computed in this process %s, resumed folds %s, resumed fits %d", res.computed_folds,
                res.resumed_folds, res.resumed_fits)
    if staging.exists():
        if not (staging / ".staging_experiment6c").exists():
            LOGGER.error("%s exists but is not this runner's staging directory", staging)
            return 9
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / ".staging_experiment6c").write_text("incomplete run output; renamed on completion\n")
    o = staging

    # ------------------------------------------------------------------ integrity of the predictions
    check("no warning raised during training / prediction (runner and all workers)", not train_warnings, json.dumps(train_warnings[:5]))
    check("every outer fold computed by a worker process and committed atomically (assembly resumed all 5 folds)",
          sorted(res.resumed_folds) == [0, 1, 2, 3, 4] and not res.computed_folds, json.dumps(
              {k: {"pid": v.get("pid"), "seconds": round(v.get("seconds", 0)), "maxrss_mb": round(v.get("maxrss_mb", 0))}
               for k, v in worker_status.items()}))
    check("fold of every row identical to experiments 5 / 5B / 6A / 6B", np.array_equal(res.fold_of_row, fold_of_row))
    check("690,147 rows and 304 matches", len(rows.y) == 690_147 and len(set(rows.match_id)) == 304, len(rows.y))
    check("every outer fold completed", sorted(c["fold_id"] for c in res.choices) == [0, 1, 2, 3, 4])
    val_counts = np.zeros(len(rows.y), dtype=int)
    for f in folds:
        val_counts[rows.mask(f.validation_match_ids)] += 1
    check("every row predicted exactly once (one validation fold per row)", bool((val_counts == 1).all()))
    tol = float(cfg["evaluation"]["probability_tolerance"])
    for name, p in [("primary", res.proba)] + [(f"replicate {r}", q) for r, q in res.proba_replicates.items()]:
        try:
            xe.check_probabilities(p, tol)
            ok = bool((p > 0).all())
        except ValueError:
            ok = False
        check(f"{name} probabilities finite, positive, 15 columns, rows sum to 1 (tol 1e-9)", ok, float(np.abs(p.sum(1) - 1).max()))
    fits = pd.DataFrame(res.fits)
    n_expected = 5 * (4 * len(cfg["lstm"]["candidates"]) + 1 + int(cfg["training"]["robustness_seeds"]))
    check(f"fit log complete: {n_expected} fits = 5 x (4 x 8 + 1 + 2), all with finite stopping losses",
          len(fits) == n_expected and bool(np.isfinite(fits["best_stop_log_loss"]).all()), len(fits))
    hist_ok = True
    for h in res.histories:
        hh = json.loads(h["history"])
        stop = [e["stop_log_loss"] for e in hh]
        frow = fits[(fits.fold_id == h["fold_id"]) & (fits.stage == h["stage"]) & (fits.held_inner_fold == h["held_inner_fold"])
                    & (fits.candidate == h["candidate"]) & (fits.seed == h["seed"])].iloc[0]
        hist_ok &= int(np.argmin(stop)) == frow.best_epoch and stop[frow.best_epoch] == frow.best_stop_log_loss \
            and len(hh) - 1 == frow.epochs_trained == min(frow.best_epoch + int(cfg["training"]["patience"]), int(cfg["training"]["max_epochs"]))
    check("every fit: best epoch = first argmin of its stopping curve (epoch 0 included), best loss = curve minimum, "
          "epochs = min(best + patience, max_epochs)", bool(hist_ok), f"{int((fits.best_epoch == 0).sum())} fits with best epoch 0")
    check("fits stopped at max_epochs (early stopping did not trigger)", True,
          f"{int(fits['hit_max_epochs'].sum())} of {len(fits)}", kind="INFO")
    check("gradient norm: largest per-batch global L2 norm over all fits (no clipping applied)", True,
          f"max {fits['grad_norm_max'].max():.4g}; batches with norm > 1: {int(fits['n_grad_norm_above_1'].sum())} of "
          f"{int(fits['n_batches'].sum())}; > 5: {int(fits['n_grad_norm_above_5'].sum())}; > 10: {int(fits['n_grad_norm_above_10'].sum())}",
          kind="INFO")
    for fm in res.fold_metrics:
        sel = res.fold_of_row == fm["fold_id"]
        ll = float(-np.log(res.proba[sel][np.arange(sel.sum()), rows.y[sel]]).mean())
        check(f"fold {fm['fold_id']} log loss reproduces from assembled predictions", abs(ll - fm["log_loss"]) < 1e-12, abs(ll - fm["log_loss"]))
    leak = [(f.fold_id, t["purpose"]) for f in folds for t in trace if t["outer_fold"] == f.fold_id
            and t["purpose"] != "outer_validation_predict" and t["match_ids"] & set(f.validation_match_ids)]
    if leak:
        raise LeakageError(f"outer-validation match used for {leak[:3]}")
    check("no outer-validation match in any inner-train / stopping / inner-validation / outer-train set", not leak,
          f"{sum(t['purpose'] != 'outer_validation_predict' for t in trace)} data sets audited")
    stop_overlap = []
    for f in folds:
        tsets = [t for t in trace if t["outer_fold"] == f.fold_id]
        for a, b in (("inner_train", "inner_stop"), ("outer_train", "outer_stop")):
            for ta, tb_ in zip([t for t in tsets if t["purpose"] == a], [t for t in tsets if t["purpose"] == b]):
                stop_overlap.append(bool(ta["match_ids"] & tb_["match_ids"]))
    check("stopping subsets are disjoint (by match) from the rows the network is trained on (one pair per fit)",
          not any(stop_overlap) and len(stop_overlap) == n_expected, f"{len(stop_overlap)} pairs")
    da6b = pd.read_csv(PROJECT_ROOT / cfg["experiment6b_results_dir"] / "data_access_audit.csv")
    mh_ = lambda ms: hashlib.sha256(",".join(map(str, sorted(ms))).encode()).hexdigest()   # noqa: E731
    mine = sorted((t["outer_fold"], t["purpose"], mh_(t["match_ids"])) for t in trace)
    theirs = sorted((int(r.outer_fold), r.purpose, r.match_ids_sha256) for r in da6b.itertuples())
    check("every distinct data set (inner train / stop / validation, outer train / stop / validation) = the SAME matches as MLP-H5 "
          "(6B access audit)", set(mine) == set(theirs), f"{len(set(mine))} distinct data sets compared")
    check("each data set used by the same number of fits as in MLP-H5 (identical multiset of 6B data-access records)",
          mine == theirs, f"{len(mine)} vs {len(theirs)} records")
    check("no final-test match in the data, folds or any data set used",
          not (dataset_matches & final_test) and not any(t["match_ids"] & final_test for t in trace) and len(final_test) == 83,
          f"{len(final_test)} final-test ids checked")

    # implementation-level: float64 NumPy LSTM of the saved weights vs torch probabilities (all rows, all replicates)
    dmax = 0.0
    for f in folds:
        va = rows.mask(f.validation_match_ids)
        cells, _, _, inv = lm.aggregate_cells(ids[va])
        for r in range(1 + int(cfg["training"]["robustness_seeds"])):
            m = lm.model_from_state(res.selected[f.fold_id], res.states[(f.fold_id, r)])
            p_np = lm.numpy_lstm_forward(m.export(), lm.one_hot_sequence(cells), m.layers)[1][inv]
            q = res.proba[va] if r == 0 else res.proba_replicates[r][va]
            dmax = max(dmax, float(np.abs(p_np - q).max()))
    check("probabilities recomputed by the float64 NumPy LSTM reference from the saved weights equal torch's on all rows and "
          "replicates (|dp| < 1e-5)", dmax < 1e-5, dmax)
    # gate-level audit of each fold's primary model on 3,000 of its validation windows
    gate = []
    for f in folds:
        va = np.flatnonzero(rows.mask(f.validation_match_ids))[:: max(1, int(rows.mask(f.validation_match_ids).sum()) // 3000)][:3000]
        m64 = lm.model_from_state(res.selected[f.fold_id], res.states[(f.fold_id, 0)]).double().eval()
        idx = torch.from_numpy(ids[va])
        z_np, _, st_np = lm.numpy_lstm_forward(m64.export(), lm.one_hot_sequence(ids[va]), m64.layers, return_states=True)
        with torch.no_grad():
            seq, (h_n, c_n) = m64.lstm(m64.encode(idx))
            cs = [m64.lstm(m64.encode(idx[:, :s]))[1][1][-1].numpy() for s in range(1, 6)]
            z64 = m64(idx).numpy()
        cnp = st_np[-1]["c"]
        tc = np.tanh(cnp)
        okm = np.abs(tc) > 1e-3
        gate.append({"fold_id": f.fold_id, "candidate": res.selected[f.fold_id]["name"], "n_windows": len(va),
                     "max_abs_h_all_positions": float(np.abs(seq.numpy() - st_np[-1]["h"]).max()),
                     "max_abs_c_all_positions": float(max(np.abs(cs[s] - cnp[:, s]).max() for s in range(5))),
                     "max_abs_output_gate_recovered": float(np.abs(seq.numpy()[okm] / tc[okm] - st_np[-1]["o"][okm]).max()),
                     "max_abs_logits": float(np.abs(z64 - z_np).max()),
                     "mean_abs_logits": float(np.abs(z64 - z_np).mean()),
                     **{f"mean_gate_{g}": float(st_np[-1][g].mean()) for g in ("i", "f", "o")}})
    gate = pd.DataFrame(gate)
    gate.to_csv(o / "gate_level_verification.csv", index=False)
    gworst = float(gate[["max_abs_h_all_positions", "max_abs_c_all_positions", "max_abs_logits"]].to_numpy().max())
    check("gate-level audit (float64) of every fold's primary model: h_s, c_s (s = 1..5), recovered output gate and logits of "
          "nn.LSTM equal the NumPy reference (< 1e-12; o_s < 1e-9)", gworst < 1e-12 and float(gate.max_abs_output_gate_recovered.max()) < 1e-9,
          gworst)
    # reproducibility: retrain fold 0's primary refit in THIS process from scratch (no checkpoint) and compare
    f0, c0 = folds[0], res.selected[0]
    ci0 = next(c["candidate_index"] for c in res.choices if c["fold_id"] == 0)
    fit_m, stop_m = lm.stopping_split(list(f0.train_match_ids), float(cfg["training"]["stopping_fraction"]),
                                      lm.derive_seed(int(cfg["training"]["seed"]), 7, 0, 0))
    t0 = time.perf_counter()
    again = lm.train_fit("repro_f0", torch.from_numpy(ids), torch.from_numpy(rows.y), np.flatnonzero(rows.mask(fit_m)),
                         np.flatnonzero(rows.mask(stop_m)), c0, cfg["training"], lm.derive_seed(int(cfg["training"]["seed"]), 0, 0, ci0, 0))
    va0 = rows.mask(f0.validation_match_ids)
    pa = lm.predict_proba(lm.model_from_state(c0, again.best_state), ids[va0])
    same_w = all(torch.equal(again.best_state[k], res.states[(0, 0)][k]) for k in again.best_state)
    rep_diff = float(np.abs(pa - res.proba[va0]).max())
    repro_seconds = time.perf_counter() - t0
    check("fresh retraining of fold 0's primary refit in the main process reproduces the worker's weights and predictions "
          "bit-identically (CPU, 1 thread)", same_w and rep_diff == 0.0, f"max |dp| = {rep_diff:.3g}; {repro_seconds:.0f} s")

    # ------------------------------------------------------------------ evaluation (same rows)
    proba = {**{m: prior[m] for m in PRIOR_MODELS}, NAME: res.proba}
    tables = xe.evaluate_models(rows.y, rows.match_id, res.fold_of_row, proba, PAIRS, cfg)
    for name, frame in tables.items():
        frame.to_csv(o / f"{name}.csv", index=False)
    mine_t = tables["model_comparison"].set_index("model")
    ref = pd.read_csv(PROJECT_ROOT / cfg["experiment6b_results_dir"] / "model_comparison.csv").set_index("model")
    recomp = []
    for mname in PRIOR_MODELS:
        cs = [c for c in ref.columns if c in mine_t.columns and pd.api.types.is_numeric_dtype(ref[c])]
        d = (mine_t.loc[mname, cs].astype(float) - ref.loc[mname, cs].astype(float)).abs()
        recomp.append({"model": mname, "n_metrics_compared": len(cs), "max_abs_difference": float(d.max()),
                       "worst_metric": str(d.idxmax()), "log_loss_recomputed": float(mine_t.at[mname, "log_loss"]),
                       "log_loss_stored": float(ref.at[mname, "log_loss"])})
    recomp = pd.DataFrame(recomp)
    recomp.to_csv(o / "baseline_recomputation_check.csv", index=False)
    check("all prior models' metrics and CIs recomputed from saved predictions equal experiment 6B's tables (1e-12)",
          bool((recomp["max_abs_difference"] < 1e-12).all()), float(recomp["max_abs_difference"].max()))
    pcm = tables["per_class_metrics"].query("model == @NAME")
    check("Shot support = 7,212 rows", int(pcm.loc[pcm["class"] == "Shot", "support"].iat[0]) == 7212,
          int(pcm.loc[pcm["class"] == "Shot", "support"].iat[0]))
    cm = tables["confusion_matrices"]
    shot = []
    for mname in proba:
        c_ = cm[cm.model == mname]
        tp = int(c_[(c_.true_class == "Shot") & (c_.predicted_class == "Shot")]["count"].sum())
        pred_n = int(c_[c_.predicted_class == "Shot"]["count"].sum())
        shot.append({"model": mname, "shot_support": 7212, "shot_predicted": pred_n, "shot_true_positives": tp,
                     "shot_recall": tp / 7212, "shot_precision": tp / pred_n if pred_n else 0.0,
                     "shot_average_precision": float(mine_t.at[mname, "shot_average_precision"])})
    pd.DataFrame(shot).to_csv(o / "shot_summary.csv", index=False)

    # ------------------------------------------------------------------ robustness seeds (descriptive; prespecified)
    yy = rows.y
    nll = lambda p: -np.log(p[np.arange(len(yy)), yy])                 # noqa: E731
    ref_nll = {m: nll(prior[m]) for m in ("XGB-H5", "MLR-H5", "MLP-H5", "MLP-H5_seed1", "MLP-H5_seed2")}
    rob = []
    for r, p in {0: res.proba, **res.proba_replicates}.items():
        ln = nll(p)
        mlp_same = ref_nll["MLP-H5" if r == 0 else f"MLP-H5_seed{r}"]
        rob.append({"replicate": r, "primary": r == 0, "pooled_log_loss": float(ln.mean()), "accuracy": float((p.argmax(1) == yy).mean()),
                    "difference_vs_MLP-H5_primary": float(ln.mean() - ref_nll["MLP-H5"].mean()),
                    "difference_vs_MLP-H5_same_replicate": float(ln.mean() - mlp_same.mean()),
                    "difference_vs_XGB-H5": float(ln.mean() - ref_nll["XGB-H5"].mean()),
                    "difference_vs_MLR-H5": float(ln.mean() - ref_nll["MLR-H5"].mean()),
                    **{f"fold{f}_log_loss": float(ln[res.fold_of_row == f].mean()) for f in range(5)}})
    rob = pd.DataFrame(rob)
    rob.to_csv(o / "seed_robustness.csv", index=False)
    check("robustness seeds: pooled log-loss range across 3 refit seeds", True,
          f"{rob['pooled_log_loss'].min():.6f} .. {rob['pooled_log_loss'].max():.6f}", kind="INFO")

    # ------------------------------------------------------------------ descriptive analyses
    strata = []
    for refm in ("MLP-H5", "XGB-H5", "MLR-H5", "markov2"):
        s = lh.nll_difference_strata(rows.y, res.proba, proba[refm], rows.seq_pos, ids[:, 4])
        s.insert(0, "reference", refm)
        s.insert(0, "model", NAME)
        strata.append(s)
    pd.concat(strata, ignore_index=True).to_csv(o / "nll_difference_by_stratum.csv", index=False)
    cur = ids[:, 4]
    byc = []
    for ci_, cname in enumerate(CLASSES):
        s = cur == ci_
        byc.append({"current_class": cname, "n_rows": int(s.sum()), "share_of_rows": float(s.mean()),
                    **{f"log_loss_{m}": float(nll(proba[m])[s].mean()) for m in (NAME, "MLP-H5", "XGB-H5", "MLR-H5", "markov2")},
                    **{f"accuracy_{m}": float((proba[m][s].argmax(1) == yy[s]).mean()) for m in (NAME, "MLP-H5", "XGB-H5")}})
    pd.DataFrame(byc).to_csv(o / "log_loss_by_current_class.csv", index=False)
    xe.permutation_importance(_RowsView(rows, ids), _NestedView({f.fold_id: _Predictor(res.selected[f.fold_id], res.states[(f.fold_id, 0)])
                                                                 for f in folds}), folds, cfg).to_csv(o / "permutation_importance_lstm_h5.csv", index=False)
    pd.DataFrame(res.choices).to_csv(o / "hyperparameter_choices.csv", index=False)
    pd.DataFrame(res.inner_scores).to_csv(o / "inner_cv_scores.csv", index=False)
    fits.to_csv(o / "fit_log.csv", index=False)
    pd.DataFrame(res.histories).to_csv(o / "training_histories.csv", index=False)
    pd.DataFrame(res.timings).to_csv(o / "timings.csv", index=False)
    pc_rows = []
    for i, c in enumerate(cfg["lstm"]["candidates"]):
        f_ = lm.parameter_count_formula(c["hidden"], c["layers"])
        pc_rows.append({"index": i, **c, "parameters_lstm_layers": json.dumps(f_["lstm_layers"]), "parameters_output": f_["output"],
                        "parameters_total": f_["total"], "parameters_pytorch": lm.count_parameters(lm.build_model(c, 0))})
    pd.DataFrame(pc_rows).to_csv(o / "search_candidates.csv", index=False)
    mlp_c = yaml.safe_load((PROJECT_ROOT / cfg["experiment6b_config"]).read_text())["mlp"]["candidates"]
    from src.sequence_models import mlp_h5_experiment as mh
    pd.DataFrame([{"model": "MLP-H5", "candidate": c["name"], "hidden": json.dumps(c["hidden"]),
                   "parameters": int(sum(p.numel() for p in mh.build_model(c, 0).parameters()))} for c in mlp_c] +
                 [{"model": NAME, "candidate": r["name"], "hidden": f"{r['hidden']} x {r['layers']} layer(s)",
                   "parameters": r["parameters_total"]} for r in pc_rows]).to_csv(o / "parameter_counts.csv", index=False)
    pop = []
    for f in folds:
        fm_, sm_ = lm.stopping_split(list(f.train_match_ids), float(cfg["training"]["stopping_fraction"]),
                                     lm.derive_seed(int(cfg["training"]["seed"]), 7, f.fold_id, 0))
        r0 = fits[(fits.fold_id == f.fold_id) & (fits.stage == "outer_refit")].iloc[0]
        trr = int(rows.mask(f.train_match_ids).sum())
        pop.append({"fold_id": f.fold_id, "outer_train_matches": len(f.train_match_ids), "outer_train_rows": trr,
                    "fit_matches": len(fm_), "fit_rows": int(r0.n_train_rows), "stop_matches": len(sm_), "stop_rows": int(r0.n_stop_rows),
                    "fit_share_of_rows": r0.n_train_rows / trr, "fit_stop_overlap_matches": len(set(fm_) & set(sm_)),
                    "fit_or_stop_in_outer_validation": len((set(fm_) | set(sm_)) & set(f.validation_match_ids)),
                    "fit_plus_stop_equals_outer_train": set(fm_) | set(sm_) == set(f.train_match_ids),
                    "selected": r0.candidate, "best_epoch": int(r0.best_epoch), "epochs_trained": int(r0.epochs_trained),
                    "best_stop_log_loss": float(r0.best_stop_log_loss), "xgb_h5_train_rows": trr})
    pop = pd.DataFrame(pop)
    pop.to_csv(o / "training_population.csv", index=False)
    check("refit stopping subsets: disjoint from fit matches, inside outer training, union = outer training",
          bool((pop.fit_stop_overlap_matches == 0).all() and (pop.fit_or_stop_in_outer_validation == 0).all() and pop.fit_plus_stop_equals_outer_train.all()))
    rv.to_csv(o / "raw_feature_verification.csv", index=False)
    rec.to_csv(o / "raw_reconstruction.csv", index=False)
    write_json(o / "mathematical_verification_pre_run.json", math_pre)
    write_json(o / "feature_verification.json", fver)
    counts = []
    for f in folds:
        trm, vam = rows.mask(f.train_match_ids), rows.mask(f.validation_match_ids)
        counts.append({"fold_id": f.fold_id, "train_matches": len(f.train_match_ids), "validation_matches": len(f.validation_match_ids),
                       "train_rows": int(trm.sum()), "validation_rows": int(vam.sum()),
                       **{f"train_n_{c}": int((rows.y[trm] == i).sum()) for i, c in enumerate(CLASSES)},
                       **{f"validation_n_{c}": int((rows.y[vam] == i).sum()) for i, c in enumerate(CLASSES)}})
    pd.DataFrame(counts).to_csv(o / "fold_counts.csv", index=False)
    pd.DataFrame([{**t, "n_matches": len(t["match_ids"]),
                   "touches_outer_validation": bool(t["match_ids"] & set(next(f for f in folds if f.fold_id == t["outer_fold"]).validation_match_ids)),
                   "touches_final_test": bool(t["match_ids"] & final_test), "match_ids_sha256": mh_(t["match_ids"])}
                  for t in trace]).drop(columns="match_ids").to_csv(o / "data_access_audit.csv", index=False)

    # ------------------------------------------------------------------ artefacts and provenance
    pred = pd.DataFrame({"event_id": rows.event_id, "match_id": rows.match_id, "fold_id": res.fold_of_row, "y": rows.y})
    for c in range(K):
        pred[f"{NAME}__p{c}"] = res.proba[:, c]
    for r, p in res.proba_replicates.items():
        for c in range(K):
            pred[f"{NAME}_seed{r}__p{c}"] = p[:, c]
    pred_path = art / "oof_predictions_lstm_h5.parquet"
    tmp = pred_path.with_name("." + pred_path.name + ".tmp")
    pred.to_parquet(tmp, index=False)
    tmp.replace(pred_path)
    arrays = {}
    for (fid, r), st in res.states.items():
        for k, v in st.items():
            arrays[f"fold{fid}_rep{r}__{k}"] = v.numpy().astype(np.float32)
    models_npz = art / "fold_models_lstm_h5.npz"
    tmpz = models_npz.with_name("." + models_npz.name + ".tmp.npz")
    np.savez(tmpz, **arrays, selected=json.dumps({int(k): v for k, v in res.selected.items()}))
    os.replace(tmpz, models_npz)
    models_pt = art / "fold_models_lstm_h5.pt"
    lm.save_atomic(models_pt, {"identifiers": identifiers, "selected": res.selected, "states": res.states})
    shutil.copy(CONFIG, o / "experiment_config.yaml")
    shutil.copy(PROJECT_ROOT / cfg["task_config"], o / "task_config.yaml")
    _, digest_raw = raw_source_hashes(part.development)
    check("raw source digest equals the canonical manifest", digest_raw == inp["manifest"]["raw_source_digest_development"], digest_raw)
    hist_after = historical_snapshot(cfg)
    check("historical results unchanged (baselines, experiments 5, 5B, 6A, 6B, 6A/6B audit: checksums, README, predictions, models, dataset)",
          hist_after == hist_before and not any(e["mismatches"] for e in hist_after["directories"].values()),
          sum(e.get("checksums.json_files", 0) for e in hist_after["directories"].values()))
    check("run status", True, f"resumed folds: {res.resumed_folds}; computed in workers: {sorted(codes)}; "
                              f"fits loaded from checkpoints: {res.resumed_fits}", kind="INFO")
    fe = Path(inp["spec"].raw["partition"]["folds_file"])
    write_json(o / "split_manifest.json", {
        "source_split_manifest": "results/markov_hmm_pilot/split_manifest.json",
        "source_split_manifest_sha256": identifiers["split_manifest_sha256"],
        "folds_file": fe.as_posix(), "folds_file_sha256": identifiers["folds_file_sha256"],
        "folds_identical_to_canonical_baselines_and_experiments_5_5b_6a_6b": True,
        "final_test_matches": len(part.final_test), "final_test_loaded": 0,
        "folds": [{"fold_id": f.fold_id, "train_match_ids": list(f.train_match_ids), "validation_match_ids": list(f.validation_match_ids),
                   "validation_rows": int(rows.mask(f.validation_match_ids).sum())} for f in folds]})
    write_json(o / "stopping_subset_manifest.json", {
        "derivation": "mlp_h5_experiment.stopping_split(train matches, 0.10, SeedSequence([20261123, 7, outer, inner + 1 | 0 for refits]))",
        "identical_to_experiment_6b": mine == theirs,
        "sets": [{"outer_fold": t["outer_fold"], "purpose": t["purpose"], "n_matches": len(t["match_ids"]),
                  "match_ids_sha256": mh_(t["match_ids"])} for t in trace if t["purpose"].endswith("stop")],
        "refits": pop.to_dict("records")})
    write_json(o / "task_manifest.json", {**inp["manifest"], "raw_source_digest_recomputed": digest_raw,
                                          "row_identity": {"n_rows": int(len(rows.y)), "n_matches": int(len(set(rows.match_id))),
                                                           "event_id_sha256": identifiers["row_event_id_sha256"],
                                                           "target_sha256": identifiers["target_sha256"],
                                                           "fold_of_row_sha256": sha256_array(res.fold_of_row),
                                                           "feature_matrix_sha256": identifiers["feature_matrix_sha256"]}})
    env = environment_info()
    env.pop("platform", None)
    env["packages"]["torch"] = torch.__version__
    write_json(o / "environment.json", {
        **env, "machine": {"processor": platform.machine(), "torch_threads_per_process": torch.get_num_threads(),
                           "parallel_fold_workers": int(cfg["training"]["parallel_fold_workers"]),
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "device": "cpu",
                           "mps_available_not_used": bool(torch.backends.mps.is_available())},
        "runtime_seconds": time.time() - started, "worker_phase_seconds": worker_seconds, "reproduction_check_seconds": repro_seconds,
        "peak_rss_mb": {"runner_before_workers": pre_maxrss, "runner_total": maxrss_mb(),
                        **{f"worker_fold{k}": v.get("maxrss_mb") for k, v in worker_status.items()}},
        "worker_status": worker_status, "checkpoint_identifiers": identifiers,
        "source_files_sha256": source_digest(ALL_SOURCES)["files"],
        "artefacts_sha256": {(p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p).as_posix(): sha256_file(p)
                             for p in (pred_path, models_npz, models_pt)},
        "run_status": {"resumed_folds": res.resumed_folds, "computed_in_workers": sorted(codes), "resumed_fits": res.resumed_fits},
        "workload": {"n_fits": len(fits), "total_fit_seconds": float(fits["seconds"].sum()),
                     "total_epochs": int(fits["epochs_trained"].sum()), "total_batches": int(fits["n_batches"].sum())},
        "historical_verification": {"before": hist_before, "after": hist_after},
        "random_seeds": {"folds": 20261027, "bootstrap": cfg["evaluation"]["bootstrap_seed"], "training_base": cfg["training"]["seed"],
                         "derivation": "numpy SeedSequence([base, outer, inner+1 (0 = refit), candidate, replicate]); per epoch "
                                       "SeedSequence([fit, 1|2|3|4, epoch]) for permutation / dropout / numpy / python (= 6B)",
                         "permutation_importance": cfg["evaluation"]["permutation_seed"],
                         "raw_reconstruction": cfg["evaluation"]["raw_reconstruction"]["seed"]},
        "commands": ["python scripts/supervise_lstm_h5_experiment6c.py --detach",
                     "python scripts/run_lstm_h5_experiment6c.py", "python scripts/verify_lstm_h5_experiment6c.py"]})
    pd.DataFrame(integrity).to_csv(o / "integrity_audit.csv", index=False)
    (staging / ".staging_experiment6c").unlink()
    write_json(o / "checksums.json", {p.relative_to(o).as_posix(): sha256_file(p) for p in sorted(o.rglob("*"))
                                      if p.is_file() and p.name not in ("checksums.json", "README.md")})
    n_fail = sum(r["result"] == "FAIL" for r in integrity)
    LOGGER.info("integrity audit: %d checks, %d FAIL", len(integrity), n_fail)
    LOGGER.info("pooled log loss: %s", {k: round(float(v), 6) for k, v in mine_t["log_loss"].items()})
    if n_fail:
        LOGGER.error("integrity audit FAILED - results left in %s (not marked complete)", staging.name)
        return 1
    staging.rename(out)          # atomic: the result directory appears only when complete
    wpath.unlink(missing_ok=True)
    LOGGER.info("done in %.0f s -> %s", time.time() - started, cfg["output_dir"])
    return 0


class _Predictor:
    def __init__(self, cand, state):
        self.model = lm.model_from_state(cand, state)

    def predict_proba(self, ids):
        return lm.predict_proba(self.model, ids)


class _RowsView:
    def __init__(self, rows, ids: np.ndarray):
        self.rows, self.ids, self.y, self.match_id = rows, ids, rows.y, rows.match_id

    def features(self, _name):
        return self.ids

    def mask(self, match_ids):
        return self.rows.mask(match_ids)


class _NestedView:
    feature_set = NAME

    def __init__(self, models):
        self.models = models


if __name__ == "__main__":
    sys.exit(main())
