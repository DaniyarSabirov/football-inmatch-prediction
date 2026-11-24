"""Experiment 6B: multilayer perceptron with a five-event history (MLP-H5) on the canonical
task actions15_360current_v1.

Same rows (690,147), outer folds and history features as experiments 5 / 5B / 6A. Final-test
matches are never loaded. The canonical dataset, the stored predictions of experiments
5 / 5B / 6A and all historical result directories are read only and hash-verified before and
after the run. Every fit is checkpointed at epoch boundaries and every completed fit / outer
fold is committed atomically; a restarted run resumes only from compatible checkpoints.

Results are written to ``<output_dir>.partial`` and renamed to ``<output_dir>`` only after the
integrity audit and the checksums are complete, so an interrupted run is never mistaken for a
complete one. A lock file prevents two concurrent runs.

Usage (normally started by scripts/supervise_mlp_h5_experiment6b.py):
    .venv/bin/python scripts/run_mlp_h5_experiment6b.py --smoke   # fold-0 inner data only: timing / memory
    .venv/bin/python scripts/run_mlp_h5_experiment6b.py           # full run
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import platform
import resource
import shutil
import sys
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
from build_canonical_task import raw_source_hashes  # noqa: E402
from run_logistic_h5_experiment6a import raw_verification  # noqa: E402  (6A's independent raw-JSON check, unchanged)
from run_xgboost_experiment5 import load_inputs  # noqa: E402  (experiment 5 loader, unchanged)

from src.sequence_models import canonical_baselines as cb  # noqa: E402
from src.sequence_models import logistic_h5_experiment as lh  # noqa: E402
from src.sequence_models import mlp_h5_experiment as mh  # noqa: E402
from src.sequence_models import xgboost_experiment as xe  # noqa: E402
from src.sequence_models.canonical_task import CLASSES, K, TASK_VERSION  # noqa: E402
from src.sequence_models.event_history import PAD, HistoryRows, history_ids, position_label  # noqa: E402
from src.sequence_models.experiment import environment_info, sha256_file, write_json  # noqa: E402
from src.sequence_models.splits import LeakageError  # noqa: E402
from src.sequence_models.xgboost_h2_experiment import sha256_array, source_digest  # noqa: E402

LOGGER = logging.getLogger("mlp_h5_experiment6b")
CONFIG = PROJECT_ROOT / "configs" / "mlp_h5_experiment6b_v1.yaml"
NAME = mh.FEATURE_SET
EXP5_MODELS = ["frequency", "markov1", "markov2", "logistic_F0", "XGB-F0", "XGB-H5"]
PRIOR_MODELS = EXP5_MODELS + ["XGB-H2", "MLR-H5"]
PAIRS = [(NAME, "XGB-H5"), (NAME, "MLR-H5"), (NAME, "markov2"), (NAME, "XGB-H2"), (NAME, "logistic_F0"),
         (NAME, "markov1"), (NAME, "XGB-F0")]
HIST_DIRS = ("baseline_results_dir", "experiment5_results_dir", "experiment5b_results_dir", "experiment6a_results_dir")
HIST_FILES = ("experiment5_predictions", "experiment5b_predictions", "experiment6a_predictions")
MODEL_SOURCES = [PROJECT_ROOT / p for p in ("src/sequence_models/mlp_h5_experiment.py", "src/sequence_models/event_history.py",
                                            "src/sequence_models/canonical_task.py")]
ALL_SOURCES = MODEL_SOURCES + [PROJECT_ROOT / p for p in (
    "src/sequence_models/xgboost_experiment.py", "src/sequence_models/evaluation_v2.py",
    "src/sequence_models/logistic_h5_experiment.py", "src/sequence_models/raw_history_check.py",
    "scripts/run_mlp_h5_experiment6b.py", "scripts/run_logistic_h5_experiment6a.py", "scripts/run_xgboost_experiment5.py",
    "scripts/supervise_mlp_h5_experiment6b.py")]


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
                "data/interim/logistic_h5_experiment6a/fold_models_mlr_h5.npz"):
        out["files"][rel] = sha256_file(PROJECT_ROOT / rel)
    return out


def load_prior_predictions(cfg: dict, rows: HistoryRows, fold_of_row: np.ndarray) -> tuple[dict, list[dict]]:
    """Stored out-of-fold predictions of experiments 5, 5B and 6A, aligned row by row."""
    checks = []
    ri = json.loads((PROJECT_ROOT / cfg["experiment5_results_dir"] / "task_manifest.json").read_text())["row_identity"]
    ri6 = json.loads((PROJECT_ROOT / cfg["experiment6a_results_dir"] / "task_manifest.json").read_text())["row_identity"]
    for label, r in (("experiment 5", ri), ("experiment 6A", ri6)):
        checks.append((f"{label} manifest: event-id hash equals this run's rows (n = {r['n_rows']})",
                       r["event_id_sha256"] == sha256_array(rows.event_id) and r["n_rows"] == len(rows.y), r["n_rows"]))
        checks.append((f"{label} manifest: target hash equals this run's targets", r["target_sha256"] == sha256_array(rows.y), None))
        checks.append((f"{label} manifest: fold-of-row hash equals this run's folds", r["fold_of_row_sha256"] == sha256_array(fold_of_row), None))
    checks.append(("experiment 6A manifest: history (feature-matrix) hash equals this run's history",
                   ri6["feature_matrix_sha256"] == sha256_array(rows.ctx), None))
    proba = {}
    for label, key, models in (("experiment 5", "experiment5_predictions", EXP5_MODELS),
                               ("experiment 5B", "experiment5b_predictions", ["XGB-H2"]),
                               ("experiment 6A", "experiment6a_predictions", ["MLR-H5"])):
        o = pd.read_parquet(PROJECT_ROOT / cfg[key])
        for col, ref in (("event_id", rows.event_id), ("match_id", rows.match_id), ("y", rows.y), ("fold_id", fold_of_row)):
            checks.append((f"{label} predictions: {col} identical row by row", np.array_equal(o[col].to_numpy(), ref), None))
        checks.append((f"{label} predictions: unique row ids", bool(o["event_id"].is_unique), len(o)))
        for m in models:
            proba[m] = o[[f"{m}__p{c}" for c in range(K)]].to_numpy()
            xe.check_probabilities(proba[m], float(cfg["evaluation"]["probability_tolerance"]))
    return proba, [{"check": c, "result": "PASS" if ok else "FAIL", "value": v} for c, ok, v in checks]


# ============================================================================ smoke test
def smoke(cfg: dict, rows: HistoryRows, folds, ids: np.ndarray, art: Path) -> None:
    """Fold-0 INNER data only (outer-validation rows of fold 0 never touched): one inner split,
    the cheapest and the most expensive candidate, 3 epochs each; timing and memory."""
    tcfg = {**cfg["training"], "max_epochs": 3}
    mh.configure_torch(tcfg)
    f = folds[0]
    inner = [g for g in folds if g.fold_id != f.fold_id]
    held = inner[0]
    itr = [m for g in inner if g is not held for m in g.validation_match_ids]
    fit_m, stop_m = mh.stopping_split(itr, float(tcfg["stopping_fraction"]), 1)
    if (set(itr) | set(held.validation_match_ids)) & set(f.validation_match_ids):
        raise LeakageError("smoke test touched outer-validation matches")
    X = torch.from_numpy(mh.column_indices(ids))
    y = torch.from_numpy(rows.y)
    tr, st = np.flatnonzero(rows.mask(fit_m)), np.flatnonzero(rows.mask(stop_m))
    report = []
    for cand in (cfg["mlp"]["candidates"][0], cfg["mlp"]["candidates"][5]):
        r = mh.train_fit(f"smoke_{cand['name']}", X, y, tr, st, cand, tcfg, 1)
        hv = np.flatnonzero(rows.mask(held.validation_match_ids))
        p = mh.predict_proba(mh.model_from_state(cand, r.best_state), ids[hv])
        xe.check_probabilities(p, float(cfg["evaluation"]["probability_tolerance"]))
        ep = [h["seconds"] for h in r.history[1:]]
        report.append({"candidate": cand, "train_rows": len(tr), "stop_rows": len(st), "seconds_per_epoch": float(np.mean(ep)),
                       "stop_log_loss_by_epoch": [h["stop_log_loss"] for h in r.history],
                       "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, "proba_shape": list(p.shape)})
        LOGGER.info("smoke %s: %.2f s/epoch, stop LL %s, maxrss %.0f MB", cand["name"], report[-1]["seconds_per_epoch"],
                    [round(v, 4) for v in report[-1]["stop_log_loss_by_epoch"]], report[-1]["max_rss_mb"])
    per_epoch = max(r["seconds_per_epoch"] for r in report)
    n_fits = 5 * (4 * len(cfg["mlp"]["candidates"]) + 1 + int(cfg["training"]["robustness_seeds"]))
    for ep in (20, 35, int(cfg["training"]["max_epochs"])):
        LOGGER.info("estimate: %d fits x %d epochs x %.2f s = %.1f h (upper bound per-epoch time)", n_fits, ep, per_epoch,
                    n_fits * ep * per_epoch / 3600)
    write_json(art / "smoke_test.json", {"note": "technical smoke test; inner data of outer fold 0 only; not a result",
                                         "fits": report, "n_fits_full_run": n_fits})


# ============================================================================ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.captureWarnings(True)
    started = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg5 = yaml.safe_load((PROJECT_ROOT / cfg["experiment5_config"]).read_text(encoding="utf-8"))
    for key in ("bootstrap_replicates", "bootstrap_seed", "probability_tolerance", "permutation_repeats", "permutation_seed"):
        if cfg["evaluation"][key] != cfg5["evaluation"][key]:
            raise SystemExit(f"evaluation.{key} differs from experiment 5")
    if cfg["feature_set"] != {"name": NAME, "n_previous": 4, "n_columns": mh.N_INPUT}:
        raise SystemExit("feature_set in the configuration differs from the implementation")
    art = PROJECT_ROOT / cfg["artefact_dir"]
    art.mkdir(parents=True, exist_ok=True)
    lock = open(art / ("runner_smoke.lock" if args.smoke else "runner.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOGGER.error("another experiment-6B runner holds the lock; refusing to start a duplicate")
        return 3
    out = PROJECT_ROOT / cfg["output_dir"]
    staging = out.with_name(out.name + ".partial")
    ckpt = PROJECT_ROOT / cfg["checkpoint_dir"]
    if not args.smoke and out.exists():
        LOGGER.error("%s exists: the experiment is complete; refusing to overwrite results", cfg["output_dir"])
        return 4

    # ------------------------------------------------------------------ inputs (verified)
    hist_before = historical_snapshot(cfg)
    bad = {d: e["mismatches"] for d, e in hist_before["directories"].items() if e["mismatches"]}
    if bad:
        LOGGER.error("historical results changed before the run: %s", bad)
        return 5
    inp = load_inputs(cfg)        # canonical hashes vs manifests, Tier B set, folds == baseline folds, no final test
    tb, folds, part = inp["tb"], inp["folds"], inp["part"]
    rows = HistoryRows.from_table(tb)
    ids = rows.ctx
    if not np.array_equal(ids, rows.features("XGB-H5")) or not np.array_equal(ids, history_ids(tb, 4)[tb["row_eligible"].to_numpy()]):
        raise RuntimeError("MLP-H5 history differs from the XGB-H5 history / second construction path")
    brows = cb.Rows.from_table(tb)
    if not (np.array_equal(ids[:, 4], brows.cur) and np.array_equal(rows.y, brows.y) and np.array_equal(rows.match_id, brows.match_id)):
        raise RuntimeError("(E_t, y, match) differ from the canonical baselines' rows")
    if len(rows.y) != inp["manifest"]["counts_tier_b"]["eligible_rows"] or len(rows.y) != 690_147 or len(set(rows.match_id)) != 304:
        LOGGER.error("row count %d / matches %d differ from the protocol", len(rows.y), len(set(rows.match_id)))
        return 6
    LOGGER.info("%s: %d rows, %d matches, history ids %s; torch %s, device cpu, %d thread(s)", TASK_VERSION, len(rows.y),
                len(set(rows.match_id)), ids.shape, torch.__version__, cfg["training"]["num_threads"])
    if args.smoke:
        smoke(cfg, rows, folds, ids, art)
        return 0

    integrity: list[dict] = []

    def check(name, ok, value=None, kind="PASS/FAIL"):
        integrity.append({"check": name, "result": ("PASS" if ok else "FAIL") if kind == "PASS/FAIL" else kind, "value": value})

    fold_of_row = np.full(len(rows.y), -1)
    for f in folds:
        fold_of_row[rows.mask(f.validation_match_ids)] = f.fold_id
    prior, prior_checks = load_prior_predictions(cfg, rows, fold_of_row)
    integrity += prior_checks
    if any(c["result"] == "FAIL" for c in prior_checks):
        pd.DataFrame(integrity).to_csv(art / "alignment_failure.csv", index=False)
        LOGGER.error("row-level alignment with experiments 5 / 5B / 6A failed; not training")
        return 7

    # ------------------------------------------------------------------ feature verification (before training)
    cols = mh.column_indices(ids)
    xs = lh.encode_sparse(ids)
    fver = {"n_columns": mh.N_INPUT, "mlp_columns_equal_6a_sparse_one_hot_on_all_rows":
                bool(np.array_equal(np.sort(cols, axis=1), xs.indices.reshape(-1, 5))),
            "e_t_never_start": bool((ids[:, 4] != PAD).all()),
            "start_count_by_lag": {position_label(lag): int((ids[:, 4 - lag] == PAD).sum()) for lag in range(1, 5)},
            "rows_with_seq_pos_below_lag": {position_label(lag): int((rows.seq_pos < lag).sum()) for lag in range(1, 5)}}
    fver["start_padding_iff_before_sequence_start"] = fver["start_count_by_lag"] == fver["rows_with_seq_pos_below_lag"]
    del xs
    check("MLP input columns equal experiment 6A's 79-column one-hot on all 690,147 rows",
          fver["mlp_columns_equal_6a_sparse_one_hot_on_all_rows"])
    check("<START> at lag l exactly when seq_pos < l; E_t never <START>",
          fver["start_padding_iff_before_sequence_start"] and fver["e_t_never_start"])
    check("history identical to XGB-H5 / two construction paths agree; (E_t, y) = canonical baseline rows", True)
    rv_cfg = yaml.safe_load((PROJECT_ROOT / cfg["experiment6a_config"]).read_text())["evaluation"]["raw_verification"]
    rv = raw_verification(rows, tb, rv_cfg)
    rv_cols = ["features_match", "one_hot_names_match", "start_padding_correct", "target_match", "history_strictly_before_target"]
    rv_ok = bool(rv[rv_cols].all().all()) and len(rv) >= 500
    rv.to_csv(art / "raw_feature_verification_pre_run.csv", index=False)
    strata_counts = rv["strata"].str.split(";").explode().value_counts().to_dict()
    check(f"raw-JSON rebuild of {len(rv)} rows (history, one-hot names, <START>, target, causality)", rv_ok, json.dumps(strata_counts))
    if not rv_ok:
        LOGGER.error("independent raw-JSON verification failed")
        return 8
    LOGGER.info("feature verification passed (%d raw rows)", len(rv))

    # ------------------------------------------------------------------ nested CV (checkpointed)
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
    trace: list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = mh.run_nested_cv(ids, rows.match_id, rows.y, folds, cfg, ckpt, identifiers, trace, LOGGER.info)
    train_warnings = [f"{w.category.__name__}: {str(w.message)[:300]}" for w in caught]
    for w in train_warnings:
        LOGGER.warning("warning during training: %s", w)
    LOGGER.info("nested CV complete: computed folds %s, resumed folds %s, resumed fits %d", res.computed_folds,
                res.resumed_folds, res.resumed_fits)
    if staging.exists():
        if not (staging / ".staging_experiment6b").exists():
            LOGGER.error("%s exists but is not this runner's staging directory", staging)
            return 9
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / ".staging_experiment6b").write_text("incomplete run output; renamed on completion\n")
    o = staging

    # ------------------------------------------------------------------ integrity of the predictions
    check("no warning raised during training / prediction", not train_warnings, json.dumps(train_warnings[:5]))
    check("fold of every row identical to experiments 5 / 5B / 6A", np.array_equal(res.fold_of_row, fold_of_row))
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
    check("all fits finished with finite losses (NumericalError otherwise)", True, f"{len(fits)} fits")
    check("fits stopped at max_epochs (early stopping did not trigger)", True,
          f"{int(fits['hit_max_epochs'].sum())} of {len(fits)}", kind="INFO")
    for fm in res.fold_metrics:
        sel = res.fold_of_row == fm["fold_id"]
        ll = float(-np.log(res.proba[sel][np.arange(sel.sum()), rows.y[sel]]).mean())
        check(f"fold {fm['fold_id']} log loss reproduces from assembled predictions", abs(ll - fm["log_loss"]) < 1e-12, abs(ll - fm["log_loss"]))
    for f in folds:
        for t in trace:
            if t["outer_fold"] == f.fold_id and t["purpose"] != "outer_validation_predict" and t["match_ids"] & set(f.validation_match_ids):
                raise LeakageError(f"fold {f.fold_id}: outer-validation match used for {t['purpose']}")
    check("no outer-validation match in any inner-train / stopping / inner-validation / outer-train set", True,
          f"{sum(t['purpose'] != 'outer_validation_predict' for t in trace)} data sets audited")
    stop_overlap = []
    for f in folds:
        tsets = [t for t in trace if t["outer_fold"] == f.fold_id]
        for a, b in (("inner_train", "inner_stop"), ("outer_train", "outer_stop")):
            for ta, tb_ in zip([t for t in tsets if t["purpose"] == a], [t for t in tsets if t["purpose"] == b]):
                stop_overlap.append(bool(ta["match_ids"] & tb_["match_ids"]))
    check("stopping subsets are disjoint (by match) from the rows the network is trained on", not any(stop_overlap),
          f"{len(stop_overlap)} pairs")
    final_test = set(part.final_test)
    check("no final-test match in the data, folds or any data set used", not (set(inp["table"]["match_id"]) & final_test)
          and not any(t["match_ids"] & final_test for t in trace) and len(final_test) == 83, f"{len(final_test)} final-test ids checked")
    # implementation-level: float64 NumPy forward pass of the saved weights vs torch probabilities
    dmax = 0.0
    for f in folds:
        va = rows.mask(f.validation_match_ids)
        m = mh.model_from_state(res.selected[f.fold_id], res.states[(f.fold_id, 0)])
        ws, bs = m.weight_matrices()
        cells, _, _, inv = mh.aggregate_cells(ids[va])
        onehot = np.zeros((len(cells), mh.N_INPUT))
        onehot[np.repeat(np.arange(len(cells)), 5), mh.column_indices(cells).ravel()] = 1.0
        dmax = max(dmax, float(np.abs(mh.numpy_forward(ws, bs, onehot)[inv] - res.proba[va]).max()))
    check("probabilities recomputed by a float64 NumPy forward pass of the saved weights equal torch's (|dp| < 1e-5)",
          dmax < 1e-5, dmax)
    # reproducibility: retrain fold 0's primary refit from scratch (no checkpoint) and compare
    f0 = folds[0]
    c0 = res.selected[0]
    ci0 = next(c["candidate_index"] for c in res.choices if c["fold_id"] == 0)
    fit_m, stop_m = mh.stopping_split(list(f0.train_match_ids), float(cfg["training"]["stopping_fraction"]),
                                      mh.derive_seed(int(cfg["training"]["seed"]), 7, 0, 0))
    t0 = time.perf_counter()
    again = mh.train_fit("repro_f0", torch.from_numpy(cols), torch.from_numpy(rows.y), np.flatnonzero(rows.mask(fit_m)),
                         np.flatnonzero(rows.mask(stop_m)), c0, cfg["training"],
                         mh.derive_seed(int(cfg["training"]["seed"]), 0, 0, ci0, 0))
    va0 = rows.mask(f0.validation_match_ids)
    pa = mh.predict_proba(mh.model_from_state(c0, again.best_state), ids[va0])
    same_w = all(torch.equal(again.best_state[k], res.states[(0, 0)][k]) for k in again.best_state)
    rep_diff = float(np.abs(pa - res.proba[va0]).max())
    check("fresh retraining of fold 0's refit reproduces weights and predictions bit-identically (CPU, 1 thread)",
          same_w and rep_diff == 0.0, f"max |dp| = {rep_diff:.3g}; {time.perf_counter() - t0:.0f} s")

    # ------------------------------------------------------------------ evaluation (9 models, same rows)
    proba = {**prior, NAME: res.proba}
    tables = xe.evaluate_models(rows.y, rows.match_id, res.fold_of_row, proba, PAIRS, cfg)
    for name, frame in tables.items():
        frame.to_csv(o / f"{name}.csv", index=False)
    mine = tables["model_comparison"].set_index("model")
    ref = pd.read_csv(PROJECT_ROOT / cfg["experiment6a_results_dir"] / "model_comparison.csv").set_index("model")
    recomp = []
    for mname in PRIOR_MODELS:
        cs = [c for c in ref.columns if c in mine.columns and pd.api.types.is_numeric_dtype(ref[c])]
        d = (mine.loc[mname, cs].astype(float) - ref.loc[mname, cs].astype(float)).abs()
        recomp.append({"model": mname, "n_metrics_compared": len(cs), "max_abs_difference": float(d.max()),
                       "worst_metric": str(d.idxmax()), "log_loss_recomputed": float(mine.at[mname, "log_loss"]),
                       "log_loss_stored": float(ref.at[mname, "log_loss"])})
    recomp = pd.DataFrame(recomp)
    recomp.to_csv(o / "baseline_recomputation_check.csv", index=False)
    check("all prior models' metrics and CIs recomputed from saved predictions equal experiment 6A's tables (1e-12)",
          bool((recomp["max_abs_difference"] < 1e-12).all()), float(recomp["max_abs_difference"].max()))

    # ------------------------------------------------------------------ robustness seeds (descriptive; prespecified)
    rob = []
    q = {0: res.proba, **res.proba_replicates}
    for r, p in q.items():
        nll = -np.log(p[np.arange(len(rows.y)), rows.y])
        rob.append({"replicate": r, "primary": r == 0, "pooled_log_loss": float(nll.mean()),
                    "accuracy": float((p.argmax(1) == rows.y).mean()),
                    "difference_vs_XGB-H5": float(nll.mean() - (-np.log(prior["XGB-H5"][np.arange(len(rows.y)), rows.y])).mean()),
                    "difference_vs_MLR-H5": float(nll.mean() - (-np.log(prior["MLR-H5"][np.arange(len(rows.y)), rows.y])).mean()),
                    **{f"fold{f}_log_loss": float(nll[res.fold_of_row == f].mean()) for f in range(5)}})
    rob = pd.DataFrame(rob)
    rob.to_csv(o / "seed_robustness.csv", index=False)
    check("robustness seeds: pooled log-loss range across 3 refit seeds", True,
          f"{rob['pooled_log_loss'].min():.6f} .. {rob['pooled_log_loss'].max():.6f}", kind="INFO")

    # ------------------------------------------------------------------ descriptive analyses
    strata = []
    for refm in ("XGB-H5", "MLR-H5", "markov2"):
        s = lh.nll_difference_strata(rows.y, res.proba, proba[refm], rows.seq_pos, ids[:, 4])
        s.insert(0, "reference", refm)
        s.insert(0, "model", NAME)
        strata.append(s)
    pd.concat(strata, ignore_index=True).to_csv(o / "nll_difference_by_stratum.csv", index=False)
    xe.permutation_importance(_RowsView(rows, ids), _NestedView({f.fold_id: _Predictor(res.selected[f.fold_id], res.states[(f.fold_id, 0)])
                                                                 for f in folds}), folds, cfg).to_csv(o / "permutation_importance_mlp_h5.csv", index=False)
    pd.DataFrame(res.choices).to_csv(o / "hyperparameter_choices.csv", index=False)
    pd.DataFrame(res.inner_scores).to_csv(o / "inner_cv_scores.csv", index=False)
    fits.to_csv(o / "fit_log.csv", index=False)
    pd.DataFrame(res.histories).to_csv(o / "training_histories.csv", index=False)
    pd.DataFrame(res.timings).to_csv(o / "timings.csv", index=False)
    pd.DataFrame([{"index": i, **c, "hidden": json.dumps(c["hidden"])} for i, c in enumerate(cfg["mlp"]["candidates"])]).to_csv(
        o / "search_candidates.csv", index=False)
    rv.to_csv(o / "raw_feature_verification.csv", index=False)
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
                   "touches_final_test": bool(t["match_ids"] & final_test),
                   "match_ids_sha256": hashlib.sha256(",".join(map(str, sorted(t["match_ids"]))).encode()).hexdigest()}
                  for t in trace]).drop(columns="match_ids").to_csv(o / "data_access_audit.csv", index=False)

    # ------------------------------------------------------------------ artefacts and provenance
    pred = pd.DataFrame({"event_id": rows.event_id, "match_id": rows.match_id, "fold_id": res.fold_of_row, "y": rows.y})
    for c in range(K):
        pred[f"{NAME}__p{c}"] = res.proba[:, c]
    for r, p in res.proba_replicates.items():
        for c in range(K):
            pred[f"{NAME}_seed{r}__p{c}"] = p[:, c]
    pred_path = art / "oof_predictions_mlp_h5.parquet"
    tmp = pred_path.with_name("." + pred_path.name + ".tmp")
    pred.to_parquet(tmp, index=False)
    tmp.replace(pred_path)
    arrays = {}
    for (fid, r), st in res.states.items():
        m = mh.model_from_state(res.selected[fid], st)
        ws, bs = m.weight_matrices()
        for i, (w, b) in enumerate(zip(ws, bs)):
            arrays[f"fold{fid}_rep{r}_W{i + 1}"], arrays[f"fold{fid}_rep{r}_b{i + 1}"] = w, b
    models_npz = art / "fold_models_mlp_h5.npz"
    np.savez(models_npz, **arrays, selected=json.dumps({int(k): v for k, v in res.selected.items()}))
    models_pt = art / "fold_models_mlp_h5.pt"
    mh.save_atomic(models_pt, {"identifiers": identifiers, "selected": res.selected, "states": res.states})
    shutil.copy(CONFIG, o / "experiment_config.yaml")
    shutil.copy(PROJECT_ROOT / cfg["task_config"], o / "task_config.yaml")
    _, digest = raw_source_hashes(part.development)
    check("raw source digest equals the canonical manifest", digest == inp["manifest"]["raw_source_digest_development"], digest)
    hist_after = historical_snapshot(cfg)
    check("historical results unchanged (baselines, experiments 5, 5B, 6A: checksums, README, predictions, dataset)",
          hist_after == hist_before and not any(e["mismatches"] for e in hist_after["directories"].values()),
          sum(e.get("checksums.json_files", 0) for e in hist_after["directories"].values()))
    check("run status", True, f"resumed folds: {res.resumed_folds or 'none'}; computed folds: {res.computed_folds}; "
                              f"fits loaded from checkpoints: {res.resumed_fits}", kind="INFO")
    fe = Path(inp["spec"].raw["partition"]["folds_file"])
    write_json(o / "split_manifest.json", {
        "source_split_manifest": "results/markov_hmm_pilot/split_manifest.json",
        "source_split_manifest_sha256": identifiers["split_manifest_sha256"],
        "folds_file": fe.as_posix(), "folds_file_sha256": identifiers["folds_file_sha256"],
        "folds_identical_to_canonical_baselines_and_experiments_5_5b_6a": True,
        "final_test_matches": len(part.final_test), "final_test_loaded": 0,
        "folds": [{"fold_id": f.fold_id, "train_match_ids": list(f.train_match_ids), "validation_match_ids": list(f.validation_match_ids),
                   "validation_rows": int(rows.mask(f.validation_match_ids).sum())} for f in folds]})
    write_json(o / "task_manifest.json", {**inp["manifest"], "raw_source_digest_recomputed": digest,
                                          "row_identity": {"n_rows": int(len(rows.y)), "n_matches": int(len(set(rows.match_id))),
                                                           "event_id_sha256": identifiers["row_event_id_sha256"],
                                                           "target_sha256": identifiers["target_sha256"],
                                                           "fold_of_row_sha256": sha256_array(res.fold_of_row),
                                                           "feature_matrix_sha256": identifiers["feature_matrix_sha256"]}})
    env = environment_info()
    env.pop("platform", None)
    env["packages"]["torch"] = torch.__version__
    write_json(o / "environment.json", {
        **env, "machine": {"processor": platform.machine(), "torch_threads": torch.get_num_threads(),
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "device": "cpu",
                           "mps_available_not_used": bool(torch.backends.mps.is_available())},
        "runtime_seconds": time.time() - started, "checkpoint_identifiers": identifiers,
        "source_files_sha256": source_digest(ALL_SOURCES)["files"],
        "artefacts_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in (pred_path, models_npz, models_pt)},
        "run_status": {"resumed_folds": res.resumed_folds, "computed_folds": res.computed_folds, "resumed_fits": res.resumed_fits},
        "workload": {"n_fits": len(fits), "total_fit_seconds": float(fits["seconds"].sum()),
                     "total_epochs": int(fits["epochs_trained"].sum())},
        "historical_verification": {"before": hist_before, "after": hist_after},
        "random_seeds": {"folds": 20261027, "bootstrap": cfg["evaluation"]["bootstrap_seed"], "training_base": cfg["training"]["seed"],
                         "derivation": "numpy SeedSequence([base, outer, inner+1 (0 = refit), candidate, replicate]); per epoch "
                                       "SeedSequence([fit, 1|2|3|4, epoch]) for permutation / dropout / numpy / python",
                         "permutation_importance": cfg["evaluation"]["permutation_seed"]},
        "commands": ["python scripts/supervise_mlp_h5_experiment6b.py --detach",
                     "python scripts/run_mlp_h5_experiment6b.py", "python scripts/verify_mlp_h5_experiment6b.py"]})
    pd.DataFrame(integrity).to_csv(o / "integrity_audit.csv", index=False)
    (staging / ".staging_experiment6b").unlink()
    write_json(o / "checksums.json", {p.relative_to(o).as_posix(): sha256_file(p) for p in sorted(o.rglob("*"))
                                      if p.is_file() and p.name not in ("checksums.json", "README.md")})
    n_fail = sum(r["result"] == "FAIL" for r in integrity)
    LOGGER.info("integrity audit: %d checks, %d FAIL", len(integrity), n_fail)
    LOGGER.info("pooled log loss: %s", {k: round(float(v), 6) for k, v in mine["log_loss"].items()})
    if n_fail:
        LOGGER.error("integrity audit FAILED - results left in %s (not marked complete)", staging.name)
        return 1
    staging.rename(out)          # atomic: the result directory appears only when complete
    LOGGER.info("done in %.0f s -> %s", time.time() - started, cfg["output_dir"])
    return 0


class _Predictor:
    def __init__(self, cand, state):
        self.model = mh.model_from_state(cand, state)

    def predict_proba(self, ids):
        return mh.predict_proba(self.model, ids)


class _RowsView:
    def __init__(self, rows: HistoryRows, ids: np.ndarray):
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
