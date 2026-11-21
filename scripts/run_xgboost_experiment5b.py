"""Experiment 5B: XGB-H2 (x_t = (E_{t-1}, E_t)) on actions15_360current_v1, the controlled
comparison with the canonical second-order Markov chain.

Same rows, outer folds, nested-CV procedure, search space, candidate seed, round cap and
selection rule as experiment 5. Final-test matches are never loaded. Experiment 5 artefacts
and the canonical dataset are read only (hash-verified). Completed outer folds are
checkpointed atomically and resumed only when every identifier matches.

Usage:
    .venv/bin/python scripts/run_xgboost_experiment5b.py --smoke         # timing; fold-0 inner sets only
    .venv/bin/python scripts/run_xgboost_experiment5b.py --audit-only    # weighted-cell audit only
    caffeinate -i .venv/bin/python scripts/run_xgboost_experiment5b.py   # full run (resumes completed folds)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import xgboost as xgb  # noqa: E402
from build_canonical_task import raw_source_hashes  # noqa: E402
from run_xgboost_experiment5 import load_inputs  # noqa: E402  (experiment 5 loader, unchanged)

from src.sequence_models import canonical_baselines as cb  # noqa: E402
from src.sequence_models import xgboost_experiment as xe  # noqa: E402
from src.sequence_models import xgboost_h2_experiment as h2  # noqa: E402
from src.sequence_models.canonical_task import CLASSES, K, TASK_VERSION  # noqa: E402
from src.sequence_models.evaluation_v2 import confusion_counts, metrics_from_confusion  # noqa: E402
from src.sequence_models.event_history import PAD, HistoryRows, history_ids  # noqa: E402
from src.sequence_models.experiment import environment_info, sha256_file, write_json  # noqa: E402
from src.sequence_models.raw_events import EVENTS_DIR  # noqa: E402
from src.sequence_models.raw_history_check import load_events, reconstruct  # noqa: E402
from src.sequence_models.splits import LeakageError  # noqa: E402

LOGGER = logging.getLogger("xgboost_experiment5b")
CONFIG = PROJECT_ROOT / "configs" / "xgboost_experiment5b_h2_v1.yaml"
ARTEFACTS = PROJECT_ROOT / "data" / "interim" / "xgboost_experiment5b"
EXP5_MODELS = ["frequency", "markov1", "markov2", "logistic_F0", "XGB-F0", "XGB-H5"]
PAIRS = [("XGB-H2", "markov2"), ("XGB-H2", "XGB-F0"), ("XGB-H2", "markov1"), ("XGB-H2", "logistic_F0"),
         ("XGB-H5", "XGB-H2"), ("XGB-H5", "markov2"), ("markov2", "markov1")]
SOURCES = [PROJECT_ROOT / p for p in (
    "src/sequence_models/xgboost_h2_experiment.py", "src/sequence_models/xgboost_experiment.py",
    "src/sequence_models/event_history.py", "src/sequence_models/canonical_task.py",
    "src/sequence_models/evaluation_v2.py", "scripts/run_xgboost_experiment5b.py",
    "scripts/run_xgboost_experiment5.py")]


def verify_exp5_unchanged(cfg: dict) -> dict:
    d = PROJECT_ROOT / cfg["experiment5_results_dir"]
    c = json.loads((d / "checksums.json").read_text())
    bad = [k for k, v in c.items() if sha256_file(d / k) != v]
    return {"files_checked": len(c), "mismatches": bad, "checksums_json_sha256": sha256_file(d / "checksums.json"),
            "readme_sha256": sha256_file(d / "README.md")}


def load_exp5_predictions(cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Stored out-of-fold predictions of experiment 5, verified against its manifest and metrics."""
    d = PROJECT_ROOT / cfg["experiment5_results_dir"]
    o = pd.read_parquet(PROJECT_ROOT / cfg["experiment5_predictions"])
    ri = json.loads((d / "task_manifest.json").read_text())["row_identity"]
    checks = [
        {"check": "exp5 predictions: event_id order", "pass": h2.sha256_array(o["event_id"].to_numpy()) == ri["event_id_sha256"]},
        {"check": "exp5 predictions: targets", "pass": h2.sha256_array(o["y"].to_numpy()) == ri["target_sha256"]},
        {"check": "exp5 predictions: fold of row", "pass": h2.sha256_array(o["fold_id"].to_numpy()) == ri["fold_of_row_sha256"]},
        {"check": "exp5 predictions: unique row ids", "pass": bool(o["event_id"].is_unique)},
    ]
    mc = pd.read_csv(d / "model_comparison.csv").set_index("model")
    y = o["y"].to_numpy()
    for m in EXP5_MODELS:
        p = o[[f"{m}__p{c}" for c in range(K)]].to_numpy()
        ll = float(-np.log(p[np.arange(len(y)), y]).mean())
        checks.append({"check": f"exp5 predictions reproduce stored {m} log loss", "value": abs(ll - mc.at[m, "log_loss"]),
                       "pass": abs(ll - mc.at[m, "log_loss"]) < 1e-12})
    return o, checks


def raw_verification(rows: HistoryRows, tb: pd.DataFrame, rv: dict) -> pd.DataFrame:
    """Independent rebuild from raw JSON: random rows, first rows of periods, last rows of periods."""
    rng = np.random.default_rng(int(rv["seed"]))
    last_pos = tb.groupby("sequence_id")["seq_pos"].max()
    is_end = rows.seq_pos == last_pos.reindex(rows.sequence_id).to_numpy() - 1   # target = last observation
    start = np.flatnonzero(rows.seq_pos == 0)
    end = np.flatnonzero(is_end)
    pick = {"random": rng.choice(len(rows.y), int(rv["n_random"]), replace=False),
            "period_start": rng.choice(start, min(int(rv["n_period_start"]), len(start)), replace=False),
            "period_end": rng.choice(end, min(int(rv["n_period_end"]), len(end)), replace=False)}
    period = dict(zip(tb["event_id"], tb["period"]))
    out, cache = [], {}
    for kind, idx in pick.items():
        for i in sorted(idx, key=lambda j: rows.match_id[j]):
            mid = int(rows.match_id[i])
            if mid not in cache:
                cache = {mid: load_events(EVENTS_DIR, mid)}
            hist, target, hid = reconstruct(cache[mid], rows.event_id[i], 1)
            out.append({"sample": kind, "event_id": rows.event_id[i], "match_id": mid,
                        "period": int(period[rows.event_id[i]]), "seq_pos": int(rows.seq_pos[i]),
                        "E_t-1_pipeline": int(rows.ctx[i, 3]), "E_t_pipeline": int(rows.ctx[i, 4]),
                        "E_t-1_raw": hist[0], "E_t_raw": hist[1], "target_pipeline": int(rows.y[i]), "target_raw": target,
                        "history_event_ids_raw": ";".join(hid),
                        "features_match": hist == rows.ctx[i, 3:].tolist(), "target_match": target == int(rows.y[i])})
    return pd.DataFrame(out)


def smoke(cfg: dict, rows: HistoryRows, ids: np.ndarray, folds) -> None:
    f = folds[0]
    inner_folds = [g for g in folds if g.fold_id != 0]
    inner = []
    for held in inner_folds:
        tr = rows.mask([m for g in inner_folds if g is not held for m in g.validation_match_ids])
        va = rows.mask(held.validation_match_ids)
        if set(rows.match_id[tr | va]) & set(f.validation_match_ids):
            raise LeakageError("smoke test touched outer-validation matches")
        dtr, _ = xe.cell_dmatrix(ids[tr], rows.y[tr])
        dva, nva = xe.cell_dmatrix(ids[va], rows.y[va])
        inner.append((dtr, dva, nva))
        LOGGER.info("XGB-H2 inner split: %d train rows -> %d cells", tr.sum(), dtr.num_row())
    cands = xe.candidate_list(cfg)
    for ci in (1, 6):            # deepest and shallowest candidates (depth 8 and 2), as in experiment 5's smoke test
        t0 = time.perf_counter()
        xe.lockstep_curve(xe.booster_params(cfg, cands[ci]), inner, 40, 10 ** 9)
        LOGGER.info("XGB-H2 candidate %d %s: %.4f s per lockstep round", ci, cands[ci], (time.perf_counter() - t0) / 40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing RESULT directory (checkpoints are kept)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg5 = yaml.safe_load((PROJECT_ROOT / cfg["experiment5_config"]).read_text(encoding="utf-8"))
    for section in ("xgboost", "search"):
        if cfg[section] != cfg5[section]:
            raise SystemExit(f"`{section}` differs from experiment 5's configuration; the controlled comparison requires identity")
    for key in ("bootstrap_replicates", "bootstrap_seed", "probability_tolerance"):
        if cfg["evaluation"][key] != cfg5["evaluation"][key]:
            raise SystemExit(f"evaluation.{key} differs from experiment 5")
    if xgb.__version__ != cfg["xgboost"]["version"]:
        raise SystemExit(f"xgboost {xgb.__version__} installed, configuration requires {cfg['xgboost']['version']}")
    out = PROJECT_ROOT / cfg["output_dir"]
    if not (args.smoke or args.audit_only) and out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{cfg['output_dir']} exists; pass --overwrite to replace it deliberately")
    ckpt = PROJECT_ROOT / cfg["checkpoint_dir"]

    # ------------------------------------------------------------------ inputs (verified)
    exp5_before = verify_exp5_unchanged(cfg)
    if exp5_before["mismatches"]:
        raise SystemExit(f"experiment 5 results changed: {exp5_before['mismatches']}")
    inp = load_inputs(cfg)            # canonical hashes vs both manifests, Tier B set, folds == baseline folds
    tb, folds, part = inp["tb"], inp["folds"], inp["part"]
    rows = HistoryRows.from_table(tb)
    ids = rows.ctx[:, 4 - h2.N_PREVIOUS:]
    ids_direct = history_ids(tb, h2.N_PREVIOUS)[tb["row_eligible"].to_numpy()]     # second construction path
    if not np.array_equal(ids, ids_direct):
        raise RuntimeError("the two H2 construction paths disagree")
    brows = cb.Rows.from_table(tb)
    if not (np.array_equal(ids[:, 1], brows.cur) and np.array_equal(np.where(ids[:, 0] == PAD, -1, ids[:, 0]), brows.prev)
            and np.array_equal(rows.y, brows.y)):
        raise RuntimeError("XGB-H2 inputs differ from the Markov-2 context (E_{t-1}, E_t)")
    m = inp["manifest"]["counts_tier_b"]
    if len(rows.y) != m["eligible_rows"] or len(set(rows.match_id)) != 304:
        raise SystemExit(f"row count {len(rows.y)} / matches {len(set(rows.match_id))} differ from the manifest")
    LOGGER.info("%s: %d rows, %d matches; H2 matrix %s", TASK_VERSION, len(rows.y), len(set(rows.match_id)), ids.shape)
    if args.smoke:
        smoke(cfg, rows, ids, folds)
        return

    # ------------------------------------------------------------------ weighted-cell audit (inner data of outer fold 0)
    a = cfg["weighted_cell_audit"]
    f0 = next(f for f in folds if f.fold_id == int(a["outer_fold"]))
    inner_folds = [g for g in folds if g.fold_id != f0.fold_id]
    held = next(g for g in inner_folds if g.fold_id == int(a["held_inner_fold"]))
    a_tr = rows.mask([mm for g in inner_folds if g is not held for mm in g.validation_match_ids])
    a_va = rows.mask(held.validation_match_ids)
    if set(rows.match_id[a_tr | a_va]) & set(f0.validation_match_ids):
        raise LeakageError("audit data contain outer-validation matches")
    audit_dir = ARTEFACTS / "weighted_cell_audit"
    if not (audit_dir / "divergence.csv").exists():
        LOGGER.info("weighted-cell audit: %d inner-train rows, %d inner-validation rows", a_tr.sum(), a_va.sum())
        aud = h2.weighted_cell_audit(ids, rows.y, a_tr, a_va, cfg, list(a["candidates"]), list(a["rounds"]))
        sel = h2.lockstep_selection_audit(ids, rows.y, rows.match_id, folds, f0.fold_id, cfg, list(a["candidates"]))
        audit_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in {**aud, "selection": sel}.items():
            frame.to_csv(audit_dir / f"{name}.csv", index=False)
        LOGGER.info("weighted-cell audit written")
    if args.audit_only:
        return

    # ------------------------------------------------------------------ nested CV with checkpoints
    identifiers = {
        "experiment": cfg["experiment_name"], "task_version": TASK_VERSION,
        "config_sha256": sha256_file(CONFIG),
        "search_and_xgboost_sha256": hashlib.sha256(json.dumps([cfg["xgboost"], cfg["search"]], sort_keys=True).encode()).hexdigest(),
        "dataset_sha256_tier_b": inp["manifest"]["dataset_sha256_tier_b"],
        "folds_file_sha256": sha256_file(PROJECT_ROOT / inp["spec"].raw["partition"]["folds_file"]),
        "split_manifest_sha256": sha256_file(PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json"),
        "row_event_id_sha256": h2.sha256_array(rows.event_id), "feature_matrix_sha256": h2.sha256_array(ids),
        "source_digest": h2.source_digest(SOURCES)["digest"], "xgboost_version": xgb.__version__,
    }
    trace: list = []
    res = h2.run_nested_cv_checkpointed(ids, rows.match_id, rows.y, folds, cfg, h2.FEATURE_SET, ckpt, identifiers,
                                        trace=trace, log=LOGGER.info)
    LOGGER.info("nested CV complete: computed folds %s, resumed folds %s", res.computed_folds, res.resumed_folds)
    out.mkdir(parents=True, exist_ok=True)
    ARTEFACTS.mkdir(parents=True, exist_ok=True)
    integrity: list[dict] = []

    def check(name, ok, value=None, kind="PASS/FAIL"):
        integrity.append({"check": name, "result": ("PASS" if ok else "FAIL") if kind == "PASS/FAIL" else kind,
                          "value": value})

    # ------------------------------------------------------------------ final audit
    exp5, exp5_checks = load_exp5_predictions(cfg)
    for c in exp5_checks:
        check(c["check"], c["pass"], c.get("value"))
    check("row ids identical to experiment 5 (same order)", np.array_equal(exp5["event_id"].to_numpy(), rows.event_id))
    check("targets identical to experiment 5", np.array_equal(exp5["y"].to_numpy(), rows.y))
    check("fold of every row identical to experiment 5", np.array_equal(exp5["fold_id"].to_numpy(), res.fold_of_row))
    check("690,147 rows and 304 matches", len(rows.y) == 690_147 and len(set(rows.match_id)) == 304, len(rows.y))
    check("every outer fold completed", sorted(c["fold_id"] for c in res.choices) == [0, 1, 2, 3, 4])
    val_counts = np.zeros(len(rows.y), dtype=int)
    for f in folds:
        val_counts[rows.mask(f.validation_match_ids)] += 1
    check("every row predicted exactly once (one validation fold per row)", bool((val_counts == 1).all()))
    check("no duplicate row ids", bool(pd.Series(rows.event_id).is_unique))
    try:
        xe.check_probabilities(res.proba, float(cfg["evaluation"]["probability_tolerance"]))
        prob_ok = bool((res.proba > 0).all())
    except ValueError:
        prob_ok = False
    check("probabilities finite, positive, 15 columns, rows sum to 1 (tol 1e-9)", prob_ok,
          float(np.abs(res.proba.sum(1) - 1).max()))
    for fm in res.fold_metrics:
        sel = res.fold_of_row == fm["fold_id"]
        p, yy = res.proba[sel], rows.y[sel]
        ll = float(-np.log(p[np.arange(len(yy)), yy]).mean())
        check(f"fold {fm['fold_id']} log loss reproduces from saved predictions", abs(ll - fm["log_loss"]) < 1e-12,
              abs(ll - fm["log_loss"]))
    for f in folds:
        for t in trace:
            if t["outer_fold"] == f.fold_id and t["purpose"] != "outer_validation_predict" and t["match_ids"] & set(f.validation_match_ids):
                raise LeakageError(f"fold {f.fold_id}: outer-validation match used for {t['purpose']}")
    check("no outer-validation match in any inner-train / inner-validation / outer-train set",
          True, f"{sum(t['purpose'] != 'outer_validation_predict' for t in trace)} data sets audited")
    final_test = set(part.final_test)
    check("no final-test match in the data, folds or any data set handed to XGBoost",
          not (set(inp["table"]["match_id"]) & final_test) and not any(t["match_ids"] & final_test for t in trace),
          f"{len(final_test)} final-test ids checked")
    # reproducibility: refit fold 0 and compare with the checkpointed predictions
    fz = folds[0]
    tr, va = rows.mask(fz.train_match_ids), rows.mask(fz.validation_match_ids)
    m0 = res.models[0]
    again = xe.XGBNextEventModel(m0.params, m0.n_rounds).fit(ids[tr], rows.y[tr]).predict_proba(ids[va])
    check("refit of fold 0 reproduces the saved predictions bit-identically", np.array_equal(again, res.proba[va]),
          float(np.abs(again - res.proba[va]).max()))
    # row-level refit of fold 0 (weighted-cell limitation, as in experiment 5)
    t0 = time.perf_counter()
    rowfit = xe.XGBNextEventModel(m0.params, m0.n_rounds).fit_rows(ids[tr], rows.y[tr]).predict_proba(ids[va])
    llc = float(-np.log(res.proba[va][np.arange(va.sum()), rows.y[va]]).mean())
    llr = float(-np.log(rowfit[np.arange(va.sum()), rows.y[va]]).mean())
    fitver = pd.DataFrame([{"feature_set": h2.FEATURE_SET, "fold_id": 0, "n_estimators": m0.n_rounds,
                            "log_loss_cells": llc, "log_loss_rows": llr, "abs_log_loss_difference": abs(llc - llr),
                            "max_abs_probability_difference": float(np.abs(rowfit - res.proba[va]).max()),
                            "argmax_agreement": float((rowfit.argmax(1) == res.proba[va].argmax(1)).mean()),
                            "row_level_fit_seconds": time.perf_counter() - t0}])
    fitver.to_csv(out / "fit_verification.csv", index=False)
    check("row-level refit of fold 0 vs weighted cells (limitation, see weighted_cell_audit)", True,
          f"|dLL|={abs(llc - llr):.2e}", kind="LIMITATION")

    # ------------------------------------------------------------------ features and raw verification
    rep, counts = h2.feature_report(ids, rows.y)
    rep["row_id_alignment_with_experiment5"] = bool(np.array_equal(exp5["event_id"].to_numpy(), rows.event_id))
    rep["inputs_equal_markov2_context"] = True
    rep["two_construction_paths_agree"] = True
    write_json(out / "feature_verification.json", rep)
    counts.to_csv(out / "feature_class_counts.csv", index=False)
    rv = raw_verification(rows, tb, cfg["evaluation"]["raw_verification"])
    rv.to_csv(out / "raw_feature_verification.csv", index=False)
    check(f"raw-JSON rebuild of {len(rv)} rows (random, period start, period end) matches",
          bool(rv["features_match"].all() and rv["target_match"].all()), int(rv["features_match"].sum()))
    h2.markov2_backoff_rates(brows.cur, brows.prev, brows.match_id, folds,
                             int(yaml.safe_load((PROJECT_ROOT / cfg["baseline_config"]).read_text())["markov2"]["min_context_count"])
                             ).to_csv(out / "markov2_backoff_rates.csv", index=False)

    # ------------------------------------------------------------------ evaluation (7 models, same rows)
    proba = {mname: exp5[[f"{mname}__p{c}" for c in range(K)]].to_numpy() for mname in EXP5_MODELS}
    proba["XGB-H2"] = res.proba
    tables = xe.evaluate_models(rows.y, rows.match_id, res.fold_of_row, proba, PAIRS, cfg)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    pooled = tables["model_comparison"].set_index("model")
    pm = h2_metrics = metrics_from_confusion(confusion_counts(rows.y, res.proba.argmax(1), K))
    check("pooled accuracy / macro-F1 reproduce from pooled predictions",
          abs(pm["accuracy"] - pooled.at["XGB-H2", "accuracy"]) < 1e-12 and abs(h2_metrics["macro_f1"] - pooled.at["XGB-H2", "macro_f1"]) < 1e-12)
    dist = tables["predicted_class_distribution"]
    dd = dist[dist.model.isin(["XGB-H2", "markov2"])].pivot(index="class", columns="model",
                                                             values=["argmax_predicted_count", "mean_predicted_probability"])
    dd.columns = [f"{a}__{b}" for a, b in dd.columns]
    dd["argmax_count_difference_h2_minus_markov2"] = dd["argmax_predicted_count__XGB-H2"] - dd["argmax_predicted_count__markov2"]
    dd.reset_index().to_csv(out / "predicted_class_distribution_h2_vs_markov2.csv", index=False)
    pd.DataFrame(res.choices).to_csv(out / "hyperparameter_choices.csv", index=False)
    pd.DataFrame(res.curves).to_csv(out / "inner_cv_curves.csv", index=False)
    pd.DataFrame(res.timings).to_csv(out / "timings.csv", index=False)
    pd.DataFrame(res.fold_metrics).to_csv(out / "fold_metrics_from_checkpoints.csv", index=False)
    pd.DataFrame([{"index": i, **c} for i, c in enumerate(xe.candidate_list(cfg))]).to_csv(out / "search_candidates.csv", index=False)
    xe.permutation_importance(_rows_view(rows, ids), _nested_view(res), folds, cfg).to_csv(
        out / "permutation_importance_h2.csv", index=False)
    pd.DataFrame([{**t, "n_matches": len(t["match_ids"]),
                   "touches_outer_validation": bool(t["match_ids"] & set(next(f for f in folds if f.fold_id == t["outer_fold"]).validation_match_ids)),
                   "match_ids_sha256": hashlib.sha256(",".join(map(str, sorted(t["match_ids"]))).encode()).hexdigest()}
                  for t in trace]).drop(columns="match_ids").to_csv(out / "data_access_audit.csv", index=False)
    for name in ("divergence", "round1_structure", "checks", "selection"):
        shutil.copy(audit_dir / f"{name}.csv", out / f"weighted_cell_audit_{name}.csv")

    # ------------------------------------------------------------------ artefacts and provenance
    pred = pd.DataFrame({"event_id": rows.event_id, "match_id": rows.match_id, "fold_id": res.fold_of_row, "y": rows.y})
    for c in range(K):
        pred[f"XGB-H2__p{c}"] = res.proba[:, c]
    pred.to_parquet(ARTEFACTS / "oof_predictions_h2.parquet", index=False)
    shutil.copy(CONFIG, out / "experiment_config.yaml")
    shutil.copy(PROJECT_ROOT / cfg["task_config"], out / "task_config.yaml")
    _, digest = raw_source_hashes(part.development)
    check("raw source digest equals the canonical manifest", digest == inp["manifest"]["raw_source_digest_development"], digest)
    exp5_after = verify_exp5_unchanged(cfg)
    check("experiment 5 results unchanged (checksums.json + README)",
          not exp5_after["mismatches"] and exp5_after == exp5_before, exp5_after["files_checked"])
    check("run status", True, "resumed folds: " + (str(res.resumed_folds) if res.resumed_folds else "none") +
          f"; computed folds: {res.computed_folds}", kind="INFO")
    fe = Path(inp["spec"].raw["partition"]["folds_file"])
    write_json(out / "split_manifest.json", {
        "source_split_manifest": "results/markov_hmm_pilot/split_manifest.json",
        "source_split_manifest_sha256": identifiers["split_manifest_sha256"],
        "folds_file": fe.as_posix(), "folds_file_sha256": identifiers["folds_file_sha256"],
        "folds_identical_to_canonical_baselines_and_experiment5": True,
        "final_test_matches": len(part.final_test), "final_test_loaded": 0,
        "folds": [{"fold_id": f.fold_id, "n_train_matches": len(f.train_match_ids),
                   "n_validation_matches": len(f.validation_match_ids),
                   "validation_rows": int(rows.mask(f.validation_match_ids).sum())} for f in folds]})
    write_json(out / "task_manifest.json", {**inp["manifest"], "raw_source_digest_recomputed": digest,
                                            "row_identity": {"n_rows": int(len(rows.y)), "n_matches": int(len(set(rows.match_id))),
                                                             "event_id_sha256": identifiers["row_event_id_sha256"],
                                                             "target_sha256": h2.sha256_array(rows.y),
                                                             "fold_of_row_sha256": h2.sha256_array(res.fold_of_row)}})
    env = environment_info()
    env.pop("platform", None)
    env["packages"]["xgboost"] = xgb.__version__
    write_json(out / "environment.json", {
        **env, "runtime_seconds": time.time() - started, "checkpoint_identifiers": identifiers,
        "source_files_sha256": h2.source_digest(SOURCES)["files"],
        "run_status": {"completed": True, "resumed_folds": res.resumed_folds, "computed_folds": res.computed_folds},
        "workload": {"n_candidates": len(xe.candidate_list(cfg)), "n_fits": res.n_fits, "n_boosting_rounds": res.n_boosting_rounds},
        "experiment5_verification": {"before": exp5_before, "after": exp5_after},
        "random_seeds": {"folds": 20261027, "bootstrap": cfg["evaluation"]["bootstrap_seed"], "xgboost": cfg["xgboost"]["seed"],
                         "candidates": cfg["search"]["candidate_seed"], "permutation": cfg["evaluation"]["permutation_seed"],
                         "raw_verification_sample": cfg["evaluation"]["raw_verification"]["seed"]},
        "commands": ["python scripts/run_xgboost_experiment5b.py --smoke",
                     "caffeinate -i python scripts/run_xgboost_experiment5b.py"]})
    pd.DataFrame(integrity).to_csv(out / "integrity_audit.csv", index=False)
    write_json(out / "checksums.json", {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*"))
                                        if p.is_file() and p.name not in ("checksums.json", "README.md")})
    n_fail = sum(r["result"] == "FAIL" for r in integrity)
    LOGGER.info("integrity audit: %d checks, %d FAIL", len(integrity), n_fail)
    LOGGER.info("done in %.0f s -> %s", time.time() - started, cfg["output_dir"])
    if n_fail:
        raise SystemExit("integrity audit FAILED - see integrity_audit.csv")


class _rows_view:
    """Adapter so experiment 5's permutation_importance can score XGB-H2 (2 positions)."""

    def __init__(self, rows: HistoryRows, ids: np.ndarray):
        self.rows, self.ids, self.y, self.match_id = rows, ids, rows.y, rows.match_id

    def features(self, _name):
        return self.ids

    def mask(self, match_ids):
        return self.rows.mask(match_ids)


def _nested_view(res):
    class V:
        feature_set = res.feature_set
        models = res.models
    return V()


if __name__ == "__main__":
    main()
