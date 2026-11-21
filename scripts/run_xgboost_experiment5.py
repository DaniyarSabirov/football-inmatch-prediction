"""Experiment 5: XGBoost (XGB-F0, XGB-H5) on the canonical task actions15_360current_v1.

Tier B (304 development matches), experiment 2's 5 match-grouped outer folds, Protocol B rows,
nested inner CV for hyperparameters and the number of boosting rounds. Final-test matches are
never loaded. The canonical dataset is READ (and its hashes verified), never rebuilt. The
canonical baselines are recomputed in memory (deterministic) for paired comparisons and checked
against the stored baseline results, which are not modified.

Usage:
    .venv/bin/python scripts/run_xgboost_experiment5.py --smoke   # timing only; no outer-validation data
    .venv/bin/python scripts/run_xgboost_experiment5.py           # full run (refuses to overwrite)
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

from src.sequence_models import canonical_baselines as cb  # noqa: E402
from src.sequence_models import xgboost_experiment as xe  # noqa: E402
from src.sequence_models.canonical_task import CLASSES, TASK_VERSION, dataset_sha256, load_task_spec  # noqa: E402
from src.sequence_models.config_v2 import load_config_v2  # noqa: E402
from src.sequence_models.event_history import FEATURE_SETS, PAD, HistoryRows  # noqa: E402
from src.sequence_models.experiment import environment_info, sha256_file, write_json  # noqa: E402
from src.sequence_models.experiment_v2 import load_partition  # noqa: E402
from src.sequence_models.raw_events import EVENTS_DIR  # noqa: E402
from src.sequence_models.raw_history_check import load_events, reconstruct  # noqa: E402
from src.sequence_models.splits import FoldAssignment, LeakageError, assert_no_final_test, verify_folds  # noqa: E402

LOGGER = logging.getLogger("xgboost_experiment5")
CONFIG = PROJECT_ROOT / "configs" / "xgboost_experiment5_v1.yaml"
ARTEFACTS = PROJECT_ROOT / "data" / "interim" / "xgboost_experiment5"   # git-ignored: models, OOF predictions

PAIRS = [("XGB-F0", "markov1"), ("XGB-F0", "logistic_F0"), ("XGB-F0", "frequency"),
         ("XGB-H5", "XGB-F0"), ("XGB-H5", "markov2"), ("XGB-H5", "markov1"), ("XGB-H5", "logistic_F0"),
         ("markov2", "markov1")]


def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes() if a.dtype != object
                          else "\n".join(map(str, a)).encode()).hexdigest()


def load_inputs(cfg: dict) -> dict:
    """Canonical Tier B table (verified against both manifests), partition and folds."""
    spec = load_task_spec(PROJECT_ROOT / cfg["task_config"])
    part = load_partition(load_config_v2())
    ddir = spec.path("dataset_dir")
    manifest = json.loads((ddir / "manifest.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((PROJECT_ROOT / cfg["baseline_results_dir"] / "task_manifest.json").read_text())
    table = pd.read_parquet(ddir / "observations.parquet")
    assert_no_final_test(table["match_id"].unique().tolist(), part.final_test)
    if set(table["task_version"]) != {TASK_VERSION}:
        raise ValueError("dataset is not actions15_360current_v1")
    dev_hash = dataset_sha256(table.drop(columns="tier"))
    tb = table[table["tier"] == "tier_b"].reset_index(drop=True)
    tb_hash = dataset_sha256(tb)
    for key, value in (("dataset_sha256_development", dev_hash), ("dataset_sha256_tier_b", tb_hash)):
        if manifest[key] != value or base_manifest[key] != value:
            raise ValueError(f"{key} differs from the canonical manifest / the baseline run")
    if sorted(tb["match_id"].unique()) != sorted(part.tier_b):
        raise LeakageError("Tier B match set differs from the partition")
    folds_payload = json.loads(spec.path("folds_file").read_text(encoding="utf-8"))
    folds = [FoldAssignment(int(f["fold_id"]), tuple(f["train_match_ids"]), tuple(f["validation_match_ids"]))
             for f in folds_payload["scenarioD15"]]
    verify_folds(folds, part.tier_b)
    for f in folds:
        assert_no_final_test(f.train_match_ids + f.validation_match_ids, part.final_test)
    base_folds = json.loads((PROJECT_ROOT / cfg["baseline_results_dir"] / "split_manifest.json").read_text())["folds"]
    if [(f.fold_id, list(f.train_match_ids), list(f.validation_match_ids)) for f in folds] != \
            [(f["fold_id"], f["train_match_ids"], f["validation_match_ids"]) for f in base_folds]:
        raise LeakageError("outer folds differ from those of the canonical baseline run")
    return {"spec": spec, "part": part, "table": table, "tb": tb, "manifest": manifest, "folds": folds}


def smoke(cfg: dict, inp: dict) -> None:
    """Technical smoke test: inner sets of outer fold 0 ONLY (outer-validation rows of fold 0 are
    never built), 2 candidates, few rounds. Prints timings and a workload estimate."""
    rows = HistoryRows.from_table(inp["tb"])
    folds = inp["folds"]
    f = folds[0]
    inner_folds = [g for g in folds if g.fold_id != f.fold_id]
    cands = xe.candidate_list(cfg)
    rounds = 40
    for fs in FEATURE_SETS:
        ids = rows.features(fs)
        inner = []
        for held in inner_folds:
            tr = rows.mask([m for g in inner_folds if g is not held for m in g.validation_match_ids])
            va = rows.mask(held.validation_match_ids)
            if set(rows.match_id[tr | va]) & set(f.validation_match_ids):
                raise LeakageError("smoke test touched outer-validation matches")
            dtr, _ = xe.cell_dmatrix(ids[tr], rows.y[tr])
            dva, nva = xe.cell_dmatrix(ids[va], rows.y[va])
            inner.append((dtr, dva, nva))
            LOGGER.info("%s inner split: %d train rows -> %d cells", fs, tr.sum(), dtr.num_row())
        for ci in sorted({int(np.argmax([c["max_depth"] for c in cands])), int(np.argmin([c["max_depth"] for c in cands]))}):
            t0 = time.perf_counter()
            curve = xe.lockstep_curve(xe.booster_params(cfg, cands[ci]), inner, rounds, 10 ** 9)
            sec = (time.perf_counter() - t0) / rounds
            LOGGER.info("%s candidate %d %s: %.4f s per lockstep round (4 boosters); LL after %d rounds %.5f",
                        fs, ci, cands[ci], sec, rounds, curve[-1])
    LOGGER.info("candidates: %s", cands)


def verify_against_raw(rows: HistoryRows, tb: pd.DataFrame, n_random: int, n_start: int, seed: int) -> pd.DataFrame:
    """Independent reconstruction of a sample of real rows from the raw JSON files."""
    rng = np.random.default_rng(seed)
    start = np.flatnonzero(rows.seq_pos < 4)
    pick = np.r_[rng.choice(len(rows.y), n_random, replace=False), rng.choice(start, n_start, replace=False)]
    period = dict(zip(tb["event_id"], tb["period"]))
    cache, out = {}, []
    for i in pick:
        mid = int(rows.match_id[i])
        if mid not in cache:
            cache = {mid: load_events(EVENTS_DIR, mid)}
        hist, target, hist_ids = reconstruct(cache[mid], rows.event_id[i], 4)
        out.append({"event_id": rows.event_id[i], "match_id": mid, "period": int(period[rows.event_id[i]]),
                    "seq_pos": int(rows.seq_pos[i]), "features_pipeline": rows.ctx[i].tolist(), "features_raw": hist,
                    "target_pipeline": int(rows.y[i]), "target_raw": target,
                    "history_event_ids_raw": ";".join(hist_ids),
                    "features_match": hist == rows.ctx[i].tolist(), "target_match": target == int(rows.y[i])})
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if xgb.__version__ != cfg["xgboost"]["version"]:
        raise SystemExit(f"xgboost {xgb.__version__} installed, configuration requires {cfg['xgboost']['version']}")
    out = PROJECT_ROOT / cfg["output_dir"]
    if not args.smoke and out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{cfg['output_dir']} exists; pass --overwrite to replace it deliberately")
    inp = load_inputs(cfg)
    if args.smoke:
        smoke(cfg, inp)
        return
    out.mkdir(parents=True, exist_ok=True)
    ARTEFACTS.mkdir(parents=True, exist_ok=True)
    tb, folds, part = inp["tb"], inp["folds"], inp["part"]

    # ------------------------------------------------------------------ rows (identical for all models)
    rows = HistoryRows.from_table(tb)
    brows = cb.Rows.from_table(tb)
    if not (np.array_equal(rows.match_id, brows.match_id) and np.array_equal(rows.y, brows.y)
            and np.array_equal(rows.ctx[:, 4], brows.cur)
            and np.array_equal(np.where(rows.ctx[:, 3] == PAD, -1, rows.ctx[:, 3]), brows.prev)):
        raise RuntimeError("XGBoost rows differ from the canonical baseline rows")
    LOGGER.info("%s: %d Tier B eligible rows, %d matches", TASK_VERSION, len(rows.y), len(set(rows.match_id)))
    raw_check = verify_against_raw(rows, tb, 400, 100, 5)
    raw_check.to_csv(out / "raw_feature_verification.csv", index=False)
    if not (raw_check["features_match"].all() and raw_check["target_match"].all()):
        raise RuntimeError("feature/target verification against raw JSON failed")
    LOGGER.info("raw-JSON verification: %d/%d rows match", int(raw_check.features_match.sum()), len(raw_check))

    # ------------------------------------------------------------------ baselines (recomputed, verified)
    bcfg = yaml.safe_load((PROJECT_ROOT / cfg["baseline_results_dir"] / "experiment_config.yaml").read_text())
    if bcfg != yaml.safe_load((PROJECT_ROOT / cfg["baseline_config"]).read_text()):
        raise RuntimeError("baseline configuration differs from the stored baseline run")
    bcv = cb.run_cv(brows, folds, bcfg)
    stored = pd.read_csv(PROJECT_ROOT / cfg["baseline_results_dir"] / "metrics_by_fold.csv")
    recheck = []
    for m in cb.MODELS:
        for f in folds:
            sel = bcv.fold_of_row == f.fold_id
            ll = -np.log(bcv.proba[m][sel][np.arange(sel.sum()), brows.y[sel]]).mean()
            ref = float(stored[(stored.model == m) & (stored.fold_id == f.fold_id)].log_loss.iloc[0])
            recheck.append({"model": m, "fold_id": f.fold_id, "recomputed_log_loss": ll, "stored_log_loss": ref,
                            "abs_difference": abs(ll - ref)})
    recheck = pd.DataFrame(recheck)
    recheck.to_csv(out / "baseline_recomputation_check.csv", index=False)
    if recheck["abs_difference"].max() > 1e-10:
        raise RuntimeError("recomputed baselines differ from the stored canonical baseline results")
    LOGGER.info("baselines recomputed; max |Δ log loss| vs stored = %.2e", recheck["abs_difference"].max())

    # ------------------------------------------------------------------ XGBoost nested CV
    trace: list = []
    nested = {}
    for fs in FEATURE_SETS:
        nested[fs] = xe.run_nested_cv(rows, folds, cfg, fs, trace=trace, log=LOGGER.info)
        if not np.array_equal(nested[fs].fold_of_row, bcv.fold_of_row):
            raise LeakageError("fold assignment of rows differs from the baselines")
    # audit of every data set handed to XGBoost
    for f in folds:
        for t in trace:
            if t["outer_fold"] == f.fold_id and t["purpose"] in ("inner_train", "inner_validation", "outer_train"):
                if t["match_ids"] & set(f.validation_match_ids):
                    raise LeakageError(f"fold {f.fold_id}: outer-validation match used for {t['purpose']}")
    pd.DataFrame([{**t, "n_matches": len(t["match_ids"]),
                   "touches_outer_validation": bool(t["match_ids"] & set(next(f for f in folds if f.fold_id == t["outer_fold"]).validation_match_ids)),
                   "match_ids_sha256": hashlib.sha256(",".join(map(str, sorted(t["match_ids"]))).encode()).hexdigest()}
                  for t in trace]).drop(columns="match_ids").to_csv(out / "data_access_audit.csv", index=False)

    # ------------------------------------------------------------------ verification fits
    checks = []
    fid = int(cfg["evaluation"]["row_level_check_fold"])
    f = next(g for g in folds if g.fold_id == fid)
    tr, va = rows.mask(f.train_match_ids), rows.mask(f.validation_match_ids)
    for fs in FEATURE_SETS:
        model = nested[fs].models[fid]
        ids = rows.features(fs)
        p_cell = nested[fs].proba[va]
        again = xe.XGBNextEventModel(model.params, model.n_rounds).fit(ids[tr], rows.y[tr]).predict_proba(ids[va])
        t0 = time.perf_counter()
        p_row = xe.XGBNextEventModel(model.params, model.n_rounds).fit_rows(ids[tr], rows.y[tr]).predict_proba(ids[va])
        ll = lambda p: float(-np.log(p[np.arange(va.sum()), rows.y[va]]).mean())  # noqa: E731
        checks.append({"feature_set": fs, "fold_id": fid, "n_estimators": model.n_rounds,
                       "refit_identical": bool(np.array_equal(again, p_cell)),
                       "log_loss_cells": ll(p_cell), "log_loss_rows": ll(p_row),
                       "abs_log_loss_difference": abs(ll(p_cell) - ll(p_row)),
                       "max_abs_probability_difference": float(np.abs(p_cell - p_row).max()),
                       "argmax_agreement": float((p_cell.argmax(1) == p_row.argmax(1)).mean()),
                       "row_level_fit_seconds": time.perf_counter() - t0})
        LOGGER.info("verification %s: %s", fs, checks[-1])
    pd.DataFrame(checks).to_csv(out / "fit_verification.csv", index=False)

    # ------------------------------------------------------------------ evaluation
    proba = {**bcv.proba, **{fs: nested[fs].proba for fs in FEATURE_SETS}}
    tables = xe.evaluate_models(rows.y, rows.match_id, bcv.fold_of_row, proba, PAIRS, cfg)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame([c for fs in FEATURE_SETS for c in nested[fs].choices]).to_csv(out / "hyperparameter_choices.csv", index=False)
    pd.DataFrame([c for fs in FEATURE_SETS for c in nested[fs].curves]).to_csv(out / "inner_cv_curves.csv", index=False)
    pd.DataFrame([c for fs in FEATURE_SETS for c in nested[fs].timings]).to_csv(out / "timings.csv", index=False)
    pd.DataFrame([{"index": i, **c} for i, c in enumerate(xe.candidate_list(cfg))]).to_csv(out / "search_candidates.csv", index=False)
    pd.DataFrame(bcv.counts).to_csv(out / "fold_counts.csv", index=False)
    xe.permutation_importance(rows, nested["XGB-H5"], folds, cfg).to_csv(out / "permutation_importance_h5.csv", index=False)
    xe.gain_importance(nested["XGB-H5"]).to_csv(out / "gain_importance_h5.csv", index=False)

    # ------------------------------------------------------------------ artefacts (git-ignored) and provenance
    pred = pd.DataFrame({"event_id": rows.event_id, "match_id": rows.match_id, "fold_id": bcv.fold_of_row, "y": rows.y})
    for name, p in proba.items():
        for c in range(len(CLASSES)):
            pred[f"{name}__p{c}"] = p[:, c].astype(np.float64)
    pred.to_parquet(ARTEFACTS / "oof_predictions.parquet", index=False)
    for fs in FEATURE_SETS:
        for fid_, model in nested[fs].models.items():
            model.booster.save_model(ARTEFACTS / f"{fs}_fold{fid_}.ubj")
    shutil.copy(CONFIG, out / "experiment_config.yaml")
    shutil.copy(PROJECT_ROOT / cfg["task_config"], out / "task_config.yaml")
    _, digest = raw_source_hashes(part.development)
    if digest != inp["manifest"]["raw_source_digest_development"]:
        raise RuntimeError("raw source files differ from the canonical manifest")
    fe = Path(inp["spec"].raw["partition"]["folds_file"])
    write_json(out / "split_manifest.json", {
        "source_split_manifest": "results/markov_hmm_pilot/split_manifest.json",
        "source_split_manifest_sha256": sha256_file(PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json"),
        "folds_file": fe.as_posix(), "folds_file_sha256": sha256_file(PROJECT_ROOT / fe),
        "folds_identical_to_canonical_baselines": True, "final_test_matches": len(part.final_test), "final_test_loaded": 0,
        "folds": [{"fold_id": f.fold_id, "n_train_matches": len(f.train_match_ids),
                   "n_validation_matches": len(f.validation_match_ids)} for f in folds]})
    write_json(out / "task_manifest.json", {
        **inp["manifest"], "raw_source_digest_recomputed": digest,
        "row_identity": {"n_rows": int(len(rows.y)), "n_matches": int(len(set(rows.match_id))),
                         "event_id_sha256": sha_array(rows.event_id), "target_sha256": sha_array(rows.y),
                         "fold_of_row_sha256": sha_array(bcv.fold_of_row),
                         "rows_identical_for": ["frequency", "markov1", "markov2", "logistic_F0", *FEATURE_SETS]},
        "h5_rows_with_padding": {f"E_{{t-{k}}}": int((rows.ctx[:, 4 - k] == PAD).sum()) for k in range(1, 5)}})
    env = environment_info()
    env.pop("platform", None)
    env["packages"]["xgboost"] = xgb.__version__
    env["xgboost_build"] = {k: xgb.build_info().get(k) for k in ("USE_OPENMP", "BUILD_WITH_SHARED_PTHREAD_LIB")}
    write_json(out / "environment.json", {
        **env, "runtime_seconds": time.time() - started,
        "workload": {fs: {"n_candidates": len(xe.candidate_list(cfg)), "n_fits": nested[fs].n_fits,
                          "n_boosting_rounds": nested[fs].n_boosting_rounds} for fs in FEATURE_SETS},
        "random_seeds": {"folds": 20261027, "bootstrap": cfg["evaluation"]["bootstrap_seed"],
                         "xgboost": cfg["xgboost"]["seed"], "candidates": cfg["search"]["candidate_seed"],
                         "permutation": cfg["evaluation"]["permutation_seed"], "raw_verification_sample": 5},
        "determinism": f"CPU hist, nthread={cfg['xgboost']['nthread']}: refits are bit-identical (fit_verification.csv)",
        "commands": ["python scripts/build_canonical_task.py   # only if data/interim/canonical_task is missing",
                     "python scripts/run_xgboost_experiment5.py --smoke",
                     "python scripts/run_xgboost_experiment5.py --overwrite"]})
    write_json(out / "checksums.json", {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*"))
                                        if p.is_file() and p.name not in ("checksums.json", "README.md")})
    LOGGER.info("done in %.0f s -> %s", time.time() - started, cfg["output_dir"])


if __name__ == "__main__":
    main()
