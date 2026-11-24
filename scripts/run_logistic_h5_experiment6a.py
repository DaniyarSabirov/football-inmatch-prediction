"""Experiment 6A: multinomial logistic regression with a five-event history (MLR-H5) on the
canonical task actions15_360current_v1.

Same rows (690,147), outer folds and nested-CV protocol as experiments 5 / 5B; same history
features as XGB-H5 (``event_history``, unchanged). Final-test matches are never loaded. The
canonical dataset, the stored predictions of experiments 5 / 5B and all historical result
directories are read only and hash-verified before and after the run. Completed outer folds
are checkpointed atomically and resumed only when every identifier matches.

Usage:
    .venv/bin/python scripts/run_logistic_h5_experiment6a.py --smoke    # fold-0 inner data only; timing/memory
    .venv/bin/python scripts/run_logistic_h5_experiment6a.py            # full run (refuses to overwrite results)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
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

import scipy  # noqa: E402
import sklearn  # noqa: E402
from build_canonical_task import raw_source_hashes  # noqa: E402
from run_xgboost_experiment5 import load_inputs  # noqa: E402  (experiment 5 loader, unchanged)

from src.sequence_models import canonical_baselines as cb  # noqa: E402
from src.sequence_models import logistic_h5_experiment as lh  # noqa: E402
from src.sequence_models import xgboost_experiment as xe  # noqa: E402
from src.sequence_models.canonical_task import CLASSES, K, TASK_VERSION  # noqa: E402
from src.sequence_models.event_history import PAD, PAD_NAME, HistoryRows, encode, history_ids, position_label  # noqa: E402
from src.sequence_models.experiment import environment_info, sha256_file, write_json  # noqa: E402
from src.sequence_models.raw_events import EVENTS_DIR  # noqa: E402
from src.sequence_models.raw_history_check import canonical_stream, is_companion, load_events, reconstruct  # noqa: E402
from src.sequence_models.splits import LeakageError  # noqa: E402
from src.sequence_models.xgboost_h2_experiment import sha256_array, source_digest  # noqa: E402

LOGGER = logging.getLogger("logistic_h5_experiment6a")
CONFIG = PROJECT_ROOT / "configs" / "logistic_h5_experiment6a_v1.yaml"
NAME = lh.FEATURE_SET
EXP5_MODELS = ["frequency", "markov1", "markov2", "logistic_F0", "XGB-F0", "XGB-H5"]
PRIOR_MODELS = EXP5_MODELS + ["XGB-H2"]
PAIRS = [(NAME, "XGB-H5"), (NAME, "logistic_F0"), (NAME, "markov2"), (NAME, "markov1"), (NAME, "XGB-H2"),
         (NAME, "XGB-F0"), ("XGB-H5", "markov2")]
# Files whose content determines the fitted models (checkpoint identity).
MODEL_SOURCES = [PROJECT_ROOT / p for p in (
    "src/sequence_models/logistic_h5_experiment.py", "src/sequence_models/logistic.py",
    "src/sequence_models/event_history.py", "src/sequence_models/canonical_task.py")]
ALL_SOURCES = MODEL_SOURCES + [PROJECT_ROOT / p for p in (
    "src/sequence_models/xgboost_experiment.py", "src/sequence_models/evaluation_v2.py",
    "src/sequence_models/raw_history_check.py", "scripts/run_logistic_h5_experiment6a.py",
    "scripts/run_xgboost_experiment5.py")]


# ============================================================================ historical artefacts
def historical_snapshot(cfg: dict) -> dict:
    """Checksum verification of every historical result directory this run reads, plus the
    hashes of the stored prediction files and the canonical dataset files."""
    out = {"directories": {}, "files": {}}
    for key in ("baseline_results_dir", "experiment5_results_dir", "experiment5b_results_dir"):
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
    for rel in (cfg["experiment5_predictions"], cfg["experiment5b_predictions"],
                "data/interim/canonical_task/actions15_360current_v1/observations.parquet",
                "data/interim/canonical_task/actions15_360current_v1/manifest.json"):
        out["files"][rel] = sha256_file(PROJECT_ROOT / rel)
    return out


def load_prior_predictions(cfg: dict, rows: HistoryRows, fold_of_row: np.ndarray) -> tuple[dict, list[dict]]:
    """Stored out-of-fold predictions of experiments 5 and 5B, aligned row by row (never refitted)."""
    checks = []
    p5 = pd.read_parquet(PROJECT_ROOT / cfg["experiment5_predictions"])
    p5b = pd.read_parquet(PROJECT_ROOT / cfg["experiment5b_predictions"])
    ri = json.loads((PROJECT_ROOT / cfg["experiment5_results_dir"] / "task_manifest.json").read_text())["row_identity"]
    checks.append(("experiment 5 row identity hash (event ids, order) equals this run's rows",
                   ri["event_id_sha256"] == sha256_array(rows.event_id) and ri["n_rows"] == len(rows.y), ri["n_rows"]))
    checks.append(("experiment 5 target hash equals this run's targets", ri["target_sha256"] == sha256_array(rows.y), None))
    checks.append(("experiment 5 fold-of-row hash equals this run's folds",
                   ri["fold_of_row_sha256"] == sha256_array(fold_of_row), None))
    for label, o in (("experiment 5", p5), ("experiment 5B", p5b)):
        checks.append((f"{label} predictions: event_id identical row by row", np.array_equal(o["event_id"].to_numpy(), rows.event_id), len(o)))
        checks.append((f"{label} predictions: match_id identical row by row", np.array_equal(o["match_id"].to_numpy(), rows.match_id), None))
        checks.append((f"{label} predictions: target identical row by row", np.array_equal(o["y"].to_numpy(), rows.y), None))
        checks.append((f"{label} predictions: fold identical row by row", np.array_equal(o["fold_id"].to_numpy(), fold_of_row), None))
        checks.append((f"{label} predictions: unique row ids", bool(o["event_id"].is_unique), None))
    proba = {m: p5[[f"{m}__p{c}" for c in range(K)]].to_numpy() for m in EXP5_MODELS}
    proba["XGB-H2"] = p5b[[f"XGB-H2__p{c}" for c in range(K)]].to_numpy()
    for m, p in proba.items():
        xe.check_probabilities(p, float(cfg["evaluation"]["probability_tolerance"]))
    return proba, [{"check": c, "result": "PASS" if ok else "FAIL", "value": v} for c, ok, v in checks]


# ============================================================================ raw-JSON verification
def raw_verification(rows: HistoryRows, tb: pd.DataFrame, rv: dict) -> pd.DataFrame:
    """Independent rebuild of history, one-hot feature names and target from raw StatsBomb JSON
    (``raw_history_check``: shares no code with the table builder or ``event_history``)."""
    rng = np.random.default_rng(int(rv["seed"]))
    st = rv["strata"]
    n = len(rows.y)
    e = tb["row_eligible"].to_numpy()
    last_pos = tb.groupby("sequence_id")["seq_pos"].transform("max").to_numpy()[e]
    prev_index = tb.groupby("sequence_id")["index"].shift(1).to_numpy()[e]
    cur_index = tb["index"].to_numpy()[e]
    first_of_match = np.r_[True, rows.match_id[1:] != rows.match_id[:-1]]
    last_of_match = np.r_[rows.match_id[1:] != rows.match_id[:-1], True]
    share = np.bincount(rows.y, minlength=K) / n
    rare = np.flatnonzero(share < 0.01)
    rare_row = np.isin(rows.y, rare) | np.isin(rows.ctx, rare).any(axis=1)
    pools = {
        "random": np.arange(n), "seq_pos_0": np.flatnonzero(rows.seq_pos == 0),
        "seq_pos_1": np.flatnonzero(rows.seq_pos == 1), "seq_pos_2_3": np.flatnonzero(np.isin(rows.seq_pos, [2, 3])),
        "period_end": np.flatnonzero(rows.seq_pos == last_pos - 1),
        "match_first_row": np.flatnonzero(first_of_match), "match_last_row": np.flatnonzero(last_of_match),
        "excluded_raw_between": np.flatnonzero((rows.seq_pos >= 1) & (cur_index - np.nan_to_num(prev_index, nan=-1) > 1)),
        "rare_class": np.flatnonzero(rare_row),
    }
    # Edge cases that need raw records: scan seeded random matches in raw JSON (independent code).
    scan = rng.choice(np.unique(rows.match_id), int(rv["n_scan_matches"]), replace=False)
    pos_of = {eid: i for i, eid in enumerate(rows.event_id)}
    comp, same_ts = [], []
    for mid in sorted(int(m) for m in scan):
        events = load_events(EVENTS_DIR, mid)
        for period in sorted({ev["period"] for ev in events if ev.get("period") != 5}):
            raw = sorted((ev for ev in events if ev.get("period") == period), key=lambda ev: ev["index"])
            companions, prev = [], None
            for ev in raw:
                if is_companion(ev, prev):
                    companions.append(ev["index"])
                prev = ev
            companions = np.asarray(companions)
            stream = canonical_stream(events, period)
            for p_, ev in enumerate(stream):
                i = pos_of.get(ev["id"])
                if i is None:
                    continue
                lo = stream[max(p_ - 4, 0)]["index"]
                if len(companions) and ((companions > lo) & (companions < ev["index"])).any():
                    comp.append(i)
                if p_ >= 1 and stream[p_ - 1]["timestamp"] == ev["timestamp"]:
                    same_ts.append(i)
    pools["after_companion"] = np.asarray(comp, dtype=int)
    pools["repeated_timestamp"] = np.asarray(same_ts, dtype=int)
    picked: dict[int, list[str]] = {}
    for name, size in st.items():
        pool = pools[name]
        for i in rng.choice(pool, min(int(size), len(pool)), replace=False):
            picked.setdefault(int(i), []).append(name)
    period_of = dict(zip(tb["event_id"], tb["period"]))
    target_eid = dict(zip(tb["event_id"], tb["target_event_id"]))
    out, cache = [], {}
    for i in sorted(picked, key=lambda j: (rows.match_id[j], j)):
        mid, eid = int(rows.match_id[i]), rows.event_id[i]
        if mid not in cache:
            cache = {mid: load_events(EVENTS_DIR, mid)}
        events = cache[mid]
        hist, target, hid = reconstruct(events, eid, 4)
        by_id = {ev["id"]: ev for ev in events}
        stream = canonical_stream(events, by_id[eid]["period"])
        pos = next(j for j, ev in enumerate(stream) if ev["id"] == eid)
        nxt = stream[pos + 1] if pos + 1 < len(stream) else None
        expected_names = [f"{position_label(4 - j)}={by_id[h]['type']['name'] if h else PAD_NAME}" for j, h in enumerate(hid)]
        x = lh.encode_sparse(rows.ctx[i:i + 1])
        pipeline_names = [lh.feature_names()[c] for c in sorted(x.indices)]
        hist_idx = [by_id[h]["index"] for h in hid if h]
        causal = (all(a < b for a, b in zip(hist_idx, hist_idx[1:])) and nxt is not None
                  and hist_idx[-1] < nxt["index"] and nxt["id"] not in hid
                  and all(by_id[h]["period"] == by_id[eid]["period"] for h in hid if h))
        n_pad_expected = max(0, 4 - pos)
        out.append({"strata": ";".join(picked[i]), "event_id": eid, "match_id": mid, "period": int(period_of[eid]),
                    "seq_pos": int(rows.seq_pos[i]), "raw_stream_position": pos,
                    "history_pipeline": ",".join(map(str, rows.ctx[i])), "history_raw": ",".join(map(str, hist)),
                    "target_pipeline": int(rows.y[i]), "target_raw": target,
                    "target_event_id_pipeline": target_eid[eid], "target_event_id_raw": nxt["id"] if nxt else None,
                    "history_event_ids_raw": ";".join(hid),
                    "features_match": hist == rows.ctx[i].tolist(),
                    "one_hot_names_match": len(pipeline_names) == 5 and set(pipeline_names) == set(expected_names),
                    "n_active_columns": int(x.nnz),
                    "start_padding_correct": int((rows.ctx[i, :4] == PAD).sum()) == n_pad_expected
                                             and pos == int(rows.seq_pos[i]),
                    "target_match": target == int(rows.y[i]) and (nxt["id"] if nxt else None) == target_eid[eid],
                    "history_strictly_before_target": bool(causal)})
    return pd.DataFrame(out)


# ============================================================================ smoke test
def smoke(cfg: dict, rows: HistoryRows, folds, ids: np.ndarray) -> None:
    """Fold-0 INNER data only (outer-validation rows of fold 0 are never touched): one inner
    split, every C of the grid; timing, iterations, gradient, memory, sparse dimensions."""
    lr = cfg["logistic"]
    f = folds[0]
    inner = [g for g in folds if g.fold_id != f.fold_id]
    held = inner[0]
    tr = rows.mask([m for g in inner if g is not held for m in g.validation_match_ids])
    va = rows.mask(held.validation_match_ids)
    if set(rows.match_id[tr | va]) & set(f.validation_match_ids):
        raise LeakageError("smoke test touched outer-validation matches")
    xs = lh.encode_sparse(ids[tr])
    LOGGER.info("smoke: inner-train %d rows, sparse %s nnz %d (%.1f MB), inner-val %d rows",
                tr.sum(), xs.shape, xs.nnz, (xs.data.nbytes + xs.indices.nbytes + xs.indptr.nbytes) / 1e6, va.sum())
    LOGGER.info("smoke: inner-train class counts %s", np.bincount(rows.y[tr], minlength=K).tolist())
    report = []
    for C in lr["C_grid"]:
        t0 = time.perf_counter()
        m = lh.MLRH5Model(float(C), lr).fit(ids[tr], rows.y[tr])
        p = m.predict_proba(ids[va])
        xe.check_probabilities(p, float(cfg["evaluation"]["probability_tolerance"]))
        info = m.fit_info
        report.append({"C": C, "n_iter": info["n_iter"], "converged": info["converged"], "warnings": info["warnings"],
                       "max_abs_gradient": info["max_abs_gradient"], "n_train_cells": info["n_train_cells"],
                       "seconds": time.perf_counter() - t0, "proba_shape": list(p.shape),
                       "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6})
        LOGGER.info("smoke C=%g: %s", C, report[-1])
    per_fit = np.mean([r["seconds"] for r in report])
    n_fits = 5 * (4 * len(lr["C_grid"]) + 1)
    LOGGER.info("smoke: mean %.1f s per fit; full run >= %d fits -> about %.0f min (+ edge extensions, evaluation)",
                per_fit, n_fits, per_fit * n_fits / 60)
    out = PROJECT_ROOT / cfg["artefact_dir"] / "smoke_test.json"
    write_json(out, {"note": "technical smoke test; inner data of outer fold 0 only; not a result", "fits": report})


# ============================================================================ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg5 = yaml.safe_load((PROJECT_ROOT / cfg["experiment5_config"]).read_text(encoding="utf-8"))
    for key in ("bootstrap_replicates", "bootstrap_seed", "probability_tolerance", "permutation_repeats", "permutation_seed"):
        if cfg["evaluation"][key] != cfg5["evaluation"][key]:
            raise SystemExit(f"evaluation.{key} differs from experiment 5")
    if cfg["feature_set"] != {"name": NAME, "n_previous": lh.N_PREVIOUS, "n_columns": lh.N_COLUMNS}:
        raise SystemExit("feature_set in the configuration differs from the implementation")
    if cfg["logistic"]["class_weight"] is not None or float(cfg["logistic"]["l1_ratio"]) != 0.0:
        raise SystemExit("the primary experiment is unweighted and pure L2")
    out = PROJECT_ROOT / cfg["output_dir"]
    art = PROJECT_ROOT / cfg["artefact_dir"]
    ckpt = PROJECT_ROOT / cfg["checkpoint_dir"]
    if not args.smoke and out.exists() and any(out.iterdir()):
        raise SystemExit(f"{cfg['output_dir']} exists and is not empty; refusing to overwrite results")

    # ------------------------------------------------------------------ inputs (verified)
    hist_before = historical_snapshot(cfg)
    bad = {d: e["mismatches"] for d, e in hist_before["directories"].items() if e["mismatches"]}
    if bad:
        raise SystemExit(f"historical results changed before the run: {bad}")
    inp = load_inputs(cfg)        # canonical hashes vs manifests, Tier B set, folds == baseline folds, no final test
    tb, folds, part = inp["tb"], inp["folds"], inp["part"]
    rows = HistoryRows.from_table(tb)
    ids = rows.ctx                                   # (n, 5): E_{t-4}, ..., E_t  (XGB-H5 = rows.features("XGB-H5"))
    if not np.array_equal(ids, rows.features("XGB-H5")):
        raise RuntimeError("MLR-H5 ids differ from the XGB-H5 ids")
    if not np.array_equal(ids, history_ids(tb, 4)[tb["row_eligible"].to_numpy()]):
        raise RuntimeError("the two history construction paths disagree")
    brows = cb.Rows.from_table(tb)
    if not (np.array_equal(ids[:, 4], brows.cur) and np.array_equal(np.where(ids[:, 3] == PAD, -1, ids[:, 3]), brows.prev)
            and np.array_equal(rows.y, brows.y) and np.array_equal(rows.match_id, brows.match_id)):
        raise RuntimeError("(E_{t-1}, E_t, y) differ from the canonical baselines' rows")
    m = inp["manifest"]["counts_tier_b"]
    if len(rows.y) != m["eligible_rows"] or len(rows.y) != 690_147 or len(set(rows.match_id)) != 304:
        raise SystemExit(f"row count {len(rows.y)} / matches {len(set(rows.match_id))} differ from the manifest")
    LOGGER.info("%s: %d rows, %d matches, history ids %s", TASK_VERSION, len(rows.y), len(set(rows.match_id)), ids.shape)
    if args.smoke:
        smoke(cfg, rows, folds, ids)
        return

    integrity: list[dict] = []

    def check(name, ok, value=None, kind="PASS/FAIL"):
        integrity.append({"check": name, "result": ("PASS" if ok else "FAIL") if kind == "PASS/FAIL" else kind,
                          "value": value})

    fold_of_row = np.full(len(rows.y), -1)
    for f in folds:
        fold_of_row[rows.mask(f.validation_match_ids)] = f.fold_id
    art.mkdir(parents=True, exist_ok=True)
    prior, prior_checks = load_prior_predictions(cfg, rows, fold_of_row)
    integrity += prior_checks
    if any(c["result"] == "FAIL" for c in prior_checks):
        pd.DataFrame(integrity).to_csv(art / "alignment_failure.csv", index=False)
        raise SystemExit("row-level alignment with experiments 5 / 5B failed; investigate before training")

    # ------------------------------------------------------------------ feature verification (before training)
    sample = np.random.default_rng(0).choice(len(ids), 20_000, replace=False)
    dense, dense_names = encode(ids[sample])
    xs = lh.encode_sparse(ids)
    fver = {
        "n_columns": int(xs.shape[1]), "names_identical_to_xgb_h5_encoder": dense_names == lh.feature_names(),
        "sparse_equals_dense_encoder_on_20000_rows": bool(np.array_equal(xs[sample].toarray(), dense.astype(np.float64))),
        "nonzeros_per_row": sorted(set(np.diff(xs.indptr).tolist())),
        "e_t_never_start": bool((ids[:, 4] != PAD).all()),
        "start_count_by_lag": {position_label(lag): int((ids[:, 4 - lag] == PAD).sum()) for lag in range(1, 5)},
        "rows_with_seq_pos_below_lag": {position_label(lag): int((rows.seq_pos < lag).sum()) for lag in range(1, 5)},
        "column_support": dict(zip(lh.feature_names(), np.asarray(xs.sum(axis=0)).ravel().astype(int).tolist())),
    }
    fver["start_padding_iff_before_sequence_start"] = fver["start_count_by_lag"] == fver["rows_with_seq_pos_below_lag"]
    del xs, dense
    check("79 position-specific one-hot columns, names identical to XGB-H5", fver["n_columns"] == 79 and fver["names_identical_to_xgb_h5_encoder"])
    check("sparse encoder equals experiment 5 dense encoder (20,000 rows)", fver["sparse_equals_dense_encoder_on_20000_rows"])
    check("exactly 5 active columns per row; E_t never <START>", fver["nonzeros_per_row"] == [5] and fver["e_t_never_start"])
    check("<START> at lag l exactly when seq_pos < l", fver["start_padding_iff_before_sequence_start"])
    check("MLR-H5 ids identical to XGB-H5 ids; two construction paths agree; (E_{t-1}, E_t, y) = Markov-2 rows", True)
    rv = raw_verification(rows, tb, cfg["evaluation"]["raw_verification"])
    rv_cols = ["features_match", "one_hot_names_match", "start_padding_correct", "target_match", "history_strictly_before_target"]
    rv_ok = bool(rv[rv_cols].all().all()) and len(rv) >= 500
    rv.to_csv(art / "raw_feature_verification_pre_run.csv", index=False)
    strata_counts = rv["strata"].str.split(";").explode().value_counts().to_dict()
    check(f"raw-JSON rebuild of {len(rv)} rows (history, one-hot names, <START>, target, causality)", rv_ok,
          json.dumps(strata_counts))
    if not rv_ok:
        raise SystemExit("independent raw-JSON verification failed; see raw_feature_verification_pre_run.csv")
    LOGGER.info("feature verification passed (%d raw rows: %s)", len(rv), strata_counts)

    # ------------------------------------------------------------------ nested CV with checkpoints
    identifiers = {
        "experiment": cfg["experiment_name"], "task_version": TASK_VERSION, "config_sha256": sha256_file(CONFIG),
        "dataset_sha256_tier_b": inp["manifest"]["dataset_sha256_tier_b"],
        "folds_file_sha256": sha256_file(PROJECT_ROOT / inp["spec"].raw["partition"]["folds_file"]),
        "split_manifest_sha256": sha256_file(PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json"),
        "row_event_id_sha256": sha256_array(rows.event_id), "feature_matrix_sha256": sha256_array(ids),
        "target_sha256": sha256_array(rows.y), "model_source_digest": source_digest(MODEL_SOURCES)["digest"],
        "sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
    }
    lr = cfg["logistic"]
    trace: list = []
    res = lh.run_nested_cv(ids, rows.match_id, rows.y, folds, lr, ckpt, identifiers, trace, LOGGER.info)
    LOGGER.info("nested CV complete: computed folds %s, resumed folds %s", res.computed_folds, res.resumed_folds)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ integrity of the predictions
    check("fold of every row identical to experiments 5 / 5B", np.array_equal(res.fold_of_row, fold_of_row))
    check("690,147 rows and 304 matches", len(rows.y) == 690_147 and len(set(rows.match_id)) == 304, len(rows.y))
    check("every outer fold completed", sorted(c["fold_id"] for c in res.choices) == [0, 1, 2, 3, 4])
    val_counts = np.zeros(len(rows.y), dtype=int)
    for f in folds:
        val_counts[rows.mask(f.validation_match_ids)] += 1
    check("every row predicted exactly once (one validation fold per row)", bool((val_counts == 1).all()))
    try:
        xe.check_probabilities(res.proba, float(cfg["evaluation"]["probability_tolerance"]))
        prob_ok = bool((res.proba > 0).all())
    except ValueError:
        prob_ok = False
    check("probabilities finite, positive, 15 columns, rows sum to 1 (tol 1e-9)", prob_ok, float(np.abs(res.proba.sum(1) - 1).max()))
    fits = pd.DataFrame(res.fits)
    check("every fit converged without warnings and passed the independent gradient check",
          bool(fits["converged"].all() and fits["gradient_ok"].all()),
          f"{len(fits)} fits; max |gradient| {fits['max_abs_gradient'].max():.2e}; max n_iter {fits['n_iter'].max()}")
    check("no class absent from any training set", all(len(a) == 0 for a in fits["absent_classes"]))
    for fm in res.fold_metrics:
        sel = res.fold_of_row == fm["fold_id"]
        ll = float(-np.log(res.proba[sel][np.arange(sel.sum()), rows.y[sel]]).mean())
        check(f"fold {fm['fold_id']} log loss reproduces from assembled predictions", abs(ll - fm["log_loss"]) < 1e-12, abs(ll - fm["log_loss"]))
    for f in folds:
        for t in trace:
            if t["outer_fold"] == f.fold_id and t["purpose"] != "outer_validation_predict" and t["match_ids"] & set(f.validation_match_ids):
                raise LeakageError(f"fold {f.fold_id}: outer-validation match used for {t['purpose']}")
    check("no outer-validation match in any inner-train / inner-validation / outer-train set", True,
          f"{sum(t['purpose'] != 'outer_validation_predict' for t in trace)} data sets audited")
    final_test = set(part.final_test)
    check("no final-test match in the data, folds or any data set used", not (set(inp["table"]["match_id"]) & final_test)
          and not any(t["match_ids"] & final_test for t in trace) and len(final_test) == 83, f"{len(final_test)} final-test ids checked")
    # implementation-level check: NumPy softmax of the saved coefficients vs scikit-learn predict_proba
    dmax = 0.0
    for f in folds:
        va = rows.mask(f.validation_match_ids)
        dmax = max(dmax, float(np.abs(res.models[f.fold_id].predict_proba(ids[va]) - res.proba[va]).max()))
    check("probabilities recomputed from saved W, b (NumPy softmax) equal scikit-learn's", dmax < 1e-12, dmax)
    fz = folds[int(cfg["evaluation"]["row_level_check_fold"])]
    tr, va = rows.mask(fz.train_match_ids), rows.mask(fz.validation_match_ids)
    c0 = next(c["C"] for c in res.choices if c["fold_id"] == fz.fold_id)
    again = lh.MLRH5Model(c0, lr).fit(ids[tr], rows.y[tr]).predict_proba(ids[va])
    rep_diff = float(np.abs(again - res.proba[va]).max())
    check(f"refit of fold {fz.fold_id} reproduces the saved predictions", rep_diff < 1e-12,
          f"max |dp| = {rep_diff:.3g} (bit-identical: {bool(rep_diff == 0)})")
    t0 = time.perf_counter()
    rowm = lh.MLRH5Model(c0, lr).fit_rows(ids[tr], rows.y[tr])
    rowp = rowm.predict_proba(ids[va])
    yv = rows.y[va]
    llc = float(-np.log(res.proba[va][np.arange(va.sum()), yv]).mean())
    llr = float(-np.log(rowp[np.arange(va.sum()), yv]).mean())
    fitver = pd.DataFrame([{"fold_id": fz.fold_id, "C": c0, "train_rows": int(tr.sum()),
                            "train_cells": int(next(r["n_train_cells"] for r in res.fits if r["fold_id"] == fz.fold_id and r["stage"] == "outer_refit")),
                            "log_loss_cells": llc, "log_loss_rows": llr, "abs_log_loss_difference": abs(llc - llr),
                            "max_abs_probability_difference": float(np.abs(rowp - res.proba[va]).max()),
                            "argmax_agreement": float((rowp.argmax(1) == res.proba[va].argmax(1)).mean()),
                            "row_fit_n_iter": rowm.fit_info["n_iter"], "row_fit_max_abs_gradient": rowm.fit_info["max_abs_gradient"],
                            "row_level_fit_seconds": time.perf_counter() - t0}])
    fitver.to_csv(out / "fit_verification.csv", index=False)
    check("unaggregated row-level refit of fold 0 equals the weighted-cell fit (|dp| < 1e-6)",
          float(fitver["max_abs_probability_difference"].iat[0]) < 1e-6 and rowm.fit_info["converged"],
          f"|dLL| = {abs(llc - llr):.2e}, max |dp| = {fitver['max_abs_probability_difference'].iat[0]:.2e}")

    # ------------------------------------------------------------------ evaluation (8 models, same rows)
    proba = {**prior, NAME: res.proba}
    tables = xe.evaluate_models(rows.y, rows.match_id, res.fold_of_row, proba, PAIRS, cfg)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    mine = tables["model_comparison"].set_index("model")
    ref = pd.read_csv(PROJECT_ROOT / cfg["experiment5b_results_dir"] / "model_comparison.csv").set_index("model")
    recomp = []
    for mname in PRIOR_MODELS:
        cols = [c for c in ref.columns if c in mine.columns and pd.api.types.is_numeric_dtype(ref[c])]
        diffs = (mine.loc[mname, cols].astype(float) - ref.loc[mname, cols].astype(float)).abs()
        recomp.append({"model": mname, "n_metrics_compared": len(cols), "max_abs_difference": float(diffs.max()),
                       "worst_metric": str(diffs.idxmax()), "log_loss_recomputed": float(mine.at[mname, "log_loss"]),
                       "log_loss_stored": float(ref.at[mname, "log_loss"])})
    recomp = pd.DataFrame(recomp)
    recomp.to_csv(out / "baseline_recomputation_check.csv", index=False)
    check("all prior models' metrics and CIs recomputed from saved predictions equal experiment 5B's tables (1e-12)",
          bool((recomp["max_abs_difference"] < 1e-12).all()), float(recomp["max_abs_difference"].max()))

    # ------------------------------------------------------------------ descriptive analyses (after selection)
    strata = []
    for refm in ("XGB-H5", "markov2", "logistic_F0"):
        s = lh.nll_difference_strata(rows.y, res.proba, proba[refm], rows.seq_pos, ids[:, 4])
        s.insert(0, "reference", refm)
        s.insert(0, "model", NAME)
        strata.append(s)
    pd.concat(strata, ignore_index=True).to_csv(out / "nll_difference_by_stratum.csv", index=False)
    coef = lh.coefficient_tables(res.models, res.support, min_support=200)
    for name, frame in coef.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    xe.permutation_importance(_RowsView(rows, ids), _NestedView(res.models), folds, cfg).to_csv(
        out / "permutation_importance_mlr_h5.csv", index=False)
    pd.DataFrame(res.choices).to_csv(out / "hyperparameter_choices.csv", index=False)
    pd.DataFrame(res.inner_scores).to_csv(out / "inner_cv_scores.csv", index=False)
    fits.assign(warnings=fits["warnings"].map(json.dumps), absent_classes=fits["absent_classes"].map(json.dumps)).to_csv(
        out / "fit_log.csv", index=False)
    pd.DataFrame(res.timings).to_csv(out / "timings.csv", index=False)
    rv.to_csv(out / "raw_feature_verification.csv", index=False)
    write_json(out / "feature_verification.json", fver)
    counts = []
    for f in folds:
        trm, vam = rows.mask(f.train_match_ids), rows.mask(f.validation_match_ids)
        counts.append({"fold_id": f.fold_id, "train_matches": len(f.train_match_ids), "validation_matches": len(f.validation_match_ids),
                       "train_rows": int(trm.sum()), "validation_rows": int(vam.sum()),
                       **{f"train_n_{c}": int((rows.y[trm] == i).sum()) for i, c in enumerate(CLASSES)},
                       **{f"validation_n_{c}": int((rows.y[vam] == i).sum()) for i, c in enumerate(CLASSES)}})
    pd.DataFrame(counts).to_csv(out / "fold_counts.csv", index=False)
    pd.DataFrame([{**t, "n_matches": len(t["match_ids"]),
                   "touches_outer_validation": bool(t["match_ids"] & set(next(f for f in folds if f.fold_id == t["outer_fold"]).validation_match_ids)),
                   "touches_final_test": bool(t["match_ids"] & final_test),
                   "match_ids_sha256": hashlib.sha256(",".join(map(str, sorted(t["match_ids"]))).encode()).hexdigest()}
                  for t in trace]).drop(columns="match_ids").to_csv(out / "data_access_audit.csv", index=False)

    # ------------------------------------------------------------------ artefacts and provenance
    pred = pd.DataFrame({"event_id": rows.event_id, "match_id": rows.match_id, "fold_id": res.fold_of_row, "y": rows.y})
    for c in range(K):
        pred[f"{NAME}__p{c}"] = res.proba[:, c]
    pred_path = art / "oof_predictions_mlr_h5.parquet"
    tmp = pred_path.with_name("." + pred_path.name + ".tmp")
    pred.to_parquet(tmp, index=False)
    tmp.replace(pred_path)
    np.savez(art / "fold_models_mlr_h5.npz", **{f"coef_fold{k}": v.coef for k, v in res.models.items()},
             **{f"intercept_fold{k}": v.intercept for k, v in res.models.items()})
    shutil.copy(CONFIG, out / "experiment_config.yaml")
    shutil.copy(PROJECT_ROOT / cfg["task_config"], out / "task_config.yaml")
    _, digest = raw_source_hashes(part.development)
    check("raw source digest equals the canonical manifest", digest == inp["manifest"]["raw_source_digest_development"], digest)
    hist_after = historical_snapshot(cfg)
    check("historical results unchanged (canonical baselines, experiments 5 and 5B: checksums, README, predictions, dataset)",
          hist_after == hist_before and not any(e["mismatches"] for e in hist_after["directories"].values()),
          sum(e.get("checksums.json_files", 0) for e in hist_after["directories"].values()))
    check("run status", True, f"resumed folds: {res.resumed_folds or 'none'}; computed folds: {res.computed_folds}", kind="INFO")
    fe = Path(inp["spec"].raw["partition"]["folds_file"])
    write_json(out / "split_manifest.json", {
        "source_split_manifest": "results/markov_hmm_pilot/split_manifest.json",
        "source_split_manifest_sha256": identifiers["split_manifest_sha256"],
        "folds_file": fe.as_posix(), "folds_file_sha256": identifiers["folds_file_sha256"],
        "folds_identical_to_canonical_baselines_and_experiments_5_5b": True,
        "final_test_matches": len(part.final_test), "final_test_loaded": 0,
        "folds": [{"fold_id": f.fold_id, "train_match_ids": list(f.train_match_ids), "validation_match_ids": list(f.validation_match_ids),
                   "validation_rows": int(rows.mask(f.validation_match_ids).sum())} for f in folds]})
    write_json(out / "task_manifest.json", {**inp["manifest"], "raw_source_digest_recomputed": digest,
                                            "row_identity": {"n_rows": int(len(rows.y)), "n_matches": int(len(set(rows.match_id))),
                                                             "event_id_sha256": identifiers["row_event_id_sha256"],
                                                             "target_sha256": identifiers["target_sha256"],
                                                             "fold_of_row_sha256": sha256_array(res.fold_of_row),
                                                             "feature_matrix_sha256": identifiers["feature_matrix_sha256"]}})
    env = environment_info()
    env.pop("platform", None)
    try:
        from threadpoolctl import threadpool_info
        env["blas"] = [{k: t.get(k) for k in ("internal_api", "version", "num_threads")} for t in threadpool_info()]
    except Exception:  # pragma: no cover
        env["blas"] = "unavailable"
    write_json(out / "environment.json", {
        **env, "runtime_seconds": time.time() - started, "checkpoint_identifiers": identifiers,
        "source_files_sha256": source_digest(ALL_SOURCES)["files"],
        "artefacts_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in
                             (pred_path, art / "fold_models_mlr_h5.npz")},
        "run_status": {"completed": True, "resumed_folds": res.resumed_folds, "computed_folds": res.computed_folds},
        "workload": {"n_fits": len(fits), "total_fit_seconds": float(fits["fit_seconds"].sum())},
        "historical_verification": {"before": hist_before, "after": hist_after},
        "random_seeds": {"folds": 20261027, "bootstrap": cfg["evaluation"]["bootstrap_seed"],
                         "permutation": cfg["evaluation"]["permutation_seed"],
                         "raw_verification_sample": cfg["evaluation"]["raw_verification"]["seed"],
                         "solver": "none (newton-cholesky is deterministic)"},
        "commands": ["python scripts/run_logistic_h5_experiment6a.py --smoke",
                     "python scripts/run_logistic_h5_experiment6a.py",
                     "python scripts/verify_logistic_h5_experiment6a.py"]})
    pd.DataFrame(integrity).to_csv(out / "integrity_audit.csv", index=False)
    write_json(out / "checksums.json", {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*"))
                                        if p.is_file() and p.name not in ("checksums.json", "README.md")})
    n_fail = sum(r["result"] == "FAIL" for r in integrity)
    LOGGER.info("integrity audit: %d checks, %d FAIL", len(integrity), n_fail)
    LOGGER.info("pooled log loss: %s", {k: round(float(v), 6) for k, v in mine["log_loss"].items()})
    LOGGER.info("done in %.0f s -> %s", time.time() - started, cfg["output_dir"])
    if n_fail:
        raise SystemExit("integrity audit FAILED - see integrity_audit.csv")


class _RowsView:
    """Adapter so experiment 5's permutation_importance scores MLR-H5 on the same 5 positions."""

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
    main()
