r"""Experiment 5B: XGB-H2, a controlled comparison with the canonical second-order Markov chain.

Both XGB-H2 and Markov 2 condition on :math:`(E_{t-1}, E_t)` and predict
:math:`P(E_{t+1} = c \mid E_{t-1}, E_t)` over the K = 15 classes, on the same 690,147 Tier B
rows and the same outer folds.

Reuse. Experiment 5's validated components are used unchanged:
- ``event_history``: history construction, the one-hot encoding with ``<START>``, cells;
- ``xgboost_experiment``: the model, candidate list, booster parameters, the lockstep inner curve,
  probability checks and evaluation.

This module adds only:

- ``run_nested_cv_checkpointed``: the same nested-CV procedure as
  ``xgboost_experiment.run_nested_cv`` (tested to give identical choices and predictions), but
  for an explicit feature matrix, and with atomic per-outer-fold checkpoints that are resumed only
  when every identifier matches;
- ``weighted_cell_audit``: row-level vs. count-weighted-cell fits on inner-split data;
- the feature report and integrity helpers.

Missing-history treatment differs between the two models and is disclosed, not changed:

- Markov 2: at :math:`t = 0`, and for contexts with fewer than 20 training rows, the prediction
  backs off to Markov 1;
- XGB-H2: :math:`E_{t-1} = \text{<START>}` is an ordinary 16th category of the
  :math:`E_{t-1}` block, and rare contexts are pooled by the tree splits.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .canonical_task import CLASSES, K
from .event_history import PAD, PAD_NAME, aggregate_cells, encode
from .splits import FoldAssignment, LeakageError
from . import xgboost_experiment as xe

FEATURE_SET = "XGB-H2"
N_PREVIOUS = 1


class CheckpointMismatch(RuntimeError):
    """A checkpoint exists but was produced under different identifiers."""


# ============================================================================ identifiers
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.asarray(a)
    if a.dtype == object:
        return sha256_bytes("\n".join(map(str, a)).encode())
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


def feature_schema_hash(names: list[str]) -> str:
    return sha256_bytes("\n".join(names).encode())


def source_digest(paths: list[Path]) -> dict:
    files = {p.name: sha256_bytes(Path(p).read_bytes()) for p in paths}
    return {"files": files, "digest": sha256_bytes(json.dumps(files, sort_keys=True).encode())}


# ============================================================================ atomic checkpoint I/O
def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def save_fold_checkpoint(ckpt_dir: Path, fold_id: int, identifiers: dict, record: dict, val_index: np.ndarray,
                         proba: np.ndarray, booster: xgb.Booster) -> None:
    """Arrays and model first; the JSON (which holds their hashes) last = commit marker."""
    arr = _npz_bytes(val_index=val_index.astype(np.int64), proba=proba.astype(np.float64))
    model = bytes(booster.save_raw(raw_format="ubj"))
    atomic_write_bytes(ckpt_dir / f"fold{fold_id}.npz", arr)
    atomic_write_bytes(ckpt_dir / f"fold{fold_id}.ubj", model)
    meta = {"identifiers": identifiers, "fold_id": fold_id, "npz_sha256": sha256_bytes(arr),
            "model_sha256": sha256_bytes(model), **record}
    atomic_write_bytes(ckpt_dir / f"fold{fold_id}.json", json.dumps(meta, indent=1, default=_json_default).encode())


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def load_fold_checkpoint(ckpt_dir: Path, fold_id: int, identifiers: dict):
    """None if no committed checkpoint; raises CheckpointMismatch if it is incompatible or corrupt."""
    jpath = ckpt_dir / f"fold{fold_id}.json"
    if not jpath.exists():
        return None
    meta = json.loads(jpath.read_text(encoding="utf-8"))
    if meta.get("identifiers") != identifiers:
        diff = sorted(k for k in set(identifiers) | set(meta.get("identifiers", {}))
                      if identifiers.get(k) != meta.get("identifiers", {}).get(k))
        raise CheckpointMismatch(f"fold {fold_id}: checkpoint identifiers differ ({diff}); refusing to mix runs")
    arr = (ckpt_dir / f"fold{fold_id}.npz").read_bytes()
    model = (ckpt_dir / f"fold{fold_id}.ubj").read_bytes()
    if sha256_bytes(arr) != meta["npz_sha256"] or sha256_bytes(model) != meta["model_sha256"]:
        raise CheckpointMismatch(f"fold {fold_id}: checkpoint files do not match their recorded hashes")
    z = np.load(io.BytesIO(arr))
    booster = xgb.Booster()
    booster.load_model(bytearray(model))
    return meta, z["val_index"], z["proba"], booster


# ============================================================================ nested CV
@dataclass
class NestedResult:
    feature_set: str
    proba: np.ndarray
    fold_of_row: np.ndarray
    models: dict[int, xe.XGBNextEventModel] = field(default_factory=dict)
    choices: list[dict] = field(default_factory=list)
    curves: list[dict] = field(default_factory=list)
    timings: list[dict] = field(default_factory=list)
    fold_metrics: list[dict] = field(default_factory=list)
    resumed_folds: list[int] = field(default_factory=list)
    computed_folds: list[int] = field(default_factory=list)
    n_fits: int = 0
    n_boosting_rounds: int = 0


def run_nested_cv_checkpointed(ids_all: np.ndarray, match_id: np.ndarray, y: np.ndarray,
                               folds: list[FoldAssignment], cfg: dict, feature_set: str,
                               ckpt_dir: Path | None = None, identifiers: dict | None = None,
                               trace: list | None = None, log=None) -> NestedResult:
    """Nested CV identical to ``xgboost_experiment.run_nested_cv`` for features ``ids_all``.

    With ``ckpt_dir``, each completed outer fold is committed atomically. A committed fold is
    loaded instead of recomputed only if ``identifiers`` match exactly (else CheckpointMismatch).
    Checkpoints never influence a choice: a loaded fold carries the choice its own run made."""
    n = len(y)
    mask = lambda ids_: np.isin(match_id, list(ids_))  # noqa: E731
    fold_of_row = np.full(n, -1)
    for f in folds:
        fold_of_row[mask(f.validation_match_ids)] = f.fold_id
    if (fold_of_row < 0).any():
        raise LeakageError("an eligible row belongs to no validation fold")
    out = NestedResult(feature_set, np.full((n, K), np.nan), fold_of_row)
    cands = xe.candidate_list(cfg)
    s = cfg["search"]

    def dm(match_ids, purpose, fold_id):
        m = mask(match_ids)
        if trace is not None:
            trace.append({"feature_set": feature_set, "outer_fold": fold_id, "purpose": purpose,
                          "match_ids": frozenset(int(x) for x in np.unique(match_id[m]))})
        return xe.cell_dmatrix(ids_all[m], y[m])

    for f in folds:
        outer_val = set(f.validation_match_ids)
        if set(f.train_match_ids) & outer_val:
            raise LeakageError(f"fold {f.fold_id}: match in training and validation")
        va = mask(f.validation_match_ids)
        if ckpt_dir is not None:
            loaded = load_fold_checkpoint(ckpt_dir, f.fold_id, identifiers)
            if loaded is not None:
                meta, val_index, proba, booster = loaded
                if not np.array_equal(val_index, np.flatnonzero(va)):
                    raise CheckpointMismatch(f"fold {f.fold_id}: checkpoint rows differ from this fold's rows")
                out.proba[va] = proba
                model = xe.XGBNextEventModel(xe.booster_params(cfg, cands[meta["choice"]["candidate"]]),
                                             meta["choice"]["n_estimators"], booster)
                out.models[f.fold_id] = model
                out.choices.append(meta["choice"])
                out.curves += meta["curves"]
                out.timings.append(meta["timing"])
                out.fold_metrics.append(meta["fold_metrics"])
                out.n_fits += meta["n_fits"]
                out.n_boosting_rounds += meta["n_boosting_rounds"]
                if trace is not None:
                    trace += [{**t, "match_ids": frozenset(t["match_ids"])} for t in meta["trace"]]
                out.resumed_folds.append(f.fold_id)
                if log:
                    log(f"{feature_set} fold {f.fold_id}: resumed from checkpoint")
                continue
        fold_trace_start = len(trace) if trace is not None else 0
        fits, rounds = 0, 0
        inner_folds = [g for g in folds if g.fold_id != f.fold_id]
        start = time.perf_counter()
        inner = []
        for held in inner_folds:
            tr_ids = [m for g in inner_folds if g is not held for m in g.validation_match_ids]
            if (set(tr_ids) | set(held.validation_match_ids)) & outer_val:
                raise LeakageError(f"fold {f.fold_id}: outer-validation match in an inner set")
            dtr, _ = dm(tr_ids, "inner_train", f.fold_id)
            dva, n_va = dm(held.validation_match_ids, "inner_validation", f.fold_id)
            inner.append((dtr, dva, n_va))
        scores, curves = [], []
        for ci, cand in enumerate(cands):
            t0 = time.perf_counter()
            curve = xe.lockstep_curve(xe.booster_params(cfg, cand), inner, int(s["max_rounds"]), int(s["patience"]))
            r = int(np.argmin(curve)) + 1
            scores.append((round(float(curve[r - 1]), 12), ci, r))
            fits += len(inner)
            rounds += len(curve) * len(inner)
            curves.append({"feature_set": feature_set, "fold_id": f.fold_id, "candidate": ci, **cand,
                           "best_rounds": r, "rounds_trained": len(curve), "inner_log_loss": float(curve[r - 1]),
                           "hit_max_rounds": len(curve) == int(s["max_rounds"]),
                           "seconds": time.perf_counter() - t0, "curve": json.dumps([round(v, 8) for v in curve])})
            if log:
                log(f"{feature_set} fold {f.fold_id} cand {ci} {cand} -> rounds {r}/{len(curve)} "
                    f"inner LL {curve[r - 1]:.6f} ({time.perf_counter() - t0:.0f}s)")
        best_score, best_ci, best_r = min(scores)
        selection = time.perf_counter() - start
        t0 = time.perf_counter()
        dm(f.train_match_ids, "outer_train", f.fold_id)          # recorded for the audit
        tr = mask(f.train_match_ids)
        model = xe.XGBNextEventModel(xe.booster_params(cfg, cands[best_ci]), best_r).fit(ids_all[tr], y[tr])
        fits += 1
        rounds += best_r
        if trace is not None:
            trace.append({"feature_set": feature_set, "outer_fold": f.fold_id, "purpose": "outer_validation_predict",
                          "match_ids": frozenset(int(x) for x in np.unique(match_id[va]))})
        p = model.predict_proba(ids_all[va])
        out.proba[va] = p
        out.models[f.fold_id] = model
        choice = {"feature_set": feature_set, "fold_id": f.fold_id, "candidate": best_ci, **cands[best_ci],
                  "n_estimators": best_r, "inner_log_loss": best_score, "rounds_at_max": best_r == int(s["max_rounds"]),
                  "train_rows": int(tr.sum()), "validation_rows": int(va.sum())}
        timing = {"feature_set": feature_set, "fold_id": f.fold_id, "selection_seconds": selection,
                  "final_fit_predict_seconds": time.perf_counter() - t0}
        yv = y[va]
        fm = {"fold_id": f.fold_id, "n_rows": int(va.sum()),
              "log_loss": float(-np.log(p[np.arange(len(yv)), yv]).mean()),
              "accuracy": float((p.argmax(1) == yv).mean())}
        out.choices.append(choice)
        out.curves += curves
        out.timings.append(timing)
        out.fold_metrics.append(fm)
        out.n_fits += fits
        out.n_boosting_rounds += rounds
        out.computed_folds.append(f.fold_id)
        if ckpt_dir is not None:
            fold_trace = [{**t, "match_ids": sorted(t["match_ids"])} for t in (trace or [])[fold_trace_start:]]
            save_fold_checkpoint(ckpt_dir, f.fold_id, identifiers,
                                 {"choice": choice, "curves": curves, "timing": timing, "fold_metrics": fm,
                                  "n_fits": fits, "n_boosting_rounds": rounds, "trace": fold_trace,
                                  "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                 np.flatnonzero(va), p, model.booster)
        if log:
            log(f"{feature_set} fold {f.fold_id} done: cand {best_ci}, M={best_r}, outer LL {fm['log_loss']:.6f}")
    if np.isnan(out.proba).any():
        raise RuntimeError(f"{feature_set}: rows without prediction")
    out.choices.sort(key=lambda c: c["fold_id"])
    out.fold_metrics.sort(key=lambda c: c["fold_id"])
    return out


# ============================================================================ weighted-cell audit
def _row_dmatrix(ids: np.ndarray, y: np.ndarray) -> xgb.DMatrix:
    x, names = encode(ids)
    return xgb.DMatrix(x, label=y, feature_names=[xe._safe(nm) for nm in names])


def _cell_dmatrix_weights(ids, y, scale=1.0):
    cx, cy, w, _ = aggregate_cells(ids, y)
    x, names = encode(cx)
    return xgb.DMatrix(x, label=cy, weight=w * scale, feature_names=[xe._safe(nm) for nm in names]), w


def _proba(booster: xgb.Booster, dm: xgb.DMatrix) -> np.ndarray:
    z = np.asarray(booster.predict(dm, output_margin=True)).reshape(dm.num_row(), -1)
    return xe.softmax64(z)


def weighted_cell_audit(ids: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, cfg: dict,
                        candidates: list[int], rounds: list[int]) -> dict[str, pd.DataFrame]:
    """Row-level vs. count-weighted-cell fits with identical parameters, seed and rounds.

    ``tr`` / ``va``: boolean masks of INNER training / INNER validation rows."""
    cands = xe.candidate_list(cfg)
    out_div, out_struct, out_checks = [], [], []
    d_row, d_val = _row_dmatrix(ids[tr], y[tr]), _row_dmatrix(ids[va], y[va])
    d_cell, w = _cell_dmatrix_weights(ids[tr], y[tr])
    yv = y[va]
    out_checks.append({"check": "cell weights sum to the number of training rows",
                       "value": float(w.sum()), "expected": float(tr.sum()), "pass": bool(w.sum() == tr.sum())})
    cx, cy, _, inv = aggregate_cells(ids[tr], y[tr])
    same_x_diff_y = int(len(cx) - len(np.unique(cx, axis=0)))
    out_checks.append({"check": "cells with identical features but different targets are kept separate",
                       "value": same_x_diff_y, "expected": "> 0 (such cells exist)", "pass": bool(same_x_diff_y > 0)})
    out_checks.append({"check": "every row maps back to its own (features, target) cell",
                       "value": bool(np.array_equal(cx[inv], ids[tr]) and np.array_equal(cy[inv], y[tr])),
                       "expected": True, "pass": bool(np.array_equal(cx[inv], ids[tr]) and np.array_equal(cy[inv], y[tr]))})
    for ci in candidates:
        params = xe.booster_params(cfg, cands[ci])
        b_row = xgb.Booster(params, [d_row])
        b_cell = xgb.Booster(params, [d_cell])
        t_row = t_cell = 0.0
        for it in range(max(rounds)):
            t0 = time.perf_counter(); b_row.update(d_row, it); t_row += time.perf_counter() - t0
            t0 = time.perf_counter(); b_cell.update(d_cell, it); t_cell += time.perf_counter() - t0
            r = it + 1
            if r == 1:
                cr, cc = json.loads(b_row.save_config()), json.loads(b_cell.save_config())
                bs_r = cr["learner"]["learner_model_param"]["base_score"]
                bs_c = cc["learner"]["learner_model_param"]["base_score"]
                dr = b_row.get_dump(dump_format="json")
                dc = b_cell.get_dump(dump_format="json")
                lr_, lc_ = _leaves(dr), _leaves(dc)
                out_struct.append({"candidate": ci, "base_score_identical": bs_r == bs_c,
                                   "round1_tree_structure_identical": [_structure(t) for t in dr] == [_structure(t) for t in dc],
                                   "round1_max_abs_leaf_difference": float(np.max(np.abs(lr_ - lc_))) if len(lr_) == len(lc_) else np.nan,
                                   "round1_n_trees": len(dr)})
            if r in rounds:
                pr, pc = _proba(b_row, d_val), _proba(b_cell, d_val)
                llr = float(-np.log(pr[np.arange(len(yv)), yv]).mean())
                llc = float(-np.log(pc[np.arange(len(yv)), yv]).mean())
                same = [_structure(a) == _structure(b) for a, b in zip(b_row.get_dump(dump_format="json"),
                                                                       b_cell.get_dump(dump_format="json"))]
                first_diff = next((i // K + 1 for i, s_ in enumerate(same) if not s_), None)
                out_div.append({"candidate": ci, **cands[ci], "rounds": r,
                                "inner_val_log_loss_rows": llr, "inner_val_log_loss_cells": llc,
                                "log_loss_difference_cells_minus_rows": llc - llr,
                                "max_abs_probability_difference": float(np.abs(pr - pc).max()),
                                "mean_abs_probability_difference": float(np.abs(pr - pc).mean()),
                                "argmax_agreement": float((pr.argmax(1) == pc.argmax(1)).mean()),
                                "first_round_with_different_tree_structure": first_diff,
                                "row_fit_seconds_cumulative": t_row, "cell_fit_seconds_cumulative": t_cell})
    # weights are used as given (no normalisation): doubling them changes the fit when lambda > 0
    params = xe.booster_params(cfg, {**cands[candidates[0]], "reg_lambda": 10.0, "min_child_weight": 1})
    d2, _ = _cell_dmatrix_weights(ids[tr], y[tr], 2.0)
    p1 = _proba(xgb.train(params, d_cell, 5), d_val)
    p2 = _proba(xgb.train(params, d2, 5), d_val)
    out_checks.append({"check": "XGBoost does not normalise instance weights (2x weights changes the fit, lambda=10)",
                       "value": float(np.abs(p1 - p2).max()), "expected": "> 0", "pass": bool(np.abs(p1 - p2).max() > 0)})
    # repeatability of both representations
    for name, dmat in (("rows", d_row), ("cells", d_cell)):
        params = xe.booster_params(cfg, cands[candidates[0]])
        a = _proba(xgb.train(params, dmat, 30), d_val)
        b = _proba(xgb.train(params, dmat, 30), d_val)
        out_checks.append({"check": f"{name}: two fits with the same seed are bit-identical (30 rounds)",
                           "value": bool(np.array_equal(a, b)), "expected": True, "pass": bool(np.array_equal(a, b))})
    perm = np.random.default_rng(0).permutation(int(tr.sum()))
    params = xe.booster_params(cfg, cands[candidates[0]])
    a = _proba(xgb.train(params, d_row, 30), d_val)
    b = _proba(xgb.train(params, _row_dmatrix(ids[tr][perm], y[tr][perm]), 30), d_val)
    out_checks.append({"check": "rows: a fit on permuted row order is bit-identical (30 rounds)",
                       "value": float(np.abs(a - b).max()), "expected": 0.0, "pass": bool(np.array_equal(a, b))})
    return {"divergence": pd.DataFrame(out_div), "round1_structure": pd.DataFrame(out_struct),
            "checks": pd.DataFrame(out_checks)}


def _walk(node, out):
    if "leaf" in node:
        out.append(("leaf",))
    else:
        out.append((node["split"], node.get("split_condition")))
        for ch in node["children"]:
            _walk(ch, out)
    return out


def _structure(tree_json: str):
    return tuple(_walk(json.loads(tree_json), []))


def _leaves(dump: list[str]) -> np.ndarray:
    vals = []

    def walk(n):
        if "leaf" in n:
            vals.append(n["leaf"])
        else:
            for ch in n["children"]:
                walk(ch)
    for t in dump:
        walk(json.loads(t))
    return np.asarray(vals, dtype=np.float64)


def lockstep_selection_audit(ids, y, match_id, folds, outer_fold, cfg, candidates) -> pd.DataFrame:
    """Inner-CV round selection with row-level vs. cell data on one outer fold's inner splits."""
    cands = xe.candidate_list(cfg)
    f = next(g for g in folds if g.fold_id == outer_fold)
    inner_folds = [g for g in folds if g.fold_id != f.fold_id]
    sets = {"rows": [], "cells": []}
    for held in inner_folds:
        trm = np.isin(match_id, [m for g in inner_folds if g is not held for m in g.validation_match_ids])
        vam = np.isin(match_id, list(held.validation_match_ids))
        if set(match_id[trm | vam]) & set(f.validation_match_ids):
            raise LeakageError("audit touched outer-validation matches")
        sets["rows"].append((_row_dmatrix(ids[trm], y[trm]), _row_dmatrix(ids[vam], y[vam]), int(vam.sum())))
        dtr, _ = xe.cell_dmatrix(ids[trm], y[trm])
        dva, nva = xe.cell_dmatrix(ids[vam], y[vam])
        sets["cells"].append((dtr, dva, nva))
    out = []
    s = cfg["search"]
    for ci in candidates:
        res = {}
        for kind, inner in sets.items():
            t0 = time.perf_counter()
            c = xe.lockstep_curve(xe.booster_params(cfg, cands[ci]), inner, int(s["max_rounds"]), int(s["patience"]))
            res[kind] = (c, time.perf_counter() - t0)
        cr, cc = res["rows"][0], res["cells"][0]
        m = min(len(cr), len(cc))
        out.append({"outer_fold": outer_fold, "candidate": ci, **cands[ci],
                    "selected_rounds_rows": int(np.argmin(cr)) + 1, "selected_rounds_cells": int(np.argmin(cc)) + 1,
                    "rounds_trained_rows": len(cr), "rounds_trained_cells": len(cc),
                    "min_inner_log_loss_rows": float(cr.min()), "min_inner_log_loss_cells": float(cc.min()),
                    "max_abs_curve_difference_common_rounds": float(np.abs(cr[:m] - cc[:m]).max()),
                    "seconds_rows": res["rows"][1], "seconds_cells": res["cells"][1]})
    return pd.DataFrame(out)


# ============================================================================ feature report
def feature_report(ids: np.ndarray, y: np.ndarray) -> tuple[dict, pd.DataFrame]:
    x, names = encode(ids)
    prev, cur = ids[:, 0], ids[:, 1]
    counts = []
    for c in range(K + 1):
        counts.append({"class_id": c, "class": CLASSES[c] if c < K else PAD_NAME,
                       "E_t_count": int((cur == c).sum()), "E_{t-1}_count": int((prev == c).sum()),
                       "target_count": int((y == c).sum()) if c < K else 0})
    contexts = np.unique(ids, axis=0)
    report = {
        "feature_set": FEATURE_SET, "n_rows": int(len(y)), "matrix_shape": list(x.shape), "dtype": str(x.dtype),
        "encoding": f"E_{{t-1}}: one-hot over 16 categories (15 classes, then {PAD_NAME} = id {PAD}); "
                    "E_t: one-hot over the 15 classes; columns ordered E_{t-1} block then E_t block",
        "feature_names": names, "feature_schema_sha256": feature_schema_hash(names),
        "missing_history_rows": int((prev == PAD).sum()),
        "unique_two_event_contexts_observed": int(len(contexts)),
        "possible_contexts": (K + 1) * K,
        "contexts_with_start_token": int((contexts[:, 0] == PAD).sum()),
        "row_sums_of_one_hot_all_2": bool(np.all(x.sum(axis=1) == 2)),
        "feature_matrix_sha256": sha256_bytes(np.ascontiguousarray(x).tobytes()),
    }
    return report, pd.DataFrame(counts)


def markov2_backoff_rates(cur: np.ndarray, prev_or_minus1: np.ndarray, match_id: np.ndarray,
                          folds: list[FoldAssignment], min_count: int) -> pd.DataFrame:
    """Share of outer-validation rows for which the canonical Markov 2 backs off to Markov 1."""
    out = []
    for f in folds:
        tr = np.isin(match_id, list(f.train_match_ids))
        va = np.isin(match_id, list(f.validation_match_ids))
        ok = tr & (prev_or_minus1 >= 0)
        ctx = np.zeros((K, K))
        np.add.at(ctx, (prev_or_minus1[ok], cur[ok]), 1)
        pv, cv = prev_or_minus1[va], cur[va]
        start = pv < 0
        rare = ~start & (ctx[np.maximum(pv, 0), cv] < min_count)
        out.append({"fold_id": f.fold_id, "validation_rows": int(va.sum()), "backoff_period_start": int(start.sum()),
                    "backoff_rare_context": int(rare.sum()), "backoff_share": float((start | rare).mean()),
                    "training_contexts_below_min_count": int(((ctx > 0) & (ctx < min_count)).sum()),
                    "training_contexts_observed": int((ctx > 0).sum())})
    return pd.DataFrame(out)


__all__ = ["FEATURE_SET", "N_PREVIOUS", "run_nested_cv_checkpointed", "weighted_cell_audit",
           "lockstep_selection_audit", "feature_report", "markov2_backoff_rates", "CheckpointMismatch",
           "load_fold_checkpoint", "save_fold_checkpoint", "sha256_array", "source_digest"]
