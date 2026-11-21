"""Tests for experiment 5B (XGB-H2 vs. Markov 2 on actions15_360current_v1).

A data/target correctness, B evaluation consistency and isolation, C model and checkpoint
correctness, D metrics against hand-computed values (not against the implementation itself).
Real-data checks run only where the local datasets exist (they are not in the repository)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.sequence_models import canonical_baselines as cb
from src.sequence_models import xgboost_experiment as xe
from src.sequence_models import xgboost_h2_experiment as h2
from src.sequence_models.canonical_task import CLASSES, build_canonical_table
from src.sequence_models.evaluation_v2 import (
    confusion_counts,
    metrics_from_confusion,
    per_class_from_confusion,
    probability_metrics,
)
from src.sequence_models.event_history import PAD, HistoryRows, encode, history_ids
from src.sequence_models.splits import FoldAssignment, LeakageError
from src.utils.paths import PROJECT_ROOT

C = {c: i for i, c in enumerate(CLASSES)}
CFG_PATH = PROJECT_ROOT / "configs/xgboost_experiment5b_h2_v1.yaml"


def raw_records(match_id, records, period=1, start=0):
    rows = []
    for j, (typ, t, h360) in enumerate(records):
        i = start + j
        rows.append({"match_id": match_id, "list_position": i, "index": i + 1, "event_id": f"{match_id}-{i}",
                     "period": period, "timestamp": f"00:00:{t:06.3f}", "t_seconds": float(t), "type": typ,
                     "has_360": h360, "has_freeze_frame": h360, "subtype": None, "outcome": None, "related_events": ""})
    return rows


def R(typ, t=0.0, h360=True):
    return (typ, t, h360)


def h2_ids(table: pd.DataFrame) -> tuple[HistoryRows, np.ndarray]:
    rows = HistoryRows.from_table(table)
    return rows, rows.ctx[:, 3:]


def random_table(n_matches=10, n_events=150, seed=0, n_types=5):
    rng = np.random.default_rng(seed)
    recs = []
    for m in range(n_matches):
        recs += raw_records(m, [R(CLASSES[int(rng.integers(0, n_types))], float(i), bool(rng.random() > 0.2))
                                for i in range(n_events)])
    return build_canonical_table(pd.DataFrame(recs))


SMALL = {
    "xgboost": {"objective": "multi:softprob", "num_class": 15, "tree_method": "hist", "eval_metric": "mlogloss",
                "nthread": 2, "seed": 1, "fixed_params": {"subsample": 1.0, "gamma": 0.0, "reg_alpha": 0.0, "max_bin": 256}},
    "search": {"n_candidates": 2, "candidate_seed": 3, "max_rounds": 12, "patience": 3,
               "space": {"max_depth": [2, 3], "learning_rate": [0.3], "min_child_weight": [1],
                         "reg_lambda": [1.0], "colsample_bytree": [1.0]}},
    "evaluation": {"bootstrap_replicates": 10, "bootstrap_seed": 1, "probability_tolerance": 1e-9,
                   "permutation_repeats": 2, "permutation_seed": 2},
}


def folds_for(ids, k=5):
    return [FoldAssignment(f, tuple(m for m in ids if m % k != f), tuple(m for m in ids if m % k == f)) for f in range(k)]


# ============================================================================ A. data and targets
def test_taxonomy_order_and_h2_encoding():
    assert len(CLASSES) == 15 and CLASSES.index("Shot") == 5 and PAD == 15
    x, names = encode(np.array([[PAD, C["Pass"]], [C["Pressure"], C["Shot"]]]))
    assert x.shape == (2, 31) and len(names) == 31
    assert names[:16] == [f"E_{{t-1}}={c}" for c in CLASSES] + ["E_{t-1}=<START>"]
    assert names[16:] == [f"E_t={c}" for c in CLASSES]
    assert [names[j] for j in np.flatnonzero(x[0])] == ["E_{t-1}=<START>", "E_t=Pass"]
    assert [names[j] for j in np.flatnonzero(x[1])] == ["E_{t-1}=Pressure", "E_t=Shot"]
    assert all(n.startswith(("E_t=", "E_{t-1}=")) for n in names)      # no ids, teams, times or locations


def test_current_previous_and_target_alignment_on_known_sequence():
    seq = ["Pass", "Carry", "Pressure", "Pass", "Shot"]
    rows, ids = h2_ids(build_canonical_table(pd.DataFrame(raw_records(1, [R(s, i) for i, s in enumerate(seq)]))))
    assert ids.tolist() == [[PAD, C["Pass"]], [C["Pass"], C["Carry"]], [C["Carry"], C["Pressure"]], [C["Pressure"], C["Pass"]]]
    assert rows.y.tolist() == [C["Carry"], C["Pressure"], C["Pass"], C["Shot"]]   # y_t = E_{t+1}


def test_first_event_of_period_and_no_history_across_period_match_or_shootout():
    p1 = raw_records(1, [R("Pass", 1), R("Carry", 2)], period=1)
    p2 = raw_records(1, [R("Shot", 1), R("Pass", 2)], period=2, start=2)
    p5 = raw_records(1, [R("Shot", 1), R("Shot", 2)], period=5, start=4)
    m2 = raw_records(2, [R("Dribble", 1), R("Pass", 2)])
    t = build_canonical_table(pd.DataFrame(p1 + p2 + p5 + m2))
    rows, ids = h2_ids(t)
    assert 5 not in set(t["period"])
    assert ids.tolist() == [[PAD, C["Pass"]], [PAD, C["Shot"]], [PAD, C["Dribble"]]]    # every first row padded
    assert rows.y.tolist() == [C["Carry"], C["Pass"], C["Pass"]]                         # targets stay in period
    assert "1-4" not in set(rows.event_id) and "1-5" not in set(rows.event_id)


def test_previous_event_without_360_is_history_and_matches_markov2_context():
    t = build_canonical_table(pd.DataFrame(raw_records(1, [R("Pass", 1, False), R("Carry", 2), R("Shot", 3)])))
    rows, ids = h2_ids(t)
    assert ids.tolist() == [[C["Pass"], C["Carry"]]]
    t = random_table()
    rows, ids = h2_ids(t)
    br = cb.Rows.from_table(t)
    assert np.array_equal(ids[:, 1], br.cur) and np.array_equal(np.where(ids[:, 0] == PAD, -1, ids[:, 0]), br.prev)
    assert np.array_equal(ids, history_ids(t, 1)[t["row_eligible"].to_numpy()])        # two construction paths


def test_no_target_or_identifier_information_in_features():
    t = random_table(3, 80, seed=4)
    base = history_ids(t, 1)
    perturbed = t.assign(target_id=0, target="Pass", target_event_id="x", target_has_360=False,
                         n_skipped_raw_events=3, match_id=999, event_id="e", index=7)
    assert np.array_equal(history_ids(perturbed, 1), base)
    later = t.copy()
    later.loc[later.index[40:], "symbol_id"] = 0
    assert np.array_equal(history_ids(later, 1)[:40], base[:40])


# ============================================================================ C. model and procedure
def test_checkpointed_cv_equals_experiment5_procedure():
    t = random_table(10, 120, seed=3)
    rows = HistoryRows.from_table(t)
    folds = folds_for(range(10))
    ref = xe.run_nested_cv(rows, folds, SMALL, "XGB-F0")
    new = h2.run_nested_cv_checkpointed(rows.features("XGB-F0"), rows.match_id, rows.y, folds, SMALL, "XGB-F0")
    assert np.array_equal(ref.proba, new.proba)
    key = lambda c: (c["fold_id"], c["candidate"], c["n_estimators"])  # noqa: E731
    assert [key(c) for c in ref.choices] == [key(c) for c in new.choices]
    assert ref.n_fits == new.n_fits == 5 * (2 * 4 + 1)


def test_h2_probabilities_valid_columns_mapped_and_start_token_handled():
    rng = np.random.default_rng(0)
    prev = rng.choice([PAD, 0, 1, 8], 4000)
    cur = rng.choice([0, 1], 4000)
    y = np.where(prev == PAD, 12, np.where(prev == 8, 5, (cur + 1) % 15))      # <START> -> class 12, Pressure -> Shot
    ids = np.column_stack([prev, cur])
    m = xe.XGBNextEventModel(xe.booster_params(SMALL, {"max_depth": 3, "learning_rate": 0.3, "min_child_weight": 1,
                                                       "reg_lambda": 1.0, "colsample_bytree": 1.0}), 40).fit(ids, y)
    p = m.predict_proba(np.array([[PAD, 0], [8, 1], [0, 0], [1, 1], [14, 0]]))
    assert p.shape == (5, 15) and np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(1), 1, atol=1e-12)
    assert p.argmax(1)[:4].tolist() == [12, 5, 1, 2]
    a = m.predict_proba(ids[:50])
    b = xe.XGBNextEventModel(m.params, 40).fit(ids, y).predict_proba(ids[:50])
    assert np.array_equal(a, b)                                                   # reproducible


def test_nested_cv_never_tunes_on_outer_validation():
    t = random_table(10, 120, seed=5)
    rows, ids = h2_ids(t)
    folds = folds_for(range(10))
    trace = []
    a = h2.run_nested_cv_checkpointed(ids, rows.match_id, rows.y, folds, SMALL, "XGB-H2", trace=trace)
    for f in folds:
        for x in (x for x in trace if x["outer_fold"] == f.fold_id):
            if x["purpose"] == "outer_validation_predict":
                assert x["match_ids"] == set(f.validation_match_ids)
            else:
                assert not x["match_ids"] & set(f.validation_match_ids)
    y2 = rows.y.copy()
    va = rows.mask(folds[0].validation_match_ids)
    y2[va] = (y2[va] + 5) % 15                                                    # scramble fold-0 validation labels
    b = h2.run_nested_cv_checkpointed(ids, rows.match_id, y2, folds, SMALL, "XGB-H2")
    assert a.choices[0]["candidate"] == b.choices[0]["candidate"] and a.choices[0]["n_estimators"] == b.choices[0]["n_estimators"]
    assert np.array_equal(a.proba[va], b.proba[va])
    bad = folds_for(range(10))
    bad[1] = FoldAssignment(1, bad[1].train_match_ids + (1,), bad[1].validation_match_ids)
    with pytest.raises(LeakageError):
        h2.run_nested_cv_checkpointed(ids, rows.match_id, rows.y, bad, SMALL, "XGB-H2")


def test_checkpoint_resume_identical_and_refuses_mismatch(tmp_path: Path, monkeypatch):
    t = random_table(10, 100, seed=6)
    rows, ids = h2_ids(t)
    folds = folds_for(range(10))
    ident = {"a": 1, "data": "x"}
    full = h2.run_nested_cv_checkpointed(ids, rows.match_id, rows.y, folds, SMALL, "XGB-H2", tmp_path, ident)
    assert full.computed_folds == [0, 1, 2, 3, 4] and not full.resumed_folds
    (tmp_path / "fold3.json").unlink()                                            # fold 3 "not committed"
    calls = []
    orig = xe.lockstep_curve
    monkeypatch.setattr(xe, "lockstep_curve", lambda *a, **k: calls.append(1) or orig(*a, **k))
    again = h2.run_nested_cv_checkpointed(ids, rows.match_id, rows.y, folds, SMALL, "XGB-H2", tmp_path, ident)
    assert again.resumed_folds == [0, 1, 2, 4] and again.computed_folds == [3]
    assert len(calls) == 2                                                        # only fold 3's 2 candidates re-searched
    assert np.array_equal(again.proba, full.proba) and again.n_fits == full.n_fits
    assert [c["n_estimators"] for c in again.choices] == [c["n_estimators"] for c in full.choices]
    with pytest.raises(h2.CheckpointMismatch):
        h2.run_nested_cv_checkpointed(ids, rows.match_id, rows.y, folds, SMALL, "XGB-H2", tmp_path, {**ident, "data": "y"})
    raw = bytearray((tmp_path / "fold0.npz").read_bytes())
    raw[-10] ^= 1
    (tmp_path / "fold0.npz").write_bytes(bytes(raw))
    with pytest.raises(h2.CheckpointMismatch):
        h2.run_nested_cv_checkpointed(ids, rows.match_id, rows.y, folds, SMALL, "XGB-H2", tmp_path, ident)
    assert not list(tmp_path.glob(".*tmp*"))                                      # no temporary files left


def test_weighted_cell_audit_checks_pass_on_synthetic_data():
    t = random_table(8, 200, seed=7)
    rows, ids = h2_ids(t)
    tr, va = rows.match_id < 6, rows.match_id >= 6
    cfg = {**SMALL, "search": {**SMALL["search"], "n_candidates": 2}}
    out = h2.weighted_cell_audit(ids, rows.y, tr, va, cfg, [0, 1], [1, 5, 20])
    assert out["checks"]["pass"].all()
    s = out["round1_structure"]
    assert s["base_score_identical"].all() and s["round1_tree_structure_identical"].all()
    assert (s["round1_max_abs_leaf_difference"] < 1e-5).all()
    d = out["divergence"]
    assert (d[d.rounds == 1]["max_abs_probability_difference"] < 1e-5).all()


def test_feature_report_counts():
    ids = np.array([[PAD, 0], [0, 1], [0, 1], [1, 5]])
    rep, counts = h2.feature_report(ids, np.array([1, 1, 5, 0]))
    assert rep["matrix_shape"] == [4, 31] and rep["missing_history_rows"] == 1
    assert rep["unique_two_event_contexts_observed"] == 3 and rep["possible_contexts"] == 240
    assert counts.set_index("class").at["<START>", "E_{t-1}_count"] == 1
    assert counts.set_index("class").at["Carry", "target_count"] == 2


# ============================================================================ D. metrics vs hand computation
def test_confusion_and_decision_metrics_hand_computed():
    y = np.array([0, 0, 1, 2])
    d = np.array([0, 1, 1, 0])
    cm = confusion_counts(y, d, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]
    pc = per_class_from_confusion(cm, ["a", "b", "c"])
    assert pc["support"].tolist() == [2, 1, 1]
    assert np.allclose(pc["precision"], [1 / 2, 1 / 2, 0]) and np.allclose(pc["recall"], [1 / 2, 1, 0])
    assert np.allclose(pc["f1"], [1 / 2, 2 / 3, 0])
    m = metrics_from_confusion(cm)
    assert math.isclose(m["accuracy"], 0.5)
    assert math.isclose(m["macro_f1"], (1 / 2 + 2 / 3 + 0) / 3)
    assert math.isclose(m["weighted_f1"], (2 * 1 / 2 + 1 * 2 / 3 + 0) / 4)


def test_log_loss_average_precision_and_shot_recall_hand_computed():
    p = np.array([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]])
    y = np.array([0, 1])
    assert math.isclose(probability_metrics(y, p)["log_loss"], (math.log(2) + math.log(4)) / 2)
    # AP: ranks 1..4 with labels 1,0,1,0 -> precision 1 at recall .5, 2/3 at recall 1
    assert math.isclose(xe.weighted_average_precision(np.array([1, 0, 1, 0]), np.array([.9, .8, .7, .6])), 0.5 + 0.5 * 2 / 3)
    # ties: two tied scores with one positive form one threshold -> precision 1/2 at recall 1
    assert math.isclose(xe.weighted_average_precision(np.array([1, 0]), np.array([.5, .5])), 0.5)
    # weights act as row multiplicities
    assert math.isclose(xe.weighted_average_precision(np.array([1, 0, 1]), np.array([.9, .8, .7]), np.array([1, 2, 1])),
                        xe.weighted_average_precision(np.array([1, 0, 0, 1]), np.array([.9, .8, .8, .7])))
    # Shot recall under argmax: 2 Shot targets, one predicted as Shot
    yy = np.array([5, 5, 0, 0])
    dd = np.array([5, 0, 0, 5])
    pc = per_class_from_confusion(confusion_counts(yy, dd, 15), CLASSES)
    assert pc.at[5, "recall"] == 0.5 and pc.at[5, "precision"] == 0.5 and pc.at[5, "support"] == 2


# ============================================================================ B. real data (local only)
DATASET = PROJECT_ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
EXP5_PRED = PROJECT_ROOT / "data/interim/xgboost_experiment5/oof_predictions.parquet"


def test_config_reuses_experiment5_search_exactly():
    cfg = yaml.safe_load(CFG_PATH.read_text())
    cfg5 = yaml.safe_load((PROJECT_ROOT / cfg["experiment5_config"]).read_text())
    assert cfg["xgboost"] == cfg5["xgboost"] and cfg["search"] == cfg5["search"]
    assert xe.candidate_list(cfg) == xe.candidate_list(cfg5)
    assert cfg["feature_set"] == {"name": "XGB-H2", "n_previous": 1}
    assert cfg["output_dir"] != cfg5["output_dir"] and "5b" in cfg["output_dir"]


@pytest.mark.skipif(not (DATASET.exists() and EXP5_PRED.exists()), reason="local datasets not available")
def test_real_rows_folds_and_isolation_match_experiment5():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_x5", PROJECT_ROOT / "scripts/run_xgboost_experiment5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = yaml.safe_load(CFG_PATH.read_text())
    inp = mod.load_inputs(cfg)
    rows, ids = h2_ids(inp["tb"])
    exp5 = pd.read_parquet(EXP5_PRED, columns=["event_id", "match_id", "fold_id", "y"])
    assert len(rows.y) == 690_147 and len(set(rows.match_id)) == 304
    assert np.array_equal(exp5["event_id"].to_numpy(), rows.event_id) and np.array_equal(exp5["y"].to_numpy(), rows.y)
    fold_of_row = np.full(len(rows.y), -1)
    for f in inp["folds"]:
        assert not set(f.train_match_ids) & set(f.validation_match_ids)
        fold_of_row[rows.mask(f.validation_match_ids)] = f.fold_id
    assert np.array_equal(fold_of_row, exp5["fold_id"].to_numpy())
    final_test = set(json.loads((PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json").read_text())["final_test_match_ids"])
    assert len(final_test) == 83 and not set(inp["table"]["match_id"]) & final_test
    assert ids.shape == (690_147, 2) and (ids[:, 0] == PAD).sum() == (rows.seq_pos == 0).sum() == 568
