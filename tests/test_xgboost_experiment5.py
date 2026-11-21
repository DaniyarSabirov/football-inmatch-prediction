"""Tests for experiment 5 (XGBoost on actions15_360current_v1): history features, target
alignment, padding, boundaries, leakage, nested-CV isolation, probabilities and metrics.

Synthetic sequences with known answers throughout; real-data checks run only when the local
canonical dataset exists (it is not part of the repository)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss as sk_log_loss

from src.sequence_models import canonical_baselines as cb
from src.sequence_models import xgboost_experiment as xe
from src.sequence_models.canonical_task import CLASSES, build_canonical_table
from src.sequence_models.evaluation_v2 import probability_metrics
from src.sequence_models.event_history import (
    FEATURE_SETS,
    PAD,
    HistoryError,
    HistoryRows,
    aggregate_cells,
    encode,
    history_ids,
)
from src.sequence_models.raw_history_check import reconstruct
from src.sequence_models.splits import FoldAssignment, LeakageError
from src.utils.paths import PROJECT_ROOT

C = {c: i for i, c in enumerate(CLASSES)}


# ----------------------------------------------------------------------------- synthetic helpers
def raw_records(match_id: int, records: list[tuple], period: int = 1, start: int = 0) -> list[dict]:
    """records: (type, t_seconds, subtype, outcome, related_to_previous, has_360)."""
    rows = []
    for j, (typ, t, sub, out, rel, h360) in enumerate(records):
        i = start + j
        rows.append({"match_id": match_id, "list_position": i, "index": i + 1, "event_id": f"{match_id}-{i}",
                     "period": period, "timestamp": f"00:00:{t:06.3f}", "t_seconds": float(t), "type": typ,
                     "has_360": h360, "has_freeze_frame": h360, "subtype": sub, "outcome": out,
                     "related_events": f"{match_id}-{i - 1}" if rel else ""})
    return rows


def R(typ, t=0.0, sub=None, out=None, rel=False, h360=True):
    return (typ, t, sub, out, rel, h360)


def table(*parts) -> pd.DataFrame:
    return build_canonical_table(pd.DataFrame([r for p in parts for r in p]))


def random_table(n_matches=10, n_events=150, seed=0, n_types=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recs = []
    for m in range(n_matches):
        recs += raw_records(m, [R(CLASSES[int(rng.integers(0, n_types))], float(i), h360=bool(rng.random() > 0.2))
                                for i in range(n_events)])
    return build_canonical_table(pd.DataFrame(recs))


SMALL_CFG = {
    "xgboost": {"objective": "multi:softprob", "num_class": 15, "tree_method": "hist", "eval_metric": "mlogloss",
                "nthread": 2, "seed": 1, "fixed_params": {"subsample": 1.0, "gamma": 0.0, "reg_alpha": 0.0, "max_bin": 256}},
    "search": {"n_candidates": 2, "candidate_seed": 3, "max_rounds": 15, "patience": 3,
               "space": {"max_depth": [2, 3], "learning_rate": [0.3], "min_child_weight": [1],
                         "reg_lambda": [1.0], "colsample_bytree": [1.0]}},
    "evaluation": {"bootstrap_replicates": 20, "bootstrap_seed": 1, "probability_tolerance": 1e-9,
                   "permutation_repeats": 3, "permutation_seed": 2},
}


def params(**kw):
    return xe.booster_params(SMALL_CFG, {"max_depth": 3, "learning_rate": 0.3, "min_child_weight": 1,
                                         "reg_lambda": 1.0, "colsample_bytree": 1.0, **kw})


def folds_for(ids, k=5):
    return [FoldAssignment(f, tuple(m for m in ids if m % k != f), tuple(m for m in ids if m % k == f)) for f in range(k)]


# ----------------------------------------------------------------------------- classes and encoding
def test_canonical_class_order_and_feature_columns():
    assert len(CLASSES) == 15 and CLASSES[0] == "Pass" and CLASSES[5] == "Shot" and CLASSES[14] == "50/50"
    assert PAD == 15 and PAD not in range(15)
    x, names = encode(np.array([[C["Shot"]]]))
    assert x.shape == (1, 15) and names == [f"E_t={c}" for c in CLASSES] and x[0, 5] == 1 and x.sum() == 1
    x, names = encode(np.array([[PAD, PAD, C["Pass"], C["Carry"], C["Shot"]]]))
    assert x.shape == (1, 79) and len(names) == 79
    on = [n for n, v in zip(names, x[0]) if v == 1]
    assert on == ["E_{t-4}=<START>", "E_{t-3}=<START>", "E_{t-2}=Pass", "E_{t-1}=Carry", "E_t=Shot"]
    with pytest.raises(HistoryError):
        encode(np.array([[C["Pass"], PAD]]))          # E_t can never be padding
    assert FEATURE_SETS == {"XGB-F0": 0, "XGB-H5": 4}


# ----------------------------------------------------------------------------- history and targets
def test_five_event_history_and_target_alignment_on_known_sequence():
    seq = ["Pass", "Carry", "Shot", "Goal Keeper", "Pass", "Carry", "Pressure"]
    t = table(raw_records(1, [R(s, float(i), sub="Shot Saved" if s == "Goal Keeper" else None) for i, s in enumerate(seq)]))
    rows = HistoryRows.from_table(t)
    ids = [C[s] for s in seq]
    assert len(rows.y) == 6                                     # last observation has no target
    assert rows.y.tolist() == ids[1:]                           # y_t = E_{t+1}
    assert rows.ctx[0].tolist() == [PAD, PAD, PAD, PAD, ids[0]]
    assert rows.ctx[2].tolist() == [PAD, PAD, ids[0], ids[1], ids[2]]
    assert rows.ctx[5].tolist() == ids[1:6]                     # E_{t-4}..E_t at t = 5
    assert rows.features("XGB-F0").tolist() == [[i] for i in ids[:6]]


def test_start_of_period_padding_and_no_history_across_period_or_match():
    p1 = raw_records(1, [R("Pass", 1), R("Carry", 2), R("Shot", 3)], period=1)
    p2 = raw_records(1, [R("Clearance", 1), R("Pass", 2)], period=2, start=3)
    m2 = raw_records(2, [R("Dribble", 1), R("Pass", 2)])
    rows = HistoryRows.from_table(table(p1, p2, m2))
    first_p2 = rows.ctx[(rows.match_id == 1) & (rows.seq_pos == 0)][1]
    assert first_p2.tolist() == [PAD, PAD, PAD, PAD, C["Clearance"]]
    first_m2 = rows.ctx[rows.match_id == 2][0]
    assert first_m2.tolist() == [PAD, PAD, PAD, PAD, C["Dribble"]]
    assert len(rows.y) == 2 + 1 + 1                             # no target across period / match
    # early-period rows are kept, not dropped
    assert (rows.seq_pos == 0).sum() == 3


def test_padding_is_monotone_and_counts_follow_sequence_position():
    rows = HistoryRows.from_table(random_table(3, 40))
    for lag in range(1, 5):
        assert np.array_equal(rows.ctx[:, 4 - lag] == PAD, rows.seq_pos < lag)


def test_history_uses_earlier_events_without_360_but_rows_require_current_360():
    t = table(raw_records(1, [R("Pass", 1, h360=False), R("Carry", 2), R("Shot", 3, h360=False), R("Pass", 4)]))
    rows = HistoryRows.from_table(t)
    assert len(rows.y) == 1                                     # only the Carry row has its own snapshot
    assert rows.ctx[0].tolist() == [PAD, PAD, PAD, C["Pass"], C["Carry"]]   # earlier no-360 Pass is history
    assert rows.y[0] == C["Shot"]                               # target needs no 360


def test_rows_identical_across_feature_sets_and_equal_to_baseline_rows():
    t = random_table()
    rows, brows = HistoryRows.from_table(t), cb.Rows.from_table(t)
    f0, h5 = rows.features("XGB-F0"), rows.features("XGB-H5")
    assert len(f0) == len(h5) == len(brows.y) and np.array_equal(f0[:, 0], h5[:, 4])
    assert np.array_equal(rows.y, brows.y) and np.array_equal(rows.match_id, brows.match_id)
    assert np.array_equal(rows.ctx[:, 4], brows.cur)
    assert np.array_equal(np.where(rows.ctx[:, 3] == PAD, -1, rows.ctx[:, 3]), brows.prev)


def test_feature_construction_deterministic_and_rejects_bad_tables():
    t = random_table(seed=4)
    assert np.array_equal(history_ids(t, 4), history_ids(t.copy(), 4))
    shuffled = t.sample(frac=1.0, random_state=0)
    with pytest.raises(HistoryError):
        history_ids(shuffled, 4)


# ----------------------------------------------------------------------------- leakage
def test_no_future_information_truncation_and_target_perturbation():
    t = random_table(2, 120, seed=9)
    full = history_ids(t, 4)
    for cut in (10, 57, 119):
        assert np.array_equal(history_ids(t.iloc[:cut].copy(), 4), full[:cut])   # later events never matter
    perturbed = t.assign(target_id=0, target="Pass", target_event_id="x", n_skipped_raw_events=7,
                         target_has_360=False, event_id="e", match_id=99, index=0)
    assert np.array_equal(history_ids(perturbed, 4), full)     # targets / identifiers are never read
    later = t.copy()
    later.loc[later.index[60:], "symbol_id"] = 0
    assert np.array_equal(history_ids(later, 4)[:60], full[:60])


# ----------------------------------------------------------------------------- model outputs
def deterministic_rows(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    cur = rng.choice([0, 1, 2, 3, 8], n)
    y = (cur + 1) % 15
    y[cur == 8] = 13                       # class 13 only after Pressure
    return cur[:, None], y


def test_probabilities_cover_15_classes_in_canonical_order_even_with_absent_classes():
    x, y = deterministic_rows()
    assert set(np.unique(y)) == {1, 2, 3, 4, 13}                # 10 classes absent in training
    m = xe.XGBNextEventModel(params(), 30).fit(x, y)
    p = m.predict_proba(np.array([[0], [1], [2], [3], [8], [14]]))
    assert p.shape == (6, 15) and p.dtype == np.float64
    assert np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(1), 1, atol=1e-12)
    assert p[:5].argmax(1).tolist() == [1, 2, 3, 4, 13]       # column c is canonical class id c
    xe.check_probabilities(p, 1e-9)
    with pytest.raises(ValueError):
        xe.check_probabilities(p[:, :14], 1e-9)


def test_weighted_cell_fit_equals_row_level_fit():
    t = random_table(8, 300, seed=2)
    rows = HistoryRows.from_table(t)
    x, y = rows.features("XGB-H5"), rows.y
    tr, va = rows.match_id < 6, rows.match_id >= 6
    cells = xe.XGBNextEventModel(params(), 20).fit(x[tr], y[tr]).predict_proba(x[va])
    each = xe.XGBNextEventModel(params(), 20).fit_rows(x[tr], y[tr]).predict_proba(x[va])
    ll = lambda p: -np.log(p[np.arange(va.sum()), y[va]]).mean()  # noqa: E731
    assert abs(ll(cells) - ll(each)) < 1e-5 and np.abs(cells - each).max() < 1e-3


def test_fits_are_reproducible_under_fixed_seed():
    rows = HistoryRows.from_table(random_table(4, 200, seed=5))
    x = rows.features("XGB-H5")
    a = xe.XGBNextEventModel(params(colsample_bytree=0.5), 15).fit(x, rows.y).predict_proba(x)
    b = xe.XGBNextEventModel(params(colsample_bytree=0.5), 15).fit(x, rows.y).predict_proba(x)
    assert np.array_equal(a, b)


def test_aggregate_cells_round_trip():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 16, (500, 5))
    ids[:, -1] = rng.integers(0, 15, 500)
    y = rng.integers(0, 15, 500)
    cx, cy, w, inv = aggregate_cells(ids, y)
    assert w.sum() == 500 and np.array_equal(cx[inv], ids) and np.array_equal(cy[inv], y)
    assert len(np.unique(np.column_stack([cx, cy]), axis=0)) == len(cx)


# ----------------------------------------------------------------------------- metrics
def test_log_loss_and_average_precision_match_sklearn():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(15), 400)
    y = rng.integers(0, 15, 400)
    assert np.isclose(probability_metrics(y, p)["log_loss"], sk_log_loss(y, p, labels=range(15)), atol=1e-12)
    s = np.round(rng.random(400), 1)                            # many ties
    yb = rng.random(400) < 0.2
    w = rng.integers(0, 4, 400).astype(float)
    assert np.isclose(xe.weighted_average_precision(yb, s), average_precision_score(yb, s), atol=1e-12)
    assert np.isclose(xe.weighted_average_precision(yb, s, w), average_precision_score(yb, s, sample_weight=w), atol=1e-12)
    z = rng.normal(size=(10, 15))
    assert np.allclose(xe.softmax64(z), np.exp(z) / np.exp(z).sum(1, keepdims=True))


# ----------------------------------------------------------------------------- nested CV
def test_nested_cv_isolates_outer_validation_and_returns_valid_predictions():
    t = random_table(10, 150, seed=3)
    rows = HistoryRows.from_table(t)
    folds = folds_for(range(10))
    trace = []
    out = xe.run_nested_cv(rows, folds, SMALL_CFG, "XGB-H5", trace=trace)
    xe.check_probabilities(out.proba, 1e-9)
    assert len(out.choices) == 5 and out.n_fits == 5 * (2 * 4 + 1)
    for f in folds:
        val = set(f.validation_match_ids)
        used = [x for x in trace if x["outer_fold"] == f.fold_id]
        assert {x["purpose"] for x in used} == {"inner_train", "inner_validation", "outer_train", "outer_validation_predict"}
        for x in used:
            if x["purpose"] != "outer_validation_predict":
                assert not x["match_ids"] & val
            else:
                assert x["match_ids"] == val
        inner_val = [x["match_ids"] for x in used if x["purpose"] == "inner_validation"]
        assert set().union(*inner_val) == set(f.train_match_ids)    # inner folds partition the outer training
    # the outer-validation rows of a fold do not influence its selection
    a = xe.run_nested_cv(rows, folds, SMALL_CFG, "XGB-H5")
    rows2 = HistoryRows.from_table(t)
    va = rows2.mask(folds[0].validation_match_ids)
    rows2.y = rows2.y.copy()
    rows2.y[va] = (rows2.y[va] + 7) % 15                        # scramble fold 0's validation targets
    b = xe.run_nested_cv(rows2, folds[:1] + folds[1:], SMALL_CFG, "XGB-H5")
    assert a.choices[0]["candidate"] == b.choices[0]["candidate"] and a.choices[0]["n_estimators"] == b.choices[0]["n_estimators"]
    assert np.array_equal(a.proba[va], b.proba[va])


def test_overlapping_folds_are_refused():
    rows = HistoryRows.from_table(random_table(10, 60))
    bad = folds_for(range(10))
    bad[0] = FoldAssignment(0, bad[0].train_match_ids + (0,), bad[0].validation_match_ids)
    with pytest.raises(LeakageError):
        xe.run_nested_cv(rows, bad, SMALL_CFG, "XGB-F0")


def test_candidate_list_is_fixed_and_distinct():
    a, b = xe.candidate_list(SMALL_CFG), xe.candidate_list(SMALL_CFG)
    assert a == b and len({tuple(c.values()) for c in a}) == 2


def test_permutation_importance_identifies_the_informative_position():
    rng = np.random.default_rng(0)
    recs = []
    for m in range(10):
        types = [CLASSES[int(rng.integers(0, 4))] for _ in range(300)]
        for i in range(2, 300):
            if rng.random() < 0.3:          # E_{t+1} depends on E_{t-1}: mark the event two steps later
                types[i] = "Shot" if types[i - 2] == "Pass" else types[i]
        recs += raw_records(m, [R(s, float(i)) for i, s in enumerate(types)])
    rows = HistoryRows.from_table(build_canonical_table(pd.DataFrame(recs)))
    folds = folds_for(range(10))
    out = xe.run_nested_cv(rows, folds, SMALL_CFG, "XGB-H5")
    imp = xe.permutation_importance(rows, out, folds, SMALL_CFG).groupby("position")["log_loss_increase_mean"].mean()
    assert imp["E_{t-1}"] > imp["E_{t-3}"] and imp["E_{t-1}"] > imp["E_{t-4}"] and imp["E_{t-1}"] > 0.01


# ----------------------------------------------------------------------------- independent raw reconstruction
def json_events(spec: list[tuple]) -> list[dict]:
    """spec: (type, timestamp seconds, detail key, detail dict, related to previous, period)."""
    out = []
    for i, (typ, ts, key, detail, rel, period) in enumerate(spec):
        e = {"id": f"e{i}", "index": i + 1, "period": period, "timestamp": f"00:00:{ts:06.3f}", "type": {"name": typ},
             "related_events": [f"e{i - 1}"] if rel else []}
        if key:
            e[key] = detail
        out.append(e)
    return out


def test_raw_reconstruction_agrees_with_canonical_pipeline_including_companions():
    spec = [("Pass", 1, None, None, False, 1), ("Ball Receipt*", 2, None, None, False, 1),
            ("Dispossessed", 3, None, None, False, 1), ("Duel", 3, "duel", {"type": {"name": "Tackle"}}, True, 1),  # K2
            ("Carry", 4, None, None, False, 1), ("Shot", 5, None, None, False, 1),
            ("Goal Keeper", 5.5, "goalkeeper", {"type": {"name": "Shot Faced"}}, True, 1),                         # K1
            ("50/50", 7, None, None, False, 1), ("50/50", 7, None, None, True, 1),                                 # K3
            ("Pass", 8, None, None, False, 1), ("Pressure", 9, None, None, False, 1),
            ("Pass", 1, None, None, False, 2), ("Carry", 2, None, None, False, 2)]
    ev = json_events(spec)
    raw = pd.DataFrame([{"match_id": 1, "list_position": i, "index": i + 1, "event_id": e["id"], "period": e["period"],
                         "timestamp": e["timestamp"], "t_seconds": float(spec[i][1]), "type": e["type"]["name"],
                         "has_360": True, "has_freeze_frame": True,
                         "subtype": (e.get("duel") or e.get("goalkeeper") or {}).get("type", {}).get("name"),
                         "outcome": None, "related_events": ";".join(e["related_events"])} for i, e in enumerate(ev)])
    rows = HistoryRows.from_table(build_canonical_table(raw))
    for i, eid in enumerate(rows.event_id):
        hist, target, _ = reconstruct(ev, eid, 4)
        assert hist == rows.ctx[i].tolist() and target == rows.y[i]
    # companions (K1-K3) and Ball Receipt* are not history positions
    assert rows.ctx[rows.event_id == "e9"][0].tolist() == [C["Dispossessed"], C["Carry"], C["Shot"], C["50/50"], C["Pass"]]
    assert "e10" not in set(rows.event_id)                      # last event of period 1: no target, no row


# ----------------------------------------------------------------------------- real data (local dataset only)
DATASET = PROJECT_ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
EXP_CONFIG = PROJECT_ROOT / "configs/xgboost_experiment5_v1.yaml"


@pytest.mark.skipif(not DATASET.exists(), reason="local canonical dataset not available")
def test_real_tier_b_rows_folds_and_final_test_isolation():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_x5", PROJECT_ROOT / "scripts/run_xgboost_experiment5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = yaml.safe_load(EXP_CONFIG.read_text())
    inp = mod.load_inputs(cfg)
    final_test = set(json.loads((PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json").read_text())["final_test_match_ids"])
    assert len(final_test) == 83 and not set(inp["table"]["match_id"]) & final_test
    rows = HistoryRows.from_table(inp["tb"])
    assert len(rows.y) == 690_147 and len(set(rows.match_id)) == 304
    assert len(rows.features("XGB-F0")) == len(rows.features("XGB-H5"))
    for f in inp["folds"]:
        assert not set(f.train_match_ids) & set(f.validation_match_ids)
        assert not (set(f.train_match_ids) | set(f.validation_match_ids)) & final_test
    assert (rows.ctx[:, 3] == PAD).sum() == (rows.seq_pos == 0).sum()
    brows = cb.Rows.from_table(inp["tb"])
    assert np.array_equal(rows.ctx[:, 4], brows.cur) and np.array_equal(rows.y, brows.y)


def test_experiment_config_is_complete_and_bounded():
    cfg = yaml.safe_load(EXP_CONFIG.read_text())
    assert cfg["task_config"] == "configs/actions15_360current_v1.yaml"
    assert cfg["xgboost"]["objective"] == "multi:softprob" and cfg["xgboost"]["num_class"] == 15
    assert cfg["xgboost"]["fixed_params"]["subsample"] == 1.0
    assert not Path(PROJECT_ROOT / cfg["output_dir"]).resolve().is_relative_to(PROJECT_ROOT / cfg["baseline_results_dir"])
    assert len(xe.candidate_list(cfg)) == cfg["search"]["n_candidates"] == 20
