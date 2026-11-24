"""Tests for experiment 6A (MLR-H5: multinomial logistic regression with a five-event history).

Synthetic sequences with known answers throughout. Real-data checks run only when the local
canonical dataset / result directories exist (they are not part of the repository)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import log_loss as sk_log_loss

from src.sequence_models import logistic_h5_experiment as lh
from src.sequence_models import xgboost_experiment as xe
from src.sequence_models.canonical_task import CLASSES, K, build_canonical_table
from src.sequence_models.evaluation_v2 import bootstrap_multiplicities, per_match_stats
from src.sequence_models.event_history import PAD, HistoryError, HistoryRows, encode, history_ids
from src.sequence_models.raw_history_check import reconstruct
from src.sequence_models.splits import FoldAssignment, LeakageError, assert_no_final_test
from src.sequence_models.xgboost_h2_experiment import CheckpointMismatch
from src.utils.paths import PROJECT_ROOT

C = {c: i for i, c in enumerate(CLASSES)}
CONFIG = PROJECT_ROOT / "configs" / "logistic_h5_experiment6a_v1.yaml"
LR = {**yaml.safe_load(CONFIG.read_text())["logistic"]}
DATASET = PROJECT_ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
RESULTS = PROJECT_ROOT / "results/logistic_h5_experiment6a_actions15_360current_v1_2026-11-23"
SMALL_LR = {**LR, "C_grid": [0.1, 1.0, 10.0], "C_edge_extension": {"upper": [100.0], "lower": [0.01]}}


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


def markov_table(n_matches=20, n_events=200, seed=0) -> pd.DataFrame:
    """Sequences with second-order structure, all 15 classes reachable."""
    rng = np.random.default_rng(seed)
    recs = []
    for m in range(n_matches):
        s = [int(rng.integers(0, 15)), int(rng.integers(0, 15))]
        for _ in range(n_events - 2):
            nxt = (s[-1] + s[-2]) % 15 if rng.random() < 0.6 else int(rng.integers(0, 15))
            s.append(nxt)
        recs += raw_records(m, [R(CLASSES[c], float(i), sub="Shot Saved" if CLASSES[c] == "Goal Keeper" else None,
                                  h360=bool(rng.random() > 0.15)) for i, c in enumerate(s)])
    return build_canonical_table(pd.DataFrame(recs))


def folds_for(ids, k=5):
    return [FoldAssignment(f, tuple(m for m in ids if m % k != f), tuple(m for m in ids if m % k == f)) for f in range(k)]


@pytest.fixture(scope="module")
def synth():
    t = markov_table()
    rows = HistoryRows.from_table(t)
    folds = folds_for(sorted(set(rows.match_id)))
    return t, rows, folds


@pytest.fixture(scope="module")
def nested(synth):
    _, rows, folds = synth
    trace: list = []
    res = lh.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL_LR, trace=trace)
    return res, trace


# ----------------------------------------------------------------------------- 1. classes / configuration
def test_canonical_class_order_and_configuration():
    assert CLASSES == ("Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed",
                       "Pressure", "Duel", "Interception", "Block", "Goal Keeper", "Foul Committed", "50/50")
    cfg = yaml.safe_load(CONFIG.read_text())
    task = yaml.safe_load((PROJECT_ROOT / cfg["task_config"]).read_text())
    assert task["classes"] == list(CLASSES) and task["task_version"] == "actions15_360current_v1"
    assert cfg["feature_set"] == {"name": "MLR-H5", "n_previous": 4, "n_columns": 79}
    assert cfg["logistic"]["l1_ratio"] == 0.0 and cfg["logistic"]["class_weight"] is None
    assert cfg["logistic"]["C_grid"] == [0.01, 0.1, 1.0, 10.0, 100.0]
    cfg5 = yaml.safe_load((PROJECT_ROOT / cfg["experiment5_config"]).read_text())
    for key in ("bootstrap_replicates", "bootstrap_seed", "probability_tolerance"):
        assert cfg["evaluation"][key] == cfg5["evaluation"][key]
    assert sum(cfg["evaluation"]["raw_verification"]["strata"].values()) >= 500


# ----------------------------------------------------------------------------- 2-8. history, encoding, boundaries
def test_target_alignment_and_five_position_history():
    seq = ["Pass", "Carry", "Shot", "Goal Keeper", "Pass", "Carry", "Pressure", "Duel"]
    t = table(raw_records(1, [R(s, float(i), sub="Shot Saved" if s == "Goal Keeper" else None) for i, s in enumerate(seq)]))
    rows = HistoryRows.from_table(t)
    ids = [C[s] for s in seq]
    assert rows.y.tolist() == ids[1:]                                  # Y_t = e_{t+1}
    assert rows.ctx[0].tolist() == [PAD] * 4 + [ids[0]]
    assert rows.ctx[3].tolist() == [PAD, ids[0], ids[1], ids[2], ids[3]]
    assert rows.ctx[6].tolist() == ids[2:7]                            # (e_{t-4}, ..., e_t)


def test_sparse_one_hot_is_position_specific_and_equals_xgb_h5_encoder():
    ids = np.array([[PAD, PAD, C["Pass"], C["Carry"], C["Shot"]], [C["Pass"]] * 5, [C["Carry"], C["Pass"], PAD, PAD, C["Pass"]]])
    ids[2] = [PAD, PAD, PAD, C["Carry"], C["Pass"]]
    x = lh.encode_sparse(ids)
    dense, names = encode(ids)
    assert x.shape == (3, 79) and lh.feature_names() == names and np.array_equal(x.toarray(), dense)
    on = [names[j] for j in x[1].indices]
    assert on == ["E_{t-4}=Pass", "E_{t-3}=Pass", "E_{t-2}=Pass", "E_{t-1}=Pass", "E_t=Pass"]   # 5 distinct columns
    assert (np.diff(x.indptr) == 5).all() and set(x.data) == {1.0}
    # same multiset of events, different order -> different vectors (not a bag of events)
    a = lh.encode_sparse(np.array([[C["Pass"], C["Carry"], C["Shot"], C["Pass"], C["Carry"]]])).toarray()
    b = lh.encode_sparse(np.array([[C["Carry"], C["Pass"], C["Pass"], C["Shot"], C["Carry"]]])).toarray()
    assert a.sum() == b.sum() == 5 and not np.array_equal(a, b)
    ft = lh.feature_table()
    assert len(ft) == 79 and ft.groupby("lag").size().to_dict() == {0: 15, 1: 16, 2: 16, 3: 16, 4: 16}


def test_start_handling_and_invalid_inputs():
    x = lh.encode_sparse(np.array([[PAD, PAD, PAD, PAD, C["Shot"]]]))
    assert [lh.feature_names()[j] for j in x.indices] == [f"E_{{t-{lag}}}=<START>" for lag in (4, 3, 2, 1)] + ["E_t=Shot"]
    with pytest.raises(HistoryError):
        lh.encode_sparse(np.array([[0, 0, 0, 0, PAD]]))                # E_t can never be <START>
    with pytest.raises(HistoryError):
        lh.encode_sparse(np.array([[16, 0, 0, 0, 0]]))
    with pytest.raises(HistoryError):
        lh.encode_sparse(np.array([0, 0, 0, 0, 0]))


def test_period_and_match_boundaries_never_share_history():
    p1 = raw_records(1, [R("Pass", 1), R("Carry", 2), R("Shot", 3), R("Pass", 4), R("Carry", 5)], period=1)
    p2 = raw_records(1, [R("Clearance", 1), R("Pass", 2), R("Carry", 3)], period=2, start=5)
    p5 = raw_records(1, [R("Shot", 1), R("Shot", 2)], period=5, start=8)       # shoot-out: excluded
    m2 = raw_records(2, [R("Dribble", 1), R("Pass", 2)])
    rows = HistoryRows.from_table(table(p1, p2, p5, m2))
    first_p2 = rows.ctx[(rows.match_id == 1) & (rows.seq_pos == 0)][1]
    assert first_p2.tolist() == [PAD] * 4 + [C["Clearance"]]          # no Carry/Pass/Shot from period 1
    second_p2 = rows.ctx[(rows.match_id == 1) & (rows.seq_pos == 1)][1]
    assert second_p2.tolist() == [PAD, PAD, PAD, C["Clearance"], C["Pass"]]
    assert rows.ctx[rows.match_id == 2][0].tolist() == [PAD] * 4 + [C["Dribble"]]
    assert len(rows.y) == 4 + 2 + 1                                    # no target across period / match; no period 5
    for lag in range(1, 5):
        assert np.array_equal(rows.ctx[:, 4 - lag] == PAD, rows.seq_pos < lag)


def test_companions_are_removed_before_history_and_raw_json_agrees():
    recs = raw_records(1, [R("Pass", 1), R("Shot", 2), R("Goal Keeper", 3, sub="Shot Faced"), R("Goal Keeper", 3, sub="Shot Saved"),
                           R("Pass", 4), R("Dispossessed", 5), R("Duel", 5, sub="Tackle", rel=True), R("Carry", 6), R("Pass", 7)])
    rows = HistoryRows.from_table(table(recs))
    # canonical: Pass, Shot, GK (Shot Saved), Pass, Dispossessed, Carry, Pass (K1 Shot Faced and K2 Tackle removed)
    assert [CLASSES[c] for c in rows.ctx[-1]] == ["Shot", "Goal Keeper", "Pass", "Dispossessed", "Carry"]
    assert rows.y[-1] == C["Pass"] and len(rows.y) == 6
    # raw JSON form of the same records -> independent reconstruction agrees
    js = []
    for r in recs:
        ev = {"id": r["event_id"], "index": r["index"], "period": r["period"], "timestamp": r["timestamp"],
              "type": {"name": r["type"]}, "related_events": [r["related_events"]] if r["related_events"] else []}
        if r["type"] == "Goal Keeper":
            ev["goalkeeper"] = {"type": {"name": r["subtype"]}}
        if r["type"] == "Duel":
            ev["duel"] = {"type": {"name": r["subtype"]}}
        js.append(ev)
    for i, eid in enumerate(rows.event_id):
        hist, target, _ = reconstruct(js, eid, 4)
        assert hist == rows.ctx[i].tolist() and target == rows.y[i]


# ----------------------------------------------------------------------------- 9. leakage (adversarial)
def test_features_never_depend_on_future_events_or_targets(synth):
    t, rows, _ = synth
    full = history_ids(t, 4)
    for cut in (17, 999, len(t) - 1):
        assert np.array_equal(history_ids(t.iloc[:cut].copy(), 4), full[:cut])
    later = t.copy()
    later.loc[later.index[500:], "symbol_id"] = (later["symbol_id"].to_numpy()[500:] + 3) % 15
    assert np.array_equal(history_ids(later, 4)[:500], full[:500])
    poisoned = t.assign(target_id=0, target="Pass", target_event_id="x", n_skipped_raw_events=9, target_has_360=False,
                        event_id="e", match_id=-1, index=0, has_360=False, usable_360=False)
    assert np.array_equal(history_ids(poisoned, 4), full)              # labels / ids / 360 never read


def test_adversarial_target_copy_cannot_reach_features(synth):
    """If the target leaked into any feature column, permuting targets would change features."""
    t, rows, _ = synth
    shuffled = t.copy()
    shuffled["target_id"] = np.random.default_rng(1).permutation(t["target_id"].to_numpy())
    assert np.array_equal(history_ids(shuffled, 4), history_ids(t, 4))
    x = lh.encode_sparse(rows.ctx).toarray()
    # no column equals the target indicator of any class (a target-copy feature would)
    for k in range(K):
        col = rows.y == k
        assert not any(np.array_equal(x[:, j].astype(bool), col) for j in range(x.shape[1]) if col.any())


def test_adversarial_period_start_shot_does_not_leak_into_next_period():
    p1 = raw_records(7, [R("Pass", 1), R("Shot", 2)], period=1)
    p2 = raw_records(7, [R("Pass", 1), R("Carry", 2)], period=2, start=2)
    rows = HistoryRows.from_table(table(p1, p2))
    r = rows.ctx[(rows.seq_pos == 0)][1]
    assert C["Shot"] not in r.tolist() and r.tolist() == [PAD] * 4 + [C["Pass"]]


# ----------------------------------------------------------------------------- 10. final test isolation
def test_final_test_assertion_and_loader_guard():
    with pytest.raises(LeakageError):
        assert_no_final_test([1, 2, 3], [3])
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import inspect

    import run_xgboost_experiment5
    src = inspect.getsource(run_xgboost_experiment5.load_inputs)
    assert src.count("assert_no_final_test") >= 2              # table and every fold are checked
    runner = (PROJECT_ROOT / "scripts/run_logistic_h5_experiment6a.py").read_text()
    assert "load_inputs(cfg)" in runner and "touches_final_test" in runner


# ----------------------------------------------------------------------------- 11-14. folds, nesting, selection
def test_outer_and_inner_folds_and_no_match_overlap(synth, nested):
    _, rows, folds = synth
    res, trace = nested
    for f in folds:
        assert set(rows.match_id[res.fold_of_row == f.fold_id]) == set(f.validation_match_ids)
        val = set(f.validation_match_ids)
        purposes = {t["purpose"] for t in trace if t["outer_fold"] == f.fold_id}
        assert purposes == {"inner_train", "inner_validation", "outer_train", "outer_validation_predict"}
        for t in trace:
            if t["outer_fold"] == f.fold_id and t["purpose"] != "outer_validation_predict":
                assert not (t["match_ids"] & val)
        inner_val = [t["match_ids"] for t in trace if t["outer_fold"] == f.fold_id and t["purpose"] == "inner_validation"]
        others = [set(g.validation_match_ids) for g in folds if g.fold_id != f.fold_id]
        assert sorted(map(sorted, inner_val)) == sorted(map(sorted, others))   # 4 inner folds = the other outer folds
    bad = [FoldAssignment(0, (0, 1), (1, 2))]
    with pytest.raises(LeakageError):
        lh.run_nested_cv(rows.ctx[:10], np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]), rows.y[:10], bad, SMALL_LR)


def test_selection_uses_training_data_only(synth, nested):
    """Corrupting ALL outer-validation targets of fold 0 must not change fold 0's chosen C,
    inner scores, coefficients or predictions (validation labels are only used for scoring)."""
    _, rows, folds = synth
    res, _ = nested
    y2 = rows.y.copy()
    va = res.fold_of_row == 0
    y2[va] = (y2[va] + 7) % 15
    out = lh.run_nested_cv(rows.ctx, rows.match_id, y2, folds, SMALL_LR)
    a = next(c for c in res.choices if c["fold_id"] == 0)
    b = next(c for c in out.choices if c["fold_id"] == 0)
    assert a["C"] == b["C"] and a["inner_log_loss"] == b["inner_log_loss"] and a["grid_evaluated"] == b["grid_evaluated"]
    assert np.array_equal(res.models[0].coef, out.models[0].coef)
    assert np.array_equal(res.proba[va], out.proba[va])
    # ... whereas fold 0's labels DO enter the other folds' training (sanity: the test can detect use)
    assert any(not np.array_equal(res.models[f].coef, out.models[f].coef) for f in (1, 2, 3, 4))


def test_select_C_grid_edge_extension_and_ties():
    calls = []

    def score(c):
        calls.append(c)
        return (np.log10(c) - 3.2) ** 2                                 # minimum above the grid
    best, scores, grid = lh.select_C(score, [0.1, 1.0, 10.0], {"upper": [100.0, 1000.0, 1e4], "lower": [0.01]})
    assert best == 1000.0 and grid == [0.1, 1.0, 10.0, 100.0, 1000.0, 1e4] and calls[-1] == 1e4
    best, _, grid = lh.select_C(lambda c: 1.0, [0.1, 1.0, 10.0], {"upper": [100.0], "lower": [0.01]})
    assert best == 0.01 and grid == [0.01, 0.1, 1.0, 10.0]             # ties -> smaller C (then lower edge rule)
    best, _, grid = lh.select_C(lambda c: (np.log10(c)) ** 2, [0.1, 1.0, 10.0], {"upper": [100.0], "lower": [0.01]})
    assert best == 1.0 and grid == [0.1, 1.0, 10.0]                    # interior: no extension
    with pytest.raises(ValueError):
        lh.select_C(lambda c: 1.0, [0.1, 1.0], {"upper": [1.0]})


# ----------------------------------------------------------------------------- 15-19. probabilities, classes, determinism
def test_probabilities_shape_normalisation_and_canonical_columns():
    rng = np.random.default_rng(0)
    n = 4000
    cur = rng.choice([0, 1, 2, 8], n)
    y = np.select([cur == 0, cur == 1, cur == 2], [5, 14, 3], 13)    # only classes 3, 5, 13, 14 present
    ids = np.column_stack([np.full((n, 4), PAD), cur])
    m = lh.MLRH5Model(1.0, LR).fit(ids, y)
    probe = np.array([[PAD] * 4 + [c] for c in (0, 1, 2, 8, 10)])
    p = m.predict_proba(probe)
    assert p.shape == (5, 15) and p.dtype == np.float64
    assert np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(1), 1, atol=1e-12)
    assert p[:4].argmax(1).tolist() == [5, 14, 3, 13]                  # column c = canonical class id c
    assert m.fit_info["absent_classes"] == [c for c in range(15) if c not in (3, 5, 13, 14)]
    assert (p[:, 0] < 1e-8).all()                                      # absent class: floor only
    xe.check_probabilities(p, 1e-9)


def test_rare_class_seen_once_is_learnt_and_log_loss_finite():
    rng = np.random.default_rng(1)
    ids = np.column_stack([np.full((3000, 4), PAD), rng.choice([0, 1], 3000)])
    y = ids[:, 4].copy()
    y[17] = C["50/50"]                                                  # a single 50/50 target
    m = lh.MLRH5Model(1.0, LR).fit(ids, y)
    p = m.predict_proba(ids)
    assert m.fit_info["absent_classes"] == [c for c in range(15) if c not in (0, 1, 14)]
    assert np.isfinite(-np.log(p[np.arange(len(y)), y])).all() and 0 < p[17, 14] < 0.01


def test_fit_deterministic_converged_and_gradient_independently_small(synth):
    _, rows, _ = synth
    a = lh.MLRH5Model(1.0, LR).fit(rows.ctx, rows.y)
    b = lh.MLRH5Model(1.0, LR).fit(rows.ctx.copy(), rows.y.copy())
    assert np.array_equal(a.predict_proba(rows.ctx), b.predict_proba(rows.ctx))
    assert a.fit_info["converged"] and a.fit_info["gradient_ok"] and a.fit_info["warnings"] == []
    assert a.fit_info["max_abs_gradient"] < 1e-8 and a.fit_info["solver"] == "newton-cholesky"


def test_cell_fit_equals_row_fit_and_frozen_softmax_equals_sklearn(synth):
    _, rows, _ = synth
    cell = lh.MLRH5Model(3.0, LR).fit(rows.ctx, rows.y)
    row = lh.MLRH5Model(3.0, LR).fit_rows(rows.ctx, rows.y)
    assert cell.fit_info["n_train_cells"] < row.fit_info["n_train_cells"] == len(rows.y)
    assert np.abs(cell.predict_proba(rows.ctx) - row.predict_proba(rows.ctx)).max() < 1e-7
    assert np.abs(cell.frozen().predict_proba(rows.ctx) - cell.predict_proba(rows.ctx)).max() < 1e-12
    # lbfgs reaches the same optimum (same objective, different solver)
    lb = lh.MLRH5Model(3.0, {**LR, "solver": "lbfgs", "max_iter": 20000, "tol": 1e-10}).fit(rows.ctx, rows.y)
    assert abs(lb.fit_info["objective"] - cell.fit_info["objective"]) < 1e-8


def test_coefficients_are_deviation_coded(synth, nested):
    res, _ = nested
    tabs = lh.coefficient_tables(res.models, res.support, min_support=10)
    ident = tabs["identifiability"]
    assert (ident["max_abs_sum_over_classes"] < 1e-6).all()
    assert (ident["max_abs_sum_over_position_block"] < 1e-4).all()
    summ = tabs["coefficient_summary"]
    assert len(summ) == 15 * 79 and set(tabs["position_strength"]["position"]) == {"E_t", "E_{t-1}", "E_{t-2}", "E_{t-3}", "E_{t-4}"}
    top = tabs["top_coefficients"]
    assert set(top["direction"]) == {"positive", "negative"}


def test_position_strength_reflects_an_additive_process():
    """y depends additively on e_t (strongly) and e_{t-1} (weakly), not on e_{t-4}."""
    rng = np.random.default_rng(5)
    n = 20000
    ids = rng.integers(0, 15, (n, 5))
    u = rng.random(n)
    y = np.where(u < 0.6, ids[:, 4], np.where(u < 0.75, ids[:, 3], rng.integers(0, 15, n)))
    m = lh.MLRH5Model(1.0, LR).fit(ids, y)
    tabs = lh.coefficient_tables({0: m.frozen()}, {0: m.support}, min_support=10)
    rms = tabs["position_strength"].query("class_id == -1").set_index("position")["rms"]
    assert rms["E_t"] > rms["E_{t-1}"] > 3 * rms["E_{t-4}"]


def test_additive_model_cannot_represent_a_pure_interaction():
    """Documented limitation: y = (e_{t-1} + e_t) mod 15 is deterministic given both positions
    but has uniform marginals given either one; an additive softmax stays near log(15)."""
    rng = np.random.default_rng(6)
    ids = np.column_stack([np.full((15000, 3), PAD), rng.integers(0, 15, (15000, 2))])
    y = (ids[:, 3] + ids[:, 4]) % 15
    p = lh.MLRH5Model(100.0, LR).fit(ids, y).predict_proba(ids)
    assert -np.log(p[np.arange(len(y)), y]).mean() > 0.95 * np.log(15)


# ----------------------------------------------------------------------------- 20-22. metrics, pairing, bootstrap
def test_metrics_and_paired_differences_match_definitions(synth, nested):
    _, rows, _ = synth
    res, _ = nested
    uniform = np.full_like(res.proba, 1 / 15)
    cfg = {"evaluation": {"bootstrap_replicates": 50, "bootstrap_seed": 7, "probability_tolerance": 1e-9}}
    tabs = xe.evaluate_models(rows.y, rows.match_id, res.fold_of_row, {"MLR-H5": res.proba, "uniform": uniform},
                              [("MLR-H5", "uniform")], cfg)
    mc = tabs["model_comparison"].set_index("model")
    assert abs(mc.at["MLR-H5", "log_loss"] - sk_log_loss(rows.y, res.proba, labels=list(range(15)))) < 1e-12
    assert abs(mc.at["uniform", "log_loss"] - np.log(15)) < 1e-12
    d = -np.log(res.proba[np.arange(len(rows.y)), rows.y]) + np.log(uniform[np.arange(len(rows.y)), rows.y])
    pdiff = tabs["paired_differences"].set_index("metric")
    assert abs(pdiff.at["log_loss", "difference"] - d.mean()) < 1e-12 and pdiff.at["log_loss", "ci_high"] < 0
    s = lh.nll_difference_strata(rows.y, res.proba, uniform, rows.seq_pos, rows.ctx[:, 4])
    for stratum in ("history_length", "current_class", "target_class"):
        assert abs(s[s.stratum == stratum]["contribution"].sum() - d.mean()) < 1e-12
        assert s[s.stratum == stratum]["n_rows"].sum() == len(rows.y)
    same = xe.evaluate_models(rows.y, rows.match_id, res.fold_of_row, {"a": res.proba, "b": res.proba.copy()}, [("a", "b")], cfg)
    ll = same["paired_differences"].set_index("metric").loc["log_loss"]
    assert ll["difference"] == 0 and ll["ci_low"] == 0 and ll["ci_high"] == 0


def test_bootstrap_resamples_whole_matches(synth, nested):
    _, rows, _ = synth
    res, _ = nested
    n_matches = len(np.unique(rows.match_id))
    mult = bootstrap_multiplicities(n_matches, 30, 7)
    assert mult.shape == (30, n_matches) and (mult.sum(1) == n_matches).all()
    assert np.array_equal(mult, bootstrap_multiplicities(n_matches, 30, 7))
    st = per_match_stats(rows.y, res.proba, {"argmax": res.proba.argmax(1)}, rows.match_id, 15)
    assert st["n"].sum() == len(rows.y) and len(st["n"]) == n_matches  # sufficient statistics per MATCH
    ll = mult @ st["ll"] / (mult @ st["n"])
    assert np.isfinite(ll).all() and ll.std() > 0


# ----------------------------------------------------------------------------- 23-25. outputs, checkpoints, provenance
def test_nested_output_schema(synth, nested):
    _, rows, folds = synth
    res, _ = nested
    assert res.proba.shape == (len(rows.y), 15) and not np.isnan(res.proba).any()
    assert sorted(c["fold_id"] for c in res.choices) == [0, 1, 2, 3, 4]
    fits = pd.DataFrame(res.fits)
    assert {"stage", "C", "n_iter", "converged", "max_abs_gradient", "warnings", "objective"} <= set(fits.columns)
    assert (fits.stage == "outer_refit").sum() == 5 and fits.converged.all()
    for f in folds:
        assert res.models[f.fold_id].coef.shape == (15, 79) and res.support[f.fold_id].shape == (79,)


def test_checkpoints_resume_identically_and_refuse_mismatch(tmp_path, synth, nested):
    _, rows, folds = synth
    ident = {"run": "a", "data": "x"}
    a = lh.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL_LR, tmp_path, ident)
    assert a.computed_folds == [0, 1, 2, 3, 4] and len(list(tmp_path.glob("fold*.json"))) == 5
    (tmp_path / "fold3.json").unlink()                                 # simulate an interruption in fold 3
    (tmp_path / "fold4.json").unlink()
    b = lh.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL_LR, tmp_path, ident)
    assert b.resumed_folds == [0, 1, 2] and b.computed_folds == [3, 4]
    assert np.array_equal(a.proba, b.proba) and a.choices == b.choices
    with pytest.raises(CheckpointMismatch):
        lh.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL_LR, tmp_path, {**ident, "data": "y"})
    data = bytearray((tmp_path / "fold0.npz").read_bytes())
    data[-10] ^= 0xFF
    (tmp_path / "fold0.npz").write_bytes(bytes(data))
    with pytest.raises(CheckpointMismatch):
        lh.run_nested_cv(rows.ctx, rows.match_id, rows.y, folds, SMALL_LR, tmp_path, ident)
    assert not list(tmp_path.glob(".*tmp*"))                           # atomic writes leave no temp files


def test_nested_results_match_nested_definition(synth, nested):
    """Independent re-derivation of fold 0: re-score the chosen C by hand."""
    _, rows, folds = synth
    res, _ = nested
    f = folds[0]
    inner = [g for g in folds if g.fold_id != 0]
    c0 = next(c for c in res.choices if c["fold_id"] == 0)["C"]
    nll, n = 0.0, 0
    for held in inner:
        tr = np.isin(rows.match_id, [m for g in inner if g is not held for m in g.validation_match_ids])
        va = np.isin(rows.match_id, held.validation_match_ids)
        p = lh.MLRH5Model(c0, SMALL_LR).fit(rows.ctx[tr], rows.y[tr]).predict_proba(rows.ctx[va])
        nll += -np.log(p[np.arange(va.sum()), rows.y[va]]).sum()
        n += va.sum()
    assert abs(nll / n - next(c for c in res.choices if c["fold_id"] == 0)["inner_log_loss"]) < 1e-12
    tr = np.isin(rows.match_id, f.train_match_ids)
    va = np.isin(rows.match_id, f.validation_match_ids)
    p = lh.MLRH5Model(c0, SMALL_LR).fit(rows.ctx[tr], rows.y[tr]).predict_proba(rows.ctx[va])
    assert np.array_equal(p, res.proba[va])


# ----------------------------------------------------------------------------- real data (skipped without local caches)
needs_data = pytest.mark.skipif(not DATASET.exists(), reason="local canonical dataset not available")
needs_results = pytest.mark.skipif(not (RESULTS / "integrity_audit.csv").exists(), reason="experiment 6A not run")


@needs_data
def test_real_rows_and_ids_equal_experiment5_rows():
    tb = pd.read_parquet(DATASET)
    tb = tb[tb["tier"] == "tier_b"].reset_index(drop=True)
    rows = HistoryRows.from_table(tb)
    assert len(rows.y) == 690_147 and len(set(rows.match_id)) == 304
    assert np.array_equal(rows.ctx, rows.features("XGB-H5"))
    p5 = PROJECT_ROOT / "data/interim/xgboost_experiment5/oof_predictions.parquet"
    if p5.exists():
        o = pd.read_parquet(p5, columns=["event_id", "y", "match_id"])
        assert np.array_equal(o["event_id"].to_numpy(), rows.event_id) and np.array_equal(o["y"].to_numpy(), rows.y)
    split = json.loads((PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json").read_text())
    final = set(split["final_test_match_ids"])
    assert len(final) == 83 and not (set(tb["match_id"]) & final)


@needs_data
def test_historical_results_unchanged():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_logistic_h5_experiment6a import historical_snapshot
    snap = historical_snapshot(yaml.safe_load(CONFIG.read_text()))
    assert all(not e["mismatches"] for e in snap["directories"].values())


@needs_results
def test_saved_experiment6a_results_are_consistent():
    audit = pd.read_csv(RESULTS / "integrity_audit.csv")
    assert not (audit["result"] == "FAIL").any()
    checks = json.loads((RESULTS / "checksums.json").read_text())
    from src.sequence_models.experiment import sha256_file
    assert all(sha256_file(RESULTS / k) == v for k, v in checks.items())
    pred = pd.read_parquet(PROJECT_ROOT / "data/interim/logistic_h5_experiment6a/oof_predictions_mlr_h5.parquet")
    p = pred[[f"MLR-H5__p{c}" for c in range(15)]].to_numpy()
    mc = pd.read_csv(RESULTS / "model_comparison.csv").set_index("model")
    assert len(pred) == 690_147 and pred["event_id"].is_unique
    assert abs(sk_log_loss(pred["y"], p, labels=list(range(15))) - mc.at["MLR-H5", "log_loss"]) < 1e-9
