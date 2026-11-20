"""Tests for the canonical task actions15_360current_v1 and its baselines.

Synthetic raw streams exercise every rule; one regression test on the real development data
runs only when the local caches exist (they are not part of the repository)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.sequence_models import canonical_baselines as cb
from src.sequence_models.canonical_task import (
    CLASSES,
    CONFIG_PATH,
    FORBIDDEN_RAW_FIELDS,
    TASK_VERSION,
    CanonicalTaskError,
    build_canonical_table,
    check_canonical_table,
    dataset_sha256,
    load_task_spec,
)
from src.sequence_models.event_classes import SCENARIO_D15_CLASSES
from src.sequence_models.feature_sets import design_matrix
from src.sequence_models.splits import FoldAssignment, LeakageError
from src.sequence_models.v3_taxonomy import D15
from src.utils.paths import PROJECT_ROOT


# ----------------------------------------------------------------------------- helpers
def stream(match_id: int, records: list[tuple], period: int = 1) -> list[dict]:
    """records: (type, t_seconds, subtype, outcome, related_to_previous, has_360, freeze_frame_nonempty)."""
    rows = []
    for i, (typ, t, sub, out, rel_prev, h360, ff) in enumerate(records):
        rows.append({"match_id": match_id, "list_position": i, "index": i + 1, "event_id": f"{match_id}-{period}-{i}",
                     "period": period, "timestamp": f"00:00:{t:06.3f}", "t_seconds": float(t), "type": typ,
                     "has_360": h360, "has_freeze_frame": ff and h360, "subtype": sub, "outcome": out,
                     "related_events": f"{match_id}-{period}-{i - 1}" if rel_prev else ""})
    return rows


def R(typ, t, sub=None, out=None, rel=False, h360=True, ff=True):
    return (typ, t, sub, out, rel, h360, ff)


def table_of(*matches) -> pd.DataFrame:
    return build_canonical_table(pd.DataFrame([r for m in matches for r in m]))


def by_type(t: pd.DataFrame) -> list[str]:
    return t["symbol"].tolist()


# ----------------------------------------------------------------------------- classes and config
def test_exactly_15_classes_with_stable_ids():
    assert len(CLASSES) == 15
    assert CLASSES == SCENARIO_D15_CLASSES == tuple(D15)
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    assert cfg["classes"] == list(CLASSES) and cfg["task_version"] == TASK_VERSION == "actions15_360current_v1"
    t = table_of(stream(1, [R(c, i) for i, c in enumerate(CLASSES)]))
    assert t["symbol_id"].tolist() == list(range(15))


@pytest.mark.parametrize("key,value", [("k4_aerial_lost_rule", True), ("target_event_requires_360", True),
                                       ("current_event_requires_usable_360", False), ("task_version", "x"),
                                       ("companion_rules", ["K1", "K2"]), ("event_order", "timestamp")])
def test_config_loader_refuses_any_other_rule(tmp_path: Path, key, value):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg[key] = value
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(CanonicalTaskError):
        load_task_spec(p)
    assert load_task_spec().task_version == TASK_VERSION


# ----------------------------------------------------------------------------- companion rules and targets
def test_companion_records_removed_and_next_event_rebuilt():
    t = table_of(stream(1, [
        R("Pass", 1.0), R("Dispossessed", 2.0), R("Duel", 2.0, "Tackle", "Won", rel=True),       # K2 -> removed
        R("Carry", 3.0), R("Shot", 4.0, out="Off T"), R("Goal Keeper", 4.5, "Shot Faced", rel=True),  # K1 -> removed
        R("Pass", 9.0), R("50/50", 10.0, out="Lost"), R("50/50", 10.0, out="Won", rel=True),       # K3 -> second removed
        R("Carry", 11.0), R("Shot", 12.0, out="Saved"), R("Goal Keeper", 12.6, "Shot Saved", rel=True),  # genuine save kept
        R("Pass", 14.0)]))
    assert by_type(t) == ["Pass", "Dispossessed", "Carry", "Shot", "Pass", "50/50", "Carry", "Shot", "Goal Keeper", "Pass"]
    assert t["target"].tolist()[:-1] == by_type(t)[1:]
    assert t.set_index("symbol").loc["Dispossessed", "target"] == "Carry"
    assert t.attrs["companion_rule_counts"] == {"K2": 1, "K1": 1, "K3": 1}


def test_unlinked_tackle_and_aerial_lost_are_kept_k4_inactive():
    t = table_of(stream(1, [R("Dispossessed", 1.0), R("Duel", 1.0, "Tackle"),          # no link -> not a companion
                            R("Duel", 3.0, "Aerial Lost"), R("Clearance", 3.0, rel=True), R("Pass", 5.0)]))
    assert by_type(t) == ["Dispossessed", "Duel", "Duel", "Clearance", "Pass"]


def test_skipped_non_vocabulary_records_do_not_break_sequences():
    t = table_of(stream(1, [R("Pass", 1.0), R("Ball Receipt*", 2.0), R("Carry", 2.0), R("Substitution", 5.0, h360=False),
                            R("Pass", 6.0)]))
    assert by_type(t) == ["Pass", "Carry", "Pass"]
    assert t["n_skipped_raw_events"].tolist()[:2] == [1, 1]


# ----------------------------------------------------------------------------- 360 eligibility
def test_current_event_360_required_target_360_not_required():
    t = table_of(stream(1, [R("Pass", 1.0, h360=False), R("Carry", 2.0), R("Pass", 3.0, ff=False),
                            R("Pressure", 4.0), R("Shot", 5.0, h360=False), R("Carry", 6.0)]))
    elig = dict(zip(t["symbol"] + t["index"].astype(str), t["row_eligible"]))
    assert elig == {"Pass1": False, "Carry2": True, "Pass3": False, "Pressure4": True, "Shot5": False, "Carry6": False}
    pressure = t[t["symbol"] == "Pressure"].iloc[0]
    assert pressure["row_eligible"] and pressure["target"] == "Shot" and not pressure["target_has_360"]


# ----------------------------------------------------------------------------- ordering and boundaries
def test_order_is_statsbomb_index_not_timestamp():
    recs = stream(1, [R("Pass", 9.0), R("Carry", 1.0), R("Shot", 5.0)])   # timestamps not monotone
    shuffled = pd.DataFrame(recs).sample(frac=1.0, random_state=0)
    t = build_canonical_table(shuffled)
    assert by_type(t) == ["Pass", "Carry", "Shot"]
    assert t["target"].tolist()[:2] == ["Carry", "Shot"]


def test_no_cross_match_or_period_transitions_and_no_shootout():
    p1 = stream(1, [R("Pass", 1.0), R("Carry", 2.0)], period=1)
    p2 = [dict(r, list_position=r["list_position"] + 2, index=r["index"] + 2)
          for r in stream(1, [R("Shot", 1.0), R("Pass", 2.0)], period=2)]
    p5 = [dict(r, list_position=r["list_position"] + 4, index=r["index"] + 4)
          for r in stream(1, [R("Shot", 1.0), R("Shot", 2.0)], period=5)]
    m2 = stream(2, [R("Pass", 1.0), R("Carry", 2.0)])
    t = build_canonical_table(pd.DataFrame(p1 + p2 + p5 + m2))
    assert 5 not in set(t["period"])
    last = t.groupby("sequence_id").tail(1)
    assert (last["target_id"] == -1).all() and not last["row_eligible"].any()
    assert len(t) == 6


def test_invariant_checker_rejects_corrupted_tables():
    t = table_of(stream(1, [R("Pass", 1.0), R("Carry", 2.0), R("Shot", 3.0)]))
    bad = t.copy()
    bad.loc[0, "target_id"] = 5
    with pytest.raises(CanonicalTaskError):
        check_canonical_table(bad)
    bad = t.copy()
    bad.loc[0, "usable_360"] = False
    with pytest.raises(CanonicalTaskError):
        check_canonical_table(bad)


# ----------------------------------------------------------------------------- leakage
def test_no_future_information_in_table_or_f0_features():
    t = table_of(stream(1, [R("Pass", 1.0, out="Incomplete"), R("Interception", 2.0, rel=True), R("Carry", 3.0)]))
    exposed = set(t.columns) & FORBIDDEN_RAW_FIELDS
    assert exposed <= {"target", "target_id", "target_event_id", "target_has_360", "n_skipped_raw_events"}
    assert "outcome" not in t.columns and "related_events" not in t.columns
    x, _ = design_matrix(t, "F0")
    x2, _ = design_matrix(t.assign(target_id=0, target="Pass", n_skipped_raw_events=9), "F0")
    assert np.array_equal(x, x2)


def test_dataset_hash_is_deterministic_and_content_sensitive():
    a = table_of(stream(1, [R("Pass", 1.0), R("Carry", 2.0), R("Shot", 3.0)]))
    b = table_of(stream(1, [R("Pass", 1.0), R("Carry", 2.0), R("Shot", 3.0)]))
    c = table_of(stream(1, [R("Pass", 1.0), R("Carry", 2.0), R("Shot", 3.0, h360=False)]))
    assert dataset_sha256(a) == dataset_sha256(b) != dataset_sha256(c)


# ----------------------------------------------------------------------------- baselines and isolation
def synthetic_rows(n_matches=10, seed=0) -> cb.Rows:
    rng = np.random.default_rng(seed)
    recs = []
    for m in range(n_matches):
        recs += stream(m, [R(CLASSES[int(rng.integers(0, 4))], float(i)) for i in range(200)])
    return cb.Rows.from_table(build_canonical_table(pd.DataFrame(recs)))


CFG = {"markov1": {"alpha_grid": [0.1, 1.0]}, "markov2": {"alpha_grid": [0.1, 1.0], "min_context_count": 5},
       "logistic": {"C_grid": [0.01, 10.0], "max_iter": 2000, "tol": 1e-8, "absent_class_floor": 1e-9},
       "evaluation": {"bootstrap_replicates": 20, "bootstrap_seed": 1, "probability_tolerance": 1e-9}}


def folds_for(ids, k=5):
    return [FoldAssignment(f, tuple(m for m in ids if m % k != f), tuple(m for m in ids if m % k == f)) for f in range(k)]


def test_baselines_produce_valid_distributions_and_isolate_folds():
    r = synthetic_rows()
    cv = cb.run_cv(r, folds_for(range(10)), CFG)
    for p in cv.proba.values():
        assert p.shape == (len(r.y), 15) and (p > 0).all() and np.allclose(p.sum(1), 1)
    tables = cb.evaluate(r, cv, CFG)
    assert set(tables["model_comparison"]["model"]) == set(cb.MODELS)
    # logistic F0 on the same rows reproduces the Markov-1 conditional distribution (theory)
    assert np.abs(cv.proba["logistic_F0"] - cv.proba["markov1"]).max() < 0.05


def test_overlapping_folds_are_refused():
    r = synthetic_rows()
    bad = folds_for(range(10))
    bad[0] = FoldAssignment(0, bad[0].train_match_ids + (0,), bad[0].validation_match_ids)
    with pytest.raises(LeakageError):
        cb.run_cv(r, bad, CFG)


def test_selection_never_sees_outer_validation_rows():
    r = synthetic_rows()
    folds = folds_for(range(10))
    seen = []
    original = cb.predict

    def spy(model, rows, tr, pred_rows, value, cfg):
        seen.append((model, set(rows.match_id[tr]), set(rows.match_id[pred_rows])))
        return original(model, rows, tr, pred_rows, value, cfg)

    cb.predict = spy
    try:
        cb.run_cv(r, folds[:5], CFG)
    finally:
        cb.predict = original
    outer_val = set(folds[0].validation_match_ids)
    # the first fold's selection calls precede its final fit; none may touch its validation matches
    n_sel = sum(len(cb.grid_of(m, CFG)) * 4 if len(cb.grid_of(m, CFG)) > 1 else 0 for m in cb.MODELS) + len(cb.MODELS)
    first_fold_calls = seen[:n_sel]
    for model, train, pred in first_fold_calls:
        assert not train & outer_val
    final = [c for c in first_fold_calls if c[2] == outer_val]
    assert len(final) == len(cb.MODELS)


# ----------------------------------------------------------------------------- real data (local caches only)
CACHE = PROJECT_ROOT / "data/interim/sequence_models/event_table_development_v2.pkl"
DETAILS = PROJECT_ROOT / "data/interim/sequence_models/v3/event_details_development.pkl"


@pytest.mark.skipif(not (CACHE.exists() and DETAILS.exists()), reason="local development caches not available")
def test_real_tier_b_counts_match_the_investigation():
    from src.sequence_models.canonical_task import load_development_raw

    raw, info = load_development_raw(load_task_spec())
    t = build_canonical_table(raw)
    part = info["partition"]
    tb = t[t["match_id"].isin(part.tier_b)]
    assert not set(t["match_id"]) & set(part.final_test)
    assert len(tb) == 782_377 and int(tb["row_eligible"].sum()) == 690_147     # results/investigation_360_taxonomy_2026-11-18
    assert (tb["symbol"] == "Duel").sum() == 8_549          # Aerial Lost only (K2 removes every linked Tackle; K4 inactive)
    assert tb["symbol"].nunique() == 15
    assert t.attrs["companion_rule_counts"] == {"K2": 12_221, "K1": 6_441, "K3": 659}   # development, periods 1-4
