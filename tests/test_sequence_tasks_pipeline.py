"""Task construction, splits, spatial features and pipeline leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml_features import load_feature_classification
from src.sequence_models.config import TaskSpec, load_config
from src.sequence_models.experiment import TaskData, cross_validate, prepare_task
from src.sequence_models.raw_events import EVENT_TABLE_COLUMNS
from src.sequence_models.spatial import SpatialScaler, spatial_observations
from src.sequence_models.splits import (
    FoldAssignment,
    LeakageError,
    assert_no_final_test,
    final_test_split,
    grouped_folds,
    pilot_subset,
    verify_folds,
)
from src.sequence_models.tasks import (
    REASON_NO_360,
    REASON_NO_TARGET,
    EventOrderError,
    UnknownEventTypeError,
    build_task_table,
    encode_symbol,
    exclusion_summary,
    validate_event_order,
)

TYPES = ("Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed")
FILTERED = TaskSpec("filtered8", "", TYPES, "vocabulary_only", None, TYPES, (5,))
FULL = TaskSpec("fullstream9", "", TYPES + ("Other",), "all_events", "Other", TYPES, (5,))


def event_table(match_id: int, events: list[tuple[int, str, bool]], shot_types: dict[int, str] | None = None) -> pd.DataFrame:
    """events: (period, type, has_360) in raw order."""
    rows = []
    for i, (period, typ, has_360) in enumerate(events):
        rows.append({c: None for c in EVENT_TABLE_COLUMNS} | {
            "match_id": match_id, "list_position": i, "index": i + 1, "event_id": f"{match_id}-{i}",
            "period": period, "timestamp": f"00:00:{i:02d}.000", "t_seconds": float(i), "type": typ,
            "team_id": 1 + (i % 2), "possession": 1, "possession_team_id": 1, "has_360": has_360,
            "shot_type": (shot_types or {}).get(i),
        })
    return pd.DataFrame(rows, columns=EVENT_TABLE_COLUMNS)


MATCH = event_table(1, [
    (1, "Starting XI", False), (1, "Pass", True), (1, "Ball Receipt*", True), (1, "Pressure", True),
    (1, "Carry", True), (1, "Shot", False), (1, "Half End", False),
    (2, "Pass", True), (2, "Shot", True), (2, "Clearance", True), (2, "Half End", False),
    (5, "Shot", True), (5, "Goal Keeper", False), (5, "Shot", True),
], shot_types={8: "Penalty", 11: "Penalty", 13: "Penalty"})


def test_filtered_task_skips_by_definition_and_counts_skipped_events():
    t = build_task_table(MATCH, FILTERED)
    p1 = t[t["period"] == 1]
    assert p1["raw_type"].tolist() == ["Pass", "Carry", "Shot"]
    assert p1["target"].tolist()[:2] == ["Carry", "Shot"]
    assert p1["n_skipped_raw_events"].tolist()[:2] == [2, 0]  # Ball Receipt*, Pressure skipped explicitly


def test_sequences_never_cross_periods_and_shootout_is_excluded():
    for task in (FILTERED, FULL):
        t = build_task_table(MATCH, task)
        assert 5 not in set(t["period"])
        last_p1 = t[t["period"] == 1].iloc[-1]
        assert last_p1["target_id"] == -1 and last_p1["exclusion_reason"] == REASON_NO_TARGET
        assert t["sequence_id"].unique().tolist() == ["1_P1", "1_P2"]


def test_ordinary_penalty_kept_while_shootout_penalties_removed():
    t = build_task_table(MATCH, FILTERED)
    assert "1-8" in set(t["event_id"])          # penalty kick in period 2
    assert not {"1-11", "1-13"} & set(t["event_id"])  # shootout
    s = exclusion_summary(MATCH, FILTERED)
    assert s.loc[s["reason"] == "excluded_period_penalty_shootout", "n_events"].sum() == 3


def test_full_stream_maps_other_and_targets_immediate_successor():
    t = build_task_table(MATCH, FULL)
    p1 = t[t["period"] == 1].set_index("event_id")
    assert p1.loc["1-1", "target"] == "Other" and p1.loc["1-1", "target_raw_type"] == "Ball Receipt*"
    assert p1.loc["1-0", "symbol"] == "Other" and not p1.loc["1-0", "is_prediction_point"]
    assert p1.loc["1-5", "target"] == "Other"  # Shot -> Half End
    assert (t["n_skipped_raw_events"].dropna() == 0).all()


def test_missing_360_makes_row_ineligible_but_keeps_sequence_continuity():
    t = build_task_table(MATCH, FILTERED).set_index("event_id")
    assert t.loc["1-5", "exclusion_reason"] == REASON_NO_TARGET
    # Carry (with 360) -> Shot (without 360): the Shot is still an observation and the Carry's target
    assert t.loc["1-4", "target_event_id"] == "1-5" and bool(t.loc["1-4", "eval_eligible"])
    other = event_table(2, [(1, "Pass", False), (1, "Carry", True), (1, "Pass", True)])
    t2 = build_task_table(other, FILTERED).set_index("event_id")
    assert not t2.loc["2-0", "eval_eligible"] and t2.loc["2-0", "exclusion_reason"] == REASON_NO_360
    assert t2.loc["2-0", "target_event_id"] == "2-1"


def test_order_is_raw_order_not_timestamp_and_violations_raise():
    swapped = MATCH.copy()
    swapped.loc[1, "t_seconds"], swapped.loc[4, "t_seconds"] = 99.0, 0.0  # timestamps disagree with order
    t = build_task_table(swapped, FILTERED)
    assert t[t["period"] == 1]["raw_type"].tolist() == ["Pass", "Carry", "Shot"]
    bad = MATCH.copy()
    bad.loc[3, "index"] = 99
    with pytest.raises(EventOrderError):
        validate_event_order(bad)
    dup = pd.concat([MATCH, MATCH.iloc[[0]]], ignore_index=True)
    with pytest.raises(EventOrderError):
        validate_event_order(dup)


def test_unknown_event_category_raises_in_filtered_vocabulary():
    with pytest.raises(UnknownEventTypeError):
        encode_symbol("Pressure", FILTERED)
    assert encode_symbol("Some Future Type", FULL) == "Other"


def test_deterministic_task_table():
    pd.testing.assert_frame_equal(build_task_table(MATCH, FILTERED), build_task_table(MATCH.copy(), FILTERED))


# ------------------------------------------------------------------ splits
def strata(n: int = 40) -> pd.Series:
    return pd.Series({1000 + i: f"comp{i % 4}" for i in range(n)})


def test_final_test_split_is_disjoint_stratified_and_deterministic():
    s = strata()
    dev, test = final_test_split(s, 0.2, 7)
    assert not set(dev) & set(test) and set(dev) | set(test) == set(s.index)
    assert len(test) == 8 and s.loc[test].value_counts().tolist() == [2, 2, 2, 2]
    assert final_test_split(s, 0.2, 7) == (dev, test)


def test_pilot_never_contains_final_test_matches():
    s = strata()
    dev, test = final_test_split(s, 0.2, 7)
    pilot = pilot_subset(dev, s, {dev[0], dev[1], test[0]}, 10, 2, 7)
    assert len(pilot) == 10 and not set(pilot) & set(test)
    assert {dev[0], dev[1]} <= set(pilot)
    with pytest.raises(LeakageError):
        assert_no_final_test(pilot + [test[0]], test)


def test_grouped_folds_partition_matches_without_overlap():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"match_id": np.repeat(np.arange(20), 30), "target_id": rng.integers(0, 4, 600)})
    folds = grouped_folds(rows, 5, 3)
    for fold in folds:
        assert not set(fold.train_match_ids) & set(fold.validation_match_ids)
    assert sorted(m for f in folds for m in f.validation_match_ids) == list(range(20))
    with pytest.raises(LeakageError):
        verify_folds([FoldAssignment(0, (1, 2), (2,))], [1, 2])


# ------------------------------------------------------------------ spatial
def production_like(rows: list[dict]) -> pd.DataFrame:
    columns = list(load_feature_classification()["columns"])
    return pd.DataFrame([{c: 0 for c in columns} | r for r in rows], columns=columns)


def test_spatial_features_are_in_the_acting_team_frame_and_missing_is_nan():
    frame = production_like([
        {"current_event_id": "a", "actor_x": 100.0, "actor_y": 10.0, "orientation_available": True, "team_normalization": "REFERENCE"},
        {"current_event_id": "b", "actor_x": 100.0, "actor_y": 10.0, "orientation_available": True, "team_normalization": "MIRRORED"},
        {"current_event_id": "c", "actor_x": np.nan, "actor_y": np.nan, "orientation_available": False, "team_normalization": "MIRRORED"},
    ])
    out = spatial_observations(frame).set_index("event_id")
    assert out.loc["a", "x_attacking"] == 100 and out.loc["a", "lateral_distance_from_centre"] == 30
    assert out.loc["b", "x_attacking"] == 20 and out.loc["b", "lateral_distance_from_centre"] == 30
    assert np.isnan(out.loc["c", "x_attacking"])


def test_scaler_uses_training_values_only():
    train = np.array([[0.0, 1.0], [2.0, np.nan], [4.0, 3.0]])
    s = SpatialScaler.fit(train)
    np.testing.assert_allclose(s.mean, [2.0, 2.0])
    np.testing.assert_allclose(s.std, [np.std([0, 2, 4]), 1.0])
    with pytest.raises(ValueError):
        SpatialScaler.fit(np.array([[1.0], [1.0]]))


# ------------------------------------------------------------------ pipeline leakage
def synthetic_task_data(n_matches: int = 10, seed: int = 0, spatial_seed: int = 1) -> tuple[TaskData, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tables = []
    for m in range(n_matches):
        events = [(p, str(rng.choice(["Pass", "Carry", "Shot", "Ball Receipt*", "Pressure"], p=[.4, .3, .1, .1, .1])), bool(rng.random() < .9))
                  for p in (1, 2) for _ in range(40)]
        tables.append(event_table(100 + m, events))
    et = pd.concat(tables, ignore_index=True)
    srng = np.random.default_rng(spatial_seed)
    spatial = pd.DataFrame({"event_id": et["event_id"], "x_attacking": srng.uniform(0, 120, len(et)),
                            "lateral_distance_from_centre": srng.uniform(0, 40, len(et))})
    return prepare_task(et, FILTERED, spatial), et


def tiny_config():
    config = load_config()
    hmm = config.hmm.__class__(**{**config.hmm.__dict__, "n_states": (2,), "seeds": (0,), "max_iter": 5, "workers": 1})
    return config.__class__(**{**config.__dict__, "hmm": hmm})


def test_cross_validation_has_no_target_or_future_leakage():
    data, _ = synthetic_task_data()
    rows = data.table[data.table["eval_eligible"]][["match_id", "target_id"]]
    folds = grouped_folds(rows, 5, 0)
    config = tiny_config()
    proba, _, fold_of_row, _ = cross_validate(data, folds, config, spatial=True, final_test=[])
    # change the LAST observation of every validation sequence of fold 0 (it is the
    # target of the second-to-last row): all other predictions must be identical
    altered = TaskData(data.task, data.table.copy(), data.seq_rows, data.seq_match)
    val = set(folds[0].validation_match_ids)
    last_rows = [rows_[-1] for sid, rows_ in data.seq_rows.items() if data.seq_match[sid] in val]
    sym = altered.table["symbol_id"].to_numpy().copy()
    sym[last_rows] = (sym[last_rows] + 1) % len(FILTERED.vocabulary)
    altered.table["symbol_id"] = sym
    proba2, _, _, _ = cross_validate(altered, folds, config, spatial=True, final_test=[])
    # (the altered matches are legitimately TRAINING data in folds 1-4, so compare fold 0's rows)
    keep = fold_of_row == 0
    keep[last_rows] = False
    for name in proba:
        np.testing.assert_array_equal(proba[name][keep], proba2[name][keep], err_msg=name)
    # every eligible row predicted exactly once, by the fold that holds out its match
    assert (fold_of_row >= 0).all()
    for fold in folds:
        held = data.table["match_id"].isin(fold.validation_match_ids).to_numpy()
        assert (fold_of_row[held] == fold.fold_id).all()


def test_cross_validation_refuses_final_test_matches():
    data, _ = synthetic_task_data(n_matches=5)
    rows = data.table[data.table["eval_eligible"]][["match_id", "target_id"]]
    folds = grouped_folds(rows, 5, 0)
    with pytest.raises(LeakageError):
        cross_validate(data, folds, tiny_config(), spatial=False, final_test=[100])


def test_config_keeps_production_taxonomy():
    config = load_config()
    assert set(config.production_event_types) == set(TYPES)
    assert config.tasks["fullstream9"].vocabulary[-1] == "Other"
    assert config.excluded_periods == (5,)
