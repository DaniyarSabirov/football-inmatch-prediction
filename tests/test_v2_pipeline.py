"""Experiment 2 pipeline: taxonomy mapping, partitions, test-set refusal and leakage."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.sequence_models.config import TaskSpec
from src.sequence_models.config_v2 import load_config_v2
from src.sequence_models.cost_sensitive import CostScheme
from src.sequence_models.experiment_v2 import (
    EXPERIMENT1_ALL_MATCH_CACHE,
    Partition,
    decide,
    fold_priors,
    load_development_events,
    load_partition,
    prepare_task,
    run_markov,
)
from src.sequence_models.raw_events import EVENT_TABLE_COLUMNS
from src.sequence_models.splits import FoldAssignment, LeakageError, grouped_folds

D15 = ("Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed",
       "Pressure", "Duel", "Interception", "Block", "Goal Keeper", "Foul Committed", "50/50")
TASK_D = TaskSpec("scenarioD15", "", D15, "vocabulary_only", None, D15, (5,))


def events(match_id: int, rows: list[tuple[int, str, bool, float]]) -> pd.DataFrame:
    out = []
    for i, (period, typ, has_360, x) in enumerate(rows):
        out.append({c: None for c in EVENT_TABLE_COLUMNS} | {
            "match_id": match_id, "list_position": i, "index": i + 1, "event_id": f"{match_id}-{i}", "period": period,
            "timestamp": f"00:00:{i % 60:02d}.000", "t_seconds": float(i), "type": typ, "team_id": 1 + i % 2,
            "possession": 1, "possession_team_id": 1, "has_360": has_360, "location_x": x, "location_y": 40.0})
    return pd.DataFrame(out, columns=EVENT_TABLE_COLUMNS)


def test_config_taxonomies_and_class_order():
    config = load_config_v2()
    assert config.tasks["scenarioD15"].vocabulary == D15  # fixed order = class indices
    assert config.tasks["filtered8"].vocabulary == D15[:8]
    assert set(config.cost_schemes) == {"uniform", "moderate", "strong"}


def test_scenario_d_mapping_skips_companions_and_admin_and_excludes_shootout():
    ev = events(1, [(1, "Pass", True, 30.0), (1, "Ball Receipt*", True, 35.0), (1, "Pressure", True, 80.0),
                    (1, "Foul Committed", True, 50.0), (1, "Foul Won", True, 70.0), (1, "Substitution", False, None),
                    (1, "Dispossessed", True, 90.0), (1, "Duel", True, 30.0), (1, "Half End", False, None),
                    (5, "Shot", True, 108.0), (5, "Goal Keeper", True, 2.0)])
    data = prepare_task(ev, TASK_D)
    t = data.table
    assert t["raw_type"].tolist() == ["Pass", "Pressure", "Foul Committed", "Dispossessed", "Duel"]
    assert t["target"].tolist()[:4] == ["Pressure", "Foul Committed", "Dispossessed", "Duel"]
    assert t["n_skipped_raw_events"].tolist()[:4] == [1, 0, 2, 0]  # Ball Receipt*; Foul Won + Substitution
    assert t["zone"].tolist() == [0, 2, 1, 2, 0]
    assert 5 not in set(t["period"])


def test_partition_is_disjoint_and_reuses_experiment_1_split():
    part = load_partition(load_config_v2())
    assert not set(part.tier_a) & set(part.tier_b)
    assert not (set(part.tier_a) | set(part.tier_b)) & set(part.final_test)
    assert set(part.tier_a) | set(part.tier_b) == set(part.development)
    assert len(part.final_test) == 83 and len(part.tier_a) == 30 and len(part.tier_b) == 304


def test_final_test_matches_and_experiment_1_cache_are_refused():
    config = load_config_v2()
    part = load_partition(config)
    with pytest.raises(LeakageError):
        load_development_events(config, part, [part.final_test[0]])
    bad = dataclasses.replace(config, event_table_cache=EXPERIMENT1_ALL_MATCH_CACHE)
    with pytest.raises(LeakageError):
        load_development_events(bad, part, part.tier_a[:1])


def synthetic(n_matches: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    tables = []
    for m in range(n_matches):
        rows = [(p, str(rng.choice(["Pass", "Carry", "Pressure", "Shot", "Ball Receipt*"], p=[.4, .3, .15, .05, .1])),
                 bool(rng.random() < .9), float(rng.uniform(0, 120))) for p in (1, 2) for _ in range(60)]
        tables.append(events(100 + m, rows))
    ev = pd.concat(tables, ignore_index=True)
    data = prepare_task(ev, TASK_D)
    folds = grouped_folds(data.table.loc[data.eval_rows, ["match_id", "target_id"]], 4, 0)
    return data, folds


def small_config():
    config = load_config_v2()
    raw = {**config.raw, "markov": {**config.raw["markov"], "concentration_grid": [1, 16]}}
    return dataclasses.replace(config, raw=raw)


def test_markov_nested_selection_and_predictions_ignore_outer_validation_labels():
    data, folds = synthetic()
    config = small_config()
    proba, choices, _ = run_markov(data, folds, config)
    # change the LAST observation of every fold-0 validation sequence: it is only the target
    # of the previous row (and the input of a row without a target)
    val = set(folds[0].validation_match_ids)
    last = [r[-1] for s, r in data.seq_rows.items() if data.seq_match[s] in val]
    altered = dataclasses.replace(data, table=data.table.copy())
    sym = altered.table["symbol_id"].to_numpy().copy()
    sym[last] = (sym[last] + 1) % len(D15)
    altered.table["symbol_id"] = sym
    proba2, choices2, _ = run_markov(altered, folds, config)
    fold0 = [c for c in choices if c["fold_id"] == 0]
    assert fold0 == [c for c in choices2 if c["fold_id"] == 0]  # nested choice unaffected
    rows = data.table.loc[data.eval_rows, "match_id"].isin(val).to_numpy()
    for name in proba:
        np.testing.assert_array_equal(proba[name][rows], proba2[name][rows], err_msg=name)


def test_cost_priors_use_training_folds_only_and_decisions_ignore_labels():
    data, folds = synthetic()
    pri = fold_priors(data, folds)
    altered = dataclasses.replace(data, table=data.table.copy())
    val = set(folds[0].validation_match_ids)
    in_val = altered.table["match_id"].isin(val).to_numpy()
    altered.table.loc[in_val, "target_id"] = 0  # rewrite every fold-0 validation label
    assert np.array_equal(fold_priors(altered, folds)[0], pri[0])
    p = np.random.default_rng(0).dirichlet(np.ones(len(D15)), size=50)
    fold_of_eval = np.zeros(50, dtype=int)
    d1, _ = decide(p, fold_of_eval, {0: pri[0]}, CostScheme("s", 1.0, 20.0), 1e-12)
    d2, _ = decide(p, fold_of_eval, {0: fold_priors(altered, folds)[0]}, CostScheme("s", 1.0, 20.0), 1e-12)
    np.testing.assert_array_equal(d1, d2)


def test_markov_run_is_deterministic():
    data, folds = synthetic()
    a, _, _ = run_markov(data, folds, small_config())
    b, _, _ = run_markov(data, folds, small_config())
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
        np.testing.assert_allclose(a[name].sum(axis=1), 1)


def test_partition_rejects_overlapping_tiers():
    with pytest.raises(LeakageError):
        from src.sequence_models.splits import assert_no_final_test
        assert_no_final_test([1, 2], [2])
    assert Partition([9], [1, 2], [1], [FoldAssignment(0, (), (1,))], [2]).tier_b == [2]
