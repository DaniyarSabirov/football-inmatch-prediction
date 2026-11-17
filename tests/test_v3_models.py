"""Experiment 3: context Markov specs, nested CV leakage, logistic regression, Shot cost layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sequence_models.backoff_markov import BackoffMarkov, build_contexts, context_cardinalities, targets_of
from src.sequence_models.splits import FoldAssignment
from src.sequence_models.v3_experiment import shot_cost_matrix, shot_metrics, threshold_for_budget
from src.sequence_models.v3_models import Checkpoint, MarkovSpec, cv_lr, fit_predict_markov, nested_markov


def synthetic(n_matches: int = 8, length: int = 120, seed: int = 0, v: int = 5) -> tuple[pd.DataFrame, list]:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        sym = rng.integers(0, v, length)
        for i in range(length):
            rows.append({"match_id": 100 + m, "sequence_id": f"{m}_P1", "symbol_id": sym[i],
                         "z3": int(rng.integers(-1, 3)), "poss_dur4": int(rng.integers(-1, 4)),
                         "eval_eligible": bool(i < length - 1 and rng.random() < 0.9),
                         "target_id": int(sym[i + 1]) if i < length - 1 else -1})
    t = pd.DataFrame(rows)
    t["cur"] = t["symbol_id"]
    prev = t.groupby("sequence_id")["symbol_id"].shift(1).fillna(v).astype(int)
    t["prev1"] = prev
    t["prev1_reset"] = prev
    for c in ("x", "lateral", "distance_to_goal", "shooting_angle", "danger_area", "penalty_area", "pf_distance_to_goal",
              "pf_shooting_angle", "pf_danger_area", "log_possession_duration",
              "log_possession_events", "possession_progression", "same_team_as_previous", "recent_possession_change",
              "log_time_since_previous", "events_last_10s", "minute_scaled"):
        t[c] = rng.normal(size=len(t))
    t["prev2"] = v
    t["period"] = 1
    folds = [FoldAssignment(i, tuple(m for m in range(100, 100 + n_matches) if m % 4 != i),
                            tuple(m for m in range(100, 100 + n_matches) if m % 4 == i)) for i in range(4)]
    return t, folds


def test_spec_levels_drop_features_from_the_end():
    s = MarkovSpec("x", ("cur", "z3", "prev1", "poss_dur4"))
    assert s.levels() == (("cur", "z3", "prev1", "poss_dur4"), ("cur", "z3", "prev1"), ("cur", "z3"), ("cur",))
    with pytest.raises(ValueError):
        MarkovSpec("bad", ("z3", "cur"))


def test_reference_spec_reproduces_experiment_2_mc2_zone():
    t, _ = synthetic()
    v = 5
    spec = MarkovSpec("reference", ("cur", "z3", "prev1"))
    _, p3 = fit_predict_markov(spec, t, t, v, concentration=16.0)
    seqs = [g["symbol_id"].to_numpy() for _, g in t.groupby("sequence_id", sort=False)]
    zones = [g["z3"].to_numpy() for _, g in t.groupby("sequence_id", sort=False)]
    ctx = build_contexts(seqs, zones, v, 3)
    m2 = BackoffMarkov("mc2_zone", v, context_cardinalities(v, 3), 16.0).fit(ctx, targets_of(seqs))
    np.testing.assert_allclose(p3, m2.predict_proba(ctx), atol=1e-15)


def test_nested_markov_has_no_validation_leakage():
    t, folds = synthetic()
    spec = MarkovSpec("p", ("cur", "z3", "prev1", "poss_dur4"))
    p, ch, _ = nested_markov(spec, t, folds, 5, [1.0, 16.0])
    t2 = t.copy()
    val = t2["match_id"].isin(folds[0].validation_match_ids)
    t2.loc[val, "target_id"] = np.where(t2.loc[val, "target_id"] >= 0, 0, -1)  # rewrite fold-0 validation labels
    p2, ch2, _ = nested_markov(spec, t2, folds, 5, [1.0, 16.0])
    rows = t.loc[t["eval_eligible"], "match_id"].isin(folds[0].validation_match_ids).to_numpy()
    np.testing.assert_array_equal(p[rows], p2[rows])
    assert ch[0] == ch2[0]
    assert np.allclose(p.sum(axis=1), 1) and not np.isnan(p).any()


def test_logistic_regression_cv_is_valid_and_ignores_validation_labels():
    t, folds = synthetic(n_matches=4, length=80)
    p, info = cv_lr(t, folds, 5, 1.0)
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all()
    t2 = t.copy()
    val = t2["match_id"].isin(folds[0].validation_match_ids)
    t2.loc[val, "target_id"] = np.where(t2.loc[val, "target_id"] >= 0, 0, -1)
    p2, _ = cv_lr(t2, folds, 5, 1.0)
    rows = t.loc[t["eval_eligible"], "match_id"].isin(folds[0].validation_match_ids).to_numpy()
    np.testing.assert_allclose(p[rows], p2[rows])


def test_lr_design_columns_do_not_depend_on_missingness():
    from src.sequence_models.v3_models import lr_design
    t, _ = synthetic(n_matches=2, length=20)
    a, names_a = lr_design(t, 5)
    t2 = t.copy()
    t2.loc[t2.index[:3], "distance_to_goal"] = np.nan
    b, names_b = lr_design(t2, 5)
    assert names_a == names_b and a.shape == b.shape


def test_shot_cost_matrix_multiplies_only_the_shot_miss_cost():
    pri = np.array([0.6, 0.3, 0.1])
    c1 = shot_cost_matrix(pri, 2, 0.5, 5.0, 1.0)
    c2 = shot_cost_matrix(pri, 2, 0.5, 5.0, 2.0)
    np.testing.assert_allclose(c1[:, :2], c2[:, :2])
    assert c2[0, 2] == pytest.approx(2 * c1[0, 2]) and c1[0, 2] == pytest.approx(min(5.0, (0.6 / 0.1) ** 0.5))
    assert (np.diag(c2) == 0).all()


def test_threshold_for_budget_and_shot_metrics_by_hand():
    y = np.array([2, 0, 0, 0, 2, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.05, 0.7, 0.0, 0.0, 0.0])
    tau = threshold_for_budget(y, score, 2, fp_per_1000=100)  # 1 false positive allowed in 10 rows
    pred = score >= tau
    assert ((pred) & (y != 2)).sum() == 1 and ((pred) & (y == 2)).sum() == 1
    m = shot_metrics(y, None, np.where(pred, 2, 0), 2)
    assert m["shot_recall"] == 0.5 and m["shot_precision"] == 0.5 and m["shot_fp_per_1000_events"] == 100


def test_checkpoint_round_trip(tmp_path):
    ck = Checkpoint(tmp_path)
    assert ck.load("m", "k") is None
    ck.save("m", "k", np.eye(3), {"a": 1})
    p, meta = ck.load("m", "k")
    np.testing.assert_array_equal(p, np.eye(3)) and meta == {"a": 1}
