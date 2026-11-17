"""Cost-sensitive decisions, decision/probability metrics, calibration and bootstrap
(experiment 2). Expected values are hand-computed or come from scikit-learn."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from src.sequence_models.cost_sensitive import (
    CostScheme,
    bayes_decisions,
    cost_matrix,
    expected_costs,
    miss_costs,
    prior_shifted,
    realised_cost,
    training_priors,
    validate_cost_matrix,
)
from src.sequence_models.evaluation_v2 import (
    bootstrap_multiplicities,
    bootstrap_summary,
    calibration,
    confusion_counts,
    metrics_from_confusion,
    per_class_from_confusion,
    per_match_stats,
    probability_metrics,
)

P = np.array([[0.6, 0.3, 0.1], [0.45, 0.1, 0.45], [0.2, 0.2, 0.6], [0.7, 0.05, 0.25]])
Y = np.array([0, 2, 2, 2])


def test_expected_cost_by_hand():
    c = np.array([[0, 1, 4], [1, 0, 4], [2, 2, 0]], dtype=float)  # C[a, k]
    r = expected_costs(P[:1], c)[0]
    # R(0) = 1*0.3 + 4*0.1 = 0.7; R(1) = 1*0.6 + 4*0.1 = 1.0; R(2) = 2*0.6 + 2*0.3 = 1.8
    np.testing.assert_allclose(r, [0.7, 1.0, 1.8])
    assert bayes_decisions(P[:1], c)[0] == 0


def test_uniform_scheme_is_argmax_and_ties_go_to_lowest_index():
    pri = training_priors(np.array([0, 0, 0, 1, 2]), 3)
    c = cost_matrix(pri, CostScheme("uniform", 0.0, 1.0))
    np.testing.assert_array_equal(c, 1 - np.eye(3))
    d = bayes_decisions(P, c)
    assert d.tolist() == [0, 0, 2, 0]  # row 1 is a tie 0.45/0.45 -> lowest index 0
    assert bayes_decisions(np.array([[1 / 3, 1 / 3, 1 / 3]]), c)[0] == 0


def test_miss_costs_from_training_priors_with_cap():
    # training targets: class 0 x 7, class 1 x 2, class 2 x 0 -> Laplace priors (8, 3, 1) / 12
    pri = training_priors(np.array([0] * 7 + [1] * 2), 3)
    np.testing.assert_allclose(pri, np.array([8, 3, 1]) / 12)
    m = miss_costs(pri, CostScheme("s", 1.0, 5.0))
    np.testing.assert_allclose(m, [1.0, 8 / 3, 5.0])  # 8/1 = 8 capped at 5
    m_half = miss_costs(pri, CostScheme("s", 0.5, 100.0))
    np.testing.assert_allclose(m_half, [1.0, math.sqrt(8 / 3), math.sqrt(8)])


def test_cost_matrix_structure_and_false_alarm_term():
    pri = np.array([0.5, 0.3, 0.2])
    c = cost_matrix(pri, CostScheme("s", 1.0, 10.0, false_alarm=0.5))
    validate_cost_matrix(c)
    # C[a, k] = m_k + f for a != k; m = (1, 5/3, 5/2)
    np.testing.assert_allclose(c, [[0, 5 / 3 + 0.5, 2.5 + 0.5], [1.5, 0, 3.0], [1.5, 5 / 3 + 0.5, 0]])
    for bad in (np.array([[1.0, 1], [1, 0]]), np.array([[0, -1.0], [1, 0]])):
        with pytest.raises(ValueError):
            validate_cost_matrix(bad)


def test_rare_cost_moves_decision_to_rare_class_by_the_threshold_rule():
    # with f = 0 the rule is argmax m_a p_a: class 2 preferred over 0 iff p2/p0 > m0/m2
    c = np.array([[0, 1, 4], [1, 0, 4], [1, 1, 0]], dtype=float)  # m = (1, 1, 4)
    assert bayes_decisions(np.array([[0.7, 0.1, 0.2]]), c)[0] == 2   # 0.2*4 = 0.8 > 0.7
    assert bayes_decisions(np.array([[0.7, 0.14, 0.16]]), c)[0] == 0  # 0.64 < 0.7


def test_realised_cost_by_hand():
    c = np.array([[0, 1, 4], [1, 0, 4], [1, 1, 0]], dtype=float)
    assert realised_cost(np.array([2, 0, 1]), np.array([0, 0, 2]), c) == pytest.approx((4 + 0 + 1) / 3)


def test_approach_b_weighted_training_equals_approach_a_decisions_but_shifts_probabilities():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5) * 0.7, size=500)
    w = np.array([1.0, 1.5, 3.0, 6.0, 12.0])
    c = np.tile(w, (5, 1))
    np.fill_diagonal(c, 0.0)
    q = prior_shifted(p, w)
    np.testing.assert_array_equal(q.argmax(axis=1), bayes_decisions(p, c))
    np.testing.assert_allclose(q.sum(axis=1), 1)
    assert np.abs(q - p).max() > 0.1  # probabilities are distorted


def test_decision_metrics_match_sklearn_and_hand_values():
    y = np.array([0, 0, 1, 2, 2, 2])
    d = np.array([0, 1, 1, 2, 0, 0])
    m = metrics_from_confusion(confusion_counts(y, d, 3))
    assert m["accuracy"] == pytest.approx(3 / 6)
    assert m["macro_f1"] == pytest.approx(f1_score(y, d, average="macro"))
    assert m["weighted_f1"] == pytest.approx(f1_score(y, d, average="weighted"))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, d))
    pc = per_class_from_confusion(confusion_counts(y, d, 3), ["a", "b", "c"]).set_index("class")
    # class c: support 3, predicted 1, TP 1 -> P 1, R 1/3, FN 2; prevalence 0.5, lift 2
    assert (pc.at["c", "true_positives"], pc.at["c", "false_positives"], pc.at["c", "false_negatives"]) == (1, 0, 2)
    assert pc.at["c", "lift"] == pytest.approx(2.0)
    # a class never predicted: precision 0 and F1 0 (zero_division=0), as scikit-learn
    pc2 = per_class_from_confusion(confusion_counts(np.array([0, 1]), np.array([0, 0]), 2), ["a", "b"]).set_index("class")
    assert pc2.at["b", "precision"] == 0 and pc2.at["b", "f1"] == 0


def test_log_loss_and_brier_by_hand():
    pm = probability_metrics(Y, P)
    assert pm["log_loss"] == pytest.approx(-(math.log(0.6) + math.log(0.45) + math.log(0.6) + math.log(0.25)) / 4)
    assert pm["log_loss"] == pytest.approx(log_loss(Y, P, labels=[0, 1, 2]))
    rows = [0.16 + 0.09 + 0.01, 0.2025 + 0.01 + 0.3025, 0.04 + 0.04 + 0.16, 0.49 + 0.0025 + 0.5625]
    assert pm["brier"] == pytest.approx(sum(rows) / 4)


def test_probability_metrics_do_not_depend_on_the_decision_rule():
    """When only the decision rule changes, log loss / Brier / calibration are identical."""
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(4), size=300)
    y = rng.integers(0, 4, 300)
    pri = training_priors(rng.integers(0, 4, 1000), 4)
    before = (probability_metrics(y, p), calibration(y, p, list("abcd"))[0])
    decisions = {s.name: bayes_decisions(p, cost_matrix(pri, s)) for s in
                 (CostScheme("u", 0, 1), CostScheme("m", 0.5, 5), CostScheme("s", 1, 20))}
    assert len({tuple(d) for d in decisions.values()}) > 1  # the rules really differ
    after = (probability_metrics(y, p), calibration(y, p, list("abcd"))[0])
    assert before == after


def test_calibration_of_a_perfectly_calibrated_bin_and_a_biased_one():
    # class-0 probability 0.25 for 4 rows of which exactly 1 is class 0 -> calibrated
    p = np.array([[0.25, 0.75]] * 4)
    y = np.array([0, 1, 1, 1])
    summary, table = calibration(y, p, ["a", "b"])
    assert summary["classwise_ece"] == pytest.approx(0.0)
    assert summary["citl_a"] == pytest.approx(0.0)
    biased, _ = calibration(np.array([1, 1, 1, 1]), p, ["a", "b"])
    assert biased["citl_a"] == pytest.approx(0.25) and biased["classwise_ece"] == pytest.approx(0.25)
    # top-label ECE: confidence 0.75, accuracy 0.75 (calibrated case)
    assert summary["top_label_ece"] == pytest.approx(0.0)


def test_match_bootstrap_from_sufficient_statistics_equals_direct_recomputation():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(6), 25)
    y = rng.integers(0, 3, 150)
    p = rng.dirichlet(np.ones(3), size=150)
    d = p.argmax(axis=1)
    stats = per_match_stats(y, p, {"uniform": d}, groups, 3)
    mult = bootstrap_multiplicities(6, 5, 3)
    reps = bootstrap_summary(stats, "uniform", mult)
    for b in range(5):
        idx = np.concatenate([np.repeat(np.flatnonzero(groups == g), int(mult[b, g])) for g in range(6)])
        direct = metrics_from_confusion(confusion_counts(y[idx], d[idx], 3))
        prob = probability_metrics(y[idx], p[idx])
        np.testing.assert_allclose(reps[b], [direct["accuracy"], direct["macro_f1"], direct["weighted_f1"],
                                             direct["balanced_accuracy"], prob["log_loss"], prob["brier"]], rtol=1e-12)
