"""Markov chains, frequency baseline and metrics (hand-computed expectations)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import log_loss as sk_log_loss

from src.sequence_models.markov import (
    FrequencyBaseline,
    MarkovChain,
    SecondOrderMarkovChain,
    load_model,
    save_model,
    transition_counts,
)
from src.sequence_models.metrics import (
    ProbabilityError,
    brier_score,
    log_loss,
    match_bootstrap,
    per_class_metrics,
    summary_metrics,
    validate_probabilities,
)

SEQS = [np.array([0, 1, 1, 2]), np.array([2, 0])]


def test_transition_counts_by_hand_and_sequence_boundaries():
    # transitions: 0->1, 1->1, 1->2 (seq 1) and 2->0 (seq 2); the boundary 2 | 2 is NOT a transition
    expected = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 0]])
    np.testing.assert_array_equal(transition_counts(SEQS, 3), expected)
    assert transition_counts(SEQS, 3)[2, 2] == 0


def test_markov_probabilities_with_laplace_smoothing():
    mc = MarkovChain(3, alpha=1.0).fit(SEQS)
    np.testing.assert_allclose(mc.transition_matrix[0], [1 / 4, 2 / 4, 1 / 4])
    np.testing.assert_allclose(mc.transition_matrix[1], [1 / 5, 2 / 5, 2 / 5])
    np.testing.assert_allclose(mc.transition_matrix.sum(axis=1), 1)
    p = mc.predict_proba([np.array([1, 0])])[0]
    np.testing.assert_allclose(p, mc.transition_matrix[[1, 0]])


def test_unseen_state_falls_back_to_frequency_baseline():
    mc = MarkovChain(4, alpha=0.5).fit(SEQS)  # symbol 3 never a transition source
    assert mc.unseen_states().tolist() == [False, False, False, True]
    p = mc.predict_proba([np.array([3])])[0][0]
    # targets in training: 1, 1, 2, 0 -> counts [1, 2, 1, 0]; (c + 0.5) / (4 + 2)
    np.testing.assert_allclose(p, np.array([1.5, 2.5, 1.5, 0.5]) / 6)
    assert mc.unseen_state_mask([np.array([3, 0])])[0].tolist() == [True, False]


def test_unknown_category_raises():
    with pytest.raises(ValueError):
        MarkovChain(3).fit([np.array([0, 3])])
    with pytest.raises(ValueError):
        MarkovChain(3).fit(SEQS).predict_proba([np.array([-1])])


def test_frequency_baseline_by_hand():
    fb = FrequencyBaseline(3, alpha=1.0).fit(SEQS)
    np.testing.assert_allclose(fb.probabilities, np.array([2, 3, 2]) / 7)  # targets 1,1,2,0 + 1 each


def test_second_order_uses_context_and_backs_off():
    seqs = [np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])]
    mc2 = SecondOrderMarkovChain(3, alpha=0.0, min_context_count=2).fit(seqs)
    p = mc2.predict_proba([np.array([0, 1, 2])])[0]
    np.testing.assert_allclose(p[1], [0, 0, 1])  # context (0,1) seen 3x -> always 2
    np.testing.assert_allclose(p[0], mc2.first_order.predict_proba([np.array([0])])[0][0])  # no previous symbol
    sparse = SecondOrderMarkovChain(3, alpha=1.0, min_context_count=99).fit(seqs)
    np.testing.assert_allclose(sparse.predict_proba([np.array([0, 1])])[0],
                               sparse.first_order.predict_proba([np.array([0, 1])])[0])


def test_model_serialisation_round_trip(tmp_path):
    for model in (FrequencyBaseline(3).fit(SEQS), MarkovChain(3).fit(SEQS), SecondOrderMarkovChain(3, 1.0, 1).fit(SEQS)):
        save_model(model, tmp_path / "m.json")
        loaded = load_model(tmp_path / "m.json")
        np.testing.assert_array_equal(np.concatenate(model.predict_proba(SEQS)), np.concatenate(loaded.predict_proba(SEQS)))


Y = np.array([0, 1, 2, 1])
P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.5, 0.25, 0.25], [0.3, 0.3, 0.4]])


def test_log_loss_and_brier_by_hand():
    expected = -(math.log(0.7) + math.log(0.8) + math.log(0.25) + math.log(0.3)) / 4
    assert log_loss(Y, P) == pytest.approx(expected)
    assert log_loss(Y, P) == pytest.approx(sk_log_loss(Y, P, labels=[0, 1, 2]))
    rows = [0.09 + 0.04 + 0.01, 0.01 + 0.04 + 0.01, 0.25 + 0.0625 + 0.5625, 0.09 + 0.49 + 0.16]
    assert brier_score(Y, P) == pytest.approx(sum(rows) / 4)


def test_zero_probability_on_true_class_gives_infinite_log_loss():
    assert math.isinf(log_loss(np.array([0]), np.array([[0.0, 1.0]])))


def test_summary_and_per_class_metrics_by_hand():
    # argmax: [0, 1, 0, 2] vs true [0, 1, 2, 1] -> 2/4 correct
    m = summary_metrics(Y, P, 3)
    assert m["accuracy"] == 0.5
    # per class: c0 P=1/2 R=1 F1=2/3; c1 P=1 R=1/2 F1=2/3; c2 P=0 R=0 F1=0
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert m["weighted_f1"] == pytest.approx((1 * 2 / 3 + 2 * 2 / 3 + 1 * 0) / 4)
    assert m["balanced_accuracy"] == pytest.approx((1 + 0.5 + 0) / 3)
    pc = per_class_metrics(Y, P, ("a", "b", "c"))
    assert pc["support"].tolist() == [1, 2, 1] and pc["n_predicted"].tolist() == [2, 1, 1]


def test_validate_probabilities_rejects_invalid_rows():
    validate_probabilities(P)
    for bad in (np.array([[0.5, 0.6]]), np.array([[-0.1, 1.1]]), np.array([[np.nan, 1.0]])):
        with pytest.raises(ProbabilityError):
            validate_probabilities(bad)


def test_match_bootstrap_resamples_matches_and_paired_difference_of_identical_models_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    p = rng.dirichlet(np.ones(3), size=200)
    groups = np.repeat(np.arange(10), 20)
    res = match_bootstrap(y, p, groups, 3, 50, 1)
    assert (res["ci_low"] <= res["estimate"] + 1e-12).all() and (res["estimate"] <= res["ci_high"] + 1e-12).all()
    diff = match_bootstrap(y, p, groups, 3, 20, 1, p_reference=p)
    assert np.allclose(diff[["estimate", "ci_low", "ci_high"]].to_numpy(), 0)
