"""Tests for the 15-class taxonomy, the F0 feature set, the multinomial logistic regression
and the experiment-4 protocol (synthetic data only; the real experiment is not run here)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sequence_models import f0_experiment as fx
from src.sequence_models.config_v2 import load_config_v2
from src.sequence_models.event_classes import (
    CLASS_TO_ID,
    HISTORICAL_FILTERED8,
    ID_TO_CLASS,
    K,
    SCENARIO_D15_CLASSES,
    TaxonomyError,
    assert_canonical_vocabulary,
    class_id,
    class_name,
)
from src.sequence_models.evaluation_v2 import confusion_counts, metrics_from_confusion, probability_metrics
from src.sequence_models.feature_sets import (
    F0,
    FORBIDDEN_COLUMNS,
    REGISTRY,
    FeatureLeakageError,
    FeatureSet,
    design_matrix,
    one_hot,
)
from src.sequence_models.logistic import MultinomialLogisticModel, OneHotLogisticChain, aggregate_one_hot
from src.sequence_models.markov import MarkovChain
from src.sequence_models.metrics import validate_probabilities
from src.sequence_models.splits import FoldAssignment, LeakageError, assert_no_final_test, verify_folds
from src.sequence_models.v3_taxonomy import D15
from src.utils.paths import PROJECT_ROOT

EXPECTED = ("Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed",
            "Pressure", "Duel", "Interception", "Block", "Goal Keeper", "Foul Committed", "50/50")


# ----------------------------------------------------------------------------- taxonomy
def test_canonical_order_and_ids():
    assert K == 15
    assert SCENARIO_D15_CLASSES == EXPECTED
    assert all(CLASS_TO_ID[c] == i and ID_TO_CLASS[i] == c for i, c in enumerate(EXPECTED))
    assert class_id("Shot") == 5 and class_name(5) == "Shot"
    assert HISTORICAL_FILTERED8 == EXPECTED[:8]
    with pytest.raises(TaxonomyError):
        class_id("Ball Receipt*")
    with pytest.raises(TaxonomyError):
        class_name(15)


def test_taxonomy_agrees_with_every_existing_definition():
    cfg2 = load_config_v2()
    assert tuple(cfg2.tasks["scenarioD15"].vocabulary) == SCENARIO_D15_CLASSES       # experiment 2 (order)
    assert tuple(D15) == SCENARIO_D15_CLASSES                                        # experiment 3 (order)
    research = json.loads((PROJECT_ROOT / "results" / "event_taxonomy_research_2026-10-27" / "scenarios.json")
                          .read_text(encoding="utf-8"))["D_expanded_selected"]
    assert set(research["target_types"]) == set(SCENARIO_D15_CLASSES)                 # research (set only)
    from src.build_statsbomb_360_dataset import MEANINGFUL_EVENT_TYPES
    assert set(HISTORICAL_FILTERED8) == set(MEANINGFUL_EVENT_TYPES)


def test_assert_canonical_vocabulary_rejects_reordering():
    assert_canonical_vocabulary(EXPECTED)
    with pytest.raises(TaxonomyError, match="different order"):
        assert_canonical_vocabulary(tuple(sorted(EXPECTED)))
    with pytest.raises(TaxonomyError):
        assert_canonical_vocabulary(EXPECTED[:8])


# ----------------------------------------------------------------------------- features
def test_one_hot_encoding():
    x = one_hot(np.array([0, 5, 14, 5]))
    assert x.shape == (4, 15)
    assert np.array_equal(x.sum(axis=1), np.ones(4))
    assert x[1, 5] == 1 and x[2, 14] == 1 and x[0, 0] == 1
    with pytest.raises(ValueError):
        one_hot(np.array([15]))
    with pytest.raises(ValueError):
        one_hot(np.array([-1]))


def test_f0_reads_only_the_current_event():
    frame = pd.DataFrame({"symbol_id": [0, 1, 5], "target_id": [1, 5, 0], "match_id": [1, 1, 1],
                          "location_x": [10.0, 50.0, 100.0], "n_skipped_raw_events": [0, 3, 1]})
    x, names = design_matrix(frame, "F0")
    assert x.shape == (3, 15) and names[5] == "E_t=Shot"
    # changing every non-input column (incl. the target) leaves X unchanged
    other = frame.assign(target_id=[9, 9, 9], match_id=[7, 7, 7], location_x=0.0, n_skipped_raw_events=99)
    assert np.array_equal(design_matrix(other, "F0")[0], x)
    assert F0.input_columns == ("symbol_id",)
    assert not set(F0.input_columns) & FORBIDDEN_COLUMNS


def test_feature_set_cannot_declare_target_columns():
    for col in ("target_id", "target", "n_skipped_raw_events", "match_id", "possession"):
        with pytest.raises(FeatureLeakageError):
            FeatureSet("bad", "", ("symbol_id", col), "planned")
    for name, fs in REGISTRY.items():
        assert not set(fs.input_columns) & FORBIDDEN_COLUMNS, name


def test_planned_feature_sets_are_not_buildable():
    frame = pd.DataFrame({"symbol_id": [0], "location_x": [1.0], "location_y": [1.0]})
    with pytest.raises(NotImplementedError):
        design_matrix(frame, "F1")


# ----------------------------------------------------------------------------- model
def _synthetic(n=6000, seed=0, absent_targets=(), concentration=0.7):
    """Sequences from a random 15-state Markov chain."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(K) * concentration, size=K)
    for c in absent_targets:
        p[:, c] = 0
    p /= p.sum(axis=1, keepdims=True)
    seqs = []
    for _ in range(n // 50):
        s = [int(rng.integers(0, K))]
        while s[-1] in absent_targets:
            s = [int(rng.integers(0, K))]
        for _ in range(49):
            s.append(int(rng.choice(K, p=p[s[-1]])))
        seqs.append(np.array(s))
    return seqs


def _rows(seqs):
    cur = np.concatenate([s[:-1] for s in seqs])
    nxt = np.concatenate([s[1:] for s in seqs])
    return cur, nxt


def test_probability_matrix_shape_sum_and_sign():
    cur, nxt = _rows(_synthetic())
    m = MultinomialLogisticModel(C=1.0).fit(one_hot(cur), nxt)
    p = m.predict_proba(one_hot(cur[:500]))
    assert p.shape == (500, 15)
    assert (p >= 0).all() and np.allclose(p.sum(axis=1), 1, atol=1e-12)
    validate_probabilities(p)
    assert m.coefficients().shape == (15, 15) and m.intercepts().shape == (15,)
    assert m.fit_info["converged"]


def test_softmax_of_logits_equals_predict_proba():
    cur, nxt = _rows(_synthetic())
    m = MultinomialLogisticModel(C=1.0).fit(one_hot(cur), nxt)
    x = np.eye(15)
    z = m.logits(x)
    soft = np.exp(z - z.max(axis=1, keepdims=True))
    soft /= soft.sum(axis=1, keepdims=True)
    assert np.allclose(soft, m.predict_proba(x), atol=1e-12)


def test_deterministic_outputs():
    cur, nxt = _rows(_synthetic())
    a = MultinomialLogisticModel(C=0.1).fit(one_hot(cur), nxt).predict_proba(np.eye(15))
    b = MultinomialLogisticModel(C=0.1).fit(one_hot(cur), nxt).predict_proba(np.eye(15))
    assert np.array_equal(a, b)


def test_unpenalised_f0_reproduces_empirical_transition_probabilities():
    """Theory (docs/thesis/02 section 6): with one-hot E_t and no penalty, softmax regression
    is the MLE of the first-order Markov chain, i.e. the empirical conditional frequencies."""
    seqs = _synthetic(n=40000, concentration=20.0)
    cur, nxt = _rows(seqs)
    counts = np.zeros((15, 15))
    np.add.at(counts, (cur, nxt), 1)
    assert (counts > 0).all()  # MLE exists only without zero cells
    mle = counts / counts.sum(axis=1, keepdims=True)
    lr = OneHotLogisticChain(C=np.inf, tol=1e-10, max_iter=10000).fit(seqs)
    assert np.abs(lr.transition_matrix() - mle).max() < 1e-5
    mc = MarkovChain(15, alpha=1.0).fit(seqs)
    assert np.abs(lr.transition_matrix() - mc.transition_matrix).max() < 0.01  # differ only by smoothing


def test_intercept_reproduces_training_marginal_for_any_penalty():
    """Stationarity in b: sum_a N_a p(c|a) = N pi_c (the intercept is not penalised)."""
    cur, nxt = _rows(_synthetic())
    for c_value in (1e-4, 1.0):
        m = MultinomialLogisticModel(C=c_value, tol=1e-10).fit(one_hot(cur), nxt)
        fitted = m.predict_proba(one_hot(cur)).mean(axis=0)
        empirical = np.bincount(nxt, minlength=15) / len(nxt)
        assert np.abs(fitted - empirical).max() < 1e-5


def test_aggregated_fit_equals_row_level_fit():
    cur, nxt = _rows(_synthetic())
    for c_value in (1e-4, 1.0):
        rows = MultinomialLogisticModel(C=c_value, tol=1e-10).fit(one_hot(cur), nxt)
        x, y, w = aggregate_one_hot(cur, nxt)
        assert w.sum() == len(cur)
        agg = MultinomialLogisticModel(C=c_value, tol=1e-10).fit(x, y, sample_weight=w)
        assert np.abs(rows.predict_proba(np.eye(15)) - agg.predict_proba(np.eye(15))).max() < 1e-6


def test_l2_stationarity_condition_and_rare_state_shrinkage():
    """First-order condition of sum_i -log p + ||W||^2/(2C):  W_{c,a} = C N_a (f_{a,c} - p_{a,c}).
    Hence |f_a - p_a| = |W_{.,a}| / (C N_a): for comparable weights, rarer states deviate more."""
    seqs = _synthetic(n=40000, concentration=5.0)
    cur, nxt = _rows(seqs)
    keep = (cur != 3) | (np.arange(len(cur)) % 20 == 0)  # make state 3 rare
    cur, nxt = cur[keep], nxt[keep]
    counts = np.zeros((15, 15))
    np.add.at(counts, (cur, nxt), 1)
    n_a = counts.sum(axis=1)
    f = counts / n_a[:, None]
    c_value = 1e-2
    m = MultinomialLogisticModel(C=c_value, tol=1e-12, max_iter=20000).fit(one_hot(cur), nxt)
    p = m.predict_proba(np.eye(15))
    implied = c_value * n_a[None, :] * (f - p).T          # (K classes c, K states a)
    assert np.abs(implied - m.coefficients()).max() < 1e-3
    assert n_a[3] < 0.1 * np.median(n_a)


def test_absent_target_class_gets_floor_probability():
    seqs = _synthetic(absent_targets=(14,))
    cur, nxt = _rows(seqs)
    assert 14 not in set(nxt)
    m = MultinomialLogisticModel(C=1.0, absent_class_floor=1e-9).fit(one_hot(cur), nxt)
    p = m.predict_proba(np.eye(15))
    assert p.shape == (15, 15) and np.allclose(p.sum(axis=1), 1)
    assert np.allclose(p[:, 14], 1e-9 / (1 + 1e-9))
    assert list(m.absent_classes) == [14]
    assert np.isnan(m.coefficients()[14]).all()


def test_unseen_current_event_predicts_intercept_only_distribution():
    seqs = _synthetic()
    cur, nxt = _rows(seqs)
    keep = cur != 7  # state 7 never seen as a current event
    m = MultinomialLogisticModel(C=1.0, tol=1e-10).fit(one_hot(cur[keep]), nxt[keep])
    assert np.allclose(m.coefficients()[:, 7], 0, atol=1e-8)
    b = m.intercepts()
    soft = np.exp(b - b.max())
    assert np.allclose(m.predict_proba(np.eye(15))[7], soft / soft.sum(), atol=1e-10)


def test_rejects_invalid_labels():
    with pytest.raises(ValueError):
        MultinomialLogisticModel().fit(np.eye(15)[:2], np.array([0, 15]))
    with pytest.raises(ValueError):
        MultinomialLogisticModel().fit(np.eye(15)[:2], np.array([3, 3]))


def test_chain_interface_matches_markov_interface():
    seqs = _synthetic()
    lr = OneHotLogisticChain(C=1.0).fit(seqs)
    mc = MarkovChain(15, 1.0).fit(seqs)
    out_lr, out_mc = lr.predict_proba(seqs[:3]), mc.predict_proba(seqs[:3])
    assert [a.shape for a in out_lr] == [a.shape for a in out_mc]
    with pytest.raises(ValueError):
        lr.predict_proba([np.array([0, 15])])


def test_compatible_with_existing_evaluation_utilities():
    seqs = _synthetic()
    cur, nxt = _rows(seqs)
    p = MultinomialLogisticModel(C=1.0).fit(one_hot(cur), nxt).predict_proba(one_hot(cur))
    pm = probability_metrics(nxt, p)
    dm = metrics_from_confusion(confusion_counts(nxt, p.argmax(axis=1), 15))
    assert np.isfinite(pm["log_loss"]) and 0 <= dm["accuracy"] <= 1


# ----------------------------------------------------------------------------- protocol
def test_markov_table_matches_markov_chain_including_unseen_rows():
    seqs = [s[s != 4] for s in _synthetic()]
    mc = MarkovChain(15, 1.0).fit(seqs)
    table = fx.markov_table(mc.counts.astype(float), 1.0)
    assert np.allclose(table[4], mc.fallback.probabilities)
    assert np.allclose(np.delete(table, 4, axis=0), np.delete(mc.transition_matrix, 4, axis=0))


def test_nested_choice_never_uses_the_outer_validation_fold():
    matches = np.arange(10)
    rng = np.random.default_rng(0)
    train_t = rng.integers(1, 5, size=(10, 15, 15)).astype(float)
    eval_t = train_t.copy()
    folds = [FoldAssignment(f, tuple(m for m in matches if m % 5 != f), tuple(m for m in matches if m % 5 == f))
             for f in range(5)]
    seen = []

    def score(tr, ev, v):
        seen.append(ev.sum())
        return 0.0, ev.sum()

    poisoned = eval_t.copy()
    poisoned[matches % 5 == 0] = 1e9  # outer validation fold 0
    fx.nested_choice(matches, train_t, poisoned, folds, folds[0], [1.0], score)
    assert max(seen) < 1e9


def test_tier_b_folds_partition_matches_and_exclude_final_test():
    cfg = fx.load_f0_config()
    folds = fx.load_folds(PROJECT_ROOT / cfg.raw["folds_file"], cfg.raw["taxonomy"])
    manifest = json.loads((PROJECT_ROOT / "results" / "markov_hmm_pilot" / "split_manifest.json").read_text())
    tier_b = sorted(set(manifest["development_match_ids"]) - set(manifest["pilot_match_ids"]))
    assert len(folds) == 5 and len(tier_b) == 304
    verify_folds(folds, tier_b)
    for f in folds:
        assert not set(f.train_match_ids) & set(f.validation_match_ids)
        assert_no_final_test(f.train_match_ids + f.validation_match_ids, manifest["final_test_match_ids"])
    with pytest.raises(LeakageError):
        assert_no_final_test([manifest["final_test_match_ids"][0]], manifest["final_test_match_ids"])


def test_config_refuses_other_taxonomies(tmp_path: Path):
    raw = (PROJECT_ROOT / "configs" / "logistic_regression_f0.yaml").read_text().replace(
        "taxonomy: scenarioD15", "taxonomy: filtered8")
    path = tmp_path / "c.yaml"
    path.write_text(raw)
    with pytest.raises(ValueError):
        fx.load_f0_config(path)
