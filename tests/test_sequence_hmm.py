"""HMM likelihood, causal filtering and training (src/sequence_models/hmm.py).

Expected values are computed by hand or by brute-force path enumeration,
independently of the forward recursion under test.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.stats import norm

from src.sequence_models.hmm import HiddenMarkovModel, HMMError

PI = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.9, 0.1], [0.2, 0.8]])


def fixed_model(n_features: int = 0) -> HiddenMarkovModel:
    model = HiddenMarkovModel(n_states=2, n_symbols=2, n_features=n_features)
    model.startprob, model.transmat, model.emissionprob = PI.copy(), A.copy(), B.copy()
    if n_features:
        model.means = np.array([[0.0], [3.0]])
        model.variances = np.array([[1.0], [4.0]])
    return model


def enumerate_joint(obs: list[int], xs: list[float] | None = None) -> float:
    """Brute force: sum over all state paths of the joint probability."""
    total = 0.0
    for path in itertools.product(range(2), repeat=len(obs)):
        p = PI[path[0]]
        for t, z in enumerate(path):
            if t:
                p *= A[path[t - 1], z]
            p *= B[z, obs[t]]
            if xs is not None and not math.isnan(xs[t]):
                p *= norm.pdf(xs[t], loc=[0.0, 3.0][z], scale=[1.0, 2.0][z])
        total += p
    return total


def test_likelihood_matches_hand_computed_value():
    # alpha_1 = [0.54, 0.08]; alpha_2 = [0.041, 0.168]; alpha_3 = [0.08631, 0.02262] -> P = 0.10893
    model = fixed_model()
    assert math.isclose(math.exp(model.score([np.array([0, 1, 0])])), 0.10893, rel_tol=1e-12)
    assert math.isclose(enumerate_joint([0, 1, 0]), 0.10893, rel_tol=1e-12)


def test_one_step_predictive_matches_hand_computed_value():
    # filtered after o1=0: [0.54, 0.08]/0.62; next state [0.661290, 0.338710];
    # P(e2=0 | o1) = 0.661290*0.9 + 0.338710*0.2 = 0.662903
    model = fixed_model()
    p = model.predict_proba([np.array([0, 1, 0])])[0]
    assert p.shape == (3, 2)
    assert p[0, 0] == pytest.approx(0.6629032258, abs=1e-9)
    assert p[0, 1] == pytest.approx(0.3370967742, abs=1e-9)
    # independent: P(o1=0, o2=v) / P(o1=0) by enumeration
    for v in (0, 1):
        assert p[0, v] == pytest.approx(enumerate_joint([0, v]) / enumerate_joint([0]), rel=1e-12)
    # and at t=2: P(o3 | o1, o2)
    for v in (0, 1):
        assert p[1, v] == pytest.approx(enumerate_joint([0, 1, v]) / enumerate_joint([0, 1]), rel=1e-12)


def test_filtered_posterior_matches_enumeration():
    model = fixed_model()
    f = model.filter([np.array([0, 1, 0])])[0]
    assert f[0] == pytest.approx([0.54 / 0.62, 0.08 / 0.62])
    assert f[2] == pytest.approx(np.array([0.08631, 0.02262]) / 0.10893)
    assert np.allclose(f.sum(axis=1), 1)


def test_mixed_emission_likelihood_and_missing_feature_marginalisation():
    model = fixed_model(n_features=1)
    obs, xs = [0, 1, 0], [0.5, float("nan"), 2.0]
    got = math.exp(model.score([np.array(obs)], [np.array(xs)[:, None]]))
    assert got == pytest.approx(enumerate_joint(obs, xs), rel=1e-10)
    # all features missing == categorical-only likelihood
    all_missing = math.exp(model.score([np.array(obs)], [np.full((3, 1), np.nan)]))
    assert all_missing == pytest.approx(enumerate_joint(obs), rel=1e-12)


def test_mixed_model_predictive_marginalises_next_spatial_feature():
    # p(e_{t+1} | o_1..t) must equal the joint over (e_{t+1}, x_{t+1}) integrated over x:
    # with x_{t+1} missing, P(o1, o2=(v, NaN)) / P(o1).
    model = fixed_model(n_features=1)
    p = model.predict_proba([np.array([0, 1])], [np.array([[0.5], [9.9]])])[0]
    for v in (0, 1):
        assert p[0, v] == pytest.approx(enumerate_joint([0, v], [0.5, float("nan")]) / enumerate_joint([0], [0.5]))


def test_predictions_do_not_depend_on_future_observations():
    rng = np.random.default_rng(3)
    model = HiddenMarkovModel(n_states=3, n_symbols=4, n_features=2, seed=1)
    model.startprob = rng.dirichlet(np.ones(3))
    model.transmat = rng.dirichlet(np.ones(3), size=3)
    model.emissionprob = rng.dirichlet(np.ones(4), size=3)
    model.means, model.variances = rng.normal(size=(3, 2)), rng.uniform(0.5, 2, size=(3, 2))
    sym = rng.integers(0, 4, size=50)
    x = rng.normal(size=(50, 2))
    base = model.predict_proba([sym], [x])[0]
    for cut in (0, 10, 37):
        sym2, x2 = sym.copy(), x.copy()
        sym2[cut + 1:] = rng.integers(0, 4, size=50 - cut - 1)
        x2[cut + 1:] = rng.normal(size=(50 - cut - 1, 2)) * 10
        changed = model.predict_proba([sym2], [x2])[0]
        np.testing.assert_array_equal(changed[: cut + 1], base[: cut + 1])
    # truncation: the prefix alone gives the same predictions
    np.testing.assert_allclose(model.predict_proba([sym[:20]], [x[:20]])[0], base[:20], rtol=0, atol=1e-15)


def test_batched_padding_equals_single_sequence_processing():
    model = fixed_model()
    seqs = [np.array([0, 1, 0, 0, 1, 1, 0]), np.array([1, 0]), np.array([0])]
    batched = model.predict_proba(seqs)
    for seq, pred in zip(seqs, batched):
        np.testing.assert_allclose(model.predict_proba([seq])[0], pred, atol=1e-15)
    assert model.score(seqs) == pytest.approx(sum(model.score([s]) for s in seqs))


def test_independent_sequences_are_not_concatenated():
    model = fixed_model()
    a, b = np.array([0, 1, 0]), np.array([1, 1])
    separate = model.score([a, b])
    assert separate == pytest.approx(math.log(enumerate_joint([0, 1, 0]) * enumerate_joint([1, 1])))
    assert separate != pytest.approx(model.score([np.concatenate([a, b])]))


def sample_hmm(rng, pi, a, b, lengths):
    seqs = []
    for n in lengths:
        z = rng.choice(len(pi), p=pi)
        out = []
        for _ in range(n):
            out.append(rng.choice(b.shape[1], p=b[z]))
            z = rng.choice(len(pi), p=a[z])
        seqs.append(np.array(out))
    return seqs


def test_em_is_monotone_deterministic_and_recovers_structure():
    rng = np.random.default_rng(0)
    true_a = np.array([[0.95, 0.05], [0.1, 0.9]])
    true_b = np.array([[0.8, 0.15, 0.05], [0.05, 0.15, 0.8]])
    seqs = sample_hmm(rng, np.array([0.5, 0.5]), true_a, true_b, [300] * 10)
    fit = lambda: HiddenMarkovModel(2, 3, seed=4, max_iter=200, tol=1e-9).fit(seqs)  # noqa: E731
    m1, m2 = fit(), fit()
    hist = np.array(m1.record.history)
    assert m1.record.monotonic and np.all(np.diff(hist) >= -1e-8 * np.abs(hist[:-1]))
    np.testing.assert_array_equal(m1.transmat, m2.transmat)
    np.testing.assert_array_equal(m1.emissionprob, m2.emissionprob)
    order = np.argsort(m1.emissionprob[:, 0])[::-1]  # state emitting symbol 0 first
    np.testing.assert_allclose(m1.emissionprob[order], true_b, atol=0.06)
    np.testing.assert_allclose(m1.transmat[np.ix_(order, order)], true_a, atol=0.04)
    assert np.allclose(m1.transmat.sum(axis=1), 1) and np.allclose(m1.emissionprob.sum(axis=1), 1)


def test_serialisation_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    seqs = [rng.integers(0, 3, size=40) for _ in range(3)]
    feats = [np.where(rng.random((40, 2)) < 0.2, np.nan, rng.normal(size=(40, 2))) for _ in range(3)]
    model = HiddenMarkovModel(2, 3, n_features=2, seed=0, max_iter=20).fit(seqs, feats)
    model.save(tmp_path / "hmm.json")
    loaded = HiddenMarkovModel.load(tmp_path / "hmm.json")
    np.testing.assert_array_equal(np.concatenate(model.predict_proba(seqs, feats)),
                                  np.concatenate(loaded.predict_proba(seqs, feats)))
    assert loaded.record.seed == 0 and loaded.record.n_iter == model.record.n_iter


def test_unknown_symbol_and_missing_features_raise():
    model = fixed_model()
    with pytest.raises(HMMError):
        model.predict_proba([np.array([0, 2])])
    with pytest.raises(HMMError):
        fixed_model(n_features=1).predict_proba([np.array([0, 1])])


def test_probabilities_sum_to_one():
    rng = np.random.default_rng(2)
    seqs = [rng.integers(0, 5, size=n) for n in (30, 7, 1)]
    model = HiddenMarkovModel(3, 5, seed=2, max_iter=10).fit(seqs)
    for p in model.predict_proba(seqs):
        assert np.all(p > 0) and np.allclose(p.sum(axis=1), 1, atol=1e-12)


def test_viterbi_matches_brute_force():
    model = fixed_model()
    obs = [0, 1, 1, 0]
    best = max(itertools.product(range(2), repeat=4), key=lambda path: np.prod(
        [PI[path[0]] * B[path[0], obs[0]]] + [A[path[t - 1], path[t]] * B[path[t], obs[t]] for t in range(1, 4)]))
    assert tuple(model.viterbi(np.array(obs))) == best
