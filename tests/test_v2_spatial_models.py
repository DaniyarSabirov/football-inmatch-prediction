"""Pitch thirds, backoff Markov chains and the categorical spatial HMM (experiment 2)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from src.sequence_models.backoff_markov import BackoffMarkov, build_contexts, context_cardinalities, targets_of
from src.sequence_models.categorical_hmm import CategoricalHMM, CategoricalHMMError
from src.sequence_models.hmm import HiddenMarkovModel
from src.sequence_models.zones import ZONE_UNKNOWN, ZoneError, zone_of_x


# ------------------------------------------------------------------ pitch thirds
def test_pitch_third_boundaries():
    x = np.array([0.0, 39.999, 40.0, 79.999, 80.0, 120.0, 120.5, -0.5, np.nan])
    assert zone_of_x(x).tolist() == [0, 0, 1, 1, 2, 2, 2, 0, ZONE_UNKNOWN]
    for bad in (121.5, -1.5):
        with pytest.raises(ZoneError):
            zone_of_x(np.array([bad]))


# ------------------------------------------------------------------ backoff Markov
SEQS = [np.array([0, 1, 0, 1, 1]), np.array([1, 0])]
ZONES = [np.array([0, 0, 2, 2, -1]), np.array([1, 1])]


def test_contexts_respect_sequence_boundaries_and_start_token():
    ctx = build_contexts(SEQS, ZONES, 2, 3)
    assert ctx["prev"].tolist() == [2, 0, 1, 0, 1, 2, 1]  # START = 2 at each sequence start
    assert ctx["zone"].tolist() == [0, 0, 2, 2, 3, 1, 1]  # UNKNOWN = 3
    assert targets_of(SEQS).tolist() == [1, 0, 1, 1, -1, 0, -1]


def test_first_order_hierarchical_smoothing_by_hand():
    ctx = build_contexts(SEQS, None, 2, 3)
    m = BackoffMarkov("mc1", 2, context_cardinalities(2, 3), concentration=2.0, base_alpha=1.0).fit(ctx, targets_of(SEQS))
    # targets: 1,0,1,1,0 -> base (2+1, 3+1)/(5+2) = (3/7, 4/7)
    # from 0: next 1,1 -> C=(0,2); from 1: next 0,1,0 -> C=(2,1)
    p = m.predict_proba({"cur": np.array([0, 1])})
    base = np.array([3, 4]) / 7
    np.testing.assert_allclose(p[0], (np.array([0, 2]) + 2 * base) / (2 + 2))
    np.testing.assert_allclose(p[1], (np.array([2, 1]) + 2 * base) / (3 + 2))
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_zone_context_backs_off_to_its_parent_when_unseen():
    ctx = build_contexts(SEQS, ZONES, 2, 3)
    t = targets_of(SEQS)
    zone = BackoffMarkov("mc1_zone", 2, context_cardinalities(2, 3), concentration=5.0).fit(ctx, t)
    first = BackoffMarkov("mc1", 2, context_cardinalities(2, 3), concentration=5.0).fit(ctx, t)
    unseen = {"cur": np.array([0]), "zone": np.array([1])}  # (cur 0, zone 1) never seen
    np.testing.assert_allclose(zone.predict_proba(unseen), first.predict_proba({"cur": np.array([0])}))
    # a seen context: (cur 0, zone 0) -> next 1 once
    seen = zone.predict_proba({"cur": np.array([0]), "zone": np.array([0])})
    parent = first.predict_proba({"cur": np.array([0])})
    np.testing.assert_allclose(seen, (np.array([[0, 1]]) + 5 * parent) / (1 + 5))


def test_unknown_context_value_raises():
    m = BackoffMarkov("mc1", 2, context_cardinalities(2, 3)).fit(build_contexts(SEQS, None, 2, 3), targets_of(SEQS))
    with pytest.raises(ValueError):
        m.predict_proba({"cur": np.array([2])})


def test_backoff_serialisation_round_trip():
    ctx = build_contexts(SEQS, ZONES, 2, 3)
    m = BackoffMarkov("mc2_zone", 2, context_cardinalities(2, 3), 3.0).fit(ctx, targets_of(SEQS))
    np.testing.assert_array_equal(BackoffMarkov.from_dict(m.to_dict()).predict_proba(ctx), m.predict_proba(ctx))


# ------------------------------------------------------------------ categorical HMM
PI = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.4, 0.6]])
BE = np.array([[0.9, 0.1], [0.2, 0.8]])
BZ = np.array([[0.5, 0.3, 0.2], [0.1, 0.3, 0.6]])
BJ = np.array([[[0.3, 0.3, 0.2], [0.1, 0.05, 0.05]], [[0.05, 0.05, 0.1], [0.2, 0.2, 0.4]]])


def model(emission: str) -> CategoricalHMM:
    m = CategoricalHMM(2, 2, emission=emission, n_zones=3 if emission != "event" else 0)
    m.startprob, m.transmat = PI.copy(), A.copy()
    m.event_emission, m.zone_emission, m.joint_emission = BE.copy(), BZ.copy(), BJ.copy()
    return m


def brute(emission: str, ev: list[int], zn: list[int]) -> float:
    def emit(s, e, z):
        if emission == "factorized":
            return BE[s, e] * (BZ[s, z] if z >= 0 else 1.0)
        return BJ[s, e, z] if z >= 0 else BJ[s, e].sum()
    total = 0.0
    for path in itertools.product(range(2), repeat=len(ev)):
        p = PI[path[0]] * emit(path[0], ev[0], zn[0])
        for t in range(1, len(ev)):
            p *= A[path[t - 1], path[t]] * emit(path[t], ev[t], zn[t])
        total += p
    return total


@pytest.mark.parametrize("emission", ["factorized", "joint"])
def test_zone_hmm_likelihood_with_missing_zone_matches_enumeration(emission):
    ev, zn = [0, 1, 0, 1], [0, -1, 2, 1]
    got = math.exp(model(emission).score([np.array(ev)], [np.array(zn)]))
    assert got == pytest.approx(brute(emission, ev, zn), rel=1e-12)


@pytest.mark.parametrize("emission", ["factorized", "joint"])
def test_zone_hmm_predictive_marginalises_the_next_zone(emission):
    m = model(emission)
    p = m.predict_proba([np.array([0, 1])], [np.array([2, 0])])[0]
    for v in (0, 1):
        assert p[0, v] == pytest.approx(brute(emission, [0, v], [2, -1]) / brute(emission, [0], [2]), rel=1e-12)
    np.testing.assert_allclose(p.sum(axis=1), 1)


@pytest.mark.parametrize("emission", ["event", "factorized", "joint"])
def test_zone_hmm_has_no_future_leakage(emission):
    rng = np.random.default_rng(5)
    ev = rng.integers(0, 4, 60)
    zn = rng.integers(-1, 3, 60)
    m = CategoricalHMM(3, 4, emission=emission, n_zones=0 if emission == "event" else 3, seed=2, max_iter=15)
    m.fit([ev], None if emission == "event" else [zn])
    base = m.predict_proba([ev], None if emission == "event" else [zn])[0]
    ev2, zn2 = ev.copy(), zn.copy()
    ev2[31:], zn2[31:] = rng.integers(0, 4, 29), rng.integers(-1, 3, 29)
    changed = m.predict_proba([ev2], None if emission == "event" else [zn2])[0]
    np.testing.assert_array_equal(changed[:31], base[:31])


def test_event_only_is_identical_to_experiment_1_hmm():
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 5, n) for n in (200, 80, 33)]
    a = HiddenMarkovModel(3, 5, seed=4, max_iter=60, tol=1e-12).fit(seqs)
    b = CategoricalHMM(3, 5, seed=4, max_iter=60, tol=1e-12).fit(seqs)
    np.testing.assert_allclose(b.transmat, a.transmat, atol=1e-12)
    np.testing.assert_allclose(b.event_emission, a.emissionprob, atol=1e-12)
    np.testing.assert_allclose(np.concatenate(b.predict_proba(seqs)), np.concatenate(a.predict_proba(seqs)), atol=1e-12)


@pytest.mark.parametrize("emission", ["factorized", "joint"])
def test_zone_hmm_em_monotone_deterministic_and_round_trips(emission, tmp_path):
    rng = np.random.default_rng(3)
    ev = [rng.integers(0, 3, n) for n in (150, 90)]
    zn = [np.where(rng.random(len(e)) < 0.1, -1, rng.integers(0, 3, len(e))) for e in ev]
    fit = lambda: CategoricalHMM(3, 3, emission=emission, n_zones=3, seed=1, max_iter=80).fit(ev, zn)  # noqa: E731
    a, b = fit(), fit()
    assert a.record.monotonic
    np.testing.assert_array_equal(a.transmat, b.transmat)
    a.save(tmp_path / "m.json")
    np.testing.assert_array_equal(np.concatenate(CategoricalHMM.load(tmp_path / "m.json").predict_proba(ev, zn)),
                                  np.concatenate(a.predict_proba(ev, zn)))
    np.testing.assert_allclose(a.event_marginal_emission().sum(axis=1), 1)


def test_zone_hmm_input_validation():
    with pytest.raises(CategoricalHMMError):
        CategoricalHMM(2, 2, emission="joint", n_zones=0)
    with pytest.raises(CategoricalHMMError):
        model("factorized").predict_proba([np.array([0, 1])])  # zones missing
    with pytest.raises(CategoricalHMMError):
        model("factorized").predict_proba([np.array([0, 2])], [np.array([0, 0])])  # unknown event
