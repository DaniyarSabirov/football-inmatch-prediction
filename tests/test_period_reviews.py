"""Tests for period-scoped orientation review decisions.

Covers the registry's separate "period_reviews" list, its use by the
display plan, the on-figure provenance text, the importer, and the real
recorded decision for match 3901833 (France W. vs Morocco W., period 2).
A period review must never be read as a match calibration or as video
verification unless that evidence was supplied.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.animate_statsbomb_360_match import orientation_provenance_text
from src.broadcast_orientation import (
    PROVENANCE_PARTIAL,
    PROVENANCE_USER_CONFIRMED,
    PROVENANCE_VERIFIED,
    SOURCE_CAMERA_RELATIVE,
    PROVENANCE_USER_CONFIRMED_AND_DERIVED,
    SOURCE_DERIVED_FROM_USER_REVIEW,
    SOURCE_LAWS_RULE,
    SOURCE_USER_CONFIRMED,
    build_display_plan,
    infer_broadcast_orientation,
)
from src.physical_orientation_calibration import (
    DEFAULT_REGISTRY_PATH,
    CalibrationRegistryError,
    load_calibration_registry,
    load_period_reviews,
    resolve_usable_physical_reference_team_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _footprint(camera_high: bool):
    near_y, far_y = (70.0, 5.0) if camera_high else (10.0, 75.0)
    return [50, near_y, 20, far_y, 100, far_y, 70, near_y]


def _rotate(poly):
    return [120 - v if i % 2 == 0 else 80 - v for i, v in enumerate(poly)]


def _inference(team_a_high_by_period, votes=25):
    events, frames = [], []
    for period, high in team_a_high_by_period.items():
        if high is None:  # no camera evidence for this period
            events += [{"team": {"id": 1}, "period": period}, {"team": {"id": 2}, "period": period}]
            frames += [{}, {}]
            continue
        poly = _footprint(high)
        for _ in range(votes):
            events += [{"team": {"id": 1}, "period": period}, {"team": {"id": 2}, "period": period}]
            frames += [{"visible_area": poly}, {"visible_area": _rotate(poly)}]
    return infer_broadcast_orientation(events, frames, (1, 2))


def _write(tmp_path, payload) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


REVIEW = {
    "match_id": 1, "period": 2, "display_reference_team": "B", "evidence_type": "user_review_of_renders",
    "reviewed_by": "owner", "date_reviewed": "2026-10-23",
}


# --- registry ---------------------------------------------------------------

def test_period_reviews_are_loaded_separately_from_calibrations(tmp_path):
    path = _write(tmp_path, {"calibrations": [], "period_reviews": [REVIEW]})
    assert load_calibration_registry(path) == {}
    review = load_period_reviews(path)[1][2]
    assert review.display_reference_team == "B"
    assert review.independently_verified is False
    assert resolve_usable_physical_reference_team_name(1, load_calibration_registry(path)) is None


def test_missing_period_reviews_list_is_empty(tmp_path):
    assert load_period_reviews(_write(tmp_path, {"calibrations": []})) == {}
    assert load_period_reviews(tmp_path / "absent.json") == {}


@pytest.mark.parametrize("change", [
    {"evidence_type": "verified"},
    {"evidence_type": "independent_video"},  # claims video without a URL
    {"period": 6},
    {"period": True},
])
def test_invalid_period_reviews_fail_loudly(tmp_path, change):
    with pytest.raises(CalibrationRegistryError):
        load_period_reviews(_write(tmp_path, {"calibrations": [], "period_reviews": [{**REVIEW, **change}]}))


def test_duplicate_period_review_fails(tmp_path):
    with pytest.raises(CalibrationRegistryError):
        load_period_reviews(_write(tmp_path, {"calibrations": [], "period_reviews": [REVIEW, REVIEW]}))


def test_video_review_with_url_is_independent(tmp_path):
    entry = {**REVIEW, "evidence_type": "independent_video", "source_url": "https://example.org/v"}
    assert load_period_reviews(_write(tmp_path, {"period_reviews": [entry]}))[1][2].independently_verified


# --- display plan -------------------------------------------------------------

def test_confirmation_that_agrees_with_camera_keeps_other_periods_unchanged():
    inference = _inference({1: None, 2: False})  # P1 no evidence; P2 camera -> team 2 unrotated
    plan = build_display_plan(inference, (1, 2), [1, 2], fallback_period_1_reference=2, confirmed_references={2: 2})
    assert plan.reference_by_period() == {1: 1, 2: 2}
    assert plan.transforms[2].source == SOURCE_USER_CONFIRMED
    # Derived from the reviewed period by the halftime rule: lineage kept, never "confirmed".
    assert plan.transforms[1].source == SOURCE_DERIVED_FROM_USER_REVIEW
    assert plan.provenance == PROVENANCE_USER_CONFIRMED_AND_DERIVED


def test_rule_derivation_from_camera_period_stays_laws_rule():
    inference = _inference({1: None, 2: False})
    plan = build_display_plan(inference, (1, 2), [1, 2], fallback_period_1_reference=2)
    assert plan.transforms[1].source == SOURCE_LAWS_RULE
    assert plan.provenance == PROVENANCE_PARTIAL


def test_confirmation_contradicting_camera_flips_camera_derived_periods():
    inference = _inference({1: True, 2: False})  # camera: P1 team 1, P2 team 2
    plan = build_display_plan(inference, (1, 2), [1, 2], confirmed_references={2: 1})
    assert plan.reference_by_period() == {1: 2, 2: 1}
    assert plan.transforms[1].source == SOURCE_CAMERA_RELATIVE


def test_all_regular_periods_confirmed_gives_user_confirmed_provenance():
    inference = _inference({1: True, 2: False})
    plan = build_display_plan(inference, (1, 2), [1, 2], confirmed_references={1: 1, 2: 2})
    assert plan.provenance == PROVENANCE_USER_CONFIRMED


def test_verified_anchor_takes_priority_over_period_reviews():
    inference = _inference({1: True, 2: False})
    plan = build_display_plan(inference, (1, 2), [1, 2], anchor_period_1_reference=1, confirmed_references={2: 1})
    assert plan.provenance == PROVENANCE_VERIFIED
    assert plan.reference_by_period() == {1: 1, 2: 2}


def test_provenance_text_is_period_specific():
    inference = _inference({1: None, 2: False})
    plan = build_display_plan(inference, (1, 2), [1, 2], fallback_period_1_reference=2, confirmed_references={2: 2})
    assert "P2 confirmed by user review" in orientation_provenance_text(plan, 2)
    assert "not independent video verification" in orientation_provenance_text(plan, 2)
    assert "derived from a user-reviewed period by the halftime rule" in orientation_provenance_text(plan, 1)
    assert "confirmed" not in orientation_provenance_text(plan, 1)


# --- importer ---------------------------------------------------------------

def _importer():
    spec = importlib.util.spec_from_file_location("importer", PROJECT_ROOT / "scripts" / "import_orientation_reviews.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importer_writes_period_scoped_review_without_video_claim():
    entries = _importer().period_reviews_from_decision({
        "match_id": 7, "decision": "proposed", "periods": [2], "evidence_type": "independent_video",
        "proposed_reference_by_period": {"1": "A", "2": "B"}, "reviewer": "owner",
    })
    assert [(e["period"], e["display_reference_team"], e["evidence_type"]) for e in entries] == [
        (2, "B", "user_review_of_renders")  # video claimed but no URL/clock -> not video
    ]


def test_importer_alternative_uses_alternative_reference():
    entries = _importer().period_reviews_from_decision({
        "match_id": 7, "decision": "alternative", "periods": [1], "alternative_reference_by_period": {"1": "B"},
    })
    assert entries[0]["display_reference_team"] == "B"


# --- the recorded France-Morocco decision -----------------------------------

def test_france_morocco_period_2_decision_is_recorded_as_user_review_only():
    reviews = load_period_reviews(DEFAULT_REGISTRY_PATH).get(3901833, {})
    assert set(reviews) == {2}  # period 1 not reviewed
    review = reviews[2]
    # Morocco attacks left -> right (unrotated) and defends the left goal; France the right.
    assert review.display_reference_team == "Morocco Women's"
    assert review.evidence_type == "user_review_of_renders" and not review.independently_verified
    assert 3901833 not in load_calibration_registry(DEFAULT_REGISTRY_PATH)
