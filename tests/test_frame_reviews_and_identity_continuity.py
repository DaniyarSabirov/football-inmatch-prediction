"""Tests for frame-level review decisions and the display-only shared-snapshot
team-identity refinement (2026-10-24).

Real-data tests are skipped when the raw StatsBomb files are absent.
"""

from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path

import pytest

from src.animate_statsbomb_360_match import (
    STATSBOMB_DATA_DIR,
    JoinedFrame,
    MatchFrames,
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    compute_orientation_results,
    compute_shared_group_info,
    display_orientation_results,
    frame_review_corrections,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    match_team_ids,
    refine_shared_sources_by_continuity,
    resolve_display_plan,
)
from src.broadcast_orientation import build_display_plan, infer_broadcast_orientation
from src.orientation_correction import ORIENTATION_NORMAL, ORIENTATION_UNCERTAIN, OrientationResult
from src.physical_orientation_calibration import (
    DEFAULT_REGISTRY_PATH,
    CalibrationRegistryError,
    load_calibration_registry,
    load_frame_reviews,
)
from src.team_identity import resolve_absolute_team_for_point

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEWED = {  # sheet suffix: (match id, period, timestamp, number of events sharing the snapshot)
    "992": (3773631, 1, "00:00:00.992", 1), "189": (3788741, 2, "00:00:00.189", 1),
    "080": (3788776, 1, "00:00:00.080", 1), "590": (3837771, 2, "00:00:00.590", 1),
    "404": (3857254, 2, "00:00:01.404", 1), "388": (3857263, 2, "00:00:00.388", 1),
    "417": (3893791, 1, "00:27:50.417", 1), "525": (3930163, 1, "00:46:15.525", 2),
    "505": (3930164, 1, "00:00:00.505", 1),
}


def _has(match_id: int) -> bool:
    return (STATSBOMB_DATA_DIR / "three-sixty" / f"{match_id}.json").is_file()


def _write(tmp_path, payload) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


ENTRY = {"match_id": 1, "period": 1, "event_ids": ["a", "b"], "snapshot_correction": "NONE",
         "evidence_type": "user_review_of_renders", "reviewed_by": "owner", "date_reviewed": "2026-10-24"}


# --- registry ---------------------------------------------------------------

def test_frame_reviews_load_per_event_and_stay_out_of_calibrations(tmp_path):
    path = _write(tmp_path, {"calibrations": [], "frame_reviews": [ENTRY]})
    reviews = load_frame_reviews(path)
    assert set(reviews) == {"a", "b"} and reviews["a"] is reviews["b"]
    assert load_calibration_registry(path) == {}


@pytest.mark.parametrize("change", [
    {"snapshot_correction": "FLIP"}, {"event_ids": []}, {"evidence_type": "verified"},
])
def test_invalid_frame_reviews_fail_loudly(tmp_path, change):
    with pytest.raises(CalibrationRegistryError):
        load_frame_reviews(_write(tmp_path, {"frame_reviews": [{**ENTRY, **change}]}))


def test_an_event_cannot_be_reviewed_twice(tmp_path):
    with pytest.raises(CalibrationRegistryError):
        load_frame_reviews(_write(tmp_path, {"frame_reviews": [ENTRY, {**ENTRY, "event_ids": ["b"]}]}))


def test_recorded_decisions_match_the_nine_sheets():
    reviews = load_frame_reviews(DEFAULT_REGISTRY_PATH)
    snapshots = {(r.match_id, r.period, r.timestamp): r for r in reviews.values()}
    assert len(snapshots) == 9 and len(reviews) == 10
    for match_id, period, timestamp, events in REVIEWED.values():
        review = snapshots[(match_id, period, timestamp)]
        assert len(review.event_ids) == events
        assert review.snapshot_correction == "NONE"  # every selection was the CURRENT drawing
        assert review.evidence_type == "user_review_of_renders"
    assert "team colours" in snapshots[(3930164, 1, "00:00:00.505")].open_issue  # identity question stays open


def test_importer_converts_current_and_alternative():
    spec = importlib.util.spec_from_file_location("importer", PROJECT_ROOT / "scripts" / "import_orientation_reviews.py")
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    base = {"match_id": 5, "period": 1, "frame_event_ids": ["e"], "current_snapshot_correction": "NONE"}
    assert importer.frame_review_from_decision({**base, "decision": "current"})["snapshot_correction"] == "NONE"
    assert importer.frame_review_from_decision({**base, "decision": "alternative"})["snapshot_correction"] == "ROTATED_180"
    assert importer.frame_review_from_decision({**base, "decision": "current", "evidence_type": "independent_video"})[
        "evidence_type"] == "user_review_of_renders"  # no URL/clock -> not video


# --- display ------------------------------------------------------------------

def _two_event_match():
    snapshot = {"visible_area": [], "freeze_frame": [{"teammate": True, "actor": True, "location": [30.0, 20.0]}]}
    events = [
        {"id": "u", "team": {"id": 1}, "period": 1, "timestamp": "00:00:01.000", "location": [30.0, 20.0]},
        {"id": "c", "team": {"id": 2}, "period": 1, "timestamp": "00:00:09.000", "location": [70.0, 50.0]},
    ]
    return MatchFrames(Path("x"), Path("y"), 2, 2, 2, 2, [JoinedFrame(e, dict(snapshot)) for e in events])


def test_review_resolves_only_uncertain_snapshots():
    match = _two_event_match()
    inference = infer_broadcast_orientation([], [], (1, 2))
    plan = build_display_plan(inference, (1, 2), [1], anchor_period_1_reference=1)
    results = [OrientationResult(ORIENTATION_UNCERTAIN, 0.1, 30.0), OrientationResult(ORIENTATION_NORMAL, 0.9, 0.0)]
    used, counts = display_orientation_results(match, results, inference, plan,
                                               frame_reviews={"u": "ROTATED_180", "c": "ROTATED_180"})
    assert used[0].orientation == "ROTATED_180" and used[0].resolved_by == "user_frame_review"
    assert used[1].orientation == "NORMAL"  # a review never overrides a confident result
    assert counts["resolved_by_user_frame_review"] == 1


@functools.lru_cache(maxsize=None)
def _pipeline(match_id: int, with_reviews: bool = True):
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan,
                                          frame_reviews=frame_review_corrections() if with_reviews else None)
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    return raw, results, used, display


@pytest.mark.parametrize("suffix", sorted(REVIEWED))
def test_reviewed_frames_are_resolved_without_changing_the_drawing(suffix):
    match_id, period, timestamp, _ = REVIEWED[suffix]
    if not _has(match_id):
        pytest.skip("raw match not available")
    raw, results, used, display = _pipeline(match_id)
    _, _, _, display_before = _pipeline(match_id, False)
    indices = [i for i, j in enumerate(raw.usable_frames) if j.event["period"] == period and j.event["timestamp"] == timestamp
               and results[i].orientation == ORIENTATION_UNCERTAIN]
    assert indices
    for i in indices:
        assert used[i].resolved_by == "user_frame_review"
        assert display.usable_frames[i].frame_360["freeze_frame"] == display_before.usable_frames[i].frame_360["freeze_frame"]


# --- shared-snapshot identity refinement ------------------------------------

def _refined(match_id: int):
    raw, _, _, display = _pipeline(match_id)
    shared = compute_shared_group_info(raw)
    raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    pair = tuple(names[t] for t in match_team_ids(raw))
    refined, switches = refine_shared_sources_by_continuity(display, shared, raw_by_id, pair)
    return raw, display, shared, refined, switches, raw_by_id, pair


def test_uruguay_ghana_keeper_is_coloured_as_ghana_after_refinement():
    if not _has(3857293):
        pytest.skip("raw match not available")
    raw, display, shared, refined, switches, raw_by_id, pair = _refined(3857293)
    assert sorted((s["period"], s["timestamp"]) for s in switches) == [(1, "00:44:24.954"), (2, "00:38:27.375")]
    index = next(i for i, j in enumerate(raw.usable_frames)
                 if j.event["timestamp"] == "00:44:24.954" and j.event["type"]["name"] == "Ball Receipt*")
    event = display.usable_frames[index].event
    keeper = next(p for p in display.usable_frames[index].frame_360["freeze_frame"] if p.get("keeper"))
    assert resolve_absolute_team_for_point(keeper, event, shared[event["id"]], raw_by_id, pair) == "Uruguay"  # before
    assert resolve_absolute_team_for_point(keeper, event, refined[event["id"]], raw_by_id, pair) == "Ghana"  # after
    assert keeper["location"][0] < 10  # standing in Ghana's own (left) goal


def test_refinement_leaves_non_shared_raw_flag_errors_alone():
    """Frame 505 (3930164 kick-off): its colours come from the raw flags of an
    unshared snapshot; the refinement must not touch it."""
    if not _has(3930164):
        pytest.skip("raw match not available")
    raw, _, shared, refined, switches, _, _ = _refined(3930164)
    kickoff = next(j.event["id"] for j in raw.usable_frames if j.event["timestamp"] == "00:00:00.505")
    assert kickoff not in shared and kickoff not in refined
    assert switches == []


def test_ml_pipeline_shares_orientation_and_identity_resolution_with_the_renderer():
    """Since 2026-10-26 (approved synchronization) the builder resolves
    orientation and team identity with the same shared functions the
    renderer uses; the display-only refinement is superseded by them."""
    builder = (PROJECT_ROOT / "src" / "build_statsbomb_360_dataset.py").read_text(encoding="utf-8")
    renderer = (PROJECT_ROOT / "src" / "animate_statsbomb_360_match.py").read_text(encoding="utf-8")
    assert "resolve_match_team_identity" in builder and "resolve_match_team_identity" in renderer
    assert "resolve_match_orientation" in builder
    assert "refine_shared_sources_by_continuity(" not in builder
