"""Tests for the StatsBomb 360 orientation classifier and correction.

Fixtures are drawn from real, previously-diagnosed cases in match 3902240
(see the forensic investigation notes), plus one synthetic ambiguous case,
so regressions in either direction (over- or under-correcting) are caught.
"""

import pytest

from src.orientation_correction import (
    CONFIDENCE_THRESHOLD,
    ORIENTATION_NORMAL,
    ORIENTATION_ROTATED_180,
    ORIENTATION_UNCERTAIN,
    apply_correction,
    classify_orientation,
    classify_orientations_for_match,
    mirror_point,
    resolve_uncertain_with_neighbors,
)

# --- Real case: frame 19, event 4f2e844a (Pass/Cross, Spain) -----------------
# Already correctly oriented; used as a known-NORMAL regression case.
FRAME_19_EVENT = {
    "id": "4f2e844a-aafe-488a-a5b2-ce512eeda302",
    "type": {"name": "Pass"},
    "location": [118.4, 16.3],
}
FRAME_19_360 = {
    "event_uuid": "4f2e844a-aafe-488a-a5b2-ce512eeda302",
    "visible_area": [
        105.08357440715477, 76.69392795503454,
        86.0705792185362, 68.92890142897971,
        105.24616249665954, 0.0,
        120.0, 0.0,
        120.0, 52.9674914148055,
        105.08357440715477, 76.69392795503454,
    ],
    "freeze_frame": [
        {"teammate": True, "actor": True, "keeper": False, "location": [118.4000015258789, 16.299999237060547]},
        {"teammate": False, "actor": False, "keeper": True, "location": [119.1372683130123, 35.81540550738862]},
        {"teammate": True, "actor": False, "keeper": False, "location": [113.5993068047155, 30.2944475997433]},
    ],
}

# --- Real case: "frame 16", event 0d3afccf (Dribbled Past, Sherida Spitse) --
# Known problematic case: event.location says the defender is still near
# (16.5, 70.0), but her freeze-frame actor point is 180 degrees away.
DRIBBLED_PAST_EVENT = {
    "id": "0d3afccf-a643-4f8c-b601-ad2c1a79fdb4",
    "type": {"name": "Dribbled Past"},
    "location": [16.5, 70.0],
}
DRIBBLED_PAST_360 = {
    "event_uuid": "0d3afccf-a643-4f8c-b601-ad2c1a79fdb4",
    "visible_area": [
        96.99773690370566, 68.30163110162141,
        78.57833858101374, 62.53777577346451,
        85.62659478473682, 0.0,
        120.0, 0.0,
        120.0, 24.85061768651143,
        96.99773690370566, 68.30163110162141,
    ],
    "freeze_frame": [
        {"teammate": True, "actor": False, "keeper": False, "location": [87.38583475800843, 14.551831557206317]},
        {"teammate": False, "actor": False, "keeper": False, "location": [89.42025206593412, 40.324675586093235]},
        {"teammate": True, "actor": True, "keeper": False, "location": [102.80090478326122, 11.024152703627603]},
        {"teammate": False, "actor": False, "keeper": False, "location": [103.5999984741211, 10.100000381469727]},
    ],
}

# --- Real borderline case: event c1bd09f8 (Ball Receipt*, near midfield) ----
# Actor signal favors ROTATED_180 (27.5 vs 3.0), but the nearest-point and
# centroid signals mildly favor NORMAL -- contradictory evidence that must
# not be blindly corrected.
BORDERLINE_EVENT = {
    "id": "c1bd09f8-0000-0000-0000-000000000000",
    "type": {"name": "Ball Receipt*"},
    "location": [57.1, 53.6],
}
BORDERLINE_360 = {
    "event_uuid": "c1bd09f8-0000-0000-0000-000000000000",
    "freeze_frame": [
        {"teammate": True, "actor": True, "keeper": False, "location": [59.9, 26.2]},
        {"teammate": True, "actor": False, "keeper": False, "location": [55.5, 51.9]},
    ],
}


def test_known_normal_case_frame_19_is_classified_normal() -> None:
    result = classify_orientation(FRAME_19_EVENT, FRAME_19_360)
    assert result.orientation == ORIENTATION_NORMAL
    assert result.confidence >= 0.45


def test_known_rotated_case_dribbled_past_is_classified_rotated() -> None:
    result = classify_orientation(DRIBBLED_PAST_EVENT, DRIBBLED_PAST_360)
    assert result.orientation == ORIENTATION_ROTATED_180
    assert result.confidence >= 0.45
    # After the fix, the actor should land close to event.location.
    assert result.best_distance < 3.0


def test_borderline_contradictory_case_is_not_blindly_corrected() -> None:
    result = classify_orientation(BORDERLINE_EVENT, BORDERLINE_360)
    # Evidence is contradictory (actor disagrees with nearest_point/centroid);
    # at the 0.45 threshold this must not be a confident ROTATED_180 call.
    assert result.orientation != ORIENTATION_ROTATED_180 or result.confidence < 0.45
    # In practice, with the conservative threshold, it resolves to UNCERTAIN.
    assert result.orientation == ORIENTATION_UNCERTAIN


def test_mirror_transformation_matches_the_documented_formula() -> None:
    assert mirror_point((0.0, 0.0)) == (120.0, 80.0)
    assert mirror_point((120.0, 80.0)) == (0.0, 0.0)
    assert mirror_point((60.0, 40.0)) == (60.0, 40.0)
    assert mirror_point((30.0, 20.0)) == (90.0, 60.0)


def test_mirror_preserves_out_of_bounds_status_exactly() -> None:
    # Regression/invariant test for a dedicated data-quality audit
    # (2026-10-13): 189,128 freeze-frame points in the raw dataset fall
    # outside [0,120]x[0,80] (confirmed, by category and magnitude, to be
    # genuine StatsBomb tracking data -- players near the touchline/corner,
    # or non-participants such as substitutes visible in the camera frame
    # -- not malformed data). Since mirror_point is a 180-degree rotation
    # about the pitch center (an isometry), it must map the pitch rectangle
    # exactly onto itself: an in-bounds point stays in-bounds, an
    # out-of-bounds point stays out-of-bounds by EXACTLY the same margin,
    # just reflected to the opposite side. This proves orientation
    # correction and team normalization (both built on mirror_point) can
    # never create, remove, or silently "fix" an out-of-bounds point.
    def in_bounds(point: tuple[float, float]) -> bool:
        return 0.0 <= point[0] <= 120.0 and 0.0 <= point[1] <= 80.0

    out_of_bounds_points = [
        (-0.84, 37.35),    # x < 0
        (120.19, 41.29),   # x > 120
        (64.67, 82.29),    # y > 80 (the dominant real-data category: touchline overshoot)
        (82.71, -0.72),    # y < 0
        (45.11, 117.30),   # far outside (e.g. a substitute/bench position visible in-frame)
    ]
    for point in out_of_bounds_points:
        assert not in_bounds(point), f"fixture error: {point} should start out of bounds"
        mirrored = mirror_point(point)
        assert not in_bounds(mirrored), (
            f"mirror_point turned an out-of-bounds point in-bounds: {point} -> {mirrored}"
        )
        # The overshoot magnitude (distance past the nearest violated
        # boundary) must be preserved exactly, not just "still out of
        # bounds by some amount."
        def overshoot(p: tuple[float, float]) -> float:
            return max(0.0, -p[0], p[0] - 120.0, -p[1], p[1] - 80.0)

        assert overshoot(mirrored) == pytest.approx(overshoot(point))

    in_bounds_points = [(0.0, 0.0), (120.0, 80.0), (60.0, 40.0), (1.0, 1.0), (119.0, 79.0)]
    for point in in_bounds_points:
        assert in_bounds(point), f"fixture error: {point} should start in bounds"
        assert in_bounds(mirror_point(point)), (
            f"mirror_point turned an in-bounds point out of bounds: {point}"
        )


def test_apply_correction_rotates_the_whole_snapshot_consistently() -> None:
    corrected = apply_correction(DRIBBLED_PAST_360, ORIENTATION_ROTATED_180)

    original_players = DRIBBLED_PAST_360["freeze_frame"]
    corrected_players = corrected["freeze_frame"]
    assert len(corrected_players) == len(original_players)
    for original, rotated in zip(original_players, corrected_players):
        ox, oy = original["location"]
        rx, ry = rotated["location"]
        assert rx == 120.0 - ox
        assert ry == 80.0 - oy
        # Flags must be untouched -- only coordinates are transformed.
        assert rotated["teammate"] == original["teammate"]
        assert rotated["actor"] == original["actor"]
        assert rotated["keeper"] == original["keeper"]

    original_area = DRIBBLED_PAST_360["visible_area"]
    corrected_area = corrected["visible_area"]
    assert len(corrected_area) == len(original_area)
    for i in range(0, len(original_area), 2):
        assert corrected_area[i] == 120.0 - original_area[i]
        assert corrected_area[i + 1] == 80.0 - original_area[i + 1]


def test_apply_correction_twice_is_not_the_same_as_once() -> None:
    """Guards against accidentally double-rotating a frame in a pipeline."""
    once = apply_correction(DRIBBLED_PAST_360, ORIENTATION_ROTATED_180)
    twice = apply_correction(once, ORIENTATION_ROTATED_180)
    # Rotating twice returns to the original (raw, uncorrected) coordinates --
    # if a caller accidentally applied the correction twice, this proves it
    # is NOT idempotent and would silently undo the fix.
    for original, twice_rotated in zip(DRIBBLED_PAST_360["freeze_frame"], twice["freeze_frame"]):
        assert twice_rotated["location"] == pytest.approx(original["location"])
    # A single application must differ from both the original and the double
    # application, so callers can detect (in their own bookkeeping) whether
    # a frame has already been corrected and must not re-apply it.
    assert once["freeze_frame"][0]["location"] != DRIBBLED_PAST_360["freeze_frame"][0]["location"]


def test_normal_snapshot_is_left_unchanged_by_apply_correction() -> None:
    corrected = apply_correction(FRAME_19_360, ORIENTATION_NORMAL)
    assert corrected["freeze_frame"] == FRAME_19_360["freeze_frame"]
    assert corrected["visible_area"] == FRAME_19_360["visible_area"]


def test_uncertain_snapshot_is_left_unchanged_by_apply_correction() -> None:
    corrected = apply_correction(BORDERLINE_360, ORIENTATION_UNCERTAIN)
    assert corrected["freeze_frame"] == BORDERLINE_360["freeze_frame"]


def test_apply_correction_does_not_mutate_the_input() -> None:
    original_snapshot = {
        "freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": [10.0, 10.0]}],
        "visible_area": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0],
    }
    import copy

    reference = copy.deepcopy(original_snapshot)
    apply_correction(original_snapshot, ORIENTATION_ROTATED_180)
    assert original_snapshot == reference


def test_resolve_uncertain_with_neighbors_only_upgrades_uncertain_cases() -> None:
    # Three consecutive frames, same period:
    #   0 "anchor"  - confidently NORMAL on geometry alone (confidence 1.0)
    #   1 "target"  - only weak, sub-threshold geometry evidence (confidence
    #                 0.40 < 0.45) that happens to lean NORMAL; the nearby
    #                 confident anchor should be enough to tip it over 0.45
    #   2 "rotated" - confidently ROTATED_180 on geometry alone; must never
    #                 be used as an anchor and must never be overturned
    events = [
        {"id": "anchor", "type": {"name": "Pass"}, "location": [56.0, 40.0]},
        {"id": "target", "type": {"name": "Ball Receipt*"}, "location": [50.0, 40.0]},
        {"id": "rotated", "type": {"name": "Dispossessed"}, "location": [10.0, 10.0]},
    ]
    frames = [
        {"freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": [56.0, 40.0]}]},
        {"freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": [56.0, 40.0]}]},
        {"freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": [110.0, 70.0]}]},
    ]
    timestamps = [10.0, 10.2, 15.0]
    periods = [1, 1, 1]

    pass1_results = [classify_orientation(events[i], frames[i]) for i in range(3)]
    assert pass1_results[0].orientation == ORIENTATION_NORMAL
    assert pass1_results[1].orientation == ORIENTATION_UNCERTAIN
    assert pass1_results[1].confidence < CONFIDENCE_THRESHOLD
    assert pass1_results[2].orientation == ORIENTATION_ROTATED_180

    refined = resolve_uncertain_with_neighbors(
        events, frames, timestamps, periods, pass1_results
    )

    # The confident NORMAL and ROTATED_180 decisions must never be overturned.
    assert refined[0].orientation == pass1_results[0].orientation
    assert refined[0].resolved_by == "geometry"
    assert refined[2].orientation == pass1_results[2].orientation
    assert refined[2].resolved_by == "geometry"
    # The weak, previously-UNCERTAIN case is pulled over the confidence
    # threshold by its confidently-NORMAL neighbor, and only by that neighbor.
    assert refined[1].orientation == ORIENTATION_NORMAL
    assert refined[1].confidence >= CONFIDENCE_THRESHOLD
    assert refined[1].resolved_by == "temporal_neighbor"


def test_resolve_uncertain_with_neighbors_never_anchors_on_non_normal_frames() -> None:
    # An UNCERTAIN frame flanked only by a ROTATED_180 neighbor (no NORMAL
    # neighbor anywhere nearby) must remain UNCERTAIN: rotated/uncertain
    # frames must never be used as anchors.
    events = [
        {"id": "rotated", "type": {"name": "Dispossessed"}, "location": [10.0, 10.0]},
        {"id": "target", "type": {"name": "Ball Receipt*"}, "location": [50.0, 40.0]},
    ]
    frames = [
        {"freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": [110.0, 70.0]}]},
        {"freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": [56.0, 40.0]}]},
    ]
    timestamps = [10.0, 10.2]
    periods = [1, 1]

    pass1_results = [classify_orientation(events[i], frames[i]) for i in range(2)]
    assert pass1_results[0].orientation == ORIENTATION_ROTATED_180
    assert pass1_results[1].orientation == ORIENTATION_UNCERTAIN

    refined = resolve_uncertain_with_neighbors(events, frames, timestamps, periods, pass1_results)
    assert refined[0].orientation == ORIENTATION_ROTATED_180
    assert refined[1].orientation == ORIENTATION_UNCERTAIN


def test_classify_orientations_for_match_matches_manual_two_pass_pipeline() -> None:
    """The shared orchestration helper must give identical results to calling
    classify_orientation() then resolve_uncertain_with_neighbors() by hand --
    it exists only to avoid re-implementing that sequence in every consumer.
    """
    events = [
        {"id": "anchor", "type": {"name": "Pass"}, "location": [56.0, 40.0]},
        {"id": "target", "type": {"name": "Ball Receipt*"}, "location": [50.0, 40.0]},
        {"id": "rotated", "type": {"name": "Dispossessed"}, "location": [10.0, 10.0]},
    ]
    frames = [
        {"freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": [56.0, 40.0]}]},
        {"freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": [56.0, 40.0]}]},
        {"freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": [110.0, 70.0]}]},
    ]
    timestamps = [10.0, 10.2, 15.0]
    periods = [1, 1, 1]

    manual_pass1 = [classify_orientation(events[i], frames[i]) for i in range(3)]
    manual_final = resolve_uncertain_with_neighbors(events, frames, timestamps, periods, manual_pass1)

    orchestrated = classify_orientations_for_match(events, frames, timestamps, periods)

    assert [r.orientation for r in orchestrated] == [r.orientation for r in manual_final]
    assert [r.confidence for r in orchestrated] == [r.confidence for r in manual_final]
