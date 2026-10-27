"""Tests for the display-only broadcast orientation layer (Problem D).

Two groups:
  - synthetic geometry / plan-resolution tests (always run);
  - regression tests on real StatsBomb matches (skipped when the raw files
    are not present locally): Belgium-Russia 3788743, Spain-Croatia 3930160,
    Iceland-France 3835342, the extra-time exception Morocco-Spain 3869220,
    and the shootout Japan-Croatia 3869219.

Passing these tests proves internal consistency and agreement with the two
registry-verified matches; it is NOT proof of broadcast alignment for any
other match.
"""

from __future__ import annotations

import functools
import math

import pytest

from src.animate_statsbomb_360_match import (
    STATSBOMB_DATA_DIR,
    DirectionInference,
    _apply_team_normalization_per_period,
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    compute_orientation_results,
    directions_from_display_plan,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    match_team_ids,
)
from src.broadcast_orientation import (
    BROADCAST_STATUS_INFERRED,
    BROADCAST_STATUS_NO_EVIDENCE,
    CAMERA_SIDE_HIGH_Y,
    CAMERA_SIDE_LOW_Y,
    CAMERA_SIDE_UNKNOWN,
    FRAME_MODEL_SHARED,
    PROVENANCE_INFERRED,
    PROVENANCE_PARTIAL,
    PROVENANCE_UNKNOWN,
    PROVENANCE_VERIFIED,
    SOURCE_CAMERA_RELATIVE,
    SOURCE_LAWS_RULE,
    build_display_plan,
    estimate_camera_side,
    infer_broadcast_orientation,
)
from src.orientation_correction import mirror_point
from src.physical_orientation_calibration import load_calibration_registry
from src.team_normalization import apply_period_end_swap, resolve_reference_team_id

# Raw visible_area of the Courtois Carry, match 3788743, P2 00:26:53.383,
# copied verbatim from data/raw/statsbomb/data/three-sixty/3788743.json.
COURTOIS_RAW_POLYGON = [
    0.0, 80.0, 0.0, 38.16023443440224, 28.10848038695826, 0.0,
    70.28121356130205, 0.0, 96.83125316828158, 80.0, 0.0, 80.0,
]


def _rotate_flat(polygon):
    flat = []
    for i in range(0, len(polygon), 2):
        flat.extend(mirror_point((polygon[i], polygon[i + 1])))
    return flat


def _footprint(camera_y, near_half_width=10.0, far_half_width=40.0, centre_x=60.0):
    """A camera footprint: narrow near edge close to camera_y, wide far edge."""
    near_y, far_y = (70.0, 5.0) if camera_y > 40 else (10.0, 75.0)
    return [
        centre_x - near_half_width, near_y, centre_x - far_half_width, far_y,
        centre_x + far_half_width, far_y, centre_x + near_half_width, near_y,
    ]


# --- geometry ---------------------------------------------------------------

def test_real_courtois_polygon_points_to_low_y_camera_in_raw_frame():
    estimate = estimate_camera_side(COURTOIS_RAW_POLYGON)
    assert estimate.side == CAMERA_SIDE_LOW_Y
    assert estimate.apex[1] < 0  # apex beyond the y=0 touchline


def test_rotation_moves_camera_to_opposite_touchline():
    assert estimate_camera_side(_rotate_flat(COURTOIS_RAW_POLYGON)).side == CAMERA_SIDE_HIGH_Y


@pytest.mark.parametrize("camera_y, expected", [(90, CAMERA_SIDE_HIGH_Y), (-10, CAMERA_SIDE_LOW_Y)])
def test_unclipped_trapezoid_narrow_side_is_camera_side(camera_y, expected):
    assert estimate_camera_side(_footprint(camera_y)).side == expected


def test_reflections_are_not_rotations_y_reflection_flips_camera_but_x_reflection_does_not():
    polygon = COURTOIS_RAW_POLYGON
    x_reflected = [v if i % 2 else 120 - v for i, v in enumerate(polygon)]
    y_reflected = [80 - v if i % 2 else v for i, v in enumerate(polygon)]
    assert estimate_camera_side(x_reflected).side == CAMERA_SIDE_LOW_Y
    assert estimate_camera_side(y_reflected).side == CAMERA_SIDE_HIGH_Y


def test_parallel_sided_polygon_gives_no_camera_side():
    assert estimate_camera_side([20, 10, 60, 10, 60, 70, 20, 70]).side == CAMERA_SIDE_UNKNOWN


def test_footprint_converging_toward_goal_line_gives_no_camera_side():
    # Behind-the-goal style footprint: lateral edges converge toward x < 0.
    assert estimate_camera_side([10, 35, 100, 5, 100, 75, 10, 45]).side == CAMERA_SIDE_UNKNOWN


def test_malformed_polygons_are_unknown():
    for value in (None, [], [1, 2, 3], [True, 1, 2, 3, 4, 5], "x"):
        assert estimate_camera_side(value).side == CAMERA_SIDE_UNKNOWN


# --- match inference and plan (synthetic) -----------------------------------

def _event(team_id, period):
    return {"team": {"id": team_id, "name": f"T{team_id}"}, "period": period}


def _synthetic_match(team_a_camera_by_period, votes=25):
    """Events where team A's own frame shows the camera at the given side;
    team B's frame shows the opposite side (OPPOSITE model) except in
    period 5, where both teams share one frame."""
    events, frames = [], []
    for period, side in team_a_camera_by_period.items():
        a_poly = _footprint(90 if side == CAMERA_SIDE_HIGH_Y else -10)
        b_poly = a_poly if period == 5 else _rotate_flat(a_poly)
        for _ in range(votes):
            events += [_event(1, period), _event(2, period)]
            frames += [{"visible_area": a_poly}, {"visible_area": b_poly}]
    return events, frames


def test_inference_needs_no_reference_team_and_observes_halftime_swap():
    events, frames = _synthetic_match({1: CAMERA_SIDE_LOW_Y, 2: CAMERA_SIDE_HIGH_Y})
    inference = infer_broadcast_orientation(events, frames, (1, 2))
    assert inference.status == BROADCAST_STATUS_INFERRED
    assert inference.reference_by_period == {1: 2, 2: 1}
    assert inference.end_swap_checks[0].swapped is True


def test_missing_halftime_swap_is_reported_as_conflicting():
    events, frames = _synthetic_match({1: CAMERA_SIDE_LOW_Y, 2: CAMERA_SIDE_LOW_Y})
    assert infer_broadcast_orientation(events, frames, (1, 2)).status == "CONFLICTING"


def test_no_polygons_means_no_evidence_and_unknown_fallback_plan():
    events = [_event(1, 1), _event(2, 1), _event(1, 2), _event(2, 2)]
    inference = infer_broadcast_orientation(events, [{}] * 4, (1, 2))
    assert inference.status == BROADCAST_STATUS_NO_EVIDENCE
    plan = build_display_plan(inference, (1, 2), [1, 2], fallback_period_1_reference=1)
    assert plan.provenance == PROVENANCE_UNKNOWN
    assert plan.reference_by_period() == {1: 1, 2: 2}  # legacy behaviour, flagged unknown


def test_shootout_uses_shared_frame_and_rotates_both_teams_together():
    events, frames = _synthetic_match({1: CAMERA_SIDE_HIGH_Y, 2: CAMERA_SIDE_LOW_Y, 5: CAMERA_SIDE_LOW_Y}, votes=5)
    events, frames = events[:], frames[:]
    more_events, more_frames = _synthetic_match({1: CAMERA_SIDE_HIGH_Y, 2: CAMERA_SIDE_LOW_Y})
    inference = infer_broadcast_orientation(events + more_events, frames + more_frames, (1, 2))
    plan = build_display_plan(inference, (1, 2), [1, 2, 5])
    shootout = plan.transforms[5]
    assert shootout.frame_model == FRAME_MODEL_SHARED
    assert plan.rotates(5, 1) is plan.rotates(5, 2) is True
    assert plan.attacking_directions(5) is None


def test_anchor_overrides_camera_but_camera_end_relations_are_kept():
    events, frames = _synthetic_match({1: CAMERA_SIDE_HIGH_Y, 2: CAMERA_SIDE_LOW_Y, 3: CAMERA_SIDE_LOW_Y, 4: CAMERA_SIDE_HIGH_Y})
    inference = infer_broadcast_orientation(events, frames, (1, 2))
    assert inference.reference_by_period == {1: 1, 2: 2, 3: 2, 4: 1}  # P3 at P2's ends
    plan = build_display_plan(inference, (1, 2), [1, 2, 3, 4], anchor_period_1_reference=2)
    assert plan.provenance == PROVENANCE_VERIFIED
    assert plan.reference_by_period() == {1: 2, 2: 1, 3: 1, 4: 2}
    assert plan.transforms[3].source == SOURCE_CAMERA_RELATIVE


def test_undecided_period_is_derived_by_rule_and_marked_partial():
    events, frames = _synthetic_match({1: CAMERA_SIDE_HIGH_Y})
    events += [_event(1, 2), _event(2, 2)]
    frames += [{}, {}]
    inference = infer_broadcast_orientation(events, frames, (1, 2))
    plan = build_display_plan(inference, (1, 2), [1, 2])
    assert plan.reference_by_period() == {1: 1, 2: 2}
    assert plan.transforms[2].source == SOURCE_LAWS_RULE
    assert plan.provenance == PROVENANCE_PARTIAL


def test_directions_follow_transform_and_labels_mark_defended_goal():
    events, frames = _synthetic_match({1: CAMERA_SIDE_HIGH_Y, 2: CAMERA_SIDE_LOW_Y})
    plan = build_display_plan(infer_broadcast_orientation(events, frames, (1, 2)), (1, 2), [1, 2])
    names = {1: "A", 2: "B"}
    result = directions_from_display_plan(plan, 2, names, DirectionInference({"A": 1, "B": -1}, ""))
    # Period 2: team 2 unrotated -> attacks +x (defends the x=0, left goal).
    assert result.directions == {"A": -1, "B": 1}
    assert "WARNING" in result.explanation  # disagreeing shot cross-check is surfaced
    assert result.provenance


# --- real StatsBomb regressions --------------------------------------------

def _has(match_id):
    return (STATSBOMB_DATA_DIR / "three-sixty" / f"{match_id}.json").is_file()


def real(match_id):
    return pytest.mark.skipif(not _has(match_id), reason=f"raw StatsBomb match {match_id} not available")


@functools.lru_cache(maxsize=None)
def _pipeline(match_id):
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    periods = sorted({j.event["period"] for j in raw.usable_frames})
    plan = build_display_plan(inference, match_team_ids(raw), periods)
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, results), plan)
    return raw, results, inference, names, plan, display


def _camera_reference_names(match_id):
    _, _, inference, names, _, _ = _pipeline(match_id)
    return {p: names[t] for p, t in inference.reference_by_period.items()}


@real(3788743)
def test_belgium_russia_camera_inference_matches_verified_broadcast():
    # Broadcast (P2, 87:43): Belgium attacks right-to-left, Russia's goal on the left.
    assert _camera_reference_names(3788743) == {1: "Belgium", 2: "Russia"}
    registry = load_calibration_registry()
    assert registry[3788743].period_1_physical_reference_team == "Belgium"


@real(3788743)
def test_belgium_russia_legacy_default_was_the_180_degree_opposite():
    raw, _, inference, names, _, _ = _pipeline(3788743)
    events = [j.event for j in raw.usable_frames]
    automatic = resolve_reference_team_id(events)
    for period in (1, 2):
        assert apply_period_end_swap(events, automatic, period) != inference.reference_by_period[period]


@real(3788743)
def test_courtois_carry_full_transform_trace():
    raw, results, _, _, plan, display = _pipeline(3788743)
    index = next(i for i, j in enumerate(raw.usable_frames) if j.event["id"] == "6d4668fb-2a8b-4f79-8d90-b7d87768a5f3")
    assert raw.usable_frames[index].event["location"] == [12.9, 32.9]
    assert raw.usable_frames[index].frame_360["visible_area"] == COURTOIS_RAW_POLYGON
    assert results[index].orientation == "NORMAL"
    shown = display.usable_frames[index]
    assert shown.event["location"] == pytest.approx([107.1, 47.1])
    actor = next(p for p in shown.frame_360["freeze_frame"] if p["actor"])
    assert actor["keeper"] and actor["location"] == pytest.approx([107.1, 47.1], abs=1e-4)
    # Polygon moved with the players: camera at the bottom of the screen.
    assert estimate_camera_side(shown.frame_360["visible_area"]).side == CAMERA_SIDE_HIGH_Y
    # Belgium (rotated in P2) attacks -x: defends the RIGHT goal where Courtois stands.
    belgium = raw.usable_frames[index].event["team"]["id"]
    assert plan.attacking_directions(2)[belgium] == -1


@real(3788743)
def test_lukaku_goal_synchronised_with_broadcast_is_at_left_goal():
    raw, _, _, _, _, display = _pipeline(3788743)
    shown = next(j for j in display.usable_frames if j.event["id"] == "b0e8f7c7-291e-4d0a-81d4-0f99f4fcd73d")
    assert (shown.event["minute"], shown.event["second"]) == (87, 45)
    assert shown.event["location"][0] < 30
    keeper = next(p for p in shown.frame_360["freeze_frame"] if p.get("keeper"))
    assert keeper["teammate"] is False and keeper["location"][0] < 10  # Russia's keeper, left goal
    assert estimate_camera_side(shown.frame_360["visible_area"]).side == CAMERA_SIDE_HIGH_Y


@real(3930160)
def test_spain_croatia_camera_inference_matches_verified_broadcast():
    # Broadcast (P1, ~31:18): Croatia goalkeeper in the right-hand goal.
    assert _camera_reference_names(3930160) == {1: "Spain", 2: "Croatia"}


@real(3835342)
def test_iceland_france_halftime_swap_and_inferred_orientation():
    assert _camera_reference_names(3835342) == {1: "Iceland Women's", 2: "France Women's"}
    assert _pipeline(3835342)[4].provenance == PROVENANCE_INFERRED


@real(3869220)
def test_extra_time_period_3_can_continue_at_period_2_ends():
    names = _camera_reference_names(3869220)
    assert names[3] == names[2] == "Morocco" and names[4] == names[1] == "Spain"


@real(3869219)
def test_shootout_both_teams_share_one_frame():
    _, _, inference, _, plan, display = _pipeline(3869219)
    evidence = inference.period_evidence(5)
    sides = [set(v) for v in evidence.votes_by_team.values()]
    assert sides[0] == sides[1] and len(sides[0]) == 1
    shots = [j.event["location"][0] for j in display.usable_frames
             if j.event["period"] == 5 and j.event["type"]["name"] == "Shot"]
    assert shots and (all(x > 60 for x in shots) or all(x < 60 for x in shots))


@real(3788743)
@pytest.mark.parametrize("period", [1, 2])
def test_display_geometry_is_consistent_for_every_frame(period):
    """Players, ball and polygon move together; keepers sit at their defended end."""
    raw, results, inference, _, plan, display = _pipeline(3788743)
    keepers_ok = keepers = camera_bottom = camera_known = 0
    for joined, result in zip(display.usable_frames, results):
        if joined.event["period"] != period or result.orientation == "UNCERTAIN":
            continue
        actor = next((p for p in joined.frame_360["freeze_frame"] if p.get("actor")), None)
        if actor and joined.event.get("location"):
            assert math.dist(actor["location"], joined.event["location"]) < 15
        side = estimate_camera_side(joined.frame_360.get("visible_area")).side
        if side != CAMERA_SIDE_UNKNOWN:
            camera_known += 1
            camera_bottom += side == CAMERA_SIDE_HIGH_Y
        unrotated = not plan.rotates(period, joined.event["team"]["id"])
        for player in joined.frame_360["freeze_frame"]:
            if player.get("keeper"):
                keepers += 1
                attacks_right = unrotated == bool(player.get("teammate"))
                keepers_ok += (player["location"][0] < 60) == attacks_right
    assert camera_bottom / camera_known > 0.99
    assert keepers_ok / keepers > 0.95


def test_ml_pipeline_does_not_use_the_display_layer():
    from pathlib import Path

    builder = Path(__file__).resolve().parents[1] / "src" / "build_statsbomb_360_dataset.py"
    source = builder.read_text(encoding="utf-8")
    assert "broadcast_orientation" not in source and "display_plan" not in source


@real(3788743)
def test_display_plan_reproduces_problem_b_transform_exactly_for_regular_periods():
    raw, results, _, _, plan, display = _pipeline(3788743)
    corrected = apply_orientation_correction_to_match(raw, results)
    legacy_math = _apply_team_normalization_per_period(corrected, plan.reference_by_period())
    for ours, theirs in zip(display.usable_frames, legacy_math.usable_frames):
        assert ours.event.get("location") == theirs.event.get("location")
        assert ours.frame_360.get("visible_area") == theirs.frame_360.get("visible_area")
        assert ours.frame_360["freeze_frame"] == theirs.frame_360["freeze_frame"]


# --- display-only resolution of UNCERTAIN snapshots -------------------------

from src.animate_statsbomb_360_match import display_orientation_results  # noqa: E402
from src.broadcast_orientation import resolve_uncertain_for_display  # noqa: E402


def test_uncertain_resolved_only_when_lean_and_camera_agree():
    in_frame = COURTOIS_RAW_POLYGON  # LOW_Y in its own frame
    assert resolve_uncertain_for_display(True, in_frame, CAMERA_SIDE_HIGH_Y) == "ROTATED_180"
    assert resolve_uncertain_for_display(False, in_frame, CAMERA_SIDE_LOW_Y) == "NORMAL"
    assert resolve_uncertain_for_display(True, in_frame, CAMERA_SIDE_LOW_Y) is None   # signals disagree
    assert resolve_uncertain_for_display(False, in_frame, CAMERA_SIDE_HIGH_Y) is None
    assert resolve_uncertain_for_display(True, [], CAMERA_SIDE_HIGH_Y) is None          # unreadable camera
    assert resolve_uncertain_for_display(True, in_frame, CAMERA_SIDE_UNKNOWN) is None   # undecided period


@pytest.mark.parametrize("match_id, timestamp", [
    (3902239, "00:23:04.965"),  # random round 3, GIF 01: actor 80 yd from the ball before the fix
    (3998846, "00:31:30.906"),  # random round 3, GIF 10: actor 109 yd from the ball before the fix
])
def test_real_uncertain_flip_frames_are_resolved_for_display(match_id, timestamp):
    if not _has(match_id):
        pytest.skip("raw match not available")
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    resolved, counts = display_orientation_results(raw, results, inference)
    index = next(i for i, j in enumerate(raw.usable_frames)
                 if j.event["timestamp"] == timestamp and j.event["type"]["name"] == "Ball Receipt*")
    assert results[index].orientation == "UNCERTAIN"  # Problem A itself is unchanged
    assert resolved[index].orientation == "ROTATED_180"
    corrected = apply_orientation_correction_to_match(raw, resolved).usable_frames[index]
    actor = next(p for p in corrected.frame_360["freeze_frame"] if p.get("actor"))
    assert math.dist(actor["location"], corrected.event["location"]) < 30
    assert counts["resolved_rotated"] >= 1
