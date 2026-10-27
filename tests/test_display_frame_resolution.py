"""Regression tests for the frame-level display resolution (2026-10-23 fix).

Covers display_orientation_results' four routes (same-moment duplicate /
duplicate consensus, kick-off own-half rule, lean + camera, camera +
temporal continuity), the plan-aware expected camera side, and the
France-Morocco period-1 derivation, using real frames that the full-dataset
audit found drawn 180 degrees flipped before the fix. Tests needing raw
StatsBomb files are skipped when those files are absent.

Passing these shows the listed frames are now drawn consistently with their
same-moment duplicates and neighbours; it is not broadcast verification.
"""

from __future__ import annotations

import functools
import math

import pytest

from src.animate_statsbomb_360_match import (
    STATSBOMB_DATA_DIR,
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    compute_orientation_results,
    display_orientation_results,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    resolve_display_plan,
)
from src.broadcast_orientation import (
    CAMERA_SIDE_HIGH_Y,
    CAMERA_SIDE_LOW_Y,
    PROVENANCE_USER_CONFIRMED_AND_DERIVED,
    SOURCE_DERIVED_FROM_USER_REVIEW,
    build_display_plan,
    expected_own_frame_camera_side,
    infer_broadcast_orientation,
    kickoff_own_half_orientation,
)
from src.team_identity import snapshot_key


def _has(match_id: int) -> bool:
    return (STATSBOMB_DATA_DIR / "three-sixty" / f"{match_id}.json").is_file()


def real(match_id: int):
    return pytest.mark.skipif(not _has(match_id), reason=f"raw StatsBomb match {match_id} not available")


@functools.lru_cache(maxsize=None)
def _pipeline(match_id: int):
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, counts = display_orientation_results(raw, results, inference, plan)
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    return raw, results, inference, plan, used, counts, display


def _index(raw, timestamp: str, event_type: str, period: int | None = None) -> int:
    return next(
        i for i, j in enumerate(raw.usable_frames)
        if j.event["timestamp"] == timestamp and j.event["type"]["name"] == event_type
        and (period is None or j.event["period"] == period)
    )


def _points(joined):
    return sorted(tuple(round(c, 6) for c in p["location"]) for p in joined.frame_360["freeze_frame"])


def _drawn_like_confident_copy(match_id: int, index: int) -> bool:
    raw, results, _, _, _, _, display = _pipeline(match_id)
    key = snapshot_key(raw.usable_frames[index].frame_360)
    copies = [k for k, j in enumerate(raw.usable_frames)
              if k != index and snapshot_key(j.frame_360) == key and results[k].orientation != "UNCERTAIN"]
    assert copies, "test frame must have a confident same-moment copy"
    return all(_points(display.usable_frames[index]) == _points(display.usable_frames[k]) for k in copies)


# --- France-Morocco period 1 (Task 1) ---------------------------------------

@real(3901833)
def test_france_morocco_period_1_is_derived_from_reviewed_period_2():
    raw, _, inference, plan, _, _, _ = _pipeline(3901833)
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    assert plan.transforms[2].source == "user_confirmed"
    assert plan.transforms[1].source == SOURCE_DERIVED_FROM_USER_REVIEW
    assert plan.provenance == PROVENANCE_USER_CONFIRMED_AND_DERIVED
    # P1: France attacks left -> right (defends the left goal); P2 the reverse.
    assert names[plan.transforms[1].reference_team_id] == "France Women's"
    assert names[plan.transforms[2].reference_team_id] == "Morocco Women's"
    # Independent camera geometry: P1's own majority supports the derivation.
    evidence = inference.period_evidence(1)
    assert not evidence.decided and evidence.support > 7 * evidence.opposition


@real(3901833)
@pytest.mark.parametrize("timestamp", [
    "00:02:32.873", "00:08:57.336", "00:09:53.531", "00:09:58.510", "00:21:54.459",
    "00:22:50.256", "00:37:25.059", "00:42:12.151", "00:43:11.000",
])
def test_france_morocco_period_1_flipped_frames_fixed(timestamp):
    raw, results, _, _, used, _, _ = _pipeline(3901833)
    index = _index(raw, timestamp, "Ball Receipt*", period=1)
    assert results[index].orientation == "UNCERTAIN"  # Problem A unchanged
    assert used[index].resolved_by == "display_duplicate_snapshot"
    assert _drawn_like_confident_copy(3901833, index)


@real(3901833)
def test_plan_supplies_expected_camera_side_for_camera_undecided_period():
    raw, _, inference, plan, _, _, _ = _pipeline(3901833)
    france = next(j.event["team"]["id"] for j in raw.usable_frames if j.event["team"]["name"] == "France Women's")
    morocco = next(j.event["team"]["id"] for j in raw.usable_frames if j.event["team"]["name"] == "Morocco Women's")
    assert expected_own_frame_camera_side(inference, 1, france) == "UNKNOWN"  # camera alone: undecided
    assert expected_own_frame_camera_side(inference, 1, france, plan) == CAMERA_SIDE_HIGH_Y
    assert expected_own_frame_camera_side(inference, 1, morocco, plan) == CAMERA_SIDE_LOW_Y


# --- duplicate evidence (Task 2) --------------------------------------------

@real(3773466)
def test_ball_receipt_flip_fixed_by_same_moment_duplicate():
    raw, results, _, _, used, _, _ = _pipeline(3773466)
    index = _index(raw, "00:13:22.642", "Ball Receipt*")
    assert results[index].orientation == "UNCERTAIN"
    assert used[index].orientation == "ROTATED_180"
    assert _drawn_like_confident_copy(3773466, index)


@real(3773526)
def test_disagreeing_confident_copies_follow_the_camera():
    raw, results, _, _, used, _, display = _pipeline(3773526)
    receipt = _index(raw, "00:07:30.149", "Ball Receipt*", period=2)
    carry = _index(raw, "00:07:30.149", "Carry", period=2)
    assert results[receipt].orientation == "NORMAL" and results[carry].orientation == "NORMAL"
    assert used[receipt].resolved_by == "display_duplicate_consensus"
    assert used[carry].orientation == "NORMAL"  # the camera-consistent copy keeps its result
    assert _points(display.usable_frames[receipt]) == _points(display.usable_frames[carry])


@pytest.mark.parametrize("match_id", [3773466, 3901833, 3788743, 3930160])
def test_every_duplicate_copy_is_drawn_identically(match_id):
    if not _has(match_id):
        pytest.skip("raw match not available")
    raw, _, _, _, _, _, display = _pipeline(match_id)
    groups: dict = {}
    for index, joined in enumerate(raw.usable_frames):
        groups.setdefault((joined.event["period"], snapshot_key(joined.frame_360)), []).append(index)
    for members in groups.values():
        drawn = {tuple(_points(display.usable_frames[m])) for m in members}
        assert len(drawn) == 1


# --- kick-off frames (Task 3) -----------------------------------------------

@pytest.mark.parametrize("match_id, period, timestamp", [
    (3773552, 1, "00:00:01.454"),
    (3788741, 1, "00:00:00.878"),
    (3877072, 2, "00:00:00.206"),
])
def test_flipped_kickoffs_fixed_by_own_half_rule(match_id, period, timestamp):
    if not _has(match_id):
        pytest.skip("raw match not available")
    raw, _, _, plan, used, _, display = _pipeline(match_id)
    index = _index(raw, timestamp, "Pass", period=period)
    assert used[index].resolved_by == "display_kickoff_own_half" and used[index].orientation == "ROTATED_180"
    # As drawn: the kicking team stands in the half it defends.
    kicking = raw.usable_frames[index].event["team"]["id"]
    attacks_right = not plan.rotates(period, kicking)
    shown = display.usable_frames[index].frame_360["freeze_frame"]
    mates = [p["location"][0] for p in shown if p.get("teammate") and not p.get("actor") and abs(p["location"][0] - 60) > 1.5]
    assert mates and all((x < 60) == attacks_right for x in mates)


@pytest.mark.parametrize("match_id, period, timestamp", [
    (3773631, 1, "00:00:00.992"),  # 5 players only; continuity supports the stored frame
    (3788741, 2, "00:00:00.189"),  # 7 of 8 players in own halves; camera reads the other way
])
def test_conflicting_kickoffs_are_left_as_stored(match_id, period, timestamp):
    if not _has(match_id):
        pytest.skip("raw match not available")
    raw, results, _, _, used, _, _ = _pipeline(match_id)
    index = _index(raw, timestamp, "Pass", period=period)
    assert used[index].orientation == "UNCERTAIN"


def test_kickoff_rule_needs_enough_decisive_players():
    def frame(xs_mates, xs_opponents):
        players = [{"teammate": True, "actor": True, "location": [60, 40]}]
        players += [{"teammate": True, "location": [x, 30]} for x in xs_mates]
        players += [{"teammate": False, "location": [x, 50]} for x in xs_opponents]
        return {"freeze_frame": players}
    assert kickoff_own_half_orientation(frame([40, 45, 50, 30], [70, 80, 90])) == "NORMAL"
    assert kickoff_own_half_orientation(frame([70, 80, 90, 75], [40, 45, 50])) == "ROTATED_180"
    assert kickoff_own_half_orientation(frame([40, 45], [70, 80])) is None          # too few
    assert kickoff_own_half_orientation(frame([40, 80, 45, 90], [70, 30, 80])) is None  # mixed
    assert kickoff_own_half_orientation(frame([60.5, 59.0, 61.0], [60.2, 59.9, 61.4])) is None  # halfway band


# --- camera + temporal continuity -------------------------------------------

@pytest.mark.parametrize("match_id, period, timestamp, event_type", [
    (3773631, 1, "00:25:39.675", "Ball Receipt*"),
    (3773587, 1, "00:07:00.007", "Dribbled Past"),
])
def test_camera_and_continuity_fix(match_id, period, timestamp, event_type):
    if not _has(match_id):
        pytest.skip("raw match not available")
    raw, results, _, _, used, _, _ = _pipeline(match_id)
    index = _index(raw, timestamp, event_type, period=period)
    assert results[index].orientation == "UNCERTAIN"
    assert used[index].resolved_by == "display_camera_continuity"


# --- no regression ------------------------------------------------------------

@real(3788743)
def test_confident_frames_unchanged_where_copies_agree():
    raw, results, _, _, used, counts, _ = _pipeline(3788743)
    assert counts["duplicate_consensus_overrides"] == 0
    assert all(u.orientation == r.orientation for u, r in zip(used, results) if r.orientation != "UNCERTAIN")


@pytest.mark.parametrize("match_id, timestamp", [(3902239, "00:23:04.965"), (3998846, "00:31:30.906")])
def test_previous_display_fixes_still_hold(match_id, timestamp):
    if not _has(match_id):
        pytest.skip("raw match not available")
    raw, _, _, _, used, _, display = _pipeline(match_id)
    index = _index(raw, timestamp, "Ball Receipt*")
    assert used[index].orientation == "ROTATED_180"
    shown = display.usable_frames[index]
    actor = next(p for p in shown.frame_360["freeze_frame"] if p.get("actor"))
    assert math.dist(actor["location"], shown.event["location"]) < 30


def test_synthetic_duplicate_algebra_across_teams():
    """An UNCERTAIN copy owned by the rotated team must get the correction
    that draws it exactly like its confident copy owned by the other team."""
    from src.animate_statsbomb_360_match import JoinedFrame, MatchFrames
    from src.orientation_correction import ORIENTATION_NORMAL, ORIENTATION_UNCERTAIN, OrientationResult
    from pathlib import Path

    snapshot = {"visible_area": [], "freeze_frame": [{"teammate": True, "actor": True, "location": [30.0, 20.0]}]}
    events = [
        {"id": "a", "team": {"id": 1}, "period": 1, "timestamp": "00:00:01.000", "location": [30.0, 20.0]},
        {"id": "b", "team": {"id": 2}, "period": 1, "timestamp": "00:00:01.000", "location": [90.0, 60.0]},
    ]
    match = MatchFrames(Path("x"), Path("y"), 2, 2, 2, 2, [JoinedFrame(e, dict(snapshot)) for e in events])
    inference = infer_broadcast_orientation([], [], (1, 2))
    plan = build_display_plan(inference, (1, 2), [1], anchor_period_1_reference=1)
    results = [OrientationResult(ORIENTATION_NORMAL, 0.9, 0.0), OrientationResult(ORIENTATION_UNCERTAIN, 0.1, 30.0)]
    used, _ = display_orientation_results(match, results, inference, plan)
    # Team 2 is rotated by the plan, so its copy must be corrected once more
    # (NORMAL xor rotated xor rotated-for-team-2 -> ROTATED_180) to be drawn like team 1's.
    assert used[1].orientation == "ROTATED_180"
    drawn = apply_display_plan_to_match(apply_orientation_correction_to_match(match, used), plan)
    assert _points(drawn.usable_frames[0]) == _points(drawn.usable_frames[1])
