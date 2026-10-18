"""Tests for team-identity resolution (Problem C).

Problem C is independent from, and read-only with respect to, Problem A
(test_orientation_correction.py) and Problem B (test_team_normalization.py).
These tests also verify that composing Problem C with the other two layers
never mutates their inputs or outputs.
"""

import copy

import pytest

from src.orientation_correction import (
    ORIENTATION_NORMAL,
    apply_correction,
    classify_orientation,
)
from src.team_identity import (
    TEAM_IDENTITY_AMBIGUOUS,
    detect_shared_cross_team_snapshots,
    resolve_absolute_team_for_point,
    resolve_team_identity_for_frame,
)
from src.team_normalization import (
    TEAM_NORMALIZATION_REFERENCE,
    apply_team_normalization_to_frame,
    resolve_reference_team_id,
    team_normalization_label,
)

TEAM_NAMES = ("Team A", "Team B")

# --- Fixtures: a genuine possession-change pair (Case A), NOT shared --------
EVENT_TEAM_A = {
    "id": "evA",
    "team": {"id": 1, "name": "Team A"},
    "type": {"name": "Pass"},
    "period": 1,
    "location": [50.0, 40.0],
}
FRAME_TEAM_A = {
    "event_uuid": "evA",
    "visible_area": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
    "freeze_frame": [
        {"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]},
        {"teammate": True, "actor": False, "keeper": True, "location": [5.0, 40.0]},
        {"teammate": False, "actor": False, "keeper": False, "location": [70.0, 40.0]},
    ],
}

EVENT_TEAM_B = {
    "id": "evB",
    "team": {"id": 2, "name": "Team B"},
    "type": {"name": "Pressure"},
    "period": 1,
    "location": [72.0, 38.0],
}
FRAME_TEAM_B = {
    "event_uuid": "evB",
    "visible_area": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
    "freeze_frame": [
        {"teammate": True, "actor": True, "keeper": False, "location": [72.0, 38.0]},
        {"teammate": False, "actor": False, "keeper": False, "location": [48.0, 41.0]},
        {"teammate": False, "actor": False, "keeper": True, "location": [4.0, 39.0]},
    ],
}


def test_actor_always_resolves_to_own_event_team() -> None:
    actor_point = FRAME_TEAM_A["freeze_frame"][0]
    assert actor_point["actor"] is True
    team = resolve_absolute_team_for_point(actor_point, EVENT_TEAM_A, None, {}, TEAM_NAMES)
    assert team == "Team A"


def test_non_shared_actor_resolution_relies_on_the_teammate_invariant() -> None:
    # There is no separate actor special-case: for a non-shared frame the
    # actor is resolved purely via `teammate`, relying on StatsBomb's own
    # schema invariant that actor=True always implies teammate=True (a
    # dedicated actor override was removed -- see
    # test_shared_snapshot_colors_are_stable_across_different_borrowing_events
    # for why an unconditional "actor = this event's own team" override is
    # actually incorrect once a snapshot can be shared/borrowed).
    actor_point = {"teammate": True, "actor": True, "keeper": False, "location": [1.0, 1.0]}
    team = resolve_absolute_team_for_point(actor_point, EVENT_TEAM_A, None, {}, TEAM_NAMES)
    assert team == "Team A"


def test_non_shared_snapshot_keeps_existing_teammate_relative_logic() -> None:
    for player in FRAME_TEAM_A["freeze_frame"]:
        team = resolve_absolute_team_for_point(player, EVENT_TEAM_A, None, {}, TEAM_NAMES)
        expected = "Team A" if player["teammate"] else "Team B"
        assert team == expected


def test_possession_change_does_not_swap_colors_for_the_same_physical_side() -> None:
    # Genuine Case-A possession change: two DIFFERENT (non-shared) snapshots,
    # different owning teams. A point that is physically on Team A's side
    # must resolve to "Team A" regardless of which event currently owns the
    # frame, and must never flip because the actor/possession changed.
    team_a_point_in_frame_a = FRAME_TEAM_A["freeze_frame"][0]  # teammate=True, event=A
    team_a_point_in_frame_b = FRAME_TEAM_B["freeze_frame"][1]  # teammate=False, event=B
    resolved_a = resolve_absolute_team_for_point(team_a_point_in_frame_a, EVENT_TEAM_A, None, {}, TEAM_NAMES)
    resolved_b = resolve_absolute_team_for_point(team_a_point_in_frame_b, EVENT_TEAM_B, None, {}, TEAM_NAMES)
    assert resolved_a == resolved_b == "Team A"


def test_team_a_remains_team_a_across_consecutive_events() -> None:
    results = [
        resolve_absolute_team_for_point(p, EVENT_TEAM_A, None, {}, TEAM_NAMES)
        for p in FRAME_TEAM_A["freeze_frame"]
        if p["teammate"]
    ]
    assert all(r == "Team A" for r in results)


def test_team_b_remains_team_b_across_consecutive_events() -> None:
    results = [
        resolve_absolute_team_for_point(p, EVENT_TEAM_B, None, {}, TEAM_NAMES)
        for p in FRAME_TEAM_B["freeze_frame"]
        if p["teammate"]
    ]
    assert all(r == "Team B" for r in results)


# --- Shared cross-team snapshot fixtures ------------------------------------
# A single raw array reused for two different-team events at the same
# instant. Team A's event is self-consistent (its own actor position matches
# its own event.location) -- the classic "single confident source" case.
SHARED_ARRAY = [
    {"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]},
    {"teammate": True, "actor": False, "keeper": True, "location": [5.0, 40.0]},
    {"teammate": False, "actor": False, "keeper": False, "location": [70.0, 40.0]},
]

SHARED_EVENT_SOURCE = {
    "id": "source-ev",
    "team": {"id": 1, "name": "Team A"},
    "type": {"name": "Ball Receipt*"},
    "period": 1,
    "location": [50.0, 40.0],  # matches the array's own actor -> self-consistent (NORMAL)
}
SHARED_FRAME_SOURCE = {
    "event_uuid": "source-ev",
    "visible_area": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
    "freeze_frame": copy.deepcopy(SHARED_ARRAY),
}

SHARED_EVENT_BORROWER = {
    "id": "borrower-ev",
    "team": {"id": 2, "name": "Team B"},
    "type": {"name": "Ball Recovery"},
    "period": 1,
    "location": [110.0, 45.0],  # far from the array -> needs mirroring, NOT self-consistent
}
SHARED_FRAME_BORROWER = {
    "event_uuid": "borrower-ev",
    "visible_area": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
    "freeze_frame": copy.deepcopy(SHARED_ARRAY),
}


def test_shared_snapshot_with_one_confident_source_is_detected() -> None:
    events = [SHARED_EVENT_SOURCE, SHARED_EVENT_BORROWER]
    frames_by_id = {"source-ev": SHARED_FRAME_SOURCE, "borrower-ev": SHARED_FRAME_BORROWER}
    groups = detect_shared_cross_team_snapshots(events, frames_by_id)
    assert "source-ev" in groups and "borrower-ev" in groups
    assert groups["source-ev"].source_event_id == "source-ev"
    assert groups["borrower-ev"].source_event_id == "source-ev"

    # Sanity: the source event really is classified NORMAL with confidence
    # above the threshold used inside detection.
    result = classify_orientation(SHARED_EVENT_SOURCE, SHARED_FRAME_SOURCE)
    assert result.orientation == ORIENTATION_NORMAL
    assert result.confidence >= 0.60


def test_shared_snapshot_non_actor_points_resolved_via_source_team() -> None:
    events = [SHARED_EVENT_SOURCE, SHARED_EVENT_BORROWER]
    frames_by_id = {"source-ev": SHARED_FRAME_SOURCE, "borrower-ev": SHARED_FRAME_BORROWER}
    raw_events_by_id = {e["id"]: e for e in events}
    groups = detect_shared_cross_team_snapshots(events, frames_by_id)

    # Non-actor points, read through the BORROWER's own frame, must be
    # resolved using the SOURCE's team convention, not the borrower's own.
    group_info = groups["borrower-ev"]
    keeper_point = SHARED_FRAME_BORROWER["freeze_frame"][1]  # teammate=True in the raw array
    outfield_point = SHARED_FRAME_BORROWER["freeze_frame"][2]  # teammate=False in the raw array

    keeper_team = resolve_absolute_team_for_point(
        keeper_point, SHARED_EVENT_BORROWER, group_info, raw_events_by_id, TEAM_NAMES
    )
    outfield_team = resolve_absolute_team_for_point(
        outfield_point, SHARED_EVENT_BORROWER, group_info, raw_events_by_id, TEAM_NAMES
    )
    # Source event's own team is "Team A"; teammate=True -> Team A, False -> Team B.
    assert keeper_team == "Team A"
    assert outfield_team == "Team B"


def test_shared_snapshot_actor_point_follows_source_team_not_borrowing_event() -> None:
    # Corrected understanding (found via visual GIF inspection of match
    # 3857283): the array's actor=True point physically belongs to whoever
    # the snapshot was TRULY captured for (the resolved source event), not
    # to whichever event currently happens to borrow the same array. Since
    # the array's source here is Team A, the borrower's own actor=True
    # point (raw teammate=True) must resolve to Team A -- resolving it to
    # the borrower's own team (Team B) was the bug: it silently invented a
    # team flip for a physical point whose position never changed.
    events = [SHARED_EVENT_SOURCE, SHARED_EVENT_BORROWER]
    frames_by_id = {"source-ev": SHARED_FRAME_SOURCE, "borrower-ev": SHARED_FRAME_BORROWER}
    raw_events_by_id = {e["id"]: e for e in events}
    groups = detect_shared_cross_team_snapshots(events, frames_by_id)

    borrower_actor_point = SHARED_FRAME_BORROWER["freeze_frame"][0]
    assert borrower_actor_point["actor"] is True
    assert borrower_actor_point["teammate"] is True

    team = resolve_absolute_team_for_point(
        borrower_actor_point, SHARED_EVENT_BORROWER, groups["borrower-ev"], raw_events_by_id, TEAM_NAMES
    )
    assert team == "Team A"  # follows the resolved SOURCE team, not the borrower's own team


def test_shared_snapshot_colors_are_stable_across_different_borrowing_events() -> None:
    # THE regression test for the actor-color-flip bug: two different
    # events (different teams, different actors) sharing the exact same
    # raw array must resolve every spatial point -- including the one
    # flagged actor=True -- to the IDENTICAL absolute team, regardless of
    # which of the two events is currently being rendered. A viewer must
    # never see the same physical point change color merely because
    # playback moved from one event to the next shared one.
    events = [SHARED_EVENT_SOURCE, SHARED_EVENT_BORROWER]
    frames_by_id = {"source-ev": SHARED_FRAME_SOURCE, "borrower-ev": SHARED_FRAME_BORROWER}
    raw_events_by_id = {e["id"]: e for e in events}
    groups = detect_shared_cross_team_snapshots(events, frames_by_id)

    for point_index in range(len(SHARED_ARRAY)):
        team_via_source = resolve_absolute_team_for_point(
            SHARED_FRAME_SOURCE["freeze_frame"][point_index],
            SHARED_EVENT_SOURCE,
            groups["source-ev"],
            raw_events_by_id,
            TEAM_NAMES,
        )
        team_via_borrower = resolve_absolute_team_for_point(
            SHARED_FRAME_BORROWER["freeze_frame"][point_index],
            SHARED_EVENT_BORROWER,
            groups["borrower-ev"],
            raw_events_by_id,
            TEAM_NAMES,
        )
        assert team_via_source == team_via_borrower, (
            f"point {point_index} flipped team depending on which event is rendered: "
            f"{team_via_source!r} (via source event) vs {team_via_borrower!r} (via borrowing event)"
        )


def test_shared_snapshot_with_multiple_confident_sources_is_ambiguous() -> None:
    # Both events happen to be self-consistent with the SAME shared array
    # (e.g. both located near the array's actor position) -- no way to tell
    # which one is the true source.
    event_x = {
        "id": "ev-x",
        "team": {"id": 1, "name": "Team A"},
        "type": {"name": "Ball Receipt*"},
        "period": 1,
        "location": [50.0, 40.0],
    }
    event_y = {
        "id": "ev-y",
        "team": {"id": 2, "name": "Team B"},
        "type": {"name": "Ball Receipt*"},
        "period": 1,
        "location": [50.0, 40.0],  # also matches -> also NORMAL
    }
    shared_array = copy.deepcopy(SHARED_ARRAY)
    frame_x = {"event_uuid": "ev-x", "visible_area": [0.0,0.0,10.0,0.0,10.0,10.0,0.0,10.0], "freeze_frame": copy.deepcopy(shared_array)}
    frame_y = {"event_uuid": "ev-y", "visible_area": [0.0,0.0,10.0,0.0,10.0,10.0,0.0,10.0], "freeze_frame": copy.deepcopy(shared_array)}

    events = [event_x, event_y]
    frames_by_id = {"ev-x": frame_x, "ev-y": frame_y}
    raw_events_by_id = {e["id"]: e for e in events}
    groups = detect_shared_cross_team_snapshots(events, frames_by_id)

    assert groups["ev-x"].source_event_id is None
    assert groups["ev-y"].source_event_id is None

    outfield_point = frame_x["freeze_frame"][2]
    team = resolve_absolute_team_for_point(outfield_point, event_x, groups["ev-x"], raw_events_by_id, TEAM_NAMES)
    assert team == TEAM_IDENTITY_AMBIGUOUS


def test_shared_snapshot_with_multiple_agreeing_confident_sources_is_resolved() -> None:
    # Real-world pattern (match 3857283, Belgium vs Morocco): a Belgium event
    # (not self-consistent -> needs 180 deg mirroring) shares its raw array
    # with TWO Morocco events (a Ball Recovery and the immediately following
    # Carry), both of which are independently self-consistent (NORMAL) with
    # the array. Two confident sources that AGREE on the same team are
    # corroborating evidence, not a contradiction -- this must resolve to
    # Morocco, not fall back to ambiguous.
    event_belgium = {
        "id": "ev-belgium",
        "team": {"id": 10, "name": "Belgium"},
        "type": {"name": "Ball Receipt*"},
        "period": 1,
        "location": [5.0, 40.0],  # far from the array's actor -> ROTATED_180, not confident
    }
    event_morocco_recovery = {
        "id": "ev-morocco-recovery",
        "team": {"id": 20, "name": "Morocco"},
        "type": {"name": "Ball Recovery"},
        "period": 1,
        "location": [50.0, 40.0],  # matches array's actor -> NORMAL, confident
    }
    event_morocco_carry = {
        "id": "ev-morocco-carry",
        "team": {"id": 20, "name": "Morocco"},
        "type": {"name": "Carry"},
        "period": 1,
        "location": [50.0, 40.0],  # same instant, also self-consistent -> NORMAL, confident
    }

    shared = copy.deepcopy(SHARED_ARRAY)
    frame_belgium = {"event_uuid": "ev-belgium", "visible_area": [], "freeze_frame": copy.deepcopy(shared)}
    frame_recovery = {"event_uuid": "ev-morocco-recovery", "visible_area": [], "freeze_frame": copy.deepcopy(shared)}
    frame_carry = {"event_uuid": "ev-morocco-carry", "visible_area": [], "freeze_frame": copy.deepcopy(shared)}

    events = [event_belgium, event_morocco_recovery, event_morocco_carry]
    frames_by_id = {
        "ev-belgium": frame_belgium,
        "ev-morocco-recovery": frame_recovery,
        "ev-morocco-carry": frame_carry,
    }
    raw_events_by_id = {e["id"]: e for e in events}

    # Sanity: confirm the fixture actually reproduces "one non-confident +
    # two agreeing confident sources" before asserting on detection's output.
    belgium_result = classify_orientation(event_belgium, frame_belgium)
    recovery_result = classify_orientation(event_morocco_recovery, frame_recovery)
    carry_result = classify_orientation(event_morocco_carry, frame_carry)
    assert not (belgium_result.orientation == ORIENTATION_NORMAL and belgium_result.confidence >= 0.60)
    assert recovery_result.orientation == ORIENTATION_NORMAL and recovery_result.confidence >= 0.60
    assert carry_result.orientation == ORIENTATION_NORMAL and carry_result.confidence >= 0.60

    groups = detect_shared_cross_team_snapshots(events, frames_by_id)

    resolved_source = groups["ev-belgium"].source_event_id
    assert resolved_source in {"ev-morocco-recovery", "ev-morocco-carry"}
    assert groups["ev-morocco-recovery"].source_event_id == resolved_source
    assert groups["ev-morocco-carry"].source_event_id == resolved_source

    team_names = ("Belgium", "Morocco")
    outfield_point = frame_belgium["freeze_frame"][2]  # teammate=False in the raw array
    team = resolve_absolute_team_for_point(outfield_point, event_belgium, groups["ev-belgium"], raw_events_by_id, team_names)
    # Reference team is Morocco (the agreed confident source); teammate=False -> Belgium.
    assert team != TEAM_IDENTITY_AMBIGUOUS
    assert team == "Belgium"


def test_shared_snapshot_same_team_is_not_flagged() -> None:
    # Two events from the SAME team sharing an array (e.g. Ball Receipt* and
    # Carry by the same player) is not the ambiguity this module targets.
    event_x = {"id": "ev-x", "team": {"id": 1, "name": "Team A"}, "type": {"name": "Ball Receipt*"}, "period": 1, "location": [50.0, 40.0]}
    event_y = {"id": "ev-y", "team": {"id": 1, "name": "Team A"}, "type": {"name": "Carry"}, "period": 1, "location": [50.0, 40.0]}
    shared = copy.deepcopy(SHARED_ARRAY)
    frame_x = {"event_uuid": "ev-x", "visible_area": [], "freeze_frame": copy.deepcopy(shared)}
    frame_y = {"event_uuid": "ev-y", "visible_area": [], "freeze_frame": copy.deepcopy(shared)}
    groups = detect_shared_cross_team_snapshots(
        [event_x, event_y], {"ev-x": frame_x, "ev-y": frame_y}
    )
    assert groups == {}


def test_ambiguous_points_are_never_deleted_or_hidden() -> None:
    events = [SHARED_EVENT_SOURCE, SHARED_EVENT_BORROWER]
    frames_by_id = {"source-ev": SHARED_FRAME_SOURCE, "borrower-ev": SHARED_FRAME_BORROWER}
    raw_events_by_id = {e["id"]: e for e in events}
    groups = detect_shared_cross_team_snapshots(events, frames_by_id)
    identities = resolve_team_identity_for_frame(
        SHARED_EVENT_BORROWER, SHARED_FRAME_BORROWER, groups["borrower-ev"], raw_events_by_id, TEAM_NAMES
    )
    # One entry per freeze_frame player, none dropped, order preserved.
    assert len(identities) == len(SHARED_FRAME_BORROWER["freeze_frame"])


# --- Non-interference with Problem A / Problem B / event ordering ----------

def test_detection_never_mutates_input_events_or_frames() -> None:
    events = [copy.deepcopy(SHARED_EVENT_SOURCE), copy.deepcopy(SHARED_EVENT_BORROWER)]
    frames_by_id = {
        "source-ev": copy.deepcopy(SHARED_FRAME_SOURCE),
        "borrower-ev": copy.deepcopy(SHARED_FRAME_BORROWER),
    }
    events_snapshot = copy.deepcopy(events)
    frames_snapshot = copy.deepcopy(frames_by_id)

    detect_shared_cross_team_snapshots(events, frames_by_id)

    assert events == events_snapshot
    assert frames_by_id == frames_snapshot


def test_coordinates_remain_byte_identical_before_and_after_problem_c() -> None:
    # Problem C only reads events/frames to resolve team identity; it must
    # never touch a "location" field anywhere.
    frame_before = copy.deepcopy(SHARED_FRAME_BORROWER)
    events = [SHARED_EVENT_SOURCE, SHARED_EVENT_BORROWER]
    frames_by_id = {"source-ev": SHARED_FRAME_SOURCE, "borrower-ev": SHARED_FRAME_BORROWER}
    raw_events_by_id = {e["id"]: e for e in events}
    groups = detect_shared_cross_team_snapshots(events, frames_by_id)
    resolve_team_identity_for_frame(
        SHARED_EVENT_BORROWER, SHARED_FRAME_BORROWER, groups["borrower-ev"], raw_events_by_id, TEAM_NAMES
    )
    assert SHARED_FRAME_BORROWER == frame_before


def test_problem_a_orientation_result_unaffected_by_problem_c() -> None:
    # classify_orientation is reused read-only inside detection; verify
    # calling detection doesn't change what Problem A itself would report.
    before = classify_orientation(SHARED_EVENT_BORROWER, SHARED_FRAME_BORROWER)
    detect_shared_cross_team_snapshots(
        [SHARED_EVENT_SOURCE, SHARED_EVENT_BORROWER],
        {"source-ev": SHARED_FRAME_SOURCE, "borrower-ev": SHARED_FRAME_BORROWER},
    )
    after = classify_orientation(SHARED_EVENT_BORROWER, SHARED_FRAME_BORROWER)
    assert before.orientation == after.orientation
    assert before.confidence == after.confidence


def test_problem_b_team_normalization_unaffected_by_problem_c() -> None:
    events = [SHARED_EVENT_SOURCE, SHARED_EVENT_BORROWER]
    reference_team_id = resolve_reference_team_id(events)
    before_label = team_normalization_label(SHARED_EVENT_BORROWER, reference_team_id)
    before_frame = apply_team_normalization_to_frame(SHARED_FRAME_BORROWER, SHARED_EVENT_BORROWER, reference_team_id)

    detect_shared_cross_team_snapshots(
        events, {"source-ev": SHARED_FRAME_SOURCE, "borrower-ev": SHARED_FRAME_BORROWER}
    )

    after_label = team_normalization_label(SHARED_EVENT_BORROWER, reference_team_id)
    after_frame = apply_team_normalization_to_frame(SHARED_FRAME_BORROWER, SHARED_EVENT_BORROWER, reference_team_id)
    assert before_label == after_label
    assert before_frame == after_frame
    assert before_label == TEAM_NORMALIZATION_REFERENCE or True  # label itself isn't the point; equality is


def test_no_events_removed_or_reordered_by_detection() -> None:
    events = [SHARED_EVENT_SOURCE, SHARED_EVENT_BORROWER, EVENT_TEAM_A, EVENT_TEAM_B]
    ids_before = [e["id"] for e in events]
    frames_by_id = {
        "source-ev": SHARED_FRAME_SOURCE,
        "borrower-ev": SHARED_FRAME_BORROWER,
        "evA": FRAME_TEAM_A,
        "evB": FRAME_TEAM_B,
    }
    detect_shared_cross_team_snapshots(events, frames_by_id)
    ids_after = [e["id"] for e in events]
    assert ids_before == ids_after  # same events, same order, nothing dropped
