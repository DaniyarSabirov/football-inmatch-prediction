"""Tests for team-based coordinate normalization (Problem B).

Problem B is independent from, and applied after, the event-level
orientation correction (Problem A, tested in test_orientation_correction.py).
These tests deliberately also exercise that Problem A keeps working
unmodified, and that the two layers compose without interfering.
"""

import pytest

from src.orientation_correction import (
    ORIENTATION_NORMAL,
    ORIENTATION_ROTATED_180,
    ORIENTATION_UNCERTAIN,
    apply_correction,
    classify_orientation,
)
from src.team_normalization import (
    PHYSICALLY_FLIPPED_PERIODS,
    TEAM_NORMALIZATION_MIRRORED,
    TEAM_NORMALIZATION_REFERENCE,
    apply_team_normalization_to_event,
    apply_team_normalization_to_frame,
    period_requires_physical_end_swap,
    resolve_physical_reference_team_id,
    resolve_reference_team_id,
    team_normalization_label,
)

REFERENCE_TEAM_ID = 100
OTHER_TEAM_ID = 200

REFERENCE_EVENT = {
    "id": "ref-event",
    "team": {"id": REFERENCE_TEAM_ID, "name": "Reference FC"},
    "type": {"name": "Pass"},
    "period": 1,
    "location": [30.0, 20.0],
    "pass": {"end_location": [50.0, 25.0]},
}

MIRRORED_EVENT = {
    "id": "other-event",
    "team": {"id": OTHER_TEAM_ID, "name": "Other FC"},
    "type": {"name": "Pass"},
    "period": 1,
    "location": [30.0, 20.0],
    "pass": {"end_location": [50.0, 25.0]},
}

SAMPLE_FRAME = {
    "event_uuid": "other-event",
    "visible_area": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
    "freeze_frame": [
        {"teammate": True, "actor": True, "keeper": False, "location": [30.0, 20.0]},
        {"teammate": True, "actor": False, "keeper": True, "location": [5.0, 40.0]},
        {"teammate": False, "actor": False, "keeper": False, "location": [70.0, 60.0]},
    ],
}


def test_events_of_two_ids_resolve_to_first_seen_team_as_reference() -> None:
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    assert resolve_reference_team_id(events) == REFERENCE_TEAM_ID
    # Order matters only in that the FIRST team id encountered becomes the
    # reference -- verify the complementary ordering picks the other team.
    assert resolve_reference_team_id([MIRRORED_EVENT, REFERENCE_EVENT]) == OTHER_TEAM_ID


def test_resolve_reference_team_id_requires_exactly_two_teams() -> None:
    with pytest.raises(ValueError):
        resolve_reference_team_id([REFERENCE_EVENT])


# --- physical_reference_team_name override -----------------------------------
# Added after the 2026-10-21 broadcast-footage investigation
# (docs/physical_pitch_orientation_validation.md): the automatic pick above
# is confirmed to have no relationship to real camera/stadium orientation
# (it is simply whichever team's first 360-matched event is chronologically
# earliest -- in practice, whichever team kicks off). This override lets a
# caller with independent, verified physical-orientation evidence (e.g. a
# broadcast screenshot) select the correct reference team explicitly,
# without changing the default (analytical) behavior for anyone who does
# not pass it.

def test_override_none_preserves_automatic_behavior() -> None:
    # Explicitly passing None must be identical to omitting the argument.
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    assert (
        resolve_reference_team_id(events)
        == resolve_reference_team_id(events, physical_reference_team_name=None)
        == REFERENCE_TEAM_ID
    )


def test_override_can_select_the_team_that_would_not_be_picked_automatically() -> None:
    # Automatic pick (first-seen) would choose REFERENCE_TEAM_ID here; the
    # override must be able to select the OTHER team instead.
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    assert resolve_reference_team_id(events) == REFERENCE_TEAM_ID
    assert (
        resolve_reference_team_id(events, physical_reference_team_name="Other FC")
        == OTHER_TEAM_ID
    )


def test_override_matches_case_insensitively() -> None:
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    assert (
        resolve_reference_team_id(events, physical_reference_team_name="other fc")
        == OTHER_TEAM_ID
    )
    assert (
        resolve_reference_team_id(events, physical_reference_team_name="OTHER FC")
        == OTHER_TEAM_ID
    )


def test_override_with_non_matching_name_raises() -> None:
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    with pytest.raises(ValueError):
        resolve_reference_team_id(events, physical_reference_team_name="Nonexistent FC")


def test_override_still_requires_exactly_two_teams_first() -> None:
    # The team-count check must run before the name match is attempted, so
    # a malformed/incomplete event list fails with the same clear error
    # regardless of whether an override was requested.
    with pytest.raises(ValueError):
        resolve_reference_team_id([REFERENCE_EVENT], physical_reference_team_name="Reference FC")


def test_override_flips_which_events_team_normalization_treats_as_mirrored() -> None:
    # End-to-end: swapping the reference team via the override must swap
    # which of the two teams' coordinates get mirrored, using the exact
    # same (already-tested) mirror formula -- no new transform is
    # introduced, only which team it is applied to changes.
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    swapped_reference_id = resolve_reference_team_id(
        events, physical_reference_team_name="Other FC"
    )
    assert swapped_reference_id == OTHER_TEAM_ID

    # Under the swapped reference, the team that was previously REFERENCE
    # (and unmirrored) is now MIRRORED, and vice versa.
    assert team_normalization_label(REFERENCE_EVENT, swapped_reference_id) == TEAM_NORMALIZATION_MIRRORED
    assert team_normalization_label(MIRRORED_EVENT, swapped_reference_id) == TEAM_NORMALIZATION_REFERENCE

    formerly_reference_normalized = apply_team_normalization_to_event(REFERENCE_EVENT, swapped_reference_id)
    assert formerly_reference_normalized["location"] == pytest.approx([90.0, 60.0])

    formerly_mirrored_normalized = apply_team_normalization_to_event(MIRRORED_EVENT, swapped_reference_id)
    assert formerly_mirrored_normalized["location"] == pytest.approx([30.0, 20.0])


# --- 1. Reference team coordinates remain unchanged -------------------------

def test_reference_team_event_is_unchanged() -> None:
    normalized = apply_team_normalization_to_event(REFERENCE_EVENT, REFERENCE_TEAM_ID)
    assert normalized["location"] == [30.0, 20.0]
    assert normalized["pass"]["end_location"] == [50.0, 25.0]
    assert team_normalization_label(REFERENCE_EVENT, REFERENCE_TEAM_ID) == TEAM_NORMALIZATION_REFERENCE


def test_reference_team_frame_is_unchanged() -> None:
    reference_frame = {**SAMPLE_FRAME, "event_uuid": "ref-event"}
    normalized = apply_team_normalization_to_frame(reference_frame, REFERENCE_EVENT, REFERENCE_TEAM_ID)
    assert normalized["freeze_frame"] == reference_frame["freeze_frame"]
    assert normalized["visible_area"] == reference_frame["visible_area"]


# --- 2. Corner-case coordinate transform -------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ([0.0, 0.0], [120.0, 80.0]),
        ([120.0, 80.0], [0.0, 0.0]),
        ([60.0, 40.0], [60.0, 40.0]),
    ],
)
def test_non_reference_event_location_transform(raw, expected) -> None:
    event = {**MIRRORED_EVENT, "location": raw}
    normalized = apply_team_normalization_to_event(event, REFERENCE_TEAM_ID)
    assert normalized["location"] == pytest.approx(expected)


# --- 3. freeze_frame player coordinates are transformed ----------------------

def test_non_reference_freeze_frame_players_are_mirrored() -> None:
    normalized = apply_team_normalization_to_frame(SAMPLE_FRAME, MIRRORED_EVENT, REFERENCE_TEAM_ID)
    original_players = SAMPLE_FRAME["freeze_frame"]
    mirrored_players = normalized["freeze_frame"]
    for original, mirrored in zip(original_players, mirrored_players):
        ox, oy = original["location"]
        mx, my = mirrored["location"]
        assert mx == pytest.approx(120.0 - ox)
        assert my == pytest.approx(80.0 - oy)


# --- 4. visible_area polygon coordinates are transformed ---------------------

def test_non_reference_visible_area_is_mirrored() -> None:
    normalized = apply_team_normalization_to_frame(SAMPLE_FRAME, MIRRORED_EVENT, REFERENCE_TEAM_ID)
    original_area = SAMPLE_FRAME["visible_area"]
    mirrored_area = normalized["visible_area"]
    for i in range(0, len(original_area), 2):
        assert mirrored_area[i] == pytest.approx(120.0 - original_area[i])
        assert mirrored_area[i + 1] == pytest.approx(80.0 - original_area[i + 1])


# --- 5. event.location is transformed ----------------------------------------

def test_non_reference_event_location_is_mirrored() -> None:
    normalized = apply_team_normalization_to_event(MIRRORED_EVENT, REFERENCE_TEAM_ID)
    assert normalized["location"] == pytest.approx([90.0, 60.0])


# --- 6. event end_location is transformed when present -----------------------

def test_non_reference_end_location_is_mirrored_when_present() -> None:
    normalized = apply_team_normalization_to_event(MIRRORED_EVENT, REFERENCE_TEAM_ID)
    assert normalized["pass"]["end_location"] == pytest.approx([70.0, 55.0])


def test_event_without_end_location_is_not_affected() -> None:
    event = {
        "id": "shot-event",
        "team": {"id": OTHER_TEAM_ID, "name": "Other FC"},
        "type": {"name": "Pressure"},
        "period": 1,
        "location": [10.0, 10.0],
    }
    normalized = apply_team_normalization_to_event(event, REFERENCE_TEAM_ID)
    assert normalized["location"] == pytest.approx([110.0, 70.0])
    assert "pass" not in normalized


# --- 7. teammate / actor / keeper flags remain unchanged ---------------------

def test_flags_are_never_touched_by_team_normalization() -> None:
    normalized = apply_team_normalization_to_frame(SAMPLE_FRAME, MIRRORED_EVENT, REFERENCE_TEAM_ID)
    for original, mirrored in zip(SAMPLE_FRAME["freeze_frame"], normalized["freeze_frame"]):
        assert mirrored["teammate"] == original["teammate"]
        assert mirrored["actor"] == original["actor"]
        assert mirrored["keeper"] == original["keeper"]


# --- 8. team_id remains unchanged --------------------------------------------

def test_team_id_and_identity_fields_are_never_touched() -> None:
    normalized = apply_team_normalization_to_event(MIRRORED_EVENT, REFERENCE_TEAM_ID)
    assert normalized["team"]["id"] == OTHER_TEAM_ID
    assert normalized["team"]["name"] == "Other FC"
    assert normalized["id"] == "other-event"
    assert normalized["type"]["name"] == "Pass"
    assert normalized["period"] == 1


def test_frame_uuid_is_never_touched() -> None:
    normalized = apply_team_normalization_to_frame(SAMPLE_FRAME, MIRRORED_EVENT, REFERENCE_TEAM_ID)
    assert normalized["event_uuid"] == SAMPLE_FRAME["event_uuid"]


# --- 9. Problem-A orientation correction still works (unmodified) -----------

def test_problem_a_classifier_and_correction_are_unaffected() -> None:
    # Same fixture family as test_orientation_correction.py's known cases --
    # a self-consistent event should still classify NORMAL and be untouched
    # by apply_correction, independent of anything in team_normalization.py.
    event = {"id": "e1", "type": {"name": "Pass"}, "location": [50.0, 40.0]}
    frame = {
        "event_uuid": "e1",
        "freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]}],
    }
    result = classify_orientation(event, frame)
    assert result.orientation == ORIENTATION_NORMAL
    assert apply_correction(frame, result.orientation)["freeze_frame"] == frame["freeze_frame"]


# --- 10. UNCERTAIN Problem-A cases are still normalized by Problem B --------

def test_uncertain_problem_a_event_is_still_team_mirrored() -> None:
    # Deliberately contradictory geometry (as in the borderline fixture from
    # test_orientation_correction.py) so Problem A resolves to UNCERTAIN --
    # Problem B must still apply its own, independent, deterministic
    # team-based mirror on top of it; it must not skip or force anything.
    event = {
        "id": "borderline",
        "team": {"id": OTHER_TEAM_ID, "name": "Other FC"},
        "type": {"name": "Ball Receipt*"},
        "location": [57.1, 53.6],
    }
    frame = {
        "event_uuid": "borderline",
        "freeze_frame": [
            {"teammate": True, "actor": True, "keeper": False, "location": [59.9, 26.2]},
            {"teammate": True, "actor": False, "keeper": False, "location": [55.5, 51.9]},
        ],
    }
    result = classify_orientation(event, frame)
    assert result.orientation == ORIENTATION_UNCERTAIN

    problem_a_frame = apply_correction(frame, result.orientation)
    assert problem_a_frame["freeze_frame"] == frame["freeze_frame"]  # untouched, as expected

    problem_b_frame = apply_team_normalization_to_frame(problem_a_frame, event, REFERENCE_TEAM_ID)
    for original, mirrored in zip(frame["freeze_frame"], problem_b_frame["freeze_frame"]):
        ox, oy = original["location"]
        mx, my = mirrored["location"]
        assert mx == pytest.approx(120.0 - ox)
        assert my == pytest.approx(80.0 - oy)

    problem_b_event = apply_team_normalization_to_event(event, REFERENCE_TEAM_ID)
    assert problem_b_event["location"] == pytest.approx([62.9, 26.4])


# --- 11 & 12. Team normalization does not depend on period ------------------

def test_team_normalization_is_identical_across_periods() -> None:
    period_1_event = {**MIRRORED_EVENT, "period": 1}
    period_2_event = {**MIRRORED_EVENT, "period": 2}
    extra_time_event = {**MIRRORED_EVENT, "period": 3}

    label_p1 = team_normalization_label(period_1_event, REFERENCE_TEAM_ID)
    label_p2 = team_normalization_label(period_2_event, REFERENCE_TEAM_ID)
    label_et = team_normalization_label(extra_time_event, REFERENCE_TEAM_ID)

    assert label_p1 == label_p2 == label_et == TEAM_NORMALIZATION_MIRRORED

    normalized_p1 = apply_team_normalization_to_event(period_1_event, REFERENCE_TEAM_ID)
    normalized_p2 = apply_team_normalization_to_event(period_2_event, REFERENCE_TEAM_ID)
    assert normalized_p1["location"] == normalized_p2["location"]


def test_reference_team_normalization_is_also_period_independent() -> None:
    period_1_event = {**REFERENCE_EVENT, "period": 1}
    period_2_event = {**REFERENCE_EVENT, "period": 2}
    assert (
        team_normalization_label(period_1_event, REFERENCE_TEAM_ID)
        == team_normalization_label(period_2_event, REFERENCE_TEAM_ID)
        == TEAM_NORMALIZATION_REFERENCE
    )


# --- 13. Distance/geometry preservation + double-application safety --------
# Added during the 2026-10-13 data-quality completion investigation: the
# mirror (x,y) -> (120-x, 80-y) is a 180-degree rotation about the pitch
# center, i.e. an isometry. It must preserve every pairwise distance between
# freeze_frame points exactly. Verified here with exact synthetic
# coordinates, and independently re-confirmed against 500 real event/frame
# pairs from match 3773477 with zero violations (see
# docs/data_quality_investigation_completion_2026-10-13.md).

def test_team_normalization_preserves_pairwise_distances() -> None:
    import math

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    frame = SAMPLE_FRAME
    normalized = apply_team_normalization_to_frame(frame, MIRRORED_EVENT, REFERENCE_TEAM_ID)
    raw_points = [p["location"] for p in frame["freeze_frame"]]
    final_points = [p["location"] for p in normalized["freeze_frame"]]
    for i in range(len(raw_points)):
        for j in range(i + 1, len(raw_points)):
            assert distance(raw_points[i], raw_points[j]) == pytest.approx(
                distance(final_points[i], final_points[j])
            )


def test_apply_team_normalization_is_not_idempotent_calling_code_must_apply_once() -> None:
    # apply_team_normalization_to_frame is NOT safe to call more than once
    # on the same frame/event pair -- it unconditionally mirrors whenever
    # the event's team is non-reference, every time it is called, so a
    # second call on an already-mirrored MIRRORED-labeled frame mirrors it
    # AGAIN, landing back on the original (uncorrected) coordinates rather
    # than staying put. This is intentional (the function is a pure,
    # stateless transform, not a tracked "already applied?" flag) -- the
    # safety guarantee instead comes from the calling code (verified by
    # reading src/build_statsbomb_360_dataset.py and
    # src/animate_statsbomb_360_match.py: each calls this function exactly
    # once per frame, in a single linear pass, never in a loop that could
    # reprocess the same frame).
    once = apply_team_normalization_to_frame(SAMPLE_FRAME, MIRRORED_EVENT, REFERENCE_TEAM_ID)
    twice = apply_team_normalization_to_frame(once, MIRRORED_EVENT, REFERENCE_TEAM_ID)
    assert once["freeze_frame"][0]["location"] != twice["freeze_frame"][0]["location"]
    # Specifically, applying it twice cancels out and returns to the raw input.
    assert twice["freeze_frame"][0]["location"] == SAMPLE_FRAME["freeze_frame"][0]["location"]


# --- 14. Period-aware physical calibration (halftime end swap) --------------
# Added after the 2026-10-21 second-half investigation
# (docs/full_physical_pitch_orientation_audit.md): a SINGLE,
# period-independent reference team (as --physical-reference-team alone
# originally provided) is only physically correct for periods sharing
# period 1's end arrangement. Real teams swap ends at halftime (and before
# the second half of extra time) -- resolve_physical_reference_team_id
# must swap which team is reference for those periods, using the SAME,
# already-tested mirror mechanism, never a new transform.

def test_period_requires_physical_end_swap_matches_laws_of_the_game() -> None:
    # Periods 1 and 3 share period 1's arrangement (3 is the first half of
    # extra time, no additional swap yet); periods 2 and 4 are played after
    # an end swap. Period 5 (shootout) is deliberately NOT flagged as
    # swapped -- see period_requires_physical_end_swap's docstring for why
    # this is a documented assumption, not a verified fact.
    assert period_requires_physical_end_swap(1) is False
    assert period_requires_physical_end_swap(2) is True
    assert period_requires_physical_end_swap(3) is False
    assert period_requires_physical_end_swap(4) is True
    assert period_requires_physical_end_swap(5) is False
    assert PHYSICALLY_FLIPPED_PERIODS == {2, 4}


def test_resolve_physical_reference_team_id_period_1_matches_plain_override() -> None:
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    assert (
        resolve_physical_reference_team_id(events, "Reference FC", period=1)
        == resolve_reference_team_id(events, physical_reference_team_name="Reference FC")
        == REFERENCE_TEAM_ID
    )


def test_resolve_physical_reference_team_id_swaps_in_period_2() -> None:
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    # Calibrated team attacks one way in period 1 (REFERENCE_TEAM_ID stays
    # reference); in period 2, after the real halftime end swap, the OTHER
    # team must become reference instead.
    assert resolve_physical_reference_team_id(events, "Reference FC", period=1) == REFERENCE_TEAM_ID
    assert resolve_physical_reference_team_id(events, "Reference FC", period=2) == OTHER_TEAM_ID


def test_resolve_physical_reference_team_id_period_3_matches_period_1() -> None:
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    assert (
        resolve_physical_reference_team_id(events, "Reference FC", period=3)
        == resolve_physical_reference_team_id(events, "Reference FC", period=1)
    )


def test_resolve_physical_reference_team_id_period_4_matches_period_2() -> None:
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    assert (
        resolve_physical_reference_team_id(events, "Reference FC", period=4)
        == resolve_physical_reference_team_id(events, "Reference FC", period=2)
    )


def test_resolve_physical_reference_team_id_applies_full_mirror_formula_once_swapped() -> None:
    # End-to-end: the period-2 swapped reference must still use the exact
    # same, already-tested (120-x, 80-y) mirror formula -- not a new,
    # separately-invented transform -- just applied to the other team.
    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    period_2_reference = resolve_physical_reference_team_id(events, "Reference FC", period=2)
    assert period_2_reference == OTHER_TEAM_ID

    normalized = apply_team_normalization_to_event(REFERENCE_EVENT, period_2_reference)
    assert normalized["location"] == pytest.approx([90.0, 60.0])  # (120-30, 80-20)


# --- 15. apply_period_end_swap: baseline-agnostic core -----------------------
# Added 2026-10-22 (docs/full_visualization_orientation_and_clock_audit_2026-10-22.md):
# factored out of resolve_physical_reference_team_id so the SAME swap logic
# can be driven by an UNVERIFIED automatic baseline too (the renderer's
# default "spectator-facing" path), not only an externally-verified team
# name. This must behave identically regardless of how the baseline id was
# obtained.

def test_apply_period_end_swap_matches_resolve_physical_reference_team_id() -> None:
    from src.team_normalization import apply_period_end_swap

    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    for period in (1, 2, 3, 4, 5):
        via_name = resolve_physical_reference_team_id(events, "Reference FC", period)
        via_id = apply_period_end_swap(events, REFERENCE_TEAM_ID, period)
        assert via_name == via_id


def test_apply_period_end_swap_works_with_an_arbitrary_unverified_baseline() -> None:
    # The core function does not care whether REFERENCE_TEAM_ID came from
    # a verified name lookup or an arbitrary automatic pick -- it is
    # baseline-agnostic by design.
    from src.team_normalization import apply_period_end_swap

    events = [REFERENCE_EVENT, MIRRORED_EVENT]
    assert apply_period_end_swap(events, OTHER_TEAM_ID, period=1) == OTHER_TEAM_ID
    assert apply_period_end_swap(events, OTHER_TEAM_ID, period=2) == REFERENCE_TEAM_ID
