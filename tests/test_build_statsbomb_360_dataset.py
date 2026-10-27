"""Tests for the next-event CSV dataset builder (Phase 5/9 audit).

Covers: event filtering (the implemented "meaningful event" definition),
next-event target construction (predicting the next MEANINGFUL_EVENT_TYPES
event in the complete raw event stream, not restricted to 360-covered
events), period-boundary non-leakage, max-time-difference filtering, CSV
round-tripping, and -- the regression target -- correct teammate/opponent
counts for events whose raw 360 snapshot is shared with a different team's
event (Problem C integration; see src/team_identity.py).
"""

import copy
import csv

import pytest

import src.build_statsbomb_360_dataset as build_statsbomb_360_dataset_module
from src.build_statsbomb_360_dataset import (
    MEANINGFUL_EVENT_TYPES,
    _match_team_names,
    build_match_dataset,
    build_rows_from_match_data,
    find_next_meaningful_event,
    resolve_actor_location,
    save_dataset,
    timestamp_to_seconds,
)
from src.match_exclusions import ExcludedMatch, ExcludedMatchError

HOME = {"id": 10, "name": "Home FC"}
AWAY = {"id": 20, "name": "Away FC"}


def _event(
    event_id,
    index,
    event_type,
    team,
    timestamp,
    period=1,
    minute=0,
    player="Player",
    possession=1,
):
    return {
        "id": event_id,
        "index": index,
        "period": period,
        "timestamp": timestamp,
        "minute": minute,
        "type": {"name": event_type},
        "team": dict(team),
        "player": {"name": player},
        "possession": possession,
    }


def _frame(event_uuid, actor_location=(50.0, 40.0), teammate_only_actor=True):
    freeze_frame = [
        {"teammate": True, "actor": True, "keeper": False, "location": list(actor_location)},
    ]
    if teammate_only_actor:
        freeze_frame.append(
            {"teammate": False, "actor": False, "keeper": False, "location": [70.0, 40.0]}
        )
    return {"event_uuid": event_uuid, "visible_area": [], "freeze_frame": freeze_frame}


def _away_filler(index=900):
    """A single 360-covered Away event, appended to fixtures that would
    otherwise only contain Home events.

    `resolve_reference_team_id` (reused from src/team_normalization.py) is
    resolved from the 360-COVERED event subset, not the raw event list as a
    whole (this matches src/animate_statsbomb_360_match.py's own behavior,
    so both consumers agree on one reference team) -- so a real match
    always needs at least one 360-covered event per team for this to
    resolve, which every actual StatsBomb 360 match satisfies but a minimal
    synthetic fixture must provide explicitly.
    """
    event = _event("away-filler", index, "Pressure", AWAY, "00:59:59.000", player="Filler")
    return event, _frame("away-filler", actor_location=(10.0, 10.0))


# --- timestamp parsing -------------------------------------------------------

def test_timestamp_to_seconds_parses_hours_minutes_seconds_microseconds() -> None:
    assert timestamp_to_seconds("00:01:02.500000") == pytest.approx(62.5)
    assert timestamp_to_seconds("00:45:12.123456") == pytest.approx(2712.123456)


# --- team name resolution ----------------------------------------------------

def test_match_team_names_returns_first_seen_order() -> None:
    events = [
        _event("e1", 1, "Pass", AWAY, "00:00:01.000"),
        _event("e2", 2, "Pass", HOME, "00:00:02.000"),
    ]
    assert _match_team_names(events) == ("Away FC", "Home FC")


def test_match_team_names_raises_when_fewer_than_two_teams() -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000")]
    with pytest.raises(ValueError):
        _match_team_names(events)


# --- find_next_meaningful_event: filtering + period boundary ----------------

def test_find_next_meaningful_event_skips_non_meaningful_types() -> None:
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Pressure", AWAY, "00:00:02.000"),  # not meaningful
        _event("e3", 3, "Duel", AWAY, "00:00:02.500"),  # not meaningful
        _event("e4", 4, "Carry", HOME, "00:00:03.000"),  # meaningful
    ]
    result = find_next_meaningful_event(events, current_position=0, current_period=1)
    assert result["id"] == "e4"


def test_find_next_meaningful_event_stops_at_period_boundary() -> None:
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000", period=1),
        _event("e2", 2, "Half End", HOME, "00:48:00.000", period=1),
        _event("e3", 3, "Carry", HOME, "00:00:01.000", period=2),  # next period
    ]
    result = find_next_meaningful_event(events, current_position=0, current_period=1)
    assert result is None  # must not leak across the half-time boundary


def test_find_next_meaningful_event_returns_none_when_none_follow() -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000")]
    assert find_next_meaningful_event(events, current_position=0, current_period=1) is None


# --- MEANINGFUL_EVENT_TYPES is the documented, implemented definition -------

def test_meaningful_event_types_definition_is_exactly_as_documented() -> None:
    # Pins down the implemented "meaningful event" definition so a future
    # change to this set is a deliberate, reviewed decision, not a silent
    # drift. Pressure/Duel/Ball Receipt*/Interception/Block/Foul Won/etc.
    # are intentionally excluded under the current definition.
    assert MEANINGFUL_EVENT_TYPES == {
        "Pass", "Carry", "Dribble", "Shot", "Miscontrol",
        "Ball Recovery", "Clearance", "Dispossessed",
    }


# --- sequence construction: current event requires 360, next does not ------

def test_current_event_without_360_frame_is_never_a_row() -> None:
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),  # no frame below
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        filler_event,
    ]
    frames = {"e2": _frame("e2"), "away-filler": filler_frame}  # e1 has no 360 coverage
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    assert all(row["current_event_id"] != "e1" for row in rows)


def test_next_event_does_not_require_360_frame() -> None:
    # Documents the implemented prediction-task framing: "next event of a
    # selected type in the complete raw event stream," not "next 360-covered
    # event." next_event_* columns are read straight from the raw event
    # object and never require frames_by_event_id coverage.
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", AWAY, "00:00:02.000"),  # no frame below
        filler_event,
    ]
    frames = {"e1": _frame("e1"), "away-filler": filler_frame}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    assert len(rows) == 1
    assert rows[0]["next_event_id"] == "e2"
    assert rows[0]["next_event_type"] == "Carry"


# --- time-difference filtering ----------------------------------------------

def test_row_dropped_when_gap_exceeds_maximum_time_difference() -> None:
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:20.000"),  # 19s later
        filler_event,
    ]
    frames = {"e1": _frame("e1"), "e2": _frame("e2"), "away-filler": filler_frame}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    assert rows == []


def test_row_kept_when_gap_within_maximum_time_difference() -> None:
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:05.000"),  # 4s later
        filler_event,
    ]
    frames = {"e1": _frame("e1"), "e2": _frame("e2"), "away-filler": filler_frame}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    assert len(rows) == 1
    assert rows[0]["time_to_next_event_seconds"] == pytest.approx(4.0)


# --- same_team / same_possession -------------------------------------------

def test_same_team_and_same_possession_flags() -> None:
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000", possession=7),
        _event("e2", 2, "Carry", HOME, "00:00:02.000", possession=7),
        _event("e3", 3, "Ball Recovery", AWAY, "00:00:03.000", possession=8),
    ]
    frames = {"e1": _frame("e1"), "e2": _frame("e2"), "e3": _frame("e3")}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row_for_e1 = next(r for r in rows if r["current_event_id"] == "e1")
    assert row_for_e1["same_team"] is True
    assert row_for_e1["same_possession"] is True

    row_for_e2 = next(r for r in rows if r["current_event_id"] == "e2")
    assert row_for_e2["same_team"] is False
    assert row_for_e2["same_possession"] is False


# --- no next-event SPATIAL data ever enters a row (leakage boundary check) --

def test_row_never_contains_next_event_spatial_fields() -> None:
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        filler_event,
    ]
    frames = {"e1": _frame("e1"), "e2": _frame("e2"), "away-filler": filler_frame}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row = rows[0]
    spatial_keywords = ("next_actor", "next_x", "next_y", "next_location", "next_freeze_frame")
    for key in row:
        assert not any(kw in key for kw in spatial_keywords), (
            f"column {key!r} would leak the next event's spatial/360 state into the input row"
        )


# --- deterministic output ----------------------------------------------------

def test_build_rows_is_deterministic_for_identical_input() -> None:
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        _event("e3", 3, "Ball Recovery", AWAY, "00:00:03.000"),
    ]
    frames = {"e1": _frame("e1"), "e2": _frame("e2"), "e3": _frame("e3")}
    rows_a = build_rows_from_match_data("m1", copy.deepcopy(events), copy.deepcopy(frames), 10.0)
    rows_b = build_rows_from_match_data("m1", copy.deepcopy(events), copy.deepcopy(frames), 10.0)
    assert rows_a == rows_b


# --- actor location extraction -----------------------------------------------

def test_actor_location_extracted_from_freeze_frame_actor_point() -> None:
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        filler_event,
    ]
    events[0]["location"] = [33.0, 21.0]  # lets the snapshot's orientation be established
    frames = {
        "e1": _frame("e1", actor_location=(33.0, 21.0)),
        "e2": _frame("e2"),
        "away-filler": filler_frame,
    }
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row = next(r for r in rows if r["current_event_id"] == "e1")
    assert row["orientation_available"] is True
    assert row["actor_x"] == pytest.approx(33.0)
    assert row["actor_y"] == pytest.approx(21.0)


def test_actor_location_unavailable_when_orientation_cannot_be_established_causally() -> None:
    """No event location -> the snapshot's frame is unknown at prediction
    time: the feature columns report the actor position as unavailable
    rather than guessing; the offline diagnostics keep the raw reading."""
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        filler_event,
    ]
    frames = {"e1": _frame("e1", actor_location=(33.0, 21.0)), "e2": _frame("e2"), "away-filler": filler_frame}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row = next(r for r in rows if r["current_event_id"] == "e1")
    assert row["orientation"] == "UNCERTAIN" and row["orientation_available"] is False
    assert row["orientation_resolved_by"] == "unresolved"
    assert row["actor_x"] is None and row["actor_y"] is None
    assert row["actor_x_offline"] == pytest.approx(33.0)


def test_actor_location_is_none_when_no_actor_point_present() -> None:
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        filler_event,
    ]
    no_actor_frame = {
        "event_uuid": "e1",
        "visible_area": [],
        "freeze_frame": [{"teammate": True, "actor": False, "keeper": False, "location": [1.0, 1.0]}],
    }
    frames = {"e1": no_actor_frame, "e2": _frame("e2"), "away-filler": filler_frame}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row = next(r for r in rows if r["current_event_id"] == "e1")
    assert row["actor_x"] is None
    assert row["actor_y"] is None


# --- teammate/opponent counts: non-shared baseline ---------------------------

def test_teammate_opponent_counts_for_non_shared_snapshot() -> None:
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        filler_event,
    ]
    frame = {
        "event_uuid": "e1",
        "visible_area": [],
        "freeze_frame": [
            {"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]},
            {"teammate": True, "actor": False, "keeper": False, "location": [45.0, 40.0]},
            {"teammate": False, "actor": False, "keeper": False, "location": [70.0, 40.0]},
            {"teammate": False, "actor": False, "keeper": True, "location": [5.0, 40.0]},
        ],
    }
    frames = {"e1": frame, "e2": _frame("e2"), "away-filler": filler_frame}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row = next(r for r in rows if r["current_event_id"] == "e1")
    assert row["visible_teammates"] == 2
    assert row["visible_opponents"] == 2
    assert row["visible_ambiguous"] == 0


# --- REGRESSION: shared cross-team snapshot must use the RESOLVED team -----
# Models the real bug found during the audit: match 3857283, Belgium's
# Ball Receipt* borrowed Morocco's Ball Recovery's raw 360 array byte-for-
# byte. Reading `teammate` relative to the borrowing event's own team
# silently inverted its visible_teammates/visible_opponents split. The
# renderer (src/animate_statsbomb_360_match.py) already fixed this via
# Problem C; before this change, build_statsbomb_360_dataset.py did not.

SHARED_RAW_ARRAY = [
    # Physically the AWAY team's own actor (the recovering player). Kept
    # off pitch-center on purpose: the classifier's "mirror" vote is
    # degenerate (zero-confidence) exactly at the pitch center (60, 40),
    # since that point mirrors onto itself.
    {"teammate": True, "actor": True, "keeper": False, "location": [90.0, 50.0]},
    # Physically the AWAY team's goalkeeper.
    {"teammate": True, "actor": False, "keeper": True, "location": [15.0, 40.0]},
    # Physically a HOME player (the only genuine opponent from Away's own
    # point of view).
    {"teammate": False, "actor": False, "keeper": False, "location": [30.0, 20.0]},
]


def _shared_snapshot_fixture():
    away_source = _event("away-src", 1, "Ball Recovery", AWAY, "00:00:05.000", player="A1")
    away_source["location"] = [90.0, 50.0]  # matches its own array's actor -> NORMAL, confident

    home_borrower = _event("home-borrow", 2, "Carry", HOME, "00:00:05.000", player="H1")
    home_borrower["location"] = [1.0, 1.0]  # far from the array either way -> UNCERTAIN, not a second source

    trailing = _event("trail", 3, "Pass", HOME, "00:00:06.000", player="H1")

    events = [away_source, home_borrower, trailing]
    frames = {
        "away-src": {
            "event_uuid": "away-src", "visible_area": [],
            "freeze_frame": copy.deepcopy(SHARED_RAW_ARRAY),
        },
        "home-borrow": {
            "event_uuid": "home-borrow", "visible_area": [],
            "freeze_frame": copy.deepcopy(SHARED_RAW_ARRAY),
        },
        "trail": _frame("trail"),
    }
    return events, frames


def test_shared_snapshot_borrower_teammate_opponent_counts_use_resolved_source_team() -> None:
    events, frames = _shared_snapshot_fixture()
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)

    borrower_row = next(r for r in rows if r["current_event_id"] == "home-borrow")
    # Correct (post-fix) answer: 2 of the 3 points are physically Away
    # players (the array's true owner), so from Home's own perspective
    # there is exactly 1 real teammate and 2 real opponents.
    assert borrower_row["visible_teammates"] == 1
    assert borrower_row["visible_opponents"] == 2
    assert borrower_row["visible_ambiguous"] == 0
    # The naive, un-fixed reading (teammate flag taken at face value
    # relative to Home) would have produced the inverted, wrong split:
    assert (borrower_row["visible_teammates"], borrower_row["visible_opponents"]) != (2, 1)


def test_shared_snapshot_source_event_unaffected_by_fix() -> None:
    events, frames = _shared_snapshot_fixture()
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)

    source_row = next(r for r in rows if r["current_event_id"] == "away-src")
    # The source event's own reading was already correct before the fix
    # (not shared/borrowed in a misleading way): 1 genuine teammate
    # (itself aside), 1 goalkeeper teammate, 1 opponent.
    assert source_row["visible_teammates"] == 2
    assert source_row["visible_opponents"] == 1
    assert source_row["visible_ambiguous"] == 0


def test_genuinely_ambiguous_shared_snapshot_is_counted_not_guessed() -> None:
    # Two different-team events BOTH self-consistent with the same shared
    # array -- genuine ambiguity (see src/team_identity.py). Must not be
    # silently assigned to either team's teammate/opponent count.
    event_x = _event("ev-x", 1, "Ball Recovery", HOME, "00:00:05.000")
    event_x["location"] = [90.0, 50.0]
    event_y = _event("ev-y", 2, "Ball Recovery", AWAY, "00:00:05.000")
    event_y["location"] = [90.0, 50.0]  # also matches -> also NORMAL, confident
    trailing = _event("trail", 3, "Pass", HOME, "00:00:06.000")

    events = [event_x, event_y, trailing]
    frames = {
        "ev-x": {"event_uuid": "ev-x", "visible_area": [], "freeze_frame": copy.deepcopy(SHARED_RAW_ARRAY)},
        "ev-y": {"event_uuid": "ev-y", "visible_area": [], "freeze_frame": copy.deepcopy(SHARED_RAW_ARRAY)},
        "trail": _frame("trail"),
    }
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row_x = next(r for r in rows if r["current_event_id"] == "ev-x")
    row_y = next(r for r in rows if r["current_event_id"] == "ev-y")
    # Offline (both directions in time): both records are ambiguous.
    assert row_x["visible_ambiguous_offline"] == 3
    assert row_x["visible_teammates_offline"] == row_x["visible_opponents_offline"] == 0
    # Causal feature columns: ev-x precedes ev-y and cannot know about the
    # conflicting copy yet; ev-y sees both and is ambiguous, not guessed.
    assert row_x["visible_ambiguous"] == 0
    assert row_y["visible_ambiguous"] == 3
    assert row_y["visible_teammates"] == row_y["visible_opponents"] == 0
    assert row_y["identity_quality"] == "uncertain"


# --- CSV round-trip -----------------------------------------------------------

def test_csv_round_trip_preserves_row_values(tmp_path) -> None:
    events, frames = _shared_snapshot_fixture()
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    output_path = tmp_path / "out.csv"
    save_dataset(rows, output_path)

    with output_path.open(newline="", encoding="utf-8") as f:
        reloaded = list(csv.DictReader(f))

    assert len(reloaded) == len(rows)
    for original, round_tripped in zip(rows, reloaded):
        assert round_tripped["current_event_id"] == original["current_event_id"]
        assert round_tripped["current_event_type"] == original["current_event_type"]
        assert int(round_tripped["visible_teammates"]) == original["visible_teammates"]
        assert int(round_tripped["visible_opponents"]) == original["visible_opponents"]
        assert round_tripped["same_team"] == str(original["same_team"])
        assert float(round_tripped["time_to_next_event_seconds"]) == pytest.approx(
            original["time_to_next_event_seconds"]
        )


def test_save_dataset_raises_on_empty_rows(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_dataset([], tmp_path / "empty.csv")


# --- resolve_actor_location: multiple/zero actor-flagged points -------------
# Modeled on a dedicated dataset-wide audit (2026-10-13): 28/1,381,467
# frames (0.002%, 11/426 matches) carry TWO freeze_frame points flagged
# actor=True. In 26/28 of those cases the two points are within ~0.22 pitch
# units of each other (floating-point/encoding noise describing the same
# physical position); in exactly 2 cases (one match) they are ~6.8 units
# apart -- genuinely ambiguous. Separately, 2/1,381,467 frames (both with a
# real, located event) have ZERO actor-flagged points -- legitimately
# missing actor information, not an error to paper over.

def test_resolve_actor_location_with_zero_actor_points_is_none_not_ambiguous() -> None:
    freeze_frame = [{"teammate": True, "actor": False, "keeper": False, "location": [1.0, 1.0]}]
    x, y, ambiguous = resolve_actor_location(freeze_frame)
    assert (x, y) == (None, None)
    assert ambiguous is False  # missing is a different condition from ambiguous


def test_resolve_actor_location_real_zero_actor_case_all_opponents_visible() -> None:
    # Modeled on match 3893793, event e43bc7e0-4353-4193-a104-29897bf67703 (a
    # 50/50) and match 4018356, event dfe42790-bef1-4e78-9632-de2e747af5cc
    # (a Dispossessed): confirmed via a dedicated audit that BOTH real
    # zero-actor frames in the full dataset have teammate=False on every
    # single visible point -- the acting player's own team (including the
    # actor) was entirely outside the 360 camera's tracked field of view at
    # that instant. Per the schema invariant actor=True => teammate=True,
    # zero teammates necessarily means zero actors; this is not a flagging
    # anomaly or an event/frame mismatch. A representative subset of the
    # real match 3893793 freeze_frame (all opponents, no actor):
    freeze_frame = [
        {"teammate": False, "actor": False, "keeper": False, "location": [69.04302884292304, 61.76283668021257]},
        {"teammate": False, "actor": False, "keeper": False, "location": [73.31509313156657, 50.31341006463371]},
        {"teammate": False, "actor": False, "keeper": False, "location": [87.5999984741211, 35.900001525878906]},
        {"teammate": False, "actor": False, "keeper": False, "location": [96.68177648364345, 53.51366349376625]},
    ]
    assert all(p["teammate"] is False for p in freeze_frame)
    x, y, ambiguous = resolve_actor_location(freeze_frame)
    assert (x, y) == (None, None)
    assert ambiguous is False


def test_resolve_actor_location_with_one_actor_point() -> None:
    freeze_frame = [{"teammate": True, "actor": True, "keeper": False, "location": [33.0, 21.0]}]
    x, y, ambiguous = resolve_actor_location(freeze_frame)
    assert (x, y) == (33.0, 21.0)
    assert ambiguous is False


def test_resolve_actor_location_with_near_duplicate_actor_points_uses_the_position() -> None:
    # Modeled on match 3773477 event 68022eee: two actor=True points at
    # byte-for-byte identical coordinates within the same raw freeze_frame.
    freeze_frame = [
        {"teammate": True, "actor": True, "keeper": False, "location": [41.099998474121094, 7.800000190734863]},
        {"teammate": True, "actor": True, "keeper": False, "location": [41.099998474121094, 7.800000190734863]},
    ]
    x, y, ambiguous = resolve_actor_location(freeze_frame)
    assert x == pytest.approx(41.099998474121094)
    assert y == pytest.approx(7.800000190734863)
    assert ambiguous is False


def test_resolve_actor_location_with_near_but_not_identical_duplicates_uses_the_position() -> None:
    # Modeled on match 3773672: two actor points ~0.22 units apart --
    # within tolerance, treated as the same physical position.
    freeze_frame = [
        {"teammate": True, "actor": True, "keeper": False, "location": [74.57040893925675, 16.818369191389067]},
        {"teammate": True, "actor": True, "keeper": False, "location": [74.5999984741211, 16.600000381469727]},
    ]
    x, y, ambiguous = resolve_actor_location(freeze_frame)
    assert (x, y) != (None, None)
    assert ambiguous is False


def test_resolve_actor_location_with_genuinely_different_actor_points_is_ambiguous() -> None:
    # Modeled on match 3802837 events 0a731280/17380117: two actor points
    # ~6.8 units apart -- genuinely ambiguous, must not guess either one.
    freeze_frame = [
        {"teammate": True, "actor": True, "keeper": False, "location": [30.144737503288656, 10.794747684926325]},
        {"teammate": True, "actor": True, "keeper": False, "location": [36.0, 7.400000095367432]},
    ]
    x, y, ambiguous = resolve_actor_location(freeze_frame)
    assert (x, y) == (None, None)
    assert ambiguous is True


def test_row_marks_actor_ambiguous_without_inventing_a_position() -> None:
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        filler_event,
    ]
    ambiguous_frame = {
        "event_uuid": "e1",
        "visible_area": [],
        "freeze_frame": [
            {"teammate": True, "actor": True, "keeper": False, "location": [10.0, 10.0]},
            {"teammate": True, "actor": True, "keeper": False, "location": [100.0, 70.0]},
        ],
    }
    frames = {"e1": ambiguous_frame, "e2": _frame("e2"), "away-filler": filler_frame}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row = next(r for r in rows if r["current_event_id"] == "e1")
    assert row["actor_x"] is None
    assert row["actor_y"] is None
    assert row["actor_ambiguous"] is True


def test_row_actor_not_ambiguous_when_no_actor_point_present() -> None:
    # Regression guard: a row with NO actor point must read actor_ambiguous
    # == False (a distinct condition from "multiple disagreeing points").
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        filler_event,
    ]
    no_actor_frame = {
        "event_uuid": "e1",
        "visible_area": [],
        "freeze_frame": [{"teammate": True, "actor": False, "keeper": False, "location": [1.0, 1.0]}],
    }
    frames = {"e1": no_actor_frame, "e2": _frame("e2"), "away-filler": filler_frame}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row = next(r for r in rows if r["current_event_id"] == "e1")
    assert row["actor_x"] is None
    assert row["actor_y"] is None
    assert row["actor_ambiguous"] is False


def test_duplicate_actor_point_inflates_visible_player_count_by_one() -> None:
    # Confirmed, documented, LOW-severity finding from the 2026-10-13
    # data-quality completion investigation: when the raw freeze_frame
    # genuinely contains two points for what is physically the same player
    # (see match 3773477 events 284b4d1b/aa92e1df, both teammate=True,
    # near-identical locations), resolve_actor_location correctly resolves
    # ONE actor position (see test_resolve_actor_location_with_near_duplicate_actor_points_uses_the_position),
    # but visible_players/visible_teammates still count BOTH raw entries --
    # they are not deduplicated as "the same player," because nothing in
    # the schema identifies non-actor points well enough to safely merge
    # them without risking merging two DIFFERENT nearby players by mistake.
    # This pins the current, intentional behavior: duplicate points are
    # rare (28/1,381,467 frames dataset-wide, 0.002%) and inflate the count
    # by at most 1, a negligible effect on spatial features, but future
    # modeling work should be aware visible_players is not a verified exact
    # physical headcount in every row.
    filler_event, filler_frame = _away_filler()
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        filler_event,
    ]
    duplicate_frame = {
        "event_uuid": "e1",
        "visible_area": [],
        "freeze_frame": [
            {"teammate": True, "actor": True, "keeper": False, "location": [41.1, 7.8]},
            {"teammate": True, "actor": True, "keeper": False, "location": [41.1, 7.8]},  # same physical player, listed twice
            {"teammate": False, "actor": False, "keeper": False, "location": [80.0, 40.0]},
        ],
    }
    frames = {"e1": duplicate_frame, "e2": _frame("e2"), "away-filler": filler_frame}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row = next(r for r in rows if r["current_event_id"] == "e1")
    assert row["visible_players"] == 3  # raw point count, duplicate included
    assert row["visible_teammates"] == 2  # both duplicate entries counted -- known, documented limitation
    assert row["visible_opponents"] == 1
    assert row["actor_ambiguous"] is False  # the actor POSITION itself is still correctly resolved


# --- exclusion manifest integration (src/match_exclusions.py) ---------------

def test_build_match_dataset_raises_excluded_match_error_before_any_file_io(monkeypatch) -> None:
    # The match id below does not exist on disk at all -- if the exclusion
    # check runs BEFORE file I/O (as required), we get ExcludedMatchError,
    # never FileNotFoundError.
    fake_excluded = ExcludedMatch(
        match_id=424242, reason="test_reason", file="nonexistent.json",
        evidence="synthetic test case", date_investigated="2026-10-13", report="docs/x.md",
    )
    monkeypatch.setattr(
        build_statsbomb_360_dataset_module, "load_exclusion_manifest", lambda: {424242: fake_excluded}
    )
    with pytest.raises(ExcludedMatchError, match="test_reason"):
        build_match_dataset(match_id=424242, maximum_time_difference=10.0)


def test_build_match_dataset_unaffected_when_manifest_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(build_statsbomb_360_dataset_module, "load_exclusion_manifest", lambda: {})
    # A real, valid match_id would proceed past the exclusion check and hit
    # the normal FileNotFoundError path for a made-up id -- proving the
    # exclusion check itself is not swallowing unrelated errors.
    with pytest.raises(FileNotFoundError):
        build_match_dataset(match_id=0, maximum_time_difference=10.0)


def test_reference_team_is_the_first_360_event_in_event_order_not_timestamp_order() -> None:
    """Regression (2026-10-27): 9 matches carry a late-indexed period-1 event
    with an early timestamp; the reference team must come from the first 360
    event in the stream (the kick-off team), never from a later event."""
    away_late = _event("late", 50, "Ball Receipt*", AWAY, "00:00:00.100")  # index 50, timestamp first
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000"),
        _event("e2", 2, "Carry", HOME, "00:00:02.000"),
        away_late,
    ]
    frames = {"e1": _frame("e1"), "e2": _frame("e2"), "late": _frame("late")}
    rows = build_rows_from_match_data("m1", events, frames, maximum_time_difference=10.0)
    row = next(r for r in rows if r["current_event_id"] == "e1")
    assert row["team_normalization"] == "REFERENCE"  # Home kicked off: it is the reference team
