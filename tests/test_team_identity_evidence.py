"""Tests for the evidence-based team identity (src/team_identity.py,
resolve_match_team_identity) and the orientation synchronization between
renderer and dataset builder (src/orientation_resolution.py).

Synthetic scenes use one fixed geometry: Team A attacks +x and defends x = 0
in its own frame; players stand >= 20 yd apart, so every pairing between
consecutive snapshots is unambiguous unless a test makes it crowded on
purpose. Real-data regressions are skipped when the raw files are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.orientation_correction import OrientationResult
from src.team_identity import (
    DECISION_CONTINUITY,
    DECISION_UNDECIDED,
    IDENTITY_CORRECTED_GOALKEEPER,
    IDENTITY_CORRECTED_VIEW,
    IDENTITY_SPEC,
    IDENTITY_UNRESOLVED,
    REASON_COPY_GROUP_UNDECIDED,
    REASON_KEEPER_GOAL_SIDE,
    REASON_POINT_CONTRADICTED,
    REASON_SIMULTANEOUS_CONTRADICTION,
    TEAM_IDENTITY_AMBIGUOUS,
    exact_structural_relation,
    resolve_match_team_identity,
    unambiguous_pairs,
)

A, B = "Team A", "Team B"
TEAMS = {A: {"id": 1, "name": A}, B: {"id": 2, "name": B}}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"

# (x, y, teammate-from-Team-A's-view, keeper) in Team A's frame.
SCENE = [
    (5.0, 40.0, True, True),       # 0: Team A goalkeeper at A's goal
    (30.0, 10.0, True, False),
    (30.0, 35.0, True, False),     # 2: Team A actor
    (30.0, 60.0, True, False),
    (55.0, 20.0, False, False),
    (55.0, 45.0, False, False),
    (55.0, 70.0, False, False),
    (115.0, 40.0, False, True),    # 7: Team B goalkeeper at B's goal
]


def _clock(ms: int) -> str:
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    return f"00:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _record(event_id, team, ms, scene=SCENE, period=1, index=None, flip_flags=(), mirror=False, shift=0.0, actor=2,
            moved=True):
    """Event + raw 360 record. `scene` is in Team A's frame; Team B records
    are stored in Team B's frame (mirrored) unless `mirror` says otherwise
    (a copy stored in the other team's frame). With `moved`, every player
    has moved independently by up to ~1 yd (seeded by the timestamp), as in
    a genuinely separate snapshot; snapshots at the same timestamp coincide."""
    in_b_frame = (team == B) != mirror
    jitter = np.random.default_rng(ms).uniform(-0.7, 0.7, (len(scene), 2)) if moved else np.zeros((len(scene), 2))
    freeze = []
    for i, (x, y, mate, keeper) in enumerate(scene):
        x, y = x + jitter[i][0], y + jitter[i][1]
        if in_b_frame:
            x, y = 120 - x, 80 - y
        teammate = mate if team == A else not mate
        if i in flip_flags:
            teammate = not teammate
        freeze.append({"teammate": teammate, "actor": i == actor, "keeper": keeper,
                       "location": [round(x + shift, 3), round(y, 3)]})
    event = {"id": event_id, "index": index if index is not None else ms, "period": period,
             "timestamp": _clock(ms), "team": dict(TEAMS[team]), "type": {"name": "Pass"}}
    return event, {"event_uuid": event_id, "freeze_frame": freeze, "visible_area": []}


def _resolve(records, orientations=None, owners=None, all_events=()):
    orientations = orientations or ["NORMAL"] * len(records)
    owners = owners or orientations
    problem_a = [OrientationResult(o, 0.0 if o == "UNCERTAIN" else 1.0, 0.0) for o in owners]
    events = [event for event, _ in records] + list(all_events)
    return resolve_match_team_identity(records, orientations, problem_a, (A, B), events)


def _swap_flags(scene):
    return [(x, y, not mate, keeper) for x, y, mate, keeper in scene]


# --- crowding-safe pairing ---------------------------------------------------

def test_unambiguous_pairs_skip_crowded_players_and_tolerate_a_constant_offset():
    a = np.array([[10.0, 10.0], [60.0, 40.0], [60.6, 40.0], [100.0, 70.0]])
    b = a + np.array([2.5, 0.0])  # whole snapshot offset by 2.5 yd
    pairs = unambiguous_pairs(a, b)
    assert (0, 0) in pairs and (3, 3) in pairs
    assert all(i not in (1, 2) for i, _ in pairs)  # 0.6 yd apart: pairing is arbitrary


def test_crowded_label_swap_is_not_flagged():
    crowded = SCENE + [(80.0, 40.0, True, False), (80.6, 40.0, False, False)]
    swapped = SCENE + [(80.0, 40.0, False, False), (80.6, 40.0, True, False)]
    records = [_record("r0", A, 0, crowded), _record("r1", A, 1000, swapped), _record("r2", A, 2000, crowded)]
    identity = _resolve(records)
    assert set(identity["r1"].statuses) == {IDENTITY_SPEC}


# --- structural detection ------------------------------------------------------

def test_exact_relation_detects_identical_mirrored_and_shifted_copies():
    a = np.array([[x, y] for x, y, _, _ in SCENE])
    assert exact_structural_relation(a, a.copy())[0] is False
    mirrored = np.column_stack([120 - a[:, 0], 80 - a[:, 1]]) + np.array([2.53, 0.0])
    assert exact_structural_relation(a, mirrored)[0] is True
    moved = a + np.random.default_rng(0).normal(0, 1.0, a.shape)  # a genuinely later snapshot
    assert exact_structural_relation(a, moved) is None


def _neighbours():
    return [_record("before", A, 4000), _record("after", A, 6000)]


def test_same_timestamp_copy_with_different_actor_is_read_in_the_evidenced_view():
    """case_01/case_05 pattern: Team B's record is a verbatim copy of Team A's
    snapshot (A's coordinates AND A's flags), only `actor` differs. Not
    byte-identical, so the legacy detector missed it."""
    no_keepers = [(x, y, mate, False) for x, y, mate, _ in SCENE]  # continuity is the only evidence
    source = _record("a", A, 5000, scene=no_keepers)
    copy_event, copy_frame = _record("b", A, 5000, scene=no_keepers, actor=5)
    copy_event = {**copy_event, "id": "b", "team": dict(TEAMS[B]), "index": 5001}
    records = [_neighbours()[0], source, (copy_event, copy_frame), _neighbours()[1]]
    identity = _resolve(records, orientations=["NORMAL", "NORMAL", "ROTATED_180", "NORMAL"])
    copy = identity["b"]
    assert copy.decision == DECISION_CONTINUITY
    assert set(copy.statuses) == {IDENTITY_CORRECTED_VIEW}
    assert copy.teams == identity["a"].teams
    assert copy.spec_teams != copy.teams  # raw spec reading preserved alongside
    assert copy.legacy_teams == copy.spec_teams  # the legacy pipeline missed the copy
    assert set(identity["a"].statuses) == {IDENTITY_SPEC}


def test_derived_record_mirrored_with_offset_and_unconverted_flags_is_corrected():
    """case_02 pattern: Team B's record re-expresses A's snapshot in B's frame
    (mirror + 2.53 yd offset) 82 ms later, but keeps A's flags."""
    derived_event, derived_frame = _record("b", B, 5000, flip_flags=range(len(SCENE)), shift=2.53)
    derived_event = {**derived_event, "timestamp": _clock(5082)}  # same scene, stamped 82 ms later
    records = [_neighbours()[0], _record("a", A, 5000), (derived_event, derived_frame), _neighbours()[1]]
    identity = _resolve(records)
    assert identity["b"].decision == DECISION_CONTINUITY
    assert set(identity["b"].statuses) == {IDENTITY_CORRECTED_VIEW}
    assert identity["b"].teams == identity["a"].teams


def test_spec_consistent_opposing_views_are_left_alone():
    records = [_record("a", A, 5000), _record("b", B, 5000)]  # mirrored + inverted: both correct
    identity = _resolve(records)
    for record in ("a", "b"):
        assert set(identity[record].statuses) == {IDENTITY_SPEC}
    assert identity["a"].teams == identity["b"].teams


def test_copy_group_without_decisive_evidence_is_unresolved_not_guessed():
    no_keepers = [(x, y, mate, False) for x, y, mate, _ in SCENE]
    copy_event, copy_frame = _record("b", A, 5000, scene=no_keepers, actor=5)
    copy_event = {**copy_event, "team": dict(TEAMS[B]), "index": 5001}
    records = [_record("a", A, 5000, scene=no_keepers), (copy_event, copy_frame)]
    identity = _resolve(records, owners=["NORMAL", "NORMAL"])  # both claim the coordinates: owner conflict
    for record in ("a", "b"):
        assert identity[record].decision == DECISION_UNDECIDED
        assert set(identity[record].teams) == {TEAM_IDENTITY_AMBIGUOUS}
        assert set(identity[record].reasons) == {REASON_COPY_GROUP_UNDECIDED}


def test_single_contradicted_player_is_unclear_without_touching_the_rest():
    records = [_record("r0", A, 0), _record("r1", A, 1000, flip_flags=(4,)), _record("r2", A, 2000)]
    identity = _resolve(records)["r1"]
    assert identity.statuses[4] == IDENTITY_UNRESOLVED and identity.reasons[4] == REASON_POINT_CONTRADICTED
    assert all(s == IDENTITY_SPEC for i, s in enumerate(identity.statuses) if i != 4)


# --- goalkeeper-only continuity rule --------------------------------------------

def _keeper_run(flip_at=2, period_of=lambda k: 1, extra_events=()):
    records = [
        _record(f"r{k}", A, 1000 * k, period=period_of(k), index=k, flip_flags=(0,) if k == flip_at else ())
        for k in range(4)
    ]
    return records, _resolve(records, all_events=extra_events)


def test_goalkeeper_only_flag_error_takes_the_verified_previous_identity():
    _, identity = _keeper_run()
    keeper = identity["r2"]
    assert keeper.spec_teams[0] == B  # raw flag says opponent
    assert keeper.teams[0] == A and keeper.statuses[0] == IDENTITY_CORRECTED_GOALKEEPER
    assert "previous verified keeper r1" in keeper.notes[0]
    assert all(s == IDENTITY_SPEC for s in keeper.statuses[1:])  # nothing else is propagated


def test_goalkeeper_rule_blocked_by_a_substitution():
    substitution = {"id": "sub", "index": 1500, "period": 1, "timestamp": _clock(1500),
                    "team": dict(TEAMS[A]), "type": {"name": "Substitution"}}
    _, identity = _keeper_run(extra_events=[substitution])
    assert identity["r2"].statuses[0] == IDENTITY_UNRESOLVED
    assert identity["r2"].reasons[0] == REASON_KEEPER_GOAL_SIDE


def test_goalkeeper_rule_never_crosses_halftime():
    _, identity = _keeper_run(period_of=lambda k: 1 if k < 2 else 2)
    assert identity["r2"].statuses[0] == IDENTITY_UNRESOLVED


def test_goalkeeper_rule_needs_an_independently_verified_previous_frame():
    _, identity = _keeper_run(flip_at=0)  # first observation: nothing verified before it
    assert identity["r0"].statuses[0] == IDENTITY_UNRESOLVED
    assert identity["r0"].teams[0] == TEAM_IDENTITY_AMBIGUOUS


def test_goalkeeper_rule_is_off_in_a_penalty_shootout():
    """Both keepers stand in the same goal in turn: continuity means nothing."""
    _, identity = _keeper_run(period_of=lambda k: 5)
    assert identity["r2"].statuses[0] == IDENTITY_SPEC
    assert identity["r2"].teams[0] == B  # raw reading kept, not "corrected"


def test_goalkeeper_rule_is_off_without_the_full_event_list():
    records = [_record(f"r{k}", A, 1000 * k, flip_flags=(0,) if k == 2 else ()) for k in range(4)]
    problem_a = [OrientationResult("NORMAL", 1.0, 0.0)] * 4
    identity = resolve_match_team_identity(records, ["NORMAL"] * 4, problem_a, (A, B), None)
    assert identity["r2"].statuses[0] == IDENTITY_UNRESOLVED


def test_unknown_orientation_gives_and_receives_no_geometric_evidence():
    records, _ = _keeper_run()
    identity = _resolve(records, orientations=["NORMAL", "NORMAL", "UNCERTAIN", "NORMAL"])
    assert identity["r2"].statuses[0] == IDENTITY_SPEC


# --- real-data regressions (the five reviewed cases and earlier ones) -----------

def _real(match_id: int):
    if not (RAW / "events" / f"{match_id}.json").is_file():
        pytest.skip("raw match not available")
    from src.orientation_resolution import match_team_ids, resolve_match_orientation

    events = json.loads((RAW / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((RAW / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    orientation = resolve_match_orientation(match_id, events, frames)
    records = orientation.records.usable_frames
    names = {r.event["team"]["id"]: r.event["team"]["name"] for r in records}
    identity = resolve_match_team_identity(
        [(r.event, r.frame_360) for r in records], [x.orientation for x in orientation.resolved],
        orientation.problem_a, tuple(names[t] for t in match_team_ids(orientation.records)), events,
    )
    return records, identity


def _at(records, identity, period, timestamp, event_type):
    return next(identity[r.event["id"]] for r in records if r.event["period"] == period
                and r.event["timestamp"] == timestamp and r.event["type"]["name"] == event_type)


@pytest.mark.parametrize("match_id, period, timestamp, event_type, expected", [
    # case_01: Costa Rica Duel, verbatim copy of Spain's record (actor differs)
    (3893791, 1, "00:18:25.761", "Duel", IDENTITY_CORRECTED_VIEW),
    (3893791, 1, "00:18:25.761", "Ball Receipt*", IDENTITY_SPEC),
    # case_02: Köln Block derived from Leverkusen's Block (mirror + 2.53 yd), flags not inverted
    (3895107, 2, "00:37:13.060", "Block", IDENTITY_CORRECTED_VIEW),
    # case_03 / case_04: crowding artefacts, not errors -> unchanged
    (3857256, 2, "00:45:23.300", "Pass", IDENTITY_SPEC),
    (3844386, 1, "00:31:58.616", "Pressure", IDENTITY_SPEC),
    # case_05: Barcelona Ball Receipt*, copy of Alavés's Block in Alavés's view
    (3773386, 1, "00:27:13.282", "Ball Receipt*", IDENTITY_CORRECTED_VIEW),
    (3773386, 1, "00:27:13.282", "Block", IDENTITY_SPEC),
])
def test_reviewed_cases(match_id, period, timestamp, event_type, expected):
    records, identity = _real(match_id)
    frame = _at(records, identity, period, timestamp, event_type)
    assert set(frame.statuses) == {expected}
    if expected == IDENTITY_SPEC:
        assert frame.teams == frame.legacy_teams  # previously correct assignment preserved


def test_reviewed_copy_cases_now_agree_with_their_partner_record():
    records, identity = _real(3893791)
    duel = _at(records, identity, 1, "00:18:25.761", "Duel")
    receipt = _at(records, identity, 1, "00:18:25.761", "Ball Receipt*")
    assert duel.teams == receipt.teams and duel.decision == receipt.decision


def test_uruguay_ghana_byte_identical_copies_follow_the_continuity_view():
    """The display-only refinement found these two groups; the shared
    resolution must reach the same colours (keeper = Ghana)."""
    records, identity = _real(3857293)
    for period, timestamp in ((1, "00:44:24.954"), (2, "00:38:27.375")):
        for event_type in ("Ball Receipt*", "Block"):
            frame = _at(records, identity, period, timestamp, event_type)
            keeper = next(i for i, p in enumerate(next(r for r in records if r.event["id"] == frame.event_id)
                                                  .frame_360["freeze_frame"]) if p.get("keeper"))
            assert frame.teams[keeper] == "Ghana"


def test_kickoff_505_raw_scramble_is_not_recoloured():
    """No structural partner and no previous snapshot: the two-neighbour
    detector cannot evaluate it, and no rule invents colours for it."""
    records, identity = _real(3930164)
    frame = _at(records, identity, 1, "00:00:00.505", "Pass")
    assert frame.teams == frame.spec_teams


# --- orientation synchronization -------------------------------------------------

def test_builder_and_renderer_use_identical_snapshot_orientations():
    match_id = 3773386
    if not (RAW / "events" / f"{match_id}.json").is_file():
        pytest.skip("raw match not available")
    from src.animate_statsbomb_360_match import (
        compute_orientation_results, display_orientation_results, frame_review_corrections,
        infer_broadcast_orientation_for_match, load_and_join_match, resolve_display_plan,
    )
    from src.build_statsbomb_360_dataset import build_match_dataset

    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    drawn, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    renderer = {j.event["id"]: r.orientation for j, r in zip(raw.usable_frames, drawn)}
    rows = build_match_dataset(match_id, 10.0)
    compared = [r for r in rows if r["current_event_id"] in renderer]
    assert compared and all(r["orientation_offline"] == renderer[r["current_event_id"]] for r in compared)


# --- defects found in the final validation (2026-10-27) ---------------------------

def test_tiny_byte_identical_cross_team_copy_keeps_the_legacy_source_view():
    """Byte-identical copies are linked whatever their size. Before the fix,
    copies with < 3 players were never linked and silently fell back to the
    spec reading, undoing legacy Problem C (5 groups / 14 records)."""
    tiny = [(55.0, 45.0, False, False), (30.0, 35.0, True, False)]
    source = _record("a", A, 5000, scene=tiny, actor=1)
    borrower_event, borrower_frame = _record("a", A, 5000, scene=tiny, actor=1)
    borrower_event = {**borrower_event, "id": "b", "team": dict(TEAMS[B]), "index": 5001}
    borrower_frame = {**borrower_frame, "event_uuid": "b"}
    identity = _resolve([source, (borrower_event, borrower_frame)], orientations=["NORMAL", "UNCERTAIN"],
                        owners=["NORMAL", "UNCERTAIN"])
    assert identity["b"].component is not None
    assert identity["b"].teams == identity["a"].teams == identity["b"].legacy_teams


def test_same_scene_copy_with_partly_different_flags_marks_only_the_disagreeing_players():
    """Identical coordinates at the same instant, 2 of 8 flags differ: the
    two records contradict each other on exactly those players."""
    a = _record("a", A, 5000)
    b_event, b_frame = _record("a", A, 5000, flip_flags=(4, 5), actor=3)
    b_event = {**b_event, "id": "b", "index": 5001}
    b_frame = {**b_frame, "event_uuid": "b"}
    identity = _resolve([_neighbours()[0], a, (b_event, b_frame), _neighbours()[1]])
    for record in ("a", "b"):
        frame = identity[record]
        assert {i for i, s in enumerate(frame.statuses) if s == IDENTITY_UNRESOLVED} == {4, 5}
        assert {frame.reasons[i] for i in (4, 5)} == {REASON_SIMULTANEOUS_CONTRADICTION}


def test_near_simultaneous_opposing_record_with_uninverted_flags_is_not_trusted():
    """Opposing-team snapshot 130 ms later, re-expressed in its own frame but
    with the flags NOT inverted (and players moved, so no exact relation):
    every player it shares with the other record is contradicted."""
    no_keepers = [(x, y, mate, False) for x, y, mate, _ in SCENE]
    other_event, other_frame = _record("b", B, 5130, scene=no_keepers, flip_flags=range(len(SCENE)))
    identity = _resolve([_record("a", A, 5000, scene=no_keepers), (other_event, other_frame)])
    assert set(identity["b"].statuses) == {IDENTITY_UNRESOLVED}
    assert set(identity["a"].statuses) == {IDENTITY_UNRESOLVED}


# --- causal (prediction-time) resolution ---------------------------------------------

def _resolve_causal(records, orientations=None, owners=None, all_events=()):
    orientations = orientations or ["NORMAL"] * len(records)
    owners = owners or orientations
    problem_a = [OrientationResult(o, 0.0 if o == "UNCERTAIN" else 1.0, 0.0) for o in owners]
    events = [event for event, _ in records] + list(all_events)
    return resolve_match_team_identity(records, orientations, problem_a, (A, B), events, causal=True)


def test_causal_mode_marks_only_the_later_record_of_a_contradiction():
    a = _record("a", A, 5000)
    b_event, b_frame = _record("a", A, 5000, flip_flags=(4,), actor=3)
    b_event, b_frame = {**b_event, "id": "b", "index": 5001}, {**b_frame, "event_uuid": "b"}
    identity = _resolve_causal([a, (b_event, b_frame)])
    assert set(identity["a"].statuses) == {IDENTITY_SPEC}  # cannot see b yet
    assert identity["b"].statuses[4] == IDENTITY_UNRESOLVED


def test_causal_mode_resolves_a_copy_from_the_previous_snapshot_only():
    no_keepers = [(x, y, mate, False) for x, y, mate, _ in SCENE]
    copy_event, copy_frame = _record("b", A, 5000, scene=no_keepers, actor=5)
    copy_event = {**copy_event, "team": dict(TEAMS[B]), "index": 5001}
    records = [_record("before", A, 4000, scene=no_keepers), _record("a", A, 5000, scene=no_keepers),
               (copy_event, copy_frame)]
    identity = _resolve_causal(records, orientations=["NORMAL", "NORMAL", "ROTATED_180"])
    assert identity["b"].decision == DECISION_CONTINUITY
    assert identity["b"].teams == identity["a"].teams


def test_causal_goalkeeper_rule_uses_only_the_past():
    records = [_record(f"r{k}", A, 1000 * k, index=k, flip_flags=(0,) if k == 2 else ()) for k in range(3)]
    identity = _resolve_causal(records)
    assert identity["r2"].statuses[0] == IDENTITY_CORRECTED_GOALKEEPER


def test_causal_resolution_of_a_record_never_depends_on_later_records():
    """Truncating the match after any record must not change that record's
    causal resolution, nor any earlier one's (real data)."""
    records, _ = _real(3773386)  # skips without raw data
    from src.orientation_resolution import match_team_ids, resolve_match_orientation

    events = json.loads((RAW / "events" / "3773386.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((RAW / "three-sixty" / "3773386.json").read_text(encoding="utf-8"))}
    orientation = resolve_match_orientation(3773386, events, frames)
    usable = orientation.records.usable_frames
    names = {r.event["team"]["id"]: r.event["team"]["name"] for r in usable}
    pair = tuple(names[t] for t in match_team_ids(orientation.records))
    inputs = [((r.event, r.frame_360), o.orientation, p) for r, o, p in zip(usable, orientation.resolved, orientation.problem_a)]
    inputs.sort(key=lambda x: (x[0][0]["period"], x[0][0]["timestamp"], x[0][0]["index"]))

    def causal(prefix):
        return resolve_match_team_identity([x[0] for x in prefix], [x[1] for x in prefix], [x[2] for x in prefix],
                                           pair, events, causal=True)

    full = causal(inputs)
    for cut in (400, 1100, 1700):  # includes the case_05 copy group at index ~1050
        truncated = causal(inputs[:cut])
        for event_id, frame in truncated.items():
            assert full[event_id].teams == frame.teams and full[event_id].statuses == frame.statuses


def test_builder_offline_columns_equal_what_the_renderer_draws():
    """Same rules, same inputs: the builder's *_offline counts are exactly the
    renderer's colours; the causal feature columns never see later records."""
    match_id = 3773386
    if not (RAW / "events" / f"{match_id}.json").is_file():
        pytest.skip("raw match not available")
    from src.animate_statsbomb_360_match import (
        compute_orientation_results, display_orientation_results, frame_review_corrections,
        infer_broadcast_orientation_for_match, load_and_join_match, load_json_records, match_team_ids,
        resolve_display_plan,
    )
    from src.build_statsbomb_360_dataset import build_match_dataset

    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    drawn, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    renderer = resolve_match_team_identity(
        [(j.event, j.frame_360) for j in raw.usable_frames], [r.orientation for r in drawn], results,
        tuple(names[t] for t in match_team_ids(raw)), load_json_records(raw.event_path, "event"),
    )
    rows = build_match_dataset(match_id, 10.0)
    assert rows
    for row in rows:
        teams = renderer[row["current_event_id"]].teams
        assert row["visible_teammates_offline"] == sum(t == row["current_team"] for t in teams)
        assert row["visible_ambiguous_offline"] == sum(t == TEAM_IDENTITY_AMBIGUOUS for t in teams)
        assert row["visible_opponents_offline"] == len(teams) - row["visible_teammates_offline"] - row["visible_ambiguous_offline"]
