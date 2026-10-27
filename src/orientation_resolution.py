"""Shared snapshot-orientation resolution for the renderer AND the dataset builder.

Moved out of src/animate_statsbomb_360_match.py on 2026-10-26 (no logic
change) so that the ML dataset builder resolves Problem A UNCERTAIN
snapshots with exactly the validated rules the renderer uses (same-moment
duplicates, kick-off own-half rule, camera side + Problem A lean, camera +
temporal continuity, reviewer frame decisions), without importing the
renderer's plotting stack. The renderer re-exports every name defined here,
so existing scripts that import them from the renderer keep working.

What this module decides is ONLY "is this snapshot stored in its own
event's frame or rotated by 180 degrees?". It never decides team identity
(src/team_identity.py) and never normalizes teams onto one pitch
(src/team_normalization.py for ML, the display plan for rendering).

`match` arguments are duck-typed: anything with a `usable_frames` list of
objects carrying `.event` and `.frame_360` (the renderer's MatchFrames, or
`OrderedRecords` below).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    from orientation_correction import (
        ORIENTATION_NORMAL,
        ORIENTATION_ROTATED_180,
        ORIENTATION_UNCERTAIN,
        OrientationResult,
        apply_correction,
        classify_orientation,
        classify_orientations_for_match,
        mirror_point,
        resolve_uncertain_with_neighbors,
    )
    from team_normalization import resolve_reference_team_id
    from physical_orientation_calibration import (
        load_calibration_registry,
        load_frame_reviews,
        load_period_reviews,
        resolve_usable_physical_reference_team_name,
    )
    from broadcast_orientation import (
        CAMERA_SIDE_UNKNOWN,
        SHARED_FRAME_PERIODS,
        PROVENANCE_PREFERENCE,
        PROVENANCE_VERIFIED,
        BroadcastOrientationInference,
        DisplayOrientationPlan,
        build_display_plan,
        estimate_camera_side,
        expected_own_frame_camera_side,
        infer_broadcast_orientation,
        kickoff_own_half_orientation,
        resolve_uncertain_for_display,
    )
    from team_identity import snapshot_key
except ImportError:
    from src.orientation_correction import (
        ORIENTATION_NORMAL,
        ORIENTATION_ROTATED_180,
        ORIENTATION_UNCERTAIN,
        OrientationResult,
        apply_correction,
        classify_orientation,
        classify_orientations_for_match,
        mirror_point,
        resolve_uncertain_with_neighbors,
    )
    from src.team_normalization import resolve_reference_team_id
    from src.physical_orientation_calibration import (
        load_calibration_registry,
        load_frame_reviews,
        load_period_reviews,
        resolve_usable_physical_reference_team_name,
    )
    from src.broadcast_orientation import (
        CAMERA_SIDE_UNKNOWN,
        SHARED_FRAME_PERIODS,
        PROVENANCE_PREFERENCE,
        PROVENANCE_VERIFIED,
        BroadcastOrientationInference,
        DisplayOrientationPlan,
        build_display_plan,
        estimate_camera_side,
        expected_own_frame_camera_side,
        infer_broadcast_orientation,
        kickoff_own_half_orientation,
        resolve_uncertain_for_display,
    )
    from src.team_identity import snapshot_key


class OrientationResolutionError(ValueError):
    """Malformed event fields that make orientation resolution impossible."""


@dataclass(frozen=True)
class OrderedRecord:
    """One event joined to its raw 360 record (renderer: JoinedFrame)."""

    event: dict[str, Any]
    frame_360: dict[str, Any]


@dataclass(frozen=True)
class OrderedRecords:
    """Minimal stand-in for the renderer's MatchFrames."""

    usable_frames: list[OrderedRecord]


def _integer_field(event: dict[str, Any], field: str) -> int:
    value = event.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OrientationResolutionError(
            f"Invalid event {event.get('id', '<unknown>')}: {field} must be an integer."
        )
    return value


def _seconds(timestamp: Any) -> float:
    if not isinstance(timestamp, str):
        raise OrientationResolutionError("Invalid event timestamp: expected a string.")
    try:
        parsed = datetime.strptime(timestamp, "%H:%M:%S.%f")
    except ValueError as error:
        raise OrientationResolutionError(f"Invalid event timestamp {timestamp!r}.") from error
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second + parsed.microsecond / 1e6


def renderer_order_key(event: dict[str, Any]) -> tuple[int, int, int, int]:
    """The renderer's chronological order (period, minute, second, index).

    Problem A's temporal refinement looks at neighbouring records by
    position, so both consumers must classify the same records in the same
    order to get the same results. Events without minute/second (synthetic
    fixtures) are placed by their period-relative timestamp instead, which
    orders a period's events the same way.
    """
    if "minute" in event and "second" in event:
        minute, second = _integer_field(event, "minute"), _integer_field(event, "second")
    else:
        elapsed = int(_seconds(event.get("timestamp")))
        minute, second = divmod(elapsed, 60)
    return (_integer_field(event, "period"), minute, second, _integer_field(event, "index"))


def ordered_usable_records(
    events: list[dict[str, Any]], frames_by_event_id: dict[str, dict[str, Any]]
) -> OrderedRecords:
    """Exactly the renderer's usable set and order: events joined to a 360
    record with a non-empty freeze_frame, sorted by `renderer_order_key`."""
    usable = [
        OrderedRecord(event, frames_by_event_id[event["id"]])
        for event in events
        if event.get("id") in frames_by_event_id
        and isinstance(frames_by_event_id[event["id"]].get("freeze_frame"), list)
        and bool(frames_by_event_id[event["id"]]["freeze_frame"])
    ]
    usable.sort(key=lambda record: renderer_order_key(record.event))
    return OrderedRecords(usable)


def compute_orientation_results(match: Any) -> list[OrientationResult]:
    """Problem A for every usable record, in `match.usable_frames` order."""
    frames = match.usable_frames
    events = [joined.event for joined in frames]
    return classify_orientations_for_match(
        events,
        [joined.frame_360 for joined in frames],
        [_seconds(event.get("timestamp")) for event in events],
        [_integer_field(event, "period") for event in events],
    )


def match_team_ids(match: Any) -> tuple[int, int]:
    """The match's two team ids, in chronological order of first appearance."""
    team_ids: list[int] = []
    for joined in match.usable_frames:
        team = joined.event.get("team")
        team_id = team.get("id") if isinstance(team, dict) else None
        if isinstance(team_id, int) and not isinstance(team_id, bool) and team_id not in team_ids:
            team_ids.append(team_id)
    if len(team_ids) != 2:
        raise OrientationResolutionError(f"Expected exactly two team ids, found {team_ids}.")
    return team_ids[0], team_ids[1]


def infer_broadcast_orientation_for_match(
    match: Any, orientation_results: list[OrientationResult]
) -> BroadcastOrientationInference:
    """Problem D: infer the camera-at-bottom reference team per period.

    `match` must be the RAW match (before Problem A correction or team
    normalization): each snapshot is put into its own event's frame here,
    using the precomputed Problem A results, and UNCERTAIN snapshots are
    left out because their frame is unknown. See
    src/broadcast_orientation.py.
    """
    pairs = [
        (joined.event, apply_correction(joined.frame_360, result.orientation))
        for joined, result in zip(match.usable_frames, orientation_results)
        if result.orientation != ORIENTATION_UNCERTAIN
    ]
    return infer_broadcast_orientation(
        [event for event, _ in pairs], [frame for _, frame in pairs], match_team_ids(match)
    )


CONTINUITY_MAX_SECONDS = 3.0
CONTINUITY_RATIO = 0.5

# Resolution routes that may read records AFTER the snapshot's own event
# (neighbours on both sides, other copies of the same snapshot in any order,
# or a reviewer who watched the whole sequence). The dataset builder flags
# such rows, because at prediction time those records do not exist yet.
# ("geometry" reads only the record itself; the kick-off and camera-side
# routes read the record plus the period's camera side, a property of the
# stadium camera estimated from the whole period.)
ROUTES_USING_LATER_RECORDS = frozenset({
    "temporal_neighbor", "display_duplicate_snapshot", "display_duplicate_consensus",
    "display_camera_continuity", "user_frame_review",
})


def display_orientation_results(
    match: Any,
    orientation_results: list[OrientationResult],
    inference: BroadcastOrientationInference,
    plan: DisplayOrientationPlan | None = None,
    use_duplicates: bool = True,
    use_kickoff_rule: bool = True,
    frame_reviews: dict[str, str] | None = None,
) -> tuple[list[OrientationResult], dict[str, int]]:
    """Problem A results with UNCERTAIN snapshots resolved. `match` must be
    the RAW match. Used by the renderer and (since 2026-10-26) by the dataset
    builder. Deterministic.

    0. Reviewer decisions (`frame_reviews`: event id -> "NONE" /
       "ROTATED_180", from the registry's frame_reviews) for UNCERTAIN
       snapshots only; route "user_frame_review". Confident results are
       never overridden by a review.
    1. Same-moment duplicates (needs `plan`). StatsBomb often attaches one
       byte-identical snapshot to several events at the same instant; all
       copies must be DRAWN identically. A copy is drawn rotated relative to
       the raw snapshot by D = c XOR r (c: its Problem A correction, r: its
       period/team display rotation). Per group of copies:
         - confident copies all imply the same D -> every UNCERTAIN copy
           gets c = D XOR r ("display_duplicate_snapshot");
         - confident copies disagree -> D is the one under which the fixed
           camera appears where it must (expected_own_frame_camera_side);
           unreadable camera -> the larger summed confidence; tie -> group
           left as it is. Copies whose c changes, confident ones included,
           are marked "display_duplicate_consensus";
         - no confident copy -> each copy goes through steps 2-3; the group
           is then aligned on the camera's D only if at least one copy's
           own evidence already chose it.
    2. Kick-off own-half rule (Law 8), unless the camera side contradicts it.
    3. Problem A's vote lean AND the fixed-camera side agree; with `plan`,
       a camera-undecided period takes its expected side from the plan.
    4. The fixed-camera side AND temporal continuity with the nearest
       confident neighbouring snapshot both say "stored rotated".

    Returns results and counts per route, plus
    `duplicate_camera_disagreements` (confident copies agreed, camera did
    not) and `duplicate_conflicts` (confident copies disagreed).
    """
    counts = {
        "resolved_by_duplicate": 0, "duplicate_consensus_overrides": 0, "resolved_by_kickoff_rule": 0,
        "resolved_by_camera_continuity": 0, "resolved_by_user_frame_review": 0,
        "resolved_rotated": 0, "resolved_normal": 0, "still_uncertain": 0,
        "duplicate_conflicts": 0, "duplicate_conflicts_unresolved": 0, "duplicate_camera_disagreements": 0,
    }
    frames = match.usable_frames

    def team_of(event: dict[str, Any]) -> int | None:
        team = event.get("team")
        return team.get("id") if isinstance(team, dict) else None

    def rotation(index: int) -> bool:
        event = frames[index].event
        return plan.rotates(_integer_field(event, "period"), team_of(event))

    def camera_d(index: int) -> bool | None:
        """D under which copy `index` shows the fixed camera where it must."""
        event = frames[index].event
        expected = expected_own_frame_camera_side(inference, _integer_field(event, "period"), team_of(event), plan)
        side = estimate_camera_side(frames[index].frame_360.get("visible_area")).side
        if CAMERA_SIDE_UNKNOWN in (side, expected):
            return None
        return (side != expected) ^ rotation(index)

    resolved = list(orientation_results)
    routes: dict[int, str] = {}

    def set_correction(index: int, rotate: bool, route: str) -> None:
        orientation = ORIENTATION_ROTATED_180 if rotate else ORIENTATION_NORMAL
        base = orientation_results[index]
        resolved[index] = OrientationResult(orientation, base.confidence, base.best_distance, base.votes, route)
        routes[index] = route

    for index, joined in enumerate(frames):
        decision = (frame_reviews or {}).get(joined.event.get("id"))
        if decision is not None and orientation_results[index].orientation == ORIENTATION_UNCERTAIN:
            set_correction(index, decision == ORIENTATION_ROTATED_180, "user_frame_review")

    uncertain_only_groups: list[list[int]] = []
    if plan is not None and use_duplicates:
        groups: dict[tuple, list[int]] = {}
        for index, joined in enumerate(frames):
            groups.setdefault((_integer_field(joined.event, "period"), snapshot_key(joined.frame_360)), []).append(index)
        for members in groups.values():
            if len(members) < 2:
                continue
            confident = [m for m in members if orientation_results[m].orientation != ORIENTATION_UNCERTAIN]
            if not confident:
                uncertain_only_groups.append(members)
                continue
            implied = {m: (orientation_results[m].orientation == ORIENTATION_ROTATED_180) ^ rotation(m) for m in confident}
            if len(set(implied.values())) == 1:
                d = next(iter(implied.values()))
                camera = camera_d(members[0])
                for m in members:
                    if orientation_results[m].orientation == ORIENTATION_UNCERTAIN and m not in routes:
                        set_correction(m, d ^ rotation(m), "display_duplicate_snapshot")
                        if camera is not None and camera != d:
                            counts["duplicate_camera_disagreements"] += 1
                continue
            counts["duplicate_conflicts"] += 1
            d = camera_d(members[0])
            if d is None:
                weight = {True: 0.0, False: 0.0}
                for m, value in implied.items():
                    weight[value] += orientation_results[m].confidence
                if weight[True] == weight[False]:
                    counts["duplicate_conflicts_unresolved"] += 1
                    continue
                d = weight[True] > weight[False]
            for m in members:
                current = (resolved[m].orientation == ORIENTATION_ROTATED_180)
                if orientation_results[m].orientation == ORIENTATION_UNCERTAIN:
                    if m not in routes:
                        set_correction(m, d ^ rotation(m), "display_duplicate_snapshot")
                elif current != (d ^ rotation(m)):
                    set_correction(m, d ^ rotation(m), "display_duplicate_consensus")
                    counts["duplicate_consensus_overrides"] += 1

    for index, (joined, result) in enumerate(zip(frames, orientation_results)):
        if result.orientation != ORIENTATION_UNCERTAIN or index in routes:
            continue
        event = joined.event
        period = _integer_field(event, "period")
        expected = expected_own_frame_camera_side(inference, period, team_of(event), plan)
        side = estimate_camera_side(joined.frame_360.get("visible_area")).side
        camera_says_rotated = None if CAMERA_SIDE_UNKNOWN in (side, expected) else side != expected
        decision, route = None, None
        if use_kickoff_rule and ((event.get("pass") or {}).get("type") or {}).get("name") == "Kick Off":
            verdict = kickoff_own_half_orientation(joined.frame_360)
            if verdict is not None and (camera_says_rotated is None or camera_says_rotated == (verdict == ORIENTATION_ROTATED_180)):
                decision, route = verdict, "display_kickoff_own_half"
        if decision is None:
            weight = sum(vote.weight for vote in result.votes)
            lean = sum(vote.vote * vote.weight for vote in result.votes) / weight if weight else 0.0
            if lean != 0:
                decision = resolve_uncertain_for_display(lean < 0, joined.frame_360.get("visible_area"), expected)
                route = "display_camera_side"
        if decision is not None:
            set_correction(index, decision == ORIENTATION_ROTATED_180, route)

    # Step 4: camera AND temporal continuity agree. Compared with the nearest
    # confidently-classified, different snapshot of the same period within
    # CONTINUITY_MAX_SECONDS, a correctly drawn frame lies close to it and a
    # flipped one close to its mirror image. Validated on 1,012 frames whose
    # correct display is known from a same-moment duplicate: "mirrored" never
    # occurred (874 continuous, 75 unclear, 63 without neighbour).
    if plan is not None:
        times = [_seconds(j.event.get("timestamp")) for j in frames]
        keys = [snapshot_key(j.frame_360) for j in frames]

        def drawn_points(index: int, rotate_display: bool) -> list[tuple[float, float]]:
            points = [tuple(p["location"]) for p in frames[index].frame_360.get("freeze_frame") or []
                      if isinstance(p.get("location"), list)]
            return [mirror_point(p) for p in points] if rotate_display else points

        def mean_nearest(a: list, b: list) -> float:
            return statistics.fmean(min(math.dist(p, q) for q in b) for p in a) if a and b else float("inf")

        for index, result in enumerate(orientation_results):
            if result.orientation != ORIENTATION_UNCERTAIN or index in routes:
                continue
            camera = camera_d(index)
            if camera is None or camera == rotation(index):
                continue  # camera unreadable, or says the stored orientation is right
            period = _integer_field(frames[index].event, "period")
            neighbours = [
                j for j in range(max(0, index - 20), min(len(frames), index + 21))
                if j != index and keys[j] != keys[index]
                and orientation_results[j].orientation != ORIENTATION_UNCERTAIN
                and _integer_field(frames[j].event, "period") == period
                and abs(times[j] - times[index]) <= CONTINUITY_MAX_SECONDS
            ]
            if not neighbours:
                continue
            j = min(neighbours, key=lambda k: abs(times[k] - times[index]))
            reference = drawn_points(j, (resolved[j].orientation == ORIENTATION_ROTATED_180) ^ rotation(j))
            as_stored = drawn_points(index, rotation(index))
            same = mean_nearest(as_stored, reference)
            mirrored = mean_nearest([mirror_point(p) for p in as_stored], reference)
            if mirrored < CONTINUITY_RATIO * same:
                set_correction(index, True, "display_camera_continuity")

    # Copies with no confident member: align on the camera's D, but only if
    # one copy's own evidence (step 2/3) already chose it.
    for members in uncertain_only_groups:
        d = camera_d(members[0]) if plan is not None else None
        chosen = {(resolved[m].orientation == ORIENTATION_ROTATED_180) ^ rotation(m) for m in members if m in routes}
        if d is None or d not in chosen:
            continue
        for m in members:
            if routes.get(m) == "user_frame_review":
                continue
            if (resolved[m].orientation == ORIENTATION_ROTATED_180) ^ rotation(m) != d or m not in routes:
                set_correction(m, d ^ rotation(m), "display_duplicate_snapshot")

    for index, result in enumerate(orientation_results):
        route = routes.get(index)
        if result.orientation != ORIENTATION_UNCERTAIN:
            continue
        if route is None:
            counts["still_uncertain"] += 1
        elif route == "display_kickoff_own_half":
            counts["resolved_by_kickoff_rule"] += 1
        elif route == "user_frame_review":
            counts["resolved_by_user_frame_review"] += 1
        elif route == "display_camera_continuity":
            counts["resolved_by_camera_continuity"] += 1
        elif route == "display_camera_side":
            counts["resolved_rotated" if resolved[index].orientation == ORIENTATION_ROTATED_180 else "resolved_normal"] += 1
        else:
            counts["resolved_by_duplicate"] += 1
    return resolved, counts


def frame_review_corrections() -> dict[str, str]:
    """Registry frame_reviews as {event_id: snapshot correction} for display."""
    return {event_id: review.snapshot_correction for event_id, review in load_frame_reviews().items()}


def resolve_display_plan(
    match_id: int,
    match: Any,
    broadcast_inference: BroadcastOrientationInference,
    physical_reference_team: str | None = None,
    presentation_reference_team: str | None = None,
) -> tuple[DisplayOrientationPlan, str, list[str]]:
    """The renderer's single source of truth for the default display plan.

    Precedence: explicit physical team > explicit presentation preference >
    verified registry calibration (period-1 anchors) > per-period reviews
    from the registry's period_reviews > camera-side inference > legacy
    automatic fallback (marked UNKNOWN). Returns the plan, the name of the
    anchor source, and notices/warnings for the caller to print. Scripts
    that must reproduce exactly what the renderer shows call this too.
    """
    events = [joined.event for joined in match.usable_frames]
    periods_present = sorted({_integer_field(event, "period") for event in events})
    messages: list[str] = []
    anchor_team, source = physical_reference_team, "cli_override" if physical_reference_team else None
    if anchor_team is None and presentation_reference_team is not None:
        anchor_team, source = presentation_reference_team, "presentation_preference"
    if anchor_team is None:
        registry_team = resolve_usable_physical_reference_team_name(match_id, load_calibration_registry())
        if registry_team is not None:
            anchor_team, source = registry_team, "registry"
    anchor_id = resolve_reference_team_id(events, anchor_team) if anchor_team is not None else None

    period_reviews = load_period_reviews().get(match_id, {})
    confirmed = {
        period: resolve_reference_team_id(events, review.display_reference_team)
        for period, review in period_reviews.items()
    }
    if period_reviews and anchor_id is not None:
        messages.append(
            f"Notice: period review(s) for periods {sorted(period_reviews)} are not applied "
            f"because a period-1 anchor ({source}) takes priority."
        )
    plan = build_display_plan(
        broadcast_inference,
        match_team_ids(match),
        periods_present,
        anchor_period_1_reference=anchor_id,
        anchor_provenance=PROVENANCE_PREFERENCE if source == "presentation_preference" else PROVENANCE_VERIFIED,
        fallback_period_1_reference=resolve_reference_team_id(events),
        confirmed_references=confirmed,
    )
    camera_period_1 = broadcast_inference.reference_by_period.get(1)
    if anchor_id is not None and camera_period_1 is not None and camera_period_1 != anchor_id:
        messages.append(
            f"Warning: the period-1 anchor ({source}, team id {anchor_id}) contradicts the visible_area "
            f"camera-side inference (team id {camera_period_1}); the anchor is used, but this match's "
            "calibration should be re-checked."
        )
    return plan, source or "camera_side_inference", messages


@dataclass(frozen=True)
class MatchOrientation:
    """Everything one match's consumers need about snapshot orientation.

    `problem_a` is the geometry classifier's output; `resolved` is the same
    list with UNCERTAIN snapshots resolved by `display_orientation_results`
    where the evidence allows (still UNCERTAIN otherwise). Both follow
    `records.usable_frames` order.
    """

    records: OrderedRecords
    problem_a: list[OrientationResult]
    resolved: list[OrientationResult]
    plan: DisplayOrientationPlan
    counts: dict[str, int]


def resolve_match_orientation(
    match_id: int,
    events: list[dict[str, Any]],
    frames_by_event_id: dict[str, dict[str, Any]],
) -> MatchOrientation:
    """The renderer's default orientation pipeline for one match, without
    rendering: usable records in renderer order, Problem A, camera-side
    inference, default display plan (registry anchors and period reviews
    included), and the display resolution of UNCERTAIN snapshots with the
    registry's frame reviews. `events`/`frames_by_event_id` must be RAW.
    """
    records = ordered_usable_records(events, frames_by_event_id)
    problem_a = compute_orientation_results(records)
    inference = infer_broadcast_orientation_for_match(records, problem_a)
    plan, _, _ = resolve_display_plan(match_id, records, inference)
    resolved, counts = display_orientation_results(
        records, problem_a, inference, plan, frame_reviews=frame_review_corrections()
    )
    return MatchOrientation(records, problem_a, resolved, plan, counts)


# ---------------------------------------------------------------------------
# Causal (prediction-time) orientation, 2026-10-27.
#
# The display resolution above may read records AFTER a snapshot's own event
# (copies in any order, neighbours on both sides, a period-level camera
# estimate, reviewer decisions made while watching whole sequences). That is
# fine for retrospective visualization but not for features of a model that
# predicts what happens next. The causal resolution decides each snapshot
# only from the record itself and records EARLIER in the event stream
# (period, index):
#
#   geometry                    Problem A on the record alone
#   causal_temporal_neighbor    Problem A's neighbour vote, earlier anchors only
#   causal_kickoff_own_half     Law 8 own-half rule on a kick-off snapshot
#                               (record only; no camera veto)
#   causal_duplicate_snapshot   an EARLIER copy of the same raw snapshot with a
#                               causal orientation: two copies' own frames are
#                               mirror images exactly when their events'
#                               teams differ (pure geometry, no camera plan)
#
# Anything else stays UNCERTAIN: the snapshot's frame is not known at
# prediction time, and consumers must treat its coordinates as unavailable
# rather than guess.
# ---------------------------------------------------------------------------

CAUSAL_ROUTES = frozenset({
    "geometry", "causal_temporal_neighbor", "causal_kickoff_own_half", "causal_duplicate_snapshot",
})
CAUSAL_UNRESOLVED_ROUTE = "unresolved"


def event_order_key(event: dict[str, Any]) -> tuple[int, int]:
    """Position in the event stream. Causal availability follows this order:
    a record is available to event e if it precedes e in the stream. (33
    adjacent pairs dataset-wide have a later index but an earlier timestamp,
    so timestamps are not used for this.)"""
    return _integer_field(event, "period"), _integer_field(event, "index")


def resolve_causal_orientations(records: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, OrientationResult]:
    """Causal orientation for every (event, RAW 360 record) pair, by event id.

    Records may come in any order; they are put into event order here. A
    record's result depends only on itself and the records before it."""
    ordered = sorted(records, key=lambda pair: event_order_key(pair[0]))
    events = [event for event, _ in ordered]
    frames = [frame for _, frame in ordered]
    pass_one = [classify_orientation(event, frame) for event, frame in ordered]
    refined = resolve_uncertain_with_neighbors(
        events, frames, [_seconds(e.get("timestamp")) for e in events],
        [_integer_field(e, "period") for e in events], pass_one, causal=True,
    )
    resolved: list[OrientationResult] = []
    first_copy: dict[tuple, int] = {}
    for k, (event, frame) in enumerate(ordered):
        result = refined[k]
        if result.orientation != ORIENTATION_UNCERTAIN:
            route = "geometry" if result.resolved_by == "geometry" else "causal_temporal_neighbor"
            result = OrientationResult(result.orientation, result.confidence, result.best_distance, result.votes, route)
        elif ((event.get("pass") or {}).get("type") or {}).get("name") == "Kick Off" \
                and (verdict := kickoff_own_half_orientation(frame)) is not None:
            result = OrientationResult(verdict, result.confidence, result.best_distance, result.votes, "causal_kickoff_own_half")
        else:
            key = (_integer_field(event, "period"), snapshot_key(frame))
            earlier = [j for j in first_copy.get(key, []) if resolved[j].orientation != ORIENTATION_UNCERTAIN]
            if earlier:
                j = earlier[0]
                rotate = resolved[j].orientation == ORIENTATION_ROTATED_180
                if key[0] not in SHARED_FRAME_PERIODS and _team_id(events[j]) != _team_id(event):
                    rotate = not rotate
                result = OrientationResult(ORIENTATION_ROTATED_180 if rotate else ORIENTATION_NORMAL, result.confidence,
                                           result.best_distance, result.votes, "causal_duplicate_snapshot")
        if result.orientation == ORIENTATION_UNCERTAIN:
            result = OrientationResult(result.orientation, result.confidence, result.best_distance, result.votes, "unresolved")
        resolved.append(result)
        first_copy.setdefault((_integer_field(event, "period"), snapshot_key(frame)), []).append(k)
    return {event["id"]: result for event, result in zip(events, resolved)}


def _team_id(event: dict[str, Any]) -> Any:
    team = event.get("team")
    return team.get("id") if isinstance(team, dict) else None
