"""Team-identity resolution for rendering (Problem C) -- StatsBomb 360 data.

Independent from, and read-only with respect to, both:
  - src/orientation_correction.py (Problem A: is this event's own freeze_frame
    internally consistent with its own event.location?)
  - src/team_normalization.py (Problem B: which team's coordinate scale do we
    standardize on?)

Problem C answers a third, different question, and touches NO coordinates:
"which ABSOLUTE team does this freeze_frame point belong to, for rendering?"

Background (forensic evidence, verified across all 426 matches with 360
data): StatsBomb sometimes attaches the exact same raw 360 snapshot
(freeze_frame + visible_area, byte-identical) to two or three different
events that belong to DIFFERENT teams and occur at the same instant (e.g. a
Ball Receipt* immediately followed by a Ball Recovery/Interception/Duel by
the other team). `teammate` is always defined relative to the event the
array was TRULY captured for, but when the same array is reused for a
second, different-team event, its `teammate` flags are not recomputed --
so reading them relative to that second event's own team produces, for
every point in the array (including whichever point happens to carry
actor=True), an absolute-team assignment that may be the OPPOSITE of the
array's true source event's reading of the exact same points. This affects
about 7.3% of all usable frames dataset-wide; within those, ~99.98% of
groups can be resolved with high confidence (see below), leaving a
residual few genuinely ambiguous groups dataset-wide.

IMPORTANT CORRECTION (found via visual GIF inspection after the initial
Problem C implementation): an event's actor is a member of that event's
OWN team only when this event is the array's TRUE source (or the array
isn't shared at all). When a DIFFERENT event merely borrows someone else's
shared array, the array's actor=True point still physically represents the
SOURCE event's own actor, not the borrowing event's (unrepresented) actor
-- so it must be resolved the same way as every other point in the array,
relative to the resolved source team, NOT forced to the borrowing event's
own team. Treating the actor as unconditionally "always this event's own
team" was itself a bug: it made the very same spatial point flip color
between two consecutive frames that shared the identical array, merely
because the currently-displayed event's team differed from the array's
true source team. The renderer's actor MARKER (a ring) is unaffected by
this and keeps tracking whichever point is flagged actor=True -- only the
FILL COLOR of that point follows the resolved absolute team, exactly like
every other point.

For points in general, resolution works like this:
  - If the raw snapshot is not shared across a different team's event: use
    the same logic already used everywhere else (`teammate` relative to
    this event's own team) -- unchanged, always reliable. (By StatsBomb's
    own schema the actor=True point always also carries teammate=True, so
    no separate actor case is needed even here.)
  - If it IS shared, collect every member that is a confident
    (`orientation == NORMAL`, `confidence >= 0.60`) self-consistent event per
    Problem A's own classifier (reused read-only; not re-implemented here).
    Confident members are grouped by which PHYSICAL TEAM they belong to
    (their own `event["team"]["id"]`), not just counted: two or more
    confident members that agree on the same team are corroborating
    evidence for that one team, not a contradiction (e.g. a team's own
    "Ball Recovery" immediately followed by that same team's own "Carry",
    both self-consistent with the identical shared array, both pointing at
    the same physical team). If every confident member agrees on exactly
    one team, that team is the array's resolved reference team, and its
    team-relative reading of `teammate` is used for every point in the
    array, for every event that shares it -- because it is literally the
    same measurement.
  - Only when confident members genuinely disagree (confident sources from
    two different teams) -- or there are none at all -- is the true source
    undeterminable without inventing information: the point is marked
    TEAM_IDENTITY_AMBIGUOUS and never guessed.

Detection MUST run on the RAW, pre-Problem-A snapshot content, because
Problem A's own correction can make two originally-identical arrays diverge
(each gets rotated to match its own event's location), which would hide the
sharing this module exists to detect.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

try:
    from orientation_correction import ORIENTATION_NORMAL, classify_orientation
except ImportError:
    from src.orientation_correction import ORIENTATION_NORMAL, classify_orientation

TEAM_IDENTITY_AMBIGUOUS = "AMBIGUOUS"

DEFAULT_CONFIDENCE_THRESHOLD = 0.60


def _team_name(event: dict[str, Any]) -> str:
    """Read an event's own team name, or 'Unknown' if absent/malformed."""
    team = event.get("team")
    if isinstance(team, dict) and isinstance(team.get("name"), str):
        return team["name"]
    return "Unknown"


def _snapshot_key(frame_360: dict[str, Any]) -> tuple:
    """A hashable key over the RAW, pre-Problem-A snapshot content."""
    freeze_frame = frame_360.get("freeze_frame") or []
    freeze_key = tuple(
        (
            player.get("teammate"),
            player.get("actor"),
            player.get("keeper"),
            tuple(player.get("location") or []),
        )
        for player in freeze_frame
        if isinstance(player, dict)
    )
    visible_area = tuple(frame_360.get("visible_area") or [])
    return (freeze_key, visible_area)


@dataclass(frozen=True)
class SharedGroupInfo:
    """One group of events that share a byte-identical raw 360 snapshot
    across more than one team.

    `source_event_id` is one event whose own team-relative reading of
    `teammate` is trusted for every point in the whole group's shared array
    -- including whichever point is flagged actor=True -- i.e. an event
    belonging to the group's resolved reference team. It is None when
    the reference team cannot be determined (zero confident members, or
    confident members that disagree across two different teams), in which
    case those points must be treated as ambiguous. Multiple confident
    members that agree on the same team are not ambiguous: any one of them
    may serve as `source_event_id`, since they all yield the identical
    reference team.
    """

    group_event_ids: tuple[str, ...]
    source_event_id: str | None


def detect_shared_cross_team_snapshots(
    events: list[dict[str, Any]],
    frames_by_event_id: dict[str, dict[str, Any]],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, SharedGroupInfo]:
    """Group RAW 360 snapshots by identical content and resolve each
    cross-team group's source event, if any.

    Returns a mapping from event_id to SharedGroupInfo for every event that
    is part of a snapshot shared with at least one OTHER team's event.
    Events not part of any such group are simply absent from the result --
    the caller should treat a missing key as "not shared" and keep using
    the existing, always-reliable teammate-relative-to-own-team logic.

    `events` and `frames_by_event_id` must be the RAW, pre-Problem-A data
    (as loaded, before any orientation correction or team normalization),
    since sharing must be detected before either correction can make two
    originally-identical arrays diverge.
    """
    groups: dict[tuple, list[str]] = defaultdict(list)
    for event in events:
        event_id = event.get("id")
        frame = frames_by_event_id.get(event_id)
        if frame is None:
            continue
        groups[_snapshot_key(frame)].append(event_id)

    events_by_id = {event["id"]: event for event in events if "id" in event}

    result: dict[str, SharedGroupInfo] = {}
    for event_ids in groups.values():
        if len(event_ids) < 2:
            continue
        team_ids = {
            events_by_id[eid]["team"]["id"]
            for eid in event_ids
            if isinstance(events_by_id.get(eid, {}).get("team"), dict)
        }
        if len(team_ids) < 2:
            continue  # shared, but within the same team -- not this phenomenon

        confident_sources = [
            eid
            for eid in event_ids
            if (
                lambda r: r.orientation == ORIENTATION_NORMAL and r.confidence >= confidence_threshold
            )(classify_orientation(events_by_id[eid], frames_by_event_id[eid]))
        ]

        # Group confident sources by which PHYSICAL TEAM they belong to, not
        # by how many events they are. Two or more confident sources that
        # agree on the same team corroborate each other (stronger evidence
        # for that team, not a contradiction); only confident sources that
        # disagree across two different teams -- or the complete absence of
        # any confident source -- leave the reference team undeterminable.
        confident_team_to_event: dict[Any, str] = {}
        for eid in confident_sources:
            team = events_by_id[eid].get("team")
            team_id = team["id"] if isinstance(team, dict) else None
            confident_team_to_event.setdefault(team_id, eid)

        source_event_id = (
            next(iter(confident_team_to_event.values()))
            if len(confident_team_to_event) == 1
            else None
        )

        info = SharedGroupInfo(group_event_ids=tuple(event_ids), source_event_id=source_event_id)
        for eid in event_ids:
            result[eid] = info

    return result


def resolve_absolute_team_for_point(
    player: dict[str, Any],
    event: dict[str, Any],
    shared_group_info: SharedGroupInfo | None,
    raw_events_by_id: dict[str, dict[str, Any]],
    team_names: tuple[str, str],
) -> str:
    """Resolve one freeze_frame player's absolute team name, or
    TEAM_IDENTITY_AMBIGUOUS if it cannot be determined without inventing
    information.

    `team_names` is the match's fixed (team_one, team_two) pair, used only
    to compute "the other team" -- it never determines which one is which;
    that mapping to colors is entirely the caller's/renderer's business.

    IMPORTANT (absolute team identity vs. event perspective): `teammate` is
    only ever a reliable signal relative to the array's TRUE physical source
    event -- the event StatsBomb actually captured this snapshot for. For a
    non-shared snapshot, that source is always this event itself, so
    reading `teammate` relative to `event`'s own team is correct, and by
    StatsBomb's own schema the array's actor=True point always also carries
    teammate=True, so no separate actor case is needed here.

    For a SHARED snapshot, a different, "borrowing" event may be currently
    rendered even though the array's content -- including whichever point
    is flagged actor=True -- still physically describes the resolved
    SOURCE event's own actor, not the borrowing event's (unrepresented)
    actor. Every point, including the actor point, must therefore be read
    relative to the resolved source team, never relative to whichever
    event happens to be on screen -- otherwise the exact same physical
    point flips color depending on which of two events sharing the array
    is currently displayed, even though nothing about the snapshot itself
    changed. (When the currently rendered event IS the resolved source,
    source_team == event's own team, so this is a no-op for that case.)

    The renderer draws the actor marker (a ring) purely from
    `player.get("actor")`, independent of this function -- so the marker
    still tracks whichever point the array flags, unaffected by this
    function's team-color resolution.
    """
    event_team = _team_name(event)

    def opponent_of(team_name: str) -> str:
        team_one, team_two = team_names
        return team_two if team_name == team_one else team_one

    teammate = player.get("teammate") is True

    if shared_group_info is None:
        return event_team if teammate else opponent_of(event_team)

    if shared_group_info.source_event_id is None:
        return TEAM_IDENTITY_AMBIGUOUS

    source_event = raw_events_by_id[shared_group_info.source_event_id]
    source_team = _team_name(source_event)
    return source_team if teammate else opponent_of(source_team)


def resolve_team_identity_for_frame(
    event: dict[str, Any],
    frame_360: dict[str, Any],
    shared_group_info: SharedGroupInfo | None,
    raw_events_by_id: dict[str, dict[str, Any]],
    team_names: tuple[str, str],
) -> list[str]:
    """Resolve absolute team identity for every freeze_frame player, in
    the same order as `frame_360["freeze_frame"]`.

    A convenience wrapper around resolve_absolute_team_for_point() for
    callers that want one pass over a whole frame. Does not touch
    coordinates or any other field.
    """
    freeze_frame = frame_360.get("freeze_frame") or []
    return [
        resolve_absolute_team_for_point(player, event, shared_group_info, raw_events_by_id, team_names)
        if isinstance(player, dict)
        else TEAM_IDENTITY_AMBIGUOUS
        for player in freeze_frame
    ]
