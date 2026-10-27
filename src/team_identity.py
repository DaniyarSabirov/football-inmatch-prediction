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


def snapshot_key(frame_360: dict[str, Any]) -> tuple:
    """Public, hashable identity of a RAW 360 snapshot's content: two
    records with equal keys are byte-identical copies of one snapshot."""
    return _snapshot_key(frame_360)


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


# ---------------------------------------------------------------------------
# Evidence-based team identity (2026-10-26)
#
# Generalizes Problem C after the hypothesis investigation in
# results/team_identity_hypothesis_investigation/. Shared by the renderer
# and the dataset builder (resolve_match_team_identity). In short:
#
# 1. The spec reading stays the default: `teammate` is relative to the
#    record's own event team (validated by goalkeeper position on 99.7% of
#    unshared records).
# 2. Structural links. Two records of the same period are the SAME scene
#    when their raw coordinates are identical, or identical up to one
#    constant offset, as stored or after a 180-degree mirror (players
#    paired by optimal assignment). Such an exact relation cannot arise
#    between two genuinely separate captures. Only `teammate` flags tell
#    the two records' flag VIEWS apart: equal flags -> same view, inverted
#    flags -> opposite views. Byte-identical copies are a special case;
#    copies that differ only in actor/keeper/visible_area, and mirrored or
#    shifted derived records, are now found too.
# 3. A linked component whose spec readings contradict these view
#    constraints (one team's view copied into the other team's record) has
#    one binary unknown: which team's view the flags are. It is chosen by
#    evidence, never by event order or by which team owns the coordinates
#    alone:
#      a. continuity: unambiguously paired players of the nearest distinct,
#         unlinked snapshots before/after (<= 2 s) agree with one view on
#         >= 4 players with at most max(1, 10%) disagreeing, and no neighbour
#         supports or leans (<= 25% vs >= 75%) towards the other view (a
#         neighbour that fits neither view carries no information); a
#         contradicting goalkeeper vetoes unless the coordinate owner sides
#         with continuity (the keeper point then goes to the goalkeeper rule);
#      b. without continuity evidence: the goalkeeper's goal side and the
#         coordinate owner (Problem A) must both name the same view;
#      c. legacy byte-identical groups with a Problem C source keep that
#         source (validated at ~98% by goalkeepers) unless (a)/(b) object;
#      d. otherwise the component's points are UNRESOLVED.
# 4. Crowding-safe contradiction detector: a player is contradicted only if
#    he is unambiguously paired (<= 3 yd, second candidate >= 1 yd further)
#    with a player in BOTH the previous and next distinct snapshot and both
#    of those carry the same, different team. Such points become UNRESOLVED;
#    the whole record only when >= 4 and >= 50% of its evaluable players are
#    contradicted. Nothing is recoloured by this step.
# 5. Narrow goalkeeper continuity rule (see _goalkeeper_correction).
# 6. Simultaneous contradictions (added 2026-10-27 after the final
#    validation): a player paired with margin >= 2 yd in two unlinked
#    snapshots <= 250 ms apart but given different teams is UNRESOLVED in
#    both -- one record is wrong about him and nothing says which. Covers
#    exact same-scene copies whose flags only partly agree and near-copies
#    of opposing teams whose flags are not inverted.
#
# Raw labels are never modified; every point keeps its spec reading next
# to its resolved team and a status.
# ---------------------------------------------------------------------------

import math as _math

import numpy as _np
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

try:
    from orientation_correction import ORIENTATION_ROTATED_180, ORIENTATION_UNCERTAIN, OrientationResult
    from broadcast_orientation import SHARED_FRAME_PERIODS
except ImportError:
    from src.orientation_correction import ORIENTATION_ROTATED_180, ORIENTATION_UNCERTAIN, OrientationResult
    from src.broadcast_orientation import SHARED_FRAME_PERIODS

# Point statuses.
IDENTITY_SPEC = "SPEC"                                    # spec reading, unchanged
IDENTITY_CORRECTED_VIEW = "CORRECTED_VIEW"                # copy/derived record read in the evidenced view
IDENTITY_CORRECTED_GOALKEEPER = "CORRECTED_GOALKEEPER"    # goalkeeper-only continuity rule
IDENTITY_UNRESOLVED = "UNRESOLVED"                        # team not established; drawn "team unclear"

# Reasons for UNRESOLVED points.
REASON_COPY_GROUP_UNDECIDED = "copy_group_undecided"
REASON_COPY_FLAG_DEVIATION = "copy_flag_deviation"
REASON_RECORD_CONTRADICTED = "record_contradicted"
REASON_POINT_CONTRADICTED = "contradicted_by_both_neighbours"
REASON_KEEPER_GOAL_SIDE = "keeper_goal_side_unverified"
REASON_GOALKEEPER_CORRECTION_CONTRADICTED = "goalkeeper_correction_contradicted"
REASON_SIMULTANEOUS_CONTRADICTION = "simultaneous_contradiction"

# Group decisions.
DECISION_SPEC = "spec_consistent"
DECISION_CONTINUITY = "continuity"
DECISION_KEEPER_AND_OWNER = "keeper_and_owner"
DECISION_LEGACY = "legacy_problem_c_source"
DECISION_UNDECIDED = "undecided"

STRUCTURE_MAX_DT_MS = 2000
EXACT_IDENTICAL_MAX_DISTANCE = 0.01
EXACT_SHIFT_MAX_RESIDUAL = 0.05
EXACT_SHIFT_MAX_OFFSET = 6.0
EXACT_MIN_POINTS = 3
EXACT_MAX_SIZE_DIFFERENCE = 2
FLAG_RELATION_SHARE = 0.9

CONTINUITY_MAX_DT_MS = 2000
PAIR_MAX_DISTANCE = 3.0
PAIR_MIN_MARGIN = 1.0
VIEW_MIN_PAIRS = 4
VIEW_MAX_DISAGREEMENT_SHARE = 0.1
VIEW_LEAN_SHARE = 0.25

SIMULTANEOUS_MAX_DT_MS = 250
SIMULTANEOUS_MIN_MARGIN = 2.0

RECORD_CONTRADICTION_MIN_POINTS = 4
RECORD_CONTRADICTION_SHARE = 0.5

KEEPER_GOAL_ZONE = 30.0
GOALKEEPER_MAX_GAP_MS = 10_000
GOALKEEPER_BASE_DISTANCE = 2.0
GOALKEEPER_SPEED_YD_PER_S = 1.5
GOALKEEPER_MAX_DISTANCE = 10.0
GOALKEEPER_TRANSITION_TYPES = frozenset(
    {"Substitution", "Player Off", "Player On", "Tactical Shift", "Half Start", "Half End"}
)


def timestamp_ms(timestamp: str) -> int:
    """'HH:MM:SS.mmm' -> integer milliseconds (StatsBomb has no finer digits)."""
    hours, minutes, rest = timestamp.split(":")
    seconds, _, fraction = rest.partition(".")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int((fraction + "000")[:3])


def unambiguous_pairs(
    a: _np.ndarray, b: _np.ndarray, max_distance: float = PAIR_MAX_DISTANCE, min_margin: float = PAIR_MIN_MARGIN
) -> list[tuple[int, int]]:
    """Optimal-assignment pairs (i in a, j in b) that are close AND unambiguous:
    the second-nearest candidate on either side is at least `min_margin`
    further away. In a crowded box the pairing of two adjacent players is
    arbitrary, so such pairs never count as evidence."""
    if len(a) == 0 or len(b) == 0:
        return []
    cost = _np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = _linear_sum_assignment(cost)
    # Consecutive snapshots are often offset as a whole by a few yards
    # (separate captures); remove the median offset of the close pairs so
    # that the offset does not masquerade as crowding.
    close = cost[rows, cols] <= max_distance + 3.0
    if close.sum() >= 3:
        offset = _np.median(b[cols[close]] - a[rows[close]], axis=0)
        if _np.linalg.norm(offset) > 0.5:
            cost = _np.linalg.norm((a + offset)[:, None, :] - b[None, :, :], axis=2)
            rows, cols = _linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(rows, cols):
        distance = cost[i, j]
        if distance > max_distance:
            continue
        others = _np.concatenate([_np.delete(cost[i], j), _np.delete(cost[:, j], i)])
        second = others.min() if len(others) else _math.inf
        if second - distance >= min_margin:
            pairs.append((int(i), int(j)))
    return pairs


def exact_structural_relation(a: _np.ndarray, b: _np.ndarray) -> tuple[bool, list[tuple[int, int]]] | None:
    """(mirrored, pairs) if raw point sets `a` and `b` are the same scene:
    identical, or identical up to one constant offset, as stored or after a
    180-degree mirror of `b`. None otherwise."""
    if min(len(a), len(b)) < EXACT_MIN_POINTS or abs(len(a) - len(b)) > EXACT_MAX_SIZE_DIFFERENCE:
        return None
    for mirrored in (False, True):
        bb = _np.column_stack([120.0 - b[:, 0], 80.0 - b[:, 1]]) if mirrored else b
        if len(a) == len(bb) and _np.abs(a.std(axis=0) - bb.std(axis=0)).max() > 0.1:
            continue  # cheap reject: spread is invariant under shift
        cost = _np.linalg.norm(a[:, None, :] - bb[None, :, :], axis=2)
        rows, cols = _linear_sum_assignment(cost)
        if cost[rows, cols].max() < EXACT_IDENTICAL_MAX_DISTANCE:
            return mirrored, list(zip(rows.tolist(), cols.tolist()))
        offset = (bb[cols] - a[rows]).mean(axis=0)
        if _np.linalg.norm(offset) < EXACT_SHIFT_MAX_OFFSET:
            shifted = a + offset
            cost = _np.linalg.norm(shifted[:, None, :] - bb[None, :, :], axis=2)
            rows, cols = _linear_sum_assignment(cost)
            if cost[rows, cols].max() < EXACT_SHIFT_MAX_RESIDUAL:
                return mirrored, list(zip(rows.tolist(), cols.tolist()))
    return None


@dataclass(frozen=True)
class FrameIdentity:
    """Resolved team identity of one record, aligned with its freeze_frame.

    `teams[i]` is a team name or TEAM_IDENTITY_AMBIGUOUS (status
    UNRESOLVED); `spec_teams[i]` is the untouched spec reading of the raw
    `teammate` flag; `legacy_teams[i]` is what the pre-2026-10-26 pipeline
    (byte-identical Problem C only) produced. Raw flags are never changed.
    """

    event_id: str
    teams: tuple[str, ...]
    spec_teams: tuple[str, ...]
    legacy_teams: tuple[str, ...]
    statuses: tuple[str, ...]
    reasons: tuple[str, ...]
    confirmed: tuple[bool, ...]          # positive independent evidence (continuity or keeper goal side)
    component: int | None                # structural component id (None: unlinked)
    decision: str                        # group decision, "" if unlinked
    keeper_evidence: tuple[str, ...]     # "consistent"/"inconsistent"/"" per point (goalkeepers near a goal)
    notes: tuple[str, ...] = ()          # per point: evidence behind a goalkeeper correction
    decision_evidence: str = ""          # group evidence summary (continuity/keeper/owner/legacy views)

    @property
    def has_unresolved(self) -> bool:
        return IDENTITY_UNRESOLVED in self.statuses

    @property
    def has_correction(self) -> bool:
        return any(s in (IDENTITY_CORRECTED_VIEW, IDENTITY_CORRECTED_GOALKEEPER) for s in self.statuses)


class _Parity:
    """Union-find with parity: view(x) = view(root) XOR parity(x)."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.parity = [0] * n
        self.inconsistent: set[int] = set()

    def find(self, x: int) -> tuple[int, int]:
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        root, acc = x, 0
        for node in reversed(path):
            acc ^= self.parity[node]
            self.parity[node] = acc
            self.parent[node] = root
        return root, (self.parity[path[0]] if path else 0)

    def union(self, a: int, b: int, relation: int) -> None:
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            if pa ^ pb != relation:
                self.inconsistent.add(ra)
            return
        self.parent[rb] = ra
        self.parity[rb] = pa ^ pb ^ relation
        if rb in self.inconsistent:
            self.inconsistent.add(ra)


class _MatchData:
    """One match's records, sorted (period, ms, index), with everything the
    identity rules read. Points are indexed by their position among the
    record's points with a usable location (`valid`)."""

    def __init__(self, records, resolved_orientations, problem_a_results, team_names, event_order: bool = False) -> None:
        # Offline: (period, timestamp, index). Causal: (period, index) -- the
        # event stream decides what is available; 33 adjacent event pairs
        # dataset-wide have a later index but an earlier timestamp.
        if event_order:
            key = lambda k: (records[k][0]["period"], records[k][0]["index"])  # noqa: E731
        else:
            key = lambda k: (records[k][0]["period"], timestamp_ms(records[k][0]["timestamp"]), records[k][0]["index"])  # noqa: E731
        order = sorted(range(len(records)), key=key)
        self.events = [records[k][0] for k in order]
        self.raws = [records[k][1] for k in order]
        self.orientations = [resolved_orientations[k] for k in order]
        self.problem_a = [problem_a_results[k] for k in order]
        self.n = len(self.events)
        self.periods = [e["period"] for e in self.events]
        self.times = [timestamp_ms(e["timestamp"]) for e in self.events]
        self.team_names = team_names
        self.team_bit = [0 if _team_name(e) == team_names[0] else 1 for e in self.events]
        self.name_of = {0: team_names[0], 1: team_names[1]}
        self.players = [list(raw.get("freeze_frame") or []) for raw in self.raws]
        self.valid = [[i for i, p in enumerate(ps) if isinstance(p, dict) and _is_pair(p.get("location"))]
                      for ps in self.players]
        self.raw_points = [
            _np.array([self.players[k][i]["location"] for i in self.valid[k]], dtype=float).reshape(-1, 2)
            for k in range(self.n)
        ]
        self.flags = [_np.array([self.players[k][i].get("teammate") is True for i in self.valid[k]], dtype=bool)
                      for k in range(self.n)]
        self.keepers = [[self.players[k][i].get("keeper") is True for i in self.valid[k]] for k in range(self.n)]
        self.keys = [_snapshot_key(raw) for raw in self.raws]
        self.has_geometry = [o != ORIENTATION_UNCERTAIN and len(p) > 0 for o, p in zip(self.orientations, self.raw_points)]
        self.own_points = [
            _np.column_stack([120.0 - p[:, 0], 80.0 - p[:, 1]]) if o == ORIENTATION_ROTATED_180 else p
            for o, p in zip(self.orientations, self.raw_points)
        ]
        self.absolute_points = [
            _np.column_stack([120.0 - p[:, 0], 80.0 - p[:, 1]])
            if self.team_bit[k] == 1 and self.periods[k] not in SHARED_FRAME_PERIODS else p
            for k, p in enumerate(self.own_points)
        ]
        self.events_by_id = {e["id"]: e for e in self.events}
        self.keeper_goal = [
            [_goal_team_bit(self.own_points[k][i][0], self.team_bit[k])
             if keeper and self.has_geometry[k] and self.periods[k] not in SHARED_FRAME_PERIODS else None
             for i, keeper in enumerate(self.keepers[k])]
            for k in range(self.n)
        ]

    def teams_for(self, k: int, record_view: int) -> list[int]:
        return [record_view if f else 1 - record_view for f in self.flags[k]]

    def neighbour(self, k: int, step: int, allowed, limit: int | None = None) -> int | None:
        """Nearest record before (step -1) or after (+1) k in the same period,
        <= CONTINUITY_MAX_DT_MS, at a different millisecond, with geometry and
        `allowed`. Never beyond index `limit` (causal mode). None in shootouts:
        keepers and takers alternate on one spot, continuity means nothing."""
        if self.periods[k] in SHARED_FRAME_PERIODS:
            return None
        j = k + step
        while 0 <= j < self.n and self.periods[j] == self.periods[k] \
                and abs(self.times[j] - self.times[k]) <= CONTINUITY_MAX_DT_MS:
            if limit is not None and j > limit:
                return None
            if self.times[j] != self.times[k] and self.has_geometry[j] and allowed(j):
                return j
            j += step
        return None

    def structural_edges(self) -> list[tuple[int, int, int, list[tuple[int, int]]]]:
        """(a, b, parity, deviating point pairs) for every exact same-scene
        relation a < b. Byte-identical records are linked whatever their size;
        other exact relations need EXACT_MIN_POINTS players."""
        edges = []
        for a in range(self.n):
            for b in range(a + 1, self.n):
                if self.periods[b] != self.periods[a] or self.times[b] - self.times[a] > STRUCTURE_MAX_DT_MS:
                    break
                if self.keys[a] == self.keys[b]:
                    edges.append((a, b, 0, []))
                    continue
                relation = exact_structural_relation(self.raw_points[a], self.raw_points[b])
                if relation is None:
                    continue
                _, pairs = relation
                equal = sum(self.flags[a][i] == self.flags[b][j] for i, j in pairs)
                if equal >= FLAG_RELATION_SHARE * len(pairs):
                    parity = 0
                elif len(pairs) - equal >= FLAG_RELATION_SHARE * len(pairs):
                    parity = 1
                else:
                    continue  # mixed flags on one scene: no view constraint (left to the detector)
                deviating = [(i, j) for i, j in pairs if (self.flags[a][i] != self.flags[b][j]) != bool(parity)]
                edges.append((a, b, parity, deviating))
        return edges

    def continuity_votes(self, rep: int, rep_parity: int, j: int, view_j: int, votes: dict) -> None:
        """Add one neighbour's verdict on the group's root view to `votes`
        (keys supports/leans/mixed). View 1 disagrees exactly where view 0
        agrees, so one pairing gives both counts."""
        pairs = unambiguous_pairs(self.absolute_points[rep], self.absolute_points[j])
        if len(pairs) < VIEW_MIN_PAIRS:
            return
        neighbour_teams = self.teams_for(j, view_j)
        mine = self.teams_for(rep, rep_parity)
        bad = {0: sum(mine[i] != neighbour_teams[jj] for i, jj in pairs)}
        bad[1] = len(pairs) - bad[0]
        for v in (0, 1):
            if bad[v] <= max(1, VIEW_MAX_DISAGREEMENT_SHARE * len(pairs)):
                votes["supports"].add(v)
            elif bad[v] <= VIEW_LEAN_SHARE * len(pairs) and bad[1 - v] >= (1 - VIEW_LEAN_SHARE) * len(pairs):
                votes["leans"].add(v)
        if not ({0, 1} & (votes["supports"] | votes["leans"])):
            votes["mixed"] = True

    def simultaneous_contradictions(self, team, status, linked, records):
        """(a, b, point in a, point in b) for every player whose team differs
        between two unlinked snapshots <= SIMULTANEOUS_MAX_DT_MS apart that pair
        him with a margin >= SIMULTANEOUS_MIN_MARGIN. One of the two records is
        wrong about him and nothing says which. For each b in `records` the
        partners are earlier records only, so passing every record yields each
        pair once and passing [b] alone stays causal."""
        for b in records:
            if not self.has_geometry[b] or self.periods[b] in SHARED_FRAME_PERIODS:
                continue
            a = b - 1
            while a >= 0 and self.periods[a] == self.periods[b] and self.times[b] - self.times[a] <= SIMULTANEOUS_MAX_DT_MS:
                if self.has_geometry[a] and not linked(a, b):
                    for i, j in unambiguous_pairs(self.absolute_points[a], self.absolute_points[b],
                                                  min_margin=SIMULTANEOUS_MIN_MARGIN):
                        if IDENTITY_UNRESOLVED not in (status[a][i], status[b][j]) and team[a][i] != team[b][j]:
                            yield a, b, i, j
                a -= 1

    def keeper_votes(self, members: list[int], parity_of) -> dict[int, int]:
        votes = {0: 0, 1: 0}
        for m in members:
            for i, goal in enumerate(self.keeper_goal[m]):
                if goal is None:
                    continue
                for v in (0, 1):
                    member_view = v ^ parity_of(m)
                    if (member_view if self.flags[m][i] else 1 - member_view) == goal:
                        votes[v] += 1
        return votes

    def owner_views(self, members: list[int], parity_of) -> set[int]:
        owners = set()
        for m in members:
            result = self.problem_a[m]
            if result.orientation == ORIENTATION_UNCERTAIN or result.confidence < DEFAULT_CONFIDENCE_THRESHOLD:
                continue
            owner = self.team_bit[m] if result.orientation != ORIENTATION_ROTATED_180 else 1 - self.team_bit[m]
            owners.add(owner ^ parity_of(m))
        return owners

    def assemble(self, view, unresolved_record, team, status, reason, agreed, goalkeeper_evidence,
                 component_of, decision, decision_evidence, legacy_groups) -> dict[str, FrameIdentity]:
        out: dict[str, FrameIdentity] = {}
        for k, event in enumerate(self.events):
            legacy_info = legacy_groups.get(event["id"])
            position = {i: p for p, i in enumerate(self.valid[k])}
            teams, spec, legacy, statuses, reasons, confirmed, keeper_check, notes = [], [], [], [], [], [], [], []
            for i, player in enumerate(self.players[k]):
                if not isinstance(player, dict):
                    spec.append(TEAM_IDENTITY_AMBIGUOUS)
                    legacy.append(TEAM_IDENTITY_AMBIGUOUS)
                else:
                    spec.append(resolve_absolute_team_for_point(player, event, None, self.events_by_id, self.team_names))
                    legacy.append(resolve_absolute_team_for_point(player, event, legacy_info, self.events_by_id, self.team_names))
                p = position.get(i)
                if p is None:  # no usable location: no evidence either way
                    unusable = not isinstance(player, dict)
                    teams.append(TEAM_IDENTITY_AMBIGUOUS if unusable or unresolved_record[k]
                                 else self.name_of[teams_for_point(player, view[k])])
                    statuses.append(IDENTITY_UNRESOLVED if unusable or unresolved_record[k]
                                    else status_for_view(view[k], self.team_bit[k]))
                    reasons.append(REASON_COPY_GROUP_UNDECIDED if unresolved_record[k] else "")
                    confirmed.append(False)
                    keeper_check.append("")
                    notes.append("")
                    continue
                evidence = goalkeeper_evidence.get((k, p))
                notes.append(
                    "" if evidence is None else
                    f"previous verified keeper {evidence['previous_event_id']} {evidence['gap_ms']} ms earlier, {evidence['distance']} yd"
                )
                statuses.append(status[k][p])
                reasons.append(reason[k][p])
                teams.append(TEAM_IDENTITY_AMBIGUOUS if status[k][p] == IDENTITY_UNRESOLVED else self.name_of[team[k][p]])
                goal = self.keeper_goal[k][p]
                # keeper_check compares the goal side with the reading BEFORE the
                # goalkeeper rule (component view), so it stays usable as evidence.
                view_team = self.teams_for(k, view[k])[p]
                keeper_check.append("" if goal is None else "consistent" if view_team == goal else "inconsistent")
                confirmed.append(status[k][p] != IDENTITY_UNRESOLVED and (agreed[k][p] or (goal is not None and team[k][p] == goal)))
            out[event["id"]] = FrameIdentity(
                event_id=event["id"], teams=tuple(teams), spec_teams=tuple(spec), legacy_teams=tuple(legacy),
                statuses=tuple(statuses), reasons=tuple(reasons), confirmed=tuple(confirmed),
                component=component_of[k], decision=decision[k], keeper_evidence=tuple(keeper_check),
                notes=tuple(notes), decision_evidence=decision_evidence[k],
            )
        return out


def _decide_group_view(votes: dict, keeper_votes: dict[int, int], owners: set[int], legacy_choice: int | None,
                       name_of: dict[int, str]) -> tuple[int | None, str, str]:
    """The evidence ladder for a group whose view constraints contradict its
    members' own teams (see the section comment). Returns (root view or
    None, decision, evidence summary)."""
    supports, leans, mixed = votes["supports"], votes["leans"], votes["mixed"]
    keeper = 0 if keeper_votes[0] and not keeper_votes[1] else 1 if keeper_votes[1] and not keeper_votes[0] else None
    owner = next(iter(owners)) if len(owners) == 1 else None

    def continuity_allows(v: int) -> bool:
        return 1 - v not in supports and 1 - v not in leans

    choice, how = None, DECISION_UNDECIDED
    if len(supports) == 1 and continuity_allows(next(iter(supports))):
        v = next(iter(supports))
        # A goalkeeper may contradict only if the coordinate owner
        # independently sides with continuity (two signals against one, and
        # the keeper flag has a documented quirk); that keeper point is then
        # handled by the goalkeeper rule.
        if keeper is None or keeper == v or owner == v:
            choice, how = v, DECISION_CONTINUITY
    elif not supports:
        if keeper is not None and owner == keeper and continuity_allows(keeper):
            choice, how = keeper, DECISION_KEEPER_AND_OWNER
        elif (legacy_choice is not None and continuity_allows(legacy_choice)
              and keeper in (None, legacy_choice) and owner in (None, legacy_choice)):
            choice, how = legacy_choice, DECISION_LEGACY
    label = lambda v: "-" if v is None else name_of[v]  # noqa: E731
    continuity_text = "/".join(name_of[v] for v in sorted(supports)) or (
        "lean " + "/".join(name_of[v] for v in sorted(leans)) if leans else "mixed" if mixed else "-")
    detail = (
        f"continuity={continuity_text};"
        f"keeper={'conflicting' if keeper_votes[0] and keeper_votes[1] else label(keeper)};"
        f"owner={'conflicting' if len(owners) > 1 else label(owner)};legacy={label(legacy_choice)}"
    )
    return choice, how, detail


def _transition_times(all_events: list[dict[str, Any]] | None) -> dict[int, list[int]]:
    times: dict[int, list[int]] = defaultdict(list)
    for event in all_events or []:
        if (event.get("type") or {}).get("name") in GOALKEEPER_TRANSITION_TYPES and isinstance(event.get("timestamp"), str):
            times[event.get("period")].append(timestamp_ms(event["timestamp"]))
    return times


def resolve_match_team_identity(
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    resolved_orientations: list[str],
    problem_a_results: list[OrientationResult],
    team_names: tuple[str, str],
    all_events: list[dict[str, Any]] | None = None,
    causal: bool = False,
) -> dict[str, FrameIdentity]:
    """Resolve every point's team for one match. See the section comment.

    `records` are (event, RAW 360 record) pairs in any order (sorted here, so
    both consumers get identical results); `resolved_orientations[k]` is the
    snapshot correction used to put record k into its own event's frame
    (UNCERTAIN = frame unknown: the record gives and receives no geometric
    evidence); `problem_a_results[k]` is the geometry classifier's output
    (coordinate-owner evidence). `all_events` (the full raw event list) is
    needed by the goalkeeper rule to rule out substitutions; without it the
    rule never fires.

    OFFLINE (default) uses evidence from both sides in time -- the best
    reconstruction of what happened, for visualization and data-quality
    analysis. With `causal=True`, record k is resolved only from records
    that precede it in the event stream (period, index) -- the information
    available when its event happens -- see _resolve_causal. Its
    `resolved_orientations` / `problem_a_results` must then be causal too
    (src/orientation_resolution.resolve_causal_orientations), otherwise
    future information enters through the geometry.
    """
    data = _MatchData(records, resolved_orientations, problem_a_results, team_names, event_order=causal)
    if causal:
        return _resolve_causal(data, all_events)
    return _resolve_offline(data, all_events)


def _resolve_offline(d: _MatchData, all_events) -> dict[str, FrameIdentity]:
    n = d.n
    legacy_groups = detect_shared_cross_team_snapshots(d.events, {e["id"]: d.raws[k] for k, e in enumerate(d.events)})

    # --- 2. structural links ---------------------------------------------
    uf = _Parity(n)
    deviations: list[tuple[int, int, int, int]] = []
    for a, b, parity, deviating in d.structural_edges():
        uf.union(a, b, parity)
        deviations.extend((a, i, b, j) for i, j in deviating)
    roots = [uf.find(k) for k in range(n)]
    members_of: dict[int, list[int]] = defaultdict(list)
    for k, (root, _) in enumerate(roots):
        members_of[root].append(k)
    component_of = [roots[k][0] if len(members_of[roots[k][0]]) > 1 else None for k in range(n)]
    parity_of = [roots[k][1] for k in range(n)]

    # --- 3. per-component view decision -----------------------------------
    view = list(d.team_bit)  # flag view per record (team bit whose `teammate=True` it means)
    unresolved_record = [False] * n
    decision = [""] * n
    decision_evidence = [""] * n
    conflicted: set[int] = set()
    for root, members in members_of.items():
        if len(members) < 2:
            continue
        if len({d.team_bit[m] ^ parity_of[m] for m in members}) == 1 and root not in uf.inconsistent:
            for m in members:
                decision[m] = DECISION_SPEC
            continue
        conflicted.add(root)

    for root in conflicted:
        members = sorted(members_of[root])
        if root in uf.inconsistent:
            choice, how, detail = None, DECISION_UNDECIDED, "inconsistent_links"
        else:
            geometric = [m for m in members if d.has_geometry[m]]
            member_set = set(members)
            clean = lambda j: component_of[j] not in conflicted and j not in member_set  # noqa: E731
            votes = {"supports": set(), "leans": set(), "mixed": False}
            if geometric:
                for rep, step in ((geometric[0], -1), (geometric[-1], 1)):
                    j = d.neighbour(rep, step, clean)
                    if j is not None:
                        d.continuity_votes(rep, parity_of[rep], j, view[j], votes)
            legacy = set()
            for m in members:
                info = legacy_groups.get(d.events[m]["id"])
                if info is not None and info.source_event_id is not None:
                    source = next(s for s in members if d.events[s]["id"] == info.source_event_id)
                    legacy.add(d.team_bit[source] ^ parity_of[source])
            choice, how, detail = _decide_group_view(
                votes, d.keeper_votes(geometric, parity_of.__getitem__), d.owner_views(members, parity_of.__getitem__),
                next(iter(legacy)) if len(legacy) == 1 else None, d.name_of,
            )
        for m in members:
            decision[m] = how
            decision_evidence[m] = detail
            if choice is None:
                unresolved_record[m] = True
            else:
                view[m] = choice ^ parity_of[m]

    # --- per-point state ----------------------------------------------------
    team = [d.teams_for(k, view[k]) for k in range(n)]           # bit per valid point
    status = [[status_for_view(view[k], d.team_bit[k])] * len(d.valid[k]) for k in range(n)]
    reason = [[""] * len(d.valid[k]) for k in range(n)]
    for k in range(n):
        if unresolved_record[k]:
            status[k] = [IDENTITY_UNRESOLVED] * len(d.valid[k])
            reason[k] = [REASON_COPY_GROUP_UNDECIDED] * len(d.valid[k])
    for a, i, b, j in deviations:
        for k, p in ((a, i), (b, j)):
            if status[k][p] != IDENTITY_UNRESOLVED:
                status[k][p] = IDENTITY_UNRESOLVED
                reason[k][p] = REASON_COPY_FLAG_DEVIATION

    def definite(k: int, i: int) -> bool:
        return status[k][i] != IDENTITY_UNRESOLVED

    # --- 4. crowding-safe contradiction detector ----------------------------
    def detect() -> tuple[list[list[bool]], list[list[bool]], list[bool]]:
        contradicted = [[False] * len(d.valid[k]) for k in range(n)]
        agreed = [[False] * len(d.valid[k]) for k in range(n)]
        suspect = [False] * n
        for k in range(n):
            if not d.has_geometry[k]:
                continue
            other = lambda j: component_of[j] is None or component_of[j] != component_of[k]  # noqa: E731
            found = [d.neighbour(k, step, other) for step in (-1, 1)]
            sides = [
                {} if j is None else {i: (j, jj) for i, jj in unambiguous_pairs(d.absolute_points[k], d.absolute_points[j])}
                for j in found
            ]
            # Two copies of one snapshot are one piece of evidence, not two.
            independent = None in found or component_of[found[0]] is None or component_of[found[0]] != component_of[found[1]]
            evaluable = bad = 0
            for i in range(len(d.valid[k])):
                if not definite(k, i):
                    continue
                partners = [side[i] for side in sides if i in side and definite(*side[i])]
                if any(team[j][jj] == team[k][i] for j, jj in partners):
                    agreed[k][i] = True
                if len(partners) == 2 and independent:
                    evaluable += 1
                    t1, t2 = (team[j][jj] for j, jj in partners)
                    if t1 == t2 != team[k][i]:
                        contradicted[k][i] = True
                        bad += 1
            suspect[k] = bad >= RECORD_CONTRADICTION_MIN_POINTS and bad >= RECORD_CONTRADICTION_SHARE * evaluable
        return contradicted, agreed, suspect

    contradicted, _, suspect = detect()

    # --- 5. goalkeeper-only continuity rule ---------------------------------
    transition_times = _transition_times(all_events)

    def verified_keeper(k: int, i: int) -> bool:
        """Keeper identity backed by two independent signals (its own raw flag
        read via an accepted view AND its goal side), not inherited."""
        return (
            status[k][i] in (IDENTITY_SPEC, IDENTITY_CORRECTED_VIEW)
            and d.keeper_goal[k][i] is not None and team[k][i] == d.keeper_goal[k][i]
            and not contradicted[k][i] and not suspect[k]
        )

    linked = lambda j, k: component_of[j] is not None and component_of[j] == component_of[k]  # noqa: E731
    goalkeeper_evidence: dict[tuple[int, int], dict[str, Any]] = {}
    for k in range(n):
        for i in range(len(d.valid[k])):
            goal = d.keeper_goal[k][i]
            if goal is None or not definite(k, i) or team[k][i] == goal:
                continue
            if all_events is not None:
                rest_bad = sum(contradicted[k][p] for p in range(len(d.valid[k])) if p != i)
                rest_ok = not suspect[k] and rest_bad <= max(1, VIEW_MAX_DISAGREEMENT_SHARE * len(d.valid[k]))
                evidence = _goalkeeper_correction(d, k, i, goal, team, linked, verified_keeper, transition_times) if rest_ok else None
                if evidence is not None:
                    goalkeeper_evidence[(k, i)] = evidence
                    continue
            status[k][i] = IDENTITY_UNRESOLVED
            reason[k][i] = REASON_KEEPER_GOAL_SIDE
    for (k, i) in goalkeeper_evidence:
        team[k][i] = d.keeper_goal[k][i]
        status[k][i] = IDENTITY_CORRECTED_GOALKEEPER

    # --- 6. simultaneous contradictions ------------------------------------
    for a, b, i, j in d.simultaneous_contradictions(team, status, lambda x, y: linked(x, y), range(n)):
        for k, p in ((a, i), (b, j)):
            if definite(k, p):
                status[k][p] = IDENTITY_UNRESOLVED
                reason[k][p] = REASON_SIMULTANEOUS_CONTRADICTION

    # --- final detector pass on the corrected state -------------------------
    contradicted, agreed, suspect = detect()
    for k in range(n):
        if suspect[k]:
            for i in range(len(d.valid[k])):
                if definite(k, i):
                    status[k][i] = IDENTITY_UNRESOLVED
                    reason[k][i] = REASON_RECORD_CONTRADICTED
            continue
        for i in range(len(d.valid[k])):
            if contradicted[k][i] and definite(k, i):
                corrected_keeper = status[k][i] == IDENTITY_CORRECTED_GOALKEEPER
                status[k][i] = IDENTITY_UNRESOLVED
                reason[k][i] = REASON_GOALKEEPER_CORRECTION_CONTRADICTED if corrected_keeper else REASON_POINT_CONTRADICTED

    return d.assemble(view, unresolved_record, team, status, reason, agreed, goalkeeper_evidence,
                      component_of, decision, decision_evidence, legacy_groups)


def _goalkeeper_correction(d: _MatchData, k: int, i: int, goal: int, team, linked, verified, transition_times):
    """The goalkeeper-only rule for keeper point i of record k whose team
    contradicts its goal side: evidence dict if the verified previous keeper
    allows the correction, else None."""
    matched = _previous_verified_keeper(
        k, i, d.periods, d.times, linked, d.keepers, d.absolute_points, d.has_geometry, verified
    )
    if matched is None:
        return None
    p, q, gap = matched
    if any(d.times[p] <= t <= d.times[k] for t in transition_times.get(d.periods[k], [])) or team[p][q] != goal:
        return None
    return {"previous_event_id": d.events[p]["id"], "gap_ms": gap,
            "distance": round(float(_np.linalg.norm(d.absolute_points[k][i] - d.absolute_points[p][q])), 2)}


def _resolve_causal(d: _MatchData, all_events) -> dict[str, FrameIdentity]:
    """Prediction-time resolution: record k sees only records 0..k in
    event-stream order (period, index) -- the current event's own snapshot and
    everything that happened before it, never the events that follow (which
    include the event a model has to predict). Same rules as offline, with
    what cannot be evaluated without the future left out:

      - a group's view is decided when each member arrives, from the members
        and neighbours seen so far (continuity: the snapshot before the
        group's earliest member only); an earlier member's view is never
        revised by a later one;
      - flag deviations mark only the later record of a pair;
      - the two-neighbour contradiction detector needs the next snapshot and
        is not run; the goalkeeper rule's "rest of the frame is consistent"
        is checked against the previous snapshot only.
    """
    n = d.n
    uf = _Parity(n)
    members_of: dict[int, list[int]] = {k: [k] for k in range(n)}
    edges_by_b: dict[int, list] = defaultdict(list)
    for a, b, parity, deviating in d.structural_edges():
        edges_by_b[b].append((a, parity, deviating))
    key_members: dict[tuple, list[int]] = defaultdict(list)
    view = list(d.team_bit)
    unresolved_record = [False] * n
    decision = [""] * n
    decision_evidence = [""] * n
    component_of: list[int | None] = [None] * n
    team: list[list[int]] = [[] for _ in range(n)]
    status: list[list[str]] = [[] for _ in range(n)]
    reason: list[list[str]] = [[] for _ in range(n)]
    agreed = [[False] * len(d.valid[k]) for k in range(n)]
    goalkeeper_evidence: dict[tuple[int, int], dict[str, Any]] = {}
    transition_times = _transition_times(all_events)
    conflicted_records: set[int] = set()  # records whose own decision was a conflicted-group decision

    def verified_keeper(k: int, i: int) -> bool:
        return (status[k][i] in (IDENTITY_SPEC, IDENTITY_CORRECTED_VIEW)
                and d.keeper_goal[k][i] is not None and team[k][i] == d.keeper_goal[k][i])

    for b in range(n):
        deviating_points = set()
        for a, parity, deviating in edges_by_b[b]:
            ra, rb = uf.find(a)[0], uf.find(b)[0]
            uf.union(a, b, parity)
            root = uf.find(b)[0]
            if ra != rb:
                merged = sorted(set(members_of.pop(ra, [a])) | set(members_of.pop(rb, [b])))
                members_of[root] = merged
            deviating_points.update(j for _, j in deviating)
        key_members[(d.periods[b], d.keys[b])].append(b)
        root, _ = uf.find(b)
        members = members_of.get(root, [b])
        parity_now = lambda m: uf.find(m)[1]  # noqa: E731
        if len(members) > 1:
            component_of[b] = root
            consistent = len({d.team_bit[m] ^ parity_now(m) for m in members}) == 1 and root not in uf.inconsistent
            if consistent:
                decision[b] = DECISION_SPEC
            elif root in uf.inconsistent:
                decision[b], decision_evidence[b], unresolved_record[b] = DECISION_UNDECIDED, "inconsistent_links", True
                conflicted_records.add(b)
            else:
                conflicted_records.add(b)
                geometric = [m for m in members if d.has_geometry[m]]
                member_set = set(members)
                clean = lambda j: j not in conflicted_records and j not in member_set  # noqa: E731
                votes = {"supports": set(), "leans": set(), "mixed": False}
                if geometric:
                    j = d.neighbour(geometric[0], -1, clean)
                    if j is not None:
                        d.continuity_votes(geometric[0], parity_now(geometric[0]), j, view[j], votes)
                legacy = set()
                for m in members:
                    group = [s for s in key_members[(d.periods[m], d.keys[m])] if s <= b]
                    teams = {d.team_bit[s] for s in group}
                    if len(teams) < 2:
                        continue
                    sources = {d.team_bit[s]: s for s in group
                               if d.problem_a[s].orientation == ORIENTATION_NORMAL
                               and d.problem_a[s].confidence >= DEFAULT_CONFIDENCE_THRESHOLD}
                    if len(sources) == 1:
                        source = next(iter(sources.values()))
                        legacy.add(d.team_bit[source] ^ parity_now(source))
                choice, how, detail = _decide_group_view(
                    votes, d.keeper_votes(geometric, parity_now), d.owner_views(members, parity_now),
                    next(iter(legacy)) if len(legacy) == 1 else None, d.name_of,
                )
                decision[b], decision_evidence[b] = how, detail
                if choice is None:
                    unresolved_record[b] = True
                else:
                    view[b] = choice ^ parity_now(b)

        team[b] = d.teams_for(b, view[b])
        if unresolved_record[b]:
            status[b] = [IDENTITY_UNRESOLVED] * len(d.valid[b])
            reason[b] = [REASON_COPY_GROUP_UNDECIDED] * len(d.valid[b])
        else:
            status[b] = [status_for_view(view[b], d.team_bit[b])] * len(d.valid[b])
            reason[b] = [""] * len(d.valid[b])
            for p in deviating_points:
                status[b][p], reason[b][p] = IDENTITY_UNRESOLVED, REASON_COPY_FLAG_DEVIATION

        # Agreement with the previous snapshot (confirmation, and the
        # goalkeeper rule's "rest of the frame" check).
        not_linked = lambda j: component_of[b] is None or uf.find(j)[0] != uf.find(b)[0]  # noqa: E731
        previous = d.neighbour(b, -1, not_linked) if d.has_geometry[b] else None
        rest_disagreeing: dict[int, int] = {}
        pairs = unambiguous_pairs(d.absolute_points[b], d.absolute_points[previous]) if previous is not None else []
        for i, jj in pairs:
            if status[b][i] == IDENTITY_UNRESOLVED or status[previous][jj] == IDENTITY_UNRESOLVED:
                continue
            same = team[previous][jj] == team[b][i]
            agreed[b][i] = same
            rest_disagreeing[i] = 0 if same else 1

        linked = lambda j, k: uf.find(j)[0] == uf.find(k)[0] and component_of[k] is not None  # noqa: E731
        for i in range(len(d.valid[b])):
            goal = d.keeper_goal[b][i]
            if goal is None or status[b][i] == IDENTITY_UNRESOLVED or team[b][i] == goal:
                continue
            if all_events is not None:
                rest_bad = sum(v for p, v in rest_disagreeing.items() if p != i)
                if rest_bad <= max(1, VIEW_MAX_DISAGREEMENT_SHARE * len(d.valid[b])):
                    evidence = _goalkeeper_correction(d, b, i, goal, team, linked, verified_keeper, transition_times)
                    if evidence is not None:
                        goalkeeper_evidence[(b, i)] = evidence
                        team[b][i] = goal
                        status[b][i] = IDENTITY_CORRECTED_GOALKEEPER
                        continue
            status[b][i], reason[b][i] = IDENTITY_UNRESOLVED, REASON_KEEPER_GOAL_SIDE

        for _, _, _, j in d.simultaneous_contradictions(team, status, linked, [b]):
            if status[b][j] != IDENTITY_UNRESOLVED:
                status[b][j], reason[b][j] = IDENTITY_UNRESOLVED, REASON_SIMULTANEOUS_CONTRADICTION

    legacy_groups = detect_shared_cross_team_snapshots(d.events, {e["id"]: d.raws[k] for k, e in enumerate(d.events)})
    return d.assemble(view, unresolved_record, team, status, reason, agreed, goalkeeper_evidence,
                      component_of, decision, decision_evidence, legacy_groups)


def teams_for_point(player: dict[str, Any], record_view: int) -> int:
    return record_view if player.get("teammate") is True else 1 - record_view


def status_for_view(record_view: int, own_bit: int) -> str:
    return IDENTITY_CORRECTED_VIEW if record_view != own_bit else IDENTITY_SPEC


def _is_pair(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def _goal_team_bit(own_frame_x: float, own_team_bit: int) -> int | None:
    """Team defending the goal a keeper stands near, from the record's own
    frame (the event's team attacks toward x = 120, so defends x = 0)."""
    if own_frame_x <= KEEPER_GOAL_ZONE:
        return own_team_bit
    if own_frame_x >= 120.0 - KEEPER_GOAL_ZONE:
        return 1 - own_team_bit
    return None


def _previous_verified_keeper(k, i, periods, times, linked, keepers, absolute_points, has_geometry, verified):
    """Nearest earlier keeper observation (same period, <= GOALKEEPER_MAX_GAP_MS,
    not a structural copy of record k, not the same millisecond) that is the
    same keeper -- within a movement allowance and at the same goal end --
    and whose identity is independently verified. Returns (record, point,
    gap_ms) or None."""
    here = absolute_points[k][i]
    j = k - 1
    while j >= 0 and periods[j] == periods[k] and times[k] - times[j] <= GOALKEEPER_MAX_GAP_MS:
        gap = times[k] - times[j]
        if gap > 0 and has_geometry[j] and not linked(j, k):
            allowance = min(GOALKEEPER_MAX_DISTANCE, GOALKEEPER_BASE_DISTANCE + GOALKEEPER_SPEED_YD_PER_S * gap / 1000)
            for q, is_keeper in enumerate(keepers[j]):
                there = absolute_points[j][q]
                if not is_keeper or (there[0] < 60) != (here[0] < 60):
                    continue
                if float(_np.linalg.norm(here - there)) <= allowance and verified(j, q):
                    return j, q, gap
        j -= 1
    return None
