"""Single source of truth for detecting and correcting 180-degree-rotated
StatsBomb 360 freeze frames.

Background
----------
Forensic analysis of match 3902240 (and reproduced on matches 3857254 and
3788741, three different competitions) showed that a subset of StatsBomb 360
freeze-frame snapshots are recorded reflected through the pitch center,
``(x, y) -> (120 - x, 80 - y)``, relative to the event's own reliable
``location`` field. This affects specific "reactive" event types (a player
losing a duel: e.g. Dispossessed, Foul Won, Dribbled Past) far more often
than "active" ones (Pass, Carry, Pressure, Shot), but the affected fraction
varies continuously by type (e.g. Duel, Dribble, Ball Receipt* are a mix) --
so event type alone is not a reliable predictor and is intentionally never
used as a direct input to the decision below.

This module classifies each event/360-frame pair as one of:

    NORMAL       - freeze_frame is already consistent with event.location
    ROTATED_180  - freeze_frame must be reflected through the pitch center
    UNCERTAIN    - evidence is insufficient or contradictory; never guess

and provides a matching correction that rotates the *entire* snapshot
(every freeze_frame player, including actor/keeper flags, and the
visible_area polygon) consistently -- never a single point in isolation.

Both src/build_statsbomb_360_dataset.py and src/animate_statsbomb_360_match.py
import this module rather than re-implementing the classifier, so any future
refinement only needs to happen here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

ORIENTATION_NORMAL = "NORMAL"
ORIENTATION_ROTATED_180 = "ROTATED_180"
ORIENTATION_UNCERTAIN = "UNCERTAIN"

# Chosen from the population separation observed during validation: correctly
# matched points land within ~0-8 units once the right hypothesis is picked,
# while genuinely mismatched orientations remain 30+ units apart. The 0.45
# confidence threshold is the more conservative value adopted after the
# false-positive review (it removes the entire 0.35-0.45 borderline band of
# ROTATED_180 calls seen during validation, at negligible cost in coverage).
CONFIDENCE_THRESHOLD = 0.45
MAX_ACCEPTABLE_DISTANCE = 15.0

# Temporal-refinement parameters (see resolve_uncertain_with_neighbors below).
NEIGHBOR_ANCHOR_WINDOW = 5      # look up to this many usable frames away
NEIGHBOR_ANCHOR_MAX_DT = 3.0    # seconds


Point = tuple[float, float]


def mirror_point(point: Point) -> Point:
    """180-degree pitch rotation: reflect through the pitch center (60, 40)."""
    x, y = point
    return (PITCH_LENGTH - x, PITCH_WIDTH - y)


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _centroid(points: list[Point]) -> Point:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def _point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def _coordinate_pair(value: Any) -> Point | None:
    """Return a numeric coordinate pair, or None when it is unavailable."""
    if not isinstance(value, list) or len(value) != 2:
        return None
    x, y = value
    if isinstance(x, bool) or isinstance(y, bool):
        return None
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return float(x), float(y)


def _visible_area_points(value: Any) -> list[Point] | None:
    """Interpret a flat visible_area sequence as ordered polygon vertices."""
    if not isinstance(value, list) or len(value) < 6 or len(value) % 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return [(float(value[i]), float(value[i + 1])) for i in range(0, len(value), 2)]


@dataclass(frozen=True)
class Vote:
    """One signal's contribution to the orientation decision."""

    name: str
    vote: float                    # in [-1, +1]; + favors NORMAL, - favors ROTATED_180
    weight: float
    distance_normal: float | None = None
    distance_rotated: float | None = None


@dataclass(frozen=True)
class OrientationResult:
    """The classifier's decision for one event/360-frame pair."""

    orientation: str
    confidence: float
    best_distance: float
    votes: tuple[Vote, ...] = field(default_factory=tuple)
    resolved_by: str = "geometry"  # "geometry" or "temporal_neighbor"


@dataclass(frozen=True)
class FrameGeometry:
    """Extracted, orientation-agnostic geometry for one event/360-frame pair."""

    event_location: Point | None
    points: tuple[Point, ...]
    actor_location: Point | None
    polygon: tuple[Point, ...] | None


def extract_geometry(event: dict[str, Any], frame_360: dict[str, Any]) -> FrameGeometry:
    """Pull out the coordinates the classifier needs, without judging them."""
    event_location = _coordinate_pair(event.get("location"))
    freeze_frame = frame_360.get("freeze_frame") or []
    points = tuple(
        point
        for player in freeze_frame
        if isinstance(player, dict)
        and (point := _coordinate_pair(player.get("location"))) is not None
    )
    actor = next(
        (player for player in freeze_frame if isinstance(player, dict) and player.get("actor") is True),
        None,
    )
    actor_location = _coordinate_pair(actor.get("location")) if actor else None
    polygon = _visible_area_points(frame_360.get("visible_area"))
    return FrameGeometry(
        event_location, points, actor_location, tuple(polygon) if polygon else None
    )


def _geometry_votes(geometry: FrameGeometry) -> list[Vote]:
    """Signals 1-4: purely spatial, derived only from this event's own record."""
    votes: list[Vote] = []
    loc = geometry.event_location
    if loc is None:
        return votes

    if geometry.actor_location is not None:
        d0 = _distance(loc, geometry.actor_location)
        d1 = _distance(loc, mirror_point(geometry.actor_location))
        votes.append(Vote("actor", (d1 - d0) / (d0 + d1 + 1e-9), 0.40, d0, d1))

    if geometry.points:
        d0 = min(_distance(loc, p) for p in geometry.points)
        d1 = min(_distance(loc, mirror_point(p)) for p in geometry.points)
        votes.append(Vote("nearest_point", (d1 - d0) / (d0 + d1 + 1e-9), 0.20, d0, d1))

        c = _centroid(list(geometry.points))
        d0c = _distance(loc, c)
        d1c = _distance(loc, mirror_point(c))
        votes.append(Vote("centroid", (d1c - d0c) / (d0c + d1c + 1e-9), 0.15, d0c, d1c))

    if geometry.polygon:
        inside_normal = _point_in_polygon(loc, list(geometry.polygon))
        inside_rotated = _point_in_polygon(mirror_point(loc), list(geometry.polygon))
        if inside_normal != inside_rotated:
            votes.append(Vote("visible_area", 1.0 if inside_normal else -1.0, 0.15))

    return votes


def _decide(geometry: FrameGeometry, votes: list[Vote]) -> OrientationResult:
    if geometry.event_location is None or not votes:
        return OrientationResult(ORIENTATION_UNCERTAIN, 0.0, float("inf"), tuple(votes))

    total_weight = sum(vote.weight for vote in votes)
    overall = sum(vote.vote * vote.weight for vote in votes) / total_weight
    confidence = abs(overall)

    primary = next((vote for vote in votes if vote.name == "actor"), None)
    if primary is None:
        primary = next((vote for vote in votes if vote.name == "nearest_point"), None)
    if primary is not None and primary.distance_normal is not None:
        best_distance = primary.distance_normal if overall >= 0 else primary.distance_rotated
    else:
        best_distance = float("inf")

    if confidence < CONFIDENCE_THRESHOLD or best_distance > MAX_ACCEPTABLE_DISTANCE:
        orientation = ORIENTATION_UNCERTAIN
    elif overall > 0:
        orientation = ORIENTATION_NORMAL
    else:
        orientation = ORIENTATION_ROTATED_180

    return OrientationResult(orientation, confidence, best_distance, tuple(votes))


def classify_orientation(event: dict[str, Any], frame_360: dict[str, Any]) -> OrientationResult:
    """Classify one event/360-frame pair as NORMAL, ROTATED_180, or UNCERTAIN.

    Geometry-only (pass 1 in the forensic analysis): uses only this event's
    own record (event.location, freeze_frame players, actor, visible_area).
    Event type is never consulted. See ``resolve_uncertain_with_neighbors``
    for the optional temporal-refinement pass over a whole match.
    """
    geometry = extract_geometry(event, frame_360)
    return _decide(geometry, _geometry_votes(geometry))


def resolve_uncertain_with_neighbors(
    ordered_events: list[dict[str, Any]],
    ordered_frames: list[dict[str, Any]],
    ordered_timestamps: list[float],
    ordered_periods: list[int],
    results: list[OrientationResult],
) -> list[OrientationResult]:
    """Temporal refinement: resolve only still-UNCERTAIN cases.

    For each UNCERTAIN result, look at nearby (<= NEIGHBOR_ANCHOR_MAX_DT
    seconds, same period) results that are already confidently NORMAL, and
    use the nearest one in time as one additional, time-decayed vote. This
    can only pull an event out of UNCERTAIN; it never overturns an existing
    NORMAL or ROTATED_180 decision, and it never anchors on a ROTATED_180 or
    UNCERTAIN neighbor, so there is no circular reasoning.

    All four input lists must be the same length and share index order (one
    entry per usable event/360-frame pair, already chronologically sorted).
    """
    length = len(ordered_events)
    if not (len(ordered_frames) == len(ordered_timestamps) == len(ordered_periods) == len(results) == length):
        raise ValueError("All inputs to resolve_uncertain_with_neighbors must have equal length.")

    geometries = [
        extract_geometry(ordered_events[i], ordered_frames[i]) for i in range(length)
    ]

    def anchor_point(geometry: FrameGeometry) -> Point | None:
        if geometry.actor_location is not None:
            return geometry.actor_location
        if geometry.points:
            return _centroid(list(geometry.points))
        return None

    refined = list(results)
    for i in range(length):
        if refined[i].orientation != ORIENTATION_UNCERTAIN:
            continue

        geometry = geometries[i]
        current_point = anchor_point(geometry)
        if current_point is None or geometry.event_location is None:
            continue

        best_dt = None
        anchor_vote: Vote | None = None
        lo = max(0, i - NEIGHBOR_ANCHOR_WINDOW)
        hi = min(length, i + NEIGHBOR_ANCHOR_WINDOW + 1)
        for j in range(lo, hi):
            if j == i or ordered_periods[j] != ordered_periods[i]:
                continue
            # Only a confidently-NORMAL neighbor may serve as an anchor:
            # never ROTATED_180, never UNCERTAIN (no circular reasoning).
            if results[j].orientation != ORIENTATION_NORMAL:
                continue
            dt = abs(ordered_timestamps[j] - ordered_timestamps[i])
            if dt > NEIGHBOR_ANCHOR_MAX_DT:
                continue
            neighbor_point = anchor_point(geometries[j])
            if neighbor_point is None:
                continue
            weight = 0.10 * max(0.0, 1 - dt / NEIGHBOR_ANCHOR_MAX_DT)
            if weight <= 0:
                continue
            if best_dt is None or dt < best_dt:
                d0 = _distance(current_point, neighbor_point)
                d1 = _distance(mirror_point(current_point), neighbor_point)
                best_dt = dt
                anchor_vote = Vote(
                    "neighbor_anchor", (d1 - d0) / (d0 + d1 + 1e-9), weight, d0, d1
                )

        if anchor_vote is None:
            continue

        votes = _geometry_votes(geometry)
        votes.append(anchor_vote)
        result = _decide(geometry, votes)
        if result.orientation != ORIENTATION_UNCERTAIN:
            refined[i] = OrientationResult(
                result.orientation, result.confidence, result.best_distance,
                result.votes, resolved_by="temporal_neighbor",
            )

    return refined


def classify_orientations_for_match(
    ordered_events: list[dict[str, Any]],
    ordered_frames: list[dict[str, Any]],
    ordered_timestamps: list[float],
    ordered_periods: list[int],
) -> list[OrientationResult]:
    """Run the full two-pass classification for one match's event/360-frame pairs.

    This is the one place that ties ``classify_orientation`` (geometry) and
    ``resolve_uncertain_with_neighbors`` (temporal refinement) together, so
    every consumer (dataset builder, animation) runs the identical pipeline
    instead of re-implementing the two-pass orchestration itself.

    All four inputs must be the same length and already sorted chronologically
    (one entry per usable event/360-frame pair).
    """
    pass1_results = [
        classify_orientation(ordered_events[i], ordered_frames[i])
        for i in range(len(ordered_events))
    ]
    return resolve_uncertain_with_neighbors(
        ordered_events, ordered_frames, ordered_timestamps, ordered_periods, pass1_results
    )


def _rotate_player(player: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of one freeze_frame player with its location mirrored.

    All flags (teammate/actor/keeper) are preserved unchanged -- only the
    coordinate is rotated, so identity/team/actor attribution is untouched.
    """
    rotated = dict(player)
    location = _coordinate_pair(player.get("location"))
    if location is not None:
        rotated["location"] = list(mirror_point(location))
    return rotated


def apply_correction(frame_360: dict[str, Any], orientation: str) -> dict[str, Any]:
    """Return a corrected COPY of a 360 frame record; never mutates the input.

    ROTATED_180: every freeze_frame player (actor, keeper, and everyone
    else) and the whole visible_area polygon are mirrored together, so the
    entire snapshot stays internally consistent -- never a single point.
    NORMAL and UNCERTAIN: returned unchanged (as a shallow copy), matching
    the policy of never forcing a correction without confident evidence.
    """
    if orientation not in (ORIENTATION_NORMAL, ORIENTATION_ROTATED_180, ORIENTATION_UNCERTAIN):
        raise ValueError(f"Unknown orientation: {orientation!r}")

    if orientation != ORIENTATION_ROTATED_180:
        return dict(frame_360)

    corrected = dict(frame_360)
    freeze_frame = frame_360.get("freeze_frame")
    if isinstance(freeze_frame, list):
        corrected["freeze_frame"] = [
            _rotate_player(player) if isinstance(player, dict) else player
            for player in freeze_frame
        ]

    visible_area = frame_360.get("visible_area")
    polygon = _visible_area_points(visible_area)
    if polygon is not None:
        rotated_flat: list[float] = []
        for x, y in polygon:
            rx, ry = mirror_point((x, y))
            rotated_flat.extend((rx, ry))
        corrected["visible_area"] = rotated_flat

    return corrected
