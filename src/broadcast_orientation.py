"""Broadcast-camera-side orientation inference (Problem D) for StatsBomb 360.

Independent from, and read-only with respect to:
  - src/orientation_correction.py (Problem A: is one event's own snapshot
    consistent with its own event.location?)
  - src/team_normalization.py (Problem B: put both teams on one scale)
  - src/team_identity.py (Problem C: which absolute team is each point?)

Problem D answers the question Problem B deliberately left open: WHICH of
the two possible shared coordinate frames is the one a television viewer
sees? Problem B only fixes the RELATIVE transform between the two teams
(one of them must be reflected through the pitch centre); the remaining
choice -- which team stays unrotated -- is a global gauge freedom that the
event coordinates alone cannot settle. Until now it was settled by an
arbitrary rule (first team to have a 360 event) or by hand.

The physical information that settles it is the visible_area polygon.
StatsBomb derives 360 snapshots from the broadcast video, so visible_area is
the ground footprint of the broadcast camera's image, expressed in the same
team-relative frame as the event that owns it. For a pinhole camera without
roll, the left and right image borders project to two ground lines that
meet at a point on the CAMERA'S side of the footprint (behind the camera's
foot point); the near and far image borders project to (nearly) parallel
lines. The polygon is clipped to the pitch, but the clipped pieces lie on
the pitch boundary and the surviving real edges keep their original slope,
so the convergence point ("apex") is recoverable from the two non-adjacent
real edges with the largest angle between them. Its y-offset from the
footprint's centroid says which touchline the camera stands behind, in that
event's own frame.

Because the main broadcast camera does not move during a match, the camera
touchline is a fixed physical fact. Choosing the presentation convention
"camera touchline at the bottom of the screen" (the viewer stands where the
camera stands, looking across the pitch) therefore determines, for every
(period, team) group of events, whether that group must be rotated by 180
degrees -- without choosing any reference team. The per-period reference
team becomes an OUTPUT of this inference, not an input. A rotation fixes
left/right and near/far together; no reflection is ever needed, because
StatsBomb coordinates and this project's display (y inverted, y=0 at the
top) have the same handedness (verified: Left Backs average y ~ 11-13,
Right Backs y ~ 69-72 in their own team's frame, i.e. y=0 is the
attacker's left, exactly as StatsBomb documents).

Two frame models, measured not assumed:
  - OPPOSITE (periods 1-4): the two teams attack opposite physical goals,
    so their raw frames differ by a 180-degree rotation and show the camera
    on opposite touchlines. One team is rotated (Problem B's transform).
  - SHARED (period 5, penalty shootout): both teams kick at the SAME goal,
    and StatsBomb records both in one frame -- every shootout in the
    dataset shows both teams' camera on the same touchline. Rotating one
    team here (as Problem B alone does) would put half the penalties at the
    wrong end; both teams are rotated together or not at all.

What this does NOT establish: that the visible_area camera is the main
broadcast camera for every snapshot (a reverse-angle or behind-goal shot
would vote for the wrong side or for no side -- the per-period majority and
agreement rate expose this), and that a given broadcaster did not mirror
its feed. Results are therefore reported as INFERRED, never as verified;
independently verified calibrations in
configs/physical_orientation_calibration.json always take priority.

Each period is inferred on its own evidence. The halftime end swap (periods
1 -> 2) and the extra-time-half swap (3 -> 4) are then CHECKED against the
data rather than assumed; no assumption is made about period 3 relative to
period 2 (extra time may start after a new coin toss).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

CAMERA_SIDE_LOW_Y = "LOW_Y"    # camera stands behind the y=0 touchline
CAMERA_SIDE_HIGH_Y = "HIGH_Y"  # camera stands behind the y=80 touchline
CAMERA_SIDE_UNKNOWN = "UNKNOWN"

# Presentation convention: this project's display draws y=0 at the top and
# y=80 at the bottom (FIXED_Y_LIMITS in the renderer), so "camera touchline
# at the bottom of the screen" means the camera sits at HIGH_Y.
DISPLAY_CAMERA_SIDE = CAMERA_SIDE_HIGH_Y

FRAME_MODEL_OPPOSITE = "OPPOSITE"
FRAME_MODEL_SHARED = "SHARED"
SHARED_FRAME_PERIODS = frozenset({5})

BROADCAST_STATUS_INFERRED = "INFERRED"
BROADCAST_STATUS_AMBIGUOUS = "AMBIGUOUS"
BROADCAST_STATUS_CONFLICTING = "CONFLICTING"
BROADCAST_STATUS_NO_EVIDENCE = "NO_EVIDENCE"

_BOUNDARY_TOLERANCE = 0.05
_MINIMUM_EDGE_LENGTH = 1.0
# Lateral (left/right image border) edges of a broadcast footprint converge
# at tens of degrees; near/far edges are parallel up to camera roll. Pairs
# below this angle are treated as near/far, never as the lateral pair.
MINIMUM_CONVERGENCE_DEGREES = 8.0
# The apex direction must point mostly across the pitch (toward a
# touchline); a footprint converging toward a goal line says nothing about
# which touchline the camera is on.
MINIMUM_TOUCHLINE_RATIO = 0.5

# A period is decided only with enough, and sufficiently unanimous, votes.
# Shootouts have few snapshots (typically 3-20) but have been unanimous in
# every match in the dataset, so they use a lower count with the same
# agreement requirement.
MINIMUM_PERIOD_VOTES = 20
MINIMUM_SHOOTOUT_VOTES = 3
MINIMUM_PERIOD_AGREEMENT = 0.90

# Pairs of periods that the Laws of the Game require to be played at
# opposite ends. Period 3 vs 2 and period 5 are deliberately absent: they
# are measured, not assumed.
REQUIRED_END_SWAPS = ((1, 2), (3, 4))

Point = tuple[float, float]


@dataclass(frozen=True)
class CameraSideEstimate:
    """Camera-side reading of one visible_area polygon, in its own frame."""

    side: str
    apex: Point | None = None
    convergence_degrees: float | None = None
    reason: str = ""


def _polygon_vertices(value: Any) -> list[Point] | None:
    if not isinstance(value, list) or len(value) < 6 or len(value) % 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    points = [(float(value[i]), float(value[i + 1])) for i in range(0, len(value), 2)]
    # StatsBomb usually repeats the first vertex to close the ring.
    while len(points) > 1 and math.dist(points[0], points[-1]) < 1e-9:
        points.pop()
    return points if len(points) >= 3 else None


def _on_pitch_boundary(p: Point, q: Point) -> bool:
    """True if segment p-q lies along one pitch boundary line (a clip edge)."""
    for axis, value in ((0, 0.0), (0, PITCH_LENGTH), (1, 0.0), (1, PITCH_WIDTH)):
        if abs(p[axis] - value) < _BOUNDARY_TOLERANCE and abs(q[axis] - value) < _BOUNDARY_TOLERANCE:
            return True
    return False


def _point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
                inside = not inside
    return inside


def estimate_camera_side(visible_area: Any) -> CameraSideEstimate:
    """Infer which touchline the camera stands behind, in the polygon's own frame."""
    points = _polygon_vertices(visible_area)
    if points is None:
        return CameraSideEstimate(CAMERA_SIDE_UNKNOWN, reason="no usable polygon")

    n = len(points)
    real_edges = [
        (i, points[i], points[(i + 1) % n])
        for i in range(n)
        if not _on_pitch_boundary(points[i], points[(i + 1) % n])
        and math.dist(points[i], points[(i + 1) % n]) > _MINIMUM_EDGE_LENGTH
    ]

    best: tuple[float, Point] | None = None
    for a in range(len(real_edges)):
        for b in range(a + 1, len(real_edges)):
            i, p1, p2 = real_edges[a]
            j, q1, q2 = real_edges[b]
            if (j - i) % n in (1, n - 1):
                continue  # adjacent edges meet at their shared vertex
            d1 = (p2[0] - p1[0], p2[1] - p1[1])
            d2 = (q2[0] - q1[0], q2[1] - q1[1])
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            sine = min(1.0, abs(cross) / (math.hypot(*d1) * math.hypot(*d2)))
            angle = math.degrees(math.asin(sine))
            if angle < MINIMUM_CONVERGENCE_DEGREES:
                continue
            t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / cross
            apex = (p1[0] + t * d1[0], p1[1] + t * d1[1])
            if best is None or angle > best[0]:
                best = (angle, apex)

    if best is None:
        return CameraSideEstimate(CAMERA_SIDE_UNKNOWN, reason="no converging lateral edge pair")

    angle, apex = best
    if _point_in_polygon(apex, points):
        return CameraSideEstimate(CAMERA_SIDE_UNKNOWN, apex, angle, "apex inside footprint")
    centroid_x = sum(p[0] for p in points) / n
    centroid_y = sum(p[1] for p in points) / n
    dx, dy = apex[0] - centroid_x, apex[1] - centroid_y
    if abs(dy) < MINIMUM_TOUCHLINE_RATIO * abs(dx):
        return CameraSideEstimate(CAMERA_SIDE_UNKNOWN, apex, angle, "footprint converges toward a goal line")
    side = CAMERA_SIDE_LOW_Y if dy < 0 else CAMERA_SIDE_HIGH_Y
    return CameraSideEstimate(side, apex, angle, "lateral edges converge toward a touchline")


def flip_camera_side(side: str) -> str:
    """The camera side after a 180-degree rotation of the frame."""
    if side == CAMERA_SIDE_LOW_Y:
        return CAMERA_SIDE_HIGH_Y
    if side == CAMERA_SIDE_HIGH_Y:
        return CAMERA_SIDE_LOW_Y
    return side


def frame_model_for_period(period: int) -> str:
    return FRAME_MODEL_SHARED if period in SHARED_FRAME_PERIODS else FRAME_MODEL_OPPOSITE


@dataclass(frozen=True)
class PeriodCameraEvidence:
    """Camera-side votes for one period, and the resulting display transform.

    `votes_by_team[team_id]` counts that team's events by the camera side
    their (orientation-corrected) snapshot shows in the team's own frame.

    OPPOSITE periods: a vote "team T's frame already has the camera at the
    bottom" is also a vote that the other team must be rotated, so both
    teams' events feed one decision, `reference_team_id` (stays unrotated).
    SHARED periods: both teams' frames coincide, so every vote is about the
    one shared frame, decided as `rotate_both`.
    """

    period: int
    frame_model: str
    votes_by_team: dict[int, dict[str, int]]
    support: int
    opposition: int
    reference_team_id: int | None = None
    rotate_both: bool | None = None

    @property
    def agreement(self) -> float:
        total = self.support + self.opposition
        return self.support / total if total else 0.0

    @property
    def decided(self) -> bool:
        return self.reference_team_id is not None or self.rotate_both is not None


@dataclass(frozen=True)
class EndSwapCheck:
    """Whether two periods were measured at opposite ends."""

    period_a: int
    period_b: int
    required: bool
    swapped: bool | None  # None when either period is undecided


@dataclass(frozen=True)
class BroadcastOrientationInference:
    """Match-level camera-side evidence, one entry per period."""

    status: str
    reference_by_period: dict[int, int]
    periods: tuple[PeriodCameraEvidence, ...]
    end_swap_checks: tuple[EndSwapCheck, ...] = field(default_factory=tuple)
    explanation: str = ""

    def period_evidence(self, period: int) -> PeriodCameraEvidence | None:
        return next((evidence for evidence in self.periods if evidence.period == period), None)


def _event_team_id(event: dict[str, Any]) -> int | None:
    team = event.get("team")
    team_id = team.get("id") if isinstance(team, dict) else None
    if isinstance(team_id, bool) or not isinstance(team_id, int):
        return None
    return team_id


def _event_period(event: dict[str, Any]) -> int | None:
    period = event.get("period")
    if isinstance(period, bool) or not isinstance(period, int):
        return None
    return period


def _decide(support: int, opposition: int, minimum_votes: int) -> bool:
    total = support + opposition
    return support != opposition and total >= minimum_votes and support / total >= MINIMUM_PERIOD_AGREEMENT


def infer_broadcast_orientation(
    events: list[dict[str, Any]],
    frames_360: list[dict[str, Any]],
    team_ids: tuple[int, int],
) -> BroadcastOrientationInference:
    """Infer, per period, the 180-degree rotations that put the camera at the bottom.

    `events[i]` must own `frames_360[i]`, and each frame must already be in
    its own event's frame -- i.e. after Problem A's orientation correction,
    and before any team normalization. Callers should drop frames whose
    Problem A classification is UNCERTAIN, since their frame is unknown.
    """
    if len(events) != len(frames_360):
        raise ValueError("events and frames_360 must have equal length.")
    first_team, second_team = team_ids
    other = {first_team: second_team, second_team: first_team}
    bottom, top = DISPLAY_CAMERA_SIDE, flip_camera_side(DISPLAY_CAMERA_SIDE)

    votes: dict[int, dict[int, Counter]] = {}
    for event, frame in zip(events, frames_360):
        team_id = _event_team_id(event)
        period = _event_period(event)
        if team_id not in other or period is None:
            continue
        side = estimate_camera_side(frame.get("visible_area")).side
        votes.setdefault(period, {first_team: Counter(), second_team: Counter()})[team_id][side] += 1

    periods: list[PeriodCameraEvidence] = []
    for period in sorted(votes):
        by_team = votes[period]
        model = frame_model_for_period(period)
        team_votes = {team: dict(by_team[team]) for team in team_ids}
        if model == FRAME_MODEL_SHARED:
            keep = by_team[first_team][bottom] + by_team[second_team][bottom]
            rotate = by_team[first_team][top] + by_team[second_team][top]
            support, opposition = max(keep, rotate), min(keep, rotate)
            decided = _decide(support, opposition, MINIMUM_SHOOTOUT_VOTES)
            periods.append(PeriodCameraEvidence(
                period, model, team_votes, support, opposition,
                rotate_both=(rotate > keep) if decided else None,
            ))
            continue
        # Evidence that a team's own frame is already camera-at-bottom.
        score = {team: by_team[team][bottom] + by_team[other[team]][top] for team in team_ids}
        leader = max(team_ids, key=lambda team: score[team])
        support, opposition = score[leader], score[other[leader]]
        decided = _decide(support, opposition, MINIMUM_PERIOD_VOTES)
        periods.append(PeriodCameraEvidence(
            period, model, team_votes, support, opposition,
            reference_team_id=leader if decided else None,
        ))

    reference_by_period = {e.period: e.reference_team_id for e in periods if e.reference_team_id is not None}
    present = {e.period for e in periods}

    def swap_check(period_a: int, period_b: int, required: bool) -> EndSwapCheck:
        swapped = (
            reference_by_period[period_a] != reference_by_period[period_b]
            if period_a in reference_by_period and period_b in reference_by_period
            else None
        )
        return EndSwapCheck(period_a, period_b, required, swapped)

    checks = [swap_check(a, b, True) for a, b in REQUIRED_END_SWAPS if a in present and b in present]
    if 2 in present and 3 in present:
        checks.append(swap_check(2, 3, False))

    regular = [e for e in periods if e.period not in SHARED_FRAME_PERIODS]
    if not periods or all(e.support + e.opposition == 0 for e in periods):
        status = BROADCAST_STATUS_NO_EVIDENCE
        explanation = "No visible_area polygon yielded a camera-side reading."
    elif any(check.required and check.swapped is False for check in checks):
        status = BROADCAST_STATUS_CONFLICTING
        explanation = "Camera side contradicts a Laws-of-the-Game end swap; physical orientation unknown."
    elif regular and all(e.decided for e in regular):
        status = BROADCAST_STATUS_INFERRED
        explanation = "Every regular period decided independently from visible_area; required end swaps observed."
    else:
        status = BROADCAST_STATUS_AMBIGUOUS
        undecided = [e.period for e in regular if not e.decided]
        explanation = f"Insufficient or split camera-side evidence in period(s) {undecided}."

    return BroadcastOrientationInference(
        status=status,
        reference_by_period=reference_by_period,
        periods=tuple(periods),
        end_swap_checks=tuple(checks),
        explanation=explanation,
    )


# --------------------------------------------------------------------------
# Display plan: the explicit, per-period presentation transform.
# --------------------------------------------------------------------------

PROVENANCE_VERIFIED = "VERIFIED"        # anchored on independent broadcast evidence
PROVENANCE_PREFERENCE = "PREFERENCE"    # anchored on an explicit user display choice
PROVENANCE_INFERRED = "INFERRED"        # every period from visible_area camera side
PROVENANCE_PARTIAL = "PARTIALLY_INFERRED"  # some periods derived by rule from inferred ones
PROVENANCE_UNKNOWN = "UNKNOWN"          # deterministic fallback, physical orientation unknown
PROVENANCE_USER_CONFIRMED = "USER_CONFIRMED"  # every regular period confirmed by a reviewer (not video-verified)
# Every regular period is either reviewer-confirmed or derived from a
# reviewer-confirmed period by the halftime / extra-time end-swap rule.
PROVENANCE_USER_CONFIRMED_AND_DERIVED = "USER_CONFIRMED_AND_DERIVED"

SOURCE_ANCHOR = "anchor"
SOURCE_CAMERA = "camera"
SOURCE_CAMERA_RELATIVE = "anchor+camera_end_relation"
SOURCE_LAWS_RULE = "laws_rule"
SOURCE_FALLBACK = "fallback"
SOURCE_USER_CONFIRMED = "user_confirmed"
SOURCE_DERIVED_FROM_USER_REVIEW = "derived_from_user_reviewed_period_and_halftime_rule"

# Rule-based fallback for periods without camera evidence: the second half
# of each pair is played at the opposite end of the first. Period 3 vs 1 is
# an ASSUMPTION used only when unmeasured (2 of 30 extra-time matches in
# the dataset start period 3 at period 2's ends instead).
_RULE_PARTNER = {1: (2, True), 2: (1, True), 3: (4, True), 4: (3, True)}


@dataclass(frozen=True)
class PeriodTransform:
    """How one period's events are rotated for display, and why."""

    period: int
    frame_model: str
    reference_team_id: int | None  # OPPOSITE: the team left unrotated
    rotate_both: bool               # SHARED: rotate both teams (or neither)
    source: str

    def rotates(self, team_id: int | None) -> bool:
        if self.frame_model == FRAME_MODEL_SHARED:
            return self.rotate_both
        return team_id != self.reference_team_id


@dataclass(frozen=True)
class DisplayOrientationPlan:
    """The complete presentation-layer transform for one match."""

    provenance: str
    transforms: dict[int, PeriodTransform]
    explanation: str
    team_ids: tuple[int, int] = (0, 0)

    def rotates(self, period: int, team_id: int | None) -> bool:
        return self.transforms[period].rotates(team_id)

    def reference_by_period(self) -> dict[int, int]:
        return {
            period: transform.reference_team_id
            for period, transform in self.transforms.items()
            if transform.reference_team_id is not None
        }

    def attacking_directions(self, period: int) -> dict[int, int] | None:
        """+1 if a team attacks toward increasing display x, -1 otherwise.

        Exact consequence of the transform (raw StatsBomb frames always
        have the event's team attacking +x): an unrotated team attacks +x.
        None for shared-frame periods, where both teams attack one goal.
        """
        transform = self.transforms.get(period)
        if transform is None or transform.frame_model == FRAME_MODEL_SHARED:
            return None
        reference = transform.reference_team_id
        return {team: 1 if team == reference else -1 for team in self.team_ids}


def build_display_plan(
    inference: BroadcastOrientationInference | None,
    team_ids: tuple[int, int],
    periods_present: list[int],
    anchor_period_1_reference: int | None = None,
    anchor_provenance: str = PROVENANCE_VERIFIED,
    fallback_period_1_reference: int | None = None,
    confirmed_references: dict[int, int] | None = None,
) -> DisplayOrientationPlan:
    """Resolve one PeriodTransform per present period.

    With an anchor (a verified calibration or an explicit preference for
    period 1), the anchor fixes the absolute frame and the camera evidence
    is used only for RELATIVE end relations between periods (which do not
    depend on the absolute side). Without one, each period comes straight
    from the camera evidence; undecided periods are derived from a decided
    partner by the Laws-of-the-Game swap; anything still unresolved uses
    `fallback_period_1_reference` (the legacy automatic pick) and the plan
    is marked UNKNOWN.

    `confirmed_references` ({period: unrotated team id}) are per-period
    reviewer decisions (src/physical_orientation_calibration.py
    period_reviews). They are ignored when an anchor is given (a verified
    calibration or explicit flag wins). Otherwise each one fixes its own
    period; if the comparable confirmations all contradict the camera, the
    camera's frame is taken to be rotated and every camera-derived period
    is flipped with it (the camera relation between periods still holds).
    They never upgrade other periods to "confirmed".
    """
    first_team, second_team = team_ids
    other = {first_team: second_team, second_team: first_team}
    camera = dict(inference.reference_by_period) if inference is not None else {}
    shared_evidence = {
        e.period: e.rotate_both for e in (inference.periods if inference else ()) if e.rotate_both is not None
    }

    transforms: dict[int, PeriodTransform] = {}
    references: dict[int, tuple[int, str]] = {}

    if anchor_period_1_reference is not None:
        references[1] = (anchor_period_1_reference, SOURCE_ANCHOR)
        for period in periods_present:
            if period in SHARED_FRAME_PERIODS or period == 1:
                continue
            if 1 in camera and period in camera:
                same_end = camera[period] == camera[1]
                references[period] = (
                    anchor_period_1_reference if same_end else other[anchor_period_1_reference],
                    SOURCE_CAMERA_RELATIVE,
                )
        # Camera and anchor disagree on period 1 => the anchor's frame is the
        # camera's frame rotated by 180 degrees; shared periods follow suit.
        anchor_flips_camera = 1 in camera and camera[1] != anchor_period_1_reference
    else:
        for period, reference in camera.items():
            references[period] = (reference, SOURCE_CAMERA)
        anchor_flips_camera = False
        confirmed = {p: t for p, t in (confirmed_references or {}).items() if p not in SHARED_FRAME_PERIODS}
        disagreements = {camera[p] != t for p, t in confirmed.items() if p in camera}
        if disagreements == {True}:
            for period, (reference, _) in list(references.items()):
                references[period] = (other[reference], SOURCE_CAMERA_RELATIVE)
            anchor_flips_camera = True
        for period, team in confirmed.items():
            references[period] = (team, SOURCE_USER_CONFIRMED)

    # Fill unmeasured regular periods from a resolved partner by rule; as a
    # last resort, chain from period 1 (anchor/fallback) through the rule.
    def rule_fill() -> None:
        changed = True
        while changed:
            changed = False
            for period in periods_present:
                if period in references or period in SHARED_FRAME_PERIODS:
                    continue
                partner, swap = _RULE_PARTNER.get(period, (1, False))
                if partner in references:
                    base, partner_source = references[partner]
                    # A period derived from a reviewer-confirmed partner keeps
                    # that lineage in its label; it is still never "confirmed".
                    source = (
                        SOURCE_DERIVED_FROM_USER_REVIEW
                        if partner_source in (SOURCE_USER_CONFIRMED, SOURCE_DERIVED_FROM_USER_REVIEW)
                        else SOURCE_LAWS_RULE
                    )
                    references[period] = (other[base] if swap else base, source)
                    changed = True
                elif period == 3 and 1 in references:
                    references[period] = (references[1][0], SOURCE_LAWS_RULE)
                    changed = True

    rule_fill()
    if any(p not in references and p not in SHARED_FRAME_PERIODS for p in periods_present):
        if fallback_period_1_reference is not None and 1 not in references:
            references[1] = (fallback_period_1_reference, SOURCE_FALLBACK)
            rule_fill()

    for period in periods_present:
        if period in SHARED_FRAME_PERIODS:
            if period in shared_evidence:
                rotate, source = shared_evidence[period] != anchor_flips_camera, (
                    SOURCE_CAMERA if anchor_period_1_reference is None else SOURCE_CAMERA_RELATIVE
                )
            else:
                rotate, source = False, SOURCE_FALLBACK
            transforms[period] = PeriodTransform(period, FRAME_MODEL_SHARED, None, rotate, source)
        elif period in references:
            reference, source = references[period]
            transforms[period] = PeriodTransform(period, FRAME_MODEL_OPPOSITE, reference, False, source)
        else:
            raise ValueError(f"Cannot resolve a display transform for period {period}.")

    sources = {t.source for t in transforms.values()}
    regular_sources = {t.source for t in transforms.values() if t.frame_model != FRAME_MODEL_SHARED}
    if anchor_period_1_reference is not None:
        provenance = anchor_provenance
    elif regular_sources == {SOURCE_USER_CONFIRMED}:
        provenance = PROVENANCE_USER_CONFIRMED
    elif SOURCE_USER_CONFIRMED in regular_sources and regular_sources <= {SOURCE_USER_CONFIRMED, SOURCE_DERIVED_FROM_USER_REVIEW}:
        provenance = PROVENANCE_USER_CONFIRMED_AND_DERIVED
    elif SOURCE_FALLBACK in {t.source for t in transforms.values() if t.frame_model == FRAME_MODEL_OPPOSITE}:
        provenance = PROVENANCE_UNKNOWN
    elif sources <= {SOURCE_CAMERA}:
        provenance = PROVENANCE_INFERRED
    else:
        provenance = PROVENANCE_PARTIAL

    explanation = "; ".join(
        f"P{t.period}: "
        + (f"rotate both teams={t.rotate_both}" if t.frame_model == FRAME_MODEL_SHARED else f"unrotated team id {t.reference_team_id}")
        + f" [{t.source}]"
        for t in transforms.values()
    )
    return DisplayOrientationPlan(provenance, transforms, explanation, team_ids)


# --------------------------------------------------------------------------
# Display-only resolution of Problem A's UNCERTAIN snapshots.
# --------------------------------------------------------------------------

def _measured_display_camera_side(
    inference: BroadcastOrientationInference, plan: "DisplayOrientationPlan"
) -> str:
    """Where the fixed camera appears ON SCREEN under `plan`, measured only
    from camera-decided periods (bottom normally; top if a verified anchor
    contradicts the camera). UNKNOWN if decided periods disagree."""
    sides = set()
    for evidence in inference.periods:
        if not evidence.decided or evidence.period not in plan.transforms:
            continue
        team = evidence.reference_team_id if evidence.reference_team_id is not None else plan.team_ids[0]
        own = _expected_from_inference(evidence, team)
        sides.add(flip_camera_side(own) if plan.rotates(evidence.period, team) else own)
    return sides.pop() if len(sides) == 1 else CAMERA_SIDE_UNKNOWN


def _expected_from_inference(evidence: PeriodCameraEvidence, team_id: int | None) -> str:
    if evidence.frame_model == FRAME_MODEL_SHARED:
        return flip_camera_side(DISPLAY_CAMERA_SIDE) if evidence.rotate_both else DISPLAY_CAMERA_SIDE
    if team_id == evidence.reference_team_id:
        return DISPLAY_CAMERA_SIDE
    return flip_camera_side(DISPLAY_CAMERA_SIDE)


def expected_own_frame_camera_side(
    inference: BroadcastOrientationInference,
    period: int,
    team_id: int | None,
    plan: "DisplayOrientationPlan | None" = None,
) -> str:
    """The camera side a snapshot must show if it is in its own event's frame.

    The camera is fixed, so every correctly-framed snapshot of one
    (period, team) group shows the same side. A camera-decided period uses
    its own measurement. An undecided period uses `plan` (if given): the
    camera's on-screen side is measured from the decided periods and
    carried to this period through the plan's own rotations -- the plan
    fixes which frame the period is in (e.g. derived by the halftime rule
    from a reviewed period), the camera measurement fixes where the camera
    is. UNKNOWN when neither is available.
    """
    evidence = inference.period_evidence(period)
    if evidence is None or not evidence.decided:
        if plan is None or period not in plan.transforms:
            return CAMERA_SIDE_UNKNOWN
        on_screen = _measured_display_camera_side(inference, plan)
        if on_screen == CAMERA_SIDE_UNKNOWN:
            return CAMERA_SIDE_UNKNOWN
        return flip_camera_side(on_screen) if plan.rotates(period, team_id) else on_screen
    if evidence.frame_model == FRAME_MODEL_SHARED:
        return flip_camera_side(DISPLAY_CAMERA_SIDE) if evidence.rotate_both else DISPLAY_CAMERA_SIDE
    if team_id == evidence.reference_team_id:
        return DISPLAY_CAMERA_SIDE
    return flip_camera_side(DISPLAY_CAMERA_SIDE)


def resolve_uncertain_for_display(
    lean_rotated: bool,
    visible_area: Any,
    expected_side: str,
) -> str | None:
    """Combine Problem A's lean with the camera side for ONE UNCERTAIN snapshot.

    Returns "ROTATED_180" when both independent signals say the snapshot is
    stored rotated (Problem A's weighted votes lean rotated AND the
    footprint shows the camera on the wrong side for its group), "NORMAL"
    when both say it is not, and None when they disagree or the camera side
    is unreadable -- the snapshot then stays uncorrected, exactly as
    Problem A left it. Measured on 120 matches: when the votes lean
    rotated, the camera side contradicts the event frame in 1085/1088
    UNCERTAIN snapshots. Display only; the dataset builder never calls it.
    """
    side = estimate_camera_side(visible_area).side
    if side == CAMERA_SIDE_UNKNOWN or expected_side == CAMERA_SIDE_UNKNOWN:
        return None
    camera_says_rotated = side != expected_side
    if lean_rotated and camera_says_rotated:
        return "ROTATED_180"
    if not lean_rotated and not camera_says_rotated:
        return "NORMAL"
    return None


# Kick-off own-half rule (Law 8: at a kick-off every player is in their own
# half). In its own event's frame the kicking team attacks +x, so its
# players stand at x < 60 and the opponents at x > 60. Players within this
# many yards of the halfway line are ignored; a verdict needs at least
# KICKOFF_MINIMUM_PLAYERS decisive players and a KICKOFF_MINIMUM_SHARE
# majority. Validated on the 464 kick-off snapshots whose camera side
# already agreed: 457 NORMAL, 7 undecided, 0 ROTATED.
KICKOFF_HALFWAY_MARGIN = 1.5
KICKOFF_MINIMUM_PLAYERS = 6
KICKOFF_MINIMUM_SHARE = 0.85


def kickoff_own_half_orientation(frame_360: dict[str, Any]) -> str | None:
    """NORMAL / ROTATED_180 for a kick-off snapshot in its event's frame, or None."""
    own = wrong = 0
    for player in frame_360.get("freeze_frame") or []:
        location = player.get("location") if isinstance(player, dict) else None
        if player.get("actor") or not isinstance(location, list) or len(location) != 2:
            continue
        x = location[0]
        if abs(x - PITCH_LENGTH / 2) < KICKOFF_HALFWAY_MARGIN:
            continue
        in_own_half = x < PITCH_LENGTH / 2 if player.get("teammate") else x > PITCH_LENGTH / 2
        own += in_own_half
        wrong += not in_own_half
    total = own + wrong
    if total < KICKOFF_MINIMUM_PLAYERS:
        return None
    if own / total >= KICKOFF_MINIMUM_SHARE:
        return "NORMAL"
    if wrong / total >= KICKOFF_MINIMUM_SHARE:
        return "ROTATED_180"
    return None
