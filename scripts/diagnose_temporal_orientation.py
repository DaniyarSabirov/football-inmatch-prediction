#!/usr/bin/env python3
"""DIAGNOSTIC-ONLY: sequence-level (temporal) StatsBomb 360 orientation check.

This is a research tool, not a production module. It does not modify
src/orientation_correction.py, src/build_statsbomb_360_dataset.py, or
src/animate_statsbomb_360_match.py -- it only imports their read-only loading
and geometry helpers, and re-uses src/orientation_correction.py's classifier
as the "current" baseline to critique.

Motivation: the existing classifier (src/orientation_correction.py) decides
each event's orientation independently, anchored on that SAME event's own
`event.location`. This script asks a different, complementary question: do
consecutive REAL freeze-frame snapshots (which must depict nearly the same
22 real players moving only a physically plausible amount in a short dt)
actually look like continuous football, or does the WHOLE configuration
(both teams, goalkeepers, visible_area) jump by a 180-degree pitch rotation
relative to its immediate neighbor -- regardless of what each frame's own
event-level classifier says?

Two independent signals are combined and compared for every consecutive pair:
  1. Pairwise player-configuration similarity (Hungarian optimal matching per
     physical team, not just the actor) under H0 (no transform) vs H1 (mirror
     the second frame), separately for each of the two teams, plus the
     goalkeeper specifically, plus visible_area polygon centroid.
  2. Physical plausibility of the implied real-world displacement given the
     real dt between the two events (elite sprint speed ~9-10 m/s is used as
     a soft plausibility reference, never a hard cutoff by itself).

It also prototypes (does NOT ship) a simple sequence-level dynamic-programming
relabeling to test whether jointly optimizing a whole chain of frames'
NORMAL/ROTATED_180 labels (event-anchor cost + inter-frame transition cost)
ever disagrees with the existing independent, per-event classifier.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from animate_statsbomb_360_match import (  # noqa: E402
    JoinedFrame,
    VisualizationError,
    coordinate_pair,
    integer_event_field,
    load_and_join_match,
    nested_name,
    resolve_interval_seconds,
    timestamp_seconds,
    visible_area_points,
)
from orientation_correction import (  # noqa: E402
    ORIENTATION_NORMAL,
    ORIENTATION_ROTATED_180,
    ORIENTATION_UNCERTAIN,
    classify_orientation,
    mirror_point,
)

Point = tuple[float, float]

# Soft physical-plausibility reference: a very fast player covers roughly
# 8-10 m/s in short bursts. Used only as a soft prior, never a hard veto --
# per the task's explicit instruction not to hard-code a "cannot move X units"
# rule in isolation.
PLAUSIBLE_SPEED_MPS = 9.0
PLAUSIBLE_MARGIN_M = 4.0  # capture-timing jitter / non-actor players cutting angles


def select_window(match_id: int, period: int, start_time: str, end_time: str) -> list[JoinedFrame]:
    match = load_and_join_match(match_id)
    start_seconds, end_seconds = resolve_interval_seconds(
        start_minute=0, end_minute=0, start_time=start_time, end_time=end_time
    )
    windowed = [
        joined
        for joined in match.usable_frames
        if integer_event_field(joined.event, "period") == period
        and start_seconds <= timestamp_seconds(joined.event.get("timestamp")) < end_seconds
    ]
    if not windowed:
        raise VisualizationError("No usable frames in the requested window.")
    return windowed


def resolve_two_teams(frames: list[JoinedFrame]) -> tuple[str, str]:
    names: list[str] = []
    for joined in frames:
        name = nested_name(joined.event, "team")
        if name != "Unknown" and name not in names:
            names.append(name)
    if len(names) != 2:
        raise VisualizationError(f"Expected exactly two teams, found {names}.")
    return names[0], names[1]


@dataclass
class TeamPoints:
    """Physical-team-labeled points extracted from one freeze_frame."""

    outfield: dict[str, list[Point]]  # team_name -> list of non-keeper points
    keeper: dict[str, Point | None]   # team_name -> single keeper point or None
    actor_team: str | None
    actor_point: Point | None


def extract_team_points(joined: JoinedFrame, team_a: str, team_b: str) -> TeamPoints:
    """Group freeze_frame players by PHYSICAL team, not by the event's own
    'teammate' flag alone -- teammate is relative to THIS event's team, so it
    is resolved into an actual team name using the event's own team.
    """
    event_team = nested_name(joined.event, "team")
    other_team = team_b if event_team == team_a else team_a
    outfield: dict[str, list[Point]] = {team_a: [], team_b: []}
    keeper: dict[str, Point | None] = {team_a: None, team_b: None}
    actor_team = None
    actor_point = None

    freeze_frame = joined.frame_360.get("freeze_frame") or []
    for player in freeze_frame:
        if not isinstance(player, dict):
            continue
        point = coordinate_pair(player.get("location"))
        if point is None:
            continue
        physical_team = event_team if player.get("teammate") is True else other_team
        if player.get("keeper") is True:
            keeper[physical_team] = point
        else:
            outfield[physical_team].append(point)
        if player.get("actor") is True:
            actor_team = physical_team
            actor_point = point

    return TeamPoints(outfield, keeper, actor_team, actor_point)


def hungarian_distances(points_a: list[Point], points_b: list[Point]) -> list[float]:
    """Optimal one-to-one matching cost between two point sets (min(|a|,|b|) pairs)."""
    if not points_a or not points_b:
        return []
    cost = np.zeros((len(points_a), len(points_b)))
    for i, pa in enumerate(points_a):
        for j, pb in enumerate(points_b):
            cost[i, j] = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
    row_idx, col_idx = linear_sum_assignment(cost)
    return [float(cost[r, c]) for r, c in zip(row_idx, col_idx)]


def mirror_all(points: list[Point]) -> list[Point]:
    return [mirror_point(p) for p in points]


def polygon_centroid(polygon: list[Point] | None) -> Point | None:
    if not polygon:
        return None
    return (sum(p[0] for p in polygon) / len(polygon), sum(p[1] for p in polygon) / len(polygon))


@dataclass
class PairAnalysis:
    dt: float
    displacements_normal: list[float]
    displacements_rotated: list[float]
    keeper_normal: dict[str, float | None]
    keeper_rotated: dict[str, float | None]
    visible_area_normal: float | None
    visible_area_rotated: float | None
    event_location_dt_speed: float | None  # ball/event.location implied speed (independent sanity check)


def analyze_pair(
    frame_i: JoinedFrame, frame_j: JoinedFrame, team_a: str, team_b: str
) -> PairAnalysis:
    """Compare two consecutive real snapshots under H0 (no transform) and
    H1 (mirror frame_j) using every physical player, both teams, the
    goalkeepers specifically, and visible_area -- not just the actor.
    """
    dt = timestamp_seconds(frame_j.event.get("timestamp")) - timestamp_seconds(
        frame_i.event.get("timestamp")
    )

    ti = extract_team_points(frame_i, team_a, team_b)
    tj = extract_team_points(frame_j, team_a, team_b)

    displacements_normal: list[float] = []
    displacements_rotated: list[float] = []
    keeper_normal: dict[str, float | None] = {}
    keeper_rotated: dict[str, float | None] = {}

    for team in (team_a, team_b):
        displacements_normal.extend(hungarian_distances(ti.outfield[team], tj.outfield[team]))
        displacements_rotated.extend(
            hungarian_distances(ti.outfield[team], mirror_all(tj.outfield[team]))
        )
        if ti.keeper[team] is not None and tj.keeper[team] is not None:
            keeper_normal[team] = math.hypot(
                ti.keeper[team][0] - tj.keeper[team][0], ti.keeper[team][1] - tj.keeper[team][1]
            )
            mirrored_keeper_j = mirror_point(tj.keeper[team])
            keeper_rotated[team] = math.hypot(
                ti.keeper[team][0] - mirrored_keeper_j[0], ti.keeper[team][1] - mirrored_keeper_j[1]
            )
        else:
            keeper_normal[team] = None
            keeper_rotated[team] = None

    polygon_i = visible_area_points(frame_i.frame_360.get("visible_area"))
    polygon_j = visible_area_points(frame_j.frame_360.get("visible_area"))
    centroid_i = polygon_centroid(polygon_i)
    centroid_j = polygon_centroid(polygon_j)
    visible_area_normal = None
    visible_area_rotated = None
    if centroid_i is not None and centroid_j is not None:
        visible_area_normal = math.hypot(centroid_i[0] - centroid_j[0], centroid_i[1] - centroid_j[1])
        mirrored_centroid_j = mirror_point(centroid_j)
        visible_area_rotated = math.hypot(
            centroid_i[0] - mirrored_centroid_j[0], centroid_i[1] - mirrored_centroid_j[1]
        )

    # Independent sanity check: does event.location -> event.location imply a
    # plausible ball speed? (event.location is not being judged for rotation
    # here -- only used as an orthogonal continuity signal.)
    loc_i = coordinate_pair(frame_i.event.get("location"))
    loc_j = coordinate_pair(frame_j.event.get("location"))
    event_location_dt_speed = None
    if loc_i is not None and loc_j is not None and dt > 0:
        event_location_dt_speed = math.hypot(loc_i[0] - loc_j[0], loc_i[1] - loc_j[1]) / dt

    return PairAnalysis(
        dt,
        displacements_normal,
        displacements_rotated,
        keeper_normal,
        keeper_rotated,
        visible_area_normal,
        visible_area_rotated,
        event_location_dt_speed,
    )


def plausible_distance(dt: float) -> float:
    """Soft plausibility bound for how far a real player can move in dt seconds."""
    return max(dt, 0.0) * PLAUSIBLE_SPEED_MPS + PLAUSIBLE_MARGIN_M


def diagnose_pair(analysis: PairAnalysis) -> tuple[str, str]:
    """Return (recommended_pair_orientation, final_diagnosis) for one transition."""
    if not analysis.displacements_normal or not analysis.displacements_rotated:
        return "UNKNOWN", "AMBIGUOUS"

    median_normal = statistics.median(analysis.displacements_normal)
    median_rotated = statistics.median(analysis.displacements_rotated)
    bound = plausible_distance(analysis.dt)

    normal_plausible = median_normal <= bound
    rotated_plausible = median_rotated <= bound
    much_better_rotated = median_rotated < median_normal * 0.5 and median_normal > bound
    much_better_normal = median_normal < median_rotated * 0.5

    if normal_plausible and not (rotated_plausible and median_rotated < median_normal):
        return "NORMAL", "CONSISTENT"
    if much_better_rotated and rotated_plausible:
        return "ROTATED_180", "LIKELY_ROTATION"
    if much_better_normal:
        return "NORMAL", "CONSISTENT"
    return "UNKNOWN", "AMBIGUOUS"


def build_transition_table(frames: list[JoinedFrame], team_a: str, team_b: str) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(frames) - 1):
        frame_i, frame_j = frames[i], frames[i + 1]
        analysis = analyze_pair(frame_i, frame_j, team_a, team_b)
        result_i = classify_orientation(frame_i.event, frame_i.frame_360)
        recommended, diagnosis = diagnose_pair(analysis)
        rows.append(
            {
                "frame_i": i + 1,
                "frame_j": i + 2,
                "event_uuid_i": frame_i.event["id"],
                "event_uuid_j": frame_j.event["id"],
                "timestamp_i": frame_i.event.get("timestamp"),
                "timestamp_j": frame_j.event.get("timestamp"),
                "dt_seconds": analysis.dt,
                "current_orientation_i": result_i.orientation,
                "current_confidence_i": result_i.confidence,
                "median_displacement_normal": statistics.median(analysis.displacements_normal)
                if analysis.displacements_normal else None,
                "median_displacement_rotated": statistics.median(analysis.displacements_rotated)
                if analysis.displacements_rotated else None,
                "keeper_normal": analysis.keeper_normal,
                "keeper_rotated": analysis.keeper_rotated,
                "visible_area_normal": analysis.visible_area_normal,
                "visible_area_rotated": analysis.visible_area_rotated,
                "plausible_bound": plausible_distance(analysis.dt),
                "recommended_pair_orientation": recommended,
                "final_diagnosis": diagnosis,
            }
        )
    return rows


def print_transition_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'i->j':<7}{'dt':>6}{'bound':>7}{'med_N':>8}{'med_R':>8}"
        f"{'gk_N':>7}{'gk_R':>7}{'va_N':>7}{'va_R':>7}"
        f"{'cur_i':<11}{'conf_i':>7}  {'diagnosis':<16}{'rec':<12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        gk_n = next((v for v in row["keeper_normal"].values() if v is not None), None)
        gk_r = next((v for v in row["keeper_rotated"].values() if v is not None), None)
        va_n = row["visible_area_normal"]
        va_r = row["visible_area_rotated"]

        def fmt(value: float | None) -> str:
            return "—" if value is None else f"{value:.1f}"

        print(
            f"{row['frame_i']:>2}->{row['frame_j']:<3}"
            f"{row['dt_seconds']:>6.2f}{row['plausible_bound']:>7.1f}"
            f"{fmt(row['median_displacement_normal']):>8}{fmt(row['median_displacement_rotated']):>8}"
            f"{fmt(gk_n):>7}{fmt(gk_r):>7}{fmt(va_n):>7}{fmt(va_r):>7}"
            f"{row['current_orientation_i']:<11}{row['current_confidence_i']:>7.2f}  "
            f"{row['final_diagnosis']:<16}{row['recommended_pair_orientation']:<12}"
        )


def print_forensic_case(frames: list[JoinedFrame], row: dict[str, Any], team_a: str, team_b: str) -> None:
    i, j = row["frame_i"] - 1, row["frame_j"] - 1
    frame_i, frame_j = frames[i], frames[j]
    event_i, event_j = frame_i.event, frame_j.event
    print(f"\n=== transition {row['frame_i']} -> {row['frame_j']} ===")
    print(f"  event_i: {event_i['id'][:8]} {nested_name(event_i,'type')} "
          f"team={nested_name(event_i,'team')} t={event_i.get('timestamp')} loc={event_i.get('location')}")
    print(f"  event_j: {event_j['id'][:8]} {nested_name(event_j,'type')} "
          f"team={nested_name(event_j,'team')} t={event_j.get('timestamp')} loc={event_j.get('location')}")
    print(f"  dt = {row['dt_seconds']:.3f}s   plausible_bound = {row['plausible_bound']:.1f} units")
    print(f"  median displacement: NORMAL={row['median_displacement_normal']:.2f}  "
          f"ROTATED={row['median_displacement_rotated']:.2f}")
    print(f"  goalkeeper displacement: NORMAL={row['keeper_normal']}  ROTATED={row['keeper_rotated']}")
    print(f"  visible_area centroid shift: NORMAL={row['visible_area_normal']}  "
          f"ROTATED={row['visible_area_rotated']}")
    result_i = classify_orientation(event_i, frame_i.frame_360)
    result_j = classify_orientation(event_j, frame_j.frame_360)
    print(f"  current per-event classifier: frame_i={result_i.orientation}({result_i.confidence:.2f})  "
          f"frame_j={result_j.orientation}({result_j.confidence:.2f})")
    print(f"  sequence-level recommendation for this pair: {row['recommended_pair_orientation']} "
          f"-> {row['final_diagnosis']}")


# --------------------------------------------------------------------------
# Task 5/9 prototype: sequence-level (chain) relabeling via dynamic programming.
# NOT used by production code. Exists only to test whether jointly optimizing
# a whole chain of frames' labels ever disagrees with the independent
# per-event classifier.
# --------------------------------------------------------------------------

def event_anchor_costs(joined: JoinedFrame) -> tuple[float, float]:
    """(cost_normal, cost_rotated): weighted distances from event.location to
    the SAME signals the production classifier uses, expressed as costs
    rather than a vote. Lower is better. Reuses only public geometry helpers.
    """
    from orientation_correction import extract_geometry

    geometry = extract_geometry(joined.event, joined.frame_360)
    loc = geometry.event_location
    if loc is None:
        return 0.0, 0.0

    weighted_normal = []
    weighted_rotated = []
    if geometry.actor_location is not None:
        weighted_normal.append((math.hypot(loc[0]-geometry.actor_location[0], loc[1]-geometry.actor_location[1]), 0.40))
        mirrored = mirror_point(geometry.actor_location)
        weighted_rotated.append((math.hypot(loc[0]-mirrored[0], loc[1]-mirrored[1]), 0.40))
    if geometry.points:
        d_n = min(math.hypot(loc[0]-p[0], loc[1]-p[1]) for p in geometry.points)
        d_r = min(math.hypot(loc[0]-mirror_point(p)[0], loc[1]-mirror_point(p)[1]) for p in geometry.points)
        weighted_normal.append((d_n, 0.20))
        weighted_rotated.append((d_r, 0.20))

    total_weight = sum(w for _, w in weighted_normal) or 1.0
    cost_normal = sum(d * w for d, w in weighted_normal) / total_weight
    cost_rotated = sum(d * w for d, w in weighted_rotated) / total_weight
    return cost_normal, cost_rotated


def sequence_level_relabel(
    frames: list[JoinedFrame], team_a: str, team_b: str, transition_weight: float = 1.0
) -> list[str]:
    """Viterbi-style dynamic program over {NORMAL, ROTATED_180} per frame.

    cost(label_1..N) = sum_i event_anchor_cost(i, label_i)
                      + transition_weight * sum_i transition_cost(i, i+1 | label_i, label_{i+1})

    transition_cost brings both frames into a COMMON hypothesized true frame
    (mirroring whichever side the label says is rotated) and then measures
    Hungarian-matched median player displacement -- i.e. it penalizes label
    combinations that would make consecutive real snapshots physically
    incoherent, on top of (not instead of) each frame's own event-anchor fit.
    """
    n = len(frames)
    labels = (ORIENTATION_NORMAL, ORIENTATION_ROTATED_180)
    anchor_costs = [event_anchor_costs(joined) for joined in frames]

    def true_points(joined: JoinedFrame, label: str) -> TeamPoints:
        points = extract_team_points(joined, team_a, team_b)
        if label == ORIENTATION_NORMAL:
            return points
        return TeamPoints(
            outfield={team: mirror_all(pts) for team, pts in points.outfield.items()},
            keeper={team: (mirror_point(p) if p else None) for team, p in points.keeper.items()},
            actor_team=points.actor_team,
            actor_point=mirror_point(points.actor_point) if points.actor_point else None,
        )

    # dp[i][label_index] = (min cost so far, backpointer)
    dp: list[list[tuple[float, int]]] = [[(0.0, -1), (0.0, -1)] for _ in range(n)]
    dp[0][0] = (anchor_costs[0][0], -1)
    dp[0][1] = (anchor_costs[0][1], -1)

    cached_true_points: dict[tuple[int, str], TeamPoints] = {}

    def get_true(i: int, label: str) -> TeamPoints:
        key = (i, label)
        if key not in cached_true_points:
            cached_true_points[key] = true_points(frames[i], label)
        return cached_true_points[key]

    for i in range(1, n):
        for cur_idx, cur_label in enumerate(labels):
            best_cost = math.inf
            best_prev = -1
            cur_points = get_true(i, cur_label)
            for prev_idx, prev_label in enumerate(labels):
                prev_points = get_true(i - 1, prev_label)
                displacements = []
                for team in (team_a, team_b):
                    displacements.extend(
                        hungarian_distances(prev_points.outfield[team], cur_points.outfield[team])
                    )
                transition_cost = statistics.median(displacements) if displacements else 0.0
                total = dp[i - 1][prev_idx][0] + transition_weight * transition_cost
                if total < best_cost:
                    best_cost = total
                    best_prev = prev_idx
            anchor_cost = anchor_costs[i][cur_idx]
            dp[i][cur_idx] = (best_cost + anchor_cost, best_prev)

    # Backtrack from whichever final state is cheaper.
    final_idx = 0 if dp[-1][0][0] <= dp[-1][1][0] else 1
    path = [final_idx]
    for i in range(n - 1, 0, -1):
        final_idx = dp[i][final_idx][1]
        path.append(final_idx)
    path.reverse()
    return [labels[idx] for idx in path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", type=int, default=3902240)
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--start-time", default="35:15")
    parser.add_argument("--end-time", default="35:58")
    parser.add_argument("--sequence-relabel", action="store_true",
                         help="Also run the DP sequence-level relabeling prototype.")
    parser.add_argument("--cross-match", type=int, nargs="*", default=[],
                         help="Additional match IDs for whole-match cross validation.")
    args = parser.parse_args()

    try:
        frames = select_window(args.match_id, args.period, args.start_time, args.end_time)
    except VisualizationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    team_a, team_b = resolve_two_teams(frames)
    print(f"Match {args.match_id}: teams = {team_a} / {team_b}")
    print(f"Window: period {args.period}, [{args.start_time}, {args.end_time}) -> {len(frames)} frames\n")

    rows = build_transition_table(frames, team_a, team_b)
    print_transition_table(rows)

    flagged_pairs = [
        (4, 5), (10, 11), (11, 12), (14, 15), (15, 16), (16, 17),
        (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25),
    ]
    print("\n\n===================== forensic detail on flagged transitions =====================")
    for i, j in flagged_pairs:
        row = next((r for r in rows if r["frame_i"] == i and r["frame_j"] == j), None)
        if row:
            print_forensic_case(frames, row, team_a, team_b)

    if args.sequence_relabel:
        print("\n\n===================== sequence-level DP relabel (prototype) =====================")
        labels = sequence_level_relabel(frames, team_a, team_b)
        disagreements = 0
        for idx, (joined, label) in enumerate(zip(frames, labels), start=1):
            current = classify_orientation(joined.event, joined.frame_360)
            agree = "" if current.orientation == label else "  <-- DISAGREES WITH PER-EVENT CLASSIFIER"
            if agree:
                disagreements += 1
            print(f"  frame {idx:>2}: per-event={current.orientation:<12} "
                  f"sequence-DP={label:<12}{agree}")
        print(f"\nTotal disagreements in this window: {disagreements}/{len(frames)}")

    for match_id in args.cross_match:
        print(f"\n\n===================== cross-match summary: {match_id} =====================")
        try:
            match = load_and_join_match(match_id)
        except VisualizationError as error:
            print(f"Error: {error}", file=sys.stderr)
            continue
        team_a2, team_b2 = resolve_two_teams(match.usable_frames)
        all_rows = build_transition_table(match.usable_frames, team_a2, team_b2)
        likely_rotation = [r for r in all_rows if r["final_diagnosis"] == "LIKELY_ROTATION"]
        ambiguous = [r for r in all_rows if r["final_diagnosis"] == "AMBIGUOUS"]
        print(f"Total consecutive pairs: {len(all_rows)}")
        print(f"LIKELY_ROTATION: {len(likely_rotation)} ({100*len(likely_rotation)/len(all_rows):.2f}%)")
        print(f"AMBIGUOUS: {len(ambiguous)} ({100*len(ambiguous)/len(all_rows):.2f}%)")
        # Cross-check against the current per-event classifier for the FIRST
        # frame of each LIKELY_ROTATION pair: does the current classifier
        # already agree (say ROTATED_180 or UNCERTAIN), or does it confidently
        # (and, per this new evidence, wrongly) call it NORMAL?
        contradicted = 0
        for r in likely_rotation:
            if r["current_orientation_i"] == ORIENTATION_NORMAL and r["current_confidence_i"] >= 0.45:
                contradicted += 1
        print(f"Of those, current classifier confidently said NORMAL for frame_i: {contradicted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
