#!/usr/bin/env python3
"""Reusable diagnostic report for the StatsBomb 360 orientation classifier.

This script contains NO classification logic of its own. All detection and
correction logic lives in src/orientation_correction.py (the single source
of truth shared with the dataset builder and the animation script); this
script only loads matches, calls that module, and prints validation reports:

  - per-event-type NORMAL / ROTATED_180 / UNCERTAIN / confidence breakdown
  - named-case inspection (e.g. specific event UUID prefixes)
  - before/after consistency across a whole ROTATED_180 population
  - a false-positive scan (near pitch-center events, borderline confidence)

Usage:
    python scripts/diagnose_orientation.py --match-ids 3902240 3857254 3788741
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from animate_statsbomb_360_match import (  # noqa: E402
    JoinedFrame,
    VisualizationError,
    integer_event_field,
    load_and_join_match,
    nested_name,
    timestamp_seconds,
)
from orientation_correction import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    ORIENTATION_NORMAL,
    ORIENTATION_ROTATED_180,
    ORIENTATION_UNCERTAIN,
    FrameGeometry,
    OrientationResult,
    Point,
    classify_orientations_for_match,
    extract_geometry,
    mirror_point,
)

Case = tuple[JoinedFrame, FrameGeometry, OrientationResult]


# Small, generic geometry utilities used only for reporting below -- NOT the
# classification algorithm itself, which lives entirely in
# src/orientation_correction.py and is never duplicated here.
def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _centroid(points: list[Point]) -> Point:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def classify_match(match_id: int) -> list[Case]:
    """Load one match and classify every usable event/360-frame pair."""
    match = load_and_join_match(match_id)
    frames = match.usable_frames
    events = [joined.event for joined in frames]
    frames_360 = [joined.frame_360 for joined in frames]

    timestamps = [timestamp_seconds(event.get("timestamp")) for event in events]
    periods = [integer_event_field(event, "period") for event in events]

    geometries = [extract_geometry(event, frame) for event, frame in zip(events, frames_360)]
    results = classify_orientations_for_match(events, frames_360, timestamps, periods)
    return list(zip(frames, geometries, results))


def print_type_summary(results: list[Case]) -> None:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "normal": 0, "rotated": 0, "uncertain": 0, "confidence_sum": 0.0}
    )
    for joined, _, result in results:
        etype = nested_name(joined.event, "type")
        row = stats[etype]
        row["total"] += 1
        row["confidence_sum"] += result.confidence
        if result.orientation == ORIENTATION_NORMAL:
            row["normal"] += 1
        elif result.orientation == ORIENTATION_ROTATED_180:
            row["rotated"] += 1
        else:
            row["uncertain"] += 1

    header = f"{'event_type':<18}{'total':>7}{'normal':>9}{'rotated_180':>13}{'uncertain':>11}{'avg_conf':>10}"
    print(header)
    print("-" * len(header))
    for etype, row in sorted(stats.items(), key=lambda kv: -kv[1]["total"]):
        avg_conf = row["confidence_sum"] / row["total"] if row["total"] else 0.0
        print(
            f"{etype:<18}{row['total']:>7}{row['normal']:>9}{row['rotated']:>13}"
            f"{row['uncertain']:>11}{avg_conf:>10.3f}"
        )


def find_by_prefix(results: list[Case], prefix: str) -> Case | None:
    return next((r for r in results if r[0].event.get("id", "").startswith(prefix)), None)


def print_case(label: str, case: Case) -> None:
    joined, geometry, result = case
    event = joined.event
    print(f"--- {label}: event {event['id'][:8]} ({nested_name(event,'type')}) at {event.get('timestamp')} ---")
    print(f"  event.location: {geometry.event_location}")
    print(f"  actor (raw):    {geometry.actor_location}")
    print(f"  decision: {result.orientation}  confidence={result.confidence:.3f}  "
          f"best_distance={result.best_distance:.2f}  resolved_by={result.resolved_by}")
    for v in result.votes:
        print(f"    signal={v.name:<16} vote={v.vote:+.3f} weight={v.weight:.2f}"
              + (f" d_normal={v.distance_normal:.2f} d_rotated={v.distance_rotated:.2f}"
                 if v.distance_normal is not None else ""))


def validate_rotation_consistency(results: list[Case], sample_size: int = 8) -> None:
    """Show that rotation improves the WHOLE snapshot, not just the actor."""
    rotated_cases = [r for r in results if r[2].orientation == ORIENTATION_ROTATED_180]
    print(f"Total ROTATED_180 cases: {len(rotated_cases)} (showing {min(sample_size, len(rotated_cases))})")
    header = (
        f"{'event':<10}{'actor_before':>14}{'actor_after':>13}"
        f"{'centroid_before':>17}{'centroid_after':>16}"
        f"{'nearest_before':>16}{'nearest_after':>15}"
    )
    print(header)
    for joined, geometry, result in rotated_cases[:sample_size]:
        loc = geometry.event_location
        actor_before = _dist(loc, geometry.actor_location) if geometry.actor_location else float("nan")
        actor_after = _dist(loc, mirror_point(geometry.actor_location)) if geometry.actor_location else float("nan")
        if geometry.points:
            c = _centroid(list(geometry.points))
            centroid_before = _dist(loc, c)
            centroid_after = _dist(loc, mirror_point(c))
            nearest_before = min(_dist(loc, p) for p in geometry.points)
            nearest_after = min(_dist(loc, mirror_point(p)) for p in geometry.points)
        else:
            centroid_before = centroid_after = nearest_before = nearest_after = float("nan")
        print(
            f"{joined.event['id'][:8]:<10}{actor_before:>14.2f}{actor_after:>13.2f}"
            f"{centroid_before:>17.2f}{centroid_after:>16.2f}"
            f"{nearest_before:>16.2f}{nearest_after:>15.2f}"
        )


def false_positive_scan(results: list[Case]) -> None:
    """Near-center / ambiguous events must not be force-classified."""
    center = (60.0, 40.0)
    near_center = [
        r for r in results
        if r[1].event_location is not None and _dist(r[1].event_location, center) < 15.0
    ]
    forced = [r for r in near_center if r[2].orientation != ORIENTATION_UNCERTAIN]
    print(f"Events within 15 units of pitch center: {len(near_center)}")
    print(f"  Classified NORMAL/ROTATED_180 (i.e. NOT left uncertain): {len(forced)}")
    for joined, geometry, result in forced[:10]:
        print(
            f"  event {joined.event['id'][:8]} loc={geometry.event_location} "
            f"-> {result.orientation} (confidence={result.confidence:.3f}, "
            f"best_distance={result.best_distance:.2f})"
        )
    low_margin = [r for r in results if 0.20 <= r[2].confidence < CONFIDENCE_THRESHOLD]
    print(f"Borderline-margin events correctly left UNCERTAIN (0.20<=confidence<{CONFIDENCE_THRESHOLD}): "
          f"{len(low_margin)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-ids", type=int, nargs="+", default=[3902240, 3857254, 3788741])
    args = parser.parse_args()

    all_results: dict[int, list[Case]] = {}
    for match_id in args.match_ids:
        try:
            all_results[match_id] = classify_match(match_id)
        except VisualizationError as error:
            print(f"Error on match {match_id}: {error}", file=sys.stderr)
            return 1

    for match_id, results in all_results.items():
        print(f"\n===================== match {match_id}: per-event-type summary =====================")
        print_type_summary(results)

    print("\n===================== named case inspection (match 3902240) =====================")
    results_3902240 = all_results.get(3902240)
    if results_3902240:
        for label, prefix in (
            ("frame 19 (Pass/Cross, Spain)", "4f2e844a"),
            ("frame 20 (Pass/Recovery, Netherlands)", "b646dfea"),
            ("frame 15 (Pressure, Sherida Spitse)", "904e8a30"),
            ("frame 16 (Dribbled Past, Sherida Spitse)", "0d3afccf"),
        ):
            case = find_by_prefix(results_3902240, prefix)
            if case:
                print_case(label, case)

    print("\n===================== before/after consistency (match 3902240) =====================")
    if results_3902240:
        validate_rotation_consistency(results_3902240)

    print("\n===================== false-positive scan (match 3902240) =====================")
    if results_3902240:
        false_positive_scan(results_3902240)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
