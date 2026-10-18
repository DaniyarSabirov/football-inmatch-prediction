#!/usr/bin/env python3
"""Reusable diagnostic table for event <-> StatsBomb 360 freeze-frame matching.

Builds, for a given match/period/time-window, exactly the diagnostic table
described in the StatsBomb 360 investigation task:

    displayed_frame_index | event_uuid | event_timestamp | event_type |
    event_location | matched_frame_uuid | frame_time_diff_seconds |
    num_freeze_frame_players | visible_area | actor_in_freeze_frame |
    actor_inside_visible_area

It reuses the exact loading/joining/sorting/timing functions from
src/animate_statsbomb_360_match.py so the diagnostic always reflects the
real behavior of the animation pipeline, not a re-implementation of it.

Usage:
    python scripts/diagnose_360_frames.py --match-id 3902240 --period 1 \
        --start-time 35:15 --end-time 35:58
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from animate_statsbomb_360_match import (  # noqa: E402
    VisualizationError,
    coordinate_pair,
    load_and_join_match,
    resolve_interval_seconds,
    timestamp_seconds,
    visible_area_points,
)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test for the visible_area polygon."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside
    return inside


def build_diagnostic_rows(
    match_id: int,
    period: int,
    start_time: str,
    end_time: str,
) -> list[dict]:
    """Build one diagnostic row per displayed (event-joined) frame in the window."""
    match = load_and_join_match(match_id)
    start_seconds, end_seconds = resolve_interval_seconds(
        start_minute=0, end_minute=0, start_time=start_time, end_time=end_time
    )

    windowed = [
        joined
        for joined in match.usable_frames
        if joined.event.get("period") == period
        and start_seconds <= timestamp_seconds(joined.event.get("timestamp")) < end_seconds
    ]
    if not windowed:
        raise VisualizationError(
            f"No usable joined frames for match {match_id}, period {period}, "
            f"[{start_time}, {end_time})."
        )

    rows = []
    previous_time = None
    for display_index, joined in enumerate(windowed, start=1):
        event = joined.event
        frame = joined.frame_360
        event_uuid = event.get("id")
        frame_uuid = frame.get("event_uuid")
        event_time = timestamp_seconds(event.get("timestamp"))
        time_diff = None if previous_time is None else round(event_time - previous_time, 3)
        previous_time = event_time

        freeze_frame = frame.get("freeze_frame", [])
        actor = next(
            (p for p in freeze_frame if isinstance(p, dict) and p.get("actor") is True),
            None,
        )
        actor_present = actor is not None
        polygon = visible_area_points(frame.get("visible_area"))
        actor_inside_area = None
        if actor_present and polygon is not None:
            actor_location = coordinate_pair(actor.get("location"))
            if actor_location is not None:
                actor_inside_area = point_in_polygon(actor_location, polygon)

        rows.append(
            {
                "displayed_frame_index": display_index,
                "event_uuid": event_uuid,
                # Matching is keyed strictly on event_uuid == frame event_uuid,
                # so these must always match; a mismatch here would be a bug.
                "event_uuid_matches_frame_uuid": event_uuid == frame_uuid,
                "event_timestamp": event.get("timestamp"),
                "event_type": event.get("type", {}).get("name"),
                "event_location": event.get("location"),
                "matched_frame_uuid": frame_uuid,
                "frame_time_diff_seconds": time_diff,
                "num_freeze_frame_players": len(freeze_frame),
                "visible_area": frame.get("visible_area"),
                "actor_in_freeze_frame": actor_present,
                "actor_inside_visible_area": actor_inside_area,
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    """Print a compact, readable diagnostic table."""
    header = (
        f"{'idx':>3} | {'event_uuid':<8} | {'timestamp':<12} | {'type':<12} | "
        f"{'location':<20} | {'frame_uuid':<8} | {'dt(s)':>7} | {'#players':>8} | "
        f"{'area_pts':>8} | {'actor?':>6} | {'in_area?':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        location = row["event_location"]
        location_str = (
            f"[{location[0]:.1f}, {location[1]:.1f}]" if location else "None"
        )
        area = row["visible_area"]
        area_points = len(area) // 2 if area else 0
        dt = row["frame_time_diff_seconds"]
        dt_str = "—" if dt is None else f"{dt:.3f}"
        mismatch = "" if row["event_uuid_matches_frame_uuid"] else " !!MISMATCH!!"
        print(
            f"{row['displayed_frame_index']:>3} | "
            f"{row['event_uuid'][:8]:<8} | "
            f"{row['event_timestamp']:<12} | "
            f"{row['event_type']:<12} | "
            f"{location_str:<20} | "
            f"{row['matched_frame_uuid'][:8]:<8} | "
            f"{dt_str:>7} | "
            f"{row['num_freeze_frame_players']:>8} | "
            f"{area_points:>8} | "
            f"{str(row['actor_in_freeze_frame']):>6} | "
            f"{str(row['actor_inside_visible_area']):>8}"
            f"{mismatch}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", type=int, required=True)
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--start-time", required=True, help="MM:SS or HH:MM:SS")
    parser.add_argument("--end-time", required=True, help="MM:SS or HH:MM:SS")
    parser.add_argument("--csv", type=Path, help="Optional path to also write the table as CSV.")
    args = parser.parse_args()

    try:
        rows = build_diagnostic_rows(
            args.match_id, args.period, args.start_time, args.end_time
        )
    except VisualizationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_table(rows)

    mismatches = [row for row in rows if not row["event_uuid_matches_frame_uuid"]]
    print()
    print(f"Rows: {len(rows)}")
    print(f"UUID mismatches (matching bug indicator): {len(mismatches)}")

    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
