"""Independent validation of the empirical flag perspective with goalkeepers.

The `keeper` flag does not depend on `teammate`, and a keeper stands near
his own team's defended goal. For every record with a keeper point standing
within 30 yd of a goal line (drawn coordinates), the keeper's team is read
from the raw `teammate` flag in both ways (own-team reading = spec, and the
opposing reading); the reading is "keeper-consistent" if it assigns the
keeper to the team the display plan has defending that goal.

`current_drawn_consistent` uses the renderer's current team for the keeper
(Problem C plus the display-only continuity refinement).

Joined with record_perspective.csv.gz this checks whether OWN / OPPOSING /
MIXED records behave as the neighbour-based inference claims, without using
neighbours at all. Output: keeper_validation.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.animate_statsbomb_360_match import (  # noqa: E402
    compute_shared_group_info,
    refine_shared_sources_by_continuity,
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    available_360_match_ids,
    compute_orientation_results,
    display_orientation_results,
    frame_review_corrections,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    match_team_ids,
    resolve_display_plan,
)
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.team_identity import resolve_absolute_team_for_point  # noqa: E402

OUT = HERE.parent


def audit_match(match_id: int) -> list[dict]:
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    team_ids = match_team_ids(raw)
    other = {names[team_ids[0]]: names[team_ids[1]], names[team_ids[1]]: names[team_ids[0]]}
    shared = compute_shared_group_info(raw)
    raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
    team_pair = tuple(names[t] for t in team_ids)
    refined, _ = refine_shared_sources_by_continuity(display, shared, raw_by_id, team_pair)
    rows = []
    for joined in display.usable_frames:
        event = joined.event
        transform = plan.transforms.get(event["period"])
        if transform is None or transform.reference_team_id is None:
            continue  # shootouts: both teams attack one goal
        defends_left = names[transform.reference_team_id]  # attacks +x, so defends x = 0
        own = event["team"]["name"]
        for player in joined.frame_360.get("freeze_frame") or []:
            if not player.get("keeper") or not isinstance(player.get("location"), list):
                continue
            x = player["location"][0]
            if 30 < x < 90:
                continue  # keeper far from both goals: no goal-side evidence
            goal_team = defends_left if x <= 30 else other[defends_left]
            own_reading = own if player.get("teammate") else other[own]
            drawn_team = resolve_absolute_team_for_point(player, event, refined.get(event["id"]), raw_by_id, team_pair)
            rows.append({"match_id": match_id, "event_id": event["id"],
                         "own_reading_consistent": own_reading == goal_team,
                         "opposing_reading_consistent": own_reading != goal_team,
                         "current_drawn_consistent": drawn_team == goal_team,
                         "current_drawn_ambiguous": drawn_team not in other})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    excluded = load_exclusion_manifest()
    ids = [m for m in available_360_match_ids() if m not in excluded]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for out in pool.map(audit_match, ids, chunksize=2) for r in out]
    with gzip.open(OUT / "keeper_validation.csv.gz", "wt", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "keeper observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
