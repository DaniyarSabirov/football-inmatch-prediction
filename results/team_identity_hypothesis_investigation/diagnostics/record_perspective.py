"""Empirical flag perspective of every 360 record (investigation only).

For each record, its RAW teammate flags are read in both possible ways --
relative to its own event team (the spec) and relative to the opposing team
-- and each reading is compared with the drawn colours of the record's
nearest neighbouring records before and after (same period, 100 ms <
|dt| <= 2000 ms, i.e. outside any same-moment cluster; the neighbours'
colours are the current renderer's). Only unambiguous pairings (optimal
assignment, <= 3 yd, second-nearest candidate >= 1 yd further) count.

  perspective = OWN       own-team reading disagrees on <= max(1, 10%) and
                          the opposing reading disagrees on >= 50%, with >= 3
                          unambiguous pairs in total
                OPPOSING  the reverse
                MIXED     both readings disagree substantially (scramble)
                UNDETERMINED  < 3 unambiguous pairs or no neighbour

Output: record_perspective.csv.gz (one row per usable record)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from snapshot_relations import ms  # noqa: E402
from src.animate_statsbomb_360_match import (  # noqa: E402
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    available_360_match_ids,
    compute_orientation_results,
    compute_shared_group_info,
    display_orientation_results,
    frame_review_corrections,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    match_team_ids,
    refine_shared_sources_by_continuity,
    resolve_display_plan,
)
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.team_identity import TEAM_IDENTITY_AMBIGUOUS, resolve_absolute_team_for_point  # noqa: E402

OUT = HERE.parent


def unambiguous_pairs(a, b):
    if len(a) == 0 or len(b) == 0:
        return []
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    out = []
    for r, c in zip(rows, cols):
        if cost[r, c] > 3.0:
            continue
        second = min(np.partition(cost[r], 1)[1] if cost.shape[1] > 1 else 99.0,
                     np.partition(cost[:, c], 1)[1] if cost.shape[0] > 1 else 99.0)
        if second - cost[r, c] >= 1.0:
            out.append((r, c))
    return out


def audit_match(match_id: int) -> list[dict]:
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    shared = compute_shared_group_info(raw)
    raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    team_pair = tuple(names[t] for t in match_team_ids(raw))
    other = {team_pair[0]: team_pair[1], team_pair[1]: team_pair[0]}
    refined, _ = refine_shared_sources_by_continuity(display, shared, raw_by_id, team_pair)
    frames = display.usable_frames
    times = [ms(j.event["timestamp"]) for j in frames]
    points, drawn_teams, flags = [], [], []
    for joined in frames:
        event = joined.event
        players = [p for p in joined.frame_360.get("freeze_frame") or [] if isinstance(p.get("location"), list)]
        points.append(np.array([p["location"] for p in players], dtype=float).reshape(-1, 2))
        drawn_teams.append([resolve_absolute_team_for_point(p, event, refined.get(event["id"]), raw_by_id, team_pair)
                            for p in players])
        flags.append([bool(p.get("teammate")) for p in players])
    rows = []
    for k, joined in enumerate(frames):
        event = joined.event
        own = event["team"]["name"]
        neighbours = []
        for step in (-1, 1):
            j = k + step
            while 0 <= j < len(frames) and frames[j].event["period"] == event["period"] and abs(times[j] - times[k]) <= 2000:
                if abs(times[j] - times[k]) > 100:
                    neighbours.append(j)
                    break
                j += step
        readings = {
            "own": [own if f else other[own] for f in flags[k]],
            "opposing": [other[own] if f else own for f in flags[k]],
        }
        totals = {"own": [0, 0], "opposing": [0, 0]}
        for j in neighbours:
            for r, c in unambiguous_pairs(points[k], points[j]):
                if drawn_teams[j][c] == TEAM_IDENTITY_AMBIGUOUS:
                    continue
                for name, colours in readings.items():
                    totals[name][0] += 1
                    totals[name][1] += colours[r] != drawn_teams[j][c]
        paired = totals["own"][0]
        bad_own, bad_opp = totals["own"][1], totals["opposing"][1]
        if paired < 3 or not neighbours:
            perspective = "UNDETERMINED"
        elif bad_own <= max(1, 0.1 * paired) and bad_opp >= 0.5 * paired:
            perspective = "OWN"
        elif bad_opp <= max(1, 0.1 * paired) and bad_own >= 0.5 * paired:
            perspective = "OPPOSING"
        else:
            perspective = "MIXED"
        current_bad = sum(1 for j in neighbours for r, c in unambiguous_pairs(points[k], points[j])
                          if TEAM_IDENTITY_AMBIGUOUS not in (drawn_teams[k][r], drawn_teams[j][c])
                          and drawn_teams[k][r] != drawn_teams[j][c])
        rows.append({
            "match_id": match_id, "event_id": event["id"], "period": event["period"], "time_ms": times[k],
            "event_type": event["type"]["name"], "team": own, "problem_a": results[k].orientation,
            "shared_problem_c": event["id"] in shared,
            "neighbours": len(neighbours), "unambiguous_pairs": paired,
            "disagree_own_reading": bad_own, "disagree_opposing_reading": bad_opp,
            "flag_perspective": perspective, "current_drawn_disagreement": current_bad,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    excluded = load_exclusion_manifest()
    ids = [m for m in available_360_match_ids() if m not in excluded]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for out in pool.map(audit_match, ids, chunksize=2) for r in out]
    with gzip.open(OUT / "record_perspective.csv.gz", "wt", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
