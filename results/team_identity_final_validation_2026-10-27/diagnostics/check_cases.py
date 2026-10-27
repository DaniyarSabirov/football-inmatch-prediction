"""Programmatic checks of the five reviewed cases with the final code.
Per case record and its previous/next snapshot (as in the review sheets):
coordinates identical with and without identity resolution (identity never
moves points), visible_area polygon present, actor point vs the event's own
location (an anchor independent of the flags), actor colour vs event team,
goalkeepers vs goal side, colour agreement with the neighbours (unambiguous
pairs) before and after the correction. Output: ../five_cases_checks.csv"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.animate_statsbomb_360_match import (  # noqa: E402
    apply_display_plan_to_match, apply_orientation_correction_to_match, compute_orientation_results,
    compute_shared_group_info, display_orientation_results, format_match_clock, frame_review_corrections,
    infer_broadcast_orientation_for_match, load_and_join_match, load_json_records, match_team_ids,
    refine_shared_sources_by_continuity, resolve_display_plan, visible_area_points,
)
from src.team_identity import TEAM_IDENTITY_AMBIGUOUS, resolve_absolute_team_for_point, resolve_match_team_identity, unambiguous_pairs  # noqa: E402

CASES = PROJECT_ROOT / "results" / "visualization_final_validation" / "team_identity_manual_review" / "review_index.csv"


def main():
    rows = []
    for case in csv.DictReader(CASES.open(encoding="utf-8")):
        match_id = int(case["match_id"])
        raw = load_and_join_match(match_id)
        results = compute_orientation_results(raw)
        inference = infer_broadcast_orientation_for_match(raw, results)
        plan, _, _ = resolve_display_plan(match_id, raw, inference)
        used, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
        display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
        names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
        pair = tuple(names[t] for t in match_team_ids(raw))
        raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
        legacy, _ = refine_shared_sources_by_continuity(display, compute_shared_group_info(raw), raw_by_id, pair)
        ids = resolve_match_team_identity([(j.event, j.frame_360) for j in raw.usable_frames], [r.orientation for r in used],
                                          results, pair, load_json_records(raw.event_path, "event"))
        frames = display.usable_frames
        k = next(i for i, j in enumerate(frames) if j.event["id"] == case["event_id"])
        prev_k = next(i for i, j in enumerate(frames) if format_match_clock(j.event) == case["previous_clock"] and i != k)
        next_k = next(i for i in range(len(frames) - 1, -1, -1) if format_match_clock(frames[i].event) == case["next_clock"] and i != k)

        def colours(i, after):
            event = frames[i].event
            if after:
                return list(ids[event["id"]].teams)
            return [resolve_absolute_team_for_point(p, event, legacy.get(event["id"]), raw_by_id, pair) for p in frames[i].frame_360["freeze_frame"]]

        def points(i):
            return np.array([p["location"] for p in frames[i].frame_360["freeze_frame"]], float)

        for tag, i in (("previous", prev_k), ("case", k), ("next", next_k)):
            event, frame = frames[i].event, frames[i].frame_360
            players = frame["freeze_frame"]
            actor = next((p for p in players if p.get("actor")), None)
            actor_index = players.index(actor) if actor else None
            out = {"case": case["case"], "match_id": match_id, "snapshot": tag, "clock": format_match_clock(event),
                   "event_type": event["type"]["name"], "event_team": event["team"]["name"], "players": len(players),
                   "polygon_present": visible_area_points(frame.get("visible_area")) is not None,
                   "orientation": used[i].orientation, "orientation_route": used[i].resolved_by,
                   "decision": ids[event["id"]].decision, "statuses": ";".join(sorted(set(ids[event["id"]].statuses)))}
            if actor is not None and isinstance(event.get("location"), list):
                out["actor_to_event_location_yd"] = round(math.dist(actor["location"], event["location"]), 2)
                out["actor_colour_before"] = colours(i, False)[actor_index]
                out["actor_colour_after"] = colours(i, True)[actor_index]
            keepers = [(n, p) for n, p in enumerate(players) if p.get("keeper")]
            checks = []
            for n, p in keepers:
                x = p["location"][0]
                transform = plan.transforms.get(event["period"])
                if transform is None or transform.reference_team_id is None or 30 < x < 90:
                    continue
                left = names[transform.reference_team_id]
                goal = left if x <= 30 else (pair[1] if left == pair[0] else pair[0])
                checks.append(f"before {'ok' if colours(i, False)[n] == goal else 'WRONG'} / after {'ok' if colours(i, True)[n] == goal else ('unclear' if colours(i, True)[n] == TEAM_IDENTITY_AMBIGUOUS else 'WRONG')}")
            out["keepers_goal_side"] = "; ".join(checks) or "no keeper near a goal"
            for side, other in (("vs_previous", prev_k), ("vs_next", next_k)):
                if other == i:
                    continue
                pairs = unambiguous_pairs(points(i), points(other))
                for when, after in (("before", False), ("after", True)):
                    a, b = colours(i, after), colours(other, after)
                    bad = sum(a[x] != b[y] for x, y in pairs if TEAM_IDENTITY_AMBIGUOUS not in (a[x], b[y]))
                    out[f"{side}_{when}"] = f"{bad}/{len(pairs)} disagree"
            rows.append(out)
    fields = sorted({key for row in rows for key in row}, key=lambda f: list(rows[0]).index(f) if f in rows[0] else 99)
    path = HERE.parent / "five_cases_checks.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print({k: v for k, v in row.items() if k not in ("match_id",)})


if __name__ == "__main__":
    main()
