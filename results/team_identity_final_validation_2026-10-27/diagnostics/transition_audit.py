"""Independent audit of the residual team-colour transitions (final validation).

Reproduces the population of scripts/audit_team_identity_continuity.py
exactly (drawn coordinates = renderer pipeline; consecutive byte-identical
copies collapsed; consecutive DISTINCT snapshots of one period with
0 <= dt <= 2 s; players paired by src/team_identity.unambiguous_pairs, <= 3 yd;
unresolved points skipped) and records, per pair:

  - context: dt, pair distance, pairing margin (2nd-best candidate minus
    paired distance), the snapshots' common offset, goalkeeper/actor flags,
    whether the two snapshots' event teams differ, structural links, status
    of both points, orientation route of both records;
  - for every colour change ("transition") additionally the same player one
    snapshot further back (partner of A in the snapshot before A) and one
    further forward (partner of B after B), if unambiguously paired.

Patterns of a transition A(i) -> B(i+1):
  blip_A        before(A) == B's colour: A alone differs (A contradicted by
                both of its neighbours in this sequence)
  blip_B        after(B) == A's colour: B alone differs
  step          before(A) == A and after(B) == B: the colour switches and stays
  unanchored    the wider context is missing or ambiguous

Independent anchors (not used by the identity resolution to decide a team):
  actor         the event's actor point is that event's own team by the
                event data itself (`team` of the event that the actor
                performed); a transition where one side is an actor point in
                an unlinked record is anchored by event semantics
  keeper_goal   a goalkeeper within 30 yd of a goal belongs to the team
                defending it (own-frame coordinates)

Outputs (../): transitions.csv.gz (every transition), pair_strata.csv
(every evaluated pair, stratified counts), match_transitions.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.animate_statsbomb_360_match import (  # noqa: E402
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    available_360_match_ids,
    compute_orientation_results,
    display_orientation_results,
    frame_review_corrections,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    load_json_records,
    match_team_ids,
    resolve_display_plan,
    timestamp_seconds,
)
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.team_identity import (  # noqa: E402
    IDENTITY_UNRESOLVED,
    PAIR_MAX_DISTANCE,
    TEAM_IDENTITY_AMBIGUOUS,
    resolve_match_team_identity,
    snapshot_key,
    unambiguous_pairs,
)

MAX_DT = 2.0


def bucket(value: float, edges: list[float]) -> str:
    for edge in edges:
        if value <= edge:
            return f"<={edge}"
    return f">{edges[-1]}"


def pair_details(a: np.ndarray, b: np.ndarray, pairs):
    """distance, margin and the common offset, measured like unambiguous_pairs."""
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    close = cost[rows, cols] <= PAIR_MAX_DISTANCE + 3.0
    offset = np.median(b[cols[close]] - a[rows[close]], axis=0) if close.sum() >= 3 else np.zeros(2)
    if np.linalg.norm(offset) <= 0.5:
        offset = np.zeros(2)
    shifted = np.linalg.norm((a + offset)[:, None, :] - b[None, :, :], axis=2)
    out = {}
    for i, j in pairs:
        others = np.concatenate([np.delete(shifted[i], j), np.delete(shifted[:, j], i)])
        out[(i, j)] = (float(shifted[i, j]), float(others.min() - shifted[i, j]) if len(others) else 99.0)
    return out, float(np.linalg.norm(offset))


def audit_match(match_id: int) -> dict:
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    pair = tuple(names[t] for t in match_team_ids(raw))
    ids = resolve_match_team_identity(
        [(j.event, j.frame_360) for j in raw.usable_frames], [r.orientation for r in used], results, pair,
        load_json_records(raw.event_path, "event"),
    )
    own = apply_orientation_correction_to_match(raw, used)  # own-event frames (keeper goal side)

    frames, previous_key = [], None
    for index, (shown, raw_joined) in enumerate(zip(display.usable_frames, raw.usable_frames)):
        key = (shown.event["period"], snapshot_key(raw_joined.frame_360))
        if key == previous_key:
            continue
        previous_key = key
        identity = ids[raw_joined.event["id"]]
        keep = [i for i, p in enumerate(shown.frame_360.get("freeze_frame") or [])
                if isinstance(p.get("location"), list) and identity.statuses[i] != IDENTITY_UNRESOLVED]
        players = shown.frame_360["freeze_frame"]
        own_players = own.usable_frames[index].frame_360["freeze_frame"]
        team = raw_joined.event["team"]["name"]
        goal = []
        for i in keep:
            if players[i].get("keeper") and used[index].orientation != "UNCERTAIN" and shown.event["period"] != 5:
                x = own_players[i]["location"][0]
                goal.append(team if x <= 30 else (pair[1] if team == pair[0] else pair[0]) if x >= 90 else "")
            else:
                goal.append("")
        frames.append({
            "index": index, "event": raw_joined.event, "period": shown.event["period"],
            "t": timestamp_seconds(shown.event["timestamp"]),
            "points": np.array([players[i]["location"] for i in keep], dtype=float).reshape(-1, 2),
            "teams": [identity.teams[i] for i in keep], "status": [identity.statuses[i] for i in keep],
            "keeper": [players[i].get("keeper") is True for i in keep],
            "actor": [players[i].get("actor") is True for i in keep],
            "flag": [players[i].get("teammate") is True for i in keep],
            "goal": goal, "linked": identity.component is not None, "component": identity.component,
            "decision": identity.decision, "route": used[index].resolved_by, "problem_a": results[index].orientation,
        })

    links = []
    for a, b in zip(frames, frames[1:]):
        if a["period"] == b["period"] and 0 <= b["t"] - a["t"] <= MAX_DT:
            links.append({i: j for i, j in unambiguous_pairs(a["points"], b["points"])})
        else:
            links.append(None)

    strata: Counter = Counter()
    transitions = []
    for k, (a, b) in enumerate(zip(frames, frames[1:])):
        mapping = links[k]
        if not mapping:
            continue
        details, offset = pair_details(a["points"], b["points"], list(mapping.items()))
        dt = b["t"] - a["t"]
        same_team_events = a["event"]["team"]["id"] == b["event"]["team"]["id"]
        same_component = a["component"] is not None and a["component"] == b["component"]
        for i, j in mapping.items():
            if TEAM_IDENTITY_AMBIGUOUS in (a["teams"][i], b["teams"][j]):
                continue
            distance, margin = details[(i, j)]
            changed = a["teams"][i] != b["teams"][j]
            key = (
                bucket(dt, [0.0, 0.25, 0.5, 1.0, 2.0]), bucket(distance, [0.5, 1.0, 1.5, 2.0, 3.0]),
                bucket(margin, [1.5, 2.0, 3.0, 5.0]), bucket(offset, [0.0, 1.0, 3.0]),
                same_team_events, a["keeper"][i] or b["keeper"][j], a["actor"][i] or b["actor"][j],
                a["linked"] or b["linked"], same_component,
                a["route"] != "geometry" or b["route"] != "geometry",
                "corrected" if "CORRECTED" in a["status"][i] + b["status"][j] else "spec",
            )
            strata[key + (changed,)] += 1
            if not changed:
                continue
            before = after = ""
            if k > 0 and links[k - 1]:
                back = {jj: ii for ii, jj in links[k - 1].items()}
                if i in back and frames[k - 1]["teams"][back[i]] != TEAM_IDENTITY_AMBIGUOUS:
                    before = frames[k - 1]["teams"][back[i]]
            if k + 1 < len(links) and links[k + 1] and j in links[k + 1]:
                c = frames[k + 2]
                if c["teams"][links[k + 1][j]] != TEAM_IDENTITY_AMBIGUOUS:
                    after = c["teams"][links[k + 1][j]]
            ta, tb = a["teams"][i], b["teams"][j]
            if before == tb and after != ta:
                pattern = "blip_A"
            elif after == ta and before != tb:
                pattern = "blip_B"
            elif before == ta and after == tb:
                pattern = "step"
            elif before == tb and after == ta:
                pattern = "alternating"
            else:
                pattern = "unanchored"
            transitions.append({
                "match_id": match_id, "period": a["period"],
                "event_a": a["event"]["id"], "type_a": a["event"]["type"]["name"], "team_a": a["event"]["team"]["name"],
                "time_a": a["event"]["timestamp"],
                "event_b": b["event"]["id"], "type_b": b["event"]["type"]["name"], "team_b": b["event"]["team"]["name"],
                "time_b": b["event"]["timestamp"], "dt_s": round(dt, 3),
                "point_a": i, "point_b": j, "x_a": round(float(a["points"][i][0]), 2), "y_a": round(float(a["points"][i][1]), 2),
                "pair_distance": round(distance, 3), "pair_margin": round(margin, 3), "common_offset": round(offset, 3),
                "colour_a": ta, "colour_b": tb, "status_a": a["status"][i], "status_b": b["status"][j],
                "flag_a": a["flag"][i], "flag_b": b["flag"][j],
                "keeper_a": a["keeper"][i], "keeper_b": b["keeper"][j], "actor_a": a["actor"][i], "actor_b": b["actor"][j],
                "goal_a": a["goal"][i], "goal_b": b["goal"][j],
                "same_event_team": same_team_events, "linked_a": a["linked"], "linked_b": b["linked"],
                "same_component": same_component, "decision_a": a["decision"], "decision_b": b["decision"],
                "route_a": a["route"], "route_b": b["route"], "problem_a_a": a["problem_a"], "problem_a_b": b["problem_a"],
                "points_a": len(a["teams"]), "points_b": len(b["teams"]),
                "before_colour": before, "after_colour": after, "pattern": pattern,
                "snapshot_transitions_in_link": sum(a["teams"][ii] != b["teams"][jj] for ii, jj in mapping.items()
                                                    if TEAM_IDENTITY_AMBIGUOUS not in (a["teams"][ii], b["teams"][jj])),
            })
    return {"match_id": match_id, "strata": strata, "transitions": transitions,
            "snapshots": len(frames), "links": sum(1 for l in links if l is not None)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--match-id", type=int, nargs="*")
    args = parser.parse_args()
    excluded = load_exclusion_manifest()
    ids = [m for m in (args.match_id or available_360_match_ids()) if m not in excluded]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(pool.map(audit_match, ids, chunksize=1))
    suffix = "" if not args.match_id else "_subset"
    rows = [t for o in outputs for t in o["transitions"]]
    with gzip.open(OUT / f"transitions{suffix}.csv.gz", "wt", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    strata = Counter()
    for o in outputs:
        strata.update(o["strata"])
    names = ["dt_s", "pair_distance", "pair_margin", "common_offset", "same_event_team", "keeper_involved",
             "actor_involved", "either_linked", "same_component", "either_orientation_not_geometry",
             "either_point_corrected", "transition", "pairs"]
    with (OUT / f"pair_strata{suffix}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(names)
        for key, count in strata.items():
            writer.writerow([*key, count])
    with (OUT / f"match_transitions{suffix}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["match_id", "distinct_snapshots", "links", "transitions"])
        for o in outputs:
            writer.writerow([o["match_id"], o["snapshots"], o["links"], len(o["transitions"])])
    print(len(ids), "matches;", sum(strata.values()), "pairs;", len(rows), "transitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
