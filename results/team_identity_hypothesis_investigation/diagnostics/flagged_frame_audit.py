"""Root-cause audit of every colour-inconsistent frame (investigation only).

Input: the frames flagged by scripts/audit_team_identity_continuity.py
(102 strong = both neighbours disagree, 2,094 weak = one neighbour) and
pairs.csv.gz from pair_audit.py.

For each flagged record, in DRAWN space (exactly what the renderer shows),
against its nearest DISTINCT snapshots before/after (same period, <= 2 s,
excluding its own same-scene partners), three candidate colourings are
evaluated -- none of them is applied anywhere:

  current   colours as drawn now
  invert    every point's team swapped (record-level perspective error:
            flags written from the opposing team's point of view)
  partner   colours transferred from the raw same-scene partner within
            100 ms (Hypothesis B): players paired by optimal assignment after
            removing the pair's constant offset; a player is transferred only
            if the pairing is unambiguous (2nd-nearest candidate >= 1 yd
            further than the chosen one) and within 1.5 yd; the rest keep
            their current colour

  owner     (Hypothesis A, structural, needs no neighbours) only when a raw
            same-scene partner of the OPPOSING team has EQUAL flags (a spec
            violation: one team's view copied into the other's record) and
            Problem A identifies exactly one of the two as holding the
            snapshot in its own frame (NORMAL, the other ROTATED_180 or
            UNCERTAIN): both records' flags are read relative to that team

An option RESOLVES the frame if, against every available neighbour, at most
max(1, 10%) of the UNAMBIGUOUSLY paired players (margin >= 1 yd) disagree
and at least 3 such players are paired. Ambiguous pairs (crowds) are
reported but not used for the verdict. Pairing to neighbours uses
optimal assignment with pairs <= 3 yd; players whose pairing margin is below
1 yd are reported separately (crowding indicator).

Output: flagged_frame_audit.csv (one row per flagged record).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import defaultdict
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
from src.team_identity import TEAM_IDENTITY_AMBIGUOUS, resolve_absolute_team_for_point, snapshot_key  # noqa: E402

OUT = HERE.parent
FLAGGED = PROJECT_ROOT / "results" / "visualization_final_validation" / "team_identity_investigation" / "colour_inconsistent_frames.csv"


def pair(a, b, max_distance):
    if len(a) == 0 or len(b) == 0:
        return []
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    out = []
    for r, c in zip(rows, cols):
        if cost[r, c] > max_distance:
            continue
        second_row = np.partition(cost[r], 1)[1] if cost.shape[1] > 1 else 99.0
        second_col = np.partition(cost[:, c], 1)[1] if cost.shape[0] > 1 else 99.0
        out.append((r, c, cost[r, c], min(second_row, second_col) - cost[r, c]))
    return out


def disagreement(points, teams, n_points, n_teams):
    pairs = pair(points, n_points, 3.0)
    paired = bad = bad_unambiguous = paired_unambiguous = 0
    for r, c, _, margin in pairs:
        if TEAM_IDENTITY_AMBIGUOUS in (teams[r], n_teams[c]):
            continue
        paired += 1
        bad += teams[r] != n_teams[c]
        if margin >= 1.0:
            paired_unambiguous += 1
            bad_unambiguous += teams[r] != n_teams[c]
    return paired, bad, paired_unambiguous, bad_unambiguous


def audit_match(job):
    match_id, targets, partners = job
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
    refined, _ = refine_shared_sources_by_continuity(display, shared, raw_by_id, team_pair)
    frames = display.usable_frames
    index = {j.event["id"]: k for k, j in enumerate(frames)}
    times = [ms(j.event["timestamp"]) for j in frames]
    keys = [snapshot_key(j.frame_360) for j in raw.usable_frames]

    def drawn(k):
        event = frames[k].event
        pts, tms = [], []
        for p in frames[k].frame_360.get("freeze_frame") or []:
            if isinstance(p.get("location"), list):
                pts.append(p["location"])
                tms.append(resolve_absolute_team_for_point(p, event, refined.get(event["id"]), raw_by_id, team_pair))
        return np.array(pts, dtype=float).reshape(-1, 2), tms

    swap = {team_pair[0]: team_pair[1], team_pair[1]: team_pair[0]}
    rows = []
    for target in targets:
        k = index[target["event_id"]]
        excluded = {index[p["id"]] for p in partners.get(target["event_id"], []) if p["id"] in index}
        neighbours = []
        for step in (-1, 1):
            j = k + step
            while 0 <= j < len(frames) and frames[j].event["period"] == frames[k].event["period"] \
                    and abs(times[j] - times[k]) <= 2000:
                if keys[j] != keys[k] and j not in excluded:
                    neighbours.append(j)
                    break
                j += step
        points, teams = drawn(k)
        inverted = [swap.get(t, t) for t in teams]
        options = {"current": teams, "invert": inverted}
        partner_used = ""
        transferred = 0
        partner_candidates = [index[p["id"]] for p in partners.get(target["event_id"], []) if p["id"] in index]
        owner_team = ""
        own_raw = raw.usable_frames[k].frame_360["freeze_frame"]
        for p in partners.get(target["event_id"], []):
            if p["cross"] and p["raw_flags"] == "EQUAL":
                mine, theirs = p["my_problem_a"], p["their_problem_a"]
                if mine == "NORMAL" and theirs != "NORMAL":
                    owner_team = frames[k].event["team"]["name"]
                elif theirs == "NORMAL" and mine != "NORMAL":
                    owner_team = p["their_team"]
                if owner_team:
                    break
        if owner_team and len(own_raw) == len(teams):
            options["owner"] = [owner_team if pl.get("teammate") else swap[owner_team] for pl in own_raw]
        if partner_candidates:
            best = None
            for q in partner_candidates:
                q_points, q_teams = drawn(q)
                if len(q_points) == 0:
                    continue
                first = pair(points, q_points, 99.0)
                if not first:
                    continue
                offset = np.mean([q_points[c] - points[r] for r, c, _, _ in first], axis=0)
                aligned = pair(points + offset, q_points, 1.5)
                if best is None or len(aligned) > len(best[1]):
                    best = (q, aligned, q_teams)
            if best is not None:
                q, aligned, q_teams = best
                colours = list(teams)
                for r, c, _, margin in aligned:
                    if margin >= 1.0 and q_teams[c] != TEAM_IDENTITY_AMBIGUOUS:
                        colours[r] = q_teams[c]
                        transferred += 1
                options["partner"] = colours
                partner_used = frames[q].event["type"]["name"] + " / " + frames[q].event["team"]["name"]
        evaluation = {}
        for name, colours in options.items():
            per = [disagreement(points, colours, *drawn(j)) for j in neighbours]
            resolves = bool(per) and all(pu >= 3 and bu <= max(1, 0.1 * pu) for _, _, pu, bu in per)
            evaluation[name] = (per, resolves)
        cur = evaluation["current"][0]
        rows.append({
            **{k_: target[k_] for k_ in ("match_id", "period", "timestamp", "match_clock", "event_id", "event_type",
                                          "team", "neighbours", "vs_previous", "vs_next", "shared_cross_team_snapshot",
                                          "problem_a")},
            "strength": "strong" if target["neighbours"] == "2" else "weak",
            "evaluated_neighbours": len(neighbours),
            "current_disagreement": "; ".join(f"{b}/{p} (unambiguous {bu}/{pu})" for p, b, pu, bu in cur),
            "current_disagreement_unambiguous_only": sum(bu for _, _, _, bu in cur),
            "current_resolves": evaluation["current"][1],
            "invert_resolves": evaluation["invert"][1],
            "invert_disagreement": "; ".join(f"{bu}/{pu} unambiguous" for _, _, pu, bu in evaluation["invert"][0]),
            "owner_team": owner_team,
            "owner_resolves": evaluation.get("owner", ([], False))[1],
            "owner_disagreement": "; ".join(f"{bu}/{pu} unambiguous" for _, _, pu, bu in evaluation.get("owner", ([], False))[0]),
            "partner": partner_used, "partner_players_transferred": transferred,
            "partner_resolves": evaluation.get("partner", ([], False))[1],
            "partner_disagreement": "; ".join(f"{bu}/{pu} unambiguous" for _, _, pu, bu in evaluation.get("partner", ([], False))[0]),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with FLAGGED.open(encoding="utf-8") as file:
        flagged = list(csv.DictReader(file))
    ids = {r["event_id"] for r in flagged}
    partners: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(OUT / "pairs.csv.gz", "rt", encoding="utf-8") as file:
        for r in csv.DictReader(file):
            if abs(int(r["dt_ms"])) > 100 or r["raw_coordinates"] == "DIFFERENT":
                continue
            for me, other in (("a", "b"), ("b", "a")):
                if r[f"event_{me}"] in ids:
                    partners[r[f"event_{me}"]].append({
                        "id": r[f"event_{other}"], "cross": r["same_team"] == "False", "raw_flags": r["raw_flags"],
                        "raw_coordinates": r["raw_coordinates"], "my_problem_a": r[f"problem_a_{me}"],
                        "their_problem_a": r[f"problem_a_{other}"], "their_team": r[f"team_{other}"],
                    })
    by_match: dict[int, list] = defaultdict(list)
    for r in flagged:
        by_match[int(r["match_id"])].append(r)
    jobs = [(m, t, {e["event_id"]: partners.get(e["event_id"], []) for e in t}) for m, t in by_match.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for out in pool.map(audit_match, jobs, chunksize=2) for r in out]
    with (OUT / "flagged_frame_audit.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "flagged records audited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
