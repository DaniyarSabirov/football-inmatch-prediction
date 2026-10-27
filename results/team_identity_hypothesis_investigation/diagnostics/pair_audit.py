"""Dataset-wide diagnostic of near-simultaneous 360 records (investigation only).

For every match and every pair of distinct 360 records in the same period
with |dt| <= 100 ms (timestamps as integer milliseconds), this records:

RAW relation (event-relative coordinates and flags, see snapshot_relations.py):
  raw_coordinates  IDENTICAL / SHIFTED (identical up to one constant offset
                   vector, residual < 0.05 yd) / NEAR / MIRROR_* / DIFFERENT
  raw_flags        EQUAL / INVERTED / MIXED over optimally paired players
  spec_consistent  same team: unmirrored + EQUAL; opposing teams: mirrored + INVERTED

DRAWN relation (what the renderer shows: Problem A + display rules + display
plan + Problem C with the display-only continuity refinement):
  drawn pairing by optimal assignment; per player the ambiguity margin =
  (2nd-nearest distance) - (paired distance), both directions;
  colour_conflicts = paired players (<= 2 yd) whose drawn team differs.

PROPAGATION feasibility (Hypothesis B), per pair:
  same_scene       drawn mean pair distance < 0.75 yd, max < 2 yd,
                   >= 90% of the smaller snapshot paired
  unambiguous      every paired player has margin >= 1.0 yd
  direction        each record's drawn colours are compared with its own
                   nearest distinct snapshots before/after (<= 2 s, not part
                   of this pair, not at the same millisecond); the record
                   whose colours agree with its neighbours wins if it
                   disagrees on <= 10% and the other on >= 50% of paired
                   players with at least 4 disagreements, else "undecided".

Outputs (results/team_identity_hypothesis_investigation/):
  pairs.csv.gz           one row per pair (all thresholds derivable from dt_ms)
  pair_audit_meta.json   counts used to verify timestamp resolution

Usage:
    .venv/bin/python results/team_identity_hypothesis_investigation/diagnostics/pair_audit.py [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
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
from src.team_identity import TEAM_IDENTITY_AMBIGUOUS, resolve_absolute_team_for_point, snapshot_key  # noqa: E402

OUT = HERE.parent
MAX_DT_MS = 100
NEIGHBOUR_MAX_MS = 2000
PAIR_MAX = 2.0
MARGIN = 1.0


def match_points(a: np.ndarray, b: np.ndarray):
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return cost, rows, cols


def raw_relation(fa: dict, fb: dict) -> dict:
    a = np.array([p["location"] for p in fa["freeze_frame"]], dtype=float)
    b = np.array([p["location"] for p in fb["freeze_frame"]], dtype=float)
    best = None
    for mirrored in (False, True):
        bb = np.column_stack([120 - b[:, 0], 80 - b[:, 1]]) if mirrored else b
        cost, rows, cols = match_points(a, bb)
        d = cost[rows, cols]
        offset = (bb[cols] - a[rows]).mean(axis=0)
        residual = np.linalg.norm(bb[cols] - offset - a[rows], axis=1).max()
        candidate = (d.mean(), d.max(), residual, float(np.linalg.norm(offset)), rows, cols, mirrored)
        if best is None or min(candidate[0], candidate[2]) < min(best[0], best[2]):
            best = candidate
    mean, worst, residual, shift, rows, cols, mirrored = best
    if worst < 0.01:
        kind = "IDENTICAL"
    elif residual < 0.05 and shift < 6:
        kind = "SHIFTED"
    elif mean < 0.75 and worst < 2.0:
        kind = "NEAR"
    else:
        kind = "DIFFERENT"
    if kind != "DIFFERENT" and mirrored:
        kind = "MIRROR_" + kind
    equal = sum(fa["freeze_frame"][r]["teammate"] == fb["freeze_frame"][c]["teammate"] for r, c in zip(rows, cols))
    n = len(rows)
    flags = "EQUAL" if equal >= 0.9 * n else "INVERTED" if n - equal >= 0.9 * n else "MIXED"
    return {"raw_coordinates": kind, "raw_flags": flags, "raw_mean_distance": round(float(mean), 3),
            "raw_shift": round(shift, 3), "raw_equal_flags": int(equal), "raw_pairs": n}


def drawn(display, refined, raw_by_id, pair, k):
    event = display.usable_frames[k].event
    points, teams = [], []
    for p in display.usable_frames[k].frame_360.get("freeze_frame") or []:
        if isinstance(p.get("location"), list):
            points.append(p["location"])
            teams.append(resolve_absolute_team_for_point(p, event, refined.get(event["id"]), raw_by_id, pair))
    return np.array(points, dtype=float).reshape(-1, 2), teams


def colour_disagreement(pa, ta, pb, tb) -> tuple[int, int]:
    if len(pa) == 0 or len(pb) == 0:
        return 0, 0
    cost, rows, cols = match_points(pa, pb)
    paired = bad = 0
    for r, c in zip(rows, cols):
        if cost[r, c] > 3.0 or TEAM_IDENTITY_AMBIGUOUS in (ta[r], tb[c]):
            continue
        paired += 1
        bad += ta[r] != tb[c]
    return paired, bad


def audit_match(match_id: int) -> dict:
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    shared = compute_shared_group_info(raw)
    raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    pair = tuple(names[t] for t in match_team_ids(raw))
    refined, _ = refine_shared_sources_by_continuity(display, shared, raw_by_id, pair)
    frames = raw.usable_frames
    times = [ms(j.event["timestamp"]) for j in frames]
    keys = [snapshot_key(j.frame_360) for j in frames]
    cache: dict[int, tuple] = {}

    def get(k):
        if k not in cache:
            cache[k] = drawn(display, refined, raw_by_id, pair, k)
        return cache[k]

    def neighbour_evidence(k, exclude):
        """(paired, disagreeing) summed over nearest distinct snapshot before and after."""
        period = frames[k].event["period"]
        total_paired = total_bad = found = 0
        for step in (-1, 1):
            j = k + step
            while 0 <= j < len(frames) and frames[j].event["period"] == period and abs(times[j] - times[k]) <= NEIGHBOUR_MAX_MS:
                if j not in exclude and times[j] != times[k] and keys[j] != keys[k]:
                    pk, tk = get(k)
                    pj, tj = get(j)
                    p, b = colour_disagreement(pk, tk, pj, tj)
                    if p:
                        total_paired += p
                        total_bad += b
                        found += 1
                    break
                j += step
        return found, total_paired, total_bad

    rows_out = []
    for i in range(len(frames)):
        for k in range(i + 1, len(frames)):
            if frames[k].event["period"] != frames[i].event["period"] or times[k] - times[i] > MAX_DT_MS:
                break
            ea, eb = frames[i].event, frames[k].event
            byte_identical = keys[i] == keys[k]
            rel = raw_relation(frames[i].frame_360, frames[k].frame_360)
            same_team = ea["team"]["id"] == eb["team"]["id"]
            mirrored = rel["raw_coordinates"].startswith("MIRROR")
            spec = (not mirrored and rel["raw_flags"] == "EQUAL") if same_team else (mirrored and rel["raw_flags"] == "INVERTED")
            pa, ta = get(i)
            pb, tb = get(k)
            if len(pa) and len(pb):
                cost, r_idx, c_idx = match_points(pa, pb)
                d = cost[r_idx, c_idx]
                close = d <= PAIR_MAX
                margins = []
                for r, c in zip(r_idx, c_idx):
                    row = np.sort(cost[r]); col = np.sort(cost[:, c])
                    second = min(row[1] if len(row) > 1 else 99, col[1] if len(col) > 1 else 99)
                    margins.append(second - cost[r, c])
                margins = np.array(margins)
                coverage = close.sum() / min(len(pa), len(pb))
                conflicts = sum(1 for r, c, ok in zip(r_idx, c_idx, close)
                                if ok and TEAM_IDENTITY_AMBIGUOUS not in (ta[r], tb[c]) and ta[r] != tb[c])
                drawn_mean, drawn_max = float(d.mean()), float(d.max())
                ambiguous = int(((margins < MARGIN) & close).sum())
            else:
                coverage = conflicts = ambiguous = 0
                drawn_mean = drawn_max = 99.0
            same_scene = drawn_mean < 0.75 and drawn_max < 2.0 and coverage >= 0.9
            direction = ""
            if conflicts:
                exclude = {i, k}
                fa = neighbour_evidence(i, exclude)
                fb = neighbour_evidence(k, exclude)
                share = lambda e: e[2] / e[1] if e[1] else None
                sa, sb = share(fa), share(fb)
                if sa is not None and sb is not None:
                    if sa <= 0.1 and sb >= 0.5 and fb[2] >= 4:
                        direction = "A_trusted"
                    elif sb <= 0.1 and sa >= 0.5 and fa[2] >= 4:
                        direction = "B_trusted"
                    else:
                        direction = "undecided"
                else:
                    direction = "no_neighbour_evidence"
                evidence = f"A {fa[2]}/{fa[1]} ({fa[0]} nb); B {fb[2]}/{fb[1]} ({fb[0]} nb)"
            else:
                evidence = ""
            rows_out.append({
                "match_id": match_id, "period": ea["period"], "dt_ms": times[k] - times[i],
                "event_a": ea["id"], "type_a": ea["type"]["name"], "team_a": ea["team"]["name"], "time_a": ea["timestamp"],
                "event_b": eb["id"], "type_b": eb["type"]["name"], "team_b": eb["team"]["name"], "time_b": eb["timestamp"],
                "same_team": same_team, "byte_identical": byte_identical,
                "shared_problem_c": ea["id"] in shared and eb["id"] in shared and shared[ea["id"]].group_event_ids == shared[eb["id"]].group_event_ids,
                "problem_a_a": results[i].orientation, "problem_a_b": results[k].orientation,
                **rel, "spec_consistent": spec,
                "drawn_mean_distance": round(drawn_mean, 3), "drawn_max_distance": round(drawn_max, 3),
                "coverage": round(float(coverage), 3), "same_scene": same_scene,
                "ambiguous_players": ambiguous, "players_a": len(pa), "players_b": len(pb),
                "colour_conflicts": int(conflicts), "direction": direction, "neighbour_evidence": evidence,
            })
    return {"match_id": match_id, "rows": rows_out,
            "nonzero_sub_ms": sum(1 for j in frames if "." in j.event["timestamp"] and len(j.event["timestamp"].split(".")[1]) > 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--match-id", type=int, nargs="*")
    args = parser.parse_args()
    excluded = load_exclusion_manifest()
    ids = [m for m in (args.match_id or available_360_match_ids()) if m not in excluded]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(pool.map(audit_match, ids, chunksize=2))
    rows = [r for o in outputs for r in o["rows"]]
    path = OUT / ("pairs.csv.gz" if not args.match_id else "pairs_subset.csv.gz")
    with gzip.open(path, "wt", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {"matches": len(ids), "pairs": len(rows),
            "timestamps_with_sub_millisecond_digits": sum(o["nonzero_sub_ms"] for o in outputs)}
    (OUT / ("pair_audit_meta.json" if not args.match_id else "pair_audit_meta_subset.json")).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
