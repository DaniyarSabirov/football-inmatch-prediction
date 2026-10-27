"""Team-colour continuity audit: are players coloured consistently frame to frame?

Players move about a yard per second, so in two consecutive DISTINCT 360
snapshots of the same period at most MAX_DT seconds apart, almost every
player can be paired with his own earlier position. This script pairs the
drawn points of consecutive snapshots with src/team_identity.unambiguous_pairs
(optimal assignment after removing the snapshots' common offset; a pair only
counts if it is within MAX_PAIR_DISTANCE yards AND the next-best candidate is
at least 1 yd further -- in a crowded box the pairing of two adjacent players
is arbitrary, which produced most earlier false positives) and counts how
often a paired player's DRAWN team colour changes. Teams are resolved exactly
as the renderer draws them with --apply-team-identity-resolution (the shared
evidence-based resolution, src/team_identity.resolve_match_team_identity;
unresolved points are skipped), positions exactly as drawn (display plan +
display_orientation_results).

Changed 2026-10-26 (crowding-safe pairing, shared identities). The earlier
outputs in results/visualization_final_validation/team_identity_investigation/
were produced by the previous version and are left untouched; this version
writes to --output-dir.

A snapshot is flagged "colour_inconsistent" when, against EVERY available
neighbour (previous and next distinct snapshot within MAX_DT), at least
MIN_MISMATCHES paired players change colour and they are at least
MIN_MISMATCH_SHARE of the pairs. Single-neighbour flags (first/last snapshot
of a run) are reported separately. Nothing is recoloured; this only measures.

Outputs (--output-dir, default results/team_identity_correction_2026-10-26/continuity_audit/):
  colour_inconsistent_frames.csv, match_summary.csv, summary.json

Usage:
    .venv/bin/python scripts/audit_team_identity_continuity.py [--match-id N ...] [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.animate_statsbomb_360_match import (  # noqa: E402
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    available_360_match_ids,
    compute_orientation_results,
    compute_shared_group_info,
    display_orientation_results,
    frame_review_corrections,
    format_match_clock,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    load_json_records,
    match_team_ids,
    resolve_display_plan,
    timestamp_seconds,
)
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.team_identity import (  # noqa: E402
    TEAM_IDENTITY_AMBIGUOUS,
    resolve_match_team_identity,
    snapshot_key,
    unambiguous_pairs,
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "team_identity_correction_2026-10-26" / "continuity_audit"
MAX_DT = 2.0
MAX_PAIR_DISTANCE = 3.0
MIN_MISMATCHES = 4
MIN_MISMATCH_SHARE = 0.3


def drawn_frames(match_id: int):
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    shared = compute_shared_group_info(raw)
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    pair = tuple(names[t] for t in match_team_ids(raw))
    # Exactly what the renderer draws (shared with the dataset builder).
    identities = resolve_match_team_identity(
        [(j.event, j.frame_360) for j in raw.usable_frames], [r.orientation for r in used], results, pair,
        load_json_records(raw.event_path, "event"),
    )
    switches: list[dict] = []
    frames = []
    previous_key = None
    for index, (shown, raw_joined) in enumerate(zip(display.usable_frames, raw.usable_frames)):
        key = (shown.event["period"], snapshot_key(raw_joined.frame_360))
        if key == previous_key:
            continue  # byte-identical copy of the previous snapshot: drawn identically
        previous_key = key
        points, teams = [], []
        resolved = identities[raw_joined.event["id"]].teams
        for i, player in enumerate(shown.frame_360.get("freeze_frame") or []):
            if not isinstance(player.get("location"), list):
                continue
            points.append(player["location"])
            teams.append(resolved[i])
        frames.append({
            "index": index, "event": shown.event, "period": shown.event["period"],
            "t": timestamp_seconds(shown.event["timestamp"]), "points": np.array(points, dtype=float).reshape(-1, 2),
            "teams": teams, "shared": raw_joined.event["id"] in shared,
            "problem_a": results[index].orientation, "display": used[index].orientation,
        })
    return frames, switches


def compare(a: dict, b: dict) -> tuple[int, int]:
    """(unambiguously paired players, those whose drawn team differs)."""
    paired = mismatched = 0
    for r, c in unambiguous_pairs(a["points"], b["points"], MAX_PAIR_DISTANCE):
        ta, tb = a["teams"][r], b["teams"][c]
        if TEAM_IDENTITY_AMBIGUOUS in (ta, tb):
            continue
        paired += 1
        mismatched += ta != tb
    return paired, mismatched


def audit_match(match_id: int) -> dict:
    frames, switches = drawn_frames(match_id)
    links = []
    for a, b in zip(frames, frames[1:]):
        if a["period"] == b["period"] and 0 <= b["t"] - a["t"] <= MAX_DT:
            links.append(compare(a, b))
        else:
            links.append(None)

    def bad(link):
        return link is not None and link[1] >= MIN_MISMATCHES and link[1] / max(link[0], 1) >= MIN_MISMATCH_SHARE

    rows, pair_stats = [], Counter()
    for link in links:
        if link is not None:
            pair_stats["pairs"] += 1
            pair_stats["paired_players"] += link[0]
            pair_stats["mismatched_players"] += link[1]
    for i, frame in enumerate(frames):
        before = links[i - 1] if i > 0 else None
        after = links[i] if i < len(links) else None
        available = [l for l in (before, after) if l is not None]
        if not available or not all(bad(l) for l in available):
            continue
        event = frame["event"]
        rows.append({
            "match_id": match_id, "period": frame["period"], "timestamp": event["timestamp"],
            "match_clock": format_match_clock(event), "seconds_into_period": round(frame["t"], 3),
            "event_id": event["id"], "event_type": event["type"]["name"], "team": event["team"]["name"],
            "is_kickoff": ((event.get("pass") or {}).get("type") or {}).get("name") == "Kick Off",
            "shared_cross_team_snapshot": frame["shared"],
            "problem_a": frame["problem_a"], "display_orientation": frame["display"],
            "neighbours": len(available),
            "vs_previous": "" if before is None else f"{before[1]}/{before[0]}",
            "vs_next": "" if after is None else f"{after[1]}/{after[0]}",
            "points": len(frame["teams"]),
        })
    return {"match_id": match_id, "rows": rows, "pair_stats": dict(pair_stats), "snapshots": len(frames),
            "switches": [{"match_id": match_id, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}}
                         for s in switches]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\\n")[0])
    parser.add_argument("--match-id", type=int, nargs="*")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir
    excluded = load_exclusion_manifest()
    match_ids = [m for m in (args.match_id or available_360_match_ids()) if m not in excluded]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(pool.map(audit_match, match_ids, chunksize=2))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for o in outputs for r in o["rows"]]
    for name, data in (("colour_inconsistent_frames.csv", rows),
                       ("match_summary.csv", [{"match_id": o["match_id"], "distinct_snapshots": o["snapshots"],
                                               "flagged": len(o["rows"]), **o["pair_stats"]} for o in outputs])):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as file:
            if data:
                writer = csv.DictWriter(file, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    switch_rows = [s for o in outputs for s in o["switches"]]
    with (output_dir / "shared_source_switches.csv").open("w", newline="", encoding="utf-8") as file:
        if switch_rows:
            writer = csv.DictWriter(file, fieldnames=list(switch_rows[0]))
            writer.writeheader()
            writer.writerows(switch_rows)
    totals = Counter()
    for o in outputs:
        totals.update(o["pair_stats"])
    summary = {
        "matches": len(match_ids),
        "distinct_snapshots": sum(o["snapshots"] for o in outputs),
        "neighbour_pairs_compared": totals["pairs"],
        "paired_players": totals["paired_players"],
        "paired_players_changing_colour": totals["mismatched_players"],
        "shared_source_switches": len(switch_rows),
        "flagged_frames": len(rows),
        "flagged_frames_both_neighbours": sum(r["neighbours"] == 2 for r in rows),
        "flagged_frames_single_neighbour": sum(r["neighbours"] == 1 for r in rows),
        "flagged_matches": len({r["match_id"] for r in rows}),
        "flagged_kickoffs": sum(r["is_kickoff"] for r in rows),
        "flagged_shared_cross_team": sum(r["shared_cross_team_snapshot"] for r in rows),
        "flagged_within_first_5s_of_period": sum(r["seconds_into_period"] <= 5 for r in rows),
        "flagged_by_event_type": dict(Counter(r["event_type"] for r in rows).most_common()),
        "thresholds": {"max_dt": MAX_DT, "max_pair_distance": MAX_PAIR_DISTANCE,
                       "min_mismatches": MIN_MISMATCHES, "min_mismatch_share": MIN_MISMATCH_SHARE,
                       "pairing": "unambiguous (margin >= 1 yd, common offset removed)"},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
