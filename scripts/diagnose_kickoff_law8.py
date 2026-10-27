"""Law 8 kick-off diagnostic (DIAGNOSTIC ONLY).

IFAB Law 8: at a kick-off every player is in his own half (the kicker on
the centre spot). A kick-off snapshot whose teammate flags put several
players deep in the wrong half points at inconsistent raw labels (frame 505,
match 3930164). This script only reports; it never recolours players,
reassigns teams, edits raw data or excludes dataset rows, and it is not
used by the dataset builder or the renderer.

Per kick-off (a Pass of type "Kick Off" with a 360 snapshot):

  inconclusive_orientation  the snapshot's frame is not known independently
                            of the flags: only the record-level geometry,
                            an EARLIER copy of the same snapshot, or an
                            earlier neighbour count (src/orientation_resolution
                            causal routes). The kick-off own-half rule is
                            circular here -- it uses flags + Law 8 itself.
  inconclusive_few_players  fewer than MIN_EVALUATED players can be judged
  consistent                no player in the wrong half beyond the margin
  minor                     wrong-half players only within DEEP_YD of the
                            line, or fewer than MIN_DEEP deep ones: plausible
                            movement after the ball was played (the snapshot
                            may be captured slightly after the kick)
  suspicious                >= MIN_DEEP players more than DEEP_YD into the
                            wrong half

Players within LINE_MARGIN_YD of the halfway line and the actor are not
judged. Outputs: <out>/kickoffs.csv, <out>/summary.json

Usage:
    .venv/bin/python scripts/diagnose_kickoff_law8.py [--out results/ml_data_integrity_final_2026-10-27/law8]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LINE_MARGIN_YD = 2.0
DEEP_YD = 5.0
MIN_DEEP = 3
MIN_EVALUATED = 6
INDEPENDENT_ROUTES = frozenset({"geometry", "causal_duplicate_snapshot", "causal_temporal_neighbor"})


def classify_kickoff(players: list[dict[str, Any]], orientation_known: bool) -> dict[str, Any]:
    """Judge one kick-off snapshot given in the KICKING team's own frame
    (that team attacks toward x = 120, so its own half is x < 60).
    `players`: freeze_frame entries (read only). Returns the verdict and the
    wrong-half players (index, depth in yards)."""
    if not orientation_known:
        return {"verdict": "inconclusive_orientation", "evaluated": 0, "wrong_half": []}
    evaluated, wrong = 0, []
    for index, player in enumerate(players):
        location = player.get("location") if isinstance(player, dict) else None
        if not isinstance(location, list) or len(location) != 2 or player.get("actor"):
            continue
        depth = location[0] - 60.0 if player.get("teammate") else 60.0 - location[0]
        if abs(location[0] - 60.0) < LINE_MARGIN_YD:
            continue
        evaluated += 1
        if depth > 0:
            wrong.append((index, round(depth, 2)))
    deep = [w for w in wrong if w[1] > DEEP_YD]
    if evaluated < MIN_EVALUATED:
        verdict = "inconclusive_few_players"
    elif len(deep) >= MIN_DEEP:
        verdict = "suspicious"
    elif wrong:
        verdict = "minor"
    else:
        verdict = "consistent"
    return {"verdict": verdict, "evaluated": evaluated, "wrong_half": wrong, "deep": len(deep)}


def diagnose_match(match_id: int) -> list[dict[str, Any]]:
    from src.orientation_correction import apply_correction
    from src.orientation_resolution import ordered_usable_records, resolve_causal_orientations

    data = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
    events = json.loads((data / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((data / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    records = ordered_usable_records(events, frames).usable_frames
    orientation = resolve_causal_orientations([(r.event, r.frame_360) for r in records])
    rows = []
    for record in records:
        event = record.event
        if ((event.get("pass") or {}).get("type") or {}).get("name") != "Kick Off":
            continue
        result = orientation[event["id"]]
        known = result.resolved_by in INDEPENDENT_ROUTES
        own = apply_correction(record.frame_360, result.orientation)["freeze_frame"] if known else record.frame_360["freeze_frame"]
        verdict = classify_kickoff(own, known)
        rows.append({
            "match_id": match_id, "period": event["period"], "timestamp": event["timestamp"], "event_id": event["id"],
            "team": event["team"]["name"], "players": len(own), "orientation_route": result.resolved_by,
            "verdict": verdict["verdict"], "evaluated": verdict["evaluated"],
            "wrong_half": len(verdict["wrong_half"]), "wrong_half_deeper_than_5yd": verdict.get("deep", 0),
            "max_depth_yd": max((d for _, d in verdict["wrong_half"]), default=0.0),
        })
    return rows


def main() -> int:
    from src.animate_statsbomb_360_match import available_360_match_ids
    from src.match_exclusions import load_exclusion_manifest

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "ml_data_integrity_final_2026-10-27" / "law8")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    excluded = load_exclusion_manifest()
    ids = [m for m in available_360_match_ids() if m not in excluded]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for out in pool.map(diagnose_match, ids, chunksize=2) for r in out]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "kickoffs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verdicts = Counter(r["verdict"] for r in rows)
    suspicious = [r for r in rows if r["verdict"] == "suspicious"]
    summary = {
        "kickoffs_with_snapshot": len(rows), "verdicts": dict(verdicts),
        "candidate_frames (orientation independently known, enough players)":
            sum(v for k, v in verdicts.items() if not k.startswith("inconclusive")),
        "suspicious_frames": len(suspicious),
        "player_observations_in_wrong_half_suspicious_frames": sum(r["wrong_half"] for r in suspicious),
        "player_observations_in_wrong_half_all_judged_frames": sum(r["wrong_half"] for r in rows if not r["verdict"].startswith("inconclusive")),
        "examples": sorted(suspicious, key=lambda r: -r["wrong_half_deeper_than_5yd"])[:10],
        "thresholds": {"line_margin_yd": LINE_MARGIN_YD, "deep_yd": DEEP_YD, "min_deep": MIN_DEEP, "min_evaluated": MIN_EVALUATED},
        "note": "diagnostic only: no recolouring, no reassignment, no exclusion; flags read as raw teammate values",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
