"""Reproducibly select 5 random (match, period, time-interval) samples for
manual visual inspection of the renderer and physical-orientation
calibration system.

Read-only: loads only from the existing match inventory
(results/physical_orientation/match_inventory.csv, itself built read-only
from raw data) and raw event/360 JSON. Never writes to data/raw,
data/processed, or configs/.

Selection algorithm (seeded, reproducible):
  1. Shuffle the 417 eligible match ids with a fixed seed.
  2. For each candidate match, in shuffled order, try each of its
     present periods (shuffled) and up to MAX_START_ATTEMPTS random
     start times within that period, looking for a ~150s window with at
     least MIN_USABLE_FRAMES usable 360 frames and at least
     MIN_DISTINCT_EVENT_TYPES distinct event types (a simple, documented
     proxy for "actual player movement and different event types").
  3. Stop once 5 matches have a valid window; record every accepted
     selection AND every rejected attempt, so the process is fully
     auditable, not just the final result.

Usage:
    .venv/bin/python scripts/select_random_visual_check_samples.py --seed 20261022
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.animate_statsbomb_360_match import load_and_join_match, timestamp_seconds  # noqa: E402

INVENTORY_CSV = PROJECT_ROOT / "results" / "physical_orientation" / "match_inventory.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "physical_orientation" / "random_visual_checks"

WINDOW_SECONDS = 150.0  # 2.5 minutes, within the requested 2-3 minute range
MIN_USABLE_FRAMES = 20
MIN_DISTINCT_EVENT_TYPES = 3
MAX_START_ATTEMPTS = 40
TARGET_COUNT = 5


def load_eligible_match_ids() -> list[int]:
    ids = []
    with INVENTORY_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["eligible"] == "True":
                ids.append(int(row["match_id"]))
    return ids


def try_match(match_id: int, rng: random.Random, log: list[dict]) -> dict | None:
    try:
        match = load_and_join_match(match_id)
    except Exception as error:  # noqa: BLE001
        log.append({"match_id": match_id, "result": "load_failed", "error": str(error)})
        return None

    periods = sorted({j.event.get("period") for j in match.usable_frames})
    rng.shuffle(periods)

    for period in periods:
        period_frames = [j for j in match.usable_frames if j.event.get("period") == period]
        times = sorted(timestamp_seconds(j.event.get("timestamp")) for j in period_frames)
        if not times or times[-1] - times[0] < WINDOW_SECONDS:
            log.append({"match_id": match_id, "period": period, "result": "period_too_short"})
            continue

        for attempt in range(MAX_START_ATTEMPTS):
            start = rng.uniform(times[0], times[-1] - WINDOW_SECONDS)
            end = start + WINDOW_SECONDS
            window_frames = [
                j for j in period_frames
                if start <= timestamp_seconds(j.event.get("timestamp")) < end
            ]
            event_types = {j.event.get("type", {}).get("name") for j in window_frames}
            if len(window_frames) >= MIN_USABLE_FRAMES and len(event_types) >= MIN_DISTINCT_EVENT_TYPES:
                log.append({
                    "match_id": match_id, "period": period, "result": "accepted",
                    "start_second": round(start, 1), "end_second": round(end, 1),
                    "usable_frames_in_window": len(window_frames),
                    "distinct_event_types": sorted(event_types),
                })
                return {
                    "match_id": match_id,
                    "period": period,
                    "start_second": round(start, 1),
                    "end_second": round(end, 1),
                    "usable_frames_in_window": len(window_frames),
                    "distinct_event_types": sorted(event_types),
                }
            log.append({
                "match_id": match_id, "period": period, "attempt": attempt, "result": "window_rejected",
                "start_second": round(start, 1), "usable_frames_in_window": len(window_frames),
                "distinct_event_types": len(event_types),
            })

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()

    eligible_ids = load_eligible_match_ids()
    rng = random.Random(args.seed)
    shuffled = list(eligible_ids)
    rng.shuffle(shuffled)

    log: list[dict] = []
    selections: list[dict] = []

    for match_id in shuffled:
        if len(selections) >= TARGET_COUNT:
            break
        result = try_match(match_id, rng, log)
        if result is not None:
            selections.append(result)
            print(f"Accepted match {match_id}: period {result['period']}, "
                  f"[{result['start_second']}, {result['end_second']}]s, "
                  f"{result['usable_frames_in_window']} frames, "
                  f"types={result['distinct_event_types']}")

    if len(selections) < TARGET_COUNT:
        print(f"WARNING: only found {len(selections)}/{TARGET_COUNT} valid samples.", file=sys.stderr)
        return 1

    # Attach readable metadata from the inventory for the final report.
    meta_by_id = {}
    with INVENTORY_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta_by_id[int(row["match_id"])] = row
    for sel in selections:
        meta = meta_by_id[sel["match_id"]]
        sel["competition"] = meta["competition"]
        sel["season"] = meta["season"]
        sel["date"] = meta["date"]
        sel["home_team"] = meta["home_team"]
        sel["away_team"] = meta["away_team"]
        sel["stadium"] = meta["stadium"]
        sel["physical_verification_status"] = meta["physical_verification_status"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "selection.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "selections": selections, "full_log": log}, f, indent=2)

    print(f"\nSaved selection to {OUTPUT_DIR / 'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
