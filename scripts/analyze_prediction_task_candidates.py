"""Read-only analysis supporting docs/prediction_task_feasibility_2026-10-14.md.

Builds the next-event dataset IN MEMORY (reusing build_rows_from_match_data
from src/build_statsbomb_360_dataset.py unchanged -- never a second
implementation of the row-construction logic) across every non-excluded
match, and reports the statistics needed to compare candidate prediction
tasks: next-event-type distribution, type-to-type transition frequencies,
time-to-next-event distribution at several horizons, 360 coverage of the
NEXT event (not computed by the production builder, added here read-only),
actor-position ambiguity frequency, and a reproducible sample of rows for
manual inspection.

Never writes to data/raw or data/processed. Writes only to results/ and
docs/ (explicitly, via --output flags), and to stdout.

Usage:
    .venv/bin/python scripts/analyze_prediction_task_candidates.py
    .venv/bin/python scripts/analyze_prediction_task_candidates.py --sample 60
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.animate_statsbomb_360_match import available_360_match_ids  # noqa: E402
from src.build_statsbomb_360_dataset import (  # noqa: E402
    MEANINGFUL_EVENT_TYPES,
    build_rows_from_match_data,
)
from src.match_exclusions import load_exclusion_manifest  # noqa: E402

EVENTS_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data" / "events"
FRAMES_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data" / "three-sixty"

# The builder applies no time window (removed 2026-10-27); the horizons
# below are applied afterwards, in Python, so the full gap distribution is
# visible and any horizon's effect can be measured rather than assumed.
HORIZONS = [2.0, 5.0, 10.0, 15.0, 30.0, 60.0]


def discover_processable_match_ids() -> list[int]:
    manifest = load_exclusion_manifest()
    return [mid for mid in available_360_match_ids() if mid not in manifest]


def load_match(match_id: int) -> tuple[list[dict], dict[str, dict]]:
    with (EVENTS_DIR / f"{match_id}.json").open(encoding="utf-8") as f:
        events = json.load(f)
    with (FRAMES_DIR / f"{match_id}.json").open(encoding="utf-8") as f:
        frames = json.load(f)
    frames_by_event_id = {fr["event_uuid"]: fr for fr in frames if "event_uuid" in fr}
    return events, frames_by_event_id


def run(match_ids: list[int]) -> dict:
    all_rows: list[dict] = []
    row_next_has_360: list[bool] = []
    errors = []

    for mid in match_ids:
        try:
            events, frames_by_event_id = load_match(mid)
            rows = build_rows_from_match_data(mid, events, frames_by_event_id)
        except Exception as error:  # noqa: BLE001
            errors.append({"match_id": mid, "error_type": type(error).__name__, "error_message": str(error)})
            continue
        for row in rows:
            row["_match_id_int"] = mid
            all_rows.append(row)
            row_next_has_360.append(row["next_event_id"] in frames_by_event_id)

    for row, has_360 in zip(all_rows, row_next_has_360):
        row["next_has_360"] = has_360

    return {"rows": all_rows, "errors": errors, "matches_processed": len(match_ids) - len(errors)}


def summarize(result: dict) -> dict:
    rows = result["rows"]
    n = len(rows)

    next_type_counts = Counter(r["next_event_type"] for r in rows)
    current_type_counts = Counter(r["current_event_type"] for r in rows)

    transitions = Counter((r["current_event_type"], r["next_event_type"]) for r in rows)

    gaps = [r["time_to_next_event_seconds"] for r in rows]
    gaps_sorted = sorted(gaps)

    def percentile(p: float) -> float:
        if not gaps_sorted:
            return float("nan")
        idx = min(len(gaps_sorted) - 1, int(p * len(gaps_sorted)))
        return gaps_sorted[idx]

    horizon_counts = {h: sum(1 for g in gaps if g <= h) for h in HORIZONS}

    actor_ambiguous_count = sum(1 for r in rows if r.get("actor_ambiguous") is True)
    next_has_360_count = sum(1 for r in rows if r.get("next_has_360"))

    # Candidate B: shot within next 10 seconds (same period, any
    # intervening meaningful events ignored -- defined precisely in the
    # report). Approximated here by scanning next_event_type=="Shot" OR,
    # more correctly, by checking whether ANY meaningful event within the
    # next 10s (not just the immediate next one) is a Shot. The immediate
    # next-event-only version is reported as "Candidate B (strict)"; a
    # chained version requires walking the event stream per row, done
    # separately below for exact sample counts.
    shot_as_immediate_next_within_10s = sum(
        1 for r in rows if r["next_event_type"] == "Shot" and r["time_to_next_event_seconds"] <= 10.0
    )

    return {
        "n_rows": n,
        "current_type_counts": dict(current_type_counts.most_common()),
        "next_type_counts": dict(next_type_counts.most_common()),
        "next_type_proportions": {k: round(v / n, 4) for k, v in next_type_counts.most_common()} if n else {},
        "top_transitions": [
            {"from": a, "to": b, "count": c, "proportion_of_from": round(c / current_type_counts[a], 4)}
            for (a, b), c in transitions.most_common(25)
        ],
        "gap_seconds": {
            "n": n,
            "mean": round(statistics.mean(gaps), 4) if gaps else None,
            "median": round(statistics.median(gaps), 4) if gaps else None,
            "p10": round(percentile(0.10), 4),
            "p25": round(percentile(0.25), 4),
            "p75": round(percentile(0.75), 4),
            "p90": round(percentile(0.90), 4),
            "p99": round(percentile(0.99), 4),
            "max": max(gaps) if gaps else None,
        },
        "horizon_counts": {str(h): horizon_counts[h] for h in HORIZONS},
        "horizon_proportions": {str(h): round(horizon_counts[h] / n, 4) for h in HORIZONS} if n else {},
        "actor_ambiguous_count": actor_ambiguous_count,
        "actor_ambiguous_proportion": round(actor_ambiguous_count / n, 6) if n else 0,
        "next_has_360_count": next_has_360_count,
        "next_has_360_proportion": round(next_has_360_count / n, 4) if n else 0,
        "shot_as_immediate_next_within_10s": shot_as_immediate_next_within_10s,
        "shot_as_immediate_next_within_10s_proportion": round(shot_as_immediate_next_within_10s / n, 6) if n else 0,
        "rows_kept_by_production_default_10s": sum(1 for g in gaps if g <= 10.0),
        "matches_processed": result["matches_processed"],
        "matches_errored": result["errors"],
    }


def build_shot_within_horizon_labels(match_ids: list[int], horizon_seconds: float = 10.0) -> dict:
    """Candidate B's TRUE target: does a Shot occur ANYWHERE in the event
    stream within `horizon_seconds` of the current event (same period),
    not just as the immediate next meaningful event. Computed directly
    from the raw, chronologically-ordered event stream (independent of
    MEANINGFUL_EVENT_TYPES row construction), so this is a distinct
    pass, not reused row-by-row from `run()`.
    """
    from src.build_statsbomb_360_dataset import timestamp_to_seconds

    total_candidates = 0
    positive = 0
    for mid in match_ids:
        try:
            events, frames_by_event_id = load_match(mid)
        except Exception:
            continue
        for i, event in enumerate(events):
            eid = event.get("id")
            if eid not in frames_by_event_id:
                continue
            if event["type"]["name"] not in MEANINGFUL_EVENT_TYPES:
                continue
            current_time = timestamp_to_seconds(event["timestamp"])
            current_period = event["period"]
            total_candidates += 1
            found_shot = False
            for candidate in events[i + 1:]:
                if candidate["period"] != current_period:
                    break
                dt = timestamp_to_seconds(candidate["timestamp"]) - current_time
                if dt > horizon_seconds:
                    break
                if candidate["type"]["name"] == "Shot":
                    found_shot = True
                    break
            if found_shot:
                positive += 1
    return {
        "horizon_seconds": horizon_seconds,
        "total_candidates": total_candidates,
        "positive": positive,
        "positive_proportion": round(positive / total_candidates, 6) if total_candidates else 0,
    }


def select_manual_inspection_sample(result: dict, n: int, seed: int) -> list[dict]:
    """Reproducible selection: stratify by a coarse bucket (event type x
    time-gap bucket x 360-availability) and sample deterministically within
    each stratum, rather than hand-picking visually interesting rows.
    """
    rows = result["rows"]

    def bucket(row: dict) -> tuple:
        gap = row["time_to_next_event_seconds"]
        gap_bucket = "short(<=2s)" if gap <= 2.0 else "medium(2-10s)" if gap <= 10.0 else "long(>10s)"
        return (row["current_event_type"], gap_bucket, row["next_has_360"], row["same_team"])

    by_bucket: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[bucket(row)].append(row)

    rng = random.Random(seed)
    strata = sorted(by_bucket.keys())
    rng.shuffle(strata)

    selected: list[dict] = []
    stratum_index = 0
    # Round-robin across strata, one row at a time, until n selected --
    # deterministic given `seed`, and guarantees diversity across type/
    # gap-length/360-availability/possession-change rather than random
    # global sampling, which would be dominated by the most common types.
    picked_per_stratum = {s: 0 for s in strata}
    while len(selected) < n and strata:
        progressed = False
        for s in strata:
            bucket_rows = sorted(by_bucket[s], key=lambda r: (r["_match_id_int"], r["current_event_id"]))
            idx = picked_per_stratum[s]
            if idx < len(bucket_rows):
                selected.append(bucket_rows[idx])
                picked_per_stratum[s] += 1
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None, help="Use a random sample of N matches instead of all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Path to save the full JSON summary.")
    parser.add_argument("--inspection-csv", type=Path, default=None, help="Path to save the manual-inspection sample as CSV.")
    parser.add_argument("--inspection-n", type=int, default=28)
    args = parser.parse_args()

    match_ids = discover_processable_match_ids()
    if args.sample is not None and args.sample < len(match_ids):
        rng = random.Random(args.seed)
        match_ids = sorted(rng.sample(match_ids, args.sample))

    print(f"Processing {len(match_ids)} non-excluded matches...", file=sys.stderr)
    result = run(match_ids)
    print(f"Done. Rows built: {len(result['rows'])}. Errors: {len(result['errors'])}", file=sys.stderr)

    summary = summarize(result)
    summary["shot_within_10s_anywhere_in_stream"] = build_shot_within_horizon_labels(match_ids, 10.0)
    summary["match_ids_used"] = match_ids

    print(json.dumps({k: v for k, v in summary.items() if k != "match_ids_used"}, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {args.output}", file=sys.stderr)

    if args.inspection_csv:
        sample_rows = select_manual_inspection_sample(result, args.inspection_n, args.seed)
        args.inspection_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "match_id", "current_event_id", "current_event_type", "current_team",
            "period", "timestamp", "next_event_id", "next_event_type", "next_team",
            "time_to_next_event_seconds", "same_team", "same_possession",
            "next_has_360", "actor_ambiguous", "visible_ambiguous", "orientation",
        ]
        with args.inspection_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sample_rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"Saved {len(sample_rows)} manual-inspection rows to {args.inspection_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
