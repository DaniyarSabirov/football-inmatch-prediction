"""Read-only data-quality audit for the StatsBomb 360 pipeline.

Produces the data-quality report described in the project's data
integrity audit (see docs/). Reads ONLY raw JSON under data/raw/; never
writes to data/raw, data/interim, or data/processed. Refuses to overwrite
an existing --output path unless --overwrite is explicitly passed.

All counts in this report describe the RAW source data (events + 360 JSON
under data/raw/statsbomb/), computed independently of, and without reading,
any file under data/processed/ -- unless --compare-csv is explicitly
passed, in which case a small, clearly-labeled read-only cross-check
section is added comparing a few recomputed raw-derived counts against an
existing processed CSV (also never written to).

Usage:
    .venv/bin/python scripts/audit_dataset.py --output results/audit.json
    .venv/bin/python scripts/audit_dataset.py --sample 40          # faster, sampled run
    .venv/bin/python scripts/audit_dataset.py --output results/audit.json --overwrite
    .venv/bin/python scripts/audit_dataset.py --compare-csv data/processed/statsbomb_360_match_3902240_next_events.csv --compare-match-id 3902240
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import traceback
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.orientation_correction import classify_orientation  # noqa: E402
from src.team_identity import detect_shared_cross_team_snapshots  # noqa: E402
from src.match_exclusions import ExcludedMatch, load_exclusion_manifest  # noqa: E402

EVENTS_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data" / "events"
FRAMES_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data" / "three-sixty"
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
MAX_EXAMPLES = 25


def discover_match_ids(sample: int | None, seed: int = 0) -> list[int]:
    ids = sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in glob.glob(str(FRAMES_DIR / "*.json")))
    if sample is not None and sample < len(ids):
        rng = random.Random(seed)
        ids = sorted(rng.sample(ids, sample))
    return ids


def classify_coordinate(value) -> str:
    """Return 'missing', 'malformed', 'non_numeric', 'in_bounds', or one or
    more of 'x<0'/'x>120'/'y<0'/'y>80' joined by '+' for out-of-bounds.

    Kept as a single function (rather than a boolean) so "malformed" (wrong
    shape/type) is never silently folded into "out of bounds" (a
    well-formed numeric pair outside the pitch rectangle) -- the two mean
    very different things for data-quality purposes.
    """
    if value is None:
        return "missing"
    if not isinstance(value, list) or len(value) != 2:
        return "malformed"
    x, y = value
    if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return "non_numeric"
    reasons = []
    if x < 0:
        reasons.append("x<0")
    if x > PITCH_LENGTH:
        reasons.append("x>120")
    if y < 0:
        reasons.append("y<0")
    if y > PITCH_WIDTH:
        reasons.append("y>80")
    return "+".join(reasons) if reasons else "in_bounds"


def overshoot_magnitude(value) -> float:
    x, y = value
    return max(0.0, -x, x - PITCH_LENGTH, -y, y - PITCH_WIDTH)


def magnitude_bucket(overshoot: float) -> str:
    for ceiling, label in [(0.5, "<=0.5"), (1.0, "<=1.0"), (2.0, "<=2.0"), (5.0, "<=5.0"), (10.0, "<=10.0")]:
        if overshoot <= ceiling:
            return label
    return ">10.0"


def audit(match_ids: list[int], exclusion_manifest: dict[int, ExcludedMatch]) -> dict:
    """Run the audit. `exclusion_manifest` (see src/match_exclusions.py) is
    required, not optional with a silent empty default -- callers (main())
    must load it explicitly so a broken manifest fails before any match is
    processed, never partway through a long run.
    """
    report = {
        "matches_discovered": len(match_ids),
        "matches_excluded": 0,
        "excluded_matches": [],  # [{match_id, reason, file, evidence, date_investigated, report}]
        "matches_loaded_ok": 0,
        "matches_failed_to_load": 0,
        "failed_matches": [],  # [{match_id, file, error_type, error_message}] -- UNEXPECTED failures only, never an excluded match
        "event_records": 0,
        "three_sixty_records": 0,
        "matched_pairs": 0,
        "matches_with_zero_matched_pairs": [],  # non-empty events+frames files sharing 0 event ids -- see note below
        "events_without_360": 0,
        "duplicate_event_ids": 0,
        "duplicate_event_uuid_in_frames": 0,
        "index_not_strictly_increasing_matches": 0,
        "events_missing_team": 0,
        "events_missing_player": 0,
        "events_missing_player_by_type": Counter(),
        "events_missing_location_by_type": Counter(),
        "event_location_category_counts": Counter(),  # missing/malformed/non_numeric/in_bounds/x<0 etc.
        "freeze_frame_point_category_counts": Counter(),
        "freeze_frame_oob_magnitude_buckets": Counter(),
        "freeze_frame_points_total": 0,
        "freeze_frame_oob_unique_matches": set(),
        "freeze_frame_oob_unique_events": set(),
        "frames_with_zero_actor_points": 0,
        "frames_with_multiple_actor_points": 0,
        "frames_empty_freeze_frame": 0,
        "zero_actor_examples": [],
        "multi_actor_examples": [],
        "event_type_counts": Counter(),
        "period_counts": Counter(),
        "orientation_counts": Counter(),
        "shared_cross_team_groups": 0,
        "shared_cross_team_resolved": 0,
        "shared_cross_team_ambiguous": 0,
    }

    for mid in match_ids:
        excluded = exclusion_manifest.get(mid)
        if excluded is not None:
            # Checked BEFORE any file I/O -- an explicitly excluded match
            # (e.g. one with a corrupted raw file) is never even opened,
            # and is reported separately from unexpected failures below.
            report["matches_excluded"] += 1
            report["excluded_matches"].append(
                {
                    "match_id": excluded.match_id,
                    "reason": excluded.reason,
                    "file": excluded.file,
                    "evidence": excluded.evidence,
                    "date_investigated": excluded.date_investigated,
                    "report": excluded.report,
                }
            )
            continue

        events_path = EVENTS_DIR / f"{mid}.json"
        frames_path = FRAMES_DIR / f"{mid}.json"
        events = None
        frames = None
        for label, path, assign in (
            ("events", events_path, "events"),
            ("three-sixty", frames_path, "frames"),
        ):
            try:
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                if assign == "events":
                    events = data
                else:
                    frames = data
            except Exception as error:  # noqa: BLE001 -- deliberately broad, but fully reported, not swallowed
                report["matches_failed_to_load"] += 1
                report["failed_matches"].append(
                    {
                        "match_id": mid,
                        "file": str(path),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                events = frames = None
                break
        if events is None or frames is None:
            continue

        report["matches_loaded_ok"] += 1
        report["event_records"] += len(events)
        report["three_sixty_records"] += len(frames)

        event_ids = [e.get("id") for e in events]
        report["duplicate_event_ids"] += len(event_ids) - len(set(event_ids))

        frame_uuids = [fr.get("event_uuid") for fr in frames]
        report["duplicate_event_uuid_in_frames"] += len(frame_uuids) - len(set(frame_uuids))

        indices = [e.get("index") for e in events if isinstance(e.get("index"), int)]
        if not all(indices[i] < indices[i + 1] for i in range(len(indices) - 1)):
            report["index_not_strictly_increasing_matches"] += 1

        frames_by_event_id = {fr["event_uuid"]: fr for fr in frames if "event_uuid" in fr}

        for event in events:
            event_type = event.get("type", {}).get("name")
            report["event_type_counts"][event_type] += 1
            report["period_counts"][event.get("period")] += 1
            if not isinstance(event.get("team"), dict):
                report["events_missing_team"] += 1
            if event.get("player") is None:
                report["events_missing_player"] += 1
                report["events_missing_player_by_type"][event_type] += 1

            loc_category = classify_coordinate(event.get("location"))
            report["event_location_category_counts"][loc_category] += 1
            if loc_category == "missing":
                report["events_missing_location_by_type"][event_type] += 1

            eid = event.get("id")
            if eid not in frames_by_event_id:
                report["events_without_360"] += 1

        for fr in frames:
            ff = fr.get("freeze_frame") or []
            eid = fr.get("event_uuid")
            if len(ff) == 0:
                report["frames_empty_freeze_frame"] += 1
            actors = []
            for player in ff:
                if not isinstance(player, dict):
                    report["freeze_frame_point_category_counts"]["malformed_player_record"] += 1
                    continue
                report["freeze_frame_points_total"] += 1
                category = classify_coordinate(player.get("location"))
                report["freeze_frame_point_category_counts"][category] += 1
                if category not in ("in_bounds", "missing"):
                    report["freeze_frame_oob_unique_matches"].add(mid)
                    report["freeze_frame_oob_unique_events"].add((mid, eid))
                    if category not in ("malformed", "non_numeric"):
                        over = overshoot_magnitude(player["location"])
                        report["freeze_frame_oob_magnitude_buckets"][magnitude_bucket(over)] += 1
                if player.get("actor") is True:
                    actors.append(player)
            if len(actors) == 0:
                report["frames_with_zero_actor_points"] += 1
                if len(report["zero_actor_examples"]) < MAX_EXAMPLES:
                    report["zero_actor_examples"].append({"match_id": mid, "event_uuid": eid})
            elif len(actors) > 1:
                report["frames_with_multiple_actor_points"] += 1
                if len(report["multi_actor_examples"]) < MAX_EXAMPLES:
                    report["multi_actor_examples"].append(
                        {"match_id": mid, "event_uuid": eid, "actor_locations": [a.get("location") for a in actors]}
                    )

        matched = [(e, frames_by_event_id[e["id"]]) for e in events if e.get("id") in frames_by_event_id]
        if events and frames and not matched:
            # Both files are non-empty (so this is NOT the "0 events"/"0
            # frames" case already visible elsewhere in this report) but
            # share zero event ids in common -- the two files do not
            # describe the same match. Found 2026-10-14: 7 matches with
            # this exact signature were previously invisible to this
            # audit (matched_pairs is only ever summed across the whole
            # dataset here, never checked per-match), yet unconditionally
            # broke src/build_statsbomb_360_dataset.py::build_match_dataset.
            report["matches_with_zero_matched_pairs"].append(mid)
        report["matched_pairs"] += len(matched)
        for event, frame in matched:
            result = classify_orientation(event, frame)
            report["orientation_counts"][result.orientation] += 1

        shared_groups = detect_shared_cross_team_snapshots(events, frames_by_event_id)
        seen_groups = set()
        for info in shared_groups.values():
            if info.group_event_ids in seen_groups:
                continue
            seen_groups.add(info.group_event_ids)
            report["shared_cross_team_groups"] += 1
            if info.source_event_id is not None:
                report["shared_cross_team_resolved"] += 1
            else:
                report["shared_cross_team_ambiguous"] += 1

    # Finalize: convert Counters/sets to JSON-serializable, deterministic forms.
    report["event_type_counts"] = dict(report["event_type_counts"].most_common())
    report["events_missing_player_by_type"] = dict(report["events_missing_player_by_type"].most_common())
    report["events_missing_location_by_type"] = dict(report["events_missing_location_by_type"].most_common())
    report["period_counts"] = dict(sorted((k, v) for k, v in report["period_counts"].items() if k is not None))
    report["orientation_counts"] = dict(report["orientation_counts"])
    report["event_location_category_counts"] = dict(report["event_location_category_counts"])
    report["freeze_frame_point_category_counts"] = dict(report["freeze_frame_point_category_counts"])
    report["freeze_frame_oob_magnitude_buckets"] = dict(report["freeze_frame_oob_magnitude_buckets"])
    report["freeze_frame_oob_unique_matches_count"] = len(report["freeze_frame_oob_unique_matches"])
    report["freeze_frame_oob_unique_events_count"] = len(report["freeze_frame_oob_unique_events"])
    del report["freeze_frame_oob_unique_matches"]
    del report["freeze_frame_oob_unique_events"]
    return report


def oob_point_total(report: dict) -> int:
    return sum(v for k, v in report["freeze_frame_point_category_counts"].items() if k not in ("in_bounds", "missing", "malformed", "non_numeric"))


def print_report(report: dict) -> None:
    print("=" * 70)
    print("STATSBOMB 360 DATA-QUALITY AUDIT (raw source data only)")
    print("=" * 70)
    print(f"Matches discovered (have three-sixty data): {report['matches_discovered']}")
    print(f"Matches explicitly excluded (manifest):     {report['matches_excluded']}")
    for excluded in report["excluded_matches"]:
        print(f"    match {excluded['match_id']}: {excluded['reason']} (see {excluded['report']})")
    print(f"Matches loaded OK:                          {report['matches_loaded_ok']}")
    print(f"Matches UNEXPECTEDLY failed to load:        {report['matches_failed_to_load']}")
    for failure in report["failed_matches"]:
        print(f"    match {failure['match_id']}: {failure['error_type']}: {failure['error_message']} ({failure['file']})")
    print()
    print(f"Event records (raw):                        {report['event_records']}")
    print(f"Three-sixty records (raw):                  {report['three_sixty_records']}")
    print(f"Matched event/frame pairs:                  {report['matched_pairs']}")
    if report["matches_with_zero_matched_pairs"]:
        print(
            f"  WARNING: {len(report['matches_with_zero_matched_pairs'])} match(es) have non-empty "
            f"events+three-sixty files that share ZERO event ids (files do not describe the same "
            f"match): {report['matches_with_zero_matched_pairs']}"
        )
    if report["event_records"]:
        pct = 100 * report["events_without_360"] / report["event_records"]
        print(f"Events without 360 coverage:                {report['events_without_360']} ({pct:.2f}%)")
    print()
    print(f"Duplicate event ids within a match:         {report['duplicate_event_ids']}")
    print(f"Duplicate event_uuid within three-sixty:    {report['duplicate_event_uuid_in_frames']}")
    print(f"Matches where raw event order != index order: {report['index_not_strictly_increasing_matches']}")
    print()
    print("Event location category counts (missing/malformed are distinct from out-of-bounds):")
    for k, v in sorted(report["event_location_category_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:15s}: {v}")
    print(f"Events missing location, by type (top 10): {dict(list(report['events_missing_location_by_type'].items())[:10])}")
    print()
    oob_total = oob_point_total(report)
    total_pts = report["freeze_frame_points_total"]
    print(f"Freeze-frame points total:                  {total_pts}")
    print(f"Freeze-frame points out of bounds:          {oob_total} ({100*oob_total/total_pts:.4f}% of total)" if total_pts else "n/a")
    print("Freeze-frame point category breakdown (a point may match >1 category, e.g. a corner overshoot in both x and y):")
    for k, v in sorted(report["freeze_frame_point_category_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:15s}: {v}")
    print("Out-of-bounds overshoot magnitude (distance past the nearest violated boundary):")
    for k in ["<=0.5", "<=1.0", "<=2.0", "<=5.0", "<=10.0", ">10.0"]:
        v = report["freeze_frame_oob_magnitude_buckets"].get(k, 0)
        print(f"  {k:8s}: {v}")
    print(f"Distinct matches with >=1 out-of-bounds point: {report['freeze_frame_oob_unique_matches_count']} / {report['matches_loaded_ok']}")
    print(f"Distinct (match, event) pairs with >=1 out-of-bounds point: {report['freeze_frame_oob_unique_events_count']}")
    print()
    print(f"Events missing team:                        {report['events_missing_team']}")
    print(f"Events missing player:                      {report['events_missing_player']}")
    print(f"  by type (top 10): {dict(list(report['events_missing_player_by_type'].items())[:10])}")
    print()
    print(f"Frames with zero actor points:               {report['frames_with_zero_actor_points']}")
    for ex in report["zero_actor_examples"]:
        print(f"    match={ex['match_id']} event={ex['event_uuid']}")
    print(f"Frames with multiple actor points:           {report['frames_with_multiple_actor_points']}")
    for ex in report["multi_actor_examples"][:10]:
        print(f"    match={ex['match_id']} event={ex['event_uuid']} actor_locations={ex['actor_locations']}")
    print(f"Frames with empty freeze_frame:              {report['frames_empty_freeze_frame']}")
    print()
    print("Orientation classification (matched pairs):")
    for k, v in report["orientation_counts"].items():
        pct = 100 * v / report["matched_pairs"] if report["matched_pairs"] else 0
        print(f"  {k:12s}: {v:8d} ({pct:.2f}%)")
    print()
    print("Shared cross-team 360 snapshots (Problem C):")
    print(f"  groups total:    {report['shared_cross_team_groups']}")
    print(f"  resolved:        {report['shared_cross_team_resolved']}")
    print(f"  ambiguous:       {report['shared_cross_team_ambiguous']}")
    print()
    print("Period distribution (event count):", report["period_counts"])
    print()
    print("Top 15 event types:")
    for t, c in list(report["event_type_counts"].items())[:15]:
        print(f"  {t:20s} {c}")


def compare_against_processed_csv(csv_path: Path, match_id: int) -> None:
    """Read-only cross-check: recompute a few counts from raw JSON for one
    match and compare against an EXISTING processed CSV. Never writes
    anything; clearly labeled as comparing raw-derived vs. processed counts.
    """
    print()
    print("=" * 70)
    print(f"CROSS-CHECK: raw-derived counts vs. existing processed CSV ({csv_path})")
    print("=" * 70)
    if not csv_path.exists():
        print(f"  CSV not found at {csv_path} -- skipping cross-check.")
        return

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("match_id") == str(match_id)]
    print(f"  Processed CSV rows for match {match_id}: {len(rows)}")

    from src.build_statsbomb_360_dataset import MEANINGFUL_EVENT_TYPES

    events_path = EVENTS_DIR / f"{match_id}.json"
    frames_path = FRAMES_DIR / f"{match_id}.json"
    with events_path.open(encoding="utf-8") as f:
        events = json.load(f)
    with frames_path.open(encoding="utf-8") as f:
        frames = json.load(f)
    frame_ids = {fr["event_uuid"] for fr in frames}
    raw_candidate_count = sum(
        1 for e in events if e["id"] in frame_ids and e["type"]["name"] in MEANINGFUL_EVENT_TYPES
    )
    print(f"  Raw MEANINGFUL_EVENT_TYPES events with 360 coverage (upper bound on CSV rows, before time/period filtering): {raw_candidate_count}")
    print(f"  CSV row count is <= this upper bound as expected: {len(rows) <= raw_candidate_count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=None, help="Audit a random sample of N matches instead of all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Path to save the report as JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing --output file.")
    parser.add_argument("--compare-csv", type=Path, default=None, help="Existing processed CSV to cross-check (read-only).")
    parser.add_argument("--compare-match-id", type=int, default=None, help="Match id to use with --compare-csv.")
    parser.add_argument(
        "--exclusion-manifest", type=Path, default=None,
        help="Override path to the exclusion manifest (default: configs/excluded_matches.json).",
    )
    args = parser.parse_args()

    if args.output is not None and args.output.exists() and not args.overwrite:
        print(
            f"Refusing to overwrite existing report at {args.output}. "
            f"Pass --overwrite to replace it, or choose a different --output path.",
            file=sys.stderr,
        )
        return 1

    from src.match_exclusions import DEFAULT_MANIFEST_PATH, ExclusionManifestError

    manifest_path = args.exclusion_manifest or DEFAULT_MANIFEST_PATH
    try:
        exclusion_manifest = load_exclusion_manifest(manifest_path)
    except ExclusionManifestError as error:
        # Fail clearly, before processing any match -- never silently treat
        # a broken manifest as "no exclusions."
        print(f"Exclusion manifest error: {error}", file=sys.stderr)
        return 1

    match_ids = discover_match_ids(args.sample, args.seed)
    report = audit(match_ids, exclusion_manifest)
    print_report(report)

    if args.compare_csv is not None:
        if args.compare_match_id is None:
            print("--compare-csv requires --compare-match-id", file=sys.stderr)
            return 1
        compare_against_processed_csv(args.compare_csv, args.compare_match_id)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved JSON report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
