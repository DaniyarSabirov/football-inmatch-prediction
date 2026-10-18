"""Read-only forensic diagnostic for the 7 matches excluded under the
'three_sixty_event_uuid_mismatch' reason (see configs/excluded_matches.json).

Never writes to data/raw, data/processed, or configs/excluded_matches.json.
Only reads raw JSON and prints/optionally saves a report.

Produces, for each affected match:
  - file existence/size/parseability
  - event-id / event_uuid counts, duplicates, null/malformed/whitespace checks
  - overlap statistics between events.json and three-sixty.json
  - sample UUIDs from both files

With --full-search, additionally builds an index of EVERY event id across
ALL events.json files and EVERY event_uuid across ALL three-sixty.json
files in the local dataset (~4,235 + ~426 files; several minutes), and
checks whether the "orphan" UUIDs from the affected matches correspond to
ANY other locally-available match. This is the authoritative, exhaustive
check -- without it, only a negative result against the 7 matches
themselves and other three-sixty files is available (much faster).

With --check-all-ratios, computes the events/three-sixty overlap ratio for
EVERY non-excluded match, to check whether any other match shows partial
(not just total) UUID mismatch contamination.

Usage:
    .venv/bin/python scripts/diagnose_excluded_match_mismatch.py
    .venv/bin/python scripts/diagnose_excluded_match_mismatch.py --full-search --output results/forensic_mismatch_full.json
    .venv/bin/python scripts/diagnose_excluded_match_mismatch.py --check-all-ratios
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EVENTS_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data" / "events"
FRAMES_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data" / "three-sixty"

AFFECTED_MATCH_IDS = [3837706, 3877115, 3877170, 3877194, 3895158, 3895266, 3895309]

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def is_valid_uuid_format(value: str) -> bool:
    return bool(_UUID_RE.match(value))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def per_match_report(match_id: int) -> dict:
    events_path = EVENTS_DIR / f"{match_id}.json"
    frames_path = FRAMES_DIR / f"{match_id}.json"
    r: dict = {
        "match_id": match_id,
        "events_path": str(events_path),
        "frames_path": str(frames_path),
        "events_exists": events_path.exists(),
        "frames_exists": frames_path.exists(),
    }
    r["events_size"] = events_path.stat().st_size if r["events_exists"] else None
    r["frames_size"] = frames_path.stat().st_size if r["frames_exists"] else None

    events, frames = [], []
    try:
        events = load_json(events_path)
        r["events_parse_ok"] = True
    except Exception as error:  # noqa: BLE001
        r["events_parse_ok"] = False
        r["events_parse_error"] = str(error)
    try:
        frames = load_json(frames_path)
        r["frames_parse_ok"] = True
    except Exception as error:  # noqa: BLE001
        r["frames_parse_ok"] = False
        r["frames_parse_error"] = str(error)

    r["event_count"] = len(events)
    r["frame_record_count"] = len(frames)

    event_ids_raw = [e.get("id") for e in events]
    frame_uuids_raw = [fr.get("event_uuid") for fr in frames]
    event_ids = [x for x in event_ids_raw if x is not None]
    frame_uuids = [x for x in frame_uuids_raw if x is not None]

    r["events_null_id_count"] = len(event_ids_raw) - len(event_ids)
    r["frames_null_uuid_count"] = len(frame_uuids_raw) - len(frame_uuids)

    event_id_set = set(event_ids)
    frame_uuid_set = set(frame_uuids)
    r["unique_event_ids"] = len(event_id_set)
    r["unique_frame_uuids"] = len(frame_uuid_set)
    r["event_id_duplicates"] = len(event_ids) - len(event_id_set)
    r["frame_uuid_duplicates"] = len(frame_uuids) - len(frame_uuid_set)

    overlap = event_id_set & frame_uuid_set
    r["overlap_count"] = len(overlap)
    r["only_in_events"] = len(event_id_set - frame_uuid_set)
    r["only_in_frames"] = len(frame_uuid_set - event_id_set)
    r["overlap_pct_of_frame_uuids"] = round(100 * len(overlap) / len(frame_uuid_set), 4) if frame_uuid_set else None

    r["events_malformed_uuid_format"] = sum(1 for x in event_ids if not is_valid_uuid_format(x))
    r["frames_malformed_uuid_format"] = sum(1 for x in frame_uuids if not is_valid_uuid_format(x))
    r["events_with_whitespace"] = sum(1 for x in event_ids if x != x.strip())
    r["frames_with_whitespace"] = sum(1 for x in frame_uuids if x != x.strip())
    r["overlap_count_case_insensitive"] = len({x.lower() for x in event_ids} & {x.lower() for x in frame_uuids})

    r["sample_event_ids"] = event_ids[:10]
    r["sample_frame_uuids"] = frame_uuids[:10]
    r["periods_present"] = sorted({e.get("period") for e in events if e.get("period") is not None})
    r["teams_in_events"] = sorted({e["team"]["name"] for e in events if isinstance(e.get("team"), dict)})

    # Coverage ratio (frame record count / event count) -- a sanity check
    # that the three-sixty content LOOKS like real coverage for SOME
    # match, not malformed/garbage data, even though it is not THIS match.
    r["frame_to_event_count_ratio"] = round(r["frame_record_count"] / r["event_count"], 4) if r["event_count"] else None

    return r


def check_all_overlap_ratios(excluded_ids: set[int]) -> list[tuple[int, float, int, int]]:
    """Compute the events/three-sixty overlap ratio for every non-excluded
    match, to check whether the mismatch pattern affects any match beyond
    the known 7 (even partially).
    """
    ratios = []
    for path in sorted(glob.glob(str(FRAMES_DIR / "*.json"))):
        mid = int(os.path.splitext(os.path.basename(path))[0])
        if mid in excluded_ids:
            continue
        try:
            frames = load_json(Path(path))
            events = load_json(EVENTS_DIR / f"{mid}.json")
        except Exception:
            continue
        event_ids = {e["id"] for e in events if e.get("id")}
        frame_uuids = {fr["event_uuid"] for fr in frames if fr.get("event_uuid")}
        overlap = event_ids & frame_uuids
        ratio = len(overlap) / len(frame_uuids) if frame_uuids else None
        if ratio is not None:
            ratios.append((mid, ratio, len(frame_uuids), len(overlap)))
    return ratios


def full_dataset_search(affected: list[dict]) -> dict:
    """Exhaustive search: does any orphan UUID from the affected matches
    correspond to ANY other match's events.json (all ~4,235 files) or
    three-sixty.json (all ~426 files) in the local dataset? Several
    minutes; only run with --full-search.
    """
    print("Building full event-id index across all events.json files (several minutes)...", file=sys.stderr)
    event_id_to_match: dict[str, list[int]] = {}
    for path in glob.glob(str(EVENTS_DIR / "*.json")):
        mid = int(os.path.splitext(os.path.basename(path))[0])
        try:
            events = load_json(Path(path))
        except Exception:
            continue
        for e in events:
            eid = e.get("id")
            if eid is not None:
                event_id_to_match.setdefault(eid, []).append(mid)
    print(f"Indexed {len(event_id_to_match)} unique event ids.", file=sys.stderr)

    print("Building full event_uuid index across all three-sixty.json files...", file=sys.stderr)
    frame_uuid_to_match: dict[str, list[int]] = {}
    for path in glob.glob(str(FRAMES_DIR / "*.json")):
        mid = int(os.path.splitext(os.path.basename(path))[0])
        try:
            frames = load_json(Path(path))
        except Exception:
            continue
        for fr in frames:
            uuid = fr.get("event_uuid")
            if uuid is not None:
                frame_uuid_to_match.setdefault(uuid, []).append(mid)
    print(f"Indexed {len(frame_uuid_to_match)} unique event_uuids.", file=sys.stderr)

    results = {}
    for r in affected:
        mid = r["match_id"]
        owners_for_frame_uuids = set()
        for uuid in r["sample_frame_uuids"]:
            owners_for_frame_uuids.update(event_id_to_match.get(uuid, []))
        # full check, not just the 10 samples
        frames = load_json(FRAMES_DIR / f"{mid}.json")
        all_frame_uuids = [fr["event_uuid"] for fr in frames if fr.get("event_uuid")]
        full_owners = set()
        for uuid in all_frame_uuids:
            full_owners.update(event_id_to_match.get(uuid, []))

        events = load_json(EVENTS_DIR / f"{mid}.json")
        all_event_ids = [e["id"] for e in events if e.get("id")]
        full_owners_rev = set()
        for eid in all_event_ids:
            full_owners_rev.update(frame_uuid_to_match.get(eid, []))

        results[mid] = {
            "frame_uuids_checked": len(all_frame_uuids),
            "owning_matches_for_frame_uuids": sorted(full_owners),
            "event_ids_checked": len(all_event_ids),
            "owning_matches_for_event_ids": sorted(full_owners_rev),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="Save the full JSON report to this path.")
    parser.add_argument("--full-search", action="store_true", help="Run the exhaustive full-dataset UUID search (several minutes).")
    parser.add_argument("--check-all-ratios", action="store_true", help="Check overlap ratio for every non-excluded match.")
    args = parser.parse_args()

    report: dict = {"affected_matches": []}
    for mid in AFFECTED_MATCH_IDS:
        r = per_match_report(mid)
        report["affected_matches"].append(r)
        print(f"=== match {mid} ===")
        print(f"  teams: {r['teams_in_events']}")
        print(f"  events: {r['event_count']} records, frames: {r['frame_record_count']} records "
              f"(coverage ratio if paired correctly: {r['frame_to_event_count_ratio']})")
        print(f"  unique event ids: {r['unique_event_ids']}  unique frame uuids: {r['unique_frame_uuids']}")
        print(f"  overlap: {r['overlap_count']}  only_in_events: {r['only_in_events']}  only_in_frames: {r['only_in_frames']}")
        print(f"  duplicates: events={r['event_id_duplicates']} frames={r['frame_uuid_duplicates']}  "
              f"malformed_format: events={r['events_malformed_uuid_format']} frames={r['frames_malformed_uuid_format']}")
        print()

    if args.check_all_ratios:
        excluded_ids = {3845506, 3923880, *AFFECTED_MATCH_IDS}
        ratios = check_all_overlap_ratios(excluded_ids)
        ratios.sort(key=lambda x: x[1])
        report["all_match_overlap_ratios_checked"] = len(ratios)
        report["lowest_10_overlap_ratios"] = ratios[:10]
        print(f"Checked overlap ratio for {len(ratios)} non-excluded matches.")
        print("10 lowest ratios:")
        for mid, ratio, n_frames, n_overlap in ratios[:10]:
            print(f"  match={mid} ratio={ratio:.4f} frame_uuids={n_frames} overlap={n_overlap}")
        print()

    if args.full_search:
        report["full_dataset_search"] = full_dataset_search(report["affected_matches"])
        print("Full-dataset search results:")
        for mid, res in report["full_dataset_search"].items():
            print(f"  match {mid}: owning_matches_for_frame_uuids={res['owning_matches_for_frame_uuids']} "
                  f"owning_matches_for_event_ids={res['owning_matches_for_event_ids']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved full report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
