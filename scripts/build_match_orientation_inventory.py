"""Build a complete inventory of all matches with local StatsBomb 360 data,
for the physical-pitch-orientation calibration feasibility study (see
docs/all_matches_physical_orientation_feasibility.md).

Read-only: never writes to data/raw, data/processed, or configs/. Writes
only to results/physical_orientation/.

For every match with a local three-sixty JSON file (426 raw; 417 after
the documented exclusions in configs/excluded_matches.json), records:
  - identity (competition, season, date, home/away team, stadium)
  - periods present and usable (360-matched, non-empty freeze_frame) count
  - the AUTOMATIC (analytical, non-physical) reference team currently
    resolved by resolve_reference_team_id -- explicitly NOT a physical
    orientation claim, see src/team_normalization.py
  - physical verification status, sourced from
    configs/physical_orientation_calibration.json if present, else
    "unverified" for every match (no value is invented here)

Usage:
    .venv/bin/python scripts/build_match_orientation_inventory.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.animate_statsbomb_360_match import (  # noqa: E402
    available_360_match_ids,
    load_and_join_match,
    load_match_metadata_index,
)
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.team_normalization import resolve_reference_team_id  # noqa: E402

STATSBOMB_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "physical_orientation"
OUTPUT_CSV = OUTPUT_DIR / "match_inventory.csv"
CALIBRATION_REGISTRY_PATH = PROJECT_ROOT / "configs" / "physical_orientation_calibration.json"


def load_calibration_status() -> dict[int, dict]:
    """Return {match_id: calibration_entry} from the registry, or {} if the
    registry does not exist yet. Never invents a verification status."""
    if not CALIBRATION_REGISTRY_PATH.exists():
        return {}
    with CALIBRATION_REGISTRY_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return {entry["match_id"]: entry for entry in data.get("calibrations", [])}


def stadium_name(match_id: int) -> str | None:
    """Read the stadium name directly from the raw per-competition match
    metadata file (not carried by MatchMetadata, which omits it)."""
    for path in (STATSBOMB_DATA_DIR / "matches").glob("*/*.json"):
        try:
            with path.open(encoding="utf-8") as f:
                records = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        for record in records:
            if record.get("match_id") == match_id:
                stadium = record.get("stadium")
                if isinstance(stadium, dict):
                    return stadium.get("name")
                return None
    return None


def main() -> int:
    manifest = load_exclusion_manifest()
    metadata_index = load_match_metadata_index()
    calibration_by_match = load_calibration_status()

    # Build a stadium lookup once (one pass over all metadata files) instead
    # of re-scanning per match -- much faster for 426 matches.
    stadium_by_match: dict[int, str | None] = {}
    for path in (STATSBOMB_DATA_DIR / "matches").glob("*/*.json"):
        try:
            with path.open(encoding="utf-8") as f:
                records = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        for record in records:
            mid = record.get("match_id")
            if isinstance(mid, int):
                stadium = record.get("stadium")
                stadium_by_match[mid] = stadium.get("name") if isinstance(stadium, dict) else None

    raw_ids = available_360_match_ids()
    rows = []
    load_errors = []

    for mid in raw_ids:
        excluded = manifest.get(mid)
        meta = metadata_index.get(mid)
        row = {
            "match_id": mid,
            "eligible": excluded is None,
            "exclusion_reason": excluded.reason if excluded else "",
            "competition": meta.competition if meta else "",
            "season": meta.season if meta else "",
            "date": meta.date if meta else "",
            "home_team": meta.home_team if meta else "",
            "away_team": meta.away_team if meta else "",
            "score": meta.score if meta else "",
            "stadium": stadium_by_match.get(mid, ""),
            "periods_present": "",
            "usable_360_snapshots": "",
            "automatic_reference_team": "",
            "physical_verification_status": "unverified",
            "physical_reference_team_period_1": "",
            "evidence_source": "",
            "confidence": "",
        }

        if excluded is None:
            try:
                match = load_and_join_match(mid)
                periods = sorted({j.event.get("period") for j in match.usable_frames})
                row["periods_present"] = ",".join(str(p) for p in periods)
                row["usable_360_snapshots"] = match.usable_count
                ref_id = resolve_reference_team_id([j.event for j in match.usable_frames])
                ref_name = next(
                    (j.event["team"]["name"] for j in match.usable_frames if j.event["team"]["id"] == ref_id),
                    "",
                )
                row["automatic_reference_team"] = ref_name
            except Exception as error:  # noqa: BLE001
                load_errors.append({"match_id": mid, "error": str(error)})

        calibration = calibration_by_match.get(mid)
        if calibration is not None:
            row["physical_verification_status"] = calibration.get("verification_status", "unverified")
            row["physical_reference_team_period_1"] = calibration.get("period_1_physical_reference_team", "")
            row["evidence_source"] = calibration.get("evidence_source", "")
            row["confidence"] = calibration.get("confidence", "")

        rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    eligible_rows = [r for r in rows if r["eligible"]]
    verified_rows = [r for r in rows if r["physical_verification_status"] == "verified"]
    competitions = Counter(r["competition"] for r in eligible_rows if r["competition"])
    stadiums = Counter(r["stadium"] for r in eligible_rows if r["stadium"])
    teams = Counter()
    for r in eligible_rows:
        if r["home_team"]:
            teams[r["home_team"]] += 1
        if r["away_team"]:
            teams[r["away_team"]] += 1

    print(f"Raw matches (local 360 file): {len(rows)}")
    print(f"Eligible matches: {len(eligible_rows)}")
    print(f"Load errors: {len(load_errors)}")
    for e in load_errors:
        print(f"  match {e['match_id']}: {e['error']}")
    print(f"Physically verified (registry): {len(verified_rows)}")
    print(f"Distinct competitions among eligible matches: {len(competitions)}")
    for comp, count in competitions.most_common():
        print(f"  {comp}: {count}")
    print(f"Distinct stadiums among eligible matches: {len(stadiums)}")
    print(f"Distinct teams among eligible matches: {len(teams)}")
    print(f"Teams appearing in >1 eligible match: {sum(1 for c in teams.values() if c > 1)}")
    print(f"\nSaved: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
