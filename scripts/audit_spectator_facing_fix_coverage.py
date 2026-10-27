"""Dataset-wide, machine-readable audit of the 2026-10-22 spectator-facing
orientation fix's SCOPE -- i.e. an internally-detectable software-defect
check, not a physical-orientation verification (which requires external
evidence per match and is explicitly out of scope here -- see
docs/full_visualization_orientation_and_clock_audit_2026-10-22.md and
docs/all_matches_physical_orientation_feasibility.md).

For every eligible match, compares the OLD default behavior (one fixed,
period-independent automatic reference team for the whole match) against
the NEW default behavior (the same automatic baseline, with the Laws-of-
the-Game halftime/extra-time-half swap applied per period) and records,
per match: which periods are present, the old single reference id, the
new per-period reference mapping, and whether any period differs (i.e.
was "affected" by the fix). This is a deterministic, provable-by-
construction property (any match with a period in {2, 4} present will
always be affected, since the swap is unconditional) -- this script exists
to confirm that fact empirically across the real dataset, with no crashes
or edge cases, rather than asserting it from code reading alone.

Never claims physical correctness for any match -- only reports the
internally-detectable fact of "the old and new DEFAULT renderer outputs
differ for this match" or not.

Usage:
    .venv/bin/python scripts/audit_spectator_facing_fix_coverage.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.animate_statsbomb_360_match import load_and_join_match  # noqa: E402
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.team_normalization import apply_period_end_swap, resolve_reference_team_id  # noqa: E402

INVENTORY_CSV = PROJECT_ROOT / "results" / "physical_orientation" / "match_inventory.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "physical_orientation" / "full_visual_audit_2026-10-22"
OUTPUT_JSON = OUTPUT_DIR / "spectator_facing_fix_coverage.json"
OUTPUT_CSV = OUTPUT_DIR / "spectator_facing_fix_coverage.csv"


def load_eligible_match_ids() -> list[int]:
    ids = []
    with INVENTORY_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["eligible"] == "True":
                ids.append(int(row["match_id"]))
    return ids


def main() -> int:
    manifest = load_exclusion_manifest()
    eligible_ids = load_eligible_match_ids()

    rows = []
    load_errors = []
    for match_id in eligible_ids:
        try:
            match = load_and_join_match(match_id)
        except Exception as error:  # noqa: BLE001
            load_errors.append({"match_id": match_id, "error": str(error)})
            continue
        all_events = [j.event for j in match.usable_frames]
        periods_present = sorted({e.get("period") for e in all_events})
        old_reference_id = resolve_reference_team_id(all_events)
        new_reference_by_period = {
            p: apply_period_end_swap(all_events, old_reference_id, p) for p in periods_present
        }
        affected_periods = [p for p, rid in new_reference_by_period.items() if rid != old_reference_id]
        rows.append({
            "match_id": match_id,
            "periods_present": periods_present,
            "old_reference_team_id": old_reference_id,
            "new_reference_by_period": new_reference_by_period,
            "affected_periods": affected_periods,
            "affected": bool(affected_periods),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"matches_checked": len(rows), "load_errors": load_errors, "results": rows}, f, indent=2)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["match_id", "periods_present", "old_reference_team_id", "new_reference_by_period", "affected_periods", "affected"])
        for r in rows:
            writer.writerow([
                r["match_id"], ";".join(map(str, r["periods_present"])), r["old_reference_team_id"],
                json.dumps(r["new_reference_by_period"]), ";".join(map(str, r["affected_periods"])), r["affected"],
            ])

    n_affected = sum(1 for r in rows if r["affected"])
    period_counts: dict[int, int] = {}
    for r in rows:
        for p in r["periods_present"]:
            period_counts[p] = period_counts.get(p, 0) + 1

    print(f"Matches checked: {len(rows)}")
    print(f"Load errors: {len(load_errors)}")
    for e in load_errors:
        print(f"  match {e['match_id']}: {e['error']}")
    print(f"Matches affected by the fix (old vs new default differ in >=1 period): {n_affected}/{len(rows)}")
    print(f"Matches UNaffected (single-period matches, or no period in {{2,4}} present): {len(rows) - n_affected}")
    print("Period presence counts across eligible matches:")
    for p in sorted(period_counts):
        print(f"  period {p}: {period_counts[p]} matches")
    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
