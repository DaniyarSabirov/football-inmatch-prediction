"""Build the production next-event dataset for every usable match.

Runs src/build_statsbomb_360_dataset.build_match_dataset (the only
row-construction logic) for every match with 360 data that is not in
configs/excluded_matches.json, and writes to data/processed/:

  statsbomb_360_match_<id>_next_events.csv     one file per match (as the builder CLI)
  statsbomb_360_all_matches_next_events.csv    all matches concatenated
  statsbomb_360_dataset_manifest.json          code commit, horizon, row counts, schema

Feature columns (visible_teammates/opponents/ambiguous, identity_quality,
identity_corrected_points) are prediction-time valid (causal team identity);
*_offline columns use later records and are for data-quality analysis only;
*_raw columns are the untouched StatsBomb teammate flags. See
docs/ml_dataset_identity_and_orientation.md.

Usage:
    .venv/bin/python scripts/build_full_dataset.py [--max-time 10] [--workers 7]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.animate_statsbomb_360_match import available_360_match_ids  # noqa: E402
from src.build_statsbomb_360_dataset import OUTPUT_DIR, build_match_dataset, save_dataset  # noqa: E402
from src.match_exclusions import load_exclusion_manifest  # noqa: E402

COMBINED = OUTPUT_DIR / "statsbomb_360_all_matches_next_events.csv"
MANIFEST = OUTPUT_DIR / "statsbomb_360_dataset_manifest.json"


def build_one(job: tuple[int, float]) -> tuple[int, int, list[str]]:
    match_id, max_time = job
    rows = build_match_dataset(match_id, max_time)
    save_dataset(rows, OUTPUT_DIR / f"statsbomb_360_match_{match_id}_next_events.csv")
    return match_id, len(rows), list(rows[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-time", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    excluded = load_exclusion_manifest()
    match_ids = [m for m in available_360_match_ids() if m not in excluded]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = sorted(pool.map(build_one, [(m, args.max_time) for m in match_ids], chunksize=1))
    schemas = {tuple(columns) for _, _, columns in results}
    if len(schemas) != 1:
        raise RuntimeError(f"Per-match files disagree on the schema: {schemas}")
    columns = list(next(iter(schemas)))
    with COMBINED.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(columns)
        for match_id, _, _ in results:
            with (OUTPUT_DIR / f"statsbomb_360_match_{match_id}_next_events.csv").open(encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader)
                writer.writerows(reader)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip())
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "src_had_uncommitted_changes": dirty,
        "max_time_seconds": args.max_time, "matches": len(results), "excluded_matches": sorted(excluded),
        "rows": sum(n for _, n, _ in results), "rows_per_match": {str(m): n for m, n, _ in results},
        "columns": columns, "combined_file": COMBINED.name,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{manifest['matches']} matches, {manifest['rows']} rows -> {COMBINED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
