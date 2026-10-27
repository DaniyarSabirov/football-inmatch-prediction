"""Full-pipeline truncation check on real rows: for each selected row, the
match is cut immediately after the row's event (its next meaningful event is
kept as bare event data -- the label -- without a 360 record), rebuilt, and
every column that is not TARGET or OFFLINE_DIAGNOSTIC is compared with the
full build for the cut row and every earlier row of the truncated build.
Rows: all 55 rows of orientation_55_rows.csv + 100 random dataset rows
(seed 20261027). Output: ../truncation_check.json
"""
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from src.build_statsbomb_360_dataset import build_rows_from_match_data  # noqa: E402
from src.ml_features import load_feature_classification  # noqa: E402
from test_ml_feature_safety import _truncate_after  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
COLUMNS = [c for c, e in load_feature_classification()["columns"].items() if e["category"] not in ("TARGET", "OFFLINE_DIAGNOSTIC")]


def same(a, b):
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


def check(job):
    match_id, event_ids = job
    events = json.loads((RAW / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((RAW / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    full = {r["current_event_id"]: r for r in build_rows_from_match_data(match_id, events, frames, 10.0)}
    out = []
    for event_id in event_ids:
        cut_events, cut_frames = _truncate_after(events, frames, event_id)
        rows = build_rows_from_match_data(match_id, cut_events, cut_frames, 10.0)
        mismatches = [(r["current_event_id"], c) for r in rows for c in COLUMNS if not same(r[c], full[r["current_event_id"]][c])]
        out.append({"match_id": match_id, "event_id": event_id, "cut_row_present": event_id in {r["current_event_id"] for r in rows},
                    "rows_compared": len(rows), "cells_compared": len(rows) * len(COLUMNS), "mismatches": mismatches[:10],
                    "mismatch_count": len(mismatches)})
    return out


def main():
    jobs = defaultdict(list)
    for r in pd.read_csv(HERE.parent / "orientation_55_rows.csv").itertuples():
        jobs[int(r.match_id)].append(r.event_id)
    data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "statsbomb_360_all_matches_next_events.csv",
                       usecols=["match_id", "current_event_id"])
    rng = random.Random(20261027)
    for i in rng.sample(range(len(data)), 100):
        jobs[int(data.match_id.iloc[i])].append(data.current_event_id.iloc[i])
    with ProcessPoolExecutor(max_workers=7) as pool:
        results = [r for out in pool.map(check, list(jobs.items()), chunksize=1) for r in out]
    summary = {"cuts": len(results), "matches": len(jobs), "cut_rows_present": sum(r["cut_row_present"] for r in results),
               "rows_compared": sum(r["rows_compared"] for r in results), "cells_compared": sum(r["cells_compared"] for r in results),
               "mismatching_cells": sum(r["mismatch_count"] for r in results), "columns_checked": COLUMNS,
               "examples": [r for r in results if r["mismatch_count"]][:10]}
    (HERE.parent / "truncation_check.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "columns_checked"}, indent=1))


if __name__ == "__main__":
    main()
