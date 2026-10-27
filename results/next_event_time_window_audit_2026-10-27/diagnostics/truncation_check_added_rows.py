"""Full-pipeline truncation check (same procedure as
results/ml_data_integrity_final_2026-10-27/diagnostics/truncation_check.py)
on 60 randomly chosen rows ADDED by removing the 10-second rule (gap > 10 s;
seed 20261027): cutting the input right after the row's event (keeping only
its label as bare event data) must leave every non-TARGET, non-offline
column unchanged. Output: ../truncation_check_added_rows.json"""
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results" / "ml_data_integrity_final_2026-10-27" / "diagnostics"))
from truncation_check import check  # noqa: E402


def main():
    eligible = pd.read_csv(HERE.parent / "eligible_events.csv.gz")
    added = eligible[eligible.outcome == "excluded_gap_over_10s"].reset_index(drop=True)
    jobs = defaultdict(list)
    for i in random.Random(20261027).sample(range(len(added)), 60):
        jobs[int(added.match_id[i])].append(added.event_id[i])
    with ProcessPoolExecutor(max_workers=7) as pool:
        results = [r for out in pool.map(check, list(jobs.items()), chunksize=1) for r in out]
    summary = {"cuts": len(results), "matches": len(jobs), "cut_rows_present": sum(r["cut_row_present"] for r in results),
               "rows_compared": sum(r["rows_compared"] for r in results), "cells_compared": sum(r["cells_compared"] for r in results),
               "mismatching_cells": sum(r["mismatch_count"] for r in results),
               "examples": [r for r in results if r["mismatch_count"]][:5]}
    (HERE.parent / "truncation_check_added_rows.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
