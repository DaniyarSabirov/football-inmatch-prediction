"""Quantify the next-event time-window filters BEFORE changing the pipeline.

Reproduces the builder's eligibility semantics with its own definitions
(src/build_statsbomb_360_dataset.py at commit 021867b): a current event is
ELIGIBLE when it has a 360 record (its event id is in the three-sixty file)
and its type is in MEANINGFUL_EVENT_TYPES; its target is the next event of a
meaningful type later in the raw event stream (event order) within the same
period (find_next_meaningful_event). The old builder then dropped rows with
  gap < 0 s   (next event's timestamp earlier than the current one), and
  gap > 10 s  (the 10-second rule).
Outcomes per eligible event:
  retained_old            kept by the old builder
  excluded_gap_over_10s   dropped only by the 10-second rule
  excluded_negative_gap   dropped by the gap < 0 rule
  no_successor_in_period  no later meaningful event in the period (no target)

Output: ../eligible_events.csv.gz (one row per eligible event), ../funnel.json
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.animate_statsbomb_360_match import available_360_match_ids  # noqa: E402
from src.build_statsbomb_360_dataset import MEANINGFUL_EVENT_TYPES, find_next_meaningful_event, timestamp_to_seconds  # noqa: E402
from src.match_exclusions import load_exclusion_manifest  # noqa: E402

DATA = ROOT / "data" / "raw" / "statsbomb" / "data"


def per_match(match_id):
    events = json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    funnel = {"events": len(events), "meaningful_type": 0, "meaningful_with_360_record": 0,
              "meaningful_without_360_record": 0}
    rows = []
    for position, event in enumerate(events):
        if event["type"]["name"] not in MEANINGFUL_EVENT_TYPES:
            continue
        funnel["meaningful_type"] += 1
        if event["id"] not in frames:
            funnel["meaningful_without_360_record"] += 1
            continue
        funnel["meaningful_with_360_record"] += 1
        nxt = find_next_meaningful_event(events, position, event["period"])
        row = {"match_id": match_id, "event_id": event["id"], "index": event["index"], "period": event["period"],
               "current_type": event["type"]["name"], "empty_freeze_frame": not frames[event["id"]].get("freeze_frame")}
        if nxt is None:
            row.update(outcome="no_successor_in_period", next_type=None, gap=None, next_has_360=None, skipped_events=None)
        else:
            gap = timestamp_to_seconds(nxt["timestamp"]) - timestamp_to_seconds(event["timestamp"])
            outcome = "excluded_negative_gap" if gap < 0 else "excluded_gap_over_10s" if gap > 10.0 else "retained_old"
            row.update(outcome=outcome, next_type=nxt["type"]["name"], gap=round(gap, 3),
                       next_has_360=nxt["id"] in frames, skipped_events=nxt["index"] - event["index"] - 1)
        rows.append(row)
    return funnel, rows


def main():
    excluded = load_exclusion_manifest()
    ids = [m for m in available_360_match_ids() if m not in excluded]
    with ProcessPoolExecutor(max_workers=7) as pool:
        out = list(pool.map(per_match, ids, chunksize=2))
    df = pd.DataFrame([r for _, rows in out for r in rows])
    df.to_csv(OUT / "eligible_events.csv.gz", index=False, compression="gzip")
    funnel = {k: sum(f[k] for f, _ in out) for k in out[0][0]}
    funnel["matches"] = len(ids)
    funnel["excluded_matches_by_manifest"] = len(excluded)
    funnel["eligible_current_events"] = len(df)
    funnel["outcomes"] = df.outcome.value_counts().to_dict()
    with_target = df[df.outcome != "no_successor_in_period"]
    funnel["eligible_with_successor"] = len(with_target)
    gaps = with_target.gap
    funnel["gap_percentiles_s"] = {q: round(float(gaps.quantile(q / 100)), 3) for q in (1, 5, 25, 50, 75, 90, 95, 99)}
    funnel["gap_percentiles_s"].update(min=float(gaps.min()), max=float(gaps.max()), mean=round(float(gaps.mean()), 3))
    funnel["gap_bins"] = pd.cut(gaps, [-1e9, -1e-9, 0, 2, 5, 10, 15, 30, 60, 120, 300, 1e9],
                                labels=["<0", "=0", "(0,2]", "(2,5]", "(5,10]", "(10,15]", "(15,30]", "(30,60]", "(60,120]", "(120,300]", ">300"]
                                ).value_counts().sort_index().to_dict()
    (OUT / "funnel.json").write_text(json.dumps(funnel, indent=1, default=str), encoding="utf-8")
    print(json.dumps(funnel, indent=1, default=str))


if __name__ == "__main__":
    main()
