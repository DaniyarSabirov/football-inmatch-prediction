"""Validate the dataset rebuilt without the 10-second rule.

- Targets are re-derived INDEPENDENTLY from the raw StatsBomb event stream
  (own loop: next event in list order, same period, type in the 8 meaningful
  types) and compared with every row.
- Old rows (backup built at 021867b) must be a subset of the new rows, with
  identical values in every non-target column (features, identity,
  orientation, offline diagnostics): removing the window must not change any
  feature.
- New rows must be exactly the events the old builder excluded only by the
  10-second rule (../eligible_events.csv.gz, computed before the change).
Output: ../rebuild_validation.json
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
from src.ml_features import load_feature_classification, select_feature_matrix, validate_schema  # noqa: E402

P = ROOT / "data" / "processed"
BACKUP = ROOT / "data" / "interim" / "processed_backup_2026-10-27_built_at_021867b"
RAW = ROOT / "data" / "raw" / "statsbomb" / "data"
TYPES = {"Pass", "Carry", "Dribble", "Shot", "Miscontrol", "Ball Recovery", "Clearance", "Dispossessed"}


def independent_targets(match_id):
    events = json.loads((RAW / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"] for f in json.loads((RAW / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    out = {}
    for i, e in enumerate(events):
        if e["type"]["name"] not in TYPES or e["id"] not in frames:
            continue
        for later in events[i + 1:]:
            if later["period"] != e["period"]:
                break
            if later["type"]["name"] in TYPES:
                out[e["id"]] = (later["id"], later["type"]["name"])
                break
    return out


def main():
    new = pd.read_csv(P / "statsbomb_360_all_matches_next_events.csv", low_memory=False)
    old = pd.read_csv(BACKUP / "statsbomb_360_all_matches_next_events.csv", low_memory=False)
    eligible = pd.read_csv(OUT / "eligible_events.csv.gz")
    classification = load_feature_classification()
    validate_schema(list(new.columns), classification)
    features = select_feature_matrix(new, classification)

    with ProcessPoolExecutor(max_workers=7) as pool:
        truth = {}
        for part in pool.map(independent_targets, sorted(new.match_id.unique()), chunksize=4):
            truth.update(part)
    target_ok = pd.Series([truth.get(e) == (n, t) for e, n, t in zip(new.current_event_id, new.next_event_id, new.next_event_type)])

    added = set(new.current_event_id) - set(old.current_event_id)
    removed = set(old.current_event_id) - set(new.current_event_id)
    expected_added = set(eligible[eligible.outcome == "excluded_gap_over_10s"].event_id)
    non_target = [c for c, e in classification["columns"].items() if e["category"] != "TARGET"]
    common = new.merge(old, on="current_event_id", suffixes=("", "_old"))
    diff_cells = {}
    for c in non_target + ["next_event_id", "next_event_type", "next_team", "next_player", "same_team", "same_possession",
                           "time_to_next_event_seconds"]:
        if c == "current_event_id":
            continue
        a, b = common[c], common[c + "_old"]
        same = (a == b) | (a.isna() & b.isna())
        diff_cells[c] = int((~same).sum())

    def dist(frame):
        return frame.next_event_type.value_counts()

    old_d, new_d = dist(old), dist(new)
    report = {
        "matches_new": int(new.match_id.nunique()), "matches_old": int(old.match_id.nunique()),
        "rows_new": len(new), "rows_old": len(old), "rows_added": len(added), "rows_removed": len(removed),
        "added_rows_equal_10s_exclusions": added == expected_added, "expected_added": len(expected_added),
        "duplicate_event_ids": int(new.current_event_id.duplicated().sum()), "duplicate_rows": int(new.duplicated().sum()),
        "targets_checked_against_raw_stream": len(new), "targets_matching_independent_derivation": int(target_ok.sum()),
        "rows_without_independent_target": int(sum(1 for e in new.current_event_id if e not in truth)),
        "eligible_without_row": len(set(truth) - set(new.current_event_id)),
        "gap_over_10s_rows_now_present": int((new.time_to_next_event_seconds > 10).sum()),
        "max_gap_s": float(new.time_to_next_event_seconds.max()), "min_gap_s": float(new.time_to_next_event_seconds.min()),
        "zero_gap_rows": int((new.time_to_next_event_seconds == 0).sum()),
        "rows_by_period_new": new.period.value_counts().sort_index().to_dict(),
        "cells_changed_in_rows_present_in_both_builds": diff_cells,
        "class_counts_old": old_d.to_dict(), "class_counts_new": new_d.to_dict(),
        "class_pct_old": (100 * old_d / old_d.sum()).round(4).to_dict(), "class_pct_new": (100 * new_d / new_d.sum()).round(4).to_dict(),
        "imbalance_ratio_old": round(old_d.max() / old_d.min(), 3), "imbalance_ratio_new": round(new_d.max() / new_d.min(), 3),
        "approved_feature_matrix_shape": list(features.shape),
        "time_gap_column_approved_as_feature": "time_to_next_event_seconds" in features.columns,
        "count_invariant_violations": int((new.visible_teammates + new.visible_opponents + new.visible_ambiguous != new.visible_players).sum()),
        "offline_count_invariant_violations": int((new.visible_teammates_offline + new.visible_opponents_offline + new.visible_ambiguous_offline != new.visible_players).sum()),
        "orientation_available_false": int((~new.orientation_available).sum()),
        "actor_present_while_orientation_unavailable": int((~new.orientation_available & new.actor_x.notna()).sum()),
        "identity_quality": new.identity_quality.value_counts().to_dict(),
        "orientation_uses_later_records": int(new.orientation_uses_later_records.sum()),
    }
    (OUT / "rebuild_validation.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
