"""Validate the rebuilt production dataset and explain every difference from
the build at commit 3423816 (backup in data/interim/).
Output: ../dataset_validation.json"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.build_statsbomb_360_dataset import MEANINGFUL_EVENT_TYPES  # noqa: E402
from src.ml_features import approved_feature_columns, load_feature_classification, select_feature_matrix, validate_schema  # noqa: E402

P = ROOT / "data" / "processed"
BACKUP = ROOT / "data" / "interim" / "processed_backup_2026-10-27_built_at_3423816"


def main():
    manifest = json.loads((P / "statsbomb_360_dataset_manifest.json").read_text())
    new = pd.read_csv(P / "statsbomb_360_all_matches_next_events.csv", low_memory=False)
    old = pd.read_csv(BACKUP / "statsbomb_360_all_matches_next_events.csv", low_memory=False)
    classification = load_feature_classification()
    validate_schema(list(new.columns), classification)
    features = select_feature_matrix(new, classification)
    per_match = {m: len(pd.read_csv(P / f"statsbomb_360_match_{m}_next_events.csv", usecols=["match_id"])) for m in manifest["rows_per_match"]}
    j = new.merge(old, on="current_event_id", suffixes=("", "_old"), how="outer", indicator=True)
    both = j[j._merge == "both"]

    def changed(a, b):
        return ~((both[a] == both[b]) | (both[a].isna() & both[b].isna()))

    actor_old_avail = both.actor_x_old.notna()
    actor_new_avail = both.actor_x.notna()
    actor_differs = changed("actor_x", "actor_x_old") | changed("actor_y", "actor_y_old")
    # offline actor now equals what the old builder put in actor_x
    offline_equals_old = ~(changed("actor_x_offline", "actor_x_old") | changed("actor_y_offline", "actor_y_old"))
    identity_cols = ["visible_teammates", "visible_opponents", "visible_ambiguous"]
    identity_changed = np.zeros(len(both), bool)
    for c in identity_cols:
        identity_changed |= changed(c, c + "_old").to_numpy()
    offline_identity_changed = np.zeros(len(both), bool)
    for c in ("visible_teammates_offline", "visible_opponents_offline", "visible_ambiguous_offline"):
        offline_identity_changed |= changed(c, c + "_old").to_numpy()
    report = {
        "matches": int(new.match_id.nunique()), "matches_manifest": manifest["matches"],
        "rows": len(new), "rows_per_match_files_total": sum(per_match.values()), "rows_manifest": manifest["rows"],
        "rows_previous_build": len(old),
        "rows_only_in_new": int((j._merge == "left_only").sum()), "rows_only_in_previous": int((j._merge == "right_only").sum()),
        "duplicate_event_ids": int(new.current_event_id.duplicated().sum()),
        "duplicate_rows": int(new.duplicated().sum()),
        "invalid_target_labels": int((~new.next_event_type.isin(MEANINGFUL_EVENT_TYPES)).sum()),
        "missing_targets": int(new.next_event_type.isna().sum()),
        "class_distribution_pct": (100 * new.next_event_type.value_counts(normalize=True)).round(4).to_dict(),
        "targets_identical_to_previous": bool((both.next_event_type == both.next_event_type_old).all()),
        "schema_classified_exactly": True,
        "approved_feature_matrix_shape": list(features.shape),
        "approved_features": approved_feature_columns(classification),
        "count_invariant_violations_causal": int((new.visible_teammates + new.visible_opponents + new.visible_ambiguous != new.visible_players).sum()),
        "count_invariant_violations_offline": int((new.visible_teammates_offline + new.visible_opponents_offline + new.visible_ambiguous_offline != new.visible_players).sum()),
        "raw_count_violations": int((new.visible_teammates_raw + new.visible_opponents_raw != new.visible_players).sum()),
        "orientation_causal": new.orientation.value_counts().to_dict(),
        "orientation_causal_routes": new.orientation_resolved_by.value_counts().to_dict(),
        "orientation_available_false": int((~new.orientation_available).sum()),
        "actor_missing_while_orientation_available": int((new.orientation_available & new.actor_x.isna()).sum()),
        "actor_present_while_orientation_unavailable": int((~new.orientation_available & new.actor_x.notna()).sum()),
        "orientation_uses_later_records": int(new.orientation_uses_later_records.sum()),
        "orientation_offline_routes": new.orientation_offline_resolved_by.value_counts().to_dict(),
        "identity_quality": new.identity_quality.value_counts().to_dict(),
        "identity_quality_offline": new.identity_quality_offline.value_counts().to_dict(),
        "vs_previous_build": {
            "actor_xy_changed": int(actor_differs.sum()),
            "actor_xy_now_unavailable": int((actor_old_avail & ~actor_new_avail).sum()),
            "actor_xy_changed_while_available": int((actor_differs & actor_new_avail & actor_old_avail).sum()),
            "offline_actor_equals_previous_actor": int(offline_equals_old.sum()),
            "offline_actor_differs_from_previous_actor": int((~offline_equals_old).sum()),
            "causal_identity_counts_changed": int(identity_changed.sum()),
            "offline_identity_counts_changed": int(offline_identity_changed.sum()),
            "orientation_offline_vs_previous_orientation_changed": int(changed("orientation_offline", "orientation_old").sum()),
        },
    }
    (OUT / "dataset_validation.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
