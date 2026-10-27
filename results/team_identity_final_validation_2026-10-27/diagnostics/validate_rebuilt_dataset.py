"""Validation of the rebuilt production dataset in data/processed/.
Output: ../rebuilt_dataset_validation.json"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
manifest = json.loads((P / "statsbomb_360_dataset_manifest.json").read_text())
df = pd.read_csv(P / "statsbomb_360_all_matches_next_events.csv", low_memory=False)
mem = pd.read_csv(OUT / "ml" / "rows.csv.gz", low_memory=False)
per_match_rows = sum(len(pd.read_csv(P / f"statsbomb_360_match_{m}_next_events.csv", usecols=["match_id"])) for m in manifest["rows_per_match"])
expected = ["visible_teammates", "visible_opponents", "visible_ambiguous", "identity_corrected_points", "identity_quality",
            "visible_teammates_raw", "visible_opponents_raw", "visible_teammates_offline", "visible_opponents_offline",
            "visible_ambiguous_offline", "identity_quality_offline", "orientation", "orientation_problem_a",
            "orientation_resolved_by", "orientation_uses_later_records", "actor_x", "actor_y", "next_event_type"]
joined = df.merge(mem, on="current_event_id", suffixes=("", "_mem"))
compare = ["visible_teammates", "visible_opponents", "visible_ambiguous", "identity_quality", "visible_teammates_offline",
           "visible_ambiguous_offline", "orientation", "next_event_type", "actor_x", "actor_y"]
mismatch = {c: int((joined[c].astype(str) != joined[c + "_mem"].astype(str)).sum()) for c in compare}
counts_ok = (df.visible_teammates + df.visible_opponents + df.visible_ambiguous == df.visible_players)
report = {
    "rows_combined": len(df), "rows_per_match_files": per_match_rows, "rows_manifest": manifest["rows"],
    "matches": int(df.match_id.nunique()), "duplicate_event_ids": int(df.current_event_id.duplicated().sum()),
    "missing_expected_columns": [c for c in expected if c not in df.columns],
    "columns": list(df.columns),
    "rows_matched_with_in_memory_validation_build": len(joined),
    "mismatches_vs_in_memory_build": mismatch,
    "count_invariant_violations": int((~counts_ok).sum()),
    "raw_count_invariant_violations": int((df.visible_teammates_raw + df.visible_opponents_raw != df.visible_players).sum()),
    "next_event_type_distribution_pct": (100 * df.next_event_type.value_counts(normalize=True)).round(4).to_dict(),
    "orientation": df.orientation.value_counts().to_dict(),
    "orientation_uses_later_records": int(df.orientation_uses_later_records.sum()),
    "identity_quality": df.identity_quality.value_counts().to_dict(),
    "identity_quality_offline": df.identity_quality_offline.value_counts().to_dict(),
    "rows_with_uncertain_player": int((df.visible_ambiguous > 0).sum()),
    "actor_missing": int(df.actor_x.isna().sum()),
    "manifest_git_commit": manifest["git_commit"], "manifest_src_dirty": manifest["src_had_uncommitted_changes"],
}
(OUT / "rebuilt_dataset_validation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "columns"}, indent=1))
