"""Breakdowns of the 10-second exclusions (input: ../eligible_events.csv.gz).
Output: ../breakdowns.json and ../tables/*.csv"""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1]
(OUT / "tables").mkdir(exist_ok=True)
df = pd.read_csv(OUT / "eligible_events.csv.gz")
t = df[df.outcome != "no_successor_in_period"].copy()
t["excluded_10s"] = t.outcome == "excluded_gap_over_10s"
S = {}


def rate(col):
    g = t.groupby(col).agg(eligible=("excluded_10s", "size"), excluded=("excluded_10s", "sum"))
    g["excluded_pct"] = (100 * g.excluded / g.eligible).round(3)
    g.to_csv(OUT / "tables" / f"exclusion_by_{col}.csv")
    return g


for col in ("current_type", "next_type", "period"):
    S[f"by_{col}"] = rate(col).reset_index().to_dict("records")
m = rate("match_id")
S["by_match"] = {"matches": len(m), "excluded_pct": m.excluded_pct.describe().round(3).to_dict(),
                 "max_match": m.excluded_pct.idxmax(), "min_match": m.excluded_pct.idxmin()}
old = t[~t.excluded_10s].next_type.value_counts()
new = t.next_type.value_counts()
cls = pd.DataFrame({"old_rows": old, "new_rows": new})
cls["added"] = cls.new_rows - cls.old_rows
cls["old_pct"] = (100 * cls.old_rows / cls.old_rows.sum()).round(4)
cls["new_pct"] = (100 * cls.new_rows / cls.new_rows.sum()).round(4)
cls["class_growth_pct"] = (100 * cls.added / cls.old_rows).round(2)
cls.to_csv(OUT / "tables" / "class_distribution_old_vs_new.csv")
S["class_distribution"] = cls.reset_index().rename(columns={"index": "next_type"}).to_dict("records")
S["imbalance_ratio_majority_to_minority"] = {"old": round(old.max() / old.min(), 3), "new": round(new.max() / new.min(), 3)}
# gaps by next type among excluded / all
S["gap_percentiles_by_next_type"] = t.groupby("next_type").gap.describe(percentiles=[.5, .9, .95, .99]).round(2).reset_index().to_dict("records")
# transition pairs most affected
pairs = t.groupby(["current_type", "next_type"]).agg(eligible=("excluded_10s", "size"), excluded=("excluded_10s", "sum"))
pairs["excluded_pct"] = (100 * pairs.excluded / pairs.eligible).round(2)
pairs.sort_values("excluded", ascending=False).to_csv(OUT / "tables" / "exclusion_by_transition.csv")
S["top_excluded_transitions"] = pairs.sort_values("excluded", ascending=False).head(10).reset_index().to_dict("records")
S["excluded_next_has_360_pct"] = round(100 * t[t.excluded_10s].next_has_360.mean(), 2)
S["retained_next_has_360_pct"] = round(100 * t[~t.excluded_10s].next_has_360.mean(), 2)
S["empty_freeze_frame_current"] = int(df.empty_freeze_frame.sum())
(OUT / "breakdowns.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
pd.set_option("display.width", 200)
for k in ("by_current_type", "by_next_type", "by_period"):
    print(k); print(pd.DataFrame(S[k]).to_string(index=False))
print(S["by_match"]); print(cls.to_string()); print(S["imbalance_ratio_majority_to_minority"])
print(pd.DataFrame(S["top_excluded_transitions"]).to_string(index=False))
print("next has 360: excluded", S["excluded_next_has_360_pct"], "retained", S["retained_next_has_360_pct"], "| empty current frames", S["empty_freeze_frame_current"])
