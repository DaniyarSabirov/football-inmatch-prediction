"""Compare event taxonomies A-D on the raw data (417 eligible matches), using
the production target semantics: an INPUT row is an event whose type is in
the scenario's input set and that has a 360 record; its TARGET is the first
later event in the raw stream (event order) whose type is in the scenario's
target set, within the same period. Nothing is trained; predictability is
measured as dataset properties.

Per scenario (../scenarios.json, ../scenario_targets.csv):
  input_rows / inputs_without_successor / matches
  target classes, counts, %, rare classes, imbalance ratio, majority share
  % of raw events whose type is in the target set / in the input set
  % of raw within-period transitions represented exactly (input -> its
    immediate next raw event is the target, i.e. nothing skipped)
  % of targets without their own 360 record
  % of rows with a zero time gap (simultaneous)
  % of rows that skip >= 1 intermediate raw event; mean skipped; top skipped types
  conditional entropy H(target | current type) in bits, the mutual information
  with the current type, the share of rows in (current type -> target) cells
  with P >= 0.9 (near-deterministic) and P >= 0.7, and the largest cells
"""
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.animate_statsbomb_360_match import available_360_match_ids  # noqa: E402
from src.build_statsbomb_360_dataset import MEANINGFUL_EVENT_TYPES  # noqa: E402
from src.match_exclusions import load_exclusion_manifest  # noqa: E402

DATA = ROOT / "data" / "raw" / "statsbomb" / "data"
CATS = json.loads((OUT / "event_type_categories.json").read_text())["types"]
ALL = set(CATS)
FOOTBALL = {t for t, c in CATS.items() if c in ("on_ball_action", "defensive_action", "goalkeeper_action")} | {
    "Foul Committed", "Ball Receipt*", "Foul Won", "Dribbled Past"}
COMPANIONS = {"Ball Receipt*", "Foul Won", "Dribbled Past"}  # >= 90% mechanical counterparts (companions.csv / README)
MIN_COUNT = 1000
TYPE_COUNTS = pd.read_csv(OUT / "type_stats.csv").set_index("event_type")["count"]
D = {t for t in FOOTBALL - COMPANIONS if TYPE_COUNTS[t] >= MIN_COUNT}
SCENARIOS = {
    "A_current_8": (set(MEANINGFUL_EVENT_TYPES), set(MEANINGFUL_EVENT_TYPES)),
    "B_all_raw_types": (ALL, ALL),
    "C_all_football_actions": (FOOTBALL, FOOTBALL),
    "D_expanded_selected": (D, D),
}


def ms(ts):
    h, m, rest = ts.split(":")
    s, _, f = rest.partition(".")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int((f + "000")[:3])


def per_match(match_id):
    events = json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"] for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    out = {}
    for name, (inputs, targets) in SCENARIOS.items():
        r = defaultdict(Counter)
        for i, e in enumerate(events):
            if e["type"]["name"] not in inputs or e["id"] not in frames:
                continue
            target, skipped = None, []
            for later in events[i + 1:]:
                if later["period"] != e["period"]:
                    break
                if later["type"]["name"] in targets:
                    target = later
                    break
                skipped.append(later["type"]["name"])
            if target is None:
                r["c"]["no_successor"] += 1
                continue
            cur, tgt = e["type"]["name"], target["type"]["name"]
            r["c"]["rows"] += 1
            r["target"][tgt] += 1
            r["cell"][(cur, tgt)] += 1
            r["c"]["target_without_360"] += target["id"] not in frames
            r["c"]["zero_gap"] += ms(target["timestamp"]) == ms(e["timestamp"])
            r["c"]["skipped_any"] += bool(skipped)
            r["c"]["skipped_total"] += len(skipped)
            r["skipped_types"].update(skipped)
        out[name] = {k: dict(v) for k, v in r.items()}
    return match_id, out


def entropy(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c)


def main():
    excluded = load_exclusion_manifest()
    ids = [m for m in available_360_match_ids() if m not in excluded]
    with ProcessPoolExecutor(max_workers=7) as pool:
        results = list(pool.map(per_match, ids, chunksize=2))
    raw = json.loads((OUT / "raw_counts.json").read_text())
    transitions = pd.read_csv(OUT / "transitions.csv")
    summary, target_rows = {}, []
    for name, (inputs, targets) in SCENARIOS.items():
        agg = defaultdict(Counter)
        matches = 0
        for _, out in results:
            o = out[name]
            matches += o.get("c", {}).get("rows", 0) > 0
            for k, v in o.items():
                agg[k].update(v)
        c, tgt, cell = agg["c"], agg["target"], agg["cell"]
        rows = c["rows"]
        by_cur = defaultdict(Counter)
        for (cur, t), n in cell.items():
            by_cur[cur][t] += n
        h_cond = sum(sum(v.values()) / rows * entropy(v) for v in by_cur.values())
        near_det = sum(n for (cur, t), n in cell.items() if n / sum(by_cur[cur].values()) >= 0.9)
        exact = transitions[transitions.current.isin(inputs) & transitions.next.isin(targets)]["count"].sum()
        counts = pd.Series(tgt).sort_values(ascending=False)
        summary[name] = {
            "input_types": sorted(inputs), "target_types": sorted(targets),
            "n_input_types": len(inputs), "n_target_types_defined": len(targets), "n_target_classes_observed": len(counts),
            "input_rows": rows, "inputs_without_successor": c["no_successor"], "matches_represented": matches,
            "rows_vs_A": None,
            "pct_raw_events_in_target_set": round(100 * sum(TYPE_COUNTS[t] for t in targets if t in TYPE_COUNTS) / raw["events"], 2),
            "pct_raw_events_in_input_set": round(100 * sum(TYPE_COUNTS[t] for t in inputs if t in TYPE_COUNTS) / raw["events"], 2),
            "pct_raw_transitions_between_selected_types_upper_bound": round(100 * exact / raw["raw_transitions_within_period"], 2),
            "pct_rows_with_no_skipped_event": round(100 * (rows - c["skipped_any"]) / rows, 2),
            "pct_targets_without_own_360": round(100 * c["target_without_360"] / rows, 2),
            "pct_zero_gap_rows": round(100 * c["zero_gap"] / rows, 2),
            "pct_rows_skipping_intermediate_events": round(100 * c["skipped_any"] / rows, 2),
            "mean_skipped_events_per_row": round(c["skipped_total"] / rows, 3),
            "top_skipped_types": dict(agg["skipped_types"].most_common(8)),
            "majority_class": counts.index[0], "majority_share_pct": round(100 * counts.iloc[0] / rows, 2),
            "smallest_class": counts.index[-1], "smallest_class_count": int(counts.iloc[-1]),
            "imbalance_ratio": round(counts.iloc[0] / counts.iloc[-1], 1),
            "classes_below_1pct": int((counts / rows < 0.01).sum()), "classes_below_0_1pct": int((counts / rows < 0.001).sum()),
            "classes_below_1000_rows": int((counts < 1000).sum()),
            "target_entropy_bits": round(entropy(Counter(tgt)), 3), "log2_classes": round(math.log2(len(counts)), 3),
            "conditional_entropy_given_current_type_bits": round(h_cond, 3),
            "pct_rows_near_deterministic_given_current_type": round(100 * near_det / rows, 2),
            "mutual_information_current_type_target_bits": round(entropy(Counter(tgt)) - h_cond, 3),
            "share_of_target_entropy_explained_by_current_type_pct": round(100 * (1 - h_cond / entropy(Counter(tgt))), 1),
            "pct_rows_in_cells_with_p_ge_0_7": round(100 * sum(n for (cur, t), n in cell.items() if n / sum(by_cur[cur].values()) >= 0.7) / rows, 2),
            "largest_cells": [(cur, t, n, round(100 * n / rows, 2), round(n / sum(by_cur[cur].values()), 3))
                              for (cur, t), n in sorted(cell.items(), key=lambda x: -x[1])[:8]],
            "near_deterministic_cells": sorted(
                [(cur, t, n, round(n / sum(by_cur[cur].values()), 3)) for (cur, t), n in cell.items()
                 if n / sum(by_cur[cur].values()) >= 0.9], key=lambda x: -x[2])[:10],
            "estimated_csv_size_mb": round(rows * 351 / 1e6),
        }
        for t, n in counts.items():
            target_rows.append({"scenario": name, "target": t, "rows": int(n), "pct": round(100 * n / rows, 4)})
    a = summary["A_current_8"]["input_rows"]
    for s in summary.values():
        s["rows_vs_A"] = f"{s['input_rows'] - a:+,} ({100 * (s['input_rows'] / a - 1):+.1f}%)"
    (OUT / "scenarios.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    pd.DataFrame(target_rows).to_csv(OUT / "scenario_targets.csv", index=False)
    for name, s in summary.items():
        print("==", name)
        for k, v in s.items():
            if k not in ("input_types", "target_types", "near_deterministic_cells", "largest_cells"):
                print(f"  {k}: {v}")
        print("  near_deterministic_cells:", s["near_deterministic_cells"][:6])
        print("  largest_cells (cur, target, rows, % of rows, P(target|cur)):", s["largest_cells"])


if __name__ == "__main__":
    main()
