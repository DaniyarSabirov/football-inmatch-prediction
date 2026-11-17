"""Independent re-check of experiment 3 (does NOT import src.sequence_models).

1. no final-test match in any experiment-3 feature table;
2. Scenario D targets and evaluation rows re-derived from the raw JSON for a sample of Tier B
   matches equal the feature table;
3. distance to goal and shooting angle recomputed from the raw locations;
4. log loss, Brier, macro-F1 (argmax) and Shot average precision recomputed with scikit-learn
   from the stored Tier B probabilities equal the reported values;
5. Shot threshold rules re-applied to the stored probabilities reproduce the reported recall / FP rate.

Usage: .venv/bin/python scripts/verify_context_aware_v3.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "context_aware_event_prediction_v3"
CACHE = ROOT / "data" / "interim" / "sequence_models" / "v3"
EVENTS = ROOT / "data" / "raw" / "statsbomb" / "data" / "events"
D15 = ["Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed", "Pressure", "Duel",
       "Interception", "Block", "Goal Keeper", "Foul Committed", "50/50"]


def main() -> int:
    split = json.loads((ROOT / "results" / "markov_hmm_pilot" / "split_manifest.json").read_text())
    final = set(split["final_test_match_ids"])
    report: dict = {}
    tables = sorted(CACHE.glob("features_*.parquet"))
    report["final_test_matches_in_feature_tables"] = sorted(
        {int(m) for p in tables for m in pd.read_parquet(p, columns=["match_id"])["match_id"].unique()} & final)

    tab = pd.read_parquet(CACHE / "features_tier_b_scenarioD15.parquet")
    sample = sorted(tab["match_id"].unique())[:5]
    target_mismatch, geo_dev = 0, 0.0
    for mid in sample:
        events = sorted(json.loads((EVENTS / f"{mid}.json").read_text()), key=lambda e: e["index"])
        kept = [e for e in events if e["period"] < 5 and e["type"]["name"] in D15]
        nxt = {}
        for a, b in zip(kept, kept[1:]):
            if a["period"] == b["period"]:
                nxt[a["id"]] = b["type"]["name"]
        sub = tab[tab["match_id"] == mid].set_index("event_id")
        for e in kept:
            want = nxt.get(e["id"])
            got = sub.at[e["id"], "target"]
            if (want is None) != (got is None or (isinstance(got, float) and math.isnan(got))) or (want and want != got):
                target_mismatch += 1
            if "location" in e:
                x, y = min(max(e["location"][0], 0), 120), min(max(e["location"][1], 0), 80)
                d = math.hypot(120 - x, 40 - y)
                ang = abs(math.atan2(44 - y, 120 - x) - math.atan2(36 - y, 120 - x))
                geo_dev = max(geo_dev, abs(d - sub.at[e["id"], "distance_to_goal"]),
                              abs(ang - sub.at[e["id"], "shooting_angle"]))
    report["targets_rederived_from_raw_json"] = {"matches": sample, "mismatches": target_mismatch}
    report["geometry_max_abs_dev"] = geo_dev

    summary = pd.read_csv(OUT / "tier_b" / "summary.csv")
    proba = np.load(CACHE / "tier_b_scenarioD15_eval_proba.npz")
    ev = tab[tab["eval_eligible"]]
    y = ev["target_id"].to_numpy()
    shot = D15.index("Shot")
    dev = {}
    for m in proba.files:
        p = proba[m]
        row = summary[(summary["task"] == "scenarioD15") & (summary["model"] == m) & (summary["decision_rule"] == "uniform")].iloc[0]
        onehot = np.eye(len(D15))[y]
        # unclipped log loss (scikit-learn clips p at machine epsilon; reported separately below)
        mine = {"log_loss": float(-np.log(p[np.arange(len(y)), y]).mean()), "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
                "macro_f1": f1_score(y, p.argmax(axis=1), labels=range(len(D15)), average="macro", zero_division=0),
                "shot_average_precision": average_precision_score(y == shot, p[:, shot])}
        dev[m] = max(abs(mine[k] - row[k]) for k in mine)
        report.setdefault("sklearn_clipped_log_loss_minus_reported", {})[m] = float(
            log_loss(y, p, labels=range(len(D15))) - row["log_loss"])
        report.setdefault("min_true_class_probability", {})[m] = float(p[np.arange(len(y)), y].min())
    report["metric_recomputation_max_abs_dev_by_model"] = dev
    thr = pd.read_csv(OUT / "tier_b" / "shot_thresholds.csv")
    thr = thr[thr["task"] == "scenarioD15"]
    tdev = 0.0
    for _, r in thr.iterrows():
        d = proba[r["model"]][:, shot] >= r["threshold_from_tier_a"]
        rec = (d & (y == shot)).sum() / (y == shot).sum()
        fp = 1000 * (d & (y != shot)).sum() / len(y)
        tdev = max(tdev, abs(rec - r["shot_recall"]), abs(fp - r["fp_per_1000_events"]))
    report["shot_threshold_rule_max_abs_dev"] = tdev
    report["passed"] = bool(not report["final_test_matches_in_feature_tables"] and target_mismatch == 0
                            and geo_dev < 1e-9 and max(dev.values()) < 1e-9 and tdev < 1e-9)
    (OUT / "independent_verification.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
