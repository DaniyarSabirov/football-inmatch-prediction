"""Independent re-check of the Markov/HMM pilot outputs.

Uses only the written artefacts (predictions.csv, metrics_summary.csv,
split_manifest.json), the raw StatsBomb event JSON and the production
per-match CSVs. It deliberately does NOT import src.sequence_models: the
first-order Markov chain is recounted with plain Python from the raw files,
and the metrics are recomputed with scikit-learn directly.

Usage:
    .venv/bin/python scripts/verify_markov_hmm_pilot.py
Writes results/markov_hmm_pilot/independent_verification.json; exits 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "markov_hmm_pilot"
EVENTS = ROOT / "data" / "raw" / "statsbomb" / "data" / "events"
TYPES = ["Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed"]


def filtered_transitions(match_id: int) -> list[tuple[str, str]]:
    events = json.loads((EVENTS / f"{match_id}.json").read_text(encoding="utf-8"))
    pairs, previous = [], {}
    for event in sorted(events, key=lambda e: e["index"]):
        period, typ = event["period"], event["type"]["name"]
        if period == 5 or typ not in TYPES:
            continue
        if period in previous:
            pairs.append((previous[period], typ))
        previous = {period: typ}  # a new period starts a new sequence
    return pairs


def main() -> int:
    manifest = json.loads((OUT / "split_manifest.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(OUT / "predictions.csv", low_memory=False)
    summary = pd.read_csv(OUT / "metrics_summary.csv")
    report: dict = {}

    report["no_final_test_match_in_predictions"] = not (set(preds["match_id"]) & set(manifest["final_test_match_ids"]))
    report["predictions_only_pilot_matches"] = set(preds["match_id"]) <= set(manifest["pilot_match_ids"])
    fold_of_match = {m: f["fold_id"] for f in manifest["pilot_folds"] for m in f["validation_match_ids"]}
    report["fold_ids_match_split_manifest"] = bool((preds["match_id"].map(fold_of_match) == preds["fold_id"]).all())

    # targets vs the production dataset
    prod = pd.concat([pd.read_csv(ROOT / "data" / "processed" / f"statsbomb_360_match_{m}_next_events.csv",
                                  usecols=["current_event_id", "next_event_type", "period"])
                      for m in manifest["pilot_match_ids"]])
    prod = prod[prod["period"] != 5].set_index("current_event_id")["next_event_type"]
    elig = preds[(preds["model_name"] == "markov_order1") & preds["eval_eligible"]].set_index("event_id")
    report["evaluation_rows_equal_production_rows_minus_shootout"] = set(elig.index) == set(prod.index)
    report["targets_equal_production_targets"] = bool((elig["true_next_event"] == prod.loc[elig.index]).all())

    # Markov chain recounted from raw JSON per fold
    max_dev = 0.0
    mc = preds[preds["model_name"] == "markov_order1"]
    for fold in manifest["pilot_folds"]:
        counts = defaultdict(lambda: np.zeros(len(TYPES)))
        for m in fold["train_match_ids"]:
            for a, b in filtered_transitions(m):
                counts[a][TYPES.index(b)] += 1
        rows = mc[mc["fold_id"] == fold["fold_id"]]
        for current, group in rows.groupby("current_event"):
            expected = (counts[current] + 1.0) / (counts[current].sum() + len(TYPES))
            got = group[[f"p_{t}" for t in TYPES]].to_numpy()
            max_dev = max(max_dev, float(np.abs(got - expected).max()))
    report["markov_order1_probabilities_recounted_from_raw_json_max_abs_dev"] = max_dev

    # metrics recomputed with scikit-learn
    deviations = {}
    for model, g in preds[preds["eval_eligible"]].groupby("model_name"):
        p = g[[f"p_{t}" for t in TYPES]].to_numpy()
        y = g["true_next_event"].map(TYPES.index).to_numpy()
        pred = p.argmax(axis=1)
        mine = {"accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, labels=range(len(TYPES)), average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "log_loss": log_loss(y, p, labels=range(len(TYPES)))}
        row = summary[(summary["task"] == "filtered8") & (summary["model"] == model)].iloc[0]
        deviations[model] = max(abs(mine[k] - row[k]) for k in mine)
        report.setdefault("probability_rows_sum_to_one_max_dev", 0.0)
        report["probability_rows_sum_to_one_max_dev"] = max(report["probability_rows_sum_to_one_max_dev"],
                                                            float(np.abs(p.sum(axis=1) - 1).max()))
    report["metric_recomputation_max_abs_dev_per_model"] = deviations

    passed = (report["no_final_test_match_in_predictions"] and report["predictions_only_pilot_matches"]
              and report["fold_ids_match_split_manifest"] and report["evaluation_rows_equal_production_rows_minus_shootout"]
              and report["targets_equal_production_targets"]
              and report["markov_order1_probabilities_recounted_from_raw_json_max_abs_dev"] < 1e-12
              and max(deviations.values()) < 1e-9 and report["probability_rows_sum_to_one_max_dev"] < 1e-9)
    report["passed"] = bool(passed)
    (OUT / "independent_verification.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
