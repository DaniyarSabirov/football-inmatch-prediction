"""Independent re-check of experiment 2 (does NOT import src.sequence_models).

From the written artefacts (predictions.parquet, cost_matrix_definitions.json,
model/cost comparison tables, split manifest) and the raw StatsBomb JSON:

1. no final-test match appears in the predictions; tiers are disjoint;
2. log loss, Brier, macro-F1 and balanced accuracy recomputed with scikit-learn
   from the stored probabilities and decisions match the reported values
   (probabilities are stored as float32: tolerance 1e-5 for log loss / Brier);
3. log loss and Brier are identical across cost schemes of the same model;
4. Bayes decisions recomputed with a plain loop over the stored cost matrices
   agree with the stored decisions (near-ties can flip only through float32 rounding);
5. the experiment-1 Laplace Markov chain of Scenario D (Tier B, fold 0) is
   recounted from the raw JSON and matches the stored probabilities.

Usage: .venv/bin/python scripts/verify_markov_hmm_cost_sensitive_v2.py
Writes results/markov_hmm_cost_sensitive_v2/independent_verification.json; exit 1 on failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "markov_hmm_cost_sensitive_v2"
EVENTS = ROOT / "data" / "raw" / "statsbomb" / "data" / "events"


def main() -> int:
    global OUT
    if len(sys.argv) > 1:  # optional alternative output directory (e.g. a smoke run)
        OUT = Path(sys.argv[1])
    split = json.loads((OUT / "split_manifest.json").read_text(encoding="utf-8"))
    costs = json.loads((OUT / "cost_matrix_definitions.json").read_text(encoding="utf-8"))
    comp = pd.read_csv(OUT / "cost_sensitive_comparison.csv")
    allm = pd.read_csv(OUT / "model_comparison.csv")
    preds = pd.read_parquet(OUT / "predictions.parquet")
    report: dict = {}
    final = set(split["final_test_match_ids"])
    report["no_final_test_match_in_predictions"] = not (set(preds["match_id"]) & final)
    report["tiers_disjoint"] = not (set(split["tier_a"]["match_ids"]) & set(split["tier_b"]["match_ids"]))

    metric_dev, invariance_dev, decision_agreement = {}, {}, {}
    for (tier, task), g in preds.groupby(["tier", "task"]):
        vocab = costs["per_tier"][tier][task]["vocabulary"]
        y = g["true_next_event"].map({c: i for i, c in enumerate(vocab)}).to_numpy()
        models = sorted({c.split("__")[1] for c in g.columns if c.startswith("p__")})
        for m in models:
            p = g[[f"p__{m}__{c}" for c in vocab]].to_numpy(dtype=float)
            if np.isnan(p).all():
                continue
            p = p / p.sum(axis=1, keepdims=True)
            ll = log_loss(y, p, labels=range(len(vocab)))
            onehot = np.eye(len(vocab))[y]
            br = float(((p - onehot) ** 2).sum(axis=1).mean())
            rows = allm[(allm["tier"] == tier) & (allm["task"] == task) & (allm["model"] == m)]
            dev = max(abs(ll - rows["log_loss"].iloc[0]), abs(br - rows["brier"].iloc[0]))
            for col in [c for c in g.columns if c.startswith(f"decision__{m}__")]:
                scheme = col.split("__")[2]
                if g[col].isna().all():  # e.g. no tuned scheme on Tier A (gamma is tuned there)
                    continue
                d = g[col].map({c: i for i, c in enumerate(vocab)}).to_numpy()
                ref = comp[(comp["tier"] == tier) & (comp["task"] == task) & (comp["model"] == m) & (comp["cost_scheme"] == scheme)]
                if ref.empty:
                    ref = allm[(allm["tier"] == tier) & (allm["task"] == task) & (allm["model"] == m)]
                dev = max(dev, abs(f1_score(y, d, labels=range(len(vocab)), average="macro", zero_division=0)
                                   - ref["macro_f1"].iloc[0]),
                          abs(balanced_accuracy_score(y, d) - ref["balanced_accuracy"].iloc[0]))
                # independent Bayes decision: loop over actions, per-fold matrices
                if scheme in ("moderate", "strong") or scheme == "uniform":
                    agree = 0
                    for f, gf in g.assign(_d=d, _i=np.arange(len(g))).groupby("fold_id"):
                        c = np.array(costs["per_tier"][tier][task]["folds"][f"fold{f}"][scheme]["matrix"])
                        pf = p[gf["_i"].to_numpy()]
                        risk = np.stack([(pf * c[a][None, :]).sum(axis=1) for a in range(len(vocab))], axis=1)
                        best = np.array([int(np.flatnonzero(r <= r.min() + 1e-12 * max(1, abs(r.min())))[0]) for r in risk])
                        agree += int((best == gf["_d"].to_numpy()).sum())
                    decision_agreement[f"{tier}|{task}|{m}|{scheme}"] = agree / len(g)
            metric_dev[f"{tier}|{task}|{m}"] = dev
            sub = comp[(comp["tier"] == tier) & (comp["task"] == task) & (comp["model"] == m)]
            if len(sub) > 1:
                invariance_dev[f"{tier}|{task}|{m}"] = float(max(sub["log_loss"].max() - sub["log_loss"].min(),
                                                                 sub["brier"].max() - sub["brier"].min()))
    report["metric_recomputation_max_abs_dev"] = max(metric_dev.values())
    report["metric_recomputation_by_model"] = metric_dev
    report["probability_metrics_invariant_across_schemes_max_dev"] = max(invariance_dev.values())
    report["bayes_decision_agreement_min"] = min(decision_agreement.values())
    report["bayes_decision_agreement"] = decision_agreement

    # Laplace Markov chain of Scenario D recounted from raw JSON (Tier B, fold 0)
    vocab = costs["per_tier"]["tier_b"]["scenarioD15"]["vocabulary"]
    fold0 = split["tier_b"]["folds"]["scenarioD15"][0]
    counts = np.zeros((len(vocab), len(vocab)))
    for mid in fold0["train_match_ids"]:
        prev = None
        for e in sorted(json.loads((EVENTS / f"{mid}.json").read_text(encoding="utf-8")), key=lambda e: e["index"]):
            if e["period"] == 5 or e["type"]["name"] not in vocab:
                continue
            cur = (e["period"], vocab.index(e["type"]["name"]))
            if prev is not None and prev[0] == cur[0]:
                counts[prev[1], cur[1]] += 1
            prev = cur
    expected = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + len(vocab))
    g = preds[(preds["tier"] == "tier_b") & (preds["task"] == "scenarioD15") & (preds["fold_id"] == 0)]
    got = g[[f"p__exp1_markov_order1__{c}" for c in vocab]].to_numpy(dtype=float)
    want = expected[g["current_event"].map({c: i for i, c in enumerate(vocab)}).to_numpy()]
    report["laplace_markov_recount_max_abs_dev_float32"] = float(np.abs(got - want).max())

    report["passed"] = bool(report["no_final_test_match_in_predictions"] and report["tiers_disjoint"]
                            and report["metric_recomputation_max_abs_dev"] < 1e-5
                            and report["probability_metrics_invariant_across_schemes_max_dev"] == 0.0
                            and report["bayes_decision_agreement_min"] > 0.9999
                            and report["laplace_markov_recount_max_abs_dev_float32"] < 1e-6)
    (OUT / "independent_verification.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
