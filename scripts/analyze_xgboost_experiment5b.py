"""Post-run analysis of experiment 5B: where does XGB-H2 differ from Markov 2?

Reads only stored artefacts (the canonical dataset, experiment 5's and 5B's out-of-fold
predictions, the fold file); trains nothing. It splits the paired log-loss difference
XGB-H2 - Markov 2 into rows where the canonical Markov 2 backs off to Markov 1
(t = 0, or a context with < 20 training rows) and rows with a full (E_{t-1}, E_t) context.
It also gives a match-level percentile bootstrap CI for each subset (1,000 replicates,
seed 7, as in the main evaluation).

Writes results/xgboost_experiment5b_h2_actions15_360current_v1_2026-11-21/h2_vs_markov2_decomposition.csv

Usage:
    .venv/bin/python scripts/analyze_xgboost_experiment5b.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_models.evaluation_v2 import bootstrap_multiplicities  # noqa: E402

OUT = PROJECT_ROOT / "results/xgboost_experiment5b_h2_actions15_360current_v1_2026-11-21"
K = 15
MIN_CONTEXT = 20       # configs/canonical_baselines_v1.yaml: markov2.min_context_count


def main() -> None:
    h = pd.read_parquet(PROJECT_ROOT / "data/interim/xgboost_experiment5b/oof_predictions_h2.parquet")
    o = pd.read_parquet(PROJECT_ROOT / "data/interim/xgboost_experiment5/oof_predictions.parquet")
    t = pd.read_parquet(PROJECT_ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet")
    tb = t[t["tier"] == "tier_b"].reset_index(drop=True)
    prev = tb.groupby("sequence_id", sort=False)["symbol_id"].shift(1).fillna(-1).astype(int).to_numpy()
    e = tb["row_eligible"].to_numpy()
    cur, pv = tb["symbol_id"].to_numpy()[e], prev[e]
    if not (np.array_equal(tb["event_id"].to_numpy()[e], h["event_id"].to_numpy())
            and np.array_equal(h["event_id"].to_numpy(), o["event_id"].to_numpy())):
        raise RuntimeError("row order differs between the dataset and the stored predictions")
    y, mid, fold = h["y"].to_numpy(), h["match_id"].to_numpy(), h["fold_id"].to_numpy()
    p2 = o[[f"markov2__p{c}" for c in range(K)]].to_numpy()
    ph = h[[f"XGB-H2__p{c}" for c in range(K)]].to_numpy()
    folds = json.loads((PROJECT_ROOT / "results/markov_hmm_cost_sensitive_v2/tiers/tier_b/scenarioD15/folds.json")
                       .read_text())["scenarioD15"]
    backoff = np.zeros(len(y), bool)
    for f in folds:
        tr = np.isin(mid, f["train_match_ids"])
        va = fold == f["fold_id"]
        ctx = np.zeros((K, K))
        ok = tr & (pv >= 0)
        np.add.at(ctx, (pv[ok], cur[ok]), 1)
        backoff[va] = (pv[va] < 0) | (ctx[np.maximum(pv[va], 0), cur[va]] < MIN_CONTEXT)
    d = -np.log(ph[np.arange(len(y)), y]) + np.log(p2[np.arange(len(y)), y])
    matches, idx = np.unique(mid, return_inverse=True)
    mult = bootstrap_multiplicities(len(matches), 1000, 7)
    total = d.sum()
    rows = []
    for name, m in [("all rows", np.ones(len(y), bool)), ("Markov-2 backoff rows (t = 0 or context < 20)", backoff),
                    ("  of which t = 0 (period start)", pv < 0), ("  of which rare context", backoff & (pv >= 0)),
                    ("full-context rows", ~backoff)]:
        s = np.bincount(idx, weights=np.where(m, d, 0.0), minlength=len(matches))
        n = np.bincount(idx, weights=m.astype(float), minlength=len(matches))
        rep = (mult @ s) / np.maximum(mult @ n, 1)
        rows.append({"subset": name, "n_rows": int(m.sum()), "mean_dLL_h2_minus_markov2": float(d[m].mean()),
                     "ci_low": float(np.percentile(rep, 2.5)), "ci_high": float(np.percentile(rep, 97.5)),
                     "contribution_to_pooled_dLL": float(d[m].sum() / len(y)), "share_of_total_difference": float(d[m].sum() / total),
                     "max_abs_probability_difference": float(np.abs(ph - p2)[m].max())})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "h2_vs_markov2_decomposition.csv", index=False)
    print(frame.to_string())


if __name__ == "__main__":
    main()
