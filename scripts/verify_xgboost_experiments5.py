"""Post-run verification of experiments 5 and 5B from their stored artefacts (trains nothing).

Checks, independently of the experiment code paths where possible:
- result checksums (the original checksums.json of both runs, never rewritten);
- row identity: stored out-of-fold predictions vs. the canonical dataset's eligible rows
  (event ids, order, targets), the outer folds file, uniqueness and completeness;
- probabilities: 15 columns in canonical order, finite, positive, rows sum to 1;
- metrics recomputed with scikit-learn (log loss, accuracy, macro/weighted F1, Shot AP,
  Shot recall) vs. the stored tables, pooled and per fold;
- final-test isolation (83 ids) and the data-access audits (outer validation only predicted);
- raw-JSON re-verification of the stored verification samples (1,100 rows), using
  ``raw_history_check`` (independent of the table builder);
- the arithmetic decomposition XGB-H5 - Markov 2 = (XGB-H5 - XGB-H2) + (XGB-H2 - Markov 2).

Writes, in the 5B result directory, ``post_run_verification.json`` and
``checksums_supplementary.json``. The latter is a separate manifest for files created after the
5B run (the original ``checksums.json`` is left as written by the run).

Usage:
    .venv/bin/python scripts/verify_xgboost_experiments5.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_models.canonical_task import CLASSES  # noqa: E402
from src.sequence_models.raw_events import EVENTS_DIR  # noqa: E402
from src.sequence_models.raw_history_check import load_events, reconstruct  # noqa: E402

EXP5 = PROJECT_ROOT / "results/xgboost_experiment5_actions15_360current_v1_2026-11-20"
EXP5B = PROJECT_ROOT / "results/xgboost_experiment5b_h2_actions15_360current_v1_2026-11-21"
PRED5 = PROJECT_ROOT / "data/interim/xgboost_experiment5/oof_predictions.parquet"
PRED5B = PROJECT_ROOT / "data/interim/xgboost_experiment5b/oof_predictions_h2.parquet"
DATASET = PROJECT_ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
FOLDS = PROJECT_ROOT / "results/markov_hmm_cost_sensitive_v2/tiers/tier_b/scenarioD15/folds.json"
SPLIT = PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json"
POST_RUN_FILES = ["h2_vs_markov2_decomposition.csv", "post_run_verification.json"]
K = 15
SHOT = CLASSES.index("Shot")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks: list[dict] = []

    def check(name, ok, value=None):
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "value": value})

    for d in (EXP5, EXP5B):
        c = json.loads((d / "checksums.json").read_text())
        bad = [k for k, v in c.items() if sha(d / k) != v]
        check(f"{d.name}: all files in checksums.json unchanged", not bad, f"{len(c)} files, mismatches {bad}")

    final_test = set(json.loads(SPLIT.read_text())["final_test_match_ids"])
    t = pd.read_parquet(DATASET)
    tb = t[t["tier"] == "tier_b"].reset_index(drop=True)
    eligible = tb[tb["row_eligible"]]
    o = pd.read_parquet(PRED5)
    h = pd.read_parquet(PRED5B)
    check("dataset contains no final-test match", not set(t["match_id"]) & final_test, f"{len(final_test)} ids")
    check("predictions contain no final-test match", not (set(o["match_id"]) | set(h["match_id"])) & final_test)
    check("690,147 rows in 304 matches", len(eligible) == len(o) == len(h) == 690_147 and o["match_id"].nunique() == 304)
    check("exp5 / 5B predictions: event ids equal the canonical eligible rows in order",
          np.array_equal(o["event_id"].to_numpy(), eligible["event_id"].to_numpy())
          and np.array_equal(h["event_id"].to_numpy(), eligible["event_id"].to_numpy()))
    check("targets equal canonical target_id", np.array_equal(o["y"].to_numpy(), eligible["target_id"].to_numpy())
          and np.array_equal(h["y"].to_numpy(), o["y"].to_numpy()))
    check("no duplicate row ids", bool(o["event_id"].is_unique and h["event_id"].is_unique))
    folds = json.loads(FOLDS.read_text())["scenarioD15"]
    fold_of = np.full(len(o), -1)
    for f in folds:
        if set(f["train_match_ids"]) & set(f["validation_match_ids"]):
            fold_of[:] = -2
        fold_of[np.isin(o["match_id"].to_numpy(), f["validation_match_ids"])] = f["fold_id"]
    check("fold of every row equals the outer folds file (each row in exactly one validation fold)",
          bool((fold_of >= 0).all()) and np.array_equal(fold_of, o["fold_id"].to_numpy())
          and np.array_equal(fold_of, h["fold_id"].to_numpy()))

    y = o["y"].to_numpy()
    proba = {m: o[[f"{m}__p{c}" for c in range(K)]].to_numpy()
             for m in ("frequency", "markov1", "markov2", "logistic_F0", "XGB-F0", "XGB-H5")}
    proba["XGB-H2"] = h[[f"XGB-H2__p{c}" for c in range(K)]].to_numpy()
    for m, p in proba.items():
        check(f"{m}: probabilities finite, positive, 15 columns, row sums within 1e-9 of 1",
              p.shape[1] == K and bool(np.isfinite(p).all() and (p > 0).all()) and float(np.abs(p.sum(1) - 1).max()) < 1e-9,
              float(np.abs(p.sum(1) - 1).max()))

    def metrics(yy, p):
        d = p.argmax(1)
        return {"log_loss": log_loss(yy, p, labels=range(K)), "accuracy": accuracy_score(yy, d),
                "macro_f1": f1_score(yy, d, average="macro", labels=range(K), zero_division=0),
                "weighted_f1": f1_score(yy, d, average="weighted", labels=range(K), zero_division=0),
                "shot_average_precision": average_precision_score(yy == SHOT, p[:, SHOT]),
                "shot_recall": recall_score(yy == SHOT, d == SHOT, zero_division=0)}

    recomputed = {}
    for d, models in ((EXP5, [m for m in proba if m != "XGB-H2"]), (EXP5B, list(proba))):
        mc = pd.read_csv(d / "model_comparison.csv").set_index("model")
        bf = pd.read_csv(d / "metrics_by_fold.csv")
        worst = 0.0
        for m in models:
            r = metrics(y, proba[m])
            recomputed[m] = r
            worst = max(worst, max(abs(r[k] - mc.at[m, k]) for k in r))
            for f in range(5):
                s = fold_of == f
                ref = bf[(bf.model == m) & (bf.fold_id == f)].iloc[0]
                worst = max(worst, abs(log_loss(y[s], proba[m][s], labels=range(K)) - ref["log_loss"]),
                            abs(accuracy_score(y[s], proba[m][s].argmax(1)) - ref["accuracy"]))
        check(f"{d.name}: pooled and per-fold metrics reproduce with scikit-learn (tol 1e-12)", worst < 1e-12, worst)

    for d in (EXP5, EXP5B):
        a = pd.read_csv(d / "data_access_audit.csv")
        ok = bool((a.loc[a.purpose != "outer_validation_predict", "touches_outer_validation"] == False).all())  # noqa: E712
        check(f"{d.name}: outer validation used only for the final predictions", ok, f"{len(a)} data sets")

    n_ok = n = 0
    for d, n_prev, cols in ((EXP5, 4, None), (EXP5B, 1, ("E_t-1_raw", "E_t_raw"))):
        rv = pd.read_csv(d / "raw_feature_verification.csv")
        if set(rv["match_id"]) & final_test:
            raise RuntimeError("verification sample contains a final-test match")
        ctx = dict(zip(eligible["event_id"], range(len(eligible))))
        hist_all = _history(tb, n_prev)
        cache: dict = {}
        for r in rv.itertuples():
            if r.match_id not in cache:
                cache = {r.match_id: load_events(EVENTS_DIR, int(r.match_id))}
            hist, target, _ = reconstruct(cache[r.match_id], r.event_id, n_prev)
            i = ctx[r.event_id]
            n += 1
            n_ok += int(hist == hist_all[i].tolist() and target == int(y[i]))
    check("raw-JSON re-verification of the stored samples (exp5: 500 H5 rows; 5B: 600 H2 rows)", n_ok == n, f"{n_ok}/{n}")

    ll = {m: recomputed[m]["log_loss"] for m in recomputed}
    total = ll["XGB-H5"] - ll["markov2"]
    hist_part, est_part = ll["XGB-H5"] - ll["XGB-H2"], ll["XGB-H2"] - ll["markov2"]
    check("arithmetic decomposition: (H5 - H2) + (H2 - M2) = H5 - M2", abs(hist_part + est_part - total) < 1e-15,
          {"XGB-H5_minus_markov2": total, "XGB-H5_minus_XGB-H2": hist_part, "XGB-H2_minus_markov2": est_part,
           "share_XGB-H5_minus_XGB-H2": hist_part / total, "share_XGB-H2_minus_markov2": est_part / total})

    n_fail = sum(c["result"] == "FAIL" for c in checks)
    (EXP5B / "post_run_verification.json").write_text(json.dumps(
        {"description": "post-run verification of experiments 5 and 5B (scripts/verify_xgboost_experiments5.py)",
         "n_checks": len(checks), "n_fail": n_fail, "checks": checks,
         "recomputed_metrics_sklearn": recomputed}, indent=1, default=float), encoding="utf-8")
    (EXP5B / "checksums_supplementary.json").write_text(json.dumps(
        {"description": "sha256 of files created AFTER the 5B run; checksums.json is the run's original manifest "
                        "and is not rewritten",
         "files": {f: sha(EXP5B / f) for f in POST_RUN_FILES},
         "generated_by": {"h2_vs_markov2_decomposition.csv": "scripts/analyze_xgboost_experiment5b.py",
                          "post_run_verification.json": "scripts/verify_xgboost_experiments5.py"}}, indent=1), encoding="utf-8")
    for c in checks:
        print(f"{c['result']:4s} {c['check']}")
    print(f"{len(checks)} checks, {n_fail} FAIL")
    if n_fail:
        raise SystemExit(1)


def _history(tb: pd.DataFrame, n_prev: int) -> np.ndarray:
    """Pipeline features of the eligible rows (event_history, as used by the experiments)."""
    from src.sequence_models.event_history import history_ids

    return history_ids(tb, n_prev)[tb["row_eligible"].to_numpy()]


if __name__ == "__main__":
    main()
