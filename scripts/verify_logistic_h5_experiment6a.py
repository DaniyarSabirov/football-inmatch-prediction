"""Independent post-run verification of experiment 6A (MLR-H5) from its stored artefacts.

Trains nothing and does not trust the runner's summary. Independently of the experiment code
paths where possible:

- result checksums (``checksums.json`` as written by the run, never rewritten);
- row identity of the stored out-of-fold predictions against the canonical dataset (eligible
  rows, order, targets), the outer folds file and the stored predictions of experiments 5/5B;
- history semantics: the 5-position history is rebuilt here with pandas ``groupby().shift()``
  (not ``event_history``), and probabilities are recomputed from the stored fold coefficients
  with a NumPy softmax on a one-hot matrix built here; they must equal the stored predictions;
- probabilities: 15 columns in canonical order, finite, positive, rows sum to 1;
- metrics recomputed with scikit-learn (log loss, Brier, accuracy, macro/weighted F1, Shot AP,
  Shot precision/recall) vs. the stored tables, pooled and per fold;
- the primary paired difference (MLR-H5 - XGB-H5 log loss) and an independent match-level
  bootstrap CI with its own seed (agreement with the stored CI is reported, not required to
  be exact);
- final-test isolation (83 ids), data-access audit, historical result directories unchanged;
- raw-JSON re-verification of the stored verification sample (``raw_history_check``).

Writes ``post_run_verification.json`` and ``checksums_supplementary.json`` in the result
directory (the run's ``checksums.json`` is left untouched).

Usage:
    .venv/bin/python scripts/verify_logistic_h5_experiment6a.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_models.canonical_task import CLASSES  # noqa: E402
from src.sequence_models.raw_events import EVENTS_DIR  # noqa: E402
from src.sequence_models.raw_history_check import load_events, reconstruct  # noqa: E402

CONFIG = PROJECT_ROOT / "configs/logistic_h5_experiment6a_v1.yaml"
DATASET = PROJECT_ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
FOLDS = PROJECT_ROOT / "results/markov_hmm_cost_sensitive_v2/tiers/tier_b/scenarioD15/folds.json"
SPLIT = PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json"
POST_RUN_FILES = ["post_run_verification.json"]
K = 15
SHOT = CLASSES.index("Shot")
NAME = "MLR-H5"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    out = PROJECT_ROOT / cfg["output_dir"]
    art = PROJECT_ROOT / cfg["artefact_dir"]
    checks: list[dict] = []

    def check(name, ok, value=None):
        checks.append({"check": name, "result": "PASS" if ok else "FAIL",
                       "value": value if not isinstance(value, (np.floating, np.integer)) else value.item()})

    # ------------------------------------------------------------------ checksums
    cs = json.loads((out / "checksums.json").read_text())
    bad = [k for k, v in cs.items() if sha(out / k) != v]
    check(f"result checksums ({len(cs)} files)", not bad, bad or len(cs))
    env = json.loads((out / "environment.json").read_text())
    for rel, h in env["artefacts_sha256"].items():
        check(f"artefact hash {rel}", sha(PROJECT_ROOT / rel) == h)
    audit = pd.read_csv(out / "integrity_audit.csv")
    check("run's own integrity audit has no FAIL", not (audit["result"] == "FAIL").any(), int((audit["result"] == "PASS").sum()))

    # ------------------------------------------------------------------ row identity
    split = json.loads(SPLIT.read_text())
    final = set(int(m) for m in split["final_test_match_ids"])
    table = pd.read_parquet(DATASET)
    check("canonical dataset contains no final-test match", not (set(table["match_id"]) & final) and len(final) == 83)
    tb = table[table["tier"] == "tier_b"].reset_index(drop=True)
    elig = tb[tb["row_eligible"]].reset_index(drop=True)
    pred = pd.read_parquet(art / "oof_predictions_mlr_h5.parquet")
    pcols = [f"{NAME}__p{c}" for c in range(K)]
    p = pred[pcols].to_numpy()
    y = pred["y"].to_numpy()
    check("690,147 rows, 304 matches", len(pred) == 690_147 and pred["match_id"].nunique() == 304, len(pred))
    check("unique row ids; no duplicated predictions", bool(pred["event_id"].is_unique))
    check("row ids, order, matches and targets equal the canonical eligible rows",
          np.array_equal(pred["event_id"].to_numpy(), elig["event_id"].to_numpy())
          and np.array_equal(pred["match_id"].to_numpy(), elig["match_id"].to_numpy())
          and np.array_equal(y, elig["target_id"].to_numpy()))
    folds = json.loads(FOLDS.read_text())["scenarioD15"]
    fold_of_match = {int(m): int(f["fold_id"]) for f in folds for m in f["validation_match_ids"]}
    check("fold of every row = outer folds file; folds partition the 304 matches",
          np.array_equal(pred["fold_id"].to_numpy(), pred["match_id"].map(fold_of_match).to_numpy())
          and len(fold_of_match) == 304 and all(not (set(f["train_match_ids"]) & set(f["validation_match_ids"])) for f in folds))
    check("no final-test match in any fold", not any((set(f["train_match_ids"]) | set(f["validation_match_ids"])) & final for f in folds))
    for label, path in (("experiment 5", cfg["experiment5_predictions"]), ("experiment 5B", cfg["experiment5b_predictions"])):
        o = pd.read_parquet(PROJECT_ROOT / path, columns=["event_id", "match_id", "fold_id", "y"])
        check(f"paired with {label}: identical event_id / match / fold / target row by row",
              all(np.array_equal(o[c].to_numpy(), pred[c].to_numpy()) for c in ("event_id", "match_id", "fold_id", "y")))

    # ------------------------------------------------------------------ probabilities
    check("15 probability columns in canonical order", list(pred.columns[4:]) == pcols)
    check("probabilities finite and strictly positive", bool(np.isfinite(p).all() and (p > 0).all()))
    check("rows sum to 1 (|dev| <= 1e-9)", float(np.abs(p.sum(1) - 1).max()) <= 1e-9, float(np.abs(p.sum(1) - 1).max()))

    # ------------------------------------------------------------------ independent history + softmax from coefficients
    hist = pd.DataFrame({f"lag{lag}": tb.groupby("sequence_id")["symbol_id"].shift(lag) for lag in range(1, 5)})
    hist = hist.fillna(15).astype(int)[tb["row_eligible"].to_numpy()].reset_index(drop=True)
    cur = elig["symbol_id"].to_numpy()
    names = [f"E_{{t-{lag}}}={c}" for lag in (4, 3, 2, 1) for c in list(CLASSES) + ["<START>"]] + [f"E_t={c}" for c in CLASSES]
    check("feature names in the stored coefficient table follow the independent column layout",
          pd.read_csv(out / "coefficient_summary.csv").query("class_id == 0")["feature"].tolist() == names)
    col = np.column_stack([16 * j + hist[f"lag{lag}"].to_numpy() for j, lag in enumerate((4, 3, 2, 1))] + [64 + cur])
    models = np.load(art / "fold_models_mlr_h5.npz")
    recomputed = np.empty_like(p)
    for f in range(5):
        sel = pred["fold_id"].to_numpy() == f
        w, b = models[f"coef_fold{f}"], models[f"intercept_fold{f}"]
        z = w.T[col[sel]].sum(axis=1) + b          # sum of the 5 active columns' weights = one-hot product
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        recomputed[sel] = e / e.sum(axis=1, keepdims=True)
    dmax = float(np.abs(recomputed - p).max())
    check("probabilities recomputed from stored W, b with an independently built history equal the stored ones (1e-10)",
          dmax < 1e-10, dmax)
    for lag in range(1, 5):
        pos = elig["seq_pos"].to_numpy()
        if not np.array_equal(hist[f"lag{lag}"].to_numpy() == 15, pos < lag):
            check(f"<START> at lag {lag} exactly when seq_pos < {lag}", False)
            break
    else:
        check("<START> at lag l exactly when seq_pos < l (no cross-period history)", True)

    # ------------------------------------------------------------------ metrics (scikit-learn)
    mc = pd.read_csv(out / "model_comparison.csv").set_index("model")
    d = p.argmax(1)
    onehot = np.eye(K)[y]
    mine = {"log_loss": log_loss(y, p, labels=list(range(K))), "brier": float(((p - onehot) ** 2).sum(1).mean()),
            "accuracy": accuracy_score(y, d),
            "macro_f1": f1_score(y, d, labels=list(range(K)), average="macro", zero_division=0),
            "weighted_f1": f1_score(y, d, labels=list(range(K)), average="weighted", zero_division=0),
            "shot_average_precision": average_precision_score(y == SHOT, p[:, SHOT]),
            "shot_recall": recall_score(y == SHOT, d == SHOT, zero_division=0),
            "shot_precision": precision_score(y == SHOT, d == SHOT, zero_division=0)}
    diffs = {k: abs(v - mc.at[NAME, k]) for k, v in mine.items()}
    check("pooled MLR-H5 metrics recomputed with scikit-learn equal the stored table (1e-9)",
          max(diffs.values()) < 1e-9, {k: round(float(v), 15) for k, v in diffs.items()})
    bf = pd.read_csv(out / "metrics_by_fold.csv").query("model == @NAME").set_index("fold_id")
    worst = 0.0
    for f in range(5):
        sel = pred["fold_id"].to_numpy() == f
        worst = max(worst, abs(log_loss(y[sel], p[sel], labels=list(range(K))) - bf.at[f, "log_loss"]),
                    abs(f1_score(y[sel], d[sel], labels=list(range(K)), average="macro", zero_division=0) - bf.at[f, "macro_f1"]))
    check("per-fold log loss and macro-F1 recomputed equal the stored table (1e-9)", worst < 1e-9, worst)
    pc = pd.read_csv(out / "per_class_metrics.csv").query("model == @NAME").set_index("class_id")
    check("per-class support equals the target counts", np.array_equal(pc["support"].to_numpy(), np.bincount(y, minlength=K)))

    # ------------------------------------------------------------------ primary paired comparison
    o5 = pd.read_parquet(PROJECT_ROOT / cfg["experiment5_predictions"], columns=[f"XGB-H5__p{c}" for c in range(K)])
    q = o5.to_numpy()
    di = -np.log(p[np.arange(len(y)), y]) + np.log(q[np.arange(len(y)), y])
    pdiff = pd.read_csv(out / "paired_differences.csv").query("model == @NAME and reference == 'XGB-H5' and metric == 'log_loss'").iloc[0]
    check("mean d_i (MLR-H5 - XGB-H5 NLL) equals the stored pooled difference (1e-12)",
          abs(di.mean() - pdiff["difference"]) < 1e-12, float(di.mean()))
    mids, inv = np.unique(pred["match_id"].to_numpy(), return_inverse=True)
    s_d, s_n = np.bincount(inv, weights=di), np.bincount(inv).astype(float)
    rng = np.random.default_rng(20261123)                 # independent seed (the run used 7)
    reps = []
    for _ in range(2000):
        m = np.bincount(rng.integers(0, len(mids), len(mids)), minlength=len(mids))
        reps.append((m @ s_d) / (m @ s_n))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    check("independent match bootstrap (2,000 reps, seed 20261123) gives the same CI sign as the stored CI",
          bool((lo > 0) == (pdiff["ci_low"] > 0) and (hi < 0) == (pdiff["ci_high"] < 0)),
          {"independent_ci": [float(lo), float(hi)], "stored_ci": [float(pdiff["ci_low"]), float(pdiff["ci_high"])]})
    folds_sign = [float(di[pred["fold_id"].to_numpy() == f].mean()) for f in range(5)]
    check("per-fold sign of the MLR-H5 - XGB-H5 difference equals the stored counts",
          sum(v < 0 for v in folds_sign) == int(pdiff["folds_negative"]) and sum(v > 0 for v in folds_sign) == int(pdiff["folds_positive"]),
          folds_sign)

    # ------------------------------------------------------------------ data access / history / historical results
    da = pd.read_csv(out / "data_access_audit.csv")
    check("data-access audit: outer validation only predicted; final test never touched",
          not da.query("purpose != 'outer_validation_predict'")["touches_outer_validation"].any()
          and not da["touches_final_test"].any(), len(da))
    fl = pd.read_csv(out / "fit_log.csv")
    check("every fit converged, no warnings, gradient check passed", bool(fl["converged"].all() and fl["gradient_ok"].all()
                                                                        and (fl["warnings"] == "[]").all()), len(fl))
    for key in ("baseline_results_dir", "experiment5_results_dir", "experiment5b_results_dir"):
        dd = PROJECT_ROOT / cfg[key]
        c = json.loads((dd / "checksums.json").read_text())
        bad = [k for k, v in c.items() if sha(dd / k) != v]
        check(f"historical results unchanged: {cfg[key]}", not bad, bad or len(c))
    before = env["historical_verification"]["before"]["files"]
    check("stored predictions of experiments 5/5B and the canonical dataset unchanged since the run",
          all(sha(PROJECT_ROOT / rel) == h for rel, h in before.items()))

    # ------------------------------------------------------------------ raw-JSON re-verification
    rv = pd.read_csv(out / "raw_feature_verification.csv")
    ok, cache = 0, {}
    for r in rv.itertuples():
        if r.match_id not in cache:
            cache = {r.match_id: load_events(EVENTS_DIR, int(r.match_id))}
        h, t, _ = reconstruct(cache[r.match_id], r.event_id, 4)
        i = pred.index[pred["event_id"] == r.event_id][0]
        ok += int(h == [*hist.iloc[i][[f"lag{lag}" for lag in (4, 3, 2, 1)]].tolist(), int(cur[i])] and t == int(y[i]))
    check(f"raw-JSON rebuild of the {len(rv)} stored verification rows equals the independently built history and target",
          ok == len(rv) and len(rv) >= 500, ok)

    n_fail = sum(c["result"] == "FAIL" for c in checks)
    report = {"verdict": "PASS" if n_fail == 0 else "FAIL", "n_checks": len(checks), "n_fail": n_fail, "checks": checks}
    (out / "post_run_verification.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "checksums_supplementary.json").write_text(json.dumps({
        "description": "sha256 of files created AFTER the 6A run; checksums.json is the run's original manifest and is not rewritten",
        "files": {f: sha(out / f) for f in POST_RUN_FILES},
        "generated_by": {f: "scripts/verify_logistic_h5_experiment6a.py" for f in POST_RUN_FILES}}, indent=1))
    for c in checks:
        print(f"{c['result']:4s}  {c['check']}  {c['value'] if c['value'] is not None else ''}")
    print(f"\nverdict: {report['verdict']} ({len(checks)} checks, {n_fail} FAIL)")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
