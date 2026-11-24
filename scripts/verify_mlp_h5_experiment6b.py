"""Independent post-run verification of experiment 6B (MLP-H5) from its stored artefacts.

Trains nothing, imports neither PyTorch nor the experiment module, and does not trust the
runner's summary:

- result checksums (``checksums.json`` as written by the run) and artefact hashes;
- the result directory is complete (not a ``.partial`` staging directory) and the run's own
  integrity audit has no FAIL;
- row identity of the stored predictions against the canonical dataset, the folds file and the
  stored predictions of experiments 5, 5B and 6A (paired row by row);
- history semantics: the 5-position history is rebuilt with pandas ``groupby().shift()``
  (not ``event_history``); probabilities are recomputed by a float64 NumPy forward pass of the
  stored network weights on a one-hot matrix built here, for the primary and every robustness
  replicate, and must match the stored ones;
- checkpoint provenance: the stored weights' configuration equals the selected configuration
  in ``hyperparameter_choices.csv``, and the checkpoint identifiers equal the run's;
- probabilities valid; metrics recomputed with scikit-learn (pooled and per fold);
- the primary paired differences (vs XGB-H5 and MLR-H5) as means of d_i, with an independent
  match bootstrap (own seed);
- data access (outer validation only predicted, final test never touched), historical results
  and inputs unchanged, raw-JSON re-verification of the stored sample.

Writes ``post_run_verification.json`` and ``checksums_supplementary.json``.

Usage:
    .venv/bin/python scripts/verify_mlp_h5_experiment6b.py
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

CONFIG = PROJECT_ROOT / "configs/mlp_h5_experiment6b_v1.yaml"
DATASET = PROJECT_ROOT / "data/interim/canonical_task/actions15_360current_v1/observations.parquet"
FOLDS = PROJECT_ROOT / "results/markov_hmm_cost_sensitive_v2/tiers/tier_b/scenarioD15/folds.json"
SPLIT = PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json"
POST_RUN_FILES = ["post_run_verification.json"]
K, SHOT, NAME = 15, CLASSES.index("Shot"), "MLP-H5"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def forward(ws, bs, x):
    h = x
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w.T + b
        if i < len(ws) - 1:
            h = np.maximum(h, 0.0)
    h = h - h.max(axis=1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(axis=1, keepdims=True)


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    out, art = PROJECT_ROOT / cfg["output_dir"], PROJECT_ROOT / cfg["artefact_dir"]
    checks: list[dict] = []

    def check(name, ok, value=None):
        checks.append({"check": name, "result": "PASS" if ok else "FAIL",
                       "value": value.item() if isinstance(value, (np.floating, np.integer)) else value})

    check("result directory complete (renamed from staging; no .partial left)",
          out.is_dir() and not out.with_name(out.name + ".partial").exists() and not (out / ".staging_experiment6b").exists())
    cs = json.loads((out / "checksums.json").read_text())
    bad = [k for k, v in cs.items() if sha(out / k) != v]
    check(f"result checksums ({len(cs)} files)", not bad, bad or len(cs))
    env = json.loads((out / "environment.json").read_text())
    for rel, h in env["artefacts_sha256"].items():
        check(f"artefact hash {rel}", sha(PROJECT_ROOT / rel) == h)
    audit = pd.read_csv(out / "integrity_audit.csv")
    check("run's own integrity audit has no FAIL", not (audit["result"] == "FAIL").any(), int((audit["result"] == "PASS").sum()))

    # ------------------------------------------------------------------ rows
    final = set(int(m) for m in json.loads(SPLIT.read_text())["final_test_match_ids"])
    table = pd.read_parquet(DATASET)
    check("canonical dataset contains no final-test match; 83 final-test ids", not (set(table["match_id"]) & final) and len(final) == 83)
    tb = table[table["tier"] == "tier_b"].reset_index(drop=True)
    elig = tb[tb["row_eligible"]].reset_index(drop=True)
    pred = pd.read_parquet(art / "oof_predictions_mlp_h5.parquet")
    y = pred["y"].to_numpy()
    fold = pred["fold_id"].to_numpy()
    check("690,147 rows, 304 matches, unique row ids", len(pred) == 690_147 and pred["match_id"].nunique() == 304
          and bool(pred["event_id"].is_unique), len(pred))
    check("row ids, order, matches and targets equal the canonical eligible rows",
          np.array_equal(pred["event_id"].to_numpy(), elig["event_id"].to_numpy())
          and np.array_equal(pred["match_id"].to_numpy(), elig["match_id"].to_numpy()) and np.array_equal(y, elig["target_id"].to_numpy()))
    folds = json.loads(FOLDS.read_text())["scenarioD15"]
    fom = {int(m): int(f["fold_id"]) for f in folds for m in f["validation_match_ids"]}
    check("fold of every row = outer folds file; no train/validation match overlap; no final-test match in folds",
          np.array_equal(fold, pred["match_id"].map(fom).to_numpy()) and len(fom) == 304
          and all(not (set(f["train_match_ids"]) & set(f["validation_match_ids"])) for f in folds)
          and not any((set(f["train_match_ids"]) | set(f["validation_match_ids"])) & final for f in folds))
    prior = {}
    for label, key, models in (("experiment 5", "experiment5_predictions", ["XGB-H5", "markov2", "logistic_F0"]),
                               ("experiment 5B", "experiment5b_predictions", ["XGB-H2"]),
                               ("experiment 6A", "experiment6a_predictions", ["MLR-H5"])):
        o = pd.read_parquet(PROJECT_ROOT / cfg[key])
        check(f"paired with {label}: identical event_id / match / fold / target row by row",
              all(np.array_equal(o[c].to_numpy(), pred[c].to_numpy()) for c in ("event_id", "match_id", "fold_id", "y")))
        for m in models:
            prior[m] = o[[f"{m}__p{c}" for c in range(K)]].to_numpy()

    # ------------------------------------------------------------------ independent history + NumPy forward pass
    hist = pd.DataFrame({lag: tb.groupby("sequence_id")["symbol_id"].shift(lag) for lag in (4, 3, 2, 1)}).fillna(15).astype(int)
    hist = hist[tb["row_eligible"].to_numpy()].reset_index(drop=True)
    cur = elig["symbol_id"].to_numpy()
    cols = np.column_stack([16 * j + hist[lag].to_numpy() for j, lag in enumerate((4, 3, 2, 1))] + [64 + cur])
    pos = elig["seq_pos"].to_numpy()
    check("<START> at lag l exactly when seq_pos < l (no cross-period history)",
          all(np.array_equal(hist[lag].to_numpy() == 15, pos < lag) for lag in (1, 2, 3, 4)))
    npz = np.load(art / "fold_models_mlp_h5.npz")
    selected = {int(k): v for k, v in json.loads(str(npz["selected"])).items()}
    hc = pd.read_csv(out / "hyperparameter_choices.csv").set_index("fold_id")
    check("checkpoint provenance: stored configuration per fold = selected configuration",
          all(selected[f]["name"] == hc.at[f, "candidate"] and json.dumps(selected[f]["hidden"]) == hc.at[f, "hidden"] for f in range(5)))
    reps = sorted({int(k.split("_rep")[1].split("_")[0]) for k in npz.files if k.startswith("fold0_rep")})
    pcols = {r: [f"{NAME}__p{c}" if r == 0 else f"{NAME}_seed{r}__p{c}" for c in range(K)] for r in reps}
    worst, shapes_ok = 0.0, True
    for f in range(5):
        sel = fold == f
        uniq, inv = np.unique(cols[sel], axis=0, return_inverse=True)
        x = np.zeros((len(uniq), 79))
        x[np.repeat(np.arange(len(uniq)), 5), uniq.ravel()] = 1.0
        for r in reps:
            n_layers = sum(1 for k in npz.files if k.startswith(f"fold{f}_rep{r}_W"))
            ws = [npz[f"fold{f}_rep{r}_W{i}"] for i in range(1, n_layers + 1)]
            bs = [npz[f"fold{f}_rep{r}_b{i}"] for i in range(1, n_layers + 1)]
            shapes_ok &= ws[0].shape[1] == 79 and ws[-1].shape[0] == 15 and [w.shape[0] for w in ws[:-1]] == selected[f]["hidden"]
            p = forward(ws, bs, x)[inv.ravel()]
            worst = max(worst, float(np.abs(p - pred.loc[sel, pcols[r]].to_numpy()).max()))
    check("stored weights have the selected architecture (79 inputs, hidden sizes, 15 logits)", bool(shapes_ok))
    check(f"probabilities recomputed by a NumPy forward pass of the stored weights on an independently built history "
          f"equal the stored ones (|dp| < 1e-5; {len(reps)} replicates)", worst < 1e-5, worst)
    p = pred[pcols[0]].to_numpy()
    check("15 probability columns in canonical order; finite, positive; rows sum to 1 (1e-9)",
          bool(np.isfinite(p).all() and (p > 0).all()) and float(np.abs(p.sum(1) - 1).max()) <= 1e-9, float(np.abs(p.sum(1) - 1).max()))

    # ------------------------------------------------------------------ metrics (scikit-learn)
    mc = pd.read_csv(out / "model_comparison.csv").set_index("model")
    d = p.argmax(1)
    mine = {"log_loss": log_loss(y, p, labels=list(range(K))), "brier": float(((p - np.eye(K)[y]) ** 2).sum(1).mean()),
            "accuracy": accuracy_score(y, d), "macro_f1": f1_score(y, d, labels=list(range(K)), average="macro", zero_division=0),
            "weighted_f1": f1_score(y, d, labels=list(range(K)), average="weighted", zero_division=0),
            "shot_average_precision": average_precision_score(y == SHOT, p[:, SHOT]),
            "shot_recall": recall_score(y == SHOT, d == SHOT, zero_division=0),
            "shot_precision": precision_score(y == SHOT, d == SHOT, zero_division=0)}
    diffs = {k: abs(v - mc.at[NAME, k]) for k, v in mine.items()}
    check("pooled MLP-H5 metrics recomputed with scikit-learn equal the stored table (1e-9)", max(diffs.values()) < 1e-9,
          {k: float(v) for k, v in diffs.items()})
    bf = pd.read_csv(out / "metrics_by_fold.csv").query("model == @NAME").set_index("fold_id")
    w = max(max(abs(log_loss(y[fold == f], p[fold == f], labels=list(range(K))) - bf.at[f, "log_loss"]),
                abs(f1_score(y[fold == f], d[fold == f], labels=list(range(K)), average="macro", zero_division=0) - bf.at[f, "macro_f1"]))
            for f in range(5))
    check("per-fold log loss and macro-F1 recomputed equal the stored table (1e-9)", w < 1e-9, w)
    pc = pd.read_csv(out / "per_class_metrics.csv").query("model == @NAME").set_index("class_id")
    check("per-class support equals the target counts", np.array_equal(pc["support"].to_numpy(), np.bincount(y, minlength=K)))
    rob = pd.read_csv(out / "seed_robustness.csv").set_index("replicate")
    check("robustness-seed log losses recompute from stored predictions (1e-12)",
          all(abs(log_loss(y, pred[pcols[r]].to_numpy(), labels=list(range(K))) - rob.at[r, "pooled_log_loss"]) < 1e-12 for r in reps))

    # ------------------------------------------------------------------ paired comparisons
    pdf = pd.read_csv(out / "paired_differences.csv")
    mids, inv = np.unique(pred["match_id"].to_numpy(), return_inverse=True)
    s_n = np.bincount(inv).astype(float)
    rng = np.random.default_rng(20261124)
    mult = np.stack([np.bincount(rng.integers(0, len(mids), len(mids)), minlength=len(mids)) for _ in range(2000)])
    for refm in ("XGB-H5", "MLR-H5", "markov2"):
        di = -np.log(p[np.arange(len(y)), y]) + np.log(prior[refm][np.arange(len(y)), y])
        row = pdf.query("model == @NAME and reference == @refm and metric == 'log_loss'").iloc[0]
        check(f"mean d_i (MLP-H5 - {refm} NLL) equals the stored pooled difference (1e-12)",
              abs(di.mean() - row["difference"]) < 1e-12, float(di.mean()))
        boot = (mult @ np.bincount(inv, weights=di)) / (mult @ s_n)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        check(f"independent match bootstrap vs {refm} (2,000 reps, seed 20261124): CI excludes zero iff the stored CI does",
              bool((lo > 0 or hi < 0) == bool(row["ci_excludes_zero"])),
              {"independent_ci": [float(lo), float(hi)], "stored_ci": [float(row["ci_low"]), float(row["ci_high"])]})
        fs = [float(di[fold == f].mean()) for f in range(5)]
        check(f"per-fold signs vs {refm} equal the stored counts",
              sum(v < 0 for v in fs) == int(row["folds_negative"]) and sum(v > 0 for v in fs) == int(row["folds_positive"]), fs)

    # ------------------------------------------------------------------ access, history, inputs
    da = pd.read_csv(out / "data_access_audit.csv")
    check("data-access audit: outer validation only predicted; stopping/training sets never touch it; final test never touched",
          not da.query("purpose != 'outer_validation_predict'")["touches_outer_validation"].any() and not da["touches_final_test"].any()
          and {"inner_train", "inner_stop", "inner_validation", "outer_train", "outer_stop"} <= set(da["purpose"]), len(da))
    for key in ("baseline_results_dir", "experiment5_results_dir", "experiment5b_results_dir", "experiment6a_results_dir"):
        dd = PROJECT_ROOT / cfg[key]
        c = json.loads((dd / "checksums.json").read_text())
        b = [k for k, v in c.items() if sha(dd / k) != v]
        check(f"historical results unchanged: {cfg[key]}", not b, b or len(c))
    before = env["historical_verification"]["before"]["files"]
    check("stored predictions of experiments 5/5B/6A and the canonical dataset unchanged since the run",
          all(sha(PROJECT_ROOT / rel) == h for rel, h in before.items()), len(before))
    rv = pd.read_csv(out / "raw_feature_verification.csv")
    ok, cache = 0, {}
    index_of = {e: i for i, e in enumerate(pred["event_id"].to_numpy())}
    for r in rv.itertuples():
        if r.match_id not in cache:
            cache = {r.match_id: load_events(EVENTS_DIR, int(r.match_id))}
        h, t, _ = reconstruct(cache[r.match_id], r.event_id, 4)
        i = index_of[r.event_id]
        ok += int(h == [*(int(hist.at[i, lag]) for lag in (4, 3, 2, 1)), int(cur[i])] and t == int(y[i]))
    check(f"raw-JSON rebuild of the {len(rv)} stored verification rows equals the independently built history and target",
          ok == len(rv) and len(rv) >= 500, ok)

    n_fail = sum(c["result"] == "FAIL" for c in checks)
    report = {"verdict": "PASS" if n_fail == 0 else "FAIL", "n_checks": len(checks), "n_fail": n_fail, "checks": checks}
    (out / "post_run_verification.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "checksums_supplementary.json").write_text(json.dumps({
        "description": "sha256 of files created AFTER the 6B run; checksums.json is the run's original manifest and is not rewritten",
        "files": {f: sha(out / f) for f in POST_RUN_FILES},
        "generated_by": {f: "scripts/verify_mlp_h5_experiment6b.py" for f in POST_RUN_FILES}}, indent=1))
    for c in checks:
        print(f"{c['result']:4s}  {c['check']}  {c['value'] if c['value'] is not None else ''}")
    print(f"\nverdict: {report['verdict']} ({len(checks)} checks, {n_fail} FAIL)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
