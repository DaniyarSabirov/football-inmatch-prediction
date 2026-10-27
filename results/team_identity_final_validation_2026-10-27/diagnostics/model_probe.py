"""Final-validation variant of results/team_identity_correction_2026-10-26/diagnostics/
model_sensitivity_probe.py (same models, folds and metrics) on ml/rows.csv.gz:
  V0 original (HEAD builder)   V1o offline-cleaned counts (uses <= 2 s after t)
  V1 causal counts (default)   V3 causal + mask uncertain   V4 no team counts
  V2 causal, uncertain rows excluded

Controlled sensitivity probe: does the team-identity handling change what a
simple next-event model learns? NOT a thesis baseline -- the project has no
predictive baseline yet (src/models/ is empty). Two standard models with
fixed settings, one match-level split shared by every variant, so the only
thing that differs between variants is the identity handling.

Task: next_event_type (8 classes, MEANINGFUL_EVENT_TYPES) from the current
row's features: current_event_type (one-hot), period, minute, actor_x,
actor_y (+ missing indicator), visible_players, visible_teammates,
visible_opponents. Rows: ml_rows.csv.gz (all 417 matches, 10 s horizon).

Variants (same folds, same test rows unless stated):
  V0 original    original pipeline features (HEAD builder: legacy counts AND
                 legacy actor coordinates), all rows
  V1 corrected   corrected counts (unresolved players counted on neither
                 side) and synchronized orientation, all rows
  V2 exclude     V1 features; training AND test rows with >= 1 unresolved
                 player removed (evaluated on the clean test rows; also
                 scored on all test rows for comparability)
  V3 mask        V1 features; for rows with >= 1 unresolved player the two
                 count features are set to missing + an uncertainty indicator
  V4 no counts   V1 without visible_teammates/visible_opponents at all (an
                 upper bound on how much these features matter)

5-fold GroupKFold by match (deterministic). Metrics: macro-F1, balanced
accuracy, MCC, per-class F1, pooled over the 5 test folds, plus per-fold
paired differences against V1.

Output: ../model_probe/metrics.csv, per_class_f1.csv, fold_differences.csv, probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "ml" / "model_probe"
SEED = 20261026


def design(ml: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "V0":
        teammates, opponents = ml.legacy_visible_teammates, ml.legacy_visible_opponents
        actor_x, actor_y = ml.legacy_actor_x, ml.legacy_actor_y
    elif variant == "V1o":
        teammates, opponents = ml.visible_teammates_offline.astype(float), ml.visible_opponents_offline.astype(float)
        actor_x, actor_y = ml.actor_x, ml.actor_y
    else:
        teammates, opponents = ml.visible_teammates.astype(float), ml.visible_opponents.astype(float)
        actor_x, actor_y = ml.actor_x, ml.actor_y
    X = pd.get_dummies(ml.current_event_type, prefix="type", dtype=float)
    X["period"] = ml.period
    X["minute"] = ml.minute
    X["actor_missing"] = actor_x.isna().astype(float)
    X["actor_x"] = actor_x.fillna(60.0)
    X["actor_y"] = actor_y.fillna(40.0)
    X["visible_players"] = ml.visible_players
    if variant != "V4":
        X["visible_teammates"] = teammates
        X["visible_opponents"] = opponents
    if variant == "V3":
        uncertain = ml.visible_ambiguous > 0
        X.loc[uncertain, ["visible_teammates", "visible_opponents"]] = np.nan
        X["identity_uncertain"] = uncertain.astype(float)
    return X


def models(name: str):
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))


def scores(y, p, labels) -> dict:
    per_class = f1_score(y, p, labels=labels, average=None, zero_division=0)
    return {"macro_f1": f1_score(y, p, labels=labels, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, p), "mcc": matthews_corrcoef(y, p),
            **{f"f1_{c}": v for c, v in zip(labels, per_class)}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["hgb"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    ml = pd.read_csv(HERE.parent / "ml" / "rows.csv.gz", low_memory=False)
    y = ml.next_event_type.to_numpy()
    labels = sorted(ml.next_event_type.unique())
    uncertain = (ml.visible_ambiguous > 0).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(ml, y, groups=ml.match_id))
    rows, fold_rows = [], []
    for model_name in args.models:
        for variant in ("V0", "V1o", "V1", "V2", "V3", "V4"):
            X = design(ml, variant)
            if model_name == "logreg":
                X = X.fillna(X.mean()) if variant == "V3" else X
            pred_all = np.empty(len(ml), dtype=object)
            for k, (train, test) in enumerate(folds):
                fit_rows = train[~uncertain[train]] if variant == "V2" else train
                model = models(model_name).fit(X.iloc[fit_rows], y[fit_rows])
                pred_all[test] = model.predict(X.iloc[test])
                evaluated = test[~uncertain[test]] if variant == "V2" else test
                fold_rows.append({"model": model_name, "variant": variant, "fold": k,
                                  **scores(y[evaluated], pred_all[evaluated], labels),
                                  "clean_test_rows_macro_f1": f1_score(y[test[~uncertain[test]]], pred_all[test[~uncertain[test]]],
                                                                       labels=labels, average="macro", zero_division=0)})
                print(model_name, variant, k, round(fold_rows[-1]["macro_f1"], 4), flush=True)
            mask = ~uncertain if variant == "V2" else np.ones(len(ml), bool)
            rows.append({"model": model_name, "variant": variant, "evaluated_rows": int(mask.sum()),
                         **scores(y[mask], pred_all[mask], labels),
                         "scored_on_all_rows_macro_f1": f1_score(y, pred_all, labels=labels, average="macro", zero_division=0),
                         "clean_rows_only_macro_f1": f1_score(y[~uncertain], pred_all[~uncertain], labels=labels, average="macro", zero_division=0),
                         "uncertain_rows_only_macro_f1": f1_score(y[uncertain], pred_all[uncertain], labels=labels, average="macro", zero_division=0)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(OUT / "fold_metrics.csv", index=False)
    diffs = []
    for model_name in args.models:
        base = folds_df[(folds_df.model == model_name) & (folds_df.variant == "V1")].set_index("fold")
        for variant in ("V0", "V1o", "V2", "V3", "V4"):
            other = folds_df[(folds_df.model == model_name) & (folds_df.variant == variant)].set_index("fold")
            for metric in ("macro_f1", "balanced_accuracy", "mcc", "clean_test_rows_macro_f1"):
                d = other[metric] - base[metric]
                diffs.append({"model": model_name, "variant_minus_V1": variant, "metric": metric,
                              "mean_difference": d.mean(), "sd_difference": d.std(ddof=1),
                              "min": d.min(), "max": d.max(),
                              "fold_sd_of_V1": base[metric].std(ddof=1)})
    pd.DataFrame(diffs).to_csv(OUT / "fold_differences.csv", index=False)
    (OUT / "probe.json").write_text(json.dumps({
        "rows": len(ml), "uncertain_rows": int(uncertain.sum()), "labels": labels, "seed": SEED,
        "folds": "GroupKFold(5) by match_id", "features": list(design(ml, "V3").columns),
    }, indent=1), encoding="utf-8")
    print(metrics.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
