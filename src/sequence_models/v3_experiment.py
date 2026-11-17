"""Experiment 3: feature tables, folds, cost layer and evaluation (see methodology.md).

Data: development matches only (experiment 2's development event cache + raw details). Tier A =
experiment 1's 30 pilot matches and folds; Tier B = the other 304 development matches with
experiment 2's Tier B folds (tiers/tier_b/filtered8/folds.json). Final-test matches are refused.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.utils.paths import INTERIM_DATA_DIR, PROJECT_ROOT

from .config_v2 import load_config_v2
from .cost_sensitive import CostScheme, bayes_decisions, miss_costs, training_priors
from .evaluation_v2 import (
    SUMMARY_KEYS,
    bootstrap_multiplicities,
    bootstrap_summary,
    calibration,
    confusion_counts,
    interval,
    metrics_from_confusion,
    per_class_from_confusion,
    per_match_stats,
    probability_metrics,
)
from .experiment_v2 import load_development_events, load_partition
from .splits import FoldAssignment, assert_no_final_test
from .v3_events import build_event_details
from .v3_features import causal_raw_features, history_tokens, possession_bins, spatial_features
from .v3_taxonomy import taxonomy_spec, taxonomy_table

LOGGER = logging.getLogger(__name__)
CONFIG_PATH = PROJECT_ROOT / "configs" / "context_aware_event_prediction_v3.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_path"] = str(path)
    cfg["_output"] = PROJECT_ROOT / cfg["output_dir"]
    cfg["_cache"] = PROJECT_ROOT / cfg["feature_cache_dir"]
    cfg["_exp2"] = PROJECT_ROOT / cfg["experiment2_dir"]
    return cfg


def partition_and_folds(cfg: dict):
    part = load_partition(load_config_v2())
    folds_b = json.loads((cfg["_exp2"] / "tiers" / "tier_b" / "filtered8" / "folds.json").read_text())["filtered8"]
    tier_b = [FoldAssignment(f["fold_id"], tuple(f["train_match_ids"]), tuple(f["validation_match_ids"])) for f in folds_b]
    return part, {"tier_a": part.tier_a_folds, "tier_b": tier_b}


def development_raw_table(cfg: dict, part) -> pd.DataFrame:
    """Raw events + details for all development matches (never a final-test match)."""
    config2 = load_config_v2()
    ev = load_development_events(config2, part, part.development)
    det = build_event_details(part.development, part.final_test, cfg["_cache"] / "event_details_development.pkl")
    t = ev.merge(det.drop(columns="match_id"), on="event_id", how="left", validate="one_to_one")
    assert_no_final_test(t["match_id"].unique().tolist(), part.final_test)
    return t


def build_feature_table(raw: pd.DataFrame, taxonomy: str) -> pd.DataFrame:
    """Taxonomy task table + causal spatial, possession, temporal and history features."""
    tab = taxonomy_table(raw, taxonomy)
    causal = causal_raw_features(raw)
    extra = raw[["event_id", "location_x", "location_y", "minute", "team_id", "subtype", "outcome"]]
    tab = tab.merge(extra, on="event_id", how="left", validate="one_to_one")
    tab = tab.merge(causal, on="event_id", how="left", validate="one_to_one")
    tab = tab.sort_values(["match_id", "period", "seq_pos"], kind="stable").reset_index(drop=True)
    sp = spatial_features(tab["location_x"].to_numpy(dtype=float), tab["location_y"].to_numpy(dtype=float))
    for k, v in sp.items():
        tab[k] = v
    for k, v in possession_bins(tab).items():
        tab[k] = v
    # possession frame: the location seen from the team causally in possession (defenders' records mirrored)
    own = (tab["team_id"] == tab["causal_possession_team"]).to_numpy()
    has_poss = tab["causal_possession_team"].notna().to_numpy()
    xp = np.where(own, tab["location_x"], 120.0 - tab["location_x"]).astype(float)
    yp = np.where(own, tab["location_y"], 80.0 - tab["location_y"]).astype(float)
    xp[~has_poss], yp[~has_poss] = np.nan, np.nan
    sp_p = spatial_features(xp, yp)
    for k in ("distance_to_goal", "shooting_angle", "danger_area", "d7", "z3d"):
        tab[f"pf_{k}"] = sp_p[k]
    tab["pf_danger_area"] = tab["pf_danger_area"].astype(float)
    v = len(taxonomy_spec(taxonomy).vocabulary)
    seq = tab["sequence_id"].to_numpy()
    poss = tab["causal_possession_id"].to_numpy()
    sym = tab["symbol_id"].to_numpy()
    tab["cur"] = sym
    for reset, suffix in ((False, ""), (True, "_reset")):
        for k, val in history_tokens(sym, seq, poss, v, depth=3, reset_at_possession=reset).items():
            tab[f"{k}{suffix}"] = val
    tab["log_possession_duration"] = np.log1p(tab["possession_duration_s"])
    tab["log_possession_events"] = np.log1p(tab["possession_events_so_far"])
    tab["log_time_since_previous"] = np.log1p(tab["time_since_previous_s"])
    tab["minute_scaled"] = tab["minute"] / 90.0
    for c in ("danger_area", "penalty_area", "same_team_as_previous", "recent_possession_change"):
        tab[c] = tab[c].astype(float)
    return tab


def feature_table(cfg: dict, raw: pd.DataFrame, tier: str, taxonomy: str, match_ids: list[int]) -> pd.DataFrame:
    path = cfg["_cache"] / f"features_{tier}_{taxonomy}.parquet"
    if path.exists():
        tab = pd.read_parquet(path)
        if sorted(tab["match_id"].unique().tolist()) == sorted(match_ids):
            return tab
    tab = build_feature_table(raw[raw["match_id"].isin(match_ids)].reset_index(drop=True), taxonomy)
    path.parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(path, index=False)
    return tab


# --------------------------------------------------------------------------- cost layer
def fold_priors(table: pd.DataFrame, folds, n: int) -> dict[int, np.ndarray]:
    elig = table["eval_eligible"].to_numpy()
    out = {}
    for f in folds:
        tr = table["match_id"].isin(f.train_match_ids).to_numpy() & elig
        out[f.fold_id] = training_priors(table["target_id"].to_numpy()[tr], n)
    return out


def shot_cost_matrix(priors: np.ndarray, shot_index: int, gamma: float, cap: float, multiplier: float) -> np.ndarray:
    """Moderate scheme with the cost of MISSING a Shot multiplied by ``multiplier`` (C[a, k] = m_k, a != k)."""
    m = miss_costs(priors, CostScheme("shot", gamma, cap)).copy()
    m[shot_index] *= multiplier
    c = np.tile(m, (len(m), 1))
    np.fill_diagonal(c, 0.0)
    return c


def decisions_for(p: np.ndarray, fold_of_eval: np.ndarray, priors: dict[int, np.ndarray], make_matrix) -> np.ndarray:
    d = np.empty(len(p), dtype=np.int64)
    for f, pri in priors.items():
        rows = fold_of_eval == f
        d[rows] = bayes_decisions(p[rows], make_matrix(pri))
    return d


# --------------------------------------------------------------------------- evaluation
def shot_metrics(y: np.ndarray, p: np.ndarray, d: np.ndarray, shot: int) -> dict[str, float]:
    is_shot = y == shot
    pred = d == shot
    tp = int((is_shot & pred).sum())
    fp = int((~is_shot & pred).sum())
    rec = tp / max(is_shot.sum(), 1)
    prec = tp / max(pred.sum(), 1) if pred.any() else 0.0
    return {"shot_recall": rec, "shot_precision": prec,
            "shot_f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
            "shot_fp_per_1000_events": 1000 * fp / len(y), "shot_predicted": int(pred.sum()), "shot_support": int(is_shot.sum())}


def ranking_metrics(y: np.ndarray, p: np.ndarray, k: int) -> dict[str, float]:
    t = (y == k).astype(int)
    return {"average_precision": float(average_precision_score(t, p[:, k])) if t.any() else np.nan,
            "roc_auc": float(roc_auc_score(t, p[:, k])) if 0 < t.sum() < len(t) else np.nan,
            "prevalence": float(t.mean())}


def threshold_for_budget(y: np.ndarray, score: np.ndarray, positive: int, fp_per_1000: float) -> float:
    """Smallest threshold whose false positives per 1,000 rows stay within the budget."""
    neg = np.sort(score[y != positive])[::-1]
    allowed = int(np.floor(fp_per_1000 * len(y) / 1000))
    if allowed >= len(neg):
        return 0.0
    return float(np.nextafter(neg[allowed], np.inf)) if allowed >= 0 else float("inf")


def evaluate(tier: str, task: str, table: pd.DataFrame, proba: dict[str, np.ndarray], folds, vocab: list[str],
             cfg: dict, reference: str, extra_pairs: list[tuple[str, str]] = (), shot_lambdas: dict | None = None,
             thresholds: dict | None = None) -> dict[str, pd.DataFrame]:
    """All metrics for models evaluated on the same rows and folds of one (tier, taxonomy)."""
    t = table[table["eval_eligible"]].reset_index(drop=True)
    y = t["target_id"].to_numpy()
    groups = t["match_id"].to_numpy()
    fold_of = {m: f.fold_id for f in folds for m in f.validation_match_ids}
    fold_of_eval = t["match_id"].map(fold_of).to_numpy()
    n = len(vocab)
    shot = vocab.index("Shot")
    priors = fold_priors(table, folds, n)
    e = cfg["evaluation"]
    mult = bootstrap_multiplicities(len(np.unique(groups)), int(e["bootstrap_replicates"]), int(e["bootstrap_seed"]))
    base = cfg["shot_cost"]["base"]
    summary, per_class, shot_rows, cal_rows, stats = [], [], [], [], {}
    for name, p in proba.items():
        if np.isnan(p).any() or np.abs(p.sum(axis=1) - 1).max() > 1e-9 or (p <= 0).any():
            raise ValueError(f"{name}: invalid probabilities")
        pm = probability_metrics(y, p)
        cal, _ = calibration(y, p, vocab)
        cal_rows.append({"tier": tier, "task": task, "model": name, **pm,
                         **{k: v for k, v in cal.items() if not k.startswith(("citl_", "ece_"))}})
        rules = {"uniform": decisions_for(p, fold_of_eval, priors, lambda pri: 1 - np.eye(n))}
        for lam in cfg["shot_cost"]["multipliers"]:
            rules[f"moderate_shot_x{lam}"] = decisions_for(
                p, fold_of_eval, priors, lambda pri, lam=lam: shot_cost_matrix(pri, shot, base["gamma"], base["cap"], lam))
        stats[name] = per_match_stats(y, p, rules, groups, n)
        rank = ranking_metrics(y, p, shot)
        for rule, d in rules.items():
            cm = confusion_counts(y, d, n)
            dm = metrics_from_confusion(cm)
            reps = bootstrap_summary(stats[name], rule, mult)
            lo, hi = interval(reps)
            row = {"tier": tier, "task": task, "model": name, "decision_rule": rule, "n": len(y), **dm, **pm,
                   **shot_metrics(y, p, d, shot), "shot_average_precision": rank["average_precision"],
                   "shot_roc_auc": rank["roc_auc"], "n_classes_predicted": int((cm.sum(axis=0) > 0).sum())}
            for i, k in enumerate(SUMMARY_KEYS):
                row[f"{k}_ci_low"], row[f"{k}_ci_high"] = float(lo[i]), float(hi[i])
            summary.append(row)
            pc = per_class_from_confusion(cm, vocab)
            pc.insert(0, "decision_rule", rule)
            pc.insert(0, "model", name)
            pc.insert(0, "task", task)
            pc.insert(0, "tier", tier)
            pc["average_precision"] = [ranking_metrics(y, p, k)["average_precision"] for k in range(n)]
            per_class.append(pc)
        if thresholds and name in thresholds:
            for budget, tau in thresholds[name].items():
                d = (p[:, shot] >= tau)
                tp, fp = int((d & (y == shot)).sum()), int((d & (y != shot)).sum())
                shot_rows.append({"tier": tier, "task": task, "model": name, "fp_budget_per_1000": float(budget),
                                  "threshold_from_tier_a": tau, "shot_recall": tp / max((y == shot).sum(), 1),
                                  "shot_precision": tp / max(d.sum(), 1), "fp_per_1000_events": 1000 * fp / len(y),
                                  "predicted_shots": int(d.sum())})
    # paired comparisons (same rows, same resampled matches), uniform decisions
    pairs = [(m, reference) for m in proba if m != reference] + list(extra_pairs)
    paired = []
    ref_rows = {r["model"]: r for r in summary if r["decision_rule"] == "uniform"}
    for model, ref in pairs:
        if model not in stats or ref not in stats:
            continue
        diff = bootstrap_summary(stats[model], "uniform", mult) - bootstrap_summary(stats[ref], "uniform", mult)
        lo, hi = interval(diff)
        for i, k in enumerate(SUMMARY_KEYS):
            a, b = ref_rows[model][k], ref_rows[ref][k]
            paired.append({"tier": tier, "task": task, "model": model, "reference": ref, "metric": k, "model_value": a,
                           "reference_value": b, "difference": a - b, "percent_change": 100 * (a - b) / b if b else np.nan,
                           "ci_low": float(lo[i]), "ci_high": float(hi[i]), "ci_excludes_zero": bool(lo[i] > 0 or hi[i] < 0)})
    return {"summary": pd.DataFrame(summary), "per_class": pd.concat(per_class, ignore_index=True),
            "paired": pd.DataFrame(paired), "calibration": pd.DataFrame(cal_rows), "shot_thresholds": pd.DataFrame(shot_rows)}


def pr_curve_points(y: np.ndarray, p: np.ndarray, k: int, max_points: int = 200) -> pd.DataFrame:
    prec, rec, thr = precision_recall_curve((y == k).astype(int), p[:, k])
    idx = np.unique(np.linspace(0, len(thr) - 1, min(max_points, len(thr))).astype(int))
    fp_rate = []
    for i in idx:
        d = p[:, k] >= thr[i]
        fp_rate.append(1000 * int((d & (y != k)).sum()) / len(y))
    return pd.DataFrame({"threshold": thr[idx], "precision": prec[idx], "recall": rec[idx], "fp_per_1000_events": fp_rate})
