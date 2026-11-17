"""Experiment 2 report stage: final tables, manifests, figures and checksums.

Reads only experiment 2's tier outputs (results/.../tiers/<tier>/) and
experiment 1's split manifest. Nothing is trained here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures_v2
from .backoff_markov import BackoffMarkov
from .config_v2 import ConfigV2
from .experiment import environment_info, sha256_file, write_json
from .experiment_v2 import load_partition
from src.utils.paths import PROCESSED_DATA_DIR

LOGGER = logging.getLogger(__name__)
TIERS = ("tier_a", "tier_b")
DECISION_MODELS = ["frequency_baseline", "mc1", "mc2", "mc1_zone", "mc2_zone", "hmm_event", "hmm_factorized", "hmm_joint"]
SPATIAL_PAIRS = {("mc1_zone", "mc1"), ("mc2_zone", "mc2"), ("hmm_factorized", "hmm_event"), ("hmm_joint", "hmm_event"),
                 ("hmm_joint", "mc1_zone"), ("hmm_factorized", "mc1_zone")}


def _tier_files(out: Path, tier: str, name: str) -> list[Path]:
    """tiers/<tier>/<name> (Tier A) and tiers/<tier>/<task>/<name> (Tier B, one dir per taxonomy)."""
    base = out / "tiers" / tier
    return sorted(p for p in [base / name, *base.glob(f"*/{name}")] if p.exists() and "_hmm_fit_cache" not in p.parts)


def _concat(out: Path, name: str) -> pd.DataFrame:
    parts = [pd.read_csv(p) for t in TIERS for p in _tier_files(out, t, name)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _json(out: Path, tier: str, name: str) -> dict:
    merged: dict = {}
    for p in _tier_files(out, tier, name):
        merged.update(json.loads(p.read_text(encoding="utf-8")))
    return merged


def display_models(summary: pd.DataFrame) -> dict[str, str]:
    """Per taxonomy: the decision model with the lowest TIER A log loss (chosen on the tuning tier)."""
    a = summary[(summary["tier"] == "tier_a") & (summary["cost_scheme"] == "uniform") & summary["model"].isin(DECISION_MODELS)]
    return {task: str(g.loc[g["log_loss"].idxmin(), "model"]) for task, g in a.groupby("task", sort=False)}


def best_hmm(summary: pd.DataFrame) -> dict[str, str]:
    a = summary[(summary["tier"] == "tier_a") & (summary["cost_scheme"] == "uniform")
                & summary["model"].isin(["hmm_event", "hmm_factorized", "hmm_joint"])]
    return {task: str(g.loc[g["log_loss"].idxmin(), "model"]) for task, g in a.groupby("task", sort=False)}


def stage_report(config: ConfigV2) -> None:
    out = config.output_dir
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    part = load_partition(config)

    summary = _concat(out, "model_comparison.csv")
    summary[summary["cost_scheme"] == "uniform"].to_csv(out / "model_comparison.csv", index=False)
    summary[summary["model"].isin(DECISION_MODELS)].to_csv(out / "cost_sensitive_comparison.csv", index=False)
    for name in ("per_class_metrics.csv", "rare_event_analysis.csv", "metrics_by_fold.csv", "calibration_metrics.csv",
                 "reliability_table.csv", "hidden_state_analysis.csv", "confusion_matrices.csv", "cross_cost.csv",
                 "paired_differences.csv", "hmm_training_log.csv", "markov_concentration_choices.csv",
                 "approach_b_check.csv"):
        frame = _concat(out, name)
        if not frame.empty:
            frame.to_csv(out / name, index=False)
    paired = pd.read_csv(out / "paired_differences.csv")
    ablation = paired[[(m, r) in SPATIAL_PAIRS for m, r in zip(paired["model"], paired["reference"])]]
    ablation.to_csv(out / "spatial_ablation.csv", index=False)

    tuning = json.loads((out / "tiers" / "tier_a" / "tuning_decisions.json").read_text(encoding="utf-8"))
    cost_defs = {"cost_family": "C(a,k) = m_k + f_a for a != k, C(k,k) = 0; m_k = min(cap, (pi_max/pi_k)^gamma); "
                                "pi from the fold's TRAINING evaluation rows (Laplace-smoothed); unit = cost of missing "
                                "the most frequent class",
                 "schemes": config.raw["cost"]["schemes"], "tuned": {**config.raw["cost"]["tuned"],
                                                                      "chosen_gamma_on_tier_a": tuning["gamma"],
                                                                      "tier_a_macro_f1_by_gamma": tuning["gamma_scores"]},
                 "tie_rule": "lowest class index (vocabulary order) among actions within the tie tolerance",
                 "per_tier": {t: {task: _json(out, t, f"cost_matrices_{task}.json") for task in config.tasks} for t in TIERS}}
    write_json(out / "cost_matrix_definitions.json", cost_defs)

    folds = {t: _json(out, t, "folds.json") for t in TIERS}
    write_json(out / "split_manifest.json", {
        "source": "results/markov_hmm_pilot/split_manifest.json (experiment 1; read, not modified)",
        "final_test_match_ids": part.final_test, "final_test_status": "UNTOUCHED: not read by experiment 2",
        "development_match_ids": part.development,
        "tier_a": {"role": "exploration and tuning (HMM K, init type, cost exponent, rare-class list)",
                   "match_ids": part.tier_a, "folds": folds["tier_a"]},
        "tier_b": {"role": "held-out development evaluation; tuned choices frozen from Tier A; Markov smoothing by "
                           "nested inner CV", "match_ids": part.tier_b, "folds": folds["tier_b"],
                   "fold_method": "StratifiedGroupKFold(5, shuffle=True, random_state=20261027), groups = match, "
                                  "y = filtered8 target; the same match folds for both taxonomies"},
    })

    # predictions: one wide file per (tier, taxonomy) -> one file with the union of columns
    frames = []
    for t in TIERS:
        for task in config.tasks:
            for path in _tier_files(out, t, f"predictions_{task}.parquet"):
                frames.append(pd.read_parquet(path))
    if frames:
        allp = pd.concat(frames, ignore_index=True)
        for col in allp.columns:
            if col.startswith("decision__"):
                allp[col] = allp[col].astype("string")
        allp.to_parquet(out / "predictions.parquet", compression="zstd", index=False)
        del allp, frames

    # models: copy the per-tier model files
    for t in TIERS:
        for task in config.tasks:
            for src in _tier_files(out, t, f"models_{task}.json"):
                (out / "models" / f"{t}_{task}.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # leakage checks
    integrity = {t: {str(p.parent.name): json.loads(p.read_text()) for p in _tier_files(out, t, "integrity_checks.json")}
                 for t in TIERS}
    audit = json.loads((out / "audit_checks.json").read_text())
    write_json(out / "leakage_checks.json", {
        "final_test_never_read": {"tier_overlap_with_final_test": {t: {k: v["final_test_overlap"] for k, v in integrity[t].items()}
                                                                    for t in TIERS},
                                  "guard": "load_development_events checks every match list against the hold-out and "
                                           "refuses experiment 1's all-match cache (tests/test_v2_pipeline.py)",
                                  "passed": all(not v["final_test_overlap"] for t in TIERS for v in integrity[t].values())},
        "tiers_disjoint": not (set(part.tier_a) & set(part.tier_b)),
        "folds_match_grouped": {t: all(not set(f["train_match_ids"]) & set(f["validation_match_ids"])
                                       for task in folds[t] for f in folds[t][task]) for t in TIERS},
        "same_rows_and_folds_within_each_comparison": "all models of a (tier, taxonomy) are evaluated by one call on "
                                                      "one row set with one fold map (evaluate_task)",
        "filtered8_targets_equal_production": {t: [v.get("filtered8_rows_and_targets_equal_production") for v in integrity[t].values()
                                                   if "filtered8_rows_and_targets_equal_production" in v] for t in TIERS},
        "scenario_d_equals_research_script": audit["scenario_d_implementation_vs_research_script_tier_a"],
        "cost_priors": "from the fold's TRAINING evaluation rows only (fold_priors; test_cost_priors_use_training_folds_only)",
        "cost_tuning": "gamma, HMM K and init type chosen on Tier A only and frozen for Tier B (tuning_decisions.json)",
        "markov_smoothing": "concentration chosen by inner CV over the outer TRAINING folds (test_markov_nested_selection...)",
        "hmm_k_tier_a": "cross-fold rule: K for fold f chosen from the other folds' validation scores",
        "calibration": "diagnostic only; no recalibration fitted, so no calibration parameter can leak",
        "future_information": "causal filtering only; zone of event t+1 marginalised (tests/test_v2_spatial_models.py)",
        "known_incident": "2026-10-28, before experiment 2 code existed: one exploratory check read experiment 1's "
                          "all-match event cache (incl. final-test matches) and filtered it to pilot matches; no "
                          "statistic of a final-test match was computed or viewed; experiment 2 code refuses that cache",
    })

    # ---------------------------------------------------------------- figures
    disp = display_models(summary)
    hbest = best_hmm(summary)
    figures_v2.class_distribution(pd.read_csv(out / "class_distribution.csv"), fig_dir / "class_distribution.png")
    figures_v2.miss_costs(cost_defs["per_tier"]["tier_b"], fig_dir / "cost_matrices.png")
    rare = pd.read_csv(out / "rare_event_analysis.csv")
    figures_v2.rare_recall(rare, disp, "tier_b", fig_dir / "rare_event_recall.png")
    figures_v2.precision_recall_tradeoff(rare, disp, "tier_b", fig_dir / "precision_recall_tradeoff.png")
    cm = pd.read_csv(out / "confusion_matrices.csv")
    for task, model in disp.items():
        vocab = list(config.tasks[task].vocabulary)
        for scheme in ("uniform", "strong", "tuned"):
            q = cm[(cm["tier"] == "tier_b") & (cm["task"] == task) & (cm["model"] == model) & (cm["cost_scheme"] == scheme)]
            if len(q):
                figures_v2.confusion(q, vocab, f"OUT-OF-FOLD EVALUATION (Tier B) - {model}, {task}, {scheme} decisions "
                                     "(row-normalised)", fig_dir / f"confusion_{task}_{model}_{scheme}.png")
    rel = pd.read_csv(out / "reliability_table.csv")
    for task in config.tasks:
        classes = ["Pass", "Carry", "Ball Recovery", "Shot"] + (["Pressure"] if task == "scenarioD15" else [])
        models = list(dict.fromkeys([disp[task], hbest[task], "mc1"]))
        figures_v2.reliability(rel, models, classes, "tier_b", task, fig_dir / f"calibration_{task}.png")
    for task in config.tasks:
        mdl = _json(out, "tier_b", f"models_{task}.json")
        m = BackoffMarkov.from_dict(mdl["markov"]["mc1|fold0"])
        vocab = list(config.tasks[task].vocabulary)
        p = m.predict_proba({"cur": np.arange(len(vocab))})
        figures_v2.heatmap(p, vocab, vocab, f"DESCRIPTIVE - 1st-order Markov transitions, {task} (Tier B fold 0 training)",
                           "next event", "current event", fig_dir / f"markov_transitions_{task}.png")
        for key, params in mdl["hmm_selected"].items():
            if key.split("|")[1] != "fold0":
                continue
            k = len(params["transmat"])
            states = [f"State {i}" for i in range(k)]
            figures_v2.heatmap(np.array(params["transmat"]), states, states,
                               f"DESCRIPTIVE - {key.split('|')[0]} (K={k}), {task}: hidden-state transitions "
                               "(Tier B fold 0 training)", "state at t+1", "state at t",
                               fig_dir / f"hmm_transitions_{task}_{key.split('|')[0]}.png")
    if (out / "hidden_state_analysis.csv").exists():
        figures_v2.state_zones(pd.read_csv(out / "hidden_state_analysis.csv"), fig_dir / "hmm_state_pitch_thirds.png")
    figures_v2.k_sensitivity(pd.read_csv(out / "hmm_training_log.csv").query("tier == 'tier_a'"), fig_dir / "hmm_k_sensitivity.png")
    ab = ablation[ablation["metric"] == "log_loss"].copy()
    ab["panel"] = ab["tier"] + " | " + ab["task"]
    ab["label"] = ab["model"] + " - " + ab["reference"]
    figures_v2.interval_comparison(ab, "label", "difference", "ci_low", "ci_high", "log-loss difference (negative = "
                                   "first model better)", "OUT-OF-FOLD EVALUATION - spatial ablations (paired, same rows)",
                                   fig_dir / "spatial_model_comparison.png", zero_line=True)
    mc = summary[(summary["cost_scheme"] == "uniform") & summary["model"].isin(DECISION_MODELS)].copy()
    mc["panel"] = mc["tier"] + " | " + mc["task"]
    for metric, xlabel in (("log_loss", "log loss, nats (lower better)"), ("brier", "Brier score (lower better)")):
        figures_v2.interval_comparison(mc[mc["model"] != "frequency_baseline"], "model", metric, f"{metric}_ci_low",
                                       f"{metric}_ci_high", xlabel, "OUT-OF-FOLD EVALUATION - model comparison "
                                       "(probabilities; taxonomies are different tasks)", fig_dir / f"model_comparison_{metric}.png")
    cs = summary[summary["model"].isin(DECISION_MODELS) & (summary["tier"] == "tier_b")].copy()
    cs["panel"] = cs["task"]
    cs["label"] = cs["model"] + " | " + cs["cost_scheme"]
    cs = cs[cs["model"].isin([disp["filtered8"], disp["scenarioD15"], "mc1", hbest["filtered8"], hbest["scenarioD15"]])]
    for metric in ("macro_f1", "balanced_accuracy"):
        figures_v2.interval_comparison(cs, "label", metric, f"{metric}_ci_low", f"{metric}_ci_high", metric,
                                       f"OUT-OF-FOLD EVALUATION (Tier B) - {metric} by model and cost scheme",
                                       fig_dir / f"cost_sensitive_{metric}.png")

    production_csv = PROCESSED_DATA_DIR / "statsbomb_360_all_matches_next_events.csv"
    write_json(out / "environment.json", {**environment_info(), "parquet_engine": "pyarrow",
                                          "production_dataset_sha256": sha256_file(production_csv),
                                          "development_event_table_cache_sha256": sha256_file(config.event_table_cache)})
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "checksums.json" and "tiers" not in p.parts)
    write_json(out / "checksums.json", {str(p.relative_to(out)): sha256_file(p) for p in files})
    LOGGER.info("report written (%d files checksummed)", len(files))
