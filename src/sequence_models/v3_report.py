"""Experiment 3 report stage: final tables, figures, reproducibility re-runs, runtime, checksums.

Reads the tier outputs of ``scripts/run_context_aware_v3.py``; nothing new is selected here.
Reproducibility: the reference and the combination are re-fitted on Tier B WITHOUT checkpoints
and compared bit-for-bit with the stored probabilities; the reference must also equal experiment
2's stored mc2_zone predictions (float32) on Tier B.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from .experiment import environment_info, sha256_file, write_json
from .figures import CATEGORICAL, GRID, NEUTRAL, SURFACE, TEXT_2, _save, plt
from .v3_experiment import partition_and_folds
from .v3_models import MarkovSpec, nested_markov
from .v3_taxonomy import taxonomy_spec

LOGGER = logging.getLogger(__name__)


def _style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def ci_plot(frame: pd.DataFrame, label: str, title: str, xlabel: str, path: Path) -> None:
    f = frame.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(f) + 1.4))
    for i, r in f.iterrows():
        colour = CATEGORICAL[0] if r["ci_high"] < 0 else (CATEGORICAL[1] if r["ci_low"] > 0 else NEUTRAL)
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(r["difference"], i, "o", color=colour, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(r["ci_high"], i, f"  {r['difference']:+.4f} ({r['percent_change']:+.2f}%)", va="center", fontsize=7,
                color=TEXT_2)
    ax.axvline(0, color=NEUTRAL, linewidth=0.8)
    ax.set_yticks(range(len(f)), f[label], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    _style(ax, "x")
    fig.text(0.01, -0.02, "Paired difference vs reference (mc2_zone), 95% match-bootstrap CI. Blue: better "
             "(CI below 0); orange: worse; grey: CI includes 0.", fontsize=7, color=TEXT_2)
    _save(fig, path)


def shot_pr_figure(curves: pd.DataFrame, ops: pd.DataFrame, thr: pd.DataFrame, models: list[str], prevalence: float,
                   title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, m in enumerate(models):
        c = curves[curves["model"] == m].sort_values("recall")
        colour = CATEGORICAL[i % len(CATEGORICAL)]
        axes[0].plot(c["recall"], c["precision"], color=colour, linewidth=1.8, label=m)
        axes[1].plot(c["fp_per_1000_events"], c["recall"], color=colour, linewidth=1.8, label=m)
        o = ops[ops["model"] == m]
        axes[0].plot(o["shot_recall"], o["shot_precision"], "o", color=colour, markersize=5, markeredgecolor=SURFACE)
        axes[1].plot(o["shot_fp_per_1000_events"], o["shot_recall"], "o", color=colour, markersize=5, markeredgecolor=SURFACE)
        t = thr[thr["model"] == m]
        axes[1].plot(t["fp_per_1000_events"], t["shot_recall"], "s", color=colour, markersize=5, markeredgecolor=SURFACE)
    axes[0].axhline(prevalence, color=NEUTRAL, linestyle=":", linewidth=1)
    axes[0].set_xlabel("Shot recall")
    axes[0].set_ylabel("Shot precision (dotted: prevalence)")
    axes[1].set_xlabel("false Shot predictions per 1,000 events")
    axes[1].set_ylabel("Shot recall")
    axes[1].set_xlim(0, 60)
    for ax in axes:
        _style(ax)
        ax.legend(fontsize=7)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.03, "Curves: one-vs-rest threshold on p(Shot). Circles: cost-sensitive multiclass decisions "
             "(moderate scheme, Shot miss cost x1, 1.25, 1.5, 2). Squares: thresholds fixed on Tier A for 5/10/20 false "
             "positives per 1,000 events.", fontsize=7, color=TEXT_2)
    _save(fig, path)


def grouped_bars(frame: pd.DataFrame, category: str, series: str, value: str, title: str, ylabel: str, path: Path,
                 log: bool = False) -> None:
    cats = list(dict.fromkeys(frame[category]))
    ser = list(dict.fromkeys(frame[series]))
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(cats)), 3.8))
    w = 0.8 / len(ser)
    for i, s in enumerate(ser):
        q = frame[frame[series] == s].set_index(category).reindex(cats)
        ax.bar(np.arange(len(cats)) + (i - (len(ser) - 1) / 2) * w, q[value], w * 0.92, color=CATEGORICAL[i], label=s)
    ax.set_xticks(range(len(cats)), cats, rotation=30, ha="right", fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=min(len(ser), 4))
    _style(ax, "y")
    _save(fig, path)


def reproducibility(cfg: dict, out: Path) -> dict:
    plan = json.loads((out / "frozen_plan.json").read_text())
    _, folds = partition_and_folds(cfg)
    task = cfg["main_taxonomy"]
    table = pd.read_parquet(cfg["_cache"] / f"features_tier_b_{task}.parquet")
    stored = np.load(cfg["_cache"] / f"tier_b_{task}_eval_proba.npz")
    vocab = list(taxonomy_spec(task).vocabulary)
    res = {}
    for name in ("reference", "combination"):
        s = plan["tier_b_markov_models"].get(name) or plan["combination"]
        spec = MarkovSpec(s["name"], tuple(s["features"]), bool(s["reset"]), s["group"])
        p, _, _ = nested_markov(spec, table, folds["tier_b"], len(vocab), cfg["markov"]["concentration_grid"])
        res[f"{name}_refit_max_abs_diff"] = float(np.abs(p - stored[name]).max())
    # reference == experiment 2's mc2_zone (stored as float32)
    t = table[table["eval_eligible"]]
    exp2 = ds.dataset(cfg["_exp2"] / "predictions.parquet").to_table(
        columns=["event_id"] + [f"p__mc2_zone__{c}" for c in vocab],
        filter=(pc.field("tier") == "tier_b") & (pc.field("task") == task)).to_pandas().set_index("event_id")
    e2 = exp2.loc[t["event_id"]].to_numpy(dtype=float)
    res["reference_vs_experiment2_mc2_zone_max_abs_diff_float32"] = float(np.abs(stored["reference"] - e2).max())
    res["passed"] = bool(res["reference_refit_max_abs_diff"] == 0 and res["combination_refit_max_abs_diff"] == 0
                         and res["reference_vs_experiment2_mc2_zone_max_abs_diff_float32"] < 1e-6)
    return res


def stage_report(cfg: dict, out: Path) -> None:
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    a, b = out / "tier_a", out / "tier_b"
    sa, sb = pd.read_csv(a / "summary.csv"), pd.read_csv(b / "summary.csv")
    pa, pb = pd.read_csv(a / "paired.csv"), pd.read_csv(b / "paired.csv")
    summary = pd.concat([sa, sb], ignore_index=True)
    paired = pd.concat([pa, pb], ignore_index=True)
    summary[summary["decision_rule"] == "uniform"].to_csv(out / "model_comparison.csv", index=False)
    summary[summary["decision_rule"].str.startswith("moderate_shot")].to_csv(out / "cost_sensitive_shot_comparison.csv", index=False)
    paired.to_csv(out / "paired_differences.csv", index=False)
    pd.concat([pd.read_csv(a / "per_class.csv"), pd.read_csv(b / "per_class.csv")]).to_csv(out / "per_class_metrics.csv", index=False)
    pd.concat([pd.read_csv(a / "calibration.csv"), pd.read_csv(b / "calibration.csv")]).to_csv(out / "calibration_metrics.csv", index=False)
    pd.concat([pd.read_csv(a / "shot_thresholds.csv"), pd.read_csv(b / "shot_thresholds.csv")]).to_csv(out / "shot_threshold_rules.csv", index=False)
    specs = {n: c for n, c in cfg["candidates"].items()}
    groups = {n: c["group"] for n, c in specs.items()}
    ll = paired[paired["metric"] == "log_loss"].copy()
    ll["group"] = ll["model"].map(groups).fillna(ll["model"].map({"combination": "combination", "mc1": "baseline"}))
    ll.loc[ll["model"].str.startswith("logistic"), "group"] = "logistic_regression"
    ll[ll["group"] == "spatial"].to_csv(out / "spatial_ablation.csv", index=False)
    ll[ll["group"].isin(["possession", "temporal"])].to_csv(out / "possession_temporal_ablation.csv", index=False)
    ch = pd.read_csv(a / "markov_concentration_choices.csv")
    order = ll[(ll["tier"] == "tier_a") & ll["group"].isin(["history", "reference"])].copy()
    sparsity = ch.groupby("model").agg(mean_contexts_most_specific=("n_contexts_most_specific", "mean"),
                                       concentration=("chosen_concentration", "median"))
    order = order.merge(sparsity, left_on="model", right_index=True, how="left")
    order.to_csv(out / "markov_order_comparison.csv", index=False)

    # ------------------------------------------------------------ figures
    task = cfg["main_taxonomy"]
    for tier, frame in (("tier_a", ll[ll["tier"] == "tier_a"]), ("tier_b", ll[ll["tier"] == "tier_b"])):
        f = frame[frame["task"] == task].sort_values("difference")
        ci_plot(f, "model", f"{tier.upper().replace('_', ' ')} ({task}): log-loss change vs mc2_zone",
                "log-loss difference, nats (negative = better)", fig / f"log_loss_vs_reference_{tier}.png")
    curves = pd.read_csv(b / f"shot_pr_curves_{task}.csv")
    ops = sb[(sb["task"] == task) & sb["decision_rule"].str.startswith("moderate_shot")]
    thr = pd.read_csv(b / "shot_thresholds.csv")
    thr = thr[thr["task"] == task]
    models = [m for m in ("reference", "spatial_g12", "combination", "logistic_C0.1", "mc1") if m in set(curves["model"])]
    prev = float(sb[(sb["task"] == task) & (sb["model"] == "reference") & (sb["decision_rule"] == "uniform")]["shot_support"].iloc[0]
                 / sb[(sb["task"] == task) & (sb["model"] == "reference") & (sb["decision_rule"] == "uniform")]["n"].iloc[0])
    shot_pr_figure(curves, ops, thr, models, prev, f"OUT-OF-FOLD (Tier B, {task}) - Shot precision-recall and "
                   "false-positive trade-off", fig / "shot_precision_recall_tier_b.png")
    da = pd.read_csv(out / "diagnostics" / "tier_a" / "rare_event_class_diagnostics.csv")
    da = da[(da["task"] == task) & (da["model"] == "mc2_zone")].copy()
    m = da.melt(id_vars="class", value_vars=["prevalence", "average_precision", "moderate_precision", "moderate_recall"])
    grouped_bars(m, "class", "variable", "value", "DIAGNOSTIC (Tier A, mc2_zone): prevalence vs ranking (AP) vs decisions",
                 "value (log scale)", fig / "rare_event_ranking_vs_decision_tier_a.png", log=True)
    cs = sb[(sb["task"] == task) & sb["decision_rule"].str.startswith("moderate_shot") & sb["model"].isin(models)].copy()
    cs["lambda"] = cs["decision_rule"].str.replace("moderate_shot_x", "")
    for metric, ylabel in (("shot_recall", "Shot recall"), ("shot_fp_per_1000_events", "false Shots per 1,000 events"),
                           ("macro_f1", "macro-F1")):
        grouped_bars(cs, "model", "lambda", metric, f"OUT-OF-FOLD (Tier B) - Shot miss-cost multiplier: {ylabel}",
                     ylabel, fig / f"shot_cost_{metric}_tier_b.png")
    tax = pd.read_csv(out / "taxonomy" / "class_distribution.csv")
    grouped_bars(tax[tax["taxonomy"] != "raw33"], "class", "taxonomy", "share", "DESCRIPTIVE - target class shares by "
                 "taxonomy (334 development matches)", "share of targets (log scale)", fig / "taxonomy_class_shares.png", log=True)
    br = pd.read_csv(out / "diagnostics" / "tier_a" / "shot_breakdowns.csv")
    br = br[(br["task"] == task) & (br["decision_rule"] == "uniform") & (br["by"] == "zone_x_possession_duration")].copy()
    br[["zone", "duration_bin"]] = br["value"].str.split("|", expand=True)
    piv = br.pivot(index="zone", columns="duration_bin", values="prevalence")
    figure, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="Blues", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), ["<5 s", "5-15 s", "15-30 s", ">30 s"][: piv.shape[1]])
    ax.set_yticks(range(piv.shape[0]), ["defensive third", "middle third", "attacking third"][: piv.shape[0]])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("causal possession duration so far")
    ax.set_title("DIAGNOSTIC (Tier A) - P(next event = Shot) by acting-team third x duration", loc="left", fontsize=9,
                 fontweight="bold")
    figure.colorbar(im, ax=ax, fraction=0.04)
    _save(figure, fig / "shot_rate_zone_x_possession_duration_tier_a.png")

    # ------------------------------------------------------------ reproducibility, leakage, runtime, checksums
    write_json(out / "reproducibility_checks.json", reproducibility(cfg, out))
    part, folds = partition_and_folds(cfg)
    feats = {f"{t}_{x}": sorted(set(pd.read_parquet(cfg["_cache"] / f"features_{t}_{x}.parquet", columns=["match_id"])["match_id"]))
             for t in ("tier_a", "tier_b") for x in ("filtered8", "scenarioD15", "actions15")}
    write_json(out / "leakage_checks.json", {
        "final_test_matches_in_any_feature_table": sorted({m for v in feats.values() for m in v} & set(part.final_test)),
        "tier_a_tier_b_overlap": sorted(set(part.tier_a) & set(part.tier_b)),
        "folds_match_grouped": all(not set(f.train_match_ids) & set(f.validation_match_ids) for k in folds for f in folds[k]),
        "plan_frozen_before_tier_b": "frozen_plan.json + sha256 written by 'freeze'; 'confirm' refuses a modified plan",
        "plan_sha256": (out / "frozen_plan.sha256").read_text().strip(),
        "plan_sha256_matches_file": hashlib.sha256((out / "frozen_plan.json").read_bytes()).hexdigest()
        == (out / "frozen_plan.sha256").read_text().strip(),
        "tier_b_already_inspected_in_experiment_2": True,
        "causal_features": "forward scans over earlier records only (tests/test_v3_features_taxonomy.py truncation test)",
        "statsbomb_possession_ids": "not used as features (look-ahead evidence in taxonomy/possession_audit.json)",
        "event_outcomes": "never inputs: only type, location and earlier records; subtype/outcome used only by the "
                          "taxonomy rules for the CURRENT record's own annotation companions",
        "scalers_and_smoothing": "LR standardisation and Markov concentration fitted on training folds only (tests)",
        "shot_thresholds": "fixed on Tier A out-of-fold predictions, applied unchanged to Tier B",
    })
    log = json.loads((out / "runtime_log.json").read_text())
    pd.DataFrame(log).to_csv(out / "runtime_report.csv", index=False)
    write_json(out / "environment.json", environment_info())
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "checksums.json")
    write_json(out / "checksums.json", {str(p.relative_to(out)): sha256_file(p) for p in files})
    LOGGER.info("report written (%d files)", len(files))
