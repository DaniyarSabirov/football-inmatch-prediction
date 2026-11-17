"""Experiment 2: cost-sensitive decisions, Scenario D, pitch thirds, categorical spatial HMM.

Stages (in order; outputs in results/markov_hmm_cost_sensitive_v2/):

  audit   taxonomy comparison (8 vs 15 types), Scenario D implementation check
          against the research script, zone-source validation, feature
          assessment; development matches only
  tier_a  Tier A (experiment 1's 30 pilot matches, its 5 folds): all models,
          HMM K grid; writes the TUNING decisions (K per emission, cost
          exponent per model, rare classes)
  tier_b  Tier B (other 304 development matches, 5 new folds): every tuned
          choice frozen from Tier A
  repro   repeat selected experiments (Tier B Markov models, Tier B HMM refits)
  report  final tables, figures, manifests, checksums

The final test set is never read; experiment 1's outputs are only read.

Usage:
    .venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py audit
    .venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py tier_a
    .venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py tier_b
    .venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py repro
    .venv/bin/python scripts/run_markov_hmm_cost_sensitive_v2.py report
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_models.categorical_hmm import CategoricalHMM  # noqa: E402
from src.sequence_models.config_v2 import load_config_v2  # noqa: E402
from src.sequence_models.cost_sensitive import cost_matrix, miss_costs  # noqa: E402
from src.sequence_models.experiment import environment_info, sha256_file, write_json  # noqa: E402
from src.sequence_models.experiment_v2 import (  # noqa: E402
    Evaluated,
    approach_b_check,
    assemble_hmm,
    cross_fold_k,
    eligible_targets,
    evaluate_task,
    fold_priors,
    fold_seq_ids,
    hidden_state_analysis,
    load_development_events,
    load_partition,
    overall_k,
    winning_init,
    prepare_task,
    run_hmm,
    run_markov,
    tune_gamma,
)
from src.sequence_models.spatial import load_production_rows  # noqa: E402
from src.sequence_models.splits import assert_no_final_test, grouped_folds  # noqa: E402
from src.sequence_models.taxonomy_audit import (  # noqa: E402
    feature_assessment,
    research_script_check,
    taxonomy_comparison,
    zone_source_validation,
)

LOGGER = logging.getLogger("markov_hmm_cost_sensitive_v2")
DECISION_MODELS = ["frequency_baseline", "mc1", "mc2", "mc1_zone", "mc2_zone", "hmm_event", "hmm_factorized", "hmm_joint"]
PAIRS = [("mc1", "frequency_baseline"), ("mc2", "mc1"), ("mc1_zone", "mc1"), ("mc2_zone", "mc2"), ("mc2_zone", "mc1"),
         ("hmm_event", "mc1"), ("hmm_factorized", "hmm_event"), ("hmm_joint", "hmm_event"), ("hmm_joint", "mc1_zone"),
         ("hmm_factorized", "mc1_zone"), ("mc1", "exp1_markov_order1"), ("mc2", "exp1_markov_order2")]


def stage_audit(config, out: Path) -> None:
    part = load_partition(config)
    events = load_development_events(config, part, part.development)
    comparison, dist = taxonomy_comparison(events, config.tasks)
    comparison.to_csv(out / "taxonomy_comparison.csv", index=False)
    dist.to_csv(out / "class_distribution.csv", index=False)
    check = research_script_check(part.tier_a, events[events["match_id"].isin(part.tier_a)], config.tasks)
    zones = zone_source_validation(part.development, events)
    from src.sequence_models.experiment_v2 import prepare_task as prep
    fa = feature_assessment(prep(events[events["match_id"].isin(part.tier_a)], config.tasks["scenarioD15"]).table)
    fa.to_csv(out / "feature_assessment.csv", index=False)
    write_json(out / "audit_checks.json", {"scope": f"{len(part.development)} development matches (no final-test match read)",
                                           "scenario_d_implementation_vs_research_script_tier_a": check,
                                           "zone_source_validation_development_production_rows": zones})
    LOGGER.info("audit written")


def prediction_sink(tdir: Path):
    """Write each (tier, taxonomy) wide prediction frame to its own Parquet file."""
    def sink(frame: pd.DataFrame) -> None:
        frame["n_skipped_raw_events"] = frame["n_skipped_raw_events"].astype("Int64")
        frame.to_parquet(tdir / f"predictions_{frame['task'].iat[0]}.parquet", compression="zstd", index=False)
    return sink


def stage_tier(tier: str, config, out: Path, overwrite: bool, task_filter: str | None = None) -> None:
    """Tier A: both taxonomies, outputs in tiers/tier_a/. Tier B: ONE taxonomy per call
    (--task), outputs in tiers/tier_b/<task>/, so a finished taxonomy is never lost. HMM
    fits are cached per fit in tiers/<tier>/_hmm_fit_cache/ so an interrupted run resumes."""
    started = time.time()
    if tier == "tier_b" and task_filter is None:
        raise SystemExit("tier_b runs one taxonomy per invocation: pass --task filtered8 or --task scenarioD15")
    task_names = list(config.tasks) if task_filter is None else [task_filter]
    tdir = out / "tiers" / tier if tier == "tier_a" else out / "tiers" / tier / task_filter
    cache_root = out / "tiers" / tier / "_hmm_fit_cache"
    if (tdir / "model_comparison.csv").exists() and not overwrite:
        raise SystemExit(f"{tier} results exist; pass --overwrite to replace them deliberately")
    tdir.mkdir(parents=True, exist_ok=True)
    part = load_partition(config)
    hcfg = config.raw["hmm"]
    tuning_path = out / "tiers" / "tier_a" / "tuning_decisions.json"
    if tier == "tier_a":
        matches, folds, workers = part.tier_a, part.tier_a_folds, hcfg["workers_tier_a"]
        tuning = {"rare_classes": {}, "n_states": {}, "gamma": {}, "gamma_scores": {}, "init_self_transition": {}}
    else:
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        matches, workers = part.tier_b, hcfg["workers_tier_b"]
    assert_no_final_test(matches, part.final_test)
    events = load_development_events(config, part, matches)
    if tier == "tier_b":  # the same match folds for both taxonomies: always built from the filtered8 rows
        f8 = prepare_task(events, config.tasks["filtered8"])
        folds = grouped_folds(f8.table.loc[f8.eval_rows, ["match_id", "target_id"]],
                              config.raw["tiers"]["tier_b_n_folds"], config.raw["tiers"]["tier_b_fold_seed"])
        del f8
    sink = prediction_sink(tdir)
    ev = Evaluated()
    fit_logs, beta_choices, approach_b, states, checks, folds_out = [], [], [], [], {}, {}
    for task_name in task_names:
        task = config.tasks[task_name]
        data = prepare_task(events, task)
        if task_name == "filtered8":  # independent target check against the production dataset
            prod = load_production_rows(matches)
            prod = prod[prod["period"] != 5]
            ours = data.table.loc[data.eval_rows]
            merged = ours.merge(prod[["current_event_id", "next_event_type"]], left_on="event_id",
                                right_on="current_event_id", how="outer", indicator=True)
            checks["filtered8_rows_and_targets_equal_production"] = bool(
                (merged["_merge"] == "both").all() and (merged["target"] == merged["next_event_type"]).all())
        if tier == "tier_a":
            share = pd.Series(data.table.loc[data.eval_rows, "target"]).value_counts(normalize=True)
            tuning["rare_classes"][task_name] = [c for c in task.vocabulary if share.get(c, 0) < config.raw["rare_class_threshold"]]
        rare = tuning["rare_classes"][task_name]
        LOGGER.info("%s %s: %d evaluation rows, %d sequences", tier, task_name, len(data.eval_rows), len(data.seq_ids))

        proba, betas, mk_fitted = run_markov(data, folds, config)
        for b in betas:
            b.update({"tier": tier, "task": task_name})
        beta_choices += betas
        if tier == "tier_a":
            grid = {e: hcfg["tier_a_n_states"] for e in hcfg["emissions"]}
        else:
            grid = {e: [tuning["n_states"][task_name][e]] for e in hcfg["emissions"]}
        inits = None if tier == "tier_a" else {e: [w] for e, w in tuning["init_self_transition"][task_name].items()}
        log, best_proba, best_params = run_hmm(data, folds, config, grid, workers, part.final_test, inits, cache_root)
        log.insert(0, "tier", tier)
        fit_logs.append(log)
        if tier == "tier_a":
            chosen = cross_fold_k(log)
            tuning["n_states"][task_name] = overall_k(log)
            tuning["init_self_transition"][task_name] = winning_init(log)
            for e in hcfg["emissions"]:
                for k in hcfg["tier_a_n_states"]:
                    proba.update(assemble_hmm(data, best_proba, {(e, f.fold_id): k for f in folds},
                                              lambda em, k=k: f"hmm_{em}_K{k}"))
        else:
            chosen = {(e, f.fold_id): tuning["n_states"][task_name][e] for e in hcfg["emissions"] for f in folds}
        proba.update(assemble_hmm(data, best_proba, chosen, lambda em: f"hmm_{em}"))
        selected_params = {f"hmm_{e}|fold{f}|K{k}": {k2: v for k2, v in best_params[(e, f, k)].items() if k2 != "_objective"}
                           for (e, f), k in chosen.items()}
        write_json(tdir / f"models_{task_name}.json", {"markov": {f"{m}|fold{f}": d for (m, f), d in mk_fitted.items()},
                                                       "hmm_selected": selected_params,
                                                       "hmm_k_per_fold": {f"{e}|fold{f}": k for (e, f), k in chosen.items()}})

        priors = fold_priors(data, folds)
        fold_of_match = {m: f.fold_id for f in folds for m in f.validation_match_ids}
        fold_of_eval = data.table.loc[data.eval_rows, "match_id"].map(fold_of_match).to_numpy()
        y = data.table.loc[data.eval_rows, "target_id"].to_numpy()
        tuned = None
        if tier == "tier_a":
            tcfg = config.raw["cost"]["tuned"]
            tuning["gamma"][task_name], tuning["gamma_scores"][task_name] = {}, {}
            for m in DECISION_MODELS:
                g, scores = tune_gamma(y, proba[m], fold_of_eval, priors, tcfg["gamma_grid"], tcfg["cap"],
                                       config.raw["cost"]["tie_tolerance"], len(task.vocabulary))
                tuning["gamma"][task_name][m] = g
                tuning["gamma_scores"][task_name][m] = {str(k): v for k, v in scores.items()}
        else:
            tuned = tuning["gamma"][task_name]
        evaluate_task(tier, data, proba, folds, config, config.cost_schemes, DECISION_MODELS, rare, PAIRS, ev, tuned,
                      DECISION_MODELS + ["exp1_markov_order1", "exp1_markov_order2"], sink)
        approach_b.append(approach_b_check(tier, data, proba["exp1_markov_order1"], folds, config.cost_schemes["strong"],
                                           config.raw["cost"]["tie_tolerance"]))
        if tier == "tier_b":
            val0 = fold_seq_ids(data, folds[0].validation_match_ids)
            for e in hcfg["emissions"]:
                k = tuning["n_states"][task_name][e]
                model = CategoricalHMM.from_dict({k2: v for k2, v in best_params[(e, 0, k)].items() if k2 != "_objective"})
                states.append(hidden_state_analysis(tier, data, model, val0, f"hmm_{e}_K{k}"))
        folds_out[task_name] = [{"fold_id": f.fold_id, "train_match_ids": list(f.train_match_ids),
                                 "validation_match_ids": list(f.validation_match_ids)} for f in folds]
        cost_defs = {f"fold{f}": {"training_priors": pri.tolist(),
                                  **{s: {"miss_costs": miss_costs(pri, sc).tolist(), "matrix": cost_matrix(pri, sc).tolist()}
                                     for s, sc in config.cost_schemes.items()}} for f, pri in priors.items()}
        write_json(tdir / f"cost_matrices_{task_name}.json", {"vocabulary": list(task.vocabulary), "folds": cost_defs})
        del proba, best_proba

    pd.DataFrame(ev.summary).to_csv(tdir / "model_comparison.csv", index=False)
    pd.DataFrame(ev.by_fold).to_csv(tdir / "metrics_by_fold.csv", index=False)
    pd.concat(ev.per_class).to_csv(tdir / "per_class_metrics.csv", index=False)
    pd.concat(ev.confusion).to_csv(tdir / "confusion_matrices.csv", index=False)
    pd.DataFrame(ev.calibration).to_csv(tdir / "calibration_metrics.csv", index=False)
    pd.concat(ev.reliability).to_csv(tdir / "reliability_table.csv", index=False)
    pd.concat(ev.rare).to_csv(tdir / "rare_event_analysis.csv", index=False)
    pd.DataFrame(ev.cross_cost).to_csv(tdir / "cross_cost.csv", index=False)
    pd.DataFrame(ev.paired).to_csv(tdir / "paired_differences.csv", index=False)
    pd.concat(fit_logs).to_csv(tdir / "hmm_training_log.csv", index=False)
    pd.DataFrame(beta_choices).to_csv(tdir / "markov_concentration_choices.csv", index=False)
    pd.DataFrame(approach_b).to_csv(tdir / "approach_b_check.csv", index=False)
    if states:
        pd.concat(states).to_csv(tdir / "hidden_state_analysis.csv", index=False)
    write_json(tdir / "folds.json", folds_out)
    write_json(tdir / "integrity_checks.json", {**checks, "matches": len(matches),
                                                "final_test_overlap": sorted(set(matches) & set(part.final_test)),
                                                "runtime_seconds": round(time.time() - started, 1)})
    if tier == "tier_a":
        write_json(tuning_path, tuning)
    LOGGER.info("%s finished in %.0f s", tier, time.time() - started)


def stage_repro(config, out: Path) -> None:
    """Repeat selected experiments from scratch and compare with the stored results."""
    part = load_partition(config)
    base = out / "tiers" / "tier_b"
    folds_json = {t: json.loads((base / t / "folds.json").read_text(encoding="utf-8"))[t] for t in config.tasks}
    tuning = json.loads((out / "tiers" / "tier_a" / "tuning_decisions.json").read_text(encoding="utf-8"))
    stored = pd.concat([pd.read_csv(base / t / "model_comparison.csv") for t in config.tasks])
    events = load_development_events(config, part, part.tier_b)
    from src.sequence_models.experiment_v2 import arrays, probability_metrics
    from src.sequence_models.splits import FoldAssignment
    report: dict = {"scope": "Tier B: Markov models of both taxonomies re-run from raw data; fold-0 selected HMMs "
                             "re-fitted from scratch with the stored seed and init"}
    for task_name, task in config.tasks.items():
        data = prepare_task(events, task)
        folds = [FoldAssignment(f["fold_id"], tuple(f["train_match_ids"]), tuple(f["validation_match_ids"]))
                 for f in folds_json[task_name]]
        proba, _, _ = run_markov(data, folds, config)
        y = data.table.loc[data.eval_rows, "target_id"].to_numpy()
        diffs = {}
        for name in ("frequency_baseline", "mc1", "mc2", "mc1_zone", "mc2_zone", "exp1_markov_order1"):
            ll = probability_metrics(y, proba[name])["log_loss"]
            ref = stored[(stored["task"] == task_name) & (stored["model"] == name) & (stored["cost_scheme"] == "uniform")]
            diffs[name] = abs(ll - float(ref["log_loss"].iloc[0]))
        models = json.loads((base / task_name / f"models_{task_name}.json").read_text(encoding="utf-8"))
        refit = {}
        train_ids = fold_seq_ids(data, folds[0].train_match_ids)
        for key, params in models["hmm_selected"].items():
            if key.split("|")[1] != "fold0":
                continue
            m = CategoricalHMM.from_dict(params)
            fresh = CategoricalHMM(m.n_states, m.n_events, m.emission, m.n_zones, m.pseudo_count, m.max_iter, m.tol,
                                   m.seed, m.init_concentration, m.init_self_transition)
            fresh.fit(arrays(data, train_ids, "symbol_id"), None if m.emission == "event" else arrays(data, train_ids, "zone"))
            refit[key] = {"max_abs_transmat_diff": float(np.abs(fresh.transmat - m.transmat).max()),
                          "max_abs_emission_diff": float(np.abs(fresh.event_marginal_emission()
                                                                - m.event_marginal_emission()).max()),
                          "n_iter_stored_vs_refit": [m.record.n_iter, fresh.record.n_iter]}
        report[task_name] = {"markov_log_loss_abs_diff_vs_stored": diffs, "hmm_fold0_refit": refit,
                             "passed": max(diffs.values()) < 1e-12 and all(
                                 v["max_abs_transmat_diff"] < 1e-10 and v["max_abs_emission_diff"] < 1e-10 for v in refit.values())}
    write_json(out / "reproducibility_checks.json", report)
    LOGGER.info("repro: %s", {k: v["passed"] for k, v in report.items() if isinstance(v, dict)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["audit", "tier_a", "tier_b", "repro", "report"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=Path, default=None, help="alternative YAML (e.g. a smoke test)")
    parser.add_argument("--task", choices=["filtered8", "scenarioD15"], default=None, help="tier_b: taxonomy to run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config_v2(args.config) if args.config else load_config_v2()
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.path, out / "experiment_config.yaml")
    if args.stage == "audit":
        stage_audit(config, out)
    elif args.stage in ("tier_a", "tier_b"):
        stage_tier(args.stage, config, out, args.overwrite, args.task)
    elif args.stage == "repro":
        stage_repro(config, out)
    else:
        from src.sequence_models.report_v2 import stage_report
        stage_report(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
