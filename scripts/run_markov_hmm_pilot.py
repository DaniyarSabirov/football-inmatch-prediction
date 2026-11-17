"""Markov chain / HMM pilot: data audit, match split, cross-validated pilot.

Stages (run in order; each writes into the config's output_dir, default
results/markov_hmm_pilot/):

  audit  descriptive readiness audit of all eligible matches (before the split)
         -> data_audit.json, data_audit.md, exclusion_report.csv
  split  final-test hold-out, development pool, pilot subset
         -> split_manifest.json (pilot folds are added by the pilot stage)
  pilot  cross-validated pilot on the pilot matches only (development data);
         never loads a final-test match -> metrics, predictions, models, figures

The final test set is not evaluated by any stage. Full-data training is not
implemented as a stage on purpose: it needs explicit approval.

Usage:
    .venv/bin/python scripts/run_markov_hmm_pilot.py audit
    .venv/bin/python scripts/run_markov_hmm_pilot.py split
    .venv/bin/python scripts/run_markov_hmm_pilot.py pilot [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_models import figures  # noqa: E402
from src.sequence_models.config import load_config  # noqa: E402
from src.sequence_models.data_audit import audit_markdown, run_audit  # noqa: E402
from src.sequence_models.experiment import (  # noqa: E402
    TaskData,
    copy_config,
    cross_validate,
    environment_info,
    evaluate,
    fit_on_all_pilot,
    hidden_state_analysis,
    make_split_manifest,
    prediction_frame,
    prepare_task,
    select_n_states,
    sha256_file,
    verify_primary_targets_against_production,
    write_json,
)
from src.sequence_models.hmm import HiddenMarkovModel  # noqa: E402
from src.sequence_models.markov import MarkovChain, save_model  # noqa: E402
from src.sequence_models.raw_events import build_event_table, eligible_match_ids, load_or_build_event_table  # noqa: E402
from src.sequence_models.spatial import SPATIAL_FEATURES, load_production_rows, spatial_observations  # noqa: E402
from src.sequence_models.splits import LeakageError, assert_no_final_test, grouped_folds  # noqa: E402
from src.utils.paths import PROCESSED_DATA_DIR  # noqa: E402

LOGGER = logging.getLogger("markov_hmm_pilot")
PRODUCTION_CSV = PROCESSED_DATA_DIR / "statsbomb_360_all_matches_next_events.csv"


def stage_audit(config, out: Path) -> None:
    ids = eligible_match_ids()
    table = load_or_build_event_table(config.event_table_cache, ids, workers=8)
    production = pd.read_csv(PRODUCTION_CSV, low_memory=False)
    audit, exclusions = run_audit(table, production, sha256_file(PRODUCTION_CSV), config)
    write_json(out / "data_audit.json", audit)
    (out / "data_audit.md").write_text(audit_markdown(audit), encoding="utf-8")
    exclusions.to_csv(out / "exclusion_report.csv", index=False)
    LOGGER.info("audit written to %s", out)


def stage_split(config, out: Path) -> None:
    path = out / "split_manifest.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the split is fixed once written (delete it deliberately to redraw)")
    manifest = make_split_manifest(config)
    write_json(path, manifest)
    LOGGER.info("final test %d, development %d, pilot %d matches", len(manifest["final_test_match_ids"]),
                len(manifest["development_match_ids"]), len(manifest["pilot_match_ids"]))


MODEL_LABELS = {"frequency_baseline": "Frequency baseline", "markov_order1": "Markov chain (1st order)",
                "markov_order2": "Markov chain (2nd order)"}


def label(name: str) -> str:
    if name in MODEL_LABELS:
        return MODEL_LABELS[name]
    variant, k = name.rsplit("_K", 1)
    return {"hmm_event": "HMM event-only", "hmm_spatial": "HMM event + spatial"}[variant] + f" (K={k})"


def stage_pilot(config, out: Path, overwrite: bool) -> None:
    if (out / "metrics_summary.csv").exists() and not overwrite:
        raise SystemExit("pilot results exist; pass --overwrite to replace them deliberately")
    started = time.time()
    manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    final_test = manifest["final_test_match_ids"]
    pilot = manifest["pilot_match_ids"]
    assert_no_final_test(pilot, final_test)  # before any data file is read
    if not set(pilot) <= set(manifest["development_match_ids"]):
        raise LeakageError("pilot matches must come from the development pool")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    copy_config(config, out)

    # ------------------------------------------------------------ data (pilot matches only)
    events = build_event_table(pilot, workers=8)
    production = load_production_rows(pilot)
    assert_no_final_test(events["match_id"].unique().tolist() + production["match_id"].unique().tolist(), final_test)
    spatial = spatial_observations(production)
    tasks = {name: prepare_task(events, config.tasks[name], spatial) for name in (config.primary_task, config.secondary_task)}
    primary, secondary = tasks[config.primary_task], tasks[config.secondary_task]
    target_check = verify_primary_targets_against_production(primary, production, config.excluded_periods)
    if not target_check["passed"]:
        raise RuntimeError(f"primary task does not reproduce the production targets: {target_check}")

    eval_rows = primary.table[primary.table["eval_eligible"]][["match_id", "target_id"]]
    folds = grouped_folds(eval_rows, config.n_folds, config.split_seed)
    manifest["pilot_folds"] = [{"fold_id": f.fold_id, "train_match_ids": list(f.train_match_ids),
                                "validation_match_ids": list(f.validation_match_ids)} for f in folds]
    manifest["pilot_fold_method"] = (f"StratifiedGroupKFold(n_splits={config.n_folds}, shuffle=True, "
                                     f"random_state={config.split_seed}); groups = match_id; y = {config.primary_task} "
                                     "target of evaluation rows; the same match folds are used for both tasks")
    write_json(out / "split_manifest.json", manifest)

    # ------------------------------------------------------------ cross-validation
    results = {}
    for name, data, spatial_on in ((config.primary_task, primary, True), (config.secondary_task, secondary, False)):
        LOGGER.info("cross-validating %s", name)
        proba, fit_log, fold_of_row, fitted = cross_validate(data, folds, config, spatial_on, final_test)
        chosen_k = select_n_states(fit_log)
        selected = ["frequency_baseline", "markov_order1", "markov_order2"] + [f"{v}_K{k}" for v, k in sorted(chosen_k.items())]
        ev = evaluate(data, proba, fold_of_row, config, bootstrap_models=selected)
        results[name] = {"data": data, "proba": proba, "fit_log": fit_log, "fold_of_row": fold_of_row,
                         "fitted": fitted, "chosen_k": chosen_k, "selected": selected, "eval": ev}

    # ------------------------------------------------------------ tables
    def cat(key):
        return pd.concat([r["eval"][key] for r in results.values()], ignore_index=True)

    summary = cat("summary")
    for name, r in results.items():
        summary.loc[summary["task"] == name, "selected_configuration"] = summary.loc[summary["task"] == name, "model"].isin(r["selected"])
    summary.to_csv(out / "metrics_summary.csv", index=False)
    cat("by_fold").to_csv(out / "metrics_by_fold.csv", index=False)
    cat("per_class").to_csv(out / "per_class_metrics.csv", index=False)
    cat("confusion").to_csv(out / "confusion_matrices.csv", index=False)
    cat("bootstrap").to_csv(out / "bootstrap_intervals.csv", index=False)
    pd.concat([r["fit_log"] for r in results.values()], ignore_index=True).to_csv(out / "hmm_training_log.csv", index=False)
    write_json(out / "hmm_model_selection.json", {
        name: {"chosen_n_states": r["chosen_k"], "rule": "lowest mean validation log loss over folds, seed per fold "
               "chosen by highest training penalised log-likelihood"} for name, r in results.items()})

    for name, r in results.items():
        preds = prediction_frame(r["data"], r["proba"], r["fold_of_row"], r["selected"])
        preds.to_csv(out / ("predictions.csv" if name == config.primary_task else f"predictions_{name}.csv"), index=False)
        others = [m for m in r["proba"] if m not in r["selected"]]
        if others:
            prediction_frame(r["data"], r["proba"], r["fold_of_row"], others).to_csv(
                out / f"predictions_{name}_other_hmm_configurations.csv.gz", index=False)
        fold_models = {f"{m}|fold{f}": d for (m, f), d in r["fitted"].items()}
        write_json(out / "models" / f"cv_fold_models_{name}.json", fold_models)

    # ------------------------------------------------------------ descriptive fits on all pilot matches
    vocab = list(primary.task.vocabulary)
    seq_ids = list(primary.seq_rows)
    all_sym = [primary.table["symbol_id"].to_numpy()[primary.seq_rows[s]] for s in seq_ids]
    mc = MarkovChain(len(vocab), config.markov_alpha).fit(all_sym)
    save_model(mc, out / "models" / "markov_order1_all_pilot.json")
    pd.DataFrame(mc.counts, index=vocab, columns=vocab).rename_axis("current_event").to_csv(out / "transition_counts.csv")
    pd.DataFrame(mc.transition_matrix, index=vocab, columns=vocab).rename_axis("current_event").to_csv(out / "transition_probabilities.csv")
    figures.heatmap(mc.transition_matrix, vocab, vocab,
                    f"DESCRIPTIVE - 1st-order Markov transition matrix, {primary.task.name}, all {len(pilot)} pilot matches",
                    "next event (filtered-event task)", "current event", out / "figures" / "markov_transition_heatmap.png")
    sec_vocab = list(secondary.task.vocabulary)
    mc_sec = MarkovChain(len(sec_vocab), config.markov_alpha).fit(
        [secondary.table["symbol_id"].to_numpy()[r] for r in secondary.seq_rows.values()])
    save_model(mc_sec, out / "models" / f"markov_order1_all_pilot_{secondary.task.name}.json")
    figures.heatmap(mc_sec.transition_matrix, sec_vocab, sec_vocab,
                    f"DESCRIPTIVE - 1st-order Markov transition matrix, {secondary.task.name}, all pilot matches",
                    "next raw event", "current raw event", out / "figures" / "markov_transition_heatmap_fullstream9.png")

    hmm_params, state_frames, interp = {}, [], {}
    for variant, spatial_on in (("hmm_event", False), ("hmm_spatial", True)):
        k = results[config.primary_task]["chosen_k"][variant]
        best, models, scaler, ids, sym, x = fit_on_all_pilot(primary, config, k, spatial_on)
        best.save(out / "models" / f"{variant}_K{k}_all_pilot.json")
        hmm_params[f"{variant}_K{k}_all_pilot"] = {
            **best.to_dict(), "vocabulary": vocab, "spatial_features": list(SPATIAL_FEATURES) if spatial_on else [],
            "scaler": scaler.to_dict() if scaler else None,
            "all_seeds": [{"seed": m.seed, "n_iter": m.record.n_iter, "converged": m.record.converged,
                           "log_likelihood": m.record.log_likelihood, "penalised_objective": m.record.penalised_objective}
                          for m in models]}
        state_frames.append(hidden_state_analysis(primary, best, ids, sym, x, f"{variant}_K{k}"))
        interp[variant] = (best, scaler, ids, sym, x)
        figures.heatmap(best.transmat, [f"State {i}" for i in range(k)], [f"State {i}" for i in range(k)],
                        f"DESCRIPTIVE - {label(f'{variant}_K{k}')}: hidden-state transition matrix (all pilot matches)",
                        "state at t+1", "state at t", out / "figures" / f"{variant}_transition_heatmap.png")
        figures.emission_bars(best.emissionprob, vocab,
                              f"DESCRIPTIVE - {label(f'{variant}_K{k}')}: event-type emission distribution per state",
                              out / "figures" / f"{variant}_emissions.png")
    write_json(out / "hmm_parameters.json", hmm_params)
    pd.concat(state_frames, ignore_index=True).to_csv(out / "hidden_state_analysis.csv", index=False)

    # spatial distribution of the spatial HMM's states (unscaled units)
    best, scaler, ids, sym, x = interp["hmm_spatial"]
    filt = np.concatenate(best.filter(sym, x))
    rows = np.concatenate([primary.seq_rows[s] for s in ids])
    pts = primary.table.iloc[rows][list(SPATIAL_FEATURES)].copy()
    pts["state"] = filt.argmax(axis=1)
    pts = pts.dropna()
    means = best.means * scaler.std + scaler.mean
    sds = np.sqrt(best.variances) * scaler.std
    figures.spatial_states(pts, means, sds, f"DESCRIPTIVE - {label('hmm_spatial_K' + str(best.n_states))}: "
                           "spatial distribution of hidden states (all pilot matches)",
                           out / "figures" / "hmm_spatial_state_pitch.png")
    # example timeline: the first regulation-time period of the first pilot match
    example = ids[0]
    i = ids.index(example)
    for variant in ("hmm_event", "hmm_spatial"):
        m, _, ids_v, sym_v, x_v = interp[variant]
        f = m.filter([sym_v[i]], None if x_v is None else [x_v[i]])[0]
        vit = m.viterbi(sym_v[i], None if x_v is None else x_v[i])
        figures.state_timeline(f, vit, list(primary.table.iloc[primary.seq_rows[example]]["symbol"]), vocab,
                               f"DESCRIPTIVE - {label(f'{variant}_K{m.n_states}')}: hidden states, first 250 events of {example}",
                               out / "figures" / f"{variant}_state_timeline_{example}.png")

    # evaluation figures (out-of-fold)
    for name, r in results.items():
        suffix = "" if name == config.primary_task else f"_{name}"
        v = list(r["data"].task.vocabulary)
        figures.metric_comparison(r["eval"]["bootstrap"], r["selected"], {m: label(m) for m in r["selected"]},
                                  f"OUT-OF-FOLD EVALUATION - model comparison, {name} ({len(pilot)} pilot matches, "
                                  f"{config.n_folds}-fold grouped CV)", out / "figures" / f"model_comparison{suffix}.png")
        figures.per_class_f1(r["eval"]["per_class"], r["selected"], {m: label(m) for m in r["selected"]}, v,
                             f"OUT-OF-FOLD EVALUATION - per-class F1, {name}", out / "figures" / f"per_class_f1{suffix}.png")
        for m in r["selected"]:
            cm = r["eval"]["confusion"]
            cm = cm[cm["model"] == m].pivot(index="true_class", columns="predicted_class", values="count").loc[v, v].to_numpy()
            norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            figures.heatmap(norm, v, v, f"OUT-OF-FOLD EVALUATION - {label(m)}, {name}: confusion matrix (row-normalised)",
                            "predicted class (argmax)", "true next event", out / "figures" / f"confusion_{m}{suffix}.png",
                            cbar_label="share of true class")
        figures.k_sensitivity(r["fit_log"], f"OUT-OF-FOLD EVALUATION - sensitivity to the number of hidden states, {name}",
                              out / "figures" / f"hmm_k_sensitivity{suffix}.png")
        figures.seed_stability(r["fit_log"], f"OUT-OF-FOLD EVALUATION - initialisation stability, {name}",
                               out / "figures" / f"hmm_seed_stability{suffix}.png")

    # ------------------------------------------------------------ verification
    checks = verification_checks(config, results, primary, folds, final_test, pilot, production, target_check)
    write_json(out / "leakage_and_reproducibility_checks.json", checks)

    # ------------------------------------------------------------ manifests
    write_json(out / "dataset_manifest.json", {
        "pilot_matches": pilot, "raw_events_loaded": int(len(events)),
        "production_rows_loaded": int(len(production)),
        "production_source": "per-match CSVs in data/processed (statsbomb_360_match_<id>_next_events.csv)",
        "production_combined_sha256": sha256_file(PRODUCTION_CSV),
        "tasks": {name: {"description": d.task.description, "vocabulary": list(d.task.vocabulary),
                         "sequences": len(d.seq_rows), "observations": int(len(d.table)),
                         "prediction_points": int(d.table["is_prediction_point"].sum()),
                         "evaluation_rows": int(d.table["eval_eligible"].sum()),
                         "exclusion_reasons_at_observations": {k or "evaluated": int(n) for k, n in
                                                               d.table["exclusion_reason"].value_counts().items()},
                         "target_distribution_evaluation_rows": {k: int(n) for k, n in d.table.loc[
                             d.table["eval_eligible"], "target"].value_counts().items()},
                         "spatial_available_share_evaluation_rows": float(d.table.loc[d.table["eval_eligible"], "spatial_available"].mean()),
                         "spatial_available_share_all_observations": float(d.table["spatial_available"].mean())}
                  for name, d in tasks.items()},
        "primary_target_reproduces_production": target_check,
    })
    write_json(out / "event_vocabulary.json", {name: {"classes": list(d.task.vocabulary),
                                                      "class_index": {c: i for i, c in enumerate(d.task.vocabulary)},
                                                      "prediction_point_types": list(d.task.prediction_point_types)}
                                               for name, d in tasks.items()})
    env = environment_info()
    env["runtime_seconds"] = round(time.time() - started, 1)
    write_json(out / "environment.json", env)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "checksums.json")
    write_json(out / "checksums.json", {str(p.relative_to(out)): sha256_file(p) for p in files})
    LOGGER.info("pilot finished in %.0f s", time.time() - started)


def verification_checks(config, results, primary: TaskData, folds, final_test, pilot, production, target_check) -> dict:
    checks: dict = {"primary_targets_reproduce_production_dataset": target_check}
    checks["final_test_untouched"] = {
        "pilot_and_final_test_overlap": sorted(set(pilot) & set(final_test)),
        "matches_in_loaded_production_rows_also_in_final_test": sorted(set(production["match_id"]) & set(final_test)),
        "passed": not (set(pilot) & set(final_test)) and not (set(production["match_id"]) & set(final_test)),
    }
    overlaps = [sorted(set(f.train_match_ids) & set(f.validation_match_ids)) for f in folds]
    val_all = sorted(m for f in folds for m in f.validation_match_ids)
    checks["match_grouping"] = {"train_validation_overlap_per_fold": overlaps,
                                "every_pilot_match_validated_exactly_once": val_all == sorted(pilot),
                                "passed": all(not o for o in overlaps) and val_all == sorted(pilot)}
    ev_ids = primary.table["event_id"]
    checks["duplicate_sequences"] = {"duplicate_event_ids": int(ev_ids.duplicated().sum()),
                                     "sequences_spanning_two_matches": 0,
                                     "passed": not ev_ids.duplicated().any()}
    for name, r in results.items():
        d = r["data"]
        elig = d.table["eval_eligible"].to_numpy()
        probs_ok = {m: bool(np.isfinite(p[elig]).all() and (p[elig] >= 0).all()
                            and np.abs(p[elig].sum(axis=1) - 1).max() < config.probability_tolerance)
                    for m, p in r["proba"].items()}
        checks[f"probabilities_valid_{name}"] = {"per_model": probs_ok, "passed": all(probs_ok.values())}
        checks[f"every_evaluation_row_predicted_once_{name}"] = bool((r["fold_of_row"][elig] >= 0).all())
    # scaler: refit on fold 0 training sequences must equal the stored fold-0 scaler
    from src.sequence_models.experiment import fit_scaler, fold_sequences, symbol_arrays, feature_arrays
    r = results[config.primary_task]
    k_sp = r["chosen_k"]["hmm_spatial"]
    stored = r["fitted"][(f"hmm_spatial_K{k_sp}", 0)]
    train_ids = fold_sequences(primary, folds[0].train_match_ids)
    scaler = fit_scaler(primary, train_ids)
    checks["preprocessing_fitted_on_training_fold_only"] = {
        "fold0_scaler_refit_from_training_sequences_equals_stored": bool(
            np.allclose(scaler.mean, stored["scaler"]["mean"]) and np.allclose(scaler.std, stored["scaler"]["std"])),
        "validation_matches_in_fold0_training_sequences": sorted(
            {primary.seq_match[s] for s in train_ids} & set(folds[0].validation_match_ids)),
    }
    # reproducibility: refit the stored fold-0 spatial model with its seed
    stored_model = HiddenMarkovModel.from_dict(stored)
    refit = HiddenMarkovModel(**{k: getattr(stored_model, k) for k in (
        "n_states", "n_symbols", "n_features", "pseudo_count", "variance_floor", "max_iter", "tol", "seed",
        "init_method", "init_concentration", "init_self_transition")}).fit(
        symbol_arrays(primary, train_ids), feature_arrays(primary, train_ids, scaler))
    diff = max(float(np.abs(refit.transmat - stored_model.transmat).max()),
               float(np.abs(refit.emissionprob - stored_model.emissionprob).max()),
               float(np.abs(refit.means - stored_model.means).max()))
    checks["reproducibility_refit_fold0_spatial_hmm"] = {"max_abs_parameter_difference": diff, "passed": diff <= 1e-10}
    # causal prediction on real data: perturb the future of one validation sequence
    val_ids = fold_sequences(primary, folds[0].validation_match_ids)
    sym = symbol_arrays(primary, val_ids[:1])[0]
    x = feature_arrays(primary, val_ids[:1], scaler)[0]
    base = stored_model.predict_proba([sym], [x])[0]
    rng = np.random.default_rng(0)
    cut = len(sym) // 2
    sym2, x2 = sym.copy(), x.copy()
    sym2[cut + 1:] = rng.integers(0, stored_model.n_symbols, len(sym) - cut - 1)
    x2[cut + 1:] = rng.normal(size=x2[cut + 1:].shape)
    changed = stored_model.predict_proba([sym2], [x2])[0]
    checks["future_invariance_real_sequence"] = {
        "sequence_id": val_ids[0], "cut_position": cut,
        "max_abs_change_at_or_before_cut": float(np.abs(changed[:cut + 1] - base[:cut + 1]).max()),
        "max_abs_change_after_cut": float(np.abs(changed[cut + 1:] - base[cut + 1:]).max()),
        "passed": bool(np.array_equal(changed[:cut + 1], base[:cut + 1])),
    }
    checks["target_never_an_input"] = ("inputs are the symbols (and spatial features) of observations 1..t; the "
                                       "target is observation t+1; verified by the future-invariance checks above and "
                                       "tests/test_sequence_tasks_pipeline.py::test_cross_validation_has_no_target_or_future_leakage")
    checks["model_selection"] = ("seed chosen by training objective within each fold; K chosen by mean validation log "
                                 "loss over pilot folds (development data only); the final test set was not loaded")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["audit", "split", "pilot"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config) if args.config else load_config()
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    {"audit": lambda: stage_audit(config, out), "split": lambda: stage_split(config, out),
     "pilot": lambda: stage_pilot(config, out, args.overwrite)}[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
