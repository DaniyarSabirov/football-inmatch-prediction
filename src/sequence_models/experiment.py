"""Cross-validated pilot runner: data preparation, training, causal inference,
evaluation and artefact writing for the frequency baseline, Markov chains and
HMMs. Pure functions take explicit inputs; ``run_pilot`` orchestrates.

Leakage controls enforced here (each raises ``LeakageError`` on violation):

- the pilot loads raw events and production rows ONLY for pilot matches, which
  are checked against the final-test list before any file is read;
- folds are match-grouped and verified (``splits.verify_folds``);
- every fitted quantity (transition counts, HMM parameters, spatial scaler) is
  computed from training-fold sequences only; validation sequences are used
  only through ``predict_proba`` (causal filtering);
- HMM seeds are selected by TRAINING objective, the number of states by mean
  validation log loss across folds (a development-only choice; the final
  test set is never touched).
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT

from .config import ExperimentConfig, TaskSpec
from .hmm import HiddenMarkovModel
from .markov import FrequencyBaseline, MarkovChain, SecondOrderMarkovChain, save_model
from .metrics import (
    confusion,
    match_bootstrap,
    per_class_metrics,
    summary_metrics,
    validate_probabilities,
)
from .raw_events import build_event_table, eligible_match_ids, load_match_metadata, max_period
from .spatial import SPATIAL_FEATURES, SpatialScaler, load_production_rows, spatial_observations
from .splits import (
    FoldAssignment,
    LeakageError,
    assert_no_final_test,
    final_test_split,
    grouped_folds,
    pilot_subset,
)
from .tasks import build_task_table, sequences

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- splits
def make_split_manifest(config: ExperimentConfig) -> dict:
    """Final-test hold-out, development pool and pilot subset. The hold-out uses
    match metadata only; the extra-time flag is read from development matches'
    event files only (test-match event files are never opened here)."""
    ids = eligible_match_ids()
    metadata = load_match_metadata()
    strata = pd.Series({m: " | ".join(str(getattr(metadata[m], f)) for f in config.stratify_by) for m in ids})
    development, final_test = final_test_split(strata, config.final_test_fraction, config.split_seed)
    assert_no_final_test(development, final_test)
    extra_time = {m for m in development if max_period(m) >= 3}
    pilot = pilot_subset(development, strata, extra_time, config.pilot_matches, config.pilot_extra_time_matches,
                         config.split_seed)
    assert_no_final_test(pilot, final_test)
    return {
        "seed": config.split_seed,
        "method": "per-stratum random draw (numpy default_rng) of round(fraction * stratum size) matches; "
                  "strata = " + ", ".join(config.stratify_by),
        "eligible_matches": len(ids),
        "final_test_fraction": config.final_test_fraction,
        "final_test_match_ids": final_test,
        "development_match_ids": development,
        "pilot_match_ids": pilot,
        "pilot_extra_time_match_ids": sorted(set(pilot) & extra_time),
        "strata": {str(m): strata[m] for m in ids},
        "final_test_status": "UNTOUCHED: not loaded by the pilot; reserved for an explicitly authorised final evaluation",
    }


# --------------------------------------------------------------------------- data
@dataclass
class TaskData:
    task: TaskSpec
    table: pd.DataFrame                  # task table (+ spatial, team columns)
    seq_rows: dict[str, np.ndarray]      # sequence_id -> row positions
    seq_match: dict[str, int]


def prepare_task(event_table: pd.DataFrame, task: TaskSpec, spatial: pd.DataFrame) -> TaskData:
    table = build_task_table(event_table, task)
    extra = event_table[["event_id", "team_id", "possession_team_id", "possession", "play_pattern"]]
    table = table.merge(extra, on="event_id", how="left", validate="one_to_one")
    table = table.merge(spatial, on="event_id", how="left", validate="one_to_one")
    table = table.sort_values(["match_id", "period", "seq_pos"], kind="stable").reset_index(drop=True)
    prev_team = table.groupby("sequence_id", sort=False)["team_id"].shift(1)
    table["team_changed_since_previous"] = (table["team_id"] != prev_team) & prev_team.notna()
    table["spatial_available"] = table[list(SPATIAL_FEATURES)].notna().all(axis=1)
    seq_rows = {table.at[rows[0], "sequence_id"]: rows for rows in sequences(table)}
    seq_match = {sid: int(table.at[rows[0], "match_id"]) for sid, rows in seq_rows.items()}
    return TaskData(task=task, table=table, seq_rows=seq_rows, seq_match=seq_match)


def verify_primary_targets_against_production(data: TaskData, production_rows: pd.DataFrame,
                                              excluded_periods: tuple[int, ...]) -> dict:
    """Independent check: the filtered task's evaluation rows must be exactly the
    production dataset rows (minus excluded periods) with the same targets."""
    prod = production_rows[~production_rows["period"].isin(excluded_periods)]
    ours = data.table[data.table["eval_eligible"]]
    merged = ours.merge(prod[["current_event_id", "next_event_id", "next_event_type"]],
                        left_on="event_id", right_on="current_event_id", how="outer", indicator=True)
    both = merged[merged["_merge"] == "both"]
    result = {
        "rows_ours": int(len(ours)), "rows_production": int(len(prod)),
        "only_ours": int((merged["_merge"] == "left_only").sum()),
        "only_production": int((merged["_merge"] == "right_only").sum()),
        "target_type_mismatches": int((both["target"] != both["next_event_type"]).sum()),
        "target_event_mismatches": int((both["target_event_id"] != both["next_event_id"]).sum()),
    }
    result["passed"] = all(result[k] == 0 for k in ("only_ours", "only_production", "target_type_mismatches",
                                                     "target_event_mismatches"))
    return result


def fold_sequences(data: TaskData, match_ids: tuple[int, ...]) -> list[str]:
    wanted = set(match_ids)
    return [sid for sid, m in data.seq_match.items() if m in wanted]


def symbol_arrays(data: TaskData, seq_ids: list[str]) -> list[np.ndarray]:
    sym = data.table["symbol_id"].to_numpy()
    return [sym[data.seq_rows[s]] for s in seq_ids]


def feature_arrays(data: TaskData, seq_ids: list[str], scaler: SpatialScaler) -> list[np.ndarray]:
    values = scaler.transform(data.table[list(SPATIAL_FEATURES)].to_numpy(dtype=float))
    return [values[data.seq_rows[s]] for s in seq_ids]


def fit_scaler(data: TaskData, train_seq_ids: list[str]) -> SpatialScaler:
    rows = np.concatenate([data.seq_rows[s] for s in train_seq_ids])
    return SpatialScaler.fit(data.table[list(SPATIAL_FEATURES)].to_numpy(dtype=float)[rows])


# --------------------------------------------------------------------------- HMM jobs
def _fit_hmm_job(args: tuple) -> dict:
    (key, n_states, n_symbols, n_features, seed, hcfg, train_sym, train_x, val_sym, val_x) = args
    start = time.perf_counter()
    model = HiddenMarkovModel(
        n_states=n_states, n_symbols=n_symbols, n_features=n_features, pseudo_count=hcfg["pseudo_count"],
        variance_floor=hcfg["variance_floor"], max_iter=hcfg["max_iter"], tol=hcfg["tol"], seed=seed,
        init_method=hcfg["init_method"], init_concentration=hcfg["init_concentration"],
        init_self_transition=hcfg["init_self_transition"],
    ).fit(train_sym, train_x)
    val_proba = model.predict_proba(val_sym, val_x) if val_sym is not None else None
    return {"key": key, "model": model.to_dict(), "val_proba": val_proba, "seconds": time.perf_counter() - start}


def run_hmm_jobs(jobs: list[tuple], workers: int) -> list[dict]:
    if workers <= 1:
        return [_fit_hmm_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_fit_hmm_job, jobs, chunksize=1))


def hmm_config_dict(config: ExperimentConfig) -> dict:
    h = config.hmm
    return {"pseudo_count": h.pseudo_count, "variance_floor": h.variance_floor, "max_iter": h.max_iter, "tol": h.tol,
            "init_method": h.init_method, "init_concentration": h.init_concentration,
            "init_self_transition": h.init_self_transition}


# --------------------------------------------------------------------------- CV
def cross_validate(data: TaskData, folds: list[FoldAssignment], config: ExperimentConfig, spatial: bool,
                   final_test: list[int]) -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray, dict]:
    """Out-of-fold probabilities for every model, for every row of the task table.

    Returns (proba by model name -> (n_rows, V) array, NaN where not predicted),
    the HMM fit log (one row per fold x K x seed x variant), the fold id of each
    row, and the fitted parameters {(model name, fold id): dict} (HMMs: selected seed;
    spatial HMMs also carry the fold's scaler).
    """
    table = data.table
    n, v = len(table), len(data.task.vocabulary)
    proba: dict[str, np.ndarray] = {}
    fold_of_row = np.full(n, -1)
    fit_log = []
    fitted: dict = {}
    scalers: dict[int, dict] = {}
    hcfg = hmm_config_dict(config)
    jobs, job_meta = [], {}
    for fold in folds:
        assert_no_final_test(fold.train_match_ids + fold.validation_match_ids, final_test)
        train_ids, val_ids = fold_sequences(data, fold.train_match_ids), fold_sequences(data, fold.validation_match_ids)
        if set(train_ids) & set(val_ids):
            raise LeakageError(f"fold {fold.fold_id}: sequence in train and validation")
        train_sym, val_sym = symbol_arrays(data, train_ids), symbol_arrays(data, val_ids)
        val_rows = np.concatenate([data.seq_rows[s] for s in val_ids])
        fold_of_row[val_rows] = fold.fold_id
        baselines = {
            "frequency_baseline": FrequencyBaseline(v, config.baseline_alpha),
            "markov_order1": MarkovChain(v, config.markov_alpha),
            "markov_order2": SecondOrderMarkovChain(v, config.markov_alpha, config.second_order_min_context_count),
        }
        for name, model in baselines.items():
            model.fit(train_sym)
            fitted[(name, fold.fold_id)] = model.to_dict()
            out = proba.setdefault(name, np.full((n, v), np.nan))
            out[val_rows] = np.concatenate(model.predict_proba(val_sym))
        variants = [("hmm_event", None, None)]
        if spatial:
            scaler = fit_scaler(data, train_ids)
            scalers[fold.fold_id] = scaler.to_dict()
            variants.append(("hmm_spatial", feature_arrays(data, train_ids, scaler), feature_arrays(data, val_ids, scaler)))
        for variant, train_x, val_x in variants:
            n_features = 0 if train_x is None else len(SPATIAL_FEATURES)
            for k in config.hmm.n_states:
                for seed in config.hmm.seeds:
                    key = (variant, fold.fold_id, k, seed)
                    jobs.append((key, k, v, n_features, seed, hcfg, train_sym, train_x, val_sym, val_x))
                    job_meta[key] = val_rows
    LOGGER.info("%s: %d HMM fits", data.task.name, len(jobs))
    results = run_hmm_jobs(jobs, config.hmm.workers)
    by_key = {r["key"]: r for r in results}
    y = table["target_id"].to_numpy()
    eligible = table["eval_eligible"].to_numpy()
    for key, r in by_key.items():
        variant, fold_id, k, seed = key
        rec = r["model"]["record"]
        rows = job_meta[key]
        p = np.concatenate(r["val_proba"])
        mask = eligible[rows]
        m = summary_metrics(y[rows][mask], p[mask], v)
        fit_log.append({"task": data.task.name, "variant": variant, "fold_id": fold_id, "n_states": k, "seed": seed,
                        "init_method": rec["init_method"], "n_iter": rec["n_iter"], "converged": rec["converged"],
                        "monotonic": rec["monotonic"], "train_log_likelihood": rec["log_likelihood"],
                        "train_penalised_objective": rec["penalised_objective"],
                        "train_n_observations": rec["n_observations"],
                        "train_log_likelihood_per_obs": rec["log_likelihood"] / rec["n_observations"],
                        "val_log_loss": m["log_loss"], "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
                        "val_balanced_accuracy": m["balanced_accuracy"], "fit_seconds": r["seconds"]})
    fit_log = pd.DataFrame(fit_log).sort_values(["variant", "n_states", "fold_id", "seed"]).reset_index(drop=True)
    # seed selection per (variant, fold, K): highest TRAINING penalised objective
    best = fit_log.loc[fit_log.groupby(["variant", "fold_id", "n_states"])["train_penalised_objective"].idxmax()]
    fit_log["selected_seed"] = fit_log.index.isin(best.index)
    for _, row in best.iterrows():
        key = (row["variant"], int(row["fold_id"]), int(row["n_states"]), int(row["seed"]))
        name = f"{row['variant']}_K{int(row['n_states'])}"
        out = proba.setdefault(name, np.full((n, v), np.nan))
        out[job_meta[key]] = np.concatenate(by_key[key]["val_proba"])
        fitted[(name, key[1])] = {**by_key[key]["model"], "scaler": scalers.get(key[1]) if key[0] == "hmm_spatial" else None}
    return proba, fit_log, fold_of_row, fitted


def select_n_states(fit_log: pd.DataFrame) -> dict[str, int]:
    """Per variant: K with the lowest mean validation log loss over folds (selected seeds)."""
    chosen = fit_log[fit_log["selected_seed"]]
    means = chosen.groupby(["variant", "n_states"])["val_log_loss"].mean().reset_index()
    return {variant: int(g.loc[g["val_log_loss"].idxmin(), "n_states"]) for variant, g in means.groupby("variant")}


# --------------------------------------------------------------------------- evaluation
def evaluate(data: TaskData, proba: dict[str, np.ndarray], fold_of_row: np.ndarray, config: ExperimentConfig,
             reference_model: str = "markov_order1", bootstrap_models: list[str] | None = None) -> dict[str, pd.DataFrame]:
    table, vocab = data.table, data.task.vocabulary
    v = len(vocab)
    eligible = table["eval_eligible"].to_numpy()
    y = table["target_id"].to_numpy()
    groups = table["match_id"].to_numpy()
    summary, by_fold, per_class, confusions, boot = [], [], [], [], []
    for name, p in proba.items():
        pe, ye = p[eligible], y[eligible]
        validate_probabilities(pe, config.probability_tolerance)
        summary.append({"task": data.task.name, "model": name, "aggregation": "pooled_out_of_fold_events",
                        **summary_metrics(ye, pe, v)})
        for fold_id in sorted(set(fold_of_row[eligible])):
            sel = eligible & (fold_of_row == fold_id)
            by_fold.append({"task": data.task.name, "model": name, "fold_id": int(fold_id),
                            **summary_metrics(y[sel], p[sel], v)})
        pc = per_class_metrics(ye, pe, vocab)
        pc.insert(0, "model", name)
        pc.insert(0, "task", data.task.name)
        per_class.append(pc)
        cm = pd.DataFrame(confusion(ye, pe, v), index=list(vocab), columns=list(vocab))
        cm = cm.rename_axis("true_class").reset_index().melt(id_vars="true_class", var_name="predicted_class",
                                                                value_name="count")
        cm.insert(0, "model", name)
        cm.insert(0, "task", data.task.name)
        confusions.append(cm)
        if bootstrap_models is None or name in bootstrap_models:
            b = match_bootstrap(ye, pe, groups[eligible], v, config.bootstrap_replicates, config.bootstrap_seed)
            b.insert(0, "comparison", "absolute")
            b.insert(0, "model", name)
            boot.append(b)
            if name != reference_model and reference_model in proba:
                d = match_bootstrap(ye, pe, groups[eligible], v, config.bootstrap_replicates, config.bootstrap_seed,
                                    p_reference=proba[reference_model][eligible])
                d.insert(0, "comparison", f"difference_vs_{reference_model}")
                d.insert(0, "model", name)
                boot.append(d)
    by_fold_df = pd.DataFrame(by_fold)
    fold_mean = by_fold_df.groupby(["task", "model"]).agg(
        **{f"{m}_fold_mean": (m, "mean") for m in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                                                    "log_loss", "brier")},
        **{f"{m}_fold_sd": (m, "std") for m in ("macro_f1", "balanced_accuracy", "log_loss")},
    ).reset_index()
    summary_df = pd.DataFrame(summary).merge(fold_mean, on=["task", "model"])
    boot_df = pd.concat(boot, ignore_index=True) if boot else pd.DataFrame()
    if not boot_df.empty:
        boot_df.insert(0, "task", data.task.name)
    return {"summary": summary_df, "by_fold": by_fold_df, "per_class": pd.concat(per_class, ignore_index=True),
            "confusion": pd.concat(confusions, ignore_index=True), "bootstrap": boot_df}


def prediction_frame(data: TaskData, proba: dict[str, np.ndarray], fold_of_row: np.ndarray,
                     models: list[str]) -> pd.DataFrame:
    """Machine-readable predictions at every production-type observation of the
    validation sequences (evaluated or not; see eval_eligible/exclusion_reason)."""
    table, vocab = data.table, data.task.vocabulary
    keep = table["raw_type"].isin(data.task.prediction_point_types).to_numpy() & (fold_of_row >= 0)
    base = table.loc[keep]
    frames = []
    for name in models:
        p = proba[name][keep]
        frame = pd.DataFrame({
            "task": data.task.name, "model_name": name, "fold_id": fold_of_row[keep],
            "match_id": base["match_id"].to_numpy(), "event_id": base["event_id"].to_numpy(),
            "sequence_id": base["sequence_id"].to_numpy(), "sequence_position": base["seq_pos"].to_numpy(),
            "prediction_period": base["period"].to_numpy(), "prediction_time": base["timestamp"].to_numpy(),
            "prediction_event_index": base["index"].to_numpy(), "current_event": base["symbol"].to_numpy(),
            "true_next_event": base["target"].to_numpy(), "true_next_event_id": base["target_event_id"].to_numpy(),
            "true_next_raw_type": base["target_raw_type"].to_numpy(),
            "n_skipped_raw_events": base["n_skipped_raw_events"].to_numpy(),
            "predicted_next_event": np.array(vocab)[p.argmax(axis=1)],
            "eval_eligible": base["eval_eligible"].to_numpy(),
            "exclusion_reason": base["exclusion_reason"].replace("", np.nan).to_numpy(),
        })
        for j, cls in enumerate(vocab):
            frame[f"p_{cls}"] = p[:, j]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- final pilot fits (interpretation)
def fit_on_all_pilot(data: TaskData, config: ExperimentConfig, n_states: int, spatial: bool):
    seq_ids = list(data.seq_rows)
    sym = symbol_arrays(data, seq_ids)
    x, scaler = None, None
    if spatial:
        scaler = fit_scaler(data, seq_ids)
        x = feature_arrays(data, seq_ids, scaler)
    hcfg = hmm_config_dict(config)
    jobs = [(("all", s), n_states, len(data.task.vocabulary), len(SPATIAL_FEATURES) if spatial else 0, s, hcfg,
             sym, x, None, None) for s in config.hmm.seeds]
    results = run_hmm_jobs(jobs, config.hmm.workers)
    models = [HiddenMarkovModel.from_dict(r["model"]) for r in results]
    best = max(models, key=lambda m: m.record.penalised_objective)
    return best, models, scaler, seq_ids, sym, x


def hidden_state_analysis(data: TaskData, model: HiddenMarkovModel, seq_ids: list[str], sym, x,
                          variant: str) -> pd.DataFrame:
    """Posterior-weighted (FILTERED, causal) state characteristics on the pilot data."""
    filtered = np.concatenate(model.filter(sym, x))
    rows = np.concatenate([data.seq_rows[s] for s in seq_ids])
    t = data.table.iloc[rows]
    records = []
    vocab = data.task.vocabulary
    for k in range(model.n_states):
        w = filtered[:, k]
        total = w.sum()
        rec = {"variant": variant, "state": f"State {k}", "posterior_mass_share": total / len(w),
               "hard_assignment_share": float((filtered.argmax(axis=1) == k).mean()),
               "stationary_self_transition": float(model.transmat[k, k]),
               "expected_dwell_observations": float(1 / max(1 - model.transmat[k, k], 1e-12))}
        for j, cls in enumerate(vocab):
            rec[f"emission_{cls}"] = float(model.emissionprob[k, j])
            rec[f"observed_share_{cls}"] = float(w[t["symbol_id"].to_numpy() == j].sum() / total)
        sp = t["spatial_available"].to_numpy()
        ws = w[sp]
        for f in SPATIAL_FEATURES:
            rec[f"weighted_mean_{f}"] = float((ws * t[f].to_numpy()[sp]).sum() / ws.sum()) if ws.sum() > 0 else np.nan
        rec["weighted_share_team_changed_since_previous"] = float((w * t["team_changed_since_previous"].to_numpy()).sum() / total)
        rec["weighted_share_actor_is_possession_team"] = float(
            (w * (t["team_id"] == t["possession_team_id"]).to_numpy()).sum() / total)
        if model.n_features:
            for d, f in enumerate(SPATIAL_FEATURES):
                rec[f"gaussian_mean_std_{f}"] = float(model.means[k, d])
                rec[f"gaussian_var_std_{f}"] = float(model.variances[k, d])
        records.append(rec)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- provenance
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict:
    import matplotlib
    import scipy
    import sklearn
    import yaml

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout
    return {
        "python": sys.version, "platform": platform.platform(), "git_commit": commit,
        "uncommitted_changes": [line for line in dirty.splitlines()],
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__},
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


def copy_config(config: ExperimentConfig, out: Path) -> None:
    shutil.copyfile(config.path, out / "experiment_config.yaml")


__all__ = [
    "TaskData", "make_split_manifest", "prepare_task", "verify_primary_targets_against_production", "cross_validate",
    "select_n_states", "evaluate", "prediction_frame", "fit_on_all_pilot", "hidden_state_analysis", "sha256_file",
    "environment_info", "write_json", "copy_config", "build_event_table", "load_production_rows",
    "spatial_observations", "grouped_folds", "save_model",
]
