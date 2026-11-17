"""Experiment 2 runner: cost-sensitive decisions, Scenario D, pitch thirds, categorical spatial HMM.

Design (see results/markov_hmm_cost_sensitive_v2/methodology.md):

- **Tier A** (tuning/exploration): experiment 1's 30 pilot matches and its 5
  match folds. It is used to choose the HMM state count K (per emission and
  taxonomy) and the cost exponent of the tuned scheme.
- **Tier B** (held-out development evaluation): the other 304 development
  matches, with 5 new match-grouped folds. Every choice tuned on Tier A is
  frozen. Markov smoothing is chosen by nested inner CV inside each Tier B
  training split.
- The final-test matches are never read: every loader checks the match list
  against experiment 1's hold-out first, and experiment 1's all-match event
  cache is refused.

Within one (tier, taxonomy), all models share the evaluation rows and folds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import INTERIM_DATA_DIR

from .backoff_markov import MODEL_LEVELS, BackoffMarkov, build_contexts, context_cardinalities, targets_of
from .categorical_hmm import CategoricalHMM
from .config import TaskSpec
from .config_v2 import ConfigV2
from .cost_sensitive import (
    CostScheme,
    bayes_decisions,
    cost_matrix,
    miss_costs,
    prior_shifted,
    training_priors,
)
from .evaluation_v2 import (
    SUMMARY_KEYS,
    bootstrap_multiplicities,
    bootstrap_precision,
    bootstrap_recall,
    bootstrap_summary,
    calibration,
    confusion_counts,
    interval,
    metrics_from_confusion,
    per_class_from_confusion,
    per_match_stats,
    probability_metrics,
)
from .markov import MarkovChain, SecondOrderMarkovChain
from .raw_events import load_or_build_event_table
from .splits import FoldAssignment, LeakageError, assert_no_final_test, grouped_folds
from .tasks import build_task_table, sequences
from .zones import ZONE_NAMES, zone_of_x

LOGGER = logging.getLogger(__name__)
EXPERIMENT1_ALL_MATCH_CACHE = INTERIM_DATA_DIR / "sequence_models" / "event_table.pkl"
N_ZONES = len(ZONE_NAMES)


# =========================================================================== data
@dataclass
class Partition:
    final_test: list[int]
    development: list[int]
    tier_a: list[int]
    tier_a_folds: list[FoldAssignment]
    tier_b: list[int]


def load_partition(config: ConfigV2) -> Partition:
    manifest = json.loads(config.split_manifest.read_text(encoding="utf-8"))
    final_test = manifest["final_test_match_ids"]
    development = manifest["development_match_ids"]
    tier_a = manifest["pilot_match_ids"]
    folds = [FoldAssignment(f["fold_id"], tuple(f["train_match_ids"]), tuple(f["validation_match_ids"]))
             for f in manifest["pilot_folds"]]
    tier_b = sorted(set(development) - set(tier_a))
    assert_no_final_test(development, final_test)
    if not set(tier_a) <= set(development):
        raise LeakageError("tier A must lie in the development pool")
    return Partition(final_test, development, tier_a, folds, tier_b)


def load_development_events(config: ConfigV2, partition: Partition, match_ids: list[int], workers: int = 8) -> pd.DataFrame:
    """Raw event table of development matches only (cached), restricted to ``match_ids``."""
    assert_no_final_test(match_ids, partition.final_test)
    if config.event_table_cache.resolve() == EXPERIMENT1_ALL_MATCH_CACHE.resolve():
        raise LeakageError("experiment 1's all-match event cache contains final-test matches; refused")
    table = load_or_build_event_table(config.event_table_cache, partition.development, workers=workers)
    assert_no_final_test(table["match_id"].unique().tolist(), partition.final_test)
    return table[table["match_id"].isin(match_ids)].reset_index(drop=True)


@dataclass
class TaskData:
    task: TaskSpec
    table: pd.DataFrame
    seq_ids: list[str]
    seq_rows: dict[str, np.ndarray]
    seq_match: dict[str, int]
    eval_rows: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # table rows evaluated
    eval_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))   # table row -> eval index / -1

    @property
    def vocab(self) -> tuple[str, ...]:
        return self.task.vocabulary


def prepare_task(events: pd.DataFrame, task: TaskSpec) -> TaskData:
    table = build_task_table(events, task)
    extra = events[["event_id", "location_x", "location_y", "team_id", "possession_team_id", "possession"]]
    table = table.merge(extra, on="event_id", how="left", validate="one_to_one")
    table = table.sort_values(["match_id", "period", "seq_pos"], kind="stable").reset_index(drop=True)
    table["zone"] = zone_of_x(table["location_x"].to_numpy(dtype=float))
    prev_team = table.groupby("sequence_id", sort=False)["team_id"].shift(1)
    table["team_changed_since_previous"] = (table["team_id"] != prev_team) & prev_team.notna()
    seq_rows = {table.at[r[0], "sequence_id"]: r for r in sequences(table)}
    seq_ids = list(seq_rows)
    eval_rows = np.flatnonzero(table["eval_eligible"].to_numpy())
    eval_pos = np.full(len(table), -1, dtype=np.int64)
    eval_pos[eval_rows] = np.arange(len(eval_rows))
    return TaskData(task, table, seq_ids, seq_rows, {s: int(table.at[r[0], "match_id"]) for s, r in seq_rows.items()},
                    eval_rows, eval_pos)


def fold_seq_ids(data: TaskData, match_ids) -> list[str]:
    wanted = set(match_ids)
    return [s for s in data.seq_ids if data.seq_match[s] in wanted]


def arrays(data: TaskData, seq_ids: list[str], column: str) -> list[np.ndarray]:
    values = data.table[column].to_numpy()
    return [values[data.seq_rows[s]].astype(np.int64) for s in seq_ids]


def rows_of(data: TaskData, seq_ids: list[str]) -> np.ndarray:
    return np.concatenate([data.seq_rows[s] for s in seq_ids])


def eligible_targets(data: TaskData, seq_ids: list[str]) -> np.ndarray:
    rows = rows_of(data, seq_ids)
    rows = rows[data.table["eval_eligible"].to_numpy()[rows]]
    return data.table["target_id"].to_numpy()[rows]


# =========================================================================== Markov
def _markov_fit_predict(model_name: str, data: TaskData, train_ids: list[str], pred_ids: list[str],
                        concentration: float, base_alpha: float) -> tuple[BackoffMarkov, np.ndarray]:
    v = len(data.vocab)
    card = context_cardinalities(v, N_ZONES)
    use_zone = "zone" in {f for level in MODEL_LEVELS[model_name] for f in level}
    tr_sym = arrays(data, train_ids, "symbol_id")
    tr_z = arrays(data, train_ids, "zone") if use_zone else None
    model = BackoffMarkov(model_name, v, card, concentration, base_alpha)
    model.fit(build_contexts(tr_sym, tr_z, v, N_ZONES), targets_of(tr_sym))
    pr_sym = arrays(data, pred_ids, "symbol_id")
    pr_z = arrays(data, pred_ids, "zone") if use_zone else None
    return model, model.predict_proba(build_contexts(pr_sym, pr_z, v, N_ZONES))


def run_markov(data: TaskData, folds: list[FoldAssignment], config: ConfigV2) -> tuple[dict, list[dict], dict]:
    """Out-of-fold probabilities (n_eval, V) for frequency baseline, experiment-1 anchors and backoff chains.

    The backoff concentration is chosen per outer fold by inner leave-one-fold-out
    CV over the outer TRAINING folds (log loss on eligible rows). Validation
    matches of the outer fold are never touched by the choice.
    """
    mcfg = config.raw["markov"]
    v = len(data.vocab)
    n_eval = len(data.eval_rows)
    elig = data.table["eval_eligible"].to_numpy()
    y_all = data.table["target_id"].to_numpy()
    proba: dict[str, np.ndarray] = {}
    choices, fitted = [], {}

    def put(name: str, rows: np.ndarray, p: np.ndarray) -> None:
        out = proba.setdefault(name, np.full((n_eval, v), np.nan))
        keep = elig[rows]
        out[data.eval_pos[rows[keep]]] = p[keep]

    for fold in folds:
        train_ids, val_ids = fold_seq_ids(data, fold.train_match_ids), fold_seq_ids(data, fold.validation_match_ids)
        if set(train_ids) & set(val_ids):
            raise LeakageError("sequence in train and validation")
        val_rows = rows_of(data, val_ids)
        tr_sym, va_sym = arrays(data, train_ids, "symbol_id"), arrays(data, val_ids, "symbol_id")
        # experiment-1 anchors (Laplace first order; hard-backoff second order) and the frequency baseline
        mc1 = MarkovChain(v, 1.0).fit(tr_sym)
        put("frequency_baseline", val_rows, np.concatenate(mc1.fallback.predict_proba(va_sym)))
        put("exp1_markov_order1", val_rows, np.concatenate(mc1.predict_proba(va_sym)))
        mc2 = SecondOrderMarkovChain(v, 1.0, 20).fit(tr_sym)
        put("exp1_markov_order2", val_rows, np.concatenate(mc2.predict_proba(va_sym)))
        fitted[("exp1_markov_order1", fold.fold_id)] = mc1.to_dict()
        # nested choice of the concentration
        inner = [f for f in folds if f.fold_id != fold.fold_id]
        for name in mcfg["models"]:
            scores = {}
            for beta in mcfg["concentration_grid"]:
                ll, n = 0.0, 0
                for held in inner:
                    in_train = fold_seq_ids(data, [m for f in inner if f is not held for m in f.validation_match_ids])
                    in_val = fold_seq_ids(data, held.validation_match_ids)
                    _, p = _markov_fit_predict(name, data, in_train, in_val, beta, mcfg["base_alpha"])
                    rows = rows_of(data, in_val)
                    keep = elig[rows]
                    yy = y_all[rows][keep]
                    ll += float(-np.log(p[keep][np.arange(len(yy)), yy]).sum())
                    n += len(yy)
                scores[beta] = ll / n
            best = min(scores, key=lambda b: (scores[b], b))
            choices.append({"model": name, "fold_id": fold.fold_id, "chosen_concentration": best,
                            **{f"inner_log_loss_beta_{b}": s for b, s in scores.items()}})
            model, p = _markov_fit_predict(name, data, train_ids, val_ids, best, mcfg["base_alpha"])
            put(name, val_rows, p)
            fitted[(name, fold.fold_id)] = model.to_dict()
    return proba, choices, fitted


# =========================================================================== HMM
def _hmm_job(args: tuple) -> dict:
    key, emission, k, v, seed, w, hcfg, tr_ev, tr_z, va_ev, va_z, cache_file = args
    start = time.perf_counter()
    model = CategoricalHMM(n_states=k, n_events=v, emission=emission, n_zones=N_ZONES if emission != "event" else 0,
                           pseudo_count=hcfg["pseudo_count"], max_iter=hcfg["max_iter"], tol=hcfg["tol"], seed=seed,
                           init_concentration=hcfg["init_concentration"], init_self_transition=w)
    model.fit(tr_ev, tr_z)
    val = model.predict_proba(va_ev, va_z)
    result = {"key": key, "model": model.to_dict(), "val_proba": val, "seconds": time.perf_counter() - start}
    if cache_file is not None:  # persist each finished fit so an interrupted run resumes
        tmp = Path(str(cache_file) + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(result, f)
        tmp.replace(cache_file)
    return result


def hmm_cache_dir(root: Path, data: TaskData, folds: list[FoldAssignment], hcfg: dict) -> Path:
    """Fit cache keyed by everything that determines a fit: task, vocabulary, data rows,
    folds and HMM settings."""
    payload = json.dumps({"task": data.task.name, "vocab": list(data.vocab), "n_rows": len(data.table),
                          "symbols_sha": hashlib.sha256(data.table["symbol_id"].to_numpy().tobytes()).hexdigest(),
                          "zones_sha": hashlib.sha256(data.table["zone"].to_numpy().tobytes()).hexdigest(),
                          "folds": [[f.fold_id, list(f.train_match_ids), list(f.validation_match_ids)] for f in folds],
                          "hmm": {k: hcfg[k] for k in ("pseudo_count", "max_iter", "tol", "init_concentration")}},
                         sort_keys=True)
    path = root / f"{data.task.name}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_hmm(data: TaskData, folds: list[FoldAssignment], config: ConfigV2, grid: dict[str, list[int]],
            workers: int, final_test: list[int], inits: dict[str, list[float]] | None = None,
            cache_root: Path | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Fit every (emission, K, init) on every fold. Returns the fit log, the
    validation probabilities of the init with the highest TRAINING objective per
    (emission, fold, K), and that init's parameters."""
    hcfg = config.raw["hmm"]
    v = len(data.vocab)
    elig = data.table["eval_eligible"].to_numpy()
    y = data.table["target_id"].to_numpy()
    jobs, meta, cached = [], {}, []
    cache = hmm_cache_dir(cache_root, data, folds, hcfg) if cache_root is not None else None
    for fold in folds:
        assert_no_final_test(fold.train_match_ids + fold.validation_match_ids, final_test)
        train_ids, val_ids = fold_seq_ids(data, fold.train_match_ids), fold_seq_ids(data, fold.validation_match_ids)
        tr_ev, va_ev = arrays(data, train_ids, "symbol_id"), arrays(data, val_ids, "symbol_id")
        tr_z, va_z = arrays(data, train_ids, "zone"), arrays(data, val_ids, "zone")
        meta[fold.fold_id] = rows_of(data, val_ids)
        for emission, ks in grid.items():
            for k in ks:
                for w in (inits or {}).get(emission, hcfg["init_self_transition"]):
                    for seed in hcfg["seeds"]:
                        key = (emission, fold.fold_id, k, float(w), seed)
                        cache_file = None if cache is None else cache / f"{emission}_f{fold.fold_id}_K{k}_w{w}_s{seed}.pkl"
                        if cache_file is not None and cache_file.exists():
                            with cache_file.open("rb") as f:
                                cached.append(pickle.load(f))
                            continue
                        zt, zv = (None, None) if emission == "event" else (tr_z, va_z)
                        jobs.append((key, emission, k, v, seed, float(w), hcfg, tr_ev, zt, va_ev, zv, cache_file))
    # longest jobs first for better packing
    jobs.sort(key=lambda j: -j[2] * (2 if j[1] != "event" else 1))
    LOGGER.info("%s: %d HMM fits on %d workers (%d loaded from cache %s)", data.task.name, len(jobs), workers,
                len(cached), cache)
    log, best_proba, best_params = [], {}, {}

    def results():
        yield from cached
        if jobs:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                yield from pool.map(_hmm_job, jobs, chunksize=1)

    for r in results():
        emission, fold_id, k, w, seed = r["key"]
        rows = meta[fold_id]
        p = np.concatenate(r["val_proba"])
        keep = elig[rows]
        yy = y[rows][keep]
        rec = r["model"]["record"]
        entry = {"task": data.task.name, "emission": emission, "fold_id": fold_id, "n_states": k,
                 "init_self_transition": w, "seed": seed, "n_iter": rec["n_iter"], "converged": rec["converged"],
                 "monotonic": rec["monotonic"], "train_log_likelihood": rec["log_likelihood"],
                 "train_penalised_objective": rec["penalised_objective"],
                 "train_n_observations": rec["n_observations"],
                 "train_log_likelihood_per_obs": rec["log_likelihood"] / rec["n_observations"],
                 "val_log_loss": float(-np.log(p[keep][np.arange(len(yy)), yy]).mean()),
                 "n_free_parameters": CategoricalHMM.from_dict(r["model"]).n_free_parameters(),
                 "fit_seconds": r["seconds"]}
        log.append(entry)
        sel = (emission, fold_id, k)
        if sel not in best_params or entry["train_penalised_objective"] > best_params[sel]["_objective"]:
            best_params[sel] = {**r["model"], "_objective": entry["train_penalised_objective"]}
            best_proba[sel] = (rows, p)
    log = pd.DataFrame(log).sort_values(["emission", "n_states", "fold_id", "init_self_transition", "seed"])
    best = log.loc[log.groupby(["emission", "fold_id", "n_states"])["train_penalised_objective"].idxmax()]
    log["selected_init"] = log.index.isin(best.index)
    return log.reset_index(drop=True), best_proba, best_params


def assemble_hmm(data: TaskData, best_proba: dict, chosen: dict[tuple[str, int], int], name_of) -> dict[str, np.ndarray]:
    """Probability arrays (n_eval, V) for given (emission, fold) -> K choices."""
    v = len(data.vocab)
    elig = data.table["eval_eligible"].to_numpy()
    out: dict[str, np.ndarray] = {}
    for (emission, fold_id), k in chosen.items():
        rows, p = best_proba[(emission, fold_id, k)]
        arr = out.setdefault(name_of(emission), np.full((len(data.eval_rows), v), np.nan))
        keep = elig[rows]
        arr[data.eval_pos[rows[keep]]] = p[keep]
    return out


def cross_fold_k(log: pd.DataFrame) -> dict[tuple[str, int], int]:
    """K for fold f = argmin over K of the mean validation log loss of the OTHER folds
    (selected inits). Fold f's own validation labels never choose its K."""
    sel = log[log["selected_init"]]
    out = {}
    for emission, g in sel.groupby("emission"):
        for f in sorted(g["fold_id"].unique()):
            others = g[g["fold_id"] != f].groupby("n_states")["val_log_loss"].mean()
            out[(emission, int(f))] = int(others.idxmin())
    return out


def winning_init(log: pd.DataFrame) -> dict[str, float]:
    """Per emission: the initialisation type that most often had the highest TRAINING
    objective (selected init) across folds and K. Uses no validation labels."""
    sel = log[log["selected_init"]]
    return {e: float(g["init_self_transition"].value_counts().sort_index().idxmax()) for e, g in sel.groupby("emission")}


def overall_k(log: pd.DataFrame) -> dict[str, int]:
    sel = log[log["selected_init"]]
    return {e: int(g.groupby("n_states")["val_log_loss"].mean().idxmin()) for e, g in sel.groupby("emission")}


# =========================================================================== cost-sensitive decisions
def fold_priors(data: TaskData, folds: list[FoldAssignment]) -> dict[int, np.ndarray]:
    """Class priors per fold from the fold's TRAINING evaluation rows only."""
    v = len(data.vocab)
    return {f.fold_id: training_priors(eligible_targets(data, fold_seq_ids(data, f.train_match_ids)), v) for f in folds}


def decide(p: np.ndarray, fold_of_eval: np.ndarray, priors: dict[int, np.ndarray],
           scheme: CostScheme, tie_tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Bayes decisions and the realised-cost matrix row lookup (per-fold cost matrices)."""
    decisions = np.empty(len(p), dtype=np.int64)
    expected = np.empty(len(p))
    for f, pri in priors.items():
        rows = fold_of_eval == f
        c = cost_matrix(pri, scheme)
        decisions[rows] = bayes_decisions(p[rows], c, tie_tolerance)
        expected[rows] = (p[rows] @ c.T)[np.arange(rows.sum()), decisions[rows]]
    return decisions, expected


def realised_costs(y: np.ndarray, decisions: np.ndarray, fold_of_eval: np.ndarray, priors: dict[int, np.ndarray],
                   scheme: CostScheme) -> np.ndarray:
    out = np.empty(len(y))
    for f, pri in priors.items():
        rows = fold_of_eval == f
        out[rows] = cost_matrix(pri, scheme)[decisions[rows], y[rows]]
    return out


def tune_gamma(y: np.ndarray, p: np.ndarray, fold_of_eval: np.ndarray, priors: dict[int, np.ndarray],
               grid: list[float], cap: float, tie_tolerance: float, n_classes: int) -> tuple[float, dict]:
    """Cost exponent maximising pooled macro-F1 on TIER A out-of-fold predictions (ties -> smaller gamma)."""
    scores = {}
    for g in grid:
        d, _ = decide(p, fold_of_eval, priors, CostScheme("tuning", g, cap), tie_tolerance)
        scores[g] = metrics_from_confusion(confusion_counts(y, d, n_classes))["macro_f1"]
    best = max(scores, key=lambda g: (round(scores[g], 12), -g))
    return best, scores


# =========================================================================== evaluation
@dataclass
class Evaluated:
    summary: list[dict] = field(default_factory=list)
    by_fold: list[dict] = field(default_factory=list)
    per_class: list[pd.DataFrame] = field(default_factory=list)
    confusion: list[pd.DataFrame] = field(default_factory=list)
    calibration: list[dict] = field(default_factory=list)
    reliability: list[pd.DataFrame] = field(default_factory=list)
    rare: list[pd.DataFrame] = field(default_factory=list)
    cross_cost: list[dict] = field(default_factory=list)
    paired: list[dict] = field(default_factory=list)


def evaluate_task(tier: str, data: TaskData, proba: dict[str, np.ndarray], folds: list[FoldAssignment],
                  config: ConfigV2, schemes: dict[str, CostScheme], decision_models: list[str],
                  rare_classes: list[str], pairs: list[tuple[str, str]], ev: Evaluated, tuned_gamma: dict[str, float] | None,
                  prediction_models: list[str], prediction_sink=None) -> None:
    """Evaluate every model of one (tier, taxonomy) on identical rows; results go to ``ev``.

    Predictions of ``prediction_models`` are passed to ``prediction_sink`` as ONE wide
    frame: one row per evaluation observation, metadata once, then per model
    ``p__<model>__<class>`` (probability) and ``decision__<model>__<scheme>`` columns."""
    vocab = data.vocab
    v = len(vocab)
    t = data.table.iloc[data.eval_rows]
    y = t["target_id"].to_numpy()
    groups = t["match_id"].to_numpy()
    fold_of_match = {m: f.fold_id for f in folds for m in f.validation_match_ids}
    fold_of_eval = t["match_id"].map(fold_of_match).to_numpy()
    priors = fold_priors(data, folds)
    tol = float(config.raw["cost"]["tie_tolerance"])
    ecfg = config.raw["evaluation"]
    mult = bootstrap_multiplicities(len(np.unique(groups)), int(ecfg["bootstrap_replicates"]), int(ecfg["bootstrap_seed"]))
    rare_idx = [vocab.index(c) for c in rare_classes]
    stats_cache: dict[str, dict] = {}
    wide = None
    if prediction_sink is not None:
        wide = t[["match_id", "event_id", "sequence_id", "seq_pos", "period", "timestamp", "index", "symbol",
                  "zone", "target", "n_skipped_raw_events"]].copy()
        wide.columns = ["match_id", "event_id", "sequence_id", "sequence_position", "prediction_period",
                        "prediction_time", "prediction_event_index", "current_event", "current_zone",
                        "true_next_event", "n_skipped_raw_events"]
        wide.insert(0, "fold_id", fold_of_eval)
        wide.insert(0, "task", data.task.name)
        wide.insert(0, "tier", tier)
        wide = wide.reset_index(drop=True)
        extra_columns: dict[str, object] = {}

    for name, p in proba.items():
        if np.isnan(p).any():
            raise ValueError(f"{name}: some evaluation rows were not predicted")
        dev = np.abs(p.sum(axis=1) - 1).max()
        if dev > ecfg["probability_tolerance"] or (p <= 0).any():
            raise ValueError(f"{name}: invalid probabilities (max deviation {dev})")
        prob = probability_metrics(y, p)
        model_schemes = dict(schemes) if name in decision_models else {"uniform": schemes["uniform"]}
        if name in decision_models and tuned_gamma is not None and name in tuned_gamma:
            model_schemes["tuned"] = CostScheme("tuned", tuned_gamma[name], float(config.raw["cost"]["tuned"]["cap"]),
                                                description=f"gamma {tuned_gamma[name]} chosen on Tier A (macro-F1)")
        decisions = {}
        for sname, scheme in model_schemes.items():
            d, expected = decide(p, fold_of_eval, priors, scheme, tol)
            decisions[sname] = d
            cm = confusion_counts(y, d, v)
            dm = metrics_from_confusion(cm)
            rc = realised_costs(y, d, fold_of_eval, priors, scheme)
            row = {"tier": tier, "task": data.task.name, "model": name, "cost_scheme": sname,
                   "gamma": scheme.gamma, "cap": scheme.cap, "n": len(y), **dm, **prob,
                   "realised_cost": float(rc.mean()), "expected_cost": float(expected.mean()),
                   "n_classes_predicted": int((cm.sum(axis=0) > 0).sum())}
            ev.summary.append(row)
            pc = per_class_from_confusion(cm, vocab)
            pc.insert(0, "cost_scheme", sname)
            pc.insert(0, "model", name)
            pc.insert(0, "task", data.task.name)
            pc.insert(0, "tier", tier)
            ev.per_class.append(pc)
            cmf = pd.DataFrame(cm, index=list(vocab), columns=list(vocab)).rename_axis("true_class").reset_index()
            cmf = cmf.melt(id_vars="true_class", var_name="predicted_class", value_name="count")
            cmf.insert(0, "cost_scheme", sname)
            cmf.insert(0, "model", name)
            cmf.insert(0, "task", data.task.name)
            cmf.insert(0, "tier", tier)
            ev.confusion.append(cmf)
            for f in sorted(priors):
                sel = fold_of_eval == f
                fm = metrics_from_confusion(confusion_counts(y[sel], d[sel], v))
                ev.by_fold.append({"tier": tier, "task": data.task.name, "model": name, "cost_scheme": sname,
                                   "fold_id": int(f), "n": int(sel.sum()), **fm, **probability_metrics(y[sel], p[sel]),
                                   "realised_cost": float(rc[sel].mean())})
            # cross-cost: these decisions under every scheme's cost matrix
            for other_name, other in model_schemes.items():
                ev.cross_cost.append({"tier": tier, "task": data.task.name, "model": name, "decision_scheme": sname,
                                      "evaluated_under": other_name,
                                      "realised_cost": float(realised_costs(y, d, fold_of_eval, priors, other).mean())})
        cal, rel = calibration(y, p, vocab)
        ev.calibration.append({"tier": tier, "task": data.task.name, "model": name, **prob, **cal})
        rel.insert(0, "model", name)
        rel.insert(0, "task", data.task.name)
        rel.insert(0, "tier", tier)
        ev.reliability.append(rel)

        # bootstrap: summaries per scheme; rare-class recall/precision and paired differences vs uniform
        stats = per_match_stats(y, p, decisions, groups, v)
        stats_cache[name] = stats
        for sname in decisions:
            reps = bootstrap_summary(stats, sname, mult)
            lo, hi = interval(reps)
            r = next(r for r in ev.summary if r["tier"] == tier and r["task"] == data.task.name
                     and r["model"] == name and r["cost_scheme"] == sname)
            for i, key in enumerate(SUMMARY_KEYS):
                r[f"{key}_ci_low"], r[f"{key}_ci_high"] = float(lo[i]), float(hi[i])
            if sname != "uniform":
                diff = reps - bootstrap_summary(stats, "uniform", mult)
                dlo, dhi = interval(diff)
                for i, key in enumerate(SUMMARY_KEYS[:4]):
                    r[f"{key}_minus_uniform"] = r[key] - next(
                        x for x in ev.summary if x["tier"] == tier and x["task"] == data.task.name
                        and x["model"] == name and x["cost_scheme"] == "uniform")[key]
                    r[f"{key}_minus_uniform_ci_low"], r[f"{key}_minus_uniform_ci_high"] = float(dlo[i]), float(dhi[i])
        if rare_idx:
            rec_u = bootstrap_recall(stats, "uniform", mult)
            for sname, d in decisions.items():
                rec = bootstrap_recall(stats, sname, mult)
                prec = bootstrap_precision(stats, sname, mult)
                cm = confusion_counts(y, d, v)
                pc = per_class_from_confusion(cm, vocab).set_index("class")
                cm_u = per_class_from_confusion(confusion_counts(y, decisions["uniform"], v), vocab).set_index("class")
                per_match = per_match_recall(y, d, decisions["uniform"], groups, rare_idx)
                for j, c in zip(rare_idx, rare_classes):
                    rlo, rhi = interval(rec[:, j])
                    plo, phi = interval(prec[:, j])
                    dlo, dhi = interval(rec[:, j] - rec_u[:, j])
                    pm = per_match[j]
                    ev.rare.append(pd.DataFrame([{
                        "tier": tier, "task": data.task.name, "model": name, "cost_scheme": sname, "class": c,
                        "support": int(pc.at[c, "support"]), "prevalence": pc.at[c, "prevalence"],
                        "n_predicted": int(pc.at[c, "n_predicted"]), "true_positives": int(pc.at[c, "true_positives"]),
                        "false_positives": int(pc.at[c, "false_positives"]),
                        "false_negatives": int(pc.at[c, "false_negatives"]),
                        "precision": pc.at[c, "precision"], "precision_ci_low": plo, "precision_ci_high": phi,
                        "recall": pc.at[c, "recall"], "recall_ci_low": rlo, "recall_ci_high": rhi,
                        "f1": pc.at[c, "f1"], "lift_over_prevalence": pc.at[c, "lift"],
                        "recall_minus_uniform": pc.at[c, "recall"] - cm_u.at[c, "recall"],
                        "recall_minus_uniform_ci_low": dlo, "recall_minus_uniform_ci_high": dhi,
                        "extra_false_positives_vs_uniform": int(pc.at[c, "false_positives"] - cm_u.at[c, "false_positives"]),
                        "matches_with_support": pm["matches"], "matches_recall_improved": pm["improved"],
                        "matches_recall_worse": pm["worse"], "median_match_recall": pm["median_recall"],
                    }]))
        if wide is not None and name in prediction_models:
            for j, c in enumerate(vocab):
                extra_columns[f"p__{name}__{c}"] = p[:, j].astype(np.float32)
            for sname, d in decisions.items():
                extra_columns[f"decision__{name}__{sname}"] = pd.Categorical.from_codes(d, categories=list(vocab))

    if wide is not None:
        prediction_sink(pd.concat([wide, pd.DataFrame(extra_columns)], axis=1))

    # paired model differences (uniform decisions + probability metrics), same rows, same resampled matches
    def uniform_row(model: str) -> dict:
        return next(r for r in ev.summary if r["tier"] == tier and r["task"] == data.task.name
                    and r["model"] == model and r["cost_scheme"] == "uniform")

    for model, reference in pairs:
        if model not in stats_cache or reference not in stats_cache:
            continue
        diff = bootstrap_summary(stats_cache[model], "uniform", mult) - bootstrap_summary(stats_cache[reference], "uniform", mult)
        lo, hi = interval(diff)
        for i, key in enumerate(SUMMARY_KEYS):
            ev.paired.append({"tier": tier, "task": data.task.name, "model": model, "reference": reference,
                              "metric": key, "model_value": uniform_row(model)[key],
                              "reference_value": uniform_row(reference)[key],
                              "difference": uniform_row(model)[key] - uniform_row(reference)[key],
                              "ci_low": float(lo[i]), "ci_high": float(hi[i]),
                              "ci_excludes_zero": bool(lo[i] > 0 or hi[i] < 0)})


def per_match_recall(y: np.ndarray, d: np.ndarray, d_uniform: np.ndarray, groups: np.ndarray,
                     rare_idx: list[int]) -> dict[int, dict]:
    """Per rare class: matches with >= 3 true occurrences, and in how many of them the
    rule's recall is above / below the uniform rule's recall."""
    frame = pd.DataFrame({"g": groups, "y": y, "hit": d == y, "hit_u": d_uniform == y})
    out = {}
    for j in rare_idx:
        f = frame[frame["y"] == j].groupby("g").agg(n=("hit", "size"), r=("hit", "mean"), ru=("hit_u", "mean"))
        f = f[f["n"] >= 3]
        out[j] = {"matches": int(len(f)), "improved": int((f["r"] > f["ru"]).sum()),
                  "worse": int((f["r"] < f["ru"]).sum()),
                  "median_recall": float(f["r"].median()) if len(f) else float("nan")}
    return out


# =========================================================================== Approach B demonstration
def approach_b_check(tier: str, data: TaskData, p_laplace: np.ndarray, folds: list[FoldAssignment],
                     scheme: CostScheme, tie_tolerance: float) -> dict:
    """Class-weighted training of a Laplace Markov chain estimates q = w p / sum(w p)
    (weights applied to counts AND pseudo-counts). Compare with Approach A."""
    t = data.table.iloc[data.eval_rows]
    y = t["target_id"].to_numpy()
    fold_of_match = {m: f.fold_id for f in folds for m in f.validation_match_ids}
    fold_of_eval = t["match_id"].map(fold_of_match).to_numpy()
    priors = fold_priors(data, folds)
    q = np.empty_like(p_laplace)
    d_a = np.empty(len(y), dtype=np.int64)
    for f, pri in priors.items():
        rows = fold_of_eval == f
        w = miss_costs(pri, scheme)
        q[rows] = prior_shifted(p_laplace[rows], w)
        d_a[rows] = bayes_decisions(p_laplace[rows], cost_matrix(pri, scheme), tie_tolerance)
    d_b = q.argmax(axis=1)
    return {"tier": tier, "task": data.task.name, "model": "exp1_markov_order1", "cost_scheme": scheme.name,
            "decision_agreement_A_vs_B": float((d_a == d_b).mean()),
            "log_loss_A_unchanged_probabilities": probability_metrics(y, p_laplace)["log_loss"],
            "log_loss_B_weighted_training": probability_metrics(y, q)["log_loss"],
            "brier_A": probability_metrics(y, p_laplace)["brier"], "brier_B": probability_metrics(y, q)["brier"],
            "max_abs_calibration_in_the_large_A": calibration(y, p_laplace, data.vocab)[0]["max_abs_calibration_in_the_large"],
            "max_abs_calibration_in_the_large_B": calibration(y, q, data.vocab)[0]["max_abs_calibration_in_the_large"]}


# =========================================================================== hidden states
def hidden_state_analysis(tier: str, data: TaskData, model: CategoricalHMM, seq_ids: list[str], label: str) -> pd.DataFrame:
    """Filtered (causal) posterior-weighted state characteristics on the given (validation) sequences."""
    ev = arrays(data, seq_ids, "symbol_id")
    zn = arrays(data, seq_ids, "zone") if model.emission != "event" else None
    post = np.concatenate(model.filter(ev, zn))
    t = data.table.iloc[rows_of(data, seq_ids)]
    sym, zone = t["symbol_id"].to_numpy(), t["zone"].to_numpy()
    m = model.event_marginal_emission()
    records = []
    for k in range(model.n_states):
        w = post[:, k]
        tot = w.sum()
        rec = {"tier": tier, "task": data.task.name, "model": label, "state": f"State {k}",
               "posterior_mass_share": tot / len(w), "self_transition": float(model.transmat[k, k]),
               "expected_dwell_events": float(1 / max(1 - model.transmat[k, k], 1e-12)),
               "mean_location_x": float((w * t["location_x"].fillna(0).to_numpy() * (zone >= 0)).sum()
                                        / max((w * (zone >= 0)).sum(), 1e-12)),
               "share_team_changed_since_previous": float((w * t["team_changed_since_previous"].to_numpy()).sum() / tot)}
        for zi, zname in enumerate(ZONE_NAMES):
            rec[f"observed_share_{zname}"] = float(w[zone == zi].sum() / max(w[zone >= 0].sum(), 1e-12))
            if model.emission == "factorized":
                rec[f"emission_{zname}"] = float(model.zone_emission[k, zi])
            elif model.emission == "joint":
                rec[f"emission_{zname}"] = float(model.joint_emission[k, :, zi].sum())
        top = np.argsort(m[k])[::-1][:3]
        rec["top_event_emissions"] = "; ".join(f"{data.vocab[i]} {m[k, i]:.2f}" for i in top)
        for i, c in enumerate(data.vocab):
            rec[f"emission_{c}"] = float(m[k, i])
            rec[f"observed_share_{c}"] = float(w[sym == i].sum() / tot)
        records.append(rec)
    return pd.DataFrame(records)
