r"""Experiment 4: multinomial logistic regression with the current event only (F0), K = 15.

Question: with exactly the information of a first-order Markov chain, :math:`E_t`, does a
softmax regression estimate a different :math:`P(E_{t+1} = c \mid E_t)`? Theory predicts no
(``docs/thesis/02_mathematical_framework.md``, section 6), so this experiment is a controlled
baseline and an implementation check, not a search for a better model.

Protocol (identical to experiment 2's Tier B, scenarioD15):

- Matches: Tier B = the 304 development matches outside the 30-match pilot. The 83
  final-test matches are refused by every loader (``assert_no_final_test``).
- Folds: experiment 2's 5 match-grouped folds, read from its ``folds.json`` and re-verified
  (every match validates exactly once; no match trains and validates in one fold).
- Training rows of a fold: every within-sequence transition (E_t, E_{t+1}) of its training
  matches, exactly the transitions a Markov chain counts. Evaluation rows: prediction
  points of the validation matches whose current event has a 360 record (``eval_eligible``),
  701,144 rows in total. All models are trained and evaluated on identical rows.
- Hyperparameters: logistic C and (supplementary) Markov alpha are chosen per outer fold by
  leave-one-inner-fold-out log loss on the outer TRAINING folds; validation labels of the
  outer fold never influence a choice.

For F0 every quantity depends on the data only through the 15 x 15 transition counts, so the
inner search fits the logistic regression on the 225 (current, next) cells with count
weights; the objective is identical (``logistic.aggregate_one_hot``). The reported outer-fold
models are fitted on the individual rows, and the agreement of both fits is checked.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from src.utils.paths import PROJECT_ROOT

from .config_v2 import load_config_v2
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
from .event_classes import K, SCENARIO_D15_CLASSES, assert_canonical_vocabulary
from .experiment_v2 import TaskData, arrays, fold_seq_ids, load_development_events, load_partition, prepare_task
from .feature_sets import FORBIDDEN_COLUMNS, REGISTRY, design_matrix, one_hot
from .logistic import MultinomialLogisticModel
from .markov import MarkovChain, SecondOrderMarkovChain, normalise_rows
from .splits import FoldAssignment, LeakageError, assert_no_final_test, verify_folds

LOGGER = logging.getLogger(__name__)

PRIMARY_MODELS = ("frequency_baseline", "markov_order1", "logistic_F0")
SUPPLEMENTARY_MODELS = ("logistic_F0_unpenalised", "markov_order1_nested_alpha", "markov_order2")
PAIRS = (("logistic_F0", "markov_order1"), ("logistic_F0", "frequency_baseline"),
         ("markov_order1", "frequency_baseline"), ("logistic_F0_unpenalised", "markov_order1"),
         ("logistic_F0", "logistic_F0_unpenalised"), ("markov_order1_nested_alpha", "markov_order1"),
         ("logistic_F0", "markov_order1_nested_alpha"), ("markov_order2", "markov_order1"))


# ============================================================================ config / data
@dataclass(frozen=True)
class F0Config:
    path: Path
    raw: dict[str, Any]
    output_dir: Path

    @property
    def lr(self) -> dict:
        return self.raw["logistic"]


def load_f0_config(path: Path = PROJECT_ROOT / "configs" / "logistic_regression_f0.yaml") -> F0Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw["taxonomy"] != "scenarioD15" or raw["feature_set"] != "F0":
        raise ValueError("experiment 4 is defined for scenarioD15 and feature set F0 only")
    if raw["logistic"]["solver"] != "lbfgs" or raw["logistic"]["l1_ratio"] != 0.0:
        raise ValueError("experiment 4 uses lbfgs with an L2 penalty (l1_ratio = 0)")
    return F0Config(Path(path), raw, PROJECT_ROOT / raw["output_dir"])


def load_folds(path: Path, taxonomy: str) -> list[FoldAssignment]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))[taxonomy]
    return [FoldAssignment(int(f["fold_id"]), tuple(int(m) for m in f["train_match_ids"]),
                           tuple(int(m) for m in f["validation_match_ids"])) for f in payload]


@dataclass
class F0Data:
    data: TaskData
    folds: list[FoldAssignment]
    final_test: list[int]
    tier_matches: list[int]
    events: pd.DataFrame          # raw event table of the Tier B matches (for independent checks)


def load_f0_data(cfg: F0Config) -> F0Data:
    cfg2 = load_config_v2(PROJECT_ROOT / cfg.raw["v2_config"])
    task = cfg2.tasks[cfg.raw["taxonomy"]]
    assert_canonical_vocabulary(task.vocabulary)
    part = load_partition(cfg2)
    folds = load_folds(PROJECT_ROOT / cfg.raw["folds_file"], cfg.raw["taxonomy"])
    verify_folds(folds, part.tier_b)          # exact partition of Tier B, no train/validation overlap
    for f in folds:
        assert_no_final_test(f.train_match_ids + f.validation_match_ids, part.final_test)
    events = load_development_events(cfg2, part, part.tier_b)
    assert_no_final_test(events["match_id"].unique().tolist(), part.final_test)
    data = prepare_task(events, task)
    assert_canonical_vocabulary(data.vocab)
    return F0Data(data, folds, part.final_test, part.tier_b, events)


# ============================================================================ count statistics
def match_count_tensors(data: TaskData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(matches, train_counts, eval_counts): per match, the 15 x 15 counts of all
    within-sequence transitions (training population) and of evaluation rows only."""
    t = data.table
    matches = np.unique(t["match_id"].to_numpy())
    m_idx = np.searchsorted(matches, t["match_id"].to_numpy())
    cur, nxt = t["symbol_id"].to_numpy(), t["target_id"].to_numpy()
    has = nxt >= 0
    elig = t["eval_eligible"].to_numpy()

    def tensor(mask: np.ndarray) -> np.ndarray:
        flat = (m_idx[mask] * K + cur[mask]) * K + nxt[mask]
        return np.bincount(flat, minlength=len(matches) * K * K).reshape(len(matches), K, K).astype(float)

    return matches, tensor(has), tensor(elig)


def counts_for(matches: np.ndarray, tensor: np.ndarray, match_ids) -> np.ndarray:
    return tensor[np.isin(matches, list(match_ids))].sum(axis=0)


def table_log_loss(counts: np.ndarray, table: np.ndarray) -> float:
    """Pooled log loss of a (K, K) conditional table on (K, K) evaluation counts."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(counts > 0, -counts * np.log(table), 0.0)
    return float(terms.sum() / counts.sum())


def lr_from_counts(counts: np.ndarray, c_value: float, lr: dict) -> MultinomialLogisticModel:
    """Fit on the non-empty (current, next) cells with count weights (same objective as the rows)."""
    cur, nxt = np.nonzero(counts)
    x, y, w = one_hot(cur, K), nxt, counts[cur, nxt]
    model = MultinomialLogisticModel(K, c_value, int(lr["max_iter"]), float(lr["tol"]), float(lr["absent_class_floor"]))
    return model.fit(x, y, sample_weight=w)


def lr_table(model: MultinomialLogisticModel) -> np.ndarray:
    return model.predict_proba(np.eye(K))


def markov_table(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Additively smoothed first-order table; rows never seen fall back to the smoothed marginal
    (``MarkovChain``'s rule)."""
    table = normalise_rows(counts, alpha)
    marginal = (counts.sum(axis=0) + alpha) / (counts.sum() + K * alpha)
    unseen = counts.sum(axis=1) == 0
    table[unseen] = marginal
    return table


def nested_choice(matches, train_t, eval_t, folds, outer: FoldAssignment, grid, score_fn) -> tuple[Any, dict]:
    """Leave-one-inner-fold-out over the outer TRAINING folds. ``score_fn(train_counts,
    eval_counts, value)`` -> (log-loss sum, n). Ties -> the first grid value (grid order)."""
    inner = [f for f in folds if f.fold_id != outer.fold_id]
    scores = {}
    for value in grid:
        total, n = 0.0, 0.0
        for held in inner:
            tr = counts_for(matches, train_t, [m for f in inner if f is not held for m in f.validation_match_ids])
            ev = counts_for(matches, eval_t, held.validation_match_ids)
            s, k = score_fn(tr, ev, value)
            total, n = total + s, n + k
        scores[value] = total / n
    best = min(grid, key=lambda v: (round(scores[v], 12), grid.index(v)))
    return best, scores


# ============================================================================ cross-validation
@dataclass
class CVResult:
    proba: dict[str, np.ndarray]
    timings: list[dict]
    lr_fits: list[dict]
    choices: list[dict]
    coefficients: list[dict]
    theory: list[dict]
    c_path: list[dict]
    checks: dict


def run_cv(f0: F0Data, cfg: F0Config) -> CVResult:
    data, folds = f0.data, f0.folds
    lr, mk = cfg.lr, cfg.raw["markov"]
    table = data.table
    n_eval = len(data.eval_rows)
    elig = table["eval_eligible"].to_numpy()
    y_all = table["target_id"].to_numpy()
    matches, train_t, eval_t = match_count_tensors(data)
    c_grid = [float(c) for c in lr["C_grid"]]
    a_grid = [float(a) for a in mk["nested_alpha_grid"]]
    proba = {m: np.full((n_eval, K), np.nan) for m in PRIMARY_MODELS + SUPPLEMENTARY_MODELS}
    timings, lr_fits, choices, coefs, theory, c_path = [], [], [], [], [], []
    checks: dict[str, Any] = {"aggregated_vs_row_level_max_abs_probability_difference": {},
                              "determinism_fold0_max_abs_difference": None}

    def lr_score(tr, ev, c_value):
        return table_log_loss(ev, lr_table(lr_from_counts(tr, c_value, lr))) * ev.sum(), ev.sum()

    def mk_score(tr, ev, alpha):
        return table_log_loss(ev, markov_table(tr, alpha)) * ev.sum(), ev.sum()

    for fold in folds:
        LOGGER.info("fold %d: %d train / %d validation matches", fold.fold_id, len(fold.train_match_ids),
                    len(fold.validation_match_ids))
        if set(fold.train_match_ids) & set(fold.validation_match_ids):
            raise LeakageError("match in train and validation")
        train_ids = fold_seq_ids(data, fold.train_match_ids)
        val_ids = fold_seq_ids(data, fold.validation_match_ids)
        tr_sym, va_sym = arrays(data, train_ids, "symbol_id"), arrays(data, val_ids, "symbol_id")
        val_rows = np.concatenate([data.seq_rows[s] for s in val_ids])
        keep = elig[val_rows]
        val_eval_rows = val_rows[keep]
        pos = data.eval_pos[val_eval_rows]

        def put(name: str, p_rows: np.ndarray) -> None:
            proba[name][pos] = p_rows

        # ---------------- count-based baselines (sequence interface of markov.py)
        start = time.perf_counter()
        mc1 = MarkovChain(K, float(mk["alpha"])).fit(tr_sym)
        t_fit = time.perf_counter() - start
        start = time.perf_counter()
        p_mc1 = np.concatenate(mc1.predict_proba(va_sym))[keep]
        t_pred = time.perf_counter() - start
        put("markov_order1", p_mc1)
        timings.append({"fold_id": fold.fold_id, "model": "markov_order1", "fit_seconds": t_fit, "predict_seconds": t_pred})
        start = time.perf_counter()
        put("frequency_baseline", np.concatenate(mc1.fallback.predict_proba(va_sym))[keep])
        timings.append({"fold_id": fold.fold_id, "model": "frequency_baseline", "fit_seconds": t_fit,
                        "predict_seconds": time.perf_counter() - start})
        start = time.perf_counter()
        mc2 = SecondOrderMarkovChain(K, float(mk["alpha"]), int(mk["second_order_min_context_count"])).fit(tr_sym)
        t_fit = time.perf_counter() - start
        start = time.perf_counter()
        put("markov_order2", np.concatenate(mc2.predict_proba(va_sym))[keep])
        timings.append({"fold_id": fold.fold_id, "model": "markov_order2", "fit_seconds": t_fit,
                        "predict_seconds": time.perf_counter() - start})

        # ---------------- training counts and consistency with the row-level Markov fit
        tr_counts = counts_for(matches, train_t, fold.train_match_ids)
        if not np.array_equal(tr_counts, mc1.counts):
            raise RuntimeError(f"fold {fold.fold_id}: per-match count tensor differs from MarkovChain counts")

        # ---------------- Markov alpha (supplementary, nested)
        alpha, a_scores = nested_choice(matches, train_t, eval_t, folds, fold, a_grid, mk_score)
        choices.append({"fold_id": fold.fold_id, "model": "markov_order1_nested_alpha", "chosen": alpha,
                        **{f"inner_log_loss_{v:g}": s for v, s in a_scores.items()}})
        put("markov_order1_nested_alpha", np.concatenate(MarkovChain(K, alpha).fit(tr_sym).predict_proba(va_sym))[keep])

        # ---------------- logistic C (nested, aggregated fits)
        start = time.perf_counter()
        c_best, c_scores = nested_choice(matches, train_t, eval_t, folds, fold, c_grid, lr_score)
        search_seconds = time.perf_counter() - start
        choices.append({"fold_id": fold.fold_id, "model": "logistic_F0", "chosen": c_best,
                        "search_seconds": search_seconds, **{f"inner_log_loss_{v:g}": s for v, s in c_scores.items()}})

        # ---------------- reported logistic models: fitted on the individual training rows
        tr_frame = table[np.isin(table["match_id"].to_numpy(), fold.train_match_ids) & (y_all >= 0)]
        x_tr, names = design_matrix(tr_frame, REGISTRY["F0"])
        y_tr = tr_frame["target_id"].to_numpy()
        x_va, _ = design_matrix(table.iloc[val_eval_rows], REGISTRY["F0"])
        if len(y_tr) != int(tr_counts.sum()):
            raise RuntimeError("logistic training rows differ from the Markov transitions")
        # The primary model is fitted on the rows. The unpenalised reference (C = inf) has no
        # finite optimum when a transition cell is empty (logits diverge, lbfgs runs to max_iter),
        # so it is fitted on the count-weighted cells: the identical objective, ~1000x cheaper.
        for name, c_value, rows in (("logistic_F0", c_best, True), ("logistic_F0_unpenalised", np.inf, False)):
            if rows:
                model = MultinomialLogisticModel(K, c_value, int(lr["max_iter"]), float(lr["tol"]),
                                                 float(lr["absent_class_floor"]), name=name).fit(x_tr, y_tr)
            else:
                model = lr_from_counts(tr_counts, c_value, lr)
                model.name = name
            model.fit_info["fitted_on"] = "rows" if rows else "count-weighted (current, next) cells"
            start = time.perf_counter()
            p = model.predict_proba(x_va)
            t_pred = time.perf_counter() - start
            put(name, p)
            timings.append({"fold_id": fold.fold_id, "model": name, "fit_seconds": model.fit_info["fit_seconds"],
                            "predict_seconds": t_pred})
            lr_fits.append({"fold_id": fold.fold_id, "model": name, **{k: v for k, v in model.fit_info.items()
                                                                       if k != "absent_classes"},
                            "absent_classes": ";".join(SCENARIO_D15_CLASSES[c] for c in model.fit_info["absent_classes"])})
            w, b = model.coefficients(), model.intercepts()
            for ci, cname in enumerate(SCENARIO_D15_CLASSES):
                coefs.append({"fold_id": fold.fold_id, "model": name, "C": c_value, "target_class": cname,
                              "intercept": b[ci], **{n: w[ci, j] for j, n in enumerate(names)}})
            if rows:
                agg = lr_from_counts(tr_counts, c_value, lr)
                diff = float(np.abs(lr_table(agg) - lr_table(model)).max())
                checks["aggregated_vs_row_level_max_abs_probability_difference"][f"fold{fold.fold_id}_{name}"] = diff
                lr_sel = model
            else:
                lr_inf = model
        if fold.fold_id == folds[0].fold_id:  # determinism: refit with identical settings
            again = MultinomialLogisticModel(K, c_best, int(lr["max_iter"]), float(lr["tol"]),
                                             float(lr["absent_class_floor"])).fit(x_tr, y_tr)
            checks["determinism_fold0_max_abs_difference"] = float(np.abs(again.predict_proba(x_va) - proba["logistic_F0"][pos]).max())

        # ---------------- theory diagnostics on the TRAINING fold (no validation data)
        n_a = tr_counts.sum(axis=1)
        mle = normalise_rows(tr_counts, 0.0)
        t_sel, t_inf, t_mc = lr_table(lr_sel), lr_table(lr_inf), markov_table(tr_counts, float(mk["alpha"]))
        fitted_marginal = (n_a[:, None] * t_sel).sum(axis=0) / n_a.sum()
        empirical_marginal = tr_counts.sum(axis=0) / tr_counts.sum()
        nz = tr_counts > 0
        for a, cname in enumerate(SCENARIO_D15_CLASSES):
            theory.append({
                "fold_id": fold.fold_id, "current_event": cname, "n_train_transitions_from_state": int(n_a[a]),
                "n_zero_count_cells": int((tr_counts[a] == 0).sum()),
                "max_abs_lr_selected_minus_mle": float(np.abs(t_sel[a] - mle[a]).max()),
                "max_abs_lr_unpenalised_minus_mle": float(np.abs(t_inf[a] - mle[a]).max()),
                "max_abs_markov_alpha1_minus_mle": float(np.abs(t_mc[a] - mle[a]).max()),
                "max_abs_lr_selected_minus_markov_alpha1": float(np.abs(t_sel[a] - t_mc[a]).max()),
                "min_lr_unpenalised_probability_on_zero_cells": float(t_inf[a][~nz[a]].min()) if (~nz[a]).any() else np.nan,
                "min_markov_alpha1_probability_on_zero_cells": float(t_mc[a][~nz[a]].min()) if (~nz[a]).any() else np.nan,
            })
        checks.setdefault("intercept_marginal_identity_max_abs", {})[f"fold{fold.fold_id}"] = \
            float(np.abs(fitted_marginal - empirical_marginal).max())

        # ---------------- descriptive C path on the outer validation fold (NOT used for selection)
        va_counts = counts_for(matches, eval_t, fold.validation_match_ids)
        for c_value in c_grid + [np.inf]:
            m = lr_from_counts(tr_counts, c_value, lr)
            tab = lr_table(m)
            c_path.append({"fold_id": fold.fold_id, "C": c_value, "validation_log_loss": table_log_loss(va_counts, tab),
                           "train_log_loss": table_log_loss(tr_counts, tab),
                           "frobenius_norm_W": float(np.sqrt(np.nansum(m.coefficients() ** 2))),
                           "n_iter": m.fit_info["n_iter"], "converged": m.fit_info["converged"],
                           "p_shot_given_pass": float(tab[0, 5]), "p_shot_given_carry": float(tab[1, 5]),
                           "p_shot_given_50_50": float(tab[14, 5])})
    for name, p in proba.items():
        if np.isnan(p).any():
            raise RuntimeError(f"{name}: evaluation rows without prediction")
    return CVResult(proba, timings, lr_fits, choices, coefs, theory, c_path, checks)


# ============================================================================ evaluation
def evaluate(f0: F0Data, cv: CVResult, cfg: F0Config) -> dict[str, pd.DataFrame]:
    data, folds = f0.data, f0.folds
    ecfg = cfg.raw["evaluation"]
    t = data.table.iloc[data.eval_rows]
    y = t["target_id"].to_numpy()
    groups = t["match_id"].to_numpy()
    fold_of_match = {m: f.fold_id for f in folds for m in f.validation_match_ids}
    fold_of_eval = t["match_id"].map(fold_of_match).to_numpy()
    mult = bootstrap_multiplicities(len(np.unique(groups)), int(ecfg["bootstrap_replicates"]), int(ecfg["bootstrap_seed"]))
    shot = SCENARIO_D15_CLASSES.index(ecfg["shot_class"])
    summary, by_fold, per_class, confusion, stats = [], [], [], [], {}
    for name, p in cv.proba.items():
        dev = np.abs(p.sum(axis=1) - 1).max()
        if dev > float(ecfg["probability_tolerance"]) or (p < 0).any() or not np.isfinite(p).all():
            raise ValueError(f"{name}: invalid probability matrix (max row-sum deviation {dev})")
        d = p.argmax(axis=1)
        cm = confusion_counts(y, d, K)
        prob = probability_metrics(y, p)
        cal, _ = calibration(y, p, SCENARIO_D15_CLASSES)
        role = "primary" if name in PRIMARY_MODELS else "supplementary"
        row = {"model": name, "role": role, "n": len(y), **prob, **metrics_from_confusion(cm),
               "shot_average_precision": float(average_precision_score(y == shot, p[:, shot])),
               "n_classes_predicted": int((cm.sum(axis=0) > 0).sum()),
               "classwise_ece": cal["classwise_ece"], "top_label_ece": cal["top_label_ece"]}
        s = per_match_stats(y, p, {"argmax": d}, groups, K)
        stats[name] = s
        lo, hi = interval(bootstrap_summary(s, "argmax", mult))
        for i, key in enumerate(SUMMARY_KEYS):
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = float(lo[i]), float(hi[i])
        summary.append(row)
        for f in sorted(set(fold_of_eval)):
            sel = fold_of_eval == f
            by_fold.append({"model": name, "fold_id": int(f), "n": int(sel.sum()),
                            **probability_metrics(y[sel], p[sel]),
                            **metrics_from_confusion(confusion_counts(y[sel], d[sel], K))})
        pc = per_class_from_confusion(cm, SCENARIO_D15_CLASSES)
        pc.insert(0, "class_id", np.arange(K))
        pc.insert(0, "model", name)
        pc["average_precision_one_vs_rest"] = [float(average_precision_score(y == c, p[:, c])) for c in range(K)]
        pc["mean_predicted_probability"] = p.mean(axis=0)
        per_class.append(pc)
        cmf = pd.DataFrame(cm.astype(int), index=list(SCENARIO_D15_CLASSES), columns=list(SCENARIO_D15_CLASSES))
        cmf = cmf.rename_axis("true_class").reset_index().melt(id_vars="true_class", var_name="predicted_class",
                                                                  value_name="count")
        cmf.insert(0, "model", name)
        confusion.append(cmf)
    paired = []
    for model, ref in PAIRS:
        diff = bootstrap_summary(stats[model], "argmax", mult) - bootstrap_summary(stats[ref], "argmax", mult)
        lo, hi = interval(diff)
        mv = next(r for r in summary if r["model"] == model)
        rv = next(r for r in summary if r["model"] == ref)
        for i, key in enumerate(SUMMARY_KEYS):
            paired.append({"model": model, "reference": ref, "metric": key, "model_value": mv[key],
                           "reference_value": rv[key], "difference": mv[key] - rv[key],
                           "ci_low": float(lo[i]), "ci_high": float(hi[i]),
                           "ci_excludes_zero": bool(lo[i] > 0 or hi[i] < 0)})
    return {"model_comparison": pd.DataFrame(summary), "metrics_by_fold": pd.DataFrame(by_fold),
            "per_class_metrics": pd.concat(per_class, ignore_index=True),
            "confusion_matrices": pd.concat(confusion, ignore_index=True), "paired_differences": pd.DataFrame(paired)}


def class_distribution(f0: F0Data) -> pd.DataFrame:
    t = f0.data.table
    has = t["target_id"].to_numpy() >= 0
    elig = t["eval_eligible"].to_numpy()
    rows = []
    for c, name in enumerate(SCENARIO_D15_CLASSES):
        rows.append({"class_id": c, "class": name,
                     "n_next_event_all_transitions": int(((t["target_id"] == c) & has).sum()),
                     "n_next_event_eval_rows": int(((t["target_id"] == c) & elig).sum()),
                     "n_current_event_eval_rows": int(((t["symbol_id"] == c) & elig).sum())})
    out = pd.DataFrame(rows)
    for col in ("n_next_event_all_transitions", "n_next_event_eval_rows", "n_current_event_eval_rows"):
        out[col.replace("n_", "share_")] = out[col] / out[col].sum()
    return out


def leakage_and_integrity_checks(f0: F0Data) -> dict:
    """Independent checks of the rows, targets and partitions (no model involved)."""
    data, folds = f0.data, f0.folds
    t = data.table
    out: dict[str, Any] = {}
    loaded = set(int(m) for m in t["match_id"].unique())
    out["matches_loaded"] = len(loaded)
    out["final_test_matches_loaded"] = len(loaded & set(f0.final_test))
    out["matches_equal_tier_b"] = loaded == set(f0.tier_matches)
    out["folds_partition_tier_b_exactly_once"] = True  # verify_folds raised otherwise
    out["train_validation_overlap_per_fold"] = {f.fold_id: len(set(f.train_match_ids) & set(f.validation_match_ids))
                                                for f in folds}
    # target = class of the NEXT observation of the same sequence (independent recomputation)
    same_seq = t["sequence_id"].to_numpy()[1:] == t["sequence_id"].to_numpy()[:-1]
    nxt = np.where(same_seq, t["symbol_id"].to_numpy()[1:], -1)
    out["target_equals_next_observation_symbol"] = bool(np.array_equal(t["target_id"].to_numpy()[:-1], nxt)
                                                        and t["target_id"].iat[-1] == -1)
    later = t["list_position"].to_numpy()[1:] > t["list_position"].to_numpy()[:-1]
    out["target_strictly_later_in_raw_stream"] = bool(later[same_seq].all())
    # independent of tasks.py: from the RAW stream, exactly one vocabulary record of the same
    # period lies after E_t up to and including E_{t+1}, i.e. no vocabulary event is skipped
    raw = f0.events.sort_values(["match_id", "list_position"], kind="stable")
    raw = raw[~raw["period"].isin([5])]
    vocab_flag = raw["type"].isin(SCENARIO_D15_CLASSES).astype(int)
    cum = vocab_flag.groupby([raw["match_id"], raw["period"]]).cumsum()
    key = pd.Series(cum.to_numpy(), index=pd.MultiIndex.from_arrays([raw["match_id"], raw["list_position"]]))
    has_t = t["target_id"].to_numpy() >= 0
    cur_key = pd.MultiIndex.from_arrays([t["match_id"][has_t], t["list_position"][has_t]])
    tgt_lp = t["list_position"].shift(-1)[has_t].astype(int)
    tgt_key = pd.MultiIndex.from_arrays([t["match_id"][has_t], tgt_lp])
    gap = key.reindex(tgt_key).to_numpy() - key.reindex(cur_key).to_numpy()
    raw_type = raw.set_index("event_id")["type"]
    out["raw_stream_next_vocabulary_record_is_target"] = bool((gap == 1).all())
    out["raw_type_of_target_event_matches_target_class"] = bool(
        (raw_type.reindex(t.loc[has_t, "target_event_id"]).to_numpy() == t.loc[has_t, "target"].to_numpy()).all())
    out["sequences_cross_period_or_match"] = int((t.groupby("sequence_id")[["match_id", "period"]].nunique() > 1).any(axis=1).sum())
    out["penalty_shootout_rows"] = int((t["period"] == 5).sum())
    elig = t["eval_eligible"].to_numpy()
    out["eval_rows"] = int(elig.sum())
    out["eval_rows_without_360"] = int((elig & ~t["has_360"].to_numpy()).sum())
    out["eval_rows_without_target"] = int((elig & (t["target_id"].to_numpy() < 0)).sum())
    out["training_transitions_total"] = int((t["target_id"].to_numpy() >= 0).sum())
    out["training_transitions_without_360_current_event"] = int(((t["target_id"].to_numpy() >= 0) & ~t["has_360"].to_numpy()).sum())
    skipped = t.loc[elig, "n_skipped_raw_events"].astype(float)
    out["eval_rows_skipping_at_least_one_raw_record_share"] = float((skipped > 0).mean())
    out["eval_rows_mean_skipped_raw_records"] = float(skipped.mean())
    out["f0_input_columns"] = list(REGISTRY["F0"].input_columns)
    out["f0_input_columns_forbidden"] = sorted(set(REGISTRY["F0"].input_columns) & FORBIDDEN_COLUMNS)
    vocab_counts = np.bincount(t["symbol_id"].to_numpy(), minlength=K)
    out["all_classes_observed_as_current_event"] = bool((vocab_counts > 0).all())
    return out
