"""Experiment 3: context-aware next-event prediction, taxonomy audit, rare-event diagnostics.

Stages (outputs in results/context_aware_event_prediction_v3/; development data only):

  taxonomy   audit of all raw types/subtypes, annotation pairs, same-timestamp records, event
             mapping, orientation and possession audits (334 development matches, descriptive)
  features   cached causal feature tables (Tier A: 3 taxonomies; Tier B: 3 taxonomies)
  diagnose   rare-event diagnostics on experiment 2's stored predictions (--tier tier_a|tier_b)
  screen     Tier A: every preregistered candidate, LR C grid, taxonomy A-vs-C, Shot costs/thresholds
  freeze     apply the preregistered selection rules -> frozen_plan.json (+ sha256); never overwritten
  confirm    Tier B: only the frozen plan
  report     tables, figures, runtime, checksums

Every configuration is checkpointed (resumable) and progress is logged with an ETA.
Usage: .venv/bin/python scripts/run_context_aware_v3.py <stage> [--tier tier_a|tier_b] [--overwrite]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_models.experiment import write_json  # noqa: E402
from src.sequence_models.v3_diagnostics import breakdown, class_diagnostics, per_match  # noqa: E402
from src.sequence_models.v3_experiment import (  # noqa: E402
    development_raw_table,
    evaluate,
    feature_table,
    load_config,
    partition_and_folds,
    pr_curve_points,
    threshold_for_budget,
)
from src.sequence_models.v3_features import causal_raw_features, orientation_audit, possession_audit  # noqa: E402
from src.sequence_models.v3_models import Checkpoint, MarkovSpec, Progress, cv_lr, nested_markov, spec_hash  # noqa: E402
from src.sequence_models.v3_taxonomy import (  # noqa: E402
    annotation_pairs,
    audit,
    event_mapping,
    taxonomy_spec,
)

LOGGER = logging.getLogger("context_aware_v3")
TAXONOMIES = ("filtered8", "scenarioD15", "actions15")


def runtime_log(out: Path, stage: str, seconds: float, extra: dict | None = None) -> None:
    path = out / "runtime_log.json"
    log = json.loads(path.read_text()) if path.exists() else []
    log.append({"stage": stage, "seconds": round(seconds, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})})
    write_json(path, log)


def vocab_of(task: str) -> list[str]:
    return list(taxonomy_spec(task).vocabulary)


# ------------------------------------------------------------------ taxonomy
def stage_taxonomy(cfg: dict, out: Path) -> None:
    part, _ = partition_and_folds(cfg)
    raw = development_raw_table(cfg, part)
    d = out / "taxonomy"
    d.mkdir(parents=True, exist_ok=True)
    summary, skipped, dist = audit(raw, ["filtered8", "scenarioD15", "actions15", "raw33"])
    summary.to_csv(d / "taxonomy_audit.csv", index=False)
    skipped.to_csv(d / "skipped_records.csv", index=False)
    dist.to_csv(d / "class_distribution.csv", index=False)
    annotation_pairs(raw).to_csv(d / "annotation_pairs.csv", index=False)
    event_mapping(raw).to_csv(d / "event_mapping.csv", index=False)
    p = raw[raw["period"] < 5]
    raw_types = p.groupby(["type", "subtype"], dropna=False).size().rename("n").reset_index()
    raw_types.to_csv(d / "raw_types_and_subtypes.csv", index=False)
    # same-timestamp audit: raw adjacent pairs sharing a timestamp, by (type -> next type)
    g = p.groupby(["match_id", "period"], sort=False)
    same = (g["t_seconds"].shift(-1) == p["t_seconds"])
    pairs = pd.DataFrame({"first": p["type"], "next": g["type"].shift(-1), "same": same}).dropna()
    st = pairs.groupby(["first", "next"]).agg(n=("same", "size"), same_timestamp_share=("same", "mean")).reset_index()
    st = st[st["n"] >= 200].sort_values("n", ascending=False)
    st.to_csv(d / "same_timestamp_audit.csv", index=False)
    causal = causal_raw_features(p.reset_index(drop=True))
    write_json(d / "orientation_audit.json", orientation_audit(raw))
    write_json(d / "possession_audit.json", possession_audit(p.reset_index(drop=True), causal))
    LOGGER.info("taxonomy audit written to %s", d)


# ------------------------------------------------------------------ features
def stage_features(cfg: dict, out: Path) -> None:
    part, _ = partition_and_folds(cfg)
    raw = development_raw_table(cfg, part)
    for tier, matches in (("tier_a", part.tier_a), ("tier_b", part.tier_b)):
        for task in TAXONOMIES:
            start = time.time()
            tab = feature_table(cfg, raw, tier, task, matches)
            LOGGER.info("features %s %s: %d rows, %d evaluation rows (%.0fs)", tier, task, len(tab),
                        int(tab["eval_eligible"].sum()), time.time() - start)


def load_features(cfg: dict, tier: str, task: str) -> pd.DataFrame:
    path = cfg["_cache"] / f"features_{tier}_{task}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}: run the 'features' stage first")
    return pd.read_parquet(path)


# ------------------------------------------------------------------ diagnostics
def stage_diagnose(cfg: dict, out: Path, tier: str) -> None:
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    d = out / "diagnostics" / tier
    d.mkdir(parents=True, exist_ok=True)
    data = ds.dataset(cfg["_exp2"] / "predictions.parquet")
    rare = cfg["evaluation"]["rare_classes"]
    all_summary, all_instead, all_thr, all_break, all_match, pr = [], [], [], [], [], []
    for task in ("scenarioD15", "filtered8"):
        vocab = vocab_of(task)
        feats = load_features(cfg, tier, task)
        feats = feats[feats["eval_eligible"]][["event_id", "z3", "d7", "danger_area", "prev1", "cur", "poss_dur4",
                                               "poss_prog4", "minute7", "match_id"]]
        for model in ("mc2_zone", "mc1", "hmm_event"):
            cols = ["event_id", "true_next_event"] + [f"p__{model}__{c}" for c in vocab] + \
                   [f"decision__{model}__{s}" for s in ("uniform", "moderate")]
            tab = data.to_table(columns=cols, filter=(pc.field("tier") == tier) & (pc.field("task") == task)).to_pandas()
            tab = tab.merge(feats, on="event_id", how="inner", validate="one_to_one")
            y = tab["true_next_event"].map({c: i for i, c in enumerate(vocab)}).to_numpy()
            p = tab[[f"p__{model}__{c}" for c in vocab]].to_numpy(dtype=float)
            p = p / p.sum(axis=1, keepdims=True)
            dec = {s: tab[f"decision__{model}__{s}"].astype(str).map({c: i for i, c in enumerate(vocab)}).to_numpy()
                   for s in ("uniform", "moderate")}
            classes = [c for c in rare if c in vocab]
            s, inst, thr = class_diagnostics(y, p, dec, vocab, classes)
            for frame in (s, inst, thr):
                frame.insert(0, "model", model)
                frame.insert(0, "task", task)
            all_summary.append(s)
            all_instead.append(inst)
            all_thr.append(thr)
            if model == "mc2_zone":
                shot = vocab.index("Shot")
                tab["prev_event"] = np.where(tab["prev1"] < len(vocab), np.array(vocab + ["START", "NEWPOSS"])[tab["prev1"]], "START")
                tab["current_event"] = np.array(vocab)[tab["cur"]]
                tab["zone_x_possession_duration"] = tab["z3"].astype(str) + "|" + tab["poss_dur4"].astype(str)
                for by in ("z3", "d7", "danger_area", "prev_event", "current_event", "poss_dur4", "poss_prog4",
                           "minute7", "zone_x_possession_duration"):
                    for rule in ("uniform", "moderate"):
                        b = breakdown(y, p, dec[rule], tab, shot, by)
                        b.insert(0, "decision_rule", rule)
                        b.insert(0, "task", task)
                        all_break.append(b)
                for rule in ("uniform", "moderate"):
                    pm = per_match(y, p, dec[rule], tab["match_id"].to_numpy(), shot)
                    pm.insert(0, "decision_rule", rule)
                    pm.insert(0, "task", task)
                    all_match.append(pm)
            curve = pr_curve_points(y, p, vocab.index("Shot"))
            curve.insert(0, "model", model)
            curve.insert(0, "task", task)
            pr.append(curve)
    pd.concat(all_summary).to_csv(d / "rare_event_class_diagnostics.csv", index=False)
    pd.concat(all_instead).to_csv(d / "rare_event_predicted_instead.csv", index=False)
    pd.concat(all_thr).to_csv(d / "rare_event_thresholds.csv", index=False)
    pd.concat(all_break).to_csv(d / "shot_breakdowns.csv", index=False)
    pd.concat(all_match).to_csv(d / "shot_per_match.csv", index=False)
    pd.concat(pr).to_csv(d / "shot_pr_curves_experiment2_models.csv", index=False)
    LOGGER.info("diagnostics written to %s", d)


# ------------------------------------------------------------------ model runs
def run_models(cfg: dict, tier: str, task: str, table: pd.DataFrame, folds, specs: list[MarkovSpec],
               lr_c: list[float], ckpt: Checkpoint) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    vocab = vocab_of(task)
    grid = cfg["markov"]["concentration_grid"]
    total = len(specs) + len(lr_c)
    prog = Progress(total, f"{tier}/{task}")
    proba, choices, lr_info = {}, [], []
    fold_key = [[f.fold_id, list(f.validation_match_ids)] for f in folds]
    for spec in specs:
        key = spec_hash({"spec": spec.to_dict(), "grid": grid, "folds": fold_key, "rows": len(table), "tier": tier,
                         "task": task})
        hit = ckpt.load(f"{tier}_{task}_{spec.name}", key)
        if hit is None:
            start = time.time()
            p, ch, models = nested_markov(spec, table, folds, len(vocab), grid, cfg["markov"]["base_alpha"])
            meta = {"choices": ch, "seconds": time.time() - start, "spec": spec.to_dict()}
            ckpt.save(f"{tier}_{task}_{spec.name}", key, p, meta)
            (ckpt.root / f"{tier}_{task}_{spec.name}__{key}_models.json").write_text(json.dumps(models))
            hit = (p, meta)
        proba[spec.name] = hit[0]
        for c in hit[1]["choices"]:
            choices.append({"tier": tier, "task": task, **c, "fit_seconds_total": hit[1]["seconds"]})
        prog.step(spec.name)
    for c in lr_c:
        name = f"logistic_C{c}"
        key = spec_hash({"lr": c, "folds": fold_key, "rows": len(table), "tier": tier, "task": task, "v": 1})
        hit = ckpt.load(f"{tier}_{task}_{name}", key)
        if hit is None:
            start = time.time()
            p, info = cv_lr(table, folds, len(vocab), c)
            hit = (p, {"info": info, "seconds": time.time() - start})
            ckpt.save(f"{tier}_{task}_{name}", key, *hit)
        proba[name] = hit[0]
        for i in hit[1]["info"]:
            lr_info.append({"tier": tier, "task": task, "model": name, **i})
        prog.step(name)
    return proba, pd.DataFrame(choices), pd.DataFrame(lr_info)


def specs_from(cfg: dict) -> list[MarkovSpec]:
    return [MarkovSpec(n, tuple(c["features"]), bool(c.get("reset", False)), c["group"]) for n, c in cfg["candidates"].items()]


def tier_thresholds(y: np.ndarray, proba: dict, shot: int, budgets: list[float]) -> dict:
    return {m: {b: threshold_for_budget(y, p[:, shot], shot, b) for b in budgets} for m, p in proba.items()}


def write_eval(d: Path, res: dict) -> None:
    for k, v in res.items():
        path = d / f"{k}.csv"
        if path.exists():
            v = pd.concat([pd.read_csv(path), v], ignore_index=True)
        v.to_csv(path, index=False)


def stage_screen(cfg: dict, out: Path) -> None:
    start = time.time()
    _, folds = partition_and_folds(cfg)
    d = out / "tier_a"
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"{d} exists; remove it deliberately to re-screen")
    d.mkdir(parents=True)
    ckpt = Checkpoint(cfg["_cache"] / "checkpoints")
    task = cfg["main_taxonomy"]
    table = load_features(cfg, "tier_a", task)
    specs = specs_from(cfg) + [MarkovSpec("mc1", ("cur",), group="baseline")]
    proba, choices, lr_info = run_models(cfg, "tier_a", task, table, folds["tier_a"], specs,
                                         cfg["logistic_regression"]["C_grid"], ckpt)
    vocab = vocab_of(task)
    y = table.loc[table["eval_eligible"], "target_id"].to_numpy()
    thr = tier_thresholds(y, proba, vocab.index("Shot"), cfg["shot_threshold"]["fp_per_1000_budgets"])
    res = evaluate("tier_a", task, table, proba, folds["tier_a"], vocab, cfg, "reference", thresholds=thr)
    write_eval(d, res)
    choices.to_csv(d / "markov_concentration_choices.csv", index=False)
    lr_info.to_csv(d / "logistic_regression_fits.csv", index=False)
    write_json(d / "shot_thresholds_from_tier_a.json", {m: {str(b): t for b, t in v.items()} for m, v in thr.items()})
    curves = []
    for m in ("reference", "mc1"):
        c = pr_curve_points(y, proba[m], vocab.index("Shot"))
        c.insert(0, "model", m)
        curves.append(c)
    pd.concat(curves).to_csv(d / "shot_pr_curves.csv", index=False)
    # taxonomy option A (Scenario D as is) vs B/C (actions15) with the reference model, Tier A
    alt = load_features(cfg, "tier_a", "actions15")
    ref_spec = [s for s in specs if s.name == "reference"]
    p_alt, ch_alt, _ = run_models(cfg, "tier_a", "actions15", alt, folds["tier_a"], ref_spec, [], ckpt)
    res_alt = evaluate("tier_a", "actions15", alt, p_alt, folds["tier_a"], vocab, cfg, "reference")
    for k in ("summary", "per_class", "calibration"):
        res_alt[k].to_csv(d / f"taxonomy_option_actions15_{k}.csv", index=False)
    runtime_log(out, "screen", time.time() - start, {"models": len(proba)})


def stage_freeze(cfg: dict, out: Path) -> None:
    plan_path = out / "frozen_plan.json"
    if plan_path.exists():
        raise SystemExit("frozen_plan.json exists and is never overwritten")
    d = out / "tier_a"
    summ = pd.read_csv(d / "summary.csv")
    paired = pd.read_csv(d / "paired.csv")
    u = summ[summ["decision_rule"] == "uniform"].set_index("model")
    specs = {s.name: s for s in specs_from(cfg)}
    groups: dict[str, list[str]] = {}
    for s in specs.values():
        groups.setdefault(s.group, []).append(s.name)
    groups["spatial"].append("reference")
    groups["history"].append("reference")
    best, included, rationale = {}, {}, []
    pll = paired[paired["metric"] == "log_loss"].set_index("model")
    for g in ("spatial", "history", "possession", "temporal"):
        names = groups[g]
        b = min(names, key=lambda n: (u.at[n, "log_loss"], n))
        best[g] = b
        if b == "reference":
            included[g] = False
            gain = 0.0
        else:
            included[g] = bool(pll.at[b, "ci_high"] < 0)
            gain = float(pll.at[b, "difference"])
        rationale.append({"group": g, "best": b, "tier_a_log_loss": float(u.at[b, "log_loss"]),
                          "delta_vs_reference": gain, "ci_high": float(pll.at[b, "ci_high"]) if b != "reference" else None,
                          "included_in_combination": included[g]})
    # combination
    spatial_feature = specs[best["spatial"]].features[1] if included["spatial"] else "z3"
    feats = ["cur", spatial_feature, "prev1"]
    extras = []
    for g in ("possession", "temporal"):
        if included[g]:
            extras.append((float(pll.at[best[g], "difference"]), specs[best[g]].features[-1]))
    hist_extra = []
    reset = False
    if included["history"]:
        hspec = specs[best["history"]]
        hist_extra = [f for f in hspec.features if f in ("prev2", "prev3")]
        reset = hspec.reset_history_at_possession
        extras.append((float(pll.at[best["history"], "difference"]), tuple(hist_extra)))
    for _, f in sorted(extras, key=lambda e: e[0]):  # most negative difference (largest gain) first
        feats += list(f) if isinstance(f, tuple) else [f]
    combo = MarkovSpec("combination", tuple(feats), reset, "combination")
    lr_rows = summ[(summ["decision_rule"] == "uniform") & summ["model"].str.startswith("logistic_C")].set_index("model")
    lr_best = min(lr_rows.index, key=lambda n: (lr_rows.at[n, "log_loss"], n))
    shot_rows = summ[summ["decision_rule"].str.startswith("moderate_shot_x")]
    lam = {}
    for m, g in shot_rows.groupby("model"):
        g = g.assign(lam=g["decision_rule"].str.replace("moderate_shot_x", "").astype(float))
        lam[m] = float(g.sort_values(["shot_f1", "lam"], ascending=[False, True]).iloc[0]["lam"])
    thr = json.loads((d / "shot_thresholds_from_tier_a.json").read_text())
    spatial_names = [n for n in groups["spatial"] if n != "reference"]
    shot_spatial = max(spatial_names, key=lambda n: (u.at[n, "shot_average_precision"], n))
    rationale.append({"group": "secondary_shot_ranking", "best": shot_spatial,
                      "tier_a_shot_average_precision": float(u.at[shot_spatial, "shot_average_precision"]),
                      "reference_shot_average_precision": float(u.at["reference", "shot_average_precision"]),
                      "included_in_combination": False})
    tier_b_markov = ["reference", "mc1"] + sorted({best[g] for g in best} - {"reference"} | {shot_spatial})
    plan = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "results/context_aware_event_prediction_v3/tier_a (out-of-fold, experiment 1 pilot folds)",
        "rules": cfg["selection_rules"], "group_selection": rationale,
        "tier_b_markov_models": {n: (specs[n].to_dict() if n in specs else {"name": "mc1", "features": ["cur"],
                                                                              "reset": False, "group": "baseline"})
                                 for n in tier_b_markov},
        "combination": combo.to_dict(), "logistic_regression_C": float(lr_best.replace("logistic_C", "")),
        "shot_lambda_star_by_model": lam, "shot_thresholds_from_tier_a": thr,
        "separate_taxonomy": {"name": cfg["separate_taxonomy"], "models": ["reference", "combination"]},
    }
    write_json(plan_path, plan)
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    (out / "frozen_plan.sha256").write_text(digest + "\n")
    LOGGER.info("plan frozen: %s (sha256 %s)", plan_path, digest[:12])


def load_plan(out: Path) -> dict:
    plan_path = out / "frozen_plan.json"
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if digest != (out / "frozen_plan.sha256").read_text().strip():
        raise SystemExit("frozen_plan.json was modified after freezing")
    return json.loads(plan_path.read_text())


def stage_confirm(cfg: dict, out: Path) -> None:
    start = time.time()
    plan = load_plan(out)
    part, folds = partition_and_folds(cfg)
    d = out / "tier_b"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("summary.csv", "per_class.csv", "paired.csv", "calibration.csv", "shot_thresholds.csv"):
        if (d / name).exists():
            raise SystemExit(f"{d / name} exists; Tier B is evaluated once (remove deliberately to rerun)")
    ckpt = Checkpoint(cfg["_cache"] / "checkpoints")
    spec_of = lambda s: MarkovSpec(s["name"], tuple(s["features"]), bool(s["reset"]), s["group"])  # noqa: E731
    specs = [spec_of(s) for s in plan["tier_b_markov_models"].values()] + [spec_of(plan["combination"])]
    for task, names in ((cfg["main_taxonomy"], None), (plan["separate_taxonomy"]["name"], plan["separate_taxonomy"]["models"])):
        table = load_features(cfg, "tier_b", task)
        assert not set(table["match_id"]) & set(part.final_test)
        use = specs if names is None else [s for s in specs if s.name in names]
        lr = [plan["logistic_regression_C"]] if names is None else []
        vocab = vocab_of(task)
        # Shot thresholds always come from TIER A out-of-fold predictions. Models without a screening
        # run on this taxonomy (the rule-defined combination; every actions15 model) are run on Tier A first.
        known = plan["shot_thresholds_from_tier_a"] if task == cfg["main_taxonomy"] else {}
        thr = {m: {float(b): t for b, t in v.items()} for m, v in known.items()}
        missing = [sp for sp in use if sp.name not in thr]
        if missing:
            ta = load_features(cfg, "tier_a", task)
            p_a, _, _ = run_models(cfg, "tier_a", task, ta, folds["tier_a"], missing, [], ckpt)
            y_a = ta.loc[ta["eval_eligible"], "target_id"].to_numpy()
            thr.update(tier_thresholds(y_a, p_a, vocab.index("Shot"), cfg["shot_threshold"]["fp_per_1000_budgets"]))
            write_json(d / f"shot_thresholds_from_tier_a_extra_{task}.json",
                       {m: {str(b): t for b, t in thr[m].items()} for m in p_a})
        proba, choices, lr_info = run_models(cfg, "tier_b", task, table, folds["tier_b"], use, lr, ckpt)
        res = evaluate("tier_b", task, table, proba, folds["tier_b"], vocab, cfg, "reference", thresholds=thr)
        write_eval(d, res)
        choices.to_csv(d / f"markov_concentration_choices_{task}.csv", index=False)
        if len(lr_info):
            lr_info.to_csv(d / f"logistic_regression_fits_{task}.csv", index=False)
        y = table.loc[table["eval_eligible"], "target_id"].to_numpy()
        curves = []
        for m in proba:
            c = pr_curve_points(y, proba[m], vocab.index("Shot"))
            c.insert(0, "model", m)
            c.insert(0, "task", task)
            curves.append(c)
        pd.concat(curves).to_csv(d / f"shot_pr_curves_{task}.csv", index=False)
        # rare-event diagnostics of the new models (same functions as the Tier A diagnostics)
        diag = []
        for m in proba:
            dec = {"uniform": proba[m].argmax(axis=1)}
            s, _, _ = class_diagnostics(y, proba[m], dec, vocab, [c for c in cfg["evaluation"]["rare_classes"] if c in vocab])
            s.insert(0, "model", m)
            s.insert(0, "task", task)
            diag.append(s)
        pd.concat(diag).to_csv(d / f"rare_event_class_diagnostics_{task}.csv", index=False)
        np.savez_compressed(cfg["_cache"] / f"tier_b_{task}_eval_proba.npz", **proba)
    runtime_log(out, "confirm", time.time() - start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["taxonomy", "features", "diagnose", "screen", "freeze", "confirm", "report"])
    parser.add_argument("--tier", default="tier_a", choices=["tier_a", "tier_b"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    out = cfg["_output"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_config.yaml").write_text(Path(cfg["_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    start = time.time()
    if args.stage == "taxonomy":
        stage_taxonomy(cfg, out)
    elif args.stage == "features":
        stage_features(cfg, out)
    elif args.stage == "diagnose":
        if args.tier == "tier_b" and not (out / "frozen_plan.json").exists():
            raise SystemExit("Tier B diagnostics run only after the plan is frozen")
        stage_diagnose(cfg, out, args.tier)
    elif args.stage == "screen":
        stage_screen(cfg, out)
    elif args.stage == "freeze":
        stage_freeze(cfg, out)
    elif args.stage == "confirm":
        stage_confirm(cfg, out)
    else:
        from src.sequence_models.v3_report import stage_report
        stage_report(cfg, out)
    if args.stage not in ("screen", "confirm"):
        runtime_log(out, args.stage + (f"_{args.tier}" if args.stage == "diagnose" else ""), time.time() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
