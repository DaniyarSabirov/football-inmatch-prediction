"""Experiment 4: multinomial logistic regression, current event only (F0), K = 15.

Runs the whole experiment on experiment 2's Tier B (304 development matches, 5 match folds)
and writes results/logistic_regression_f0_2026-11-17/. The final-test matches are never read.
An existing result directory is never overwritten (pass --overwrite deliberately).

Usage:
    .venv/bin/python scripts/run_logistic_f0.py            # ~10 minutes on an M1
    .venv/bin/python scripts/run_logistic_f0.py --overwrite
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

from src.sequence_models.event_classes import CLASS_TO_ID, K, SCENARIO_D15_CLASSES, TAXONOMY_NAME  # noqa: E402
from src.sequence_models.experiment import environment_info, sha256_file, write_json  # noqa: E402
from src.sequence_models.f0_experiment import (  # noqa: E402
    class_distribution,
    evaluate,
    leakage_and_integrity_checks,
    load_f0_config,
    load_f0_data,
    run_cv,
)

LOGGER = logging.getLogger("logistic_f0")
BLUE, GRAY, INK, INK2 = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e"


def verify_against_experiment2(summary: pd.DataFrame, cfg) -> dict:
    ref = cfg.raw["reproduce_experiment2"]
    tol = float(ref["tolerance"])
    got = summary.set_index("model")
    pairs = {"frequency_baseline": ref["frequency_baseline_log_loss"],
             "markov_order1": ref["exp1_markov_order1_log_loss"],
             "markov_order2": ref["exp1_markov_order2_log_loss"]}
    out = {"n_eval_rows": {"expected": ref["n_eval_rows"], "got": int(got["n"].iat[0])}}
    for model, expected in pairs.items():
        out[model] = {"expected_log_loss": expected, "got": float(got.at[model, "log_loss"]),
                      "abs_difference": abs(float(got.at[model, "log_loss"]) - expected)}
    out["all_match"] = bool(out["n_eval_rows"]["expected"] == out["n_eval_rows"]["got"]
                            and all(v["abs_difference"] <= tol for k, v in out.items() if isinstance(v, dict) and "abs_difference" in v))
    return out


def figures(out: Path, c_path: pd.DataFrame, coefs: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    # 1. regularisation path (mean over folds of the outer-validation log loss; descriptive only)
    path = c_path.groupby("C")["validation_log_loss"].mean()
    finite = path[np.isfinite(path.index)]
    mc1 = float(summary.set_index("model").at["markov_order1", "log_loss"])
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=200)
    ax.plot(finite.index, finite.values, color=BLUE, lw=2, marker="o", ms=5, label="logistic F0, validation log loss")
    ax.axhline(mc1, color=GRAY, lw=1.5, ls="--", label="first-order Markov (alpha = 1)")
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse L2 penalty strength), log scale")
    ax.set_ylabel("log loss (mean of 5 folds)")
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Regularisation path, Tier B, K = 15 (descriptive; C chosen by nested CV)", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "regularisation_path.png")
    plt.close(fig)
    # 2. coefficient matrix W (fold 0, selected C): rows = target class c, columns = current event
    sel = coefs[(coefs["fold_id"] == 0) & (coefs["model"] == "logistic_F0")]
    feat = [c for c in sel.columns if c.startswith("E_t=")]
    w = sel[feat].to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(w)))
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=200)
    im = ax.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(K), [f.removeprefix("E_t=") for f in feat], rotation=60, ha="right")
    ax.set_yticks(range(K), list(sel["target_class"]))
    ax.set_xlabel("current event E_t (one-hot feature)")
    ax.set_ylabel("next-event class c")
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("weight w_c[E_t] (logit contribution)")
    ax.set_title("Logistic F0 weights W (fold 0, selected C); red raises, blue lowers the logit", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "coefficients_fold0.png")
    plt.close(fig)


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Markdown table without optional dependencies (pandas.to_markdown needs tabulate)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}g}" if (abs(v) >= 1e4 or (0 < abs(v) < 1e-3)) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in frame.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def summary_markdown(out: Path, summary: pd.DataFrame, per_class: pd.DataFrame, choices: pd.DataFrame,
                     fits: pd.DataFrame, timings: pd.DataFrame, verification: dict) -> None:
    lines = ["# Experiment 4 — generated summary (do not edit; see README.md)", ""]
    cols = ["model", "role", "log_loss", "log_loss_ci_low", "log_loss_ci_high", "brier", "accuracy", "macro_f1",
            "weighted_f1", "balanced_accuracy", "shot_average_precision", "n_classes_predicted"]
    lines += ["## Pooled out-of-fold metrics (argmax decisions; 95% match-bootstrap CI for log loss)", "",
              md_table(summary[cols]), ""]
    shot = per_class[per_class["class"] == "Shot"][["model", "support", "n_predicted", "precision", "recall", "f1",
                                                    "average_precision_one_vs_rest"]]
    lines += ["## Shot", "", md_table(shot), ""]
    lines += ["## Hyperparameter choices (nested inner CV)", "",
              md_table(choices[["fold_id", "model", "chosen"]]), ""]
    lines += ["## Logistic fits", "", md_table(fits[["fold_id", "model", "C", "n_iter", "converged", "fit_seconds",
                                                     "n_train_rows"]]), ""]
    lines += ["## Mean timings per fold (seconds)", "",
              md_table(timings.groupby("model")[["fit_seconds", "predict_seconds"]].mean().reset_index()), ""]
    lines += ["## Reproduction of experiment 2 (same rows and folds)", "", "```json", json.dumps(verification, indent=1), "```"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    cfg = load_f0_config()
    out = cfg.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it deliberately")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.path, out / "experiment_config.yaml")

    f0 = load_f0_data(cfg)
    LOGGER.info("data: %d task rows, %d evaluation rows, %d matches", len(f0.data.table), len(f0.data.eval_rows),
                f0.data.table["match_id"].nunique())
    checks = leakage_and_integrity_checks(f0)
    class_distribution(f0).to_csv(out / "class_distribution.csv", index=False)
    write_json(out / "class_mapping.json", {"taxonomy": TAXONOMY_NAME, "K": K, "class_to_id": CLASS_TO_ID,
                                            "order": list(SCENARIO_D15_CLASSES),
                                            "source": "src/sequence_models/event_classes.py"})

    cv = run_cv(f0, cfg)
    tables = evaluate(f0, cv, cfg)
    summary = tables["model_comparison"]
    verification = verify_against_experiment2(summary, cfg)
    if not verification["all_match"]:
        write_json(out / "FAILED_reproduction_of_experiment2.json", verification)
        raise SystemExit("evaluation rows/folds do not reproduce experiment 2; see FAILED_reproduction_of_experiment2.json")
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame(cv.lr_fits).to_csv(out / "logistic_fits.csv", index=False)
    pd.DataFrame(cv.choices).to_csv(out / "hyperparameter_choices.csv", index=False)
    pd.DataFrame(cv.timings).to_csv(out / "timings.csv", index=False)
    pd.DataFrame(cv.coefficients).to_csv(out / "logistic_coefficients.csv", index=False)
    pd.DataFrame(cv.theory).to_csv(out / "theory_markov_equivalence.csv", index=False)
    pd.DataFrame(cv.c_path).to_csv(out / "regularisation_path.csv", index=False)
    write_json(out / "leakage_checks.json", checks)
    write_json(out / "reproducibility_checks.json", {**cv.checks, "experiment2_reproduction": verification})

    t = f0.data.table.iloc[f0.data.eval_rows]
    pred = t[["match_id", "event_id", "sequence_id", "seq_pos", "symbol", "target"]].reset_index(drop=True)
    for model in ("markov_order1", "logistic_F0"):
        for j, c in enumerate(SCENARIO_D15_CLASSES):
            pred[f"p__{model}__{c}"] = cv.proba[model][:, j].astype(np.float32)
    pred.to_parquet(out / "predictions_f0.parquet", compression="zstd", index=False)  # git-ignored

    figures(out, pd.DataFrame(cv.c_path), pd.DataFrame(cv.coefficients), summary)
    summary_markdown(out, summary, tables["per_class_metrics"], pd.DataFrame(cv.choices), pd.DataFrame(cv.lr_fits),
                     pd.DataFrame(cv.timings), verification)
    write_json(out / "environment.json", {**environment_info(), "runtime_seconds": time.time() - started})
    write_json(out / "checksums.json", {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*"))
                                        if p.is_file() and p.name != "checksums.json"})
    LOGGER.info("done in %.0f s -> %s", time.time() - started, out)


if __name__ == "__main__":
    main()
