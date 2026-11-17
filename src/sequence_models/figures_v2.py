"""Figures for experiment 2 (static PNG, matplotlib).

Titles state the tier and whether a figure is an OUT-OF-FOLD EVALUATION or
DESCRIPTIVE. Colours follow experiment 1 (``figures.py``): fixed categorical
order, one-hue sequential ramp. No freeze frame is drawn, so no visible-area
polygon applies; zones are relative to the acting team (attacking left ->
right).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .figures import CATEGORICAL, GRID, NEUTRAL, SEQUENTIAL, SURFACE, TEXT, TEXT_2, _save, heatmap, plt

SCHEME_COLOURS = {"uniform": NEUTRAL, "moderate": CATEGORICAL[0], "strong": CATEGORICAL[1], "tuned": CATEGORICAL[2]}
SCHEMES = ["uniform", "moderate", "strong", "tuned"]


def _style(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def class_distribution(dist: pd.DataFrame, path: Path) -> None:
    taxonomies = list(dict.fromkeys(dist["taxonomy"]))
    fig, axes = plt.subplots(1, len(taxonomies), figsize=(11, 5.2), gridspec_kw={"width_ratios": [8, 15]})
    for ax, tax in zip(axes, taxonomies):
        d = dist[dist["taxonomy"] == tax].sort_values("share")
        colours = [CATEGORICAL[1] if r else CATEGORICAL[0] for r in d["rare_lt_5pct"]]
        ax.barh(d["class"], d["share"] * 100, color=colours, height=0.7)
        for y, (s, n) in enumerate(zip(d["share"], d["evaluation_rows_as_target"])):
            ax.text(s * 100 * 1.15, y, f"{s:.2%} (n={n:,})", va="center", fontsize=7, color=TEXT_2)
        ax.set_xscale("log")
        ax.set_xlim(0.08, 400)
        ax.set_xlabel("share of evaluation rows as target, % (log scale)")
        ax.set_title(f"{tax} ({len(d)} classes)", loc="left", fontsize=9)
        _style(ax, "x")
    fig.suptitle("DESCRIPTIVE - target class distribution, 334 development matches (orange: rare, < 5%)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, path)


def miss_costs(defs: dict, path: Path) -> None:
    """Miss cost m_k per class and scheme (Tier B, fold 0 priors)."""
    tasks = list(defs)
    fig, axes = plt.subplots(len(tasks), 1, figsize=(11, 3.2 * len(tasks)))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        vocab = defs[task]["vocabulary"]
        fold0 = defs[task]["folds"]["fold0"]
        schemes = [s for s in SCHEMES if s in fold0]
        width = 0.8 / len(schemes)
        x = np.arange(len(vocab))
        for i, s in enumerate(schemes):
            ax.bar(x + (i - (len(schemes) - 1) / 2) * width, fold0[s]["miss_costs"], width * 0.92,
                   color=SCHEME_COLOURS[s], label=s)
        ax.set_xticks(x, vocab, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("miss cost m_k\n(units: cost of missing the majority class)")
        ax.set_title(task, loc="left", fontsize=9)
        ax.legend(ncol=len(schemes))
        _style(ax)
    fig.suptitle("Cost schemes - cost of missing each class, C(a,k) = m_k for a != k (Tier B fold 0 training priors)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, path)


def rare_recall(rare: pd.DataFrame, model_of: dict[str, str], tier: str, path: Path) -> None:
    tasks = list(model_of)
    fig, axes = plt.subplots(len(tasks), 1, figsize=(11, 3.4 * len(tasks)))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        r = rare[(rare["tier"] == tier) & (rare["task"] == task) & (rare["model"] == model_of[task])]
        classes = list(dict.fromkeys(r["class"]))
        schemes = [s for s in SCHEMES if s in set(r["cost_scheme"])]
        width = 0.8 / len(schemes)
        x = np.arange(len(classes))
        for i, s in enumerate(schemes):
            q = r[r["cost_scheme"] == s].set_index("class").loc[classes]
            pos = x + (i - (len(schemes) - 1) / 2) * width
            ax.bar(pos, q["recall"], width * 0.92, color=SCHEME_COLOURS[s], label=s)
            ax.errorbar(pos, q["recall"], yerr=[q["recall"] - q["recall_ci_low"], q["recall_ci_high"] - q["recall"]],
                        fmt="none", ecolor=TEXT, elinewidth=0.8, capsize=2)
        ax.set_xticks(x, classes, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("recall (out-of-fold)")
        ax.set_ylim(0, 1)
        ax.set_title(f"{task} - {model_of[task]}", loc="left", fontsize=9)
        ax.legend(ncol=len(schemes))
        _style(ax)
    fig.suptitle(f"OUT-OF-FOLD EVALUATION ({tier}) - rare-event recall by cost scheme (95% match-bootstrap CI)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, path)


def precision_recall_tradeoff(rare: pd.DataFrame, model_of: dict[str, str], tier: str, path: Path) -> None:
    """Small multiples: one panel per (taxonomy, rare class); points = cost schemes."""
    panels = [(task, c) for task in model_of for c in dict.fromkeys(
        rare[(rare["tier"] == tier) & (rare["task"] == task) & (rare["model"] == model_of[task])]["class"])]
    ncol = 6
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.5 * nrow), squeeze=False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax, (task, c) in zip(axes.ravel(), panels):
        q = rare[(rare["tier"] == tier) & (rare["task"] == task) & (rare["model"] == model_of[task]) & (rare["class"] == c)]
        q = q.set_index("cost_scheme").loc[[s for s in SCHEMES if s in set(q["cost_scheme"])]]
        ax.plot(q["recall"], q["precision"], color=NEUTRAL, linewidth=1, zorder=1)
        for sname, row in q.iterrows():
            ax.plot(row["recall"], row["precision"], "o", color=SCHEME_COLOURS[sname], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1, zorder=2, label=sname)
        ax.axhline(q["prevalence"].iloc[0], color=TEXT_2, linewidth=0.7, linestyle=":")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{c} ({'8' if task == 'filtered8' else '15'} cl.)", loc="left", fontsize=8)
        ax.tick_params(labelsize=6)
        _style(ax, "both")
    axes[0, 0].legend(fontsize=6, loc="upper right")
    fig.supxlabel("recall", fontsize=8, color=TEXT_2)
    fig.supylabel("precision (dotted: prevalence = precision of random guessing)", fontsize=8, color=TEXT_2)
    fig.suptitle(f"OUT-OF-FOLD EVALUATION ({tier}) - precision/recall of rare classes under each cost scheme "
                 f"({', '.join(f'{t}: {m}' for t, m in model_of.items())})", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    _save(fig, path)


def confusion(cm_long: pd.DataFrame, vocab: list[str], title: str, path: Path) -> None:
    cm = cm_long.pivot(index="true_class", columns="predicted_class", values="count").loc[vocab, vocab].to_numpy()
    heatmap(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), vocab, vocab, title, "predicted class",
            "true next event", path, cbar_label="share of true class")


def reliability(rel: pd.DataFrame, models: list[str], classes: list[str], tier: str, task: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(classes), figsize=(3.2 * len(classes), 3.4), sharey=False)
    for ax, c in zip(np.atleast_1d(axes), classes):
        ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=0.8, linestyle="--")
        for i, m in enumerate(models):
            q = rel[(rel["tier"] == tier) & (rel["task"] == task) & (rel["model"] == m) & (rel["class"] == c) & (rel["n"] > 0)]
            ax.plot(q["mean_predicted"], q["observed_frequency"], color=CATEGORICAL[i], marker="o", markersize=4,
                    linewidth=1.5, label=m)
        ax.set_title(c, loc="left", fontsize=9)
        ax.set_xlabel("mean predicted probability (bin)")
        ax.set_ylabel("observed frequency")
        q_all = rel[(rel["tier"] == tier) & (rel["task"] == task) & (rel["class"] == c) & (rel["n"] > 0)
                    & rel["model"].isin(models)]
        lim = max(0.05, float(q_all[["mean_predicted", "observed_frequency"]].max().max()) * 1.1)
        ax.set_xlim(0, min(1, lim))
        ax.set_ylim(0, min(1, lim))
        _style(ax, "both")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(f"OUT-OF-FOLD EVALUATION ({tier}, {task}) - reliability (10 equal-width bins; diagonal = calibrated)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, path)


def interval_comparison(frame: pd.DataFrame, label_col: str, value: str, low: str, high: str, xlabel: str,
                        title: str, path: Path, zero_line: bool = False) -> None:
    groups = list(dict.fromkeys(frame["panel"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups) + 2.4, 0.42 * frame.groupby("panel").size().max() + 1.6),
                             sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, g in zip(np.atleast_1d(axes), groups):
        f = frame[frame["panel"] == g].reset_index(drop=True)
        for i, row in f.iterrows():
            ax.plot([row[low], row[high]], [i, i], color=CATEGORICAL[0], linewidth=2, solid_capstyle="round")
            ax.plot(row[value], i, "o", color=CATEGORICAL[0], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.text(row[value], i - 0.22, f"{row[value]:.4f}", fontsize=6.5, color=TEXT_2, ha="center")
        if zero_line:
            ax.axvline(0, color=NEUTRAL, linewidth=0.8)
        ax.set_yticks(range(len(f)), f[label_col], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(g, loc="left", fontsize=9)
        _style(ax, "x")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.03, "Point: pooled out-of-fold estimate. Line: 95% match-level bootstrap interval.",
             fontsize=8, color=TEXT_2)
    _save(fig, path)


def k_sensitivity(log: pd.DataFrame, path: Path) -> None:
    tasks = list(dict.fromkeys(log["task"]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.6 * len(tasks), 3.6))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        sel = log[(log["task"] == task) & log["selected_init"]]
        for i, (e, g) in enumerate(sel.groupby("emission")):
            mean = g.groupby("n_states")["val_log_loss"].mean()
            sd = g.groupby("n_states")["val_log_loss"].std()
            ax.errorbar(mean.index, mean, yerr=sd, color=CATEGORICAL[i], marker="o", markersize=5, linewidth=1.8,
                        capsize=3, label=e)
        ax.set_xlabel("number of hidden states K")
        ax.set_ylabel("validation log loss (fold mean ± SD)")
        ax.set_title(task, loc="left", fontsize=9)
        ax.set_xticks(sorted(sel["n_states"].unique()))
        ax.legend()
        _style(ax, "both")
    fig.suptitle("OUT-OF-FOLD EVALUATION (Tier A) - HMM sensitivity to K (init with best training objective)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, path)


def state_zones(states: pd.DataFrame, path: Path) -> None:
    labels = list(dict.fromkeys(states["task"] + " | " + states["model"]))
    n_rows = states.groupby(["task", "model"]).size().max()
    fig, axes = plt.subplots(len(labels), 1, figsize=(11, (0.24 * n_rows + 0.9) * len(labels)))
    fig.subplots_adjust(hspace=0.55)
    zones = ["Defensive third", "Middle third", "Attacking third"]
    for ax, lab in zip(np.atleast_1d(axes), labels):
        task, model = lab.split(" | ")
        s = states[(states["task"] == task) & (states["model"] == model)].reset_index(drop=True)
        left = np.zeros(len(s))
        for i, z in enumerate(zones):
            vals = s[f"observed_share_{z}"].to_numpy()
            ax.barh(range(len(s)), vals, left=left, color=[CATEGORICAL[0], CATEGORICAL[2], CATEGORICAL[1]][i],
                    height=0.7, label=z if lab == labels[0] else None, edgecolor=SURFACE, linewidth=1)
            left += vals
        ax.set_yticks(range(len(s)), [f"{st}: {top}" for st, top in zip(s["state"], s["top_event_emissions"])], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(lab, loc="left", fontsize=9)
    np.atleast_1d(axes)[0].legend(ncol=3, loc="lower right", fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("filtered-posterior-weighted share of observations in each pitch third")
    fig.suptitle("DESCRIPTIVE (Tier B fold 0 validation matches) - hidden states: pitch thirds and top event emissions",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, path)
