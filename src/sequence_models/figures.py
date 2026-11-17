"""Static figures for the Markov/HMM pilot (matplotlib, PNG).

Every figure states in its title whether it is DESCRIPTIVE (parameters or
states fitted on all pilot matches; not a performance estimate) or an
OUT-OF-FOLD EVALUATION (cross-validated validation predictions).
Colours: fixed categorical order (never cycled); one-hue sequential ramp
for heatmaps. Pitch plots use the acting team's attacking frame (attacking
left -> right, x = 0 own goal line, 120 opponent goal line), which is the
frame of the spatial features; no freeze frame is drawn, so no visible-area
polygon applies.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8986"
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "font.size": 9,
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def heatmap(matrix: np.ndarray, rows: list[str], cols: list[str], title: str, xlabel: str, ylabel: str,
            path: Path, fmt: str = "{:.2f}", cbar_label: str = "probability") -> None:
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(cols), 1.0 + 0.5 * len(rows)))
    im = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=max(float(np.nanmax(matrix)), 1e-12), aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    vmax = np.nanmax(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > 0.55 * vmax else TEXT)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)
    _save(fig, path)


def emission_bars(emission: np.ndarray, vocab: list[str], title: str, path: Path) -> None:
    k = emission.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(vocab)), 3.4))
    width = 0.8 / k
    x = np.arange(len(vocab))
    for s in range(k):
        ax.bar(x + (s - (k - 1) / 2) * width, emission[s], width * 0.92, color=CATEGORICAL[s], label=f"State {s}")
    ax.set_xticks(x, vocab, rotation=30, ha="right")
    ax.set_ylabel("P(event type | state)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=k, loc="upper right")
    ax.set_title(title, loc="left")
    _save(fig, path)


def metric_comparison(boot: pd.DataFrame, models: list[str], labels: dict[str, str], title: str, path: Path) -> None:
    metrics = [("macro_f1", "Macro-F1 (higher better)"), ("balanced_accuracy", "Balanced accuracy (higher better)"),
               ("log_loss", "Log loss, nats (lower better)"), ("accuracy", "Accuracy (higher better)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 0.45 * len(models) + 1.4), sharey=True)
    b = boot[boot["comparison"] == "absolute"]
    for ax, (metric, label) in zip(axes, metrics):
        for i, m in enumerate(models):
            row = b[(b["model"] == m) & (b["metric"] == metric)].iloc[0]
            ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=CATEGORICAL[0], linewidth=2, solid_capstyle="round")
            ax.plot(row["estimate"], i, "o", color=CATEGORICAL[0], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.text(row["ci_high"], i + 0.28, f"{row['estimate']:.3f}", fontsize=7, color=TEXT_2, ha="right")
        ax.set_yticks(range(len(models)), [labels[m] for m in models])
        ax.set_xlabel(label)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.04, "Point: pooled out-of-fold estimate. Line: 95% match-level bootstrap interval.",
             fontsize=8, color=TEXT_2)
    _save(fig, path)


def per_class_f1(per_class: pd.DataFrame, models: list[str], labels: dict[str, str], vocab: list[str],
                 title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(vocab)), 3.6))
    width = 0.8 / len(models)
    x = np.arange(len(vocab))
    for i, m in enumerate(models):
        f1 = per_class[per_class["model"] == m].set_index("class").loc[vocab, "f1"].to_numpy()
        ax.bar(x + (i - (len(models) - 1) / 2) * width, f1, width * 0.92, color=CATEGORICAL[i], label=labels[m])
    support = per_class[per_class["model"] == models[0]].set_index("class").loc[vocab, "support"]
    ax.set_xticks(x, [f"{c}\n(n={support[c]:,})" for c in vocab], rotation=0, fontsize=7)
    ax.set_ylabel("F1 (out-of-fold)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(models), loc="upper right")
    ax.set_title(title, loc="left")
    _save(fig, path)


def k_sensitivity(fit_log: pd.DataFrame, title: str, path: Path) -> None:
    variants = sorted(fit_log["variant"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for vi, variant in enumerate(variants):
        f = fit_log[fit_log["variant"] == variant]
        sel = f[f["selected_seed"]]
        mean = sel.groupby("n_states")["val_log_loss"].mean()
        sd = sel.groupby("n_states")["val_log_loss"].std()
        axes[0].errorbar(mean.index + (vi - 0.5) * 0.08, mean, yerr=sd, color=CATEGORICAL[vi], marker="o",
                         linewidth=2, markersize=6, capsize=3, label=variant)
        tr = sel.groupby("n_states")["train_log_likelihood_per_obs"].mean()
        axes[1].plot(tr.index, tr, color=CATEGORICAL[vi], marker="o", linewidth=2, markersize=6, label=variant)
    axes[0].set_xlabel("number of hidden states K")
    axes[0].set_ylabel("validation log loss (fold mean ± SD)")
    axes[1].set_xlabel("number of hidden states K")
    axes[1].set_ylabel("training log-likelihood per observation")
    for ax in axes:
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend()
        ax.set_xticks(sorted(fit_log["n_states"].unique()))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, path)


def seed_stability(fit_log: pd.DataFrame, title: str, path: Path) -> None:
    variants = sorted(fit_log["variant"].unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(4.5 * len(variants), 3.4), squeeze=False)
    for ax, variant in zip(axes[0], variants):
        f = fit_log[fit_log["variant"] == variant]
        for i, k in enumerate(sorted(f["n_states"].unique())):
            g = f[f["n_states"] == k]
            jitter = (g["seed"].to_numpy() - 2) * 0.03
            ax.scatter(g["fold_id"] + jitter + (i - 1.5) * 0.18, g["val_log_loss"], s=14, color=CATEGORICAL[i],
                       label=f"K={k}", edgecolor=SURFACE, linewidth=0.5)
        ax.set_xlabel("fold")
        ax.set_ylabel("validation log loss (every seed)")
        ax.set_title(variant, loc="left", fontsize=9)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, path)


def _pitch(ax) -> None:
    ax.add_patch(Rectangle((0, 0), 120, 40, fill=False, edgecolor=NEUTRAL, linewidth=1))
    ax.plot([60, 60], [0, 40], color=NEUTRAL, linewidth=0.8)
    ax.add_patch(Rectangle((102, 0), 18, 22, fill=False, edgecolor=NEUTRAL, linewidth=0.8))
    ax.add_patch(Rectangle((0, 0), 18, 22, fill=False, edgecolor=NEUTRAL, linewidth=0.8))
    ax.set_xlim(-2, 122)
    ax.set_ylim(-2, 42)
    ax.set_aspect("equal")


def spatial_states(points: pd.DataFrame, means: np.ndarray, sds: np.ndarray, title: str, path: Path) -> None:
    """points: x_attacking, lateral_distance_from_centre, state (filtered argmax)."""
    k = means.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(3.6 * k, 2.9), squeeze=False)
    for s, ax in enumerate(axes[0]):
        _pitch(ax)
        p = points[points["state"] == s]
        ax.scatter(p["x_attacking"], p["lateral_distance_from_centre"], s=1.5, alpha=0.15, color=CATEGORICAL[s],
                   rasterized=True)
        for n_sd, ls in ((1, "-"), (2, "--")):
            ax.add_patch(Ellipse((means[s, 0], means[s, 1]), 2 * n_sd * sds[s, 0], 2 * n_sd * sds[s, 1],
                                 fill=False, edgecolor=TEXT, linewidth=1, linestyle=ls))
        ax.set_title(f"State {s} (n={len(p):,})", loc="left", fontsize=9)
    fig.supxlabel("x in the acting team's attacking frame (pitch units; 0 = own goal line, 120 = opponent goal line)",
                  fontsize=8, color=TEXT_2)
    fig.supylabel("|y - 40|  (0 = centre, 40 = touchline)", fontsize=8, color=TEXT_2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.1, "Dots: actor points, coloured by the most probable FILTERED state. Ellipses: fitted "
             "Gaussian emission (1 and 2 SD).", fontsize=8, color=TEXT_2)
    _save(fig, path)


def state_timeline(filtered: np.ndarray, viterbi: np.ndarray, symbols: list[str], vocab: list[str], title: str,
                   path: Path, window: int = 250) -> None:
    """First ``window`` observations of one sequence (filtering is causal, so the
    window's posteriors are identical to those of the full sequence)."""
    filtered, viterbi, symbols = filtered[:window], viterbi[:window], symbols[:window]
    t = np.arange(len(filtered))
    k = filtered.shape[1]
    fig, axes = plt.subplots(3, 1, figsize=(11, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 0.5, 1.6]})
    axes[0].stackplot(t, filtered.T, colors=CATEGORICAL[:k], labels=[f"State {s}" for s in range(k)], linewidth=0)
    axes[0].set_ylabel("filtered P(state | events so far)")
    axes[0].set_ylim(0, 1)
    axes[0].legend(ncol=k, loc="upper right", fontsize=7)
    axes[1].imshow(viterbi[None, :], aspect="auto", cmap=matplotlib.colors.ListedColormap(CATEGORICAL[:k]),
                   vmin=-0.5, vmax=k - 0.5, interpolation="nearest", extent=(-0.5, len(t) - 0.5, 0, 1))
    axes[1].set_yticks([])
    axes[1].set_ylabel("Viterbi\n(retro.)", rotation=0, ha="right", va="center", fontsize=7)
    idx = [vocab.index(s) for s in symbols]
    axes[2].scatter(t, idx, s=8, color=TEXT_2)
    axes[2].set_yticks(range(len(vocab)), vocab, fontsize=7)
    axes[2].set_xlabel("observation number in the sequence (event order)")
    axes[2].set_ylabel("event")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.03, "Top: causal filtered posterior (what the predictor uses). Middle: Viterbi path of the "
             "whole period - uses future events, for description only, never for evaluation.",
             fontsize=8, color=TEXT_2)
    _save(fig, path)
