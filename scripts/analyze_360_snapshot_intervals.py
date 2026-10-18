#!/usr/bin/env python3
"""Analyze the temporal resolution of StatsBomb 360 snapshots.

Answers one question: how frequently does StatsBomb provide a 360 freeze-frame
snapshot, across the entire local raw dataset (all matches under
data/raw/statsbomb/data/three-sixty/)?

Data sources (see also the "Data integrity" printout at the top of the run):
  - data/raw/statsbomb/data/three-sixty/{match_id}.json
        One record per snapshot: {event_uuid, freeze_frame, visible_area}.
        This file has NO timestamp of its own -- a snapshot's time comes from
        the event it is attached to.
  - data/raw/statsbomb/data/events/{match_id}.json
        event["id"] == three-sixty record's "event_uuid" is the join key
        (exactly the join already used throughout src/animate_statsbomb_360_match.py
        and src/build_statsbomb_360_dataset.py -- reused here, not
        reimplemented). event["timestamp"] ("HH:MM:SS.ffffff") and
        event["period"] give the snapshot's time.

Critical data-integrity facts, verified directly against the raw data before
writing this script (see the printed "Data integrity checks" section):
  - event["timestamp"] is PERIOD-RELATIVE, not match-absolute: period 2 (and
    3/4 for matches with extra time) restarts near 00:00:00, it does not
    continue from where period 1 left off. Consecutive-interval differencing
    is therefore computed independently within each (match_id, period) group,
    in chronological order (StatsBomb's own event "index" order within a
    period, which is already timestamp-ordered) -- never across a period
    boundary, and never across two different matches.
  - Multiple rows CAN share the same event_uuid's timestamp: StatsBomb
    sometimes logs two or three events (belonging to different teams) at the
    literal same instant (interval == 0.000s) -- e.g. a Ball Receipt*
    immediately followed by the opposing team's Ball Recovery/Interception.
    This is a genuine, previously-investigated StatsBomb data characteristic
    (see src/team_identity.py), not a bug in this script or in the join.

No existing processed file already contains a "time to next 360 snapshot"
column for the full 426-match corpus (the closest existing artifact,
data/processed/statsbomb_360_match_*_next_events.csv, is a different,
narrower thing: "time to next MEANINGFUL EVENT of specific types" for a
handful of already-processed matches, not "time to the next 360 snapshot" for
every match). This script computes it directly from the join above, reusing
existing loading utilities (`load_and_join_match`, `timestamp_seconds`,
`integer_event_field`) rather than re-parsing JSON by hand.

Outputs (all under results/):
  statsbomb_360_interval_distribution_all.png/.pdf       (Variant A: includes 0.000s)
  statsbomb_360_interval_distribution_filtered.png/.pdf  (Variant B: 0.0001s < interval < 60s)
  statsbomb_360_intervals_per_match_boxplot.png/.pdf     (distribution of valid intervals per match)
  statsbomb_360_interval_per_match_statistics.csv        (one row per match, both variants)

Snapshots vs. intervals per match: NOT simply "snapshots = intervals + 1".
That identity only holds for a single, unbroken chronological sequence. Since
intervals are deliberately computed WITHIN each period separately (see above),
a match with P periods loses one interval per period boundary, so the true
relationship is `snapshots = intervals + P` (modulo the rare non-chronological
row excluded in a few matches, logged as a warning and accounted for
explicitly below). This is verified against the data, not assumed -- see the
printed "Snapshot/interval relationship check".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from animate_statsbomb_360_match import (  # noqa: E402
    VisualizationError,
    available_360_match_ids,
    integer_event_field,
    load_and_join_match,
    timestamp_seconds,
)

RESULTS_DIR = PROJECT_ROOT / "results"

# Variant B thresholds, exactly as specified.
MIN_VALID_INTERVAL = 0.0001   # excludes the same-instant duplicated-snapshot rows
MAX_VALID_INTERVAL = 60.0     # excludes rare, non-representative long gaps

# Ordered bin edges. Inspecting the raw distribution first (see the printed
# "bin coverage" check) confirms the overwhelming majority of intervals are
# well under 5 seconds, so bins are kept at 1-second resolution through 5s
# and coarser beyond that, exactly where the data itself has little mass --
# this is discovered from the data, not assumed up front.
BIN_EDGES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 60.0]
BIN_LABELS = ["0-1s", "1-2s", "2-3s", "3-4s", "4-5s", "5-10s", "10-20s", "20-30s", "30-60s"]

PRIMARY_BLUE = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID_COLOR = "#e3e2dd"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.titleweight": "bold",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


@dataclass
class MatchIntervals:
    match_id: int
    intervals: np.ndarray  # seconds, one per consecutive pair within a period, all periods concatenated
    total_snapshots: int   # true count of usable 360 frames, counted directly -- NOT derived from len(intervals)
    n_periods: int
    n_negative_excluded: int  # non-chronological rows excluded (see the warning printed when found)


def compute_all_match_intervals() -> tuple[list[MatchIntervals], list[int]]:
    """Load every available match and compute consecutive-snapshot intervals
    strictly within each (match_id, period) group.

    Returns (per-match interval arrays, list of match_ids that failed to load).
    """
    match_ids = available_360_match_ids()
    results: list[MatchIntervals] = []
    failed: list[int] = []

    for match_id in match_ids:
        try:
            match = load_and_join_match(match_id)
        except VisualizationError as error:
            print(f"  [match {match_id}] skipped (load error): {error}", file=sys.stderr)
            failed.append(match_id)
            continue

        periods = sorted({integer_event_field(joined.event, "period") for joined in match.usable_frames})
        match_intervals: list[float] = []
        n_negative_excluded = 0
        for period in periods:
            # usable_frames is already chronologically sorted (see
            # load_and_join_match's chronological_key); filtering to one
            # period preserves that order.
            times = [
                timestamp_seconds(joined.event.get("timestamp"))
                for joined in match.usable_frames
                if integer_event_field(joined.event, "period") == period
            ]
            diffs = np.diff(np.asarray(times, dtype=float))
            negative = diffs[diffs < 0]
            if len(negative):
                print(
                    f"  [match {match_id}, period {period}] WARNING: {len(negative)} negative "
                    f"interval(s) found (non-chronological timestamps) -- excluded from analysis",
                    file=sys.stderr,
                )
                diffs = diffs[diffs >= 0]
                n_negative_excluded += len(negative)
            match_intervals.extend(diffs.tolist())

        results.append(
            MatchIntervals(
                match_id=match_id,
                intervals=np.asarray(match_intervals, dtype=float),
                total_snapshots=len(match.usable_frames),
                n_periods=len(periods),
                n_negative_excluded=n_negative_excluded,
            )
        )

    return results, failed


def bin_counts(intervals: np.ndarray) -> np.ndarray:
    """Count intervals into BIN_EDGES; anything >= last edge is dropped (the
    caller is expected to have already applied MAX_VALID_INTERVAL for
    Variant B, or to accept that Variant A's histogram intentionally caps
    its last bin at 60s and reports overflow separately)."""
    counts, _ = np.histogram(intervals, bins=BIN_EDGES)
    return counts


def summarize(intervals: np.ndarray) -> dict:
    if len(intervals) == 0:
        return {key: np.nan for key in (
            "n", "mean", "median", "std", "min", "max", "p25", "p75", "p90", "p95", "p99",
            "pct_below_1s", "pct_below_2s", "pct_below_3s", "pct_below_5s", "pct_below_10s",
        )}
    return {
        "n": len(intervals),
        "mean": float(np.mean(intervals)),
        "median": float(np.median(intervals)),
        "std": float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0,
        "min": float(np.min(intervals)),
        "max": float(np.max(intervals)),
        "p25": float(np.percentile(intervals, 25)),
        "p75": float(np.percentile(intervals, 75)),
        "p90": float(np.percentile(intervals, 90)),
        "p95": float(np.percentile(intervals, 95)),
        "p99": float(np.percentile(intervals, 99)),
        "pct_below_1s": float(100 * np.mean(intervals < 1.0)),
        "pct_below_2s": float(100 * np.mean(intervals < 2.0)),
        "pct_below_3s": float(100 * np.mean(intervals < 3.0)),
        "pct_below_5s": float(100 * np.mean(intervals < 5.0)),
        "pct_below_10s": float(100 * np.mean(intervals < 10.0)),
    }


def build_per_match_table(per_match: list[MatchIntervals]) -> pd.DataFrame:
    rows = []
    for m in per_match:
        all_intervals = m.intervals
        filtered = all_intervals[(all_intervals > MIN_VALID_INTERVAL) & (all_intervals < MAX_VALID_INTERVAL)]

        # total_snapshots is counted directly from the loaded match, NOT
        # derived as len(intervals) + 1 -- that identity only holds for a
        # single unbroken period. With intervals computed within each period
        # separately, the correct relationship is
        # snapshots == intervals + n_periods + n_negative_excluded (see the
        # "Snapshot/interval relationship check" printed by main()).
        row: dict = {
            "match_id": m.match_id,
            "total_snapshots": m.total_snapshots,
            "n_periods": m.n_periods,
            "n_negative_excluded": m.n_negative_excluded,
        }

        counts_all = bin_counts(all_intervals)
        counts_filtered = bin_counts(filtered)
        n_all = len(all_intervals)
        n_filtered = len(filtered)

        row["n_intervals_all"] = n_all
        row["n_zero_intervals"] = int(np.sum(all_intervals <= MIN_VALID_INTERVAL))
        for label, count in zip(BIN_LABELS, counts_all):
            row[f"all_{label}_n"] = int(count)
            row[f"all_{label}_pct"] = float(100 * count / n_all) if n_all else np.nan
        stats_all = summarize(all_intervals)
        for key, value in stats_all.items():
            row[f"all_{key}"] = value

        row["n_intervals_filtered"] = n_filtered
        for label, count in zip(BIN_LABELS, counts_filtered):
            row[f"filtered_{label}_n"] = int(count)
            row[f"filtered_{label}_pct"] = float(100 * count / n_filtered) if n_filtered else np.nan
        stats_filtered = summarize(filtered)
        for key, value in stats_filtered.items():
            row[f"filtered_{key}"] = value

        rows.append(row)
    return pd.DataFrame(rows)


def plot_global_histogram(intervals: np.ndarray, title: str, subtitle: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    counts, edges = np.histogram(intervals, bins=BIN_EDGES)
    total = counts.sum()
    x_positions = np.arange(len(BIN_LABELS))

    bars = ax.bar(x_positions, counts, color=PRIMARY_BLUE, width=0.72, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.set_ylim(0, counts.max() * 1.18)

    for x, count in zip(x_positions, counts):
        pct = 100 * count / total if total else 0
        ax.text(
            x, count + total * 0.012, f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8.5, color=TEXT_SECONDARY,
        )

    mean_val = np.mean(intervals)
    median_val = np.median(intervals)

    def interval_to_x(value: float) -> float:
        # Map a raw-second value onto the categorical bin x-axis by linear
        # interpolation within whichever bin it falls in.
        for i in range(len(BIN_EDGES) - 1):
            if BIN_EDGES[i] <= value < BIN_EDGES[i + 1] or (i == len(BIN_EDGES) - 2 and value >= BIN_EDGES[i + 1]):
                span = BIN_EDGES[i + 1] - BIN_EDGES[i]
                frac = (value - BIN_EDGES[i]) / span if span else 0
                return i - 0.5 + frac
        return len(BIN_LABELS) - 1

    ax.axvline(interval_to_x(mean_val), color=TEXT_PRIMARY, linestyle="--", linewidth=1.4, zorder=4)
    ax.axvline(interval_to_x(median_val), color="#e34948", linestyle="--", linewidth=1.4, zorder=4)
    # A small fixed-position stats box (rather than text pinned to the line's
    # own height) avoids colliding with the per-bar count/percentage labels,
    # whose heights vary with the data.
    stats_box = dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=GRID_COLOR, linewidth=1.0)
    ax.text(0.985, 0.975, f"— mean = {mean_val:.2f}s", transform=ax.transAxes, fontsize=9.5,
            va="top", ha="right", color=TEXT_PRIMARY, fontweight="bold", bbox=stats_box)
    ax.text(0.985, 0.895, f"— median = {median_val:.2f}s", transform=ax.transAxes, fontsize=9.5,
            va="top", ha="right", color="#e34948", fontweight="bold", bbox=stats_box)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Time interval to next 360 snapshot (seconds)", fontsize=11)
    ax.set_ylabel("Number of snapshots", fontsize=11)
    ax.set_title(title, fontsize=14, pad=28)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY, ha="left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def compute_boxplot_outliers(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standard 1.5*IQR rule. Returns (outlier values, lower_fence, upper_fence)."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = values[(values < lower_fence) | (values > upper_fence)]
    return outliers, lower_fence, upper_fence


def plot_intervals_per_match_boxplot(per_match_valid_counts: np.ndarray, output_path: Path) -> None:
    """One horizontal boxplot: the distribution, across matches, of each
    match's TOTAL number of valid (Variant B) 360 snapshot intervals.

    Exactly one distribution -- no interval-duration bins, no per-match bars,
    no percentages. Each match contributes exactly one number.
    """
    values = per_match_valid_counts
    n = len(values)
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    std_val = float(np.std(values, ddof=1))
    p10, p90 = np.percentile(values, [10, 90])
    outliers, lower_fence, upper_fence = compute_boxplot_outliers(values)

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    box = ax.boxplot(
        values, vert=False, widths=0.5, patch_artist=True,
        boxprops={"facecolor": "#cde2fb", "edgecolor": PRIMARY_BLUE, "linewidth": 1.6},
        medianprops={"color": "#e34948", "linewidth": 2.0},
        whiskerprops={"color": TEXT_PRIMARY, "linewidth": 1.3},
        capprops={"color": TEXT_PRIMARY, "linewidth": 1.3},
        flierprops={
            "marker": "o", "markerfacecolor": "none", "markeredgecolor": "#e34948",
            "markersize": 5, "linewidth": 1.0, "alpha": 0.8,
        },
    )

    ax.set_yticks([])
    ax.set_xlabel("Number of valid 360 snapshot intervals per match", fontsize=11)
    ax.set_title("Distribution of valid StatsBomb 360 snapshot intervals per match", fontsize=14, pad=34)
    ax.text(
        0.0, 1.10,
        f"One value per match ({MIN_VALID_INTERVAL}s < interval < {MAX_VALID_INTERVAL:.0f}s, Variant B) -- "
        f"n = {n} matches",
        transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY, ha="left",
    )

    stats_lines = (
        f"n = {n} matches\n"
        f"mean = {mean_val:,.0f}\n"
        f"median = {median_val:,.0f}\n"
        f"Q1 = {q1:,.0f}   Q3 = {q3:,.0f}\n"
        f"IQR = {iqr:,.0f}\n"
        f"min = {values.min():,.0f}   max = {values.max():,.0f}\n"
        f"outliers (1.5×IQR) = {len(outliers)}"
    )
    ax.text(
        0.985, 0.94, stats_lines, transform=ax.transAxes, fontsize=9.2,
        va="top", ha="right", color=TEXT_PRIMARY,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": SURFACE, "edgecolor": GRID_COLOR, "linewidth": 1.0},
    )

    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def print_overall_summary(label: str, intervals: np.ndarray, n_matches: int) -> None:
    stats = summarize(intervals)
    print(f"--- Overall summary: {label} ---")
    print(f"  matches: {n_matches}")
    print(f"  intervals (n): {stats['n']:,}")
    print(f"  mean={stats['mean']:.3f}s  median={stats['median']:.3f}s  std={stats['std']:.3f}s")
    print(f"  min={stats['min']:.3f}s  max={stats['max']:.3f}s")
    print(f"  p25={stats['p25']:.3f}  p75={stats['p75']:.3f}  p90={stats['p90']:.3f}  "
          f"p95={stats['p95']:.3f}  p99={stats['p99']:.3f}")
    print(f"  % < 1s: {stats['pct_below_1s']:.1f}   % < 2s: {stats['pct_below_2s']:.1f}   "
          f"% < 3s: {stats['pct_below_3s']:.1f}   % < 5s: {stats['pct_below_5s']:.1f}   "
          f"% < 10s: {stats['pct_below_10s']:.1f}")
    print()


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Data integrity checks")
    print("=" * 70)
    print("Snapshot source: data/raw/statsbomb/data/three-sixty/{match_id}.json")
    print("  columns: event_uuid, freeze_frame, visible_area (no timestamp)")
    print("Timestamp source: data/raw/statsbomb/data/events/{match_id}.json")
    print("  join key: event_uuid == event['id']; time columns: timestamp, period")
    print("Match ID: three-sixty filename stem (== events filename stem)")
    print("Timestamps are PERIOD-RELATIVE (verified: period 2 restarts near 0s) --")
    print("  intervals are computed within (match_id, period) groups only, never")
    print("  across a period boundary and never across two different matches.")
    print()

    print("Loading all matches and computing consecutive-snapshot intervals...")
    per_match, failed = compute_all_match_intervals()
    print(f"Loaded {len(per_match)} matches successfully; {len(failed)} skipped (load errors): {failed}")
    print()

    all_intervals = np.concatenate([m.intervals for m in per_match]) if per_match else np.array([])
    filtered_intervals = all_intervals[(all_intervals > MIN_VALID_INTERVAL) & (all_intervals < MAX_VALID_INTERVAL)]

    zero_count = int(np.sum(all_intervals <= MIN_VALID_INTERVAL))
    outlier_count = int(np.sum(all_intervals >= MAX_VALID_INTERVAL))
    print("=" * 70)
    print("Zero-interval / outlier handling")
    print("=" * 70)
    print(f"Variant A (all intervals): n={len(all_intervals):,}")
    print(f"  of which interval <= {MIN_VALID_INTERVAL}s (duplicated/same-instant): "
          f"{zero_count:,} ({100*zero_count/len(all_intervals):.2f}%)")
    print(f"Variant B (filtered): {MIN_VALID_INTERVAL}s < interval < {MAX_VALID_INTERVAL:.0f}s -> n={len(filtered_intervals):,}")
    print(f"  excluded as duplicated/same-instant (<= {MIN_VALID_INTERVAL}s): {zero_count:,}")
    print(f"  excluded as outliers (>= {MAX_VALID_INTERVAL:.0f}s): {outlier_count:,}")
    print()

    print("=" * 70)
    print("Bin coverage check (Variant B) -- confirms the chosen bin widths")
    print("=" * 70)
    counts_check = bin_counts(filtered_intervals)
    for label, count in zip(BIN_LABELS, counts_check):
        print(f"  {label:<8} {count:>8,}  ({100*count/len(filtered_intervals):.2f}%)")
    print(f"  (below 5s total: {100*np.sum(counts_check[:5])/len(filtered_intervals):.1f}% "
          f"of all Variant-B intervals -- confirms 1s-wide bins there resolve real structure)")
    print()

    print_overall_summary("Variant A (all intervals, includes 0.000s)", all_intervals, len(per_match))
    print_overall_summary(f"Variant B ({MIN_VALID_INTERVAL}s < interval < {MAX_VALID_INTERVAL:.0f}s)", filtered_intervals, len(per_match))

    print("Building per-match statistics table...")
    per_match_df = build_per_match_table(per_match)
    csv_path = RESULTS_DIR / "statsbomb_360_interval_per_match_statistics.csv"
    per_match_df.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path} ({len(per_match_df)} rows)")
    print()

    print("=" * 70)
    print("Snapshot/interval relationship check")
    print("=" * 70)
    print("NOT simply 'snapshots = intervals + 1' -- that only holds for a single")
    print("unbroken period. Intervals are computed WITHIN each period separately,")
    print("so one interval is lost per period boundary:")
    print("  snapshots == intervals(all, unfiltered) + n_periods + n_negative_excluded")
    total_snapshots_sum = int(per_match_df["total_snapshots"].sum())
    total_intervals_all_sum = int(per_match_df["n_intervals_all"].sum())
    total_periods_sum = int(per_match_df["n_periods"].sum())
    total_negative_excluded_sum = int(per_match_df["n_negative_excluded"].sum())
    predicted = total_intervals_all_sum + total_periods_sum + total_negative_excluded_sum
    print(f"  total snapshots (counted directly): {total_snapshots_sum:,}")
    print(f"  total intervals (Variant A) + total periods + excluded non-chronological rows:")
    print(f"    {total_intervals_all_sum:,} + {total_periods_sum:,} + {total_negative_excluded_sum:,} = {predicted:,}")
    print(f"  identity holds exactly: {total_snapshots_sum == predicted}")
    print()

    print("Rendering global histograms...")
    plot_global_histogram(
        all_intervals,
        "Temporal resolution of StatsBomb 360 snapshots",
        f"Variant A: all intervals, including duplicated same-instant snapshots (n={len(all_intervals):,} intervals, {len(per_match)} matches)",
        RESULTS_DIR / "statsbomb_360_interval_distribution_all.png",
    )
    plot_global_histogram(
        filtered_intervals,
        "Temporal resolution of StatsBomb 360 snapshots (filtered)",
        f"Variant B: {MIN_VALID_INTERVAL}s < interval < {MAX_VALID_INTERVAL:.0f}s, duplicated/outlier intervals excluded (n={len(filtered_intervals):,})",
        RESULTS_DIR / "statsbomb_360_interval_distribution_filtered.png",
    )
    print(f"  saved: {RESULTS_DIR / 'statsbomb_360_interval_distribution_all.png'} (+ .pdf)")
    print(f"  saved: {RESULTS_DIR / 'statsbomb_360_interval_distribution_filtered.png'} (+ .pdf)")
    print()

    print("=" * 70)
    print("Per-match distribution: total valid (Variant B) intervals per match")
    print("=" * 70)
    per_match_valid_counts = per_match_df["n_intervals_filtered"].to_numpy(dtype=float)
    n_matches_box = len(per_match_valid_counts)
    mean_box = float(np.mean(per_match_valid_counts))
    median_box = float(np.median(per_match_valid_counts))
    std_box = float(np.std(per_match_valid_counts, ddof=1))
    q1_box, q3_box = np.percentile(per_match_valid_counts, [25, 75])
    iqr_box = q3_box - q1_box
    p10_box, p90_box = np.percentile(per_match_valid_counts, [10, 90])
    outliers_box, lower_fence, upper_fence = compute_boxplot_outliers(per_match_valid_counts)
    print(f"  matches: {n_matches_box}")
    print(f"  mean={mean_box:,.1f}  median={median_box:,.1f}  std={std_box:,.1f}")
    print(f"  min={per_match_valid_counts.min():,.0f}  max={per_match_valid_counts.max():,.0f}")
    print(f"  Q1={q1_box:,.1f}  Q3={q3_box:,.1f}  IQR={iqr_box:,.1f}")
    print(f"  P10={p10_box:,.1f}  P90={p90_box:,.1f}")
    print(f"  1.5*IQR fences: [{lower_fence:,.1f}, {upper_fence:,.1f}]")
    print(f"  outliers by the 1.5*IQR rule: {len(outliers_box)} match(es)")
    if len(outliers_box):
        outlier_ids = per_match_df.loc[
            per_match_df["n_intervals_filtered"].isin(outliers_box), "match_id"
        ].tolist()
        print(f"    match IDs: {outlier_ids}")
    print()

    print("Rendering per-match boxplot...")
    plot_intervals_per_match_boxplot(
        per_match_valid_counts, RESULTS_DIR / "statsbomb_360_intervals_per_match_boxplot.png"
    )
    print(f"  saved: {RESULTS_DIR / 'statsbomb_360_intervals_per_match_boxplot.png'} (+ .pdf)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
