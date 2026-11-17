"""Taxonomy comparison (8 vs 15 types), zone validation and feature assessment.

Runs on DEVELOPMENT matches only (experiment 2 never reads final-test data),
so its numbers differ from the all-match numbers of the earlier taxonomy
research, which predate the split.
"""

from __future__ import annotations

import importlib.util
import math
from collections import Counter

import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT

from .config import TaskSpec
from .spatial import load_production_rows, spatial_observations
from .tasks import build_task_table
from .zones import ZONE_NAMES, zone_of_x

RESEARCH_SCRIPT = PROJECT_ROOT / "results" / "event_taxonomy_research_2026-10-27" / "diagnostics" / "compare_scenarios.py"


def _entropy(counts) -> float:
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def _conditional_entropy(frame: pd.DataFrame, by: list[str]) -> float:
    n = len(frame)
    return float(sum(len(g) / n * _entropy(g["target"].value_counts().to_numpy()) for _, g in frame.groupby(by)))


def taxonomy_comparison(events: pd.DataFrame, tasks: dict[str, TaskSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per taxonomy, class distribution rows). ``events``: development raw table."""
    in_play = events[~events["period"].isin([5])]
    rows, dist = [], []
    raw_pairs = int(sum(len(g) - 1 for _, g in in_play.groupby(["match_id", "period"])))
    for name, task in tasks.items():
        t = build_task_table(events, task)
        t = t.merge(events[["event_id", "location_x"]], on="event_id", how="left")
        t["zone"] = zone_of_x(t["location_x"].to_numpy(dtype=float))
        ev = t[t["eval_eligible"]].copy()
        nxt = t.groupby("sequence_id")["t_seconds"].shift(-1)
        ev["gap"] = (nxt - t["t_seconds"]).loc[ev.index]
        counts = ev["target"].value_counts().reindex(list(task.vocabulary), fill_value=0)
        shares = counts / counts.sum()
        obs = t
        skipped = ev["n_skipped_raw_events"].astype(float)
        cells = ev.groupby(["symbol", "target"]).size()
        by_cur = ev.groupby("symbol").size()
        cell_p = cells / by_cur.reindex(cells.index.get_level_values(0)).to_numpy()
        v = len(task.vocabulary)
        rows.append({
            "taxonomy": name, "n_classes": v, "target_definition": "next event of the taxonomy in the same period (filtered-event task)",
            "matches": int(ev["match_id"].nunique()), "sequences": int(t["sequence_id"].nunique()),
            "observations_in_sequences": int(len(obs)), "mean_sequence_length": float(obs.groupby("sequence_id").size().mean()),
            "evaluation_rows": int(len(ev)),
            "raw_events_in_taxonomy_share": float(len(obs) / len(in_play)),
            "observations_with_360_share": float(obs["has_360"].mean()),
            "targets_without_own_360_share": float((~t.set_index("event_id").loc[ev["target_event_id"], "has_360"].to_numpy()).mean()),
            "rows_skipping_raw_events_share": float((skipped >= 1).mean()), "mean_skipped_raw_events": float(skipped.mean()),
            "raw_within_period_adjacent_pairs": raw_pairs,
            "rows_with_zero_time_gap_share": float((ev["gap"] == 0).mean()),
            "majority_class": str(counts.idxmax()), "majority_share": float(shares.max()),
            "smallest_class": str(counts.idxmin()), "smallest_class_rows": int(counts.min()),
            "imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
            "classes_below_5pct": int((shares < 0.05).sum()), "classes_below_1pct": int((shares < 0.01).sum()),
            "target_entropy_bits": _entropy(counts.to_numpy()),
            "H_target_given_current_bits": _conditional_entropy(ev, ["symbol"]),
            "H_target_given_current_and_zone_bits": _conditional_entropy(ev, ["symbol", "zone"]),
            "H_target_given_previous_and_current_bits": _conditional_entropy(
                ev.assign(prev=t.groupby("sequence_id")["symbol"].shift(1).loc[ev.index].fillna("START")), ["prev", "symbol"]),
            "rows_in_cells_with_p_ge_0_9_share": float(cells[cell_p >= 0.9].sum() / len(ev)),
            "largest_cell": f"{cells.idxmax()[0]} -> {cells.idxmax()[1]} ({cells.max() / len(ev):.3f})",
            "zone_unknown_share": float((ev["zone"] < 0).mean()),
            "params_markov_order1": v * (v - 1), "params_markov_order2": (v + 1) * v * (v - 1),
            "params_markov_order1_zone": 4 * v * (v - 1), "params_markov_order2_zone": 4 * (v + 1) * v * (v - 1),
            "params_hmm_event_K8": 7 + 8 * 7 + 8 * (v - 1),
        })
        top = Counter()
        for c in task.vocabulary:
            sub = ev[ev["target"] == c]
            src = obs[obs["raw_type"] == c]
            dist.append({"taxonomy": name, "class": c, "evaluation_rows_as_target": int(counts[c]),
                         "share": float(shares[c]), "rare_lt_5pct": bool(shares[c] < 0.05),
                         "observations_as_input": int(len(src)), "input_360_coverage": float(src["has_360"].mean()),
                         "matches_with_class_as_target": int(sub["match_id"].nunique())})
        if name == "scenarioD15":
            dd = ev[(ev["symbol"] == "Dispossessed")]
            rows[-1]["dispossessed_to_duel_rows"] = int((dd["target"] == "Duel").sum())
            rows[-1]["dispossessed_to_duel_probability"] = float((dd["target"] == "Duel").mean())
        del top
    return pd.DataFrame(rows), pd.DataFrame(dist)


def research_script_check(match_ids: list[int], events: pd.DataFrame, tasks: dict[str, TaskSpec]) -> dict:
    """Independent implementation check: the research's own per-match routine
    (compare_scenarios.per_match; periods NOT excluded there) vs build_task_table
    with no excluded period, on the given development matches."""
    spec = importlib.util.spec_from_file_location("compare_scenarios", RESEARCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = {}
    for name, key in (("filtered8", "A_current_8"), ("scenarioD15", "D_expanded_selected")):
        task = tasks[name]
        ours = build_task_table(events, TaskSpec(**{**task.__dict__, "excluded_periods": ()}))
        ours = ours[ours["eval_eligible"]]
        theirs_rows, theirs_targets = 0, Counter()
        for m in match_ids:
            _, res = module.per_match(m)
            theirs_rows += res[key].get("c", {}).get("rows", 0)
            theirs_targets.update(res[key].get("target", {}))
        ours_targets = Counter(ours["target"].tolist())
        out[name] = {"matches": len(match_ids), "rows_ours": int(len(ours)), "rows_research_script": int(theirs_rows),
                     "target_counts_equal": dict(ours_targets) == dict(theirs_targets),
                     "passed": int(len(ours)) == int(theirs_rows) and dict(ours_targets) == dict(theirs_targets)}
    return out


def zone_source_validation(match_ids: list[int], events: pd.DataFrame) -> dict:
    """Zone from the raw event location vs zone from the approved causal 360 actor point
    (acting-team frame, experiment 1 transform) on production rows."""
    prod = load_production_rows(match_ids)
    prod = prod[prod["period"] != 5]
    sp = spatial_observations(prod).merge(events[["event_id", "location_x"]], on="event_id", how="left")
    both = sp["x_attacking"].notna() & sp["location_x"].notna()
    z_event = zone_of_x(sp.loc[both, "location_x"].to_numpy())
    z_actor = zone_of_x(np.clip(sp.loc[both, "x_attacking"].to_numpy(), 0, 120))
    agree = z_event == z_actor
    confusion = pd.crosstab(pd.Series(z_event, name="event_location_zone"), pd.Series(z_actor, name="actor_point_zone"))
    return {"rows_compared": int(both.sum()), "zone_agreement": float(agree.mean()),
            "disagreements": int((~agree).sum()),
            "confusion_event_zone_rows_vs_actor_zone_cols": {ZONE_NAMES[i]: {ZONE_NAMES[j]: int(confusion.loc[i, j])
                                                                            for j in confusion.columns}
                                                              for i in confusion.index},
            "actor_point_missing": int((~sp["x_attacking"].notna()).sum())}


def feature_assessment(data_table: pd.DataFrame) -> pd.DataFrame:
    """Measured availability of candidate causal features on a task table (one taxonomy)."""
    t = data_table
    gap = t.groupby("sequence_id")["t_seconds"].diff()
    loc = t["location_x"].notna()
    rows = [
        ("distance_to_attacking_goal", float(loc.mean()), "event location (acting frame)", "none (own attribute of event t)",
         "continuous: needs a density (the experiment-1 failure mode) or discretisation; pitch thirds are the discretised form",
         "covered by pitch thirds; finer zones next"),
        ("time_since_previous_event", float((gap.notna()).mean()),
         f"zero gap {float((gap == 0).mean()):.3f}, negative gap {int((gap < 0).sum())} rows", "none if previous event only",
         "heavy-tailed continuous; ties and corrupted timestamps (experiment 1 audit)", "defer; bin if used"),
        ("possession_change_indicator", 1.0, "team of event t vs event t-1 in the sequence", "none",
         "binary context: compatible with Markov contexts and as a categorical HMM emission",
         f"mechanical in D (defensive actions by the other team): changed in {float(t['team_changed_since_previous'].mean()):.3f} of observations"),
        ("current_possession_duration", float(t["possession"].notna().mean()) if "possession" in t else float("nan"),
         "StatsBomb possession id", "MODERATE: possession ids are assigned by the annotator and are not monotone (9 decreases)",
         "continuous", "not used"),
        ("previous_event_types", 1.0, "sequence", "none", "native to second-order chains / HMM state", "used (mc2, HMM)"),
        ("visible_teammates_opponents", float(t["has_360"].mean()), "360 visible counts (8 production types only)",
         "none (causal identity)", "count depends on camera framing; unobserved != absent",
         "not used: not available for the 7 added Scenario D types in the production dataset"),
        ("nearest_visible_opponent_distance", float(t["has_360"].mean()), "360 freeze frame + visible_area",
         "none if causal orientation/identity", "censored by the visible area (unobserved players)",
         "not built: needs a new causal feature build with visible-area censoring"),
    ]
    return pd.DataFrame(rows, columns=["feature", "availability_share", "source", "leakage_risk", "model_compatibility",
                                       "decision"])
