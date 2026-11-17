"""Readiness audit of the raw event stream and the production dataset.

Descriptive only, run BEFORE the match split: it reads all eligible matches
(including those later held out as the final test set) to characterise the
data, but nothing computed here is used to fit or select any model.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pandas as pd

from src.match_exclusions import load_exclusion_manifest

from .config import ExperimentConfig
from .raw_events import available_360_match_ids, load_match_metadata
from .spatial import spatial_observations
from .tasks import build_task_table, exclusion_summary


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def task_statistics(task_table: pd.DataFrame, vocabulary: tuple[str, ...]) -> dict:
    rows = task_table[task_table["eval_eligible"]]
    dist = rows["target"].value_counts().reindex(list(vocabulary), fill_value=0)
    joint = pd.crosstab(rows["symbol"], rows["target"])
    h_cond = sum(len(g) / len(rows) * _entropy(g["target"].value_counts().to_numpy())
                 for _, g in rows.groupby("symbol"))
    skipped = rows["n_skipped_raw_events"].astype(float)
    return {
        "evaluation_rows": int(len(rows)),
        "matches": int(rows["match_id"].nunique()),
        "sequences": int(task_table["sequence_id"].nunique()),
        "observations_in_sequences": int(len(task_table)),
        "target_distribution": {k: int(v) for k, v in dist.items()},
        "target_share": {k: float(v / len(rows)) for k, v in dist.items()},
        "majority_class_share": float(dist.max() / len(rows)),
        "imbalance_ratio_max_over_min": float(dist.max() / max(dist[dist > 0].min(), 1)),
        "target_entropy_bits": _entropy(dist.to_numpy()),
        "conditional_entropy_given_current_bits": float(h_cond),
        "largest_transition_cell": {
            "from": str(joint.stack().idxmax()[0]), "to": str(joint.stack().idxmax()[1]),
            "share_of_rows": float(joint.to_numpy().max() / len(rows)),
        },
        "rows_skipping_at_least_one_raw_event": float((skipped >= 1).mean()),
        "mean_skipped_raw_events": float(skipped.mean()),
        "rows_by_period": {str(k): int(v) for k, v in rows["period"].value_counts().sort_index().items()},
    }


def run_audit(event_table: pd.DataFrame, production: pd.DataFrame, production_sha256: str,
              config: ExperimentConfig) -> tuple[dict, pd.DataFrame]:
    t = event_table
    types = config.production_event_types
    excluded = load_exclusion_manifest()
    metadata = load_match_metadata()
    ids = sorted(t["match_id"].unique().tolist())
    audit: dict = {}

    comps = Counter(f"{metadata[m].competition} {metadata[m].season}" for m in ids)
    audit["matches"] = {
        "three_sixty_files": len(available_360_match_ids()),
        "excluded_by_registry": {str(k): v.reason for k, v in sorted(excluded.items())},
        "eligible": len(ids),
        "by_competition_season": dict(sorted(comps.items())),
        "max_period_per_match": {str(k): int(v) for k, v in t.groupby("match_id")["period"].max().value_counts().sort_index().items()},
    }

    dt = t.groupby(["match_id", "period"])["t_seconds"].diff()
    neg = t[dt < 0]
    content_hash = t.groupby("match_id").apply(
        lambda g: hashlib.sha256(pd.util.hash_pandas_object(
            g[["period", "timestamp", "type", "location_x", "location_y"]], index=False).values.tobytes()).hexdigest(),
        include_groups=False)
    meta_frame = pd.DataFrame([metadata[m].__dict__ for m in ids])
    audit["event_stream"] = {
        "raw_events": int(len(t)),
        "event_types": int(t["type"].nunique()),
        "events_by_period": {str(k): int(v) for k, v in t["period"].value_counts().sort_index().items()},
        "index_equals_raw_array_order": bool((t["index"] == t["list_position"] + 1).all()),
        "duplicate_event_ids": int(t["event_id"].duplicated().sum()),
        "period_decreases_along_stream": int(t.groupby("match_id")["period"].apply(lambda s: (s.diff() < 0).sum()).sum()),
        "adjacent_pairs_with_zero_timestamp_gap": int((dt == 0).sum()),
        "adjacent_pairs_with_negative_timestamp_gap": int(len(neg)),
        "negative_gap_event_types": {k: int(v) for k, v in neg["type"].value_counts().items()},
        "negative_gap_below_minus_60s": int((dt[dt < 0] < -60).sum()),
        "possession_id_decreases_along_stream": int(t.groupby("match_id")["possession"].apply(lambda s: (s.diff() < 0).sum()).sum()),
        "first_event_type_of_period": {k: int(v) for k, v in t.groupby(["match_id", "period"]).head(1)["type"].value_counts().items()},
        "last_event_type_of_period": {k: int(v) for k, v in t.groupby(["match_id", "period"]).tail(1)["type"].value_counts().items()},
        "duplicate_match_content_hashes": int(content_hash.duplicated().sum()),
        "duplicate_date_home_away": int(meta_frame.duplicated(["match_date", "home_team", "away_team"]).sum()),
    }
    p5 = t[t["period"] == 5]
    audit["penalties"] = {
        "period_5_matches": int(p5["match_id"].nunique()),
        "period_5_event_types": {k: int(v) for k, v in p5["type"].value_counts().items()},
        "period_5_shot_types": {k: int(v) for k, v in p5["shot_type"].value_counts().items()},
        "period_5_is_shootout": bool(set(p5.loc[p5["type"] == "Shot", "shot_type"]) == {"Penalty"}),
        "ordinary_penalty_shots_periods_1_to_4": int(((t["shot_type"] == "Penalty") & (t["period"] < 5)).sum()),
        "production_rows_in_period_5": int((production["period"] == 5).sum()),
    }
    sel = t[t["type"].isin(types)]
    audit["production_event_types"] = {
        "types": list(types),
        "raw_events_of_these_types": int(len(sel)),
        "share_of_raw_events": float(len(sel) / len(t)),
        "with_360_record": int(sel["has_360"].sum()),
        "with_location": float(sel["has_location"].mean()),
        "raw_events_of_other_types": {k: int(v) for k, v in t.loc[~t["type"].isin(types), "type"].value_counts().items()},
    }

    # production dataset
    prod_ids = set(production["current_event_id"])
    filtered = build_task_table(t[t["period"].isin([1, 2, 3, 4, 5])], config.tasks[config.primary_task].__class__(
        **{**config.tasks[config.primary_task].__dict__, "excluded_periods": ()}))
    derived = filtered[filtered["eval_eligible"]]
    merged = derived.merge(production[["current_event_id", "next_event_id", "next_event_type"]],
                           left_on="event_id", right_on="current_event_id", how="outer", indicator=True)
    both = merged[merged["_merge"] == "both"]
    spatial = spatial_observations(production)
    loc = t.set_index("event_id")[["location_x", "location_y"]]
    sp = spatial.join(loc, on="event_id")
    mirrored_y = np.where(production["team_normalization"].to_numpy() == "MIRRORED",
                          80 - production["actor_y"].to_numpy(), production["actor_y"].to_numpy())
    dist = np.hypot(sp["x_attacking"] - sp["location_x"], mirrored_y - sp["location_y"])
    raw_dist = np.hypot(production["actor_x"].to_numpy() - sp["location_x"], production["actor_y"].to_numpy() - sp["location_y"])
    audit["production_dataset"] = {
        "sha256": production_sha256,
        "rows": int(len(production)),
        "columns": int(production.shape[1]),
        "matches": int(production["match_id"].nunique()),
        "independent_rederivation": {
            "rows_derived": int(len(derived)), "only_derived": int((merged["_merge"] == "left_only").sum()),
            "only_production": int((merged["_merge"] == "right_only").sum()),
            "target_type_mismatches": int((both["target"] != both["next_event_type"]).sum()),
            "target_event_id_mismatches": int((both["target_event_id"] != both["next_event_id"]).sum()),
            "production_ids_unique": bool(len(prod_ids) == len(production)),
        },
        "actor_x_missing": int(production["actor_x"].isna().sum()),
        "orientation_unavailable": int((~production["orientation_available"].astype(bool)).sum()),
        "visible_players_summary": production["visible_players"].describe().round(3).to_dict(),
        "visible_ambiguous_rows": int((production["visible_ambiguous"] > 0).sum()),
        "identity_quality": {k: int(v) for k, v in production["identity_quality"].value_counts().items()},
        "visible_area_polygon_in_dataset": False,
    }
    ok = dist.notna()
    audit["spatial_transform_validation"] = {
        "description": "distance between the acting-frame actor point (from actor_x/actor_y + team_normalization) "
                       "and the raw event location (StatsBomb: acting team's attacking frame)",
        "rows_compared": int(ok.sum()),
        "share_within_1_unit": float((dist[ok] <= 1).mean()),
        "share_within_5_units": float((dist[ok] <= 5).mean()),
        "median_distance": float(dist[ok].median()),
        "max_distance": float(dist[ok].max()),
        "share_within_5_units_without_transform": float((raw_dist[ok] <= 5).mean()),
    }

    audit["tasks"] = {}
    exclusion_frames = []
    for name, task in config.tasks.items():
        table = build_task_table(t, task)
        audit["tasks"][name] = {"description": task.description, **task_statistics(table, task.vocabulary)}
        exclusion_frames.append(exclusion_summary(t, task))
    skipped = derived.loc[derived["period"] != 5]
    audit["filtered_task_skips"] = {
        "skipped_types_between_current_and_target": _skipped_type_counts(t, skipped),
    }
    return audit, pd.concat(exclusion_frames, ignore_index=True)


def _skipped_type_counts(t: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """Raw types lying strictly between a filtered-task row and its target."""
    types = t["type"].to_numpy()
    pos_of = pd.Series(np.arange(len(t)), index=t["event_id"])
    start = pos_of.loc[rows["event_id"]].to_numpy()
    end = pos_of.loc[rows["target_event_id"]].to_numpy()
    counts: Counter = Counter()
    for s, e in zip(start, end):
        if e - s > 1:
            counts.update(types[s + 1:e])
    return {k: int(v) for k, v in counts.most_common()}


def audit_markdown(audit: dict) -> str:
    m, s, pen, prod, sp = (audit["matches"], audit["event_stream"], audit["penalties"], audit["production_dataset"],
                           audit["spatial_transform_validation"])
    rd = prod["independent_rederivation"]
    lines = [
        "# Data readiness audit (Markov / HMM pilot)", "",
        "Descriptive audit of all eligible matches, run before the match split. Generated by "
        "`scripts/run_markov_hmm_pilot.py audit` (`src/sequence_models/data_audit.py`); numbers are copied from "
        "`data_audit.json`.", "",
        "## Matches",
        f"- 360 files: {m['three_sixty_files']}; excluded by `configs/excluded_matches.json`: "
        f"{len(m['excluded_by_registry'])}; **eligible: {m['eligible']}**.",
        f"- Max period per match: {m['max_period_per_match']} (3/4 = extra time, 5 = penalty shootout).",
        f"- Competitions: {m['by_competition_season']}.", "",
        "## Raw event stream",
        f"- Raw events: **{s['raw_events']:,}**, {s['event_types']} types; by period {s['events_by_period']}.",
        f"- `index` equals raw array order: **{s['index_equals_raw_array_order']}**; duplicate event ids: "
        f"{s['duplicate_event_ids']}; period decreases: {s['period_decreases_along_stream']}.",
        f"- Adjacent raw pairs with identical timestamps: {s['adjacent_pairs_with_zero_timestamp_gap']:,}; with a "
        f"negative gap: {s['adjacent_pairs_with_negative_timestamp_gap']} ({s['negative_gap_event_types']}; "
        f"{s['negative_gap_below_minus_60s']} jump back more than 60 s, i.e. corrupted timestamps near the end of a "
        "period). **Event order, not the clock, defines the sequence.**",
        f"- Possession id decreases: {s['possession_id_decreases_along_stream']} (annotation interleaving around "
        "referee ball-drops).",
        f"- Every period starts with {s['first_event_type_of_period']} and ends with {s['last_event_type_of_period']}.",
        f"- Duplicate matches: identical content hashes {s['duplicate_match_content_hashes']}, identical "
        f"(date, home, away) {s['duplicate_date_home_away']}.", "",
        "## Penalties",
        f"- Period 5 in {pen['period_5_matches']} matches; event types {pen['period_5_event_types']}; all shots are "
        f"penalties: **{pen['period_5_is_shootout']}** -> period 5 is the shootout and is excluded.",
        f"- Ordinary penalty shots in periods 1-4: {pen['ordinary_penalty_shots_periods_1_to_4']} (kept).",
        f"- Production rows in period 5: {pen['production_rows_in_period_5']} (dropped by the pilot task definition; "
        "the production file is not modified).", "",
        "## Production dataset",
        f"- SHA-256 `{prod['sha256']}`; {prod['rows']:,} rows x {prod['columns']} columns; {prod['matches']} matches.",
        f"- Independent re-derivation of every row and target from the raw stream: only-derived {rd['only_derived']}, "
        f"only-production {rd['only_production']}, target type mismatches {rd['target_type_mismatches']}, target id "
        f"mismatches {rd['target_event_id_mismatches']}.",
        f"- actor_x missing: {prod['actor_x_missing']}; causal orientation unavailable: {prod['orientation_unavailable']}.",
        f"- Rows with unresolved-identity players: {prod['visible_ambiguous_rows']:,}; identity quality "
        f"{prod['identity_quality']}.",
        "- The dataset stores visible-player counts, not the visible_area polygon or player coordinates; spatial "
        "pressure / nearest-opponent features would need a new feature build (not done in this pilot).", "",
        "## Spatial transform validation",
        f"- Acting-frame actor point vs raw event location ({sp['rows_compared']:,} rows): within 1 unit "
        f"{sp['share_within_1_unit']:.4%}, within 5 units {sp['share_within_5_units']:.4%}, median "
        f"{sp['median_distance']:.2g}, max {sp['max_distance']:.1f}. Without the transform: within 5 units "
        f"{sp['share_within_5_units_without_transform']:.2%}.", "",
        "## Candidate targets (all eligible matches, periods 1-4, evaluation rows)", "",
        "| | " + " | ".join(audit["tasks"]) + " |", "|---|" + "---|" * len(audit["tasks"]),
    ]
    keys = [("evaluation_rows", "{:,}"), ("majority_class_share", "{:.3f}"), ("imbalance_ratio_max_over_min", "{:.1f}"),
            ("target_entropy_bits", "{:.3f}"), ("conditional_entropy_given_current_bits", "{:.3f}"),
            ("rows_skipping_at_least_one_raw_event", "{:.3f}"), ("mean_skipped_raw_events", "{:.3f}")]
    for key, fmt in keys:
        lines.append(f"| {key} | " + " | ".join(fmt.format(v[key]) for v in audit["tasks"].values()) + " |")
    lines.append("| largest cell | " + " | ".join(
        f"{v['largest_transition_cell']['from']} -> {v['largest_transition_cell']['to']} "
        f"({v['largest_transition_cell']['share_of_rows']:.3f})" for v in audit["tasks"].values()) + " |")
    lines += ["", "Target distributions:", ""]
    for name, v in audit["tasks"].items():
        lines.append(f"- **{name}**: " + ", ".join(f"{k} {n:,} ({v['target_share'][k]:.2%})"
                                                   for k, n in v["target_distribution"].items()))
    lines += ["", "Raw event types skipped between a filtered8 row and its target (all periods 1-4):", "",
              ", ".join(f"{k} {v:,}" for k, v in list(audit["filtered_task_skips"]
                                                      ["skipped_types_between_current_and_target"].items())[:12])]
    return "\n".join(lines) + "\n"
