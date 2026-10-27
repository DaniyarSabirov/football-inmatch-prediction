"""Quantify the team-identity problem and its ML exposure from the audit
outputs (audit_identity_correction.py). Units are kept apart: matches,
events, snapshots (usable 360 records), player observations (freeze-frame
points) and ML rows. Every percentage names its denominator.

Output: ../summary.json, ../tables/*.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
TABLES = OUT / "tables"
PREVIOUS = HERE.parents[1] / "team_identity_hypothesis_investigation"


def pct(part, whole) -> float:
    return round(100.0 * part / whole, 4) if whole else 0.0


def share(n, d, unit):
    return {"count": int(n), "denominator": int(d), "unit": unit, "percent": pct(n, d)}


def main() -> int:
    TABLES.mkdir(exist_ok=True)
    rec = pd.read_csv(OUT / "records.csv.gz")
    kp = pd.read_csv(OUT / "keepers.csv.gz")
    ml = pd.read_csv(OUT / "ml_rows.csv.gz", low_memory=False)
    gk = pd.read_csv(OUT / "goalkeeper_rule.csv") if (OUT / "goalkeeper_rule.csv").exists() else pd.DataFrame()
    me = pd.read_csv(OUT / "match_events.csv")
    S: dict = {}

    points = int(rec.points.sum())
    snapshots = len(rec)
    S["population"] = {
        "matches": int(me.match_id.nunique()), "events": int(me.events.sum()),
        "events_with_360_record": int(me.events_with_360_record.sum()),
        "usable_snapshots": snapshots, "player_observations": points,
        "ml_rows": len(ml), "legacy_ml_rows": int(me.legacy_ml_rows.sum()),
        "legacy_rows_missing_in_new": int(me.legacy_rows_missing_in_new.sum()),
    }

    # --- player observations ---------------------------------------------
    changed = int(rec.changed_team_vs_legacy.sum())
    gained = int(rec.legacy_ambiguous_now_definite.sum())
    lost = int(rec.legacy_definite_now_unresolved.sum())
    unresolved = int(rec.unresolved.sum())
    reasons = {c.replace("unresolved_", ""): int(rec[c].sum()) for c in rec.columns if c.startswith("unresolved_")}
    suspected = sum(v for k, v in reasons.items() if k != "copy_group_undecided")
    confirmed = int(rec.confirmed.sum())
    S["player_observations"] = {
        "confirmed_correct (positive independent evidence: neighbour agreement or keeper goal side)": share(confirmed, points, "player observations"),
        "corrected_vs_original_pipeline (team changed on evidence)": share(changed, points, "player observations"),
        "  of which goalkeeper rule": share(int(rec.corrected_goalkeeper.sum()), points, "player observations"),
        "  of which view correction of copies/derived records": share(changed - int(rec.corrected_goalkeeper.sum()), points, "player observations"),
        "original_unclear_now_resolved": share(gained, points, "player observations"),
        "original_definite_now_unresolved": share(lost, points, "player observations"),
        "unresolved_after_correction (all)": share(unresolved, points, "player observations"),
        "  unresolved: structurally proven inconsistency, direction undecided (copy_group_undecided)": share(reasons["copy_group_undecided"], points, "player observations"),
        "  unresolved: suspected label errors (other reasons)": share(suspected, points, "player observations"),
        "unresolved_by_reason": reasons,
        "unverified (no evidence either way, kept as read)": share(points - confirmed - unresolved, points, "player observations"),
        "status_counts": {k: int(rec[k].sum()) for k in ("spec", "corrected_view", "corrected_goalkeeper", "unresolved")},
        "note_corrected_view": "corrected_view counts every point read in another view than its own event's spec "
                               "reading, INCLUDING byte-identical copies already handled by the original pipeline; "
                               "changes relative to the original pipeline are 'corrected_vs_original_pipeline'.",
        "original_pipeline_unclear_points": share(int(rec.legacy_ambiguous.sum()), points, "player observations"),
    }

    # --- snapshots -----------------------------------------------------------
    linked = rec[rec.linked]
    conflicted = rec[rec.decision.fillna("").isin(["continuity", "keeper_and_owner", "legacy_problem_c_source", "undecided"])]
    S["snapshots"] = {
        "with_any_change_vs_original": share((rec.changed_team_vs_legacy > 0).sum(), snapshots, "snapshots"),
        "with_any_unresolved_point": share((rec.unresolved > 0).sum(), snapshots, "snapshots"),
        "fully_unresolved": share((rec.unresolved == rec.points).sum(), snapshots, "snapshots"),
        "structurally_linked_to_another_record": share(len(linked), snapshots, "snapshots"),
        "structurally_inconsistent (flag view contradicts its own team)": share(len(conflicted), snapshots, "snapshots"),
        "structural_decisions": conflicted.decision.value_counts().to_dict(),
        "scrambled_raw_labels (>=1 point contradicted by both neighbours)": share(
            ((rec.unresolved_contradicted_by_both_neighbours > 0) | (rec.unresolved_record_contradicted > 0)).sum(), snapshots, "snapshots"),
        "record_level_contradiction (whole record unclear)": share((rec.unresolved_record_contradicted > 0).sum(), snapshots, "snapshots"),
        "orientation_resolved_by": rec.resolved_by.value_counts().to_dict(),
        "orientation_still_uncertain": share((rec.resolved_orientation == "UNCERTAIN").sum(), snapshots, "snapshots"),
    }

    # --- goalkeepers -----------------------------------------------------------
    kp["gk_only_original_error"] = (
        ~kp.legacy_consistent & ~kp.legacy_ambiguous
        & (kp.record_other_points_unresolved == 0) & (kp.record_other_points_changed_vs_legacy == 0)
    )
    original_wrong = kp[~kp.legacy_consistent & ~kp.legacy_ambiguous]
    gk_only = kp[kp.gk_only_original_error]
    S["goalkeepers"] = {
        "observations_near_a_goal": len(kp),
        "original_pipeline_wrong_keeper_colour": share(len(original_wrong), len(kp), "keeper observations near a goal"),
        "goalkeeper_only_errors (rest of record unchanged and resolved)": share(len(gk_only), len(kp), "keeper observations near a goal"),
        "goalkeeper_only_outcome": gk_only.new_status.value_counts().to_dict(),
        "goalkeeper_only_unresolved_reasons": gk_only[gk_only.new_status == "UNRESOLVED"].new_reason.value_counts().to_dict(),
        "new_pipeline_wrong_keeper_colour": share(int((~kp.new_consistent & (kp.new_status != "UNRESOLVED")).sum()), len(kp), "keeper observations near a goal"),
        "new_pipeline_unresolved_keepers": share(int((kp.new_status == "UNRESOLVED").sum()), len(kp), "keeper observations near a goal"),
        "goalkeeper_rule_forward_validation": gk.forward_validation.value_counts().to_dict() if len(gk) else {},
    }
    # Reconcile with the earlier 957 figure (keeper-wrong records whose other
    # points agree with their neighbours: record_perspective == OWN).
    try:
        kv = pd.read_csv(PREVIOUS / "keeper_validation.csv.gz")
        rp = pd.read_csv(PREVIOUS / "record_perspective.csv.gz", usecols=["event_id", "flag_perspective", "current_drawn_disagreement"])
        wrong = kv[~kv.current_drawn_ambiguous & ~kv.current_drawn_consistent].merge(rp, on="event_id", how="left")
        previous_set = wrong[wrong.flag_perspective == "OWN"]
        joined = previous_set.merge(kp, on="event_id", how="left")
        S["goalkeepers"]["reconciliation_with_previous_957"] = {
            "previous_keeper_wrong_records_reproduced": len(wrong),
            "of_which_rest_agrees_with_neighbours (flag_perspective OWN)": len(previous_set),
            "note": "957 could not be reproduced exactly from the saved files; the same definition gives the "
                    "number above (the earlier text may have used an unsaved variant). Outcomes below are for this set.",
            "outcome_in_new_pipeline": joined.new_status.fillna("not a keeper-near-goal observation now").value_counts().to_dict(),
        }
    except FileNotFoundError:
        pass

    # --- ML rows ---------------------------------------------------------------
    n = len(ml)
    ml["uncertain"] = ml.visible_ambiguous > 0
    ml["counts_changed"] = (ml.visible_teammates != ml.legacy_visible_teammates) | (ml.visible_opponents != ml.legacy_visible_opponents)
    ml["actor_changed"] = ~np.isclose(ml.actor_x.fillna(-999), ml.legacy_actor_x.fillna(-999)) | ~np.isclose(ml.actor_y.fillna(-999), ml.legacy_actor_y.fillna(-999))
    ml["orientation_changed"] = ml.orientation != ml.legacy_orientation
    ml["legacy_uncertain"] = ml.legacy_visible_ambiguous > 0
    S["ml_rows"] = {
        "rows": n,
        "identity_quality": ml.identity_quality.value_counts().to_dict(),
        "rows_with_>=1_uncertain_player": share(ml.uncertain.sum(), n, "ML rows"),
        "rows_with_>=1_uncertain_player_in_original": share(ml.legacy_uncertain.sum(), n, "ML rows"),
        "rows_whose_teammate/opponent_counts_change_vs_original": share(ml.counts_changed.sum(), n, "ML rows"),
        "rows_whose_counts_change_and_are_certain": share((ml.counts_changed & ~ml.uncertain).sum(), n, "ML rows"),
        "rows_excluded_if_unresolved_rows_dropped": share(ml.uncertain.sum(), n, "ML rows"),
        "uncertain_players_per_uncertain_row": ml.loc[ml.uncertain, "visible_ambiguous"].describe().round(3).to_dict(),
        "rows_with_entire_freeze_frame_uncertain": share(((ml.visible_ambiguous == ml.visible_players) & ml.uncertain).sum(), n, "ML rows"),
        "teammates_change_distribution": (ml.visible_teammates - ml.legacy_visible_teammates)[ml.counts_changed].value_counts().sort_index().to_dict(),
        "opponents_change_distribution": (ml.visible_opponents - ml.legacy_visible_opponents)[ml.counts_changed].value_counts().sort_index().to_dict(),
        "orientation_sync": {
            "rows_orientation_label_changed": share(ml.orientation_changed.sum(), n, "ML rows"),
            "rows_actor_xy_changed": share(ml.actor_changed.sum(), n, "ML rows"),
            "legacy_uncertain_rows (the earlier 535)": int((ml.legacy_orientation == "UNCERTAIN").sum()),
            "legacy_uncertain_rows_now_resolved": int(((ml.legacy_orientation == "UNCERTAIN") & (ml.orientation != "UNCERTAIN")).sum()),
            "legacy_uncertain_rows_now_rotated (actor_x/y mirrored)": int(((ml.legacy_orientation == "UNCERTAIN") & (ml.orientation == "ROTATED_180")).sum()),
            "legacy_uncertain_rows_now_normal (coordinates unchanged)": int(((ml.legacy_orientation == "UNCERTAIN") & (ml.orientation == "NORMAL")).sum()),
            "legacy_confident_rows_changed": int(((ml.legacy_orientation != "UNCERTAIN") & ml.orientation_changed).sum()),
            "transitions": ml[ml.orientation_changed].groupby(["legacy_orientation", "orientation"]).size().rename("rows").reset_index().to_dict("records"),
            "resolved_by": ml.loc[ml.legacy_orientation == "UNCERTAIN", "orientation_resolved_by"].value_counts().to_dict(),
        },
    }

    # --- concentration and selection bias ----------------------------------------
    by_match = ml.groupby("match_id").agg(rows=("uncertain", "size"), uncertain=("uncertain", "sum"), changed=("counts_changed", "sum"))
    by_match["uncertain_share"] = by_match.uncertain / by_match.rows
    by_match.sort_values("uncertain", ascending=False).to_csv(TABLES / "ml_uncertainty_by_match.csv")
    top = by_match.sort_values("uncertain", ascending=False)
    total_unc = by_match.uncertain.sum()
    S["concentration"] = {
        "matches_with_>=1_uncertain_row": share((by_match.uncertain > 0).sum(), len(by_match), "matches"),
        "matches_with_>=1_changed_row": share((by_match.changed > 0).sum(), len(by_match), "matches"),
        "share_of_uncertain_rows_in_top_10_matches": pct(top.uncertain.head(10).sum(), total_unc),
        "share_of_uncertain_rows_in_top_10pct_matches": pct(top.uncertain.head(max(1, len(top) // 10)).sum(), total_unc),
        "per_match_uncertain_share": by_match.uncertain_share.describe().round(5).to_dict(),
        "max_match": {"match_id": int(top.index[0]), "uncertain_rows": int(top.uncertain.iloc[0]), "rows": int(top.rows.iloc[0])},
    }
    for column, name in (("current_event_type", "by_current_event_type"), ("next_event_type", "by_next_event_type"), ("period", "by_period")):
        table = ml.groupby(column).agg(rows=("uncertain", "size"), uncertain=("uncertain", "sum"), counts_changed=("counts_changed", "sum"))
        table["uncertain_pct_of_group"] = (100 * table.uncertain / table.rows).round(3)
        table["share_of_all_rows_pct"] = (100 * table.rows / n).round(3)
        table["share_after_exclusion_pct"] = (100 * (table.rows - table.uncertain) / (n - ml.uncertain.sum())).round(3)
        table.to_csv(TABLES / f"ml_uncertainty_{name}.csv")
        chi2, p, dof, _ = stats.chi2_contingency(np.vstack([table.uncertain, table.rows - table.uncertain]))
        S["concentration"][name] = {"chi2": round(float(chi2), 2), "dof": int(dof), "p_value": float(p),
                                   "cramers_v": round(float(np.sqrt(chi2 / (n * 1))), 4),
                                   "table": table.reset_index().to_dict("records")}
    # minute buckets and pitch zones (game situation)
    ml["minute_bucket"] = pd.cut(ml.minute, [-1, 15, 30, 45, 60, 75, 90, 200])
    ml["zone"] = pd.cut(ml.actor_x, [-1, 40, 80, 121], labels=["defensive third", "middle third", "attacking third"])
    for column in ("minute_bucket", "zone"):
        table = ml.groupby(column, observed=True).agg(rows=("uncertain", "size"), uncertain=("uncertain", "sum"))
        table["uncertain_pct_of_group"] = (100 * table.uncertain / table.rows).round(3)
        S["concentration"][f"by_{column}"] = {str(k): v for k, v in table.to_dict("index").items()}

    # class imbalance before/after exclusion
    before = ml.next_event_type.value_counts(normalize=True)
    after = ml.loc[~ml.uncertain, "next_event_type"].value_counts(normalize=True)
    imbalance = pd.DataFrame({"share_all_pct": (100 * before).round(4), "share_after_exclusion_pct": (100 * after).round(4)})
    imbalance["relative_change_pct"] = (100 * (imbalance.share_after_exclusion_pct / imbalance.share_all_pct - 1)).round(3)
    imbalance["rows_removed"] = ml[ml.uncertain].next_event_type.value_counts().reindex(imbalance.index).fillna(0).astype(int)
    imbalance["rows_removed_pct_of_class"] = (100 * imbalance.rows_removed / ml.next_event_type.value_counts().reindex(imbalance.index)).round(3)
    imbalance.to_csv(TABLES / "class_balance_before_after_exclusion.csv")
    S["class_balance"] = imbalance.reset_index().rename(columns={"index": "next_event_type"}).to_dict("records")
    S["class_balance_imbalance_ratio"] = {
        "majority_to_minority_all": round(float(before.max() / before.min()), 3),
        "majority_to_minority_after_exclusion": round(float(after.max() / after.min()), 3),
    }

    # feature distributions: excluded vs kept, and original vs corrected
    feats = ["actor_x", "actor_y", "minute", "visible_players", "visible_teammates", "visible_opponents"]
    S["excluded_vs_kept_features"] = {
        f: {"mean_kept": round(float(ml.loc[~ml.uncertain, f].mean()), 3), "mean_excluded": round(float(ml.loc[ml.uncertain, f].mean()), 3),
            "ks_statistic": round(float(stats.ks_2samp(ml.loc[~ml.uncertain, f].dropna(), ml.loc[ml.uncertain, f].dropna()).statistic), 4)}
        for f in feats
    }
    S["original_vs_corrected_features"] = {
        f: {"mean_original": round(float(ml[f"legacy_{f}"].mean()), 5), "mean_corrected": round(float(ml[f].mean()), 5),
            "ks_statistic": round(float(stats.ks_2samp(ml[f"legacy_{f}"].dropna(), ml[f].dropna()).statistic), 6),
            "rows_changed": int((ml[f"legacy_{f}"].fillna(-999) != ml[f].fillna(-999)).sum())}
        for f in ("visible_teammates", "visible_opponents", "actor_x", "actor_y")
    }
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: S[k] for k in ("population", "player_observations", "snapshots", "goalkeepers", "ml_rows")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
