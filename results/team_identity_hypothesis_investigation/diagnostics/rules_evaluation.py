"""Evaluate candidate team-identity rules against goalkeeper ground truth.

Ground truth (keeper_validation.csv.gz): per record with a keeper near a
goal, whether the CURRENT drawn colours, the OWN-team reading (spec) and the
OPPOSING reading put the keeper at the goal his team defends. A record is
counted once (majority over its keeper points).

Pairs (pairs.csv.gz) supply, per near-simultaneous record pair, the RAW
relation (coordinates and flags), Problem A results, teams, drawn same-scene
test and current colour conflicts.

Rules evaluated per threshold T (|dt| <= T ms):
  R1 current     today's renderer (Problem C + display-only refinement)
  R2 owner       for a cross-team raw same-scene partner (any coordinate
                 relation except DIFFERENT) with EQUAL flags, and Problem A
                 NORMAL for exactly one of the two: read the record's flags
                 relative to that record's team; otherwise keep current
  R3 majority    every raw same-scene member within T votes for the reading
                 implied by the raw flag relation (same team + EQUAL or cross
                 team + INVERTED -> same view as the member's own; otherwise
                 the opposite), the record itself voting for its own team's
                 view; ties fall back to R2, then to current
  R4 naive B     for drawn same-scene pairs with colour conflicts, the later
                 record takes the earlier record's current colours

For each rule: records whose colouring changes, how many of those were wrong
and become right (fixed), right and become wrong (broken), and changes
without keeper evidence.

Outputs: temporal_threshold_analysis.csv, rule_outcomes_by_threshold.csv
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1]
THRESHOLDS = [("0 s", 0.0), ("0.0001 s", 0.1), ("0.0003 s", 0.3), ("0.001 s", 1.0), ("0.01 s", 10.0), ("0.1 s", 100.0)]


def main() -> int:
    pairs = pd.read_csv(OUT / "pairs.csv.gz")
    pairs["adt"] = pairs.dt_ms.abs()
    keepers = pd.read_csv(OUT / "keeper_validation.csv.gz")
    truth = keepers[~keepers.current_drawn_ambiguous].groupby("event_id").agg(
        current_ok=("current_drawn_consistent", "mean"), own_ok=("own_reading_consistent", "mean"))
    truth = (truth >= 0.5)
    current_ok = truth.current_ok.to_dict()
    own_ok = truth.own_ok.to_dict()
    flagged = pd.read_csv(OUT.parent / "visualization_final_validation" / "team_identity_investigation" / "colour_inconsistent_frames.csv")
    strong = set(flagged[flagged.neighbours == 2].event_id)
    weak = set(flagged[flagged.neighbours == 1].event_id)

    def reading_ok(event_id, view_is_own):
        if event_id not in own_ok:
            return None
        return own_ok[event_id] if view_is_own else not own_ok[event_id]

    threshold_rows, rule_rows = [], []
    for label, t in THRESHOLDS:
        p = pairs[pairs.adt <= t]
        raw_scene = p[p.raw_coordinates != "DIFFERENT"]
        cross = raw_scene[~raw_scene.same_team]
        # Per record: R2 owner view and R3 votes.
        owner_view: dict[str, bool] = {}
        votes: dict[str, list[bool]] = defaultdict(list)
        for row in raw_scene.itertuples(index=False):
            for me, other, my_pa, other_pa, my_team, other_team in (
                (row.event_a, row.event_b, row.problem_a_a, row.problem_a_b, row.team_a, row.team_b),
                (row.event_b, row.event_a, row.problem_a_b, row.problem_a_a, row.team_b, row.team_a),
            ):
                if row.raw_flags == "MIXED":
                    continue
                same_view = (row.same_team and row.raw_flags == "EQUAL") or (not row.same_team and row.raw_flags == "INVERTED")
                votes[me].append(same_view)  # vote: "my own view" if the partner's own view maps to it
                if not row.same_team and row.raw_flags == "EQUAL" and me not in owner_view:
                    if my_pa == "NORMAL" and other_pa != "NORMAL":
                        owner_view[me] = True
                    elif other_pa == "NORMAL" and my_pa != "NORMAL":
                        owner_view[me] = False
        # Current view per record = own if current colouring equals own reading.
        records = set(votes) | set(owner_view)

        def tally(new_view_for):
            changed = fixed = broken = unknown = 0
            fixed_strong = fixed_weak = 0
            for event_id in records:
                view = new_view_for(event_id)
                if view is None or event_id not in own_ok:
                    if view is not None:
                        unknown += 1  # counted only if it would differ; undeterminable without keeper
                    continue
                current_view = (current_ok[event_id] == own_ok[event_id])
                if view == current_view:
                    continue
                changed += 1
                after = reading_ok(event_id, view)
                if not current_ok[event_id] and after:
                    fixed += 1
                    fixed_strong += event_id in strong
                    fixed_weak += event_id in weak
                elif current_ok[event_id] and not after:
                    broken += 1
            return changed, fixed, broken, unknown, fixed_strong, fixed_weak

        def r2(event_id):
            return owner_view.get(event_id)

        def r3(event_id):
            v = votes.get(event_id, [])
            own_votes = sum(v) + 1  # the record itself
            opp_votes = len(v) - sum(v)
            if own_votes > opp_votes:
                return True
            if opp_votes > own_votes:
                return False
            return owner_view.get(event_id)

        r2_out, r3_out = tally(r2), tally(r3)
        # R4 naive propagation: later record takes the earlier one's colours.
        drawn_conflict = p[p.same_scene & (p.colour_conflicts > 0)]
        r4 = defaultdict(int)
        for row in drawn_conflict.itertuples(index=False):
            earlier, later = (row.event_a, row.event_b) if row.dt_ms >= 0 else (row.event_b, row.event_a)
            if later not in current_ok or earlier not in current_ok:
                r4["unknown"] += 1
                continue
            r4["changed"] += 1
            if not current_ok[later] and current_ok[earlier]:
                r4["fixed"] += 1
            elif current_ok[later] and not current_ok[earlier]:
                r4["broken"] += 1
        for name, out in (("R2 owner", r2_out), ("R3 majority", r3_out)):
            rule_rows.append({"threshold": label, "rule": name, "records_changed_with_keeper_evidence": out[0],
                              "fixed": out[1], "broken": out[2], "applicable_records_without_keeper_evidence": out[3],
                              "fixed_strong_flagged": out[4], "fixed_weak_flagged": out[5]})
        rule_rows.append({"threshold": label, "rule": "R4 naive earlier->later", "records_changed_with_keeper_evidence": r4["changed"],
                          "fixed": r4["fixed"], "broken": r4["broken"], "applicable_records_without_keeper_evidence": r4["unknown"],
                          "fixed_strong_flagged": "", "fixed_weak_flagged": ""})
        same_scene_drawn = p[p.same_scene]
        conflicts = same_scene_drawn[same_scene_drawn.colour_conflicts > 0]
        unambiguous = conflicts[conflicts.ambiguous_players == 0]
        ids_in_conflict = set(conflicts.event_a) | set(conflicts.event_b)
        threshold_rows.append({
            "threshold": label, "threshold_ms": t,
            "candidate_pairs": len(p), "cross_team_pairs": int((~p.same_team).sum()),
            "identical_coordinates": int(p.raw_coordinates.isin(["IDENTICAL", "MIRROR_IDENTICAL"]).sum()),
            "nearly_identical_coordinates": int(p.raw_coordinates.isin(["NEAR", "SHIFTED", "MIRROR_NEAR", "MIRROR_SHIFTED"]).sum()),
            "cross_team_copies_equal_flags": int(((cross.raw_flags == "EQUAL")).sum()),
            "drawn_same_scene_pairs": int(p.same_scene.sum()),
            "conflicting_colour_pairs": len(conflicts),
            "conflicting_pairs_4plus_players": int((conflicts.colour_conflicts >= 4).sum()),
            "conflicting_pairs_spatially_unambiguous": len(unambiguous),
            "conflicting_pairs_ambiguous_or_unsafe": len(conflicts) - len(unambiguous),
            "strong_flagged_frames_in_a_conflicting_pair": len(ids_in_conflict & strong),
            "weak_flagged_frames_in_a_conflicting_pair": len(ids_in_conflict & weak),
            "R3_fixed_keeper_verified": r3_out[1], "R3_broken_keeper_verified": r3_out[2],
            "R4_fixed_keeper_verified": r4["fixed"], "R4_broken_keeper_verified": r4["broken"],
        })
    pd.DataFrame(threshold_rows).to_csv(OUT / "temporal_threshold_analysis.csv", index=False)
    pd.DataFrame(rule_rows).to_csv(OUT / "rule_outcomes_by_threshold.csv", index=False)
    print(pd.DataFrame(threshold_rows).to_string())
    print(pd.DataFrame(rule_rows).to_string())
    print("keeper-verified records:", len(truth), "currently wrong:", int((~truth.current_ok).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
