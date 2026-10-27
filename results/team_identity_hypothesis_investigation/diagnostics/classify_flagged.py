"""Root-cause classification of the 102 strong + 2,094 weak colour-inconsistent
frames (investigation only). Inputs: flagged_frame_audit.csv, pairs.csv.gz,
record_perspective.csv.gz, keeper_validation.csv.gz. Output:
full_dataset_audit.csv and a printed summary.

Categories (first match wins; structural evidence is used even when the
temporal neighbours are too few, e.g. a same-millisecond cluster):
  crowding_artefact            no unambiguous colour change: against every
                               evaluated neighbour at most max(1, 10%) of the
                               unambiguously paired players disagree
  disagrees_with_one_neighbour_only
                               0 unambiguous disagreements against one
                               neighbour, real ones against the other: the
                               other snapshot is the likelier culprit
  copy_missed_by_problem_c     an opposing-team record within 100 ms with
                               IDENTICAL/NEAR coordinates and EQUAL flags,
                               not byte-identical (only actor/keeper/visible
                               area differ), so the copy detection misses it
  shared_wrong_perspective     byte-identical shared group (Problem C applied)
                               but the colours still contradict neighbours
  derived_record_flags_not_reexpressed
                               a record within 100 ms related by a mirror or
                               a constant offset whose flags violate the spec
                               (coordinates re-expressed, flags copied)
  record_level_inversion       no such partner, and swapping every point's
                               team makes the frame consistent
  raw_label_scramble           real unambiguous disagreements with both
                               neighbours, no structural explanation, and no
                               whole-record inversion fits (frame 505 type)
  insufficient_evidence        no neighbour with >= 3 unambiguous pairs
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1]


def parse(text):
    out = []
    for part in str(text).split(";"):
        m = re.search(r"(\d+)/(\d+) \(unambiguous (\d+)/(\d+)\)", part)
        if m:
            out.append(tuple(map(int, m.groups())))
    return out


def main() -> int:
    flagged = pd.read_csv(OUT / "flagged_frame_audit.csv")
    pairs = pd.read_csv(OUT / "pairs.csv.gz")
    pairs = pairs[(pairs.dt_ms.abs() <= 100) & (pairs.raw_coordinates != "DIFFERENT")]
    ids = set(flagged.event_id)
    rel = pd.concat([
        pairs[pairs.event_a.isin(ids)].rename(columns={"event_a": "eid", "event_b": "other"}),
        pairs[pairs.event_b.isin(ids)].rename(columns={"event_b": "eid", "event_a": "other"}),
    ])
    perspective = pd.read_csv(OUT / "record_perspective.csv.gz", usecols=["event_id", "flag_perspective"]).set_index("event_id")
    keepers = pd.read_csv(OUT / "keeper_validation.csv.gz")
    keeper_ok = keepers[~keepers.current_drawn_ambiguous].groupby("event_id").current_drawn_consistent.mean() >= 0.5

    rows = []
    for r in flagged.itertuples(index=False):
        per = parse(r.current_disagreement)
        mine = rel[rel.eid == r.event_id]
        cross = mine[~mine.same_team]
        copy_missed = cross[cross.raw_coordinates.isin(["IDENTICAL", "NEAR"]) & (cross.raw_flags == "EQUAL") & ~cross.byte_identical]
        derived = mine[mine.raw_coordinates.isin(["SHIFTED", "MIRROR_SHIFTED", "MIRROR_NEAR", "MIRROR_IDENTICAL"]) & ~mine.spec_consistent]
        unamb = [(pu, bu) for _, _, bu, pu in per]  # parse() yields (bad, paired, bad_unambiguous, paired_unambiguous)
        enough = [x for x in unamb if x[0] >= 3]
        consistent = [bu <= max(1, 0.1 * pu) for pu, bu in enough]
        if enough and all(consistent):
            category = "crowding_artefact"
        elif enough and any(bu == 0 for pu, bu in enough):
            category = "disagrees_with_one_neighbour_only"
        elif len(copy_missed):
            category = "copy_missed_by_problem_c"
        elif str(r.shared_cross_team_snapshot) == "True":
            category = "shared_wrong_perspective"
        elif len(derived):
            category = "derived_record_flags_not_reexpressed"
        elif not enough:
            category = "insufficient_evidence"
        elif bool(r.invert_resolves):
            category = "record_level_inversion"
        else:
            category = "raw_label_scramble"
        resolvers = [name for name, ok in (("invert", r.invert_resolves), ("partner_transfer", r.partner_resolves),
                                            ("owner_rule", r.owner_resolves)) if bool(ok)]
        rows.append({
            "match_id": r.match_id, "period": r.period, "match_clock": r.match_clock, "timestamp": r.timestamp,
            "event_id": r.event_id, "event_type": r.event_type, "team": r.team, "strength": r.strength,
            "problem_a": r.problem_a, "shared_problem_c": r.shared_cross_team_snapshot,
            "unambiguous_disagreement_per_neighbour": "; ".join(f"{bu}/{pu}" for pu, bu in unamb),
            "same_scene_partners_within_100ms": len(mine),
            "partner_relations": ",".join(sorted(set(mine.raw_coordinates + "+" + mine.raw_flags + mine.same_team.map({True: "/same", False: "/cross"})))),
            "neighbour_based_perspective": perspective.flag_perspective.get(r.event_id, ""),
            "keeper_check_current": {True: "consistent", False: "inconsistent"}.get(keeper_ok.get(r.event_id), "no keeper evidence"),
            "root_cause": category,
            "candidate_corrections_that_resolve_it": ",".join(resolvers),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "full_dataset_audit.csv", index=False)
    print(pd.crosstab(out.root_cause, out.strength, margins=True))
    print(pd.crosstab(out.root_cause, out.candidate_corrections_that_resolve_it.replace("", "none")))
    print(pd.crosstab(out.root_cause, out.keeper_check_current))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
