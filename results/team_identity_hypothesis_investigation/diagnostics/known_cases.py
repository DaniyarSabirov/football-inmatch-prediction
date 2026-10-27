"""Build hypothesis_comparison.csv for the manually reviewed / investigated cases
from raw JSON plus the investigation outputs (investigation only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from snapshot_relations import load, ms, relate  # noqa: E402
from src.animate_statsbomb_360_match import compute_orientation_results, compute_shared_group_info, load_and_join_match  # noqa: E402

OUT = HERE.parent
CASES = [
    ("case_01", 3893791, 1, "00:18:25.761", "Duel", "copy_missed_by_problem_c",
     "Colours inverted for all 18 players in the Costa Rica Duel copy; Spain's two records at the same ms are consistent.",
     "Hypothesis A (shared-snapshot handling too strict: copy differs only in the actor flag)"),
    ("case_02", 3895107, 2, "00:37:13.060", "Block", "derived_record_flags_not_reexpressed",
     "Köln Block record: coordinates mirrored into Köln's frame (constant 2.53 yd offset from Leverkusen's Block 82 ms earlier) but teammate flags still Leverkusen's view.",
     "Raw record violates the spec in a structured way (flags not re-expressed); detectable, not random"),
    ("case_03", 3857256, 2, "00:45:23.300", "Pass", "raw_label_scramble (weak)",
     "Only 2 unambiguously paired players change colour against each neighbour; rest is goal-mouth crowding.",
     "Mostly a detection artefact; at most a 2-player raw label inconsistency"),
    ("case_04", 3844386, 1, "00:31:58.616", "Pressure", "crowding_artefact",
     "0/4 and 1/2 unambiguous disagreements; keeper coloured consistently.",
     "Not a colour error (pairing artefact in a crowded box)"),
    ("case_05", 3773386, 1, "00:27:13.282", "Ball Receipt*", "copy_missed_by_problem_c",
     "Barcelona Ball Receipt* and Alavés Block are verbatim copies (actor differs); Barcelona record drawn with inverted colours (6/6 unambiguous).",
     "Hypothesis A (copy missed); flags are Alavés's view for both records"),
    ("505", 3930164, 1, "00:00:00.505", "Pass", "raw_label_scramble",
     "Kick-off snapshot, no copy; 10/17 unambiguous players disagree with the next snapshot; Law 8 violated by 8 points.",
     "Genuinely inconsistent raw labels (no pipeline explanation found)"),
    ("3857293_a", 3857293, 1, "00:44:24.954", "Ball Receipt*", "shared_wrong_perspective (fixed in display)",
     "Byte-identical copy: coordinates in Ghana's frame but flags in Uruguay's view; Problem C chose Ghana.",
     "Hypothesis A (Problem C couples flag view to the coordinate owner)"),
    ("3857293_b", 3857293, 2, "00:38:27.375", "Ball Receipt*", "shared_wrong_perspective (fixed in display)",
     "Same structure as 3857293_a.", "Hypothesis A (as above)"),
]


def main() -> int:
    audit = pd.read_csv(OUT / "full_dataset_audit.csv")
    rows = []
    for name, mid, period, ts, etype, cause, evidence, verdict in CASES:
        events, frames, joined = load(mid)
        raw = load_and_join_match(mid)
        results = {j.event["id"]: r for j, r in zip(raw.usable_frames, compute_orientation_results(raw))}
        shared = compute_shared_group_info(raw)
        target = next(e for e in joined if e["period"] == period and e["timestamp"] == ts and e["type"]["name"] == etype)
        t = ms(ts)
        nearby = [e for e in joined if e["period"] == period and abs(ms(e["timestamp"]) - t) <= 2000 and e is not target]
        relations = []
        for e in nearby:
            r = relate(frames[target["id"]], frames[e["id"]])
            if r.coordinates != "DIFFERENT" or abs(ms(e["timestamp"]) - t) <= 150:
                relations.append(f"{e['type']['name']} ({e['team']['name']}, team {e['team']['id']}, dt {ms(e['timestamp']) - t} ms): "
                                 f"{r.coordinates}/{r.flags} {r.equal_flags}/{r.pairs} equal flags, mean {r.mean_distance} yd")
        info = shared.get(target["id"])
        row = audit[(audit.match_id == mid) & (audit.timestamp == ts) & (audit.event_type == etype)]
        rows.append({
            "case": name, "match_id": mid, "period": period, "timestamp": ts, "event_id": target["id"],
            "event_type": etype, "event_team": target["team"]["name"], "event_team_id": target["team"]["id"],
            "raw_teammate_true": sum(p["teammate"] for p in frames[target["id"]]["freeze_frame"]),
            "raw_teammate_false": sum(not p["teammate"] for p in frames[target["id"]]["freeze_frame"]),
            "problem_a": results[target["id"]].orientation,
            "problem_c_shared": info is not None,
            "problem_c_source_team": next((e["team"]["name"] for e in events if info and e["id"] == info.source_event_id), ""),
            "near_simultaneous_relations": " | ".join(relations),
            "unambiguous_disagreement_per_neighbour": row.unambiguous_disagreement_per_neighbour.iloc[0] if len(row) else "",
            "keeper_check_current": row.keeper_check_current.iloc[0] if len(row) else "",
            "audit_root_cause": row.root_cause.iloc[0] if len(row) else "",
            "assessed_cause": cause, "evidence": evidence, "best_explanation": verdict,
        })
    pd.DataFrame(rows).to_csv(OUT / "hypothesis_comparison.csv", index=False)
    print(pd.DataFrame(rows)[["case", "event_team_id", "problem_a", "problem_c_shared", "problem_c_source_team", "audit_root_cause"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
