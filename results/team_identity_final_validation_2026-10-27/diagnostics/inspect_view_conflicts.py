"""Raw-data inspection of near-simultaneous (<= 0.25 s) snapshot pairs where
>= 4 unambiguously paired players change colour together (whole-view
conflicts that survived the correction). For each: raw coordinate relation
(as stored / mirrored, mean and max pair distance after removing the common
offset), raw flag relation, Problem A, and which of the two records agrees
with its OWN other neighbour (independent of the pair).

Output: ../view_conflicts.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.orientation_resolution import match_team_ids, resolve_match_orientation  # noqa: E402
from src.team_identity import resolve_match_team_identity, unambiguous_pairs  # noqa: E402

DATA = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"


def relation(a, b):
    best = None
    for mirrored in (False, True):
        bb = np.column_stack([120 - b[:, 0], 80 - b[:, 1]]) if mirrored else b
        cost = np.linalg.norm(a[:, None] - bb[None], axis=2)
        r, c = linear_sum_assignment(cost)
        offset = np.median(bb[c] - a[r], axis=0)
        cost2 = np.linalg.norm((a + offset)[:, None] - bb[None], axis=2)
        r, c = linear_sum_assignment(cost2)
        d = cost2[r, c]
        if best is None or d.mean() < best[0]:
            best = (d.mean(), d.max(), mirrored, list(zip(r, c)), float(np.linalg.norm(offset)))
    return best


def main():
    t = pd.read_csv(OUT / "transitions.csv.gz")
    z = t[t.dt_s <= 0.25]
    links = z.groupby(["match_id", "event_a", "event_b"]).size().rename("flips").reset_index()
    links = links[links.flips >= 4]
    rows = []
    for match_id, group in links.groupby("match_id"):
        events = json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
        frames = {f["event_uuid"]: f for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
        mo = resolve_match_orientation(match_id, events, frames)
        recs = mo.records.usable_frames
        names = {r.event["team"]["id"]: r.event["team"]["name"] for r in recs}
        pair = tuple(names[x] for x in match_team_ids(mo.records))
        ids = resolve_match_team_identity([(r.event, r.frame_360) for r in recs], [x.orientation for x in mo.resolved],
                                          mo.problem_a, pair, events)
        index = {r.event["id"]: k for k, r in enumerate(recs)}
        for row in group.itertuples():
            ka, kb = index[row.event_a], index[row.event_b]
            ea, eb = recs[ka].event, recs[kb].event
            pa = np.array([p["location"] for p in recs[ka].frame_360["freeze_frame"]], float)
            pb = np.array([p["location"] for p in recs[kb].frame_360["freeze_frame"]], float)
            mean, worst, mirrored, mapping, offset = relation(pa, pb)
            fa = [p["teammate"] for p in recs[ka].frame_360["freeze_frame"]]
            fb = [p["teammate"] for p in recs[kb].frame_360["freeze_frame"]]
            equal = sum(fa[i] == fb[j] for i, j in mapping)
            rows.append({
                "match_id": match_id, "period": ea["period"], "time_a": ea["timestamp"], "time_b": eb["timestamp"],
                "type_a": ea["type"]["name"], "team_a": ea["team"]["name"], "type_b": eb["type"]["name"], "team_b": eb["team"]["name"],
                "flips": row.flips, "players_a": len(pa), "players_b": len(pb),
                "raw_relation": ("MIRROR " if mirrored else "") + f"mean {mean:.2f} max {worst:.2f} offset {offset:.2f}",
                "raw_equal_flags": f"{equal}/{len(mapping)}",
                "problem_a_a": mo.problem_a[ka].orientation, "problem_a_b": mo.problem_a[kb].orientation,
                "resolved_a": mo.resolved[ka].orientation, "resolved_b": mo.resolved[kb].orientation,
                "status_a": ",".join(sorted(set(ids[ea["id"]].statuses))), "status_b": ",".join(sorted(set(ids[eb["id"]].statuses))),
                "decision_a": ids[ea["id"]].decision, "decision_b": ids[eb["id"]].decision,
                "event_a": ea["id"], "event_b": eb["id"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "view_conflicts.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[["match_id", "time_a", "type_a", "team_a", "type_b", "team_b", "flips", "players_a", "raw_relation",
               "raw_equal_flags", "problem_a_a", "problem_a_b", "resolved_a", "resolved_b"]].to_string())


main()
