"""Read-only: (1) is the 360 'actor' point an independent player position or the event location?
(2) do consecutive raw records share an identical 360 snapshot? Tier B only.

Writes ../tables/p1b_actor_vs_event_location_by_type.csv, ../tables/p1b_shared_snapshot_pairs.csv,
../summary_actor.json.

Usage: .venv/bin/python results/investigation_360_taxonomy_2026-11-18/diagnostics/actor_and_shared_frames.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/statsbomb/data"


def one(mid: int):
    events = json.loads((RAW / "events" / f"{mid}.json").read_text(encoding="utf-8"))
    frames = {}
    for r in json.loads((RAW / "three-sixty" / f"{mid}.json").read_text(encoding="utf-8")):
        ff = r.get("freeze_frame") or []
        key = hashlib.sha1(json.dumps(sorted((round(p["location"][0], 3), round(p["location"][1], 3)) for p in ff)).encode()).hexdigest()
        act = [p["location"] for p in ff if p.get("actor")]
        frames[r["event_uuid"]] = (key, act[0] if act else None, len(ff))
    rows, pairs = [], Counter()
    prev = None
    for e in events:
        fr = frames.get(e["id"])
        loc = e.get("location")
        if fr and fr[1] is not None and isinstance(loc, list):
            d = float(np.hypot(fr[1][0] - loc[0], fr[1][1] - loc[1]))
            d_m = float(np.hypot(fr[1][0] - (120 - loc[0]), fr[1][1] - (80 - loc[1])))
            rows.append((e["type"]["name"], d, d_m))
        if prev is not None and prev[2] == e["period"]:
            same = fr is not None and prev[1] is not None and fr[0] == prev[1]
            pairs[(prev[0], e["type"]["name"], same)] += 1
        prev = (e["type"]["name"], fr[0] if fr else None, e["period"])
    return rows, pairs


def main() -> None:
    m = json.loads((ROOT / "results/markov_hmm_pilot/split_manifest.json").read_text())
    tier_b = sorted(set(m["development_match_ids"]) - set(m["pilot_match_ids"]))
    assert not set(tier_b) & set(m["final_test_match_ids"])
    with ProcessPoolExecutor(6) as pool:
        res = list(pool.map(one, tier_b))
    d = pd.DataFrame([r for rows, _ in res for r in rows], columns=["type", "dist", "dist_mirrored"])
    pairs = Counter()
    for _, p in res:
        pairs.update(p)
    by = d.groupby("type").agg(n=("dist", "size"), share_actor_equals_event_location=("dist", lambda x: (x < 0.05).mean()),
                               share_equals_mirrored_location=("dist_mirrored", lambda x: (x < 0.05).mean()),
                               median_dist=("dist", "median"), p90_dist=("dist", lambda x: x.quantile(0.9))).reset_index()
    by.sort_values("n", ascending=False).to_csv(OUT / "tables/p1b_actor_vs_event_location_by_type.csv", index=False)
    pr = pd.DataFrame([(a, b, s, n) for (a, b, s), n in pairs.items()], columns=["first", "second", "identical_snapshot", "n"])
    tot = pr.groupby(["first", "second"])["n"].transform("sum")
    pr["share_of_pair"] = pr["n"] / tot
    pr = pr[pr["identical_snapshot"]].sort_values("n", ascending=False)
    pr.head(40).to_csv(OUT / "tables/p1b_shared_snapshot_pairs.csv", index=False)
    s = {"360_events_with_actor_and_location": int(len(d)),
         "share_actor_point_equals_event_location_lt_0_05": float((d["dist"] < 0.05).mean()),
         "share_actor_point_equals_mirrored_location": float((d["dist_mirrored"] < 0.05).mean()),
         "consecutive_raw_pairs_with_identical_snapshot": int(pr["n"].sum())}
    (OUT / "summary_actor.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    print(by.sort_values("n", ascending=False).head(18).to_string(index=False))
    print(pr.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
