"""Measure every StatsBomb event type and the raw event-transition structure
(417 eligible matches; raw data read-only; nothing in production touched).

Outputs (../):
  type_stats.csv        per type: count, % of events, matches, with 360 record,
                        with non-empty freeze frame, actor point in its own
                        snapshot, has location, has player, category
  transitions.csv       immediate raw transitions (current type -> next raw
                        event in the same period, event order): count and
                        share at the same timestamp (gap 0)
  companions.csv        per type: share of occurrences whose immediately
                        preceding raw event (same period) is at the SAME
                        timestamp, and the most common such predecessor --
                        evidence for "derived companion" events
  raw_counts.json       totals
"""
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.animate_statsbomb_360_match import available_360_match_ids  # noqa: E402
from src.match_exclusions import load_exclusion_manifest  # noqa: E402

DATA = ROOT / "data" / "raw" / "statsbomb" / "data"


def ms(ts):
    h, m, rest = ts.split(":")
    s, _, f = rest.partition(".")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int((f + "000")[:3])


def per_match(match_id):
    events = json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    stats = defaultdict(Counter)
    trans = Counter()
    trans_zero = Counter()
    companion = defaultdict(Counter)
    for i, e in enumerate(events):
        t = e["type"]["name"]
        s = stats[t]
        s["count"] += 1
        s["has_location"] += isinstance(e.get("location"), list)
        s["has_player"] += isinstance(e.get("player"), dict)
        f = frames.get(e["id"])
        if f is not None:
            s["with_360_record"] += 1
            ff = f.get("freeze_frame") or []
            if ff:
                s["nonempty_freeze_frame"] += 1
                s["actor_point"] += any(isinstance(p, dict) and p.get("actor") for p in ff)
        if i + 1 < len(events) and events[i + 1]["period"] == e["period"]:
            n = events[i + 1]
            same = ms(n["timestamp"]) == ms(e["timestamp"])
            trans[(t, n["type"]["name"])] += 1
            trans_zero[(t, n["type"]["name"])] += same
        if i > 0 and events[i - 1]["period"] == e["period"]:
            p = events[i - 1]
            if ms(p["timestamp"]) == ms(e["timestamp"]):
                companion[t][p["type"]["name"]] += 1
    return match_id, {k: dict(v) for k, v in stats.items()}, dict(trans), dict(trans_zero), {k: dict(v) for k, v in companion.items()}


def main():
    excluded = load_exclusion_manifest()
    ids = [m for m in available_360_match_ids() if m not in excluded]
    with ProcessPoolExecutor(max_workers=7) as pool:
        results = list(pool.map(per_match, ids, chunksize=2))
    cats = json.loads((OUT / "event_type_categories.json").read_text())["types"]
    stats, matches = defaultdict(Counter), Counter()
    trans, trans_zero, companion = Counter(), Counter(), defaultdict(Counter)
    for _, s, t, tz, c in results:
        for k, v in s.items():
            stats[k].update(v)
            matches[k] += 1
        trans.update(t)
        trans_zero.update(tz)
        for k, v in c.items():
            companion[k].update(v)
    total = sum(v["count"] for v in stats.values())
    rows = []
    for t, v in sorted(stats.items(), key=lambda x: -x[1]["count"]):
        n = v["count"]
        rows.append({
            "event_type": t, "category": cats.get(t, "UNCATEGORISED"), "count": n, "pct_of_events": round(100 * n / total, 4),
            "matches": matches[t], "with_360_record": v["with_360_record"], "pct_with_360_record": round(100 * v["with_360_record"] / n, 2),
            "without_360_record": n - v["with_360_record"], "nonempty_freeze_frame": v["nonempty_freeze_frame"],
            "actor_point_in_own_snapshot": v["actor_point"],
            "pct_actor_point_of_360": round(100 * v["actor_point"] / v["nonempty_freeze_frame"], 2) if v["nonempty_freeze_frame"] else None,
            "pct_has_location": round(100 * v["has_location"] / n, 2), "pct_has_player": round(100 * v["has_player"] / n, 2),
        })
    pd.DataFrame(rows).to_csv(OUT / "type_stats.csv", index=False)
    pd.DataFrame([{"current": a, "next": b, "count": c, "same_timestamp": trans_zero[(a, b)]} for (a, b), c in trans.items()]
                 ).sort_values("count", ascending=False).to_csv(OUT / "transitions.csv", index=False)
    comp_rows = []
    for t, v in stats.items():
        c = companion.get(t, Counter())
        same = sum(c.values())
        top, top_n = (c.most_common(1)[0] if c else ("", 0))
        comp_rows.append({"event_type": t, "count": v["count"], "same_timestamp_predecessor": same,
                          "pct_same_timestamp_predecessor": round(100 * same / v["count"], 2),
                          "top_same_timestamp_predecessor": top, "pct_with_top_predecessor": round(100 * top_n / v["count"], 2)})
    pd.DataFrame(comp_rows).sort_values("count", ascending=False).to_csv(OUT / "companions.csv", index=False)
    raw = {"matches": len(ids), "events": total, "event_types": len(stats),
           "uncategorised_types": [t for t in stats if t not in cats],
           "raw_transitions_within_period": sum(trans.values()),
           "raw_transitions_same_timestamp": sum(trans_zero.values())}
    (OUT / "raw_counts.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
    print(json.dumps(raw, indent=1))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(comp_rows).sort_values("count", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
