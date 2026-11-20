"""Read-only: residual paired records under actions15 and same-timestamp transitions (Tier B).

Reuses the raw parser of investigate.py (same directory). Writes ../tables/p3b_*.csv,
../examples/ex_residual_pairs.csv and ../summary_pairs.json.

Usage: .venv/bin/python results/investigation_360_taxonomy_2026-11-18/diagnostics/pairs_extra.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from investigate import D15, EXAMPLES, OUT, TABLES, companion_flags, load, save, sequence, window  # noqa: E402


def linked(ids, related) -> np.ndarray:
    return np.array([isinstance(i, str) and i in r.split(";") for i, r in zip(ids, related)])


def main() -> None:
    ev = load()
    rules = companion_flags(ev)
    s = sequence(ev, (ev["type"].isin(D15) & (rules == "")).to_numpy(), "actions15")
    s["next_related_back"] = linked(s["event_id"], s["next_event_id"].map(ev.set_index("event_id")["related_events"]).fillna(""))
    s["link_next"] = linked(s["next_event_id"], s["related_events"]) | s["next_related_back"]
    tr = s[s["has_target"]].copy()
    tr["same_ts"] = tr["next_t"] == tr["t"]
    tr["same_team"] = tr["next_team_id"] == tr["team_id"]
    out = {}
    # 1. transitions at an identical timestamp: linked (one contest) vs unlinked (separate records)
    z = tr[tr["same_ts"]]
    out["actions15_transitions"] = int(len(tr))
    out["same_timestamp_transitions"] = int(len(z))
    out["same_timestamp_share"] = float(len(z) / len(tr))
    out["same_timestamp_linked_share"] = float(z["link_next"].mean())
    top = z.groupby(["type", "next_type", "link_next", "same_team"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    save(top.head(40), "p3b_actions15_same_timestamp_pairs.csv")
    # 2. residual two-record contests kept by actions15
    rows = []
    for a, b, cond, label in [
        ("Duel", None, lambda x: x["subtype"] == "Aerial Lost", "Aerial Lost -> linked aerial winner record"),
        ("Shot", "Goal Keeper", lambda x: x["next_subtype"].isin(["Shot Saved", "Shot Saved to Post", "Shot Saved Off Target", "Penalty Saved", "Penalty Saved to Post"]), "Shot -> keeper save"),
        ("Shot", "Block", lambda x: x["outcome"] == "Blocked", "Blocked shot -> Block"),
        ("Dribble", None, lambda x: x["outcome"] == "Incomplete", "Incomplete Dribble -> next"),
        ("Pass", "Interception", lambda x: x["outcome"] == "Incomplete", "Incomplete pass -> Interception"),
    ]:
        x = tr[(tr["type"] == a) & cond(tr)]
        y = x if b is None else x[x["next_type"] == b]
        rows.append({"pattern": label, "n_first": int(len(x)), "n_pairs": int(len(y)),
                     "P(second|first)": float(len(y) / max(len(x), 1)),
                     "next_types_top": x["next_type"].value_counts(normalize=True).head(4).round(3).to_dict(),
                     "same_timestamp": float(y["same_ts"].mean()) if len(y) else np.nan,
                     "linked": float(y["link_next"].mean()) if len(y) else np.nan,
                     "opposite_team": float((~y["same_team"]).mean()) if len(y) else np.nan,
                     "dt_median_s": float((y["next_t"] - y["t"]).median()) if len(y) else np.nan})
    # aerial: where is the winner's record relative to Aerial Lost (raw neighbours)?
    al = ev[(ev["type"] == "Duel") & (ev["subtype"] == "Aerial Lost")]
    by_id = ev.set_index("event_id")
    rel = []
    for _, r in al.iterrows():
        w = [x for x in r["related_events"].split(";") if x in by_id.index and by_id.at[x, "aerial_won"]]
        if w:
            o = by_id.loc[w[0]]
            rel.append({"winner_type": o["type"], "winner_before": o["pos"] < r["pos"], "dt": r["t"] - o["t"],
                        "raw_distance": int(abs(o["pos"] - r["pos"])), "winner_has_360": bool(o["has_360"]),
                        "loser_has_360": bool(r["has_360"])})
    rel = pd.DataFrame(rel)
    out["aerial_lost_with_linked_winner"] = {"n_aerial_lost": int(len(al)), "with_linked_aerial_won_record": int(len(rel)),
                                            "winner_record_before_loser": float(rel["winner_before"].mean()),
                                            "same_timestamp": float((rel["dt"] == 0).mean()),
                                            "adjacent_raw_records": float((rel["raw_distance"] == 1).mean()),
                                            "winner_types": rel["winner_type"].value_counts().to_dict()}
    save(pd.DataFrame(rows), "p3b_residual_pairs_actions15.csv")
    # 3. examples: separate actions at identical timestamps (unlinked) vs linked residual pairs
    ex = []
    rng = np.random.default_rng(5)
    def pick(mask, tag, n):
        idx = np.flatnonzero(mask)
        for j in rng.choice(idx, size=min(n, len(idx)), replace=False):
            i = ev.index[ev["event_id"] == tr["event_id"].iat[j]][0]
            ex.append(window(ev, i, 1, 2, tag))
    pick((tr["same_ts"] & ~tr["link_next"]).to_numpy(), "same timestamp, NOT linked (separate records)", 5)
    pick((tr["same_ts"] & tr["link_next"] & (tr["subtype"] == "Aerial Lost")).to_numpy(), "residual pair: Aerial Lost + aerial winner", 3)
    pick(((tr["type"] == "Shot") & (tr["next_subtype"] == "Shot Saved")).to_numpy(), "residual pair: Shot -> Shot Saved", 2)
    save(pd.concat(ex), "ex_residual_pairs_and_same_timestamp.csv", EXAMPLES)
    (OUT / "summary_pairs.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
