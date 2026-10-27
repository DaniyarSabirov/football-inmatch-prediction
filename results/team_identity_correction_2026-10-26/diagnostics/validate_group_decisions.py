"""Independent cross-checks of the copy/derived-group view decisions.

Each decision rule uses some signals; the others are held out here:
  continuity decisions  -> agreement with the coordinate owner and, where
                           the keeper did not veto, the goalkeeper
  keeper_and_owner      -> agreement with continuity is impossible (none);
                           agreement with the legacy Problem C source
  legacy                -> agreement with the goalkeeper (not available by
                           construction when it disagrees) -- reported only
Also: how many continuity/keeper decisions overturned the legacy choice and
what the goalkeeper says about those overturns.
Output: ../tables/group_decision_validation.csv (+ printed)
"""
import re
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1]
rec = pd.read_csv(OUT / "records.csv.gz", usecols=["match_id", "event_id", "decision", "decision_evidence", "linked"])
g = rec[rec.decision.isin(["continuity", "keeper_and_owner", "legacy_problem_c_source", "undecided"])].copy()
# one row per group: the evidence string is identical for every member
g = g.drop_duplicates(["match_id", "decision_evidence", "decision"]) if False else g


def parse(ev):
    d = dict(part.split("=", 1) for part in ev.split(";"))
    return pd.Series({k: d.get(k, "-") for k in ("continuity", "keeper", "owner", "legacy")})


g = pd.concat([g, g.decision_evidence.apply(parse)], axis=1)
definite = lambda v: v not in ("-", "mixed", "conflicting") and not v.startswith("lean")  # noqa: E731
rows = []
for decision, sub in g.groupby("decision"):
    for signal in ("continuity", "keeper", "owner", "legacy"):
        for other in ("continuity", "keeper", "owner", "legacy"):
            if other <= signal:
                continue
            both = sub[sub[signal].map(definite) & sub[other].map(definite)]
            if len(both):
                rows.append({"decision": decision, "signal_a": signal, "signal_b": other, "snapshots_with_both": len(both),
                             "agree": int((both[signal] == both[other]).sum()),
                             "agree_pct": round(100 * (both[signal] == both[other]).mean(), 2)})
table = pd.DataFrame(rows)
table.to_csv(OUT / "tables" / "group_decision_validation.csv", index=False)
print(table.to_string())
cont = g[g.decision == "continuity"]
overturn = cont[cont.legacy.map(definite) & (cont.continuity != cont.legacy)]
print("continuity decisions overturning the legacy source:", len(overturn),
      "keeper available:", int(overturn.keeper.map(definite).sum()),
      "keeper agrees with continuity:", int((overturn.keeper == overturn.continuity).sum()))
