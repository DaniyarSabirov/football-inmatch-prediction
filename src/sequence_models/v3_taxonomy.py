"""Experiment 3: taxonomy audit and the meaningful-action taxonomy ``actions15``.

Taxonomies (all filtered-event tasks except ``raw33``, see ``methodology.md``):

- ``filtered8``   : the 8 production types (experiments 1-2).
- ``scenarioD15`` : Scenario D, 15 types (experiment 2's recommended taxonomy).
- ``actions15``   : Scenario D's 15 class names, but records that are an *annotation of the same
  contest* as their predecessor are not separate actions. Measured rules (development audit):

  K1  Goal Keeper records without a keeper action: subtype Shot Faced, Goal Conceded or
      Penalty Conceded (outcome markers of the preceding shot; Shot Faced has no outcome at all).
      Saves, claims, punches, sweeping and smothers stay "Goal Keeper".
  K2  a Duel/Tackle that is the companion of the immediately preceding Dispossessed or
      incomplete Dribble (same timestamp, linked in related_events; 100% of those Dispossessed and
      4,280 of 4,830 incomplete Dribbles). The contest is represented once, by its first record.
  K3  the second record of a 50/50 pair (immediately preceding 50/50, same timestamp, linked).

  Skipped companions become non-vocabulary records, so they are counted in
  ``n_skipped_raw_events`` like every other skipped record. Composite labelling (option B) and
  skipping (option C) give the same sequence; the class keeps the first record's name.
- ``raw33``       : every raw type, immediate next raw record (diagnostic reference only).
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT

from .config import TaskSpec
from .tasks import build_task_table

CATEGORIES = json.loads((PROJECT_ROOT / "results" / "event_taxonomy_research_2026-10-27" /
                         "event_type_categories.json").read_text(encoding="utf-8"))["types"]
FILTERED8 = ("Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed")
D15 = FILTERED8 + ("Pressure", "Duel", "Interception", "Block", "Goal Keeper", "Foul Committed", "50/50")
GK_NO_ACTION = ("Shot Faced", "Goal Conceded", "Penalty Conceded")
COMPANION = "__companion__"


def companion_rules(t: pd.DataFrame) -> pd.Series:
    """K1/K2/K3 label per raw event ('' if none). ``t``: raw event table + details, raw order."""
    g = t.groupby(["match_id", "period"], sort=False)
    prev_type, prev_t = g["type"].shift(1), g["t_seconds"].shift(1)
    prev_id, prev_outcome = g["event_id"].shift(1), g["outcome"].shift(1)
    related = t["related_events"].fillna("")
    linked = np.array([isinstance(p, str) and p in r for p, r in zip(prev_id, related)])
    same_ts = (prev_t == t["t_seconds"]).to_numpy()
    k1 = (t["type"] == "Goal Keeper") & t["subtype"].isin(GK_NO_ACTION)
    k2 = ((t["type"] == "Duel") & (t["subtype"] == "Tackle") & same_ts & linked
          & ((prev_type == "Dispossessed") | ((prev_type == "Dribble") & (prev_outcome == "Incomplete"))))
    k3 = (t["type"] == "50/50") & (prev_type == "50/50") & same_ts & linked
    return pd.Series(np.select([k1, k2, k3], ["K1", "K2", "K3"], ""), index=t.index)


def taxonomy_spec(name: str, all_types: list[str] | None = None) -> TaskSpec:
    if name == "filtered8":
        types = FILTERED8
    elif name in ("scenarioD15", "actions15"):
        types = D15
    elif name == "raw33":
        types = tuple(sorted(all_types))
    else:
        raise ValueError(name)
    return TaskSpec(name, name, types, "vocabulary_only", None, types, (5,))


def taxonomy_event_table(t: pd.DataFrame, name: str) -> pd.DataFrame:
    """Event table for one taxonomy: for actions15, companion records are renamed COMPANION
    (skipped by definition). Other columns unchanged."""
    if name != "actions15":
        return t
    out = t.copy()
    rules = companion_rules(out)
    out.loc[rules != "", "type"] = COMPANION
    return out


def taxonomy_table(t: pd.DataFrame, name: str) -> pd.DataFrame:
    spec = taxonomy_spec(name, sorted(set(t["type"]) - {COMPANION}))
    return build_task_table(taxonomy_event_table(t, name), spec)


def audit(t: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(taxonomy summary, skipped-record categories, class distribution) on periods 1-4."""
    t = t[t["period"] < 5].reset_index(drop=True)
    rules = companion_rules(t)
    t_by_id = t.set_index("event_id")
    summary, skipped_rows, dist = [], [], []
    n_raw = len(t)
    for name in names:
        tt = taxonomy_table(t, name)
        ev = tt[tt["is_prediction_point"]]
        nxt_t = tt.groupby("sequence_id")["t_seconds"].shift(-1)
        gap = (nxt_t - tt["t_seconds"]).loc[ev.index]
        counts = ev["target"].value_counts()
        cells = ev.groupby(["symbol", "target"]).size()
        cell_p = cells / ev.groupby("symbol").size().reindex(cells.index.get_level_values(0)).to_numpy()
        h = lambda c: float(-(c / c.sum() * np.log2(c / c.sum())).sum())  # noqa: E731
        h_cond = float(sum(len(x) / len(ev) * h(x["target"].value_counts()) for _, x in ev.groupby("symbol")))
        skipped = tt["n_skipped_raw_events"].astype(float).loc[ev.index]
        summary.append({
            "taxonomy": name, "n_classes": int(counts.size), "observations": int(len(tt)),
            "prediction_points": int(len(ev)), "raw_event_coverage": len(tt) / n_raw,
            "rows_skipping_raw_events": float((skipped >= 1).mean()), "mean_skipped": float(skipped.mean()),
            "zero_gap_rows": float((gap == 0).mean()), "gap_le_0_1s_rows": float((gap.abs() <= 0.1).mean()),
            "gap_le_0_5s_rows": float((gap.abs() <= 0.5).mean()), "negative_gap_rows": float((gap < 0).mean()),
            "majority_class": str(counts.idxmax()), "majority_share": float(counts.max() / counts.sum()),
            "smallest_class": str(counts.idxmin()), "smallest_class_rows": int(counts.min()),
            "imbalance_ratio": float(counts.max() / counts.min()),
            "target_entropy_bits": h(counts), "H_target_given_current_bits": h_cond,
            "rows_in_cells_p_ge_0_9": float(cells[cell_p >= 0.9].sum() / len(ev)),
            "rows_in_cells_p_ge_0_99": float(cells[cell_p >= 0.99].sum() / len(ev)),
            "near_deterministic_cells": "; ".join(f"{a} -> {b} ({p:.3f}, n={cells[(a, b)]:,})"
                                                 for (a, b), p in cell_p[cell_p >= 0.9].items() if cells[(a, b)] >= 50),
        })
        # what is skipped between an observation and its target
        pos = pd.Series(np.arange(n_raw), index=t["event_id"])
        start, end = pos.loc[ev["event_id"]].to_numpy(), pos.loc[ev["target_event_id"]].to_numpy()
        types, rl = t["type"].to_numpy(), rules.to_numpy()
        c = Counter()
        for s, e in zip(start, end):
            for j in range(s + 1, e):
                c[(types[j], rl[j])] += 1
        for (typ, rule), n in c.most_common():
            skipped_rows.append({"taxonomy": name, "skipped_type": typ,
                                 "category": CATEGORIES.get(typ, "unknown"),
                                 "skipped_as_companion_rule": rule or "", "n": n})
        for cls, n in counts.items():
            dist.append({"taxonomy": name, "class": cls, "rows": int(n), "share": float(n / counts.sum())})
    del t_by_id
    return pd.DataFrame(summary), pd.DataFrame(skipped_rows), pd.DataFrame(dist)


def annotation_pairs(t: pd.DataFrame) -> pd.DataFrame:
    """Raw-stream pairs A -> next raw record B: probability, same timestamp, StatsBomb link, B's subtypes."""
    t = t[t["period"] < 5]
    g = t.groupby(["match_id", "period"], sort=False)
    nt, nsub, ntime, nid = g["type"].shift(-1), g["subtype"].shift(-1), g["t_seconds"].shift(-1), g["event_id"].shift(-1)
    linked = np.array([isinstance(i, str) and i in (r or "") for i, r in zip(nid, t["related_events"])])
    rows = []
    for a, b in [("Dispossessed", "Duel"), ("Dribble", "Duel"), ("Shot", "Goal Keeper"), ("Shot", "Block"),
                 ("Foul Committed", "Foul Won"), ("Dribbled Past", "Dribble"), ("Pass", "Ball Receipt*"),
                 ("50/50", "50/50"), ("Miscontrol", "Ball Recovery"), ("Interception", "Ball Recovery")]:
        is_a = (t["type"] == a).to_numpy()
        hit = is_a & (nt == b).to_numpy()
        rows.append({"first": a, "next_raw": b, "n_first": int(is_a.sum()), "p_next_is_b": hit.sum() / max(is_a.sum(), 1),
                     "same_timestamp_given_pair": float((ntime == t["t_seconds"]).to_numpy()[hit].mean()) if hit.any() else np.nan,
                     "linked_in_related_events_given_pair": float(linked[hit].mean()) if hit.any() else np.nan,
                     "subtypes_of_b": "; ".join(f"{k} {v:.3f}" for k, v in nsub[hit].value_counts(normalize=True).head(6).items())})
    return pd.DataFrame(rows)


def event_mapping(t: pd.DataFrame) -> pd.DataFrame:
    """(type, subtype) -> class in every taxonomy, with counts."""
    rules = companion_rules(t)
    m = t.assign(rule=rules).groupby(["type", "subtype", "rule"], dropna=False).size().rename("n").reset_index()
    m["category"] = m["type"].map(CATEGORIES)
    m["filtered8"] = np.where(m["type"].isin(FILTERED8), m["type"], "skipped")
    m["scenarioD15"] = np.where(m["type"].isin(D15), m["type"], "skipped")
    m["actions15"] = np.where(m["type"].isin(D15) & (m["rule"] == ""), m["type"],
                              np.where(m["rule"] != "", "skipped (companion " + m["rule"] + ")", "skipped"))
    m["raw33"] = m["type"]
    return m.sort_values(["type", "n"], ascending=[True, False])
