"""Read-only investigation of `Aerial Lost` duels and the proposed companion rule K4 (Tier B).

Candidate rule K4 (NOT active in any task version; implemented here only):
    a Duel with subtype 'Aerial Lost' is a companion record when ALL of
      (a) its related_events contain a record with `aerial_won = true`   (explicit relationship),
      (b) that record has the same timestamp                                (same moment),
      (c) that record belongs to the opposite team                          (two contestants),
      (d) that record is the adjacent raw record (index +-1)               (documented pattern),
      (e) that record is a retained canonical action (Pass/Clearance/Shot/Miscontrol ...).

Compares the canonical task actions15_360current_v1 with a hypothetical variant "+K4".
Reads only development caches and the raw JSON of Tier B matches; final-test matches are
never opened. Writes ../tables/*.csv, ../examples/*.csv, ../summary.json.

Usage: .venv/bin/python results/aerial_lost_k4_investigation_2026-11-19/diagnostics/k4_investigation.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sequence_models.canonical_task import (  # noqa: E402
    CLASSES, COMPANION, build_canonical_table, load_development_raw, load_task_spec, task_spec_for_builder)
from src.sequence_models.raw_events import EVENTS_DIR  # noqa: E402
from src.sequence_models.tasks import build_task_table  # noqa: E402
from src.sequence_models.v3_taxonomy import companion_rules  # noqa: E402


def raw_extra(mid: int) -> list[dict]:
    """player id and aerial_won flag per event (fields not in the project's caches)."""
    rows = []
    for e in json.loads((EVENTS_DIR / f"{mid}.json").read_text(encoding="utf-8")):
        won = any(isinstance(v, dict) and v.get("aerial_won") for v in e.values())
        rows.append({"event_id": e["id"], "player_id": (e.get("player") or {}).get("id"), "aerial_won": bool(won)})
    return rows


def k4_flags(raw: pd.DataFrame) -> pd.DataFrame:
    """Per Aerial Lost record: the evidence for each K4 criterion."""
    by_id = raw.set_index("event_id")
    pos = raw.groupby("match_id").cumcount()
    raw = raw.assign(pos=pos)
    al = raw[(raw["type"] == "Duel") & (raw["subtype"] == "Aerial Lost")]
    out = []
    for _, r in al.iterrows():
        rel = [x for x in str(r["related_events"]).split(";") if x and x in by_id.index]
        won = [x for x in rel if by_id.at[x, "aerial_won"]]
        w = by_id.loc[won[0]] if won else None
        rec = {"event_id": r["event_id"], "match_id": r["match_id"], "period": r["period"], "index": r["index"],
               "timestamp": r["timestamp"], "team_id": r["team_id"], "player_id": r["player_id"],
               "has_360": bool(r["has_360"] and r["has_freeze_frame"]),
               "n_related": len(rel), "related_types": ",".join(sorted(by_id.at[x, "type"] for x in rel)),
               "a_linked_aerial_won": w is not None}
        if w is not None:
            rec.update({"winner_event_id": won[0], "winner_type": w["type"], "winner_index": int(w["index"]),
                        "winner_timestamp": w["timestamp"], "winner_team_id": w["team_id"], "winner_player_id": w["player_id"],
                        "winner_has_360": bool(w["has_360"] and w["has_freeze_frame"]),
                        "b_same_timestamp": w["t_seconds"] == r["t_seconds"],
                        "c_opposite_team": w["team_id"] != r["team_id"],
                        "d_adjacent_raw": abs(int(w["index"]) - int(r["index"])) == 1,
                        "winner_first": int(w["index"]) < int(r["index"]),
                        "e_winner_is_canonical_action": w["type"] in CLASSES,
                        "back_link": r["event_id"] in str(w["related_events"]).split(";")})
        out.append(rec)
    f = pd.DataFrame(out)
    for c in ("b_same_timestamp", "c_opposite_team", "d_adjacent_raw", "e_winner_is_canonical_action", "back_link", "winner_first"):
        f[c] = f[c].fillna(False).astype(bool)
    f["k4_all_criteria"] = f[["a_linked_aerial_won", "b_same_timestamp", "c_opposite_team", "d_adjacent_raw",
                              "e_winner_is_canonical_action"]].all(axis=1)
    return f


def variant_table(raw: pd.DataFrame, extra_companions: set[str]) -> pd.DataFrame:
    """Canonical construction with additional companion records (the hypothetical +K4 variant)."""
    raw = raw.sort_values(["match_id", "index"], kind="stable").reset_index(drop=True)
    rules = companion_rules(raw)
    rel = raw.copy()
    rel.loc[((rules != "") | raw["event_id"].isin(extra_companions)).to_numpy(), "type"] = COMPANION
    t = build_task_table(rel, task_spec_for_builder())
    usable = raw.set_index("event_id")
    usable = usable["has_360"].astype(bool) & usable["has_freeze_frame"].astype(bool)
    t["usable_360"] = t["event_id"].map(usable).astype(bool).to_numpy()
    t["row_eligible"] = (t["target_id"].to_numpy() >= 0) & t["usable_360"].to_numpy()
    return t


def summary_of(t: pd.DataFrame, name: str) -> dict:
    tr = t[t["target_id"] >= 0]
    ev = t[t["row_eligible"]]
    cell = tr.groupby(["symbol", "target"]).size()
    p = cell / tr.groupby("symbol").size().reindex(cell.index.get_level_values(0)).to_numpy()
    h = lambda c: float(-(c / c.sum() * np.log2(c / c.sum())).sum())  # noqa: E731
    hc = float(sum(len(x) / len(ev) * h(x["target"].value_counts()) for _, x in ev.groupby("symbol")))
    return {"variant": name, "observations": len(t), "transitions": len(tr), "eligible_rows": len(ev),
            "classes_with_observations": int(t["symbol"].nunique()), "duel_observations": int((t["symbol"] == "Duel").sum()),
            "duel_eligible_rows_as_current": int((ev["symbol"] == "Duel").sum()),
            "duel_eligible_rows_as_target": int((ev["target"] == "Duel").sum()),
            "rows_in_cells_P_ge_0_9": float(cell[p >= 0.9].sum() / len(tr)),
            "max_cell_probability_n_ge_50": float(p[cell >= 50].max()),
            "H_target_bits": h(ev["target"].value_counts()), "H_target_given_current_bits": hc}


def main() -> None:
    spec = load_task_spec()
    raw_dev, info = load_development_raw(spec)
    part = info["partition"]
    raw = raw_dev[raw_dev["match_id"].isin(part.tier_b)].copy()
    assert not set(raw["match_id"]) & set(part.final_test)
    with ProcessPoolExecutor(6) as pool:
        extra = pd.DataFrame([r for chunk in pool.map(raw_extra, sorted(part.tier_b)) for r in chunk])
    raw = raw.merge(extra, on="event_id", how="left", validate="one_to_one")
    raw = raw.sort_values(["match_id", "index"]).reset_index(drop=True)
    f = k4_flags(raw[raw["period"] != 5])
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "examples").mkdir(exist_ok=True)
    f.to_csv(OUT / "tables/aerial_lost_records_with_k4_evidence.csv", index=False)
    s = {"aerial_lost_records_periods_1_4": int(len(f))}
    for c in ("a_linked_aerial_won", "b_same_timestamp", "c_opposite_team", "d_adjacent_raw", "e_winner_is_canonical_action",
              "back_link", "winner_first", "k4_all_criteria"):
        s[c] = {"n": int(f[c].sum()), "pct": float(100 * f[c].mean())}
    s["related_types_distribution"] = f["related_types"].value_counts().head(8).to_dict()
    s["winner_types"] = f["winner_type"].value_counts().to_dict()
    s["aerial_lost_with_usable_360"] = int(f["has_360"].sum())
    s["winner_with_usable_360_when_loser_has_360"] = float(f.loc[f["has_360"], "winner_has_360"].mean())
    s["same_player_both_records"] = int((f["player_id"] == f["winner_player_id"]).sum())
    # all duel subtypes in the canonical v1 sequence (K2 already removes linked tackles)
    canon = build_canonical_table(raw.drop(columns=["player_id", "aerial_won"]))
    duel_sub = canon[canon["symbol"] == "Duel"].merge(raw[["event_id", "subtype"]], on="event_id")["subtype"].value_counts()
    s["canonical_v1_duel_subtypes"] = duel_sub.to_dict()
    # immediate transitions in the canonical v1 sequence
    nxt = canon.assign(target_event_id=canon["target_event_id"])
    al_ids = set(f["event_id"])
    win = dict(zip(f["event_id"], f["winner_event_id"]))
    rows_al = nxt[nxt["event_id"].isin(al_ids)]
    s["v1_aerial_lost_next_is_its_winner"] = int((rows_al["target_event_id"] == rows_al["event_id"].map(win)).sum())
    win_ids = {v: k for k, v in win.items() if isinstance(v, str)}
    rows_w = nxt[nxt["event_id"].isin(win_ids)]
    s["v1_winner_next_is_its_aerial_lost"] = int((rows_w["target_event_id"] == rows_w["event_id"].map(win_ids)).sum())
    s["v1_next_after_aerial_lost"] = rows_al["target"].value_counts(normalize=True).round(4).head(6).to_dict()
    s["v1_previous_before_aerial_lost"] = canon.groupby("sequence_id")["symbol"].shift(1)[canon["event_id"].isin(al_ids)].value_counts(normalize=True).round(4).head(6).to_dict()
    # variant comparison
    k4_ids = set(f.loc[f["k4_all_criteria"], "event_id"])
    var = variant_table(raw.drop(columns=["player_id", "aerial_won"]), k4_ids)
    comp = pd.DataFrame([summary_of(canon, "actions15_360current_v1 (K4 off)"), summary_of(var, "hypothetical +K4")])
    comp.to_csv(OUT / "tables/variant_comparison.csv", index=False)
    m = canon[canon["row_eligible"]][["event_id", "target_event_id", "target"]].merge(
        var[var["row_eligible"]][["event_id", "target_event_id", "target"]], on="event_id", how="left", suffixes=("_v1", "_k4"))
    s["k4_effect"] = {"records_removed": len(k4_ids),
                      "v1_eligible_rows_removed_as_current": int(m["target_event_id_k4"].isna().sum()),
                      "v1_eligible_rows_target_event_changed": int(((m["target_event_id_k4"] != m["target_event_id_v1"]) & m["target_event_id_k4"].notna()).sum()),
                      "v1_eligible_rows_target_class_changed": int(((m["target_k4"] != m["target_v1"]) & m["target_k4"].notna()).sum()),
                      "eligible_rows_v1": int(canon["row_eligible"].sum()), "eligible_rows_k4": int(var["row_eligible"].sum())}
    dist = pd.DataFrame({"v1_eligible_targets": canon.loc[canon["row_eligible"], "target"].value_counts(),
                         "k4_eligible_targets": var.loc[var["row_eligible"], "target"].value_counts()}).reindex(list(CLASSES)).fillna(0).astype(int)
    dist["v1_share_pct"] = 100 * dist["v1_eligible_targets"] / dist["v1_eligible_targets"].sum()
    dist["k4_share_pct"] = 100 * dist["k4_eligible_targets"] / dist["k4_eligible_targets"].sum()
    dist.rename_axis("class").reset_index().to_csv(OUT / "tables/class_distribution_v1_vs_k4.csv", index=False)
    # examples: 12 records + their linked winner, sampled across winner types and order
    rng = np.random.default_rng(20261119)
    picks = []
    for typ in ("Pass", "Clearance", "Shot", "Miscontrol"):
        idx = f.index[(f["winner_type"] == typ) & ~f["winner_first"]]
        picks += list(rng.choice(idx, size=min(2, len(idx)), replace=False))
    idx = f.index[f["winner_first"]]
    picks += list(rng.choice(idx, size=4, replace=False))
    ex = []
    cols = ["match_id", "period", "index", "event_id", "timestamp", "type", "subtype", "team_id", "player_id",
            "aerial_won", "related_events", "has_360"]
    for n, i in enumerate(picks, 1):
        r = f.loc[i]
        lo, hi = min(r["index"], r["winner_index"]) - 1, max(r["index"], r["winner_index"]) + 1
        w = raw[(raw["match_id"] == r["match_id"]) & raw["index"].between(lo, hi)][cols].copy()
        w.insert(0, "role", np.where(w["event_id"] == r["event_id"], "AERIAL LOST",
                                     np.where(w["event_id"] == r["winner_event_id"], "WINNER (aerial_won)", "context")))
        w.insert(0, "example", n)
        ex.append(w)
    pd.concat(ex).to_csv(OUT / "examples/aerial_lost_examples.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))
    print(comp.T.to_string())


if __name__ == "__main__":
    main()
