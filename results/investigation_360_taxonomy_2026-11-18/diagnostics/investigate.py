"""Read-only investigation: 360 coverage, paired records and target taxonomy (Tier B, 304 matches).

Reads ONLY: results/markov_hmm_pilot/split_manifest.json (match lists),
experiment 2's Tier B folds, experiment 4's per-class metrics, and the raw StatsBomb
events / three-sixty JSON of the 304 Tier B matches. Final-test matches are never opened
(asserted). The raw stream is parsed here independently of the project pipeline; the
project's own actions15 rule (src/sequence_models/v3_taxonomy.companion_rules) is imported
only to compare it with the independent re-implementation below.

Writes CSV/JSON into ../tables, ../examples and ../summary.json (this directory's parent).

Usage (from the repository root, ~1-2 min):
    .venv/bin/python results/investigation_360_taxonomy_2026-11-18/diagnostics/investigate.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/statsbomb/data"
TABLES, EXAMPLES = OUT / "tables", OUT / "examples"
D15 = ("Pass", "Carry", "Ball Recovery", "Clearance", "Dribble", "Shot", "Miscontrol", "Dispossessed",
       "Pressure", "Duel", "Interception", "Block", "Goal Keeper", "Foul Committed", "50/50")
C12 = tuple(c for c in D15 if c not in ("Goal Keeper", "Duel", "50/50"))
GK_NO_ACTION = ("Shot Faced", "Goal Conceded", "Penalty Conceded")
STANDARD_KEYS = {"id", "index", "period", "timestamp", "minute", "second", "type", "possession", "possession_team",
                 "play_pattern", "team", "player", "position", "location", "duration", "related_events",
                 "under_pressure", "off_camera", "out", "counterpress", "tactics"}
SUMMARY: dict = {}


# ============================================================================ parsing
def _secs(ts: str) -> float:
    d = datetime.strptime(ts, "%H:%M:%S.%f")
    return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6


def parse(mid: int) -> list[dict]:
    events = json.loads((RAW / "events" / f"{mid}.json").read_text(encoding="utf-8"))
    frames = {}
    for r in json.loads((RAW / "three-sixty" / f"{mid}.json").read_text(encoding="utf-8")):
        ff = r.get("freeze_frame") or []
        actors = [p for p in ff if p.get("actor")]
        frames[r["event_uuid"]] = (len(ff), len(actors), actors[0]["location"] if actors else None,
                                   len(r.get("visible_area") or []))
    rows = []
    for pos, e in enumerate(events):
        detail_key = next((k for k, v in e.items() if k not in STANDARD_KEYS and isinstance(v, dict)), None)
        detail = e.get(detail_key, {}) if detail_key else {}
        sub, out = detail.get("type"), detail.get("outcome")
        loc = e.get("location") if isinstance(e.get("location"), list) else None
        end = detail.get("end_location") if isinstance(detail.get("end_location"), list) else None
        fr = frames.get(e["id"])
        rows.append({
            "match_id": mid, "pos": pos, "index": e["index"], "event_id": e["id"], "period": e["period"],
            "timestamp": e["timestamp"], "t": _secs(e["timestamp"]), "minute": e["minute"], "second": e["second"],
            "type": e["type"]["name"], "team_id": (e.get("team") or {}).get("id"),
            "player_id": (e.get("player") or {}).get("id"), "possession": e.get("possession"),
            "play_pattern": (e.get("play_pattern") or {}).get("name"),
            "x": loc[0] if loc else np.nan, "y": loc[1] if loc else np.nan,
            "end_x": end[0] if end else np.nan, "end_y": end[1] if end else np.nan,
            "subtype": sub.get("name") if isinstance(sub, dict) else None,
            "outcome": out.get("name") if isinstance(out, dict) else None,
            "recipient_id": (detail.get("recipient") or {}).get("id") if e["type"]["name"] == "Pass" else None,
            "aerial_won": bool(detail.get("aerial_won", False)),
            "related_events": ";".join(e.get("related_events") or []),
            "off_camera": bool(e.get("off_camera", False)),
            "has_360": fr is not None, "ff_len": fr[0] if fr else 0, "n_actor": fr[1] if fr else 0,
            "actor_x": fr[2][0] if fr and fr[2] else np.nan, "actor_y": fr[2][1] if fr and fr[2] else np.nan,
            "visible_area_len": fr[3] if fr else 0,
        })
    orphan = len(set(frames) - {e["id"] for e in events})
    rows.append({"match_id": mid, "_orphan_frames": orphan, "_n_frames": len(frames)})
    return rows


def load() -> pd.DataFrame:
    manifest = json.loads((ROOT / "results/markov_hmm_pilot/split_manifest.json").read_text())
    test = set(manifest["final_test_match_ids"])
    tier_b = sorted(set(manifest["development_match_ids"]) - set(manifest["pilot_match_ids"]))
    assert len(tier_b) == 304 and not set(tier_b) & test, "Tier B must be 304 development matches"
    with ProcessPoolExecutor(6) as pool:
        rows = [r for chunk in pool.map(parse, tier_b) for r in chunk]
    meta = pd.DataFrame([r for r in rows if "_orphan_frames" in r])
    ev = pd.DataFrame([r for r in rows if "_orphan_frames" not in r])
    assert set(ev["match_id"]) == set(tier_b) and not set(ev["match_id"]) & test
    SUMMARY["tier_b_matches"] = len(tier_b)
    SUMMARY["raw_events"] = len(ev)
    SUMMARY["three_sixty_records"] = int(meta["_n_frames"].sum())
    SUMMARY["three_sixty_records_without_matching_event"] = int(meta["_orphan_frames"].sum())
    SUMMARY["index_equals_array_position_plus_1"] = bool((ev["index"] == ev["pos"] + 1).all())
    ev["usable_360"] = ev["has_360"] & (ev["ff_len"] > 0)
    return ev.sort_values(["match_id", "pos"]).reset_index(drop=True)


# ============================================================================ task construction
def sequence(ev: pd.DataFrame, keep: np.ndarray, name: str) -> pd.DataFrame:
    """Observations = rows with keep (period != 5 enforced), next observation in the same
    (match, period) by raw order. Adds prev/next fields."""
    t = ev[keep & (ev["period"] != 5).to_numpy()].copy()
    g = t.groupby(["match_id", "period"], sort=False)
    for col in ("type", "event_id", "pos", "t", "team_id", "has_360", "usable_360", "player_id", "subtype"):
        t[f"next_{col}"] = g[col].shift(-1)
        t[f"prev_{col}"] = g[col].shift(1)
    t["has_target"] = t["next_type"].notna()
    t["skipped"] = t["next_pos"] - t["pos"] - 1
    t["taxonomy"] = name
    return t


def companion_flags(ev: pd.DataFrame) -> pd.Series:
    """Independent re-implementation of the actions15 K1-K3 rules on the RAW stream
    (previous raw record of the same match & period)."""
    g = ev.groupby(["match_id", "period"], sort=False)
    p_type, p_t, p_id, p_out = g["type"].shift(1), g["t"].shift(1), g["event_id"].shift(1), g["outcome"].shift(1)
    linked = np.array([isinstance(p, str) and p in r.split(";") for p, r in zip(p_id, ev["related_events"])])
    same = (p_t == ev["t"]).to_numpy()
    k1 = (ev["type"] == "Goal Keeper") & ev["subtype"].isin(GK_NO_ACTION)
    k2 = ((ev["type"] == "Duel") & (ev["subtype"] == "Tackle") & same & linked
          & ((p_type == "Dispossessed") | ((p_type == "Dribble") & (p_out == "Incomplete"))))
    k3 = (ev["type"] == "50/50") & (p_type == "50/50") & same & linked
    return pd.Series(np.select([k1, k2, k3], ["K1", "K2", "K3"], ""), index=ev.index)


# ============================================================================ helpers
def save(frame: pd.DataFrame, name: str, folder: Path = TABLES) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / name, index=False)


def transitions(t: pd.DataFrame, rows: np.ndarray, label: str) -> pd.DataFrame:
    x = t[rows]
    cnt = x.groupby(["type", "next_type"]).size().rename("n").reset_index()
    cnt["p_next_given_current"] = cnt["n"] / cnt.groupby("type")["n"].transform("sum")
    cnt.insert(0, "set", label)
    return cnt


EX_COLS = ["match_id", "period", "index", "event_id", "timestamp", "type", "subtype", "outcome", "team_id", "player_id",
           "x", "y", "end_x", "end_y", "has_360", "ff_len", "n_actor", "actor_x", "actor_y", "related_events"]


def window(ev: pd.DataFrame, i: int, before: int = 2, after: int = 3, tag: str = "") -> pd.DataFrame:
    m, p = ev.at[i, "match_id"], ev.at[i, "pos"]
    w = ev[(ev["match_id"] == m) & ev["pos"].between(p - before, p + after)][EX_COLS].copy()
    w.insert(0, "example", tag)
    w.insert(1, "focus", w["event_id"] == ev.at[i, "event_id"])
    w["related_events"] = w["related_events"].str.slice(0, 80)
    return w


# ============================================================================ part 1
def part1(ev: pd.DataFrame, a: pd.DataFrame) -> None:
    in_d15 = ev["type"].isin(D15)
    p5 = ev["period"] == 5
    key = ev["event_id"]
    t_ids = set(a.loc[a["has_target"], "event_id"])
    eval_ids = set(a.loc[a["has_target"] & a["usable_360"], "event_id"])
    tgt_any = set(a.loc[a["has_target"], "next_event_id"])
    tgt_eval = set(a.loc[a["has_target"] & a["usable_360"], "next_event_id"])
    rows = []
    for typ, g in ev.groupby("type"):
        n = len(g)
        k = key[g.index]
        r = {"type": typ, "in_scenarioD15": typ in D15, "raw_events": n,
             "with_360_record": int(g["has_360"].sum()), "pct_with_360": 100 * g["has_360"].mean(),
             "without_360_record": int((~g["has_360"]).sum()), "pct_without_360": 100 * (~g["has_360"]).mean(),
             "360_record_but_empty_freeze_frame": int((g["has_360"] & (g["ff_len"] == 0)).sum()),
             "360_record_without_actor": int((g["has_360"] & (g["n_actor"] == 0)).sum()),
             "360_record_without_visible_area": int((g["has_360"] & (g["visible_area_len"] == 0)).sum()),
             "excluded_by_preprocessing_period5": int(p5[g.index].sum()),
             "excluded_by_taxonomy": 0 if typ in D15 else int((~p5[g.index]).sum()),
             "retained_in_scenarioD15": int((in_d15[g.index] & ~p5[g.index]).sum()),
             "training_inputs_all_transitions": int(k.isin(t_ids).sum()),
             "prediction_inputs_eval_rows": int(k.isin(eval_ids).sum()),
             "targets_of_any_transition": int(k.isin(tgt_any).sum()),
             "targets_of_eval_rows": int(k.isin(tgt_eval).sum()),
             "off_camera_flag": int(g["off_camera"].sum()),
             "pct_without_360_among_off_camera": 100 * (~g.loc[g["off_camera"], "has_360"]).mean() if g["off_camera"].any() else np.nan}
        # mutually exclusive categories for the 360-bearing events
        h = g["has_360"] & (g["ff_len"] > 0)
        r["cat1_no_360_record"] = int((~g["has_360"]).sum())
        r["cat2_360_but_preprocessing_filtered(period5)"] = int((h & p5[g.index]).sum())
        r["cat3_360_but_not_in_taxonomy"] = int((h & ~p5[g.index]).sum()) if typ not in D15 else 0
        r["cat4_360_in_taxonomy_but_no_target(last_in_period)"] = int((h & ~p5[g.index] & ~k.isin(t_ids)).sum()) if typ in D15 else 0
        r["cat4b_360_record_but_empty_freeze_frame"] = int((g["has_360"] & (g["ff_len"] == 0)).sum())
        r["used_as_eval_input"] = r["prediction_inputs_eval_rows"]
        rows.append(r)
    cov = pd.DataFrame(rows).sort_values("raw_events", ascending=False)
    save(cov, "p1_event_coverage_by_type.csv")
    SUMMARY["p1_events_with_360_pct"] = float(100 * ev["has_360"].mean())
    SUMMARY["p1_360_records_with_empty_freeze_frame"] = int((ev["has_360"] & (ev["ff_len"] == 0)).sum())
    SUMMARY["p1_360_records_without_actor"] = int((ev["has_360"] & (ev["n_actor"] == 0)).sum())
    SUMMARY["p1_360_records_with_more_than_one_actor"] = int((ev["n_actor"] > 1).sum())
    # why no 360: by play pattern and position relative to stoppages (D15 events, periods 1-4)
    d = ev[ev["type"].isin(D15) & ~p5].copy()
    d["prev_raw_type"] = ev.groupby(["match_id", "period"])["type"].shift(1)[d.index]
    by_pp = d.groupby("play_pattern")["has_360"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "share_with_360"})
    save(by_pp.reset_index(), "p1_no360_by_play_pattern_d15.csv")
    by_prev = d.groupby("prev_raw_type")["has_360"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "share_with_360"})
    save(by_prev[by_prev["n"] >= 200].sort_values("share_with_360").reset_index(), "p1_no360_by_previous_raw_type_d15.csv")
    # runs of consecutive no-360 events (replays / camera off) vs isolated gaps
    d["run"] = (d["has_360"] != d.groupby(["match_id", "period"])["has_360"].shift(1)).cumsum()
    runs = d[~d["has_360"]].groupby("run").agg(n=("event_id", "size"), dur=("t", lambda s: s.max() - s.min()))
    SUMMARY["p1_no360_d15_events_in_runs"] = {"runs": int(len(runs)), "events": int(runs["n"].sum()),
                                             "share_events_in_runs_len_ge_5": float(runs.loc[runs["n"] >= 5, "n"].sum() / runs["n"].sum()),
                                             "share_events_isolated_len_1": float(runs.loc[runs["n"] == 1, "n"].sum() / runs["n"].sum()),
                                             "median_run_length": float(runs["n"].median())}
    per_match = ev.groupby("match_id")["has_360"].mean()
    SUMMARY["p1_per_match_360_share"] = {"min": float(per_match.min()), "median": float(per_match.median()),
                                        "max": float(per_match.max())}


def ball_receipt(ev: pd.DataFrame) -> None:
    g = ev.groupby(["match_id", "period"], sort=False)
    nxt_type, nxt_i = g["type"].shift(-1), g["pos"].shift(-1)
    passes = ev[(ev["type"] == "Pass") & (ev["period"] != 5)]
    br = ev[ev["type"] == "Ball Receipt*"]
    by_id = ev.set_index("event_id")
    # receipt linked to which pass? use related_events
    rec = []
    for i, r in br.iterrows():
        rel = [x for x in r["related_events"].split(";") if x]
        linked = [x for x in rel if x in by_id.index and by_id.at[x, "type"] == "Pass"]
        p = by_id.loc[linked[0]] if linked else None
        rec.append({"receipt_id": r["event_id"],
                    "receipt_outcome": r["outcome"] if isinstance(r["outcome"], str) else "Complete(no outcome)",
                    "receipt_has_360": r["has_360"], "linked_pass": p is not None,
                    "dt_from_pass": (r["t"] - p["t"]) if p is not None else np.nan,
                    "same_team": (r["team_id"] == p["team_id"]) if p is not None else np.nan,
                    "receiver_is_pass_recipient": (r["player_id"] == p["recipient_id"]) if p is not None else np.nan,
                    "dist_receipt_loc_to_pass_end": float(np.hypot(r["x"] - p["end_x"], r["y"] - p["end_y"])) if p is not None else np.nan,
                    "pass_outcome": (p["outcome"] if isinstance(p["outcome"], str) else "Complete") if p is not None else None,
                    "pass_has_360": p["has_360"] if p is not None else np.nan})
    rec = pd.DataFrame(rec)
    save(rec.groupby(["receipt_outcome", "pass_outcome"], dropna=False).agg(
        n=("receipt_id", "size"), receipt_has_360=("receipt_has_360", "mean"),
        median_dt=("dt_from_pass", "median"), share_same_team=("same_team", "mean"),
        share_receiver_is_recipient=("receiver_is_pass_recipient", "mean"),
        median_dist_to_pass_end=("dist_receipt_loc_to_pass_end", "median")).reset_index(),
        "p1_ball_receipt_vs_linked_pass.csv")
    pass_next = nxt_type[passes.index]
    by_out = pd.DataFrame({"pass_outcome": passes["outcome"].fillna("Complete"), "next_raw_is_receipt": pass_next == "Ball Receipt*"})
    save(by_out.groupby("pass_outcome")["next_raw_is_receipt"].agg(["size", "mean"]).reset_index(),
         "p1_pass_next_raw_is_ball_receipt_by_pass_outcome.csv")
    SUMMARY["p1_ball_receipt"] = {
        "n": int(len(br)), "with_360": float(br["has_360"].mean()),
        "linked_to_a_pass": float(rec["linked_pass"].mean()),
        "complete_receipts": int((br["outcome"].isna()).sum()), "incomplete_receipts": int((br["outcome"] == "Incomplete").sum()),
        "share_passes_next_raw_receipt": float((pass_next == "Ball Receipt*").mean()),
        "receiver_is_pass_recipient_given_complete": float(rec.loc[rec["receipt_outcome"].str.startswith("Complete"), "receiver_is_pass_recipient"].mean()),
        "median_dt_receipt_minus_pass_s": float(rec["dt_from_pass"].median()),
        "median_dist_receipt_location_to_pass_end_location": float(rec["dist_receipt_loc_to_pass_end"].median()),
        "share_dist_le_1": float((rec["dist_receipt_loc_to_pass_end"] <= 1).mean()),
        "next_retained_after_receipt": ev.loc[br.index].pipe(lambda b: nxt_type[b.index]).value_counts(normalize=True).head(6).to_dict(),
    }
    # examples: complete pass with 360 on both, incomplete pass, long pass, pass without receipt
    ex = []
    pick = passes[(pass_next == "Ball Receipt*")]
    rng = np.random.default_rng(20261118)
    sel = []
    for name, mask in [("complete_short", pick["outcome"].isna() & (np.hypot(pick["end_x"] - pick["x"], pick["end_y"] - pick["y"]) < 15)),
                       ("complete_long", pick["outcome"].isna() & (np.hypot(pick["end_x"] - pick["x"], pick["end_y"] - pick["y"]) > 40)),
                       ("incomplete_pass", pick["outcome"] == "Incomplete"),
                       ("complete_pass_receipt_without_360", pick["outcome"].isna()),
                       ("complete_mid", pick["outcome"].isna())]:
        idx = pick.index[mask.to_numpy()]
        if name == "complete_pass_receipt_without_360":
            idx = [i for i in idx[:20000] if not ev.at[i + 1, "has_360"]]
        sel.append((name, int(rng.choice(idx))))
    for name, i in sel:
        ex.append(window(ev, i, 1, 2, f"pass_receipt:{name}"))
    save(pd.concat(ex), "ex_pass_ball_receipt.csv", EXAMPLES)


# ============================================================================ part 2
def part2(ev: pd.DataFrame, a: pd.DataFrame) -> None:
    folds = json.loads((ROOT / "results/markov_hmm_cost_sensitive_v2/tiers/tier_b/scenarioD15/folds.json").read_text())["scenarioD15"]
    c = sequence(ev, (ev["type"].isin(D15) & ev["usable_360"]).to_numpy(), "C_360_only_sequence")
    A_train = a["has_target"].to_numpy()
    A_eval = (a["has_target"] & a["usable_360"]).to_numpy()
    B_train = A_eval
    C_rows = c["has_target"].to_numpy()
    SUMMARY["p2_definitions"] = {
        "A_matches_with_360_coverage": int(ev.groupby("match_id")["has_360"].any().sum()),
        "B_events_with_usable_360_all_types": int(ev["usable_360"].sum()),
        "B_d15_events_with_usable_360_periods_1_4": int((ev["type"].isin(D15) & ev["usable_360"] & (ev["period"] != 5)).sum()),
        "C_rows_current_360_any_target_360": int(A_eval.sum()),
        "C_rows_whose_target_also_has_360": int((A_eval & a["next_usable_360"].astype(bool).to_numpy()).sum()),
    }
    prot = []
    for name, t, rows_train, rows_eval in (("A_all_transitions", a, A_train, A_eval),
                                           ("B_360_only_training", a, B_train, A_eval),
                                           ("C_360_only_sequence", c, C_rows, C_rows)):
        prot.append({"protocol": name, "observations_in_sequence": len(t), "training_rows": int(rows_train.sum()),
                     "evaluation_rows": int(rows_eval.sum()), "matches": int(t["match_id"].nunique()),
                     "eval_share_skipping_ge1_raw": float((t.loc[rows_eval, "skipped"] > 0).mean()),
                     "eval_mean_skipped_raw": float(t.loc[rows_eval, "skipped"].mean()),
                     "train_share_skipping_ge1_raw": float((t.loc[rows_train, "skipped"] > 0).mean()),
                     "eval_share_zero_time_gap": float((t.loc[rows_eval, "next_t"] == t.loc[rows_eval, "t"]).mean()),
                     "eval_median_time_gap_s": float((t.loc[rows_eval, "next_t"] - t.loc[rows_eval, "t"]).median()),
                     "eval_p90_time_gap_s": float((t.loc[rows_eval, "next_t"] - t.loc[rows_eval, "t"]).quantile(0.9))})
    save(pd.DataFrame(prot), "p2_protocol_sizes.csv")
    # class distributions (targets and currents) per protocol
    dist = []
    for name, t, rows in (("A_training", a, A_train), ("A_and_B_evaluation", a, A_eval), ("B_training", a, B_train),
                          ("C_training_and_evaluation", c, C_rows)):
        tc = t.loc[rows, "next_type"].value_counts()
        cc = t.loc[rows, "type"].value_counts()
        for cls in D15:
            dist.append({"set": name, "class": cls, "n_target": int(tc.get(cls, 0)), "share_target_pct": 100 * tc.get(cls, 0) / tc.sum(),
                         "n_current": int(cc.get(cls, 0)), "share_current_pct": 100 * cc.get(cls, 0) / cc.sum()})
    dist = pd.DataFrame(dist)
    save(dist, "p2_class_distribution_by_protocol.csv")
    pv = dist.pivot(index="class", columns="set", values="share_target_pct")
    shift = pd.DataFrame({"class": pv.index, "A_train_target_pct": pv["A_training"], "B_train_target_pct": pv["B_training"],
                          "B_minus_A_pp": pv["B_training"] - pv["A_training"],
                          "eval_target_pct": pv["A_and_B_evaluation"], "C_target_pct": pv["C_training_and_evaluation"],
                          "C_minus_eval_pp": pv["C_training_and_evaluation"] - pv["A_and_B_evaluation"]}).reset_index(drop=True)
    pc = dist.pivot(index="class", columns="set", values="share_current_pct")
    shift["A_train_current_pct"], shift["B_train_current_pct"] = pc["A_training"].to_numpy(), pc["B_training"].to_numpy()
    shift["current_B_minus_A_pp"] = shift["B_train_current_pct"] - shift["A_train_current_pct"]
    save(shift, "p2_distribution_shift_A_vs_B.csv")
    tr = pd.concat([transitions(a, A_train, "A_training"), transitions(a, A_eval, "A_B_evaluation"),
                    transitions(c, C_rows, "C")])
    save(tr, "p2_transitions_by_protocol.csv")
    # what the no-360 training rows are
    no360 = a[A_train & ~a["usable_360"].to_numpy()]
    save(no360["type"].value_counts().rename("n").reset_index().assign(
        share=lambda f: f["n"] / f["n"].sum()), "p2_training_rows_without_360_by_current_type.csv")
    # C vs A: targets that change for rows with current 360
    m = a.loc[A_eval, ["event_id", "next_event_id", "next_type"]].merge(
        c.loc[C_rows, ["event_id", "next_event_id", "next_type"]], on="event_id", how="left", suffixes=("_A", "_C"))
    SUMMARY["p2_C_vs_A"] = {"A_eval_rows": int(A_eval.sum()), "rows_also_in_C": int(m["next_event_id_C"].notna().sum()),
                            "rows_target_event_changed": int((m["next_event_id_A"] != m["next_event_id_C"]).sum()),
                            "rows_target_class_changed": int((m["next_type_A"] != m["next_type_C"]).sum()),
                            "share_target_class_changed": float((m["next_type_A"] != m["next_type_C"]).mean())}
    # Markov first order (alpha = 1): A-training vs B-training, same evaluation rows and folds (counting only)
    K = len(D15)
    ids = {cls: i for i, cls in enumerate(D15)}
    fold_of = {m_: f["fold_id"] for f in folds for m_ in f["validation_match_ids"]}
    cur = a["type"].map(ids).to_numpy(); nxt = a["next_type"].map(ids).fillna(-1).astype(int).to_numpy()
    fold = a["match_id"].map(fold_of).to_numpy()
    res = {"A": 0.0, "B": 0.0}
    n_eval = int(A_eval.sum())
    per_fold = []
    for f in folds:
        trm = np.isin(a["match_id"].to_numpy(), f["train_match_ids"])
        va = (fold == f["fold_id"]) & A_eval
        row = {"fold": f["fold_id"]}
        for name, rows in (("A", A_train), ("B", B_train)):
            sel = trm & rows
            C = np.zeros((K, K)); np.add.at(C, (cur[sel], nxt[sel]), 1)
            tab = (C + 1) / (C.sum(1, keepdims=True) + K)
            ll = -np.log(tab[cur[va], nxt[va]]).sum()
            res[name] += ll
            row[f"log_loss_{name}"] = ll / va.sum()
        per_fold.append(row)
    SUMMARY["p2_markov1_alpha1_log_loss"] = {"A_all_transitions": res["A"] / n_eval, "B_360_only": res["B"] / n_eval,
                                             "B_minus_A": (res["B"] - res["A"]) / n_eval, "per_fold": per_fold,
                                             "note": "count-based re-estimation for reproduction of audit R1; same rows and folds as experiment 4"}


# ============================================================================ part 3
def neighbours(ev_seq: pd.DataFrame, cls: str) -> dict:
    x = ev_seq[ev_seq["type"] == cls]
    out = {"occurrences": int(len(x)), "with_360": int(x["has_360"].sum()), "without_360": int((~x["has_360"]).sum())}
    prev_dt = x["t"] - x["prev_t"]
    next_dt = x["next_t"] - x["t"]
    out.update({
        "prev_retained_top": x["prev_type"].value_counts(normalize=True).head(6).round(4).to_dict(),
        "next_retained_top": x["next_type"].value_counts(normalize=True).head(6).round(4).to_dict(),
        "share_prev_same_timestamp": float((prev_dt == 0).mean()), "share_prev_within_0_5s": float((prev_dt.abs() <= 0.5).mean()),
        "share_next_same_timestamp": float((next_dt == 0).mean()), "share_next_within_0_5s": float((next_dt.abs() <= 0.5).mean()),
        "share_prev_same_team": float((x["prev_team_id"] == x["team_id"]).mean()),
        "share_next_same_team": float((x["next_team_id"] == x["team_id"]).mean()),
        "share_prev_same_type": float((x["prev_type"] == cls).mean()), "share_next_same_type": float((x["next_type"] == cls).mean()),
        "prev_dt_quantiles_s": prev_dt.quantile([0.1, 0.5, 0.9]).round(3).tolist(),
        "next_dt_quantiles_s": next_dt.quantile([0.1, 0.5, 0.9]).round(3).tolist(),
        "share_linked_to_prev_retained": float(np.mean([isinstance(p, str) and p in r.split(";") for p, r in zip(x["prev_event_id"], x["related_events"])])),
        "share_linked_to_next_retained": float(np.mean([isinstance(n, str) and n in r.split(";") for n, r in zip(x["next_event_id"], x["related_events"])])),
    })
    return out


def part3(ev: pd.DataFrame, a: pd.DataFrame) -> None:
    # a has prev/next of the retained D15 sequence; add t/related on the observation rows
    s = a.copy()
    s["prev_t"] = s.groupby(["match_id", "period"])["t"].shift(1)
    res = {cls: neighbours(s, cls) for cls in ("Goal Keeper", "Duel", "50/50")}
    # conditional probabilities in both directions over all retained transitions
    tr = s[s["has_target"]]
    pair = tr.groupby(["type", "next_type"]).size()
    p_next = pair / tr.groupby("type").size().reindex(pair.index.get_level_values(0)).to_numpy()
    p_prev = pair / tr.groupby("next_type").size().reindex(pair.index.get_level_values(1)).to_numpy()
    rows = [{"current": cur, "next": nxt, "n": int(pair[(cur, nxt)]), "P(next|current)": float(p_next[(cur, nxt)]),
             "P(previous=current | event=next)": float(p_prev[(cur, nxt)])}
            for (cur, nxt) in pair.index if cur in ("Goal Keeper", "Duel", "50/50") or nxt in ("Goal Keeper", "Duel", "50/50")]
    save(pd.DataFrame(rows).sort_values("n", ascending=False), "p3_conditional_probabilities_both_directions.csv")
    res["Goal Keeper"]["P(next=GK | current=Shot)"] = float(p_next.get(("Shot", "Goal Keeper"), 0))
    res["Goal Keeper"]["P(previous=Shot | current=GK)"] = float(p_prev.get(("Shot", "Goal Keeper"), 0))
    # GK subtypes
    gk = s[s["type"] == "Goal Keeper"].copy()
    gk["after_shot"] = gk["prev_type"] == "Shot"
    gk["prev_same_ts"] = gk["prev_t"] == gk["t"]
    gk["linked_prev"] = [isinstance(p, str) and p in r.split(";") for p, r in zip(gk["prev_event_id"], gk["related_events"])]
    sub = gk.groupby("subtype", dropna=False).agg(n=("event_id", "size"), with_360=("has_360", "mean"),
                                                  after_shot=("after_shot", "mean"), prev_same_timestamp=("prev_same_ts", "mean"),
                                                  linked_to_prev=("linked_prev", "mean"),
                                                  is_actions15_companion=("subtype", lambda x: x.isin(GK_NO_ACTION).mean())).reset_index()
    sub["share"] = sub["n"] / sub["n"].sum()
    save(sub.sort_values("n", ascending=False), "p3_goal_keeper_subtypes.csv")
    shots = s[(s["type"] == "Shot") & s["has_target"]]
    ct = pd.crosstab(shots["outcome"].fillna("?"), shots["next_type"].where(shots["next_type"] == "Goal Keeper", "other") + ":" +
                     shots["next_subtype"].where(shots["next_type"] == "Goal Keeper", "").fillna(""))
    save(ct.reset_index(), "p3_shot_outcome_vs_next_goal_keeper_subtype.csv")
    # 50/50 consecutive pairs
    f = s[(s["type"] == "50/50") & (s["next_type"] == "50/50")]
    linked = np.array([isinstance(n, str) and n in r.split(";") for n, r in zip(f["next_event_id"], f["related_events"])])
    res["50/50"].update({
        "P(next=50/50 | current=50/50)": float(p_next.get(("50/50", "50/50"), 0)),
        "consecutive_pairs": int(len(f)), "pairs_same_timestamp": float((f["next_t"] == f["t"]).mean()),
        "pairs_opposite_teams": float((f["next_team_id"] != f["team_id"]).mean()),
        "pairs_linked_related_events": float(linked.mean()),
        "pairs_one_contest(same_ts & opposite & linked)": float(((f["next_t"] == f["t"]) & (f["next_team_id"] != f["team_id"]) & linked).mean()),
        "outcome_pairs": (f["outcome"].fillna("?") + " | " + ev.set_index("event_id").loc[f["next_event_id"], "outcome"].fillna("?").to_numpy()).value_counts().head(6).to_dict(),
        "singletons(not in a consecutive pair)": int(((s["type"] == "50/50") & (s["prev_type"] != "50/50") & (s["next_type"] != "50/50")).sum()),
    })
    # Duel: subtype, partner records
    du = s[s["type"] == "Duel"].copy()
    du["linked_prev"] = [isinstance(p, str) and p in r.split(";") for p, r in zip(du["prev_event_id"], du["related_events"])]
    du["prev_same_ts"] = du["prev_t"] == du["t"]
    by_id = ev.set_index("event_id")
    def partner(row):
        rel = [x for x in row["related_events"].split(";") if x in by_id.index]
        types = [by_id.at[x, "type"] + ("(aerial_won)" if by_id.at[x, "aerial_won"] else "") for x in rel]
        return ",".join(sorted(types)) if types else "(none)"
    du["related_types"] = du.apply(partner, axis=1)
    dsub = du.groupby(["subtype", "outcome"], dropna=False).agg(n=("event_id", "size"), with_360=("has_360", "mean"),
                                                                 prev_same_ts=("prev_same_ts", "mean"), linked_prev=("linked_prev", "mean")).reset_index()
    save(dsub.sort_values("n", ascending=False), "p3_duel_subtypes.csv")
    save(du.groupby(["subtype", "related_types"]).size().rename("n").reset_index().sort_values("n", ascending=False).head(30),
         "p3_duel_related_event_types.csv")
    dd = s[(s["type"] == "Duel") & (s["next_type"] == "Duel")]
    res["Duel"].update({"P(next=Duel | current=Duel)": float(p_next.get(("Duel", "Duel"), 0)),
                        "consecutive_duel_pairs": int(len(dd)), "duel_pairs_same_timestamp": float((dd["next_t"] == dd["t"]).mean()),
                        "duel_pairs_opposite_teams": float((dd["next_team_id"] != dd["team_id"]).mean()),
                        "P(next=Duel | current=Dispossessed)": float(p_next.get(("Dispossessed", "Duel"), 0)),
                        "P(previous=Dispossessed | current=Duel)": float(p_prev.get(("Dispossessed", "Duel"), 0)),
                        "subtype_counts": du["subtype"].value_counts().to_dict()})
    SUMMARY["p3"] = res
    # ---------------- examples (support and contradict)
    ex = []
    rng = np.random.default_rng(7)
    def pick(mask, tag, n):
        idx = np.flatnonzero(mask)
        for j in rng.choice(idx, size=min(n, len(idx)), replace=False):
            i = ev.index[(ev["event_id"] == s["event_id"].iat[j])][0]
            ex.append(window(ev, i, 2, 2, tag))
    gkm = s["type"].to_numpy() == "Goal Keeper"
    pick(gkm & (s["subtype"] == "Shot Faced").to_numpy(), "GK support: Shot Faced right after Shot (annotation)", 2)
    pick(gkm & (s["subtype"] == "Goal Conceded").to_numpy(), "GK support: Goal Conceded (annotation)", 1)
    pick(gkm & (s["subtype"] == "Shot Saved").to_numpy(), "GK contradict: Shot Saved = real save", 2)
    pick(gkm & s["subtype"].isin(["Collected", "Punch", "Keeper Sweeper"]).to_numpy() & (s["prev_type"] != "Shot").to_numpy(),
         "GK contradict: claim/punch/sweep not after a shot", 2)
    dm = s["type"].to_numpy() == "Duel"
    pick(dm & (s["prev_type"] == "Dispossessed").to_numpy() & (s["subtype"] == "Tackle").to_numpy(), "Duel support: Tackle companion of Dispossessed", 2)
    pick(dm & (s["subtype"] == "Tackle").to_numpy() & (s["prev_type"] == "Dribble").to_numpy(), "Duel support: Tackle companion of Dribble", 1)
    pick(dm & (s["subtype"] == "Aerial Lost").to_numpy(), "Duel: Aerial Lost (loser's record; winner recorded separately)", 2)
    pick(dm & (s["next_type"] == "Duel").to_numpy() & (s["next_t"] != s["t"]).to_numpy(), "Duel contradict: consecutive duels at different times", 2)
    fm = s["type"].to_numpy() == "50/50"
    pick(fm & (s["next_type"] == "50/50").to_numpy() & (s["next_t"] == s["t"]).to_numpy(), "50/50 support: same-timestamp linked pair", 3)
    pick(fm & (s["prev_type"] != "50/50").to_numpy() & (s["next_type"] != "50/50").to_numpy(), "50/50 contradict: single record (no partner)", 3)
    save(pd.concat(ex), "ex_goal_keeper_duel_5050.csv", EXAMPLES)


# ============================================================================ part 4
def part4(ev: pd.DataFrame, a: pd.DataFrame) -> None:
    rules = companion_flags(ev)
    # the project's own rule for comparison (read-only import)
    sys.path.insert(0, str(ROOT))
    from src.sequence_models.v3_taxonomy import companion_rules  # noqa: E402
    proj = companion_rules(ev.rename(columns={"t": "t_seconds"}).assign(related_events=ev["related_events"]))
    SUMMARY["p4_independent_rules_equal_project_rules"] = bool((proj.to_numpy() == rules.to_numpy()).all())
    SUMMARY["p4_rule_counts_all_periods"] = rules[rules != ""].value_counts().to_dict()
    SUMMARY["p4_rule_counts_periods_1_4"] = rules[(rules != "") & (ev["period"] != 5)].value_counts().to_dict()
    taxa = {"scenarioD15": ev["type"].isin(D15).to_numpy(),
            "actions15": (ev["type"].isin(D15) & (rules == "")).to_numpy(),
            "alt12_no_GK_Duel_5050": ev["type"].isin(C12).to_numpy()}
    seqs = {n: (a if n == "scenarioD15" else sequence(ev, k, n)) for n, k in taxa.items()}
    base = seqs["scenarioD15"]
    base_eval = base[base["has_target"] & base["usable_360"]]
    comp = []
    per_class = []
    for n, t in seqs.items():
        tr = t[t["has_target"]]
        evr = t[t["has_target"] & t["usable_360"]]
        m = base_eval[["event_id", "next_event_id", "next_type"]].merge(evr[["event_id", "next_event_id", "next_type"]],
                                                                        on="event_id", how="left", suffixes=("_d15", "_x"))
        cell = tr.groupby(["type", "next_type"]).size()
        p = cell / tr.groupby("type").size().reindex(cell.index.get_level_values(0)).to_numpy()
        h = lambda c: float(-(c / c.sum() * np.log2(c / c.sum())).sum())  # noqa: E731
        hcond = float(sum(len(x) / len(evr) * h(x["next_type"].value_counts()) for _, x in evr.groupby("type")))
        seq_aff = set(m.loc[m["next_event_id_x"] != m["next_event_id_d15"], "event_id"])
        aff = base_eval[base_eval["event_id"].isin(seq_aff)].groupby(["match_id", "period"]).ngroups
        comp.append({"taxonomy": n, "n_classes": t["type"].nunique(), "retained_events": len(t),
                     "removed_vs_scenarioD15": len(base) - len(t), "pct_removed_vs_scenarioD15": 100 * (len(base) - len(t)) / len(base),
                     "transitions": len(tr), "eval_rows_current_360": len(evr),
                     "d15_eval_rows_dropped_as_current": int(m["next_event_id_x"].isna().sum()),
                     "d15_eval_rows_with_changed_target_event": int(((m["next_event_id_x"] != m["next_event_id_d15"]) & m["next_event_id_x"].notna()).sum()),
                     "d15_eval_rows_with_changed_target_class": int(((m["next_type_x"] != m["next_type_d15"]) & m["next_event_id_x"].notna()).sum()),
                     "pct_sequences_affected": 100 * aff / base_eval.groupby(["match_id", "period"]).ngroups,
                     "eval_share_skipping_ge1_raw": float((evr["skipped"] > 0).mean()),
                     "eval_share_zero_time_gap": float((evr["next_t"] == evr["t"]).mean()),
                     "rows_in_cells_P_ge_0_9": float(cell[p >= 0.9].sum() / len(tr)),
                     "near_deterministic_cells": "; ".join(f"{i}->{j} {v:.3f}" for (i, j), v in p[(p >= 0.9) & (cell >= 50)].items()),
                     "H_target_bits": h(evr["next_type"].value_counts()), "H_target_given_current_bits": hcond})
        vc, ec = t["type"].value_counts(), evr["next_type"].value_counts()
        for i, cls in enumerate(D15):
            per_class.append({"taxonomy": n, "class": cls, "canonical_id": i if (n != "alt12_no_GK_Duel_5050" or cls in C12) else None,
                              "id_within_taxonomy": (C12.index(cls) if cls in C12 else None) if n.startswith("alt12") else i,
                              "retained_events": int(vc.get(cls, 0)), "eval_targets": int(ec.get(cls, 0)),
                              "eval_target_share_pct": 100 * ec.get(cls, 0) / ec.sum()})
    save(pd.DataFrame(comp), "p4_taxonomy_comparison.csv")
    save(pd.DataFrame(per_class), "p4_taxonomy_class_counts.csv")
    # legitimate actions removed by the 12-class alternative
    keep_d15 = ev["type"].isin(D15) & (ev["period"] != 5)
    leg = ev[keep_d15 & ev["type"].isin(["Goal Keeper", "Duel", "50/50"]) & (rules == "")]
    save(leg.groupby(["type", "subtype"], dropna=False).size().rename("n").reset_index().sort_values("n", ascending=False),
         "p4_records_removed_by_alt12_but_kept_by_actions15.csv")
    # before/after sequences
    ex = []
    for rule, n in (("K1", 2), ("K2", 2), ("K3", 2)):
        idx = np.flatnonzero((rules == rule).to_numpy() & (ev["period"] != 5).to_numpy())
        for j in np.random.default_rng(11).choice(idx, n, replace=False):
            w = window(ev, ev.index[j], 3, 2, f"companion {rule}")
            w["scenarioD15"] = np.where(w["type"].isin(D15), w["type"], "(skipped)")
            w["actions15"] = np.where(w["type"].isin(D15) & (rules[w.index] == "").to_numpy(), w["type"], "(skipped)")
            w["alt12"] = np.where(w["type"].isin(C12), w["type"], "(skipped)")
            ex.append(w)
    save(pd.concat(ex), "ex_before_after_taxonomies.csv", EXAMPLES)


# ============================================================================ part 5
def part5() -> None:
    pc = pd.read_csv(ROOT / "results/logistic_regression_f0_2026-11-17/per_class_metrics.csv")
    lr = pc[pc["model"] == "logistic_F0"][["class", "support", "precision", "recall", "f1", "n_predicted", "true_positives"]]
    save(lr, "p5_per_class_experiment4_logistic_F0.csv")
    three = lr["class"].isin(["Goal Keeper", "Duel", "50/50"])
    SUMMARY["p5"] = {"macro_f1_15": float(lr["f1"].mean()),
                     "macro_f1_12_other_classes": float(lr.loc[~three, "f1"].sum() / 12),
                     "share_of_f1_sum_from_3": float(lr.loc[three, "f1"].sum() / lr["f1"].sum()),
                     "share_of_correct_predictions_from_3": float(lr.loc[three, "true_positives"].sum() / lr["true_positives"].sum()),
                     "total_correct": int(lr["true_positives"].sum()), "correct_from_3": int(lr.loc[three, "true_positives"].sum()),
                     "accuracy_check": float(lr["true_positives"].sum() / lr["support"].sum())}


def main() -> None:
    ev = load()
    a = sequence(ev, ev["type"].isin(D15).to_numpy(), "scenarioD15")
    SUMMARY["scenarioD15_observations"] = len(a)
    SUMMARY["scenarioD15_transitions"] = int(a["has_target"].sum())
    SUMMARY["scenarioD15_eval_rows_has_360"] = int((a["has_target"] & a["has_360"]).sum())
    SUMMARY["scenarioD15_eval_rows_usable_360"] = int((a["has_target"] & a["usable_360"]).sum())
    part1(ev, a)
    ball_receipt(ev)
    part2(ev, a)
    part3(ev, a)
    part4(ev, a)
    part5()
    (OUT / "summary.json").write_text(json.dumps(SUMMARY, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(SUMMARY, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
