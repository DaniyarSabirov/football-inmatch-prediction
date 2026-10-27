"""Dataset-wide before/after audit of the evidence-based team identity and
the synchronized orientation (investigation output; writes only here).

For every usable match (417), with the CURRENT code:
  - new pipeline: src/orientation_resolution.resolve_match_orientation +
    src/team_identity.resolve_match_team_identity (what renderer and builder
    now share) and src/build_statsbomb_360_dataset.build_rows_from_match_data;
  - legacy pipeline: the builder as committed at HEAD 5c21ad3
    (legacy_builder_head.py, byte-for-byte `git show HEAD:...`) and the legacy
    Problem C team of every point (FrameIdentity.legacy_teams).

Outputs (../):
  records.csv.gz        one row per usable 360 record (snapshot)
  keepers.csv.gz        one row per goalkeeper observation near a goal
  goalkeeper_rule.csv   every goalkeeper correction with forward validation
  ml_rows.csv.gz        legacy and new ML features per row (joined on event id)
  match_events.csv      events per match (denominators)

Usage: .venv/bin/python results/team_identity_correction_2026-10-26/diagnostics/audit_identity_correction.py [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import legacy_builder_head  # noqa: E402
from src.animate_statsbomb_360_match import available_360_match_ids  # noqa: E402
from src.build_statsbomb_360_dataset import build_rows_from_match_data  # noqa: E402
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.orientation_resolution import match_team_ids, resolve_match_orientation  # noqa: E402
from src.team_identity import (  # noqa: E402
    GOALKEEPER_BASE_DISTANCE,
    GOALKEEPER_MAX_DISTANCE,
    GOALKEEPER_MAX_GAP_MS,
    GOALKEEPER_SPEED_YD_PER_S,
    IDENTITY_CORRECTED_GOALKEEPER,
    IDENTITY_CORRECTED_VIEW,
    IDENTITY_SPEC,
    IDENTITY_UNRESOLVED,
    KEEPER_GOAL_ZONE,
    TEAM_IDENTITY_AMBIGUOUS,
    resolve_match_team_identity,
    timestamp_ms,
)

DATA = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
REASONS = [
    "copy_group_undecided", "copy_flag_deviation", "record_contradicted",
    "contradicted_by_both_neighbours", "keeper_goal_side_unverified", "goalkeeper_correction_contradicted",
]
ML_FIELDS = [
    "current_event_id", "current_event_type", "current_team", "period", "minute", "actor_x", "actor_y",
    "orientation", "visible_players", "visible_teammates", "visible_opponents", "visible_ambiguous",
    "next_event_type", "same_team", "same_possession", "time_to_next_event_seconds",
]


def own_and_absolute(frame: dict, orientation: str, team_bit: int, period: int):
    points = [p.get("location") for p in frame.get("freeze_frame") or []]
    own, absolute = [], []
    for loc in points:
        if not (isinstance(loc, list) and len(loc) == 2):
            own.append(None)
            absolute.append(None)
            continue
        x, y = loc
        if orientation == "ROTATED_180":
            x, y = 120 - x, 80 - y
        own.append((x, y))
        absolute.append((120 - x, 80 - y) if team_bit == 1 and period != 5 else (x, y))
    return own, absolute


def audit_match(match_id: int) -> dict:
    events = json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    mo = resolve_match_orientation(match_id, events, frames)
    records = mo.records.usable_frames
    names = {r.event["team"]["id"]: r.event["team"]["name"] for r in records}
    pair = tuple(names[t] for t in match_team_ids(mo.records))
    ids = resolve_match_team_identity(
        [(r.event, r.frame_360) for r in records], [x.orientation for x in mo.resolved], mo.problem_a, pair, events
    )
    other = {pair[0]: pair[1], pair[1]: pair[0]}

    rec_rows, keeper_rows, geometry = [], [], {}
    for k, record in enumerate(records):
        event, frame = record.event, record.frame_360
        f = ids[event["id"]]
        orientation = mo.resolved[k].orientation
        team = event["team"]["name"]
        own, absolute = own_and_absolute(frame, orientation, 0 if team == pair[0] else 1, event["period"])
        geometry[event["id"]] = (absolute, orientation != "UNCERTAIN")
        st = Counter(f.statuses)
        rs = Counter(r for r in f.reasons if r)
        legacy_definite = [t != TEAM_IDENTITY_AMBIGUOUS for t in f.legacy_teams]
        new_definite = [s != IDENTITY_UNRESOLVED for s in f.statuses]
        row = {
            "match_id": match_id, "event_id": event["id"], "event_type": event["type"]["name"], "team": team,
            "period": event["period"], "time_ms": timestamp_ms(event["timestamp"]),
            "problem_a": mo.problem_a[k].orientation, "resolved_orientation": orientation,
            "resolved_by": mo.resolved[k].resolved_by, "points": len(f.statuses),
            "spec": st[IDENTITY_SPEC], "corrected_view": st[IDENTITY_CORRECTED_VIEW],
            "corrected_goalkeeper": st[IDENTITY_CORRECTED_GOALKEEPER], "unresolved": st[IDENTITY_UNRESOLVED],
            **{f"unresolved_{r}": rs[r] for r in REASONS},
            "confirmed": sum(f.confirmed),
            "legacy_ambiguous": sum(not d for d in legacy_definite),
            "changed_team_vs_legacy": sum(ld and nd and a != b for a, b, ld, nd in zip(f.teams, f.legacy_teams, legacy_definite, new_definite)),
            "legacy_definite_now_unresolved": sum(ld and not nd for ld, nd in zip(legacy_definite, new_definite)),
            "legacy_ambiguous_now_definite": sum(nd and not ld for ld, nd in zip(legacy_definite, new_definite)),
            "changed_team_vs_spec": sum(nd and a != b for a, b, nd in zip(f.teams, f.spec_teams, new_definite)),
            "linked": f.component is not None, "decision": f.decision, "decision_evidence": f.decision_evidence,
        }
        rec_rows.append(row)
        for i, player in enumerate(frame.get("freeze_frame") or []):
            if not (isinstance(player, dict) and player.get("keeper") is True) or own[i] is None or event["period"] == 5:
                continue
            if orientation == "UNCERTAIN":
                continue
            x = own[i][0]
            goal = team if x <= KEEPER_GOAL_ZONE else other[team] if x >= 120 - KEEPER_GOAL_ZONE else None
            if goal is None:
                continue
            keeper_rows.append({
                "match_id": match_id, "event_id": event["id"], "point": i, "event_type": event["type"]["name"],
                "period": event["period"], "time_ms": row["time_ms"], "goal_team": goal,
                "spec_consistent": f.spec_teams[i] == goal, "legacy_consistent": f.legacy_teams[i] == goal,
                "legacy_ambiguous": f.legacy_teams[i] == TEAM_IDENTITY_AMBIGUOUS,
                "new_status": f.statuses[i], "new_reason": f.reasons[i],
                "new_consistent": f.teams[i] == goal, "linked": f.component is not None,
                "record_other_points_unresolved": sum(s == IDENTITY_UNRESOLVED for j, s in enumerate(f.statuses) if j != i),
                "record_other_points_changed_vs_legacy": sum(
                    a != b and b != TEAM_IDENTITY_AMBIGUOUS and a != TEAM_IDENTITY_AMBIGUOUS
                    for j, (a, b) in enumerate(zip(f.teams, f.legacy_teams)) if j != i),
                "note": f.notes[i] if f.notes else "",
            })

    # Forward (held-out) validation of every goalkeeper correction: the next
    # keeper observation of the same goalkeeper (same allowance as the rule)
    # whose identity is verified on its own (spec reading, goal-side
    # consistent, not itself corrected). The rule only looked backwards.
    keeper_index = [(r["event_id"], r["point"], r["time_ms"], r["period"]) for r in keeper_rows]
    gk_rows = []
    for r in keeper_rows:
        if r["new_status"] != IDENTITY_CORRECTED_GOALKEEPER:
            continue
        here = np.array(geometry[r["event_id"]][0][r["point"]])
        verdict, nxt = "no_forward_evidence", ""
        for eid, point, t, period in keeper_index:
            gap = t - r["time_ms"]
            if period != r["period"] or gap <= 0 or gap > GOALKEEPER_MAX_GAP_MS or eid == r["event_id"]:
                continue
            there = np.array(geometry[eid][0][point])
            allowance = min(GOALKEEPER_MAX_DISTANCE, GOALKEEPER_BASE_DISTANCE + GOALKEEPER_SPEED_YD_PER_S * gap / 1000)
            if (here[0] < 60) != (there[0] < 60) or np.linalg.norm(here - there) > allowance:
                continue
            o = next(x for x in keeper_rows if x["event_id"] == eid and x["point"] == point)
            if o["new_status"] in (IDENTITY_SPEC, IDENTITY_CORRECTED_VIEW) and o["new_consistent"]:
                verdict, nxt = ("agrees" if o["goal_team"] == r["goal_team"] else "disagrees"), eid
                break
        gk_rows.append({**r, "forward_validation": verdict, "forward_event_id": nxt})

    # ML rows: legacy (HEAD builder) and new builder, same horizon (10 s).
    try:
        legacy = legacy_builder_head.build_rows_from_match_data(match_id, events, frames, 10.0)
        new = build_rows_from_match_data(match_id, events, frames, 10.0)
    except ValueError:
        legacy, new = [], []
    legacy_by_id = {r["current_event_id"]: r for r in legacy}
    ml_rows = []
    for r in new:
        old = legacy_by_id.get(r["current_event_id"], {})
        ml_rows.append({
            "match_id": match_id, **{f: r[f] for f in ML_FIELDS},
            "orientation_problem_a": r["orientation_problem_a"], "orientation_resolved_by": r["orientation_resolved_by"],
            "visible_teammates_raw": r["visible_teammates_raw"], "visible_opponents_raw": r["visible_opponents_raw"],
            "identity_corrected_points": r["identity_corrected_points"], "identity_quality": r["identity_quality"],
            **{f"legacy_{f}": old.get(f) for f in ("orientation", "actor_x", "actor_y", "visible_teammates",
                                                   "visible_opponents", "visible_ambiguous", "next_event_type")},
            "in_legacy": bool(old),
        })
    missing_in_new = len(set(legacy_by_id) - {r["current_event_id"] for r in new})
    event_types = Counter(e["type"]["name"] for e in events)
    return {
        "records": rec_rows, "keepers": keeper_rows, "gk": gk_rows, "ml": ml_rows,
        "match": {"match_id": match_id, "events": len(events), "events_with_360_record": sum(e["id"] in frames for e in events),
                  "usable_records": len(records), "ml_rows": len(new), "legacy_ml_rows": len(legacy),
                  "legacy_rows_missing_in_new": missing_in_new,
                  "substitution_events": event_types["Substitution"]},
    }


def write(path: Path, rows: list[dict]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--match-id", type=int, nargs="*")
    args = parser.parse_args()
    excluded = load_exclusion_manifest()
    match_ids = [m for m in (args.match_id or available_360_match_ids()) if m not in excluded]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(pool.map(audit_match, match_ids, chunksize=1))
    suffix = "" if not args.match_id else "_subset"
    write(OUT / f"records{suffix}.csv.gz", [r for o in outputs for r in o["records"]])
    write(OUT / f"keepers{suffix}.csv.gz", [r for o in outputs for r in o["keepers"]])
    gk = [r for o in outputs for r in o["gk"]]
    if gk:
        write(OUT / f"goalkeeper_rule{suffix}.csv", gk)
    write(OUT / f"ml_rows{suffix}.csv.gz", [r for o in outputs for r in o["ml"]])
    write(OUT / f"match_events{suffix}.csv", [o["match"] for o in outputs])
    print(len(match_ids), "matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
