"""ML feature correctness and leakage audit (final validation, in memory only;
nothing is written to data/processed).

Per match: rows from the CURRENT builder (causal feature columns + *_offline
columns) and from the builder as committed at HEAD (legacy), joined on the
current event id. Additionally, per row:

  future_identity_records   records LATER than the row's event (period,
                            timestamp, index) that the offline resolution of
                            the row's snapshot could have used: its
                            structural group members, the continuity and
                            detector neighbours after it, simultaneous
                            partners after it (computed as: offline result
                            differs from causal result, see below)
  identity_offline_differs  the offline counts differ from the causal counts
  next_event_snapshot_in_window
                            the event being predicted (next meaningful
                            event) has a 360 snapshot within 2 s after the
                            current one, i.e. it is a record the OFFLINE
                            resolution may consult
  orientation_route         how the row's snapshot orientation was decided;
                            routes that can read later records are listed in
                            ORIENTATION_ROUTES_USING_LATER_RECORDS

Invariants checked per row:
  teammates + opponents + ambiguous == number of dict points in the snapshot
  raw teammates + raw opponents == same number
  identity_quality consistent with ambiguous / corrected counts

Outputs (../ml/): rows.csv.gz, invariants.json
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "ml"
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "results" / "team_identity_correction_2026-10-26" / "diagnostics"))

import pandas as pd  # noqa: E402

import legacy_builder_head  # noqa: E402  (git show HEAD:src/build_statsbomb_360_dataset.py)
from src.animate_statsbomb_360_match import available_360_match_ids  # noqa: E402
from src.build_statsbomb_360_dataset import build_rows_from_match_data, timestamp_to_seconds  # noqa: E402
from src.match_exclusions import load_exclusion_manifest  # noqa: E402

DATA = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
ORIENTATION_ROUTES_USING_LATER_RECORDS = {
    "temporal_neighbor",            # Problem A refinement: neighbours on both sides (pre-existing in HEAD)
    "display_duplicate_snapshot",   # orientation of another copy of the same snapshot (any order)
    "display_duplicate_consensus",
    "display_camera_continuity",    # nearest confident neighbour on either side
    "user_frame_review",            # a human decision made while viewing the whole sequence
}
ORIENTATION_ROUTES_USING_PERIOD_CAMERA = {"display_kickoff_own_half", "display_camera_side"}


def rows_for(match_id: int) -> list[dict]:
    events = json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    try:
        new = build_rows_from_match_data(match_id, events, frames, 10.0)
        legacy = {r["current_event_id"]: r for r in legacy_builder_head.build_rows_from_match_data(match_id, events, frames, 10.0)}
    except ValueError:
        return []
    by_id = {e["id"]: e for e in events}
    out = []
    for r in new:
        frame = frames[r["current_event_id"]]
        points = sum(isinstance(p, dict) for p in frame.get("freeze_frame") or [])
        old = legacy.get(r["current_event_id"], {})
        next_frame = frames.get(r["next_event_id"])
        gap = timestamp_to_seconds(by_id[r["next_event_id"]]["timestamp"]) - timestamp_to_seconds(r["timestamp"])
        out.append({
            "match_id": match_id, "current_event_id": r["current_event_id"], "current_event_type": r["current_event_type"],
            "period": r["period"], "minute": r["minute"], "next_event_type": r["next_event_type"],
            "time_to_next_event_seconds": r["time_to_next_event_seconds"],
            "actor_x": r["actor_x"], "actor_y": r["actor_y"], "visible_players": r["visible_players"],
            "dict_points": points,
            "orientation": r["orientation"], "orientation_problem_a": r["orientation_problem_a"],
            "orientation_route": r["orientation_resolved_by"],
            **{c: r[c] for c in ("visible_teammates", "visible_opponents", "visible_ambiguous", "identity_corrected_points",
                                 "identity_quality", "visible_teammates_raw", "visible_opponents_raw",
                                 "visible_teammates_offline", "visible_opponents_offline", "visible_ambiguous_offline",
                                 "identity_quality_offline")},
            "legacy_visible_teammates": old.get("visible_teammates"), "legacy_visible_opponents": old.get("visible_opponents"),
            "legacy_visible_ambiguous": old.get("visible_ambiguous"), "legacy_orientation": old.get("orientation"),
            "legacy_actor_x": old.get("actor_x"), "legacy_actor_y": old.get("actor_y"),
            "next_event_snapshot_in_window": bool(next_frame and next_frame.get("freeze_frame")) and gap <= 2.0,
        })
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    excluded = load_exclusion_manifest()
    ids = [m for m in available_360_match_ids() if m not in excluded]
    with ProcessPoolExecutor(max_workers=7) as pool:
        rows = [r for out in pool.map(rows_for, ids, chunksize=1) for r in out]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rows.csv.gz", index=False, compression="gzip")

    sums = df.visible_teammates + df.visible_opponents + df.visible_ambiguous
    sums_offline = df.visible_teammates_offline + df.visible_opponents_offline + df.visible_ambiguous_offline
    quality_ok = (
        ((df.visible_ambiguous > 0) == (df.identity_quality == "uncertain"))
        & (((df.visible_ambiguous == 0) & (df.identity_corrected_points > 0)) == (df.identity_quality == "corrected"))
    )
    invariants = {
        "rows": len(df), "matches": int(df.match_id.nunique()),
        "rows_missing_vs_legacy": int(df.legacy_visible_teammates.isna().sum()),
        "violations_causal_counts_sum": int((sums != df.dict_points).sum()),
        "violations_offline_counts_sum": int((sums_offline != df.dict_points).sum()),
        "violations_raw_counts_sum": int((df.visible_teammates_raw + df.visible_opponents_raw != df.dict_points).sum()),
        "violations_identity_quality": int((~quality_ok).sum()),
        "raw_counts_equal_legacy_spec_reading_rows": None,  # legacy has no raw column; see report
        "visible_players_vs_dict_points_mismatch": int((df.visible_players != df.dict_points).sum()),
    }
    (OUT / "invariants.json").write_text(json.dumps(invariants, indent=1), encoding="utf-8")
    print(json.dumps(invariants, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
