"""Row-by-row audit of the 55 dataset rows whose orientation (as built at
commit 3423816) came from a route that reads later records
(orientation_uses_later_records). For each: why the record needed a
non-geometric route, what the route used, the offline vs causal orientation
and actor position under the CURRENT builder, and the identity features.

Input: data/processed/statsbomb_360_all_matches_next_events.csv as built at
3423816 (read before the rebuild; a copy of the 55 rows is saved first).
Output: ../orientation_55_rows.csv, ../orientation_55_summary.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.build_statsbomb_360_dataset import build_match_dataset  # noqa: E402
from src.orientation_correction import classify_orientation  # noqa: E402

PREVIOUS = PROJECT_ROOT / "data" / "processed" / "statsbomb_360_all_matches_next_events.csv"
SNAPSHOT = OUT / "orientation_55_rows_as_built_at_3423816.csv"
DATA = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
WHY = {
    "display_duplicate_snapshot": "UNCERTAIN; another copy of the same raw snapshot attached to another event (possibly LATER in the stream) had a confident orientation",
    "user_frame_review": "UNCERTAIN; decided by a human reviewer who watched the rendered sequence (including later frames)",
    "temporal_neighbor": "Problem A's neighbour vote used a confidently NORMAL neighbour that may lie AFTER the record",
    "display_duplicate_consensus": "confident copies disagreed; the group was aligned on the period camera side",
    "display_camera_continuity": "camera side + nearest confident neighbour on either side",
}


def main() -> int:
    if not SNAPSHOT.exists():
        previous = pd.read_csv(PREVIOUS, low_memory=False)
        previous[previous.orientation_uses_later_records].to_csv(SNAPSHOT, index=False)
    old = pd.read_csv(SNAPSHOT)
    rows = []
    for match_id, group in old.groupby("match_id"):
        new = {r["current_event_id"]: r for r in build_match_dataset(int(match_id), 10.0)}
        events = {e["id"]: e for e in json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))}
        frames = {f["event_uuid"]: f for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
        for r in group.itertuples():
            n = new[r.current_event_id]
            event, frame = events[r.current_event_id], frames[r.current_event_id]
            geometry = classify_orientation(event, frame)
            later_copies = sum(
                1 for f_id, f in frames.items()
                if f_id != r.current_event_id and f_id in events and f.get("freeze_frame") == frame.get("freeze_frame")
                and events[f_id]["period"] == event["period"] and events[f_id]["index"] > event["index"]
            )
            earlier_copies = sum(
                1 for f_id, f in frames.items()
                if f_id != r.current_event_id and f_id in events and f.get("freeze_frame") == frame.get("freeze_frame")
                and events[f_id]["period"] == event["period"] and events[f_id]["index"] < event["index"]
            )
            rows.append({
                "match_id": match_id, "event_id": r.current_event_id, "event_index": r.current_event_index,
                "period": r.period, "timestamp": r.timestamp, "event_type": r.current_event_type,
                "team": r.current_team, "next_event_type": r.next_event_type,
                "offline_route": r.orientation_resolved_by, "why_offline_needed_more": WHY.get(r.orientation_resolved_by, ""),
                "geometry_alone": geometry.orientation, "geometry_confidence": round(geometry.confidence, 3),
                "event_location": event.get("location"), "earlier_identical_copies": earlier_copies,
                "later_identical_copies": later_copies,
                "offline_orientation": n["orientation_offline"], "causal_orientation": n["orientation"],
                "causal_route": n["orientation_resolved_by"],
                "actor_xy_at_3423816": (r.actor_x, r.actor_y), "actor_xy_offline_now": (n["actor_x_offline"], n["actor_y_offline"]),
                "actor_xy_causal_now": (n["actor_x"], n["actor_y"]),
                "visible_teammates_causal": n["visible_teammates"], "visible_opponents_causal": n["visible_opponents"],
                "visible_ambiguous_causal": n["visible_ambiguous"],
                "visible_teammates_offline": n["visible_teammates_offline"], "visible_opponents_offline": n["visible_opponents_offline"],
            })
    df = pd.DataFrame(rows)
    df["offline_rotated_raw"] = df.offline_orientation == "ROTATED_180"
    df["causal_matches_offline"] = df.causal_orientation == df.offline_orientation
    df["causal_unavailable"] = df.causal_orientation == "UNCERTAIN"
    df["identity_causal_equals_offline"] = (
        (df.visible_teammates_causal == df.visible_teammates_offline) & (df.visible_opponents_causal == df.visible_opponents_offline)
    )
    df.to_csv(OUT / "orientation_55_rows.csv", index=False)
    summary = {
        "rows": len(df), "matches": int(df.match_id.nunique()),
        "offline_route": Counter(df.offline_route), "event_types": Counter(df.event_type),
        "geometry_alone": Counter(df.geometry_alone),
        "rows_with_a_later_identical_copy": int((df.later_identical_copies > 0).sum()),
        "rows_with_an_earlier_identical_copy": int((df.earlier_identical_copies > 0).sum()),
        "offline_orientation": Counter(df.offline_orientation),
        "causal_route": Counter(df.causal_route),
        "causal_matches_offline": int(df.causal_matches_offline.sum()),
        "causal_unavailable_actor_xy_empty": int(df.causal_unavailable.sum()),
        "causal_decided_but_differs_from_offline": int((~df.causal_unavailable & ~df.causal_matches_offline).sum()),
        "identity_causal_equals_offline": int(df.identity_causal_equals_offline.sum()),
    }
    (OUT / "orientation_55_summary.json").write_text(json.dumps(summary, indent=1, default=dict), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=dict))
    pd.set_option("display.width", 250)
    print(df[["match_id", "timestamp", "event_type", "offline_route", "geometry_alone", "earlier_identical_copies",
              "later_identical_copies", "offline_orientation", "causal_orientation", "causal_route"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
