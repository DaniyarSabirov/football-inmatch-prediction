"""Import exported review decisions into the physical-orientation registry.

Input: the JSON file exported by results/physical_orientation/
final_orientation_audit/review/review.html (Export decisions button).

Period-scoped decisions (a decision object with a "periods" list, e.g.
[2]) are written to the registry's separate "period_reviews" list, one
entry per period -- never to "calibrations" -- with evidence_type
"user_review_of_renders" unless the reviewer explicitly chose
"independent_video" AND supplied source_url and match_clock. Periods not
listed are left untouched.

Match-level rules (decision without "periods"; never invent a calibration):
  - decision "proposed"/"alternative" WITH a source_url AND a match_clock
    -> verification_status "verified" (the reviewer watched identifiable
       footage); the period-1 reference team is taken from the option chosen.
  - the same decision without that evidence -> "partially_verified" (kept,
    never auto-applied by the renderer).
  - "cannot_tell" -> "ambiguous" (kept for the record, never auto-applied).
  - an existing registry entry for the match is never overwritten unless
    --replace is given.

Default is a dry run that prints what would change; pass --write to save.

Usage:
    .venv/bin/python scripts/import_orientation_reviews.py decisions.json [--write] [--replace]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.physical_orientation_calibration import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    EVIDENCE_INDEPENDENT_VIDEO,
    EVIDENCE_USER_RENDER_REVIEW,
    load_calibration_registry,
    load_frame_reviews,
    load_period_reviews,
)

REPORT = "docs/final_statsbomb_360_orientation_research.md"


def entry_from_decision(decision: dict) -> dict | None:
    choice = decision.get("decision")
    if choice not in ("proposed", "alternative", "cannot_tell"):
        return None
    has_evidence = bool((decision.get("source_url") or "").strip()) and bool((decision.get("match_clock") or "").strip())
    if choice == "cannot_tell":
        status, team = "ambiguous", decision.get("proposed_period_1_reference")
    else:
        status = "verified" if has_evidence else "partially_verified"
        team = decision.get(f"{choice}_period_1_reference")
    if not team:
        return None
    landmark = "; ".join(
        part for part in (
            f"clock {decision.get('match_clock')}" if decision.get("match_clock") else "",
            decision.get("goal_side_evidence") or "",
            decision.get("camera_side_evidence") or "",
        ) if part
    )
    return {
        "match_id": int(decision["match_id"]),
        "period_1_physical_reference_team": team,
        "period_handling_rule": "per_period_visible_area_end_relations",
        "horizontal_orientation_verified": status == "verified",
        "vertical_orientation_verified": status == "verified",
        "vertical_orientation_note": "A 180-degree choice fixes left/right and near/far together (no reflection exists in a proper map).",
        "source_url": decision.get("source_url") or None,
        "source_description": f"Manual review via review.html; video timestamp {decision.get('video_timestamp') or 'n/a'}.",
        "evidence_landmark": landmark or "not recorded",
        "verification_method": f"Human comparison of real footage with the proposed/alternative renders; decision={choice}.",
        "confidence": "high" if status == "verified" else "low",
        "verification_status": status,
        "verified_by": decision.get("reviewer") or "unrecorded reviewer",
        "date_verified": (decision.get("reviewed_at") or date.today().isoformat())[:10],
        "reviewer_notes": decision.get("notes") or "",
        "report": REPORT,
    }


def period_reviews_from_decision(decision: dict) -> list[dict]:
    """One period_reviews entry per listed period; [] when not applicable."""
    choice = decision.get("decision")
    periods = decision.get("periods") or []
    if choice not in ("proposed", "alternative") or not periods:
        return []
    references = decision.get(f"{choice}_reference_by_period") or {}
    has_video = (
        decision.get("evidence_type") == EVIDENCE_INDEPENDENT_VIDEO
        and bool((decision.get("source_url") or "").strip())
        and bool((decision.get("match_clock") or "").strip())
    )
    entries = []
    for period in periods:
        team = references.get(str(period)) or references.get(period)
        if not team:
            raise ValueError(f"match {decision.get('match_id')}: no {choice} reference recorded for period {period}")
        entries.append({
            "match_id": int(decision["match_id"]),
            "period": int(period),
            "display_reference_team": team,
            "decision": choice,
            "evidence_type": EVIDENCE_INDEPENDENT_VIDEO if has_video else EVIDENCE_USER_RENDER_REVIEW,
            "statement": decision.get("statement") or "",
            "source_url": decision.get("source_url") or None,
            "reviewed_by": decision.get("reviewer") or "unrecorded reviewer",
            "date_reviewed": (decision.get("reviewed_at") or date.today().isoformat())[:10],
            "notes": decision.get("notes") or "",
            "report": REPORT,
        })
    return entries


def frame_review_from_decision(decision: dict) -> dict:
    """A frame-level decision ("frame_event_ids" present) -> one frame_reviews entry.

    "current" keeps the snapshot as currently drawn; "alternative" flips the
    correction. `current_snapshot_correction` is the correction the renderer
    applied when the sheet was made ("NONE" for uncorrected snapshots). Never
    video evidence unless the reviewer says so and gives a URL and clock.
    """
    choice = decision["decision"]
    if choice not in ("current", "alternative"):
        raise ValueError(f"frame decision must be current/alternative, got {choice!r}")
    stored = decision.get("current_snapshot_correction", "NONE")
    correction = stored if choice == "current" else ("NONE" if stored == "ROTATED_180" else "ROTATED_180")
    has_video = bool(decision.get("evidence_type") == EVIDENCE_INDEPENDENT_VIDEO and decision.get("source_url")
                     and decision.get("match_clock"))
    return {
        "match_id": int(decision["match_id"]), "period": int(decision["period"]),
        "event_ids": list(decision["frame_event_ids"]), "timestamp": decision.get("timestamp", ""),
        "selected_panel": decision.get("selected_panel", ""), "decision": choice,
        "snapshot_correction": correction,
        "evidence_type": EVIDENCE_INDEPENDENT_VIDEO if has_video else EVIDENCE_USER_RENDER_REVIEW,
        "reviewed_by": decision.get("reviewer") or "unrecorded reviewer",
        "date_reviewed": (decision.get("reviewed_at") or date.today().isoformat())[:10],
        "sheet": decision.get("sheet", ""), "open_issue": decision.get("open_issue", ""),
        "notes": decision.get("notes", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    load_calibration_registry(args.registry)  # fail loudly on a broken registry first
    registry = json.loads(args.registry.read_text(encoding="utf-8")) if args.registry.exists() else {"calibrations": []}
    existing = {entry["match_id"]: index for index, entry in enumerate(registry["calibrations"])}
    load_period_reviews(args.registry)
    reviews = registry.setdefault("period_reviews", [])
    existing_reviews = {(e["match_id"], e["period"]): i for i, e in enumerate(reviews)}

    changes = 0
    load_frame_reviews(args.registry)
    frame_reviews = registry.setdefault("frame_reviews", [])
    reviewed_events = {e: i for i, entry in enumerate(frame_reviews) for e in entry["event_ids"]}
    for decision in json.loads(args.decisions.read_text(encoding="utf-8")):
        if decision.get("frame_event_ids"):
            entry = frame_review_from_decision(decision)
            existing_index = next((reviewed_events[e] for e in entry["event_ids"] if e in reviewed_events), None)
            if existing_index is not None and not args.replace:
                print(f"keep frame review {entry['match_id']} {entry['timestamp']}: already recorded (use --replace)")
                continue
            print(f"{'replace' if existing_index is not None else 'add'} frame review {entry['match_id']} P{entry['period']} "
                  f"{entry['timestamp']}: {entry['decision']} -> snapshot_correction {entry['snapshot_correction']}")
            if existing_index is not None:
                frame_reviews[existing_index] = entry
            else:
                for e in entry["event_ids"]:
                    reviewed_events[e] = len(frame_reviews)
                frame_reviews.append(entry)
            changes += 1
            continue
        if decision.get("periods"):
            for review in period_reviews_from_decision(decision):
                key = (review["match_id"], review["period"])
                if key in existing_reviews and not args.replace:
                    print(f"keep {key}: period review already recorded (use --replace)")
                    continue
                print(f"{'replace' if key in existing_reviews else 'add'} period review {key}: "
                      f"{review['evidence_type']}, display reference {review['display_reference_team']}")
                if key in existing_reviews:
                    reviews[existing_reviews[key]] = review
                else:
                    existing_reviews[key] = len(reviews)
                    reviews.append(review)
                changes += 1
            continue
        entry = entry_from_decision(decision)
        if entry is None:
            print(f"skip {decision.get('match_id')}: no usable decision")
            continue
        match_id = entry["match_id"]
        if match_id in existing and not args.replace:
            print(f"keep {match_id}: registry already has an entry (use --replace)")
            continue
        print(f"{'replace' if match_id in existing else 'add'} {match_id}: {entry['verification_status']}, "
              f"period-1 reference {entry['period_1_physical_reference_team']}")
        if match_id in existing:
            registry["calibrations"][existing[match_id]] = entry
        else:
            registry["calibrations"].append(entry)
        changes += 1

    if args.write and changes:
        args.registry.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        load_calibration_registry(args.registry)  # re-validate what was written
        load_period_reviews(args.registry)
        load_frame_reviews(args.registry)
        print(f"wrote {changes} change(s) to {args.registry}")
    elif changes:
        print(f"dry run: {changes} change(s) not written (pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
