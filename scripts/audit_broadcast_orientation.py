"""Dataset-wide audit of broadcast-camera-side orientation (Problem D).

For every StatsBomb 360 match file, this script:
  1. loads and joins the match with the production loader;
  2. runs Problem A (orientation_correction) exactly as the renderer does;
  3. infers the camera-at-bottom reference team PER PERIOD from visible_area
     (src/broadcast_orientation.py), with no reference-team input;
  4. compares that with the legacy automatic default (kickoff-team baseline
     + halftime swap) and with any registry calibration;
  5. applies the inferred transform and re-checks the result with signals
     that the inference never used: goalkeeper positions, shot origins,
     actor/ball distance, ball-inside-visible-area, and the camera side of
     the TRANSFORMED polygons;
  6. classifies the match into one of five categories.

Writes (under results/physical_orientation/final_orientation_audit/):
  per_match_audit.json     machine-readable, one object per match
  per_match_audit.csv      one flat row per match
  audit_summary.json       dataset-wide counts

Read-only with respect to raw data, the registry, and the ML pipeline.

Usage:
    .venv/bin/python scripts/audit_broadcast_orientation.py [--match-id N ...] [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.animate_statsbomb_360_match import (  # noqa: E402
    VisualizationError,
    apply_display_plan_to_match,
    available_360_match_ids,
    compute_shared_group_info,
    compute_orientation_results,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    load_match_metadata_index,
    apply_orientation_correction_to_match,
    match_team_ids,
)
from src.broadcast_orientation import (  # noqa: E402
    BROADCAST_STATUS_INFERRED,
    build_display_plan,
    CAMERA_SIDE_HIGH_Y,
    CAMERA_SIDE_UNKNOWN,
    _point_in_polygon,
    _polygon_vertices,
    estimate_camera_side,
)
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.orientation_correction import ORIENTATION_UNCERTAIN  # noqa: E402
from src.physical_orientation_calibration import load_calibration_registry, load_period_reviews  # noqa: E402
from src.team_identity import TEAM_IDENTITY_AMBIGUOUS, resolve_absolute_team_for_point  # noqa: E402
from src.team_normalization import apply_period_end_swap, resolve_reference_team_id  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "physical_orientation" / "final_orientation_audit"

CATEGORY_VERIFIED = "physically_verified"
CATEGORY_INFERRED = "high_confidence_inferred"
# Every camera-undecided regular period is reviewer-confirmed or derived from a
# reviewer-confirmed period by the halftime rule, and that derivation is
# corroborated by the period's own camera majority and the independent checks.
CATEGORY_USER_REVIEWED_DERIVED = "user_reviewed_and_derived"
CATEGORY_AMBIGUOUS = "geometrically_consistent_physically_ambiguous"
CATEGORY_CONFLICTING = "conflicting_or_unreliable"
CATEGORY_EXCLUDED = "excluded_insufficient_data"

# Independent post-transform checks must pass at these rates.
MINIMUM_KEEPER_CONSISTENCY = 0.90
MINIMUM_CAMERA_BOTTOM_RATE = 0.90


def _team(event: dict[str, Any]) -> tuple[int | None, str]:
    team = event.get("team") or {}
    return team.get("id"), team.get("name", "Unknown")


def _goalkeepers(events_path: Path) -> dict[str, list[str]]:
    with events_path.open(encoding="utf-8") as file:
        events = json.load(file)
    keepers: dict[str, list[str]] = {}
    for event in events:
        if (event.get("position") or {}).get("name") == "Goalkeeper" and event.get("player"):
            names = keepers.setdefault(_team(event)[1], [])
            if event["player"]["name"] not in names:
                names.append(event["player"]["name"])
    return keepers


def _post_transform_checks(display_frames, raw_frames, shared, reference_by_period, team_names) -> dict[int, dict[str, Any]]:
    """Signals NOT used by the inference, evaluated in the display frame.

    Keeper team identity comes from the production Problem C resolver
    (src/team_identity.py), so reused cross-team snapshots do not show up
    as false orientation errors; unresolvable keepers are skipped.
    """
    per_period: dict[int, dict[str, Any]] = {}
    raw_events_by_id = {j.event["id"]: j.event for j in raw_frames}
    pair = tuple(team_names[t] for t in sorted(team_names))
    for joined, raw_joined in zip(display_frames, raw_frames):
        event = joined.event
        period = event.get("period")
        if period not in reference_by_period:
            continue
        stats = per_period.setdefault(
            period,
            {"keeper_ok": 0, "keeper_total": 0, "camera_bottom": 0, "camera_known": 0,
             "ball_in_polygon": 0, "polygon_total": 0, "actor_distances": [], "shots": {}},
        )
        team_id, _ = _team(event)
        reference = reference_by_period[period]
        frame = joined.frame_360

        side = estimate_camera_side(frame.get("visible_area")).side
        if side != CAMERA_SIDE_UNKNOWN:
            stats["camera_known"] += 1
            stats["camera_bottom"] += side == CAMERA_SIDE_HIGH_Y

        location = event.get("location")
        polygon = _polygon_vertices(frame.get("visible_area"))
        if isinstance(location, list) and len(location) == 2 and polygon:
            stats["polygon_total"] += 1
            stats["ball_in_polygon"] += _point_in_polygon((location[0], location[1]), polygon)

        for player in frame.get("freeze_frame") or []:
            point = player.get("location")
            if not isinstance(point, list) or len(point) != 2:
                continue
            if player.get("actor") and isinstance(location, list) and len(location) == 2:
                stats["actor_distances"].append(((point[0] - location[0]) ** 2 + (point[1] - location[1]) ** 2) ** 0.5)
            if player.get("keeper"):
                keeper_team = resolve_absolute_team_for_point(
                    player, raw_joined.event, shared.get(event["id"]), raw_events_by_id, pair
                )
                if keeper_team == TEAM_IDENTITY_AMBIGUOUS:
                    continue
                keeper_team_is_reference = keeper_team == team_names[reference]
                # The reference team attacks +x, so its keeper defends x < 60.
                stats["keeper_total"] += 1
                stats["keeper_ok"] += (point[0] < 60) == keeper_team_is_reference

        if (event.get("type") or {}).get("name") == "Shot" and isinstance(location, list):
            stats["shots"].setdefault(team_names[team_id], []).append(location[0])

    result: dict[int, dict[str, Any]] = {}
    for period, stats in per_period.items():
        reference_name = team_names[reference_by_period[period]]
        shot_means = {team: round(statistics.fmean(xs), 1) for team, xs in stats["shots"].items()}
        shots_consistent = None
        if len(shot_means) == 2 and all(len(xs) >= 2 for xs in stats["shots"].values()):
            shots_consistent = all((mean > 60) == (team == reference_name) for team, mean in shot_means.items())
        result[period] = {
            "keeper_consistency": round(stats["keeper_ok"] / stats["keeper_total"], 4) if stats["keeper_total"] else None,
            "keeper_points": stats["keeper_total"],
            "camera_bottom_rate": round(stats["camera_bottom"] / stats["camera_known"], 4) if stats["camera_known"] else None,
            "ball_in_polygon_rate": round(stats["ball_in_polygon"] / stats["polygon_total"], 4) if stats["polygon_total"] else None,
            "median_actor_ball_distance": round(statistics.median(stats["actor_distances"]), 3) if stats["actor_distances"] else None,
            "shot_mean_x": shot_means,
            "shots_consistent_with_direction": shots_consistent,
            "reference_team_attacks_right": reference_name,
        }
    return result


def _representative_event_ids(display_frames, reference_by_period) -> dict[int, str]:
    """Per period, the snapshot nearest the period's middle that shows a keeper."""
    by_period: dict[int, list] = {}
    for joined in display_frames:
        by_period.setdefault(joined.event.get("period"), []).append(joined)
    chosen: dict[int, str] = {}
    for period, frames in by_period.items():
        if period not in reference_by_period:
            continue
        with_keeper = [j for j in frames if any(p.get("keeper") for p in j.frame_360.get("freeze_frame") or [])] or frames
        middle = frames[len(frames) // 2].event.get("timestamp", "")
        chosen[period] = min(with_keeper, key=lambda j: abs(_seconds(j.event.get("timestamp", "")) - _seconds(middle))).event["id"]
    return chosen


def _seconds(timestamp: str) -> float:
    try:
        hours, minutes, seconds = timestamp.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return 0.0


def audit_match(match_id: int, excluded: dict, registry: dict, metadata, period_reviews: dict | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"match_id": match_id}
    if metadata is not None:
        row.update(
            competition=metadata.competition, season=metadata.season, date=metadata.date,
            home_team=metadata.home_team, away_team=metadata.away_team, score=metadata.score,
        )
    if match_id in excluded:
        row.update(category=CATEGORY_EXCLUDED, reason=f"excluded manifest: {excluded[match_id].reason}")
        return row
    try:
        match = load_and_join_match(match_id)
        team_ids = match_team_ids(match)
    except (VisualizationError, ValueError) as error:
        row.update(category=CATEGORY_EXCLUDED, reason=f"load failure: {error}")
        return row

    team_names = {}
    for joined in match.usable_frames:
        team_id, name = _team(joined.event)
        team_names.setdefault(team_id, name)
    row["teams"] = [team_names[t] for t in team_ids]
    row["goalkeepers"] = _goalkeepers(match.event_path)
    row["usable_snapshots"] = match.usable_count
    row["snapshots_with_visible_area"] = sum(bool(j.frame_360.get("visible_area")) for j in match.usable_frames)

    orientation_results = compute_orientation_results(match)
    row["problem_a_counts"] = dict(Counter(r.orientation for r in orientation_results))
    inference = infer_broadcast_orientation_for_match(match, orientation_results)
    events = [j.event for j in match.usable_frames]
    periods_present = sorted({e.get("period") for e in events})
    row["periods"] = periods_present

    automatic = resolve_reference_team_id(events)
    legacy = {p: apply_period_end_swap(events, automatic, p) for p in periods_present}
    row["automatic_reference_team"] = team_names[automatic]
    row["legacy_reference_by_period"] = {p: team_names[t] for p, t in legacy.items()}

    row["camera_status"] = inference.status
    row["camera_explanation"] = inference.explanation
    row["camera_reference_by_period"] = {p: team_names[t] for p, t in inference.reference_by_period.items()}
    row["camera_period_evidence"] = {
        e.period: {
            "support": e.support, "opposition": e.opposition, "agreement": round(e.agreement, 4),
            "decided": e.decided,
            "votes_by_team": {team_names[t]: v for t, v in e.votes_by_team.items()},
        }
        for e in inference.periods
    }
    row["end_swap_checks"] = [
        {"periods": [c.period_a, c.period_b], "required_by_laws": c.required, "swapped": c.swapped}
        for c in inference.end_swap_checks
    ]
    decided_regular = [p for p in inference.reference_by_period if p in legacy and p != 5]
    row["legacy_default_matches_camera"] = (
        all(legacy[p] == inference.reference_by_period[p] for p in decided_regular) if decided_regular else None
    )
    row["legacy_periods_disagreeing"] = [p for p in decided_regular if legacy[p] != inference.reference_by_period[p]]

    calibration = registry.get(match_id)
    row["registry_status"] = calibration.verification_status if calibration else None
    registry_agrees = None
    if calibration is not None and 1 in inference.reference_by_period:
        registry_agrees = (
            team_names[inference.reference_by_period[1]].casefold()
            == calibration.period_1_physical_reference_team.casefold()
        )
    row["registry_period_1_reference"] = calibration.period_1_physical_reference_team if calibration else None
    row["registry_agrees_with_camera"] = registry_agrees

    reviews = period_reviews or {}
    confirmed = {p: resolve_reference_team_id(events, r.display_reference_team) for p, r in reviews.items()}
    row["period_reviews"] = {
        p: {"display_reference_team": r.display_reference_team, "evidence_type": r.evidence_type,
            "independently_verified": r.independently_verified, "reviewed_by": r.reviewed_by,
            "date_reviewed": r.date_reviewed,
            "agrees_with_camera": (inference.reference_by_period.get(p) == confirmed[p]) if p in inference.reference_by_period else None}
        for p, r in reviews.items()
    }
    plan = build_display_plan(
        inference, team_ids, periods_present, fallback_period_1_reference=automatic, confirmed_references=confirmed,
    )
    row["display_plan_provenance"] = plan.provenance
    row["display_reference_by_period"] = {p: team_names[t] for p, t in plan.reference_by_period().items()}
    row["display_source_by_period"] = {p: t.source for p, t in plan.transforms.items()}
    row["display_plan"] = plan.explanation
    checks: dict[int, dict[str, Any]] = {}
    if plan.reference_by_period():
        corrected = apply_orientation_correction_to_match(match, orientation_results)
        keep = [
            i for i, (j, r) in enumerate(zip(corrected.usable_frames, orientation_results))
            if r.orientation != ORIENTATION_UNCERTAIN and j.event.get("period") in plan.transforms
        ]
        display = apply_display_plan_to_match(corrected, plan).usable_frames
        display = [display[i] for i in keep]
        raw_kept = [match.usable_frames[i] for i in keep]
        checks = _post_transform_checks(
            display, raw_kept, compute_shared_group_info(match), plan.reference_by_period(), team_names
        )
        row["representative_event_ids"] = _representative_event_ids(display, plan.reference_by_period())
    row["post_transform_checks"] = checks

    # A period resolved through a user-reviewed neighbour is not decided by the
    # camera, so its own camera reading must only SUPPORT the result (majority),
    # not reach the 0.90 decision threshold.
    reviewed_sources = ("user_confirmed", "derived_from_user_reviewed_period_and_halftime_rule")
    independent_ok = bool(checks) and all(
        (c["keeper_consistency"] is None or c["keeper_consistency"] >= MINIMUM_KEEPER_CONSISTENCY)
        and (c["camera_bottom_rate"] is None or c["camera_bottom_rate"] >= (
            0.5 if plan.transforms[p].source in reviewed_sources else MINIMUM_CAMERA_BOTTOM_RATE))
        and c["shots_consistent_with_direction"] is not False
        for p, c in checks.items() if p != 5
    )
    row["independent_checks_pass"] = independent_ok

    if registry_agrees is False:
        category, reason = CATEGORY_CONFLICTING, "camera inference contradicts registry calibration"
    elif calibration is not None and calibration.verification_status == "verified" and registry_agrees:
        category, reason = CATEGORY_VERIFIED, "registry-verified and camera inference agrees"
    elif inference.status != "CONFLICTING" and independent_ok and _undecided_periods_reviewed_or_derived(inference, plan):
        category, reason = CATEGORY_USER_REVIEWED_DERIVED, (
            "camera-undecided period(s) resolved by a user-reviewed period and the halftime rule; "
            "own camera majority and independent checks agree"
        )
    elif inference.status == BROADCAST_STATUS_INFERRED and independent_ok:
        category, reason = CATEGORY_INFERRED, "all regular periods decided; independent checks pass"
    elif inference.status in ("CONFLICTING",) or (inference.status == BROADCAST_STATUS_INFERRED and not independent_ok):
        category, reason = CATEGORY_CONFLICTING, inference.explanation if inference.status != BROADCAST_STATUS_INFERRED else "independent post-transform check failed"
    elif inference.status == "NO_EVIDENCE" and match.usable_count < 50:
        category, reason = CATEGORY_EXCLUDED, "too little 360 coverage"
    else:
        category, reason = CATEGORY_AMBIGUOUS, inference.explanation
    if reviews:
        reason += "; period review(s): " + ", ".join(
            f"P{p} {r.evidence_type}" for p, r in sorted(reviews.items())
        )
    row.update(category=category, reason=reason)
    return row


def _undecided_periods_reviewed_or_derived(inference, plan) -> bool:
    undecided = [e for e in inference.periods if not e.decided and e.frame_model == "OPPOSITE"]
    if not undecided:
        return False
    for evidence in undecided:
        transform = plan.transforms.get(evidence.period)
        if transform is None or transform.source not in ("user_confirmed", "derived_from_user_reviewed_period_and_halftime_rule"):
            return False
        first, second = plan.team_ids
        other = {first: second, second: first}
        score = {t: evidence.votes_by_team[t].get(CAMERA_SIDE_HIGH_Y, 0) + evidence.votes_by_team[other[t]].get("LOW_Y", 0)
                 for t in plan.team_ids}
        if max(plan.team_ids, key=lambda t: score[t]) != transform.reference_team_id or score[first] == score[second]:
            return False
    return True


def _worker(args):
    match_id, excluded, registry, metadata, period_reviews = args
    try:
        return audit_match(match_id, excluded, registry, metadata, period_reviews)
    except Exception as error:  # an audit must report, not abort, on one bad match
        return {"match_id": match_id, "category": CATEGORY_CONFLICTING, "reason": f"audit error: {error!r}"}


CSV_FIELDS = [
    "match_id", "competition", "season", "date", "home_team", "away_team", "score", "periods",
    "usable_snapshots", "snapshots_with_visible_area", "category", "reason", "camera_status",
    "camera_reference_p1", "camera_reference_p2", "camera_reference_p3", "camera_reference_p4", "camera_reference_p5",
    "agreement_p1", "agreement_p2", "agreement_p3", "agreement_p4", "agreement_p5",
    "automatic_reference_team", "legacy_default_matches_camera", "legacy_periods_disagreeing",
    "registry_status", "registry_agrees_with_camera", "user_reviewed_periods", "display_plan_provenance",
    "independent_checks_pass",
    "min_keeper_consistency", "min_camera_bottom_rate", "goalkeepers",
]


def flat_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: row.get(key) for key in CSV_FIELDS}
    for period in range(1, 6):
        flat[f"camera_reference_p{period}"] = (row.get("camera_reference_by_period") or {}).get(period)
        evidence = (row.get("camera_period_evidence") or {}).get(period)
        flat[f"agreement_p{period}"] = evidence["agreement"] if evidence else None
    checks = row.get("post_transform_checks") or {}
    keeper = [c["keeper_consistency"] for p, c in checks.items() if p != 5 and c["keeper_consistency"] is not None]
    camera = [c["camera_bottom_rate"] for p, c in checks.items() if p != 5 and c["camera_bottom_rate"] is not None]
    flat["min_keeper_consistency"] = min(keeper) if keeper else None
    flat["min_camera_bottom_rate"] = min(camera) if camera else None
    for key in ("periods", "legacy_periods_disagreeing"):
        flat[key] = ",".join(map(str, row.get(key) or []))
    flat["user_reviewed_periods"] = "; ".join(
        f"P{p}: {r['evidence_type']}" for p, r in (row.get("period_reviews") or {}).items()
    )
    flat["goalkeepers"] = "; ".join(f"{t}: {', '.join(n)}" for t, n in (row.get("goalkeepers") or {}).items())
    return flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--match-id", type=int, nargs="*")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    excluded = load_exclusion_manifest()
    registry = load_calibration_registry()
    metadata_index = load_match_metadata_index()
    match_ids = args.match_id or available_360_match_ids()
    period_reviews = load_period_reviews()
    jobs = [(m, excluded, registry, metadata_index.get(m), period_reviews.get(m)) for m in match_ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = sorted(pool.map(_worker, jobs, chunksize=4), key=lambda r: r["match_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_match_audit.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    with (args.output_dir / "per_match_audit.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(flat_row(r) for r in rows)

    eligible = [r for r in rows if r["category"] != CATEGORY_EXCLUDED]
    with_p3 = [r for r in eligible if 3 in (r.get("camera_reference_by_period") or {}) and 2 in (r.get("camera_reference_by_period") or {})]
    summary = {
        "match_files": len(rows),
        "categories": dict(Counter(r["category"] for r in rows)),
        "camera_status": dict(Counter(r.get("camera_status") for r in eligible)),
        "legacy_default_matches_camera": dict(Counter(str(r.get("legacy_default_matches_camera")) for r in eligible)),
        "registry_agreement": {r["match_id"]: r.get("registry_agrees_with_camera") for r in rows if r.get("registry_status")},
        "extra_time_period_3_same_end_as_period_2": dict(
            Counter(r["camera_reference_by_period"][3] == r["camera_reference_by_period"][2] for r in with_p3)
        ),
        "end_swap_1_2_observed": dict(Counter(
            str(next((c["swapped"] for c in r.get("end_swap_checks", []) if c["periods"] == [1, 2]), None)) for r in eligible
        )),
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
