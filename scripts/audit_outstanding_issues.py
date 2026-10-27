"""Outstanding-issues audit of the StatsBomb 360 visualization (read-only).

Recomputes, with the CURRENT renderer code, a per-period status for every
eligible match: how the display orientation is determined (the renderer's
own resolve_display_plan, including registry calibrations and period
reviews), camera-side evidence, Problem A UNCERTAIN snapshots and their
display-only resolution, unusual camera footprints, internal geometric
inconsistencies, data-quality counts, clock sanity, and a render smoke test
(first/middle/last frame of each period through render_event_frame).

Never modifies raw data, the registry, the ML pipeline, or source code.

Outputs (results/physical_orientation/outstanding_issues/):
  all_period_status.csv      one row per (match, period)
  uncertain_frames.csv       every Problem A UNCERTAIN snapshot and its display outcome
  unusual_camera_frames.csv  every decided snapshot whose footprint reads camera-at-top after display
  inconsistent_frames.csv    snapshots whose actor sits far from its own event after display
  unresolved_matches.csv     one row per (match, period, issue)
  manual_review_priority.csv prioritized review tasks
  summary.json               all headline counts (reconciled with the CSVs)

Usage:
    .venv/bin/python scripts/audit_outstanding_issues.py [--workers 8] [--match-id N ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.animate_statsbomb_360_match import (  # noqa: E402
    DirectionInference,
    VisualizationError,
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    available_360_match_ids,
    build_timing_plan,
    compute_orientation_results,
    compute_shared_group_info,
    create_landscape_figure,
    determine_team_style,
    display_orientation_results,
    frame_review_corrections,
    format_match_clock,
    get_visual_theme,
    infer_attacking_directions,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    load_match_metadata_index,
    match_team_ids,
    render_event_frame,
    resolve_display_plan,
    timestamp_seconds,
)
from src.broadcast_orientation import (  # noqa: E402
    kickoff_own_half_orientation,
    CAMERA_SIDE_HIGH_Y,
    CAMERA_SIDE_LOW_Y,
    CAMERA_SIDE_UNKNOWN,
    MINIMUM_PERIOD_AGREEMENT,
    _point_in_polygon,
    _polygon_vertices,
    estimate_camera_side,
    expected_own_frame_camera_side,
)
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.orientation_correction import ORIENTATION_UNCERTAIN, mirror_point  # noqa: E402
from src.physical_orientation_calibration import load_calibration_registry, load_period_reviews  # noqa: E402
from src.team_identity import TEAM_IDENTITY_AMBIGUOUS, _snapshot_key, resolve_absolute_team_for_point  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "physical_orientation" / "outstanding_issues"
PERIOD_MINUTE_RANGE = {1: (0, 60), 2: (45, 110), 3: (90, 115), 4: (105, 130), 5: (120, 135)}
SPARSE_PERIOD_SNAPSHOTS = 300
INCONSISTENT_ACTOR_DISTANCE = 15.0


def _team_id(event):
    team = event.get("team") or {}
    return team.get("id")


def _lean(result) -> float:
    weight = sum(v.weight for v in result.votes)
    return sum(v.vote * v.weight for v in result.votes) / weight if weight else 0.0


def _actor_distances(frame, event):
    actor = next((p for p in frame.get("freeze_frame") or [] if p.get("actor") and p.get("location")), None)
    location = event.get("location")
    if actor is None or not location:
        return None, None
    d = math.dist(actor["location"], location)
    return d, math.dist(mirror_point(tuple(actor["location"])), location)


def _points(frame) -> list[tuple[float, float]]:
    return [tuple(p["location"]) for p in frame.get("freeze_frame") or [] if isinstance(p.get("location"), list)]


def _mean_nearest(a, b) -> float:
    return statistics.fmean(min(math.dist(p, q) for q in b) for p in a) if a and b else float("inf")


def sibling_check(index, display, raw, results, siblings_by_key) -> str:
    """Ground truth from a byte-identical raw snapshot owned by a CONFIDENT event.

    The same physical snapshot must be drawn identically for both events;
    if the uncertain one is drawn as the 180-degree image of its confident
    sibling, its flip is confirmed (and vice versa).
    """
    key = _snapshot_key(raw.usable_frames[index].frame_360)
    siblings = [j for j in siblings_by_key.get(key, []) if j != index and results[j].orientation != ORIENTATION_UNCERTAIN]
    if not siblings:
        return "no_confident_sibling"
    shown = _points(display.usable_frames[index].frame_360)
    sibling = _points(display.usable_frames[siblings[0]].frame_360)
    same = _mean_nearest(shown, sibling)
    mirrored = _mean_nearest([mirror_point(p) for p in shown], sibling)
    if same < 1.0 and mirrored > 5.0:
        return "confirmed_correct"
    if mirrored < 1.0 and same > 5.0:
        return "confirmed_flipped"
    return "sibling_inconclusive"


def duplicate_conflict(index, raw, results, plan, siblings_by_key) -> bool:
    """True when confident byte-identical copies imply different corrections."""
    event = raw.usable_frames[index].event
    implied = set()
    for j in siblings_by_key.get(_snapshot_key(raw.usable_frames[index].frame_360), []):
        if j == index or results[j].orientation == ORIENTATION_UNCERTAIN:
            continue
        other = raw.usable_frames[j].event
        if other.get("period") != event.get("period"):
            continue
        implied.add((results[j].orientation == "ROTATED_180")
                    ^ plan.rotates(other.get("period"), _team_id(other)) ^ plan.rotates(event.get("period"), _team_id(event)))
    return len(implied) > 1


def continuity_check(index, display, results, by_time) -> str:
    """Compare the drawn players with the nearest confidently-classified,
    different snapshot within 3 s of the same period: players move a few
    yards, so a correct frame is near it and a flipped one near its mirror."""
    period, t = by_time[index]
    candidates = [
        j for j in range(max(0, index - 20), min(len(by_time), index + 21))
        if j != index and by_time[j][0] == period and abs(by_time[j][1] - t) <= 3.0
        and results[j].orientation != ORIENTATION_UNCERTAIN
        and display.usable_frames[j].frame_360.get("freeze_frame") != display.usable_frames[index].frame_360.get("freeze_frame")
    ]
    if not candidates:
        return "no_neighbour"
    j = min(candidates, key=lambda k: abs(by_time[k][1] - t))
    shown = _points(display.usable_frames[index].frame_360)
    neighbour = _points(display.usable_frames[j].frame_360)
    same = _mean_nearest(shown, neighbour)
    mirrored = _mean_nearest([mirror_point(p) for p in shown], neighbour)
    if same < 0.5 * mirrored:
        return "as_drawn"
    if mirrored < 0.5 * same:
        return "mirrored"
    return "unclear"


def duplicate_display_check(display, results, siblings_by_key) -> list[dict]:
    """Every group of byte-identical raw snapshots (confident members too):
    each member must be DRAWN like the group's first confident member. A
    member drawn as its mirror image is a visible 180-degree jump."""
    rows = []
    for members in siblings_by_key.values():
        if len(members) < 2:
            continue
        confident = [m for m in members if results[m].orientation != ORIENTATION_UNCERTAIN]
        reference = confident[0] if confident else members[0]
        ref_points = _points(display.usable_frames[reference].frame_360)
        for member in members:
            if member == reference:
                continue
            shown = _points(display.usable_frames[member].frame_360)
            same = _mean_nearest(shown, ref_points)
            mirrored = _mean_nearest([mirror_point(p) for p in shown], ref_points)
            rows.append({
                "event_id": display.usable_frames[member].event["id"],
                "reference_confident": bool(confident),
                "member_confident": results[member].orientation != ORIENTATION_UNCERTAIN,
                "state": "identical" if same < 1.0 else "mirrored" if mirrored < 1.0 else "different",
                "timestamp": display.usable_frames[member].event.get("timestamp"),
                "period": display.usable_frames[member].event.get("period"),
            })
    return rows


def audit_match(match_id: int, metadata, registry, reviews) -> dict[str, Any]:
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, anchor_source, messages = resolve_display_plan(match_id, raw, inference)
    display_results, display_counts = display_orientation_results(
        raw, results, inference, plan, frame_reviews=frame_review_corrections())
    corrected = apply_orientation_correction_to_match(raw, display_results)
    display = apply_display_plan_to_match(corrected, plan)
    shared = compute_shared_group_info(raw)
    raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
    team_ids = match_team_ids(raw)
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    name_pair = tuple(names[t] for t in team_ids)
    style = determine_team_style(display.usable_frames, "#4aa8ff", "#ff5a5f")
    match_reviews = reviews.get(match_id, {})
    calibration = registry.get(match_id)
    verified = calibration is not None and calibration.verification_status == "verified"

    siblings_by_key: dict[tuple, list[int]] = {}
    for position, joined in enumerate(raw.usable_frames):
        siblings_by_key.setdefault(_snapshot_key(joined.frame_360), []).append(position)
    by_time = [(j.event.get("period"), timestamp_seconds(j.event.get("timestamp"))) for j in raw.usable_frames]
    periods: dict[int, dict[str, Any]] = {}
    uncertain_rows, unusual_rows, inconsistent_rows, override_rows = [], [], [], []
    by_period_frames: dict[int, list] = {}
    for index, (shown, raw_joined, base, used) in enumerate(zip(display.usable_frames, raw.usable_frames, results, display_results)):
        event = shown.event
        period = event.get("period")
        by_period_frames.setdefault(period, []).append(shown)
        s = periods.setdefault(period, Counter())
        s["snapshots"] += 1
        clock = format_match_clock(event)
        frame = shown.frame_360
        polygon = _polygon_vertices(frame.get("visible_area"))
        if not frame.get("visible_area"):
            s["missing_visible_area"] += 1
        elif polygon is None:
            s["malformed_visible_area"] += 1
        players = [p for p in frame.get("freeze_frame") or [] if isinstance(p.get("location"), list)]
        if len(players) < 3:
            s["sparse_freeze_frames"] += 1
        s["player_points"] += len(players)
        s["player_points_out_of_bounds"] += sum(
            not (-1 <= p["location"][0] <= 121 and -1 <= p["location"][1] <= 81) for p in players
        )
        # Points beyond the renderer's fixed viewport (FIXED_X/Y_LIMITS) are clipped, i.e. not drawn.
        s["player_points_outside_viewport"] += sum(
            not (-3 <= p["location"][0] <= 123 and -3 <= p["location"][1] <= 83) for p in players
        )
        location = event.get("location")
        if location and not (0 <= location[0] <= 120 and 0 <= location[1] <= 80):
            s["ball_out_of_bounds"] += 1
        minute = event.get("minute")
        low, high = PERIOD_MINUTE_RANGE.get(period, (0, 200))
        if not isinstance(minute, int) or not low <= minute <= high:
            s["clock_out_of_period_range"] += 1
        if "+" in clock:
            s["stoppage_time_snapshots"] += 1

        # Team / keeper identity (Problem C).
        keepers_by_team = Counter()
        for player in players:
            team = resolve_absolute_team_for_point(player, raw_joined.event, shared.get(raw_joined.event["id"]), raw_by_id, name_pair)
            if team == TEAM_IDENTITY_AMBIGUOUS:
                s["ambiguous_team_points"] += 1
            if player.get("keeper"):
                if team == TEAM_IDENTITY_AMBIGUOUS:
                    s["keeper_identity_ambiguous"] += 1
                    continue
                keepers_by_team[team] += 1
                transform = plan.transforms.get(period)
                if transform is not None and transform.reference_team_id is not None:
                    s["keeper_checked"] += 1
                    attacks_right = team == names[transform.reference_team_id]
                    s["keeper_at_defended_end"] += (player["location"][0] < 60) == attacks_right
        if any(count > 1 for count in keepers_by_team.values()):
            s["frames_two_keepers_same_team"] += 1

        side = estimate_camera_side(frame.get("visible_area")).side
        if base.orientation == ORIENTATION_UNCERTAIN:
            s["uncertain_problem_a"] += 1
            outcome = (
                "resolved_rotated" if used.orientation == "ROTATED_180"
                else "resolved_normal" if used.orientation == "NORMAL" else "left_uncorrected"
            )
            s[outcome] += 1
            if outcome != "left_uncorrected":
                s[f"route:{used.resolved_by}"] += 1
            raw_side = estimate_camera_side(raw_joined.frame_360.get("visible_area")).side
            expected = expected_own_frame_camera_side(inference, period, _team_id(raw_joined.event), plan)
            lean = _lean(base)
            d, dm = _actor_distances(raw_joined.frame_360, raw_joined.event)
            is_kickoff = ((raw_joined.event.get("pass") or {}).get("type") or {}).get("name") == "Kick Off"
            kickoff_verdict = kickoff_own_half_orientation(raw_joined.frame_360) if is_kickoff else None
            continuity = "not_checked"
            if outcome == "left_uncorrected":
                continuity = continuity_check(index, display, results, by_time)
                if duplicate_conflict(index, raw, results, plan, siblings_by_key):
                    reason = "confident duplicates disagree"
                elif raw_side == CAMERA_SIDE_UNKNOWN:
                    reason = "camera side unreadable"
                elif expected == CAMERA_SIDE_UNKNOWN:
                    reason = "period camera side undecided"
                elif lean == 0:
                    reason = "Problem A has no lean"
                else:
                    reason = "signals disagree"
                s[f"left_uncorrected:{reason}"] += 1
            else:
                reason = {
                    "display_duplicate_snapshot": "same-moment confident duplicate",
                    "display_kickoff_own_half": "kick-off own-half rule (camera not contradicting)",
                    "user_frame_review": "user review of the comparison sheet (renders, not video)",
                }.get(used.resolved_by, "lean and camera agree")
            d_shown, dm_shown = _actor_distances(frame, event)
            uncertain_rows.append({
                "match_id": match_id, "period": period, "timestamp": event.get("timestamp"), "match_clock": clock,
                "event_id": event["id"], "event_type": (event.get("type") or {}).get("name"), "team": names.get(_team_id(event)),
                "problem_a_lean": round(lean, 3), "problem_a_lean_direction": "rotated" if lean < 0 else "normal" if lean > 0 else "none",
                "raw_camera_side": raw_side, "expected_own_frame_side": expected,
                "raw_actor_ball_distance": None if d is None else round(d, 1),
                "raw_mirrored_actor_ball_distance": None if dm is None else round(dm, 1),
                "display_outcome": outcome, "reason": reason,
                "display_route": used.resolved_by if outcome != "left_uncorrected" else "",
                "is_kickoff": is_kickoff, "kickoff_own_half": kickoff_verdict or "",
                "continuity_with_confident_neighbour": continuity,
                "sibling_check": sibling_check(index, display, raw, results, siblings_by_key),
                "display_camera_side": (
                    "unknown" if raw_side == CAMERA_SIDE_UNKNOWN or expected == CAMERA_SIDE_UNKNOWN
                    else "top" if (raw_side != expected) != (outcome == "resolved_rotated") else "bottom"
                ),
                "shown_actor_ball_distance": None if d_shown is None else round(d_shown, 1),
                "likely_display_state": (
                    "consistent" if d_shown is not None and d_shown <= INCONSISTENT_ACTOR_DISTANCE
                    else "likely_rotated_frame" if d_shown is not None and dm_shown is not None and dm_shown < d_shown
                    else "unknown_no_actor" if d_shown is None else "inconsistent_unclear"
                ),
            })
        else:
            if side == CAMERA_SIDE_LOW_Y:
                s["unusual_camera_top"] += 1
                unusual_rows.append({
                    "match_id": match_id, "period": period, "timestamp": event.get("timestamp"), "match_clock": clock,
                    "event_id": event["id"], "event_type": (event.get("type") or {}).get("name"),
                    "team": names.get(_team_id(event)), "problem_a": base.orientation,
                    "apex": estimate_camera_side(frame.get("visible_area")).apex,
                    "ball_in_polygon": bool(polygon and location and _point_in_polygon(tuple(location), polygon)),
                })
            elif side == CAMERA_SIDE_HIGH_Y:
                s["camera_bottom"] += 1
            else:
                s["camera_unknown"] += 1
        if base.orientation != ORIENTATION_UNCERTAIN and used.resolved_by == "display_duplicate_consensus":
            d_raw, dm_raw = _actor_distances(raw_joined.frame_360, raw_joined.event)
            d_now, dm_now = _actor_distances(frame, event)
            override_rows.append({
                "match_id": match_id, "period": period, "timestamp": event.get("timestamp"), "match_clock": clock,
                "event_id": event["id"], "event_type": (event.get("type") or {}).get("name"),
                "team": names.get(_team_id(event)), "problem_a": base.orientation,
                "problem_a_confidence": round(base.confidence, 3), "display_orientation": used.orientation,
                "display_camera_side_now": side,
                "actor_ball_distance_now": None if d_now is None else round(d_now, 1),
                "actor_ball_distance_problem_a": None if d_raw is None else round(min(d_raw, dm_raw) if base.orientation == "ROTATED_180" else d_raw, 1),
            })
        d_shown, dm_shown = _actor_distances(frame, event)
        if d_shown is not None and d_shown > INCONSISTENT_ACTOR_DISTANCE:
            s["actor_far_from_event"] += 1
            if dm_shown < d_shown:
                s["actor_far_mirror_closer"] += 1
            inconsistent_rows.append({
                "match_id": match_id, "period": period, "timestamp": event.get("timestamp"), "match_clock": clock,
                "event_id": event["id"], "event_type": (event.get("type") or {}).get("name"),
                "problem_a": base.orientation, "display_orientation": used.orientation,
                "actor_ball_distance": round(d_shown, 1), "mirrored_distance": round(dm_shown, 1),
            })
        if polygon and location:
            s["ball_polygon_checked"] += 1
            s["ball_in_polygon"] += _point_in_polygon(tuple(location), polygon)

    # Render smoke test + shot cross-check per period.
    theme = get_visual_theme("dark")
    period_rows = []
    for period, frames in sorted(by_period_frames.items()):
        s = periods[period]
        transform = plan.transforms[period]
        evidence = inference.period_evidence(period)
        shot_check = infer_attacking_directions(display.usable_frames, period, style)
        plan_dirs = plan.attacking_directions(period)
        shot_agrees = None
        if shot_check.directions and plan_dirs:
            shot_agrees = shot_check.directions == {names[t]: d for t, d in plan_dirs.items()}
        render_errors = []
        for sample in {0, len(frames) // 2, len(frames) - 1}:
            figure = None
            try:
                figure, ax = create_landscape_figure(theme)
                timing = build_timing_plan([frames[sample]], "fixed", 250, 1.0, 80, 1500)[0]
                directions = {names[t]: d for t, d in (plan_dirs or {}).items()}
                render_event_frame(ax, match_id, frames[sample], 1, 1, timing, "fixed", "camera-overlay", style,
                                   DirectionInference(directions, ""), theme, False, True, shared, raw_by_id)
                figure.canvas.draw()
            except Exception as error:  # the audit records, never hides, a rendering failure
                render_errors.append(f"{frames[sample].event.get('timestamp')}: {error!r}")
            finally:
                if figure is not None:
                    plt.close(figure)
        review = match_reviews.get(period)
        if verified:
            status = "independently_verified"
        elif review is not None and review.independently_verified:
            status = "independently_verified_period"
        elif review is not None:
            status = "user_reviewed"
        elif transform.source == "camera":
            status = "high_confidence_inferred"
        elif transform.source == "derived_from_user_reviewed_period_and_halftime_rule":
            status = "derived_from_user_reviewed_period"
        elif transform.source in ("laws_rule", "camera_relative", "anchor+camera_end_relation"):
            status = "rule_or_relation_derived"
        else:
            status = "unresolved_fallback"
        conflicts = []
        if evidence and evidence.decided and transform.reference_team_id is not None and evidence.reference_team_id is not None \
                and evidence.reference_team_id != transform.reference_team_id:
            conflicts.append("display transform contradicts camera evidence")
        if shot_agrees is False:
            conflicts.append("shot-origin direction disagrees with transform")
        keeper_rate = s["keeper_at_defended_end"] / s["keeper_checked"] if s["keeper_checked"] else None
        if keeper_rate is not None and keeper_rate < 0.9 and period != 5:
            conflicts.append(f"keeper consistency {keeper_rate:.2f}")
        for message in messages:
            conflicts.append(message)
        # Does this period's OWN camera majority (decided or not) point to the
        # same unrotated team as the display plan? Independent of the plan.
        camera_majority_team = ""
        if evidence is not None and evidence.frame_model == "OPPOSITE":
            other = {team_ids[0]: team_ids[1], team_ids[1]: team_ids[0]}
            score = {t: evidence.votes_by_team[t].get(CAMERA_SIDE_HIGH_Y, 0) + evidence.votes_by_team[other[t]].get(CAMERA_SIDE_LOW_Y, 0)
                     for t in team_ids}
            if score[team_ids[0]] != score[team_ids[1]]:
                camera_majority_team = names[max(team_ids, key=lambda t: score[t])]
        plan_team = names.get(transform.reference_team_id) if transform.reference_team_id else ""
        derivation_supported = (
            transform.source == "derived_from_user_reviewed_period_and_halftime_rule"
            and camera_majority_team == plan_team
            and (keeper_rate is None or keeper_rate >= 0.95)
            and shot_agrees is not False
        )
        period_rows.append({
            "match_id": match_id,
            "competition": metadata.competition if metadata else "", "season": metadata.season if metadata else "",
            "home_team": metadata.home_team if metadata else "", "away_team": metadata.away_team if metadata else "",
            "period": period,
            "first_clock": format_match_clock(frames[0].event), "last_clock": format_match_clock(frames[-1].event),
            "frame_model": transform.frame_model,
            "orientation_method": transform.source,
            "display_reference_team": names.get(transform.reference_team_id) if transform.reference_team_id else "",
            "shootout_rotate_both": transform.rotate_both if transform.frame_model == "SHARED" else "",
            "plan_provenance": plan.provenance,
            "status": status,
            "rule_derived": transform.source == "laws_rule",
            "camera_decided": bool(evidence and evidence.decided),
            "camera_support": evidence.support if evidence else 0,
            "camera_opposition": evidence.opposition if evidence else 0,
            "camera_agreement": round(evidence.agreement, 4) if evidence else None,
            "below_threshold": bool(evidence and evidence.agreement < MINIMUM_PERIOD_AGREEMENT),
            "user_review": f"{review.decision or 'confirmed'} ({review.evidence_type})" if review else "",
            "camera_majority_unrotated_team": camera_majority_team,
            "derivation_supported_by_independent_evidence": derivation_supported if transform.source.startswith("derived") else "",
            "registry_verified": verified,
            "shot_cross_check": {None: "unavailable", True: "agrees", False: "disagrees"}[shot_agrees],
            "keeper_consistency": None if keeper_rate is None else round(keeper_rate, 4),
            "conflicts": " | ".join(conflicts),
            "render_errors": " | ".join(render_errors),
            **{key: s.get(key, 0) for key in (
                "snapshots", "uncertain_problem_a", "resolved_rotated", "resolved_normal", "left_uncorrected",
                "route:display_duplicate_snapshot", "route:display_kickoff_own_half", "route:display_camera_side",
                "unusual_camera_top", "camera_unknown", "actor_far_from_event", "actor_far_mirror_closer",
                "missing_visible_area", "malformed_visible_area", "sparse_freeze_frames",
                "player_points", "player_points_out_of_bounds", "player_points_outside_viewport",
                "ball_out_of_bounds", "ambiguous_team_points", "keeper_identity_ambiguous", "frames_two_keepers_same_team", "clock_out_of_period_range",
                "stoppage_time_snapshots", "ball_in_polygon", "ball_polygon_checked",
            )},
            "left_uncorrected_reasons": "; ".join(f"{k.split(':', 1)[1]}={v}" for k, v in s.items() if k.startswith("left_uncorrected:")),
        })
    duplicates = [{"match_id": match_id, **row} for row in duplicate_display_check(display, results, siblings_by_key)]
    return {"periods": period_rows, "uncertain": uncertain_rows, "unusual": unusual_rows,
            "inconsistent": inconsistent_rows, "duplicates": duplicates, "display_counts": display_counts,
            "overrides": override_rows}


def frame_signals(row: dict) -> tuple[list[str], list[str]]:
    """Independent per-frame signals for an uncorrected snapshot as DRAWN:
    (signals saying 'flipped', signals saying 'correct')."""
    flipped, correct = [], []
    d, dm = row["shown_actor_ball_distance"], None
    if row["likely_display_state"] == "likely_rotated_frame":
        flipped.append("actor nearer mirrored event location")
    elif row["likely_display_state"] == "consistent" and row["raw_actor_ball_distance"] != row["raw_mirrored_actor_ball_distance"]:
        correct.append("actor at event location")  # (not for centre-spot kick-offs: rotation-invariant)
    if row["display_camera_side"] == "top":
        flipped.append("camera at top")
    elif row["display_camera_side"] == "bottom":
        correct.append("camera at bottom")
    if row["kickoff_own_half"] == "ROTATED_180":
        flipped.append("kick-off players in wrong halves")
    elif row["kickoff_own_half"] == "NORMAL":
        correct.append("kick-off players in own halves")
    if row["continuity_with_confident_neighbour"] == "mirrored":
        flipped.append("mirror of neighbouring frame")
    elif row["continuity_with_confident_neighbour"] == "as_drawn":
        correct.append("continuous with neighbouring frame")
    return flipped, correct


def visual_class(row: dict) -> str:
    """How a Problem A UNCERTAIN snapshot is DRAWN.

    confirmed_flipped / confirmed_correct: compared with a confident
      byte-identical copy at the same instant (ground truth).
    resolved_*: resolved by the display rules (see display_route).
    For other uncorrected snapshots, independent signals are counted
    (frame_signals): likely_flipped (more say flipped), likely_correct
    (more say correct), conflicting (tie), undetermined (no signal).
    """
    if row["sibling_check"] == "confirmed_flipped":
        return "confirmed_flipped"
    if row["display_outcome"] != "left_uncorrected":
        return row["display_outcome"]
    if row["sibling_check"] == "confirmed_correct":
        return "confirmed_correct"
    flipped, correct = frame_signals(row)
    if len(flipped) > len(correct):
        return "likely_flipped"
    if len(correct) > len(flipped):
        return "likely_correct"
    return "conflicting" if flipped else "undetermined"


def defect_types(uncertain: list[dict]) -> dict[str, int]:
    """Confirmed/likely software defects, by mechanism -> affected frames."""
    flipped = [u for u in uncertain if u["visual_class"] in ("likely_flipped", "confirmed_flipped")]
    gap = [u for u in flipped if u["reason"] == "period camera side undecided"]
    other = [u for u in flipped if u["reason"] != "period camera side undecided"]
    types = {
        "display_rule_skipped_in_rule_derived_period": {
            "confirmed": sum(u["visual_class"] == "confirmed_flipped" for u in gap),
            "likely": sum(u["visual_class"] == "likely_flipped" for u in gap)},
        "uncorrected_uncertain_snapshot_drawn_flipped": {
            "confirmed": sum(u["visual_class"] == "confirmed_flipped" for u in other),
            "likely": sum(u["visual_class"] == "likely_flipped" for u in other)},
    }
    return {name: counts for name, counts in types.items() if counts["confirmed"] or counts["likely"]}


def _teams(row: dict) -> str:
    return f"{row['home_team']} vs {row['away_team']}"


BASELINE_DIR = OUTPUT_DIR / "baseline_before_fix"


def compare_with_baseline(uncertain: list[dict]) -> dict[str, Any]:
    """Before/after for every Problem A UNCERTAIN snapshot (joined by event id)."""
    path = BASELINE_DIR / "uncertain_frames.csv"
    if not path.exists():
        return {"baseline": "missing"}
    with path.open(encoding="utf-8") as file:
        before = {r["event_id"]: r for r in csv.DictReader(file)}
    now = {u["event_id"]: u for u in uncertain}
    was_flipped = {e for e, r in before.items() if r["visual_class"] == "confirmed_flipped"}
    was_likely = {e for e, r in before.items() if r["visual_class"] == "likely_flipped"}
    was_correct = {e for e, r in before.items() if r["sibling_check"] == "confirmed_correct" and r["visual_class"] != "confirmed_flipped"}
    rotated_before = {e for e, r in before.items() if r["display_outcome"] == "resolved_rotated"}
    rotated_now = {e for e, u in now.items() if u["display_outcome"] == "resolved_rotated"}
    changed = (rotated_before ^ rotated_now) & set(now)
    unexplained = sorted(changed - was_flipped - was_likely)
    return {
        "frames_compared": len(set(before) & set(now)),
        "confirmed_flipped_before": len(was_flipped),
        "confirmed_flipped_after": sum(now[e]["visual_class"] == "confirmed_flipped" for e in was_flipped if e in now)
        + sum(u["visual_class"] == "confirmed_flipped" for e, u in now.items() if e not in was_flipped),
        "confirmed_flipped_fixed": sum(now[e]["sibling_check"] == "confirmed_correct" for e in was_flipped if e in now),
        "likely_flipped_before": len(was_likely),
        "likely_flipped_now_rotated": sum(e in rotated_now for e in was_likely),
        "previously_confirmed_correct": len(was_correct),
        "previously_confirmed_correct_still_correct": sum(now[e]["sibling_check"] == "confirmed_correct" for e in was_correct if e in now),
        "display_changed_frames": len(changed),
        "display_changed_outside_previous_flips": len(unexplained),
        "display_changed_outside_previous_flips_event_ids": unexplained[:50],
        "fix_routes": dict(Counter(now[e]["display_route"] for e in changed)),
    }


def build_reports(periods: list[dict], uncertain: list[dict], unusual: list[dict], matches: int,
                  duplicates: list[dict] | None = None, display_counts: list[dict] | None = None) -> None:
    by_key = {(int(r["match_id"]), int(r["period"])): r for r in periods}
    issues: list[dict] = []

    def issue(row, kind, detail, why, likely, evidence, manual, priority, when=""):
        issues.append({
            "match_id": row["match_id"], "teams": _teams(row), "period": row["period"], "issue": kind,
            "time": when or f"{row['first_clock']} - {row['last_clock']}", "what_is_uncertain": detail,
            "why": why, "current_visualization_likely_correct": likely, "evidence_needed": evidence,
            "manual_inspection_required": manual, "priority": priority,
        })

    for r in periods:
        if r["orientation_method"] == "derived_from_user_reviewed_period_and_halftime_rule":
            supported = str(r["derivation_supported_by_independent_evidence"]) == "True"
            issue(r, "derived_from_user_reviewed_period",
                  f"derived from the user-reviewed neighbouring period by the halftime rule; own camera majority "
                  f"{r['camera_support']} vs {r['camera_opposition']} points to {r['camera_majority_unrotated_team'] or 'nobody'}",
                  "own camera evidence below the 0.90 decision threshold",
                  "yes" if supported else "unknown",
                  "none (derivation corroborated by camera majority, keepers, shots)" if supported else "footage",
                  "no" if supported else "yes", "P3" if supported else "P1")
            continue
        if r["below_threshold"] or not r["camera_decided"]:
            issue(r, "ambiguous_camera_evidence",
                  f"camera agreement {r['camera_agreement']} < {MINIMUM_PERIOD_AGREEMENT} ({r['camera_support']} vs {r['camera_opposition']})",
                  "split visible_area camera-side evidence (alternate camera angles)",
                  "probably (majority and halftime rule agree)", "broadcast footage of this period", "yes", "P1")
        if r["orientation_method"] == "laws_rule":
            issue(r, "rule_derived_orientation", "orientation derived from the other half by the halftime rule",
                  "this period's own camera evidence is undecided",
                  "probably", "broadcast footage of this period", "yes", "P1")
        if r["orientation_method"] == "fallback":
            issue(r, "fallback_orientation", "legacy fallback used", "no camera evidence", "unknown", "footage", "yes", "P1")
        if r["conflicts"]:
            issue(r, "conflicting_signals", r["conflicts"], "independent checks disagree", "unknown", "footage", "yes", "P1")
        if r["render_errors"]:
            issue(r, "render_error", r["render_errors"], "exception while rendering", "no", "fix code", "no", "P0")
        if int(r["frames_two_keepers_same_team"]):
            issue(r, "two_keepers_same_team", f"{r['frames_two_keepers_same_team']} frame(s)",
                  "StatsBomb keeper flag on two points of one resolved team", "data as recorded", "none", "no", "P3")
        if int(r["ambiguous_team_points"]):
            issue(r, "ambiguous_team_identity", f"{r['ambiguous_team_points']} point(s) drawn in the neutral colour",
                  "shared cross-team snapshot without a confident source event (Problem C)", "yes (shown as unclear)", "none", "no", "P3")
        if int(r["clock_out_of_period_range"]):
            issue(r, "clock_outside_expected_range", f"{r['clock_out_of_period_range']} snapshot(s); clock {r['last_clock']}",
                  "long penalty shootout; StatsBomb minute keeps counting", "yes (data as recorded)", "none", "no", "P3")
        decided_frames = int(r["snapshots"]) - int(r["uncertain_problem_a"])
        if decided_frames and int(r["unusual_camera_top"]) / decided_frames >= 0.05:
            issue(r, "frequent_unusual_camera_polygons",
                  f"{r['unusual_camera_top']} of {decided_frames} decided footprints read camera-at-top",
                  "alternate (behind-goal / reverse-angle) camera footprints; players and ball are Problem-A-consistent",
                  "yes", "none required", "no", "P3")

    flipped_by_period: dict[tuple, list] = {}
    undetermined_by_period: dict[tuple, list] = {}
    for u in uncertain:
        key = (int(u["match_id"]), int(u["period"]))
        if u["visual_class"] in ("likely_flipped", "confirmed_flipped"):
            flipped_by_period.setdefault(key, []).append(u)
        elif u["visual_class"] in ("undetermined", "conflicting"):
            undetermined_by_period.setdefault(key, []).append(u)
    for key, rows in sorted(flipped_by_period.items()):
        r = by_key[key]
        gap = any(u["reason"] == "period camera side undecided" for u in rows)
        confirmed = sum(u["visual_class"] == "confirmed_flipped" for u in rows)
        issue(r, "flipped_single_frames",
              f"{len(rows)} snapshot(s) drawn rotated by 180 degrees for one frame ({confirmed} confirmed by a same-snapshot confident sibling, {len(rows) - confirmed} likely)",
              "display rule does not run in rule-derived periods (code gap)" if gap else
              "Problem A UNCERTAIN and the camera/lean signals disagree or the camera side is unreadable",
              "no (those frames)",
              "none (confirmed by sibling)" if confirmed == len(rows) else "spot check of the unconfirmed frame(s)",
              "no" if confirmed == len(rows) else "yes (unconfirmed frames)", "P0" if confirmed else "P1",
              when="; ".join(u["match_clock"] for u in rows))
    for key, rows in sorted(undetermined_by_period.items()):
        r = by_key[key]
        issue(r, "undetermined_uncertain_frames",
              f"{len(rows)} UNCERTAIN snapshot(s) left as stored: " + "; ".join(
                  f"{u['match_clock']} {u['event_type']} [flipped: {u['signals_flipped'] or '-'} | correct: {u['signals_correct'] or '-'}]"
                  for u in rows),
              "independent signals conflict or are absent; no duplicate copy exists", "unknown (single frames)",
              "none practical (one frame each)", "no", "P3", when="; ".join(u["match_clock"] for u in rows))
    write_csv(OUTPUT_DIR / "unresolved_matches.csv", issues)

    tasks = [{
        "rank": 1, "priority": "P1" if any(i["priority"] == "P1" and i["match_id"] == 3901833 and int(i["period"]) == 1 for i in issues) else "resolved", "match_id": 3901833, "teams": "France Women's vs Morocco Women's", "period": 1,
        "match_clock": by_key[(3901833, 1)]["first_clock"] + " - " + by_key[(3901833, 1)]["last_clock"] if (3901833, 1) in by_key else "",
        "what_to_check": "which team attacks left->right on a broadcast frame of the 1st half (use the scoreboard clock)",
        "how": "open results/physical_orientation/final_orientation_audit/review/review.html (match 3901833, P1 proposed/alternative), or render with the standard flags and --period 1",
        "alternatives": "PROPOSED: France Women's attacks left->right (defends left goal) | ALTERNATIVE: Morocco Women's attacks left->right",
        "record": "review page: decision proposed/alternative, tick P1 only, evidence 'my review' or 'independent video' (+URL, clock); then scripts/import_orientation_reviews.py <export> --write",
    }]
    rank = 2
    flipped = sorted((u for u in uncertain if u["visual_class"] == "likely_flipped"), key=lambda u: (u["match_id"], u["period"], u["timestamp"]))
    sample = [u for i, u in enumerate(flipped) if i % max(1, len(flipped) // 10) == 0][:10]
    for u in sample:
        seconds = timestamp_seconds(u["timestamp"])
        tasks.append({
            "rank": rank, "priority": "P1", "match_id": u["match_id"], "teams": _teams(by_key[(int(u["match_id"]), int(u["period"]))]),
            "period": u["period"], "match_clock": u["match_clock"],
            "what_to_check": f"is this single {u['event_type']} frame drawn 180-degree rotated relative to its neighbours? ({u['reason']})",
            "how": (f".venv/bin/python src/animate_statsbomb_360_match.py --match-id {u['match_id']} --period {u['period']} "
                    f"--start-second {max(0, seconds - 4):.1f} --end-second {seconds + 4:.1f} --apply-orientation-correction "
                    "--apply-team-normalization --apply-team-identity-resolution --view camera-overlay --timing-mode real-time "
                    f"--output results/physical_orientation/outstanding_issues/spot_check_{u['match_id']}_{u['period']}_{int(seconds)}.gif"),
            "alternatives": "frame jumps (players/keepers on the wrong ends for one frame) | frame continuous with neighbours",
            "record": "note yes/no per frame; if most are flips, approve extending the display rule (code change)",
        })
        rank += 1
    for match_id, period, label in ((3869220, 3, "Morocco vs Spain"), (4018357, 3, "France Women's vs Germany Women's")):
        r = by_key.get((match_id, period))
        if r:
            tasks.append({
                "rank": rank, "priority": "P2", "match_id": match_id, "teams": label, "period": period,
                "match_clock": f"{r['first_clock']} - {r['last_clock']}",
                "what_to_check": "extra time 1st half continues at the 2nd half's ends (unusual)",
                "how": "review.html (queued: extra time) or render --period 3",
                "alternatives": f"PROPOSED: {r['display_reference_team']} attacks left->right | ALTERNATIVE: the other team",
                "record": "review page, tick the extra-time period, import with scripts/import_orientation_reviews.py",
            })
            rank += 1
    tasks.append({
        "rank": rank, "priority": "P2", "match_id": "", "teams": "26 other queued validation matches", "period": "",
        "match_clock": "", "what_to_check": "stratified/lowest-agreement/shootout validation sample (not ambiguous)",
        "how": "results/physical_orientation/final_orientation_audit/review/review.html",
        "alternatives": "proposed | alternative | cannot tell", "record": "review page export -> import script",
    })
    tasks = [t for t in tasks if t["priority"] != "resolved"]
    for position, task in enumerate(tasks, start=1):
        task["rank"] = position
    write_csv(OUTPUT_DIR / "manual_review_priority.csv", tasks)

    status = Counter(r["status"] for r in periods)
    classes = Counter(u["visual_class"] for u in uncertain)
    summary = {
        "generated_with": "scripts/audit_outstanding_issues.py (current code)",
        "eligible_matches": matches,
        "periods_examined": len(periods),
        "regular_periods": sum(r["frame_model"] == "OPPOSITE" for r in periods),
        "shootout_periods": sum(r["frame_model"] == "SHARED" for r in periods),
        "independently_verified_matches": len({r["match_id"] for r in periods if r["status"] == "independently_verified"}),
        "period_status": dict(status),
        "user_reviewed_periods": status.get("user_reviewed", 0),
        "high_confidence_inferred_periods": status.get("high_confidence_inferred", 0),
        "ambiguous_periods": sum(bool(r["below_threshold"]) or not r["camera_decided"] for r in periods),
        "rule_derived_periods": sum(r["orientation_method"] == "laws_rule" for r in periods),
        "fallback_periods": sum(r["orientation_method"] == "fallback" for r in periods),
        "contradicted_by_independent_evidence": sum(bool(r["conflicts"]) for r in periods),
        "snapshots": sum(int(r["snapshots"]) for r in periods),
        "uncertain_snapshots_problem_a": len(uncertain),
        "uncertain_outcomes": dict(Counter(u["display_outcome"] for u in uncertain)),
        "uncertain_signal_disagreements": sum(u["reason"] == "signals disagree" for u in uncertain),
        "uncertain_visual_class": dict(classes),
        "unresolved_event_level_cases": sum(u["display_outcome"] == "left_uncorrected" for u in uncertain),
        "likely_flipped_frames": classes.get("likely_flipped", 0),
        "confirmed_flipped_frames": classes.get("confirmed_flipped", 0),
        "flipped_frames_total": classes.get("likely_flipped", 0) + classes.get("confirmed_flipped", 0),
        "flipped_frame_matches": len({u["match_id"] for u in uncertain if u["visual_class"] in ("likely_flipped", "confirmed_flipped")}),
        "sibling_check": dict(Counter(u["sibling_check"] for u in uncertain)),
        "sibling_check_by_outcome": {
            f"{o}|{c}": n for (o, c), n in Counter((u["display_outcome"], u["sibling_check"]) for u in uncertain).items()
        },
        "unusual_camera_frames": len(unusual),
        "unusual_camera_ball_inside_polygon": sum(str(u["ball_in_polygon"]) == "True" for u in unusual),
        "render_errors": sum(bool(r["render_errors"]) for r in periods),
        "shot_cross_check": dict(Counter(r["shot_cross_check"] for r in periods)),
        "known_visualization_defects": len(defect_types(uncertain)),
        "known_visualization_defect_types": defect_types(uncertain),
        "issues_by_priority": dict(Counter(i["priority"] for i in issues)),
        "issues_by_type": dict(Counter(i["issue"] for i in issues)),
        "manual_review_tasks": dict(Counter(t["priority"] for t in tasks)),
        "data_quality": {k: sum(int(r[k]) for r in periods) for k in (
            "missing_visible_area", "malformed_visible_area", "sparse_freeze_frames", "player_points",
            "player_points_out_of_bounds", "player_points_outside_viewport", "ball_out_of_bounds", "ambiguous_team_points", "keeper_identity_ambiguous",
            "frames_two_keepers_same_team", "clock_out_of_period_range", "stoppage_time_snapshots")},
    }
    duplicates = duplicates or []
    totals = Counter()
    for counts in display_counts or []:
        totals.update(counts)
    summary["display_resolution_totals"] = dict(totals)
    summary["before_after"] = compare_with_baseline(uncertain)
    summary["duplicate_snapshot_consistency"] = {
        "copies_checked": len(duplicates),
        "states": dict(Counter(d["state"] for d in duplicates)),
        "mirrored_with_confident_reference": sum(d["state"] == "mirrored" and d["reference_confident"] for d in duplicates),
        "mirrored_confident_member": sum(d["state"] == "mirrored" and d["member_confident"] for d in duplicates),
    }
    summary["periods_requiring_manual_orientation_review"] = sum(
        1 for i in issues if i["priority"] == "P1" and i["issue"] in (
            "ambiguous_camera_evidence", "rule_derived_orientation", "fallback_orientation",
            "conflicting_signals", "derived_from_user_reviewed_period")
    )
    summary["orientation_conflicts"] = summary["contradicted_by_independent_evidence"]
    summary["suspicious_kickoff_frames"] = sum(
        u["is_kickoff"] is True and u["display_outcome"] == "left_uncorrected" and u["visual_class"] != "likely_correct"
        for u in uncertain
    )
    write_csv(OUTPUT_DIR / "duplicate_snapshot_inconsistencies.csv", [d for d in duplicates if d["state"] != "identical"])
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def _worker(job):
    match_id, metadata, registry, reviews = job
    try:
        return audit_match(match_id, metadata, registry, reviews)
    except (VisualizationError, ValueError, KeyError) as error:
        return {"error": f"{match_id}: {error!r}", "periods": [], "uncertain": [], "unusual": [], "inconsistent": [],
                "duplicates": [], "display_counts": {}}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        if not rows:
            file.write("")
            return
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--match-id", type=int, nargs="*")
    args = parser.parse_args()

    excluded = load_exclusion_manifest()
    registry = load_calibration_registry()
    reviews = load_period_reviews()
    metadata = load_match_metadata_index()
    match_ids = [m for m in (args.match_id or available_360_match_ids()) if m not in excluded]
    jobs = [(m, metadata.get(m), registry, reviews) for m in match_ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(pool.map(_worker, jobs, chunksize=2))

    errors = [o["error"] for o in outputs if "error" in o]
    periods = sorted((r for o in outputs for r in o["periods"]), key=lambda r: (r["match_id"], r["period"]))
    uncertain = [r for o in outputs for r in o["uncertain"]]
    unusual = [r for o in outputs for r in o["unusual"]]
    inconsistent = [r for o in outputs for r in o["inconsistent"]]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for row in uncertain:
        # Signals describe the frame as DRAWN, so only uncorrected frames get them.
        flipped, correct = frame_signals(row) if row["display_outcome"] == "left_uncorrected" else ([], [])
        row["signals_flipped"] = "; ".join(flipped)
        row["signals_correct"] = "; ".join(correct)
        row["visual_class"] = visual_class(row)
    write_csv(OUTPUT_DIR / "all_period_status.csv", periods)
    write_csv(OUTPUT_DIR / "uncertain_frames.csv", uncertain)
    build_reports(periods, uncertain, unusual, len(match_ids),
                  [d for o in outputs for d in o.get("duplicates", [])], [o.get("display_counts", {}) for o in outputs])
    write_csv(OUTPUT_DIR / "unusual_camera_frames.csv", unusual)
    write_csv(OUTPUT_DIR / "inconsistent_frames.csv", inconsistent)
    overrides = [r for o in outputs for r in o.get("overrides", [])]
    write_csv(OUTPUT_DIR / "duplicate_consensus_overrides.csv", overrides)
    (OUTPUT_DIR / "raw_counts.json").write_text(json.dumps({
        "matches_audited": len(match_ids), "audit_errors": errors, "periods": len(periods),
        "uncertain_rows": len(uncertain), "unusual_rows": len(unusual), "inconsistent_rows": len(inconsistent),
    }, indent=1), encoding="utf-8")
    print(json.dumps({"matches": len(match_ids), "errors": errors, "periods": len(periods),
                      "uncertain": len(uncertain), "unusual": len(unusual), "inconsistent": len(inconsistent),
                      "status": Counter(r["status"] for r in periods),
                      "uncertain_outcomes": Counter(r["display_outcome"] for r in uncertain)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
