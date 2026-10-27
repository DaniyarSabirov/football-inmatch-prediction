"""Forensic orientation audit for Belgium vs Russia (match 3788743).

Produces, under results/physical_orientation/belgium_russia_broadcast_audit/:
  transformation_trace.json   raw JSON -> displayed coordinates, every stage,
                              for the Courtois Carry (P2 26:53.383) and the
                              broadcast-synchronised Lukaku goal (87:45)
  variant_*.png               one render per candidate transform
  comparison_sheet.png        all variants side by side
  pitch_diagram.png           defended goals / attack directions / camera side
  lukaku_goal_camera_view.png the synchronised broadcast moment, new view

Every variant transforms players, actor, ball/event location, end_location,
visible_area, goal labels and arrows with the SAME map. Nothing here edits
raw data, the registry, or the ML pipeline.

Usage:
    .venv/bin/python scripts/belgium_russia_broadcast_audit.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from src.animate_statsbomb_360_match import (  # noqa: E402
    DirectionInference,
    JoinedFrame,
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    build_timing_plan,
    compute_orientation_results,
    create_landscape_figure,
    determine_team_style,
    get_visual_theme,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    match_team_ids,
    orientation_provenance_text,
    render_event_frame,
)
from src.broadcast_orientation import build_display_plan, estimate_camera_side  # noqa: E402
from src.team_normalization import apply_period_end_swap, resolve_reference_team_id  # noqa: E402

MATCH_ID = 3788743
COURTOIS_CARRY = "6d4668fb-2a8b-4f79-8d90-b7d87768a5f3"
LUKAKU_GOAL = "b0e8f7c7-291e-4d0a-81d4-0f99f4fcd73d"
OUTPUT_DIR = PROJECT_ROOT / "results" / "physical_orientation" / "belgium_russia_broadcast_audit"

# Candidate maps applied on top of the PREVIOUS (legacy) display frame, i.e.
# the frame of the screenshot the project owner reported.
VARIANTS = {
    "identity": (lambda x, y: (x, y), 1, "Previous rendering (legacy default)"),
    "rotate_180": (lambda x, y: (120 - x, 80 - y), -1, "180-degree rotation"),
    "x_reflection": (lambda x, y: (120 - x, y), -1, "X reflection (mirror image)"),
    "y_reflection": (lambda x, y: (x, 80 - y), 1, "Y reflection (mirror image)"),
}


def _map_pair(value, fn):
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
        return list(fn(float(value[0]), float(value[1])))
    return value


def transform_joined(joined: JoinedFrame, fn) -> JoinedFrame:
    event = dict(joined.event)
    event["location"] = _map_pair(event.get("location"), fn)
    for key, value in list(event.items()):
        if isinstance(value, dict) and "end_location" in value:
            event[key] = {**value, "end_location": _map_pair(value["end_location"], fn)}
    frame = dict(joined.frame_360)
    frame["freeze_frame"] = [{**p, "location": _map_pair(p.get("location"), fn)} for p in frame.get("freeze_frame") or []]
    area = frame.get("visible_area") or []
    mapped: list[float] = []
    for i in range(0, len(area) - 1, 2):
        mapped.extend(fn(area[i], area[i + 1]))
    frame["visible_area"] = mapped
    return JoinedFrame(event=event, frame_360=frame)


def frame_summary(joined: JoinedFrame) -> dict:
    keeper = [p["location"] for p in joined.frame_360["freeze_frame"] if p.get("keeper")]
    actor = [p["location"] for p in joined.frame_360["freeze_frame"] if p.get("actor")]
    area = joined.frame_360.get("visible_area") or []
    camera = estimate_camera_side(area)
    return {
        "event_location": joined.event.get("location"),
        "actor_location": actor[0] if actor else None,
        "keeper_locations": keeper,
        "freeze_frame": [
            {k: p.get(k) for k in ("teammate", "actor", "keeper")} | {"location": [round(c, 3) for c in p["location"]]}
            for p in joined.frame_360["freeze_frame"]
        ],
        "visible_area_vertices": [[round(area[i], 3), round(area[i + 1], 3)] for i in range(0, len(area) - 1, 2)],
        "camera_side": camera.side,
        "camera_apex": [round(c, 2) for c in camera.apex] if camera.apex else None,
    }


def render(joined: JoinedFrame, team_style, directions, path: Path, title_note: str, provenance: str = "") -> None:
    theme = get_visual_theme("dark")
    figure, ax = create_landscape_figure(theme)
    timing = build_timing_plan([joined], "fixed", 250, 1.0, 80, 1500)[0]
    render_event_frame(
        ax, MATCH_ID, joined, 1, 1, timing, "fixed", "camera-overlay", team_style,
        DirectionInference(directions, "", provenance), theme,
    )
    ax.text(1.0, 1.055, title_note, transform=ax.transAxes, ha="right", va="bottom",
            color="#ffd166", fontsize=13, fontweight="bold", clip_on=False)
    figure.savefig(path, dpi=80, facecolor=theme.figure_background)
    plt.close(figure)


def pitch_diagram(path: Path, left_defender: str, right_defender: str, colors: dict[str, str]) -> None:
    figure, ax = plt.subplots(figsize=(12, 7.6), facecolor="white")
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=True, fc="#e8f3e8", ec="#2d572c", lw=2))
    ax.plot([60, 60], [0, 80], color="#2d572c")
    for x in (-3, 120):
        ax.add_patch(Rectangle((x, 36), 3, 8, fc="#cccccc", ec="#333333"))
    ax.text(-5, 40, f"{left_defender}\ndefends\nLEFT goal", ha="right", va="center", color=colors[left_defender], fontsize=12, fontweight="bold")
    ax.text(125, 40, f"{right_defender}\ndefends\nRIGHT goal", ha="left", va="center", color=colors[right_defender], fontsize=12, fontweight="bold")
    ax.add_patch(FancyArrowPatch((80, 28), (30, 28), arrowstyle="-|>", mutation_scale=25, color=colors[right_defender], lw=3))
    ax.text(55, 24, f"{right_defender} attacks right -> left", ha="center", color=colors[right_defender], fontsize=12)
    ax.add_patch(FancyArrowPatch((40, 52), (90, 52), arrowstyle="-|>", mutation_scale=25, color=colors[left_defender], lw=3))
    ax.text(65, 58, f"{left_defender} attacks left -> right", ha="center", color=colors[left_defender], fontsize=12)
    ax.text(60, -4, "FAR touchline (StatsBomb display y = 0)", ha="center", fontsize=11)
    ax.text(60, 86.5, "NEAR touchline = broadcast camera side (display y = 80, bottom of new view)\n"
            "camera side: inferred from visible_area apex; consistent with broadcast at 87:43 (not independently filmed)",
            ha="center", va="top", fontsize=10)
    ax.text(60, 70, "Previous view: the bottom of the screen showed the FAR touchline (camera at top)\n"
            "and left/right were swapped -- both errors are one 180-degree rotation.", ha="center", fontsize=10, color="#7a1f1f")
    ax.set_xlim(-28, 148)
    ax.set_ylim(92, -10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Belgium vs Russia, 2nd half (match 3788743) -- verified display orientation", fontsize=13)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_and_join_match(MATCH_ID)
    results = compute_orientation_results(raw)
    corrected = apply_orientation_correction_to_match(raw, results)
    events = [j.event for j in raw.usable_frames]
    team_ids = match_team_ids(raw)
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    periods = sorted({e["period"] for e in events})

    automatic = resolve_reference_team_id(events)
    legacy_reference = {p: apply_period_end_swap(events, automatic, p) for p in periods}
    inference = infer_broadcast_orientation_for_match(raw, results)
    camera_plan = build_display_plan(inference, team_ids, periods, fallback_period_1_reference=automatic)

    index = {j.event["id"]: i for i, j in enumerate(raw.usable_frames)}
    team_style = determine_team_style(raw.usable_frames, "#4aa8ff", "#ff5a5f")
    colors = {team_style.team_one: team_style.team_one_color, team_style.team_two: team_style.team_two_color}

    from src.animate_statsbomb_360_match import _apply_team_normalization_per_period  # noqa: E402
    legacy_display = _apply_team_normalization_per_period(corrected, legacy_reference)
    camera_display = apply_display_plan_to_match(corrected, camera_plan)

    trace = {
        "match_id": MATCH_ID,
        "teams_by_id": names,
        "handedness_check": "StatsBomb y=0 is the attacker's LEFT (own-frame mean y: Left Back ~13, Right Back ~72); display draws y=0 at top -> proper, non-mirrored map",
        "automatic_reference_team": names[automatic],
        "legacy_reference_by_period": {p: names[t] for p, t in legacy_reference.items()},
        "camera_inference": {
            "status": inference.status,
            "reference_by_period": {p: names[t] for p, t in inference.reference_by_period.items()},
            "periods": [
                {"period": e.period, "support": e.support, "opposition": e.opposition,
                 "agreement": round(e.agreement, 4),
                 "votes_by_team": {names[t]: v for t, v in e.votes_by_team.items()}}
                for e in inference.periods
            ],
        },
        "camera_plan": camera_plan.explanation,
        "events": {},
    }
    for label, event_id in (("courtois_carry_p2_26_53_383", COURTOIS_CARRY), ("lukaku_goal_p2_87_45", LUKAKU_GOAL)):
        i = index[event_id]
        period = raw.usable_frames[i].event["period"]
        team_id = raw.usable_frames[i].event["team"]["id"]
        trace["events"][label] = {
            "event_id": event_id,
            "type": raw.usable_frames[i].event["type"]["name"],
            "player": raw.usable_frames[i].event["player"]["name"],
            "team": names[team_id],
            "period": period,
            "minute_second": [raw.usable_frames[i].event["minute"], raw.usable_frames[i].event["second"]],
            "1_raw": frame_summary(raw.usable_frames[i]),
            "2_problem_a": {"orientation": results[i].orientation, "confidence": round(results[i].confidence, 3)},
            "3_after_problem_a": frame_summary(corrected.usable_frames[i]),
            "4_legacy_display": {
                "reference_team_this_period": names[legacy_reference[period]],
                "rotated": legacy_reference[period] != team_id,
                **frame_summary(legacy_display.usable_frames[i]),
            },
            "5_camera_display": {
                "reference_team_this_period": names[camera_plan.transforms[period].reference_team_id],
                "rotated": camera_plan.rotates(period, team_id),
                "attacking_directions": {names[t]: d for t, d in camera_plan.attacking_directions(period).items()},
                **frame_summary(camera_display.usable_frames[i]),
            },
        }

    # Variants of the Courtois frame, all relative to the previous view.
    i = index[COURTOIS_CARRY]
    legacy_dirs = {names[legacy_reference[2]]: 1, names[next(t for t in team_ids if t != legacy_reference[2])]: -1}
    paths = []
    raw_path = OUTPUT_DIR / "variant_0_raw_coordinates.png"
    render(raw.usable_frames[i], team_style, {}, raw_path, "RAW StatsBomb coordinates (Belgium's own frame; no labels)")
    paths.append(raw_path)
    trace["variants"] = {}
    for n, (key, (fn, sign, title)) in enumerate(VARIANTS.items(), start=1):
        joined = transform_joined(legacy_display.usable_frames[i], fn)
        directions = {team: d * sign for team, d in legacy_dirs.items()}
        path = OUTPUT_DIR / f"variant_{n}_{key}.png"
        render(joined, team_style, directions, path, title)
        paths.append(path)
        trace["variants"][key] = {
            "courtois": joined.event["location"],
            "camera_side": estimate_camera_side(joined.frame_360["visible_area"]).side,
            "belgium_direction": directions["Belgium"],
            "proper_rotation": key in ("identity", "rotate_180"),
        }
    new_path = OUTPUT_DIR / "variant_5_new_camera_view.png"
    dirs = {names[t]: d for t, d in camera_plan.attacking_directions(2).items()}
    render(camera_display.usable_frames[i], team_style, dirs, new_path,
           "NEW default: camera touchline at bottom (== 180-degree rotation of previous)",
           orientation_provenance_text(camera_plan))
    paths.append(new_path)

    images = [Image.open(p) for p in paths]
    w, h = images[0].size
    sheet = Image.new("RGB", (w * 2, h * 3), "black")
    for k, image in enumerate(images):
        sheet.paste(image, ((k % 2) * w, (k // 2) * h))
    sheet.save(OUTPUT_DIR / "comparison_sheet.png")

    g = index[LUKAKU_GOAL]
    render(camera_display.usable_frames[g], team_style, dirs, OUTPUT_DIR / "lukaku_goal_camera_view.png",
           "Lukaku goal 87:45 (broadcast 87:43, BEL 2-0 RUS) -- new view", orientation_provenance_text(camera_plan))
    window = [
        j for j in camera_display.usable_frames
        if j.event["period"] == 2 and 87 * 60 + 35 <= j.event["minute"] * 60 + j.event["second"] <= 87 * 60 + 46
    ]
    trace["broadcast_window_87_35_to_87_46"] = {
        "snapshots": len(window),
        "mean_event_x": round(statistics.fmean(j.event["location"][0] for j in window), 2),
        "camera_sides": sorted({estimate_camera_side(j.frame_360["visible_area"]).side for j in window}),
        "mean_visible_area_x": round(statistics.fmean(
            statistics.fmean(j.frame_360["visible_area"][k] for k in range(0, len(j.frame_360["visible_area"]), 2))
            for j in window
        ), 2),
    }

    pitch_diagram(OUTPUT_DIR / "pitch_diagram.png", "Russia", "Belgium", colors)
    (OUTPUT_DIR / "transformation_trace.json").write_text(json.dumps(trace, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: trace[k] for k in ("legacy_reference_by_period", "camera_plan", "variants", "broadcast_window_87_35_to_87_46")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
