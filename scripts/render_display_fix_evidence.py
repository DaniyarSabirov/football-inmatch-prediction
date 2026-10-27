"""Before/after evidence for the 2026-10-23 frame-level display fixes.

BEFORE = the pre-fix display logic: display_orientation_results without the
display plan, without duplicate evidence and without the kick-off rule (the
code path the renderer used before the fix). AFTER = the current renderer
logic (resolve_display_plan + display_orientation_results with the plan).
Both are drawn by the production render_event_frame in the camera-overlay
view, so every panel shows the freeze-frame players, actor, ball and the
visible_area polygon.

Each case is one PNG with BEFORE (top) and AFTER (bottom); where a same-moment
confident copy or a neighbouring frame exists it is shown as a third REFERENCE
panel (current logic). A contact sheet combines all cases.

Usage:
    .venv/bin/python scripts/render_display_fix_evidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.animate_statsbomb_360_match import (  # noqa: E402
    DirectionInference,
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    build_timing_plan,
    compute_orientation_results,
    compute_shared_group_info,
    create_landscape_figure,
    determine_team_style,
    display_orientation_results,
    get_visual_theme,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    orientation_provenance_text,
    render_event_frame,
    resolve_display_plan,
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "physical_orientation" / "outstanding_issues" / "fix_evidence"

# (label, match, period, timestamp, event type, reference (timestamp, type) or None, description)
CASES = [
    ("01_france_morocco_p1_ball_receipt", 3901833, 1, "00:09:58.510", "Ball Receipt*",
     ("00:09:58.510", "Ball Recovery"), "France-Morocco P1: flipped Ball Receipt* fixed by its same-moment duplicate"),
    ("02_ball_receipt_duplicate_fix", 3773466, 1, "00:13:22.642", "Ball Receipt*",
     ("00:13:20.733", "Pressure"), "Celta-Barcelona: flipped Ball Receipt* fixed by its same-moment duplicate"),
    ("03_previously_correct_confident_frame", 3901833, 1, "00:09:58.510", "Ball Recovery",
     None, "Previously correct confident frame: unchanged (no regression)"),
    ("04_previously_correct_resolved_frame", 3902239, 1, "00:23:04.965", "Ball Receipt*",
     None, "Frame fixed in round 3 by lean+camera: unchanged (no regression)"),
    ("05_kickoff_own_half_fix", 3773552, 1, "00:00:01.454", "Pass",
     None, "Kick-off: all players in the wrong halves before; fixed by the own-half rule"),
    ("06_duplicate_consensus_fix", 3773526, 2, "00:07:30.149", "Ball Receipt*",
     ("00:07:30.149", "Carry"), "Weak confident Ball Receipt* copy disagreed with its duplicates; camera decides"),
    ("07_camera_continuity_fix", 3773631, 1, "00:25:39.675", "Ball Receipt*",
     None, "No duplicate: camera side and temporal continuity both said flipped"),
]


def build(match_id: int):
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    before_results, _ = display_orientation_results(raw, results, inference, None, use_duplicates=False, use_kickoff_rule=False)
    after_results, _ = display_orientation_results(raw, results, inference, plan)
    before = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, before_results), plan)
    after = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, after_results), plan)
    return raw, plan, before, after, before_results, after_results


def render_panel(match_id, joined, plan, style, shared, raw_by_id, note, path):
    theme = get_visual_theme("dark")
    figure, ax = create_landscape_figure(theme)
    timing = build_timing_plan([joined], "fixed", 250, 1.0, 80, 1500)[0]
    period = joined.event["period"]
    render_event_frame(
        ax, match_id, joined, 1, 1, timing, "fixed", "camera-overlay", style,
        DirectionInference(style_directions(plan, period, style), "", orientation_provenance_text(plan, period)),
        theme, False, True, shared, raw_by_id,
    )
    ax.text(1.0, 1.055, note, transform=ax.transAxes, ha="right", va="bottom",
            color="#ffd166", fontsize=13, fontweight="bold", clip_on=False)
    figure.savefig(path, dpi=60, facecolor=theme.figure_background)
    plt.close(figure)


def style_directions(plan, period, style):
    by_id = plan.attacking_directions(period) or {}
    return {TEAM_NAMES[team_id]: direction for team_id, direction in by_id.items()}


TEAM_NAMES: dict[int, str] = {}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    sheets = []
    for label, match_id, period, timestamp, event_type, reference, description in CASES:
        if match_id not in cache:
            cache[match_id] = build(match_id)
        raw, plan, before, after, before_results, after_results = cache[match_id]
        TEAM_NAMES.clear()
        TEAM_NAMES.update({j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames})
        style = determine_team_style(raw.usable_frames, "#4aa8ff", "#ff5a5f")
        shared = compute_shared_group_info(raw)
        raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}

        def find(ts, kind):
            return next(i for i, j in enumerate(raw.usable_frames)
                        if j.event["period"] == period and j.event["timestamp"] == ts and j.event["type"]["name"] == kind)

        index = find(timestamp, event_type)
        panels = []
        b_note = f"BEFORE ({before_results[index].orientation} {before_results[index].resolved_by})"
        a_note = f"AFTER ({after_results[index].orientation} via {after_results[index].resolved_by})"
        for tag, display, note in (("before", before, b_note), ("after", after, a_note)):
            path = OUTPUT_DIR / f"{label}_{tag}.png"
            render_panel(match_id, display.usable_frames[index], plan, style, shared, raw_by_id, note, path)
            panels.append(path)
        if reference is not None:
            ref_index = find(*reference)
            path = OUTPUT_DIR / f"{label}_reference.png"
            render_panel(match_id, after.usable_frames[ref_index], plan, style, shared, raw_by_id,
                         f"REFERENCE: {reference[1]} {reference[0][3:]} (confident)", path)
            panels.append(path)
        images = [Image.open(p).convert("RGB") for p in panels]
        width, height = images[0].size
        combined = Image.new("RGB", (width, height * len(images)), "black")
        for position, image in enumerate(images):
            combined.paste(image, (0, position * height))
        combined.save(OUTPUT_DIR / f"{label}.png")
        sheets.append((label, combined))
        print(f"{label}: {description} -> {OUTPUT_DIR / (label + '.png')}")

    thumbs = [image.resize((image.width // 2, image.height // 2)) for _, image in sheets]
    width = max(t.width for t in thumbs)
    sheet = Image.new("RGB", (width * len(thumbs), max(t.height for t in thumbs)), "black")
    for position, thumb in enumerate(thumbs):
        sheet.paste(thumb, (position * width, 0))
    sheet.save(OUTPUT_DIR / "contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
