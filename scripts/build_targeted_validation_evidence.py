"""Targeted manual-review evidence for the final visualization validation.

Writes, under results/visualization_final_validation/ (random GIFs are made
separately by scripts/sample_final_validation_gifs.py):

  uncertain_frames/   one sheet per remaining UNCERTAIN snapshot (from the
                      outstanding-issues audit): previous frame | the frame as
                      currently drawn | the 180-degree alternative | next
                      frame, all through the production render_event_frame
                      (camera-overlay); index.csv
  extra_time/         CLI GIFs, standard flags: last 120 s of period 2 and the
                      first 60 s of period 3 for 3869220 and 4018357; index.csv
  france_morocco/     CLI GIFs of 3901833 around the first France shot of each
                      half; the review-page P1/P2 snapshots; index.csv
  corrected_frame_examples/  copies of the existing before/after evidence
  reliability_sample_index.csv  the 26 queued validation matches (review.html)

Read-only with respect to code, registry, raw and ML data.

Usage:
    .venv/bin/python scripts/build_targeted_validation_evidence.py
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from generate_random_visual_checks import RENDER_FLAGS  # noqa: E402
from src.animate_statsbomb_360_match import (  # noqa: E402
    DirectionInference,
    JoinedFrame,
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    build_timing_plan,
    compute_orientation_results,
    compute_shared_group_info,
    create_landscape_figure,
    determine_team_style,
    display_orientation_results,
    frame_review_corrections,
    format_match_clock,
    get_visual_theme,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    orientation_provenance_text,
    render_event_frame,
    resolve_display_plan,
    timestamp_seconds,
)
from src.orientation_correction import ORIENTATION_ROTATED_180, apply_correction  # noqa: E402
from src.team_identity import snapshot_key  # noqa: E402
from src.team_normalization import rotate_event_180  # noqa: E402

OUT = PROJECT_ROOT / "results" / "visualization_final_validation"
AUDIT = PROJECT_ROOT / "results" / "physical_orientation" / "outstanding_issues"
QUEUE = PROJECT_ROOT / "results" / "physical_orientation" / "final_orientation_audit" / "manual_review_queue.json"
PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def pipeline(match_id: int):
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    return raw, plan, used, display


def render_png(match_id, joined, plan, style, shared, raw_by_id, names, note, path) -> None:
    theme = get_visual_theme("dark")
    figure, ax = create_landscape_figure(theme)
    period = joined.event["period"]
    directions = {names[t]: d for t, d in (plan.attacking_directions(period) or {}).items()}
    timing = build_timing_plan([joined], "fixed", 250, 1.0, 80, 1500)[0]
    render_event_frame(ax, match_id, joined, 1, 1, timing, "fixed", "camera-overlay", style,
                       DirectionInference(directions, "", orientation_provenance_text(plan, period)),
                       theme, False, True, shared, raw_by_id)
    ax.text(1.0, 1.055, note, transform=ax.transAxes, ha="right", va="bottom",
            color="#ffd166", fontsize=13, fontweight="bold", clip_on=False)
    figure.savefig(path, dpi=55, facecolor=theme.figure_background)
    plt.close(figure)


def grid(paths: list[Path], target: Path) -> None:
    images = [Image.open(p).convert("RGB") for p in paths]
    w, h = images[0].size
    sheet = Image.new("RGB", (w * 2, h * ((len(images) + 1) // 2)), "black")
    for i, image in enumerate(images):
        sheet.paste(image, ((i % 2) * w, (i // 2) * h))
    sheet.save(target)
    for p in paths:
        p.unlink()


def uncertain_frames() -> list[dict]:
    out = OUT / "uncertain_frames"
    out.mkdir(parents=True, exist_ok=True)
    with (AUDIT / "uncertain_frames.csv").open(encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if r["display_outcome"] == "left_uncorrected"]
    index, done = [], set()
    cache: dict[int, tuple] = {}
    for row in rows:
        match_id = int(row["match_id"])
        if match_id not in cache:
            cache[match_id] = pipeline(match_id)
        raw, plan, used, display = cache[match_id]
        i = next(k for k, j in enumerate(raw.usable_frames) if j.event["id"] == row["event_id"])
        key = (match_id, snapshot_key(raw.usable_frames[i].frame_360))
        events_sharing = [r for r in rows if int(r["match_id"]) == match_id and r["timestamp"] == row["timestamp"]]
        if key in done:
            continue
        done.add(key)
        period = raw.usable_frames[i].event["period"]
        same_period = [k for k, j in enumerate(raw.usable_frames) if j.event["period"] == period]
        position = same_period.index(i)
        own_key = snapshot_key(raw.usable_frames[i].frame_360)
        prev = next((k for k in reversed(same_period[:position]) if snapshot_key(raw.usable_frames[k].frame_360) != own_key), None)
        nxt = next((k for k in same_period[position + 1:] if snapshot_key(raw.usable_frames[k].frame_360) != own_key), None)
        names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
        style = determine_team_style(raw.usable_frames, "#4aa8ff", "#ff5a5f")
        shared = compute_shared_group_info(raw)
        raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
        shown = display.usable_frames[i]
        alternative = JoinedFrame(rotate_event_180(shown.event), apply_correction(shown.frame_360, ORIENTATION_ROTATED_180))
        label = f"{match_id}_p{period}_{row['timestamp'].replace(':', '').replace('.', '_')}"
        panels = []
        for tag, joined, note in (
            ("a", display.usable_frames[prev] if prev is not None else None, "PREVIOUS frame (as drawn)"),
            ("b", shown, "THIS frame AS DRAWN (current)"),
            ("c", alternative, "THIS frame rotated 180 (alternative)"),
            ("d", display.usable_frames[nxt] if nxt is not None else None, "NEXT frame (as drawn)"),
        ):
            if joined is None:
                continue
            path = out / f"{label}_{tag}.png"
            render_png(match_id, joined, plan, style, shared, raw_by_id, names, note, path)
            panels.append(path)
        sheet = out / f"{label}.png"
        grid(panels, sheet)
        index.append({
            "sheet": str(sheet.relative_to(PROJECT_ROOT)), "match_id": match_id,
            "teams": f"{raw.usable_frames[0].event['team']['name']} / {next(n for n in names.values() if n != raw.usable_frames[0].event['team']['name'])}",
            "period": period, "match_clock": row["match_clock"],
            "events": "; ".join(f"{r['event_type']} ({r['team']})" for r in events_sharing),
            "is_kickoff": row["is_kickoff"], "class": row["visual_class"],
            "signals_flipped": row["signals_flipped"], "signals_correct": row["signals_correct"],
            "previous_clock": format_match_clock(raw.usable_frames[prev].event) if prev is not None else "",
            "next_clock": format_match_clock(raw.usable_frames[nxt].event) if nxt is not None else "",
            "question_for_reviewer": "Is THIS frame AS DRAWN continuous with PREVIOUS/NEXT (players in plausible places, keepers at their own goals)? If the ALTERNATIVE fits better, it is a flip.",
        })
        print("uncertain sheet", sheet.name)
    write(out / "index.csv", index)
    return index


def cli_gif(match_id: int, period: int, start: float, end: float, target: Path) -> dict:
    command = [PYTHON, str(PROJECT_ROOT / "src" / "animate_statsbomb_360_match.py"),
               "--match-id", str(match_id), "--period", str(period),
               "--start-second", f"{start:.3f}", "--end-second", f"{end:.3f}", *RENDER_FLAGS,
               "--output", str(target), "--event-frame-manifest", str(target.with_suffix(".csv"))]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:])
    stdout = completed.stdout
    frames = int(next(l.split(":")[1] for l in stdout.splitlines() if l.startswith("Selected real event frames")))
    direction = next((l for l in stdout.splitlines() if l.startswith("Attacking-direction inference")), "")
    print("gif", target.name, frames)
    return {"gif": str(target.relative_to(PROJECT_ROOT)), "frames_360": frames, "direction_check": direction,
            "camera_layer_active": "Display orientation layer: provenance=" in stdout}


def extra_time() -> list[dict]:
    out = OUT / "extra_time"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for match_id in (3869220, 4018357):
        raw, plan, _, display = pipeline(match_id)
        names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
        for period, where in ((2, "end"), (3, "start"), (4, "start")):
            frames = [j for j in display.usable_frames if j.event["period"] == period]
            if not frames:
                continue
            times = [timestamp_seconds(j.event["timestamp"]) for j in frames]
            start, end = (times[-1] - 120, times[-1] + 0.001) if where == "end" else (times[0], times[0] + 60)
            target = out / f"{match_id}_p{period}_{where}.gif"
            info = cli_gif(match_id, period, start, end, target)
            selected = [j for j, t in zip(frames, times) if start <= t < end]
            index.append({
                **info, "match_id": match_id, "period": period, "segment": f"{where} of period",
                "match_clock": f"{format_match_clock(selected[0].event)} - {format_match_clock(selected[-1].event)}",
                "displayed_left_to_right_team": names[plan.transforms[period].reference_team_id],
                "plan_source": plan.transforms[period].source,
                "question_for_reviewer": "Same team attacks left->right at the end of P2 and the start of P3 (extra time continued at P2's ends), and the ends swap again in P4?",
            })
    write(out / "index.csv", index)
    return index


def france_morocco() -> list[dict]:
    out = OUT / "france_morocco"
    out.mkdir(parents=True, exist_ok=True)
    raw, plan, _, display = pipeline(3901833)
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    index = []
    for period in (1, 2):
        shot = next(j for j in display.usable_frames
                    if j.event["period"] == period and j.event["type"]["name"] == "Shot" and j.event["team"]["name"] == "France Women's")
        t = timestamp_seconds(shot.event["timestamp"])
        target = out / f"3901833_p{period}_around_first_france_shot.gif"
        info = cli_gif(3901833, period, t - 40, t + 10, target)
        index.append({
            **info, "match_id": 3901833, "period": period,
            "first_france_shot_clock": format_match_clock(shot.event),
            "shot_location_as_drawn": shot.event["location"],
            "displayed_left_to_right_team": names[plan.transforms[period].reference_team_id],
            "plan_source": plan.transforms[period].source,
            "question_for_reviewer": "P1: France attacks left->right (shot near the right goal); P2: France attacks right->left (shot near the left goal). Keepers at their own goals; camera polygon narrow at the bottom in both halves?",
        })
    review = PROJECT_ROOT / "results" / "physical_orientation" / "final_orientation_audit" / "review" / "3901833"
    for name in ("p1_proposed.png", "p2_proposed.png"):
        if (review / name).exists():
            shutil.copy(review / name, out / f"review_snapshot_{name}")
    write(out / "index.csv", index)
    return index


def corrected_examples() -> None:
    out = OUT / "corrected_frame_examples"
    out.mkdir(parents=True, exist_ok=True)
    source = AUDIT / "fix_evidence"
    for name in ("01_france_morocco_p1_ball_receipt.png", "02_ball_receipt_duplicate_fix.png",
                 "03_previously_correct_confident_frame.png", "04_previously_correct_resolved_frame.png",
                 "05_kickoff_own_half_fix.png", "06_duplicate_consensus_fix.png", "07_camera_continuity_fix.png",
                 "continuity_3773631_neighbours.png", "contact_sheet.png"):
        shutil.copy(source / name, out / name)


def reliability_index() -> None:
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    rows = [{
        "match_id": e["match_id"], "teams": f"{e['home_team']} vs {e['away_team']}",
        "competition": f"{e['competition']} {e['season']}", "why_queued": e["why"],
        "proposed_reference_by_period": json.dumps(e.get("proposed_reference_by_period")),
        "snapshots": "; ".join(e["images"].values()),
        "review_page": "results/physical_orientation/final_orientation_audit/review/review.html",
    } for e in queue if not e["why"].startswith("extra time")]
    write(OUT / "reliability_sample_index.csv", rows)


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    uncertain_frames()
    extra_time()
    france_morocco()
    corrected_examples()
    reliability_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
