"""FINAL-validation renders of the five reviewed cases (final code; new files
only -- results/team_identity_correction_2026-10-26/case_visualizations/ and the
earlier case_NN evidence in results/visualization_final_validation/
team_identity_manual_review/ is not touched).

Per case (../case_visualizations/case_NN/):
  before_after.png  rows = previous / reviewed / next snapshot (as in the
                    earlier review); left = colours as drawn before
                    (legacy Problem C + display-only refinement), right =
                    colours now (shared evidence-based identity). Players and
                    the visible camera-area polygon are always drawn;
                    corrected points carry a dashed ring, unresolved points
                    the neutral "team unclear" fill.
  context.gif       production CLI with the established flags (orientation
                    correction, display plan, team identity, camera overlay,
                    real time), same window as the earlier context.gif
  points.csv        per point of the reviewed snapshot: raw flag, spec
                    reading, original colour, new colour, status, reason

Usage: .venv/bin/python results/team_identity_final_validation_2026-10-27/diagnostics/render_final_cases.py
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from generate_random_visual_checks import RENDER_FLAGS  # noqa: E402
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
    frame_review_corrections,
    get_visual_theme,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    load_json_records,
    match_team_ids,
    orientation_provenance_text,
    refine_shared_sources_by_continuity,
    render_event_frame,
    resolve_display_plan,
)
from src.team_identity import resolve_absolute_team_for_point, resolve_match_team_identity  # noqa: E402

PREVIOUS = PROJECT_ROOT / "results" / "visualization_final_validation" / "team_identity_manual_review"
OUT = HERE.parent / "case_visualizations"


def pipeline(match_id: int):
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    pair = tuple(names[t] for t in match_team_ids(raw))
    raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
    legacy, _ = refine_shared_sources_by_continuity(display, compute_shared_group_info(raw), raw_by_id, pair)
    identities = resolve_match_team_identity(
        [(j.event, j.frame_360) for j in raw.usable_frames], [r.orientation for r in used], results, pair,
        load_json_records(raw.event_path, "event"),
    )
    style = determine_team_style(raw.usable_frames, "#4aa8ff", "#ff5a5f")
    return plan, display, legacy, raw_by_id, identities, names, pair, style


def panel(match_id, joined, plan, style, names, note, path, legacy=None, raw_by_id=None, identities=None):
    theme = get_visual_theme("dark")
    figure, ax = create_landscape_figure(theme)
    period = joined.event["period"]
    directions = {names[t]: d for t, d in (plan.attacking_directions(period) or {}).items()}
    timing = build_timing_plan([joined], "fixed", 250, 1.0, 80, 1500)[0]
    render_event_frame(ax, match_id, joined, 1, 1, timing, "fixed", "camera-overlay", style,
                       DirectionInference(directions, "", orientation_provenance_text(plan, period)),
                       theme, False, True, legacy, raw_by_id, frame_identities=identities)
    ax.text(1.0, 1.055, note, transform=ax.transAxes, ha="right", va="bottom",
            color="#ffd166", fontsize=12, fontweight="bold", clip_on=False)
    figure.savefig(path, dpi=55, facecolor=theme.figure_background)
    plt.close(figure)


def main() -> int:
    with (PREVIOUS / "review_index.csv").open(encoding="utf-8") as file:
        cases = list(csv.DictReader(file))
    index_rows = []
    for case in cases:
        out = OUT / case["case"]
        out.mkdir(parents=True, exist_ok=True)
        match_id = int(case["match_id"])
        plan, display, legacy, raw_by_id, identities, names, pair, style = pipeline(match_id)
        frames = display.usable_frames
        k = next(i for i, j in enumerate(frames) if j.event["id"] == case["event_id"])
        prev_k = next(i for i, j in enumerate(frames) if format_clock_match(j, case["previous_clock"]) and i != k)
        next_k = next(i for i in range(len(frames) - 1, -1, -1) if format_clock_match(frames[i], case["next_clock"]) and i != k)
        paths = []
        for tag, i in (("prev", prev_k), ("case", k), ("next", next_k)):
            event = frames[i].event
            label = f"{tag.upper()} {event['timestamp'][3:]} {event['type']['name']} ({event['team']['name']})"
            for side, kwargs in (("before", {"legacy": legacy, "raw_by_id": raw_by_id}), ("after", {"identities": identities})):
                path = out / f"_{tag}_{side}.png"
                panel(match_id, frames[i], plan, style, names, f"{side.upper()}: {label}", path, **kwargs)
                paths.append(path)
        images = [Image.open(p).convert("RGB") for p in paths]
        w, h = images[0].size
        sheet = Image.new("RGB", (w * 2, h * 3), "black")
        for n, image in enumerate(images):
            sheet.paste(image, ((n % 2) * w, (n // 2) * h))
        sheet.save(out / "before_after.png")
        for p in paths:
            p.unlink()

        identity = identities[case["event_id"]]
        event = frames[k].event
        points = []
        for i, player in enumerate(frames[k].frame_360["freeze_frame"]):
            points.append({
                "point": i, "x_drawn": round(player["location"][0], 2), "y_drawn": round(player["location"][1], 2),
                "raw_teammate_flag": player.get("teammate"), "actor": player.get("actor"), "keeper": player.get("keeper"),
                "spec_reading": identity.spec_teams[i],
                "original_colour_team": resolve_absolute_team_for_point(player, event, legacy.get(event["id"]), raw_by_id, pair),
                "new_colour_team": identity.teams[i], "status": identity.statuses[i], "reason": identity.reasons[i],
            })
        with (out / "points.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)

        start, end = case["gif_window_period_seconds"].split("-")
        command = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), str(PROJECT_ROOT / "src" / "animate_statsbomb_360_match.py"),
                   "--match-id", str(match_id), "--period", case["period"], "--start-second", start, "--end-second", end,
                   *RENDER_FLAGS, "--theme", "dark", "--output", str(out / "context.gif"), "--event-frame-manifest", str(out / "context_frames.csv")]
        completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr[-2000:])
        changed = sum(p["original_colour_team"] != p["new_colour_team"] for p in points)
        index_rows.append({
            "case": case["case"], "match_id": match_id, "teams": case["teams"], "period": case["period"],
            "match_clock": case["match_clock"], "event_type": case["event_type"], "event_team": case["event_team"],
            "decision": identity.decision, "decision_evidence": identity.decision_evidence,
            "points": len(points), "points_recoloured": changed,
            "statuses": "; ".join(f"{s}={identity.statuses.count(s)}" for s in sorted(set(identity.statuses))),
            "identity_line": next((l for l in completed.stdout.splitlines() if l.startswith("Evidence-based team identity")), ""),
        })
        print(index_rows[-1])
    with (OUT / "index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    return 0


def format_clock_match(joined, clock: str) -> bool:
    from src.animate_statsbomb_360_match import format_match_clock

    return format_match_clock(joined.event) == clock


if __name__ == "__main__":
    raise SystemExit(main())
