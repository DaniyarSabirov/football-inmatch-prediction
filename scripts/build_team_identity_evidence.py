"""Visual evidence for the 2026-10-24 team-identity investigation.

frame_505/ (match 3930164, Slovakia vs Belgium, P1 00:00.505 kick-off):
  sheet.png       the frame as drawn | the next two snapshots | the frame
                  again with every player whose raw `teammate` flag
                  contradicts his paired position 1.1 s later ringed
  points.csv      every freeze-frame point: raw flag, drawn team, paired
                  next-frame team, distance, Law 8 half check
  opening_10s.gif CLI render (standard flags) of the first 10 s of P1

shared_source_3857293/ (Uruguay vs Ghana, the two groups the display-only
continuity rule switched): per group, previous neighbour | colours with the
Problem C source | colours with the refined source | next neighbour.

Nothing here changes data, registry or code; the "refined" panels are what
the renderer now draws.

Usage:
    .venv/bin/python scripts/build_team_identity_evidence.py
"""

from __future__ import annotations

import csv
import math
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
from scipy.optimize import linear_sum_assignment  # noqa: E402

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
    match_team_ids,
    orientation_provenance_text,
    refine_shared_sources_by_continuity,
    render_event_frame,
    resolve_display_plan,
)
from src.team_identity import resolve_absolute_team_for_point  # noqa: E402

OUT = PROJECT_ROOT / "results" / "visualization_final_validation" / "team_identity_investigation"


def pipeline(match_id: int):
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    shared = compute_shared_group_info(raw)
    raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    pair = tuple(names[t] for t in match_team_ids(raw))
    style = determine_team_style(raw.usable_frames, "#4aa8ff", "#ff5a5f")
    return raw, plan, display, shared, raw_by_id, names, pair, style


def panel(match_id, joined, plan, style, shared, raw_by_id, names, note, path, rings=()):
    theme = get_visual_theme("dark")
    figure, ax = create_landscape_figure(theme)
    period = joined.event["period"]
    directions = {names[t]: d for t, d in (plan.attacking_directions(period) or {}).items()}
    timing = build_timing_plan([joined], "fixed", 250, 1.0, 80, 1500)[0]
    render_event_frame(ax, match_id, joined, 1, 1, timing, "fixed", "camera-overlay", style,
                       DirectionInference(directions, "", orientation_provenance_text(plan, period)),
                       theme, False, True, shared, raw_by_id)
    for x, y in rings:
        ax.scatter([x], [y], s=900, facecolors="none", edgecolors="#ffffff", linewidths=2.5, zorder=12)
    ax.text(1.0, 1.055, note, transform=ax.transAxes, ha="right", va="bottom",
            color="#ffd166", fontsize=12, fontweight="bold", clip_on=False)
    figure.savefig(path, dpi=55, facecolor=theme.figure_background)
    plt.close(figure)


def sheet(paths, target, columns=2):
    images = [Image.open(p).convert("RGB") for p in paths]
    w, h = images[0].size
    out = Image.new("RGB", (w * columns, h * ((len(images) + columns - 1) // columns)), "black")
    for i, image in enumerate(images):
        out.paste(image, ((i % columns) * w, (i // columns) * h))
    out.save(target)
    for p in paths:
        p.unlink()


def drawn_teams(joined, info, raw_by_id, pair):
    return [(tuple(p["location"]), resolve_absolute_team_for_point(p, joined.event, info, raw_by_id, pair), p)
            for p in joined.frame_360["freeze_frame"]]


def frame_505() -> None:
    out = OUT / "frame_505"
    out.mkdir(parents=True, exist_ok=True)
    match_id = 3930164
    raw, plan, display, shared, raw_by_id, names, pair, style = pipeline(match_id)
    frames = display.usable_frames
    i = next(k for k, j in enumerate(frames) if j.event["timestamp"] == "00:00:00.505")
    nxt = [k for k in range(i + 1, len(frames)) if frames[k].event["timestamp"] != frames[i].event["timestamp"]]
    nxt = [nxt[0], next(k for k in nxt if frames[k].event["timestamp"] != frames[nxt[0]].event["timestamp"])]
    a = drawn_teams(frames[i], None, raw_by_id, pair)
    b = drawn_teams(frames[nxt[0]], None, raw_by_id, pair)
    cost = [[math.dist(p[0], q[0]) for q in b] for p in a]
    rows, cols = linear_sum_assignment(cost)
    kicking = frames[i].event["team"]["name"]
    attacks_right = plan.attacking_directions(1)[frames[i].event["team"]["id"]] == 1
    records, rings = [], []
    for r, c in zip(rows, cols):
        location, team, player = a[r]
        own_half = (location[0] < 60) == ((team == kicking) == attacks_right)
        contradicts = cost[r][c] <= 3.0 and team != b[c][1]
        if contradicts:
            rings.append(location)
        records.append({
            "x_drawn": round(location[0], 2), "y_drawn": round(location[1], 2),
            "raw_teammate_flag": player.get("teammate"), "actor": player.get("actor"), "keeper": player.get("keeper"),
            "drawn_team": team, "paired_next_frame_team": b[c][1], "pair_distance_yd": round(cost[r][c], 2),
            "colour_contradicts_next_frame": contradicts,
            "in_own_half_by_drawn_team (Law 8 at kick-off)": "on halfway line" if abs(location[0] - 60) < 1.5 else own_half,
        })
    with (out / "points.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    paths = []
    for tag, k, note in (("a", i, "00:00.505 KICK-OFF as drawn (raw teammate flags)"),
                         ("b", nxt[0], "NEXT snapshot (1.1 s later)"),
                         ("c", nxt[1], "Following snapshot")):
        path = out / f"_{tag}.png"
        panel(match_id, frames[k], plan, style, shared, raw_by_id, names, note, path)
        paths.append(path)
    path = out / "_d.png"
    panel(match_id, frames[i], plan, style, shared, raw_by_id, names,
          f"00:00.505 ringed: {len(rings)} players whose colour contradicts the next frame", path, rings)
    paths.append(path)
    sheet(paths, out / "sheet.png")
    command = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), str(PROJECT_ROOT / "src" / "animate_statsbomb_360_match.py"),
               "--match-id", str(match_id), "--period", "1", "--start-second", "0", "--end-second", "10",
               *RENDER_FLAGS, "--output", str(out / "opening_10s.gif"), "--event-frame-manifest", str(out / "opening_10s.csv")]
    subprocess.run(command, check=True, capture_output=True, cwd=PROJECT_ROOT)
    print("frame_505:", len(rings), "contradicting points;", out)


def shared_source_3857293() -> None:
    out = OUT / "shared_source_3857293"
    out.mkdir(parents=True, exist_ok=True)
    match_id = 3857293
    raw, plan, display, shared, raw_by_id, names, pair, style = pipeline(match_id)
    refined, switches = refine_shared_sources_by_continuity(display, shared, raw_by_id, pair)
    frames = display.usable_frames
    for switch in switches:
        first = min(k for k, j in enumerate(frames) if j.event["id"] in switch["events"])
        last = max(k for k, j in enumerate(frames) if j.event["id"] in switch["events"])
        before = next(k for k in range(first - 1, -1, -1) if frames[k].event["id"] not in shared)
        after = next(k for k in range(last + 1, len(frames)) if frames[k].event["id"] not in shared)
        label = f"p{switch['period']}_{switch['timestamp'].replace(':', '').replace('.', '_')}"
        paths = []
        for tag, k, groups, note in (
            ("a", before, shared, "PREVIOUS snapshot (not shared)"),
            ("b", first, shared, f"BEFORE: colours read via {switch['previous_source_team']} (Problem C source)"),
            ("c", first, refined, f"AFTER: colours read via {switch['new_source_team']} (continuity rule)"),
            ("d", after, shared, "NEXT snapshot (not shared)"),
        ):
            path = out / f"_{label}_{tag}.png"
            panel(match_id, frames[k], plan, style, groups, raw_by_id, names, note, path)
            paths.append(path)
        sheet(paths, out / f"{label}.png")
        print("3857293:", label)


def main() -> int:
    frame_505()
    shared_source_3857293()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
