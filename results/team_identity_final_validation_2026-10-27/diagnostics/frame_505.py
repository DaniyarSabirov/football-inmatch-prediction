"""Frame 505 (match 3930164, Slovakia kick-off, P1 00:00:00.505) diagnostics.

  frame_505/panels.png   505 and the next two distinct snapshots, drawn
                         exactly as the renderer draws them (established
                         flags); on the 505 panel, white rings mark players
                         whose drawn team stands in the wrong half under Law 8
                         (>= 2 yd from the halfway line, actor excluded).
                         DIAGNOSTIC labels on every panel.
  frame_505/context.gif  production CLI, established flags, P1 0-8 s
  frame_505/facts.json   raw flags, orientation route, Law 8 and continuity
                         counts under the drawn reading, the mirrored frame
                         and the inverted flags
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "frame_505"
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from generate_random_visual_checks import RENDER_FLAGS  # noqa: E402
from src.animate_statsbomb_360_match import (  # noqa: E402
    DirectionInference, apply_display_plan_to_match, apply_orientation_correction_to_match, build_timing_plan,
    compute_orientation_results, create_landscape_figure, determine_team_style, display_orientation_results,
    frame_review_corrections, get_visual_theme, infer_broadcast_orientation_for_match, load_and_join_match,
    load_json_records, match_team_ids, orientation_provenance_text, render_event_frame, resolve_display_plan,
)
from src.orientation_correction import apply_correction  # noqa: E402
from src.team_identity import resolve_match_team_identity, unambiguous_pairs  # noqa: E402

MATCH_ID = 3930164


def law8(points, flags, actor):
    """(Slovakia players beyond halfway, Belgium players in Slovakia's half) in the kicker's own frame."""
    own = sum(1 for (x, _), f, a in zip(points, flags, actor) if f and not a and x > 62)
    opp = sum(1 for (x, _), f, a in zip(points, flags, actor) if not f and not a and x < 58)
    return own, opp


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_and_join_match(MATCH_ID)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(MATCH_ID, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan, frame_reviews=frame_review_corrections())
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    pair = tuple(names[t] for t in match_team_ids(raw))
    ids = resolve_match_team_identity([(j.event, j.frame_360) for j in raw.usable_frames], [r.orientation for r in used],
                                      results, pair, load_json_records(raw.event_path, "event"))
    k = next(i for i, j in enumerate(raw.usable_frames) if j.event["timestamp"] == "00:00:00.505")
    distinct = [k]
    for i in range(k + 1, len(raw.usable_frames)):
        if raw.usable_frames[i].frame_360["freeze_frame"] != raw.usable_frames[distinct[-1]].frame_360["freeze_frame"]:
            distinct.append(i)
        if len(distinct) == 3:
            break

    record = raw.usable_frames[k]
    own = apply_correction(record.frame_360, used[k].orientation)["freeze_frame"]
    points = [p["location"] for p in own]
    flags = [p["teammate"] for p in own]
    actor = [p.get("actor") is True for p in own]
    mirrored = [(120 - x, 80 - y) for x, y in points]
    nxt = apply_correction(raw.usable_frames[distinct[1]].frame_360, used[distinct[1]].orientation)["freeze_frame"]
    q = np.array([p["location"] for p in nxt], float)
    qflags = [p["teammate"] for p in nxt]

    def continuity(pts, f):
        pairs = unambiguous_pairs(np.array(pts, float), q)
        return {"unambiguous_pairs": len(pairs), "flag_disagreements": sum(f[i] != qflags[j] for i, j in pairs)}

    facts = {
        "event": {k_: record.event[k_] for k_ in ("id", "index", "timestamp", "location")} | {
            "type": record.event["type"]["name"], "pass_type": record.event["pass"]["type"]["name"],
            "team": record.event["team"]["name"], "player": record.event["player"]["name"],
            "recipient": record.event["pass"]["recipient"]["name"], "end_location": record.event["pass"]["end_location"]},
        "previous_events": "Starting XI x2, Half Start x2 (no 360 record before it in the period)",
        "points": len(points), "teammate_true": sum(flags), "keepers_flagged": sum(p.get("keeper") is True for p in own),
        "problem_a": results[k].orientation, "resolved_orientation": used[k].orientation, "route": used[k].resolved_by,
        "status_now": sorted(set(ids[record.event["id"]].statuses)),
        "law8_as_drawn (slovakia_beyond_halfway, belgium_in_slovakia_half)": law8(points, flags, actor),
        "law8_if_frame_mirrored": law8(mirrored, flags, actor),
        "law8_if_flags_inverted": law8(points, [not f for f in flags], actor),
        "continuity_with_next_distinct_snapshot_as_drawn": continuity(points, flags),
        "continuity_if_flags_inverted": continuity(points, [not f for f in flags]),
        "continuity_if_frame_mirrored": continuity(mirrored, flags),
        "next_snapshot": raw.usable_frames[distinct[1]].event["timestamp"],
    }
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")

    theme = get_visual_theme("dark")
    style = determine_team_style(raw.usable_frames, "#4aa8ff", "#ff5a5f")
    paths = []
    for n, i in enumerate(distinct):
        joined = display.usable_frames[i]
        figure, ax = create_landscape_figure(theme)
        period = joined.event["period"]
        directions = {names[t]: d for t, d in (plan.attacking_directions(period) or {}).items()}
        timing = build_timing_plan([joined], "fixed", 250, 1.0, 80, 1500)[0]
        render_event_frame(ax, MATCH_ID, joined, 1, 1, timing, "fixed", "camera-overlay", style,
                           DirectionInference(directions, "", orientation_provenance_text(plan, period)),
                           theme, False, True, None, None, frame_identities=ids)
        note = "DIAGNOSTIC: frame 505 (rings = wrong half under Law 8)" if n == 0 else f"DIAGNOSTIC: next distinct snapshot {n}"
        if n == 0:
            drawn = joined.frame_360["freeze_frame"]
            for p, (x, _), f, a in zip(drawn, points, flags, actor):
                if not a and ((f and x > 62) or (not f and x < 58)):
                    ax.scatter([p["location"][0]], [p["location"][1]], s=700, facecolors="none", edgecolors="white", linewidths=2.2, zorder=12)
        ax.text(1.0, 1.055, note, transform=ax.transAxes, ha="right", va="bottom", color="#ffd166",
                fontsize=12, fontweight="bold", clip_on=False)
        path = OUT / f"_{n}.png"
        figure.savefig(path, dpi=55, facecolor=theme.figure_background)
        plt.close(figure)
        paths.append(path)
    images = [Image.open(p).convert("RGB") for p in paths]
    w, h = images[0].size
    sheet = Image.new("RGB", (w, h * len(images)), "black")
    for n, image in enumerate(images):
        sheet.paste(image, (0, n * h))
    sheet.save(OUT / "panels.png")
    for p in paths:
        p.unlink()
    command = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), str(PROJECT_ROOT / "src" / "animate_statsbomb_360_match.py"),
               "--match-id", str(MATCH_ID), "--period", "1", "--start-second", "0", "--end-second", "8", *RENDER_FLAGS,
               "--theme", "dark", "--output", str(OUT / "context.gif"), "--event-frame-manifest", str(OUT / "context_frames.csv")]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:])
    print(json.dumps(facts, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
