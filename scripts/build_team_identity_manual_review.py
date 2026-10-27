"""Visual-review package for suspicious team-colour frames (investigation only).

Selection (reproducible, --seed, default 20261024) from the STRONG cases of
scripts/audit_team_identity_continuity.py (colour contradicts BOTH
neighbouring snapshots), excluding frame 505 (3930164 P1 00:00.505):

  stratum "full inversion"        >= 90% of paired players contradict both
                                  neighbours                         -> 2 cases
  stratum "mixed, 4 players"      min contradictions vs a neighbour == 4 -> 1
  stratum "mixed, 5-7 players"                                       -> 1
  stratum "mixed, >= 8 players"                                      -> 1

Within a stratum the candidates are shuffled with the seed and the first one
is taken whose match is not yet used and whose competition is not yet used
(competition is relaxed if no such candidate exists). If the final set lacks
period 1 or period 2, the last mixed pick is redrawn under that requirement.

Per case (case_NN/):
  context.gif     production CLI, mandatory flags, from 8 s before the
                  previous neighbour to 8 s after the next neighbour
  comparison.png  previous | suspicious | next snapshot, plus the suspicious
                  snapshot with rings: white = colour contradicts BOTH
                  neighbours' paired player, orange = contradicts one
  points.csv      per point: drawn team, paired teams before/after, distances

Nothing is recoloured; colours are exactly what the renderer draws.

Usage:
    .venv/bin/python scripts/build_team_identity_manual_review.py [--seed N]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
    format_match_clock,
    frame_review_corrections,
    get_visual_theme,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    match_team_ids,
    orientation_provenance_text,
    refine_shared_sources_by_continuity,
    render_event_frame,
    resolve_display_plan,
    timestamp_seconds,
)
from src.team_identity import TEAM_IDENTITY_AMBIGUOUS, resolve_absolute_team_for_point, snapshot_key  # noqa: E402

AUDIT = PROJECT_ROOT / "results" / "visualization_final_validation" / "team_identity_investigation" / "colour_inconsistent_frames.csv"
MATCHES = PROJECT_ROOT / "results" / "physical_orientation" / "final_orientation_audit" / "per_match_audit.json"
OUT = PROJECT_ROOT / "results" / "visualization_final_validation" / "team_identity_manual_review"
EXCLUDED = {("3930164", "00:00:00.505")}
MAX_PAIR_DISTANCE = 3.0
CONTEXT_SECONDS = 8.0


def ratio(text: str) -> tuple[int, int]:
    mismatched, paired = text.split("/")
    return int(mismatched), int(paired)


def stratum(row: dict) -> str:
    before, after = ratio(row["vs_previous"]), ratio(row["vs_next"])
    if min(before[0] / before[1], after[0] / after[1]) >= 0.9:
        return "full inversion"
    low = min(before[0], after[0])
    return "mixed, 4 players" if low == 4 else "mixed, 5-7 players" if low <= 7 else "mixed, >= 8 players"


def select(seed: int) -> tuple[list[dict], list[str]]:
    with AUDIT.open(encoding="utf-8") as file:
        strong = [r for r in csv.DictReader(file)
                  if r["neighbours"] == "2" and (r["match_id"], r["timestamp"]) not in EXCLUDED]
    competitions = {str(r["match_id"]): r.get("competition", "") for r in json.loads(MATCHES.read_text(encoding="utf-8"))}
    plan = [("full inversion", 2), ("mixed, 4 players", 1), ("mixed, 5-7 players", 1), ("mixed, >= 8 players", 1)]
    rng = random.Random(seed)
    pools = {name: sorted((r for r in strong if stratum(r) == name), key=lambda r: (r["match_id"], r["timestamp"]))
             for name, _ in plan}
    for name in pools:
        rng.shuffle(pools[name])
    log = [f"seed {seed}; strong candidates {len(strong)}; stratum sizes " +
           ", ".join(f"{n}: {len(p)}" for n, p in pools.items())]
    chosen: list[dict] = []

    def pick(name: str, need_period: str | None = None) -> None:
        used_matches = {c["match_id"] for c in chosen}
        used_comps = {competitions[c["match_id"]] for c in chosen}
        for relax in (False, True):
            for row in pools[name]:
                if row["match_id"] in used_matches or (need_period and row["period"] != need_period):
                    continue
                if not relax and competitions[row["match_id"]] in used_comps:
                    continue
                chosen.append({**row, "stratum": name, "competition": competitions[row["match_id"]],
                               "selection_note": f"first candidate in seeded order of stratum '{name}'"
                               + ("" if not relax else " (competition repeat allowed: no unused competition left)")
                               + (f"; redrawn to include period {need_period}" if need_period else "")})
                return

    for name, count in plan:
        for _ in range(count):
            pick(name)
    periods = {c["period"] for c in chosen}
    for needed in ("1", "2"):
        if needed not in periods:
            removed = chosen.pop()
            log.append(f"set lacked period {needed}: replaced {removed['match_id']} {removed['match_clock']}")
            pick(removed["stratum"], need_period=needed)
            periods = {c["period"] for c in chosen}
    return chosen, log


def drawn_match(match_id: int):
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
    refined, _ = refine_shared_sources_by_continuity(display, shared, raw_by_id, pair)  # exactly what the CLI draws
    style = determine_team_style(raw.usable_frames, "#4aa8ff", "#ff5a5f")
    return raw, plan, display, refined, raw_by_id, names, pair, style


def neighbours(raw, display, index):
    """Nearest DISTINCT snapshot before/after in the same period (as in the audit)."""
    key = snapshot_key(raw.usable_frames[index].frame_360)
    period = display.usable_frames[index].event["period"]
    before = next(k for k in range(index - 1, -1, -1)
                  if snapshot_key(raw.usable_frames[k].frame_360) != key and display.usable_frames[k].event["period"] == period)
    after = next(k for k in range(index + 1, len(display.usable_frames))
                 if snapshot_key(raw.usable_frames[k].frame_360) != key and display.usable_frames[k].event["period"] == period)
    return before, after


def teams(display, refined, raw_by_id, pair, k):
    event = display.usable_frames[k].event
    return [(tuple(p["location"]), resolve_absolute_team_for_point(p, event, refined.get(event["id"]), raw_by_id, pair), p)
            for p in display.usable_frames[k].frame_360["freeze_frame"]]


def pairing(a, b):
    cost = [[math.dist(p[0], q[0]) for q in b] for p in a]
    rows, cols = linear_sum_assignment(cost)
    return {r: (c, cost[r][c]) for r, c in zip(rows, cols)}


def panel(match_id, joined, plan, style, refined, raw_by_id, names, note, path, rings=()):
    theme = get_visual_theme("dark")
    figure, ax = create_landscape_figure(theme)
    period = joined.event["period"]
    directions = {names[t]: d for t, d in (plan.attacking_directions(period) or {}).items()}
    timing = build_timing_plan([joined], "fixed", 250, 1.0, 80, 1500)[0]
    render_event_frame(ax, match_id, joined, 1, 1, timing, "fixed", "camera-overlay", style,
                       DirectionInference(directions, "", orientation_provenance_text(plan, period)),
                       theme, False, True, refined, raw_by_id)
    for (x, y), colour in rings:
        ax.scatter([x], [y], s=900, facecolors="none", edgecolors=colour, linewidths=2.6, zorder=12)
    ax.text(1.0, 1.055, note, transform=ax.transAxes, ha="right", va="bottom",
            color="#ffd166", fontsize=12, fontweight="bold", clip_on=False)
    figure.savefig(path, dpi=55, facecolor=theme.figure_background)
    plt.close(figure)


def build_case(number: int, row: dict) -> dict:
    out = OUT / f"case_{number:02d}"
    out.mkdir(parents=True, exist_ok=True)
    match_id = int(row["match_id"])
    raw, plan, display, refined, raw_by_id, names, pair, style = drawn_match(match_id)
    index = next(k for k, j in enumerate(display.usable_frames) if j.event["id"] == row["event_id"])
    before, after = neighbours(raw, display, index)
    s = teams(display, refined, raw_by_id, pair, index)
    b = teams(display, refined, raw_by_id, pair, before)
    a = teams(display, refined, raw_by_id, pair, after)
    pb, pa = pairing(s, b), pairing(s, a)
    points, rings, both = [], [], 0
    for i, (location, team, player) in enumerate(s):
        jb, db = pb.get(i, (None, None))
        ja, da = pa.get(i, (None, None))
        team_b = b[jb][1] if jb is not None and db <= MAX_PAIR_DISTANCE else ""
        team_a = a[ja][1] if ja is not None and da <= MAX_PAIR_DISTANCE else ""
        contra_b = bool(team_b) and TEAM_IDENTITY_AMBIGUOUS not in (team, team_b) and team != team_b
        contra_a = bool(team_a) and TEAM_IDENTITY_AMBIGUOUS not in (team, team_a) and team != team_a
        if contra_b and contra_a:
            rings.append((location, "#ffffff"))
            both += 1
        elif contra_b or contra_a:
            rings.append((location, "#ff9f1c"))
        points.append({
            "x_drawn": round(location[0], 2), "y_drawn": round(location[1], 2),
            "raw_teammate_flag": player.get("teammate"), "actor": player.get("actor"), "keeper": player.get("keeper"),
            "drawn_team": team,
            "paired_team_previous": team_b, "pair_distance_previous_yd": "" if db is None else round(db, 2),
            "paired_team_next": team_a, "pair_distance_next_yd": "" if da is None else round(da, 2),
            "contradicts": "both" if contra_b and contra_a else "previous" if contra_b else "next" if contra_a else "",
        })
    with (out / "points.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    frames = display.usable_frames
    paths = []
    for tag, k, note, marks in (
        ("a", before, f"PREVIOUS {format_match_clock(frames[before].event)} ({frames[before].event['type']['name']})", ()),
        ("b", index, f"SUSPICIOUS {format_match_clock(frames[index].event)} (as drawn)", ()),
        ("c", after, f"NEXT {format_match_clock(frames[after].event)} ({frames[after].event['type']['name']})", ()),
        ("d", index, f"SUSPICIOUS ringed: white = vs both ({both}), orange = vs one", rings),
    ):
        path = out / f"_{tag}.png"
        panel(match_id, frames[k], plan, style, refined, raw_by_id, names, note, path, marks)
        paths.append(path)
    images = [Image.open(p).convert("RGB") for p in paths]
    w, h = images[0].size
    sheet = Image.new("RGB", (w * 2, h * 2), "black")
    for i, image in enumerate(images):
        sheet.paste(image, ((i % 2) * w, (i // 2) * h))
    sheet.save(out / "comparison.png")
    for p in paths:
        p.unlink()

    start = max(0.0, timestamp_seconds(frames[before].event["timestamp"]) - CONTEXT_SECONDS)
    end = timestamp_seconds(frames[after].event["timestamp"]) + CONTEXT_SECONDS
    command = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), str(PROJECT_ROOT / "src" / "animate_statsbomb_360_match.py"),
               "--match-id", str(match_id), "--period", str(row["period"]),
               "--start-second", f"{start:.3f}", "--end-second", f"{end:.3f}", *RENDER_FLAGS,
               "--output", str(out / "context.gif"), "--event-frame-manifest", str(out / "context_frames.csv")]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:])
    gif_frames = int(next(l.split(":")[1] for l in completed.stdout.splitlines() if l.startswith("Selected real event frames")))
    with (out / "context_frames.csv").open(encoding="utf-8") as file:
        manifest = list(csv.DictReader(file))
    suspicious_frame = next((m["frame_number"] for m in manifest if m["event_id"] == row["event_id"]), "")
    event = frames[index].event
    home_away = f"{pair[0]} vs {pair[1]}"
    return {
        "case": f"case_{number:02d}", "match_id": match_id, "competition": row["competition"], "teams": home_away,
        "period": row["period"], "match_clock": row["match_clock"], "timestamp": row["timestamp"],
        "event_id": row["event_id"], "event_type": row["event_type"], "event_team": row["team"],
        "shared_cross_team_snapshot": row["shared_cross_team_snapshot"], "problem_a": row["problem_a"],
        "points_in_snapshot": len(s),
        "suspicious_players_vs_both_neighbours": both,
        "suspicious_players_vs_one_neighbour": sum(1 for p in points if p["contradicts"] in ("previous", "next")),
        "audit_mismatch_vs_previous": row["vs_previous"], "audit_mismatch_vs_next": row["vs_next"],
        "previous_clock": format_match_clock(frames[before].event), "next_clock": format_match_clock(frames[after].event),
        "stratum": row["stratum"], "selection_reason": row["selection_note"],
        "gif_frames": gif_frames, "suspicious_frame_number_in_gif": suspicious_frame,
        "gif_window_period_seconds": f"{start:.3f}-{end:.3f}",
        "possession_team": event.get("possession_team", {}).get("name", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    chosen, log = select(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [build_case(n, row) for n, row in enumerate(chosen, start=1)]
    with (OUT / "review_index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "selection_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    for r in rows:
        print(r["case"], r["match_id"], r["teams"], f"P{r['period']}", r["match_clock"], r["event_type"], r["stratum"],
              "both:", r["suspicious_players_vs_both_neighbours"], "one:", r["suspicious_players_vs_one_neighbour"],
              "gif frames:", r["gif_frames"], "frame#", r["suspicious_frame_number_in_gif"])
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
