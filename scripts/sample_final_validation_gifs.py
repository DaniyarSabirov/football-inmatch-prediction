"""Reproducible random GIF sample for the final manual visualization validation.

Five slots fix the COVERAGE, randomness fixes everything else:

  slot 1  period 1, early third     slot 4  period 2, late third
  slot 2  period 1, late third      slot 5  extra time (period 3 or 4), any third
  slot 3  period 2, middle third

For each slot, a match is drawn uniformly (seeded) from the eligible matches
of the final orientation audit (period present, >= --minimum-snapshots
usable snapshots), excluding every match already shown in an earlier
manual-review sample or evidence set, and distinct across slots. Then a
duration in [120, 240] s and a start uniformly inside the slot's third.
A draw is REJECTED (and logged with the reason) when the interval has fewer
than --minimum-frames usable snapshots or almost no ball movement (total
event-location path < --minimum-ball-travel yards, or x-range < 20 yards);
the slot then redraws. Nothing is hand-picked.

Each GIF is rendered by the production CLI with the mandatory flags
(--apply-orientation-correction --apply-team-normalization
--apply-team-identity-resolution --view camera-overlay --timing-mode
real-time; --display-orientation camera is the default and stated
explicitly). Automatic checks per GIF: frame count, polygon and player
pixels in sampled frames, data-level camera side / flips / UNCERTAIN frames
(validate_data from generate_random_visual_checks.py), duplicate snapshots
drawn identically with identical team colours.

Usage:
    .venv/bin/python scripts/sample_final_validation_gifs.py [--seed N] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import secrets
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_random_visual_checks import RENDER_FLAGS, validate_data, validate_gif  # noqa: E402

from src.animate_statsbomb_360_match import (  # noqa: E402
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    compute_orientation_results,
    compute_shared_group_info,
    display_orientation_results,
    format_match_clock,
    get_visual_theme,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    match_team_ids,
    resolve_display_plan,
    timestamp_seconds,
)
from src.team_identity import resolve_absolute_team_for_point, snapshot_key  # noqa: E402

AUDIT = PROJECT_ROOT / "results" / "physical_orientation" / "final_orientation_audit" / "per_match_audit.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "visualization_final_validation"
SLOTS = [
    ("P1 early", (1,), 0),
    ("P1 late", (1,), 2),
    ("P2 middle", (2,), 1),
    ("P2 late", (2,), 2),
    ("extra time", (3, 4), None),
]
MAX_DRAWS_PER_SLOT = 40

# Matches the reviewer has already seen in earlier samples or evidence sets.
ALREADY_SHOWN = {
    3788743, 3930160, 3835342, 3901833, 3869219, 3869220, 4018357,
    3773466, 3773526, 3773552, 3773631, 3773587, 3906390, 3902239,
}


def previously_sampled() -> set[int]:
    ids = set(ALREADY_SHOWN)
    for path in PROJECT_ROOT.glob("results/physical_orientation/*/random_match_*_manifest.csv"):
        with path.open(encoding="utf-8") as file:
            ids.update(int(row["match_id"]) for row in csv.DictReader(file))
    for path in PROJECT_ROOT.glob("results/physical_orientation/random_visual_checks_round*/manifest.csv"):
        with path.open(encoding="utf-8") as file:
            ids.update(int(row["match_id"]) for row in csv.DictReader(file))
    return ids


def ball_travel(frames) -> tuple[float, float]:
    locations = [j.event["location"] for j in frames if j.event.get("location")]
    path = sum(math.dist(a, b) for a, b in zip(locations, locations[1:]))
    x_range = max(x for x, _ in locations) - min(x for x, _ in locations) if locations else 0.0
    return path, x_range


def draw(seed: int, args, rows: dict[int, dict], excluded: set[int]):
    rng = random.Random(seed)
    chosen, rejections, used = [], [], set(excluded)
    for slot_number, (label, periods, third) in enumerate(SLOTS, start=1):
        pool = sorted(
            m for m, r in rows.items()
            if r["category"] != "excluded_insufficient_data"
            and (r.get("usable_snapshots") or 0) >= args.minimum_snapshots
            and any(str(p) in map(str, r["periods"]) or p in r["periods"] for p in periods)
        )
        for attempt in range(1, MAX_DRAWS_PER_SLOT + 1):
            candidates = [m for m in pool if m not in used]
            if not candidates:
                rejections.append({"slot": slot_number, "slot_label": label, "attempt": attempt, "match_id": "",
                                   "reason": "no unused eligible match left for this slot"})
                break
            match_id = rng.choice(candidates)
            available = [p for p in periods if p in [int(x) for x in rows[match_id]["periods"]]]
            period = rng.choice(available)
            span_third = third if third is not None else rng.randrange(3)
            duration = rng.uniform(120.0, 240.0)
            match = load_and_join_match(match_id)
            frames = [j for j in match.usable_frames if j.event.get("period") == period]
            first, last = timestamp_seconds(frames[0].event["timestamp"]), timestamp_seconds(frames[-1].event["timestamp"])
            low = first + (last - first) * span_third / 3
            high = first + (last - first) * (span_third + 1) / 3 - duration
            start = rng.uniform(low, high) if high > low else low
            end = start + duration
            selected = [j for j in frames if start <= timestamp_seconds(j.event["timestamp"]) < end]
            travel, x_range = ball_travel(selected)
            base = {"slot": slot_number, "slot_label": label, "attempt": attempt, "match_id": match_id, "period": period,
                    "start_period_seconds": round(start, 3), "end_period_seconds": round(end, 3),
                    "frames": len(selected), "ball_travel_yards": round(travel, 1), "x_range_yards": round(x_range, 1)}
            if len(selected) < args.minimum_frames:
                rejections.append({**base, "reason": f"only {len(selected)} usable 360 frames (< {args.minimum_frames})"})
                continue
            if travel < args.minimum_ball_travel or x_range < 20:
                rejections.append({**base, "reason": f"almost no ball movement (path {travel:.0f} yd, x-range {x_range:.0f} yd)"})
                continue
            used.add(match_id)
            chosen.append({**base, "third": ["early", "middle", "late"][span_third],
                           "start_clock": format_match_clock(selected[0].event),
                           "end_clock": format_match_clock(selected[-1].event)})
            break
    return chosen, rejections


def duplicate_and_colour_check(match_id: int, period: int, start: float, end: float) -> dict:
    """Copies of one snapshot inside the interval: same drawn points AND same team colour per point."""
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, _, _ = resolve_display_plan(match_id, raw, inference)
    used, _ = display_orientation_results(raw, results, inference, plan)
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    shared = compute_shared_group_info(raw)
    raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    pair = tuple(names[t] for t in match_team_ids(raw))
    groups: dict = {}
    for index, joined in enumerate(raw.usable_frames):
        t = timestamp_seconds(joined.event["timestamp"])
        if joined.event.get("period") == period and start <= t < end:
            groups.setdefault(snapshot_key(joined.frame_360), []).append(index)
    copies = geometry_mismatch = colour_mismatch = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        drawn = []
        for m in members:
            points = []
            for player in display.usable_frames[m].frame_360["freeze_frame"]:
                team = resolve_absolute_team_for_point(
                    player, raw.usable_frames[m].event, shared.get(raw.usable_frames[m].event["id"]), raw_by_id, pair)
                points.append((tuple(round(c, 6) for c in player["location"]), team))
            drawn.append(sorted(points))
        copies += len(members) - 1
        geometry_mismatch += sum(sorted(p for p, _ in d) != sorted(p for p, _ in drawn[0]) for d in drawn[1:])
        colour_mismatch += sum(d != drawn[0] for d in drawn[1:])
    return {"duplicate_copies": copies, "duplicate_geometry_mismatches": geometry_mismatch,
            "duplicate_colour_mismatches": colour_mismatch}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="select and log only; no rendering")
    parser.add_argument("--minimum-snapshots", type=int, default=1500)
    parser.add_argument("--minimum-frames", type=int, default=40)
    parser.add_argument("--minimum-ball-travel", type=float, default=150.0)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else secrets.randbits(32)
    rows = {r["match_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))}
    excluded = previously_sampled()
    chosen, rejections = draw(seed, args, rows, excluded)
    print(json.dumps({"seed": seed, "excluded_previously_shown": len(excluded),
                      "chosen": [(c["match_id"], c["period"], c["start_clock"], c["end_clock"]) for c in chosen],
                      "rejections": len(rejections)}, indent=1))
    if args.dry_run:
        return 0

    out = OUTPUT_DIR / "random_gifs"
    out.mkdir(parents=True, exist_ok=True)
    theme = get_visual_theme("dark")
    manifest = []
    for number, c in enumerate(chosen, start=1):
        gif = out / f"random_{number:02d}.gif"
        frames_csv = out / f"random_{number:02d}_frames.csv"
        command = [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"), str(PROJECT_ROOT / "src" / "animate_statsbomb_360_match.py"),
            "--match-id", str(c["match_id"]), "--period", str(c["period"]),
            "--start-second", f"{c['start_period_seconds']}", "--end-second", f"{c['end_period_seconds']}",
            *RENDER_FLAGS, "--output", str(gif), "--event-frame-manifest", str(frames_csv),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            raise RuntimeError(f"random_{number:02d} failed: {completed.stderr[-2000:]}")
        stdout = completed.stdout
        rendered = int(next(line.split(":")[1] for line in stdout.splitlines() if line.startswith("Selected real event frames")))
        warnings = [line for line in completed.stderr.splitlines() if "Warning" in line or "Notice" in line]
        direction = next((line for line in stdout.splitlines() if line.startswith("Attacking-direction inference")), "")
        if "WARNING" in direction:
            warnings.append(direction)
        data = validate_data(c["match_id"], c["period"], c["start_period_seconds"], c["end_period_seconds"])
        pixels = validate_gif(gif, rendered, theme)
        duplicates = duplicate_and_colour_check(c["match_id"], c["period"], c["start_period_seconds"], c["end_period_seconds"])
        row = rows[c["match_id"]]
        flags = set(RENDER_FLAGS)
        manifest.append({
            "gif": str(gif.relative_to(PROJECT_ROOT)),
            "match_id": c["match_id"], "competition": row.get("competition"), "season": row.get("season"),
            "home_team": row.get("home_team"), "away_team": row.get("away_team"),
            "period": c["period"], "slot": c["slot_label"], "third_of_period": c["third"],
            "match_clock_start": c["start_clock"], "match_clock_end": c["end_clock"],
            "period_seconds_start": c["start_period_seconds"], "period_seconds_end": c["end_period_seconds"],
            "interval_minutes": round((c["end_period_seconds"] - c["start_period_seconds"]) / 60, 2),
            "random_seed": seed, "sampling_attempt": c["attempt"],
            "rejected_draws_for_slot": sum(r["slot"] == c["slot"] for r in rejections),
            "frames_360": rendered, "gif_frames": pixels["gif_frames"],
            "ball_travel_yards": c["ball_travel_yards"],
            "required_flags_present": all(f in command for f in (
                "--apply-orientation-correction", "--apply-team-normalization",
                "--apply-team-identity-resolution", "camera-overlay", "real-time")) and bool(flags),
            "camera_layer_active": "Display orientation layer: provenance=" in stdout,
            "orientation_provenance": data["plan_provenance"], "period_source": data["period_source"],
            "team_attacking_left_to_right": data["display_reference_team"],
            "camera_side_agreement": data["camera_agreement"],
            "frames_with_players": data["frames_with_players"], "frames_with_visible_area": data["frames_with_polygon"],
            "polygon_pixels_in_sampled_frames": all(p["polygon_edge_pixels"] > 0 for p in pixels["pixel_checks"]),
            "player_pixels_in_sampled_frames": pixels["players_visible_in_samples"],
            "camera_top_snapshots": data["camera_top"], "suspected_180_flips": data["suspected_180_flips"],
            "uncertain_left_uncorrected": data["uncertain_problem_a"],
            "keeper_side_consistency": f"{data['keeper_ok']}/{data['keeper_total']}",
            "max_actor_ball_distance": round(data["max_actor_distance"], 1),
            **duplicates,
            "warnings": " | ".join(warnings),
            "command": " ".join(command[1:]).replace(str(PROJECT_ROOT) + "/", ""),
        })
        print(f"random_{number:02d}: {c['match_id']} P{c['period']} {c['start_clock']}-{c['end_clock']} frames={rendered}")

    fields = list(manifest[0])
    with (OUTPUT_DIR / "sampling_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    with (OUTPUT_DIR / "sampling_rejections.csv").open("w", newline="", encoding="utf-8") as file:
        names = ["slot", "slot_label", "attempt", "match_id", "period", "start_period_seconds", "end_period_seconds",
                 "frames", "ball_travel_yards", "x_range_yards", "reason"]
        writer = csv.DictWriter(file, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rejections)
    (OUTPUT_DIR / "sampling_seed.json").write_text(json.dumps({
        "seed": seed, "command": f".venv/bin/python scripts/sample_final_validation_gifs.py --seed {seed}",
        "excluded_previously_shown_match_ids": sorted(excluded),
        "minimum_frames": args.minimum_frames, "minimum_ball_travel_yards": args.minimum_ball_travel,
        "minimum_snapshots": args.minimum_snapshots,
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
