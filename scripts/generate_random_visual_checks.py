"""Generate and validate random visual-check GIFs with the camera-side display orientation.

Selection (all random, from --seed; a fresh seed is drawn and reported when omitted):
  - population: eligible matches in the final orientation audit with at least
    --minimum-snapshots usable 360 snapshots, minus every match used by an
    earlier random sample (read from the earlier sample manifests) and the
    --exclude list;
  - N distinct matches; for each, a period drawn uniformly from its periods
    1-4 with 360 data (shootouts are excluded: there is no 2-3 minute run of
    play); if the draw contains no period 1 or no period 2 at all, the
    periods are redrawn and the redraw is logged;
  - duration uniform in [120, 180] s and a start uniform over the period's
    360-covered span; an interval with fewer than --minimum-frames usable
    snapshots is resampled and the reason logged.

Rendering: the production CLI with --apply-orientation-correction
--apply-team-normalization --apply-team-identity-resolution --view
camera-overlay --timing-mode real-time --display-orientation camera.

Validation (after rendering): every GIF is reopened (frame count vs the
selection), sampled frames are checked for team-colour and polygon pixels,
and the displayed data is re-derived in process with the renderer's own
resolve_display_plan: players present, polygon present, camera side of every
polygon (a LOW_Y reading inside a period would be a 180-degree flip),
ball inside polygon, actor-ball distance, keeper side.

Usage:
    .venv/bin/python scripts/generate_random_visual_checks.py --output-dir results/physical_orientation/random_visual_checks_round3
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

from PIL import Image  # noqa: E402

from src.animate_statsbomb_360_match import (  # noqa: E402
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    compute_orientation_results,
    compute_shared_group_info,
    display_orientation_results,
    frame_review_corrections,
    format_match_clock,
    get_visual_theme,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    timestamp_seconds,
    resolve_display_plan,
)
from src.broadcast_orientation import (  # noqa: E402
    CAMERA_SIDE_HIGH_Y,
    CAMERA_SIDE_LOW_Y,
    _point_in_polygon,
    _polygon_vertices,
    estimate_camera_side,
)
from src.orientation_correction import ORIENTATION_UNCERTAIN  # noqa: E402
from src.physical_orientation_calibration import load_calibration_registry, load_period_reviews  # noqa: E402
from src.team_identity import TEAM_IDENTITY_AMBIGUOUS, resolve_absolute_team_for_point  # noqa: E402

AUDIT = PROJECT_ROOT / "results" / "physical_orientation" / "final_orientation_audit" / "per_match_audit.json"
PREVIOUS_SAMPLE_GLOB = "results/physical_orientation/*/random_match_*_manifest.csv"
RENDER_FLAGS = [
    "--apply-orientation-correction", "--apply-team-normalization", "--apply-team-identity-resolution",
    "--view", "camera-overlay", "--timing-mode", "real-time", "--display-orientation", "camera",
]
MAX_INTERVAL_ATTEMPTS = 25


def previous_sample_match_ids(output_dir: Path) -> set[int]:
    ids = set()
    for path in PROJECT_ROOT.glob(PREVIOUS_SAMPLE_GLOB):
        if path.parent.resolve() == output_dir.resolve():
            continue
        with path.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                ids.add(int(row["match_id"]))
                break
    return ids


def period_seconds(joined) -> float:
    return timestamp_seconds(joined.event.get("timestamp"))


def choose(rng: random.Random, args, rows: dict[int, dict], excluded: set[int], log: list[str]):
    population = sorted(
        m for m, r in rows.items()
        if r["category"] != "excluded_insufficient_data"
        and (r.get("usable_snapshots") or 0) >= args.minimum_snapshots
        and m not in excluded
    )
    log.append(f"population: {len(population)} eligible matches after excluding {len(excluded)} previously used/listed")
    match_ids = rng.sample(population, args.count)

    def regular_periods(match_id):
        return [int(p) for p in rows[match_id]["periods"] if int(p) in (1, 2, 3, 4)]

    for attempt in range(10):
        periods = [rng.choice(regular_periods(m)) for m in match_ids]
        if {1, 2} <= set(periods):
            break
        log.append(f"period redraw {attempt + 1}: draw {periods} lacked a first or second half")
    return list(zip(match_ids, periods))


def pick_interval(rng: random.Random, match, period: int, minimum_frames: int, log: list[str], label: str):
    frames = [j for j in match.usable_frames if j.event.get("period") == period]
    first, last = period_seconds(frames[0]), period_seconds(frames[-1])
    for attempt in range(1, MAX_INTERVAL_ATTEMPTS + 1):
        duration = rng.uniform(120.0, 180.0)
        if last - first <= duration:
            start = first
        else:
            start = rng.uniform(first, last - duration)
        end = start + duration
        selected = [j for j in frames if start <= period_seconds(j) < end]
        if len(selected) >= minimum_frames:
            return round(start, 3), round(end, 3), selected, attempt
        log.append(f"{label}: resampled interval {start:.1f}-{end:.1f}s (P{period}): only {len(selected)} usable frames")
    raise RuntimeError(f"{label}: no interval with >= {minimum_frames} frames after {MAX_INTERVAL_ATTEMPTS} attempts")


def colour_pixels(image: Image.Image, hex_colour: str, tolerance: int = 40) -> int:
    target = tuple(int(hex_colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    rgb = image.convert("RGB").resize((image.width // 2, image.height // 2))
    return sum(1 for p in rgb.get_flattened_data() if all(abs(p[k] - target[k]) <= tolerance for k in range(3)))


def validate_gif(path: Path, expected_frames: int, theme) -> dict:
    with Image.open(path) as gif:
        n = gif.n_frames
        checks = []
        for index in sorted({0, n // 2, n - 1}):
            gif.seek(index)
            frame = gif.copy()
            checks.append({
                "frame": index,
                "team_one_pixels": colour_pixels(frame, theme.default_team_one),
                "team_two_pixels": colour_pixels(frame, theme.default_team_two),
                "polygon_edge_pixels": colour_pixels(frame, theme.camera_edge, 30),
            })
    return {
        "gif_frames": n,
        "frame_count_matches": n == expected_frames,
        "pixel_checks": checks,
        "players_visible_in_samples": all(c["team_one_pixels"] + c["team_two_pixels"] > 200 for c in checks),
    }


def validate_data(match_id: int, period: int, start: float, end: float) -> dict:
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan, source, messages = resolve_display_plan(match_id, raw, inference)
    results, _ = display_orientation_results(  # exactly what the renderer applies
        raw, results, inference, plan, frame_reviews=frame_review_corrections())
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, results), plan)
    shared = compute_shared_group_info(raw)
    raw_by_id = {j.event["id"]: j.event for j in raw.usable_frames}
    names = tuple(sorted({j.event["team"]["name"] for j in raw.usable_frames}))
    transform = plan.transforms[period]
    reference_name = next(j.event["team"]["name"] for j in raw.usable_frames if j.event["team"]["id"] == transform.reference_team_id)

    previous_location = None
    stats = dict(suspected_180_flips=0, frames=0, frames_with_players=0, frames_with_polygon=0, camera_bottom=0, camera_top=0,
                 uncertain_problem_a=0, ball_in_polygon=0, keeper_ok=0, keeper_total=0, max_actor_distance=0.0)
    for shown, raw_joined, result in zip(display.usable_frames, raw.usable_frames, results):
        if shown.event.get("period") != period or not start <= period_seconds(shown) < end:
            continue
        stats["frames"] += 1
        location = shown.event.get("location")
        if location and previous_location:
            # A 180-degree flip of the whole picture moves the ball to (near)
            # the mirror image of where it just was; ordinary play does not.
            mirrored = (120 - previous_location[0], 80 - previous_location[1])
            if math.dist(location, previous_location) > 40 and math.dist(location, mirrored) < 15:
                stats["suspected_180_flips"] += 1
        previous_location = location or previous_location
        players = [p for p in shown.frame_360.get("freeze_frame") or [] if p.get("location")]
        stats["frames_with_players"] += bool(players)
        polygon = _polygon_vertices(shown.frame_360.get("visible_area"))
        if polygon:
            stats["frames_with_polygon"] += 1
            location = shown.event.get("location")
            if location:
                stats["ball_in_polygon"] += _point_in_polygon(tuple(location), polygon)
        if result.orientation == ORIENTATION_UNCERTAIN:
            stats["uncertain_problem_a"] += 1
        else:
            side = estimate_camera_side(shown.frame_360.get("visible_area")).side
            stats["camera_bottom"] += side == CAMERA_SIDE_HIGH_Y
            stats["camera_top"] += side == CAMERA_SIDE_LOW_Y
        for player in players:
            if player.get("actor") and shown.event.get("location"):
                stats["max_actor_distance"] = max(stats["max_actor_distance"], math.dist(player["location"], shown.event["location"]))
            if player.get("keeper"):
                team = resolve_absolute_team_for_point(player, raw_joined.event, shared.get(raw_joined.event["id"]), raw_by_id, names)
                if team != TEAM_IDENTITY_AMBIGUOUS:
                    stats["keeper_total"] += 1
                    stats["keeper_ok"] += (player["location"][0] < 60) == (team == reference_name)
    evidence = inference.period_evidence(period)
    return {
        **stats,
        "plan_provenance": plan.provenance,
        "period_source": transform.source,
        "display_reference_team": reference_name,
        "camera_agreement": round(evidence.agreement, 4) if evidence else None,
        "anchor_source": source,
        "messages": messages,
        "plan": plan.explanation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--minimum-snapshots", type=int, default=1500)
    parser.add_argument("--minimum-frames", type=int, default=40)
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = random.Random(seed)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {r["match_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))}
    log: list[str] = [f"seed: {seed}"]
    excluded = previous_sample_match_ids(output_dir) | set(args.exclude)
    picks = choose(rng, args, rows, excluded, log)
    registry = load_calibration_registry()
    reviews = load_period_reviews()
    theme = get_visual_theme("dark")

    manifest = []
    for number, (match_id, period) in enumerate(picks, start=1):
        label = f"random_match_{number:02d}"
        row = rows[match_id]
        match = load_and_join_match(match_id)
        start, end, selected, attempts = pick_interval(rng, match, period, args.minimum_frames, log, label)
        gif = output_dir / f"{label}.gif"
        frames_csv = output_dir / f"{label}_frames.csv"
        command = [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"), str(PROJECT_ROOT / "src" / "animate_statsbomb_360_match.py"),
            "--match-id", str(match_id), "--period", str(period),
            "--start-second", f"{start}", "--end-second", f"{end}",
            *RENDER_FLAGS, "--output", str(gif), "--event-frame-manifest", str(frames_csv),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            raise RuntimeError(f"{label} render failed: {completed.stderr[-2000:]}")
        stdout = completed.stdout
        rendered = int(next(line.split(":")[1] for line in stdout.splitlines() if line.startswith("Selected real event frames")))
        direction_line = next((line for line in stdout.splitlines() if line.startswith("Attacking-direction inference")), "")
        warnings = [line for line in (completed.stderr or "").splitlines() if "Warning" in line or "Notice" in line]
        if "WARNING" in direction_line:
            warnings.append(direction_line)
        if attempts > 1:
            warnings.append(f"interval resampled {attempts - 1}x for sparse coverage")

        data = validate_data(match_id, period, start, end)
        gif_check = validate_gif(gif, rendered, theme)
        review = reviews.get(match_id, {}).get(period)
        verified = match_id in registry and registry[match_id].verification_status == "verified"
        if data["uncertain_problem_a"]:
            warnings.append(
                f"{data['uncertain_problem_a']} snapshot(s) left uncorrected (Problem A UNCERTAIN and camera side disagrees)"
            )
        if data["suspected_180_flips"]:
            warnings.append(f"{data['suspected_180_flips']} suspected 180-degree flip(s) of play")
        if data["camera_top"]:
            warnings.append(
                f"{data['camera_top']} visible_area footprint(s) read as camera-at-top "
                "(alternate camera angle; ball/player continuity checked separately)"
            )
        manifest.append({
            "gif_filename": gif.name,
            "match_id": match_id,
            "competition": row.get("competition"),
            "season": row.get("season"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "period": period,
            "start_period_seconds": start,
            "end_period_seconds": end,
            "start_match_time": format_match_clock(selected[0].event),
            "end_match_time": format_match_clock(selected[-1].event),
            "actual_360_frames": rendered,
            "gif_frames": gif_check["gif_frames"],
            "orientation_decision": f"{data['display_reference_team']} attacks left->right (P{period} source: {data['period_source']})",
            "orientation_provenance": data["plan_provenance"],
            "camera_side_confidence": data["camera_agreement"],
            "independent_broadcast_verification": "yes (registry verified)" if verified else (
                "period review: " + review.evidence_type if review else "no"),
            "frames_with_players": data["frames_with_players"],
            "frames_with_visible_area": data["frames_with_polygon"],
            "camera_bottom_snapshots": data["camera_bottom"],
            "camera_top_snapshots": data["camera_top"],
            "suspected_180_flips": data["suspected_180_flips"],
            "interval_minutes": round((end - start) / 60, 2),
            "ball_in_visible_area": data["ball_in_polygon"],
            "max_actor_ball_distance": round(data["max_actor_distance"], 3),
            "keeper_side_consistency": f"{data['keeper_ok']}/{data['keeper_total']}",
            "gif_frame_count_ok": gif_check["frame_count_matches"],
            "gif_players_visible": gif_check["players_visible_in_samples"],
            # The camera layer reports itself on stdout; legacy mode never does.
            "camera_orientation_layer_active": "Display orientation layer: provenance=" in stdout,
            "uncertain_snapshots_left_uncorrected": data["uncertain_problem_a"],
            "warnings": " | ".join(warnings),
        })
        print(f"{label}: match {match_id} P{period} {manifest[-1]['start_match_time']}-{manifest[-1]['end_match_time']} "
              f"frames={rendered} gif={gif_check['gif_frames']} provenance={data['plan_provenance']} "
              f"camera_top={data['camera_top']} warnings={len(warnings)}")

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (output_dir / "selection_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
