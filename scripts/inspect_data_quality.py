"""Visual/text inspection tool for StatsBomb 360 data-quality audit findings.

Read-only: never modifies data/raw/, never overwrites an existing file
under data/processed/ or results/ outside its own dedicated output
directory (results/data_quality_inspections/). Reuses the project's own
pitch-drawing, orientation-correction, team-normalization, and
team-identity logic -- it never re-implements coordinate transformations.

Examples
--------
  # Inspect one event by UUID (prints a full report; saves a diagnostic PNG)
  .venv/bin/python scripts/inspect_data_quality.py --match 3773477 --event 49e3ba44-0b26-4a81-8721-dcc7f4f3831e

  # Visualize every out-of-bounds freeze-frame point for a match
  .venv/bin/python scripts/inspect_data_quality.py --match 3764440 --issue out-of-bounds

  # Same, scoped to one event
  .venv/bin/python scripts/inspect_data_quality.py --match 3764440 --issue out-of-bounds --event 1393e9fb-...

  # List/plot every zero-actor and multiple-actor frame (recomputed live from source)
  .venv/bin/python scripts/inspect_data_quality.py --issue actor-anomalies

  # Same, scoped to one match
  .venv/bin/python scripts/inspect_data_quality.py --match 3773477 --issue actor-anomalies

  # Inspect orientation classification for one match (first ROTATED_180 case)
  .venv/bin/python scripts/inspect_data_quality.py --match 3773477 --issue orientation --orientation ROTATED_180

  # Inspect orientation for one specific event
  .venv/bin/python scripts/inspect_data_quality.py --match 3773477 --issue orientation --event <uuid>

  # Diagnose the corrupted match (works even though the JSON cannot be parsed)
  .venv/bin/python scripts/inspect_data_quality.py --match 3845506 --issue corrupted-json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.animate_statsbomb_360_match import (  # noqa: E402
    coordinate_pair,
    create_landscape_figure,
    draw_pitch,
    get_visual_theme,
    point_inside_pitch,
    raw_input_paths,
    set_viewport,
    visible_area_points,
)
from src.build_statsbomb_360_dataset import (  # noqa: E402
    ACTOR_DUPLICATE_TOLERANCE_UNITS,
    MEANINGFUL_EVENT_TYPES,
    resolve_actor_location,
)
from src.build_statsbomb_360_dataset import OUTPUT_DIR as PROCESSED_DIR  # noqa: E402
from src.orientation_correction import (  # noqa: E402
    ORIENTATION_NORMAL,
    ORIENTATION_ROTATED_180,
    ORIENTATION_UNCERTAIN,
    apply_correction,
    classify_orientation,
    mirror_point,
)
from src.team_identity import (  # noqa: E402
    TEAM_IDENTITY_AMBIGUOUS,
    detect_shared_cross_team_snapshots,
    resolve_absolute_team_for_point,
)
from src.team_normalization import (  # noqa: E402
    TEAM_NORMALIZATION_REFERENCE,
    apply_team_normalization_to_frame,
    resolve_reference_team_id,
    team_normalization_label,
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "data_quality_inspections"
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# Known cases from the second-stage audit (docs/data_quality_investigation_2026-10-13.md).
# Used only as a documented fallback for --issue actor-anomalies when a live
# recompute is skipped; the default behavior always recomputes from source
# (see find_actor_anomalies), never relies solely on this list.
KNOWN_ZERO_ACTOR_CASES = [
    (3893793, "e43bc7e0-4353-4193-a104-29897bf67703"),
    (4018356, "dfe42790-bef1-4e78-9632-de2e747af5cc"),
]


# --------------------------------------------------------------------------
# Loading (read-only)
# --------------------------------------------------------------------------


@dataclass
class LoadResult:
    events: list[dict] | None
    frames: list[dict] | None
    events_error: str | None
    frames_error: str | None


def load_match_raw(match_id: int) -> LoadResult:
    """Load one match's raw events + 360 JSON. Never raises on bad JSON --
    returns the parse error message instead, so callers (notably the
    corrupted-json diagnostic) can report it rather than crash.
    """
    events_path, frames_path = raw_input_paths(match_id)
    events: list[dict] | None = None
    frames: list[dict] | None = None
    events_error = None
    frames_error = None

    if not events_path.exists():
        events_error = f"File not found: {events_path}"
    else:
        try:
            with events_path.open("r", encoding="utf-8") as f:
                events = json.load(f)
        except json.JSONDecodeError as error:
            events_error = str(error)

    if not frames_path.exists():
        frames_error = f"File not found: {frames_path}"
    else:
        try:
            with frames_path.open("r", encoding="utf-8") as f:
                frames = json.load(f)
        except json.JSONDecodeError as error:
            frames_error = str(error)

    return LoadResult(events, frames, events_error, frames_error)


def build_lookup(events: list[dict], frames: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    events_by_id = {e["id"]: e for e in events if "id" in e}
    frames_by_event_id = {fr["event_uuid"]: fr for fr in frames if "event_uuid" in fr}
    return events_by_id, frames_by_event_id


def describe_uuid_lookup_failure(event_uuid: str, candidate_ids: list[str], container_desc: str) -> str:
    """Build a clear 'not found' message, with an explicit suggestion when
    the given value looks like a truncated/prefix form of exactly one real
    id -- this tool's own reports print 8-char UUID prefixes (e.g. in
    'event=49e3ba44'), so a user pasting one back in is a likely mistake to
    guard against rather than silently fail or silently guess.
    """
    message = f"Event {event_uuid!r} was not found in {container_desc} ({len(candidate_ids)} events)."
    prefix_matches = [cid for cid in candidate_ids if cid.startswith(event_uuid)]
    if event_uuid not in candidate_ids and 0 < len(prefix_matches) <= 3:
        if len(prefix_matches) == 1:
            message += f"\nThis looks like a truncated UUID. Did you mean the full UUID: {prefix_matches[0]} ?"
        else:
            message += (
                f"\nThis looks like a truncated UUID matching {len(prefix_matches)} candidates "
                f"(ambiguous, not auto-resolved): {prefix_matches}"
            )
    return message


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def write_report(name: str, text: str) -> Path:
    path = ensure_output_dir() / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def write_json(name: str, payload: dict) -> Path:
    path = ensure_output_dir() / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Mode A: inspect one event
# --------------------------------------------------------------------------


def inspect_event(match_id: int, event_uuid: str, theme_name: str) -> int:
    load = load_match_raw(match_id)
    if load.events is None:
        print(f"Could not load events for match {match_id}: {load.events_error}")
        return 1
    if load.frames is None:
        print(
            f"Three-sixty file for match {match_id} could not be parsed: {load.frames_error}\n"
            f"Event-level metadata can still be shown from events.json, but no freeze_frame "
            f"is available. Use --issue corrupted-json to diagnose the three-sixty file."
        )
        frames: list[dict] = []
    else:
        frames = load.frames

    events_by_id, frames_by_event_id = build_lookup(load.events, frames)
    event = events_by_id.get(event_uuid)
    if event is None:
        print(describe_uuid_lookup_failure(event_uuid, list(events_by_id.keys()), f"match {match_id}'s events.json"))
        return 1

    lines = []
    lines.append(f"Match ID:        {match_id}")
    lines.append(f"Event UUID:      {event_uuid}")
    lines.append(f"Event index:     {event.get('index')}")
    lines.append(f"Event type:      {event.get('type', {}).get('name')}")
    lines.append(f"Period:          {event.get('period')}")
    lines.append(f"Timestamp:       {event.get('timestamp')}")
    lines.append(f"Team:            {(event.get('team') or {}).get('name')}")
    lines.append(f"Player:          {(event.get('player') or {}).get('name')}")
    lines.append(f"Event location:  {event.get('location')}")

    frame = frames_by_event_id.get(event_uuid)
    if frame is None:
        lines.append("")
        lines.append(
            "No matching 360 frame found for this event (event_uuid not present in "
            "three-sixty JSON). Reporting this explicitly rather than substituting a "
            "different frame."
        )
        print("\n".join(lines))
        write_report(f"match{match_id}_event_{event_uuid[:8]}", "\n".join(lines))
        return 0

    freeze_frame = frame.get("freeze_frame") or []
    lines.append("")
    lines.append(f"Freeze-frame player count: {len(freeze_frame)}")
    for i, player in enumerate(freeze_frame):
        loc = player.get("location")
        in_bounds = coordinate_pair(loc) is not None and point_inside_pitch(coordinate_pair(loc))
        lines.append(
            f"  [{i:2d}] location={loc}  teammate={player.get('teammate')}  "
            f"actor={player.get('actor')}  keeper={player.get('keeper')}  "
            f"in_bounds={in_bounds}"
        )

    visible_area = frame.get("visible_area")
    lines.append(f"visible_area ({len(visible_area) if isinstance(visible_area, list) else 0} values): {visible_area}")

    # Shared-snapshot status (Problem C), computed on the RAW match.
    shared_groups = detect_shared_cross_team_snapshots(load.events, frames_by_event_id)
    shared_info = shared_groups.get(event_uuid)
    lines.append("")
    if shared_info is None:
        lines.append("Shared snapshot: NO -- this event's raw 360 snapshot is not shared with a different team's event.")
    else:
        is_source = shared_info.source_event_id == event_uuid
        lines.append(
            f"Shared snapshot: YES -- shared with {len(shared_info.group_event_ids) - 1} other event(s): "
            f"{[eid for eid in shared_info.group_event_ids if eid != event_uuid]}"
        )
        lines.append(f"  This event IS the resolved source: {is_source}")
        lines.append(f"  Resolved source event id: {shared_info.source_event_id}")

    # Orientation classification.
    orientation = classify_orientation(event, frame)
    lines.append("")
    lines.append(f"Orientation classification: {orientation.orientation} (confidence={orientation.confidence:.3f})")
    lines.append(f"  resolved_by: {orientation.resolved_by}")

    # Current-pipeline transformations (reusing the exact same functions the
    # dataset builder and renderer use -- never a second implementation).
    reference_team_id = resolve_reference_team_id(load.events)
    label = team_normalization_label(event, reference_team_id)
    corrected_frame = apply_correction(frame, orientation.orientation)
    normalized_frame = apply_team_normalization_to_frame(corrected_frame, event, reference_team_id)
    lines.append("")
    lines.append(f"Team normalization: {label} (reference_team_id={reference_team_id})")
    lines.append(
        f"Orientation correction applied: {orientation.orientation == ORIENTATION_ROTATED_180}"
    )
    lines.append(
        f"Team-normalization mirror applied: {label != TEAM_NORMALIZATION_REFERENCE}"
    )

    # Out-of-bounds check, raw vs. final.
    oob_raw = sum(
        1 for p in freeze_frame
        if coordinate_pair(p.get("location")) is not None and not point_inside_pitch(coordinate_pair(p.get("location")))
    )
    final_ff = normalized_frame.get("freeze_frame") or []
    oob_final = sum(
        1 for p in final_ff
        if coordinate_pair(p.get("location")) is not None and not point_inside_pitch(coordinate_pair(p.get("location")))
    )
    lines.append(f"Out-of-bounds points (raw): {oob_raw} / {len(freeze_frame)}")
    lines.append(f"Out-of-bounds points (after current pipeline): {oob_final} / {len(final_ff)}")

    # Actor resolution (dataset-builder logic).
    actor_x, actor_y, actor_ambiguous = resolve_actor_location(freeze_frame)
    lines.append("")
    lines.append(f"Dataset-builder actor resolution: actor_x={actor_x}, actor_y={actor_y}, actor_ambiguous={actor_ambiguous}")

    report_text = "\n".join(lines)
    print(report_text)

    name = f"match{match_id}_event_{event_uuid[:8]}"
    report_path = write_report(name, report_text)
    render_event_panel(event, frame, orientation, normalized_frame, theme_name, name)
    print(f"\nSaved text report: {report_path}")
    print(f"Saved PNG:         {OUTPUT_DIR / (name + '.png')}")
    return 0


def render_event_panel(event, frame, orientation, normalized_frame, theme_name: str, name: str) -> None:
    theme = get_visual_theme(theme_name)
    figure, (ax_raw, ax_final) = plt.subplots(
        1, 2, figsize=(18, 9), facecolor=theme.figure_background
    )
    for ax, title, source_frame, show_transform_note in (
        (ax_raw, "RAW (as loaded)", frame, None),
        (ax_final, "Current pipeline (orientation + team normalization)", normalized_frame, True),
    ):
        draw_pitch(ax, theme)
        set_viewport(ax)
        freeze_frame = source_frame.get("freeze_frame") or []
        for player in freeze_frame:
            point = coordinate_pair(player.get("location"))
            if point is None:
                continue
            in_bounds = point_inside_pitch(point)
            color = theme.default_team_one if player.get("teammate") else theme.default_team_two
            marker = "D" if player.get("keeper") else "o"
            edge = theme.actor_accent if player.get("actor") else theme.marker_edge
            ax.scatter(
                [point[0]], [point[1]], c=color if in_bounds else theme.ambiguous_fill,
                marker=marker, edgecolors=edge, linewidths=2.2 if player.get("actor") else 1.2,
                s=170 if player.get("actor") else 130, zorder=6,
            )
        polygon = visible_area_points(source_frame.get("visible_area"))
        if polygon:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            ax.plot(xs, ys, color=theme.camera_edge, lw=1.2, zorder=3, alpha=0.7)
        ax.set_title(title, color=theme.primary_text, fontsize=12)
    caption = (
        f"{event.get('type', {}).get('name')} | {event.get('team', {}).get('name')} | "
        f"orientation={orientation.orientation} ({orientation.confidence:.2f})"
    )
    figure.suptitle(caption, color=theme.primary_text, fontsize=13)
    out_path = ensure_output_dir() / f"{name}.png"
    figure.savefig(out_path, dpi=110, facecolor=theme.figure_background)
    plt.close(figure)


# --------------------------------------------------------------------------
# Mode B: out-of-bounds points
# --------------------------------------------------------------------------


def overshoot_magnitude(point: tuple[float, float]) -> float:
    x, y = point
    return max(0.0, -x, x - PITCH_LENGTH, -y, y - PITCH_WIDTH)


def inspect_out_of_bounds(match_id: int, event_uuid: str | None, theme_name: str) -> int:
    load = load_match_raw(match_id)
    if load.events is None or load.frames is None:
        print(f"Could not load match {match_id}: events_error={load.events_error} frames_error={load.frames_error}")
        return 1
    events_by_id, frames_by_event_id = build_lookup(load.events, load.frames)

    target_frames: list[tuple[str, dict]]
    if event_uuid is not None:
        frame = frames_by_event_id.get(event_uuid)
        if frame is None:
            print(describe_uuid_lookup_failure(event_uuid, list(frames_by_event_id.keys()), f"match {match_id}'s three-sixty.json"))
            return 1
        target_frames = [(event_uuid, frame)]
    else:
        target_frames = list(frames_by_event_id.items())

    offenders = []
    for eid, frame in target_frames:
        event = events_by_id.get(eid, {})
        for player in frame.get("freeze_frame") or []:
            point = coordinate_pair(player.get("location"))
            if point is None:
                continue
            if point_inside_pitch(point):
                continue
            offenders.append(
                {
                    "event_uuid": eid,
                    "event_type": event.get("type", {}).get("name"),
                    "point": point,
                    "overshoot": overshoot_magnitude(point),
                    "teammate": player.get("teammate"),
                    "actor": player.get("actor"),
                    "keeper": player.get("keeper"),
                }
            )

    lines = [f"Match {match_id}: {len(offenders)} out-of-bounds point(s) found"
             + (f" in event {event_uuid}" if event_uuid else " across all events")]
    for o in offenders:
        lines.append(
            f"  event={o['event_uuid'][:8]} type={o['event_type']:18s} point={o['point']} "
            f"overshoot={o['overshoot']:.3f} teammate={o['teammate']} actor={o['actor']} keeper={o['keeper']}"
        )
    report_text = "\n".join(lines)
    print(report_text)

    theme = get_visual_theme(theme_name)
    figure = plt.figure(figsize=(16, 10.5), facecolor=theme.figure_background)
    ax = figure.add_axes([0.03, 0.12, 0.94, 0.78])
    draw_pitch(ax, theme)
    set_viewport(ax)

    # In-bounds points (lightly, for spatial context) and out-of-bounds
    # points (highlighted) are drawn with visibly distinct markers.
    for eid, frame in target_frames:
        for player in frame.get("freeze_frame") or []:
            point = coordinate_pair(player.get("location"))
            if point is None or not point_inside_pitch(point):
                continue
            ax.scatter([point[0]], [point[1]], c=theme.secondary_text, s=16, alpha=0.18, zorder=2)

    # Per-point text annotations become unreadable past a few dozen offenders
    # on a whole-match view; past that threshold, markers alone (still
    # color-coded by severity) remain, and exact coordinates are always in
    # the saved text/JSON report -- use --event to zoom into one event.
    annotate = len(offenders) <= 40
    for o in offenders:
        point = o["point"]
        is_extreme = o["overshoot"] > 10.0
        color = "#FF3B30" if is_extreme else "#FFA500"  # red=far outside, orange=touchline overshoot
        ax.scatter(
            [point[0]], [point[1]], c=color, marker="X", s=170, zorder=8,
            edgecolors=theme.primary_text, linewidths=1.0,
        )
        if annotate:
            ax.annotate(
                f"{point[0]:.1f},{point[1]:.1f} (+{o['overshoot']:.1f})",
                point, color=theme.primary_text, fontsize=7, zorder=9,
                xytext=(5, 5), textcoords="offset points",
            )

    ax.set_title(
        f"Match {match_id} -- out-of-bounds freeze-frame points"
        + (f" (event {event_uuid[:8]})" if event_uuid else ""),
        color=theme.primary_text, fontsize=13, pad=10,
    )

    n_extreme = sum(1 for o in offenders if o["overshoot"] > 10.0)
    legend_text = (
        f"{len(offenders)} out-of-bounds point(s) shown ({n_extreme} far outside, >10 units)."
        + ("" if annotate else "  Per-point labels suppressed above 40 points -- see the saved .txt/.json report for exact coordinates, or re-run with --event to zoom into one event.")
        + "\nX orange = touchline overshoot (<=10 units)   X red = far outside pitch (>10 units)   "
          "dim dots = in-bounds points (context)\n"
        + "Dashed/solid box = padded render viewport around the true pitch [0,120]x[0,80]"
    )
    figure.text(0.03, 0.05, legend_text, color=theme.primary_text, fontsize=8.5, va="top")

    name = f"match{match_id}_out_of_bounds" + (f"_{event_uuid[:8]}" if event_uuid else "")
    out_path = ensure_output_dir() / f"{name}.png"
    figure.savefig(out_path, dpi=110, facecolor=theme.figure_background)
    plt.close(figure)

    report_path = write_report(name, report_text)
    write_json(name, {"match_id": match_id, "event_uuid": event_uuid, "offenders": offenders})
    print(f"\nSaved PNG:  {out_path}")
    print(f"Saved text: {report_path}")
    return 0


# --------------------------------------------------------------------------
# Mode C: actor anomalies (zero-actor / multiple-actor frames)
# --------------------------------------------------------------------------


def find_actor_anomalies(match_id: int | None) -> tuple[list[dict], list[dict]]:
    """Recompute zero-actor and multiple-actor cases live from source JSON.

    Scoped to one match if given, otherwise scans every match with 360 data
    (a plain flag-count pass -- no orientation classification required, so
    this is fast even dataset-wide).
    """
    from src.animate_statsbomb_360_match import available_360_match_ids

    match_ids = [match_id] if match_id is not None else available_360_match_ids()
    zero_actor = []
    multi_actor = []
    for mid in match_ids:
        load = load_match_raw(mid)
        if load.events is None or load.frames is None:
            continue
        events_by_id, _ = build_lookup(load.events, load.frames)
        for frame in load.frames:
            ff = frame.get("freeze_frame") or []
            actors = [p for p in ff if isinstance(p, dict) and p.get("actor") is True]
            eid = frame.get("event_uuid")
            event = events_by_id.get(eid)
            if len(actors) == 0:
                zero_actor.append({"match_id": mid, "event_uuid": eid, "event": event})
            elif len(actors) > 1:
                multi_actor.append(
                    {
                        "match_id": mid,
                        "event_uuid": eid,
                        "event": event,
                        "actor_locations": [a.get("location") for a in actors],
                    }
                )
    return zero_actor, multi_actor


def _pairwise_max_distance(points: list[list[float]]) -> float:
    best = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            best = max(best, (dx * dx + dy * dy) ** 0.5)
    return best


def inspect_actor_anomalies(match_id: int | None, theme_name: str) -> int:
    zero_actor, multi_actor = find_actor_anomalies(match_id)

    lines = [f"Zero-actor frames: {len(zero_actor)}", f"Multiple-actor frames: {len(multi_actor)}", ""]
    for case in zero_actor:
        e = case["event"] or {}
        lines.append(
            f"ZERO ACTOR  match={case['match_id']} event={case['event_uuid']}  "
            f"type={e.get('type', {}).get('name')} team={e.get('team', {}).get('name')} "
            f"location={e.get('location')}"
        )
    lines.append("")
    for case in multi_actor:
        e = case["event"] or {}
        max_dist = _pairwise_max_distance([p for p in case["actor_locations"] if p])
        lines.append(
            f"MULTI ACTOR match={case['match_id']} event={case['event_uuid']}  "
            f"type={e.get('type', {}).get('name')} team={e.get('team', {}).get('name')} "
            f"actor_locations={case['actor_locations']} max_pairwise_distance={max_dist:.4f} "
            f"within_tolerance({ACTOR_DUPLICATE_TOLERANCE_UNITS})={max_dist <= ACTOR_DUPLICATE_TOLERANCE_UNITS}"
        )

    report_text = "\n".join(lines)
    print(report_text)

    # Per-case detail + builder cross-check, only computed for cases whose
    # match we can cheaply re-load (always true here, since find_actor_anomalies
    # already loaded them -- but we reload per match to also get the
    # resolve_actor_location()/shared-snapshot answer using the SAME
    # functions the dataset builder uses).
    detail_lines = []
    plotted = []
    theme = get_visual_theme(theme_name)
    cases_to_plot = (zero_actor + multi_actor) if match_id is not None else (zero_actor + multi_actor)[:12]
    by_match: dict[int, list[dict]] = {}
    for case in cases_to_plot:
        by_match.setdefault(case["match_id"], []).append(case)

    for mid, cases in by_match.items():
        load = load_match_raw(mid)
        if load.events is None or load.frames is None:
            continue
        events_by_id, frames_by_event_id = build_lookup(load.events, load.frames)
        shared_groups = detect_shared_cross_team_snapshots(load.events, frames_by_event_id)
        for case in cases:
            eid = case["event_uuid"]
            frame = frames_by_event_id.get(eid)
            if frame is None:
                continue
            freeze_frame = frame.get("freeze_frame") or []
            actor_x, actor_y, actor_ambiguous = resolve_actor_location(freeze_frame)
            shared_info = shared_groups.get(eid)
            detail_lines.append(
                f"case match={mid} event={eid}: builder actor_x={actor_x} actor_y={actor_y} "
                f"actor_ambiguous={actor_ambiguous}; shared={shared_info is not None}"
            )
            plotted.append((mid, eid, frame, freeze_frame, actor_ambiguous))

    print("\n".join(detail_lines))

    # One combined figure: up to 12 small panels, one per case, each showing
    # the FULL freeze_frame with every actor candidate marked distinctly --
    # never arbitrarily picking one to display.
    n = len(plotted)
    if n > 0:
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        figure, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), facecolor=theme.figure_background)
        axes_flat = [axes] if n == 1 else axes.flatten()
        for ax, (mid, eid, frame, freeze_frame, actor_ambiguous) in zip(axes_flat, plotted):
            draw_pitch(ax, theme)
            set_viewport(ax)
            for player in freeze_frame:
                point = coordinate_pair(player.get("location"))
                if point is None:
                    continue
                is_actor = player.get("actor") is True
                color = theme.actor_accent if is_actor else (
                    theme.default_team_one if player.get("teammate") else theme.default_team_two
                )
                ax.scatter(
                    [point[0]], [point[1]], c=color, s=160 if is_actor else 70,
                    marker="*" if is_actor else "o", zorder=7 if is_actor else 5,
                    edgecolors=theme.primary_text if is_actor else theme.marker_edge,
                    linewidths=1.6 if is_actor else 0.8,
                )
            n_actors = sum(1 for p in freeze_frame if p.get("actor") is True)
            status = "AMBIGUOUS" if actor_ambiguous else ("ZERO-ACTOR" if n_actors == 0 else "resolved")
            ax.set_title(f"m{mid} {eid[:8]} ({status})", color=theme.primary_text, fontsize=9)
        for ax in axes_flat[n:]:
            ax.axis("off")
        name = f"actor_anomalies" + (f"_match{match_id}" if match_id else "_all")
        out_path = ensure_output_dir() / f"{name}.png"
        figure.savefig(out_path, dpi=100, facecolor=theme.figure_background)
        plt.close(figure)
        print(f"\nSaved PNG:  {out_path}")
        report_path = write_report(name, report_text + "\n\n" + "\n".join(detail_lines))
        print(f"Saved text: {report_path}")
    return 0


# --------------------------------------------------------------------------
# Mode D: orientation inspection
# --------------------------------------------------------------------------


def inspect_orientation(
    match_id: int, event_uuid: str | None, orientation_filter: str | None,
    period_filter: int | None, event_type_filter: str | None, theme_name: str,
) -> int:
    load = load_match_raw(match_id)
    if load.events is None or load.frames is None:
        print(f"Could not load match {match_id}: events_error={load.events_error} frames_error={load.frames_error}")
        return 1
    events_by_id, frames_by_event_id = build_lookup(load.events, load.frames)

    if event_uuid is not None:
        candidates = [event_uuid] if event_uuid in frames_by_event_id else []
        if not candidates:
            print(describe_uuid_lookup_failure(event_uuid, list(frames_by_event_id.keys()), f"match {match_id}'s three-sixty.json"))
            return 1
    else:
        candidates = []
        for eid, event in events_by_id.items():
            if eid not in frames_by_event_id:
                continue
            if period_filter is not None and event.get("period") != period_filter:
                continue
            if event_type_filter is not None and event.get("type", {}).get("name") != event_type_filter:
                continue
            if orientation_filter is not None:
                result = classify_orientation(event, frames_by_event_id[eid])
                if result.orientation != orientation_filter:
                    continue
            candidates.append(eid)
        if not candidates:
            print("No events match the given --period/--event-type/--orientation filters.")
            return 1
        print(f"{len(candidates)} matching event(s). Showing the first; pass --event to pick a specific one.")
        print("First 10 candidate event UUIDs:")
        for eid in candidates[:10]:
            print(f"  {eid}")

    eid = candidates[0]
    event = events_by_id[eid]
    frame = frames_by_event_id[eid]
    orientation = classify_orientation(event, frame)
    reference_team_id = resolve_reference_team_id(load.events)
    corrected_frame = apply_correction(frame, orientation.orientation)
    normalized_frame = apply_team_normalization_to_frame(corrected_frame, event, reference_team_id)

    lines = [
        f"Match {match_id}, event {eid}",
        f"Type: {event.get('type', {}).get('name')}  Period: {event.get('period')}  Team: {event.get('team', {}).get('name')}",
        f"Orientation: {orientation.orientation} (confidence={orientation.confidence:.3f}, resolved_by={orientation.resolved_by})",
        f"Transformation applied (orientation mirror): {orientation.orientation == ORIENTATION_ROTATED_180}",
    ]
    if orientation.orientation == ORIENTATION_UNCERTAIN:
        lines.append(
            "UNCERTAIN: evidence is insufficient or contradictory. No transformation is "
            "applied -- raw coordinates are shown in BOTH panels below. This is NOT a "
            "guess presented as ground truth; it is the documented 'never guess' policy."
        )
    report_text = "\n".join(lines)
    print(report_text)

    name = f"match{match_id}_orientation_{eid[:8]}"
    render_event_panel(event, frame, orientation, normalized_frame, theme_name, name)
    report_path = write_report(name, report_text)
    print(f"\nSaved PNG:  {OUTPUT_DIR / (name + '.png')}")
    print(f"Saved text: {report_path}")
    return 0


# --------------------------------------------------------------------------
# Mode E: corrupted-json diagnostic (works without a valid parse)
# --------------------------------------------------------------------------


def find_nul_runs(data: bytes) -> list[tuple[int, int]]:
    runs = []
    i = 0
    n = len(data)
    while i < n:
        if data[i] == 0:
            start = i
            while i < n and data[i] == 0:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1
    return runs


def extract_top_level_objects(data: bytes) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Scan a `[ {...}, {...}, ... ]` byte stream for top-level object byte
    ranges, string-literal-aware (so braces inside string values never
    affect depth). Returns (candidate_ranges, failed_ranges_undetermined).

    This never guesses content -- it only locates syntactic boundaries. Any
    object whose byte range fails `json.loads` on its own is reported as
    unrecoverable, never repaired or invented.
    """
    ranges: list[tuple[int, int]] = []
    depth = 0
    in_string = False
    escape = False
    obj_start = None
    i = 0
    n = len(data)
    try:
        while i < n:
            b = data[i]
            if in_string:
                if escape:
                    escape = False
                elif b == 0x5C:  # backslash
                    escape = True
                elif b == 0x22:  # double quote
                    in_string = False
            else:
                if b == 0x22:
                    in_string = True
                elif b == 0x7B:  # {
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif b == 0x7D:  # }
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        ranges.append((obj_start, i + 1))
                        obj_start = None
                    if depth < 0:
                        break
            i += 1
    except Exception:
        pass
    return ranges, []


def inspect_corrupted_json(match_id: int, theme_name: str) -> int:
    events_path, frames_path = raw_input_paths(match_id)

    lines = [f"Match {match_id} -- corrupted-json diagnostic", "=" * 60]

    # Sibling events.json check -- never touched, read-only.
    if events_path.exists():
        try:
            with events_path.open(encoding="utf-8") as f:
                events = json.load(f)
            lines.append(f"events.json: OK, {len(events)} records ({events_path})")
        except json.JSONDecodeError as error:
            lines.append(f"events.json: FAILED to parse: {error} ({events_path})")
    else:
        lines.append(f"events.json: not found at {events_path}")

    lines.append("")
    if not frames_path.exists():
        lines.append(f"three-sixty.json: not found at {frames_path}")
        print("\n".join(lines))
        write_report(f"match{match_id}_corrupted_json", "\n".join(lines))
        return 1

    raw_bytes = frames_path.read_bytes()
    lines.append(f"three-sixty.json: {frames_path}")
    lines.append(f"File size: {len(raw_bytes)} bytes")

    try:
        json.loads(raw_bytes.decode("utf-8"))
        lines.append("JSON parses successfully -- no corruption detected (file may have been fixed upstream).")
        print("\n".join(lines))
        write_report(f"match{match_id}_corrupted_json", "\n".join(lines))
        return 0
    except json.JSONDecodeError as error:
        lines.append(f"JSONDecodeError: {error}")
        error_offset = error.pos

    nul_runs = find_nul_runs(raw_bytes)
    lines.append(f"\nNUL-byte (0x00) runs found: {len(nul_runs)}")
    for start, length in nul_runs:
        lines.append(f"  offset={start} length={length} (ends at offset {start + length})")

    if nul_runs:
        start, length = nul_runs[0]
        context_before = raw_bytes[max(0, start - 60):start]
        context_after = raw_bytes[start + length: start + length + 60]
        lines.append("")
        lines.append("Context immediately before the first corrupted run:")
        lines.append(f"  text:  {context_before.decode('utf-8', errors='replace')!r}")
        lines.append(f"  hex:   {context_before.hex(' ')}")
        lines.append("Context immediately after the first corrupted run:")
        lines.append(f"  text:  {context_after.decode('utf-8', errors='replace')!r}")
        lines.append(f"  hex:   {context_after.hex(' ')}")

    lines.append(f"\nParser's reported error offset: {error_offset} ({raw_bytes[max(0,error_offset-30):error_offset+30]!r})")

    # Safe, non-guessing partial-record recovery attempt.
    lines.append("\nAttempting safe top-level-record extraction (string-literal-aware brace matching)...")
    ranges, _ = extract_top_level_objects(raw_bytes)
    recovered = 0
    failed = 0
    recovered_records = []
    for start, end in ranges:
        chunk = raw_bytes[start:end]
        try:
            record = json.loads(chunk.decode("utf-8"))
            recovered += 1
            recovered_records.append(record)
        except Exception:
            failed += 1
    lines.append(f"  Candidate top-level objects located: {len(ranges)}")
    lines.append(f"  Successfully parsed independently: {recovered}")
    lines.append(f"  Failed to parse (contain or follow the corrupted region): {failed}")

    lines.append(
        "\nWhy a complete animation cannot be generated: the corrupted 16KB region "
        "overwrites real freeze_frame content inside at least one 360 record with NUL "
        "bytes. That content is permanently lost -- there is no way to reconstruct the "
        "original player positions without inventing data, which this project's policy "
        "explicitly forbids. The surrounding records (those that still parse "
        "independently) are NOT guaranteed to be complete or correctly ordered relative "
        "to the lost one(s), so any output built from them must be clearly labeled "
        "partial and potentially incomplete."
    )

    report_text = "\n".join(lines)
    print(report_text)
    report_path = write_report(f"match{match_id}_corrupted_json", report_text)
    print(f"\nSaved text report: {report_path}")

    if recovered_records:
        partial_path = ensure_output_dir() / f"match{match_id}_PARTIAL_recovered_three_sixty.json"
        partial_path.write_text(
            json.dumps(
                {
                    "WARNING": "PARTIAL AND POTENTIALLY INCOMPLETE -- recovered from a corrupted "
                               "source file by independently re-parsing syntactically complete "
                               "top-level records only. Records overlapping the corrupted region "
                               "are permanently missing, not reconstructed.",
                    "source_file": str(frames_path),
                    "records_recovered": recovered,
                    "records_failed": failed,
                    "data": recovered_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved PARTIAL recovered data (clearly labeled, NOT a repaired source file): {partial_path}")
    else:
        print("Safe partial recovery did not yield any independently-parseable records; none saved.")

    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--match", type=int, default=None, help="StatsBomb match id.")
    parser.add_argument("--event", type=str, default=None, help="Event UUID to inspect.")
    parser.add_argument(
        "--issue",
        choices=["out-of-bounds", "actor-anomalies", "orientation", "corrupted-json"],
        default=None,
        help="Issue category to inspect. Omit together with --event to just inspect one event.",
    )
    parser.add_argument(
        "--orientation", choices=[ORIENTATION_NORMAL, ORIENTATION_ROTATED_180, ORIENTATION_UNCERTAIN],
        default=None, help="Filter orientation candidates (--issue orientation only).",
    )
    parser.add_argument("--period", type=int, default=None, help="Filter by period (--issue orientation only).")
    parser.add_argument("--event-type", type=str, default=None, help="Filter by event type name (--issue orientation only).")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark", help="Visual theme (matches the main animation tool).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.issue == "corrupted-json":
        if args.match is None:
            print("--issue corrupted-json requires --match.")
            return 1
        return inspect_corrupted_json(args.match, args.theme)

    if args.issue == "actor-anomalies":
        return inspect_actor_anomalies(args.match, args.theme)

    if args.issue == "out-of-bounds":
        if args.match is None:
            print("--issue out-of-bounds requires --match.")
            return 1
        return inspect_out_of_bounds(args.match, args.event, args.theme)

    if args.issue == "orientation":
        if args.match is None:
            print("--issue orientation requires --match.")
            return 1
        return inspect_orientation(
            args.match, args.event, args.orientation, args.period, args.event_type, args.theme
        )

    if args.event is not None:
        if args.match is None:
            print("--event requires --match.")
            return 1
        return inspect_event(args.match, args.event, args.theme)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
