import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

try:
    # Flat sibling import: how this module sees it when run directly as a
    # script (e.g. `python src/build_statsbomb_360_dataset.py`), which puts
    # this file's own directory (src/) at sys.path[0].
    from orientation_correction import apply_correction, classify_orientations_for_match
    from team_normalization import (
        apply_team_normalization_to_frame,
        resolve_reference_team_id,
        team_normalization_label,
    )
    from team_identity import (
        TEAM_IDENTITY_AMBIGUOUS,
        detect_shared_cross_team_snapshots,
        resolve_absolute_team_for_point,
    )
    from match_exclusions import check_not_excluded, load_exclusion_manifest
except ImportError:
    # Package-style import: how this module sees it when run as
    # `python -m src.build_statsbomb_360_dataset` from the project root,
    # which puts the project root (not src/) at sys.path[0].
    from src.orientation_correction import apply_correction, classify_orientations_for_match
    from src.team_normalization import (
        apply_team_normalization_to_frame,
        resolve_reference_team_id,
        team_normalization_label,
    )
    from src.team_identity import (
        TEAM_IDENTITY_AMBIGUOUS,
        detect_shared_cross_team_snapshots,
        resolve_absolute_team_for_point,
    )
    from src.match_exclusions import check_not_excluded, load_exclusion_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATSBOMB_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "statsbomb"
    / "data"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MEANINGFUL_EVENT_TYPES = {
    "Pass",
    "Carry",
    "Dribble",
    "Shot",
    "Miscontrol",
    "Ball Recovery",
    "Clearance",
    "Dispossessed",
}

# A rare raw-data quirk (confirmed in a dedicated audit: 28/1,381,467 frames,
# 0.002%, across 11/426 matches) has TWO freeze_frame points flagged
# actor=True in the same snapshot. In 26/28 of those cases the two points
# are within ~0.22 pitch units of each other (floating-point/encoding noise
# around what is clearly the same physical position); in exactly 2 cases
# (one match, both referencing the same underlying pair of events) they are
# ~6.8 units apart -- genuinely ambiguous. ACTOR_DUPLICATE_TOLERANCE_UNITS
# separates "effectively the same point" (safe to pick either) from
# "genuinely ambiguous" (must not guess), based on that measured gap.
ACTOR_DUPLICATE_TOLERANCE_UNITS = 1.0


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def timestamp_to_seconds(timestamp: str) -> float:
    parsed = datetime.strptime(timestamp, "%H:%M:%S.%f")

    return (
        parsed.hour * 3600
        + parsed.minute * 60
        + parsed.second
        + parsed.microsecond / 1_000_000
    )


def resolve_actor_location(freeze_frame: list[dict]) -> tuple[float | None, float | None, bool]:
    """Return (actor_x, actor_y, ambiguous) for one freeze_frame.

    Zero actor-flagged points -> (None, None, False): legitimately missing
    actor information (the array may genuinely contain no actor=True
    point; see the audit for confirmed real examples), not invented.

    Two or more actor-flagged points that are effectively the same
    position (within ACTOR_DUPLICATE_TOLERANCE_UNITS) are treated as one
    -- the first is used, since they already agree to within floating-point
    noise. Two or more actor-flagged points that meaningfully disagree are
    genuinely ambiguous: returns (None, None, True) rather than arbitrarily
    picking one, per the project's "never guess" policy.
    """
    actor_points = [
        player
        for player in freeze_frame
        if isinstance(player, dict) and player.get("actor") is True
    ]
    if not actor_points:
        return None, None, False

    first_location = actor_points[0].get("location")
    if not isinstance(first_location, list) or len(first_location) != 2:
        return None, None, False

    for other in actor_points[1:]:
        other_location = other.get("location")
        if not isinstance(other_location, list) or len(other_location) != 2:
            continue
        distance = (
            (first_location[0] - other_location[0]) ** 2
            + (first_location[1] - other_location[1]) ** 2
        ) ** 0.5
        if distance > ACTOR_DUPLICATE_TOLERANCE_UNITS:
            return None, None, True

    return first_location[0], first_location[1], False


def _match_team_names(events: list[dict]) -> tuple[str, str]:
    """Return the match's two team names, in first-seen chronological order.

    Mirrors `resolve_reference_team_id`'s own first-seen scan (by id); here
    we need names, not ids, purely to hand `resolve_absolute_team_for_point`
    a (team_one, team_two) pair it can compute "the other team" from.
    """
    names: list[str] = []
    for event in events:
        team = event.get("team")
        name = team.get("name") if isinstance(team, dict) else None
        if isinstance(name, str) and name not in names:
            names.append(name)
        if len(names) == 2:
            break
    if len(names) != 2:
        raise ValueError(f"Expected exactly two team names in events, found {names}.")
    return names[0], names[1]


def find_next_meaningful_event(
    events: list[dict],
    current_position: int,
    current_period: int,
):
    for candidate in events[current_position + 1:]:
        if candidate["period"] != current_period:
            return None

        if candidate["type"]["name"] in MEANINGFUL_EVENT_TYPES:
            return candidate

    return None


def build_match_dataset(
    match_id: int,
    maximum_time_difference: float,
) -> list[dict]:
    """Load one match's raw JSON from disk and build its next-event dataset.

    Thin I/O wrapper around `build_rows_from_match_data`, which contains all
    the actual logic and takes already-loaded data -- kept separate so the
    logic can be unit-tested against small synthetic fixtures without
    touching real files on disk.

    Checks `configs/excluded_matches.json` (see src/match_exclusions.py)
    BEFORE any file I/O, so an explicitly excluded match (e.g. one with a
    corrupted raw file) never reaches the code path that would otherwise
    raise a less informative error -- callers get a clear
    `ExcludedMatchError` naming the documented reason instead.
    """
    check_not_excluded(match_id, load_exclusion_manifest())

    events_path = (
        STATSBOMB_DATA_DIR
        / "events"
        / f"{match_id}.json"
    )

    frames_path = (
        STATSBOMB_DATA_DIR
        / "three-sixty"
        / f"{match_id}.json"
    )

    if not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found: {events_path}"
        )

    if not frames_path.exists():
        raise FileNotFoundError(
            f"Three-sixty file not found: {frames_path}"
        )

    events = load_json(events_path)
    frames = load_json(frames_path)

    frames_by_event_id = {
        frame["event_uuid"]: frame
        for frame in frames
    }

    return build_rows_from_match_data(match_id, events, frames_by_event_id, maximum_time_difference)


def build_rows_from_match_data(
    match_id: int,
    events: list[dict],
    frames_by_event_id: dict[str, dict],
    maximum_time_difference: float,
) -> list[dict]:
    """Build one match's next-event dataset rows from already-loaded data.

    `events` must be StatsBomb's raw, unmodified events list (already in
    `index` order -- verified dataset-wide to match the raw file's own
    array order, so no additional chronological sort is performed here).
    `frames_by_event_id` must be the RAW (pre-orientation-correction) 360
    frames, keyed by `event_uuid`, exactly as loaded from the three-sixty
    JSON file.
    """
    # Classify every matched event/360-frame pair up front, in chronological
    # order, so the temporal-refinement pass (see
    # src/orientation_correction.py) has the full match timeline available as
    # potential neighbors -- not just the subset of MEANINGFUL_EVENT_TYPES
    # rows this dataset ultimately keeps.
    matched_pairs = sorted(
        (
            (event, frames_by_event_id[event["id"]])
            for event in events
            if event["id"] in frames_by_event_id
        ),
        key=lambda pair: (
            pair[0]["period"],
            timestamp_to_seconds(pair[0]["timestamp"]),
            pair[0]["index"],
        ),
    )
    ordered_events = [pair[0] for pair in matched_pairs]
    ordered_frames = [pair[1] for pair in matched_pairs]
    ordered_timestamps = [
        timestamp_to_seconds(event["timestamp"]) for event in ordered_events
    ]
    ordered_periods = [event["period"] for event in ordered_events]
    orientation_by_event_id = {
        event["id"]: result
        for event, result in zip(
            ordered_events,
            classify_orientations_for_match(
                ordered_events, ordered_frames, ordered_timestamps, ordered_periods
            ),
        )
    }

    # Step 2 (Problem B, independent of and applied after Step 1 above): the
    # two teams' raw coordinates use opposite attack-direction conventions
    # (see src/team_normalization.py for the forensic evidence). This is a
    # separate question from Step 1's "is this event internally consistent
    # with its own location?" -- it never overrides orientation_result, it
    # only decides whether this event's (already Step-1-corrected) frame
    # needs mirroring to sit on the same scale as the reference team.
    #
    # Resolved from `ordered_events` (chronological, 360-joined), not the
    # full raw `events` list: early events such as "Starting XI"/"Half
    # Start" have no 360 data and can belong to a different team than the
    # first 360-joined event, which would otherwise make this pick a
    # different reference team than the animation script (which only ever
    # sees 360-joined events) -- the two consumers must agree on one team.
    reference_team_id = resolve_reference_team_id(ordered_events)

    # Step 3 (Problem C, independent of Steps 1-2, resolved on the RAW
    # events/frames -- see src/team_identity.py): StatsBomb sometimes
    # attaches the exact same raw 360 snapshot to two or three different
    # events belonging to DIFFERENT teams at the same instant. Reading
    # `teammate` relative to a borrowing event's own team, instead of the
    # snapshot's true source team, silently flips the teammate/opponent
    # split for that row. Must run before any orientation/normalization
    # mirroring, which would make originally-identical arrays diverge.
    shared_group_by_event_id = detect_shared_cross_team_snapshots(events, frames_by_event_id)
    raw_events_by_id = {event["id"]: event for event in events if "id" in event}
    team_names = _match_team_names(events)

    rows = []

    for current_position, current_event in enumerate(events):
        current_event_id = current_event["id"]
        current_event_type = current_event["type"]["name"]

        if current_event_id not in frames_by_event_id:
            continue

        if current_event_type not in MEANINGFUL_EVENT_TYPES:
            continue

        next_event = find_next_meaningful_event(
            events=events,
            current_position=current_position,
            current_period=current_event["period"],
        )

        if next_event is None:
            continue

        current_time = timestamp_to_seconds(
            current_event["timestamp"]
        )

        next_time = timestamp_to_seconds(
            next_event["timestamp"]
        )

        time_difference = next_time - current_time

        if time_difference < 0:
            continue

        if time_difference > maximum_time_difference:
            continue

        # Some StatsBomb 360 snapshots are recorded reflected through the
        # pitch center relative to the event's own location (see
        # src/orientation_correction.py for the forensic evidence and the
        # geometry-based detection method). Never assume event type alone
        # decides this; each snapshot was classified on its own evidence
        # above, and a correction is never forced when that evidence is weak
        # or contradictory (orientation == UNCERTAIN keeps raw coordinates).
        orientation_result = orientation_by_event_id[current_event_id]
        frame = apply_correction(
            frames_by_event_id[current_event_id], orientation_result.orientation
        )
        frame = apply_team_normalization_to_frame(frame, current_event, reference_team_id)
        freeze_frame = frame.get("freeze_frame", [])
        team_normalization = team_normalization_label(current_event, reference_team_id)

        # Resolve each point's ABSOLUTE team via Problem C instead of the
        # naive "teammate relative to this event's own team" reading, which
        # is only reliable when this event's raw snapshot isn't shared with
        # a different team's event (see the module-level comment above).
        # Coordinates (already orientation/normalization-corrected above)
        # are never touched by this -- only which team each point counts
        # toward. Genuinely ambiguous points (no confident source) are
        # counted separately rather than guessed either way.
        shared_group_info = shared_group_by_event_id.get(current_event_id)
        current_team_name = current_event["team"]["name"]
        teammates = []
        opponents = []
        ambiguous_points = []
        for player in freeze_frame:
            if not isinstance(player, dict):
                continue
            resolved_team = resolve_absolute_team_for_point(
                player, current_event, shared_group_info, raw_events_by_id, team_names
            )
            if resolved_team == TEAM_IDENTITY_AMBIGUOUS:
                ambiguous_points.append(player)
            elif resolved_team == current_team_name:
                teammates.append(player)
            else:
                opponents.append(player)

        actor_x, actor_y, actor_ambiguous = resolve_actor_location(freeze_frame)

        current_team_id = current_event["team"]["id"]
        next_team_id = next_event["team"]["id"]

        same_team = current_team_id == next_team_id

        same_possession = (
            current_event.get("possession")
            == next_event.get("possession")
        )

        rows.append({
            "match_id": match_id,
            "current_event_id": current_event_id,
            "current_event_index": current_event["index"],
            "current_event_type": current_event_type,
            "current_team": current_event["team"]["name"],
            "current_player": current_event.get(
                "player", {}
            ).get("name"),
            "period": current_event["period"],
            "minute": current_event["minute"],
            "timestamp": current_event["timestamp"],
            "actor_x": actor_x,
            "actor_y": actor_y,
            "actor_ambiguous": actor_ambiguous,
            "orientation": orientation_result.orientation,
            "orientation_confidence": round(orientation_result.confidence, 3),
            "team_normalization": team_normalization,
            "visible_players": len(freeze_frame),
            "visible_teammates": len(teammates),
            "visible_opponents": len(opponents),
            "visible_ambiguous": len(ambiguous_points),
            "next_event_id": next_event["id"],
            "next_event_type": next_event["type"]["name"],
            "next_team": next_event["team"]["name"],
            "next_player": next_event.get(
                "player", {}
            ).get("name"),
            "same_team": same_team,
            "same_possession": same_possession,
            "time_to_next_event_seconds": round(
                time_difference,
                3,
            ),
        })

    return rows


def save_dataset(
    rows: list[dict],
    output_path: Path,
):
    if not rows:
        raise ValueError(
            "No dataset rows were created."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a next-event dataset from "
            "StatsBomb event and 360 data."
        )
    )

    parser.add_argument(
        "--match-id",
        type=int,
        required=True,
        help="StatsBomb match ID.",
    )

    parser.add_argument(
        "--max-time",
        type=float,
        default=10.0,
        help=(
            "Maximum allowed number of seconds "
            "between current and next event."
        ),
    )

    args = parser.parse_args()

    rows = build_match_dataset(
        match_id=args.match_id,
        maximum_time_difference=args.max_time,
    )

    output_path = (
        OUTPUT_DIR
        / f"statsbomb_360_match_{args.match_id}_next_events.csv"
    )

    save_dataset(
        rows=rows,
        output_path=output_path,
    )

    print(f"Rows created: {len(rows)}")
    print(f"Output file: {output_path}")

    print("\nFirst ten event sequences:")

    for row in rows[:10]:
        print(
            f'{row["current_event_type"]:<15}'
            f' -> {row["next_event_type"]:<15}'
            f' | same team: {str(row["same_team"]):<5}'
            f' | same possession: '
            f'{str(row["same_possession"]):<5}'
            f' | time: '
            f'{row["time_to_next_event_seconds"]} s'
        )


if __name__ == "__main__":
    main()
