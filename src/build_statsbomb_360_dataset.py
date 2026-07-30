import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


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

        frame = frames_by_event_id[current_event_id]
        freeze_frame = frame.get("freeze_frame", [])

        teammates = [
            player
            for player in freeze_frame
            if player.get("teammate") is True
        ]

        opponents = [
            player
            for player in freeze_frame
            if player.get("teammate") is False
        ]

        actor = next(
            (
                player
                for player in freeze_frame
                if player.get("actor") is True
            ),
            None,
        )

        actor_location = (
            actor.get("location", [None, None])
            if actor is not None
            else [None, None]
        )

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
            "actor_x": actor_location[0],
            "actor_y": actor_location[1],
            "visible_players": len(freeze_frame),
            "visible_teammates": len(teammates),
            "visible_opponents": len(opponents),
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
