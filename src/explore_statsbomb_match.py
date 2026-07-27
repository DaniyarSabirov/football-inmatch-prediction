"""Inspect one StatsBomb event file and export its events as a CSV table."""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from pprint import pprint
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "statsbomb_match_15946.csv"


def find_match_file(raw_data_dir: Path) -> Path:
    """Return the only JSON match file in ``raw_data_dir``.

    Raises:
        FileNotFoundError: If the raw data directory contains no JSON file.
        RuntimeError: If more than one JSON file is present.
    """
    match_files = sorted(raw_data_dir.glob("*.json"))

    if not match_files:
        raise FileNotFoundError(f"No JSON file found in {raw_data_dir}")
    if len(match_files) > 1:
        names = ", ".join(path.name for path in match_files)
        raise RuntimeError(f"Expected one JSON file, found: {names}")

    return match_files[0]


def load_events(match_file: Path) -> list[dict[str, Any]]:
    """Load and validate a StatsBomb event JSON file."""
    try:
        with match_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {match_file}: {error}") from error
    except OSError as error:
        raise OSError(f"Could not read {match_file}: {error}") from error

    if not isinstance(payload, list):
        raise TypeError("Expected the top-level JSON value to be a list of events")
    if not all(isinstance(event, dict) for event in payload):
        raise TypeError("Every event in the JSON list must be an object")

    return payload


def event_type_name(event: dict[str, Any]) -> str | None:
    """Extract an event type name from the StatsBomb nested type object."""
    event_type = event.get("type")
    if isinstance(event_type, dict):
        name = event_type.get("name")
        return name if isinstance(name, str) else None
    return event_type if isinstance(event_type, str) else None


def print_event_summary(match_file: Path, events: list[dict[str, Any]]) -> None:
    """Print the requested JSON-level exploratory information."""
    event_types = sorted(
        {
            name
            for event in events
            if (name := event_type_name(event)) is not None
        }
    )
    top_level_keys = sorted({key for event in events for key in event})

    print(f"Match file name: {match_file.name}")
    print(f"Number of events: {len(events)}")
    print("\nFirst five events:")
    pprint(events[:5], sort_dicts=False)
    print("\nUnique event types:")
    pprint(event_types)
    print("\nTop-level JSON keys:")
    pprint(top_level_keys)


def print_dataframe_summary(dataframe: pd.DataFrame) -> None:
    """Print the requested tabular schema information."""
    print(f"\nDataFrame shape: {dataframe.shape}")
    print("\nColumn names:")
    pprint(dataframe.columns.tolist())
    print("\nData types:")
    print(dataframe.dtypes.to_string())


def main() -> None:
    """Run the StatsBomb match exploration and CSV export."""
    match_file = find_match_file(RAW_DATA_DIR)
    events = load_events(match_file)
    print_event_summary(match_file, events)

    dataframe = pd.DataFrame(events)
    print_dataframe_summary(dataframe)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved processed data to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
