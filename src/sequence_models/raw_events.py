"""Flat table of the raw StatsBomb event stream (read-only on raw data).

One row per raw event of a match, in the raw file's array order. Nothing is
filtered, re-ordered or corrected here: sequence boundaries, vocabularies and
exclusions are decided later, explicitly, in ``tasks``. The table keeps both
the raw array position (``list_position``) and StatsBomb's ``index`` so the
audit can verify that they agree.

Paths come from ``src.utils.paths``; the excluded-match registry is read only
through ``src.match_exclusions`` (one source of truth with the dataset builder).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.match_exclusions import load_exclusion_manifest
from src.utils.paths import RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)

STATSBOMB_DATA_DIR = RAW_DATA_DIR / "statsbomb" / "data"
EVENTS_DIR = STATSBOMB_DATA_DIR / "events"
THREE_SIXTY_DIR = STATSBOMB_DATA_DIR / "three-sixty"
MATCHES_DIR = STATSBOMB_DATA_DIR / "matches"

EVENT_TABLE_COLUMNS = [
    "match_id", "list_position", "index", "event_id", "period", "timestamp", "t_seconds",
    "minute", "second", "type", "team_id", "possession", "possession_team_id", "play_pattern",
    "has_location", "location_x", "location_y", "shot_type", "has_360", "has_freeze_frame",
]


@dataclass(frozen=True)
class MatchMetadata:
    match_id: int
    competition: str
    season: str
    match_date: str
    home_team: str
    away_team: str
    competition_stage: str


def timestamp_to_seconds(timestamp: str) -> float:
    """Period-relative ``HH:MM:SS.fff`` timestamp in seconds."""
    parsed = datetime.strptime(timestamp, "%H:%M:%S.%f")
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second + parsed.microsecond / 1_000_000


def available_360_match_ids(three_sixty_dir: Path = THREE_SIXTY_DIR) -> list[int]:
    """Match ids that have a three-sixty JSON file (same rule as the builder's caller)."""
    return sorted(int(path.stem) for path in three_sixty_dir.glob("*.json") if path.stem.isdigit())


def eligible_match_ids() -> list[int]:
    """Matches with 360 data that are not in configs/excluded_matches.json."""
    excluded = load_exclusion_manifest()
    return [match_id for match_id in available_360_match_ids() if match_id not in excluded]


def _name(value: Any) -> str | None:
    return value.get("name") if isinstance(value, dict) else None


def _id(value: Any) -> int | None:
    return value.get("id") if isinstance(value, dict) else None


def event_records(match_id: int, events: list[dict[str, Any]], frames_by_event_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten one match's raw events (already loaded) into table rows."""
    rows = []
    for position, event in enumerate(events):
        location = event.get("location")
        has_location = isinstance(location, list) and len(location) >= 2
        frame = frames_by_event_id.get(event["id"])
        rows.append({
            "match_id": match_id,
            "list_position": position,
            "index": event.get("index"),
            "event_id": event["id"],
            "period": event.get("period"),
            "timestamp": event.get("timestamp"),
            "t_seconds": timestamp_to_seconds(event["timestamp"]) if isinstance(event.get("timestamp"), str) else None,
            "minute": event.get("minute"),
            "second": event.get("second"),
            "type": _name(event.get("type")),
            "team_id": _id(event.get("team")),
            "possession": event.get("possession"),
            "possession_team_id": _id(event.get("possession_team")),
            "play_pattern": _name(event.get("play_pattern")),
            "has_location": has_location,
            "location_x": float(location[0]) if has_location else None,
            "location_y": float(location[1]) if has_location else None,
            "shot_type": _name(event.get("shot", {}).get("type")) if isinstance(event.get("shot"), dict) else None,
            "has_360": frame is not None,
            "has_freeze_frame": bool(frame.get("freeze_frame")) if frame is not None else False,
        })
    return rows


def load_match_event_records(match_id: int) -> list[dict[str, Any]]:
    """Read one match's raw events and 360 file and flatten them."""
    with (EVENTS_DIR / f"{match_id}.json").open(encoding="utf-8") as file:
        events = json.load(file)
    frames_path = THREE_SIXTY_DIR / f"{match_id}.json"
    frames: list[dict[str, Any]] = []
    if frames_path.exists():
        with frames_path.open(encoding="utf-8") as file:
            frames = json.load(file)
    return event_records(match_id, events, {frame["event_uuid"]: frame for frame in frames})


def max_period(match_id: int) -> int:
    """Highest period in one match's events file (3/4 = extra time, 5 = shootout)."""
    with (EVENTS_DIR / f"{match_id}.json").open(encoding="utf-8") as file:
        return max(int(event["period"]) for event in json.load(file))


def build_event_table(match_ids: list[int], workers: int = 1) -> pd.DataFrame:
    """Event table for the given matches, sorted by (match_id, list_position)."""
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(load_match_event_records, match_ids, chunksize=1))
    else:
        chunks = [load_match_event_records(match_id) for match_id in match_ids]
    table = pd.DataFrame([row for chunk in chunks for row in chunk], columns=EVENT_TABLE_COLUMNS)
    return table.sort_values(["match_id", "list_position"], kind="stable").reset_index(drop=True)


def load_or_build_event_table(cache_path: Path, match_ids: list[int], workers: int = 1) -> pd.DataFrame:
    """Read the cached event table if it covers exactly ``match_ids``; else rebuild it.

    The cache is a derived, regenerable file (``data/interim``); it is
    rebuilt whenever its match set differs, never silently partially reused.
    """
    if cache_path.exists():
        table = pd.read_pickle(cache_path)
        if sorted(table["match_id"].unique().tolist()) == sorted(match_ids):
            LOGGER.info("event table: cache hit %s (%d rows)", cache_path, len(table))
            return table
        LOGGER.info("event table: cache %s covers other matches; rebuilding", cache_path)
    table = build_event_table(match_ids, workers=workers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(cache_path)
    LOGGER.info("event table: built %d rows for %d matches -> %s", len(table), len(match_ids), cache_path)
    return table


def load_match_metadata(matches_dir: Path = MATCHES_DIR) -> dict[int, MatchMetadata]:
    """Competition/season/date/teams per match id from matches/<comp>/<season>.json."""
    metadata: dict[int, MatchMetadata] = {}
    for path in sorted(matches_dir.glob("*/*.json")):
        with path.open(encoding="utf-8") as file:
            for record in json.load(file):
                match_id = record.get("match_id")
                if not isinstance(match_id, int):
                    continue
                metadata[match_id] = MatchMetadata(
                    match_id=match_id,
                    competition=str(_name(record.get("competition")) or record.get("competition", {}).get("competition_name", "Unknown")),
                    season=str(record.get("season", {}).get("season_name", "Unknown")),
                    match_date=str(record.get("match_date", "Unknown")),
                    home_team=str(record.get("home_team", {}).get("home_team_name", "Unknown")),
                    away_team=str(record.get("away_team", {}).get("away_team_name", "Unknown")),
                    competition_stage=str(_name(record.get("competition_stage")) or "Unknown"),
                )
    return metadata
