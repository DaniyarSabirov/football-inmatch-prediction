"""Experiment 3: raw event details (subtypes, outcomes, related events) for development matches.

The experiment-2 event table keeps only the primary type. The taxonomy audit needs the
type-specific detail of every event, e.g. ``goalkeeper.type`` (Shot Faced, Collected, ...),
``duel.type`` (Tackle, Aerial Lost), ``pass.type`` (Corner, Goal Kick, ...) and the outcome,
plus ``related_events`` (StatsBomb's own links between records of the same moment).

Read-only on raw data; cached in data/interim/sequence_models/v3/. Callers must pass
development match ids; ``build_event_details`` refuses final-test matches.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from .raw_events import EVENTS_DIR
from .splits import assert_no_final_test

LOGGER = logging.getLogger(__name__)
STANDARD_KEYS = {"id", "index", "period", "timestamp", "minute", "second", "type", "possession", "possession_team",
                 "play_pattern", "team", "player", "position", "location", "duration", "related_events",
                 "under_pressure", "off_camera", "out", "counterpress", "tactics"}
DETAIL_COLUMNS = ["event_id", "match_id", "subtype", "outcome", "related_events", "n_related", "under_pressure",
                  "counterpress", "duration"]


def event_details(match_id: int, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for e in events:
        detail = next((v for k, v in e.items() if k not in STANDARD_KEYS and isinstance(v, dict)), {})
        sub = detail.get("type")
        out = detail.get("outcome")
        related = e.get("related_events") or []
        rows.append({
            "event_id": e["id"], "match_id": match_id,
            "subtype": sub.get("name") if isinstance(sub, dict) else None,
            "outcome": out.get("name") if isinstance(out, dict) else None,
            "related_events": ";".join(related), "n_related": len(related),
            "under_pressure": bool(e.get("under_pressure", False)), "counterpress": bool(e.get("counterpress", False)),
            "duration": e.get("duration"),
        })
    return rows


def _load(match_id: int) -> list[dict[str, Any]]:
    with (EVENTS_DIR / f"{match_id}.json").open(encoding="utf-8") as f:
        return event_details(match_id, json.load(f))


def build_event_details(match_ids: list[int], final_test: list[int], cache: Path, workers: int = 4) -> pd.DataFrame:
    """Details for exactly ``match_ids`` (development only), cached; the cache is rebuilt if the set differs."""
    assert_no_final_test(match_ids, final_test)
    if cache.exists():
        table = pd.read_pickle(cache)
        if sorted(table["match_id"].unique().tolist()) == sorted(match_ids):
            return table
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(_load, match_ids, chunksize=4))
    table = pd.DataFrame([r for c in chunks for r in c], columns=DETAIL_COLUMNS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(cache)
    LOGGER.info("event details: %d rows for %d matches -> %s", len(table), len(match_ids), cache)
    return table
