"""Causality and safety of the ML feature pipeline (2026-10-27).

- The whole feature pipeline (build_rows_from_match_data) is truncation
  invariant: cutting the input immediately after an event -- keeping only
  that event's NEXT meaningful event as bare event data (the label), with no
  360 record and nothing in between -- leaves every column that is not a
  TARGET or OFFLINE_DIAGNOSTIC unchanged, for that row and every earlier row.
  Real rows include several of the 55 rows whose offline orientation reads
  later records.
- configs/ml_feature_classification.json covers the dataset schema exactly,
  and the training allowlist only admits approved causal features.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.build_statsbomb_360_dataset import MEANINGFUL_EVENT_TYPES, build_rows_from_match_data
from src.ml_features import (
    FeatureSafetyError,
    approved_feature_columns,
    load_feature_classification,
    select_feature_matrix,
    validate_feature_columns,
    validate_schema,
)
from src.orientation_resolution import resolve_causal_orientations

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "statsbomb" / "data"
CLASSIFICATION = load_feature_classification()
TRUNCATION_INVARIANT = [
    c for c, e in CLASSIFICATION["columns"].items() if e["category"] not in ("TARGET", "OFFLINE_DIAGNOSTIC")
]


def _load(match_id):
    path = RAW / "events" / f"{match_id}.json"
    if not path.is_file():
        pytest.skip("raw match not available")
    events = json.loads(path.read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((RAW / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    return events, frames


def _truncate_after(events, frames, event_id):
    """Everything up to and including `event_id`, plus its next meaningful
    event (same period) as event data only -- the label, never its snapshot."""
    position = next(i for i, e in enumerate(events) if e["id"] == event_id)
    kept = events[: position + 1]
    label = next((e for e in events[position + 1:] if e["period"] == kept[-1]["period"]
                  and e["type"]["name"] in MEANINGFUL_EVENT_TYPES), None)
    if label is not None:
        kept = kept + [label]
    return kept, {e["id"]: frames[e["id"]] for e in events[: position + 1] if e["id"] in frames}


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


# match, event id of a row whose OFFLINE orientation read later records (from the 55-row audit)
CUT_ROWS = [
    (3773477, "00:46:48.258", "Carry"),     # later identical copy decided offline -> causally unavailable
    (3794688, "00:04:44.513", "Carry"),     # earlier copy -> causal_duplicate_snapshot
    (3893832, "00:21:40.764", "Miscontrol"),  # Problem A neighbour vote
    (3930164, "00:00:00.505", "Pass"),      # reviewer decision (frame 505)
    (3773386, "00:27:13.282", "Pass"),      # not among the 55: an ordinary row near the case_05 copy group
]


@pytest.mark.parametrize("match_id, timestamp, event_type", CUT_ROWS)
def test_feature_pipeline_is_truncation_invariant(match_id, timestamp, event_type):
    events, frames = _load(match_id)
    target = next((e for e in events if e["timestamp"] == timestamp and e["type"]["name"] == event_type), None)
    if target is None:  # the ordinary row may not exist at exactly that time: take the nearest Pass
        target = min((e for e in events if e["type"]["name"] == event_type and e["id"] in frames),
                     key=lambda e: abs(e["index"] - 1100))
    full = {r["current_event_id"]: r for r in build_rows_from_match_data(match_id, events, frames)}
    cut_events, cut_frames = _truncate_after(events, frames, target["id"])
    truncated = build_rows_from_match_data(match_id, cut_events, cut_frames)
    by_id = {r["current_event_id"]: r for r in truncated}
    assert target["id"] in by_id and target["id"] in full
    compared = 0
    for event_id, row in by_id.items():
        reference = full[event_id]
        for column in TRUNCATION_INVARIANT:
            assert _same(row[column], reference[column]), (event_id, column, row[column], reference[column])
        compared += 1
    assert compared >= 1  # the cut row itself; a cut right after the first event leaves only it


def test_causal_orientation_never_uses_later_records():
    events, frames = _load(3794688)
    records = [(e, frames[e["id"]]) for e in events if e["id"] in frames and frames[e["id"]].get("freeze_frame")]
    full = resolve_causal_orientations(records)
    for cut in (300, 1200, 2500):
        part = resolve_causal_orientations(records[:cut])
        assert all(full[k].orientation == v.orientation and full[k].resolved_by == v.resolved_by for k, v in part.items())


def test_causal_and_offline_values_live_in_separate_columns():
    events, frames = _load(3773477)
    rows = build_rows_from_match_data(3773477, events, frames)
    row = next(r for r in rows if r["timestamp"] == "00:46:48.258" and r["current_event_type"] == "Carry")
    assert row["orientation_uses_later_records"] is True
    assert row["orientation_available"] is False and row["actor_x"] is None  # causal: unavailable
    assert row["orientation_offline"] == "ROTATED_180" and row["actor_x_offline"] is not None  # offline kept apart


# --- classification and allowlist --------------------------------------------------

def test_classification_covers_the_builder_schema_exactly():
    events, frames = _load(3773386)
    rows = build_rows_from_match_data(3773386, events, frames)
    validate_schema(list(rows[0]), CLASSIFICATION)


def test_allowlist_contains_only_approved_causal_features():
    approved = approved_feature_columns(CLASSIFICATION)
    assert approved == ["current_event_type", "period", "minute", "actor_x", "actor_y", "actor_ambiguous",
                        "orientation_available", "visible_players", "visible_teammates", "visible_opponents",
                        "visible_ambiguous"]
    for column in approved:
        assert CLASSIFICATION["columns"][column]["category"] == "CAUSAL_FEATURE"
        assert not column.endswith("_offline")
    assert CLASSIFICATION["target"] not in approved


@pytest.mark.parametrize("unsafe", [
    "next_event_type", "same_team", "time_to_next_event_seconds", "visible_teammates_offline",
    "actor_x_offline", "orientation_uses_later_records", "match_id", "current_player", "orientation",
    "some_new_column",
])
def test_unsafe_or_unknown_columns_are_rejected(unsafe):
    with pytest.raises(FeatureSafetyError):
        validate_feature_columns(approved_feature_columns(CLASSIFICATION) + [unsafe], CLASSIFICATION)


def test_select_feature_matrix_returns_the_allowlist_only():
    pd = pytest.importorskip("pandas")
    events, frames = _load(3773386)
    frame = pd.DataFrame(build_rows_from_match_data(3773386, events, frames))
    matrix = select_feature_matrix(frame, CLASSIFICATION)
    assert list(matrix.columns) == approved_feature_columns(CLASSIFICATION)
    with pytest.raises(FeatureSafetyError):
        select_feature_matrix(frame.assign(leaked=1), CLASSIFICATION)  # unclassified column -> schema error


def test_a_non_causal_column_cannot_be_approved(tmp_path):
    broken = json.loads(json.dumps(CLASSIFICATION))
    broken["columns"]["visible_teammates_offline"]["approved_for_training"] = True
    path = tmp_path / "classification.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(FeatureSafetyError):
        load_feature_classification(path)
