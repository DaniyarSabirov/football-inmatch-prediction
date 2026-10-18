"""Tests for the visual data-quality inspection tool (scripts/inspect_data_quality.py).

Synthetic, deterministic -- no dependency on the real dataset. `raw_input_paths`
is monkeypatched to point at tmp_path fixtures (the standard pattern already
used in tests/test_audit_dataset.py), and OUTPUT_DIR is monkeypatched to a
tmp_path so these tests never write into the real
results/data_quality_inspections/ directory.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_data_quality.py"
_spec = importlib.util.spec_from_file_location("inspect_data_quality", SCRIPT_PATH)
idq = importlib.util.module_from_spec(_spec)
sys.modules["inspect_data_quality"] = idq
_spec.loader.exec_module(idq)

HOME = {"id": 1, "name": "Home FC"}
AWAY = {"id": 2, "name": "Away FC"}


def _event(event_id, index, event_type, team, timestamp, period=1, location=None, player="Player"):
    event = {
        "id": event_id,
        "index": index,
        "period": period,
        "timestamp": timestamp,
        "minute": 0,
        "type": {"name": event_type},
        "team": dict(team),
        "player": {"name": player},
    }
    if location is not None:
        event["location"] = location
    return event


def _frame(event_uuid, freeze_frame, visible_area=None):
    return {
        "event_uuid": event_uuid,
        "visible_area": visible_area or [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
        "freeze_frame": freeze_frame,
    }


def _write_match(tmp_path, match_id, events, frames):
    events_dir = tmp_path / "events"
    frames_dir = tmp_path / "three-sixty"
    events_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    (events_dir / f"{match_id}.json").write_text(json.dumps(events), encoding="utf-8")
    (frames_dir / f"{match_id}.json").write_text(json.dumps(frames), encoding="utf-8")
    return events_dir, frames_dir


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Redirect raw_input_paths() and OUTPUT_DIR to tmp_path for one test."""
    events_dir = tmp_path / "events"
    frames_dir = tmp_path / "three-sixty"
    events_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)

    def fake_raw_input_paths(match_id):
        return events_dir / f"{match_id}.json", frames_dir / f"{match_id}.json"

    monkeypatch.setattr(idq, "raw_input_paths", fake_raw_input_paths)
    output_dir = tmp_path / "inspections_out"
    monkeypatch.setattr(idq, "OUTPUT_DIR", output_dir)
    return tmp_path


def _away_filler():
    # resolve_reference_team_id (reused from src/team_normalization.py)
    # requires exactly two distinct team ids somewhere in the events list;
    # a real match always has both teams' events, but a minimal synthetic
    # fixture must supply this explicitly. No frame/location needed -- the
    # function only scans event["team"]["id"].
    return _event("away-filler", 900, "Pressure", AWAY, "00:59:59.000")


def _write(tmp_path, match_id, events, frames):
    events_path = tmp_path / "events" / f"{match_id}.json"
    frames_path = tmp_path / "three-sixty" / f"{match_id}.json"
    events_path.write_text(json.dumps(events), encoding="utf-8")
    frames_path.write_text(json.dumps(frames), encoding="utf-8")


# --- CLI argument parsing ----------------------------------------------------

def test_build_parser_exposes_expected_arguments() -> None:
    parser = idq.build_parser()
    args = parser.parse_args(["--match", "123", "--event", "abc", "--issue", "orientation", "--theme", "light"])
    assert args.match == 123
    assert args.event == "abc"
    assert args.issue == "orientation"
    assert args.theme == "light"


def test_build_parser_rejects_unknown_issue() -> None:
    parser = idq.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--issue", "not-a-real-issue"])


def test_main_with_no_arguments_prints_help_and_returns_nonzero(capsys) -> None:
    exit_code = idq.main([])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


def test_main_issue_requiring_match_without_match_fails_clearly(capsys) -> None:
    exit_code = idq.main(["--issue", "out-of-bounds"])
    assert exit_code == 1
    assert "requires --match" in capsys.readouterr().out


# --- Mode A: inspect one event ------------------------------------------------

def test_inspect_event_valid_event_and_matching_frame(patched_paths, capsys) -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000", location=[50.0, 40.0]), _away_filler()]
    frames = [_frame("e1", [
        {"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]},
    ])]
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_event(1, "e1", "dark")
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Match ID:        1" in out
    assert "Event UUID:      e1" in out
    assert "Orientation classification:" in out
    assert (idq.OUTPUT_DIR / "match1_event_e1.png").exists()
    assert (idq.OUTPUT_DIR / "match1_event_e1.txt").exists()


def test_inspect_event_missing_event_reports_explicitly_not_substituted(patched_paths, capsys) -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000", location=[50.0, 40.0])]
    frames = [_frame("e1", [{"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]}])]
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_event(1, "does-not-exist", "dark")
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "was not found" in out
    # Nothing should have been written for a failed lookup.
    assert not (idq.OUTPUT_DIR / "match1_event_does-not-exist.png").exists()


def test_inspect_event_truncated_uuid_suggests_the_full_one(patched_paths, capsys) -> None:
    events = [_event("abcdef12-full-uuid", 1, "Pass", HOME, "00:00:01.000", location=[1.0, 1.0])]
    frames = []
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_event(1, "abcdef12", "dark")
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "truncated UUID" in out
    assert "abcdef12-full-uuid" in out


def test_inspect_event_without_matching_frame_is_reported_not_guessed(patched_paths, capsys) -> None:
    # Event exists but has no 360 coverage -- must say so explicitly, never
    # silently pick a different event's frame.
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000", location=[50.0, 40.0]),
        _event("e2", 2, "Pass", AWAY, "00:00:02.000", location=[10.0, 10.0]),
    ]
    frames = [_frame("e2", [{"teammate": True, "actor": True, "keeper": False, "location": [10.0, 10.0]}])]
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_event(1, "e1", "dark")
    assert exit_code == 0  # event metadata alone is still a valid, complete report
    out = capsys.readouterr().out
    assert "No matching 360 frame found for this event" in out


# --- Mode B: out-of-bounds points ---------------------------------------------

def test_inspect_out_of_bounds_detects_and_reports_offenders(patched_paths, capsys) -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000", location=[50.0, 40.0])]
    frames = [_frame("e1", [
        {"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]},   # in bounds
        {"teammate": False, "actor": False, "keeper": False, "location": [-2.0, 40.0]},  # x<0, overshoot 2.0
        {"teammate": False, "actor": False, "keeper": False, "location": [60.0, 95.0]},  # y>80, overshoot 15.0
    ])]
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_out_of_bounds(1, None, "dark")
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "2 out-of-bounds point(s) found" in out
    assert "overshoot=2.000" in out
    assert "overshoot=15.000" in out
    assert (idq.OUTPUT_DIR / "match1_out_of_bounds.png").exists()

    with (idq.OUTPUT_DIR / "match1_out_of_bounds.json").open() as f:
        payload = json.load(f)
    assert len(payload["offenders"]) == 2


def test_inspect_out_of_bounds_scoped_to_one_event(patched_paths, capsys) -> None:
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000", location=[50.0, 40.0]),
        _event("e2", 2, "Pass", HOME, "00:00:02.000", location=[50.0, 40.0]),
    ]
    frames = [
        _frame("e1", [{"teammate": True, "actor": True, "keeper": False, "location": [-5.0, 40.0]}]),
        _frame("e2", [{"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]}]),
    ]
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_out_of_bounds(1, "e1", "dark")
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "1 out-of-bounds point(s) found in event e1" in out


def test_overshoot_magnitude_matches_distance_past_boundary() -> None:
    assert idq.overshoot_magnitude((60.0, 40.0)) == pytest.approx(0.0)
    assert idq.overshoot_magnitude((-3.0, 40.0)) == pytest.approx(3.0)
    assert idq.overshoot_magnitude((123.0, 40.0)) == pytest.approx(3.0)
    assert idq.overshoot_magnitude((60.0, 90.0)) == pytest.approx(10.0)


# --- Mode C: actor anomalies ---------------------------------------------------

def test_find_actor_anomalies_detects_zero_actor_case(patched_paths) -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000", location=[50.0, 40.0])]
    frames = [_frame("e1", [{"teammate": True, "actor": False, "keeper": False, "location": [50.0, 40.0]}])]
    _write(patched_paths, 1, events, frames)

    zero_actor, multi_actor = idq.find_actor_anomalies(1)
    assert len(zero_actor) == 1
    assert zero_actor[0]["event_uuid"] == "e1"
    assert multi_actor == []


def test_find_actor_anomalies_detects_multiple_actor_case(patched_paths) -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000", location=[50.0, 40.0])]
    frames = [_frame("e1", [
        {"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]},
        {"teammate": True, "actor": True, "keeper": False, "location": [90.0, 70.0]},
    ])]
    _write(patched_paths, 1, events, frames)

    zero_actor, multi_actor = idq.find_actor_anomalies(1)
    assert zero_actor == []
    assert len(multi_actor) == 1
    assert multi_actor[0]["actor_locations"] == [[50.0, 40.0], [90.0, 70.0]]


def test_inspect_actor_anomalies_reports_ambiguity_distance_and_builder_result(patched_paths, capsys) -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000", location=[50.0, 40.0])]
    frames = [_frame("e1", [
        {"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]},
        {"teammate": True, "actor": True, "keeper": False, "location": [90.0, 70.0]},  # far apart -> ambiguous
    ])]
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_actor_anomalies(1, "dark")
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Multiple-actor frames: 1" in out
    assert "within_tolerance(1.0)=False" in out
    assert "actor_ambiguous=True" in out  # builder cross-check never silently resolves it
    assert (idq.OUTPUT_DIR / "actor_anomalies_match1.png").exists()


def test_inspect_actor_anomalies_near_duplicate_is_resolved_not_ambiguous(patched_paths, capsys) -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000", location=[50.0, 40.0])]
    frames = [_frame("e1", [
        {"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]},
        {"teammate": True, "actor": True, "keeper": False, "location": [50.1, 40.1]},  # near-duplicate
    ])]
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_actor_anomalies(1, "dark")
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "within_tolerance(1.0)=True" in out
    assert "actor_ambiguous=False" in out


# --- Mode D: orientation -------------------------------------------------------

def test_inspect_orientation_uncertain_shows_raw_in_both_panels(patched_paths, capsys) -> None:
    # No event.location at all -> classify_orientation's geometry votes are
    # empty -> UNCERTAIN, confidence 0.0, deterministically.
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000"), _away_filler()]  # no location
    frames = [_frame("e1", [{"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]}])]
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_orientation(1, "e1", None, None, None, "dark")
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Orientation: UNCERTAIN" in out
    assert "Transformation applied (orientation mirror): False" in out
    assert "No transformation is applied" in out
    assert (idq.OUTPUT_DIR / "match1_orientation_e1.png").exists()


def test_inspect_orientation_filters_by_period_and_event_type(patched_paths, capsys) -> None:
    events = [
        _event("e1", 1, "Pass", HOME, "00:00:01.000", period=1, location=[50.0, 40.0]),
        _event("e2", 2, "Carry", HOME, "00:00:02.000", period=2, location=[50.0, 40.0]),
        _away_filler(),
    ]
    frames = [
        _frame("e1", [{"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]}]),
        _frame("e2", [{"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]}]),
    ]
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_orientation(1, None, None, 2, "Carry", "dark")
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "event e2" in out  # only the period-2 Carry event should be selected


def test_inspect_orientation_no_matches_reports_clearly(patched_paths, capsys) -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000", period=1, location=[50.0, 40.0])]
    frames = [_frame("e1", [{"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]}])]
    _write(patched_paths, 1, events, frames)

    exit_code = idq.inspect_orientation(1, None, None, 99, None, "dark")
    assert exit_code == 1
    assert "No events match" in capsys.readouterr().out


# --- Mode E: corrupted JSON ----------------------------------------------------

def test_inspect_corrupted_json_detects_nul_bytes_without_crashing(patched_paths, capsys) -> None:
    events_dir = patched_paths / "events"
    frames_dir = patched_paths / "three-sixty"
    events_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    (events_dir / "1.json").write_text(json.dumps([_event("e1", 1, "Pass", HOME, "00:00:01.000")]))

    good_record = json.dumps({"event_uuid": "e1", "visible_area": [], "freeze_frame": []}).encode()
    corrupted_bytes = b"[" + good_record + b", {\"event_uuid\": \"e2\", \"freeze_frame\": [" + (b"\x00" * 50) + b"]}]"
    (frames_dir / "1.json").write_bytes(corrupted_bytes)

    exit_code = idq.inspect_corrupted_json(1, "dark")
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "NUL-byte (0x00) runs found: 1" in out
    assert "offset=" in out
    assert "length=50" in out
    assert "events.json: OK" in out
    # Safe recovery should find the first (complete, unaffected) record.
    assert "Successfully parsed independently: 1" in out


def test_inspect_corrupted_json_never_modifies_the_source_file(patched_paths) -> None:
    events_dir = patched_paths / "events"
    frames_dir = patched_paths / "three-sixty"
    events_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    (events_dir / "1.json").write_text("[]")
    corrupted_bytes = b"[{\"a\": " + (b"\x00" * 20) + b"}]"
    frame_path = frames_dir / "1.json"
    frame_path.write_bytes(corrupted_bytes)

    before = frame_path.read_bytes()
    idq.inspect_corrupted_json(1, "dark")
    after = frame_path.read_bytes()
    assert before == after


def test_inspect_corrupted_json_on_valid_file_reports_no_corruption(patched_paths, capsys) -> None:
    events_dir = patched_paths / "events"
    frames_dir = patched_paths / "three-sixty"
    events_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    (events_dir / "1.json").write_text("[]")
    (frames_dir / "1.json").write_text("[]")

    exit_code = idq.inspect_corrupted_json(1, "dark")
    assert exit_code == 0
    assert "no corruption detected" in capsys.readouterr().out


def test_find_nul_runs_locates_exact_offsets_and_lengths() -> None:
    data = b"abc" + b"\x00" * 5 + b"def" + b"\x00" * 2 + b"ghi"
    runs = idq.find_nul_runs(data)
    assert runs == [(3, 5), (11, 2)]


def test_find_nul_runs_empty_when_no_corruption() -> None:
    assert idq.find_nul_runs(b"no nulls here") == []


def test_extract_top_level_objects_ignores_braces_inside_strings() -> None:
    data = b'[{"a": "value with } brace"}, {"b": 2}]'
    ranges, _ = idq.extract_top_level_objects(data)
    assert len(ranges) == 2
    for start, end in ranges:
        json.loads(data[start:end])  # each extracted range must be independently valid


# --- Never modifies raw input files (general guarantee) -----------------------

def test_tool_never_writes_into_its_own_input_directories(patched_paths) -> None:
    events = [_event("e1", 1, "Pass", HOME, "00:00:01.000", location=[50.0, 40.0]), _away_filler()]
    frames = [_frame("e1", [{"teammate": True, "actor": True, "keeper": False, "location": [50.0, 40.0]}])]
    _write(patched_paths, 1, events, frames)

    events_path = patched_paths / "events" / "1.json"
    frames_path = patched_paths / "three-sixty" / "1.json"
    before_events = events_path.read_bytes()
    before_frames = frames_path.read_bytes()

    idq.inspect_event(1, "e1", "dark")
    idq.inspect_out_of_bounds(1, None, "dark")
    idq.inspect_actor_anomalies(1, "dark")
    idq.inspect_orientation(1, "e1", None, None, None, "dark")

    assert events_path.read_bytes() == before_events
    assert frames_path.read_bytes() == before_frames
