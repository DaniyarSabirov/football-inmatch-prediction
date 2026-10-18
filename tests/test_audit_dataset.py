"""Tests for the read-only data-quality audit script (scripts/audit_dataset.py).

Synthetic, deterministic -- no dependency on the real dataset, so these run
fast and do not require data/raw to be populated.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_dataset.py"
_spec = importlib.util.spec_from_file_location("audit_dataset", SCRIPT_PATH)
audit_dataset = importlib.util.module_from_spec(_spec)
sys.modules["audit_dataset"] = audit_dataset
_spec.loader.exec_module(audit_dataset)


def test_classify_coordinate_distinguishes_missing_malformed_non_numeric_and_bounds() -> None:
    assert audit_dataset.classify_coordinate(None) == "missing"
    assert audit_dataset.classify_coordinate([1.0]) == "malformed"
    assert audit_dataset.classify_coordinate("not a list") == "malformed"
    assert audit_dataset.classify_coordinate([True, 1.0]) == "non_numeric"
    assert audit_dataset.classify_coordinate(["a", "b"]) == "non_numeric"
    assert audit_dataset.classify_coordinate([60.0, 40.0]) == "in_bounds"
    assert audit_dataset.classify_coordinate([0.0, 0.0]) == "in_bounds"
    assert audit_dataset.classify_coordinate([120.0, 80.0]) == "in_bounds"


def test_classify_coordinate_reports_every_violated_boundary() -> None:
    assert audit_dataset.classify_coordinate([-1.0, 40.0]) == "x<0"
    assert audit_dataset.classify_coordinate([121.0, 40.0]) == "x>120"
    assert audit_dataset.classify_coordinate([60.0, -1.0]) == "y<0"
    assert audit_dataset.classify_coordinate([60.0, 81.0]) == "y>80"
    # A corner overshoot can violate two boundaries simultaneously.
    assert audit_dataset.classify_coordinate([121.0, -1.0]) == "x>120+y<0"


def test_overshoot_magnitude_is_distance_past_nearest_boundary() -> None:
    assert audit_dataset.overshoot_magnitude([60.0, 40.0]) == pytest.approx(0.0)
    assert audit_dataset.overshoot_magnitude([-2.0, 40.0]) == pytest.approx(2.0)
    assert audit_dataset.overshoot_magnitude([122.0, 40.0]) == pytest.approx(2.0)
    assert audit_dataset.overshoot_magnitude([60.0, 90.0]) == pytest.approx(10.0)


def test_magnitude_bucket_boundaries() -> None:
    assert audit_dataset.magnitude_bucket(0.0) == "<=0.5"
    assert audit_dataset.magnitude_bucket(0.5) == "<=0.5"
    assert audit_dataset.magnitude_bucket(0.51) == "<=1.0"
    assert audit_dataset.magnitude_bucket(10.0) == "<=10.0"
    assert audit_dataset.magnitude_bucket(10.01) == ">10.0"


def test_audit_reports_failed_match_with_diagnostic_detail(tmp_path, monkeypatch) -> None:
    # A match whose events.json is malformed must be reported with the
    # actual exception type/message, not silently counted and discarded.
    events_dir = tmp_path / "events"
    frames_dir = tmp_path / "three-sixty"
    events_dir.mkdir()
    frames_dir.mkdir()
    (events_dir / "999.json").write_text("{not valid json", encoding="utf-8")
    (frames_dir / "999.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(audit_dataset, "EVENTS_DIR", events_dir)
    monkeypatch.setattr(audit_dataset, "FRAMES_DIR", frames_dir)

    report = audit_dataset.audit([999], {})
    assert report["matches_failed_to_load"] == 1
    assert report["matches_loaded_ok"] == 0
    assert len(report["failed_matches"]) == 1
    failure = report["failed_matches"][0]
    assert failure["match_id"] == 999
    assert failure["error_type"] == "JSONDecodeError"
    assert "999.json" in failure["file"]
    assert failure["error_message"]  # never empty -- the real message is preserved


def test_audit_never_writes_to_input_directories(tmp_path, monkeypatch) -> None:
    events_dir = tmp_path / "events"
    frames_dir = tmp_path / "three-sixty"
    events_dir.mkdir()
    frames_dir.mkdir()

    event = {
        "id": "e1", "index": 1, "period": 1, "timestamp": "00:00:01.000", "minute": 0,
        "type": {"name": "Pass"}, "team": {"id": 1, "name": "A"}, "player": {"name": "P"},
        "location": [10.0, 10.0],
    }
    frame = {"event_uuid": "e1", "visible_area": [], "freeze_frame": [
        {"teammate": True, "actor": True, "keeper": False, "location": [10.0, 10.0]},
    ]}
    import json
    (events_dir / "1.json").write_text(json.dumps([event]), encoding="utf-8")
    (frames_dir / "1.json").write_text(json.dumps([frame]), encoding="utf-8")

    monkeypatch.setattr(audit_dataset, "EVENTS_DIR", events_dir)
    monkeypatch.setattr(audit_dataset, "FRAMES_DIR", frames_dir)

    before_events = (events_dir / "1.json").read_text()
    before_frames = (frames_dir / "1.json").read_text()
    audit_dataset.audit([1], {})
    assert (events_dir / "1.json").read_text() == before_events
    assert (frames_dir / "1.json").read_text() == before_frames


def test_main_refuses_to_overwrite_existing_output_without_flag(tmp_path, monkeypatch) -> None:
    events_dir = tmp_path / "events"
    frames_dir = tmp_path / "three-sixty"
    events_dir.mkdir()
    frames_dir.mkdir()
    monkeypatch.setattr(audit_dataset, "EVENTS_DIR", events_dir)
    monkeypatch.setattr(audit_dataset, "FRAMES_DIR", frames_dir)

    output_path = tmp_path / "report.json"
    output_path.write_text('{"sentinel": true}', encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["audit_dataset.py", "--output", str(output_path)])
    exit_code = audit_dataset.main()
    assert exit_code == 1
    assert output_path.read_text() == '{"sentinel": true}'  # untouched


def test_main_overwrites_when_flag_is_passed(tmp_path, monkeypatch) -> None:
    events_dir = tmp_path / "events"
    frames_dir = tmp_path / "three-sixty"
    events_dir.mkdir()
    frames_dir.mkdir()
    monkeypatch.setattr(audit_dataset, "EVENTS_DIR", events_dir)
    monkeypatch.setattr(audit_dataset, "FRAMES_DIR", frames_dir)

    output_path = tmp_path / "report.json"
    output_path.write_text('{"sentinel": true}', encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["audit_dataset.py", "--output", str(output_path), "--overwrite"])
    exit_code = audit_dataset.main()
    assert exit_code == 0
    assert output_path.read_text() != '{"sentinel": true}'


def test_audit_detects_nonempty_files_with_zero_overlapping_event_ids(tmp_path, monkeypatch) -> None:
    # Found 2026-10-14: 7 real matches had non-empty, well-formed events.json
    # and three-sixty.json files that shared zero event ids -- invisible to
    # the old aggregate-only matched_pairs count. This is a dedicated
    # regression guard for that detection.
    import json

    events_dir = tmp_path / "events"
    frames_dir = tmp_path / "three-sixty"
    events_dir.mkdir()
    frames_dir.mkdir()

    event = {
        "id": "event-from-this-match", "index": 1, "period": 1, "timestamp": "00:00:01.000",
        "minute": 0, "type": {"name": "Pass"}, "team": {"id": 1, "name": "A"}, "player": {"name": "P"},
        "location": [10.0, 10.0],
    }
    mismatched_frame = {
        "event_uuid": "event-from-a-totally-different-match",
        "visible_area": [], "freeze_frame": [],
    }
    (events_dir / "1.json").write_text(json.dumps([event]), encoding="utf-8")
    (frames_dir / "1.json").write_text(json.dumps([mismatched_frame]), encoding="utf-8")

    monkeypatch.setattr(audit_dataset, "EVENTS_DIR", events_dir)
    monkeypatch.setattr(audit_dataset, "FRAMES_DIR", frames_dir)

    report = audit_dataset.audit([1], {})
    assert report["matches_with_zero_matched_pairs"] == [1]
    assert report["matched_pairs"] == 0


def test_audit_does_not_flag_a_match_with_genuine_zero_coverage(tmp_path, monkeypatch) -> None:
    # A match with an EMPTY three-sixty.json ([]) is a different, already
    # well-understood case (zero 360 coverage, not a file-pairing mismatch)
    # and must not be flagged by the new check.
    import json

    events_dir = tmp_path / "events"
    frames_dir = tmp_path / "three-sixty"
    events_dir.mkdir()
    frames_dir.mkdir()
    event = {
        "id": "e1", "index": 1, "period": 1, "timestamp": "00:00:01.000", "minute": 0,
        "type": {"name": "Pass"}, "team": {"id": 1, "name": "A"}, "player": {"name": "P"},
        "location": [10.0, 10.0],
    }
    (events_dir / "1.json").write_text(json.dumps([event]), encoding="utf-8")
    (frames_dir / "1.json").write_text(json.dumps([]), encoding="utf-8")

    monkeypatch.setattr(audit_dataset, "EVENTS_DIR", events_dir)
    monkeypatch.setattr(audit_dataset, "FRAMES_DIR", frames_dir)

    report = audit_dataset.audit([1], {})
    assert report["matches_with_zero_matched_pairs"] == []
