"""Tests for the physical pitch-orientation calibration registry
(src/physical_orientation_calibration.py), added as part of the
all-matches feasibility study -- see
docs/all_matches_physical_orientation_feasibility.md.

Synthetic fixtures only (except one deliberate exception below that
validates the actual shipped registry's structure), mirroring the
precedent set by tests/test_match_exclusions.py.
"""

import json

import pytest

from src.physical_orientation_calibration import (
    DEFAULT_REGISTRY_PATH,
    CalibrationRegistryError,
    load_calibration_registry,
    resolve_usable_physical_reference_team_name,
)

VALID_ENTRY = {
    "match_id": 123,
    "period_1_physical_reference_team": "Team A",
    "verification_status": "verified",
    "confidence": "high",
    "evidence_landmark": "Goalkeeper kit color, right-side goal",
    "verification_method": "human visual comparison against broadcast screenshot",
    "date_verified": "2026-10-21",
    "report": "docs/full_physical_pitch_orientation_audit.md",
}


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- missing file is NOT an error (unlike the exclusion manifest) -----------

def test_missing_registry_file_returns_empty_mapping_not_an_error(tmp_path) -> None:
    missing_path = tmp_path / "does_not_exist.json"
    assert load_calibration_registry(missing_path) == {}


# --- malformed file IS an error, matching the match-exclusions precedent ---

def test_invalid_json_raises(tmp_path) -> None:
    path = tmp_path / "registry.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CalibrationRegistryError):
        load_calibration_registry(path)


def test_missing_required_field_raises(tmp_path) -> None:
    path = tmp_path / "registry.json"
    entry = dict(VALID_ENTRY)
    del entry["evidence_landmark"]
    _write(path, {"calibrations": [entry]})
    with pytest.raises(CalibrationRegistryError):
        load_calibration_registry(path)


def test_invalid_verification_status_raises(tmp_path) -> None:
    path = tmp_path / "registry.json"
    entry = {**VALID_ENTRY, "verification_status": "definitely_correct_trust_me"}
    _write(path, {"calibrations": [entry]})
    with pytest.raises(CalibrationRegistryError):
        load_calibration_registry(path)


def test_duplicate_match_id_raises(tmp_path) -> None:
    path = tmp_path / "registry.json"
    _write(path, {"calibrations": [VALID_ENTRY, VALID_ENTRY]})
    with pytest.raises(CalibrationRegistryError):
        load_calibration_registry(path)


def test_non_integer_match_id_raises(tmp_path) -> None:
    path = tmp_path / "registry.json"
    entry = {**VALID_ENTRY, "match_id": "123"}
    _write(path, {"calibrations": [entry]})
    with pytest.raises(CalibrationRegistryError):
        load_calibration_registry(path)


def test_wrong_top_level_shape_raises(tmp_path) -> None:
    path = tmp_path / "registry.json"
    _write(path, {"not_calibrations": []})
    with pytest.raises(CalibrationRegistryError):
        load_calibration_registry(path)


# --- valid registry loads correctly, with defaults for optional fields -----

def test_valid_registry_loads_correctly(tmp_path) -> None:
    path = tmp_path / "registry.json"
    _write(path, {"calibrations": [VALID_ENTRY]})
    registry = load_calibration_registry(path)
    assert set(registry.keys()) == {123}
    entry = registry[123]
    assert entry.period_1_physical_reference_team == "Team A"
    assert entry.verification_status == "verified"
    assert entry.horizontal_orientation_verified is False  # default when field omitted
    assert entry.source_url is None  # default when field omitted


# --- resolve_usable_physical_reference_team_name: the auto-apply gate ------

def test_verified_entry_is_usable(tmp_path) -> None:
    path = tmp_path / "registry.json"
    _write(path, {"calibrations": [VALID_ENTRY]})
    registry = load_calibration_registry(path)
    assert resolve_usable_physical_reference_team_name(123, registry) == "Team A"


@pytest.mark.parametrize("status", ["partially_verified", "unverified", "ambiguous"])
def test_non_verified_entries_are_never_auto_applied(status, tmp_path) -> None:
    path = tmp_path / "registry.json"
    entry = {**VALID_ENTRY, "verification_status": status}
    _write(path, {"calibrations": [entry]})
    registry = load_calibration_registry(path)
    # The entry loads successfully (so review tooling can still display it)...
    assert registry[123].verification_status == status
    # ...but is never returned as a usable, auto-applicable calibration.
    assert resolve_usable_physical_reference_team_name(123, registry) is None


def test_absent_match_id_returns_none() -> None:
    assert resolve_usable_physical_reference_team_name(999999, {}) is None


# --- the actual shipped registry must itself be valid -----------------------

def test_real_shipped_registry_is_structurally_valid() -> None:
    if not DEFAULT_REGISTRY_PATH.exists():
        pytest.skip("No calibration registry has been created yet.")
    registry = load_calibration_registry()
    for match_id, entry in registry.items():
        assert entry.match_id == match_id
        assert entry.verification_status in {"verified", "partially_verified", "unverified", "ambiguous"}


def test_match_3930160_is_verified_in_the_real_registry() -> None:
    if not DEFAULT_REGISTRY_PATH.exists():
        pytest.skip("No calibration registry has been created yet.")
    registry = load_calibration_registry()
    assert resolve_usable_physical_reference_team_name(3930160, registry) == "Spain"
