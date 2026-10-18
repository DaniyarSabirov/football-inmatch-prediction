"""Tests for the central match-exclusion manifest (src/match_exclusions.py).

Synthetic fixtures only -- these tests never depend on the real
configs/excluded_matches.json so they stay valid even if that file's
contents change. One test (test_real_manifest_...) is a deliberate
exception: it validates the actual shipped manifest's structure.
"""

import json

import pytest

from src.match_exclusions import (
    DEFAULT_MANIFEST_PATH,
    ExcludedMatch,
    ExcludedMatchError,
    ExclusionManifestError,
    check_not_excluded,
    load_exclusion_manifest,
)


def _write_manifest(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


VALID_ENTRY = {
    "match_id": 123,
    "reason": "test_reason",
    "file": "data/raw/statsbomb/data/three-sixty/123.json",
    "evidence": "some evidence",
    "date_investigated": "2026-10-13",
    "report": "docs/some_report.md",
}


# --- loading a well-formed manifest ------------------------------------------

def test_load_exclusion_manifest_parses_valid_entries(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    _write_manifest(path, {"excluded_matches": [VALID_ENTRY]})

    manifest = load_exclusion_manifest(path)
    assert set(manifest.keys()) == {123}
    entry = manifest[123]
    assert isinstance(entry, ExcludedMatch)
    assert entry.reason == "test_reason"
    assert entry.file == "data/raw/statsbomb/data/three-sixty/123.json"


def test_load_exclusion_manifest_with_empty_list_is_valid(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    _write_manifest(path, {"excluded_matches": []})
    assert load_exclusion_manifest(path) == {}


def test_load_exclusion_manifest_ignores_unknown_top_level_keys(tmp_path) -> None:
    # e.g. the shipped manifest's "_comment" field.
    path = tmp_path / "excluded.json"
    _write_manifest(path, {"_comment": "docs", "excluded_matches": [VALID_ENTRY]})
    manifest = load_exclusion_manifest(path)
    assert set(manifest.keys()) == {123}


# --- fail-closed behavior on every malformed-manifest case -------------------

def test_missing_manifest_file_raises_clearly(tmp_path) -> None:
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(ExclusionManifestError, match="not found"):
        load_exclusion_manifest(missing_path)


def test_invalid_json_raises_clearly(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ExclusionManifestError, match="not valid JSON"):
        load_exclusion_manifest(path)


def test_wrong_top_level_shape_raises_clearly(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    _write_manifest(path, {"wrong_key": []})
    with pytest.raises(ExclusionManifestError, match="excluded_matches"):
        load_exclusion_manifest(path)


def test_top_level_not_an_object_raises_clearly(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    _write_manifest(path, [1, 2, 3])
    with pytest.raises(ExclusionManifestError):
        load_exclusion_manifest(path)


def test_entry_missing_required_field_raises_clearly(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    incomplete = dict(VALID_ENTRY)
    del incomplete["evidence"]
    _write_manifest(path, {"excluded_matches": [incomplete]})
    with pytest.raises(ExclusionManifestError, match="missing required field"):
        load_exclusion_manifest(path)


def test_entry_with_non_integer_match_id_raises_clearly(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    bad = dict(VALID_ENTRY, match_id="not-an-int")
    _write_manifest(path, {"excluded_matches": [bad]})
    with pytest.raises(ExclusionManifestError, match="non-integer match_id"):
        load_exclusion_manifest(path)


def test_entry_with_boolean_match_id_raises_clearly(tmp_path) -> None:
    # bool is technically an int subclass in Python -- must be rejected explicitly.
    path = tmp_path / "excluded.json"
    bad = dict(VALID_ENTRY, match_id=True)
    _write_manifest(path, {"excluded_matches": [bad]})
    with pytest.raises(ExclusionManifestError, match="non-integer match_id"):
        load_exclusion_manifest(path)


def test_duplicate_match_id_raises_clearly(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    _write_manifest(path, {"excluded_matches": [VALID_ENTRY, dict(VALID_ENTRY)]})
    with pytest.raises(ExclusionManifestError, match="duplicate match_id"):
        load_exclusion_manifest(path)


def test_entry_not_an_object_raises_clearly(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    _write_manifest(path, {"excluded_matches": ["not an object"]})
    with pytest.raises(ExclusionManifestError, match="not an object"):
        load_exclusion_manifest(path)


# --- check_not_excluded -------------------------------------------------------

def test_check_not_excluded_is_a_no_op_for_an_unlisted_match(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    _write_manifest(path, {"excluded_matches": [VALID_ENTRY]})
    manifest = load_exclusion_manifest(path)
    check_not_excluded(999, manifest)  # must not raise


def test_check_not_excluded_raises_excluded_match_error_with_reason(tmp_path) -> None:
    path = tmp_path / "excluded.json"
    _write_manifest(path, {"excluded_matches": [VALID_ENTRY]})
    manifest = load_exclusion_manifest(path)
    with pytest.raises(ExcludedMatchError) as exc_info:
        check_not_excluded(123, manifest)
    assert exc_info.value.excluded.match_id == 123
    assert "test_reason" in str(exc_info.value)


def test_check_not_excluded_never_touches_disk(tmp_path) -> None:
    # Pure in-memory check -- must not read or write anything.
    manifest = {123: ExcludedMatch(123, "r", "f", "e", "d", "rep")}
    check_not_excluded(999, manifest)
    with pytest.raises(ExcludedMatchError):
        check_not_excluded(123, manifest)


# --- the real, shipped manifest -----------------------------------------------

def test_real_manifest_is_valid_and_contains_documented_exclusions() -> None:
    manifest = load_exclusion_manifest(DEFAULT_MANIFEST_PATH)
    assert 3845506 in manifest
    assert manifest[3845506].reason == "corrupted_three_sixty_json"
    assert 3923880 in manifest
    assert manifest[3923880].reason == "insufficient_three_sixty_coverage_for_team_reference_resolution"
    for entry in manifest.values():
        # Every entry must cite a report that actually exists on disk --
        # an exclusion must be independently verifiable, not just asserted.
        from pathlib import Path

        report_path = DEFAULT_MANIFEST_PATH.resolve().parents[1] / entry.report
        assert report_path.exists(), f"cited report does not exist: {entry.report}"
