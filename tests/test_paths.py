"""Tests for canonical project paths."""

from src.utils.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR


def test_project_directories_are_resolved_from_repository_root() -> None:
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RESULTS_DIR == PROJECT_ROOT / "results"
    assert PROJECT_ROOT.name == "football-inmatch-prediction"
