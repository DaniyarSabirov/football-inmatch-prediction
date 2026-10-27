"""The Law 8 kick-off diagnostic only reports (scripts/diagnose_kickoff_law8.py)."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_kickoff_law8 import classify_kickoff  # noqa: E402


def _kickoff(slovakia_x, belgium_x):
    """Kicking team (teammate=True) in its own frame: own half is x < 60."""
    players = [{"teammate": True, "actor": True, "keeper": False, "location": [60.0, 40.0]}]
    players += [{"teammate": True, "actor": False, "keeper": False, "location": [x, 10.0 + 6 * i]} for i, x in enumerate(slovakia_x)]
    players += [{"teammate": False, "actor": False, "keeper": False, "location": [x, 12.0 + 6 * i]} for i, x in enumerate(belgium_x)]
    return players


def test_lined_up_kickoff_is_consistent():
    assert classify_kickoff(_kickoff([40, 45, 50, 55], [65, 70, 75, 80]), True)["verdict"] == "consistent"


def test_slight_encroachment_is_only_minor_not_evidence_of_an_error():
    verdict = classify_kickoff(_kickoff([40, 45, 50, 63], [65, 70, 57, 80]), True)
    assert verdict["verdict"] == "minor" and verdict["deep"] == 0


def test_several_players_deep_in_the_wrong_half_is_suspicious():
    verdict = classify_kickoff(_kickoff([40, 75, 80, 85], [30, 70, 75, 80]), True)
    assert verdict["verdict"] == "suspicious" and verdict["deep"] == 4


def test_unknown_orientation_or_too_few_players_is_inconclusive():
    assert classify_kickoff(_kickoff([75, 80, 85, 90], [30, 35]), False)["verdict"] == "inconclusive_orientation"
    assert classify_kickoff(_kickoff([75, 80], [30]), True)["verdict"] == "inconclusive_few_players"


def test_players_on_the_halfway_line_and_the_actor_are_not_judged():
    verdict = classify_kickoff(_kickoff([40, 45, 50, 61], [59, 70, 75, 80]), True)
    assert verdict["verdict"] == "consistent" and verdict["evaluated"] == 6


def test_diagnostic_never_modifies_its_input():
    players = _kickoff([40, 75, 80, 85], [30, 70, 75, 80])
    before = copy.deepcopy(players)
    classify_kickoff(players, True)
    assert players == before


def test_production_code_does_not_use_the_diagnostic():
    root = Path(__file__).resolve().parents[1] / "src"
    for path in root.glob("*.py"):
        assert "diagnose_kickoff_law8" not in path.read_text(encoding="utf-8"), path
