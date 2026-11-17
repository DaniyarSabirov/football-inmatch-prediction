"""Tests for the --match-id / --minute / --event-type selection of scripts/demo_explain_prediction.py.

Unit tests use a synthetic prediction table. Integration tests read the stored experiment-2 results
(read-only) and are skipped when they are not present.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "demo_explain_prediction.py"
_spec = importlib.util.spec_from_file_location("demo_explain_prediction", SCRIPT_PATH)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

VOCAB = ["Pass", "Carry", "Shot", "Goal Keeper"]


def _rows() -> pd.DataFrame:
    recs = [  # match, event, fold, period, time, index, current, next
        (10, "e1", 0, 1, "00:12:01.000", 5, "Pass", "Carry"),
        (10, "e2", 0, 1, "00:12:30.500", 6, "Carry", "Shot"),
        (10, "e3", 0, 1, "00:12:59.999", 7, "Shot", "Goal Keeper"),
        (10, "e4", 0, 1, "00:47:10.000", 90, "Pass", "Pass"),      # first-half stoppage time: minute 47
        (10, "e5", 0, 2, "00:02:05.000", 120, "Carry", "Pass"),    # second half 02:05 -> minute 47
        (10, "e6", 0, 2, "00:30:00.000", 300, "Pass", "Carry"),    # minute 75
        (20, "e7", 1, 1, "00:12:10.000", 8, "Carry", "Pass"),
        (20, "e8", 1, 3, "00:01:00.000", 900, "Shot", "Goal Keeper"),  # extra time 1 -> minute 91
    ]
    cols = ["match_id", "event_id", "fold_id", "prediction_period", "prediction_time",
            "prediction_event_index", "current_event", "true_next_event"]
    return pd.DataFrame(recs, columns=cols).sample(frac=1.0, random_state=0)  # shuffled on purpose


SPLIT = {"final_test_match_ids": [99], "final_test_status": "UNTOUCHED",
         "development_match_ids": [10, 20, 30], "tier_a": {"match_ids": [10]}, "tier_b": {"match_ids": [10, 20, 30]}}


# ---------------------------------------------------------------- minute interpretation
@pytest.mark.parametrize("period, ts, expected", [
    (1, "00:00:00.000", 0), (1, "00:00:59.999", 0), (1, "00:12:18.859", 12), (1, "00:47:10.000", 47),
    (2, "00:00:00.000", 45), (2, "00:02:05.000", 47), (3, "00:01:00.000", 91), (4, "00:14:59.000", 119),
    (1, "01:00:00.000", 60),
])
def test_match_clock_minute(period, ts, expected):
    assert demo.match_clock_minute(period, ts) == expected


def test_match_clock_minute_rejects_unknown_period():
    with pytest.raises(ValueError):
        demo.match_clock_minute(7, "00:00:01.000")


# ---------------------------------------------------------------- event type
def test_event_type_is_case_insensitive_and_canonicalised():
    assert demo.canonical_event_type("goal keeper", VOCAB, "t") == "Goal Keeper"


def test_event_type_outside_vocabulary_is_explained():
    with pytest.raises(demo.SelectionError, match="not in the t vocabulary.*Valid types: Pass, Carry"):
        demo.canonical_event_type("Foul Won", VOCAB, "t")


# ---------------------------------------------------------------- match permission
def test_final_test_match_is_refused():
    with pytest.raises(demo.SelectionError, match="refused: match 99 is a final-test match"):
        demo.check_match_permitted(99, SPLIT, "tier_b")


def test_unknown_match_is_explained():
    with pytest.raises(demo.SelectionError, match="not in the development set"):
        demo.check_match_permitted(55, SPLIT, "tier_b")


def test_match_outside_tier_is_explained():
    with pytest.raises(demo.SelectionError, match="not part of tier_a"):
        demo.check_match_permitted(20, SPLIT, "tier_a")
    demo.check_match_permitted(20, SPLIT, "tier_b")  # no error


# ---------------------------------------------------------------- selection
def test_unique_match():
    out = demo.select_candidates(_rows(), match_id=10, minute=12, event_type="Shot")
    assert out["event_id"].tolist() == ["e3"]


def test_minute_uses_current_event_timestamp_floor():
    out = demo.select_candidates(_rows(), match_id=10, minute=12)
    assert out["event_id"].tolist() == ["e1", "e2", "e3"]  # 12:59.999 is still minute 12


def test_multiple_matches_are_sorted_deterministically_and_include_both_periods():
    a = demo.select_candidates(_rows(), minute=47)
    b = demo.select_candidates(_rows().iloc[::-1], minute=47)
    assert a["event_id"].tolist() == b["event_id"].tolist() == ["e4", "e5"]
    assert a["prediction_period"].tolist() == [1, 2]


def test_event_type_matches_current_not_next_event():
    out = demo.select_candidates(_rows(), event_type="Shot")
    assert set(out["current_event"]) == {"Shot"}
    assert out["event_id"].tolist() == ["e3", "e8"]  # e2 has true next = Shot but current = Carry


def test_filters_combine_with_event_id():
    assert demo.select_candidates(_rows(), event_id="e2", match_id=10)["event_id"].tolist() == ["e2"]
    with pytest.raises(demo.SelectionError, match="match 20 has no rows.*containing event e2"):
        demo.select_candidates(_rows(), event_id="e2", match_id=20)


def test_no_rows_for_match():
    with pytest.raises(demo.SelectionError, match="match 30 has no rows"):
        demo.select_candidates(_rows(), match_id=30)


def test_no_rows_for_minute_reports_available_range():
    with pytest.raises(demo.SelectionError, match=r"minute 200 .*minutes with rows: 12-75"):
        demo.select_candidates(_rows(), match_id=10, minute=200)


def test_negative_minute():
    with pytest.raises(demo.SelectionError, match=">= 0"):
        demo.select_candidates(_rows(), minute=-1)


def test_minute_present_but_event_type_absent_explains_where_it_occurs():
    with pytest.raises(demo.SelectionError, match=r"minute 75 has 1 .*none with current event Shot.*minutes \[12\]"):
        demo.select_candidates(_rows(), match_id=10, minute=75, event_type="Shot")


def test_event_type_absent_in_match():
    with pytest.raises(demo.SelectionError, match="no prediction row in match 20 has current event Goal Keeper"):
        demo.select_candidates(_rows(), match_id=20, event_type="Goal Keeper")


def test_format_candidates_lists_ids_and_timestamps_and_truncates():
    df = demo.select_candidates(_rows())
    text = demo.format_candidates(df, limit=3)
    assert text.startswith("8 prediction rows match") and "first 3 shown" in text
    assert "e1" in text and "00:12:01.000" in text and "e8" not in text


# ---------------------------------------------------------------- integration with stored results (read-only)
EXP2 = PROJECT_ROOT / "results" / "markov_hmm_cost_sensitive_v2"
needs_results = pytest.mark.skipif(not (EXP2 / "predictions.parquet").exists() or
                                   not (EXP2 / "split_manifest.json").exists(),
                                   reason="experiment-2 results not present")


@needs_results
def test_cli_refuses_final_test_match():
    split = json.loads((EXP2 / "split_manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(SystemExit) as exc:
        demo.main(["--match-id", str(split["final_test_match_ids"][0])])
    assert "final-test match" in str(exc.value.code)


@needs_results
def test_cli_lists_candidates_when_ambiguous(capsys):
    split = json.loads((EXP2 / "split_manifest.json").read_text(encoding="utf-8"))
    match_id = split["tier_b"]["match_ids"][0]
    with pytest.raises(SystemExit) as exc:
        demo.main(["--match-id", str(match_id), "--minute", "10"])
    assert exc.value.code == demo.EXIT_AMBIGUOUS
    out = capsys.readouterr().out
    assert "prediction rows match; rerun with --event-id" in out and str(match_id) in out


@needs_results
def test_stored_predictions_contain_no_final_test_match():
    import pyarrow.dataset as ds
    split = json.loads((EXP2 / "split_manifest.json").read_text(encoding="utf-8"))
    ids = set(ds.dataset(EXP2 / "predictions.parquet").to_table(columns=["match_id"])["match_id"].to_pylist())
    assert not ids & set(split["final_test_match_ids"])
