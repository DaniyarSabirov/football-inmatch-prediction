"""Experiment 3: causal features, pitch geometry and the actions15 companion rules."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.sequence_models.v3_features import (
    DANGER_MARGIN,
    DANGER_X,
    causal_raw_features,
    distance_to_goal,
    history_tokens,
    in_danger_area,
    possession_bins,
    shooting_angle,
    spatial_features,
)
from src.sequence_models.v3_taxonomy import COMPANION, companion_rules, taxonomy_event_table, taxonomy_table


def raw(rows):
    """rows: (type, team, t, x, subtype, outcome, related_to_previous)."""
    out = []
    for i, (typ, team, t, x, sub, outc, rel) in enumerate(rows):
        out.append({"match_id": 1, "period": 1, "list_position": i, "index": i + 1, "event_id": f"e{i}", "type": typ,
                    "team_id": team, "t_seconds": t, "timestamp": f"00:00:{int(t):02d}.000", "location_x": x,
                    "location_y": 40.0, "subtype": sub, "outcome": outc, "related_events": f"e{i - 1}" if rel else "",
                    "has_360": True, "minute": 0, "possession": 1, "possession_team_id": team})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ geometry
def test_distance_and_angle_by_hand():
    assert distance_to_goal(np.array([108.0]), np.array([40.0]))[0] == pytest.approx(12.0)
    assert distance_to_goal(np.array([117.0]), np.array([36.0]))[0] == pytest.approx(5.0)
    # penalty spot (108, 40): angle = 2 * atan(4 / 12)
    assert shooting_angle(np.array([108.0]), np.array([40.0]))[0] == pytest.approx(2 * math.atan(4 / 12))
    # on the goal line beside the post the angle is 0; centred far away it shrinks
    assert shooting_angle(np.array([120.0]), np.array([30.0]))[0] == pytest.approx(0.0, abs=1e-12)
    assert shooting_angle(np.array([60.0]), np.array([40.0]))[0] < shooting_angle(np.array([100.0]), np.array([40.0]))[0]


def test_danger_area_uses_five_metres_in_statsbomb_units():
    assert DANGER_MARGIN == pytest.approx(5 / 0.9144)
    assert DANGER_X == pytest.approx(102 - 5 / 0.9144)
    x = np.array([96.6, 96.4, 110.0, 110.0, 110.0])
    y = np.array([40.0, 40.0, 12.6, 12.4, 67.4])
    assert in_danger_area(x, y).tolist() == [True, False, True, False, True]


def test_zone_bins_and_unknown():
    f = spatial_features(np.array([19.9, 20.0, 99.9, 100.0, np.nan]), np.array([40.0, 10.0, 40.0, 70.0, np.nan]))
    assert f["b6"].tolist() == [0, 1, 4, 5, -1]
    assert f["z3"].tolist() == [0, 0, 2, 2, -1]
    assert f["g12"].tolist() == [1, 2, 9, 10, -1]        # band*2 + central(|y-40|<=22)
    assert f["d7"][-1] == -1 and f["z3d"][-1] == -1


# ------------------------------------------------------------------ causal possession / time
SEQ = [("Pass", 1, 0.0, 30.0, None, None, False), ("Ball Receipt*", 1, 1.0, 40.0, None, None, True),
       ("Pressure", 2, 1.5, 70.0, None, None, False), ("Carry", 1, 1.0, 45.0, None, None, False),
       ("Pass", 1, 4.0, 60.0, None, None, False), ("Ball Recovery", 2, 6.0, 50.0, None, None, False),
       ("Carry", 2, 6.0, 55.0, None, None, False), ("Shot", 2, 9.0, 100.0, None, None, False)]


def test_causal_possession_by_hand():
    f = causal_raw_features(raw(SEQ))
    assert f["causal_possession_id"].tolist() == [1, 1, 1, 1, 1, 2, 2, 2]
    # Pressure by team 2 does not change possession; its x is mirrored into team 1's frame
    assert f["x_possession_frame"].iloc[2] == pytest.approx(50.0)
    assert f["possession_progression"].iloc[4] == pytest.approx(30.0)      # 60 - 30
    assert f["possession_progression"].iloc[7] == pytest.approx(50.0)      # 100 - 50
    assert f["possession_events_so_far"].tolist() == [1, 2, 3, 4, 5, 1, 2, 3]
    assert f["possession_duration_s"].iloc[7] == pytest.approx(3.0)
    # the Carry stamped 1.0 after the Pressure at 1.5 runs backwards: duration unknown, not negative
    assert np.isnan(f["possession_duration_s"].iloc[3]) and bool(f["timestamp_corrupted"].iloc[3])


def test_causal_features_are_truncation_invariant():
    full = causal_raw_features(raw(SEQ))
    for cut in range(1, len(SEQ)):
        part = causal_raw_features(raw(SEQ[:cut]))
        pd.testing.assert_frame_equal(part, full.iloc[:cut].reset_index(drop=True), check_dtype=False)


def test_possession_bins_cover_unknown():
    f = causal_raw_features(raw(SEQ)).assign(minute=[0] * 8, period=[1] * 8)
    b = possession_bins(f)
    assert b["poss_dur4"][3] == -1 and b["poss_n4"].tolist() == [0, 1, 1, 2, 2, 0, 1, 1]


def test_history_tokens_start_and_possession_reset():
    sym = np.array([0, 1, 2, 3, 4])
    seq = np.array(["a", "a", "a", "a", "b"])
    poss = np.array([1, 1, 2, 2, 2])
    h = history_tokens(sym, seq, poss, n_symbols=10, depth=3)
    assert h["prev1"].tolist() == [10, 0, 1, 2, 10] and h["prev3"].tolist() == [10, 10, 10, 0, 10]
    r = history_tokens(sym, seq, poss, n_symbols=10, depth=3, reset_at_possession=True)
    assert r["prev1"].tolist() == [10, 0, 11, 2, 10]          # event 2 starts a new possession
    assert r["prev2"].tolist() == [10, 10, 11, 11, 10]


# ------------------------------------------------------------------ taxonomy
COMP = [("Shot", 1, 10.0, 105.0, "Open Play", "Saved", False), ("Goal Keeper", 2, 11.0, 2.0, "Shot Saved", None, True),
        ("Goal Keeper", 2, 12.0, 2.0, "Shot Faced", None, True), ("Dispossessed", 1, 20.0, 50.0, None, None, False),
        ("Duel", 2, 20.0, 70.0, "Tackle", "Won", True), ("Dribble", 1, 25.0, 60.0, None, "Incomplete", False),
        ("Duel", 2, 25.0, 60.0, "Tackle", "Won", True), ("Duel", 2, 30.0, 60.0, "Aerial Lost", None, False),
        ("50/50", 1, 35.0, 50.0, None, "Won", False), ("50/50", 2, 35.0, 70.0, None, "Lost", True),
        ("Pass", 1, 36.0, 50.0, None, None, False)]


def test_companion_rules_by_construction():
    rules = companion_rules(raw(COMP)).tolist()
    assert rules == ["", "", "K1", "", "K2", "", "K2", "", "", "K3", ""]


def test_actions15_skips_companions_but_keeps_actions():
    t = taxonomy_table(raw(COMP), "actions15")
    assert t["raw_type"].tolist() == ["Shot", "Goal Keeper", "Dispossessed", "Dribble", "Duel", "50/50", "Pass"]
    assert taxonomy_event_table(raw(COMP), "actions15")["type"].tolist().count(COMPANION) == 4
    d = taxonomy_table(raw(COMP), "scenarioD15")
    assert len(d) == len(COMP)       # Scenario D keeps every record
