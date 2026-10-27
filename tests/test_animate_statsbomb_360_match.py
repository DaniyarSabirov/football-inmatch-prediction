"""Tests for src/animate_statsbomb_360_match.py's period-aware physical
pitch orientation calibration and match-time display (added 2026-10-21;
see docs/full_physical_pitch_orientation_audit.md for the forensic
evidence: a single, period-independent --physical-reference-team renders
the second half backwards relative to a real broadcast camera, because
real teams swap ends at halftime).

Covers the pure-data transformation/formatting functions -- not
matplotlib rendering, CLI parsing, or file I/O, which were verified
manually/visually (actual PNG frames extracted from real generated GIFs
and inspected; see the audit doc's verification section).
"""

from pathlib import Path

import pytest

from src.animate_statsbomb_360_match import (
    JoinedFrame,
    MatchFrames,
    apply_team_normalization_to_match_spectator_facing,
    apply_team_normalization_to_match_with_physical_calibration,
    determine_team_style,
    format_match_clock,
    infer_attacking_directions,
    period_display_name,
    summarize_physical_team_normalization,
)
from src.team_normalization import TEAM_NORMALIZATION_MIRRORED, TEAM_NORMALIZATION_REFERENCE

HOME_ID = 100
AWAY_ID = 200


def _event(event_id, team_id, team_name, period, location, minute=0, second=0, timestamp="00:00:00.000"):
    return {
        "id": event_id,
        "team": {"id": team_id, "name": team_name},
        "type": {"name": "Pass"},
        "period": period,
        "minute": minute,
        "second": second,
        "timestamp": timestamp,
        "location": location,
    }


def _frame(event_id, actor_location):
    return {
        "event_uuid": event_id,
        "visible_area": [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
        "freeze_frame": [
            {"teammate": True, "actor": True, "keeper": False, "location": actor_location},
        ],
    }


def _match_frames(joined_list):
    return MatchFrames(
        event_path=Path("unused-events.json"),
        three_sixty_path=Path("unused-360.json"),
        event_count=len(joined_list),
        three_sixty_count=len(joined_list),
        matched_count=len(joined_list),
        usable_count=len(joined_list),
        usable_frames=joined_list,
    )


# --- format_match_clock -------------------------------------------------

def test_format_match_clock_period_1_matches_timestamp_minute() -> None:
    event = _event(
        "e1", HOME_ID, "Home", period=1, location=[10.0, 10.0],
        minute=31, second=24, timestamp="00:31:24.988",
    )
    assert format_match_clock(event) == "31:24.988"


def test_format_match_clock_period_2_uses_true_match_clock_not_plus_45() -> None:
    # StatsBomb's own minute/second already account for first-half added
    # time -- period 2 can legitimately start at minute=45 even if period 1
    # ran past 45 (confirmed empirically: match 3930160's period 1 ran to
    # minute 47 after 2:57 of stoppage time). A naive "period-relative +
    # 45" reconstruction would be wrong whenever first-half stoppage time
    # was non-zero; using the real minute/second fields directly avoids
    # that entirely.
    event = _event(
        "e2", HOME_ID, "Home", period=2, location=[10.0, 10.0],
        minute=55, second=34, timestamp="00:10:34.456",
    )
    assert format_match_clock(event) == "55:34.456"


def test_format_match_clock_never_hour_wraps_past_minute_59() -> None:
    # Regression test for the exact bug reported against match 3835342:
    # the clock showed "1:05:59.487" for minute=65 because an earlier
    # implementation used divmod(minute, 60) to introduce an hours
    # component. Football match clocks never do this -- minute=65 must
    # render as "65:59", not "1:05:59". period=2 here so minute=65 is far
    # below period 2's nominal end (90) and is NOT stoppage time.
    event = _event(
        "e3b", HOME_ID, "Home", period=2, location=[10.0, 10.0],
        minute=65, second=59, timestamp="00:20:59.487",
    )
    assert format_match_clock(event) == "65:59.487"


def test_format_match_clock_does_not_hour_wrap_even_for_long_shootouts() -> None:
    # Period 5 has no PERIOD_NOMINAL_END_MINUTE entry, so no stoppage-time
    # "+" notation is applied -- the raw minute is shown directly, still
    # never hour-wrapped.
    event = _event(
        "e3", HOME_ID, "Home", period=5, location=[10.0, 10.0],
        minute=125, second=5, timestamp="00:05:05.000",
    )
    assert format_match_clock(event) == "125:05.000"


def test_format_match_clock_zero_pads_single_digit_minutes() -> None:
    event = _event(
        "e5", HOME_ID, "Home", period=1, location=[10.0, 10.0],
        minute=5, second=12, timestamp="00:05:12.000",
    )
    assert format_match_clock(event) == "05:12.000"


@pytest.mark.parametrize(
    "period,minute,second,expected",
    [
        (1, 44, 59, "44:59"),     # just before the nominal first-half end: plain
        (1, 45, 0, "45:00"),      # exactly at the nominal end: still plain, not "+0"
        (1, 46, 23, "45+1:23"),   # 1 minute of first-half stoppage
        (1, 48, 15, "45+3:15"),   # 3 minutes of first-half stoppage
        (2, 45, 0, "45:00"),      # second half kickoff: plain, NOT first-half stoppage
        (2, 65, 59, "65:59"),     # ordinary second-half time, far from 90
        (2, 89, 59, "89:59"),     # just before the nominal second-half end: plain
        (2, 90, 0, "90:00"),      # exactly at the nominal end: plain
        (2, 92, 8, "90+2:08"),    # 2 minutes of second-half stoppage
        (3, 104, 59, "104:59"),   # extra time, 1st half, just before nominal end: plain
        (3, 106, 30, "105+1:30"), # extra time, 1st half stoppage
        (4, 119, 59, "119:59"),   # extra time, 2nd half, just before nominal end: plain
        (4, 121, 45, "120+1:45"), # extra time, 2nd half stoppage
    ],
)
def test_format_match_clock_stoppage_time_notation(period, minute, second, expected) -> None:
    event = _event(
        "e-stoppage", HOME_ID, "Home", period=period, location=[10.0, 10.0],
        minute=minute, second=second, timestamp=f"00:{minute:02d}:{second:02d}.000",
    )
    assert format_match_clock(event) == f"{expected}.000"


def test_format_match_clock_keys_stoppage_off_actual_period_not_raw_minute() -> None:
    # A period-2 event at minute=46 must NOT be misread as first-half
    # stoppage time (period 1's nominal end is 45) merely because 46 is
    # numerically close to 45 -- it must be compared against period 2's
    # OWN nominal end (90), where 46 is plain, ordinary second-half time.
    period_1_event = _event(
        "e-p1", HOME_ID, "Home", period=1, location=[10.0, 10.0],
        minute=46, second=0, timestamp="00:01:00.000",
    )
    period_2_event = _event(
        "e-p2", HOME_ID, "Home", period=2, location=[10.0, 10.0],
        minute=46, second=0, timestamp="00:01:00.000",
    )
    assert format_match_clock(period_1_event) == "45+1:00.000"
    assert format_match_clock(period_2_event) == "46:00.000"


def test_format_match_clock_missing_fields_returns_unknown() -> None:
    event = {"id": "e4", "timestamp": "00:01:00.000"}
    assert format_match_clock(event) == "Unknown"


# --- apply_team_normalization_to_match_with_physical_calibration --------

def test_physical_calibration_applies_period_1_reference_unchanged() -> None:
    period_1_home = _event("p1h", HOME_ID, "Home", period=1, location=[20.0, 30.0])
    period_1_away = _event("p1a", AWAY_ID, "Away", period=1, location=[20.0, 30.0])
    match = _match_frames([
        JoinedFrame(event=period_1_home, frame_360=_frame("p1h", [20.0, 30.0])),
        JoinedFrame(event=period_1_away, frame_360=_frame("p1a", [20.0, 30.0])),
    ])
    all_events = [period_1_home, period_1_away]

    normalized, reference_by_period = apply_team_normalization_to_match_with_physical_calibration(
        match, all_events, "Home"
    )

    assert reference_by_period == {1: HOME_ID}
    home_result = next(j for j in normalized.usable_frames if j.event["id"] == "p1h")
    away_result = next(j for j in normalized.usable_frames if j.event["id"] == "p1a")
    assert home_result.event["location"] == [20.0, 30.0]  # reference team: unchanged
    assert away_result.event["location"] == pytest.approx([100.0, 50.0])  # mirrored (120-20, 80-30)


def test_physical_calibration_swaps_reference_team_in_period_2() -> None:
    period_1_home = _event("p1h", HOME_ID, "Home", period=1, location=[20.0, 30.0])
    period_2_home = _event("p2h", HOME_ID, "Home", period=2, location=[20.0, 30.0], minute=46)
    period_2_away = _event("p2a", AWAY_ID, "Away", period=2, location=[20.0, 30.0], minute=46)
    match = _match_frames([
        JoinedFrame(event=period_1_home, frame_360=_frame("p1h", [20.0, 30.0])),
        JoinedFrame(event=period_2_home, frame_360=_frame("p2h", [20.0, 30.0])),
        JoinedFrame(event=period_2_away, frame_360=_frame("p2a", [20.0, 30.0])),
    ])
    all_events = [period_1_home, period_2_home, period_2_away]

    normalized, reference_by_period = apply_team_normalization_to_match_with_physical_calibration(
        match, all_events, "Home"
    )

    # "Home" is calibrated as reference for period 1 (its own raw
    # convention already matches physical reality there). After the real
    # halftime end swap, "Away" must become reference for period 2 --
    # Home's OWN period-2 events must now be MIRRORED, even though Home
    # was unmirrored in period 1. This is the exact behavior that was
    # missing before this fix (match 3930160 kept Spain unmirrored in both
    # halves, rendering the second half backwards).
    assert reference_by_period == {1: HOME_ID, 2: AWAY_ID}
    period_2_home_result = next(j for j in normalized.usable_frames if j.event["id"] == "p2h")
    period_2_away_result = next(j for j in normalized.usable_frames if j.event["id"] == "p2a")
    assert period_2_home_result.event["location"] == pytest.approx([100.0, 50.0])  # now mirrored
    assert period_2_away_result.event["location"] == [20.0, 30.0]  # now reference


def test_summarize_physical_team_normalization_counts_per_period_correctly() -> None:
    period_1_home = _event("p1h", HOME_ID, "Home", period=1, location=[20.0, 30.0])
    period_2_home = _event("p2h", HOME_ID, "Home", period=2, location=[20.0, 30.0], minute=46)
    period_2_away = _event("p2a", AWAY_ID, "Away", period=2, location=[20.0, 30.0], minute=46)
    match = _match_frames([
        JoinedFrame(event=period_1_home, frame_360=_frame("p1h", [20.0, 30.0])),
        JoinedFrame(event=period_2_home, frame_360=_frame("p2h", [20.0, 30.0])),
        JoinedFrame(event=period_2_away, frame_360=_frame("p2a", [20.0, 30.0])),
    ])
    reference_by_period = {1: HOME_ID, 2: AWAY_ID}

    counts = summarize_physical_team_normalization(match, reference_by_period)

    # period 1 Home (reference) + period 2 Away (reference) = 2 REFERENCE;
    # period 2 Home (mirrored, since Away is period-2's reference) = 1 MIRRORED.
    assert counts[TEAM_NORMALIZATION_REFERENCE] == 2
    assert counts[TEAM_NORMALIZATION_MIRRORED] == 1


# --- period_display_name -------------------------------------------------

@pytest.mark.parametrize(
    "period,expected",
    [
        (1, "1st Half"),
        (2, "2nd Half"),
        (3, "Extra Time, 1st Half"),
        (4, "Extra Time, 2nd Half"),
        (5, "Penalty Shootout"),
        (6, "Period 6"),
        (None, "Unknown Period"),
        (True, "Unknown Period"),  # bool must never be treated as a valid period
    ],
)
def test_period_display_name(period, expected) -> None:
    assert period_display_name(period) == expected


# --- apply_team_normalization_to_match_spectator_facing ------------------
# Root-cause fix (2026-10-22, docs/full_visualization_orientation_and_clock_audit_2026-10-22.md):
# the UNCALIBRATED default rendering path must ALSO apply the real,
# Laws-of-the-Game halftime end swap -- "teams change ends at halftime"
# is true regardless of whether any match's absolute left/right has been
# physically verified. Reproduces the exact Iceland vs France (match
# 3835342) scenario with synthetic data: period-1 kickoff team (the
# automatic baseline) must become MIRRORED in period 2, even with NO
# physical calibration supplied.

def test_spectator_facing_default_swaps_reference_team_in_period_2_without_calibration() -> None:
    period_1_kickoff_team = _event("p1", HOME_ID, "Home", period=1, location=[20.0, 30.0])
    period_2_kickoff_team = _event("p2", AWAY_ID, "Away", period=2, location=[20.0, 30.0], minute=46)
    match = _match_frames([
        JoinedFrame(event=period_1_kickoff_team, frame_360=_frame("p1", [20.0, 30.0])),
        JoinedFrame(event=period_2_kickoff_team, frame_360=_frame("p2", [20.0, 30.0])),
    ])
    all_events = [period_1_kickoff_team, period_2_kickoff_team]

    normalized, reference_by_period = apply_team_normalization_to_match_spectator_facing(match, all_events)

    # Automatic baseline (first-seen team, Home/HOME_ID) is reference for
    # period 1, but must be swapped to Away/AWAY_ID for period 2 -- NO
    # --physical-reference-team or registry entry was supplied here.
    assert reference_by_period == {1: HOME_ID, 2: AWAY_ID}

    p1_result = next(j for j in normalized.usable_frames if j.event["id"] == "p1")
    p2_result = next(j for j in normalized.usable_frames if j.event["id"] == "p2")
    assert p1_result.event["location"] == [20.0, 30.0]  # period-1 baseline team: unchanged
    # period-2 kickoff team (Away) is now period 2's reference: ALSO unchanged
    assert p2_result.event["location"] == [20.0, 30.0]


def test_spectator_facing_default_mirrors_period_1_baseline_team_in_period_2() -> None:
    # The team that WAS reference (unmirrored) in period 1 must become
    # MIRRORED once rendered in period 2 -- this is the exact defect
    # reported for match 3835342 (Iceland, period-1 kickoff/baseline team,
    # rendered on the wrong side in period 2 before this fix).
    period_1_baseline = _event("p1", HOME_ID, "Home", period=1, location=[20.0, 30.0])
    period_2_same_team = _event("p2same", HOME_ID, "Home", period=2, location=[20.0, 30.0], minute=46)
    period_2_other_team = _event("p2other", AWAY_ID, "Away", period=2, location=[20.0, 30.0], minute=46)
    match = _match_frames([
        JoinedFrame(event=period_1_baseline, frame_360=_frame("p1", [20.0, 30.0])),
        JoinedFrame(event=period_2_same_team, frame_360=_frame("p2same", [20.0, 30.0])),
        JoinedFrame(event=period_2_other_team, frame_360=_frame("p2other", [20.0, 30.0])),
    ])
    all_events = [period_1_baseline, period_2_same_team, period_2_other_team]

    normalized, reference_by_period = apply_team_normalization_to_match_spectator_facing(match, all_events)

    assert reference_by_period[1] == HOME_ID
    assert reference_by_period[2] == AWAY_ID  # swapped

    p2_home_result = next(j for j in normalized.usable_frames if j.event["id"] == "p2same")
    assert p2_home_result.event["location"] == pytest.approx([100.0, 50.0])  # now mirrored (120-20, 80-30)


def test_spectator_facing_default_period_1_only_is_unaffected() -> None:
    # A period-1-only render must be byte-identical to before this fix --
    # the swap only activates for periods in PHYSICALLY_FLIPPED_PERIODS
    # (2, 4), never for period 1 itself.
    period_1_home = _event("p1h", HOME_ID, "Home", period=1, location=[20.0, 30.0])
    period_1_away = _event("p1a", AWAY_ID, "Away", period=1, location=[20.0, 30.0])
    match = _match_frames([
        JoinedFrame(event=period_1_home, frame_360=_frame("p1h", [20.0, 30.0])),
        JoinedFrame(event=period_1_away, frame_360=_frame("p1a", [20.0, 30.0])),
    ])
    all_events = [period_1_home, period_1_away]

    normalized, reference_by_period = apply_team_normalization_to_match_spectator_facing(match, all_events)

    assert reference_by_period == {1: HOME_ID}
    home_result = next(j for j in normalized.usable_frames if j.event["id"] == "p1h")
    away_result = next(j for j in normalized.usable_frames if j.event["id"] == "p1a")
    assert home_result.event["location"] == [20.0, 30.0]
    assert away_result.event["location"] == pytest.approx([100.0, 50.0])


# --- Real-data regression tests (2026-10-22) ------------------------------
# Added after the Russia vs Belgium "orientation regression" report
# (docs/russia_belgium_orientation_regression_2026-10-22.md). The first
# round of investigation concluded (incorrectly -- see
# docs/belgium_russia_broadcast_orientation_forensic_audit.md) that no
# coordinate change was needed, based only on a goalkeeper-position
# self-consistency check (Courtois near his own labeled goal). That check
# is necessary but NOT sufficient: a uniformly-flipped render would pass
# it too, because a team's own keeper is always drawn next to that same
# team's own label regardless of which absolute side the match is
# rendered on. A second round, using independent opponent-goal evidence
# (the goalkeeper StatsBomb names in the shot that beat him, cross-checked
# against the real broadcast via clock + scoreline), established that the
# automatic/default calibration is actually backwards for this match, and
# that Belgium (not the automatic default of Russia) must be the period-1
# physical reference team. See
# docs/belgium_russia_broadcast_orientation_forensic_audit.md.

def test_russia_belgium_courtois_lands_at_own_goal_under_automatic_default_too() -> None:
    # Match 3788743 (Russia vs Belgium, UEFA Euro 2020), event
    # bf4fa305-e059-461b-881d-b068506dcfb3: Thibaut Courtois (Belgium's
    # own goalkeeper) taking a pass from inside his own box, period 2,
    # 71:54.418. This documents that the WITHIN-TEAM self-consistency
    # check (keeper near his own team's goal label) passes under the
    # automatic/uncalibrated default -- this was real, but it was not
    # proof of absolute broadcast orientation, only of internal
    # consistency. Kept as a regression guard on that narrower property.
    from src.animate_statsbomb_360_match import load_and_join_match

    match = load_and_join_match(3788743)
    all_events = [j.event for j in match.usable_frames]
    normalized, reference_by_period = apply_team_normalization_to_match_spectator_facing(match, all_events)

    joined = next(
        j for j in normalized.usable_frames
        if j.event["id"] == "bf4fa305-e059-461b-881d-b068506dcfb3"
    )
    assert joined.event["team"]["name"] == "Belgium"
    x, y = joined.event["location"]
    assert x < 20.0  # near a goal line, within his own defensive third

    team_style = determine_team_style(
        match.usable_frames, "#27A7E8", "#FF5A5F"
    )
    di = infer_attacking_directions(normalized.usable_frames, 2, team_style)
    # Whichever side Belgium's keeper lands on, the whole-period shot
    # direction inference must agree Belgium attacks the OTHER way --
    # i.e. this is a test of internal consistency, not of broadcast truth.
    keeper_side = 1 if x < 60.0 else -1
    assert di.directions.get("Belgium") == keeper_side


def test_russia_belgium_lukaku_87th_goal_matches_broadcast_under_verified_calibration() -> None:
    # Match 3788743, event b0e8f7c7-291e-4d0a-81d4-0f99f4fcd73d: Romelu
    # Lukaku's 87th-minute goal (period 2, timestamp 00:42:45.554),
    # Belgium's 3rd and last goal (final score 3-0). This is the
    # independent, non-circular evidence anchor: the shot's own embedded
    # freeze_frame names the facing goalkeeper as Anton Shunin (Russia),
    # and the real broadcast (clock ~87:43, scoreline BEL 2-0 RUS just
    # before this exact goal) shows Russia's goal on the LEFT of the
    # screen. Under the physically-calibrated path with Belgium as the
    # period-1 reference team (configs/physical_orientation_calibration.json,
    # match_id 3788743), this goal must land on the LEFT (small x),
    # matching the broadcast -- unlike the automatic default, which puts
    # it on the right (see the other test in this module).
    from src.animate_statsbomb_360_match import load_and_join_match

    match = load_and_join_match(3788743)
    all_events = [j.event for j in match.usable_frames]
    normalized, reference_by_period = apply_team_normalization_to_match_with_physical_calibration(
        match, all_events, physical_reference_team_name="Belgium"
    )

    joined = next(
        j for j in normalized.usable_frames
        if j.event["id"] == "b0e8f7c7-291e-4d0a-81d4-0f99f4fcd73d"
    )
    assert joined.event["team"]["name"] == "Belgium"
    x, y = joined.event["location"]
    # Raw location [102.5, 49.2] must be mirrored to land on the LEFT
    # (small x), matching the broadcast's "Russia's goal is on screen-left".
    assert x == pytest.approx(120.0 - 102.5, abs=0.1)
    assert y == pytest.approx(80.0 - 49.2, abs=0.1)
    assert x < 20.0

    # Internal consistency is preserved under the corrected calibration
    # too: Belgium's keeper (Courtois) must still land at Belgium's own
    # defended goal, now on the opposite (right) side from period 1.
    courtois = next(
        j for j in normalized.usable_frames
        if j.event["id"] == "bf4fa305-e059-461b-881d-b068506dcfb3"
    )
    cx, _ = courtois.event["location"]
    assert cx > 100.0


def test_iceland_france_sandy_baltimore_matches_broadcast_corroborated_side() -> None:
    # Match 3835342 (Iceland vs France, UEFA Women's Euro 2022), event
    # 42d38c43-a105-4207-a12d-f8753efcc416: Sandy Baltimore (France) pass,
    # period 2, 65:59.487 -- the case independently confirmed by the
    # project owner against real broadcast footage
    # (docs/full_physical_pitch_orientation_audit.md).
    from src.animate_statsbomb_360_match import load_and_join_match

    match = load_and_join_match(3835342)
    all_events = [j.event for j in match.usable_frames]
    normalized, reference_by_period = apply_team_normalization_to_match_spectator_facing(match, all_events)

    joined = next(
        j for j in normalized.usable_frames
        if j.event["id"] == "42d38c43-a105-4207-a12d-f8753efcc416"
    )
    assert joined.event["team"]["name"] == "France Women's"
    # Raw location [115.7, 11.3] (France's own raw convention, deep in
    # their attacking third) -- confirmed correct when France is period 2's
    # reference (unmirrored), matching the broadcast-corroborated result.
    x, y = joined.event["location"]
    assert x == pytest.approx(115.7, abs=0.1)
    assert y == pytest.approx(11.3, abs=0.1)
    assert reference_by_period[2] != reference_by_period[1]  # halftime swap is active


# --- Unambiguous goal-label text (2026-10-22) -----------------------------

def test_goal_label_text_is_unambiguous_defending_goal() -> None:
    # Regression test for the exact misreading reported for match 3788743:
    # a bare "{team} goal" label is genuinely ambiguous (could mean
    # "defends" or "attacks into") and was the most likely real cause of a
    # visual-inspection failure, even though the underlying coordinates
    # were already correct. The label must always say "defending goal"
    # explicitly, never bare "goal".
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.animate_statsbomb_360_match import add_orientation_annotations, DirectionInference, TeamStyle

    team_style = TeamStyle(
        team_one="Russia", team_two="Belgium",
        team_one_color="#27A7E8", team_two_color="#FF5A5F",
        team_one_label="RUS", team_two_label="BEL",
    )
    direction_inference = DirectionInference({"Russia": -1, "Belgium": 1}, "test")
    fig, ax = plt.subplots()
    add_orientation_annotations(ax, team_style, direction_inference)
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)

    assert "Belgium defending goal" in texts
    assert "Russia defending goal" in texts
    assert "Belgium goal" not in texts  # the old, ambiguous text must be gone
    assert "Russia goal" not in texts
