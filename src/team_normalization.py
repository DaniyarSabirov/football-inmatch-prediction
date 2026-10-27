"""Team-based coordinate normalization (Problem B) for StatsBomb 360 data.

Independent from, and applied AFTER, the event-level orientation correction in
src/orientation_correction.py (Problem A). That module answers: "is THIS
event's own freeze_frame internally consistent with its own event.location?"
This module answers a different question: "how do we place BOTH teams'
events onto one shared pitch coordinate system?"

Forensic evidence (verified across match 3902240, 3857254, 3788741; every
period including extra time; event.location, freeze_frame, and visible_area;
same-instant event pairs and related_events-linked pairs):

  Raw StatsBomb event/360 data records each event using a TEAM-RELATIVE
  attack-direction convention -- a team's own events (location AND the
  accompanying 360 snapshot) are always expressed as if that team attacks
  toward increasing x, regardless of which physical end it actually defends.
  Two different teams' events are therefore never on the same coordinate
  scale unless one side is reflected through the pitch center. This holds
  for the SAME team_id throughout the ENTIRE match (both halves, extra
  time) -- it is not period-dependent, because it is a data-bookkeeping
  convention, not a physical-side rule.

Fix: pick one team_id as the match's reference (arbitrary but deterministic
-- see resolve_reference_team_id). Every event belonging to the OTHER
team_id has ALL of its spatial data reflected through the pitch center,
(x, y) -> (120-x, 80-y): event.location, any nested end_location (pass,
carry, shot, ...), every freeze_frame player (actor/keeper/outfield alike),
and visible_area. No other field (ids, flags, type, timestamp, period,
player identity) is touched. The reference team's own events are returned
unchanged.

The (120-x, 80-y) transform itself is never re-implemented here: both the
event-level mirror (mirror_point) and the whole-snapshot rotation
(apply_correction) are reused directly from orientation_correction.py, so
there is exactly one place in the codebase that knows the mirror formula.
"""

from __future__ import annotations

from typing import Any

try:
    # Flat sibling import: how production consumers (src/*.py, run directly
    # as scripts, e.g. `python src/build_statsbomb_360_dataset.py`) see it.
    from orientation_correction import (
        ORIENTATION_ROTATED_180,
        apply_correction,
        mirror_point,
    )
except ImportError:
    # Package-style import: how the test suite (`from src.team_normalization
    # import ...`, run from the project root) sees it.
    from src.orientation_correction import (
        ORIENTATION_ROTATED_180,
        apply_correction,
        mirror_point,
    )

TEAM_NORMALIZATION_REFERENCE = "REFERENCE"
TEAM_NORMALIZATION_MIRRORED = "MIRRORED"


def resolve_reference_team_id(
    events: list[dict[str, Any]],
    physical_reference_team_name: str | None = None,
) -> int:
    """Pick one of the match's two team ids as the normalization reference.

    Deterministic, not arbitrary in the sense of being unstable: by default
    (`physical_reference_team_name=None`), the first team id encountered
    while scanning `events` in the given (chronological) order is used, so
    the same match always resolves to the same reference regardless of
    which consumer (dataset builder, animation) computes it, as long as
    both are handed a chronologically-ordered event list -- which
    StatsBomb's raw events.json, and this project's chronologically-sorted
    usable_frames, both are.

    Which of the two teams ends up "the reference" under this default rule
    is NOT itself physically meaningful (Problem B does not attempt to
    recover true physical sides) -- only that exactly one is picked and
    used consistently for the whole match. Verified directly (see
    docs/physical_pitch_orientation_validation.md): the automatic pick is
    simply whichever team happens to have the chronologically-first
    360-matched event, which in practice is just whichever team kicks off
    -- unrelated to stadium geometry or broadcast camera position. This
    means the automatically-normalized coordinate system is suitable for
    ANALYTICAL use (e.g. ML features, where only internal cross-team
    consistency matters) but is NOT validated or guaranteed to match a
    real broadcast camera's left/right for VISUALIZATION.

    `physical_reference_team_name`, when given, overrides the automatic
    pick with an explicit, externally-verified choice (e.g. confirmed
    against broadcast footage for one specific match) -- the named team
    becomes the reference (unmirrored) team instead of whichever team the
    automatic rule would have picked. Matched case-insensitively against
    the two teams found in `events`; raises ValueError if it does not
    match exactly one of them. Passing None (the default) preserves the
    automatic behavior exactly, unchanged -- existing callers (notably
    src/build_statsbomb_360_dataset.py's analytical pipeline) are
    unaffected unless they explicitly opt in.
    """
    team_ids, team_names_by_id = _two_team_ids_and_names(events)

    if physical_reference_team_name is None:
        return team_ids[0]

    target = physical_reference_team_name.casefold()
    matches = [tid for tid in team_ids if team_names_by_id.get(tid, "").casefold() == target]
    if len(matches) != 1:
        raise ValueError(
            f"physical_reference_team_name={physical_reference_team_name!r} must exactly match "
            f"(case-insensitively) exactly one of the match's two teams: "
            f"{[team_names_by_id.get(tid) for tid in team_ids]}."
        )
    return matches[0]


def _two_team_ids_and_names(events: list[dict[str, Any]]) -> tuple[list[int], dict[int, str]]:
    """Scan `events` (chronological order) for exactly two distinct team ids
    and their names. Shared by resolve_reference_team_id and
    resolve_physical_reference_team_id so both use identical team discovery.
    """
    team_ids: list[int] = []
    team_names_by_id: dict[int, str] = {}
    for event in events:
        team = event.get("team")
        if not isinstance(team, dict):
            continue
        team_id = team.get("id")
        if isinstance(team_id, bool) or not isinstance(team_id, int):
            continue
        if team_id not in team_ids:
            team_ids.append(team_id)
            name = team.get("name")
            if isinstance(name, str):
                team_names_by_id[team_id] = name
        if len(team_ids) == 2:
            break
    if len(team_ids) != 2:
        raise ValueError(f"Expected exactly two team ids in events, found {team_ids}.")
    return team_ids, team_names_by_id


# Periods in which, per the Laws of the Game, the two teams have swapped
# physical ends relative to period 1's arrangement: the second half of
# normal time (2) and the second half of extra time (4) -- both are played
# after an end-swap. Periods 1 and 3 share period 1's arrangement (3 is the
# FIRST half of extra time, played without an additional swap relative to
# period 1 -- teams swap again only going INTO period 4).
#
# Period 5 (penalty shootout) is deliberately NOT included: shootouts are
# conventionally taken at one end chosen by the referee (commonly, though
# not universally, the same end associated with period 1's arrangement),
# but this is a refereeing/venue decision, not a Law-of-the-Game-guaranteed
# swap -- treating it as "same arrangement as period 1" here is a
# documented, explicit assumption, not a verified fact, and is flagged as
# such in docs/full_physical_pitch_orientation_audit.md. It is NOT silently
# asserted as equivalent to the period-1/3 case.
#
# MEASURED LIMITS (2026-10-23, docs/final_statsbomb_360_orientation_research.md):
# the visible_area camera side shows the 1->2 and 3->4 swaps in every
# match, but period 3 starts at period 2's ends (not period 1's) in 2 of
# 30 extra-time matches, and in a shootout (period 5) both teams share ONE
# frame, so mirroring one team there puts half the kicks at the wrong goal.
# The renderer's default display layer (src/broadcast_orientation.py)
# therefore measures each period instead; this rule remains only for the
# --display-orientation legacy path and as a fallback for unmeasured periods.
PHYSICALLY_FLIPPED_PERIODS = {2, 4}


def period_requires_physical_end_swap(period: int) -> bool:
    """Return True if `period` is played after a halftime-style end swap
    relative to period 1's physical arrangement (see PHYSICALLY_FLIPPED_PERIODS).
    """
    return period in PHYSICALLY_FLIPPED_PERIODS


def resolve_physical_reference_team_id(
    events: list[dict[str, Any]],
    physical_reference_team_name: str,
    period: int,
) -> int:
    """Resolve the reference (unmirrored) team id for ONE SPECIFIC PERIOD,
    accounting for the real, physical end swap at halftime (and before the
    second half of extra time).

    `physical_reference_team_name` must name the team confirmed (e.g. via
    broadcast footage) to attack toward increasing visualized x DURING
    PERIOD 1's arrangement -- i.e. the same calibration already established
    for `resolve_reference_team_id`'s override. For a period in
    PHYSICALLY_FLIPPED_PERIODS, the OTHER team is returned instead, because
    in a real match the two teams swap which end they attack at that point
    -- a single, period-independent reference choice (as
    resolve_reference_team_id alone provides) is therefore only physically
    correct for periods sharing period 1's arrangement; it is systematically
    backwards for the others. See
    docs/full_physical_pitch_orientation_audit.md for the forensic evidence
    (confirmed on match 3930160: the renderer kept showing Spain attacking
    the same screen side in period 2 as in period 1, when real teams swap
    ends at halftime).

    This function is used ONLY by the renderer's optional physical-
    calibration path; it does not affect, and is never called by,
    src/build_statsbomb_360_dataset.py's analytical pipeline, which remains
    exactly as before (one fixed, period-independent reference for the
    whole match -- appropriate for its purpose, see
    resolve_reference_team_id's docstring).
    """
    base_reference_id = resolve_reference_team_id(events, physical_reference_team_name)
    return apply_period_end_swap(events, base_reference_id, period)


def apply_period_end_swap(events: list[dict[str, Any]], baseline_reference_team_id: int, period: int) -> int:
    """Core, id-based (no team-name lookup) period-aware swap: given a
    reference team id that is correct for PERIOD 1's arrangement (however
    that baseline was chosen -- explicit physical calibration, OR the
    automatic/analytical pick), return the correct reference team id for
    `period`, applying the same real, Law-of-the-Game halftime (and
    extra-time-half) end swap.

    This is deliberately id-based and baseline-agnostic: it does not care
    HOW `baseline_reference_team_id` was determined, only that it is valid
    for period 1. This lets both the physically-calibrated path
    (resolve_physical_reference_team_id, baseline from a verified team
    name) and the renderer's default "spectator-facing" path (baseline
    from the automatic, unverified resolve_reference_team_id pick) share
    the identical, tested swap logic -- see
    docs/full_visualization_orientation_and_clock_audit_2026-10-22.md for
    why the swap must not be restricted to only the physically-calibrated
    case: "teams change ends at halftime" is a universal rule of the
    sport, true whether or not any specific match's absolute left/right
    has been verified against real footage.
    """
    team_ids, _ = _two_team_ids_and_names(events)
    if not period_requires_physical_end_swap(period):
        return baseline_reference_team_id
    return next(tid for tid in team_ids if tid != baseline_reference_team_id)


def team_normalization_label(event: dict[str, Any], reference_team_id: int) -> str:
    """Return REFERENCE if `event` belongs to the reference team, else MIRRORED."""
    team = event.get("team")
    team_id = team.get("id") if isinstance(team, dict) else None
    return TEAM_NORMALIZATION_REFERENCE if team_id == reference_team_id else TEAM_NORMALIZATION_MIRRORED


def _mirror_xy(value: Any) -> Any:
    """Mirror a [x, y] location field; anything else is returned unchanged."""
    if not isinstance(value, list) or len(value) != 2:
        return value
    x, y = value
    if isinstance(x, bool) or isinstance(y, bool):
        return value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return value
    mirrored_x, mirrored_y = mirror_point((float(x), float(y)))
    return [mirrored_x, mirrored_y]


def apply_team_normalization_to_event(event: dict[str, Any], reference_team_id: int) -> dict[str, Any]:
    """Return a copy of `event` with its spatial fields mirrored if it
    belongs to the non-reference team; an unchanged (shallow) copy otherwise.

    Only `location` and any nested `...["end_location"]` (pass.end_location,
    carry.end_location, shot.end_location, etc. -- found generically by key,
    so every qualifier type is covered without hard-coding each one) are
    touched. team/type/timestamp/period/player/ids are never modified.
    """
    if team_normalization_label(event, reference_team_id) == TEAM_NORMALIZATION_REFERENCE:
        return dict(event)
    return rotate_event_180(event)


def rotate_event_180(event: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `event` with every spatial field rotated through
    the pitch centre: `location` and every nested `end_location`. Shared by
    team normalization and the display-only orientation layer
    (src/broadcast_orientation.py), so both rotate events identically.
    """
    normalized = dict(event)
    if "location" in normalized:
        normalized["location"] = _mirror_xy(normalized["location"])

    for key, value in list(normalized.items()):
        if isinstance(value, dict) and "end_location" in value:
            nested_qualifier = dict(value)
            nested_qualifier["end_location"] = _mirror_xy(nested_qualifier["end_location"])
            normalized[key] = nested_qualifier

    return normalized


def apply_team_normalization_to_frame(
    frame_360: dict[str, Any], event: dict[str, Any], reference_team_id: int
) -> dict[str, Any]:
    """Return a copy of a 360 frame mirrored if its OWNING event belongs to
    the non-reference team; an unchanged (shallow) copy otherwise.

    Reuses apply_correction() from orientation_correction.py to perform the
    actual whole-snapshot rotation (every freeze_frame player -- actor,
    keeper, and everyone else -- plus the whole visible_area polygon,
    together, never a single point in isolation). teammate/actor/keeper
    flags are never touched, only coordinates.
    """
    if team_normalization_label(event, reference_team_id) == TEAM_NORMALIZATION_REFERENCE:
        return dict(frame_360)
    return apply_correction(frame_360, ORIENTATION_ROTATED_180)
