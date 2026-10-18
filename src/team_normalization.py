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


def resolve_reference_team_id(events: list[dict[str, Any]]) -> int:
    """Pick one of the match's two team ids as the normalization reference.

    Deterministic, not arbitrary in the sense of being unstable: the first
    team id encountered while scanning `events` in the given (chronological)
    order is used, so the same match always resolves to the same reference
    regardless of which consumer (dataset builder, animation) computes it,
    as long as both are handed a chronologically-ordered event list -- which
    StatsBomb's raw events.json, and this project's chronologically-sorted
    usable_frames, both are.

    Which of the two teams ends up "the reference" is not itself meaningful
    (Problem B does not attempt to recover true physical sides) -- only that
    exactly one is picked and used consistently for the whole match.
    """
    team_ids: list[int] = []
    for event in events:
        team = event.get("team")
        if not isinstance(team, dict):
            continue
        team_id = team.get("id")
        if isinstance(team_id, bool) or not isinstance(team_id, int):
            continue
        if team_id not in team_ids:
            team_ids.append(team_id)
        if len(team_ids) == 2:
            break
    if len(team_ids) != 2:
        raise ValueError(f"Expected exactly two team ids in events, found {team_ids}.")
    return team_ids[0]


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
